"""
Schema-based tool name inference for Worker datasets.

Infers tool names from JSON arguments using tool schema matching to increase
tool_name_ids coverage from ~0.9% to 20-40%.

This is Phase 4 of the dataset readiness pipeline - a pre-training gate that
must pass before starting the expensive training run.

Author: @darianrosebrook
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, FrozenSet

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.schema_registry import ToolSchemaRegistry, get_registry
from training.json_repair import extract_json_from_text, repair_json


def load_tokenizer(tokenizer_path: Optional[str] = None):
    """Load tokenizer for decoding token IDs."""
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"[infer_tool_name] Loaded tokenizer from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"[infer_tool_name] WARN: Failed to load tokenizer from {tokenizer_path}: {e}")
            return None
    else:
        print(f"[infer_tool_name] WARN: No tokenizer path provided")
        return None


def build_schema_key_map(registry: ToolSchemaRegistry) -> Dict[FrozenSet[str], List[str]]:
    """
    Build map from JSON key sets to tool names.
    
    Extracts required/optional fields from each tool schema and creates
    a mapping that can match JSON objects to tools.
    
    Args:
        registry: Tool schema registry
        
    Returns:
        Dictionary mapping frozenset of keys to list of tool names
    """
    schema_map: Dict[FrozenSet[str], List[str]] = {}
    
    for tool_name, schema in registry.all().items():
        # Extract keys from schema
        keys = set()
        
        # Handle different schema structures
        properties = schema.get("properties", {})
        
        # Check if schema has nested "arguments" structure
        if "arguments" in properties:
            # Schema like: {"name": "tool", "arguments": {"path": "...", "query": "..."}}
            args_props = properties.get("arguments", {})
            if isinstance(args_props, dict) and "properties" in args_props:
                # Nested properties
                keys.update(args_props["properties"].keys())
            elif isinstance(args_props, dict):
                # Direct properties dict
                keys.update(args_props.keys())
        else:
            # Direct properties (e.g., {"q": "...", "top_k": ...})
            keys.update(properties.keys())
        
        # Also check required fields
        required = schema.get("required", [])
        for req_field in required:
            if req_field not in ["name", "arguments"]:  # Skip structural fields
                keys.add(req_field)
        
        # Create key set (frozenset for hashability)
        key_set = frozenset(keys)
        
        if key_set not in schema_map:
            schema_map[key_set] = []
        schema_map[key_set].append(tool_name)
    
    return schema_map


def extract_json_keys(json_obj: Dict[str, Any]) -> FrozenSet[str]:
    """
    Extract keys from JSON object, handling nested structures.
    
    Args:
        json_obj: JSON object (may have nested "arguments" field)
        
    Returns:
        Frozenset of keys
    """
    keys = set()
    
    # Handle nested structure: {"name": "tool", "arguments": {"path": "...", "query": "..."}}
    if "arguments" in json_obj and isinstance(json_obj["arguments"], dict):
        keys.update(json_obj["arguments"].keys())
    else:
        # Direct structure: {"path": "...", "query": "..."}
        keys.update(json_obj.keys())
        # Remove structural fields
        keys.discard("name")
        keys.discard("arguments")
    
    return frozenset(keys)


def match_json_to_tool(
    json_obj: Dict[str, Any],
    schema_map: Dict[FrozenSet[str], List[str]],
    registry: ToolSchemaRegistry
) -> Optional[Tuple[str, str]]:
    """
    Match JSON object to tool name using schema key matching.
    
    Args:
        json_obj: JSON object to match
        schema_map: Map from key sets to tool names
        registry: Tool schema registry
        
    Returns:
        (tool_name, confidence) tuple where confidence is:
        - "exact": Exact key match, unambiguous
        - "partial": Partial match (subset of required keys)
        - "ambiguous": Multiple tools match
        None if no match
    """
    # Extract keys from JSON
    json_keys = extract_json_keys(json_obj)
    
    if not json_keys:
        return None
    
    # Find exact matches
    exact_matches = []
    for key_set, tool_names in schema_map.items():
        if json_keys == key_set:
            exact_matches.extend(tool_names)
    
    if len(exact_matches) == 1:
        return (exact_matches[0], "exact")
    elif len(exact_matches) > 1:
        return (exact_matches[0], "ambiguous")  # Return first, but mark ambiguous
    
    # Try partial matches (JSON has subset of schema keys)
    partial_matches = []
    for key_set, tool_names in schema_map.items():
        if json_keys.issubset(key_set) and len(json_keys) >= len(key_set) * 0.5:
            # At least 50% of keys match
            partial_matches.extend(tool_names)
    
    if len(partial_matches) == 1:
        return (partial_matches[0], "partial")
    elif len(partial_matches) > 1:
        # Try to disambiguate using tool name in JSON if present
        if "name" in json_obj:
            tool_name_in_json = json_obj["name"]
            if tool_name_in_json in partial_matches:
                return (tool_name_in_json, "partial")
        return (partial_matches[0], "ambiguous")
    
    return None


def decode_json_from_token_ids(
    token_ids: List[int],
    tokenizer
) -> Optional[Dict[str, Any]]:
    """
    Decode token IDs to JSON object.
    
    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer for decoding
        
    Returns:
        Parsed JSON dict if successful, None otherwise
    """
    if tokenizer is None or not token_ids:
        return None
    
    try:
        # Decode tokens to text
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Extract JSON from text (may have surrounding text)
        json_str = extract_json_from_text(text, require_valid=True)
        
        if json_str is None:
            # Try repair if JSON is malformed
            json_str = extract_json_from_text(text, require_valid=False)
            if json_str:
                success, repaired_obj, _ = repair_json(json_str, use_jsonrepair=True)
                if success and repaired_obj:
                    return repaired_obj
        
        if json_str:
            # Parse JSON
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
    except Exception as e:
        # Decoding or parsing failed
        pass
    
    return None


def extract_json_from_integration_mask(
    teacher_text: str,
    integration_mask: List[int],
    tokenizer
) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from teacher_text using integration_mask spans.
    
    Args:
        teacher_text: Full teacher text
        integration_mask: Mask indicating integration spans (as token IDs or boolean mask)
        tokenizer: Tokenizer for decoding if needed
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    if not teacher_text or not integration_mask:
        return None
    
    # Try to extract JSON from text directly
    # Integration mask might point to spans containing JSON
    json_str = extract_json_from_text(teacher_text, require_valid=True)
    
    if json_str:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    
    # If direct extraction fails, try repair
    json_str = extract_json_from_text(teacher_text, require_valid=False)
    if json_str:
        success, repaired_obj, _ = repair_json(json_str, use_jsonrepair=True)
        if success and repaired_obj:
            return repaired_obj
    
    return None


def infer_tool_name_from_text(
    teacher_text: str,
    tool_names: List[str]
) -> Optional[str]:
    """
    Infer tool name from text using heuristic matching.
    
    Searches for tool names in text (exact match, quoted, backticked).
    
    Args:
        teacher_text: Teacher output text
        tool_names: List of valid tool names
        
    Returns:
        Tool name if found unambiguously, None otherwise
    """
    if not teacher_text or not tool_names:
        return None
    
    found_tools = []
    
    for tool_name in tool_names:
        # Try various patterns
        patterns = [
            rf'\b{re.escape(tool_name)}\b',  # Word boundary
            rf'"{re.escape(tool_name)}"',  # Quoted
            rf'`{re.escape(tool_name)}`',  # Backticked
            rf"'name'\s*:\s*['"]{re.escape(tool_name)}['"]",  # JSON name field
        ]
        
        for pattern in patterns:
            if re.search(pattern, teacher_text, re.IGNORECASE):
                found_tools.append(tool_name)
                break
    
    # Return only if unambiguous
    if len(found_tools) == 1:
        return found_tools[0]
    
    return None


def infer_tool_name_for_sample(
    sample: Dict[str, Any],
    tokenizer,
    schema_map: Dict[FrozenSet[str], List[str]],
    registry: ToolSchemaRegistry,
    tool_names: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Infer tool name for a single sample using priority ordering.
    
    Priority:
    1. gold_json_text_ids → decode → match schema
    2. integration_mask → extract JSON → match schema
    3. heuristic text matching
    
    Args:
        sample: Dataset sample
        tokenizer: Tokenizer for decoding
        schema_map: Schema key map
        registry: Tool schema registry
        tool_names: List of valid tool names
        
    Returns:
        (tool_name, source, confidence) tuple
        source: "schema" | "mask+schema" | "heuristic" | "original" | None
        confidence: "exact" | "partial" | "ambiguous" | "low" | None
    """
    # Check if already has tool_name_ids
    if sample.get("tool_name_ids"):
        # Extract tool name from existing data if possible
        if "tool_name" in sample:
            return (sample["tool_name"], "original", "exact")
        return (None, "original", None)
    
    teacher_text = sample.get("teacher_text", "")
    
    # Priority 1: gold_json_text_ids
    if "gold_json_text_ids" in sample:
        json_obj = decode_json_from_token_ids(sample["gold_json_text_ids"], tokenizer)
        if json_obj:
            match = match_json_to_tool(json_obj, schema_map, registry)
            if match:
                tool_name, confidence = match
                return (tool_name, "schema", confidence)
    
    # Priority 2: integration_mask
    if "integration_mask" in sample and teacher_text:
        json_obj = extract_json_from_integration_mask(
            teacher_text, sample["integration_mask"], tokenizer
        )
        if json_obj:
            match = match_json_to_tool(json_obj, schema_map, registry)
            if match:
                tool_name, confidence = match
                return (tool_name, "mask+schema", confidence)
    
    # Priority 3: heuristic text matching
    if teacher_text:
        tool_name = infer_tool_name_from_text(teacher_text, tool_names)
        if tool_name:
            return (tool_name, "heuristic", "low")
    
    return (None, None, None)


def infer_tool_names_for_dataset(
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer_path: str,
    registry: ToolSchemaRegistry,
    min_coverage: float = 0.20,
    max_ambiguity: float = 0.05,
    validate_registry: bool = True
) -> Dict[str, Any]:
    """
    Infer tool names for all samples in dataset.
    
    Args:
        input_jsonl: Input dataset file
        output_jsonl: Output dataset file
        tokenizer_path: Path to tokenizer
        registry: Tool schema registry
        min_coverage: Minimum coverage threshold (default: 0.20)
        max_ambiguity: Maximum ambiguity rate (default: 0.05)
        validate_registry: Whether to validate inferred tool names exist in registry
        
    Returns:
        Statistics dictionary
        
    Raises:
        ValueError: If coverage/ambiguity gates fail
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer from {tokenizer_path}")
    
    # Build schema map
    schema_map = build_schema_key_map(registry)
    tool_names = registry.list_tools()
    
    print(f"[infer_tool_name] Built schema map with {len(schema_map)} key patterns")
    print(f"[infer_tool_name] Found {len(tool_names)} tools in registry")
    
    # Load samples
    print(f"[infer_tool_name] Loading samples from {input_jsonl}")
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[infer_tool_name] WARN: Invalid JSON line: {e}")
                continue
    
    print(f"[infer_tool_name] Loaded {len(samples)} samples")
    
    # Identify tool-use samples
    tool_use_samples = [
        s for s in samples
        if s.get("task_type") in ["tool_use", "caws_tool"]
        or s.get("tool_name_ids")
        or s.get("gold_json_text_ids")
        or s.get("integration_mask")
    ]
    
    print(f"[infer_tool_name] Found {len(tool_use_samples)} tool-use samples")
    
    # Statistics
    stats = {
        "total_samples": len(samples),
        "tool_use_samples": len(tool_use_samples),
        "had_tool_name_ids": 0,
        "inferred_from_json": 0,
        "inferred_from_mask": 0,
        "inferred_from_text": 0,
        "ambiguous": 0,
        "no_match": 0,
        "json_parse_errors": 0,
        "invalid_tool_names": 0,
    }
    
    # Process samples
    updated_samples = []
    for sample in samples:
        updated_sample = sample.copy()
        
        # Check if already has tool_name_ids
        if sample.get("tool_name_ids"):
            stats["had_tool_name_ids"] += 1
            updated_samples.append(updated_sample)
            continue
        
        # Skip if not a tool-use sample
        if sample not in tool_use_samples:
            updated_samples.append(updated_sample)
            continue
        
        # Infer tool name
        tool_name, source, confidence = infer_tool_name_for_sample(
            sample, tokenizer, schema_map, registry, tool_names
        )
        
        if tool_name:
            # Validate tool name exists in registry
            if validate_registry and tool_name not in tool_names:
                stats["invalid_tool_names"] += 1
                print(f"[infer_tool_name] WARN: Inferred tool name '{tool_name}' not in registry")
                updated_samples.append(updated_sample)
                continue
            
            # Encode tool name to token IDs
            try:
                tool_name_ids = tokenizer.encode(tool_name, add_special_tokens=False)
                tool_name_mask = [1] * len(tool_name_ids)
                
                # Update sample
                updated_sample["tool_name"] = tool_name
                updated_sample["tool_name_ids"] = tool_name_ids
                updated_sample["tool_name_mask"] = tool_name_mask
                updated_sample["tool_name_source"] = source
                updated_sample["tool_name_confidence"] = confidence
                
                # Track statistics
                if source == "schema":
                    stats["inferred_from_json"] += 1
                elif source == "mask+schema":
                    stats["inferred_from_mask"] += 1
                elif source == "heuristic":
                    stats["inferred_from_text"] += 1
                
                if confidence == "ambiguous":
                    stats["ambiguous"] += 1
            except Exception as e:
                print(f"[infer_tool_name] WARN: Failed to encode tool name '{tool_name}': {e}")
                stats["json_parse_errors"] += 1
        else:
            stats["no_match"] += 1
        
        updated_samples.append(updated_sample)
    
    # Calculate coverage
    total_with_tool_name = (
        stats["had_tool_name_ids"] +
        stats["inferred_from_json"] +
        stats["inferred_from_mask"] +
        stats["inferred_from_text"]
    )
    
    coverage = total_with_tool_name / stats["tool_use_samples"] if stats["tool_use_samples"] > 0 else 0.0
    ambiguity_rate = stats["ambiguous"] / stats["tool_use_samples"] if stats["tool_use_samples"] > 0 else 0.0
    parse_error_rate = stats["json_parse_errors"] / stats["tool_use_samples"] if stats["tool_use_samples"] > 0 else 0.0
    
    print(f"\n[infer_tool_name] Inference Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Tool-use samples: {stats['tool_use_samples']}")
    print(f"  Already had tool_name_ids: {stats['had_tool_name_ids']}")
    print(f"  Inferred from JSON: {stats['inferred_from_json']}")
    print(f"  Inferred from mask: {stats['inferred_from_mask']}")
    print(f"  Inferred from text: {stats['inferred_from_text']}")
    print(f"  Ambiguous matches: {stats['ambiguous']}")
    print(f"  No match: {stats['no_match']}")
    print(f"  JSON parse errors: {stats['json_parse_errors']}")
    print(f"  Invalid tool names: {stats['invalid_tool_names']}")
    print(f"\n[infer_tool_name] Coverage: {total_with_tool_name}/{stats['tool_use_samples']} ({coverage:.1%})")
    print(f"[infer_tool_name] Ambiguity rate: {ambiguity_rate:.1%}")
    print(f"[infer_tool_name] Parse error rate: {parse_error_rate:.1%}")
    
    # Validate gates
    if coverage < min_coverage:
        raise ValueError(
            f"Coverage {coverage:.1%} below threshold {min_coverage:.1%}. "
            f"Need {int(stats['tool_use_samples'] * min_coverage)} samples with tool_name, "
            f"got {total_with_tool_name}"
        )
    
    if ambiguity_rate > max_ambiguity:
        raise ValueError(
            f"Ambiguity rate {ambiguity_rate:.1%} above threshold {max_ambiguity:.1%}. "
            f"Found {stats['ambiguous']} ambiguous matches"
        )
    
    if parse_error_rate > 0.02:  # 2% threshold
        print(f"[infer_tool_name] WARN: Parse error rate {parse_error_rate:.1%} above 2% threshold")
    
    # Write output
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in updated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n[infer_tool_name] Wrote {len(updated_samples)} samples to {output_jsonl}")
    
    # Add calculated metrics to stats
    stats["coverage"] = coverage
    stats["ambiguity_rate"] = ambiguity_rate
    stats["parse_error_rate"] = parse_error_rate
    
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Infer tool names from JSON arguments using schema matching",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input Worker dataset JSONL (e.g., worker_production_final.jsonl)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output dataset JSONL (e.g., worker_production_tools_v1.jsonl)",
    )
    ap.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to student tokenizer",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.20,
        help="Minimum coverage threshold (default: 0.20)",
    )
    ap.add_argument(
        "--max-ambiguity",
        type=float,
        default=0.05,
        help="Maximum ambiguity rate (default: 0.05)",
    )
    ap.add_argument(
        "--validate-registry",
        action="store_true",
        default=True,
        help="Validate inferred tool names exist in registry (default: True)",
    )
    ap.add_argument(
        "--no-validate-registry",
        dest="validate_registry",
        action="store_false",
        help="Skip registry validation",
    )
    
    args = ap.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"[infer_tool_name] ERROR: Input file not found: {input_file}")
        return 1
    
    # Load registry
    try:
        registry = get_registry()
    except Exception as e:
        print(f"[infer_tool_name] ERROR: Failed to load tool registry: {e}")
        return 1
    
    # Run inference
    try:
        stats = infer_tool_names_for_dataset(
            input_file,
            output_file,
            args.tokenizer_path,
            registry,
            min_coverage=args.min_coverage,
            max_ambiguity=args.max_ambiguity,
            validate_registry=args.validate_registry,
        )
        
        print(f"\n[infer_tool_name] SUCCESS: All gates passed")
        return 0
        
    except ValueError as e:
        print(f"\n[infer_tool_name] ERROR: Gate validation failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[infer_tool_name] ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

