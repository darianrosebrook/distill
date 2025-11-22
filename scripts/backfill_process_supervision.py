"""
Backfill process-step supervision fields for Worker samples missing them.

Extracts tool_name_ids, gold_json_text_ids, and integration_mask from
teacher_text using the student tokenizer.

Author: @darianrosebrook
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.extractors import (
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)
from tools.schema_registry import get_registry


def load_tokenizer(tokenizer_path: Optional[str] = None):
    """Load tokenizer for process-step supervision extraction."""
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"[backfill_process_supervision] Loaded tokenizer from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"[backfill_process_supervision] WARN: Failed to load tokenizer from {tokenizer_path}: {e}")
            print(f"[backfill_process_supervision] Process-step supervision will be skipped")
            return None
    else:
        print(f"[backfill_process_supervision] WARN: No tokenizer path provided, skipping process-step supervision")
        return None


def extract_process_step_targets(
    teacher_text: str,
    tokenizer,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract process-step supervision targets from teacher output.
    
    Returns dict with:
    - tool_name_ids: List[int] or None
    - gold_json_text_ids: List[int] or None
    - integration_mask: List[int] or None
    """
    if tokenizer is None:
        return {}

    targets = {}

    # Extract tool name span
    tool_name_span = extract_tool_name_span(teacher_text, tool_names)
    if tool_name_span:
        start_char, end_char = tool_name_span
        tool_name_text = teacher_text[start_char:end_char]
        tool_name_ids = tokenizer.encode(tool_name_text, add_special_tokens=False)
        targets["tool_name_ids"] = tool_name_ids
        targets["tool_name_mask"] = [1] * len(tool_name_ids)

    # Extract JSON argument spans
    json_spans = extract_json_argument_spans(teacher_text)
    if json_spans:
        all_json_ids = []
        all_json_mask = []
        for start_char, end_char in json_spans:
            json_text = teacher_text[start_char:end_char]
            json_ids = tokenizer.encode(json_text, add_special_tokens=False)
            all_json_ids.extend(json_ids)
            all_json_mask.extend([1] * len(json_ids))
        if all_json_ids:
            targets["gold_json_text_ids"] = all_json_ids
            targets["mask_valid_json_tokens"] = all_json_mask

    # Extract integration spans
    integration_spans = identify_integration_spans(teacher_text)
    if integration_spans:
        all_integration_ids = []
        all_integration_mask = []
        for start_char, end_char in integration_spans:
            integration_text = teacher_text[start_char:end_char]
            integration_ids = tokenizer.encode(integration_text, add_special_tokens=False)
            all_integration_ids.extend(integration_ids)
            all_integration_mask.extend([1] * len(integration_ids))
        if all_integration_ids:
            targets["integration_mask"] = all_integration_mask
            # Store integration text IDs separately if needed
            targets["integration_text_ids"] = all_integration_ids

    return targets


def get_tool_names_from_registry() -> List[str]:
    """Get list of tool names from schema registry."""
    try:
        registry = get_registry()
        if hasattr(registry, 'list_tools'):
            return registry.list_tools()
        elif hasattr(registry, '_schemas'):
            return list(registry._schemas.keys())
        else:
            return []
    except Exception as e:
        print(f"[backfill_process_supervision] WARN: Failed to load tool registry: {e}")
        return []


def backfill_sample(
    sample: Dict[str, Any],
    tokenizer,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Backfill process-step supervision for a single sample."""
    sample = sample.copy()
    teacher_text = sample.get("teacher_text", "")
    
    if not teacher_text:
        return sample
    
    # Check if already has process-step supervision
    has_tool_name_ids = bool(sample.get("tool_name_ids"))
    has_gold_json_text_ids = bool(sample.get("gold_json_text_ids"))
    has_integration_mask = bool(sample.get("integration_mask"))
    
    # Skip if already complete
    if has_tool_name_ids and has_gold_json_text_ids and has_integration_mask:
        return sample
    
    # Extract process-step targets
    targets = extract_process_step_targets(teacher_text, tokenizer, tool_names)
    
    # Backfill missing fields
    if not has_tool_name_ids and "tool_name_ids" in targets:
        sample["tool_name_ids"] = targets["tool_name_ids"]
        sample["tool_name_mask"] = targets.get("tool_name_mask", [1] * len(targets["tool_name_ids"]))
    
    if not has_gold_json_text_ids and "gold_json_text_ids" in targets:
        sample["gold_json_text_ids"] = targets["gold_json_text_ids"]
        sample["mask_valid_json_tokens"] = targets.get("mask_valid_json_tokens", [1] * len(targets["gold_json_text_ids"]))
    
    if not has_integration_mask and "integration_mask" in targets:
        sample["integration_mask"] = targets["integration_mask"]
        if "integration_text_ids" in targets:
            sample["integration_text_ids"] = targets["integration_text_ids"]
    
    return sample


def main():
    ap = argparse.ArgumentParser(
        description="Backfill process-step supervision for Worker samples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input Worker dataset JSONL")
    ap.add_argument("--output", dest="output_file", required=True, help="Output JSONL file with backfilled supervision")
    ap.add_argument(
        "--tokenizer-path",
        help="Path to student tokenizer (required for process-step supervision)",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum coverage threshold to report (default: 0.8)",
    )
    args = ap.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output_file)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    tool_names = get_tool_names_from_registry() if tokenizer else None
    
    # Load samples
    print(f"[backfill_process_supervision] Loading samples from {input_file}")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[backfill_process_supervision] WARN: Invalid JSON: {e}")
                continue
    
    print(f"[backfill_process_supervision] Loaded {len(samples)} samples")
    
    # Identify tool-use samples
    tool_use_samples = [
        s for s in samples
        if s.get("task_type") in ["tool_use", "caws_tool"]
        or s.get("tool_name_ids")
        or s.get("gold_json_text_ids")
        or s.get("integration_mask")
    ]
    
    print(f"[backfill_process_supervision] Found {len(tool_use_samples)} tool-use samples")
    
    # Backfill process-step supervision
    backfilled_samples = []
    stats = {
        "total": len(samples),
        "tool_use": len(tool_use_samples),
        "backfilled_tool_name_ids": 0,
        "backfilled_gold_json_text_ids": 0,
        "backfilled_integration_mask": 0,
    }
    
    for sample in samples:
        if sample in tool_use_samples:
            original_sample = sample.copy()
            backfilled_sample = backfill_sample(sample, tokenizer, tool_names)
            
            # Track what was backfilled
            if not original_sample.get("tool_name_ids") and backfilled_sample.get("tool_name_ids"):
                stats["backfilled_tool_name_ids"] += 1
            if not original_sample.get("gold_json_text_ids") and backfilled_sample.get("gold_json_text_ids"):
                stats["backfilled_gold_json_text_ids"] += 1
            if not original_sample.get("integration_mask") and backfilled_sample.get("integration_mask"):
                stats["backfilled_integration_mask"] += 1
            
            backfilled_samples.append(backfilled_sample)
        else:
            backfilled_samples.append(sample)
    
    # Calculate final coverage
    final_tool_name_ids = sum(1 for s in tool_use_samples if backfilled_samples[samples.index(s)].get("tool_name_ids"))
    final_gold_json_text_ids = sum(1 for s in tool_use_samples if backfilled_samples[samples.index(s)].get("gold_json_text_ids"))
    final_integration_mask = sum(1 for s in tool_use_samples if backfilled_samples[samples.index(s)].get("integration_mask"))
    
    tool_name_coverage = final_tool_name_ids / len(tool_use_samples) if tool_use_samples else 0
    gold_json_coverage = final_gold_json_text_ids / len(tool_use_samples) if tool_use_samples else 0
    integration_coverage = final_integration_mask / len(tool_use_samples) if tool_use_samples else 0
    
    print(f"\n[backfill_process_supervision] Backfill Statistics:")
    print(f"  Backfilled tool_name_ids: {stats['backfilled_tool_name_ids']}")
    print(f"  Backfilled gold_json_text_ids: {stats['backfilled_gold_json_text_ids']}")
    print(f"  Backfilled integration_mask: {stats['backfilled_integration_mask']}")
    print(f"\n[backfill_process_supervision] Final Coverage (tool-use samples):")
    print(f"  tool_name_ids: {final_tool_name_ids}/{len(tool_use_samples)} ({tool_name_coverage*100:.1f}%)")
    print(f"  gold_json_text_ids: {final_gold_json_text_ids}/{len(tool_use_samples)} ({gold_json_coverage*100:.1f}%)")
    print(f"  integration_mask: {final_integration_mask}/{len(tool_use_samples)} ({integration_coverage*100:.1f}%)")
    
    # Check if coverage meets threshold
    if tool_use_samples:
        if tool_name_coverage < args.min_coverage:
            print(f"  WARN: tool_name_ids coverage ({tool_name_coverage*100:.1f}%) below threshold ({args.min_coverage*100:.1f}%)")
        if gold_json_coverage < args.min_coverage:
            print(f"  WARN: gold_json_text_ids coverage ({gold_json_coverage*100:.1f}%) below threshold ({args.min_coverage*100:.1f}%)")
        if integration_coverage < args.min_coverage:
            print(f"  WARN: integration_mask coverage ({integration_coverage*100:.1f}%) below threshold ({args.min_coverage*100:.1f}%)")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in backfilled_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n[backfill_process_supervision] Wrote {len(backfilled_samples)} samples to {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

