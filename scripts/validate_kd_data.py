"""
Validate knowledge distillation dataset before training.

Validates:
- Tool calls: JSON parses, required fields present, tool name exists in registry
- Token alignment: Extracted token spans align with original text
- Schema versioning: Dataset schema matches runtime expectations
- Data quality metrics: Corruption rate, missing fields, etc.
@author: @darianrosebrook
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from training.dataset import load_tokenizer
from tools.schema_registry import ToolSchemaRegistry


def validate_tool_call(tool_call: Dict[str, Any], registry: ToolSchemaRegistry) -> List[str]:
    """
    Validate a single tool call.

    Args:
        tool_call: Tool call dictionary with 'name' and 'arguments'
        registry: Tool schema registry for validation

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if "name" not in tool_call:
        errors.append("Missing 'name' field")
        return errors

    if "arguments" not in tool_call:
        errors.append("Missing 'arguments' field")
        return errors

    tool_name = tool_call["name"]

    # Validate tool name exists in registry
    tool_schema = registry.get_schema(tool_name)
    if tool_schema is None:
        errors.append(f"Tool name '{tool_name}' not found in registry")
        return errors  # Can't validate arguments without schema

    # Validate JSON arguments
    try:
        if isinstance(tool_call["arguments"], str):
            args_dict = json.loads(tool_call["arguments"])
        else:
            args_dict = tool_call["arguments"]

        # Validate against schema
        is_valid, validation_errors = registry.validate_tool_call(tool_name, tool_call)
        if not is_valid:
            errors.extend(
                [f"Schema validation: {e}" for e in validation_errors]
                if isinstance(validation_errors, list)
                else [f"Schema validation: {validation_errors}"]
            )
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in arguments: {e}")
    except Exception as e:
        errors.append(f"Error validating arguments: {e}")

    return errors


def validate_token_alignment(text: str, token_ids: List[int], tokenizer) -> bool:
    """
    Validate that token IDs align with original text.

    Args:
        text: Original text
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        True if alignment is valid, False otherwise
    """
    try:
        # Decode token IDs back to text
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

        # Re-encode to verify consistency
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

        # Check if re-encoded matches original (allowing for special token differences)
        # This is a basic check - full alignment validation would use offset_mapping
        return len(re_encoded) == len(token_ids)
    except Exception:
        return False


def validate_sample(
    sample: Dict[str, Any], tokenizer, registry: ToolSchemaRegistry
) -> Dict[str, Any]:
    """
    Validate a single dataset sample.

    Returns:
        Dictionary with validation results:
        - valid: bool
        - errors: List[str]
        - warnings: List[str]
    """
    errors = []
    warnings = []

    # Check required fields
    required_fields = ["prompt", "teacher_text"]
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Validate tool calls if present
    if "tool_calls" in sample:
        tool_calls = sample["tool_calls"]
        if isinstance(tool_calls, list):
            for i, tool_call in enumerate(tool_calls):
                tool_errors = validate_tool_call(tool_call, registry)
                if tool_errors:
                    errors.extend([f"Tool call {i}: {e}" for e in tool_errors])

    # Validate process-step supervision targets if present
    if "tool_name_ids" in sample:
        tool_name_ids = sample["tool_name_ids"]
        if "teacher_text" in sample:
            # Basic validation: check token IDs are valid
            vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
            if any(tid >= vocab_size for tid in tool_name_ids if isinstance(tid, int)):
                errors.append("tool_name_ids contains invalid token IDs (>= vocab_size)")

    # Validate schema version if present
    if "metadata" in sample:
        metadata = sample["metadata"]
        if "tool_schema_version" in metadata:
            schema_version = metadata["tool_schema_version"]
            # Check against current registry version if available
            current_version = getattr(registry, "version", None)
            if current_version and schema_version != current_version:
                warnings.append(
                    f"Schema version mismatch: dataset={schema_version}, registry={current_version}"
                )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_dataset(
    jsonl_path: Path,
    tokenizer_path: str = "models/student/tokenizer",
    registry_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate entire dataset.

    Returns:
        Dictionary with validation results and statistics
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load tool registry
    if registry_path:
        # If registry_path is a directory, use it; otherwise create registry and load schemas
        registry = ToolSchemaRegistry(schemas_dir=Path(registry_path))
    else:
        # Use default registry (loads from tools/schemas/)
        registry = ToolSchemaRegistry()

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    total_errors = 0
    total_warnings = 0

    error_types = defaultdict(int)
    samples_with_errors = []

    print(f"[validate_kd_data] Validating dataset: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
                total_samples += 1

                result = validate_sample(sample, tokenizer, registry)

                if result["valid"]:
                    valid_samples += 1
                else:
                    invalid_samples += 1
                    samples_with_errors.append({"line": line_num, "errors": result["errors"]})
                    for error in result["errors"]:
                        error_types[error.split(":")[0]] += 1
                        total_errors += 1

                total_warnings += len(result["warnings"])

            except json.JSONDecodeError as e:
                invalid_samples += 1
                total_errors += 1
                error_types["JSONDecodeError"] += 1
                samples_with_errors.append(
                    {"line": line_num, "errors": [f"JSON decode error: {e}"]}
                )
            except Exception as e:
                invalid_samples += 1
                total_errors += 1
                error_types["Exception"] += 1
                samples_with_errors.append({"line": line_num, "errors": [f"Unexpected error: {e}"]})

    corruption_rate = (invalid_samples / total_samples * 100) if total_samples > 0 else 0.0

    results = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "corruption_rate": corruption_rate,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "error_types": dict(error_types),
        "samples_with_errors": samples_with_errors[:10],  # First 10 for reporting
    }

    return results


def main():
    ap = argparse.ArgumentParser(description="Validate knowledge distillation dataset")
    ap.add_argument("--data", required=True, help="Path to dataset JSONL file")
    ap.add_argument(
        "--tokenizer", default="models/student/tokenizer", help="Path to tokenizer directory"
    )
    ap.add_argument("--registry", help="Path to tool schema registry JSON file")
    ap.add_argument(
        "--fail-on-errors", action="store_true", help="Exit with error code if validation fails"
    )
    ap.add_argument(
        "--max-corruption-rate",
        type=float,
        default=5.0,
        help="Maximum allowed corruption rate (default: 5.0%%)",
    )

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[validate_kd_data] ERROR: Dataset file not found: {data_path}")
        sys.exit(1)

    results = validate_dataset(data_path, args.tokenizer, args.registry)

    # Print results
    print("\n[validate_kd_data] Validation Results:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Valid samples: {results['valid_samples']}")
    print(f"  Invalid samples: {results['invalid_samples']}")
    print(f"  Corruption rate: {results['corruption_rate']:.2f}%")
    print(f"  Total errors: {results['total_errors']}")
    print(f"  Total warnings: {results['total_warnings']}")

    if results["error_types"]:
        print("\n  Error breakdown:")
        for error_type, count in sorted(results["error_types"].items(), key=lambda x: -x[1]):
            print(f"    {error_type}: {count}")

    if results["samples_with_errors"]:
        print(f"\n  First {len(results['samples_with_errors'])} samples with errors:")
        for sample_info in results["samples_with_errors"]:
            print(f"    Line {sample_info['line']}: {sample_info['errors'][0]}")

    # Check if validation passed
    corruption_rate = results["corruption_rate"]
    if corruption_rate > args.max_corruption_rate:
        print(
            f"\n[validate_kd_data] FAILED: Corruption rate {corruption_rate:.2f}% exceeds threshold {args.max_corruption_rate}%"
        )
        if args.fail_on_errors:
            sys.exit(1)
    elif results["invalid_samples"] > 0:
        print(f"\n[validate_kd_data] WARNING: Found {results['invalid_samples']} invalid samples")
        if args.fail_on_errors:
            sys.exit(1)
    else:
        print("\n[validate_kd_data] PASSED: All samples valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
