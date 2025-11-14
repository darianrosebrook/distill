"""
Extract process-step supervision targets from teacher outputs and add token spans.

Author: @darianrosebrook
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.extractors import (  # noqa: E402
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)
from tools.schema_registry import ToolSchemaRegistry, validate_args  # noqa: E402
from scripts.util_token_spans import (  # noqa: E402
    bytes_to_token_span,
    normalize_text_for_alignment,
)


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    try:
        from transformers import AutoTokenizer
        from training.safe_model_loading import safe_from_pretrained_tokenizer

        return safe_from_pretrained_tokenizer(tokenizer_path, use_fast=True)
    except ImportError:
        raise RuntimeError("transformers required for token span extraction")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def extract_process_step_targets(
    teacher_text: str,
    tokenizer,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract process-step supervision targets from teacher output.

    Returns:
        Dictionary with:
        - tool_name_ids: Token IDs for tool name span (if found)
        - tool_name_mask: Mask for tool name tokens
        - tool_name_span_bytes: Byte offsets [start, end]
        - tool_name_span_tokens: Token offsets [start, end]
        - gold_json_text_ids: Token IDs for JSON argument spans
        - mask_valid_json_tokens: Mask for valid JSON tokens
        - json_args_span_bytes: Byte offsets [start, end]
        - json_args_span_tokens: Token offsets [start, end]
        - integration_token_ids: Token IDs for integration spans (separate from tool_result_fields dict)
        - integration_mask: Mask for integration spans
        - integration_spans_bytes: List of byte offsets [[start, end], ...]
        - integration_spans_tokens: List of token offsets [[start, end], ...]
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
        targets["tool_name_span_bytes"] = [start_char, end_char]

        # Add token spans
        token_span = bytes_to_token_span(teacher_text, start_char, end_char, tokenizer)
        if token_span:
            targets["tool_name_span_tokens"] = list(token_span)

    # Extract JSON argument spans
    json_spans = extract_json_argument_spans(teacher_text)
    if json_spans:
        all_json_ids = []
        all_json_mask = []
        json_bytes_spans = []
        json_token_spans = []

        for start_char, end_char in json_spans:
            json_text = teacher_text[start_char:end_char]
            json_ids = tokenizer.encode(json_text, add_special_tokens=False)
            all_json_ids.extend(json_ids)
            all_json_mask.extend([1] * len(json_ids))
            json_bytes_spans.append([start_char, end_char])

            # Add token spans
            token_span = bytes_to_token_span(teacher_text, start_char, end_char, tokenizer)
            if token_span:
                json_token_spans.append(list(token_span))

        if all_json_ids:
            targets["gold_json_text_ids"] = all_json_ids
            targets["mask_valid_json_tokens"] = all_json_mask
            targets["json_args_span_bytes"] = json_bytes_spans[0] if json_bytes_spans else None
            if json_token_spans:
                targets["json_args_span_tokens"] = json_token_spans[0]

    # Extract integration spans (only if not a control case)
    # Note: We preserve existing integration_spans_bytes from generator if present
    # Only re-extract if missing and this is not a control case
    integration_spans = identify_integration_spans(teacher_text)
    if integration_spans:
        all_integration_ids = []
        all_integration_mask = []
        integration_bytes_spans = []
        integration_token_spans = []

        for start_char, end_char in integration_spans:
            integration_text = teacher_text[start_char:end_char]
            integration_ids = tokenizer.encode(integration_text, add_special_tokens=False)
            all_integration_ids.extend(integration_ids)
            all_integration_mask.extend([1] * len(integration_ids))
            integration_bytes_spans.append([start_char, end_char])

            # Add token spans
            token_span = bytes_to_token_span(teacher_text, start_char, end_char, tokenizer)
            if token_span:
                integration_token_spans.append(list(token_span))

        if all_integration_ids:
            # Don't overwrite tool_result_fields - it's a dict from the generator
            # Store integration token IDs separately if needed for training
            targets["integration_token_ids"] = all_integration_ids
            targets["integration_mask"] = all_integration_mask
            # Only add spans if we found them (generator should have set them, but preserve if missing)
            if integration_bytes_spans:
                targets["integration_spans_bytes"] = integration_bytes_spans
            if integration_token_spans:
                targets["integration_spans_tokens"] = integration_token_spans

    return targets


def validate_json_args(
    teacher_text: str, json_span_bytes: List[int], reg: ToolSchemaRegistry
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate JSON arguments against schema registry.

    Returns:
        (is_valid, tool_name, arguments_dict) tuple
    """
    if not json_span_bytes or len(json_span_bytes) < 2:
        return False, None, None

    try:
        json_text = teacher_text[json_span_bytes[0] : json_span_bytes[1]]
        # Find JSON object in the span
        start_idx = json_text.find("{")
        end_idx = json_text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return False, None, None

        json_obj = json.loads(json_text[start_idx : end_idx + 1])
        tool_name = json_obj.get("name")
        args = json_obj.get("arguments", {})

        if not tool_name:
            return False, None, None

        schema = reg.get(tool_name)
        if not schema:
            return False, tool_name, args

        is_valid, errors = validate_args(schema, args)
        return is_valid, tool_name, args

    except Exception:
        return False, None, None


def extract_integration_fields(
    teacher_text: str, integration_spans_bytes: List[List[int]], call_sequence: List[Dict[str, Any]]
) -> List[str]:
    """
    Extract integration field names from tool results that appear in teacher text.

    Returns:
        List of field names that were integrated
    """
    integration_fields = []

    if not integration_spans_bytes or not call_sequence:
        return integration_fields

    # Look for common tool result field patterns
    # This is a heuristic - in practice, you'd match against actual tool result schemas
    for span in integration_spans_bytes:
        if len(span) >= 2:
            span_text = teacher_text[span[0] : span[1]]
            # Look for patterns like "summary", "results", "content", etc.
            common_fields = ["summary", "results", "content", "data", "output"]
            for field in common_fields:
                if field.lower() in span_text.lower():
                    integration_fields.append(field)

    return integration_fields


def process_sample(
    item: Dict[str, Any],
    tokenizer,
    reg: ToolSchemaRegistry,
    add_token_spans: bool = True,
) -> Dict[str, Any]:
    """
    Process a single sample: extract targets, add token spans, validate.

    Returns:
        Enhanced item dict with process-step targets
    """
    teacher_text = item.get("teacher_text", "")
    metadata = item.get("metadata", {})
    call_sequence = metadata.get("call_sequence", [])

    # Normalize text to match format used when computing byte spans
    text_norm = metadata.get("text_norm")
    line_endings = metadata.get("line_endings")
    teacher_text_normalized = normalize_text_for_alignment(
        teacher_text, text_norm=text_norm, line_endings=line_endings
    )

    # Determine which buffer spans target (default: teacher)
    spans_target = metadata.get("spans_target", "teacher")

    # Get tool names from call sequence
    tool_names = [call.get("name") for call in call_sequence if call.get("name")]

    # Extract process-step targets (use normalized text for span extraction)
    targets = extract_process_step_targets(
        teacher_text=teacher_text_normalized,
        tokenizer=tokenizer,
        tool_names=tool_names if tool_names else None,
    )

    # Validate JSON arguments if we have a span (use normalized text)
    arg_semantics_valid = True
    if "json_args_span_bytes" in targets:
        is_valid, tool_name, args = validate_json_args(
            teacher_text_normalized, targets["json_args_span_bytes"], reg
        )
        arg_semantics_valid = is_valid
        if tool_name:
            targets["validated_tool_name"] = tool_name
        if args:
            targets["validated_arguments"] = args

    # Extract integration fields (use normalized text)
    integration_spans_bytes = targets.get("integration_spans_bytes", [])
    integration_fields = extract_integration_fields(
        teacher_text_normalized, integration_spans_bytes, call_sequence
    )
    if integration_fields:
        targets["integration_fields"] = integration_fields

    # Add to metadata, but preserve existing tool_result_fields if present
    existing_tool_result_fields = metadata.get("tool_result_fields")

    # For control cases, ensure no integration spans
    expected_behaviour = metadata.get("expected_behaviour", "normal")
    is_control = expected_behaviour in {"no_tool", "decline"}

    if is_control:
        # Remove any integration spans that might have been extracted
        targets.pop("integration_spans_bytes", None)
        targets.pop("integration_spans_tokens", None)
        targets.pop("integration_token_ids", None)
        targets.pop("integration_mask", None)
        targets.pop("integration_fields", None)
        # Ensure metadata has empty array
        targets["integration_spans_bytes"] = []

    metadata.update(targets)
    if existing_tool_result_fields and isinstance(existing_tool_result_fields, dict):
        # Preserve the original tool_result_fields dict from generator
        metadata["tool_result_fields"] = existing_tool_result_fields

    # Add spans_target metadata if not present
    if "spans_target" not in metadata:
        metadata["spans_target"] = spans_target

    metadata["arg_semantics_valid"] = arg_semantics_valid

    # Update item
    item["metadata"] = metadata

    return item


def main():
    ap = argparse.ArgumentParser(description="Extract process-step targets and add token spans")
    ap.add_argument("--in", dest="input_file", required=True, help="Input JSONL file")
    ap.add_argument("--out", dest="output_file", required=True, help="Output JSONL file")
    ap.add_argument(
        "--tokenizer-path",
        default="models/student/tokenizer",
        help="Path to tokenizer",
    )
    ap.add_argument(
        "--add-token-spans",
        action="store_true",
        default=True,
        help="Add token span alignments (default: True)",
    )
    args = ap.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Load schema registry
    reg = ToolSchemaRegistry()

    # Process samples (preserve header if present)
    processed = []
    dataset_header = None
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Check if this is the header line
            if item.get("__header__") is True:
                dataset_header = item
                # Preserve header, don't process it
                continue

            processed_item = process_sample(
                item, tokenizer, reg, add_token_spans=args.add_token_spans
            )
            processed.append(processed_item)

    # Write output (preserve header as first line)
    with open(args.output_file, "w", encoding="utf-8") as f:
        # Write header first if present
        if dataset_header:
            f.write(json.dumps(dataset_header, ensure_ascii=False, separators=(",", ":")) + "\n")
        # Write processed items
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"[extract_process_targets] Processed {len(processed)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
