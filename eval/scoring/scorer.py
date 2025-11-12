"""Scoring module that reuses verifier logic for model evaluation."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

# Import verifier utilities
from scripts.verify_contextual_set import (
    contains_grounding,
    grounded_values_in_span_strict,
    compute_integration_f1,
    compute_integration_f1_strict,
    check_privacy,
)
from tools.schema_registry import ToolSchemaRegistry, validate_args


def extract_integration_spans(text: str) -> List[List[int]]:
    """
    Extract integration spans from model output using regex.
    
    Args:
        text: Model output text
        
    Returns:
        List of [start, end] byte spans for integration regions
    """
    spans = []
    # Use same regex as verifier: Integration: ... (sentence)
    for m in re.finditer(r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)', text, flags=re.UNICODE):
        spans.append([m.start(1), m.end(1)])
    return spans


def extract_tool_calls_from_trace(tool_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from tool_trace (already parsed).
    
    Args:
        tool_trace: List of tool call dicts with name, arguments, result
        
    Returns:
        List of call dicts compatible with verifier format
    """
    calls = []
    for tc in tool_trace:
        calls.append({
            "name": tc.get("name", ""),
            "arguments": tc.get("arguments", {}),
        })
    return calls


def build_tool_result_fields(tool_trace: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build tool_result_fields dict from tool_trace results.
    
    Args:
        tool_trace: List of tool call dicts with result fields
        
    Returns:
        Dict mapping field names to string values (for grounding checks)
    """
    fields = {}
    for tc in tool_trace:
        result = tc.get("result", {})
        if isinstance(result, dict):
            for key, value in result.items():
                fields[key] = str(value)
    return fields


def score_item(
    item: Dict[str, Any],
    model_output: str,
    tool_trace: List[Dict[str, Any]],
    integration_span_cap: int = 3,
) -> Dict[str, Any]:
    """
    Score a single evaluation item using verifier-parity logic.
    
    Args:
        item: Input item (with metadata, expected call_sequence, etc.)
        model_output: Model's generated text
        tool_trace: List of tool calls with brokered results
        integration_span_cap: Maximum integration spans allowed
        
    Returns:
        Dict with per-item scores
    """
    meta = item.get("metadata", {})
    expected_behaviour = meta.get("expected_behaviour", "normal")
    negative_control = meta.get("negative_control", False)
    locale_probe = meta.get("locale_probe", False)
    
    # Extract expected call sequence
    expected_calls = meta.get("call_sequence", [])
    
    # Extract integration spans from model output
    integration_spans_bytes = extract_integration_spans(model_output)
    
    # Apply integration span cap
    integration_spans_exceeded_cap = len(integration_spans_bytes) > integration_span_cap
    if integration_spans_exceeded_cap:
        integration_spans_bytes = integration_spans_bytes[:integration_span_cap]
    
    # Build tool_result_fields from tool_trace results
    tool_result_fields = build_tool_result_fields(tool_trace)
    
    # Extract observed tool calls
    observed_calls = extract_tool_calls_from_trace(tool_trace)
    
    # Determine eligibility (non-control, has tool calls)
    is_eligible = (
        expected_behaviour not in {"no_tool", "decline"}
        and len(observed_calls) > 0
        and len(integration_spans_bytes) > 0
    )
    
    # Control contamination check
    controls_with_integration = 0
    if expected_behaviour in {"no_tool", "decline"}:
        if integration_spans_bytes or observed_calls:
            controls_with_integration = 1
    
    # Negative control check
    negative_control_ok = True
    if negative_control:
        # Check if integration spans are grounded (should be false for negative controls)
        grounded_lax = contains_grounding(model_output, integration_spans_bytes, tool_result_fields)
        if grounded_lax:
            negative_control_ok = False
    
    # Compute integration F1 (lax and strict)
    integration_f1_lax = None
    integration_precision_lax = None
    integration_recall_lax = None
    integration_f1_strict = None
    integration_precision_strict = None
    integration_recall_strict = None
    
    if is_eligible and integration_spans_bytes and tool_result_fields:
        # Lax F1
        prec_lax, rec_lax, f1_lax = compute_integration_f1(
            model_output, integration_spans_bytes, tool_result_fields
        )
        integration_f1_lax = f1_lax
        integration_precision_lax = prec_lax
        integration_recall_lax = rec_lax
        
        # Strict F1
        prec_strict, rec_strict, f1_strict = compute_integration_f1_strict(
            model_output, integration_spans_bytes, tool_result_fields
        )
        integration_f1_strict = f1_strict
        integration_precision_strict = prec_strict
        integration_recall_strict = rec_strict
    
    # Grounding checks
    integration_grounded_lax = False
    integration_grounded_strict = False
    if integration_spans_bytes and tool_result_fields:
        integration_grounded_lax = contains_grounding(
            model_output, integration_spans_bytes, tool_result_fields
        )
        # Strict grounding check
        for span in integration_spans_bytes:
            if len(span) >= 2:
                seg = model_output[span[0]:span[1]]
                if grounded_values_in_span_strict(seg, tool_result_fields):
                    integration_grounded_strict = True
                    break
    
    # Multi-call parity check
    multi_call_parity_ok = True
    if len(expected_calls) > 1:
        # Multi-call: require one observed call per expected call
        if len(observed_calls) != len(expected_calls):
            multi_call_parity_ok = False
        else:
            # Check that tool names match (order matters)
            for exp_call, obs_call in zip(expected_calls, observed_calls):
                if exp_call.get("name") != obs_call.get("name"):
                    multi_call_parity_ok = False
                    break
    
    # JSON args validity check
    json_args_valid = True
    json_args_errors = []
    reg = ToolSchemaRegistry()
    for call in observed_calls:
        tool_name = call.get("name", "")
        args_obj = call.get("arguments", {})
        schema = reg.get(tool_name)
        if schema:
            sem_ok, errs = validate_args(schema, args_obj)
            if not sem_ok:
                json_args_valid = False
                json_args_errors.extend(errs)
        else:
            json_args_valid = False
            json_args_errors.append(f"unknown_tool:{tool_name}")
    
    # Privacy check
    privacy_check = check_privacy(model_output)
    
    # Build scores dict
    scores = {
        "integration_f1_eligible": is_eligible,
        "integration_f1_lax": integration_f1_lax,
        "integration_precision_lax": integration_precision_lax,
        "integration_recall_lax": integration_recall_lax,
        "integration_f1_strict": integration_f1_strict,
        "integration_precision_strict": integration_precision_strict,
        "integration_recall_strict": integration_recall_strict,
        "integration_grounded_lax": integration_grounded_lax,
        "integration_grounded_strict": integration_grounded_strict,
        "integration_spans_count": len(integration_spans_bytes),
        "integration_spans_exceeded_cap": integration_spans_exceeded_cap,
        "multi_call_parity_ok": multi_call_parity_ok,
        "json_args_valid": json_args_valid,
        "json_args_errors": json_args_errors,
        "controls_with_integration": controls_with_integration,
        "negative_control_ok": negative_control_ok,
        "privacy_ok": privacy_check.get("privacy_ok", True),
        "privacy_check": privacy_check,
    }
    
    return scores

