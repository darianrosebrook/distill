"""Scoring module that reuses verifier logic for model evaluation."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

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
    integration_spans_exceeded_cap = len(
        integration_spans_bytes) > integration_span_cap
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
        grounded_lax = contains_grounding(
            model_output, integration_spans_bytes, tool_result_fields)
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


def evaluate_speed_gates(
    current_metrics: Dict[str, Dict[str, float]],
    baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    hardware_match: bool = True,
    regression_threshold: float = 0.05,  # 5% regression threshold
    current_hw_profile_key: Optional[str] = None,
    baseline_hw_profile_key: Optional[str] = None,
    current_ane_residency: Optional[Dict[str, float]] = None,
    baseline_ane_residency: Optional[Dict[str, float]] = None,
    min_ane_pct: float = 0.80,
    max_ane_regression_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluate speed gates: relative gates vs baseline on same hardware.

    Also checks ANE residency gates to ensure ANE is being used.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 5

    Args:
        current_metrics: Current speed metrics dict with ttft_ms, tps, ttfa_tokens, ttfa_ms
        baseline_metrics: Baseline metrics from last blessed report (same hardware)
        hardware_match: Whether hardware matches baseline (if False, gates are skipped)
        regression_threshold: Maximum allowed regression (default: 0.05 = 5%)
        current_hw_profile_key: Current hardware profile key (e.g., "m1-max-64g")
        baseline_hw_profile_key: Baseline hardware profile key
        current_ane_residency: Current ANE residency measurements (optional)
        baseline_ane_residency: Baseline ANE residency measurements (optional)
        min_ane_pct: Minimum ANE percentage threshold (default: 0.80 = 80%)
        max_ane_regression_pct: Maximum allowed ANE regression (default: 0.10 = 10%)

    Returns:
        Dictionary with gate evaluation results:
        - gates_passed: bool
        - ttft_regression: dict with p50/p95 regression info
        - tps_regression: dict with p50/p95 regression info
        - ttfa_gate: dict with pass/fail info
        - ane_gates: dict with ANE residency gate results
        - errors: list of error messages
        - warnings: list of warning messages
    """
    errors = []
    gates_passed = True

    # Check hardware profile match (preferred over hardware_match flag)
    if current_hw_profile_key and baseline_hw_profile_key:
        if current_hw_profile_key != baseline_hw_profile_key:
            try:
                from eval.hw_profile import require_same_profile
                require_same_profile(current_hw_profile_key,
                                     baseline_hw_profile_key)
            except SystemExit:
                return {
                    "gates_passed": False,
                    "ttft_regression": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "tps_regression": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "ttfa_gate": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "errors": [f"Hardware profile mismatch: {current_hw_profile_key} vs {baseline_hw_profile_key}"],
                }

    if not hardware_match:
        return {
            "gates_passed": True,  # Skip gates if hardware doesn't match
            "ttft_regression": {"skipped": True, "reason": "hardware_mismatch"},
            "tps_regression": {"skipped": True, "reason": "hardware_mismatch"},
            "ttfa_gate": {"skipped": True, "reason": "hardware_mismatch"},
            "errors": [],
        }

    if baseline_metrics is None:
        # No baseline: gates pass but warn
        return {
            "gates_passed": True,
            "ttft_regression": {"skipped": True, "reason": "no_baseline"},
            "tps_regression": {"skipped": True, "reason": "no_baseline"},
            "ttfa_gate": {"skipped": True, "reason": "no_baseline"},
            "errors": ["No baseline metrics provided; speed gates skipped"],
        }

    # TTFT regression check (p50 and p95)
    ttft_regression = {"p50": {"passed": True}, "p95": {"passed": True}}
    for percentile in ["p50", "p95"]:
        current_val = current_metrics.get("ttft_ms", {}).get(percentile, 0.0)
        baseline_val = baseline_metrics.get("ttft_ms", {}).get(percentile, 0.0)

        if baseline_val > 0:
            regression = (current_val - baseline_val) / baseline_val
            ttft_regression[percentile] = {
                "passed": regression <= regression_threshold,
                "regression": regression,
                "current": current_val,
                "baseline": baseline_val,
            }
            if regression > regression_threshold:
                gates_passed = False
                errors.append(
                    f"TTFT {percentile} regression: {regression*100:.1f}% "
                    f"(current={current_val:.1f}ms, baseline={baseline_val:.1f}ms)"
                )

    # TPS regression check (p50 and p95)
    tps_regression = {"p50": {"passed": True}, "p95": {"passed": True}}
    for percentile in ["p50", "p95"]:
        current_val = current_metrics.get("tps", {}).get(percentile, 0.0)
        baseline_val = baseline_metrics.get("tps", {}).get(percentile, 0.0)

        if baseline_val > 0:
            regression = (baseline_val - current_val) / \
                baseline_val  # TPS: lower is worse
            tps_regression[percentile] = {
                "passed": regression <= regression_threshold,
                "regression": regression,
                "current": current_val,
                "baseline": baseline_val,
            }
            if regression > regression_threshold:
                gates_passed = False
                errors.append(
                    f"TPS {percentile} regression: {regression*100:.1f}% "
                    f"(current={current_val:.1f} tok/s, baseline={baseline_val:.1f} tok/s)"
                )

    # TTFA gate: p95 tokens ≤ 25
    ttfa_tokens_p95 = current_metrics.get(
        "ttfa_tokens", {}).get("p95", float('inf'))
    ttfa_gate = {
        "passed": ttfa_tokens_p95 <= 25.0,
        "ttfa_tokens_p95": ttfa_tokens_p95,
        "threshold": 25.0,
    }
    if ttfa_tokens_p95 > 25.0:
        gates_passed = False
        errors.append(
            f"TTFA gate failed: p95 tokens={ttfa_tokens_p95:.1f} > 25"
        )

    # ANE residency gates (if provided)
    ane_gates = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "current_ane_pct": None,
        "baseline_ane_pct": None,
        "meets_threshold": None,
        "regression_within_limit": None,
    }

    if current_ane_residency:
        current_ane_pct = current_ane_residency.get("ane_time_pct", 0.0)
        ane_gates["current_ane_pct"] = current_ane_pct

        # Check threshold
        if current_ane_pct < min_ane_pct:
            ane_gates["passed"] = False
            ane_gates["meets_threshold"] = False
            ane_gates["errors"].append(
                f"ANE residency {current_ane_pct:.1%} below threshold ({min_ane_pct:.1%})"
            )
            gates_passed = False
        else:
            ane_gates["meets_threshold"] = True

        # Compare with baseline if available
        if baseline_ane_residency:
            baseline_ane_pct = baseline_ane_residency.get("ane_time_pct", 0.0)
            ane_gates["baseline_ane_pct"] = baseline_ane_pct

            if baseline_ane_pct > 0:
                regression = baseline_ane_pct - current_ane_pct
                regression_pct = regression / baseline_ane_pct if baseline_ane_pct > 0 else 0.0

                if regression_pct > max_ane_regression_pct:
                    ane_gates["passed"] = False
                    ane_gates["regression_within_limit"] = False
                    ane_gates["errors"].append(
                        f"ANE residency regression {regression_pct:.1%} exceeds limit ({max_ane_regression_pct:.1%})"
                    )
                    gates_passed = False
                else:
                    ane_gates["regression_within_limit"] = True
        else:
            ane_gates["warnings"].append(
                "No baseline ANE residency for comparison")

    # Combine speed gates and ANE gates
    all_errors = errors + ane_gates["errors"]
    all_warnings = ane_gates["warnings"]

    return {
        "gates_passed": gates_passed,
        "ttft_regression": ttft_regression,
        "tps_regression": tps_regression,
        "ttfa_gate": ttfa_gate,
        "ane_gates": ane_gates,
        "errors": all_errors,
        "warnings": all_warnings,
    }


def load_baseline_speed_metrics(baseline_report_path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load baseline speed metrics from a previous report.

    Args:
        baseline_report_path: Path to baseline report JSON file

    Returns:
        Baseline metrics dict or None if not found
    """
    if not baseline_report_path.exists():
        return None

    try:
        with open(baseline_report_path, 'r') as f:
            report = json.load(f)
            return report.get("speed_metrics")
    except Exception as e:
        print(f"[scorer] WARN: Failed to load baseline metrics: {e}")
        return None
