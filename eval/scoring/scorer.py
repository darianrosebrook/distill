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
    for m in re.finditer(r"Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)", text, flags=re.UNICODE):
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
        calls.append(
            {
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", {}),
            }
        )
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
    latent_spans_used: int = 0,
    refinement_loops: int = 1,
    halt_logits: Optional[List[float]] = None,
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
    meta.get("locale_probe", False)

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
                seg = model_output[span[0] : span[1]]
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

    # Code-mode detection and CES metrics
    # Detect TS API usage vs direct tool calls
    used_code_mode = False
    used_direct_tool = False

    ts_api_patterns = [
        "from './servers",
        'from "./servers',
        "callMCPTool(",
        "import * as",
    ]
    direct_tool_patterns = [
        "<|tool_call|>",
        "<|tool_result|>",
    ]

    if any(pattern in model_output for pattern in ts_api_patterns):
        used_code_mode = True
    if any(pattern in model_output for pattern in direct_tool_patterns):
        used_direct_tool = True

    # CES accounting: count what the model actually read/returned
    # Includes: prompt, model output, tool definitions loaded, tool results in context,
    # file reads (progressive disclosure), and log returns

    # Prompt tokens (input)
    # Rough estimate - actual would use tokenizer
    tokens_in = len(model_output.split())

    # Model output tokens
    tokens_out = len(model_output.split())  # Rough estimate

    # Tool definition tokens (loaded on demand in code-mode)
    tool_def_tokens = sum(
        len(str(tc.get("name", ""))) + len(str(tc.get("arguments", {}))) for tc in observed_calls
    )
    tool_def_tokens = tool_def_tokens // 4  # Rough token estimate

    # Tool result tokens that hit context (not sandbox-isolated)
    tool_result_tokens = 0
    if used_direct_tool:
        # Direct tool calls echo results into context
        tool_result_tokens = sum(len(str(tc.get("result", {}))) for tc in tool_trace)
        tool_result_tokens = tool_result_tokens // 4
    elif used_code_mode:
        # Code-mode: results stay in sandbox, only summaries/logs returned
        # Estimate log return tokens (console.log outputs)
        log_pattern = re.compile(r"console\.log\([^)]+\)", re.IGNORECASE)
        log_matches = log_pattern.findall(model_output)
        tool_result_tokens = sum(len(match) for match in log_matches) // 4

    # File read tokens (progressive disclosure - files read by sandbox)
    # This would be instrumented by sandbox runtime; for now estimate from metadata
    file_read_tokens = 0
    intermediate_sizes = meta.get("intermediate_sizes", [])
    if used_code_mode and intermediate_sizes:
        # In code-mode, large files are read but not echoed
        # Count as "read" but not "in context"
        file_read_bytes = sum(intermediate_sizes)
        file_read_tokens = file_read_bytes // 4  # Rough estimate

    # CES tokens: total context efficiency tokens
    # Includes: prompt + model_out + tool_defs_loaded + tool_results_in_context + file_reads + log_returns
    ces_tokens_total = (
        tokens_in + tokens_out + tool_def_tokens + tool_result_tokens + file_read_tokens
    )
    ces_tokens_direct_tool = tool_result_tokens if used_direct_tool else 0
    ces_tokens_code_mode = (tool_result_tokens + file_read_tokens) if used_code_mode else 0

    # Data leak detection (PII in tokens when bindings exist)
    # Hardened: only count leaks when binding path exists AND snippet length > threshold
    data_leak = False
    # Minimum length to avoid false positives (e.g., docstrings)
    MIN_SNIPPET_LEN = 24

    if tool_result_fields:
        # Compiled PII patterns (hardened, international support)
        pii_patterns = {
            "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
            "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b"),
            "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        }

        for pii_type, pattern in pii_patterns.items():
            matches_in_output = pattern.findall(model_output)
            matches_in_results = []

            # Extract PII from tool results (binding path exists)
            for field_value in tool_result_fields.values():
                matches_in_results.extend(pattern.findall(str(field_value)))

            # Check if any match appears in both output and results (leak)
            # Only count if: (has_binding_path && matched && snippet_len >= MIN_SNIPPET_LEN)
            for match in matches_in_output:
                if len(match) >= MIN_SNIPPET_LEN and match in matches_in_results:
                    data_leak = True
                    break

            if data_leak:
                break

    # Check eligibility for code-mode
    eligible_for_code_mode = meta.get("eligible_for_code_mode", False)
    if not eligible_for_code_mode:
        # Auto-detect eligibility: min_tools >= 2 OR large intermediates OR PII
        tool_count = len(observed_calls)
        intermediate_sizes = meta.get("intermediate_sizes", [])
        pii_tags_present = meta.get("pii_tags_present", False)
        eligible_for_code_mode = (
            tool_count >= 2
            or (max(intermediate_sizes) >= 10000 if intermediate_sizes else False)
            or pii_tags_present
        )

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
        # Code-mode metrics
        "ces_tokens_total": ces_tokens_total,
        "ces_tokens_direct_tool": ces_tokens_direct_tool,
        "ces_tokens_code_mode": ces_tokens_code_mode,
        "data_leak_events": 1 if data_leak else 0,
        "used_code_mode": used_code_mode,
        "used_direct_tool": used_direct_tool,
        "eligible_for_code_mode": eligible_for_code_mode,
        # Latent reasoning metrics
        "latent_spans_used": latent_spans_used,
        "refinement_loops": refinement_loops,
    }

    # Add halt logits if provided
    if halt_logits is not None:
        scores["halt_logits"] = halt_logits
        # Compute halt probability from logits [continue, halt]
        if len(halt_logits) == 2:
            import numpy as np

            # Softmax to get probabilities
            exp_logits = np.exp(halt_logits)
            probs = exp_logits / exp_logits.sum()
            scores["halt_probability"] = float(probs[1])  # Probability of halting

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

                require_same_profile(current_hw_profile_key, baseline_hw_profile_key)
            except SystemExit:
                return {
                    "gates_passed": False,
                    "ttft_regression": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "tps_regression": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "ttfa_gate": {"skipped": True, "reason": "hardware_profile_mismatch"},
                    "errors": [
                        f"Hardware profile mismatch: {current_hw_profile_key} vs {baseline_hw_profile_key}"
                    ],
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
                    f"TTFT {percentile} regression: {regression * 100:.1f}% "
                    f"(current={current_val:.1f}ms, baseline={baseline_val:.1f}ms)"
                )

    # TPS regression check (p50 and p95)
    tps_regression = {"p50": {"passed": True}, "p95": {"passed": True}}
    for percentile in ["p50", "p95"]:
        current_val = current_metrics.get("tps", {}).get(percentile, 0.0)
        baseline_val = baseline_metrics.get("tps", {}).get(percentile, 0.0)

        if baseline_val > 0:
            regression = (baseline_val - current_val) / baseline_val  # TPS: lower is worse
            tps_regression[percentile] = {
                "passed": regression <= regression_threshold,
                "regression": regression,
                "current": current_val,
                "baseline": baseline_val,
            }
            if regression > regression_threshold:
                gates_passed = False
                errors.append(
                    f"TPS {percentile} regression: {regression * 100:.1f}% "
                    f"(current={current_val:.1f} tok/s, baseline={baseline_val:.1f} tok/s)"
                )

    # TTFA gate: p95 tokens ≤ 25
    ttfa_tokens_p95 = current_metrics.get("ttfa_tokens", {}).get("p95", float("inf"))
    ttfa_gate = {
        "passed": ttfa_tokens_p95 <= 25.0,
        "ttfa_tokens_p95": ttfa_tokens_p95,
        "threshold": 25.0,
    }
    if ttfa_tokens_p95 > 25.0:
        gates_passed = False
        errors.append(f"TTFA gate failed: p95 tokens={ttfa_tokens_p95:.1f} > 25")

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
            ane_gates["warnings"].append("No baseline ANE residency for comparison")

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


def capture_baseline_ces_metrics(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Capture baseline CES metrics from direct-tool runner for comparison.

    Stores baseline metrics by task/seed for later comparison.

    Args:
        results: Evaluation results from direct-tool runner
        output_path: Optional path to save baseline JSON file

    Returns:
        Baseline metrics dict with ces_tokens_total and other CES metrics
    """
    baseline_metrics = {
        "ces_tokens_total": 0,
        "ces_tokens_direct_tool": 0,
        "ces_tokens_code_mode": 0,
        "data_leak_events": 0,
        "code_mode_adoption_rate": 0.0,
        "eligible_count": 0,
        "total_count": len(results),
    }

    eligible_results = []
    for result in results:
        scores = result.get("scores", {})
        if scores.get("eligible_for_code_mode", False):
            eligible_results.append(result)
            baseline_metrics["ces_tokens_total"] += scores.get("ces_tokens_total", 0)
            baseline_metrics["ces_tokens_direct_tool"] += scores.get("ces_tokens_direct_tool", 0)
            baseline_metrics["ces_tokens_code_mode"] += scores.get("ces_tokens_code_mode", 0)
            baseline_metrics["data_leak_events"] += scores.get("data_leak_events", 0)

    baseline_metrics["eligible_count"] = len(eligible_results)

    if eligible_results:
        code_adopted = sum(
            1 for r in eligible_results if r.get("scores", {}).get("used_code_mode", False)
        )
        baseline_metrics["code_mode_adoption_rate"] = code_adopted / len(eligible_results)

    # Save baseline if output path provided
    if output_path:
        import json
        from pathlib import Path

        baseline_path = Path(output_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)

    return baseline_metrics


def evaluate_code_mode_gates(
    results: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    baseline_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate code-mode gates: CES, no-leakage, and adoption rate.

    Reference: code-mode-latent-reasoning.md Milestone 1

    Args:
        results: List of per-item result dicts with scores
        config: Optional gate configuration dict
        baseline_summary: Optional baseline summary for comparison

    Returns:
        Dictionary with gate evaluation results:
        - gates_passed: bool
        - ces_gate: dict with pass/fail info
        - no_leakage_gate: dict with pass/fail info
        - adoption_gate: dict with pass/fail info
        - errors: list of error messages
    """
    errors = []
    gates_passed = True

    # Load gate config (defaults if not provided)
    if config is None:
        config = {}

    gates_cfg = config.get("gates", {})
    context_efficiency_cfg = gates_cfg.get("context_efficiency", {})
    no_leakage_cfg = gates_cfg.get("no_leakage", {})
    code_mode_adoption_cfg = gates_cfg.get("code_mode_adoption", {})

    # Aggregate code-mode metrics
    ces_total = 0
    ces_direct = 0
    ces_code = 0
    leaks = 0
    code_adopted = 0
    code_eligible = 0

    for result in results:
        scores = result.get("scores", {})
        ces_total += scores.get("ces_tokens_total", 0)
        ces_direct += scores.get("ces_tokens_direct_tool", 0)
        ces_code += scores.get("ces_tokens_code_mode", 0)
        leaks += scores.get("data_leak_events", 0)

        if scores.get("eligible_for_code_mode", False):
            code_eligible += 1
            if scores.get("used_code_mode", False):
                code_adopted += 1

    # Compute absolute CES and %Δ vs baseline

    if baseline_summary:
        baseline_ces = baseline_summary.get("ces_tokens_total", 0)
        if baseline_ces > 0:
            ces_delta = ces_total - baseline_ces
            (ces_delta / baseline_ces) * 100

    # Context Efficiency Score gate
    ces_gate = {"passed": True, "skipped": True}
    if context_efficiency_cfg.get("enabled", False):
        ces_gate["skipped"] = False
        min_improvement = context_efficiency_cfg.get("min_improvement", 0.25)

        if baseline_summary:
            baseline_ces = baseline_summary.get("ces_tokens_total", 0)
            if baseline_ces > 0:
                improvement = (baseline_ces - ces_total) / baseline_ces
                ces_gate["passed"] = improvement >= min_improvement
                ces_gate["improvement"] = improvement
                ces_gate["current"] = ces_total
                ces_gate["baseline"] = baseline_ces
                ces_gate["min_improvement"] = min_improvement

                if not ces_gate["passed"]:
                    gates_passed = False
                    errors.append(
                        f"CES gate failed: improvement {improvement:.1%} < {min_improvement:.1%} "
                        f"(current={ces_total}, baseline={baseline_ces})"
                    )
        else:
            ces_gate["warn"] = "No baseline for CES comparison"

    # No-Leakage gate
    no_leakage_gate = {"passed": True, "skipped": True}
    if no_leakage_cfg.get("enabled", False):
        no_leakage_gate["skipped"] = False
        no_leakage_gate["passed"] = leaks == 0
        no_leakage_gate["leak_count"] = leaks

        if leaks > 0:
            gates_passed = False
            errors.append(f"No-leakage gate failed: {leaks} data leak events detected")

    # Code-Mode Adoption gate
    adoption_gate = {"passed": True, "skipped": True}
    if code_mode_adoption_cfg.get("enabled", False):
        adoption_gate["skipped"] = False
        min_rate = code_mode_adoption_cfg.get("min_rate", 0.60)

        if code_eligible > 0:
            adoption_rate = code_adopted / code_eligible
            adoption_gate["passed"] = adoption_rate >= min_rate
            adoption_gate["adoption_rate"] = adoption_rate
            adoption_gate["code_adopted"] = code_adopted
            adoption_gate["code_eligible"] = code_eligible
            adoption_gate["min_rate"] = min_rate

            if not adoption_gate["passed"]:
                gates_passed = False
                errors.append(
                    f"Code-mode adoption gate failed: {adoption_rate:.1%} < {min_rate:.1%} "
                    f"({code_adopted}/{code_eligible} eligible cases)"
                )
        else:
            adoption_gate["warn"] = "No eligible cases for code-mode"

    return {
        "gates_passed": gates_passed,
        "ces_gate": ces_gate,
        "no_leakage_gate": no_leakage_gate,
        "adoption_gate": adoption_gate,
        "ces_tokens_total": ces_total,
        "ces_tokens_direct_tool": ces_direct,
        "ces_tokens_code_mode": ces_code,
        "data_leak_events": leaks,
        "code_mode_adoption_rate": code_adopted / max(1, code_eligible),
        "errors": errors,
    }


# Duplicate function definition removed
def _evaluate_claim_quality(results: List[Dict[str, Any]], claim_extractor: Any) -> Dict[str, Any]:
    """
    Evaluate claim extraction quality across results.

    Returns:
        Dict with claim quality metrics:
        - avg_claim_count: Average claims per item
        - avg_success_rate: Average claim extraction success rate
        - supported_claim_ratio: Ratio of supported claims
        - claim_confidence_avg: Average claim confidence
    """
    if not results:
        return {
            "avg_claim_count": 0,
            "avg_success_rate": 0,
            "supported_claim_ratio": 0,
            "claim_confidence_avg": 0,
        }

    total_claims = 0
    total_success_rate = 0
    supported_claims = 0
    total_confidence = 0
    total_items = 0

    for result in results:
        item_meta = result.get("metadata", {})
        student_output = result.get("model_output", "")
        teacher_output = item_meta.get("teacher_text", "")

        if student_output and teacher_output:
            total_items += 1

            # Extract claims
            student_claims = claim_extractor.extract_claims(student_output)
            teacher_claims = claim_extractor.extract_claims(teacher_output)

            total_claims += len(student_claims)
            total_success_rate += claim_extractor.extract_claim_success_rate(student_output)

            # Check claim support
            for claim in student_claims:
                total_confidence += claim.confidence
                # Simplified support check (would need more sophisticated logic)
                supported = any(
                    claim.statement.lower() in teacher_claim.lower()
                    or teacher_claim.lower() in claim.statement.lower()
                    for teacher_claim in teacher_claims
                )
                if supported:
                    supported_claims += 1

    avg_claim_count = total_claims / max(1, total_items)
    avg_success_rate = total_success_rate / max(1, total_items)
    supported_claim_ratio = supported_claims / max(1, total_claims)
    claim_confidence_avg = total_confidence / max(1, total_claims)

    return {
        "avg_claim_count": avg_claim_count,
        "avg_success_rate": avg_success_rate,
        "supported_claim_ratio": supported_claim_ratio,
        "claim_confidence_avg": claim_confidence_avg,
        "total_items_evaluated": total_items,
    }


def _evaluate_caws_compliance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate CAWS compliance across results.

    Returns:
        Dict with CAWS compliance metrics:
        - compliance_gate_passed: bool
        - avg_budget_penalty: Average budget violation penalty
        - avg_quality_penalty: Average quality violation penalty
        - tier_compliance_rate: % of results within tier limits
    """
    from training.losses import caws_compliance_loss
    from training.claim_extraction import SimpleClaimExtractor

    if not results:
        return {
            "compliance_gate_passed": False,
            "avg_budget_penalty": 0,
            "avg_quality_penalty": 0,
            "tier_compliance_rate": 0,
        }

    claim_extractor = SimpleClaimExtractor()
    total_budget_penalty = 0
    total_quality_penalty = 0
    compliant_items = 0
    total_items = 0

    for result in results:
        item_meta = result.get("metadata", {})
        student_output = result.get("model_output", "")

        if student_output:
            total_items += 1

            # Evaluate compliance loss components
            compliance_loss = caws_compliance_loss(
                student_output=student_output,
                teacher_output=item_meta.get("teacher_text", ""),
                claim_extractor=claim_extractor,
            )

            # Extract penalty components (simplified - would need to expose individual penalties)
            loss_value = compliance_loss.item()

            # Rough approximation: budget penalties tend to be higher for severe violations
            if loss_value > 1.5:  # High loss likely indicates budget violation
                total_budget_penalty += loss_value * 0.7
                total_quality_penalty += loss_value * 0.3
            else:
                total_budget_penalty += loss_value * 0.3
                total_quality_penalty += loss_value * 0.7

            # Check tier compliance (simplified)
            latent_spans = student_output.count("<bot>")
            if latent_spans <= 3:  # Tier 3 max
                compliant_items += 1

    avg_budget_penalty = total_budget_penalty / max(1, total_items)
    avg_quality_penalty = total_quality_penalty / max(1, total_items)
    tier_compliance_rate = compliant_items / max(1, total_items)

    # Compliance gate: low penalties and high tier compliance
    compliance_gate_passed = (
        avg_budget_penalty < 0.5 and avg_quality_penalty < 0.8 and tier_compliance_rate > 0.8
    )

    return {
        "compliance_gate_passed": compliance_gate_passed,
        "avg_budget_penalty": avg_budget_penalty,
        "avg_quality_penalty": avg_quality_penalty,
        "tier_compliance_rate": tier_compliance_rate,
        "total_items_evaluated": total_items,
    }


def _compute_overall_efficiency(
    all_metrics: Dict[str, Any], baseline_summary: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute overall system efficiency combining all metrics.

    Returns:
        Dict with overall efficiency assessment:
        - efficiency_score: Composite efficiency score (0-1)
        - recommendations: List of improvement recommendations
        - bottlenecks: Identified bottlenecks
    """
    score_components = []
    recommendations = []
    bottlenecks = []

    # Code-mode efficiency
    code_mode = all_metrics.get("code_mode", {})
    if code_mode.get("gates_passed", False):
        score_components.append(0.9)  # High score for passing gates
    else:
        score_components.append(0.3)  # Lower score for failing gates
        recommendations.append("Improve code-mode CES efficiency and adoption rates")
        bottlenecks.append("code_mode_efficiency")

    # Latent efficiency
    latent_eff = all_metrics.get("latent_efficiency", {})
    if latent_eff.get("gates_passed", False):
        score_components.append(0.9)
    else:
        score_components.append(0.4)
        recommendations.append("Optimize latent reasoning token reduction and accuracy")
        bottlenecks.append("latent_efficiency")

    # CAWS compliance
    caws_comp = all_metrics.get("caws_compliance", {})
    if caws_comp.get("compliance_gate_passed", False):
        score_components.append(0.95)
    else:
        score_components.append(0.5)
        recommendations.append("Address CAWS budget and quality violations")
        bottlenecks.append("caws_compliance")

    # Claim quality
    claim_qual = all_metrics.get("claim_quality", {})
    claim_score = min(
        1.0,
        claim_qual.get("supported_claim_ratio", 0) * 0.8
        + claim_qual.get("claim_confidence_avg", 0) * 0.2,
    )
    score_components.append(claim_score)

    if claim_score < 0.7:
        recommendations.append("Improve claim extraction and support validation")
        bottlenecks.append("claim_quality")

    # Overall score (weighted average)
    if score_components:
        weights = [0.3, 0.3, 0.2, 0.2]  # Equal weights
        efficiency_score = sum(s * w for s, w in zip(score_components, weights))
    else:
        efficiency_score = 0.0

    # Baseline comparison
    baseline_comparison = {}
    if baseline_summary:
        baseline_efficiency = baseline_summary.get("overall", {}).get("efficiency_score", 0.5)
        improvement = efficiency_score - baseline_efficiency
        baseline_comparison = {
            "baseline_efficiency": baseline_efficiency,
            "current_efficiency": efficiency_score,
            "improvement": improvement,
            "improvement_pct": (improvement / max(0.01, baseline_efficiency)) * 100,
        }

        if improvement < 0:
            recommendations.append(
                f"Efficiency regressed by {abs(improvement):.1f} points vs baseline"
            )
            bottlenecks.append("regression_vs_baseline")

    return {
        "efficiency_score": efficiency_score,
        "score_components": score_components,
        "recommendations": recommendations,
        "bottlenecks": bottlenecks,
        "baseline_comparison": baseline_comparison,
    }


def load_baseline_speed_metrics(
    baseline_report_path: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
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
        with open(baseline_report_path, "r") as f:
            report = json.load(f)
            return report.get("speed_metrics")
    except Exception as e:
        print(f"[scorer] WARN: Failed to load baseline metrics: {e}")
        return None
