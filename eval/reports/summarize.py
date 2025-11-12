"""Summarize evaluation results into report."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List


def _eligible(r: Dict[str, Any]) -> bool:
    """Check if result is eligible for integration F1."""
    scores = r.get("scores", {})
    return scores.get("integration_f1_eligible", False)


def _f1(r: Dict[str, Any], mode: str) -> float:
    """Extract F1 score for given mode."""
    scores = r.get("scores", {})
    if mode == "lax":
        return scores.get("integration_f1_lax", 0.0) or 0.0
    else:  # strict
        return scores.get("integration_f1_strict", 0.0) or 0.0


def _prec(r: Dict[str, Any], mode: str) -> float:
    """Extract precision for given mode."""
    scores = r.get("scores", {})
    if mode == "lax":
        return scores.get("integration_precision_lax", 0.0) or 0.0
    else:  # strict
        return scores.get("integration_precision_strict", 0.0) or 0.0


def _rec(r: Dict[str, Any], mode: str) -> float:
    """Extract recall for given mode."""
    scores = r.get("scores", {})
    if mode == "lax":
        return scores.get("integration_recall_lax", 0.0) or 0.0
    else:  # strict
        return scores.get("integration_recall_strict", 0.0) or 0.0


def macro_f1(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged F1 over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_f1(r, mode) for r in elig) / len(elig)


def macro_precision(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged precision over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_prec(r, mode) for r in elig) / len(elig)


def macro_recall(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged recall over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_rec(r, mode) for r in elig) / len(elig)


def micro_f1(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute micro-averaged F1 over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    # Micro over items: compute global P/R, then F1
    prec = sum(_prec(r, mode) for r in elig) / len(elig) if elig else 0.0
    rec = sum(_rec(r, mode) for r in elig) / len(elig) if elig else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def per_tool_deltas(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-tool lax/strict macro-F1 and compute delta = lax - strict.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        tool_trace = r.get("tool_trace", [])
        tool_names = {tc.get("name") for tc in tool_trace if isinstance(tc, dict) and tc.get("name")}
        for t in tool_names:
            buckets.setdefault(t, []).append(r)
    
    out: Dict[str, Dict[str, float]] = {}
    for tool, bucket in buckets.items():
        if not bucket:
            continue
        # Filter to eligible items only
        elig_bucket = [r for r in bucket if _eligible(r)]
        if not elig_bucket:
            continue
        lax = macro_f1(elig_bucket, mode="lax")
        strict = macro_f1(elig_bucket, mode="strict")
        out[tool] = {
            "f1_lax_macro": round(lax, 3),
            "f1_strict_macro": round(strict, 3),
            "delta_lax_minus_strict": round(lax - strict, 3),
            "sample_count": len(elig_bucket),
        }
    return out


def summarize_results(
    results: List[Dict[str, Any]],
    report_version: str,
    dataset_header: Optional[Dict[str, Any]],
    dataset_sha256: str,
    tool_registry_sha256: Optional[str],
    tokenizer_fingerprint: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    wall_time_sec: float,
    gates_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize evaluation results into report.
    
    Args:
        results: List of per-item result dicts
        report_version: Report schema version
        dataset_header: Dataset header dict
        dataset_sha256: SHA256 of dataset
        tool_registry_sha256: Tool registry SHA256
        tokenizer_fingerprint: Tokenizer fingerprint dict
        config: Evaluation config (runner, model, seed, etc.)
        wall_time_sec: Wall time for evaluation
        gates_overrides: Optional gate threshold overrides
        
    Returns:
        Report dict with summary, gates, and fingerprints
    """
    total = len(results)
    eligible_results = [r for r in results if _eligible(r)]
    num_eligible = len(eligible_results)
    
    # Count controls and negative controls
    num_controls = sum(
        1 for r in results
        if r.get("scores", {}).get("controls_with_integration", 0) > 0
    )
    num_negative_controls = sum(
        1 for r in results
        if r.get("scores", {}).get("negative_control_ok", True) is False
    )
    
    # Integration F1 metrics
    avg_integration_f1_macro_lax = macro_f1(results, mode="lax") if eligible_results else None
    avg_integration_f1_micro_lax = micro_f1(results, mode="lax") if eligible_results else None
    avg_integration_f1_macro_strict = macro_f1(results, mode="strict") if eligible_results else None
    avg_integration_f1_micro_strict = micro_f1(results, mode="strict") if eligible_results else None
    
    # Integration span histogram
    integration_span_count_histogram = defaultdict(int)
    integration_spans_over_cap_count = 0
    for r in results:
        scores = r.get("scores", {})
        span_count = scores.get("integration_spans_count", 0)
        integration_span_count_histogram[span_count] += 1
        if scores.get("integration_spans_exceeded_cap", False):
            integration_spans_over_cap_count += 1
    
    # Multi-call parity
    multi_call_parity_ok = sum(1 for r in results if r.get("scores", {}).get("multi_call_parity_ok", True))
    multi_call_parity_total = total
    multi_call_parity_rate = round(multi_call_parity_ok / multi_call_parity_total, 3) if multi_call_parity_total > 0 else None
    
    # JSON args validity
    json_args_valid_count = sum(1 for r in results if r.get("scores", {}).get("json_args_valid", True))
    json_args_valid_rate = round(json_args_valid_count / total, 3) if total > 0 else None
    
    # Controls with integration
    controls_with_integration = sum(1 for r in results if r.get("scores", {}).get("controls_with_integration", 0) > 0)
    
    # Privacy
    privacy_ok_count = sum(1 for r in results if r.get("scores", {}).get("privacy_ok", True))
    privacy_ok_rate = round(privacy_ok_count / total, 3) if total > 0 else None
    
    # Per-tool deltas
    per_tool = per_tool_deltas(results)
    
    # Define gates (mirror verifier gates)
    min_eligible = gates_overrides.get("min_eligible_for_gates", 15) if gates_overrides else 15
    gates = {
        "integration_f1_macro_lax": {"threshold": 0.90, "policy": "count_based_misses", "misses_allowed_pct": 0.05, "min_eligible": min_eligible},
        "integration_f1_macro_strict": {"threshold": 0.75, "policy": "warning_only"},
        "multi_call_parity_rate": {"threshold": 0.95, "policy": "count_based_misses", "misses_allowed_pct": 0.05},
        "json_args_valid_rate": {"threshold": 0.98, "policy": "hard_fail"},
        "controls_with_integration": {"threshold": 0, "policy": "hard_fail"},
        "privacy_ok_rate": {"threshold": 1.0, "policy": "hard_fail"},
    }
    
    # Check gates
    gates_ok = True
    inconclusive = False
    
    # Integration F1 gate (lax)
    if num_eligible >= min_eligible:
        f1_threshold = gates["integration_f1_macro_lax"]["threshold"]
        misses_allowed_pct = gates["integration_f1_macro_lax"]["misses_allowed_pct"]
        misses_allowed = max(1, math.ceil(misses_allowed_pct * num_eligible))
        
        # Count misses (items with F1 < threshold)
        misses_count = sum(1 for r in eligible_results if (_f1(r, "lax") or 0.0) < f1_threshold)
        
        if avg_integration_f1_macro_lax and avg_integration_f1_macro_lax < f1_threshold and misses_count > misses_allowed:
            gates_ok = False
    elif num_eligible > 0:
        inconclusive = True
    
    # Controls gate
    if controls_with_integration > 0:
        gates_ok = False
    
    # Privacy gate
    if privacy_ok_rate and privacy_ok_rate < 1.0:
        gates_ok = False
    
    # Multi-call parity gate
    if multi_call_parity_rate and multi_call_parity_rate < gates["multi_call_parity_rate"]["threshold"]:
        misses_allowed_pct = gates["multi_call_parity_rate"]["misses_allowed_pct"]
        misses_allowed = max(1, math.ceil(misses_allowed_pct * total))
        misses_count = total - multi_call_parity_ok
        if misses_count > misses_allowed:
            gates_ok = False
    
    # Build summary
    summary = {
        "total": total,
        "num_eligible": num_eligible,
        "num_controls": num_controls,
        "num_negative_controls": num_negative_controls,
        "controls_with_integration": controls_with_integration,
        "avg_integration_f1_macro_lax": round(avg_integration_f1_macro_lax, 3) if avg_integration_f1_macro_lax is not None else None,
        "avg_integration_f1_micro_lax": round(avg_integration_f1_micro_lax, 3) if avg_integration_f1_micro_lax is not None else None,
        "avg_integration_f1_macro_strict": round(avg_integration_f1_macro_strict, 3) if avg_integration_f1_macro_strict is not None else None,
        "avg_integration_f1_micro_strict": round(avg_integration_f1_micro_strict, 3) if avg_integration_f1_micro_strict is not None else None,
        "multi_call_parity_rate": multi_call_parity_rate,
        "json_args_valid_rate": json_args_valid_rate,
        "privacy_ok_rate": privacy_ok_rate,
        "integration_span_count_histogram": dict(sorted(integration_span_count_histogram.items())),
        "integration_spans_over_cap_count": integration_spans_over_cap_count,
        "per_tool_deltas": per_tool,
        "inconclusive": inconclusive,
        "wall_time_sec": round(wall_time_sec, 2),
    }
    
    # Build report header
    report_header = {
        "report_version": report_version,
        "dataset_sha256": dataset_sha256,
        "tool_registry_sha256": tool_registry_sha256,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "config": config,
        "gates": gates,
    }
    
    # Build full report
    report = {
        "header": report_header,
        "summary": summary,
        "gates_ok": gates_ok,
    }
    
    return report

