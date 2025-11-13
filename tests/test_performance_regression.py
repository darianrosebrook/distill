"""
Performance regression gate tests.

Validates that performance metrics (TTFT, tokens/s) don't regress beyond thresholds.
@author: @darianrosebrook
"""
import pytest
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    ttft_p50_ms: float  # Time to first token, 50th percentile
    ttft_p95_ms: float  # Time to first token, 95th percentile
    tokens_per_second_p50: float  # Tokens per second, 50th percentile
    tokens_per_second_p95: float  # Tokens per second, 95th percentile
    ane_residency_pct: float  # ANE residency percentage
    memory_mb: float  # Memory usage in MB


def load_baseline_metrics(baseline_path: Path) -> Dict[str, PerformanceMetrics]:
    """Load baseline performance metrics."""
    with open(baseline_path, 'r') as f:
        data = json.load(f)
    
    baselines = {}
    for model_key, metrics_dict in data.items():
        baselines[model_key] = PerformanceMetrics(
            ttft_p50_ms=metrics_dict["ttft_p50_ms"],
            ttft_p95_ms=metrics_dict["ttft_p95_ms"],
            tokens_per_second_p50=metrics_dict["tokens_per_second_p50"],
            tokens_per_second_p95=metrics_dict["tokens_per_second_p95"],
            ane_residency_pct=metrics_dict.get("ane_residency_pct", 0.0),
            memory_mb=metrics_dict.get("memory_mb", 0.0),
        )
    
    return baselines


def check_performance_regression(
    current: PerformanceMetrics,
    baseline: PerformanceMetrics,
    ttft_regression_threshold: float = 0.20,  # 20% regression threshold
    tps_regression_threshold: float = 0.15,  # 15% regression threshold
) -> tuple[bool, list[str]]:
    """
    Check if current metrics show regression compared to baseline.
    
    Args:
        current: Current performance metrics
        baseline: Baseline performance metrics
        ttft_regression_threshold: Maximum allowed TTFT regression (default: 20%)
        tps_regression_threshold: Maximum allowed tokens/s regression (default: 15%)
        
    Returns:
        Tuple of (passes: bool, errors: List[str])
    """
    errors = []
    
    # Check TTFT regression
    ttft_p50_regression = (current.ttft_p50_ms - baseline.ttft_p50_ms) / baseline.ttft_p50_ms
    ttft_p95_regression = (current.ttft_p95_ms - baseline.ttft_p95_ms) / baseline.ttft_p95_ms
    
    if ttft_p50_regression > ttft_regression_threshold:
        errors.append(
            f"TTFT P50 regression {ttft_p50_regression:.1%} exceeds threshold {ttft_regression_threshold:.1%} "
            f"({baseline.ttft_p50_ms:.1f}ms -> {current.ttft_p50_ms:.1f}ms)"
        )
    
    if ttft_p95_regression > ttft_regression_threshold:
        errors.append(
            f"TTFT P95 regression {ttft_p95_regression:.1%} exceeds threshold {ttft_regression_threshold:.1%} "
            f"({baseline.ttft_p95_ms:.1f}ms -> {current.ttft_p95_ms:.1f}ms)"
        )
    
    # Check tokens/s regression
    tps_p50_regression = (baseline.tokens_per_second_p50 - current.tokens_per_second_p50) / baseline.tokens_per_second_p50
    tps_p95_regression = (baseline.tokens_per_second_p95 - current.tokens_per_second_p95) / baseline.tokens_per_second_p95
    
    if tps_p50_regression > tps_regression_threshold:
        errors.append(
            f"Tokens/s P50 regression {tps_p50_regression:.1%} exceeds threshold {tps_regression_threshold:.1%} "
            f"({baseline.tokens_per_second_p50:.1f} -> {current.tokens_per_second_p50:.1f})"
        )
    
    if tps_p95_regression > tps_regression_threshold:
        errors.append(
            f"Tokens/s P95 regression {tps_p95_regression:.1%} exceeds threshold {tps_regression_threshold:.1%} "
            f"({baseline.tokens_per_second_p95:.1f} -> {current.tokens_per_second_p95:.1f})"
        )
    
    # Check ANE residency (should not drop significantly)
    ane_drop = baseline.ane_residency_pct - current.ane_residency_pct
    if ane_drop > 0.10:  # 10% drop threshold
        errors.append(
            f"ANE residency dropped {ane_drop:.1%} ({baseline.ane_residency_pct:.1%} -> {current.ane_residency_pct:.1%})"
        )
    
    return len(errors) == 0, errors


def test_performance_regression_gates(
    current_metrics_path: str,
    baseline_metrics_path: str,
    model_key: str = "student_9b",
):
    """
    Test performance regression gates.
    
    Args:
        current_metrics_path: Path to current performance metrics JSON
        baseline_metrics_path: Path to baseline performance metrics JSON
        model_key: Model key to test
    """
    current_path = Path(current_metrics_path)
    baseline_path = Path(baseline_metrics_path)
    
    if not current_path.exists():
        pytest.skip(f"Current metrics not found: {current_metrics_path}")
    if not baseline_path.exists():
        pytest.skip(f"Baseline metrics not found: {baseline_metrics_path}")
    
    # Load metrics
    with open(current_path, 'r') as f:
        current_data = json.load(f)
    
    baselines = load_baseline_metrics(baseline_path)
    
    if model_key not in current_data:
        pytest.skip(f"Model key '{model_key}' not found in current metrics")
    if model_key not in baselines:
        pytest.skip(f"Model key '{model_key}' not found in baseline metrics")
    
    current_metrics_dict = current_data[model_key]
    current = PerformanceMetrics(
        ttft_p50_ms=current_metrics_dict["ttft_p50_ms"],
        ttft_p95_ms=current_metrics_dict["ttft_p95_ms"],
        tokens_per_second_p50=current_metrics_dict["tokens_per_second_p50"],
        tokens_per_second_p95=current_metrics_dict["tokens_per_second_p95"],
        ane_residency_pct=current_metrics_dict.get("ane_residency_pct", 0.0),
        memory_mb=current_metrics_dict.get("memory_mb", 0.0),
    )
    
    baseline = baselines[model_key]
    
    # Check regression
    passes, errors = check_performance_regression(current, baseline)
    
    if not passes:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        pytest.fail(f"Performance regression detected:\n{error_msg}")


def test_memory_budget_compliance():
    """Test that memory usage stays within budget."""
    # This would load memory budget config and verify actual usage
    pytest.skip("Memory budget compliance test requires runtime memory measurement")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

