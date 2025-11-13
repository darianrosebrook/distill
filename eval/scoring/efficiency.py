"""
Efficiency metrics for latent reasoning evaluation.

Implements:
- Accuracy vs generated_tokens curve
- Accuracy vs wall_clock time curve
- Token reduction percentage calculation
- Baseline comparison (direct CoT vs latent reasoning)
"""
# @author: @darianrosebrook

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for a single run."""
    accuracy: float
    generated_tokens: int
    wall_clock_time_ms: float
    latent_spans_used: int
    refinement_loops: int
    baseline_accuracy: Optional[float] = None
    baseline_tokens: Optional[int] = None
    baseline_time_ms: Optional[float] = None


@dataclass
class EfficiencyCurve:
    """Efficiency curve data points."""
    accuracies: List[float]
    token_counts: List[int]
    time_ms: List[float]
    labels: List[str]  # Labels for each point (e.g., "baseline", "latent")


def calculate_token_reduction(
    baseline_tokens: int,
    current_tokens: int,
) -> float:
    """
    Calculate token reduction percentage.
    
    Args:
        baseline_tokens: Baseline token count
        current_tokens: Current token count
    
    Returns:
        Token reduction percentage (0-1, where 1.0 = 100% reduction)
    """
    if baseline_tokens == 0:
        return 0.0
    reduction = (baseline_tokens - current_tokens) / baseline_tokens
    return max(0.0, min(1.0, reduction))  # Clamp to [0, 1]


def calculate_time_reduction(
    baseline_time_ms: float,
    current_time_ms: float,
) -> float:
    """
    Calculate time reduction percentage.
    
    Args:
        baseline_time_ms: Baseline time in milliseconds
        current_time_ms: Current time in milliseconds
    
    Returns:
        Time reduction percentage (0-1, where 1.0 = 100% reduction)
    """
    if baseline_time_ms == 0:
        return 0.0
    reduction = (baseline_time_ms - current_time_ms) / baseline_time_ms
    return max(0.0, min(1.0, reduction))  # Clamp to [0, 1]


def compute_efficiency_curves(
    metrics_list: List[EfficiencyMetrics],
    baseline_metrics: Optional[EfficiencyMetrics] = None,
) -> Dict[str, EfficiencyCurve]:
    """
    Compute efficiency curves from metrics.
    
    Args:
        metrics_list: List of efficiency metrics
        baseline_metrics: Optional baseline metrics for comparison
    
    Returns:
        Dict with curves:
        - accuracy_vs_tokens: EfficiencyCurve
        - accuracy_vs_time: EfficiencyCurve
    """
    accuracies = []
    token_counts = []
    time_ms = []
    labels = []
    
    # Add baseline if provided
    if baseline_metrics:
        accuracies.append(baseline_metrics.accuracy)
        token_counts.append(baseline_metrics.generated_tokens)
        time_ms.append(baseline_metrics.wall_clock_time_ms)
        labels.append("baseline")
    
    # Add current metrics
    for i, metrics in enumerate(metrics_list):
        accuracies.append(metrics.accuracy)
        token_counts.append(metrics.generated_tokens)
        time_ms.append(metrics.wall_clock_time_ms)
        labels.append(f"latent_{i+1}")
    
    accuracy_vs_tokens = EfficiencyCurve(
        accuracies=accuracies,
        token_counts=token_counts,
        time_ms=time_ms,
        labels=labels,
    )
    
    accuracy_vs_time = EfficiencyCurve(
        accuracies=accuracies,
        token_counts=token_counts,
        time_ms=time_ms,
        labels=labels,
    )
    
    return {
        "accuracy_vs_tokens": accuracy_vs_tokens,
        "accuracy_vs_time": accuracy_vs_time,
    }


def compare_with_baseline(
    current_metrics: EfficiencyMetrics,
    baseline_metrics: EfficiencyMetrics,
) -> Dict[str, Any]:
    """
    Compare current metrics with baseline.
    
    Args:
        current_metrics: Current efficiency metrics
        baseline_metrics: Baseline efficiency metrics
    
    Returns:
        Dict with comparison results:
        - token_reduction: float (0-1)
        - time_reduction: float (0-1)
        - accuracy_delta: float
        - accuracy_maintained: bool
        - meets_efficiency_target: bool (≥25-40% token reduction)
    """
    token_reduction = calculate_token_reduction(
        baseline_metrics.generated_tokens,
        current_metrics.generated_tokens,
    )
    
    time_reduction = calculate_time_reduction(
        baseline_metrics.wall_clock_time_ms,
        current_metrics.wall_clock_time_ms,
    )
    
    accuracy_delta = current_metrics.accuracy - baseline_metrics.accuracy
    accuracy_maintained = accuracy_delta >= -0.01  # Allow small regression
    
    # Target: ≥25-40% token reduction
    meets_efficiency_target = token_reduction >= 0.25
    
    return {
        "token_reduction": token_reduction,
        "time_reduction": time_reduction,
        "accuracy_delta": accuracy_delta,
        "accuracy_maintained": accuracy_maintained,
        "meets_efficiency_target": meets_efficiency_target,
        "token_reduction_percent": token_reduction * 100,
        "time_reduction_percent": time_reduction * 100,
    }


def aggregate_efficiency_metrics(
    metrics_list: List[EfficiencyMetrics],
) -> Dict[str, Any]:
    """
    Aggregate efficiency metrics across multiple runs.
    
    Args:
        metrics_list: List of efficiency metrics
    
    Returns:
        Dict with aggregated statistics:
        - mean_accuracy: float
        - mean_tokens: float
        - mean_time_ms: float
        - mean_loops: float
        - mean_latent_spans: float
        - std_accuracy: float
        - std_tokens: float
    """
    if not metrics_list:
        return {}
    
    accuracies = [m.accuracy for m in metrics_list]
    tokens = [m.generated_tokens for m in metrics_list]
    times = [m.wall_clock_time_ms for m in metrics_list]
    loops = [m.refinement_loops for m in metrics_list]
    latent_spans = [m.latent_spans_used for m in metrics_list]
    
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_tokens": float(np.mean(tokens)),
        "mean_time_ms": float(np.mean(times)),
        "mean_loops": float(np.mean(loops)),
        "mean_latent_spans": float(np.mean(latent_spans)),
        "std_accuracy": float(np.std(accuracies)),
        "std_tokens": float(np.std(tokens)),
        "std_time_ms": float(np.std(times)),
        "count": len(metrics_list),
    }


def evaluate_efficiency_gates(
    current_metrics: EfficiencyMetrics,
    baseline_metrics: EfficiencyMetrics,
    min_token_reduction: float = 0.25,
    max_token_reduction: float = 0.40,
    max_accuracy_regression: float = 0.01,
    max_loop_increase: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate efficiency gates.
    
    Gates:
    - ≥ baseline accuracy with ≥25-40% fewer generated tokens on long chains
    - Average refinement loops ≤ current self-refine
    - Tail latency not worse than baseline
    
    Args:
        current_metrics: Current efficiency metrics
        baseline_metrics: Baseline efficiency metrics
        min_token_reduction: Minimum token reduction (default: 0.25 = 25%)
        max_token_reduction: Maximum token reduction (default: 0.40 = 40%)
        max_accuracy_regression: Maximum allowed accuracy regression (default: 0.01)
        max_loop_increase: Maximum allowed loop increase (if None, no limit)
    
    Returns:
        Dict with gate results:
        - all_gates_passed: bool
        - token_reduction_gate: bool
        - accuracy_gate: bool
        - loop_gate: bool
        - latency_gate: bool
        - details: Dict with individual gate details
    """
    comparison = compare_with_baseline(current_metrics, baseline_metrics)
    
    token_reduction = comparison["token_reduction"]
    accuracy_delta = comparison["accuracy_delta"]
    
    # Gate 1: Token reduction ≥25-40%
    token_reduction_gate = (
        token_reduction >= min_token_reduction and
        token_reduction <= max_token_reduction
    )
    
    # Gate 2: Accuracy maintained (≥ baseline - small regression)
    accuracy_gate = accuracy_delta >= -max_accuracy_regression
    
    # Gate 3: Loop count ≤ baseline (or within limit)
    loop_gate = True
    if max_loop_increase is not None:
        loop_increase = current_metrics.refinement_loops - baseline_metrics.refinement_loops
        loop_gate = loop_increase <= max_loop_increase
    else:
        loop_gate = current_metrics.refinement_loops <= baseline_metrics.refinement_loops
    
    # Gate 4: Latency not worse (time reduction ≥ 0 or small increase)
    time_delta = current_metrics.wall_clock_time_ms - baseline_metrics.wall_clock_time_ms
    latency_gate = time_delta <= 0 or (time_delta / baseline_metrics.wall_clock_time_ms) <= 0.1  # ≤10% increase
    
    all_gates_passed = (
        token_reduction_gate and
        accuracy_gate and
        loop_gate and
        latency_gate
    )
    
    return {
        "all_gates_passed": all_gates_passed,
        "token_reduction_gate": token_reduction_gate,
        "accuracy_gate": accuracy_gate,
        "loop_gate": loop_gate,
        "latency_gate": latency_gate,
        "details": {
            "token_reduction": token_reduction,
            "accuracy_delta": accuracy_delta,
            "loop_delta": current_metrics.refinement_loops - baseline_metrics.refinement_loops,
            "time_delta_ms": time_delta,
            "time_delta_percent": (time_delta / baseline_metrics.wall_clock_time_ms) * 100 if baseline_metrics.wall_clock_time_ms > 0 else 0,
        },
    }

