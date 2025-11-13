"""
Baseline capture and loading for efficiency gate evaluation.

Handles:
- Saving direct CoT baseline artifacts per task/seed
- Loading baselines for comparison
- Persisting efficiency curves and summary deltas
"""
# @author: @darianrosebrook

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from eval.scoring.efficiency import EfficiencyMetrics, aggregate_efficiency_metrics


@dataclass
class BaselineArtifact:
    """Baseline artifact for a single run."""

    run_id: str
    seed: int
    task_name: str
    metrics: EfficiencyMetrics
    config: Dict[str, Any]
    timestamp: str


def save_baseline(
    results: List[Dict[str, Any]],
    run_id: str,
    seed: int,
    task_name: str,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save direct CoT baseline artifact for later comparison.

    Args:
        results: Evaluation results from direct CoT run (latent disabled)
        run_id: Unique run identifier
        seed: Random seed used
        task_name: Task name/identifier
        output_path: Path to save baseline JSON file
        config: Optional config dict to save with baseline

    Returns:
        Saved baseline metadata dict
    """
    from datetime import datetime

    # Extract efficiency metrics from results
    metrics_list = []
    for result in results:
        scores = result.get("scores", {})
        metrics = EfficiencyMetrics(
            accuracy=scores.get("f1_lax", 0.0),
            generated_tokens=scores.get("generated_tokens", 0),
            wall_clock_time_ms=scores.get("wall_clock_time_ms", 0.0),
            latent_spans_used=0,  # Baseline has no latent spans
            refinement_loops=scores.get("refinement_loops", 1),
        )
        metrics_list.append(metrics)

    # Aggregate metrics
    aggregated = aggregate_efficiency_metrics(metrics_list)

    # Create baseline metrics
    baseline_metrics = EfficiencyMetrics(
        accuracy=aggregated.get("mean_accuracy", 0.0),
        generated_tokens=int(aggregated.get("mean_tokens", 0)),
        wall_clock_time_ms=aggregated.get("mean_time_ms", 0.0),
        latent_spans_used=0,
        refinement_loops=int(aggregated.get("mean_loops", 1)),
    )

    # Create baseline artifact
    artifact = BaselineArtifact(
        run_id=run_id,
        seed=seed,
        task_name=task_name,
        metrics=baseline_metrics,
        config=config or {},
        timestamp=datetime.now().isoformat(),
    )

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "run_id": artifact.run_id,
                "seed": artifact.seed,
                "task_name": artifact.task_name,
                "metrics": asdict(artifact.metrics),
                "config": artifact.config,
                "timestamp": artifact.timestamp,
                "aggregated": aggregated,
            },
            f,
            indent=2,
        )

    return {
        "run_id": artifact.run_id,
        "seed": artifact.seed,
        "task_name": artifact.task_name,
        "metrics": asdict(artifact.metrics),
        "output_path": str(output_path),
    }


def load_baseline(baseline_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load baseline artifact from file.

    Args:
        baseline_path: Path to baseline JSON file

    Returns:
        Baseline dict or None if not found
    """
    if not baseline_path.exists():
        return None

    try:
        with open(baseline_path, "r") as f:
            baseline = json.load(f)

        # Convert metrics dict back to EfficiencyMetrics
        metrics_dict = baseline.get("metrics", {})
        baseline["metrics"] = EfficiencyMetrics(**metrics_dict)

        return baseline
    except Exception as e:
        print(f"[baseline] WARN: Failed to load baseline from {baseline_path}: {e}")
        return None


def find_baseline(
    task_name: str,
    seed: int,
    baseline_dir: Path,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Find baseline artifact for a given task/seed.

    Args:
        task_name: Task name/identifier
        seed: Random seed
        baseline_dir: Directory containing baseline artifacts
        run_id: Optional specific run_id to match

    Returns:
        Path to baseline file or None if not found
    """
    if not baseline_dir.exists():
        return None

    # Look for baseline files matching pattern: {task_name}_seed{seed}_*.json
    pattern = f"{task_name}_seed{seed}_*.json"
    matches = list(baseline_dir.glob(pattern))

    if not matches:
        return None

    # If run_id specified, try to match it
    if run_id:
        for match in matches:
            try:
                baseline = load_baseline(match)
                if baseline and baseline.get("run_id") == run_id:
                    return match
            except Exception:
                continue

    # Return most recent match
    return max(matches, key=lambda p: p.stat().st_mtime)


def save_efficiency_curves(
    current_metrics: List[EfficiencyMetrics],
    baseline_metrics: Optional[EfficiencyMetrics],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Save efficiency curves (accuracy vs tokens/time) to file.

    Args:
        current_metrics: List of current efficiency metrics
        baseline_metrics: Optional baseline metrics for comparison
        output_path: Path to save curves JSON file

    Returns:
        Curves dict with accuracy vs tokens/time data
    """
    curves = {
        "current": {
            "accuracy": [m.accuracy for m in current_metrics],
            "tokens": [m.generated_tokens for m in current_metrics],
            "time_ms": [m.wall_clock_time_ms for m in current_metrics],
        },
    }

    if baseline_metrics:
        curves["baseline"] = {
            "accuracy": baseline_metrics.accuracy,
            "tokens": baseline_metrics.generated_tokens,
            "time_ms": baseline_metrics.wall_clock_time_ms,
        }

        # Compute deltas
        aggregated = aggregate_efficiency_metrics(current_metrics)
        curves["deltas"] = {
            "accuracy_delta": aggregated.get("mean_accuracy", 0.0) - baseline_metrics.accuracy,
            "token_reduction": (
                baseline_metrics.generated_tokens - aggregated.get("mean_tokens", 0)
            )
            / baseline_metrics.generated_tokens
            if baseline_metrics.generated_tokens > 0
            else 0.0,
            "time_delta_percent": (
                (aggregated.get("mean_time_ms", 0.0) - baseline_metrics.wall_clock_time_ms)
                / baseline_metrics.wall_clock_time_ms
                * 100.0
            )
            if baseline_metrics.wall_clock_time_ms > 0
            else 0.0,
        }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(curves, f, indent=2)

    return curves
