"""
Batch policy enforcement for M-series Apple Silicon optimization.

Automatically selects optimal batch size based on workload type (interactive vs offline)
and hardware profile. Enforces policies to optimize latency vs throughput trade-off.

Reference: docs/M_SERIES_ADVANCED_OPTIMIZATIONS.md Phase 12
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BatchPolicyConfig:
    """Configuration for batch policy."""

    interactive_default: int = 1  # Always 1 for interactive
    # Allowed batch sizes for offline (e.g., [2, 4])
    offline_allowed: List[int] = None
    tps_improvement_threshold: float = 0.10  # Minimum 10% TPS improvement
    latency_penalty_threshold: float = 0.10  # Maximum 10% p95 latency penalty

    def __post_init__(self):
        if self.offline_allowed is None:
            self.offline_allowed = [2, 4]


class BatchPolicy:
    """
    Enforce batch policy based on workload type.

    Benefits:
    - Automatic optimization (no manual tuning)
    - Workload-aware (interactive vs offline)
    - Hardware-specific (uses hardware profile)

    Policy:
    - Interactive: Always batch=1 (lowest latency)
    - Offline: Use batch 2-4 if TPS improves ≥10% with <10% p95 latency penalty

    Usage:
        policy = BatchPolicy(hardware_profile)
        batch_size = policy.select_batch_size(workload_type="interactive")
    """

    def __init__(
        self,
        hardware_profile: Optional[Dict[str, Any]] = None,
        config: Optional[BatchPolicyConfig] = None,
    ):
        """
        Initialize batch policy.

        Args:
            hardware_profile: Hardware profile dict (from eval.hw_profile)
            config: Optional BatchPolicyConfig (uses defaults if None)
        """
        self.hardware_profile = hardware_profile or {}

        # Extract config from hardware profile if available
        if config is None:
            profile_config = self.hardware_profile.get("config", {})
            batch_policy_config = profile_config.get("batch_policy", {})

            config = BatchPolicyConfig(
                interactive_default=batch_policy_config.get("interactive_default", 1),
                offline_allowed=batch_policy_config.get("offline_allowed", [2, 4]),
                tps_improvement_threshold=batch_policy_config.get(
                    "tps_improvement_threshold", 0.10
                ),
                latency_penalty_threshold=batch_policy_config.get(
                    "latency_penalty_threshold", 0.10
                ),
            )

        self.config = config

        # Cached optimization results
        self._offline_optimal_batch: Optional[int] = None
        self._optimization_results: Optional[Dict[int, Dict[str, float]]] = None

    def select_batch_size(
        self,
        workload_type: str = "interactive",
        force_batch_size: Optional[int] = None,
    ) -> int:
        """
        Select batch size based on workload type.

        Args:
            workload_type: "interactive" or "offline"
            force_batch_size: Optional override (for testing)

        Returns:
            Optimal batch size
        """
        if force_batch_size is not None:
            return force_batch_size

        if workload_type == "interactive":
            return self.config.interactive_default  # Always 1 for interactive
        else:
            # Offline: use optimized batch size
            return self._optimize_offline_batch()

    def _optimize_offline_batch(self) -> int:
        """
        Optimize batch size for offline workloads.

        Returns:
            Optimal batch size (defaults to first allowed if not optimized)
        """
        # If already optimized, return cached result
        if self._offline_optimal_batch is not None:
            return self._offline_optimal_batch

        # Default: return first allowed batch size
        # In production, this would run benchmarks to find optimal batch
        optimal = self.config.offline_allowed[0] if self.config.offline_allowed else 1
        self._offline_optimal_batch = optimal

        return optimal

    def optimize_from_benchmarks(
        self,
        benchmark_results: Dict[int, Dict[str, float]],
    ) -> int:
        """
        Optimize batch size from benchmark results.

        Args:
            benchmark_results: Dict mapping batch_size to metrics:
                {
                    batch_size: {
                        "tps": float,  # Tokens per second
                        "p95_latency_ms": float,  # P95 latency in milliseconds
                    }
                }

        Returns:
            Optimal batch size
        """
        if not benchmark_results:
            return self.config.interactive_default

        # Get baseline (batch=1) metrics
        baseline = benchmark_results.get(1, {})
        baseline_tps = baseline.get("tps", 0.0)
        baseline_p95 = baseline.get("p95_latency_ms", float("inf"))

        if baseline_tps == 0 or baseline_p95 == float("inf"):
            return self.config.interactive_default

        best_batch = 1
        best_score = 0.0

        # Evaluate each allowed batch size
        for batch_size in self.config.offline_allowed:
            if batch_size not in benchmark_results:
                continue

            metrics = benchmark_results[batch_size]
            tps = metrics.get("tps", 0.0)
            p95_latency = metrics.get("p95_latency_ms", float("inf"))

            if tps == 0 or p95_latency == float("inf"):
                continue

            # Check thresholds
            tps_improvement = (tps - baseline_tps) / baseline_tps if baseline_tps > 0 else 0.0
            latency_penalty = (
                (p95_latency - baseline_p95) / baseline_p95 if baseline_p95 > 0 else 0.0
            )

            # Must meet both thresholds
            if (
                tps_improvement >= self.config.tps_improvement_threshold
                and latency_penalty <= self.config.latency_penalty_threshold
            ):
                # Score: prioritize TPS improvement, penalize latency penalty
                score = tps_improvement - (latency_penalty * 0.5)

                if score > best_score:
                    best_score = score
                    best_batch = batch_size

        self._offline_optimal_batch = best_batch
        self._optimization_results = benchmark_results

        return best_batch

    def should_use_batch(
        self,
        batch_size: int,
        workload_type: str = "interactive",
    ) -> Tuple[bool, str]:
        """
        Check if a batch size is allowed for the workload type.

        Args:
            batch_size: Batch size to check
            workload_type: "interactive" or "offline"

        Returns:
            Tuple of (is_allowed, reason)
        """
        if workload_type == "interactive":
            if batch_size == 1:
                return True, "Interactive workloads always use batch=1"
            else:
                return False, f"Interactive workloads must use batch=1, got {batch_size}"
        else:
            if batch_size in self.config.offline_allowed:
                return True, f"Batch size {batch_size} allowed for offline workloads"
            else:
                return (
                    False,
                    f"Batch size {batch_size} not in allowed list {self.config.offline_allowed}",
                )

    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get policy summary for reporting.

        Returns:
            Dictionary with policy configuration and current optimal batch sizes
        """
        return {
            "interactive_batch": self.config.interactive_default,
            "offline_allowed": self.config.offline_allowed,
            "offline_optimal": self._offline_optimal_batch or self.config.offline_allowed[0]
            if self.config.offline_allowed
            else 1,
            "tps_improvement_threshold": self.config.tps_improvement_threshold,
            "latency_penalty_threshold": self.config.latency_penalty_threshold,
            "optimization_results": self._optimization_results,
        }


def create_batch_policy(
    hardware_profile: Optional[Dict[str, Any]] = None,
    interactive_default: int = 1,
    offline_allowed: Optional[List[int]] = None,
) -> BatchPolicy:
    """
    Convenience function to create batch policy.

    Args:
        hardware_profile: Optional hardware profile dict
        interactive_default: Default batch size for interactive (default: 1)
        offline_allowed: Allowed batch sizes for offline (default: [2, 4])

    Returns:
        BatchPolicy instance
    """
    config = BatchPolicyConfig(
        interactive_default=interactive_default,
        offline_allowed=offline_allowed or [2, 4],
    )
    return BatchPolicy(hardware_profile=hardware_profile, config=config)
