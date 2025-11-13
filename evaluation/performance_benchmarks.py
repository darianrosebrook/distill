"""
Performance Benchmarks for Arbiter Stack Models

This module defines performance targets and evaluation metrics for worker,
judge, and drafter models in the arbiter stack.

Author: @darianrosebrook
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class ModelRole(Enum):
    """Model roles in the arbiter stack"""

    WORKER = "worker"
    JUDGE = "judge"
    DRAFTER = "drafter"


@dataclass
class PerformanceTargets:
    """Performance targets for each model role"""

    # Latency targets (milliseconds)
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput targets (tokens per second)
    tokens_per_second: float

    # Quality targets
    min_accuracy: float
    min_confidence: float

    # Resource targets
    max_memory_mb: float
    max_cpu_percent: float


# Performance targets by model role
PERFORMANCE_TARGETS = {
    ModelRole.WORKER: PerformanceTargets(
        p50_latency_ms=2000.0,  # 2s for typical code edits
        p95_latency_ms=5000.0,  # 5s for complex tasks
        p99_latency_ms=10000.0,  # 10s for very complex tasks
        tokens_per_second=50.0,  # ~50 tokens/sec for 9B model
        min_accuracy=0.85,  # 85% accuracy on code generation
        min_confidence=0.7,  # 70% confidence threshold
        max_memory_mb=18000.0,  # ~18GB for 9B model (FP16)
        max_cpu_percent=80.0,
    ),
    ModelRole.JUDGE: PerformanceTargets(
        p50_latency_ms=30.0,  # 30ms for typical evaluation
        p95_latency_ms=50.0,  # 50ms target (critical requirement)
        p99_latency_ms=100.0,  # 100ms worst case
        tokens_per_second=200.0,  # ~200 tokens/sec for 3-4B model
        min_accuracy=0.90,  # 90% accuracy on CAWS compliance
        min_confidence=0.75,  # 75% confidence threshold
        max_memory_mb=8000.0,  # ~8GB for 7B model (FP16)
        max_cpu_percent=60.0,
    ),
    ModelRole.DRAFTER: PerformanceTargets(
        p50_latency_ms=50.0,  # 50ms per token
        p95_latency_ms=100.0,  # 100ms per token
        p99_latency_ms=200.0,  # 200ms per token
        tokens_per_second=500.0,  # ~500 tokens/sec for 4B model
        min_accuracy=0.70,  # 70% acceptance rate for speculative decoding
        min_confidence=0.65,  # 65% confidence threshold
        max_memory_mb=8000.0,  # ~8GB for 4B model (FP16)
        max_cpu_percent=70.0,
    ),
}


@dataclass
class PerformanceMetrics:
    """Measured performance metrics"""

    latency_ms: float
    tokens_per_second: float
    memory_mb: float
    cpu_percent: float
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    tokens_generated: Optional[int] = None


class PerformanceBenchmark:
    """Performance benchmark evaluator"""

    def __init__(self, model_role: ModelRole):
        self.model_role = model_role
        self.targets = PERFORMANCE_TARGETS[model_role]

    def evaluate_latency(
        self,
        latencies: List[float],
    ) -> Dict[str, bool]:
        """
        Evaluate latency against targets.

        Args:
            latencies: List of latency measurements in milliseconds

        Returns:
            Dictionary with pass/fail for each percentile target
        """
        if not latencies:
            return {
                "p50": False,
                "p95": False,
                "p99": False,
            }

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1]
        p99 = sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1]

        return {
            "p50": p50 <= self.targets.p50_latency_ms,
            "p95": p95 <= self.targets.p95_latency_ms,
            "p99": p99 <= self.targets.p99_latency_ms,
            "p50_value": p50,
            "p95_value": p95,
            "p99_value": p99,
        }

    def evaluate_throughput(
        self,
        tokens_per_second: float,
    ) -> Dict[str, bool]:
        """
        Evaluate throughput against target.

        Args:
            tokens_per_second: Measured tokens per second

        Returns:
            Dictionary with pass/fail and measured value
        """
        return {
            "pass": tokens_per_second >= self.targets.tokens_per_second,
            "target": self.targets.tokens_per_second,
            "measured": tokens_per_second,
        }

    def evaluate_quality(
        self,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, bool]:
        """
        Evaluate quality metrics against targets.

        Args:
            accuracy: Measured accuracy (0.0 to 1.0)
            confidence: Measured confidence (0.0 to 1.0)

        Returns:
            Dictionary with pass/fail for quality metrics
        """
        results = {}

        if accuracy is not None:
            results["accuracy_pass"] = accuracy >= self.targets.min_accuracy
            results["accuracy_value"] = accuracy
            results["accuracy_target"] = self.targets.min_accuracy

        if confidence is not None:
            results["confidence_pass"] = confidence >= self.targets.min_confidence
            results["confidence_value"] = confidence
            results["confidence_target"] = self.targets.min_confidence

        return results

    def evaluate_resources(
        self,
        memory_mb: float,
        cpu_percent: float,
    ) -> Dict[str, bool]:
        """
        Evaluate resource usage against targets.

        Args:
            memory_mb: Measured memory usage in MB
            cpu_percent: Measured CPU usage percentage

        Returns:
            Dictionary with pass/fail for resource metrics
        """
        return {
            "memory_pass": memory_mb <= self.targets.max_memory_mb,
            "memory_value": memory_mb,
            "memory_target": self.targets.max_memory_mb,
            "cpu_pass": cpu_percent <= self.targets.max_cpu_percent,
            "cpu_value": cpu_percent,
            "cpu_target": self.targets.max_cpu_percent,
        }

    def evaluate_all(
        self,
        metrics: PerformanceMetrics,
        latencies: Optional[List[float]] = None,
    ) -> Dict[str, any]:
        """
        Evaluate all performance metrics.

        Args:
            metrics: Performance metrics to evaluate
            latencies: Optional list of latency measurements for percentile analysis

        Returns:
            Comprehensive evaluation results
        """
        results = {
            "model_role": self.model_role.value,
            "targets": {
                "p50_latency_ms": self.targets.p50_latency_ms,
                "p95_latency_ms": self.targets.p95_latency_ms,
                "p99_latency_ms": self.targets.p99_latency_ms,
                "tokens_per_second": self.targets.tokens_per_second,
                "min_accuracy": self.targets.min_accuracy,
                "min_confidence": self.targets.min_confidence,
            },
        }

        # Evaluate latency
        if latencies:
            results["latency"] = self.evaluate_latency(latencies)
        elif metrics.latency_ms:
            results["latency"] = {
                "single_measurement": metrics.latency_ms,
                "pass": metrics.latency_ms <= self.targets.p95_latency_ms,
            }

        # Evaluate throughput
        if metrics.tokens_per_second:
            results["throughput"] = self.evaluate_throughput(metrics.tokens_per_second)

        # Evaluate quality
        results["quality"] = self.evaluate_quality(
            accuracy=metrics.accuracy,
            confidence=metrics.confidence,
        )

        # Evaluate resources
        results["resources"] = self.evaluate_resources(
            memory_mb=metrics.memory_mb,
            cpu_percent=metrics.cpu_percent,
        )

        # Overall pass/fail
        all_pass = (
            results.get("latency", {}).get("pass", True)
            and results.get("throughput", {}).get("pass", True)
            and results.get("quality", {}).get("accuracy_pass", True)
            and results.get("quality", {}).get("confidence_pass", True)
            and results.get("resources", {}).get("memory_pass", True)
            and results.get("resources", {}).get("cpu_pass", True)
        )
        results["overall_pass"] = all_pass

        return results


def measure_inference_time(
    model_fn,
    *args,
    **kwargs,
) -> Tuple[float, any]:
    """
    Measure inference time for a model function.

    Args:
        model_fn: Model inference function
        *args: Positional arguments for model_fn
        **kwargs: Keyword arguments for model_fn

    Returns:
        Tuple of (latency_ms, output)
    """
    start_time = time.time()
    output = model_fn(*args, **kwargs)
    end_time = time.time()

    latency_ms = (end_time - start_time) * 1000.0

    return latency_ms, output


def calculate_tokens_per_second(
    tokens_generated: int,
    latency_ms: float,
) -> float:
    """
    Calculate tokens per second from tokens generated and latency.

    Args:
        tokens_generated: Number of tokens generated
        latency_ms: Latency in milliseconds

    Returns:
        Tokens per second
    """
    if latency_ms <= 0:
        return 0.0

    return (tokens_generated / latency_ms) * 1000.0


# Example usage
if __name__ == "__main__":
    # Test judge model benchmark
    judge_benchmark = PerformanceBenchmark(ModelRole.JUDGE)

    # Simulate latency measurements
    test_latencies = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

    results = judge_benchmark.evaluate_all(
        metrics=PerformanceMetrics(
            latency_ms=45.0,
            tokens_per_second=180.0,
            memory_mb=7500.0,
            cpu_percent=55.0,
            accuracy=0.92,
            confidence=0.78,
        ),
        latencies=test_latencies,
    )

    print("JUDGE MODEL PERFORMANCE EVALUATION:")
    print(f"Overall Pass: {results['overall_pass']}")
    print(
        f"Latency P95: {results['latency']['p95_value']:.2f}ms (target: {results['targets']['p95_latency_ms']}ms)"
    )
    print(
        f"Throughput: {results['throughput']['measured']:.2f} tokens/sec (target: {results['targets']['tokens_per_second']} tokens/sec)"
    )
    print(
        f"Accuracy: {results['quality']['accuracy_value']:.2%} (target: {results['targets']['min_accuracy']:.2%})"
    )
    print(
        f"Memory: {results['resources']['memory_value']:.0f}MB (target: {results['targets']['max_memory_mb']:.0f}MB)"
    )
