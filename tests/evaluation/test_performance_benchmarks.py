"""
Tests for evaluation/performance_benchmarks.py - Performance benchmarks for arbiter stack models.

Tests performance targets, metrics evaluation, and benchmark functionality.
"""
# @author: @darianrosebrook


import pytest

from evaluation.performance_benchmarks import (
    ModelRole,
    PerformanceTargets,
    PERFORMANCE_TARGETS,
    PerformanceMetrics,
    PerformanceBenchmark,
    measure_inference_time,
    calculate_tokens_per_second,
)


class TestModelRole:
    """Test ModelRole enum."""

    def test_model_role_values(self):
        """Test ModelRole enum values."""
        assert ModelRole.WORKER.value == "worker"
        assert ModelRole.JUDGE.value == "judge"
        assert ModelRole.DRAFTER.value == "drafter"

    def test_model_role_count(self):
        """Test that we have 3 model roles."""
        assert len(ModelRole) == 3


class TestPerformanceTargets:
    """Test PerformanceTargets dataclass."""

    def test_performance_targets_creation(self):
        """Test creating PerformanceTargets."""
        targets = PerformanceTargets(
            p50_latency_ms=100.0,
            p95_latency_ms=200.0,
            p99_latency_ms=300.0,
            tokens_per_second=50.0,
            min_accuracy=0.85,
            min_confidence=0.7,
            max_memory_mb=8000.0,
            max_cpu_percent=70.0,
        )

        assert targets.p50_latency_ms == 100.0
        assert targets.p95_latency_ms == 200.0
        assert targets.p99_latency_ms == 300.0
        assert targets.tokens_per_second == 50.0
        assert targets.min_accuracy == 0.85
        assert targets.min_confidence == 0.7
        assert targets.max_memory_mb == 8000.0
        assert targets.max_cpu_percent == 70.0


class TestPerformanceTargetsConstants:
    """Test PERFORMANCE_TARGETS dictionary."""

    def test_performance_targets_all_roles(self):
        """Test that all model roles have targets."""
        assert ModelRole.WORKER in PERFORMANCE_TARGETS
        assert ModelRole.JUDGE in PERFORMANCE_TARGETS
        assert ModelRole.DRAFTER in PERFORMANCE_TARGETS

    def test_worker_targets(self):
        """Test worker model targets."""
        worker_targets = PERFORMANCE_TARGETS[ModelRole.WORKER]
        assert worker_targets.p50_latency_ms == 2000.0
        assert worker_targets.p95_latency_ms == 5000.0
        assert worker_targets.p99_latency_ms == 10000.0
        assert worker_targets.tokens_per_second == 50.0
        assert worker_targets.min_accuracy == 0.85
        assert worker_targets.min_confidence == 0.7

    def test_judge_targets(self):
        """Test judge model targets."""
        judge_targets = PERFORMANCE_TARGETS[ModelRole.JUDGE]
        assert judge_targets.p50_latency_ms == 30.0
        assert judge_targets.p95_latency_ms == 50.0
        assert judge_targets.p99_latency_ms == 100.0
        assert judge_targets.tokens_per_second == 200.0
        assert judge_targets.min_accuracy == 0.90
        assert judge_targets.min_confidence == 0.75

    def test_drafter_targets(self):
        """Test drafter model targets."""
        drafter_targets = PERFORMANCE_TARGETS[ModelRole.DRAFTER]
        assert drafter_targets.p50_latency_ms == 50.0
        assert drafter_targets.p95_latency_ms == 100.0
        assert drafter_targets.p99_latency_ms == 200.0
        assert drafter_targets.tokens_per_second == 500.0
        assert drafter_targets.min_accuracy == 0.70
        assert drafter_targets.min_confidence == 0.65


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            latency_ms=100.0,
            tokens_per_second=50.0,
            memory_mb=4000.0,
            cpu_percent=60.0,
            accuracy=0.9,
            confidence=0.8,
            tokens_generated=1000,
        )

        assert metrics.latency_ms == 100.0
        assert metrics.tokens_per_second == 50.0
        assert metrics.memory_mb == 4000.0
        assert metrics.cpu_percent == 60.0
        assert metrics.accuracy == 0.9
        assert metrics.confidence == 0.8
        assert metrics.tokens_generated == 1000

    def test_performance_metrics_optional_fields(self):
        """Test PerformanceMetrics with optional fields."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            tokens_per_second=100.0,
            memory_mb=2000.0,
            cpu_percent=50.0,
        )

        assert metrics.accuracy is None
        assert metrics.confidence is None
        assert metrics.tokens_generated is None


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def test_benchmark_initialization_worker(self):
        """Test benchmark initialization for worker role."""
        benchmark = PerformanceBenchmark(ModelRole.WORKER)
        assert benchmark.model_role == ModelRole.WORKER
        assert benchmark.targets == PERFORMANCE_TARGETS[ModelRole.WORKER]

    def test_benchmark_initialization_judge(self):
        """Test benchmark initialization for judge role."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        assert benchmark.model_role == ModelRole.JUDGE
        assert benchmark.targets == PERFORMANCE_TARGETS[ModelRole.JUDGE]

    def test_benchmark_initialization_drafter(self):
        """Test benchmark initialization for drafter role."""
        benchmark = PerformanceBenchmark(ModelRole.DRAFTER)
        assert benchmark.model_role == ModelRole.DRAFTER
        assert benchmark.targets == PERFORMANCE_TARGETS[ModelRole.DRAFTER]


class TestEvaluateLatency:
    """Test evaluate_latency method."""

    def test_evaluate_latency_empty_list(self):
        """Test latency evaluation with empty list."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_latency([])

        assert result["p50"] is False
        assert result["p95"] is False
        assert result["p99"] is False

    def test_evaluate_latency_single_measurement(self):
        """Test latency evaluation with single measurement."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        latencies = [25.0]  # Below p50 target of 30ms

        result = benchmark.evaluate_latency(latencies)

        assert result["p50"] is True
        assert result["p95"] is True
        assert result["p99"] is True
        assert result["p50_value"] == 25.0
        assert result["p95_value"] == 25.0
        assert result["p99_value"] == 25.0

    def test_evaluate_latency_multiple_measurements(self):
        """Test latency evaluation with multiple measurements."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Create latencies: 20, 25, 30, 35, 40, 45, 50, 55, 60, 65
        latencies = [float(i) for i in range(20, 70, 5)]

        result = benchmark.evaluate_latency(latencies)

        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        assert "p50_value" in result
        assert "p95_value" in result
        assert "p99_value" in result

    def test_evaluate_latency_exceeds_target(self):
        """Test latency evaluation when measurements exceed targets."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # All latencies above p50 target of 30ms
        # For small lists (n <= 20), p95 uses last value; n <= 100, p99 uses last value
        # So p99 will be 100.0, which is exactly at the target (100.0 <= 100.0)
        latencies = [60.0, 70.0, 80.0, 90.0, 100.0]

        result = benchmark.evaluate_latency(latencies)

        assert result["p50"] is False  # 80.0 > 30.0
        assert result["p95"] is False  # 100.0 (last value) > 50.0
        assert result["p99"] is True  # 100.0 (last value) <= 100.0 (exactly at target)

    def test_evaluate_latency_worker_targets(self):
        """Test latency evaluation with worker targets."""
        benchmark = PerformanceBenchmark(ModelRole.WORKER)
        # Worker has higher latency targets
        # p50 is index 2 (2500.0), p95 and p99 use last value (4000.0) for small lists
        latencies = [1500.0, 2000.0, 2500.0, 3000.0, 4000.0]

        result = benchmark.evaluate_latency(latencies)

        # p50 target is 2000ms, but actual p50 (index 2) is 2500.0 > 2000.0
        assert result["p50"] is False  # 2500.0 > 2000.0
        assert result["p95"] is True  # 4000.0 (last value) <= 5000.0
        assert result["p99"] is True  # 4000.0 (last value) <= 10000.0


class TestEvaluateThroughput:
    """Test evaluate_throughput method."""

    def test_evaluate_throughput_passes(self):
        """Test throughput evaluation when target is met."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Judge target is 200 tokens/sec
        result = benchmark.evaluate_throughput(250.0)

        assert result["pass"] is True
        assert result["target"] == 200.0
        assert result["measured"] == 250.0

    def test_evaluate_throughput_fails(self):
        """Test throughput evaluation when target is not met."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Judge target is 200 tokens/sec
        result = benchmark.evaluate_throughput(150.0)

        assert result["pass"] is False
        assert result["target"] == 200.0
        assert result["measured"] == 150.0

    def test_evaluate_throughput_worker(self):
        """Test throughput evaluation for worker role."""
        benchmark = PerformanceBenchmark(ModelRole.WORKER)
        # Worker target is 50 tokens/sec
        result = benchmark.evaluate_throughput(60.0)

        assert result["pass"] is True
        assert result["target"] == 50.0

    def test_evaluate_throughput_drafter(self):
        """Test throughput evaluation for drafter role."""
        benchmark = PerformanceBenchmark(ModelRole.DRAFTER)
        # Drafter target is 500 tokens/sec
        result = benchmark.evaluate_throughput(600.0)

        assert result["pass"] is True
        assert result["target"] == 500.0


class TestEvaluateQuality:
    """Test evaluate_quality method."""

    def test_evaluate_quality_with_accuracy(self):
        """Test quality evaluation with accuracy only."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Judge min_accuracy is 0.90
        result = benchmark.evaluate_quality(accuracy=0.95)

        assert result["accuracy_pass"] is True
        assert result["accuracy_value"] == 0.95
        assert result["accuracy_target"] == 0.90

    def test_evaluate_quality_with_confidence(self):
        """Test quality evaluation with confidence only."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Judge min_confidence is 0.75
        result = benchmark.evaluate_quality(confidence=0.80)

        assert result["confidence_pass"] is True
        assert result["confidence_value"] == 0.80
        assert result["confidence_target"] == 0.75

    def test_evaluate_quality_with_both(self):
        """Test quality evaluation with both accuracy and confidence."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(accuracy=0.92, confidence=0.78)

        assert result["accuracy_pass"] is True
        assert result["confidence_pass"] is True
        assert result["accuracy_value"] == 0.92
        assert result["confidence_value"] == 0.78

    def test_evaluate_quality_fails(self):
        """Test quality evaluation when targets are not met."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(accuracy=0.85, confidence=0.70)

        assert result["accuracy_pass"] is False  # 0.85 < 0.90
        assert result["confidence_pass"] is False  # 0.70 < 0.75

    def test_evaluate_quality_none_values(self):
        """Test quality evaluation with None values."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality()

        assert len(result) == 0


class TestEvaluateResources:
    """Test evaluate_resources method."""

    def test_evaluate_resources_passes(self):
        """Test resource evaluation when targets are met."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Judge max_memory_mb is 8000, max_cpu_percent is 60
        result = benchmark.evaluate_resources(memory_mb=6000.0, cpu_percent=50.0)

        assert result["memory_pass"] is True
        assert result["cpu_pass"] is True
        assert result["memory_value"] == 6000.0
        assert result["cpu_value"] == 50.0

    def test_evaluate_resources_fails(self):
        """Test resource evaluation when targets are exceeded."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_resources(memory_mb=10000.0, cpu_percent=70.0)

        assert result["memory_pass"] is False  # 10000 > 8000
        assert result["cpu_pass"] is False  # 70 > 60

    def test_evaluate_resources_worker(self):
        """Test resource evaluation for worker role."""
        benchmark = PerformanceBenchmark(ModelRole.WORKER)
        # Worker max_memory_mb is 18000, max_cpu_percent is 80
        result = benchmark.evaluate_resources(memory_mb=15000.0, cpu_percent=75.0)

        assert result["memory_pass"] is True
        assert result["cpu_pass"] is True


class TestEvaluateAll:
    """Test evaluate_all method."""

    def test_evaluate_all_with_latencies(self):
        """Test comprehensive evaluation with latency list."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            accuracy=0.92,
            confidence=0.78,
        )
        latencies = [25.0, 30.0, 35.0, 40.0, 45.0]

        result = benchmark.evaluate_all(metrics, latencies=latencies)

        assert result["model_role"] == "judge"
        assert "targets" in result
        assert "latency" in result
        assert "throughput" in result
        assert "quality" in result
        assert "resources" in result

    def test_evaluate_all_without_latencies(self):
        """Test comprehensive evaluation without latency list."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics)

        assert "latency" in result
        assert result["latency"]["single_measurement"] == 40.0
        assert result["latency"]["pass"] is True  # 40 <= 50 (p95 target)

    def test_evaluate_all_worker(self):
        """Test comprehensive evaluation for worker role."""
        benchmark = PerformanceBenchmark(ModelRole.WORKER)
        metrics = PerformanceMetrics(
            latency_ms=3000.0,
            tokens_per_second=55.0,
            memory_mb=16000.0,
            cpu_percent=75.0,
            accuracy=0.87,
            confidence=0.72,
        )

        result = benchmark.evaluate_all(metrics)

        assert result["model_role"] == "worker"
        assert result["throughput"]["pass"] is True  # 55 >= 50
        assert result["quality"]["accuracy_pass"] is True  # 0.87 >= 0.85
        assert result["resources"]["memory_pass"] is True  # 16000 <= 18000

    def test_evaluate_all_drafter(self):
        """Test comprehensive evaluation for drafter role."""
        benchmark = PerformanceBenchmark(ModelRole.DRAFTER)
        metrics = PerformanceMetrics(
            latency_ms=80.0,
            tokens_per_second=550.0,
            memory_mb=7500.0,
            cpu_percent=65.0,
            accuracy=0.72,
            confidence=0.68,
        )

        result = benchmark.evaluate_all(metrics)

        assert result["model_role"] == "drafter"
        assert result["throughput"]["pass"] is True  # 550 >= 500
        assert result["quality"]["accuracy_pass"] is True  # 0.72 >= 0.70
        assert result["quality"]["confidence_pass"] is True  # 0.68 >= 0.65


class TestPerformanceBenchmarkIntegration:
    """Test integration of performance benchmark components."""

    def test_complete_benchmark_workflow(self):
        """Test complete benchmark evaluation workflow."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)

        # Simulate performance measurements
        latencies = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            accuracy=0.92,
            confidence=0.78,
        )

        # Evaluate all metrics
        result = benchmark.evaluate_all(metrics, latencies=latencies)

        # Verify all evaluations are present
        assert "latency" in result
        assert "throughput" in result
        assert "quality" in result
        assert "resources" in result

        # Verify targets are included
        assert "targets" in result
        assert result["targets"]["p95_latency_ms"] == 50.0
        assert result["targets"]["tokens_per_second"] == 200.0

    def test_benchmark_comparison_across_roles(self):
        """Test comparing benchmarks across different roles."""
        worker_benchmark = PerformanceBenchmark(ModelRole.WORKER)
        judge_benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        drafter_benchmark = PerformanceBenchmark(ModelRole.DRAFTER)

        # Same metrics for all
        metrics = PerformanceMetrics(
            latency_ms=100.0,
            tokens_per_second=100.0,
            memory_mb=5000.0,
            cpu_percent=50.0,
        )

        worker_result = worker_benchmark.evaluate_all(metrics)
        judge_result = judge_benchmark.evaluate_all(metrics)
        drafter_result = drafter_benchmark.evaluate_all(metrics)

        # Judge should pass latency (100 <= 50... wait, that's wrong)
        # Actually, judge p95 is 50ms, so 100ms fails
        assert judge_result["latency"]["pass"] is False

        # Worker should pass latency (100 <= 5000)
        assert worker_result["latency"]["pass"] is True

        # Throughput: 100 tokens/sec
        # Worker: 100 >= 50 -> pass
        # Judge: 100 >= 200 -> fail
        # Drafter: 100 >= 500 -> fail
        assert worker_result["throughput"]["pass"] is True
        assert judge_result["throughput"]["pass"] is False
        assert drafter_result["throughput"]["pass"] is False


class TestMeasureInferenceTime:
    """Test measure_inference_time helper function."""

    def test_measure_inference_time_basic(self):
        """Test basic inference time measurement."""
        def mock_model_fn(x):
            return x * 2

        latency_ms, output = measure_inference_time(mock_model_fn, 5)

        assert output == 10
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0

    def test_measure_inference_time_with_args(self):
        """Test inference time measurement with positional arguments."""
        def mock_model_fn(a, b):
            return a + b

        latency_ms, output = measure_inference_time(mock_model_fn, 3, 4)

        assert output == 7
        assert latency_ms >= 0

    def test_measure_inference_time_with_kwargs(self):
        """Test inference time measurement with keyword arguments."""
        def mock_model_fn(x, multiplier=2):
            return x * multiplier

        latency_ms, output = measure_inference_time(mock_model_fn, 5, multiplier=3)

        assert output == 15
        assert latency_ms >= 0

    def test_measure_inference_time_returns_ms(self):
        """Test that latency is returned in milliseconds."""
        import time

        def slow_model_fn():
            time.sleep(0.001)  # 1ms sleep
            return "result"

        latency_ms, output = measure_inference_time(slow_model_fn)

        assert output == "result"
        assert latency_ms >= 1.0  # Should be at least 1ms


class TestCalculateTokensPerSecond:
    """Test calculate_tokens_per_second helper function."""

    def test_calculate_tokens_per_second_basic(self):
        """Test basic tokens per second calculation."""
        tokens = 100
        latency_ms = 200.0  # 200ms = 0.2s

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec == 500.0  # 100 / 0.2 = 500

    def test_calculate_tokens_per_second_fast(self):
        """Test tokens per second calculation for fast inference."""
        tokens = 50
        latency_ms = 100.0  # 100ms = 0.1s

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec == 500.0  # 50 / 0.1 = 500

    def test_calculate_tokens_per_second_slow(self):
        """Test tokens per second calculation for slow inference."""
        tokens = 10
        latency_ms = 1000.0  # 1000ms = 1s

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec == 10.0  # 10 / 1 = 10

    def test_calculate_tokens_per_second_zero_latency(self):
        """Test tokens per second calculation with zero latency."""
        tokens = 100
        latency_ms = 0.0

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec == 0.0

    def test_calculate_tokens_per_second_negative_latency(self):
        """Test tokens per second calculation with negative latency."""
        tokens = 100
        latency_ms = -10.0

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec == 0.0

    def test_calculate_tokens_per_second_very_small_latency(self):
        """Test tokens per second calculation with very small latency."""
        tokens = 100
        latency_ms = 0.001  # 0.001ms

        tokens_per_sec = calculate_tokens_per_second(tokens, latency_ms)

        assert tokens_per_sec > 0
        assert tokens_per_sec == (tokens / latency_ms) * 1000.0


class TestEvaluateAllEdgeCases:
    """Test evaluate_all method edge cases."""

    def test_evaluate_all_with_zero_tokens_per_second(self):
        """Test evaluate_all with metrics with zero tokens_per_second."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=0.0,  # Zero tokens per second (falsy value)
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics)

        # Should not evaluate throughput when tokens_per_second is 0 (falsy check)
        assert "throughput" not in result

    def test_evaluate_all_without_quality_metrics(self):
        """Test evaluate_all with metrics without quality metrics."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            # No accuracy or confidence
        )

        result = benchmark.evaluate_all(metrics)

        assert "quality" in result
        assert len(result["quality"]) == 0  # Empty dict when no quality metrics

    def test_evaluate_all_overall_pass_calculation(self):
        """Test evaluate_all calculates overall_pass correctly."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # All metrics pass
        metrics = PerformanceMetrics(
            latency_ms=40.0,  # Passes (40 <= 50)
            tokens_per_second=220.0,  # Passes (220 >= 200)
            memory_mb=7000.0,  # Passes (7000 <= 8000)
            cpu_percent=55.0,  # Passes (55 <= 60)
            accuracy=0.92,  # Passes (0.92 >= 0.90)
            confidence=0.78,  # Passes (0.78 >= 0.75)
        )

        result = benchmark.evaluate_all(metrics)

        assert result["overall_pass"] is True

    def test_evaluate_all_overall_pass_fails(self):
        """Test evaluate_all calculates overall_pass as False when any metric fails."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Latency fails
        metrics = PerformanceMetrics(
            latency_ms=60.0,  # Fails (60 > 50)
            tokens_per_second=220.0,  # Passes
            memory_mb=7000.0,  # Passes
            cpu_percent=55.0,  # Passes
            accuracy=0.92,  # Passes
            confidence=0.78,  # Passes
        )

        result = benchmark.evaluate_all(metrics)

        assert result["overall_pass"] is False

    def test_evaluate_all_latency_percentile_calculation(self):
        """Test evaluate_all uses correct percentile calculation for latencies."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Create 100 latencies to test p99 calculation
        latencies = [float(i) for i in range(20, 120)]  # 20-119ms

        metrics = PerformanceMetrics(
            latency_ms=50.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics, latencies=latencies)

        assert "latency" in result
        assert "p50" in result["latency"]
        assert "p95" in result["latency"]
        assert "p99" in result["latency"]

    def test_evaluate_all_latency_small_list(self):
        """Test evaluate_all with small latency list (tests p95/p99 fallback)."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Only 5 latencies - p95 and p99 should use last value
        latencies = [25.0, 30.0, 35.0, 40.0, 45.0]

        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics, latencies=latencies)

        # p95 and p99 should use last value (45.0) when list is small
        assert result["latency"]["p95_value"] == 45.0
        assert result["latency"]["p99_value"] == 45.0


class TestLatencyEvaluationEdgeCases:
    """Test edge cases in latency evaluation."""

    def test_evaluate_latency_empty_list(self):
        """Test evaluate_latency with empty list."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_latency([])

        assert result["p50"] is False
        assert result["p95"] is False
        assert result["p99"] is False

    def test_evaluate_latency_single_value(self):
        """Test evaluate_latency with single value."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_latency([40.0])

        # Single value should be used for all percentiles
        assert result["p50_value"] == 40.0
        assert result["p95_value"] == 40.0
        assert result["p99_value"] == 40.0

    def test_evaluate_latency_two_values(self):
        """Test evaluate_latency with two values."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_latency([30.0, 50.0])

        # Should use sorted values
        # n = 2, int(n * 0.50) = int(1.0) = 1
        # sorted_latencies = [30.0, 50.0]
        # sorted_latencies[1] = 50.0
        # So p50 should be 50.0
        assert result["p50_value"] == 50.0
        # p95 and p99 should use last value (50.0) when list is small (n <= 20)
        assert result["p95_value"] == 50.0
        assert result["p99_value"] == 50.0

    def test_evaluate_latency_exactly_at_threshold(self):
        """Test evaluate_latency when latency is exactly at threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # p95 threshold is 50.0ms
        result = benchmark.evaluate_latency([50.0] * 100)

        # Should pass (<= threshold)
        assert result["p95"] is True
        assert result["p95_value"] == 50.0

    def test_evaluate_latency_just_below_threshold(self):
        """Test evaluate_latency when latency is just below threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # p95 threshold is 50.0ms
        result = benchmark.evaluate_latency([49.99] * 100)

        # Should pass (< threshold)
        assert result["p95"] is True

    def test_evaluate_latency_just_above_threshold(self):
        """Test evaluate_latency when latency is just above threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # p95 threshold is 50.0ms
        result = benchmark.evaluate_latency([50.01] * 100)

        # Should fail (> threshold)
        assert result["p95"] is False

    def test_evaluate_latency_boundary_percentile_calculation(self):
        """Test percentile calculation at boundaries."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Create exactly 100 values
        latencies = [float(i) for i in range(100)]

        result = benchmark.evaluate_latency(latencies)

        # p50 should be at index 50
        assert result["p50_value"] == 50.0
        # p95 should be at index 95
        assert result["p95_value"] == 95.0
        # p99 should be at index 99
        assert result["p99_value"] == 99.0

    def test_evaluate_latency_large_list(self):
        """Test evaluate_latency with large list (1000+ values)."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Create 1000 latencies
        latencies = [float(i) for i in range(1000)]

        result = benchmark.evaluate_latency(latencies)

        # Should calculate percentiles correctly
        assert result["p50_value"] == 500.0
        assert result["p95_value"] == 950.0
        assert result["p99_value"] == 990.0

    def test_evaluate_latency_unsorted_list(self):
        """Test evaluate_latency with unsorted list (should sort internally)."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Unsorted latencies
        latencies = [50.0, 20.0, 80.0, 10.0, 90.0]

        result = benchmark.evaluate_latency(latencies)

        # Should sort internally and calculate correctly
        assert result["p50_value"] == 50.0
        assert result["p95_value"] == 90.0
        assert result["p99_value"] == 90.0

    def test_evaluate_latency_duplicate_values(self):
        """Test evaluate_latency with duplicate values."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # All values are the same
        latencies = [40.0] * 100

        result = benchmark.evaluate_latency(latencies)

        # All percentiles should be the same
        assert result["p50_value"] == 40.0
        assert result["p95_value"] == 40.0
        assert result["p99_value"] == 40.0

    def test_evaluate_latency_all_three_roles(self):
        """Test evaluate_latency for all three model roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)
            latencies = [float(i) for i in range(100)]

            result = benchmark.evaluate_latency(latencies)

            # Should return results for all roles
            assert "p50" in result
            assert "p95" in result
            assert "p99" in result


class TestThroughputEvaluationEdgeCases:
    """Test edge cases in throughput evaluation."""

    def test_evaluate_throughput_zero_tokens_per_second(self):
        """Test evaluate_throughput with zero tokens per second."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_throughput(0.0)

        # Should fail (0 < target of 200)
        assert result["pass"] is False
        assert result["measured"] == 0.0

    def test_evaluate_throughput_exactly_at_threshold(self):
        """Test evaluate_throughput when exactly at threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Target is 200.0 tokens/sec
        result = benchmark.evaluate_throughput(200.0)

        # Should pass (>= threshold)
        assert result["pass"] is True
        assert result["measured"] == 200.0

    def test_evaluate_throughput_just_below_threshold(self):
        """Test evaluate_throughput when just below threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Target is 200.0 tokens/sec
        result = benchmark.evaluate_throughput(199.99)

        # Should fail (< threshold)
        assert result["pass"] is False

    def test_evaluate_throughput_just_above_threshold(self):
        """Test evaluate_throughput when just above threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # Target is 200.0 tokens/sec
        result = benchmark.evaluate_throughput(200.01)

        # Should pass (> threshold)
        assert result["pass"] is True

    def test_evaluate_throughput_very_high_value(self):
        """Test evaluate_throughput with very high value."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_throughput(1000.0)

        # Should pass (well above threshold)
        assert result["pass"] is True
        assert result["measured"] == 1000.0

    def test_evaluate_throughput_all_three_roles(self):
        """Test evaluate_throughput for all three model roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)
            # Use target value for each role
            target = PERFORMANCE_TARGETS[role].tokens_per_second

            result = benchmark.evaluate_throughput(target)

            # Should pass at threshold
            assert result["pass"] is True
            assert result["measured"] == target
            assert result["target"] == target


class TestQualityEvaluationEdgeCases:
    """Test edge cases in quality evaluation."""

    def test_evaluate_quality_no_metrics(self):
        """Test evaluate_quality with no metrics."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality()

        # Should return empty dict
        assert result == {}

    def test_evaluate_quality_only_accuracy(self):
        """Test evaluate_quality with only accuracy."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(accuracy=0.92)

        # Should have accuracy metrics only
        assert "accuracy_pass" in result
        assert "accuracy_value" in result
        assert "accuracy_target" in result
        assert "confidence_pass" not in result

    def test_evaluate_quality_only_confidence(self):
        """Test evaluate_quality with only confidence."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(confidence=0.78)

        # Should have confidence metrics only
        assert "confidence_pass" in result
        assert "confidence_value" in result
        assert "confidence_target" in result
        assert "accuracy_pass" not in result

    def test_evaluate_quality_exactly_at_threshold(self):
        """Test evaluate_quality when exactly at threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # min_accuracy is 0.90, min_confidence is 0.75
        result = benchmark.evaluate_quality(accuracy=0.90, confidence=0.75)

        # Should pass (>= threshold)
        assert result["accuracy_pass"] is True
        assert result["confidence_pass"] is True

    def test_evaluate_quality_just_below_threshold(self):
        """Test evaluate_quality when just below threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # min_accuracy is 0.90, min_confidence is 0.75
        result = benchmark.evaluate_quality(accuracy=0.8999, confidence=0.7499)

        # Should fail (< threshold)
        assert result["accuracy_pass"] is False
        assert result["confidence_pass"] is False

    def test_evaluate_quality_just_above_threshold(self):
        """Test evaluate_quality when just above threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # min_accuracy is 0.90, min_confidence is 0.75
        result = benchmark.evaluate_quality(accuracy=0.9001, confidence=0.7501)

        # Should pass (> threshold)
        assert result["accuracy_pass"] is True
        assert result["confidence_pass"] is True

    def test_evaluate_quality_zero_accuracy(self):
        """Test evaluate_quality with zero accuracy."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(accuracy=0.0)

        # Should fail (0 < min_accuracy of 0.90)
        assert result["accuracy_pass"] is False

    def test_evaluate_quality_one_accuracy(self):
        """Test evaluate_quality with accuracy of 1.0."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_quality(accuracy=1.0)

        # Should pass (1.0 >= min_accuracy of 0.90)
        assert result["accuracy_pass"] is True

    def test_evaluate_quality_all_three_roles(self):
        """Test evaluate_quality for all three model roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)
            targets = PERFORMANCE_TARGETS[role]

            # Test at threshold
            result = benchmark.evaluate_quality(
                accuracy=targets.min_accuracy, confidence=targets.min_confidence
            )

            # Should pass at threshold
            assert result["accuracy_pass"] is True
            assert result["confidence_pass"] is True


class TestResourceEvaluationEdgeCases:
    """Test edge cases in resource evaluation."""

    def test_evaluate_resources_zero_memory(self):
        """Test evaluate_resources with zero memory."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_resources(memory_mb=0.0, cpu_percent=55.0)

        # Should pass (0 <= max_memory_mb of 8000)
        assert result["memory_pass"] is True
        assert result["memory_value"] == 0.0

    def test_evaluate_resources_zero_cpu(self):
        """Test evaluate_resources with zero CPU."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_resources(memory_mb=7000.0, cpu_percent=0.0)

        # Should pass (0 <= max_cpu_percent of 60)
        assert result["cpu_pass"] is True
        assert result["cpu_value"] == 0.0

    def test_evaluate_resources_exactly_at_threshold(self):
        """Test evaluate_resources when exactly at threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # max_memory_mb is 8000.0, max_cpu_percent is 60.0
        result = benchmark.evaluate_resources(memory_mb=8000.0, cpu_percent=60.0)

        # Should pass (<= threshold)
        assert result["memory_pass"] is True
        assert result["cpu_pass"] is True

    def test_evaluate_resources_just_below_threshold(self):
        """Test evaluate_resources when just below threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # max_memory_mb is 8000.0, max_cpu_percent is 60.0
        result = benchmark.evaluate_resources(memory_mb=7999.99, cpu_percent=59.99)

        # Should pass (< threshold)
        assert result["memory_pass"] is True
        assert result["cpu_pass"] is True

    def test_evaluate_resources_just_above_threshold(self):
        """Test evaluate_resources when just above threshold."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        # max_memory_mb is 8000.0, max_cpu_percent is 60.0
        result = benchmark.evaluate_resources(memory_mb=8000.01, cpu_percent=60.01)

        # Should fail (> threshold)
        assert result["memory_pass"] is False
        assert result["cpu_pass"] is False

    def test_evaluate_resources_very_high_values(self):
        """Test evaluate_resources with very high values."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        result = benchmark.evaluate_resources(memory_mb=100000.0, cpu_percent=100.0)

        # Should fail (well above thresholds)
        assert result["memory_pass"] is False
        assert result["cpu_pass"] is False

    def test_evaluate_resources_all_three_roles(self):
        """Test evaluate_resources for all three model roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)
            targets = PERFORMANCE_TARGETS[role]

            # Test at threshold
            result = benchmark.evaluate_resources(
                memory_mb=targets.max_memory_mb, cpu_percent=targets.max_cpu_percent
            )

            # Should pass at threshold
            assert result["memory_pass"] is True
            assert result["cpu_pass"] is True


class TestEvaluateAllEdgeCasesExtended:
    """Test additional edge cases in evaluate_all method."""

    def test_evaluate_all_with_none_latency_ms(self):
        """Test evaluate_all with None latency_ms (falsy value)."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=0.0,  # Falsy value
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics)

        # Should not evaluate latency when latency_ms is falsy
        assert "latency" not in result

    def test_evaluate_all_with_single_latency_measurement(self):
        """Test evaluate_all with single latency measurement (no latencies list)."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=45.0,  # Single measurement
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics)

        # Should use single_measurement format
        assert "latency" in result
        assert "single_measurement" in result["latency"]
        assert result["latency"]["single_measurement"] == 45.0
        assert result["latency"]["pass"] is True  # 45.0 <= 50.0

    def test_evaluate_all_overall_pass_with_none_quality(self):
        """Test evaluate_all overall_pass when quality metrics are None."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            # No accuracy or confidence
        )

        result = benchmark.evaluate_all(metrics)

        # Quality should be empty, but overall_pass should still work
        assert result["quality"] == {}
        # overall_pass should be True (None quality metrics are treated as passing)
        assert result["overall_pass"] is True

    def test_evaluate_all_overall_pass_fails_on_memory(self):
        """Test evaluate_all overall_pass fails when memory fails."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=9000.0,  # Fails (9000 > 8000)
            cpu_percent=55.0,
        )

        result = benchmark.evaluate_all(metrics)

        assert result["resources"]["memory_pass"] is False
        assert result["overall_pass"] is False

    def test_evaluate_all_overall_pass_fails_on_cpu(self):
        """Test evaluate_all overall_pass fails when CPU fails."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=65.0,  # Fails (65 > 60)
        )

        result = benchmark.evaluate_all(metrics)

        assert result["resources"]["cpu_pass"] is False
        assert result["overall_pass"] is False

    def test_evaluate_all_overall_pass_fails_on_accuracy(self):
        """Test evaluate_all overall_pass fails when accuracy fails."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            accuracy=0.85,  # Fails (0.85 < 0.90)
            confidence=0.78,
        )

        result = benchmark.evaluate_all(metrics)

        assert result["quality"]["accuracy_pass"] is False
        assert result["overall_pass"] is False

    def test_evaluate_all_overall_pass_fails_on_confidence(self):
        """Test evaluate_all overall_pass fails when confidence fails."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)
        metrics = PerformanceMetrics(
            latency_ms=40.0,
            tokens_per_second=220.0,
            memory_mb=7000.0,
            cpu_percent=55.0,
            accuracy=0.92,
            confidence=0.70,  # Fails (0.70 < 0.75)
        )

        result = benchmark.evaluate_all(metrics)

        assert result["quality"]["confidence_pass"] is False
        assert result["overall_pass"] is False

    def test_evaluate_all_all_three_roles(self):
        """Test evaluate_all for all three model roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)
            targets = PERFORMANCE_TARGETS[role]

            metrics = PerformanceMetrics(
                latency_ms=targets.p95_latency_ms,
                tokens_per_second=targets.tokens_per_second,
                memory_mb=targets.max_memory_mb,
                cpu_percent=targets.max_cpu_percent,
                accuracy=targets.min_accuracy,
                confidence=targets.min_confidence,
            )

            result = benchmark.evaluate_all(metrics)

            # Should pass at thresholds
            assert result["overall_pass"] is True
            assert result["model_role"] == role.value


class TestMeasureInferenceTimeEdgeCases:
    """Test edge cases in measure_inference_time function."""

    def test_measure_inference_time_immediate_return(self):
        """Test measure_inference_time with function that returns immediately."""
        def immediate_fn():
            return "result"

        latency_ms, output = measure_inference_time(immediate_fn)

        # Should measure very small latency
        assert latency_ms >= 0.0
        assert latency_ms < 1000.0  # Should be less than 1 second
        assert output == "result"

    def test_measure_inference_time_with_args(self):
        """Test measure_inference_time with function that takes args."""
        def add_fn(a, b):
            return a + b

        latency_ms, output = measure_inference_time(add_fn, 1, 2)

        assert output == 3
        assert latency_ms >= 0.0

    def test_measure_inference_time_with_kwargs(self):
        """Test measure_inference_time with function that takes kwargs."""
        def multiply_fn(a, b):
            return a * b

        latency_ms, output = measure_inference_time(multiply_fn, a=3, b=4)

        assert output == 12
        assert latency_ms >= 0.0

    def test_measure_inference_time_with_args_and_kwargs(self):
        """Test measure_inference_time with function that takes args and kwargs."""
        def combine_fn(a, b, c):
            return a + b + c

        latency_ms, output = measure_inference_time(combine_fn, 1, 2, c=3)

        assert output == 6
        assert latency_ms >= 0.0

    def test_measure_inference_time_returns_none(self):
        """Test measure_inference_time with function that returns None."""
        def none_fn():
            return None

        latency_ms, output = measure_inference_time(none_fn)

        assert output is None
        assert latency_ms >= 0.0

    def test_measure_inference_time_raises_exception(self):
        """Test measure_inference_time with function that raises exception."""
        def error_fn():
            raise ValueError("Test error")

        # Should propagate exception
        with pytest.raises(ValueError, match="Test error"):
            measure_inference_time(error_fn)


class TestCalculateTokensPerSecondEdgeCases:
    """Test edge cases in calculate_tokens_per_second function."""

    def test_calculate_tokens_per_second_zero_latency(self):
        """Test calculate_tokens_per_second with zero latency."""
        result = calculate_tokens_per_second(tokens_generated=100, latency_ms=0.0)

        # Should return 0.0 (division by zero protection)
        assert result == 0.0

    def test_calculate_tokens_per_second_negative_latency(self):
        """Test calculate_tokens_per_second with negative latency."""
        result = calculate_tokens_per_second(tokens_generated=100, latency_ms=-1.0)

        # Should return 0.0 (negative latency protection)
        assert result == 0.0

    def test_calculate_tokens_per_second_zero_tokens(self):
        """Test calculate_tokens_per_second with zero tokens."""
        result = calculate_tokens_per_second(tokens_generated=0, latency_ms=1000.0)

        # Should return 0.0 (0 tokens / 1000ms = 0 tokens/sec)
        assert result == 0.0

    def test_calculate_tokens_per_second_normal_case(self):
        """Test calculate_tokens_per_second with normal values."""
        result = calculate_tokens_per_second(tokens_generated=100, latency_ms=1000.0)

        # 100 tokens / 1000ms = 0.1 tokens/ms = 100 tokens/sec
        assert result == 100.0

    def test_calculate_tokens_per_second_high_throughput(self):
        """Test calculate_tokens_per_second with high throughput."""
        result = calculate_tokens_per_second(tokens_generated=200, latency_ms=1000.0)

        # 200 tokens / 1000ms = 200 tokens/sec
        assert result == 200.0

    def test_calculate_tokens_per_second_low_latency(self):
        """Test calculate_tokens_per_second with low latency."""
        result = calculate_tokens_per_second(tokens_generated=100, latency_ms=100.0)

        # 100 tokens / 100ms = 1 token/ms = 1000 tokens/sec
        assert result == 1000.0

    def test_calculate_tokens_per_second_fractional_latency(self):
        """Test calculate_tokens_per_second with fractional latency."""
        result = calculate_tokens_per_second(tokens_generated=100, latency_ms=500.5)

        # 100 tokens / 500.5ms ≈ 0.1998 tokens/ms ≈ 199.8 tokens/sec
        assert result == pytest.approx(199.8, abs=0.1)

    def test_calculate_tokens_per_second_large_tokens(self):
        """Test calculate_tokens_per_second with large token count."""
        result = calculate_tokens_per_second(tokens_generated=10000, latency_ms=1000.0)

        # 10000 tokens / 1000ms = 10000 tokens/sec
        assert result == 10000.0


class TestPerformanceBenchmarkInitialization:
    """Test PerformanceBenchmark initialization edge cases."""

    def test_performance_benchmark_all_roles(self):
        """Test PerformanceBenchmark initialization for all roles."""
        for role in ModelRole:
            benchmark = PerformanceBenchmark(role)

            assert benchmark.model_role == role
            assert benchmark.targets == PERFORMANCE_TARGETS[role]

    def test_performance_benchmark_targets_access(self):
        """Test that PerformanceBenchmark can access all target fields."""
        benchmark = PerformanceBenchmark(ModelRole.JUDGE)

        assert benchmark.targets.p50_latency_ms == 30.0
        assert benchmark.targets.p95_latency_ms == 50.0
        assert benchmark.targets.p99_latency_ms == 100.0
        assert benchmark.targets.tokens_per_second == 200.0
        assert benchmark.targets.min_accuracy == 0.90
        assert benchmark.targets.min_confidence == 0.75
        assert benchmark.targets.max_memory_mb == 8000.0
        assert benchmark.targets.max_cpu_percent == 60.0

