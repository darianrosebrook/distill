"""
Tests for evaluation/performance_benchmarks.py - Performance benchmarks for arbiter stack models.

Tests performance targets, metrics evaluation, and benchmark functionality.
"""
# @author: @darianrosebrook


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

