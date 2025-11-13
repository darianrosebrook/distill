"""
Unit tests for batch policy enforcement.

Tests automatic workload-aware batch size selection for M-series Apple Silicon optimization.
"""
import pytest

try:
    from coreml.runtime.batch_policy import (
        BatchPolicy,
        BatchPolicyConfig,
        create_batch_policy,
    )
    BATCH_POLICY_AVAILABLE = True
except ImportError:
    BATCH_POLICY_AVAILABLE = False


@pytest.mark.skipif(not BATCH_POLICY_AVAILABLE, reason="Batch policy not available")
class TestBatchPolicyConfig:
    """Tests for BatchPolicyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BatchPolicyConfig()

        assert config.interactive_default == 1
        assert config.offline_allowed == [2, 4]
        assert config.tps_improvement_threshold == 0.10
        assert config.latency_penalty_threshold == 0.10

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchPolicyConfig(
            interactive_default=1,
            offline_allowed=[2, 3, 4],
            tps_improvement_threshold=0.15,
            latency_penalty_threshold=0.05,
        )

        assert config.interactive_default == 1
        assert config.offline_allowed == [2, 3, 4]
        assert config.tps_improvement_threshold == 0.15
        assert config.latency_penalty_threshold == 0.05


@pytest.mark.skipif(not BATCH_POLICY_AVAILABLE, reason="Batch policy not available")
class TestBatchPolicy:
    """Tests for BatchPolicy class."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        policy = BatchPolicy()

        assert policy.config.interactive_default == 1
        assert policy.config.offline_allowed == [2, 4]

    def test_initialization_with_hardware_profile(self):
        """Test initialization with hardware profile."""
        hardware_profile = {
            "key": "m1-max-64g",
            "config": {
                "batch_policy": {
                    "interactive_default": 1,
                    "offline_allowed": [2, 4],
                }
            }
        }

        policy = BatchPolicy(hardware_profile=hardware_profile)

        assert policy.config.interactive_default == 1
        assert policy.config.offline_allowed == [2, 4]

    def test_select_batch_size_interactive(self):
        """Test batch size selection for interactive workload."""
        policy = BatchPolicy()

        batch_size = policy.select_batch_size(workload_type="interactive")

        assert batch_size == 1

    def test_select_batch_size_offline(self):
        """Test batch size selection for offline workload."""
        policy = BatchPolicy()

        batch_size = policy.select_batch_size(workload_type="offline")

        # Should return first allowed batch size (default optimization)
        assert batch_size in [2, 4]

    def test_select_batch_size_force(self):
        """Test forced batch size override."""
        policy = BatchPolicy()

        batch_size = policy.select_batch_size(
            workload_type="interactive",
            force_batch_size=4
        )

        assert batch_size == 4

    def test_optimize_from_benchmarks(self):
        """Test optimization from benchmark results."""
        policy = BatchPolicy()

        benchmark_results = {
            1: {"tps": 30.0, "p95_latency_ms": 500.0},
            # 16.7% TPS improvement, 4% latency penalty
            2: {"tps": 35.0, "p95_latency_ms": 520.0},
            # 33% TPS improvement, 20% latency penalty (too high)
            4: {"tps": 40.0, "p95_latency_ms": 600.0},
        }

        optimal = policy.optimize_from_benchmarks(benchmark_results)

        # Should select batch=2 (meets thresholds)
        assert optimal == 2

    def test_optimize_from_benchmarks_no_improvement(self):
        """Test optimization when no batch size meets thresholds."""
        policy = BatchPolicy()

        benchmark_results = {
            1: {"tps": 30.0, "p95_latency_ms": 500.0},
            # Only 3% TPS improvement, 20% latency penalty
            2: {"tps": 31.0, "p95_latency_ms": 600.0},
        }

        optimal = policy.optimize_from_benchmarks(benchmark_results)

        # Should default to batch=1 (no improvement meets thresholds)
        assert optimal == 1

    def test_should_use_batch_interactive(self):
        """Test batch size validation for interactive workload."""
        policy = BatchPolicy()

        allowed, reason = policy.should_use_batch(
            1, workload_type="interactive")
        assert allowed is True
        assert "interactive" in reason.lower()

        allowed, reason = policy.should_use_batch(
            2, workload_type="interactive")
        assert allowed is False
        assert "must use batch=1" in reason.lower()

    def test_should_use_batch_offline(self):
        """Test batch size validation for offline workload."""
        policy = BatchPolicy()

        allowed, reason = policy.should_use_batch(2, workload_type="offline")
        assert allowed is True
        assert "offline" in reason.lower()

        allowed, reason = policy.should_use_batch(5, workload_type="offline")
        assert allowed is False
        assert "not in allowed list" in reason.lower()

    def test_get_policy_summary(self):
        """Test policy summary generation."""
        policy = BatchPolicy()

        summary = policy.get_policy_summary()

        assert "interactive_batch" in summary
        assert "offline_allowed" in summary
        assert "offline_optimal" in summary
        assert summary["interactive_batch"] == 1


@pytest.mark.skipif(not BATCH_POLICY_AVAILABLE, reason="Batch policy not available")
class TestCreateBatchPolicy:
    """Tests for create_batch_policy convenience function."""

    def test_create_batch_policy(self):
        """Test creating batch policy."""
        policy = create_batch_policy(
            interactive_default=1,
            offline_allowed=[2, 4],
        )

        assert isinstance(policy, BatchPolicy)
        assert policy.config.interactive_default == 1
        assert policy.config.offline_allowed == [2, 4]

    def test_create_batch_policy_with_profile(self):
        """Test creating batch policy with hardware profile."""
        hardware_profile = {
            "key": "m1-max-64g",
            "config": {
                "batch_policy": {
                    "interactive_default": 1,
                    "offline_allowed": [2, 4],
                }
            }
        }

        policy = create_batch_policy(hardware_profile=hardware_profile)

        assert isinstance(policy, BatchPolicy)
        assert policy.hardware_profile == hardware_profile
