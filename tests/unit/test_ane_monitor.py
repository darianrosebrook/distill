"""
Unit tests for ANE residency monitoring.

Tests ANE residency measurement and gating for M-series Apple Silicon optimization.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

try:
    from coreml.runtime.ane_monitor import ANEResidencyMonitor, measure_model_residency

    ANE_MONITOR_AVAILABLE = True
except ImportError:
    ANE_MONITOR_AVAILABLE = False


@pytest.mark.skipif(not ANE_MONITOR_AVAILABLE, reason="ANE monitor not available")
class TestANEResidencyMonitor:
    """Tests for ANEResidencyMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ANEResidencyMonitor(model_mlpackage_path="test.mlpackage")

        assert monitor.model_path == "test.mlpackage"
        assert len(monitor.samples) == 0

    def test_measure_wall_clock(self):
        """Test wall-clock residency measurement."""
        monitor = ANEResidencyMonitor()

        # Create a fast, consistent inference function (simulates ANE)
        def fast_inference():
            time.sleep(0.001)  # 1ms - fast and consistent

        residency = monitor._measure_wall_clock(fast_inference, num_samples=10)

        assert "ane_time_pct" in residency
        assert "gpu_time_pct" in residency
        assert "cpu_time_pct" in residency
        assert "total_samples" in residency
        assert residency["total_samples"] == 10
        assert 0.0 <= residency["ane_time_pct"] <= 1.0
        assert 0.0 <= residency["gpu_time_pct"] <= 1.0
        assert 0.0 <= residency["cpu_time_pct"] <= 1.0

    def test_measure_residency(self):
        """Test full residency measurement."""
        monitor = ANEResidencyMonitor()

        def inference_fn():
            time.sleep(0.001)

        residency = monitor.measure_residency(inference_fn, num_samples=10, warmup_samples=2)

        assert "ane_time_pct" in residency
        assert "total_samples" in residency
        assert residency["total_samples"] == 10

    def test_check_residency_threshold_pass(self):
        """Test threshold check passing."""
        monitor = ANEResidencyMonitor()

        residency = {
            "ane_time_pct": 0.85,
            "gpu_time_pct": 0.10,
            "cpu_time_pct": 0.05,
        }

        passed, message = monitor.check_residency_threshold(residency, min_ane_pct=0.80)

        assert passed is True
        assert "meets threshold" in message.lower()

    def test_check_residency_threshold_fail(self):
        """Test threshold check failing."""
        monitor = ANEResidencyMonitor()

        residency = {
            "ane_time_pct": 0.70,
            "gpu_time_pct": 0.20,
            "cpu_time_pct": 0.10,
        }

        passed, message = monitor.check_residency_threshold(residency, min_ane_pct=0.80)

        assert passed is False
        assert "below threshold" in message.lower()

    def test_compare_with_baseline_no_regression(self):
        """Test baseline comparison with no regression."""
        monitor = ANEResidencyMonitor()

        current = {"ane_time_pct": 0.85}
        baseline = {"ane_time_pct": 0.90}

        passed, message = monitor.compare_with_baseline(current, baseline, max_regression_pct=0.10)

        # Regression is 5.6% (5% / 90%), which is < 10%
        assert passed is True
        assert "within limit" in message.lower()

    def test_compare_with_baseline_regression(self):
        """Test baseline comparison with regression."""
        monitor = ANEResidencyMonitor()

        current = {"ane_time_pct": 0.70}
        baseline = {"ane_time_pct": 0.90}

        passed, message = monitor.compare_with_baseline(current, baseline, max_regression_pct=0.10)

        # Regression is 22% (20% / 90%), which is > 10%
        assert passed is False
        assert "exceeds limit" in message.lower()

    def test_get_model_ops_info_no_model(self):
        """Test getting ops info without model."""
        monitor = ANEResidencyMonitor()

        info = monitor.get_model_ops_info()

        assert "error" in info

    @patch("coreml.runtime.ane_monitor.COREML_AVAILABLE", True)
    def test_get_model_ops_info_with_model(self):
        """Test getting ops info with model."""
        # Mock CoreML model
        mock_spec = Mock()
        mock_spec.WhichOneof.return_value = "mlProgram"
        mock_spec.mlProgram.functions = {
            "main": Mock(
                block=Mock(
                    operations=[
                        Mock(type="matmul"),
                        Mock(type="add"),
                        Mock(type="layernorm"),
                    ]
                )
            )
        }

        mock_model = Mock()
        mock_model.get_spec.return_value = mock_spec
        mock_ct = Mock()
        mock_ct.models.MLModel.return_value = mock_model

        monitor = ANEResidencyMonitor(model_mlpackage_path="test.mlpackage")

        # Patch coremltools.models.MLModel where it's actually used
        with patch("coremltools.models.MLModel", return_value=mock_model):
            info = monitor.get_model_ops_info()

        assert "total_ops" in info
        assert "ane_friendly_ops" in info
        assert "is_mlprogram" in info


@pytest.mark.skipif(not ANE_MONITOR_AVAILABLE, reason="ANE monitor not available")
class TestMeasureModelResidency:
    """Tests for measure_model_residency convenience function."""

    def test_measure_model_residency(self):
        """Test measuring residency for a model."""
        # Mock model and adapter
        mock_model = Mock()
        mock_adapter = Mock()

        def prepare_state(prompt):
            return {}

        def first_step(model, prompt, state):
            return np.random.randn(32000), {}

        def next_step(model, token, state):
            return np.random.randn(32000), {}

        mock_adapter.prepare_state = prepare_state
        mock_adapter.first_step = first_step
        mock_adapter.next_step = next_step

        prompts = [
            np.array([[1, 2, 3]], dtype=np.int32),
            np.array([[4, 5, 6]], dtype=np.int32),
        ]

        residency = measure_model_residency(
            model=mock_model,
            adapter=mock_adapter,
            prompts=prompts,
            num_samples=5,
        )

        assert "ane_time_pct" in residency
        assert "total_samples" in residency
        assert residency["total_samples"] == 5
