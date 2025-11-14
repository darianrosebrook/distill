"""
Tests for training/performance_monitor.py - Performance monitoring and profiling utilities.

Tests memory usage tracking, timing, and performance metrics.
"""
# @author: @darianrosebrook

import pytest
import time
from unittest.mock import patch
from training.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    time_operation,
    TrainingProfiler,
    profile_memory_usage,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            wall_time_seconds=10.5,
            cpu_percent=50.0,
            memory_mb=1024.0,
            gpu_memory_mb=2048.0,
            tokens_processed=1000,
            throughput_tokens_per_sec=100.0,
        )
        assert metrics.wall_time_seconds == 10.5
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_mb == 1024.0
        assert metrics.gpu_memory_mb == 2048.0
        assert metrics.tokens_processed == 1000
        assert metrics.throughput_tokens_per_sec == 100.0

    def test_performance_metrics_optional_fields(self):
        """Test PerformanceMetrics with optional fields."""
        metrics = PerformanceMetrics(
            wall_time_seconds=5.0,
            cpu_percent=25.0,
            memory_mb=512.0,
        )
        assert metrics.gpu_memory_mb is None
        assert metrics.tokens_processed is None
        assert metrics.throughput_tokens_per_sec is None


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.process is not None
        assert monitor.start_time > 0

    def test_get_metrics_basic(self):
        """Test getting basic metrics."""
        monitor = PerformanceMonitor()
        time.sleep(0.1)  # Small delay to ensure time passes
        metrics = monitor.get_metrics()

        assert metrics.wall_time_seconds > 0
        assert metrics.cpu_percent >= 0
        assert metrics.memory_mb > 0

    def test_get_metrics_with_tokens(self):
        """Test getting metrics with tokens processed."""
        monitor = PerformanceMonitor()
        time.sleep(0.1)
        metrics = monitor.get_metrics(tokens_processed=1000)

        assert metrics.tokens_processed == 1000
        assert metrics.throughput_tokens_per_sec is not None
        assert metrics.throughput_tokens_per_sec > 0

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    def test_get_metrics_with_gpu(self, mock_memory, mock_available):
        """Test getting metrics with GPU available."""
        mock_available.return_value = True
        mock_memory.return_value = 1024 * 1024 * 1024  # 1GB in bytes

        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()

        assert metrics.gpu_memory_mb is not None
        assert metrics.gpu_memory_mb > 0

    @patch("torch.cuda.is_available")
    def test_get_metrics_no_gpu(self, mock_available):
        """Test getting metrics without GPU."""
        mock_available.return_value = False

        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()

        assert metrics.gpu_memory_mb is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    def test_get_metrics_gpu_exception(self, mock_memory, mock_available):
        """Test getting metrics when GPU memory allocation raises exception."""
        mock_available.return_value = True
        mock_memory.side_effect = RuntimeError("CUDA error")

        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()

        # Should handle exception gracefully and return None
        assert metrics.gpu_memory_mb is None

    def test_reset(self):
        """Test resetting performance monitor."""
        monitor = PerformanceMonitor()
        initial_time = monitor.start_time

        time.sleep(0.1)
        monitor.reset()

        assert monitor.start_time > initial_time


class TestTimeOperation:
    """Test time_operation context manager."""

    def test_time_operation_basic(self, capsys):
        """Test timing a basic operation."""
        with time_operation("test_operation"):
            time.sleep(0.1)

        captured = capsys.readouterr()
        assert "[PERF]" in captured.out
        assert "test_operation" in captured.out
        assert "took" in captured.out

    def test_time_operation_with_exception(self, capsys):
        """Test timing operation that raises exception."""
        with pytest.raises(ValueError):
            with time_operation("failing_operation"):
                raise ValueError("Test error")

        captured = capsys.readouterr()
        assert "[PERF]" in captured.out
        assert "failing_operation" in captured.out


class TestTrainingProfiler:
    """Test TrainingProfiler class."""

    @pytest.fixture
    def profiler(self, tmp_path):
        """Create a TrainingProfiler instance."""
        return TrainingProfiler(log_dir=tmp_path)

    def test_training_profiler_initialization(self, tmp_path):
        """Test TrainingProfiler initialization."""
        profiler = TrainingProfiler(log_dir=tmp_path)
        assert profiler.log_dir == tmp_path
        assert profiler.monitor is not None
        assert len(profiler.step_metrics) == 0

    def test_training_profiler_no_log_dir(self):
        """Test TrainingProfiler without log directory."""
        profiler = TrainingProfiler()
        assert profiler.log_dir is None

    def test_log_step(self, profiler, capsys):
        """Test logging a training step."""
        profiler.log_step(step=1, loss=0.5, lr=1e-4, tokens_processed=1000)

        assert len(profiler.step_metrics) == 1
        step_data = profiler.step_metrics[0]
        assert step_data["step"] == 1
        assert step_data["loss"] == 0.5
        assert step_data["learning_rate"] == 1e-4

        captured = capsys.readouterr()
        assert "[STEP 1]" in captured.out
        assert "Loss: 0.5000" in captured.out

    def test_log_step_multiple(self, profiler):
        """Test logging multiple steps."""
        for i in range(5):
            profiler.log_step(step=i, loss=0.5 - i * 0.1, lr=1e-4)

        assert len(profiler.step_metrics) == 5

    def test_log_step_without_tokens(self, profiler, capsys):
        """Test logging step without tokens processed."""
        profiler.log_step(step=1, loss=0.5, lr=1e-4)

        captured = capsys.readouterr()
        assert "N/A" in captured.out or "tok/s" in captured.out

    def test_save_profile(self, profiler, tmp_path):
        """Test saving profiling data."""
        profiler.log_step(step=1, loss=0.5, lr=1e-4)
        profiler.log_step(step=2, loss=0.4, lr=1e-4)

        output_path = tmp_path / "profile.json"
        profiler.save_profile(output_path)

        assert output_path.exists()
        import json

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "total_training_time" in data
        assert "total_steps" in data
        assert "step_metrics" in data
        assert data["total_steps"] == 2

    def test_save_profile_default_path(self, profiler, tmp_path):
        """Test saving profile with default path."""
        profiler.log_dir = tmp_path
        profiler.log_step(step=1, loss=0.5, lr=1e-4)

        profiler.save_profile()

        default_path = tmp_path / "training_profile.json"
        assert default_path.exists()

    def test_save_profile_no_path(self, profiler):
        """Test saving profile without path."""
        profiler.log_dir = None
        profiler.log_step(step=1, loss=0.5, lr=1e-4)

        # Should not raise error, just return
        profiler.save_profile()

    def test_get_summary_stats(self, profiler):
        """Test getting summary statistics."""
        profiler.log_step(step=1, loss=0.5, lr=1e-4, tokens_processed=1000)
        profiler.log_step(step=2, loss=0.4, lr=1e-4, tokens_processed=2000)
        profiler.log_step(step=3, loss=0.3, lr=1e-4, tokens_processed=3000)

        stats = profiler.get_summary_stats()

        assert stats["total_steps"] == 3
        assert stats["final_loss"] == 0.3
        assert stats["min_loss"] == 0.3
        assert stats["total_time_seconds"] > 0
        assert "peak_memory_mb" in stats

    def test_get_summary_stats_empty(self, profiler):
        """Test getting summary stats with no steps."""
        stats = profiler.get_summary_stats()
        assert stats == {}


class TestProfileMemoryUsage:
    """Test profile_memory_usage decorator."""

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.max_memory_allocated")
    def test_profile_memory_usage_with_gpu(
        self, mock_max_memory, mock_memory, mock_reset, mock_available, capsys
    ):
        """Test memory profiling with GPU available."""
        mock_available.return_value = True
        mock_memory.side_effect = [1024 * 1024 * 100, 1024 * 1024 * 200]  # 100MB -> 200MB
        mock_max_memory.return_value = 1024 * 1024 * 250  # 250MB peak

        @profile_memory_usage("test_function")
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        captured = capsys.readouterr()
        assert "[MEMORY]" in captured.out
        assert "test_function" in captured.out

    @patch("torch.cuda.is_available")
    def test_profile_memory_usage_no_gpu(self, mock_available):
        """Test memory profiling without GPU."""
        mock_available.return_value = False

        @profile_memory_usage("test_function")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"


class TestPerformanceMonitorIntegration:
    """Test integration of performance monitoring components."""

    def test_complete_profiling_workflow(self, tmp_path):
        """Test complete profiling workflow."""
        profiler = TrainingProfiler(log_dir=tmp_path)

        # Log several steps
        for i in range(3):
            profiler.log_step(step=i, loss=0.5 - i * 0.1, lr=1e-4, tokens_processed=(i + 1) * 1000)

        # Get summary
        stats = profiler.get_summary_stats()
        assert stats["total_steps"] == 3

        # Save profile
        output_path = tmp_path / "profile.json"
        profiler.save_profile(output_path)
        assert output_path.exists()

    def test_monitor_with_profiler(self):
        """Test using PerformanceMonitor with TrainingProfiler."""
        monitor = PerformanceMonitor()
        metrics1 = monitor.get_metrics(tokens_processed=1000)

        time.sleep(0.1)
        metrics2 = monitor.get_metrics(tokens_processed=2000)

        assert metrics2.wall_time_seconds > metrics1.wall_time_seconds
        assert metrics2.tokens_processed > metrics1.tokens_processed

    def test_time_operation_with_profiler(self, tmp_path):
        """Test using time_operation with profiler."""
        profiler = TrainingProfiler(log_dir=tmp_path)

        with time_operation("model_forward"):
            profiler.log_step(step=1, loss=0.5, lr=1e-4)

        assert len(profiler.step_metrics) == 1


