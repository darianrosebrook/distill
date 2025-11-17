"""
Tests for training/monitoring.py - Testing actual implementation methods.

This file tests the actual methods that exist in monitoring.py, which differ
from the methods tested in test_monitoring.py.
"""
# @author: @darianrosebrook

import json
import time
import threading
from unittest.mock import Mock, patch

import pytest

from training.monitoring import (
    HealthStatus,
    MetricsCollector,
    HealthChecker,
    SystemHealthChecks,
    TrainingMonitor,
    initialize_monitoring,
)


class TestMetricsCollectorActual:
    """Test MetricsCollector with actual implementation methods."""

    @pytest.fixture
    def collector(self):
        """Create a metrics collector instance."""
        return MetricsCollector(max_points=100)

    def test_metrics_collector_initialization(self, collector):
        """Test metrics collector initialization."""
        assert len(collector.metrics) == 0
        assert collector.max_points == 100
        assert hasattr(collector, 'lock') and collector.lock is not None

    def test_record_metric(self, collector):
        """Test recording a metric."""
        collector.record_metric("loss", 0.5, epoch=1, batch=10)

        assert len(collector.metrics) == 1
        assert collector.metrics[0].name == "loss"
        assert collector.metrics[0].value == 0.5
        assert collector.metrics[0].tags["epoch"] == 1
        assert collector.metrics[0].tags["batch"] == 10

    def test_record_metric_eviction(self, collector):
        """Test metric eviction when max_points is reached."""
        collector.max_points = 3

        # Add 5 points
        for i in range(5):
            collector.record_metric(f"metric_{i}", float(i))

        # Should only keep the last 3
        assert len(collector.metrics) == 3
        assert collector.metrics[0].name == "metric_2"
        assert collector.metrics[1].name == "metric_3"
        assert collector.metrics[2].name == "metric_4"

    def test_get_metrics_no_filter(self, collector):
        """Test getting all metrics without filters."""
        collector.record_metric("loss", 0.5)
        collector.record_metric("accuracy", 0.9)
        collector.record_metric("loss", 0.3)

        all_metrics = collector.get_metrics()
        assert len(all_metrics) == 3

    def test_get_metrics_by_name(self, collector):
        """Test getting metrics filtered by name."""
        collector.record_metric("loss", 0.5)
        collector.record_metric("accuracy", 0.9)
        collector.record_metric("loss", 0.3)

        loss_metrics = collector.get_metrics(name="loss")
        assert len(loss_metrics) == 2
        assert all(m.name == "loss" for m in loss_metrics)

    def test_get_metrics_by_time_range(self, collector):
        """Test getting metrics filtered by time range."""
        time.time()
        collector.record_metric("metric1", 1.0)
        time.sleep(0.01)
        mid_time = time.time()
        collector.record_metric("metric2", 2.0)
        time.sleep(0.01)
        collector.record_metric("metric3", 3.0)
        end_time = time.time()

        # Get metrics in time range
        filtered = collector.get_metrics(start_time=mid_time, end_time=end_time)
        assert len(filtered) >= 1
        assert all(mid_time <= m.timestamp <= end_time for m in filtered)

    def test_get_summary_stats(self, collector):
        """Test getting summary statistics."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for value in values:
            collector.record_metric("loss", value)

        stats = collector.get_summary_stats("loss")

        assert stats["count"] == 5
        assert stats["mean"] == 0.3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["latest"] == 0.5
        assert stats["first"] == 0.1

    def test_get_summary_stats_empty(self, collector):
        """Test getting summary stats for non-existent metric."""
        stats = collector.get_summary_stats("nonexistent")
        assert stats == {}

    def test_save_to_file(self, collector, tmp_path):
        """Test saving metrics to file."""
        collector.record_metric("loss", 0.5, epoch=1)
        collector.record_metric("accuracy", 0.9, epoch=1)

        output_path = tmp_path / "metrics.json"
        collector.save_to_file(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "exported_at" in data
        assert "metrics" in data
        assert len(data["metrics"]) == 2

    def test_thread_safety(self, collector):
        """Test that metrics collector is thread-safe."""
        import concurrent.futures

        def record_metrics_worker(worker_id):
            for i in range(10):
                collector.record_metric(f"worker_{worker_id}_metric_{i}", float(i))

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(record_metrics_worker, i) for i in range(3)]
            concurrent.futures.wait(futures)

        assert len(collector.metrics) == 30


class TestHealthCheckerActual:
    """Test HealthChecker with actual implementation methods."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance."""
        return HealthChecker()

    def test_health_checker_initialization(self, health_checker):
        """Test health checker initialization."""
        assert health_checker.metrics_collector is None
        assert len(health_checker.checks) == 0
        assert health_checker.check_interval == 60

    def test_register_check(self, health_checker):
        """Test registering a health check."""
        def dummy_check():
            return HealthStatus(
                timestamp=time.time(), component="test", status="healthy", message="OK"
            )

        health_checker.register_check("test_check", dummy_check)

        assert "test_check" in health_checker.checks
        assert callable(health_checker.checks["test_check"])

    def test_run_checks(self, health_checker):
        """Test running all health checks."""
        def check1():
            return HealthStatus(time.time(), "comp1", "healthy", "OK")

        def check2():
            return HealthStatus(time.time(), "comp2", "warning", "Warning")

        health_checker.register_check("check1", check1)
        health_checker.register_check("check2", check2)

        results = health_checker.run_checks()

        assert len(results) == 2
        assert any(r.component == "comp1" and r.status == "healthy" for r in results)
        assert any(r.component == "comp2" and r.status == "warning" for r in results)

    def test_run_checks_with_metrics_collector(self, health_checker):
        """Test running checks with metrics collector."""
        collector = MetricsCollector()
        health_checker.metrics_collector = collector

        def healthy_check():
            return HealthStatus(time.time(), "test", "healthy", "OK")

        health_checker.register_check("test", healthy_check)
        health_checker.run_checks()

        # Should have recorded health check metric
        health_metrics = collector.get_metrics(name="health_check")
        assert len(health_metrics) == 1
        assert health_metrics[0].value == 1  # healthy = 1

    def test_run_checks_exception_handling(self, health_checker):
        """Test that exceptions in health checks are handled."""
        def failing_check():
            raise ValueError("Check failed")

        health_checker.register_check("failing", failing_check)
        results = health_checker.run_checks()

        assert len(results) == 1
        assert results[0].component == "failing"
        assert results[0].status == "error"
        assert "Health check failed" in results[0].message

    def test_get_overall_health(self, health_checker):
        """Test getting overall health status."""
        def healthy_check():
            return HealthStatus(time.time(), "comp1", "healthy", "OK")

        def warning_check():
            return HealthStatus(time.time(), "comp2", "warning", "Warning")

        def error_check():
            return HealthStatus(time.time(), "comp3", "error", "Error")

        # Test with all healthy
        health_checker.register_check("healthy", healthy_check)
        assert health_checker.get_overall_health() == "healthy"

        # Test with warning
        health_checker.register_check("warning", warning_check)
        assert health_checker.get_overall_health() == "warning"

        # Test with error
        health_checker.register_check("error", error_check)
        assert health_checker.get_overall_health() == "error"


class TestSystemHealthChecks:
    """Test SystemHealthChecks static methods."""

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_healthy(self, mock_memory):
        """Test memory check when usage is healthy."""
        mock_memory.return_value.percent = 50.0

        status = SystemHealthChecks.check_memory_usage()

        assert status.component == "memory"
        assert status.status == "healthy"
        assert "50.0%" in status.message

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_warning(self, mock_memory):
        """Test memory check when usage is high."""
        mock_memory.return_value.percent = 85.0

        status = SystemHealthChecks.check_memory_usage()

        assert status.component == "memory"
        assert status.status == "warning"
        assert "85.0%" in status.message

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_error(self, mock_memory):
        """Test memory check when usage is critical."""
        mock_memory.return_value.percent = 95.0

        status = SystemHealthChecks.check_memory_usage()

        assert status.component == "memory"
        assert status.status == "error"
        assert "95.0%" in status.message

    @patch("psutil.disk_usage")
    def test_check_disk_usage_healthy(self, mock_disk):
        """Test disk usage check when healthy."""
        mock_disk.return_value.percent = 60.0

        status = SystemHealthChecks.check_disk_usage()

        assert status.component == "disk"
        assert status.status == "healthy"
        assert "60.0%" in status.message

    @patch("psutil.disk_usage")
    def test_check_disk_usage_warning(self, mock_disk):
        """Test disk usage check when high."""
        mock_disk.return_value.percent = 90.0

        status = SystemHealthChecks.check_disk_usage()

        assert status.component == "disk"
        assert status.status == "warning"
        assert "90.0%" in status.message

    @patch("psutil.disk_usage")
    def test_check_disk_usage_error(self, mock_disk):
        """Test disk usage check when critical."""
        mock_disk.return_value.percent = 96.0

        status = SystemHealthChecks.check_disk_usage()

        assert status.component == "disk"
        assert status.status == "error"
        assert "96.0%" in status.message

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_with_gpu_healthy(
        self, mock_device_props, mock_memory_reserved, mock_memory_allocated, mock_cuda_available
    ):
        """Test GPU memory check when GPU is available and healthy."""
        mock_cuda_available.return_value = True
        # Simulate 50% usage: allocated=4GB, reserved=4GB, total=8GB
        allocated_bytes = 4 * 1024**3
        reserved_bytes = 4 * 1024**3
        total_bytes = 8 * 1024**3
        mock_memory_allocated.return_value = allocated_bytes
        mock_memory_reserved.return_value = reserved_bytes
        mock_props = Mock()
        mock_props.total_memory = total_bytes
        mock_device_props.return_value = mock_props

        status = SystemHealthChecks.check_gpu_memory()

        assert status.component == "gpu_memory"
        assert status.status == "healthy"
        assert "normal" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_with_gpu_warning(
        self, mock_device_props, mock_memory_reserved, mock_memory_allocated, mock_cuda_available
    ):
        """Test GPU memory check when GPU memory is high."""
        mock_cuda_available.return_value = True
        # Simulate 90% usage: allocated=7.2GB, reserved=7.2GB, total=8GB
        allocated_bytes = 7.2 * 1024**3
        reserved_bytes = 7.2 * 1024**3
        total_bytes = 8 * 1024**3
        mock_memory_allocated.return_value = allocated_bytes
        mock_memory_reserved.return_value = reserved_bytes
        mock_props = Mock()
        mock_props.total_memory = total_bytes
        mock_device_props.return_value = mock_props

        status = SystemHealthChecks.check_gpu_memory()

        assert status.component == "gpu_memory"
        assert status.status == "warning"
        assert "high" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_with_gpu_error(
        self, mock_device_props, mock_memory_reserved, mock_memory_allocated, mock_cuda_available
    ):
        """Test GPU memory check when GPU memory is critical."""
        mock_cuda_available.return_value = True
        # Simulate 96% usage: allocated=7.68GB, reserved=7.68GB, total=8GB
        allocated_bytes = 7.68 * 1024**3
        reserved_bytes = 7.68 * 1024**3
        total_bytes = 8 * 1024**3
        mock_memory_allocated.return_value = allocated_bytes
        mock_memory_reserved.return_value = reserved_bytes
        mock_props = Mock()
        mock_props.total_memory = total_bytes
        mock_device_props.return_value = mock_props

        status = SystemHealthChecks.check_gpu_memory()

        assert status.component == "gpu_memory"
        assert status.status == "error"
        assert "critical" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_exception(self, mock_device_props, mock_memory_reserved, mock_memory_allocated, mock_cuda_available):
        """Test GPU memory check when exception occurs."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.side_effect = RuntimeError("CUDA error")

        status = SystemHealthChecks.check_gpu_memory()

        assert status.component == "gpu_memory"
        assert status.status == "warning"
        assert "failed" in status.message.lower()

    @patch("torch.cuda.is_available")
    def test_check_gpu_memory_no_gpu(self, mock_cuda_available):
        """Test GPU memory check when no GPU is available."""
        mock_cuda_available.return_value = False

        status = SystemHealthChecks.check_gpu_memory()

        assert status.component == "gpu_memory"
        assert status.status == "healthy"
        assert "not available" in status.message.lower()


class TestTrainingMonitorActual:
    """Test TrainingMonitor with actual implementation methods."""

    @pytest.fixture
    def training_monitor(self, tmp_path):
        """Create a training monitor instance."""
        return TrainingMonitor(log_dir=tmp_path)

    def test_training_monitor_initialization(self, tmp_path):
        """Test training monitor initialization."""
        monitor = TrainingMonitor(log_dir=tmp_path)

        assert monitor.log_dir == tmp_path
        assert isinstance(monitor.metrics, MetricsCollector)
        assert isinstance(monitor.health_checker, HealthChecker)
        assert monitor.training_start_time is None

    def test_start_training(self, training_monitor):
        """Test starting training monitoring."""
        config = {"model": "test", "epochs": 10}
        training_monitor.start_training(config)

        assert training_monitor.training_start_time is not None
        # Should have recorded training_started metric
        start_metrics = training_monitor.metrics.get_metrics(name="training_started")
        assert len(start_metrics) == 1

    def test_record_step(self, training_monitor):
        """Test recording training step."""
        training_monitor.record_step(step=1, loss=0.5, lr=1e-4, tokens_processed=1000)

        loss_metrics = training_monitor.metrics.get_metrics(name="loss")
        lr_metrics = training_monitor.metrics.get_metrics(name="learning_rate")
        token_metrics = training_monitor.metrics.get_metrics(name="tokens_processed")

        assert len(loss_metrics) == 1
        assert len(lr_metrics) == 1
        assert len(token_metrics) == 1
        assert loss_metrics[0].value == 0.5

    def test_record_step_with_gpu_memory(self, training_monitor):
        """Test recording step with GPU memory."""
        training_monitor.record_step(
            step=1, loss=0.5, lr=1e-4, gpu_memory_mb=4000.0
        )

        gpu_metrics = training_monitor.metrics.get_metrics(name="gpu_memory_mb")
        assert len(gpu_metrics) == 1
        assert gpu_metrics[0].value == 4000.0

    def test_end_training(self, training_monitor, tmp_path):
        """Test ending training monitoring."""
        training_monitor.start_training({"model": "test"})
        training_monitor.record_step(step=1, loss=0.5, lr=1e-4)

        training_monitor.end_training(final_loss=0.3, total_steps=100)

        # Should have saved metrics file
        metrics_files = list(tmp_path.glob("training_metrics_*.json"))
        assert len(metrics_files) == 1

        # Should have saved summary file
        summary_file = tmp_path / "training_summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)
        assert summary["total_steps"] == 100
        assert summary["final_loss"] == 0.3

    def test_end_training_without_start(self, training_monitor):
        """Test ending training without starting."""
        # Should not raise error
        training_monitor.end_training(final_loss=0.3, total_steps=100)

    def test_get_status(self, training_monitor):
        """Test getting monitoring status."""
        training_monitor.start_training({"model": "test"})
        training_monitor.record_step(step=1, loss=0.5, lr=1e-4)

        status = training_monitor.get_status()

        assert isinstance(status, dict)
        assert "training_started" in status
        assert "total_steps" in status
        assert "metrics_count" in status

    def test_run_health_checks_periodic(self, training_monitor):
        """Test that health checks run periodically."""
        training_monitor.health_check_interval = 0.1  # Very short interval
        training_monitor.start_training({"model": "test"})

        # Record multiple steps quickly
        for i in range(5):
            training_monitor.record_step(step=i, loss=0.5, lr=1e-4)
            time.sleep(0.05)

        # Should have run health checks
        health_metrics = training_monitor.metrics.get_metrics(name="health_check")
        assert len(health_metrics) > 0

    @patch("training.monitoring.print")
    def test_run_health_checks_logs_warnings(self, mock_print, training_monitor):
        """Test that health check warnings are logged."""
        # Register a warning check
        def warning_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test",
                status="warning",
                message="Test warning",
            )

        training_monitor.health_checker.register_check("test_warning", warning_check)
        training_monitor.health_check_interval = 0.0  # Run immediately
        training_monitor.start_training({"model": "test"})

        # Trigger health check
        training_monitor.record_step(step=1, loss=0.5, lr=1e-4)

        # Should have logged warning
        assert mock_print.called
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("⚠️" in str(call) or "warning" in str(call).lower() for call in print_calls)

        # Should have recorded health issue metric
        health_issues = training_monitor.metrics.get_metrics(name="health_issue")
        assert len(health_issues) > 0


class TestInitializeMonitoring:
    """Test initialize_monitoring function."""

    @patch("training.monitoring.TrainingMonitor")
    def test_initialize_monitoring(self, mock_training_monitor, tmp_path):
        """Test initializing monitoring."""
        mock_monitor = Mock()
        mock_training_monitor.return_value = mock_monitor

        result = initialize_monitoring(log_dir=tmp_path)

        assert result == mock_monitor
        mock_training_monitor.assert_called_once_with(log_dir=tmp_path)

    @patch("training.monitoring.TrainingMonitor")
    def test_initialize_monitoring_default(self, mock_training_monitor):
        """Test initializing monitoring with default log dir."""
        mock_monitor = Mock()
        mock_training_monitor.return_value = mock_monitor

        result = initialize_monitoring()

        assert result == mock_monitor
        mock_training_monitor.assert_called_once()

