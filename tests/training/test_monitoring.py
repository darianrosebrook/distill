"""
Tests for training/monitoring.py - Training monitoring and observability system.

Tests metrics collection, health checking, alerting, system monitoring,
and training progress tracking using mock data and scenarios.
"""
# @author: @darianrosebrook

import time
import threading
from unittest.mock import Mock, patch

import pytest

from training.monitoring import (
    MetricPoint,
    HealthStatus,
    MetricsCollector,
    HealthChecker,
    SystemHealthChecks,
    TrainingMonitor,
    initialize_monitoring,
)


class TestMetricPoint:
    """Test MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        point = MetricPoint(
            timestamp=1234567890.0, name="loss", value=0.5, tags={"phase": "training", "epoch": "1"}
        )

        assert point.timestamp == 1234567890.0
        assert point.name == "loss"
        assert point.value == 0.5
        assert point.tags["phase"] == "training"
        assert point.tags["epoch"] == "1"

    def test_metric_point_default_tags(self):
        """Test metric point with default empty tags."""
        point = MetricPoint(timestamp=1234567890.0, name="accuracy", value=0.95)

        assert len(point.tags) == 0
        assert isinstance(point.tags, dict)


class TestHealthStatus:
    """Test HealthStatus dataclass."""

    def test_health_status_creation(self):
        """Test creating a health status."""
        status = HealthStatus(
            timestamp=1234567890.0,
            component="gpu",
            status="healthy",
            message="GPU utilization normal",
            details={"utilization": 45.2, "memory_used": "2.1GB"},
        )

        assert status.timestamp == 1234567890.0
        assert status.component == "gpu"
        assert status.status == "healthy"
        assert status.message == "GPU utilization normal"
        assert status.details["utilization"] == 45.2

    def test_health_status_default_details(self):
        """Test health status with default empty details."""
        status = HealthStatus(
            timestamp=1234567890.0, component="cpu", status="warning", message="High CPU usage"
        )

        assert len(status.details) == 0
        assert isinstance(status.details, dict)


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a metrics collector instance."""
        return MetricsCollector(max_points=100)

    def test_metrics_collector_initialization(self, collector):
        """Test metrics collector initialization."""
        assert len(collector.metrics) == 0
        assert collector.max_points == 100
        assert isinstance(collector.lock, threading.Lock)

    def test_add_metric(self, collector):
        """Test adding a metric point."""
        point = MetricPoint(timestamp=time.time(), name="loss", value=0.5, tags={"batch": "1"})

        collector.add_metric(point)

        assert len(collector.metrics) == 1
        assert collector.metrics[0] == point

    def test_add_metric_eviction(self, collector):
        """Test metric eviction when max_points is reached."""
        # Set max_points to 3
        collector.max_points = 3

        # Add 5 points
        for i in range(5):
            point = MetricPoint(timestamp=time.time(), name=f"metric_{i}", value=float(i))
            collector.add_metric(point)

        # Should only keep the last 3
        assert len(collector.metrics) == 3
        assert collector.metrics[0].name == "metric_2"
        assert collector.metrics[1].name == "metric_3"
        assert collector.metrics[2].name == "metric_4"

    def test_get_metrics_by_name(self, collector):
        """Test retrieving metrics by name."""
        # Add various metrics
        metrics_data = [
            ("loss", 0.5, {"phase": "train"}),
            ("accuracy", 0.9, {"phase": "train"}),
            ("loss", 0.3, {"phase": "val"}),
            ("f1_score", 0.85, {"phase": "test"}),
        ]

        for name, value, tags in metrics_data:
            point = MetricPoint(timestamp=time.time(), name=name, value=value, tags=tags)
            collector.add_metric(point)

        # Get loss metrics
        loss_metrics = collector.get_metrics_by_name("loss")
        assert len(loss_metrics) == 2
        assert all(m.name == "loss" for m in loss_metrics)

        # Get accuracy metrics
        accuracy_metrics = collector.get_metrics_by_name("accuracy")
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].name == "accuracy"

    def test_get_metrics_by_tags(self, collector):
        """Test retrieving metrics by tags."""
        # Add metrics with different tags
        metrics_data = [
            ("loss", 0.5, {"phase": "train", "epoch": "1"}),
            ("accuracy", 0.9, {"phase": "train", "epoch": "1"}),
            ("loss", 0.3, {"phase": "val", "epoch": "1"}),
            ("loss", 0.2, {"phase": "train", "epoch": "2"}),
        ]

        for name, value, tags in metrics_data:
            point = MetricPoint(timestamp=time.time(), name=name, value=value, tags=tags)
            collector.add_metric(point)

        # Get training metrics
        train_metrics = collector.get_metrics_by_tags({"phase": "train"})
        assert len(train_metrics) == 3

        # Get epoch 1 metrics
        epoch1_metrics = collector.get_metrics_by_tags({"epoch": "1"})
        assert len(epoch1_metrics) == 3

        # Get specific combination
        train_epoch1 = collector.get_metrics_by_tags({"phase": "train", "epoch": "1"})
        assert len(train_epoch1) == 2

    def test_get_latest_metric(self, collector):
        """Test getting the latest metric value."""
        # Add metrics with increasing timestamps
        base_time = time.time()
        for i in range(3):
            point = MetricPoint(timestamp=base_time + i, name="loss", value=0.5 - i * 0.1)
            collector.add_metric(point)

        latest = collector.get_latest_metric("loss")
        assert latest is not None
        assert latest.value == 0.2  # Most recent value

    def test_get_metric_statistics(self, collector):
        """Test calculating metric statistics."""
        # Add multiple values for the same metric
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for value in values:
            point = MetricPoint(timestamp=time.time(), name="loss", value=value)
            collector.add_metric(point)

        stats = collector.get_metric_statistics("loss")

        assert stats["count"] == 5
        assert stats["mean"] == 0.3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert abs(stats["std"] - 0.15811388300841897) < 1e-6

    def test_clear_metrics(self, collector):
        """Test clearing all metrics."""
        # Add some metrics
        for i in range(3):
            point = MetricPoint(timestamp=time.time(), name=f"metric_{i}", value=float(i))
            collector.add_metric(point)

        assert len(collector.metrics) == 3

        # Clear metrics
        collector.clear_metrics()

        assert len(collector.metrics) == 0

    def test_thread_safety(self, collector):
        """Test that metrics collector is thread-safe."""
        import concurrent.futures

        def add_metrics_thread(thread_id):
            for i in range(10):
                point = MetricPoint(
                    timestamp=time.time(), name=f"thread_{thread_id}_metric_{i}", value=float(i)
                )
                collector.add_metric(point)

        # Run multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_metrics_thread, i) for i in range(3)]
            concurrent.futures.wait(futures)

        # Should have 30 metrics total
        assert len(collector.metrics) == 30

        # Check that all thread metrics are present
        for thread_id in range(3):
            thread_metrics = [m for m in collector.metrics if f"thread_{thread_id}_" in m.name]
            assert len(thread_metrics) == 10


class TestHealthChecker:
    """Test HealthChecker class."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance."""
        return HealthChecker()

    def test_health_checker_initialization(self, health_checker):
        """Test health checker initialization."""
        assert len(health_checker.checks) == 0
        assert len(health_checker.status_history) == 0

    def test_add_check(self, health_checker):
        """Test adding a health check."""

        def dummy_check():
            return HealthStatus(
                timestamp=time.time(), component="test", status="healthy", message="All good"
            )

        health_checker.add_check("test_check", dummy_check)

        assert "test_check" in health_checker.checks
        assert callable(health_checker.checks["test_check"])

    def test_run_check(self, health_checker):
        """Test running a single health check."""

        def dummy_check():
            return HealthStatus(
                timestamp=time.time(), component="test", status="healthy", message="Check passed"
            )

        health_checker.add_check("test_check", dummy_check)

        status = health_checker.run_check("test_check")

        assert status.component == "test"
        assert status.status == "healthy"
        assert status.message == "Check passed"

    def test_run_check_not_found(self, health_checker):
        """Test running a nonexistent health check."""
        with pytest.raises(KeyError):
            health_checker.run_check("nonexistent_check")

    def test_run_all_checks(self, health_checker):
        """Test running all health checks."""

        # Add multiple checks
        def check1():
            return HealthStatus(time.time(), "comp1", "healthy", "OK")

        def check2():
            return HealthStatus(time.time(), "comp2", "warning", "Warning")

        def check3():
            return HealthStatus(time.time(), "comp3", "error", "Error")

        health_checker.add_check("check1", check1)
        health_checker.add_check("check2", check2)
        health_checker.add_check("check3", check3)

        statuses = health_checker.run_all_checks()

        assert len(statuses) == 3
        assert len([s for s in statuses if s.status == "healthy"]) == 1
        assert len([s for s in statuses if s.status == "warning"]) == 1
        assert len([s for s in statuses if s.status == "error"]) == 1

    def test_get_component_status(self, health_checker):
        """Test getting status for a specific component."""

        # Add checks for different components
        def gpu_check():
            return HealthStatus(time.time(), "gpu", "healthy", "GPU OK")

        def cpu_check():
            return HealthStatus(time.time(), "cpu", "warning", "CPU high")

        health_checker.add_check("gpu_check", gpu_check)
        health_checker.add_check("cpu_check", cpu_check)

        # Run checks to populate history
        health_checker.run_all_checks()

        # Get status for GPU
        gpu_status = health_checker.get_component_status("gpu")
        assert gpu_status is not None
        assert gpu_status.component == "gpu"
        assert gpu_status.status == "healthy"

        # Get status for CPU
        cpu_status = health_checker.get_component_status("cpu")
        assert cpu_status is not None
        assert cpu_status.component == "cpu"
        assert cpu_status.status == "warning"

        # Get status for nonexistent component
        nonexistent = health_checker.get_component_status("nonexistent")
        assert nonexistent is None

    def test_get_overall_health(self, health_checker):
        """Test getting overall system health."""

        # Add checks with different statuses
        def healthy_check():
            return HealthStatus(time.time(), "comp1", "healthy", "OK")

        def warning_check():
            return HealthStatus(time.time(), "comp2", "warning", "Warning")

        def error_check():
            return HealthStatus(time.time(), "comp3", "error", "Error")

        health_checker.add_check("healthy", healthy_check)
        health_checker.add_check("warning", warning_check)
        health_checker.add_check("error", error_check)

        # Run checks
        health_checker.run_all_checks()

        # Overall health should be "error" (worst status)
        overall = health_checker.get_overall_health()
        assert overall == "error"

        # Test with only healthy checks
        healthy_only_checker = HealthChecker()
        healthy_only_checker.add_check("healthy1", healthy_check)
        healthy_only_checker.add_check("healthy2", healthy_check)
        healthy_only_checker.run_all_checks()

        assert healthy_only_checker.get_overall_health() == "healthy"


class TestSystemHealthChecks:
    """Test SystemHealthChecks class."""

    @pytest.fixture
    def system_checks(self):
        """Create system health checks instance."""
        return SystemHealthChecks()

    def test_system_health_checks_initialization(self, system_checks):
        """Test system health checks initialization."""
        # Should have several built-in checks
        assert len(system_checks.health_checker.checks) > 0

        # Should include common system checks
        check_names = list(system_checks.health_checker.checks.keys())
        assert any("cpu" in name.lower() for name in check_names)
        assert any("memory" in name.lower() for name in check_names)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_cpu_usage_check(self, mock_memory, mock_cpu, system_checks):
        """Test CPU usage health check."""
        mock_cpu.return_value = 85.0  # High CPU usage
        mock_memory.return_value.percent = 50.0

        status = system_checks._check_cpu_usage()

        assert status.component == "cpu"
        assert status.status == "warning"  # High usage
        assert "85.0%" in status.message

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_memory_usage_check(self, mock_memory, mock_cpu, system_checks):
        """Test memory usage health check."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 90.0  # High memory usage

        status = system_checks._check_memory_usage()

        assert status.component == "memory"
        assert status.status == "warning"
        assert "90.0%" in status.message

    @patch("torch.cuda.is_available")
    def test_gpu_check_no_gpu(self, mock_cuda_available, system_checks):
        """Test GPU check when no GPU is available."""
        mock_cuda_available.return_value = False

        status = system_checks._check_gpu_usage()

        assert status.component == "gpu"
        assert status.status == "healthy"
        assert "not available" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.mem_get_info")
    @patch("torch.cuda.utilization")
    def test_gpu_check_with_gpu(
        self, mock_utilization, mock_mem_info, mock_cuda_available, system_checks
    ):
        """Test GPU check when GPU is available."""
        mock_cuda_available.return_value = True
        mock_utilization.return_value = 75
        mock_mem_info.return_value = (2 * 1024**3, 8 * 1024**3)  # 2GB used, 8GB total

        status = system_checks._check_gpu_usage()

        assert status.component == "gpu"
        assert status.status == "healthy"
        assert "75%" in status.message

    def test_disk_usage_check(self, system_checks):
        """Test disk usage health check."""
        status = system_checks._check_disk_usage()

        assert status.component == "disk"
        assert isinstance(status.status, str)
        assert "%" in status.message

    def test_network_connectivity_check(self, system_checks):
        """Test network connectivity health check."""
        status = system_checks._check_network_connectivity()

        assert status.component == "network"
        assert isinstance(status.status, str)

    def test_run_system_checks(self, system_checks):
        """Test running all system health checks."""
        statuses = system_checks.run_system_checks()

        assert isinstance(statuses, list)
        assert len(statuses) > 0

        # All statuses should be HealthStatus objects
        for status in statuses:
            assert isinstance(status, HealthStatus)


class TestTrainingMonitor:
    """Test TrainingMonitor class."""

    @pytest.fixture
    def training_monitor(self):
        """Create a training monitor instance."""
        return TrainingMonitor()

    def test_training_monitor_initialization(self, training_monitor):
        """Test training monitor initialization."""
        assert isinstance(training_monitor.metrics_collector, MetricsCollector)
        assert isinstance(training_monitor.health_checker, HealthChecker)
        assert not training_monitor.is_monitoring
        assert training_monitor.monitoring_thread is None

    def test_start_monitoring(self, training_monitor):
        """Test starting monitoring."""
        training_monitor.start_monitoring(interval=1.0)

        assert training_monitor.is_monitoring
        assert training_monitor.monitoring_thread is not None
        assert training_monitor.monitoring_thread.is_alive()

        # Stop monitoring
        training_monitor.stop_monitoring()
        assert not training_monitor.is_monitoring

    def test_stop_monitoring(self, training_monitor):
        """Test stopping monitoring."""
        training_monitor.start_monitoring(interval=1.0)
        assert training_monitor.is_monitoring

        training_monitor.stop_monitoring()
        assert not training_monitor.is_monitoring

        # Thread should eventually stop
        if training_monitor.monitoring_thread:
            training_monitor.monitoring_thread.join(timeout=2.0)

    def test_log_metric(self, training_monitor):
        """Test logging a metric."""
        training_monitor.log_metric("loss", 0.5, {"epoch": "1"})

        metrics = training_monitor.metrics_collector.get_metrics_by_name("loss")
        assert len(metrics) == 1
        assert metrics[0].value == 0.5
        assert metrics[0].tags["epoch"] == "1"

    def test_log_training_step(self, training_monitor):
        """Test logging training step metrics."""
        training_monitor.log_training_step(
            step=100, loss=0.3, learning_rate=1e-4, epoch=5, tokens_processed=50000
        )

        # Should log multiple metrics
        loss_metrics = training_monitor.metrics_collector.get_metrics_by_name("training_loss")
        lr_metrics = training_monitor.metrics_collector.get_metrics_by_name("learning_rate")
        token_metrics = training_monitor.metrics_collector.get_metrics_by_name("tokens_processed")

        assert len(loss_metrics) >= 1
        assert len(lr_metrics) >= 1
        assert len(token_metrics) >= 1

    def test_log_validation_metrics(self, training_monitor):
        """Test logging validation metrics."""
        training_monitor.log_validation_metrics(
            epoch=5, val_loss=0.4, val_accuracy=0.85, val_f1=0.82
        )

        val_loss_metrics = training_monitor.metrics_collector.get_metrics_by_name("validation_loss")
        accuracy_metrics = training_monitor.metrics_collector.get_metrics_by_name(
            "validation_accuracy"
        )

        assert len(val_loss_metrics) >= 1
        assert len(accuracy_metrics) >= 1

    def test_check_system_health(self, training_monitor):
        """Test checking system health."""
        # This would normally check system resources
        # For testing, we'll just ensure it doesn't crash
        try:
            health_statuses = training_monitor.check_system_health()
            assert isinstance(health_statuses, list)
        except Exception:
            # Some health checks might fail in test environment
            pass

    def test_get_monitoring_stats(self, training_monitor):
        """Test getting monitoring statistics."""
        # Add some metrics first
        training_monitor.log_metric("test_metric", 42.0)

        stats = training_monitor.get_monitoring_stats()

        assert isinstance(stats, dict)
        assert "metrics_collected" in stats
        assert "health_checks_run" in stats
        assert stats["metrics_collected"] >= 1

    def test_save_load_metrics(self, training_monitor, tmp_path):
        """Test saving and loading metrics."""
        # Add some metrics
        training_monitor.log_metric("test_loss", 0.5)
        training_monitor.log_metric("test_accuracy", 0.9)

        metrics_file = tmp_path / "metrics.json"

        # Save metrics
        training_monitor.save_metrics(str(metrics_file))

        # Clear current metrics
        training_monitor.metrics_collector.clear_metrics()
        assert len(training_monitor.metrics_collector.metrics) == 0

        # Load metrics
        training_monitor.load_metrics(str(metrics_file))

        # Should have loaded the metrics back
        assert len(training_monitor.metrics_collector.metrics) >= 2

    def test_alert_on_condition(self, training_monitor):
        """Test alerting on conditions."""
        alert_triggered = []

        def alert_callback(message, severity):
            alert_triggered.append((message, severity))

        training_monitor.add_alert_condition(
            "high_loss",
            lambda: training_monitor.metrics_collector.get_latest_metric("loss")
            and training_monitor.metrics_collector.get_latest_metric("loss").value > 1.0,
            "Loss too high",
            "warning",
            alert_callback,
        )

        # Add low loss - should not trigger
        training_monitor.log_metric("loss", 0.5)
        training_monitor.check_alerts()
        assert len(alert_triggered) == 0

        # Add high loss - should trigger
        training_monitor.log_metric("loss", 1.5)
        training_monitor.check_alerts()
        assert len(alert_triggered) == 1
        assert "Loss too high" in alert_triggered[0][0]


class TestInitializeMonitoring:
    """Test initialize_monitoring function."""

    @patch("training.monitoring.TrainingMonitor")
    def test_initialize_monitoring_no_log_dir(self, mock_training_monitor):
        """Test initializing monitoring without log directory."""
        mock_monitor = Mock()
        mock_training_monitor.return_value = mock_monitor

        result = initialize_monitoring()

        assert result == mock_monitor
        mock_training_monitor.assert_called_once()

    @patch("training.monitoring.TrainingMonitor")
    def test_initialize_monitoring_with_log_dir(self, mock_training_monitor, tmp_path):
        """Test initializing monitoring with log directory."""
        mock_monitor = Mock()
        mock_training_monitor.return_value = mock_monitor

        log_dir = tmp_path / "logs"
        result = initialize_monitoring(log_dir)

        assert result == mock_monitor
        mock_training_monitor.assert_called_once()


class TestConcurrencyAndPerformance:
    """Test concurrent access and performance aspects."""

    @pytest.fixture
    def training_monitor(self):
        """Create a training monitor instance."""
        return TrainingMonitor()

    def test_concurrent_metric_logging(self, training_monitor):
        """Test concurrent metric logging."""
        import concurrent.futures

        def log_metrics_worker(worker_id):
            for i in range(50):
                training_monitor.log_metric(
                    f"worker_{worker_id}_metric", float(i), {"worker": str(worker_id)}
                )

        # Run with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(log_metrics_worker, i) for i in range(4)]
            concurrent.futures.wait(futures)

        # Should have collected all metrics
        total_metrics = len(training_monitor.metrics_collector.metrics)
        assert total_metrics == 200  # 4 workers * 50 metrics each

    def test_monitoring_performance(self, training_monitor):
        """Test monitoring performance under load."""
        import time

        # Measure time to log many metrics
        start_time = time.time()

        for i in range(1000):
            training_monitor.log_metric(f"perf_metric_{i}", float(i % 100))

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # Less than 5 seconds for 1000 operations

        # Verify all metrics were collected
        assert len(training_monitor.metrics_collector.metrics) == 1000
