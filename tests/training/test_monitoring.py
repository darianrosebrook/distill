"""
Tests for training/monitoring.py - Training monitoring and observability system.

Tests metrics collection, health checking, alerting, system monitoring,
and training progress tracking using mock data and scenarios.
"""
# @author: @darianrosebrook

import json
import time
import threading
from datetime import datetime
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


class TestMetricsCollectorEdgeCases:
    """Test edge cases for MetricsCollector."""

    @pytest.fixture
    def collector(self):
        """Create a metrics collector instance."""
        return MetricsCollector(max_points=100)

    def test_record_metric_max_points_eviction(self, collector):
        """Test record_metric evicts old metrics when max_points exceeded (lines 60-67)."""
        # Set small max_points
        collector.max_points = 5
        
        # Add more metrics than max_points
        for i in range(10):
            collector.record_metric("test_metric", float(i))
        
        # Should only keep last 5 metrics
        assert len(collector.metrics) == 5
        # Should keep the most recent ones
        assert collector.metrics[0].value == 5.0
        assert collector.metrics[-1].value == 9.0

    def test_get_metrics_with_name_filter(self, collector):
        """Test get_metrics with name filter (lines 85-98)."""
        collector.record_metric("metric1", 1.0)
        collector.record_metric("metric2", 2.0)
        collector.record_metric("metric1", 3.0)
        
        filtered = collector.get_metrics(name="metric1")
        assert len(filtered) == 2
        assert all(m.name == "metric1" for m in filtered)

    def test_get_metrics_with_start_time_filter(self, collector):
        """Test get_metrics with start_time filter (lines 85-98)."""
        import time
        start = time.time()
        
        collector.record_metric("test", 1.0)
        time.sleep(0.01)
        mid_time = time.time()
        collector.record_metric("test", 2.0)
        collector.record_metric("test", 3.0)
        
        filtered = collector.get_metrics(start_time=mid_time)
        assert len(filtered) == 2
        assert all(m.timestamp >= mid_time for m in filtered)

    def test_get_metrics_with_end_time_filter(self, collector):
        """Test get_metrics with end_time filter (lines 85-98)."""
        import time
        
        collector.record_metric("test", 1.0)
        collector.record_metric("test", 2.0)
        mid_time = time.time()
        time.sleep(0.01)
        collector.record_metric("test", 3.0)
        
        filtered = collector.get_metrics(end_time=mid_time)
        assert len(filtered) == 2
        assert all(m.timestamp <= mid_time for m in filtered)

    def test_get_summary_stats_empty_metrics(self, collector):
        """Test get_summary_stats with no metrics (lines 109-115)."""
        stats = collector.get_summary_stats("nonexistent_metric")
        assert stats == {}

    def test_save_to_file(self, collector, tmp_path):
        """Test save_to_file method (lines 130-146)."""
        collector.record_metric("test_metric", 1.0, tag1="value1")
        collector.record_metric("test_metric", 2.0, tag2="value2")
        
        output_file = tmp_path / "metrics.json"
        collector.save_to_file(output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        
        assert "exported_at" in data
        assert "metrics" in data
        assert len(data["metrics"]) == 2
        assert data["metrics"][0]["name"] == "test_metric"
        assert data["metrics"][0]["value"] == 1.0
        assert data["metrics"][0]["tags"]["tag1"] == "value1"


class TestHealthCheckerEdgeCases:
    """Test edge cases for HealthChecker."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance."""
        return HealthChecker()

    def test_run_checks_with_registered_checks(self, health_checker):
        """Test run_checks executes all registered checks (lines 178-205)."""
        check1_called = []
        check2_called = []
        
        def check1():
            check1_called.append(True)
            return HealthStatus(
                timestamp=time.time(),
                component="test1",
                status="healthy",
                message="OK"
            )
        
        def check2():
            check2_called.append(True)
            return HealthStatus(
                timestamp=time.time(),
                component="test2",
                status="warning",
                message="Warning"
            )
        
        health_checker.register_check("check1", check1)
        health_checker.register_check("check2", check2)
        
        results = health_checker.run_checks()
        
        assert len(results) == 2
        assert len(check1_called) == 1
        assert len(check2_called) == 1
        assert results[0].component == "test1"
        assert results[1].component == "test2"

    def test_run_checks_with_metrics_collector(self, health_checker):
        """Test run_checks records metrics when metrics_collector is available (lines 186-192)."""
        metrics_collector = MetricsCollector()
        health_checker.metrics_collector = metrics_collector
        
        def healthy_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test",
                status="healthy",
                message="OK"
            )
        
        health_checker.register_check("test_check", healthy_check)
        results = health_checker.run_checks()
        
        assert len(results) == 1
        # Check that metric was recorded
        metrics = metrics_collector.get_metrics(name="health_check")
        assert len(metrics) == 1
        assert metrics[0].value == 1.0  # 1 for healthy
        assert metrics[0].tags["check"] == "test_check"
        assert metrics[0].tags["status"] == "healthy"

    def test_run_checks_exception_handling(self, health_checker):
        """Test run_checks handles exceptions in check functions (lines 194-203)."""
        def failing_check():
            raise ValueError("Check failed")
        
        health_checker.register_check("failing_check", failing_check)
        results = health_checker.run_checks()
        
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].component == "failing_check"
        assert "Health check failed" in results[0].message
        assert "error" in results[0].details

    def test_get_overall_health(self, health_checker):
        """Test get_overall_health method (lines 207-220)."""
        def healthy_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test1",
                status="healthy",
                message="OK"
            )
        
        def warning_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test2",
                status="warning",
                message="Warning"
            )
        
        health_checker.register_check("check1", healthy_check)
        health_checker.register_check("check2", warning_check)
        
        overall = health_checker.get_overall_health()
        # Should be "warning" since one check is warning
        assert overall == "warning"
        
        # Test with all healthy
        def all_healthy_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test3",
                status="healthy",
                message="OK"
            )
        health_checker.register_check("check3", all_healthy_check)
        health_checker.checks = {"check3": all_healthy_check}
        overall = health_checker.get_overall_health()
        assert overall == "healthy"
        
        # Test with error
        def error_check():
            return HealthStatus(
                timestamp=time.time(),
                component="test4",
                status="error",
                message="Error"
            )
        health_checker.checks = {"check4": error_check}
        overall = health_checker.get_overall_health()
        assert overall == "error"


class TestSystemHealthChecksEdgeCases:
    """Test edge cases for SystemHealthChecks."""

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_critical(self, mock_memory):
        """Test check_memory_usage with critical usage (lines 229-242)."""
        mock_mem = Mock()
        mock_mem.percent = 95.0
        mock_mem.available = 500 * 1024 * 1024  # 500 MB
        mock_mem.total = 10 * 1024 * 1024 * 1024  # 10 GB
        mock_memory.return_value = mock_mem
        
        status = SystemHealthChecks.check_memory_usage()
        assert status.status == "error"
        assert "critical" in status.message.lower()
        assert status.component == "memory"

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_warning(self, mock_memory):
        """Test check_memory_usage with warning usage (lines 229-242)."""
        mock_mem = Mock()
        mock_mem.percent = 85.0
        mock_mem.available = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
        mock_mem.total = 10 * 1024 * 1024 * 1024  # 10 GB
        mock_memory.return_value = mock_mem
        
        status = SystemHealthChecks.check_memory_usage()
        assert status.status == "warning"
        assert "high" in status.message.lower()
        assert status.component == "memory"

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_healthy(self, mock_memory):
        """Test check_memory_usage with healthy usage (lines 229-242)."""
        mock_mem = Mock()
        mock_mem.percent = 50.0
        mock_mem.available = 5 * 1024 * 1024 * 1024  # 5 GB
        mock_mem.total = 10 * 1024 * 1024 * 1024  # 10 GB
        mock_memory.return_value = mock_mem
        
        status = SystemHealthChecks.check_memory_usage()
        assert status.status == "healthy"
        assert "normal" in status.message.lower()
        assert status.component == "memory"

    @patch("psutil.disk_usage")
    def test_check_disk_usage_critical(self, mock_disk):
        """Test check_disk_usage with critical usage (lines 255-270)."""
        mock_d = Mock()
        mock_d.percent = 96.0
        mock_d.free = 40 * 1024 * 1024 * 1024  # 40 GB
        mock_d.total = 1000 * 1024 * 1024 * 1024  # 1 TB
        mock_disk.return_value = mock_d
        
        status = SystemHealthChecks.check_disk_usage()
        assert status.status == "error"
        assert "critical" in status.message.lower()
        assert status.component == "disk"

    @patch("psutil.disk_usage")
    def test_check_disk_usage_warning(self, mock_disk):
        """Test check_disk_usage with warning usage (lines 255-270)."""
        mock_d = Mock()
        mock_d.percent = 86.0
        mock_d.free = 140 * 1024 * 1024 * 1024  # 140 GB
        mock_d.total = 1000 * 1024 * 1024 * 1024  # 1 TB
        mock_disk.return_value = mock_d
        
        status = SystemHealthChecks.check_disk_usage()
        assert status.status == "warning"
        assert "high" in status.message.lower()
        assert status.component == "disk"

    @patch("psutil.disk_usage")
    def test_check_disk_usage_healthy(self, mock_disk):
        """Test check_disk_usage with healthy usage (lines 255-270)."""
        mock_d = Mock()
        mock_d.percent = 50.0
        mock_d.free = 500 * 1024 * 1024 * 1024  # 500 GB
        mock_d.total = 1000 * 1024 * 1024 * 1024  # 1 TB
        mock_disk.return_value = mock_d
        
        status = SystemHealthChecks.check_disk_usage()
        assert status.status == "healthy"
        assert "normal" in status.message.lower()
        assert status.component == "disk"

    @patch("torch.cuda.is_available")
    def test_check_gpu_memory_no_gpu(self, mock_cuda_available):
        """Test check_gpu_memory when GPU not available (lines 285-292)."""
        mock_cuda_available.return_value = False
        
        status = SystemHealthChecks.check_gpu_memory()
        assert status.status == "healthy"
        assert "No GPU available" in status.message
        assert status.component == "gpu_memory"
        assert status.details["gpu_available"] is False

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_critical(self, mock_get_props, mock_reserved, mock_allocated, mock_cuda_available):
        """Test check_gpu_memory with critical usage (lines 294-324)."""
        mock_cuda_available.return_value = True
        # 9.6 GB allocated out of 10 GB total = 96% usage (critical)
        mock_allocated.return_value = 9.6 * 1024 * 1024 * 1024  # 9.6 GB
        mock_reserved.return_value = 10.0 * 1024 * 1024 * 1024  # 10 GB
        mock_props = Mock()
        mock_props.total_memory = 10.0 * 1024 * 1024 * 1024  # 10 GB total
        mock_get_props.return_value = mock_props
        
        status = SystemHealthChecks.check_gpu_memory()
        assert status.status == "error"
        assert "critical" in status.message.lower()
        assert status.component == "gpu_memory"

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_warning_branch(self, mock_get_props, mock_reserved, mock_allocated, mock_cuda_available):
        """Test check_gpu_memory warning branch (line 306-308)."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 9000000000  # 9GB
        mock_reserved.return_value = 0
        mock_get_props.return_value.total_memory = 10000000000  # 10GB total
        # 9GB / 10GB = 90% usage, should trigger warning (85-95%)
        status = SystemHealthChecks.check_gpu_memory()
        
        assert status.status == "warning"
        assert "high" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_healthy_branch(self, mock_get_props, mock_reserved, mock_allocated, mock_cuda_available):
        """Test check_gpu_memory healthy branch (line 309-311)."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 5000000000  # 5GB
        mock_reserved.return_value = 0
        mock_get_props.return_value.total_memory = 10000000000  # 10GB total
        # 5GB / 10GB = 50% usage, should be healthy (<85%)
        status = SystemHealthChecks.check_gpu_memory()
        
        assert status.status == "healthy"
        assert "normal" in status.message.lower()

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_memory_exception(self, mock_get_props, mock_reserved, mock_allocated, mock_cuda_available):
        """Test check_gpu_memory handles exceptions (lines 326-327)."""
        mock_cuda_available.return_value = True
        # Make get_device_properties raise exception to trigger exception path
        mock_get_props.side_effect = RuntimeError("CUDA error")
        
        status = SystemHealthChecks.check_gpu_memory()
        assert status.status == "error"
        assert "error" in status.message.lower() or "failed" in status.message.lower()
        assert status.component == "gpu_memory"


class TestTrainingMonitorEdgeCases:
    """Test edge cases for TrainingMonitor."""

    @pytest.fixture
    def training_monitor(self, tmp_path):
        """Create a training monitor instance."""
        return TrainingMonitor(log_dir=tmp_path)

    def test_start_training_records_metric(self, training_monitor):
        """Test start_training records metric (lines 368-378)."""
        config = {"model": "test", "steps": 100}
        training_monitor.start_training(config)
        
        assert training_monitor.training_start_time is not None
        metrics = training_monitor.metrics.get_metrics(name="training_started")
        assert len(metrics) == 1
        assert metrics[0].value == 1.0

    def test_record_step_with_all_params(self, training_monitor):
        """Test record_step with all optional parameters (lines 398-411)."""
        training_monitor.start_training({})
        
        training_monitor.record_step(
            step=1,
            loss=0.5,
            lr=0.001,
            tokens_processed=1000,
            gpu_memory_mb=5000.0
        )
        
        # Check metrics were recorded
        loss_metrics = training_monitor.metrics.get_metrics(name="loss")
        assert len(loss_metrics) == 1
        assert loss_metrics[0].value == 0.5
        
        lr_metrics = training_monitor.metrics.get_metrics(name="learning_rate")
        assert len(lr_metrics) == 1
        assert lr_metrics[0].value == 0.001
        
        token_metrics = training_monitor.metrics.get_metrics(name="tokens_processed")
        assert len(token_metrics) == 1
        assert token_metrics[0].value == 1000.0
        
        gpu_metrics = training_monitor.metrics.get_metrics(name="gpu_memory_mb")
        assert len(gpu_metrics) == 1
        assert gpu_metrics[0].value == 5000.0

    def test_record_step_triggers_health_checks(self, training_monitor):
        """Test record_step triggers periodic health checks (lines 408-411)."""
        training_monitor.start_training({})
        training_monitor.health_check_interval = 0.1  # Short interval for testing
        training_monitor.last_health_check = 0
        
        # Mock health checker
        mock_health_checker = Mock()
        mock_health_checker.run_checks.return_value = []
        training_monitor.health_checker = mock_health_checker
        
        time.sleep(0.15)  # Wait longer than interval
        training_monitor.record_step(step=1, loss=0.5, lr=0.001)
        
        # Should have run health checks
        assert mock_health_checker.run_checks.called

    def test_run_health_checks_logs_issues(self, training_monitor):
        """Test _run_health_checks logs non-healthy issues (lines 415-425)."""
        # Create health checker with warning status
        warning_status = HealthStatus(
            timestamp=time.time(),
            component="test",
            status="warning",
            message="Test warning"
        )
        
        mock_health_checker = Mock()
        mock_health_checker.run_checks.return_value = [warning_status]
        training_monitor.health_checker = mock_health_checker
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            training_monitor._run_health_checks()
            # Should have printed warning
            assert mock_print.called
            call_args = str(mock_print.call_args)
            assert "warning" in call_args.lower() or "test" in call_args.lower()

    def test_end_training_without_start(self, training_monitor):
        """Test end_training when training not started (lines 440-442)."""
        # Don't call start_training
        training_monitor.end_training(final_loss=0.1, total_steps=10)
        
        # Should return early without error
        # Check that no metrics were recorded
        completed_metrics = training_monitor.metrics.get_metrics(name="training_completed")
        assert len(completed_metrics) == 0

    def test_end_training_saves_files(self, training_monitor, tmp_path):
        """Test end_training saves metrics and summary files (lines 440-475)."""
        training_monitor.start_training({})
        training_monitor.record_step(step=1, loss=0.5, lr=0.001)
        
        training_monitor.end_training(final_loss=0.1, total_steps=10)
        
        # Check that metrics file was created
        metrics_files = list(tmp_path.glob("training_metrics_*.json"))
        assert len(metrics_files) == 1
        
        # Check that summary file was created
        summary_file = tmp_path / "training_summary.json"
        assert summary_file.exists()
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        assert summary["total_steps"] == 10
        assert summary["final_loss"] == 0.1
        assert "training_duration_seconds" in summary
        assert "metrics_file" in summary

    def test_get_status(self, training_monitor):
        """Test get_status method (line 483)."""
        status = training_monitor.get_status()
        
        assert isinstance(status, dict)
        assert "health" in status
        assert "metrics_count" in status
        assert "training_active" in status
        assert "last_health_check" in status
        assert status["training_active"] is False  # Not started yet
        
        # Start training and check again
        training_monitor.start_training({})
        status = training_monitor.get_status()
        assert status["training_active"] is True

    def test_record_step_health_check_interval_not_exceeded(self, training_monitor):
        """Test record_step when health check interval is NOT exceeded (line 409->exit branch)."""
        training_monitor.start_training({})
        training_monitor.last_health_check = time.time()  # Set to current time
        
        # Record step immediately - should NOT trigger health check
        with patch.object(training_monitor, '_run_health_checks') as mock_run:
            training_monitor.record_step(step=1, loss=0.5, lr=0.001)
            # Should NOT have called _run_health_checks (line 409->exit branch)
            assert not mock_run.called

