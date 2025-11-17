"""
Monitoring and observability system for training.

Provides metrics collection, health checks, and alerting capabilities.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import torch


@dataclass
class MetricPoint:
    """Single metric measurement."""

    timestamp: float
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""

    timestamp: float
    component: str
    status: str  # "healthy", "warning", "error"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collect and store training metrics."""

    def __init__(self, max_points: int = 10000):
        """Initialize metrics collector.

        Args:
            max_points: Maximum number of metric points to store
        """
        self.metrics: List[MetricPoint] = []
        self.max_points = max_points
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float, **tags) -> None:
        """Record a metric measurement.

        Args:
            name: Metric name
            value: Metric value
            **tags: Additional tags for the metric
        """
        point = MetricPoint(timestamp=time.time(), name=name, value=value, tags=tags)

        with self.lock:
            self.metrics.append(point)

            # Maintain max size
            if len(self.metrics) > self.max_points:
                self.metrics = self.metrics[-self.max_points :]

    def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[MetricPoint]:
        """Get metrics with optional filtering.

        Args:
            name: Filter by metric name
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp

        Returns:
            List of matching metric points
        """
        with self.lock:
            metrics = self.metrics.copy()

        # Apply filters
        if name:
            metrics = [m for m in metrics if m.name == name]

        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]

        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        return metrics

    def get_summary_stats(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_metrics(name=name)
        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "first": values[0],
        }

    # Convenience aliases for test compatibility
    def add_metric(self, point: MetricPoint) -> None:
        """Add a metric point (alias for record_metric compatibility).

        Args:
            point: MetricPoint to add
        """
        with self.lock:
            self.metrics.append(point)
            if len(self.metrics) > self.max_points:
                self.metrics = self.metrics[-self.max_points :]

    def get_metrics_by_name(self, name: str) -> List[MetricPoint]:
        """Get all metrics with a specific name.

        Args:
            name: Metric name to filter by

        Returns:
            List of matching metric points
        """
        return self.get_metrics(name=name)

    def get_metrics_by_tags(self, tags: Dict[str, str]) -> List[MetricPoint]:
        """Get metrics matching specific tags.

        Args:
            tags: Dictionary of tags to match

        Returns:
            List of matching metric points
        """
        with self.lock:
            metrics = self.metrics.copy()

        # Filter by all tags
        for key, value in tags.items():
            metrics = [m for m in metrics if m.tags.get(key) == value]

        return metrics

    def get_latest_metric(self, name: str) -> Optional[MetricPoint]:
        """Get the latest metric with a specific name.

        Args:
            name: Metric name

        Returns:
            Latest MetricPoint or None if not found
        """
        metrics = self.get_metrics(name=name)
        return metrics[-1] if metrics else None

    def get_metric_statistics(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric (alias for get_summary_stats).

        Args:
            name: Metric name

        Returns:
            Dictionary with statistics
        """
        return self.get_summary_stats(name)

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self.lock:
            self.metrics = []

    def save_to_file(self, output_path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            output_path: Path to save metrics
        """
        with self.lock:
            data = {
                "exported_at": time.time(),
                "metrics": [
                    {
                        "timestamp": m.timestamp,
                        "name": m.name,
                        "value": m.value,
                        "tags": m.tags,
                    }
                    for m in self.metrics
                ],
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class HealthChecker:
    """Monitor system and training health."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize health checker.

        Args:
            metrics_collector: Optional metrics collector for health metrics
        """
        self.metrics_collector = metrics_collector
        self.checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.status_history: List[HealthStatus] = []
        self.last_check_time = 0
        self.check_interval = 60  # Check every 60 seconds

    def register_check(self, name: str, check_func: Callable[[], HealthStatus]) -> None:
        """Register a health check function.

        Args:
            name: Check name
            check_func: Function that returns HealthStatus
        """
        self.checks[name] = check_func

    def add_check(self, name: str, check_func: Callable[[], HealthStatus]) -> None:
        """Add a health check function (alias for register_check).

        Args:
            name: Check name
            check_func: Function that returns HealthStatus
        """
        self.register_check(name, check_func)

    def run_checks(self) -> List[HealthStatus]:
        """Run all registered health checks.

        Returns:
            List of health status results
        """
        results = []

        for name, check_func in self.checks.items():
            try:
                status = check_func()
                results.append(status)
                # Track in history
                self.status_history.append(status)

                # Record as metric
                if self.metrics_collector:
                    self.metrics_collector.record_metric(
                        "health_check",
                        1 if status.status == "healthy" else 0,
                        check=name,
                        status=status.status,
                    )

            except Exception as e:
                # Check function itself failed
                error_status = HealthStatus(
                    timestamp=time.time(),
                    component=name,
                    status="error",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                )
                results.append(error_status)
                self.status_history.append(error_status)

        self.last_check_time = time.time()
        return results

    def run_check(self, name: str) -> HealthStatus:
        """Run a single health check by name.

        Args:
            name: Name of the check to run

        Returns:
            HealthStatus result

        Raises:
            KeyError: If check name not found
        """
        if name not in self.checks:
            raise KeyError(f"Health check '{name}' not found")

        check_func = self.checks[name]
        try:
            status = check_func()
            self.status_history.append(status)

            # Record as metric
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    "health_check",
                    1 if status.status == "healthy" else 0,
                    check=name,
                    status=status.status,
                )

            self.last_check_time = time.time()
            return status

        except Exception as e:
            error_status = HealthStatus(
                timestamp=time.time(),
                component=name,
                status="error",
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )
            self.status_history.append(error_status)
            return error_status

    def run_all_checks(self) -> List[HealthStatus]:
        """Run all registered health checks (alias for run_checks).

        Returns:
            List of health status results
        """
        return self.run_checks()

    def get_component_status(self, component: str) -> Optional[HealthStatus]:
        """Get the latest health status for a specific component.

        Args:
            component: Component name to get status for

        Returns:
            Latest HealthStatus for component, or None if not found
        """
        # Search history in reverse to find most recent
        for status in reversed(self.status_history):
            if status.component == component:
                return status
        return None

    def get_overall_health(self) -> str:
        """Get overall system health status.

        Returns:
            Overall health status: "healthy", "warning", or "error"
        """
        results = self.run_checks()

        if any(r.status == "error" for r in results):
            return "error"
        elif any(r.status == "warning" for r in results):
            return "warning"
        else:
            return "healthy"


class SystemHealthChecks:
    """Standard system health checks."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize system health checks.

        Args:
            metrics_collector: Optional metrics collector for health metrics
        """
        self.health_checker = HealthChecker(metrics_collector)
        # Register all system checks
        self.health_checker.add_check("cpu", self._check_cpu_usage)
        self.health_checker.add_check("memory", self._check_memory_usage)
        self.health_checker.add_check("gpu", self._check_gpu_usage)
        self.health_checker.add_check("disk", self._check_disk_usage)
        self.health_checker.add_check("network", self._check_network_connectivity)

    def _check_cpu_usage(self) -> HealthStatus:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent > 90:
            status = "error"
            message = f"CPU usage critical: {cpu_percent:.1f}%"
        elif cpu_percent > 80:
            status = "warning"
            message = f"CPU usage high: {cpu_percent:.1f}%"
        else:
            status = "healthy"
            message = f"CPU usage normal: {cpu_percent:.1f}%"

        return HealthStatus(
            timestamp=time.time(),
            component="cpu",
            status=status,
            message=message,
            details={"usage_percent": cpu_percent},
        )

    def _check_memory_usage(self) -> HealthStatus:
        """Check system memory usage."""
        return self.check_memory_usage()

    def _check_gpu_usage(self) -> HealthStatus:
        """Check GPU usage."""
        if not torch.cuda.is_available():
            return HealthStatus(
                timestamp=time.time(),
                component="gpu",
                status="healthy",
                message="GPU not available",
                details={"gpu_available": False},
            )

        try:
            # Try to get GPU utilization and memory info
            utilization = 0
            try:
                # torch.cuda.utilization() may not be available on all systems
                if hasattr(torch.cuda, 'utilization'):
                    utilization = torch.cuda.utilization()
            except:
                pass

            mem_info = torch.cuda.mem_get_info(0)
            used_gb = (mem_info[1] - mem_info[0]) / 1024**3
            total_gb = mem_info[1] / 1024**3
            usage_percent = (used_gb / total_gb * 100) if total_gb > 0 else 0

            # Use utilization if available and > 0, otherwise use memory usage
            display_percent = utilization if utilization > 0 else usage_percent
            
            # Format as integer if whole number, otherwise 1 decimal place
            if display_percent == int(display_percent):
                percent_str = f"{int(display_percent)}%"
            else:
                percent_str = f"{display_percent:.1f}%"
            
            if display_percent > 95:
                status = "error"
                message = f"GPU usage critical: {percent_str}"
            elif display_percent > 85:
                status = "warning"
                message = f"GPU usage high: {percent_str}"
            else:
                status = "healthy"
                message = f"GPU usage normal: {percent_str}"

            return HealthStatus(
                timestamp=time.time(),
                component="gpu",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "utilization": utilization,
                    "used_gb": used_gb,
                    "total_gb": total_gb,
                },
            )
        except Exception as e:
            return HealthStatus(
                timestamp=time.time(),
                component="gpu",
                status="warning",
                message=f"GPU check failed: {e}",
                details={"error": str(e)},
            )

    def _check_disk_usage(self) -> HealthStatus:
        """Check disk usage."""
        return self.check_disk_usage()

    def _check_network_connectivity(self) -> HealthStatus:
        """Check network connectivity."""
        try:
            import socket
            # Try to connect to a well-known host
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return HealthStatus(
                timestamp=time.time(),
                component="network",
                status="healthy",
                message="Network connectivity OK",
                details={"connected": True},
            )
        except Exception:
            return HealthStatus(
                timestamp=time.time(),
                component="network",
                status="warning",
                message="Network connectivity check failed",
                details={"connected": False},
            )

    def run_system_checks(self) -> List[HealthStatus]:
        """Run all system health checks.

        Returns:
            List of health status results
        """
        return self.health_checker.run_all_checks()

    @staticmethod
    def check_memory_usage() -> HealthStatus:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        if usage_percent > 90:
            status = "error"
            message = f"Memory usage critical: {usage_percent:.1f}%"
        elif usage_percent > 80:
            status = "warning"
            message = f"Memory usage high: {usage_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Memory usage normal: {usage_percent:.1f}%"

        return HealthStatus(
            timestamp=time.time(),
            component="memory",
            status=status,
            message=message,
            details={
                "usage_percent": usage_percent,
                "available_mb": memory.available / 1024 / 1024,
                "total_mb": memory.total / 1024 / 1024,
            },
        )

    @staticmethod
    def check_disk_usage() -> HealthStatus:
        """Check disk usage."""
        disk = psutil.disk_usage("/")
        usage_percent = disk.percent

        if usage_percent > 95:
            status = "error"
            message = f"Disk usage critical: {usage_percent:.1f}%"
        elif usage_percent > 85:
            status = "warning"
            message = f"Disk usage high: {usage_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Disk usage normal: {usage_percent:.1f}%"

        return HealthStatus(
            timestamp=time.time(),
            component="disk",
            status=status,
            message=message,
            details={
                "usage_percent": usage_percent,
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "total_gb": disk.total / 1024 / 1024 / 1024,
            },
        )

    @staticmethod
    def check_gpu_memory() -> HealthStatus:
        """Check GPU memory usage."""
        if not torch.cuda.is_available():
            return HealthStatus(
                timestamp=time.time(),
                component="gpu_memory",
                status="healthy",
                message="No GPU available",
                details={"gpu_available": False},
            )

        try:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB

            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024

            usage_percent = (allocated / total_memory) * 100 if total_memory > 0 else 0

            if usage_percent > 95:
                status = "error"
                message = f"GPU memory critical: {usage_percent:.1f}%"
            elif usage_percent > 85:
                status = "warning"
                message = f"GPU memory high: {usage_percent:.1f}%"
            else:
                status = "healthy"
                message = f"GPU memory normal: {usage_percent:.1f}%"

            return HealthStatus(
                timestamp=time.time(),
                component="gpu_memory",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total_memory,
                },
            )

        except Exception as e:
            return HealthStatus(
                timestamp=time.time(),
                component="gpu_memory",
                status="warning",
                message=f"GPU memory check failed: {e}",
                details={"error": str(e)},
            )


class TrainingMonitor:
    """Comprehensive training monitoring system."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize training monitor.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics)

        # Register standard health checks
        self.health_checker.register_check("memory", SystemHealthChecks.check_memory_usage)
        self.health_checker.register_check("disk", SystemHealthChecks.check_disk_usage)
        self.health_checker.register_check("gpu_memory", SystemHealthChecks.check_gpu_memory)

        # Monitoring state
        self.training_start_time = None
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes

    def start_training(self, config: Dict[str, Any]) -> None:
        """Start training monitoring.

        Args:
            config: Training configuration
        """
        self.training_start_time = time.time()

        # Record training start
        self.metrics.record_metric(
            "training_started",
            1,
            config_hash=hash(str(config)),
            timestamp=datetime.now().isoformat(),
        )

        print(f"📊 Training monitoring started at {datetime.now().isoformat()}")

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_processed: Optional[int] = None,
        gpu_memory_mb: Optional[float] = None,
    ) -> None:
        """Record training step metrics.

        Args:
            step: Current training step
            loss: Training loss
            lr: Learning rate
            tokens_processed: Total tokens processed
            gpu_memory_mb: GPU memory usage in MB
        """
        # Record metrics
        self.metrics.record_metric("loss", loss, step=step)
        self.metrics.record_metric("learning_rate", lr, step=step)

        if tokens_processed is not None:
            self.metrics.record_metric("tokens_processed", tokens_processed, step=step)

        if gpu_memory_mb is not None:
            self.metrics.record_metric("gpu_memory_mb", gpu_memory_mb, step=step)

        # Periodic health checks
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            self._run_health_checks()
            self.last_health_check = current_time

    def _run_health_checks(self) -> None:
        """Run periodic health checks."""
        results = self.health_checker.run_checks()

        # Log any issues
        for result in results:
            if result.status != "healthy":
                print(
                    f"⚠️  Health check {result.component}: {result.status.upper()} - {result.message}"
                )

                # Record health issues as metrics
                self.metrics.record_metric(
                    "health_issue",
                    1,
                    component=result.component,
                    status=result.status,
                    message=result.message,
                )

    def end_training(self, final_loss: float, total_steps: int) -> None:
        """End training monitoring.

        Args:
            final_loss: Final training loss
            total_steps: Total training steps completed
        """
        if not self.training_start_time:
            return

        training_duration = time.time() - self.training_start_time

        # Record training end
        self.metrics.record_metric(
            "training_completed",
            1,
            duration_seconds=training_duration,
            final_loss=final_loss,
            total_steps=total_steps,
        )

        # Save final metrics
        metrics_file = self.log_dir / f"training_metrics_{int(time.time())}.json"
        self.metrics.save_to_file(metrics_file)

        # Generate summary
        summary = {
            "training_duration_seconds": training_duration,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "average_loss": self.metrics.get_summary_stats("loss").get("mean", 0),
            "health_status": self.health_checker.get_overall_health(),
            "metrics_file": str(metrics_file),
        }

        summary_file = self.log_dir / "training_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Training completed in {training_duration:.1f}s")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Metrics saved to: {metrics_file}")
        print(f"   Summary saved to: {summary_file}")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Dictionary with current status
        """
        return {
            "health": self.health_checker.get_overall_health(),
            "metrics_count": len(self.metrics.metrics),
            "training_active": self.training_start_time is not None,
            "last_health_check": self.last_health_check,
        }


# Global training monitor instance
training_monitor = TrainingMonitor()


def initialize_monitoring(log_dir: Optional[Path] = None) -> TrainingMonitor:
    """Initialize training monitoring system.

    Args:
        log_dir: Directory for monitoring logs

    Returns:
        Configured training monitor
    """
    global training_monitor
    training_monitor = TrainingMonitor(log_dir)
    return training_monitor


if __name__ == "__main__":
    # Example usage
    monitor = initialize_monitoring(Path("example_logs"))

    # Simulate training
    monitor.start_training({"model": "test", "steps": 100})

    for step in range(10):
        monitor.record_step(
            step=step,
            loss=1.0 / (step + 1),  # Decreasing loss
            lr=0.001 * (0.9**step),  # Decreasing LR
            tokens_processed=(step + 1) * 1000,
        )
        time.sleep(0.1)  # Simulate training time

    monitor.end_training(final_loss=0.1, total_steps=10)

    print(f"Final status: {monitor.get_status()}")
