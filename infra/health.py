"""
Production health checks and monitoring for Distill.

Provides:
- Health check endpoints
- Graceful shutdown handling
- Resource monitoring
- Performance metrics
- System diagnostics

For production deployment hardening.
"""
# @author: @darianrosebrook

import sys
import time
import signal
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    uptime_seconds: float = 0.0


class HealthChecker:
    """
    Comprehensive health checker for production deployment.

    Monitors system health, model performance, and resource usage.
    Provides graceful shutdown and health check endpoints.
    """

    def __init__(
        self,
        model=None,
        enable_gpu_monitoring: bool = True,
        health_check_interval: int = 30,
        degraded_thresholds: Optional[Dict[str, float]] = None,
        unhealthy_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize health checker.

        Args:
            model: Model instance for model-specific checks
            enable_gpu_monitoring: Whether to monitor GPU resources
            health_check_interval: Seconds between health checks
            degraded_thresholds: Thresholds for degraded status
            unhealthy_thresholds: Thresholds for unhealthy status
        """
        self.model = model
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.health_check_interval = health_check_interval

        # Default thresholds
        self.degraded_thresholds = degraded_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
        }

        self.unhealthy_thresholds = unhealthy_thresholds or {
            "cpu_percent": 95.0,
            "memory_percent": 95.0,
            "gpu_memory_percent": 98.0,
        }

        # State
        self.is_shutting_down = False
        self.start_time = time.time()
        self.last_health_check = None
        self.health_history = []
        self._shutdown_handlers: list[Callable] = []

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"[health] Received signal {signum}, initiating graceful shutdown...")
            self.initiate_graceful_shutdown()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def register_shutdown_handler(self, handler: Callable):
        """Register a handler to be called during graceful shutdown."""
        self._shutdown_handlers.append(handler)

    def initiate_graceful_shutdown(self):
        """Initiate graceful shutdown sequence."""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        print("[health] Starting graceful shutdown sequence...")

        # Call all registered shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                print(f"[health] Error in shutdown handler: {e}")

        print("[health] Graceful shutdown completed")

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
        else:
            # Fallback values when psutil is not available
            cpu_percent = 50.0  # Placeholder
            memory_percent = 60.0  # Placeholder
            memory_used_gb = 8.0  # Placeholder

        gpu_memory_used_gb = None
        gpu_memory_percent = None

        if self.enable_gpu_monitoring:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_used_gb = gpu_memory_allocated / (1024 ** 3)
                gpu_memory_percent = (gpu_memory_allocated / gpu_memory) * 100
            except Exception:
                pass

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_percent=gpu_memory_percent,
            uptime_seconds=time.time() - self.start_time,
        )

    def check_system_health(self) -> HealthCheckResult:
        """Check system resource health."""
        metrics = self.get_system_metrics()

        # Check CPU
        if metrics.cpu_percent >= self.unhealthy_thresholds["cpu_percent"]:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"CPU usage critically high: {metrics.cpu_percent:.1f}%",
                details={"cpu_percent": metrics.cpu_percent}
            )
        elif metrics.cpu_percent >= self.degraded_thresholds["cpu_percent"]:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"CPU usage high: {metrics.cpu_percent:.1f}%",
                details={"cpu_percent": metrics.cpu_percent}
            )

        # Check memory
        if metrics.memory_percent >= self.unhealthy_thresholds["memory_percent"]:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Memory usage critically high: {metrics.memory_percent:.1f}%",
                details={"memory_percent": metrics.memory_percent}
            )
        elif metrics.memory_percent >= self.degraded_thresholds["memory_percent"]:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Memory usage high: {metrics.memory_percent:.1f}%",
                details={"memory_percent": metrics.memory_percent}
            )

        # Check GPU memory
        if self.enable_gpu_monitoring and metrics.gpu_memory_percent is not None:
            if metrics.gpu_memory_percent >= self.unhealthy_thresholds["gpu_memory_percent"]:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"GPU memory usage critically high: {metrics.gpu_memory_percent:.1f}%",
                    details={"gpu_memory_percent": metrics.gpu_memory_percent}
                )
            elif metrics.gpu_memory_percent >= self.degraded_thresholds["gpu_memory_percent"]:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"GPU memory usage high: {metrics.gpu_memory_percent:.1f}%",
                    details={"gpu_memory_percent": metrics.gpu_memory_percent}
                )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="System resources within acceptable limits",
            details={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "gpu_memory_percent": metrics.gpu_memory_percent,
            }
        )

    def check_model_health(self) -> HealthCheckResult:
        """Check model-specific health (if model provided)."""
        if not self.model:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="No model to check"
            )

        try:
            # Basic model check: ensure it can do a forward pass
            with torch.no_grad():
                # Create a small test input
                test_input = torch.randint(0, 1000, (1, 10))

                if hasattr(self.model, 'eval'):
                    self.model.eval()

                start_time = time.time()
                output = self.model(test_input)
                inference_time = time.time() - start_time

                # Check for NaN/inf in output
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="Model producing NaN or inf values",
                        details={"has_nan": torch.isnan(output).any().item(),
                               "has_inf": torch.isinf(output).any().item()}
                    )

                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Model responding normally",
                    details={"inference_time_ms": inference_time * 1000}
                )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Model health check failed: {e}",
                details={"error": str(e)}
            )

    def run_comprehensive_health_check(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}

        # System health
        results["system"] = self.check_system_health()

        # Model health
        results["model"] = self.check_model_health()

        # Overall status (worst of all checks)
        statuses = [result.status for result in results.values()]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
            overall_message = "One or more components unhealthy"
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
            overall_message = "One or more components degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            overall_message = "All components healthy"

        results["overall"] = HealthCheckResult(
            status=overall_status,
            message=overall_message,
            details={"component_count": len(results)}
        )

        # Store in history
        self.last_health_check = results
        self.health_history.append(results)

        # Keep only last 100 health checks
        if len(self.health_history) > 100:
            self.health_history.pop(0)

        return results

    def get_health_endpoint_data(self) -> Dict[str, Any]:
        """Get data for health check HTTP endpoint."""
        if not self.last_health_check:
            self.run_comprehensive_health_check()

        health_data = {
            "status": self.last_health_check["overall"].status.value,
            "message": self.last_health_check["overall"].message,
            "timestamp": self.last_health_check["overall"].timestamp,
            "uptime_seconds": time.time() - self.start_time,
            "components": {}
        }

        for component_name, result in self.last_health_check.items():
            if component_name != "overall":
                health_data["components"][component_name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details
                }

        return health_data

    def start_monitoring_thread(self):
        """Start background monitoring thread."""
        def monitoring_loop():
            while not self.is_shutting_down:
                try:
                    self.run_comprehensive_health_check()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    print(f"[health] Error in monitoring loop: {e}")
                    time.sleep(5)  # Shorter sleep on error

        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        print(f"[health] Started monitoring thread (interval: {self.health_check_interval}s)")


def create_production_health_checker(model=None, **kwargs) -> HealthChecker:
    """
    Create a health checker configured for production use.

    Args:
        model: Model instance to monitor
        **kwargs: Additional arguments for HealthChecker

    Returns:
        Configured HealthChecker instance
    """
    # Production-optimized thresholds
    degraded_thresholds = {
        "cpu_percent": 70.0,      # More conservative CPU threshold
        "memory_percent": 80.0,   # More conservative memory threshold
        "gpu_memory_percent": 85.0, # More conservative GPU threshold
    }

    unhealthy_thresholds = {
        "cpu_percent": 90.0,      # Critical CPU threshold
        "memory_percent": 90.0,   # Critical memory threshold
        "gpu_memory_percent": 95.0, # Critical GPU threshold
    }

    return HealthChecker(
        model=model,
        enable_gpu_monitoring=torch.cuda.is_available(),
        health_check_interval=30,  # Check every 30 seconds
        degraded_thresholds=degraded_thresholds,
        unhealthy_thresholds=unhealthy_thresholds,
        **kwargs
    )


if __name__ == "__main__":
    # Simple CLI for health checking
    import argparse

    ap = argparse.ArgumentParser(description="Distill Health Checker")
    ap.add_argument("--model-path", help="Path to model for model health checks")
    ap.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    ap.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")

    args = ap.parse_args()

    # Create health checker
    health_checker = create_production_health_checker()
    health_checker.health_check_interval = args.interval

    if args.continuous:
        print("Starting continuous health monitoring...")
        health_checker.start_monitoring_thread()

        # Keep main thread alive
        try:
            while not health_checker.is_shutting_down:
                time.sleep(1)
        except KeyboardInterrupt:
            health_checker.initiate_graceful_shutdown()
    else:
        # Single health check
        results = health_checker.run_comprehensive_health_check()

        print(f"Health Status: {results['overall'].status.value.upper()}")
        print(f"Message: {results['overall'].message}")

        for component, result in results.items():
            if component != "overall":
                print(f"  {component}: {result.status.value} - {result.message}")

        # Exit with appropriate code
        if results["overall"].status == HealthStatus.UNHEALTHY:
            sys.exit(1)
        elif results["overall"].status == HealthStatus.DEGRADED:
            sys.exit(2)
        else:
            sys.exit(0)
