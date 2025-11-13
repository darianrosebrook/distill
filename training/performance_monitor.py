"""
Performance monitoring and profiling utilities.

Provides memory usage tracking, timing, and performance metrics.
"""

import time
import psutil
import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    wall_time_seconds: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    tokens_processed: Optional[int] = None
    throughput_tokens_per_sec: Optional[float] = None


class PerformanceMonitor:
    """Monitor system and GPU performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_cpu_times = self.process.cpu_times()

    def get_metrics(self, tokens_processed: Optional[int] = None) -> PerformanceMetrics:
        """Get current performance metrics.

        Args:
            tokens_processed: Number of tokens processed since start

        Returns:
            PerformanceMetrics object
        """
        current_time = time.time()
        wall_time = current_time - self.start_time

        # CPU metrics
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        # GPU metrics
        gpu_memory_mb = None
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except Exception:
                gpu_memory_mb = None

        # Throughput
        throughput = None
        if tokens_processed and wall_time > 0:
            throughput = tokens_processed / wall_time

        return PerformanceMetrics(
            wall_time_seconds=wall_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            tokens_processed=tokens_processed,
            throughput_tokens_per_sec=throughput,
        )

    def reset(self) -> None:
        """Reset performance counters."""
        self.start_time = time.time()
        self.start_cpu_times = self.process.cpu_times()


@contextmanager
def time_operation(operation_name: str) -> Generator[None, None, None]:
    """Context manager to time an operation.

    Args:
        operation_name: Name of the operation being timed

    Example:
        with time_operation("model_forward"):
            output = model(input)
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"[PERF] {operation_name} took {elapsed:.3f}s")


class TrainingProfiler:
    """Profile training performance and resource usage."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize training profiler.

        Args:
            log_dir: Directory to save profiling data
        """
        self.log_dir = log_dir
        self.monitor = PerformanceMonitor()
        self.step_metrics = []
        self.start_time = time.time()

    def log_step(
        self, step: int, loss: float, lr: float, tokens_processed: Optional[int] = None
    ) -> None:
        """Log metrics for a training step.

        Args:
            step: Current training step
            loss: Current loss value
            lr: Current learning rate
            tokens_processed: Total tokens processed
        """
        metrics = self.monitor.get_metrics(tokens_processed)

        step_data = {
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "timestamp": time.time(),
            "metrics": {
                "wall_time_seconds": metrics.wall_time_seconds,
                "cpu_percent": metrics.cpu_percent,
                "memory_mb": metrics.memory_mb,
                "gpu_memory_mb": metrics.gpu_memory_mb,
                "tokens_processed": metrics.tokens_processed,
                "throughput_tokens_per_sec": metrics.throughput_tokens_per_sec,
            },
        }

        self.step_metrics.append(step_data)

        # Print summary
        throughput_str = (
            f"{metrics.throughput_tokens_per_sec:.1f} tok/s"
            if metrics.throughput_tokens_per_sec
            else "N/A"
        )
        gpu_str = f"{metrics.gpu_memory_mb:.1f}MB" if metrics.gpu_memory_mb else "N/A"

        print(
            f"[STEP {step}] Loss: {loss:.4f}, LR: {lr:.2e}, "
            f"Throughput: {throughput_str}, GPU: {gpu_str}"
        )

    def save_profile(self, output_path: Optional[Path] = None) -> None:
        """Save profiling data to file.

        Args:
            output_path: Path to save profile data
        """
        if not output_path and self.log_dir:
            output_path = self.log_dir / "training_profile.json"
        elif not output_path:
            return

        profile_data = {
            "total_training_time": time.time() - self.start_time,
            "total_steps": len(self.step_metrics),
            "step_metrics": self.step_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(profile_data, f, indent=2, default=str)

        print(f"[PROFILER] Profile saved to {output_path}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from profiling data.

        Returns:
            Dictionary with summary statistics
        """
        if not self.step_metrics:
            return {}

        losses = [m["loss"] for m in self.step_metrics]
        throughputs = [
            m["metrics"]["throughput_tokens_per_sec"]
            for m in self.step_metrics
            if m["metrics"]["throughput_tokens_per_sec"]
        ]

        return {
            "total_steps": len(self.step_metrics),
            "total_time_seconds": time.time() - self.start_time,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "avg_throughput_tok_per_sec": sum(throughputs) / len(throughputs)
            if throughputs
            else None,
            "peak_memory_mb": max(m["metrics"]["memory_mb"] for m in self.step_metrics),
        }


def profile_memory_usage(func_name: str = "operation") -> None:
    """Decorator to profile memory usage of a function.

    Args:
        func_name: Name to use in profiling output
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()

                print(f"[MEMORY] {func_name}:")
                print(f"  Start: {start_memory / 1024 / 1024:.1f}MB")
                print(f"  End: {end_memory / 1024 / 1024:.1f}MB")
                print(f"  Peak: {peak_memory / 1024 / 1024:.1f}MB")

            return result

        return wrapper

    return decorator
