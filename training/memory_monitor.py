"""
Memory and swap monitoring for training loops.

Logs memory usage, swap usage, and PyTorch device memory at regular intervals
to detect memory leaks, swap thrashing, and resource constraints.

Usage:
    monitor = MemoryMonitor(log_interval=10)
    for step in range(total_steps):
        # ... training step ...
        monitor.log_step(step)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MemoryMonitor:
    """Monitor system and PyTorch memory usage during training."""

    def __init__(
        self,
        log_interval: int = 10,
        output_path: Optional[str] = None,
        red_zone_mem_percent: float = 90.0,
        red_zone_swap_gb: float = 5.0,
    ):
        """
        Initialize memory monitor.

        Args:
            log_interval: Log memory every N steps
            output_path: Path to JSONL file for logging (if None, no file output)
            red_zone_mem_percent: Warn if memory usage exceeds this percentage
            red_zone_swap_gb: Warn if swap usage exceeds this GB
        """
        self.log_interval = log_interval
        self.output_path = Path(output_path) if output_path else None
        self.red_zone_mem_percent = red_zone_mem_percent
        self.red_zone_swap_gb = red_zone_swap_gb
        
        self.metrics = []
        
        if not PSUTIL_AVAILABLE:
            print("[MemoryMonitor] WARN: psutil not available, memory monitoring disabled")
            print("[MemoryMonitor] Install with: pip install psutil")

    def get_system_memory(self) -> Dict[str, Any]:
        """Get system memory and swap usage."""
        if not PSUTIL_AVAILABLE:
            return {
                "mem_total_gb": 0.0,
                "mem_used_gb": 0.0,
                "mem_percent": 0.0,
                "swap_total_gb": 0.0,
                "swap_used_gb": 0.0,
                "swap_percent": 0.0,
            }
        
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "mem_total_gb": vm.total / (1024 ** 3),
            "mem_used_gb": vm.used / (1024 ** 3),
            "mem_available_gb": vm.available / (1024 ** 3),
            "mem_percent": vm.percent,
            "swap_total_gb": swap.total / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3),
            "swap_percent": swap.percent,
        }

    def get_torch_memory(self, device: Optional[Any] = None) -> Dict[str, Any]:
        """Get PyTorch device memory usage."""
        if not TORCH_AVAILABLE:
            return {}
        
        metrics = {}
        
        # CPU memory (always available)
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()
            metrics["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            metrics["cuda_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)
            metrics["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        # MPS memory (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have direct memory query API, but we can estimate
            # from allocated tensors
            try:
                # Get all MPS tensors and sum their memory
                mps_allocated = 0
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.device.type == "mps":
                        mps_allocated += obj.numel() * obj.element_size()
                metrics["mps_allocated_gb"] = mps_allocated / (1024 ** 3)
            except Exception:
                pass
        
        return metrics

    def log_step(self, step: int, device: Optional[Any] = None) -> Dict[str, Any]:
        """
        Log memory metrics for current step.

        Args:
            step: Current training step
            device: PyTorch device (optional, for device-specific metrics)

        Returns:
            Dict with memory metrics
        """
        if step % self.log_interval != 0:
            return {}
        
        timestamp = time.time()
        
        # Get system memory
        sys_mem = self.get_system_memory()
        
        # Get PyTorch memory
        torch_mem = self.get_torch_memory(device)
        
        # Combine metrics
        metrics = {
            "step": step,
            "timestamp": timestamp,
            **sys_mem,
            **torch_mem,
        }
        
        # Check red zones
        warnings = []
        if sys_mem["mem_percent"] > self.red_zone_mem_percent:
            warnings.append(f"High memory usage: {sys_mem['mem_percent']:.1f}%")
        
        if sys_mem["swap_used_gb"] > self.red_zone_swap_gb:
            warnings.append(f"High swap usage: {sys_mem['swap_used_gb']:.2f} GB")
        
        if warnings:
            metrics["warnings"] = warnings
            print(f"[MemoryMonitor] Step {step}: {'; '.join(warnings)}")
        
        # Store metrics
        self.metrics.append(metrics)
        
        # Write to file if output path specified
        if self.output_path:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged metrics."""
        if not self.metrics:
            return {}
        
        mem_percents = [m.get("mem_percent", 0) for m in self.metrics]
        swap_used = [m.get("swap_used_gb", 0) for m in self.metrics]
        
        return {
            "total_logged_steps": len(self.metrics),
            "avg_mem_percent": sum(mem_percents) / len(mem_percents) if mem_percents else 0,
            "max_mem_percent": max(mem_percents) if mem_percents else 0,
            "avg_swap_gb": sum(swap_used) / len(swap_used) if swap_used else 0,
            "max_swap_gb": max(swap_used) if swap_used else 0,
            "red_zone_warnings": sum(1 for m in self.metrics if "warnings" in m),
        }

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        if not summary:
            print("[MemoryMonitor] No metrics logged yet")
            return
        
        print("\n=== Memory Monitor Summary ===")
        print(f"Logged steps: {summary['total_logged_steps']}")
        print(f"Avg memory usage: {summary['avg_mem_percent']:.1f}%")
        print(f"Max memory usage: {summary['max_mem_percent']:.1f}%")
        print(f"Avg swap usage: {summary['avg_swap_gb']:.2f} GB")
        print(f"Max swap usage: {summary['max_swap_gb']:.2f} GB")
        print(f"Red zone warnings: {summary['red_zone_warnings']}")


# Import gc for MPS memory tracking
import gc

