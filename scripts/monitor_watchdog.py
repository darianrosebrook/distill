"""
Watchdog daemon monitoring script.

Monitors watchdog daemon status and detects blocked threads that could
cause kernel panics due to watchdog timeouts.

@author: @darianrosebrook
"""

import argparse
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class WatchdogMonitor:
    """Monitor watchdog daemon and detect blocked threads."""

    def __init__(self, check_interval: float = 1.0, alert_threshold: int = 50):
        """
        Initialize watchdog monitor.

        Args:
            check_interval: How often to check watchdog status (seconds)
            alert_threshold: Alert if watchdog time remaining is below this (seconds)
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def _get_watchdog_status(self) -> Dict[str, Any]:
        """
        Get current watchdog daemon status.

        Returns:
            Dictionary with watchdog status information
        """
        status = {
            "timestamp": time.time(),
            "platform": sys.platform,
        }

        # macOS-specific watchdog monitoring
        if sys.platform == "darwin":
            try:
                # Check for watchdogd process
                result = subprocess.run(
                    ["pgrep", "-f", "watchdogd"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                status["watchdogd_running"] = result.returncode == 0

                # Try to get watchdog timeout (may require root)
                try:
                    result = subprocess.run(
                        ["sysctl", "kern.watchdog.period"],
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )
                    if result.returncode == 0:
                        # Parse output like "kern.watchdog.period: 60"
                        parts = result.stdout.strip().split(":")
                        if len(parts) == 2:
                            status["watchdog_period"] = int(parts[1].strip())
                except (subprocess.TimeoutExpired, ValueError, IndexError):
                    status["watchdog_period"] = None

            except Exception as e:
                status["error"] = str(e)

        # Check for blocked threads (simplified - would need more sophisticated detection)
        try:
            import psutil

            # Get all Python processes
            python_processes = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if "python" in proc.info["name"].lower():
                        python_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            status["python_processes"] = len(python_processes)

            # Check for processes that might be blocking
            # This is a simplified check - real detection would need more analysis
            blocking_candidates = []
            for proc_info in python_processes:
                try:
                    proc = psutil.Process(proc_info["pid"])
                    # Check if process has been in same state for a long time
                    # This is a heuristic, not definitive
                    if proc.status() == psutil.STATUS_SLEEPING:
                        # Could be blocked in I/O wait
                        blocking_candidates.append(proc_info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            status["blocking_candidates"] = blocking_candidates

        except ImportError:
            status["psutil_available"] = False
        except Exception as e:
            status["error"] = str(e)

        return status

    def _monitor_loop(self):
        """Main monitoring loop."""
        last_status = None
        consecutive_alerts = 0

        while self.is_monitoring:
            try:
                status = self._get_watchdog_status()

                # Check for watchdog timeout approaching
                if status.get("watchdog_period"):
                    # Estimate time remaining (simplified - would need actual watchdog state)
                    # This is a placeholder - real implementation would track watchdog state
                    pass

                # Alert if we detect potential blocking
                if status.get("blocking_candidates"):
                    consecutive_alerts += 1
                    if consecutive_alerts >= 3:
                        print(
                            f"\n[ALERT] Potential blocked threads detected: "
                            f"{status['blocking_candidates']}"
                        )
                        print(
                            f"[ALERT] This could cause watchdog timeout. "
                            f"Consider terminating blocking processes."
                        )
                else:
                    consecutive_alerts = 0

                last_status = status
                time.sleep(self.check_interval)

            except Exception as e:
                print(f"[ERROR] Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def start(self):
        """Start monitoring."""
        if self.is_monitoring:
            print("[WARNING] Monitor is already running")
            return

        print("[INFO] Starting watchdog monitor...")
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[INFO] Monitor started (check interval: {self.check_interval}s)")

    def stop(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return

        print("[INFO] Stopping watchdog monitor...")
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("[INFO] Monitor stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current watchdog status."""
        return self._get_watchdog_status()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor watchdog daemon and detect blocked threads"
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="Check interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=int,
        default=50,
        help="Alert threshold in seconds (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Monitor for specified duration in seconds (default: run until interrupted)",
    )
    parser.add_argument(
        "--output",
        help="Output file for status (JSON format)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (don't monitor continuously)",
    )

    args = parser.parse_args()

    monitor = WatchdogMonitor(
        check_interval=args.check_interval, alert_threshold=args.alert_threshold
    )

    if args.once:
        # Single check
        status = monitor.get_status()
        print(json.dumps(status, indent=2, default=str))

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
            print(f"\n[INFO] Status saved to: {output_path}")

        return

    # Continuous monitoring
    try:
        monitor.start()

        if args.duration:
            # Monitor for specified duration
            time.sleep(args.duration)
            monitor.stop()
        else:
            # Monitor until interrupted
            print("[INFO] Monitoring... Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user")
                monitor.stop()

        # Save final status if output specified
        if args.output:
            status = monitor.get_status()
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
            print(f"\n[INFO] Final status saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        monitor.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()


