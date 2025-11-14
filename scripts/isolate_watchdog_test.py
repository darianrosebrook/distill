"""
Test isolation script for identifying watchdog timeout issues.

Runs tests individually with monitoring and timeout detection to identify
which test is causing kernel panics due to watchdog timeouts.

@author: @darianrosebrook
"""

import argparse
import subprocess
import sys
import time
import signal
import os
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class TestIsolationRunner:
    """Run tests individually with monitoring and timeout detection."""

    def __init__(
        self,
        test_file: str,
        test_name: Optional[str] = None,
        timeout: int = 30,
        check_interval: float = 1.0,
        max_watchdog_time: int = 60,
    ):
        """
        Initialize test isolation runner.

        Args:
            test_file: Path to test file
            test_name: Optional specific test name to run
            timeout: Test timeout in seconds
            check_interval: How often to check test status (seconds)
            max_watchdog_time: Maximum time before watchdog timeout (seconds)
        """
        self.test_file = Path(test_file)
        self.test_name = test_name
        self.timeout = timeout
        self.check_interval = check_interval
        self.max_watchdog_time = max_watchdog_time
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.should_stop = False

    def _build_pytest_command(self) -> List[str]:
        """Build pytest command for running the test."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_file),
            "-v",
            f"--timeout={self.timeout}",
            "--tb=short",
        ]

        if self.test_name:
            # Use -k to filter by test name (more flexible than :: which requires full path)
            cmd.extend(["-k", self.test_name])

        return cmd

    def _monitor_process(self):
        """Monitor test process and system state."""
        while not self.should_stop and self.process:
            if self.process.poll() is not None:
                # Process has finished
                break

            elapsed = time.time() - (self.start_time or 0)

            # Check if we're approaching watchdog timeout
            if elapsed > (self.max_watchdog_time * 0.8):
                print(
                    f"\n[WARNING] Test has been running for {elapsed:.1f}s, "
                    f"approaching watchdog timeout ({self.max_watchdog_time}s)"
                )
                print("[WARNING] Consider terminating test to prevent kernel panic")

            # Check if test timeout exceeded
            if elapsed > self.timeout:
                print(
                    f"\n[ERROR] Test timeout exceeded ({self.timeout}s), "
                    "terminating to prevent watchdog timeout"
                )
                self._terminate_process()
                break

            time.sleep(self.check_interval)

    def _terminate_process(self):
        """Terminate the test process safely."""
        if self.process:
            print("\n[INFO] Terminating test process...")
            try:
                # Try graceful termination first
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    print("[WARNING] Graceful termination failed, forcing kill")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                print(f"[ERROR] Error terminating process: {e}")

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for logging."""
        state = {
            "timestamp": time.time(),
            "cpu_count": os.cpu_count(),
        }

        try:
            import psutil

            process = psutil.Process()
            state["process_cpu_percent"] = process.cpu_percent(interval=0.1)
            state["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)

            # System-wide metrics (non-blocking)
            if hasattr(psutil, "cpu_percent"):
                try:
                    # Use non-blocking call to avoid blocking watchdog
                    state["system_cpu_percent"] = psutil.cpu_percent(interval=None)
                except Exception:
                    state["system_cpu_percent"] = None

            state["system_memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            state["psutil_available"] = False
        except Exception as e:
            state["error"] = str(e)

        return state

    def run(self) -> Dict[str, Any]:
        """
        Run the test with monitoring.

        Returns:
            Dictionary with test results and system state
        """
        if not self.test_file.exists():
            return {
                "success": False,
                "error": f"Test file not found: {self.test_file}",
            }

        print(f"[INFO] Running test: {self.test_file}")
        if self.test_name:
            print(f"[INFO] Test name: {self.test_name}")
        print(f"[INFO] Timeout: {self.timeout}s")
        print(f"[INFO] Max watchdog time: {self.max_watchdog_time}s")
        print("-" * 80)

        # Capture initial system state
        initial_state = self._capture_system_state()
        print(f"[INFO] Initial CPU: {initial_state.get('system_cpu_percent', 'N/A')}%")
        print(
            f"[INFO] Initial Memory: {initial_state.get('system_memory_percent', 'N/A')}%"
        )

        # Build command
        cmd = self._build_pytest_command()
        print(f"[INFO] Command: {' '.join(cmd)}")
        print("-" * 80)

        # Start test process
        self.start_time = time.time()
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd(),
            )

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
            self.monitor_thread.start()

            # Wait for process to complete
            stdout, stderr = self.process.communicate(timeout=self.max_watchdog_time)

            elapsed = time.time() - self.start_time

            # Capture final system state
            final_state = self._capture_system_state()

            result = {
                "success": self.process.returncode == 0,
                "returncode": self.process.returncode,
                "elapsed_time": elapsed,
                "stdout": stdout,
                "stderr": stderr,
                "initial_state": initial_state,
                "final_state": final_state,
                "test_file": str(self.test_file),
                "test_name": self.test_name,
            }

            if self.process.returncode == 0:
                print(f"\n[SUCCESS] Test passed in {elapsed:.2f}s")
            else:
                print(f"\n[FAILURE] Test failed with return code {self.process.returncode}")
                print(f"[INFO] Elapsed time: {elapsed:.2f}s")
                if stderr:
                    print(f"[ERROR] Stderr:\n{stderr[:500]}")

            return result

        except subprocess.TimeoutExpired:
            elapsed = time.time() - self.start_time
            print(f"\n[ERROR] Test exceeded maximum watchdog time ({self.max_watchdog_time}s)")
            print(f"[ERROR] Elapsed time: {elapsed:.2f}s")
            self._terminate_process()

            return {
                "success": False,
                "error": f"Test exceeded maximum watchdog time ({self.max_watchdog_time}s)",
                "elapsed_time": elapsed,
                "test_file": str(self.test_file),
                "test_name": self.test_name,
            }

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            self.should_stop = True
            self._terminate_process()
            return {
                "success": False,
                "error": "Interrupted by user",
                "test_file": str(self.test_file),
                "test_name": self.test_name,
            }

        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            self._terminate_process()
            return {
                "success": False,
                "error": str(e),
                "test_file": str(self.test_file),
                "test_name": self.test_name,
            }

        finally:
            self.should_stop = True
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Isolate watchdog timeout issues by running tests individually"
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to test file",
    )
    parser.add_argument(
        "--test-name",
        help="Specific test name to run (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Test timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="Check interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-watchdog-time",
        type=int,
        default=60,
        help="Maximum time before watchdog timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)",
    )

    args = parser.parse_args()

    runner = TestIsolationRunner(
        test_file=args.test_file,
        test_name=args.test_name,
        timeout=args.timeout,
        check_interval=args.check_interval,
        max_watchdog_time=args.max_watchdog_time,
    )

    result = runner.run()

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[INFO] Results saved to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()

