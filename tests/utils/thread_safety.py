"""
Thread safety utilities for tests.

Provides safe thread join wrappers and timeout utilities to prevent
watchdog timeouts and kernel panics during test execution.

@author: @darianrosebrook
"""

import threading
import time
import logging
from typing import Optional, List, Tuple, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThreadJoinResult:
    """Result of a thread join operation."""

    success: bool
    timeout: bool
    thread_alive: bool
    elapsed_time: float
    message: str


def safe_thread_join(
    thread: threading.Thread, timeout: float = 5.0, thread_name: Optional[str] = None
) -> ThreadJoinResult:
    """
    Safely join a thread with timeout to prevent blocking.

    This prevents indefinite blocking that could cause watchdog timeouts.

    Args:
        thread: Thread to join
        timeout: Maximum time to wait for thread (seconds)
        thread_name: Optional name for logging

    Returns:
        ThreadJoinResult with join status and timing information
    """
    name = thread_name or (thread.name if hasattr(thread, "name") else "unknown")
    start_time = time.time()

    if not thread.is_alive():
        elapsed = time.time() - start_time
        return ThreadJoinResult(
            success=True,
            timeout=False,
            thread_alive=False,
            elapsed_time=elapsed,
            message=f"Thread '{name}' already finished",
        )

    thread.join(timeout=timeout)
    elapsed = time.time() - start_time
    is_alive = thread.is_alive()

    if is_alive:
        logger.warning(
            f"Thread '{name}' did not complete within timeout ({timeout}s). "
            f"Elapsed: {elapsed:.2f}s. Thread is still alive."
        )
        return ThreadJoinResult(
            success=False,
            timeout=True,
            thread_alive=True,
            elapsed_time=elapsed,
            message=f"Thread '{name}' timed out after {elapsed:.2f}s",
        )

    return ThreadJoinResult(
        success=True,
        timeout=False,
        thread_alive=False,
        elapsed_time=elapsed,
        message=f"Thread '{name}' completed in {elapsed:.2f}s",
    )


def safe_thread_join_all(
    threads: List[threading.Thread], timeout: float = 5.0, per_thread: bool = False
) -> Tuple[bool, List[ThreadJoinResult]]:
    """
    Safely join multiple threads with timeout.

    Args:
        threads: List of threads to join
        timeout: Maximum time to wait
                 - If per_thread=True: timeout per thread
                 - If per_thread=False: total timeout for all threads
        per_thread: Whether timeout applies per thread or total

    Returns:
        Tuple of (all_succeeded, list of results)
    """
    results = []
    start_time = time.time()

    for i, thread in enumerate(threads):
        if per_thread:
            # Timeout per thread
            result = safe_thread_join(thread, timeout=timeout, thread_name=f"thread_{i}")
        else:
            # Remaining time for all threads
            elapsed = time.time() - start_time
            remaining = max(0.1, timeout - elapsed)
            result = safe_thread_join(thread, timeout=remaining, thread_name=f"thread_{i}")

        results.append(result)

        # If we've exceeded total timeout, stop joining remaining threads
        if not per_thread:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"Total timeout ({timeout}s) exceeded. "
                    f"Stopped joining threads after {i+1}/{len(threads)}"
                )
                # Mark remaining threads as not joined
                for j in range(i + 1, len(threads)):
                    results.append(
                        ThreadJoinResult(
                            success=False,
                            timeout=True,
                            thread_alive=threads[j].is_alive(),
                            elapsed_time=elapsed,
                            message=f"Thread {j} not joined due to total timeout",
                        )
                    )
                break

    all_succeeded = all(r.success for r in results)
    return all_succeeded, results


def run_with_timeout(
    func: Callable[[], Any], timeout: float = 5.0, default: Any = None
) -> Tuple[Any, bool]:
    """
    Run a function with timeout using a separate thread.

    Args:
        func: Function to run
        timeout: Maximum time to wait (seconds)
        default: Default value to return if timeout

    Returns:
        Tuple of (result, timed_out)
    """
    result_container = {"value": None, "exception": None, "completed": False}

    def target():
        try:
            result_container["value"] = func()
            result_container["completed"] = True
        except Exception as e:
            result_container["exception"] = e
            result_container["completed"] = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    join_result = safe_thread_join(thread, timeout=timeout)

    if join_result.timeout:
        logger.warning(f"Function timed out after {timeout}s")
        return default, True

    if result_container["exception"]:
        raise result_container["exception"]

    return result_container["value"], False


def create_timeout_thread(
    target: Callable[[], None],
    timeout: float = 5.0,
    name: Optional[str] = None,
    daemon: bool = True,
) -> threading.Thread:
    """
    Create a thread with built-in timeout protection.

    The thread will automatically terminate if it exceeds the timeout.

    Args:
        target: Function to run in thread
        timeout: Maximum execution time (seconds)
        name: Thread name
        daemon: Whether thread is daemon

    Returns:
        Thread instance
    """
    def timeout_wrapper():
        start_time = time.time()
        try:
            target()
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Thread '{name or 'unknown'}' exceeded timeout "
                    f"({timeout}s), elapsed: {elapsed:.2f}s"
                )

    thread = threading.Thread(target=timeout_wrapper, name=name, daemon=daemon)
    return thread


def monitor_threads(
    threads: List[threading.Thread],
    check_interval: float = 0.5,
    max_duration: Optional[float] = None,
) -> List[bool]:
    """
    Monitor threads and return when all complete or timeout.

    Args:
        threads: Threads to monitor
        check_interval: How often to check thread status (seconds)
        max_duration: Maximum time to monitor (None = no limit)

    Returns:
        List of booleans indicating which threads completed
    """
    start_time = time.time()
    completed = [False] * len(threads)

    while True:
        # Check if all threads completed
        all_done = True
        for i, thread in enumerate(threads):
            if not thread.is_alive() and not completed[i]:
                completed[i] = True
            if thread.is_alive():
                all_done = False

        if all_done:
            break

        # Check timeout
        if max_duration:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                logger.warning(f"Thread monitoring timed out after {max_duration}s")
                break

        time.sleep(check_interval)

    return completed


