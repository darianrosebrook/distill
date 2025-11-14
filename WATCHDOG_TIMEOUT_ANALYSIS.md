# Watchdog Timeout Analysis

## Summary

We've implemented comprehensive timeout protection and isolation tools to identify tests causing kernel panics due to watchdog timeouts. Individual high-risk tests complete quickly when run in isolation, suggesting the issue may occur:

1. During full test suite execution (resource accumulation)
2. With specific test combinations
3. Under specific system conditions
4. When multiple tests create threads/resources that aren't properly cleaned up

## Implemented Solutions

### 1. Test Isolation Tools

- **`scripts/isolate_watchdog_test.py`**: Run individual tests with monitoring and timeout detection
- **`scripts/monitor_watchdog.py`**: Monitor watchdog daemon status and detect blocked threads
- **`tests/utils/thread_safety.py`**: Safe thread join utilities with timeouts

### 2. Timeout Protection

- **Global pytest timeout**: 30 seconds per test (configured in `pytest.ini`)
- **Thread join timeouts**: All thread joins now use timeouts (5 seconds)
- **Fixed high-risk test**: `test_feature_manager_not_thread_safe` now uses `safe_thread_join_all()`

### 3. Verified Mock Coverage

- **CoreML tests**: All CoreML operations properly mocked (no real hardware calls)
- **ANE monitor tests**: All ANE operations properly mocked (no real hardware calls)
- **Health checker**: Uses non-blocking CPU percent calls

## Test Results

### High-Risk Tests Tested

1. ✅ `test_feature_manager_not_thread_safe` - Passed (2.80s)
2. ✅ `test_entailment_calibration` - Passed (4.94s)
3. ✅ `test_ane_monitor` - Passed (4.37s)
4. ✅ `test_start_monitoring` - Completed quickly (2.85s)
5. ✅ `test_stop_monitoring` - Completed quickly (2.67s)

All high-risk tests complete quickly when run individually with timeouts.

## Potential Problem Areas

### 1. Health Checker CPU Monitoring

**Location**: `infra/health.py::get_system_metrics()`

**Issue**: First call to `psutil.cpu_percent()` uses `interval=0.05` (50ms). While this is better than 1 second, it could still block if:
- System is heavily loaded
- Multiple tests create HealthChecker instances
- Error handling causes repeated initialization attempts

**Risk**: MEDIUM - Non-blocking on subsequent calls, but first call blocks

**Recommendation**: Consider using `interval=None` even on first call if psutil is already initialized elsewhere

### 2. Monitoring Threads

**Location**: `tests/training/test_monitoring.py::test_start_monitoring`

**Issue**: Tests create monitoring threads that call `get_system_metrics()`. If multiple tests run in sequence and threads aren't properly cleaned up, they could accumulate.

**Risk**: MEDIUM - Tests have cleanup, but might not be reliable under all conditions

**Recommendation**: Ensure all monitoring threads are properly stopped and joined with timeouts

### 3. ThreadPoolExecutor Tests

**Location**: Multiple tests using `concurrent.futures.ThreadPoolExecutor`

**Issue**: Tests use ThreadPoolExecutor but rely on context manager cleanup. If tests fail or are interrupted, threads might not be properly cleaned up.

**Risk**: LOW - Context managers should handle cleanup, but worth monitoring

### 4. Subprocess Calls

**Location**: `tests/test_entailment_calibration.py`

**Issue**: Uses real `subprocess.run()` without timeout. If the script hangs, the test could block indefinitely.

**Risk**: LOW - Test completed quickly, but no timeout protection

**Recommendation**: Add timeout to subprocess calls

## Next Steps

### 1. Run Full Test Suite with Monitoring

```bash
# Run full test suite with timeout protection
source venv/bin/activate
python -m pytest tests/ -v --timeout=30 --tb=short 2>&1 | tee /tmp/test_run.log

# If it hangs, check which test was running
tail -50 /tmp/test_run.log
```

### 2. Run Tests in Batches

```bash
# Run tests by category to isolate problematic area
python -m pytest tests/conversion/ -v --timeout=30
python -m pytest tests/training/ -v --timeout=30
python -m pytest tests/evaluation/ -v --timeout=30
```

### 3. Monitor System Resources

```bash
# Monitor watchdog and system resources during test run
python scripts/monitor_watchdog.py --duration 300 &
python -m pytest tests/ -v --timeout=30
```

### 4. Check for Resource Leaks

Look for tests that:
- Create threads without proper cleanup
- Create processes without proper cleanup
- Use blocking system calls without timeouts
- Create HealthChecker instances that start monitoring threads
- Use psutil functions with blocking intervals

### 5. Add Subprocess Timeout

```python
# In test_entailment_calibration.py
proc = subprocess.run(
    [sys.executable, str(SCRIPT), "--in", str(DATA), "--out", str(OUT)],
    capture_output=True,
    text=True,
    timeout=30,  # Add timeout
)
```

## Recommendations

1. **Add timeouts to all subprocess calls** - Prevent hangs from external scripts
2. **Ensure all monitoring threads are properly stopped** - Add cleanup in test teardown
3. **Monitor test execution time** - Identify tests that take longer than expected
4. **Run tests in isolated processes** - Use pytest-xdist or similar to isolate test execution
5. **Add resource monitoring** - Track thread count, memory usage, CPU usage during test runs

## Files Modified

- `scripts/isolate_watchdog_test.py` - Test isolation script
- `scripts/monitor_watchdog.py` - Watchdog monitoring script
- `tests/utils/thread_safety.py` - Thread safety utilities
- `tests/training/test_feature_flags.py` - Fixed thread join timeout
- `requirements-dev.txt` - Added pytest-timeout
- `pytest.ini` - Added timeout configuration
- `tests/conftest.py` - Added timeout documentation

## Conclusion

All individual high-risk tests complete quickly with timeout protection. The kernel panic likely occurs during full test suite execution when:
- Multiple tests create resources that aren't properly cleaned up
- System resources are exhausted
- A specific test combination triggers a blocking operation

Continue monitoring test execution and use the isolation tools to identify problematic test combinations.


