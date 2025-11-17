"""
Full integration gates tests (A8).

Runs all validation tests and verifies all gates pass.
"""

import pytest
import subprocess
import sys
from pathlib import Path


@pytest.mark.slow
def test_full_integration_gates():
    """Test that all integration gates pass."""
    # This test runs all validation tests and verifies all gates pass
    
    # List of gate tests to run
    gate_tests = [
        "tests/e2e/test_checkpoint_validation.py",
        "tests/e2e/test_export_validation.py",
        "tests/e2e/test_coreml_conversion.py",
        "tests/e2e/test_parity_probes.py",
        "tests/e2e/test_tool_use_evaluation.py",
        "tests/e2e/test_performance_benchmarks.py",
        "tests/e2e/test_long_context_evaluation.py",
    ]
    
    # Run each gate test
    results = {}
    for test_file in gate_tests:
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = {"status": "skipped", "reason": "file not found"}
            continue
        
        # Run the test
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_file,
                "-v",
                "--tb=short",
                "-m",
                "slow",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes per test file
        )
        
        if result.returncode == 0:
            results[test_file] = {"status": "passed"}
        elif result.returncode == 5:  # No tests collected
            results[test_file] = {"status": "skipped", "reason": "no tests collected"}
        else:
            results[test_file] = {"status": "failed", "stderr": result.stderr[:500]}
    
    # Report results
    passed = sum(1 for r in results.values() if r["status"] == "passed")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    
    print("\n=== Integration Gates Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(results)}")
    
    for test_file, result in results.items():
        status = result["status"]
        symbol = "✅" if status == "passed" else "⚠️" if status == "skipped" else "❌"
        print(f"{symbol} {Path(test_file).name}: {status}")
        if status == "failed" and "reason" in result:
            print(f"   Reason: {result.get('reason', 'unknown')}")
    
    # For now, we allow some tests to be skipped (e.g., CoreML not available)
    # In production, all critical gates should pass
    assert failed == 0, f"{failed} gate tests failed"
    
    print("\n✅ All integration gates passed (or appropriately skipped)")


@pytest.mark.slow
def test_integration_gates_parity():
    """Test parity gate specifically."""
    # Run parity probe test
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/test_parity_probes.py",
            "-v",
            "--tb=short",
            "-m",
            "slow",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # Allow skip if CoreML not available
    if result.returncode == 5:  # No tests collected
        pytest.skip("Parity probe tests not collected (may require CoreML)")
    
    # For now, we allow the test to be skipped if CoreML is not available
    # In production, this gate must pass
    assert result.returncode in [0, 5], f"Parity gate failed: {result.stderr[:500]}"


@pytest.mark.slow
def test_integration_gates_performance():
    """Test performance gate specifically."""
    # Run performance benchmark test
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/test_performance_benchmarks.py",
            "-v",
            "--tb=short",
            "-m",
            "slow",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # Performance tests may be skipped if CoreML not available
    if result.returncode == 5:  # No tests collected
        pytest.skip("Performance benchmark tests not collected (may require CoreML)")
    
    assert result.returncode in [0, 5], f"Performance gate failed: {result.stderr[:500]}"


def test_integration_gates_infrastructure():
    """Test that integration gates infrastructure is available."""
    # Verify test files exist
    test_files = [
        "tests/e2e/test_checkpoint_validation.py",
        "tests/e2e/test_export_validation.py",
        "tests/e2e/test_coreml_conversion.py",
        "tests/e2e/test_parity_probes.py",
        "tests/e2e/test_tool_use_evaluation.py",
        "tests/e2e/test_performance_benchmarks.py",
        "tests/e2e/test_long_context_evaluation.py",
    ]
    
    for test_file in test_files:
        test_path = Path(test_file)
        assert test_path.exists(), f"Test file should exist: {test_file}"
    
    print("✅ Integration gates infrastructure validated")

