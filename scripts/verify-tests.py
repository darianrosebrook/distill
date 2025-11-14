#!/usr/bin/env python3
"""
Test execution verification script for production readiness.

Runs full pytest suite and parses results to verify:
- All tests pass (no failures)
- No tests are skipped inappropriately
- Test execution completes successfully

Usage:
    python scripts/verify-tests.py --output docs/internal/audits/readiness/evidence/test-execution/
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def run_pytest(output_dir: Path) -> tuple[bool, str, subprocess.CompletedProcess]:
    """
    Run pytest suite and capture output.

    Args:
        output_dir: Directory to save evidence files

    Returns:
        Tuple of (success, error_message)
    """
    pytest_output_file = output_dir / "pytest-output.txt"

    try:
        # Run pytest with verbose output and JSON report
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",  # Run all tests in tests/ directory
            "-v",  # Verbose output
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Strict marker validation
            "--durations=10",  # Show 10 slowest tests
            # Note: JSON report requires pytest-json-report plugin
            # We'll parse the text output instead
            "--maxfail=5",  # Stop after 5 failures to avoid long runs
        ]

        print(f"[verify-tests] Running: {' '.join(cmd)}")
        print(f"[verify-tests] Output will be saved to: {pytest_output_file}")

        # Run pytest and capture output
        pytest_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=Path.cwd(),
        )

        # Save raw output
        with open(pytest_output_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {pytest_result.returncode}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n" + "=" * 80 + "\nSTDOUT:\n" + "=" * 80 + "\n")
            f.write(pytest_result.stdout)
            if pytest_result.stderr:
                f.write("\n" + "=" * 80 + "\nSTDERR:\n" + "=" * 80 + "\n")
                f.write(pytest_result.stderr)

        print(f"[verify-tests] Pytest completed with return code: {pytest_result.returncode}")

        return True, "", pytest_result

    except subprocess.TimeoutExpired:
        error_msg = "Test execution timed out after 30 minutes"
        print(f"[verify-tests] ERROR: {error_msg}")
        with open(pytest_output_file, "w") as f:
            f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
        return False, error_msg

    except Exception as e:
        error_msg = f"Failed to run pytest: {e}"
        print(f"[verify-tests] ERROR: {error_msg}")
        with open(pytest_output_file, "w") as f:
            f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
        return False, error_msg


def parse_pytest_results(
    output_dir: Path, returncode: int, stdout: str, stderr: str
) -> Optional[Dict[str, Any]]:
    """
    Parse pytest text output and extract key metrics.

    Args:
        output_dir: Directory to save parsed results
        returncode: pytest exit code
        stdout: pytest stdout output
        stderr: pytest stderr output

    Returns:
        Dictionary with parsed results, or None if parsing failed
    """
    try:
        # Extract key metrics from pytest output
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        warnings = 0
        duration_seconds = 0.0

        # Parse collection line: "collecting ... collected 705 items"
        collection_match = re.search(r"collected\s+(\d+)\s+items", stdout)
        if collection_match:
            total_tests = int(collection_match.group(1))

        # Parse summary line: "============= 5 failed, 38 passed, 2 skipped, 2 warnings in 4.20s =============="
        summary_match = re.search(
            r"=+\s+(\d+)\s+failed[,\s]+(\d+)\s+passed[,\s]+(\d+)\s+skipped[,\s]+(\d+)\s+warnings\s+in\s+([\d.]+)s\s+=+",
            stdout,
        )
        if summary_match:
            failed = int(summary_match.group(1))
            passed = int(summary_match.group(2))
            skipped = int(summary_match.group(3))
            warnings = int(summary_match.group(4))
            duration_seconds = float(summary_match.group(5))

        # Extract failed test details from FAILURES section
        failed_tests = []
        in_failures_section = False
        current_failure = None

        for line in stdout.split("\n"):
            line = line.strip()

            if line.startswith(
                "=================================== FAILURES ==================================="
            ):
                in_failures_section = True
                continue
            elif line.startswith(
                "=================================== warnings summary ==============================="
            ):
                in_failures_section = False
                continue

            if in_failures_section:
                # Look for failure start: "___________________________ test_name ___________________________"
                failure_match = re.match(r"^_{20,}\s+(.+?)\s+_{20,}", line)
                if failure_match:
                    if current_failure:
                        failed_tests.append(current_failure)
                    current_failure = {
                        "nodeid": failure_match.group(1),
                        "outcome": "failed",
                        "duration": 0.0,
                        "traceback": "",
                    }
                elif current_failure and line.startswith("tests/"):
                    # Extract test path and line info
                    test_info_match = re.match(r"(tests/.*?\.py):(\d+):", line)
                    if test_info_match:
                        current_failure["nodeid"] = (
                            f"{test_info_match.group(1)}::{current_failure['nodeid']}"
                        )
                elif current_failure and line.startswith("E "):
                    # Add traceback line
                    current_failure["traceback"] += line[2:] + "\n"
                elif current_failure and line.startswith("> "):
                    # Add assertion line
                    current_failure["traceback"] += line[2:] + "\n"

        # Add the last failure if exists
        if current_failure:
            failed_tests.append(current_failure)

        # Extract platform info from header
        python_version = "unknown"
        pytest_version = "unknown"
        platform_match = re.search(
            r"platform\s+(.*?)\s+--\s+Python\s+([\d.]+),\s+pytest-([\d.]+)", stdout
        )
        if platform_match:
            python_version = platform_match.group(2)
            pytest_version = platform_match.group(3)

        # Build results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,  # Pytest doesn't separate errors from failures in summary
            "warnings": warnings,
            "duration_seconds": duration_seconds,
            "exit_code": returncode,
            "python_version": python_version,
            "pytest_version": pytest_version,
        }

        # Add pass/fail assessment
        results["all_passed"] = (
            results["failed"] == 0 and results["errors"] == 0 and results["exit_code"] == 0
        )

        results["failed_tests"] = failed_tests

        # Save parsed results
        results_file = output_dir / "test-results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[verify-tests] Parsed results saved to: {results_file}")

        # Save failed tests summary
        if failed_tests:
            failed_file = output_dir / "failed-tests.txt"
            with open(failed_file, "w") as f:
                f.write(f"Failed Tests Summary ({len(failed_tests)} failures)\n")
                f.write("=" * 80 + "\n\n")
                for test in failed_tests:
                    f.write(f"Test: {test['nodeid']}\n")
                    f.write(f"Outcome: {test['outcome']}\n")
                    f.write(f"Duration: {test['duration']:.2f}s\n")
                    if test.get("traceback"):
                        f.write("Traceback:\n")
                        f.write(test["traceback"][:500] + "...\n")
                    f.write("-" * 40 + "\n\n")

            print(f"[verify-tests] Failed tests saved to: {failed_file}")

        return results

    except Exception as e:
        print(f"[verify-tests] WARNING: Failed to parse pytest output: {e}")
        return None


def generate_summary(output_dir: Path, results: Optional[Dict[str, Any]]) -> None:
    """
    Generate a human-readable summary of test results.

    Args:
        output_dir: Directory to save summary
        results: Parsed test results (or None)
    """
    summary_file = output_dir / "test-summary.txt"

    with open(summary_file, "w") as f:
        f.write("Test Execution Verification Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        if results:
            f.write("Test Results:\n")
            f.write(f"  Total tests: {results['total_tests']}\n")
            f.write(f"  Passed: {results['passed']}\n")
            f.write(f"  Failed: {results['failed']}\n")
            f.write(f"  Skipped: {results['skipped']}\n")
            f.write(f"  Errors: {results['errors']}\n")
            f.write(f"  Warnings: {results['warnings']}\n")
            f.write(f"  Duration: {results['duration_seconds']:.1f}s\n")
            f.write(f"  Exit code: {results['exit_code']}\n")
            f.write(f"  Python version: {results['python_version']}\n")
            f.write(f"  Pytest version: {results['pytest_version']}\n\n")

            # Assessment
            if results["all_passed"]:
                f.write("✅ ASSESSMENT: PASS - All tests pass\n")
                f.write("   Production readiness criterion: All unit tests passing - MET\n")
            else:
                f.write("❌ ASSESSMENT: FAIL - Tests failed or had errors\n")
                f.write("   Production readiness criterion: All unit tests passing - NOT MET\n")
                f.write(f"   Failed tests: {len(results.get('failed_tests', []))}\n")

            # Check for inappropriate skipping
            if results["skipped"] > results["total_tests"] * 0.1:  # More than 10% skipped
                f.write(
                    f"⚠️  WARNING: High skip rate ({results['skipped']}/{results['total_tests']} = {(results['skipped'] / max(1, results['total_tests'])) * 100:.1f}%)\n"
                )
                f.write("   Review skipped tests to ensure they're not masking failures\n")
        else:
            f.write("❌ ASSESSMENT: Unable to parse test results\n")
            f.write("   Check pytest-output.txt for manual review\n")

    print(f"[verify-tests] Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify test execution for production readiness")
    parser.add_argument("--output", required=True, help="Output directory for evidence files")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[verify-tests] Starting test execution verification...")
    print(f"[verify-tests] Output directory: {output_dir}")

    # Run pytest
    success, error, pytest_result = run_pytest(output_dir)

    if not success:
        print(f"[verify-tests] CRITICAL: Failed to run tests: {error}")
        sys.exit(1)

    # Parse results from the pytest result object
    results = parse_pytest_results(
        output_dir, pytest_result.returncode, pytest_result.stdout, pytest_result.stderr
    )

    # Generate summary
    generate_summary(output_dir, results)

    # Final assessment
    if results and results.get("all_passed"):
        print("[verify-tests] ✅ SUCCESS: All tests pass")
        sys.exit(0)
    else:
        print("[verify-tests] ❌ FAILURE: Tests failed or could not be parsed")
        print("   Check evidence files for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
