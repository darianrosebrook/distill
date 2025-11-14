#!/usr/bin/env python3
"""
Coverage verification script for production readiness.

Runs pytest with coverage and analyzes results to verify:
- Line coverage >= 80%
- Branch coverage >= 90%
- Coverage meets production thresholds

Usage:
    python scripts/verify-coverage.py --output docs/internal/audits/readiness/evidence/coverage/
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def run_coverage(output_dir: Path) -> tuple[bool, str]:
    """
    Run pytest with coverage analysis.

    Args:
        output_dir: Directory to save coverage evidence

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Run pytest with coverage for key modules
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=training",  # Training module
            "--cov=models",  # Model definitions
            "--cov=evaluation",  # Evaluation harness
            "--cov=conversion",  # Conversion utilities
            "--cov-report=json:coverage.json",  # JSON report for parsing
            "--cov-report=html:htmlcov",  # HTML report for manual review
            "--cov-report=term-missing",  # Terminal summary
            "--cov-fail-under=0",  # Don't fail on low coverage (we check manually)
            "tests/",  # Run all tests
        ]

        print(f"[verify-coverage] Running: {' '.join(cmd)}")
        print(f"[verify-coverage] Coverage reports will be saved to: {output_dir}")

        # Run coverage and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2400,  # 40 minute timeout
            cwd=Path.cwd(),
        )

        # Save raw output
        raw_output_file = output_dir / "pytest-coverage-output.txt"
        with open(raw_output_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n" + "=" * 80 + "\nSTDOUT:\n" + "=" * 80 + "\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n" + "=" * 80 + "\nSTDERR:\n" + "=" * 80 + "\n")
                f.write(result.stderr)

        print(
            f"[verify-coverage] Coverage analysis completed with return code: {result.returncode}"
        )

        # Copy coverage files to output directory
        coverage_json = Path("coverage.json")
        htmlcov_dir = Path("htmlcov")

        if coverage_json.exists():
            import shutil

            shutil.copy2(coverage_json, output_dir / "coverage-report.json")
            print(
                f"[verify-coverage] Coverage JSON copied to: {output_dir / 'coverage-report.json'}"
            )

        if htmlcov_dir.exists():
            import shutil

            htmlcov_target = output_dir / "htmlcov"
            if htmlcov_target.exists():
                shutil.rmtree(htmlcov_target)
            shutil.copytree(htmlcov_dir, htmlcov_target)
            print(f"[verify-coverage] HTML coverage report copied to: {htmlcov_target}")

        return True, ""

    except subprocess.TimeoutExpired:
        error_msg = "Coverage analysis timed out after 40 minutes"
        print(f"[verify-coverage] ERROR: {error_msg}")
        raw_output_file = output_dir / "pytest-coverage-output.txt"
        with open(raw_output_file, "w") as f:
            f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
        return False, error_msg

    except Exception as e:
        error_msg = f"Failed to run coverage analysis: {e}"
        print(f"[verify-coverage] ERROR: {error_msg}")
        raw_output_file = output_dir / "pytest-coverage-output.txt"
        with open(raw_output_file, "w") as f:
            f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
        return False, error_msg


def parse_coverage_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Parse coverage JSON report and extract key metrics.

    Args:
        output_dir: Directory containing coverage files

    Returns:
        Dictionary with parsed coverage results, or None if parsing failed
    """
    coverage_file = output_dir / "coverage-report.json"

    if not coverage_file.exists():
        print("[verify-coverage] WARNING: No coverage JSON file found")
        return None

    try:
        with open(coverage_file, "r") as f:
            data = json.load(f)

        # Extract totals
        totals = data.get("totals", {})
        results = {
            "timestamp": datetime.now().isoformat(),
            "line_coverage_percent": totals.get("percent_covered", 0.0),
            "line_covered": totals.get("covered_lines", 0),
            "line_total": totals.get("num_statements", 0),
            "line_missing": totals.get("missing_lines", 0),
            "branch_coverage_percent": totals.get("percent_covered_branch", 0.0),
            "branch_covered": totals.get("covered_branches", 0),
            "branch_total": totals.get("num_branches", 0),
            "branch_missing": totals.get("missing_branches", 0),
        }

        # Extract per-module coverage
        module_coverage = {}
        for file_path, file_data in data.get("files", {}).items():
            # Use the path as-is from coverage report
            module_coverage[file_path] = {
                "line_percent": file_data.get("summary", {}).get("percent_covered", 0.0),
                "line_covered": file_data.get("summary", {}).get("covered_lines", 0),
                "line_total": file_data.get("summary", {}).get("num_statements", 0),
                "branch_percent": file_data.get("summary", {}).get("percent_covered_branch", 0.0),
                "branch_covered": file_data.get("summary", {}).get("covered_branches", 0),
                "branch_total": file_data.get("summary", {}).get("num_branches", 0),
            }

        results["module_coverage"] = module_coverage

        # Assessment against thresholds
        LINE_THRESHOLD = 80.0
        BRANCH_THRESHOLD = 90.0

        results["line_threshold_met"] = results["line_coverage_percent"] >= LINE_THRESHOLD
        results["branch_threshold_met"] = results["branch_coverage_percent"] >= BRANCH_THRESHOLD
        results["overall_pass"] = results["line_threshold_met"] and results["branch_threshold_met"]

        results["thresholds"] = {
            "line_coverage_required": LINE_THRESHOLD,
            "branch_coverage_required": BRANCH_THRESHOLD,
        }

        # Save parsed results
        parsed_file = output_dir / "coverage-by-module.json"
        with open(parsed_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[verify-coverage] Parsed coverage saved to: {parsed_file}")

        return results

    except Exception as e:
        print(f"[verify-coverage] ERROR: Failed to parse coverage results: {e}")
        return None


def generate_summary(output_dir: Path, results: Optional[Dict[str, Any]]) -> None:
    """
    Generate human-readable coverage summary.

    Args:
        output_dir: Directory to save summary
        results: Parsed coverage results
    """
    summary_file = output_dir / "coverage-summary.txt"

    with open(summary_file, "w") as f:
        f.write("Coverage Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        if results:
            f.write("Overall Coverage:\n")
            f.write(".1f")
            f.write(".1f")
            f.write(f"  Line coverage: {results['line_covered']}/{results['line_total']} lines\n")
            f.write(
                f"  Branch coverage: {results['branch_covered']}/{results['branch_total']} branches\n\n"
            )

            # Threshold assessment
            results["thresholds"]
            f.write("Threshold Assessment:\n")
            f.write(".1f")
            if results["line_threshold_met"]:
                f.write(" ✅ MET\n")
            else:
                f.write(" ❌ NOT MET\n")

            f.write(".1f")
            if results["branch_threshold_met"]:
                f.write(" ✅ MET\n")
            else:
                f.write(" ❌ NOT MET\n")

            f.write("\nProduction Readiness Criteria:\n")
            if results["overall_pass"]:
                f.write("✅ PASS - Complete unit test coverage (80%+ line, 90%+ branch)\n")
            else:
                f.write("❌ FAIL - Coverage thresholds not met\n")
                if not results["line_threshold_met"]:
                    f.write(".1f")
                if not results["branch_threshold_met"]:
                    f.write(".1f")

            # Top 10 modules by coverage (lowest first)
            f.write("\nCoverage by Module (Top 10 Lowest):\n")
            f.write("-" * 80 + "\n")
            f.write("<25")
            f.write("-" * 80 + "\n")

            sorted_modules = sorted(
                results["module_coverage"].items(), key=lambda x: x[1]["line_percent"]
            )[:10]

            for module_path, coverage in sorted_modules:
                f.write("<25")
                f.write("\n")

        else:
            f.write("❌ ERROR: Could not parse coverage results\n")
            f.write(
                "   Check coverage-report.json and pytest-coverage-output.txt for manual review\n"
            )

    print(f"[verify-coverage] Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify code coverage for production readiness")
    parser.add_argument("--output", required=True, help="Output directory for coverage evidence")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[verify-coverage] Starting coverage verification...")
    print(f"[verify-coverage] Output directory: {output_dir}")

    # Run coverage analysis
    success, error = run_coverage(output_dir)

    if not success:
        print(f"[verify-coverage] CRITICAL: Failed to run coverage: {error}")
        sys.exit(1)

    # Parse results
    results = parse_coverage_results(output_dir)

    # Generate summary
    generate_summary(output_dir, results)

    # Final assessment
    if results and results.get("overall_pass"):
        print("[verify-coverage] ✅ SUCCESS: Coverage thresholds met")
        sys.exit(0)
    else:
        print("[verify-coverage] ❌ FAILURE: Coverage thresholds not met")
        print("   Check coverage-summary.txt for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
