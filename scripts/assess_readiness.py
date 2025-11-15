#!/usr/bin/env python3
"""
Readiness Assessment Script

Runs comprehensive baseline assessment of test status, coverage, TODOs, and readiness
for training/conversion workflows. Generates structured reports and comparison with
previous baselines.

@author: @darianrosebrook
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform


def get_git_info() -> Dict[str, Any]:
    """Get git commit, branch, and dirty status."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True
        ).returncode != 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def run_unit_tests() -> Dict[str, Any]:
    """Run unit tests and collect results."""
    print("Running unit tests...")
    try:
        # Run all tests (markers may not be used, so run all and filter by path/name)
        # Use -q for quieter output and --timeout=0 to disable per-test timeouts
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--tb=line", "--timeout=0"],
            capture_output=True,
            text=True,
            timeout=3600  # 60 minutes for full test suite
        )

        # Parse stdout for test results
        unit_data = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "failures": [],
            "duration_seconds": 0.0
        }

        stdout = result.stdout + result.stderr
        import re

        # Extract test counts from pytest output
        # Look for patterns like "131 passed in 5.48s" or "5 failed, 126 passed"
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        failed_match = re.search(r'(\d+)\s+failed', stdout)
        skipped_match = re.search(r'(\d+)\s+skipped', stdout)
        error_match = re.search(r'(\d+)\s+error', stdout)
        deselected_match = re.search(r'(\d+)\s+deselected', stdout)

        # Debug: save output if no matches found
        if not passed_match and not failed_match:
            debug_file = Path("reports/readiness/unit_test_debug.txt")
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w") as f:
                f.write(stdout[-2000:])  # Last 2000 chars

        if passed_match:
            unit_data["passed"] = int(passed_match.group(1))
        if failed_match:
            unit_data["failed"] = int(failed_match.group(1))
        if skipped_match:
            unit_data["skipped"] = int(skipped_match.group(1))
        if error_match:
            unit_data["errors"] = int(error_match.group(1))

        unit_data["total"] = unit_data["passed"] + \
            unit_data["failed"] + unit_data["skipped"] + unit_data["errors"]

        # Extract duration
        duration_match = re.search(r'(\d+\.\d+)s', stdout)
        if duration_match:
            unit_data["duration_seconds"] = float(duration_match.group(1))

        # Extract failure details from stdout
        failure_pattern = re.compile(
            r'(FAILED|ERROR)\s+(tests/[^\s]+::[^\s]+)')
        for match in failure_pattern.finditer(stdout):
            unit_data["failures"].append({
                "test": match.group(2),
                "error": match.group(1),
                "message": ""
            })

        return unit_data
    except subprocess.TimeoutExpired:
        return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 1, "failures": [{"test": "timeout", "error": "Test execution timed out"}], "duration_seconds": 1800.0}
    except Exception as e:
        return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 1, "failures": [{"test": "error", "error": str(e)}], "duration_seconds": 0.0}


def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests and collect results."""
    print("Running integration tests...")
    # For now, integration tests are the same as unit tests since markers aren't used
    # In a real scenario, you'd filter by test file patterns or markers
    # Skip integration tests for now to avoid duplicate runs - use unit test results
    # In production, you'd filter by test file patterns (e.g., test_integration_*.py)
    # Return empty results since we're using unit test results for both
    # This avoids running tests twice
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "failures": [],
        "duration_seconds": 0.0
    }


def run_mutation_tests(skip: bool = False) -> Dict[str, Any]:
    """Run mutation tests on critical modules."""
    if skip:
        print("Skipping mutation tests...")
        return {"modules_tested": 0, "scores": {}, "survivors": {}, "timeouts": 0, "targets_met": 0, "targets_missed": []}

    print("Running mutation tests on critical modules...")
    try:
        result = subprocess.run(
            ["python", "scripts/run_mutation_testing.py",
                "--all-critical", "--mode", "s", "-n", "10"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour
        )

        mutation_data = {
            "modules_tested": 0,
            "scores": {},
            "survivors": {},
            "timeouts": 0,
            "targets_met": 0,
            "targets_missed": []
        }

        # Parse mutation test output (simplified - would need actual parsing)
        stdout = result.stdout
        # This is a placeholder - actual parsing would depend on mutatest output format
        mutation_data["modules_tested"] = stdout.count(
            "Running mutation testing on:")

        return mutation_data
    except subprocess.TimeoutExpired:
        return {"modules_tested": 0, "scores": {}, "survivors": {}, "timeouts": 1, "targets_met": 0, "targets_missed": []}
    except Exception as e:
        print(f"Warning: Mutation tests failed: {e}")
        return {"modules_tested": 0, "scores": {}, "survivors": {}, "timeouts": 0, "targets_met": 0, "targets_missed": []}


def collect_coverage() -> Dict[str, Any]:
    """Run coverage collection and parse results."""
    print("Collecting coverage data...")
    try:
        result = subprocess.run(
            [
                "python", "-m", "pytest",
                "--cov=training",
                "--cov=models",
                "--cov=evaluation",
                "--cov=conversion",
                "--cov-report=json:reports/readiness/coverage.json",
                "--cov-report=term",
                "tests/"
            ],
            capture_output=True,
            text=True,
            timeout=2400  # 40 minutes
        )

        coverage_data = {
            "overall": {
                "line_percent": 0.0,
                "branch_percent": 0.0,
                "lines_covered": 0,
                "lines_total": 0,
                "branches_covered": 0,
                "branches_total": 0
            },
            "by_module": {},
            "thresholds": {"line": 80.0, "branch": 90.0},
            "meets_thresholds": False,
            "critical_modules_below_threshold": []
        }

        # Coverage file might be in current dir or reports/readiness/
        coverage_file = Path("reports/readiness/coverage.json")
        if not coverage_file.exists():
            coverage_file = Path("coverage.json")

        if coverage_file.exists():
            with open(coverage_file) as f:
                report = json.load(f)
                totals = report.get("totals", {})

                coverage_data["overall"]["line_percent"] = totals.get(
                    "percent_covered", 0.0)
                coverage_data["overall"]["branch_percent"] = totals.get(
                    "percent_covered_branches", 0.0)
                coverage_data["overall"]["lines_covered"] = totals.get(
                    "covered_lines", 0)
                coverage_data["overall"]["lines_total"] = totals.get(
                    "num_statements", 0)
                coverage_data["overall"]["branches_covered"] = totals.get(
                    "covered_branches", 0)
                coverage_data["overall"]["branches_total"] = totals.get(
                    "num_branches", 0)

                # Extract per-module coverage
                files = report.get("files", {})
                critical_modules = [
                    "training/distill_kd.py",
                    "training/losses.py",
                    "conversion/export_onnx.py",
                    "conversion/convert_coreml.py"
                ]

                for file_path, file_data in files.items():
                    module_coverage = {
                        "line_percent": file_data.get("summary", {}).get("percent_covered", 0.0),
                        "branch_percent": file_data.get("summary", {}).get("percent_covered_branches", 0.0),
                        "lines_covered": file_data.get("summary", {}).get("covered_lines", 0),
                        "lines_total": file_data.get("summary", {}).get("num_statements", 0)
                    }
                    coverage_data["by_module"][file_path] = module_coverage

                    # Check if critical module is below threshold
                    if any(crit in file_path for crit in critical_modules):
                        if module_coverage["line_percent"] < 80.0 or module_coverage["branch_percent"] < 90.0:
                            coverage_data["critical_modules_below_threshold"].append(
                                file_path)

                coverage_data["meets_thresholds"] = (
                    coverage_data["overall"]["line_percent"] >= 80.0 and
                    coverage_data["overall"]["branch_percent"] >= 90.0
                )

        return coverage_data
    except Exception as e:
        print(f"Warning: Coverage collection failed: {e}")
        return {
            "overall": {"line_percent": 0.0, "branch_percent": 0.0, "lines_covered": 0, "lines_total": 0, "branches_covered": 0, "branches_total": 0},
            "by_module": {},
            "thresholds": {"line": 80.0, "branch": 90.0},
            "meets_thresholds": False,
            "critical_modules_below_threshold": []
        }


def analyze_todos() -> Dict[str, Any]:
    """Run TODO analyzer and parse results."""
    print("Analyzing TODOs...")
    try:
        result = subprocess.run(
            [
                "python", "scripts/todo_analyzer.py",
                "--root", ".",
                "--output-json", "reports/readiness/todos.json",
                "--output-md", "reports/readiness/todos_report.md",
                "--min-confidence", "0.7"
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        todos_data = {
            "total": 0,
            "blocking": 0,
            "critical": 0,
            "high_confidence": 0,
            "in_training_path": [],
            "in_conversion_path": [],
            "placeholders": 0,
            "mock_data": 0
        }

        todos_file = Path("reports/readiness/todos.json")
        if todos_file.exists():
            with open(todos_file) as f:
                report = json.load(f)
                summary = report.get("summary", {})

                todos_data["total"] = summary.get("total_hidden_todos", 0)
                todos_data["high_confidence"] = summary.get(
                    "high_confidence_todos", 0)

                # Filter TODOs in training/conversion paths
                training_paths = ["training/", "configs/"]
                conversion_paths = ["conversion/", "coreml/"]

                files = report.get("files", {})
                for file_path, file_data in files.items():
                    # Skip mutants directory and test files
                    if "mutants/" in file_path or "/test_" in file_path or "/tests/" in file_path:
                        continue

                    is_training = any(tp in file_path for tp in training_paths)
                    is_conversion = any(
                        cp in file_path for cp in conversion_paths)

                    hidden_todos = file_data.get("hidden_todos", {})
                    for line_num, todo_data in hidden_todos.items():
                        todo_info = {
                            "file": file_path,
                            "line": line_num,
                            "text": todo_data.get("comment", "")[:200],
                            "confidence": todo_data.get("confidence_score", 0.0)
                        }

                        if is_training:
                            todos_data["in_training_path"].append(todo_info)
                        if is_conversion:
                            todos_data["in_conversion_path"].append(todo_info)

                        # Check for blocking indicators
                        comment = todo_data.get("comment", "").lower()
                        if "blocking" in comment or "critical" in comment:
                            todos_data["blocking"] += 1
                            todos_data["critical"] += 1

                        if "placeholder" in comment:
                            todos_data["placeholders"] += 1
                        if "mock" in comment or "fake" in comment:
                            todos_data["mock_data"] += 1

        return todos_data
    except Exception as e:
        print(f"Warning: TODO analysis failed: {e}")
        return {
            "total": 0,
            "blocking": 0,
            "critical": 0,
            "high_confidence": 0,
            "in_training_path": [],
            "in_conversion_path": [],
            "placeholders": 0,
            "mock_data": 0
        }


def calculate_readiness_score(test_status: Dict, coverage: Dict, todos: Dict) -> float:
    """Calculate overall readiness score (0-100)."""
    # Test Health (30%)
    unit_total = test_status["unit"]["total"] or 1
    unit_score = (test_status["unit"]["passed"] /
                  unit_total) * 100 if unit_total > 0 else 0

    integration_total = test_status["integration"]["total"] or 1
    integration_score = (test_status["integration"]["passed"] /
                         integration_total) * 100 if integration_total > 0 else 0

    mutation_score = 50.0  # Default if not available
    if test_status["mutation"]["modules_tested"] > 0:
        scores = test_status["mutation"]["scores"].values()
        if scores:
            mutation_score = sum(scores) / len(scores)

    test_health = (unit_score * 0.4 + integration_score *
                   0.4 + mutation_score * 0.2)

    # Coverage (30%)
    coverage_score = (coverage["overall"]["line_percent"]
                      * 0.6 + coverage["overall"]["branch_percent"] * 0.4)

    # TODO Blocker Ratio (25%)
    total_todos = max(todos["total"], 1)
    blocker_ratio = (1 - todos["blocking"] / total_todos) * 100

    # Critical Path Health (15%)
    critical_path_health = 100.0
    if todos["in_training_path"] or todos["in_conversion_path"]:
        # Check if any are blocking
        blocking_in_path = sum(1 for t in todos["in_training_path"] + todos["in_conversion_path"]
                               if "blocking" in t.get("text", "").lower() or "critical" in t.get("text", "").lower())
        if blocking_in_path > 0:
            critical_path_health = 0.0
        else:
            critical_path_health = 50.0  # Warnings but not blockers

    score = (test_health * 0.3 + coverage_score * 0.3 +
             blocker_ratio * 0.25 + critical_path_health * 0.15)
    return round(score, 1)


def assess_readiness(test_status: Dict, coverage: Dict, todos: Dict) -> Dict[str, Any]:
    """Assess overall readiness and identify blockers."""
    blockers = []
    warnings = []
    recommendations = []

    # Check test failures
    if test_status["unit"]["failed"] > 0:
        blockers.append({
            "category": "tests",
            "description": f"{test_status['unit']['failed']} unit test(s) failing",
            "severity": "high"
        })

    if test_status["integration"]["failed"] > 0:
        blockers.append({
            "category": "tests",
            "description": f"{test_status['integration']['failed']} integration test(s) failing",
            "severity": "high"
        })

    # Check coverage thresholds
    if not coverage["meets_thresholds"]:
        blockers.append({
            "category": "coverage",
            "description": f"Coverage below thresholds (line: {coverage['overall']['line_percent']:.1f}%, branch: {coverage['overall']['branch_percent']:.1f}%)",
            "severity": "medium"
        })

    if coverage["critical_modules_below_threshold"]:
        blockers.append({
            "category": "coverage",
            "description": f"Critical modules below threshold: {', '.join(coverage['critical_modules_below_threshold'][:3])}",
            "severity": "high"
        })

    # Check blocking TODOs
    if todos["blocking"] > 0:
        blockers.append({
            "category": "todos",
            "description": f"{todos['blocking']} blocking TODO(s) found",
            "severity": "critical"
        })

    if todos["in_training_path"]:
        blockers.append({
            "category": "todos",
            "description": f"{len(todos['in_training_path'])} TODO(s) in training path",
            "severity": "high"
        })

    if todos["in_conversion_path"]:
        blockers.append({
            "category": "todos",
            "description": f"{len(todos['in_conversion_path'])} TODO(s) in conversion path",
            "severity": "high"
        })

    # Determine status
    if blockers and any(b["severity"] == "critical" for b in blockers):
        status = "blocked"
    elif blockers:
        status = "partial"
    else:
        status = "ready"

    # Generate recommendations
    if test_status["unit"]["failed"] > 0:
        recommendations.append("Fix failing unit tests before proceeding")
    if coverage["overall"]["line_percent"] < 80.0:
        recommendations.append(
            f"Increase line coverage from {coverage['overall']['line_percent']:.1f}% to 80%")
    if todos["blocking"] > 0:
        recommendations.append("Resolve blocking TODOs in critical paths")

    score = calculate_readiness_score(test_status, coverage, todos)

    return {
        "status": status,
        "score": score,
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": recommendations
    }


def compare_baselines(current: Dict, previous: Optional[Dict]) -> Dict[str, Any]:
    """Compare current baseline with previous baseline."""
    if not previous:
        return {"has_previous": False}

    comparison = {
        "has_previous": True,
        "readiness_score_delta": current["readiness"]["score"] - previous.get("readiness", {}).get("score", 0),
        "test_failures_delta": {
            "unit": current["test_status"]["unit"]["failed"] - previous.get("test_status", {}).get("unit", {}).get("failed", 0),
            "integration": current["test_status"]["integration"]["failed"] - previous.get("test_status", {}).get("integration", {}).get("failed", 0)
        },
        "coverage_delta": {
            "line": current["coverage"]["overall"]["line_percent"] - previous.get("coverage", {}).get("overall", {}).get("line_percent", 0),
            "branch": current["coverage"]["overall"]["branch_percent"] - previous.get("coverage", {}).get("overall", {}).get("branch_percent", 0)
        },
        "todos_delta": {
            "total": current["todos"]["total"] - previous.get("todos", {}).get("total", 0),
            "blocking": current["todos"]["blocking"] - previous.get("todos", {}).get("blocking", 0)
        }
    }

    return comparison


def generate_summary_report(baseline: Dict, comparison: Optional[Dict], output_file: Path):
    """Generate human-readable markdown summary report."""
    with open(output_file, "w") as f:
        f.write("# Readiness Assessment Summary\n\n")
        f.write(f"**Timestamp**: {baseline['timestamp']}\n")
        f.write(f"**Git Commit**: {baseline['git']['commit'][:8]}\n")
        f.write(f"**Branch**: {baseline['git']['branch']}\n")
        f.write(f"**Status**: {baseline['readiness']['status'].upper()}\n")
        f.write(
            f"**Readiness Score**: {baseline['readiness']['score']}/100\n\n")

        if comparison and comparison.get("has_previous"):
            f.write("## Comparison with Previous Baseline\n\n")
            score_delta = comparison["readiness_score_delta"]
            trend = "↑" if score_delta > 0 else "↓" if score_delta < 0 else "→"
            f.write(
                f"- **Readiness Score**: {trend} {abs(score_delta):.1f} points\n")
            f.write(
                f"- **Unit Test Failures**: {comparison['test_failures_delta']['unit']:+d}\n")
            f.write(
                f"- **Coverage (Line)**: {comparison['coverage_delta']['line']:+.1f}%\n")
            f.write(
                f"- **TODOs**: {comparison['todos_delta']['total']:+d} (blocking: {comparison['todos_delta']['blocking']:+d})\n\n")

        f.write("## Test Status\n\n")
        f.write(
            f"- **Unit Tests**: {baseline['test_status']['unit']['passed']}/{baseline['test_status']['unit']['total']} passed")
        if baseline['test_status']['unit']['failed'] > 0:
            f.write(f" ({baseline['test_status']['unit']['failed']} failed)")
        f.write("\n")
        f.write(
            f"- **Integration Tests**: {baseline['test_status']['integration']['passed']}/{baseline['test_status']['integration']['total']} passed")
        if baseline['test_status']['integration']['failed'] > 0:
            f.write(
                f" ({baseline['test_status']['integration']['failed']} failed)")
        f.write("\n\n")

        f.write("## Coverage\n\n")
        f.write(
            f"- **Line Coverage**: {baseline['coverage']['overall']['line_percent']:.1f}% (threshold: 80%)\n")
        f.write(
            f"- **Branch Coverage**: {baseline['coverage']['overall']['branch_percent']:.1f}% (threshold: 90%)\n")
        if baseline['coverage']['critical_modules_below_threshold']:
            f.write(
                f"- **Critical Modules Below Threshold**: {len(baseline['coverage']['critical_modules_below_threshold'])}\n")
        f.write("\n")

        f.write("## TODOs\n\n")
        f.write(f"- **Total**: {baseline['todos']['total']}\n")
        f.write(f"- **Blocking**: {baseline['todos']['blocking']}\n")
        f.write(
            f"- **In Training Path**: {len(baseline['todos']['in_training_path'])}\n")
        f.write(
            f"- **In Conversion Path**: {len(baseline['todos']['in_conversion_path'])}\n\n")

        f.write("## Blockers\n\n")
        if baseline['readiness']['blockers']:
            for blocker in baseline['readiness']['blockers']:
                f.write(
                    f"- **{blocker['category']}** ({blocker['severity']}): {blocker['description']}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write("## Recommendations\n\n")
        for rec in baseline['readiness']['recommendations']:
            f.write(f"- {rec}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Assess project readiness for training/conversion")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with previous baseline")
    parser.add_argument(
        "--output-dir", default="reports/readiness", help="Output directory")
    parser.add_argument("--skip-mutation", action="store_true",
                        help="Skip mutation tests")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    # Run assessments
    unit_tests = run_unit_tests()
    integration_tests = run_integration_tests()
    mutation_tests = run_mutation_tests(skip=args.skip_mutation)
    coverage = collect_coverage()
    todos = analyze_todos()

    # Assess readiness
    readiness = assess_readiness(
        {"unit": unit_tests, "integration": integration_tests,
            "mutation": mutation_tests},
        coverage,
        todos
    )

    # Build baseline
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "git": get_git_info(),
        "test_status": {
            "unit": unit_tests,
            "integration": integration_tests,
            "mutation": mutation_tests
        },
        "coverage": coverage,
        "todos": todos,
        "readiness": readiness,
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "assessment_duration_seconds": (datetime.now() - start_time).total_seconds()
        }
    }

    # Load previous baseline if comparing
    previous_baseline = None
    latest_file = output_dir / "readiness_baseline_latest.json"
    if args.compare and latest_file.exists():
        with open(latest_file) as f:
            previous_baseline = json.load(f)

    comparison = compare_baselines(baseline, previous_baseline)

    # Save baseline
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_file = output_dir / f"readiness_baseline_{timestamp_str}.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)

    # Update latest symlink (create copy since symlinks may not work on all systems)
    import shutil
    shutil.copy(baseline_file, latest_file)

    # Append to history
    history_file = output_dir / "history.jsonl"
    with open(history_file, "a") as f:
        f.write(json.dumps(baseline) + "\n")

    # Generate reports
    summary_file = output_dir / f"readiness_summary_{timestamp_str}.md"
    generate_summary_report(baseline, comparison, summary_file)

    print(f"\n✅ Assessment complete!")
    print(f"📊 Readiness Score: {readiness['score']}/100")
    print(f"📁 Baseline saved: {baseline_file}")
    print(f"📄 Summary report: {summary_file}")

    if comparison.get("has_previous"):
        print(f"\n📈 Comparison with previous baseline:")
        print(f"   Score change: {comparison['readiness_score_delta']:+.1f}")

    return 0 if readiness["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
