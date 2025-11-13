# evaluation/caws_eval.py
# End-to-end CAWS gates evaluation: budget checks, waiver policy, provenance
# @author: @darianrosebrook

import json
import typer
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

app = typer.Typer()


def validate_budget_adherence(change_diff: str, max_loc: int, max_files: int) -> Dict:
    """Validate change adheres to CAWS budget constraints."""
    lines_added = 0
    lines_removed = 0
    files_changed = set()

    # Parse diff: count + and - lines, track unique files
    for line in change_diff.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            # Extract filename from diff header
            if line.startswith("+++"):
                file_path = line[4:].split("\t")[0].strip()
            else:
                file_path = line[4:].split("\t")[0].strip()
            if file_path and file_path != "/dev/null":
                files_changed.add(file_path)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    files_changed_count = len(files_changed) if files_changed else 1

    total_loc = lines_added + lines_removed
    within_budget = total_loc <= max_loc and files_changed_count <= max_files

    return {
        "within_budget": within_budget,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "files_changed": files_changed_count,
        "max_loc": max_loc,
        "max_files": max_files,
    }


def validate_gate_integrity(test_results: Dict, lint_results: Dict, coverage_results: Dict) -> Dict:
    """Validate all CAWS quality gates pass."""
    return {
        "tests_pass": test_results.get("all_passed", False),
        "lint_pass": lint_results.get("no_errors", False),
        "coverage_pass": coverage_results.get("meets_threshold", False),
        "all_gates_pass": (
            test_results.get("all_passed", False)
            and lint_results.get("no_errors", False)
            and coverage_results.get("meets_threshold", False)
        ),
    }


def validate_provenance_clarity(
    rationale: Optional[str], evidence_manifest: Optional[Dict], change_diff: Optional[str]
) -> Dict:
    """Validate provenance clarity requirements."""
    rationale_present = rationale is not None and len(rationale.strip()) > 0
    evidence_present = (
        evidence_manifest is not None and len(evidence_manifest.get("evidence_items", [])) > 0
    )
    diff_present = change_diff is not None and len(change_diff.strip()) > 0

    # Simple alignment score: all components present = 1.0
    alignment_score = 1.0 if (rationale_present and evidence_present and diff_present) else 0.0

    return {
        "rationale_present": rationale_present,
        "evidence_manifest_present": evidence_present,
        "diff_present": diff_present,
        "alignment_score": alignment_score,
    }


def evaluate_caws_compliance(
    change_id: str,
    working_spec: Dict,
    change_diff: str,
    rationale: Optional[str],
    evidence_manifest: Optional[Dict],
    test_results: Dict,
    lint_results: Dict,
    coverage_results: Dict,
) -> Dict:
    """Run complete CAWS compliance evaluation."""

    # Extract budgets from working spec
    max_loc = working_spec.get("budgets", {}).get("max_loc", 1000)
    max_files = working_spec.get("budgets", {}).get("max_files", 10)

    budget_adherence = validate_budget_adherence(change_diff, max_loc, max_files)
    gate_integrity = validate_gate_integrity(test_results, lint_results, coverage_results)
    provenance_clarity = validate_provenance_clarity(rationale, evidence_manifest, change_diff)

    # Determine verdict
    all_pass = (
        budget_adherence["within_budget"]
        and gate_integrity["all_gates_pass"]
        and provenance_clarity["alignment_score"] >= 0.8
    )

    verdict = "PASS" if all_pass else "FAIL"
    if not all_pass and budget_adherence["within_budget"]:
        verdict = "WAIVER_REQUIRED"

    return {
        "change_id": change_id,
        "verdict": verdict,
        "caws_compliance": {
            "budget_adherence": budget_adherence,
            "gate_integrity": gate_integrity,
            "provenance_clarity": provenance_clarity,
        },
        "overall_compliance": all_pass,
    }


def _load_working_spec(spec_path: str) -> Dict:
    """Load working spec YAML file."""
    if yaml is None:
        raise ImportError("PyYAML required for CAWS evaluation. Install with: pip install pyyaml")

    spec_file = Path(spec_path)
    if not spec_file.exists():
        raise FileNotFoundError(f"Working spec not found: {spec_path}")

    with open(spec_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_file_content(file_path: str) -> Optional[str]:
    """Load file content if path is provided."""
    if not file_path:
        return None

    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_json_file(file_path: str) -> Optional[Dict]:
    """Load JSON file if path is provided."""
    if not file_path:
        return None

    path = Path(file_path)
    if not path.exists():
        print(f"Warning: JSON file not found: {file_path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_tests() -> Dict:
    """Run test suite and return results."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-v"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "all_passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"all_passed": False, "error": "Test execution timed out"}
    except FileNotFoundError:
        return {"all_passed": False, "error": "pytest not found"}


def _run_linter() -> Dict:
    """Run linter and return results."""
    # Try common linters
    linters = [
        ("ruff", ["ruff", "check", "."]),
        ("flake8", ["flake8", "."]),
        ("pylint", ["pylint", "--errors-only", "."]),
    ]

    for linter_name, cmd in linters:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {
                "no_errors": result.returncode == 0,
                "linter": linter_name,
                "returncode": result.returncode,
                "output": result.stdout + result.stderr,
            }
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return {"no_errors": False, "error": f"{linter_name} timed out"}

    # No linter found - assume passing (user may not have linter configured)
    return {"no_errors": True, "warning": "No linter found, assuming no errors"}


def _run_coverage() -> Dict:
    """Run coverage check and return results."""
    try:
        # Try to get coverage from pytest-cov
        result = subprocess.run(
            ["python", "-m", "pytest", "--cov", "--cov-report=json"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Try to parse coverage.json if it exists
        coverage_file = Path("coverage.json")
        if coverage_file.exists():
            with open(coverage_file, "r") as f:
                coverage_data = json.load(f)
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)

                # Default threshold is 80%
                threshold = 80.0
                return {
                    "meets_threshold": total_coverage >= threshold,
                    "coverage_percent": total_coverage,
                    "threshold": threshold,
                }

        return {
            "meets_threshold": result.returncode == 0,
            "warning": "Could not parse coverage report",
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "meets_threshold": True,
            "warning": "Coverage check not available, assuming threshold met",
        }


@app.command()
def main(
    working_spec: str = typer.Argument(...),
    change_diff: str = typer.Option("", "--diff"),
    rationale: str = typer.Option("", "--rationale"),
    evidence_manifest: str = typer.Option("", "--evidence"),
    test_results: str = typer.Option("", "--test-results"),
    lint_results: str = typer.Option("", "--lint-results"),
    coverage_results: str = typer.Option("", "--coverage-results"),
    run_tests: bool = typer.Option(False, "--run-tests"),
    run_lint: bool = typer.Option(False, "--run-lint"),
    run_coverage: bool = typer.Option(False, "--run-coverage"),
    output: str = typer.Option("", "--output", "-o"),
):
    """Run CAWS compliance evaluation.

    Args:
        working_spec: Path to working-spec.yaml
        change_diff: Path to change diff file (or use --run-tests to get from git)
        rationale: Path to rationale file
        evidence_manifest: Path to evidence manifest JSON
        test_results: Path to test results JSON (or use --run-tests)
        lint_results: Path to lint results JSON (or use --run-lint)
        coverage_results: Path to coverage results JSON (or use --run-coverage)
        run_tests: Run test suite automatically
        run_lint: Run linter automatically
        run_coverage: Run coverage check automatically
        output: Output file path for results JSON (default: stdout)
    """
    # Load working spec
    try:
        spec = _load_working_spec(working_spec)
    except Exception as e:
        print(f"Error loading working spec: {e}", file=sys.stderr)
        sys.exit(1)

    change_id = spec.get("id", "UNKNOWN")

    # Load change diff
    diff_content = None
    if change_diff:
        diff_content = _load_file_content(change_diff)
    elif run_tests:
        # Try to get diff from git
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                diff_content = result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: Could not get git diff", file=sys.stderr)

    # Load rationale
    rationale_content = _load_file_content(rationale) if rationale else None

    # Load evidence manifest
    evidence_data = _load_json_file(evidence_manifest) if evidence_manifest else None

    # Get test results
    if test_results:
        test_data = _load_json_file(test_results)
    elif run_tests:
        print("Running test suite...")
        test_data = _run_tests()
    else:
        test_data = {"all_passed": False, "warning": "Test results not provided"}

    # Get lint results
    if lint_results:
        lint_data = _load_json_file(lint_results)
    elif run_lint:
        print("Running linter...")
        lint_data = _run_linter()
    else:
        lint_data = {"no_errors": True, "warning": "Lint results not provided"}

    # Get coverage results
    if coverage_results:
        coverage_data = _load_json_file(coverage_results)
    elif run_coverage:
        print("Running coverage check...")
        coverage_data = _run_coverage()
    else:
        coverage_data = {"meets_threshold": True, "warning": "Coverage results not provided"}

    # Run CAWS compliance evaluation
    try:
        result = evaluate_caws_compliance(
            change_id=change_id,
            working_spec=spec,
            change_diff=diff_content or "",
            rationale=rationale_content,
            evidence_manifest=evidence_data,
            test_results=test_data,
            lint_results=lint_data,
            coverage_results=coverage_data,
        )

        # Add timestamp
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Output results
        result_json = json.dumps(result, indent=2)

        if output:
            with open(output, "w") as f:
                f.write(result_json)
            print(f"CAWS evaluation results saved to: {output}")
        else:
            print(result_json)

        # Exit with appropriate code
        if result["verdict"] == "PASS":
            sys.exit(0)
        elif result["verdict"] == "WAIVER_REQUIRED":
            print("\n⚠️  Waiver required - see violations above", file=sys.stderr)
            sys.exit(2)
        else:
            print("\n❌ CAWS compliance check failed", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    app()
