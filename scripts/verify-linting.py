#!/usr/bin/env python3
"""
Linting verification script for production readiness.

Runs linters and parses results to verify:
- Zero linting errors (warnings allowed)
- Code follows style guidelines
- No critical code quality issues

Usage:
    python scripts/verify-linting.py --output docs/internal/audits/readiness/evidence/linting/
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def run_linter(output_dir: Path) -> tuple[bool, str]:
    """
    Run linter and capture output.

    Tries ruff first (configured in pyproject.toml), then flake8, then pylint.

    Args:
        output_dir: Directory to save linting evidence

    Returns:
        Tuple of (success, error_message)
    """
    linters = [
        ("ruff", ["ruff", "check", ".", "--output-format=json"]),
        ("flake8", ["flake8", ".", "--format=json"]),
        ("pylint", ["pylint", "--errors-only", "--output-format=json", "."]),
    ]

    for linter_name, cmd in linters:
        try:
            print(f"[verify-linting] Trying {linter_name}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path.cwd(),
            )

            # Save raw output
            output_file = output_dir / f"{linter_name}-output.txt"
            with open(output_file, "w") as f:
                f.write(f"Linter: {linter_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("\n" + "=" * 80 + "\nSTDOUT:\n" + "=" * 80 + "\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "=" * 80 + "\nSTDERR:\n" + "=" * 80 + "\n")
                    f.write(result.stderr)

            print(f"[verify-linting] {linter_name} completed with return code: {result.returncode}")
            print(f"[verify-linting] Output saved to: {output_file}")

            # Return success even if linter found issues (we parse them below)
            return True, ""

        except FileNotFoundError:
            print(f"[verify-linting] {linter_name} not found, trying next linter...")
            continue
        except subprocess.TimeoutExpired:
            error_msg = f"{linter_name} timed out after 5 minutes"
            print(f"[verify-linting] ERROR: {error_msg}")
            output_file = output_dir / f"{linter_name}-output.txt"
            with open(output_file, "w") as f:
                f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to run {linter_name}: {e}"
            print(f"[verify-linting] ERROR: {error_msg}")
            output_file = output_dir / f"{linter_name}-output.txt"
            with open(output_file, "w") as f:
                f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
            return False, error_msg

    # No linter found
    error_msg = "No supported linter found (tried ruff, flake8, pylint)"
    print(f"[verify-linting] ERROR: {error_msg}")
    output_file = output_dir / "no-linter-output.txt"
    with open(output_file, "w") as f:
        f.write(f"ERROR: {error_msg}\nTimestamp: {datetime.now().isoformat()}\n")
    return False, error_msg


def parse_linter_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Parse linter output and extract error metrics.

    Args:
        output_dir: Directory containing linter output

    Returns:
        Dictionary with parsed results, or None if parsing failed
    """
    # Find the linter output file
    linter_files = list(output_dir.glob("*-output.txt"))
    if not linter_files:
        print("[verify-linting] WARNING: No linter output files found")
        return None

    # Use the first (most preferred) linter that ran
    output_file = linter_files[0]
    linter_name = output_file.stem.replace("-output", "")

    try:
        with open(output_file, "r") as f:
            content = f.read()

        # Extract JSON if present (for structured output)
        json_start = content.find("[")
        if json_start >= 0:
            try:
                json_content = content[json_start:]
                issues = json.loads(json_content)
            except json.JSONDecodeError:
                # Fall back to text parsing
                issues = parse_text_output(content)
        else:
            # Parse text output
            issues = parse_text_output(content)

        # Categorize issues
        error_count = 0
        warning_count = 0
        issues_by_type = {}
        issues_by_file = {}

        for issue in issues:
            issue_type = issue.get("code", issue.get("type", "unknown"))
            issue_file = issue.get("filename", issue.get("file", "unknown"))
            issue_severity = issue.get("severity", "unknown")

            # Count errors vs warnings
            if issue_severity in ["error", "E", "F"]:  # flake8/ruff error codes
                error_count += 1
            elif issue_severity in ["warning", "W", "C", "R"]:  # warnings and conventions
                warning_count += 1

            # Group by type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

            # Group by file
            if issue_file not in issues_by_file:
                issues_by_file[issue_file] = []
            issues_by_file[issue_file].append(issue)

        results = {
            "timestamp": datetime.now().isoformat(),
            "linter": linter_name,
            "total_issues": len(issues),
            "errors": error_count,
            "warnings": warning_count,
            "zero_errors": error_count == 0,
            "issues_by_type": issues_by_type,
            "issues_by_file": issues_by_file,
            "raw_issues": issues[:100],  # Limit for file size
        }

        # Save parsed results
        parsed_file = output_dir / "lint-errors.json"
        with open(parsed_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[verify-linting] Parsed results saved to: {parsed_file}")

        return results

    except Exception as e:
        print(f"[verify-linting] ERROR: Failed to parse linter results: {e}")
        return None


def parse_text_output(content: str) -> list:
    """
    Parse text-based linter output into structured format.

    This is a fallback for linters that don't support JSON output.

    Args:
        content: Raw linter output text

    Returns:
        List of issue dictionaries
    """
    issues = []
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("=") or "Command:" in line:
            continue

        # Try to extract file:line:col: message format
        if ":" in line and line.count(":") >= 3:
            try:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    filename, line_num, col, message = parts
                    issues.append(
                        {
                            "filename": filename.strip(),
                            "line": int(line_num) if line_num.isdigit() else 0,
                            "column": int(col) if col.isdigit() else 0,
                            "message": message.strip(),
                            "severity": "unknown",
                            "code": "unknown",
                        }
                    )
            except (ValueError, IndexError):
                continue

    return issues


def generate_summary(output_dir: Path, results: Optional[Dict[str, Any]]) -> None:
    """
    Generate human-readable linting summary.

    Args:
        output_dir: Directory to save summary
        results: Parsed linting results
    """
    summary_file = output_dir / "lint-summary.json"

    if results:
        # Save structured summary
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        # Also create text summary
        text_summary_file = output_dir / "lint-summary.txt"
        with open(text_summary_file, "w") as f:
            f.write("Linting Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Linter: {results['linter']}\n\n")

            f.write("Issue Summary:\n")
            f.write(f"  Total issues: {results['total_issues']}\n")
            f.write(f"  Errors: {results['errors']}\n")
            f.write(f"  Warnings: {results['warnings']}\n\n")

            if results["zero_errors"]:
                f.write("✅ ASSESSMENT: PASS - Zero linting errors\n")
                f.write("   Production readiness criterion: Zero linting errors - MET\n")
            else:
                f.write("❌ ASSESSMENT: FAIL - Linting errors found\n")
                f.write("   Production readiness criterion: Zero linting errors - NOT MET\n")
                f.write(f"   Must fix {results['errors']} errors before production\n")

            # Show top issue types
            if results["issues_by_type"]:
                f.write("\nTop Issue Types:\n")
                sorted_types = sorted(
                    results["issues_by_type"].items(), key=lambda x: len(x[1]), reverse=True
                )[:10]

                for issue_type, issues in sorted_types:
                    f.write(f"  {issue_type}: {len(issues)} issues\n")

            # Show files with most issues
            if results["issues_by_file"]:
                f.write("\nFiles with Most Issues:\n")
                sorted_files = sorted(
                    results["issues_by_file"].items(), key=lambda x: len(x[1]), reverse=True
                )[:10]

                for file_path, issues in sorted_files:
                    f.write(f"  {file_path}: {len(issues)} issues\n")
    else:
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "error": "Could not parse linting results",
                },
                f,
                indent=2,
            )

        text_summary_file = output_dir / "lint-summary.txt"
        with open(text_summary_file, "w") as f:
            f.write("Linting Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("❌ ERROR: Could not parse linting results\n")
            f.write("   Check linter output files for manual review\n")

    print(f"[verify-linting] Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify linting for production readiness")
    parser.add_argument("--output", required=True, help="Output directory for linting evidence")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[verify-linting] Starting linting verification...")
    print(f"[verify-linting] Output directory: {output_dir}")

    # Run linter
    success, error = run_linter(output_dir)

    if not success:
        print(f"[verify-linting] CRITICAL: Failed to run linter: {error}")
        sys.exit(1)

    # Parse results
    results = parse_linter_results(output_dir)

    # Generate summary
    generate_summary(output_dir, results)

    # Final assessment
    if results and results.get("zero_errors"):
        print("[verify-linting] ✅ SUCCESS: Zero linting errors")
        sys.exit(0)
    else:
        print("[verify-linting] ❌ FAILURE: Linting errors found")
        print("   Check lint-summary.txt for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
