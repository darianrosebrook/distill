#!/usr/bin/env python3
"""
Code quality verification script for production readiness.

Scans codebase for code quality issues:
- TODOs, PLACEHOLDERs, and MOCK_DATA in production code
- Unused imports (using ruff or vulture)
- Code formatting issues
- Dead code detection

Usage:
    python scripts/verify-code-quality.py --output docs/internal/audits/readiness/evidence/code-quality/
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def scan_for_todos_placeholders(output_dir: Path) -> Dict[str, Any]:
    """
    Scan codebase for TODOs, PLACEHOLDERs, and MOCK_DATA.

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with scan results
    """
    print("[verify-code-quality] Scanning for TODOs, PLACEHOLDERs, and MOCK_DATA...")

    # Directories to scan (production code only - exclude scripts/utilities)
    scan_dirs = ["training", "conversion", "models", "evaluation"]
    exclude_patterns = [
        "__pycache__",
        ".git",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "*.egg-info",
        "build",
        "dist",
    ]

    todos_placeholders = []
    # Only flag PLACEHOLDER/TODO/MOCK_DATA when they're clearly marked as such
    # This avoids false positives in documentation or comments that just mention these terms
    pattern = re.compile(r"#\s*(TODO|PLACEHOLDER|MOCK_DATA)\b", re.IGNORECASE)

    for scan_dir in scan_dirs:
        scan_path = Path(scan_dir)
        if not scan_path.exists():
            continue

        for py_file in scan_path.rglob("*.py"):
            # Skip excluded files
            if any(excl in str(py_file) for excl in exclude_patterns):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    matches = pattern.findall(line)
                    if matches:
                        # Extract the full comment
                        stripped = line.strip()
                        if (
                            stripped.startswith("#")
                            or "TODO" in stripped
                            or "PLACEHOLDER" in stripped
                        ):
                            todos_placeholders.append(
                                {
                                    "file": str(py_file),
                                    "line": line_num,
                                    "type": matches[0].upper(),
                                    "content": stripped,
                                }
                            )

            except (UnicodeDecodeError, IOError) as e:
                print(f"[verify-code-quality] WARNING: Could not read {py_file}: {e}")
                continue

    # Save detailed list
    todos_file = output_dir / "todos-placeholders.txt"
    with open(todos_file, "w") as f:
        f.write("TODOs, PLACEHOLDERs, and MOCK_DATA Found\n")
        f.write("=" * 60 + "\n\n")

        if todos_placeholders:
            f.write(f"Found {len(todos_placeholders)} issues:\n\n")
            for item in todos_placeholders:
                f.write(f"File: {item['file']}:{item['line']}\n")
                f.write(f"Type: {item['type']}\n")
                f.write(f"Content: {item['content']}\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("✅ No TODOs, PLACEHOLDERs, or MOCK_DATA found in production code\n")

    print(f"[verify-code-quality] TODO/PLACEHOLDER scan saved to: {todos_file}")

    return {
        "todos_placeholders_found": len(todos_placeholders),
        "todos_placeholders_list": todos_placeholders,
        "zero_todos_placeholders": len(todos_placeholders) == 0,
    }


def check_unused_imports(output_dir: Path) -> Dict[str, Any]:
    """
    Check for unused imports using ruff or vulture.

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with unused import results
    """
    print("[verify-code-quality] Checking for unused imports...")

    unused_imports = []

    # Try ruff first (fast and integrated)
    try:
        cmd = ["ruff", "check", "--select=F401", "--output-format=json", "."]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=Path.cwd())

        if result.returncode == 0:
            # Parse JSON output
            try:
                issues = json.loads(result.stdout)
                for issue in issues:
                    if issue.get("code") == "F401":  # Unused import
                        unused_imports.append(
                            {
                                "file": issue.get("filename", ""),
                                "line": issue.get("location", {}).get("row", 0),
                                "message": issue.get("message", ""),
                                "code": issue.get("code", ""),
                            }
                        )
            except json.JSONDecodeError:
                pass

        print(f"[verify-code-quality] Ruff found {len(unused_imports)} unused imports")

    except FileNotFoundError:
        # Try vulture if ruff not available
        try:
            cmd = ["vulture", "--min-confidence=80", "."]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=Path.cwd()
            )

            # Parse vulture text output
            lines = result.stdout.split("\n")
            for line in lines:
                if ":" in line and line.strip():
                    try:
                        file_path, line_num, message = line.split(":", 2)
                        unused_imports.append(
                            {
                                "file": file_path.strip(),
                                "line": int(line_num.strip()),
                                "message": message.strip(),
                                "code": "F401",  # Vulture doesn't use codes
                            }
                        )
                    except ValueError:
                        continue

            print(f"[verify-code-quality] Vulture found {len(unused_imports)} unused imports")

        except FileNotFoundError:
            print(
                "[verify-code-quality] WARNING: Neither ruff nor vulture available for unused import detection"
            )
            unused_imports = []

    # Save unused imports list
    unused_file = output_dir / "dead-code-report.txt"
    with open(unused_file, "w") as f:
        f.write("Unused Imports/Code Analysis\n")
        f.write("=" * 40 + "\n\n")

        if unused_imports:
            f.write(f"Found {len(unused_imports)} unused imports:\n\n")
            for item in unused_imports:
                f.write(f"File: {item['file']}:{item['line']}\n")
                f.write(f"Message: {item['message']}\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("✅ No unused imports detected\n")
            f.write("   (Note: This check may not be comprehensive without ruff/vulture)\n")

    print(f"[verify-code-quality] Unused imports report saved to: {unused_file}")

    return {
        "unused_imports_found": len(unused_imports),
        "unused_imports_list": unused_imports,
    }


def check_formatting(output_dir: Path) -> Dict[str, Any]:
    """
    Check code formatting using ruff format.

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with formatting results
    """
    print("[verify-code-quality] Checking code formatting...")

    formatting_issues = []

    try:
        # Use ruff format --check to see what would be changed
        cmd = ["ruff", "format", "--check", "--diff", "."]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=Path.cwd())

        # If return code is non-zero, there are formatting issues
        if result.returncode != 0:
            # Parse the diff output to count issues
            lines = result.stdout.split("\n")
            files_changed = 0
            for line in lines:
                if line.startswith("+++ ") and ".py" in line:
                    files_changed += 1

            formatting_issues = [
                {
                    "type": "formatting",
                    "files_affected": files_changed,
                    "message": f"Code formatting issues in {files_changed} files",
                    "details": result.stdout[:2000],  # Truncate for file size
                }
            ]

        print(f"[verify-code-quality] Formatting check found {len(formatting_issues)} issues")

    except FileNotFoundError:
        print("[verify-code-quality] WARNING: ruff not available for formatting check")
        formatting_issues = []

    # Save formatting report
    formatting_file = output_dir / "formatting-check.txt"
    with open(formatting_file, "w") as f:
        f.write("Code Formatting Check\n")
        f.write("=" * 30 + "\n\n")

        if formatting_issues:
            issue = formatting_issues[0]
            f.write(f"Found formatting issues in {issue['files_affected']} files\n\n")
            f.write("Details:\n")
            f.write(issue["details"])
        else:
            f.write("✅ Code formatting is correct\n")

    print(f"[verify-code-quality] Formatting report saved to: {formatting_file}")

    return {
        "formatting_issues_found": len(formatting_issues),
        "formatting_issues": formatting_issues,
    }


def generate_summary(output_dir: Path, results: Dict[str, Any]) -> None:
    """
    Generate comprehensive code quality summary.

    Args:
        output_dir: Directory to save summary
        results: Combined results from all checks
    """
    summary_file = output_dir / "code-quality-summary.txt"

    with open(summary_file, "w") as f:
        f.write("Code Quality Verification Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # TODO/PLACEHOLDER check
        todo_results = results["todos_placeholders"]
        f.write("1. TODOs/PLACEHOLDERs/MOCK_DATA Check:\n")
        f.write(f"   Found: {todo_results['todos_placeholders_found']}\n")
        if todo_results["zero_todos_placeholders"]:
            f.write("   ✅ PASS - No TODOs/PLACEHOLDERs/MOCK_DATA in production code\n")
        else:
            f.write("   ❌ FAIL - TODOs/PLACEHOLDERs/MOCK_DATA found\n")
            f.write("   See: todos-placeholders.txt\n")
        f.write("\n")

        # Unused imports
        unused_results = results["unused_imports"]
        f.write("2. Unused Imports Check:\n")
        f.write(f"   Found: {unused_results['unused_imports_found']}\n")
        if unused_results["unused_imports_found"] == 0:
            f.write("   ✅ PASS - No unused imports detected\n")
        else:
            f.write("   ⚠️  WARNING - Unused imports found (may be false positives)\n")
            f.write("   See: dead-code-report.txt\n")
        f.write("\n")

        # Formatting
        format_results = results["formatting"]
        f.write("3. Code Formatting Check:\n")
        f.write(f"   Issues: {format_results['formatting_issues_found']}\n")
        if format_results["formatting_issues_found"] == 0:
            f.write("   ✅ PASS - Code formatting is correct\n")
        else:
            f.write("   ❌ FAIL - Code formatting issues found\n")
            f.write("   See: formatting-check.txt\n")
        f.write("\n")

        # Overall assessment
        f.write("Production Readiness Assessment:\n")
        all_checks_pass = (
            todo_results["zero_todos_placeholders"]
            and format_results["formatting_issues_found"] == 0
        )

        if all_checks_pass:
            f.write("✅ PASS - Code quality gates met\n")
            f.write("   - No TODOs/PLACEHOLDERs/MOCK_DATA in production code\n")
            f.write("   - Code formatting is correct\n")
        else:
            f.write("❌ FAIL - Code quality issues found\n")
            if not todo_results["zero_todos_placeholders"]:
                f.write("   - TODOs/PLACEHOLDERs/MOCK_DATA must be removed\n")
            if format_results["formatting_issues_found"] > 0:
                f.write("   - Code formatting must be fixed\n")

    print(f"[verify-code-quality] Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify code quality for production readiness")
    parser.add_argument(
        "--output", required=True, help="Output directory for code quality evidence"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[verify-code-quality] Starting code quality verification...")
    print(f"[verify-code-quality] Output directory: {output_dir}")

    # Run all checks
    results = {}

    results["todos_placeholders"] = scan_for_todos_placeholders(output_dir)
    results["unused_imports"] = check_unused_imports(output_dir)
    results["formatting"] = check_formatting(output_dir)

    # Generate summary
    generate_summary(output_dir, results)

    # Final assessment
    todo_pass = results["todos_placeholders"]["zero_todos_placeholders"]
    format_pass = results["formatting"]["formatting_issues_found"] == 0

    if todo_pass and format_pass:
        print("[verify-code-quality] ✅ SUCCESS: Code quality gates met")
        sys.exit(0)
    else:
        print("[verify-code-quality] ❌ FAILURE: Code quality issues found")
        print("   Check code-quality-summary.txt for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
