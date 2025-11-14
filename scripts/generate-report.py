#!/usr/bin/env python3
"""
Report generation script for production readiness verification.

Reads all evidence files and generates comprehensive markdown reports
showing production readiness status across all criteria.

Usage:
    python scripts/generate-report.py --evidence-dir docs/internal/audits/readiness/evidence/ --output docs/internal/audits/readiness/reports/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def load_evidence_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse an evidence JSON file.

    Args:
        file_path: Path to evidence file

    Returns:
        Parsed evidence data, or None if file doesn't exist or can't be parsed
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_evidence_files(evidence_dir: Path) -> Dict[str, Any]:
    """
    Load all evidence files from the evidence directory.

    Args:
        evidence_dir: Base evidence directory

    Returns:
        Dictionary with all evidence data organized by category
    """
    evidence = {}

    # Test execution
    test_results_file = evidence_dir / "test-execution/test-results.json"
    evidence["tests"] = load_evidence_file(test_results_file) or {}

    # Coverage
    coverage_file = evidence_dir / "coverage/coverage-by-module.json"
    evidence["coverage"] = load_evidence_file(coverage_file) or {}

    # Linting
    lint_file = evidence_dir / "linting/lint-errors.json"
    evidence["linting"] = load_evidence_file(lint_file) or {}

    # Code quality
    # For code quality, we need to derive from the summary since it's not JSON
    code_quality_summary = {}
    todos_file = evidence_dir / "code-quality/todos-placeholders.txt"
    if todos_file.exists():
        try:
            with open(todos_file, "r") as f:
                content = f.read()
                # Simple parsing - count occurrences
                code_quality_summary["todos_found"] = content.count("File:")
                code_quality_summary["zero_todos"] = "✅ No TODOs" in content
        except IOError:
            pass
    evidence["code_quality"] = code_quality_summary

    # Security
    security_summary_file = evidence_dir / "security/security-summary.txt"
    security_summary = {}
    if security_summary_file.exists():
        try:
            with open(security_summary_file, "r") as f:
                content = f.read()
                security_summary["content"] = content
                # Simple parsing for key indicators
                security_summary["dep_pass"] = "✅ PASS" in content and "Dependency" in content
                security_summary["sast_pass"] = (
                    "✅ PASS" in content and "Static Analysis" in content
                )
        except IOError:
            pass
    evidence["security"] = security_summary

    return evidence


def generate_production_readiness_report(evidence: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate the main production readiness report.

    Args:
        evidence: All evidence data
        output_dir: Output directory for reports
    """
    report_file = output_dir / "production-readiness-report.md"

    with open(report_file, "w") as f:
        f.write("# Production Readiness Verification Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write(
            "**Status**: Implementation Complete - Production readiness verification results\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        # Calculate overall status
        tests_pass = evidence.get("tests", {}).get("all_passed", False)
        coverage_pass = evidence.get("coverage", {}).get("overall_pass", False)
        lint_pass = evidence.get("linting", {}).get("zero_errors", False)
        code_quality_pass = evidence.get("code_quality", {}).get("zero_todos", False)

        automated_checks_pass = all([tests_pass, coverage_pass, lint_pass, code_quality_pass])

        if automated_checks_pass:
            f.write(
                "✅ **AUTOMATED CHECKS PASSED** - Production readiness criteria met for automated verification\n\n"
            )
        else:
            f.write("❌ **AUTOMATED CHECKS FAILED** - Issues found requiring attention\n\n")

        f.write("### Verification Results\n\n")
        f.write("| Criteria | Status | Details |\n")
        f.write("|----------|--------|---------|\n")

        # Test execution
        test_data = evidence.get("tests", {})
        if test_data.get("all_passed"):
            f.write("All tests passing | ✅ PASS | ")
            f.write(f"{test_data.get('passed', 0)} passed, {test_data.get('failed', 0)} failed |\n")
        else:
            f.write("All tests passing | ❌ FAIL | ")
            f.write(f"{test_data.get('passed', 0)} passed, {test_data.get('failed', 0)} failed |\n")

        # Coverage
        cov_data = evidence.get("coverage", {})
        line_pct = cov_data.get("line_coverage_percent", 0)
        branch_pct = cov_data.get("branch_coverage_percent", 0)
        if cov_data.get("overall_pass"):
            f.write(
                f"Coverage ≥80%/90% | ✅ PASS | {line_pct:.1f}% line, {branch_pct:.1f}% branch |\n"
            )
        else:
            f.write(
                f"Coverage ≥80%/90% | ❌ FAIL | {line_pct:.1f}% line, {branch_pct:.1f}% branch |\n"
            )

        # Linting
        lint_data = evidence.get("linting", {})
        if lint_data.get("zero_errors"):
            f.write(
                f"Zero linting errors | ✅ PASS | {lint_data.get('errors', 0)} errors, {lint_data.get('warnings', 0)} warnings |\n"
            )
        else:
            f.write(
                f"Zero linting errors | ❌ FAIL | {lint_data.get('errors', 0)} errors, {lint_data.get('warnings', 0)} warnings |\n"
            )

        # Code quality
        cq_data = evidence.get("code_quality", {})
        todos_found = cq_data.get("todos_found", 0)
        if cq_data.get("zero_todos"):
            f.write(f"No TODOs/PLACEHOLDERs | ✅ PASS | {todos_found} TODOs/PLACEHOLDERs found |\n")
        else:
            f.write(f"No TODOs/PLACEHOLDERs | ❌ FAIL | {todos_found} TODOs/PLACEHOLDERs found |\n")

        # Security
        sec_data = evidence.get("security", {})
        dep_pass = sec_data.get("dep_pass", False)
        sast_pass = sec_data.get("sast_pass", False)
        if dep_pass and sast_pass:
            f.write("Security scans | ✅ PASS | No critical vulnerabilities |\n")
        else:
            f.write("Security scans | ⚠️  REVIEW | Check security summary |\n")

        # Hardware (manual)
        f.write("Hardware verification | ⏳ MANUAL | Requires M1 Max execution |\n")

        f.write("\n")

        # Detailed Results
        f.write("## Detailed Results\n\n")

        # Tests
        f.write("### Testing & Quality Assurance\n\n")
        if test_data:
            f.write("**Test Execution:**\n")
            f.write(f"- Total tests: {test_data.get('total_tests', 0)}\n")
            f.write(f"- Passed: {test_data.get('passed', 0)}\n")
            f.write(f"- Failed: {test_data.get('failed', 0)}\n")
            f.write(f"- Skipped: {test_data.get('skipped', 0)}\n")
            f.write(f"- Duration: {test_data.get('duration_seconds', 0):.1f}s\n")
            if test_data.get("all_passed"):
                f.write("- ✅ **PASS**: All tests pass\n")
            else:
                f.write("- ❌ **FAIL**: Tests failed\n")
            f.write("\n")

        if cov_data:
            f.write("**Code Coverage:**\n")
            f.write(f"- Line coverage: {line_pct:.1f}%\n")
            f.write(f"- Branch coverage: {branch_pct:.1f}%\n")
            f.write(
                f"- Required: ≥{cov_data.get('thresholds', {}).get('line_coverage_required', 80)}% line, "
            )
            f.write(
                f"≥{cov_data.get('thresholds', {}).get('branch_coverage_required', 90)}% branch\n"
            )
            if cov_data.get("overall_pass"):
                f.write("- ✅ **PASS**: Coverage thresholds met\n")
            else:
                f.write("- ❌ **FAIL**: Coverage below thresholds\n")
            f.write("\n")

        # Code Quality
        f.write("### Code Quality Gates\n\n")
        if lint_data:
            f.write("**Linting:**\n")
            f.write(f"- Tool: {lint_data.get('linter', 'unknown')}\n")
            f.write(f"- Errors: {lint_data.get('errors', 0)}\n")
            f.write(f"- Warnings: {lint_data.get('warnings', 0)}\n")
            if lint_data.get("zero_errors"):
                f.write("- ✅ **PASS**: Zero linting errors\n")
            else:
                f.write("- ❌ **FAIL**: Linting errors found\n")
            f.write("\n")

        if cq_data:
            f.write("**Code Quality:**\n")
            f.write(f"- TODOs/PLACEHOLDERs found: {todos_found}\n")
            if cq_data.get("zero_todos"):
                f.write("- ✅ **PASS**: No TODOs/PLACEHOLDERs in production code\n")
            else:
                f.write("- ❌ **FAIL**: TODOs/PLACEHOLDERs found\n")
            f.write("\n")

        # Security
        f.write("### Security & Reliability\n\n")
        if sec_data.get("content"):
            f.write("**Security Assessment:**\n")
            # Extract key lines from security summary
            lines = sec_data["content"].split("\n")
            for line in lines:
                if any(
                    keyword in line for keyword in ["PASS", "FAIL", "WARNING", "found", "scanned"]
                ):
                    f.write(f"- {line.strip()}\n")
            f.write("\n")

        # Infrastructure & Persistence
        f.write("### Infrastructure & Persistence\n\n")
        f.write("**Database & Persistence:**\n")
        f.write("- ⏳ **MANUAL**: Requires verification of real database connections\n")
        f.write("- ⏳ **MANUAL**: Requires verification of migration scripts\n")
        f.write("- ⏳ **MANUAL**: Requires verification of error handling\n\n")

        # Documentation
        f.write("### Documentation & Reality Alignment\n\n")
        f.write("**Documentation Verification:**\n")
        f.write("- ⏳ **MANUAL**: Requires verification that docs match implementation\n")
        f.write("- ⏳ **MANUAL**: Requires verification of API documentation accuracy\n")
        f.write("- ⏳ **MANUAL**: Requires verification of deployment docs\n\n")

        # Hardware
        f.write("### Hardware Verification (Apple Silicon)\n\n")
        f.write("**CoreML/ANE Execution:**\n")
        f.write("- ⏳ **MANUAL**: Execute CoreML models on M1 Max hardware\n")
        f.write("- ⏳ **MANUAL**: Verify ANE residency >90%\n")
        f.write("- ⏳ **MANUAL**: Measure latency/throughput against targets\n")
        f.write("- ⏳ **MANUAL**: Verify memory budgets on real hardware\n\n")

        # Next Steps
        f.write("## Next Steps\n\n")

        if automated_checks_pass:
            f.write("✅ **Automated checks passed!**\n\n")
            f.write("To complete production readiness verification:\n\n")
            f.write("1. **Complete manual verification:**\n")
            f.write("   - Hardware verification on M1 Max\n")
            f.write("   - Infrastructure verification\n")
            f.write("   - Documentation verification\n\n")
            f.write("2. **Update audit document:**\n")
            f.write("   - Update `docs/AUDIT_END_TO_END_READINESS.md` with verified status\n")
            f.write("   - Consider claiming production-ready status\n\n")
        else:
            f.write("❌ **Automated checks failed**\n\n")
            f.write("Issues requiring attention:\n\n")

            if not tests_pass:
                f.write("- Fix failing tests (check `evidence/test-execution/failed-tests.txt`)\n")
            if not coverage_pass:
                f.write(
                    "- Improve code coverage (check `evidence/coverage/coverage-summary.txt`)\n"
                )
            if not lint_pass:
                f.write("- Fix linting errors (check `evidence/linting/lint-summary.txt`)\n")
            if not code_quality_pass:
                f.write(
                    "- Remove TODOs/PLACEHOLDERs (check `evidence/code-quality/todos-placeholders.txt`)\n"
                )

            f.write("\nAfter fixing issues:\n")
            f.write("- Re-run verification: `bash scripts/run-verification.sh`\n")
            f.write("- Complete manual verification steps\n")
            f.write("- Update audit document\n\n")

        # Evidence Location
        f.write("## Evidence Location\n\n")
        f.write("All evidence files are stored in:\n")
        f.write("```\n")
        f.write("docs/internal/audits/readiness/evidence/\n")
        f.write("```\n\n")

        f.write("### Key Evidence Files\n\n")
        f.write("- `test-execution/test-results.json` - Test execution results\n")
        f.write("- `coverage/coverage-by-module.json` - Coverage analysis\n")
        f.write("- `linting/lint-errors.json` - Linting results\n")
        f.write("- `code-quality/todos-placeholders.txt` - TODO/PLACEHOLDER scan\n")
        f.write("- `security/security-summary.txt` - Security assessment\n\n")

    print(f"[generate-report] Main report saved to: {report_file}")


def generate_criteria_reports(evidence: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate individual reports for each criteria category.

    Args:
        evidence: All evidence data
        output_dir: Output directory for reports
    """
    # Code Quality Gates Report
    code_quality_file = output_dir / "code-quality-gates.md"
    with open(code_quality_file, "w") as f:
        f.write("# Code Quality Gates Verification\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        # Linting results
        lint_data = evidence.get("linting", {})
        if lint_data:
            f.write("## Linting Results\n\n")
            f.write(f"**Tool**: {lint_data.get('linter', 'unknown')}\n\n")
            f.write(f"- Errors: {lint_data.get('errors', 0)}\n")
            f.write(f"- Warnings: {lint_data.get('warnings', 0)}\n")
            f.write(f"- Total issues: {lint_data.get('total_issues', 0)}\n\n")

            if lint_data.get("zero_errors"):
                f.write("✅ **PASS**: Zero linting errors\n")
            else:
                f.write("❌ **FAIL**: Linting errors found\n")

        # Code quality results
        cq_data = evidence.get("code_quality", {})
        if cq_data:
            f.write("\n## Code Quality Results\n\n")
            f.write(f"- TODOs/PLACEHOLDERs found: {cq_data.get('todos_found', 0)}\n\n")

            if cq_data.get("zero_todos"):
                f.write("✅ **PASS**: No TODOs/PLACEHOLDERs in production code\n")
            else:
                f.write("❌ **FAIL**: TODOs/PLACEHOLDERs found in production code\n")

    # Testing & QA Report
    testing_file = output_dir / "testing-qa.md"
    with open(testing_file, "w") as f:
        f.write("# Testing & Quality Assurance Verification\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        # Test results
        test_data = evidence.get("tests", {})
        if test_data:
            f.write("## Test Execution Results\n\n")
            f.write(f"- Total tests: {test_data.get('total_tests', 0)}\n")
            f.write(f"- Passed: {test_data.get('passed', 0)}\n")
            f.write(f"- Failed: {test_data.get('failed', 0)}\n")
            f.write(f"- Skipped: {test_data.get('skipped', 0)}\n")
            f.write(f"- Duration: {test_data.get('duration_seconds', 0):.1f}s\n\n")

            if test_data.get("all_passed"):
                f.write("✅ **PASS**: All unit tests passing\n")
            else:
                f.write("❌ **FAIL**: Tests failed\n")

        # Coverage results
        cov_data = evidence.get("coverage", {})
        if cov_data:
            f.write("\n## Code Coverage Results\n\n")
            line_pct = cov_data.get("line_coverage_percent", 0)
            branch_pct = cov_data.get("branch_coverage_percent", 0)
            f.write(f"- Line coverage: {line_pct:.1f}%\n")
            f.write(f"- Branch coverage: {branch_pct:.1f}%\n")
            f.write(
                f"- Required: ≥{cov_data.get('thresholds', {}).get('line_coverage_required', 80)}% line, "
            )
            f.write(
                f"≥{cov_data.get('thresholds', {}).get('branch_coverage_required', 90)}% branch\n\n"
            )

            if cov_data.get("overall_pass"):
                f.write("✅ **PASS**: Complete unit test coverage (80%+ line, 90%+ branch)\n")
            else:
                f.write("❌ **FAIL**: Coverage thresholds not met\n")

    # Security Report
    security_file = output_dir / "security-reliability.md"
    with open(security_file, "w") as f:
        f.write("# Security & Reliability Verification\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        sec_data = evidence.get("security", {})
        if sec_data.get("content"):
            f.write(sec_data["content"])
        else:
            f.write("Security verification not completed.\n")

    # Infrastructure & Persistence Report
    infra_file = output_dir / "infrastructure-persistence.md"
    with open(infra_file, "w") as f:
        f.write("# Infrastructure & Persistence Verification\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("**Status**: Manual verification required\n\n")
        f.write("This report requires manual verification of:\n\n")
        f.write("- Actual database persistence (not in-memory mocks)\n")
        f.write("- Database integration tests with real database\n")
        f.write("- Migration scripts tested and working\n")
        f.write("- Data consistency and rollback capabilities\n")
        f.write("- Connection pooling and error handling\n\n")

    # Documentation Report
    docs_file = output_dir / "documentation-alignment.md"
    with open(docs_file, "w") as f:
        f.write("# Documentation & Reality Alignment Verification\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("**Status**: Manual verification required\n\n")
        f.write("This report requires manual verification that:\n\n")
        f.write("- Documentation matches implementation reality\n")
        f.write("- API documentation current and accurate\n")
        f.write("- Deployment and operational docs exist\n")
        f.write("- Architecture diagrams reflect actual implementation\n")
        f.write("- README and changelogs accurate\n\n")

    print(f"[generate-report] Individual reports saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate production readiness verification reports"
    )
    parser.add_argument("--evidence-dir", required=True, help="Directory containing evidence files")
    parser.add_argument("--output", required=True, help="Output directory for generated reports")

    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[generate-report] Starting report generation...")
    print(f"[generate-report] Evidence directory: {evidence_dir}")
    print(f"[generate-report] Output directory: {output_dir}")

    # Load evidence
    evidence = load_evidence_files(evidence_dir)

    # Generate reports
    generate_production_readiness_report(evidence, output_dir)
    generate_criteria_reports(evidence, output_dir)

    print("[generate-report] Report generation completed successfully")


if __name__ == "__main__":
    main()
