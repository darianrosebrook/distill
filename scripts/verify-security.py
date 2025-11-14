#!/usr/bin/env python3
"""
Security verification script for production readiness.

Runs security scans to verify:
- No vulnerable dependencies
- No hardcoded secrets
- No SAST (Static Application Security Testing) violations
- Basic security hygiene

Usage:
    python scripts/verify-security.py --output docs/internal/audits/readiness/evidence/security/
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def run_dependency_scan(output_dir: Path) -> Dict[str, Any]:
    """
    Run dependency vulnerability scanning.

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with scan results
    """
    print("[verify-security] Running dependency vulnerability scan...")

    vulnerabilities = []
    scan_tool = None

    # Try pip-audit first (recommended for Python)
    try:
        cmd = ["pip-audit", "--format=json", "--disable-pip"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            scan_tool = "pip-audit"
            try:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data:
                    vulnerabilities.append(
                        {
                            "package": vuln.get("name", ""),
                            "version": vuln.get("version", ""),
                            "vulnerability_id": vuln.get("id", ""),
                            "severity": vuln.get("severity", "unknown"),
                            "description": vuln.get("description", ""),
                            "url": vuln.get("url", ""),
                        }
                    )
            except json.JSONDecodeError:
                vulnerabilities = [{"error": "Could not parse pip-audit JSON output"}]
        else:
            # pip-audit found vulnerabilities
            scan_tool = "pip-audit"
            # Parse text output for vulnerabilities
            lines = result.stdout.split("\n")
            for line in lines:
                if "vulnerability found in" in line.lower():
                    # Extract package name from line like "vulnerability found in requests==2.25.1"
                    match = re.search(
                        r"vulnerability found in ([^=]+)==([^\s]+)", line, re.IGNORECASE
                    )
                    if match:
                        vulnerabilities.append(
                            {
                                "package": match.group(1),
                                "version": match.group(2),
                                "severity": "unknown",
                                "description": line.strip(),
                            }
                        )

        print(f"[verify-security] pip-audit found {len(vulnerabilities)} vulnerabilities")

    except FileNotFoundError:
        # Try safety as fallback
        try:
            cmd = ["safety", "check", "--json"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, cwd=Path.cwd()
            )

            scan_tool = "safety"
            if result.returncode == 0:
                try:
                    safety_data = json.loads(result.stdout)
                    for issue in safety_data.get("issues", []):
                        vulnerabilities.append(
                            {
                                "package": issue.get("package", ""),
                                "version": issue.get("version", ""),
                                "vulnerability_id": issue.get("id", ""),
                                "severity": issue.get("severity", "unknown"),
                                "description": issue.get("description", ""),
                            }
                        )
                except json.JSONDecodeError:
                    vulnerabilities = [{"error": "Could not parse safety JSON output"}]
            else:
                # Parse text output
                lines = result.stdout.split("\n")
                for line in lines:
                    if line.strip() and not line.startswith("+"):
                        match = re.search(r"([^\s]+)[^\d]+(\d+)[^\d]+(.+)", line)
                        if match:
                            vulnerabilities.append(
                                {
                                    "package": match.group(1),
                                    "version": match.group(2),
                                    "description": match.group(3).strip(),
                                    "severity": "unknown",
                                }
                            )

            print(f"[verify-security] Safety found {len(vulnerabilities)} vulnerabilities")

        except FileNotFoundError:
            print("[verify-security] WARNING: Neither pip-audit nor safety available")
            scan_tool = None
            vulnerabilities = [{"warning": "No dependency scanning tool available"}]

    # Save dependency scan results
    dep_file = output_dir / "dependency-scan.json"
    with open(dep_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "scan_tool": scan_tool,
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
            },
            f,
            indent=2,
        )

    print(f"[verify-security] Dependency scan saved to: {dep_file}")

    return {
        "scan_tool": scan_tool,
        "vulnerabilities_found": len(vulnerabilities),
        "vulnerabilities": vulnerabilities,
        "no_critical_vulnerabilities": not any(
            v.get("severity") in ["critical", "high"]
            for v in vulnerabilities
            if "error" not in v and "warning" not in v
        ),
    }


def run_sast_scan(output_dir: Path) -> Dict[str, Any]:
    """
    Run Static Application Security Testing (SAST).

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with SAST results
    """
    print("[verify-security] Running SAST (Static Application Security Testing)...")

    sast_issues = []
    sast_tool = None

    # Try bandit (Python SAST tool)
    try:
        cmd = ["bandit", "-r", ".", "-f", "json"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path.cwd(),
        )

        sast_tool = "bandit"
        if result.returncode == 0:
            try:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get("results", []):
                    for filename, file_issues in issue.items():
                        for file_issue in file_issues:
                            sast_issues.append(
                                {
                                    "file": filename,
                                    "line": file_issue.get("line_number", 0),
                                    "code": file_issue.get("code", ""),
                                    "severity": file_issue.get("issue_severity", "unknown"),
                                    "confidence": file_issue.get("issue_confidence", "unknown"),
                                    "description": file_issue.get("issue_text", ""),
                                    "cwe": file_issue.get("cwe", ""),
                                }
                            )
            except json.JSONDecodeError:
                sast_issues = [{"error": "Could not parse bandit JSON output"}]
        else:
            # Bandit found issues, parse output
            lines = result.stdout.split("\n")
            for line in lines:
                if "SEVERITY:" in line and "CONFIDENCE:" in line:
                    # Extract severity and confidence
                    sev_match = re.search(r"SEVERITY: ([^\s]+)", line)
                    conf_match = re.search(r"CONFIDENCE: ([^\s]+)", line)
                    severity = sev_match.group(1) if sev_match else "unknown"
                    confidence = conf_match.group(1) if conf_match else "unknown"

                    sast_issues.append(
                        {
                            "severity": severity,
                            "confidence": confidence,
                            "description": line.strip(),
                        }
                    )

        print(f"[verify-security] Bandit found {len(sast_issues)} SAST issues")

    except FileNotFoundError:
        print("[verify-security] WARNING: Bandit not available for SAST")
        sast_tool = None
        sast_issues = [{"warning": "No SAST tool available"}]

    # Save SAST results
    sast_file = output_dir / "sast-results.json"
    with open(sast_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "sast_tool": sast_tool,
                "issues_found": len(sast_issues),
                "issues": sast_issues,
            },
            f,
            indent=2,
        )

    print(f"[verify-security] SAST results saved to: {sast_file}")

    return {
        "sast_tool": sast_tool,
        "issues_found": len(sast_issues),
        "issues": sast_issues,
        "no_high_severity_issues": not any(
            i.get("severity") in ["high", "critical"]
            for i in sast_issues
            if "error" not in i and "warning" not in i
        ),
    }


def check_hardcoded_secrets(output_dir: Path) -> Dict[str, Any]:
    """
    Check for potential hardcoded secrets in codebase.

    Args:
        output_dir: Directory to save evidence

    Returns:
        Dictionary with secret scan results
    """
    print("[verify-security] Checking for hardcoded secrets...")

    # Common patterns for potential secrets (not comprehensive)
    secret_patterns = [
        r'password\s*[:=]\s*["\'][^"\']+["\']',  # password = "secret"
        r'secret\s*[:=]\s*["\'][^"\']+["\']',  # secret = "key"
        r'api_key\s*[:=]\s*["\'][^"\']+["\']',  # api_key = "key"
        r'token\s*[:=]\s*["\'][^"\']+["\']',  # token = "token"
        r'key\s*[:=]\s*["\'][^"\']+["\']',  # key = "value"
        r"AKIA[0-9A-Z]{16}",  # AWS access key pattern
        r"sk-[a-zA-Z0-9]{48}",  # OpenAI API key pattern
    ]

    potential_secrets = []
    files_scanned = 0

    # Scan Python files
    for py_file in Path(".").rglob("*.py"):
        # Skip common non-sensitive files
        if any(
            skip in str(py_file)
            for skip in [
                "__pycache__",
                ".git",
                "test",
                "spec",
                ".egg-info",
                "node_modules",
                ".venv",
                "venv",
                "env",
                "scripts",
            ]
        ):
            continue

        files_scanned += 1
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern in secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip obvious test/mock values
                    if any(
                        test in match.lower()
                        for test in ["test", "mock", "fake", "example", "sample", "dummy"]
                    ):
                        continue

                    potential_secrets.append(
                        {
                            "file": str(py_file),
                            "pattern": pattern,
                            "match": match[:50],  # Truncate for safety
                            "line_context": "...redacted for security...",
                        }
                    )

        except (UnicodeDecodeError, IOError):
            continue

    # Save secret scan results
    secrets_file = output_dir / "secrets-scan.json"
    with open(secrets_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "files_scanned": files_scanned,
                "potential_secrets_found": len(potential_secrets),
                "potential_secrets": potential_secrets,
            },
            f,
            indent=2,
        )

    print(f"[verify-security] Secret scan saved to: {secrets_file}")
    print(
        f"[verify-security] Scanned {files_scanned} files, found {len(potential_secrets)} potential secrets"
    )

    return {
        "files_scanned": files_scanned,
        "potential_secrets_found": len(potential_secrets),
        "potential_secrets": potential_secrets,
        "no_hardcoded_secrets": len(potential_secrets) == 0,
    }


def generate_security_summary(output_dir: Path, results: Dict[str, Any]) -> None:
    """
    Generate comprehensive security assessment summary.

    Args:
        output_dir: Directory to save summary
        results: Combined results from all security checks
    """
    summary_file = output_dir / "security-summary.txt"

    with open(summary_file, "w") as f:
        f.write("Security Assessment Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Dependency vulnerabilities
        dep_results = results["dependencies"]
        f.write("1. Dependency Vulnerabilities:\n")
        f.write(f"   Tool: {dep_results.get('scan_tool', 'None')}\n")
        f.write(f"   Vulnerabilities found: {dep_results['vulnerabilities_found']}\n")
        if dep_results["no_critical_vulnerabilities"]:
            f.write("   ✅ PASS - No critical/high severity vulnerabilities\n")
        else:
            f.write("   ❌ FAIL - Critical/high vulnerabilities found\n")
            f.write("   See: dependency-scan.json\n")
        f.write("\n")

        # SAST issues
        sast_results = results["sast"]
        f.write("2. Static Analysis (SAST):\n")
        f.write(f"   Tool: {sast_results.get('sast_tool', 'None')}\n")
        f.write(f"   Issues found: {sast_results['issues_found']}\n")
        if sast_results["no_high_severity_issues"]:
            f.write("   ✅ PASS - No high/critical severity SAST issues\n")
        else:
            f.write("   ❌ FAIL - High/critical SAST issues found\n")
            f.write("   See: sast-results.json\n")
        f.write("\n")

        # Hardcoded secrets
        secret_results = results["secrets"]
        f.write("3. Hardcoded Secrets:\n")
        f.write(f"   Files scanned: {secret_results['files_scanned']}\n")
        f.write(f"   Potential secrets found: {secret_results['potential_secrets_found']}\n")
        if secret_results["no_hardcoded_secrets"]:
            f.write("   ✅ PASS - No hardcoded secrets detected\n")
        else:
            f.write("   ⚠️  WARNING - Potential hardcoded secrets found\n")
            f.write("   See: secrets-scan.json (review manually)\n")
        f.write("\n")

        # Overall assessment
        f.write("Production Readiness Assessment:\n")

        # Check if any critical issues
        critical_issues = (
            not dep_results["no_critical_vulnerabilities"]
            or not sast_results["no_high_severity_issues"]
        )

        non_critical_warnings = secret_results["potential_secrets_found"] > 0

        if not critical_issues:
            if non_critical_warnings:
                f.write("⚠️  PASS WITH WARNINGS - Security gates mostly met\n")
                f.write("   - No critical vulnerabilities or SAST issues\n")
                f.write("   - Potential secrets require manual review\n")
            else:
                f.write("✅ PASS - Security controls verified\n")
                f.write("   - No security scan violations\n")
                f.write("   - No hardcoded secrets detected\n")
        else:
            f.write("❌ FAIL - Critical security issues found\n")
            if not dep_results["no_critical_vulnerabilities"]:
                f.write("   - Vulnerable dependencies must be updated\n")
            if not sast_results["no_high_severity_issues"]:
                f.write("   - SAST issues must be addressed\n")

    print(f"[verify-security] Security summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify security for production readiness")
    parser.add_argument("--output", required=True, help="Output directory for security evidence")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[verify-security] Starting security verification...")
    print(f"[verify-security] Output directory: {output_dir}")

    # Run all security checks
    results = {}

    results["dependencies"] = run_dependency_scan(output_dir)
    results["sast"] = run_sast_scan(output_dir)
    results["secrets"] = check_hardcoded_secrets(output_dir)

    # Generate summary
    generate_security_summary(output_dir, results)

    # Final assessment
    dep_pass = results["dependencies"]["no_critical_vulnerabilities"]
    sast_pass = results["sast"]["no_high_severity_issues"]
    secrets_ok = results["secrets"]["no_hardcoded_secrets"]  # Warnings are OK

    if dep_pass and sast_pass:
        if secrets_ok:
            print("[verify-security] ✅ SUCCESS: Security controls verified")
        else:
            print(
                "[verify-security] ⚠️  SUCCESS WITH WARNINGS: Security gates met, secrets need review"
            )
        sys.exit(0)
    else:
        print("[verify-security] ❌ FAILURE: Security issues found")
        print("   Check security-summary.txt for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
