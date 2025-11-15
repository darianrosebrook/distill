#!/usr/bin/env python3
"""
TODO Risk Analysis Script

Analyzes TODO patterns from todos.json to identify production risks:
1. Critical path TODOs (training/conversion)
2. "For now" implementation risk assessment
3. Generate actionable resolution plan

@author: @darianrosebrook
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def load_todos_data(todos_file: Path) -> Dict[str, Any]:
    """Load TODO data from JSON file."""
    print(f"Loading TODO data from {todos_file}...")
    with open(todos_file, 'r') as f:
        return json.load(f)


def is_critical_path(file_path: str) -> tuple[bool, str]:
    """
    Check if file is in critical path (training/conversion).
    Returns (is_critical, path_type).
    """
    if "mutants/" in file_path:
        return False, ""
    if "/test_" in file_path or "/tests/" in file_path:
        return False, ""

    if file_path.startswith("training/"):
        return True, "training"
    if file_path.startswith("conversion/") or file_path.startswith("coreml/"):
        return True, "conversion"

    return False, ""


def categorize_risk(todo: Dict[str, Any], file_path: str, path_type: str) -> str:
    """
    Categorize TODO by risk level.
    Returns: CRITICAL, HIGH, MEDIUM, or LOW
    """
    comment = todo.get("comment", "").lower()
    confidence = todo.get("confidence_score", 0.0)

    # Critical files that affect model correctness
    critical_files = [
        "training/distill_kd.py",
        "training/losses.py",
        "conversion/convert_coreml.py",
        "conversion/export_pytorch.py",
    ]

    # Check for blocking indicators
    blocking_keywords = ["blocking", "critical",
                         "must", "required", "production"]
    if any(kw in comment for kw in blocking_keywords):
        return "CRITICAL"

    # Check if in critical file
    if any(cf in file_path for cf in critical_files):
        # Check for correctness-affecting keywords
        correctness_keywords = [
            "quantize", "loss", "training", "conversion", "export",
            "model", "weights", "embedding", "correctness", "accuracy"
        ]
        if any(kw in comment for kw in correctness_keywords):
            return "HIGH"
        return "MEDIUM"

    # Check for "for now" in critical paths
    if "for now" in comment and path_type in ["training", "conversion"]:
        # Assess impact based on context
        if any(kw in comment for kw in ["quantize", "loss", "weight", "model"]):
            return "HIGH"
        return "MEDIUM"

    # PLACEHOLDER in critical paths
    if "placeholder" in comment and path_type in ["training", "conversion"]:
        return "HIGH"

    # Simplified implementations
    if "simplified" in comment and path_type in ["training", "conversion"]:
        return "MEDIUM"

    # Default risk based on path
    if path_type in ["training", "conversion"]:
        return "MEDIUM"

    return "LOW"


def extract_critical_path_todos(todos_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract TODOs from critical paths (training/conversion)."""
    critical_todos = {
        "training": [],
        "conversion": []
    }

    files = todos_data.get("files", {})

    for file_path, file_data in files.items():
        is_critical, path_type = is_critical_path(file_path)

        if not is_critical:
            continue

        hidden_todos = file_data.get("hidden_todos", {})

        for line_num, todo_data in hidden_todos.items():
            todo_info = {
                "file": file_path,
                "line": int(line_num),
                "comment": todo_data.get("comment", ""),
                "confidence": todo_data.get("confidence_score", 0.0),
                "matches": todo_data.get("matches", {}),
                "context_score": todo_data.get("context_score", 0.0),
            }

            # Categorize risk
            risk = categorize_risk(todo_data, file_path, path_type)
            todo_info["risk"] = risk

            critical_todos[path_type].append(todo_info)

    return critical_todos


def analyze_for_now_patterns(todos_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze distribution of 'for now' implementations."""
    for_now_analysis = {
        "total": 0,
        "by_path": defaultdict(int),
        "by_file": defaultdict(int),
        "by_context": defaultdict(int),
        "high_risk": [],
        "confidence_distribution": defaultdict(int),
    }

    files = todos_data.get("files", {})

    for file_path, file_data in files.items():
        if "mutants/" in file_path:
            continue

        is_critical, path_type = is_critical_path(file_path)
        hidden_todos = file_data.get("hidden_todos", {})

        for line_num, todo_data in hidden_todos.items():
            comment = todo_data.get("comment", "").lower()

            if "for now" in comment:
                for_now_analysis["total"] += 1

                if path_type:
                    for_now_analysis["by_path"][path_type] += 1
                else:
                    for_now_analysis["by_path"]["other"] += 1

                for_now_analysis["by_file"][file_path] += 1

                # Categorize by context
                if "quantize" in comment or "quantization" in comment:
                    for_now_analysis["by_context"]["quantization"] += 1
                elif "loss" in comment:
                    for_now_analysis["by_context"]["loss"] += 1
                elif "weight" in comment or "embedding" in comment:
                    for_now_analysis["by_context"]["weights"] += 1
                elif "data" in comment or "dataset" in comment:
                    for_now_analysis["by_context"]["data"] += 1
                elif "training" in comment:
                    for_now_analysis["by_context"]["training"] += 1
                elif "conversion" in comment or "export" in comment:
                    for_now_analysis["by_context"]["conversion"] += 1
                else:
                    for_now_analysis["by_context"]["other"] += 1

                # Track high-risk instances
                confidence = todo_data.get("confidence_score", 0.0)
                if confidence >= 0.9 and is_critical:
                    for_now_analysis["high_risk"].append({
                        "file": file_path,
                        "line": int(line_num),
                        "comment": todo_data.get("comment", "")[:200],
                        "confidence": confidence,
                        "path_type": path_type,
                    })

                # Confidence distribution
                conf_bucket = int(confidence * 10) / 10
                for_now_analysis["confidence_distribution"][conf_bucket] += 1

    return for_now_analysis


def read_file_context(file_path: str, line_num: int, context_lines: int = 5) -> Optional[str]:
    """Read code context around a TODO line."""
    try:
        full_path = Path(file_path)
        if not full_path.exists():
            return None

        with open(full_path, 'r') as f:
            lines = f.readlines()

        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)

        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context.append(f"{prefix}{i+1:4d}: {lines[i]}")

        return "".join(context)
    except Exception as e:
        return f"Error reading context: {e}"


def generate_risk_report(
    critical_todos: Dict[str, List[Dict[str, Any]]],
    for_now_analysis: Dict[str, Any],
    output_file: Path
) -> None:
    """Generate comprehensive risk assessment report."""

    # Count by risk level
    risk_counts = defaultdict(int)
    for path_type in ["training", "conversion"]:
        for todo in critical_todos[path_type]:
            risk_counts[todo["risk"]] += 1

    with open(output_file, 'w') as f:
        f.write("# TODO Risk Analysis Report\n\n")
        f.write(
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            f"- **Total Critical Path TODOs**: {len(critical_todos['training']) + len(critical_todos['conversion'])}\n")
        f.write(f"  - Training path: {len(critical_todos['training'])}\n")
        f.write(f"  - Conversion path: {len(critical_todos['conversion'])}\n")
        f.write(
            f"- **\"For Now\" Implementations**: {for_now_analysis['total']}\n")
        f.write(
            f"- **High-Risk \"For Now\" in Critical Paths**: {len(for_now_analysis['high_risk'])}\n\n")

        f.write("### Risk Distribution\n\n")
        for risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = risk_counts.get(risk_level, 0)
            if count > 0:
                f.write(f"- **{risk_level}**: {count}\n")
        f.write("\n")

        # Critical Path TODO Inventory
        f.write("## Critical Path TODO Inventory\n\n")

        for path_type in ["training", "conversion"]:
            f.write(
                f"### {path_type.capitalize()} Path ({len(critical_todos[path_type])} TODOs)\n\n")

            # Group by file
            by_file = defaultdict(list)
            for todo in critical_todos[path_type]:
                by_file[todo["file"]].append(todo)

            for file_path in sorted(by_file.keys()):
                todos_in_file = by_file[file_path]
                f.write(f"#### `{file_path}` ({len(todos_in_file)} TODOs)\n\n")

                for todo in sorted(todos_in_file, key=lambda x: x["line"]):
                    f.write(
                        f"**Line {todo['line']}** - Risk: **{todo['risk']}**\n\n")
                    f.write(f"```\n{todo['comment'][:500]}\n```\n\n")
                    f.write(f"- Confidence: {todo['confidence']:.2f}\n")
                    f.write(
                        f"- Context Score: {todo.get('context_score', 0.0):.2f}\n")

                    # Add code context
                    context = read_file_context(todo['file'], todo['line'])
                    if context:
                        f.write(
                            f"\n**Code Context:**\n\n```python\n{context}\n```\n\n")
                    f.write("\n")

            f.write("\n")

        # "For Now" Analysis
        f.write("## \"For Now\" Implementation Risk Analysis\n\n")

        f.write(f"### Summary\n\n")
        f.write(
            f"- **Total \"For Now\" Instances**: {for_now_analysis['total']}\n")
        f.write(
            f"- **In Critical Paths**: {for_now_analysis['by_path']['training'] + for_now_analysis['by_path']['conversion']}\n")
        f.write(f"  - Training: {for_now_analysis['by_path']['training']}\n")
        f.write(
            f"  - Conversion: {for_now_analysis['by_path']['conversion']}\n")
        f.write(f"  - Other: {for_now_analysis['by_path']['other']}\n\n")

        f.write("### Distribution by Context\n\n")
        for context, count in sorted(for_now_analysis['by_context'].items(), key=lambda x: -x[1]):
            f.write(f"- **{context}**: {count}\n")
        f.write("\n")

        f.write("### Top Files with \"For Now\" Implementations\n\n")
        top_files = sorted(
            for_now_analysis['by_file'].items(), key=lambda x: -x[1])[:10]
        for file_path, count in top_files:
            f.write(f"- `{file_path}`: {count}\n")
        f.write("\n")

        f.write("### High-Risk \"For Now\" Instances in Critical Paths\n\n")
        if for_now_analysis['high_risk']:
            for item in for_now_analysis['high_risk'][:20]:  # Top 20
                f.write(f"**`{item['file']}`** (Line {item['line']})\n\n")
                f.write(f"```\n{item['comment']}\n```\n\n")
                f.write(f"- Confidence: {item['confidence']:.2f}\n")
                f.write(f"- Path Type: {item['path_type']}\n\n")
        else:
            f.write("None found.\n\n")

        # Risk Heat Map
        f.write("## Risk Heat Map by Module\n\n")

        module_risks = defaultdict(lambda: defaultdict(int))
        for path_type in ["training", "conversion"]:
            for todo in critical_todos[path_type]:
                module = todo["file"].split(
                    "/")[-1] if "/" in todo["file"] else todo["file"]
                module_risks[module][todo["risk"]] += 1

        f.write("| Module | CRITICAL | HIGH | MEDIUM | LOW | Total |\n")
        f.write("|--------|----------|------|--------|-----|-------|\n")

        for module in sorted(module_risks.keys()):
            risks = module_risks[module]
            total = sum(risks.values())
            f.write(f"| `{module}` | {risks.get('CRITICAL', 0)} | {risks.get('HIGH', 0)} | "
                    f"{risks.get('MEDIUM', 0)} | {risks.get('LOW', 0)} | {total} |\n")

        f.write("\n")

        # Production Readiness Impact
        f.write("## Production Readiness Impact Assessment\n\n")

        critical_count = risk_counts.get("CRITICAL", 0)
        high_count = risk_counts.get("HIGH", 0)

        if critical_count > 0:
            f.write(
                f"**BLOCKER**: {critical_count} CRITICAL risk TODOs must be resolved before production.\n\n")

        if high_count > 0:
            f.write(
                f"**WARNING**: {high_count} HIGH risk TODOs should be addressed before production.\n\n")

        f.write("### Impact by Category\n\n")

        # Analyze impact by functionality
        impact_categories = {
            "Model Correctness": ["quantize", "loss", "weight", "embedding", "model"],
            "Training Stability": ["training", "dataset", "data"],
            "Conversion Success": ["conversion", "export", "coreml"],
            "Performance": ["performance", "speed", "optimization"],
        }

        for category, keywords in impact_categories.items():
            count = 0
            for path_type in ["training", "conversion"]:
                for todo in critical_todos[path_type]:
                    comment = todo["comment"].lower()
                    if any(kw in comment for kw in keywords):
                        if todo["risk"] in ["CRITICAL", "HIGH"]:
                            count += 1

            if count > 0:
                f.write(f"- **{category}**: {count} high-risk TODOs\n")

        f.write("\n")


def generate_resolution_plan(
    critical_todos: Dict[str, List[Dict[str, Any]]],
    for_now_analysis: Dict[str, Any],
    output_file: Path
) -> None:
    """Generate actionable resolution plan."""

    with open(output_file, 'w') as f:
        f.write("# TODO Resolution Plan\n\n")
        f.write(
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write(
            "This plan prioritizes TODO resolution by risk level and production impact.\n\n")

        # Group TODOs by risk and priority
        todos_by_risk = defaultdict(list)
        for path_type in ["training", "conversion"]:
            for todo in critical_todos[path_type]:
                todos_by_risk[todo["risk"]].append(todo)

        # Resolution milestones
        f.write("## Resolution Milestones\n\n")

        milestone_num = 1

        # Milestone 1: Critical Blockers
        if todos_by_risk.get("CRITICAL"):
            f.write(f"### Milestone {milestone_num}: Critical Blockers\n\n")
            f.write("**Priority**: IMMEDIATE\n")
            f.write("**Target**: Resolve all CRITICAL risk TODOs\n")
            f.write("**Estimated Effort**: 1-2 weeks\n\n")

            f.write("**TODOs to Resolve**:\n\n")
            for todo in sorted(todos_by_risk["CRITICAL"], key=lambda x: (x["file"], x["line"])):
                f.write(f"1. **`{todo['file']}`** (Line {todo['line']})\n")
                f.write(f"   - Risk: CRITICAL\n")
                f.write(f"   - Comment: {todo['comment'][:150]}...\n")
                f.write(f"   - Estimated Effort: 2-4 days\n")
                f.write(f"   - Dependencies: None\n")
                f.write(
                    f"   - Acceptance Criteria: TODO resolved, tests pass, no production blockers\n\n")

            milestone_num += 1

        # Milestone 2: High Priority
        if todos_by_risk.get("HIGH"):
            f.write(f"### Milestone {milestone_num}: High Priority TODOs\n\n")
            f.write("**Priority**: HIGH\n")
            f.write("**Target**: Resolve HIGH risk TODOs in critical paths\n")
            f.write("**Estimated Effort**: 2-3 weeks\n\n")

            # Group by file for better organization
            by_file = defaultdict(list)
            for todo in todos_by_risk["HIGH"]:
                by_file[todo["file"]].append(todo)

            f.write("**TODOs to Resolve**:\n\n")
            for file_path in sorted(by_file.keys()):
                todos_in_file = by_file[file_path]
                f.write(f"#### `{file_path}` ({len(todos_in_file)} TODOs)\n\n")

                for todo in sorted(todos_in_file, key=lambda x: x["line"]):
                    f.write(f"1. **Line {todo['line']}**\n")
                    f.write(f"   - Comment: {todo['comment'][:150]}...\n")
                    f.write(f"   - Estimated Effort: 1-3 days\n")
                    f.write(
                        f"   - Acceptance Criteria: TODO resolved, functionality verified\n\n")

            milestone_num += 1

        # Milestone 3: "For Now" Implementations
        if for_now_analysis['high_risk']:
            f.write(
                f"### Milestone {milestone_num}: High-Risk \"For Now\" Implementations\n\n")
            f.write("**Priority**: MEDIUM\n")
            f.write(
                "**Target**: Replace high-risk \"for now\" implementations with proper code\n")
            f.write("**Estimated Effort**: 3-4 weeks\n\n")

            f.write(
                f"**Total High-Risk \"For Now\" Instances**: {len(for_now_analysis['high_risk'])}\n\n")

            f.write("**Focus Areas**:\n\n")
            for context, count in sorted(for_now_analysis['by_context'].items(), key=lambda x: -x[1])[:5]:
                if count > 0:
                    f.write(f"- **{context}**: {count} instances\n")

            f.write("\n**Action Items**:\n\n")
            f.write("1. Review each high-risk \"for now\" implementation\n")
            f.write("2. Document intended behavior vs current behavior\n")
            f.write("3. Implement proper solution\n")
            f.write("4. Add tests to verify correctness\n")
            f.write("5. Remove \"for now\" comment\n\n")

        # General Recommendations
        f.write("## General Recommendations\n\n")

        f.write("### Immediate Actions\n\n")
        f.write(
            "1. **Review CRITICAL TODOs**: Assess each CRITICAL TODO for actual production impact\n")
        f.write("2. **Document \"For Now\" Decisions**: For TODOs that will remain, document why and when they should be addressed\n")
        f.write("3. **Add Tests**: Ensure all TODO resolutions are covered by tests\n")
        f.write("4. **Track Progress**: Use this plan to track resolution progress\n\n")

        f.write("### Long-term Strategy\n\n")
        f.write("1. **Prevent Accumulation**: Establish code review practices to prevent new \"for now\" implementations\n")
        f.write(
            "2. **Regular Audits**: Schedule quarterly TODO audits to track resolution progress\n")
        f.write("3. **Documentation**: Maintain documentation of temporary implementations and their intended replacements\n")
        f.write(
            "4. **Automated Detection**: Consider adding linting rules to flag new \"for now\" patterns\n\n")


def main():
    """Main execution."""
    todos_file = Path("reports/readiness/todos.json")
    output_dir = Path("reports/readiness")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    todos_data = load_todos_data(todos_file)

    # Extract critical path TODOs
    print("Extracting critical path TODOs...")
    critical_todos = extract_critical_path_todos(todos_data)

    print(f"Found {len(critical_todos['training'])} training TODOs")
    print(f"Found {len(critical_todos['conversion'])} conversion TODOs")

    # Analyze "for now" patterns
    print("Analyzing 'for now' patterns...")
    for_now_analysis = analyze_for_now_patterns(todos_data)

    print(f"Found {for_now_analysis['total']} 'for now' implementations")
    print(f"High-risk in critical paths: {len(for_now_analysis['high_risk'])}")

    # Save critical path TODOs as JSON
    critical_path_file = output_dir / "todo_critical_path.json"
    with open(critical_path_file, 'w') as f:
        json.dump(critical_todos, f, indent=2)
    print(f"Saved critical path TODOs to {critical_path_file}")

    # Generate reports
    print("Generating risk assessment report...")
    risk_report_file = output_dir / "TODO_RISK_ANALYSIS.md"
    generate_risk_report(critical_todos, for_now_analysis, risk_report_file)
    print(f"Saved risk report to {risk_report_file}")

    print("Generating resolution plan...")
    resolution_plan_file = output_dir / "TODO_RESOLUTION_PLAN.md"
    generate_resolution_plan(
        critical_todos, for_now_analysis, resolution_plan_file)
    print(f"Saved resolution plan to {resolution_plan_file}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
