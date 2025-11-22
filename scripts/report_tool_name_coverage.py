"""
Generate coverage report for tool name inference.

Analyzes dataset after tool name inference and generates a markdown report
with coverage statistics, quality metrics, and acceptance criteria checklist.

Author: @darianrosebrook
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.schema_registry import get_registry


def load_samples(input_jsonl: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


def analyze_coverage(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tool name coverage statistics."""
    total_samples = len(samples)
    
    # Identify tool-use samples
    tool_use_samples = [
        s for s in samples
        if s.get("task_type") in ["tool_use", "caws_tool"]
        or s.get("tool_name_ids")
        or s.get("gold_json_text_ids")
        or s.get("integration_mask")
    ]
    
    # Count samples with tool_name
    has_tool_name = [s for s in tool_use_samples if s.get("tool_name_ids")]
    
    # Count by source
    source_counts = Counter()
    confidence_counts = Counter()
    tool_name_counts = Counter()
    
    for sample in tool_use_samples:
        if sample.get("tool_name_ids"):
            source = sample.get("tool_name_source", "unknown")
            confidence = sample.get("tool_name_confidence", "unknown")
            tool_name = sample.get("tool_name", "unknown")
            
            source_counts[source] += 1
            confidence_counts[confidence] += 1
            tool_name_counts[tool_name] += 1
    
    # Calculate metrics
    tool_use_count = len(tool_use_samples)
    coverage_count = len(has_tool_name)
    coverage_rate = coverage_count / tool_use_count if tool_use_count > 0 else 0.0
    
    # Ambiguity rate
    ambiguous_count = confidence_counts.get("ambiguous", 0)
    ambiguity_rate = ambiguous_count / tool_use_count if tool_use_count > 0 else 0.0
    
    # Parse error rate (estimate from samples without tool_name but with JSON)
    samples_with_json = [
        s for s in tool_use_samples
        if s.get("gold_json_text_ids") and not s.get("tool_name_ids")
    ]
    parse_error_rate = len(samples_with_json) / tool_use_count if tool_use_count > 0 else 0.0
    
    return {
        "total_samples": total_samples,
        "tool_use_samples": tool_use_count,
        "coverage_count": coverage_count,
        "coverage_rate": coverage_rate,
        "source_counts": dict(source_counts),
        "confidence_counts": dict(confidence_counts),
        "tool_name_counts": dict(tool_name_counts),
        "ambiguous_count": ambiguous_count,
        "ambiguity_rate": ambiguity_rate,
        "parse_error_rate": parse_error_rate,
    }


def validate_registry(tool_names: List[str]) -> Dict[str, Any]:
    """Validate that all tool names exist in registry."""
    try:
        registry = get_registry()
        registry_tools = set(registry.list_tools())
        
        invalid_tools = []
        for tool_name in tool_names:
            if tool_name and tool_name != "unknown" and tool_name not in registry_tools:
                invalid_tools.append(tool_name)
        
        return {
            "total_unique_tools": len(set(tool_names)),
            "invalid_tools": invalid_tools,
            "all_valid": len(invalid_tools) == 0,
        }
    except Exception as e:
        return {
            "error": str(e),
            "all_valid": False,
        }


def generate_report(
    input_jsonl: Path,
    output_md: Path,
    min_coverage: float = 0.20,
    max_ambiguity: float = 0.05,
) -> Dict[str, Any]:
    """Generate coverage report."""
    print(f"[report_coverage] Loading samples from {input_jsonl}")
    samples = load_samples(input_jsonl)
    
    print(f"[report_coverage] Analyzing coverage...")
    stats = analyze_coverage(samples)
    
    # Validate registry
    tool_names = [s.get("tool_name") for s in samples if s.get("tool_name")]
    registry_validation = validate_registry(tool_names)
    
    # Check acceptance criteria
    coverage_pass = stats["coverage_rate"] >= min_coverage
    ambiguity_pass = stats["ambiguity_rate"] <= max_ambiguity
    registry_pass = registry_validation.get("all_valid", False)
    parse_error_pass = stats["parse_error_rate"] <= 0.02
    
    all_pass = coverage_pass and ambiguity_pass and registry_pass and parse_error_pass
    
    # Generate markdown report
    report_lines = [
        "# Tool Name Inference Coverage Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset**: `{input_jsonl.name}`",
        "",
        "## Executive Summary",
        "",
        f"- **Total Samples**: {stats['total_samples']:,}",
        f"- **Tool-Use Samples**: {stats['tool_use_samples']:,}",
        f"- **Coverage**: {stats['coverage_count']:,}/{stats['tool_use_samples']:,} ({stats['coverage_rate']:.1%})",
        f"- **Ambiguity Rate**: {stats['ambiguity_rate']:.1%}",
        f"- **Parse Error Rate**: {stats['parse_error_rate']:.1%}",
        "",
        "## Acceptance Criteria",
        "",
        "| Criterion | Threshold | Actual | Status |",
        "|-----------|-----------|--------|--------|",
        f"| Tool-use samples with `tool_name` | ≥ {min_coverage:.0%} | {stats['coverage_rate']:.1%} | {'✅ PASS' if coverage_pass else '❌ FAIL'} |",
        f"| Ambiguity rate | < {max_ambiguity:.0%} | {stats['ambiguity_rate']:.1%} | {'✅ PASS' if ambiguity_pass else '❌ FAIL'} |",
        f"| All `tool_name` values in registry | 100% | {len(registry_validation.get('invalid_tools', []))} invalid | {'✅ PASS' if registry_pass else '❌ FAIL'} |",
        f"| JSON parse error rate | < 2% | {stats['parse_error_rate']:.1%} | {'✅ PASS' if parse_error_pass else '❌ FAIL'} |",
        "",
        f"**Overall Status**: {'✅ **ALL CRITERIA PASSED**' if all_pass else '❌ **CRITERIA FAILED**'}",
        "",
        "## Coverage Breakdown",
        "",
        "### By Source",
        "",
        "| Source | Count | Percentage |",
        "|--------|-------|------------|",
    ]
    
    for source, count in sorted(stats["source_counts"].items(), key=lambda x: -x[1]):
        pct = count / stats["coverage_count"] if stats["coverage_count"] > 0 else 0.0
        report_lines.append(f"| `{source}` | {count:,} | {pct:.1%} |")
    
    report_lines.extend([
        "",
        "### By Confidence",
        "",
        "| Confidence | Count | Percentage |",
        "|------------|-------|------------|",
    ])
    
    for confidence, count in sorted(stats["confidence_counts"].items(), key=lambda x: -x[1]):
        pct = count / stats["coverage_count"] if stats["coverage_count"] > 0 else 0.0
        report_lines.append(f"| `{confidence}` | {count:,} | {pct:.1%} |")
    
    report_lines.extend([
        "",
        "## Tool Name Distribution",
        "",
        "Top 10 most common tool names:",
        "",
        "| Tool Name | Count |",
        "|-----------|-------|",
    ])
    
    for tool_name, count in stats["tool_name_counts"].most_common(10):
        report_lines.append(f"| `{tool_name}` | {count:,} |")
    
    if len(stats["tool_name_counts"]) > 10:
        report_lines.append(f"\n*({len(stats['tool_name_counts']) - 10} more tool names)*")
    
    report_lines.extend([
        "",
        "## Registry Validation",
        "",
    ])
    
    if registry_validation.get("error"):
        report_lines.append(f"**Error**: {registry_validation['error']}")
    else:
        report_lines.append(f"- **Total Unique Tools**: {registry_validation.get('total_unique_tools', 0)}")
        invalid_tools = registry_validation.get("invalid_tools", [])
        if invalid_tools:
            report_lines.append(f"- **Invalid Tools**: {len(invalid_tools)}")
            for tool in invalid_tools:
                report_lines.append(f"  - `{tool}`")
        else:
            report_lines.append("- **All tools valid** ✅")
    
    report_lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    
    if not coverage_pass:
        report_lines.append(
            f"- ⚠️ Coverage ({stats['coverage_rate']:.1%}) below threshold ({min_coverage:.0%}). "
            f"Consider improving schema matching or adding more heuristics."
        )
    
    if not ambiguity_pass:
        report_lines.append(
            f"- ⚠️ Ambiguity rate ({stats['ambiguity_rate']:.1%}) above threshold ({max_ambiguity:.0%}). "
            f"Consider refining schema key matching or adding disambiguation logic."
        )
    
    if not registry_pass:
        report_lines.append(
            f"- ⚠️ Found {len(registry_validation.get('invalid_tools', []))} invalid tool names. "
            f"Review inference logic or update tool registry."
        )
    
    if not parse_error_pass:
        report_lines.append(
            f"- ⚠️ Parse error rate ({stats['parse_error_rate']:.1%}) above 2% threshold. "
            f"Consider improving JSON extraction or repair logic."
        )
    
    if all_pass:
        report_lines.append("- ✅ All acceptance criteria passed. Dataset ready for training.")
    
    # Write report
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[report_coverage] Generated report: {output_md}")
    
    return {
        "stats": stats,
        "registry_validation": registry_validation,
        "acceptance_criteria": {
            "coverage": coverage_pass,
            "ambiguity": ambiguity_pass,
            "registry": registry_pass,
            "parse_error": parse_error_pass,
            "all_pass": all_pass,
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate tool name inference coverage report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input dataset JSONL (after inference)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output markdown report file",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.20,
        help="Minimum coverage threshold (default: 0.20)",
    )
    ap.add_argument(
        "--max-ambiguity",
        type=float,
        default=0.05,
        help="Maximum ambiguity rate (default: 0.05)",
    )
    
    args = ap.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.out)
    
    if not input_file.exists():
        print(f"[report_coverage] ERROR: Input file not found: {input_file}")
        return 1
    
    try:
        result = generate_report(
            input_file,
            output_file,
            min_coverage=args.min_coverage,
            max_ambiguity=args.max_ambiguity,
        )
        
        if result["acceptance_criteria"]["all_pass"]:
            print(f"\n[report_coverage] SUCCESS: All acceptance criteria passed")
            return 0
        else:
            print(f"\n[report_coverage] WARN: Some acceptance criteria failed")
            return 1
            
    except Exception as e:
        print(f"\n[report_coverage] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

