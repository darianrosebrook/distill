"""
Generate comprehensive quality report for production datasets.

Analyzes dataset quality metrics, distributions, and coverage to produce
a human-readable report for review.

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime

from scripts.audit_datasets import load_jsonl, audit_worker_dataset, audit_judge_dataset, audit_drafter_dataset


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from character count."""
    return len(text) // chars_per_token


def analyze_worker_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Worker dataset distribution."""
    task_types = Counter(s.get("task_type", "unknown") for s in samples)
    caws_levels = Counter(s.get("caws_level", 0) for s in samples)
    sources = Counter(s.get("source", "unknown") for s in samples)

    # Process-step supervision coverage
    has_tool_ids = sum(1 for s in samples if s.get("tool_name_ids"))
    has_json_ids = sum(1 for s in samples if s.get("gold_json_text_ids"))
    has_integration = sum(1 for s in samples if s.get("integration_mask"))

    # CAWS context coverage
    has_caws = sum(1 for s in samples if s.get("caws_context"))
    has_evidence = sum(1 for s in samples if s.get("evidence_manifest"))
    has_provenance = sum(1 for s in samples if s.get("provenance_chain"))

    # Long-context samples
    long_context = sum(1 for s in samples if s.get("task_type") == "long_context" or s.get("metadata", {}).get("long_context"))

    # Token length distribution
    token_lengths = []
    for s in samples:
        text = s.get("teacher_text", "")
        if text:
            token_lengths.append(estimate_tokens(text))

    token_lengths.sort()
    p50 = token_lengths[len(token_lengths) // 2] if token_lengths else 0
    p95 = token_lengths[int(len(token_lengths) * 0.95)] if token_lengths else 0

    total = len(samples)

    return {
        "total_samples": total,
        "task_type_distribution": {
            "counts": dict(task_types),
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in task_types.items()},
        },
        "caws_level_distribution": {
            "counts": dict(caws_levels),
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in caws_levels.items()},
        },
        "source_distribution": {
            "counts": dict(sources),
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in sources.items()},
        },
        "process_step_supervision": {
            "tool_name_ids": {"count": has_tool_ids, "percentage": (has_tool_ids / total * 100) if total > 0 else 0},
            "gold_json_text_ids": {"count": has_json_ids, "percentage": (has_json_ids / total * 100) if total > 0 else 0},
            "integration_mask": {"count": has_integration, "percentage": (has_integration / total * 100) if total > 0 else 0},
        },
        "caws_coverage": {
            "caws_context": {"count": has_caws, "percentage": (has_caws / total * 100) if total > 0 else 0},
            "evidence_manifest": {"count": has_evidence, "percentage": (has_evidence / total * 100) if total > 0 else 0},
            "provenance_chain": {"count": has_provenance, "percentage": (has_provenance / total * 100) if total > 0 else 0},
        },
        "long_context_samples": {
            "count": long_context,
            "percentage": (long_context / total * 100) if total > 0 else 0,
        },
        "token_length_stats": {
            "min": min(token_lengths) if token_lengths else 0,
            "p50": p50,
            "p95": p95,
            "max": max(token_lengths) if token_lengths else 0,
        },
    }


def analyze_judge_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Judge dataset distribution."""
    winners = Counter(s.get("winner", "unknown") for s in samples)
    
    # Clause coverage
    all_clauses = []
    for s in samples:
        for side in ["a", "b"]:
            clauses = s.get(side, {}).get("clauses", [])
            all_clauses.extend(clauses)
    clause_counts = Counter(all_clauses)

    # Debate scores coverage
    has_debate_scores = sum(1 for s in samples if s.get("debate_scores"))
    
    # Adjudication stages coverage
    stages = Counter(s.get("adjudication_stage", "unknown") for s in samples)

    total = len(samples)

    return {
        "total_samples": total,
        "winner_distribution": {
            "counts": dict(winners),
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in winners.items()},
        },
        "clause_distribution": {
            "counts": dict(clause_counts),
            "percentages": {k: (v / (total * 2) * 100) if total > 0 else 0 for k, v in clause_counts.items()},
        },
        "debate_scores_coverage": {
            "count": has_debate_scores,
            "percentage": (has_debate_scores / total * 100) if total > 0 else 0,
        },
        "adjudication_stages": {
            "counts": dict(stages),
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in stages.items()},
        },
    }


def generate_markdown_report(
    worker_analysis: Optional[Dict[str, Any]],
    judge_analysis: Optional[Dict[str, Any]],
    drafter_analysis: Optional[Dict[str, Any]],
    output_file: Path,
) -> None:
    """Generate markdown quality report."""
    report_lines = [
        "# Dataset Quality Report - Production",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Author**: @darianrosebrook",
        "",
        "## Executive Summary",
        "",
    ]

    if worker_analysis:
        report_lines.extend([
            f"- **Worker Dataset**: {worker_analysis['total_samples']} samples",
            f"  - Process-step supervision: {worker_analysis['process_step_supervision']['tool_name_ids']['percentage']:.1f}%",
            f"  - CAWS context: {worker_analysis['caws_coverage']['caws_context']['percentage']:.1f}%",
            f"  - Long-context: {worker_analysis['long_context_samples']['percentage']:.1f}%",
            "",
        ])

    if judge_analysis:
        report_lines.extend([
            f"- **Judge Dataset**: {judge_analysis['total_samples']} samples",
            f"  - Debate scores: {judge_analysis['debate_scores_coverage']['percentage']:.1f}%",
            f"  - Adjudication stages: {len(judge_analysis['adjudication_stages']['counts'])} stages",
            "",
        ])

    if drafter_analysis:
        report_lines.extend([
            f"- **Drafter Dataset**: {drafter_analysis.get('total_samples', 0)} samples",
            "",
        ])

    # Worker details
    if worker_analysis:
        report_lines.extend([
            "## Worker Dataset Analysis",
            "",
            "### Distribution",
            "",
            "**Task Types**:",
        ])
        for task_type, pct in worker_analysis["task_type_distribution"]["percentages"].items():
            count = worker_analysis["task_type_distribution"]["counts"][task_type]
            report_lines.append(f"- {task_type}: {count} ({pct:.1f}%)")

        report_lines.extend([
            "",
            "**CAWS Levels**:",
        ])
        for level, pct in worker_analysis["caws_level_distribution"]["percentages"].items():
            count = worker_analysis["caws_level_distribution"]["counts"][level]
            report_lines.append(f"- Level {level}: {count} ({pct:.1f}%)")

        report_lines.extend([
            "",
            "### Process-Step Supervision Coverage",
            "",
            f"- `tool_name_ids`: {worker_analysis['process_step_supervision']['tool_name_ids']['count']}/{worker_analysis['total_samples']} ({worker_analysis['process_step_supervision']['tool_name_ids']['percentage']:.1f}%)",
            f"- `gold_json_text_ids`: {worker_analysis['process_step_supervision']['gold_json_text_ids']['count']}/{worker_analysis['total_samples']} ({worker_analysis['process_step_supervision']['gold_json_text_ids']['percentage']:.1f}%)",
            f"- `integration_mask`: {worker_analysis['process_step_supervision']['integration_mask']['count']}/{worker_analysis['total_samples']} ({worker_analysis['process_step_supervision']['integration_mask']['percentage']:.1f}%)",
            "",
            "### CAWS Coverage",
            "",
            f"- `caws_context`: {worker_analysis['caws_coverage']['caws_context']['count']}/{worker_analysis['total_samples']} ({worker_analysis['caws_coverage']['caws_context']['percentage']:.1f}%)",
            f"- `evidence_manifest`: {worker_analysis['caws_coverage']['evidence_manifest']['count']}/{worker_analysis['total_samples']} ({worker_analysis['caws_coverage']['evidence_manifest']['percentage']:.1f}%)",
            f"- `provenance_chain`: {worker_analysis['caws_coverage']['provenance_chain']['count']}/{worker_analysis['total_samples']} ({worker_analysis['caws_coverage']['provenance_chain']['percentage']:.1f}%)",
            "",
            "### Token Length Statistics",
            "",
            f"- Min: {worker_analysis['token_length_stats']['min']} tokens",
            f"- P50: {worker_analysis['token_length_stats']['p50']} tokens",
            f"- P95: {worker_analysis['token_length_stats']['p95']} tokens",
            f"- Max: {worker_analysis['token_length_stats']['max']} tokens",
            "",
        ])

    # Judge details
    if judge_analysis:
        report_lines.extend([
            "## Judge Dataset Analysis",
            "",
            "### Distribution",
            "",
            "**Winners**:",
        ])
        for winner, pct in judge_analysis["winner_distribution"]["percentages"].items():
            count = judge_analysis["winner_distribution"]["counts"][winner]
            report_lines.append(f"- {winner}: {count} ({pct:.1f}%)")

        report_lines.extend([
            "",
            "**CAWS Clauses**:",
        ])
        for clause, pct in sorted(judge_analysis["clause_distribution"]["percentages"].items(), key=lambda x: -x[1])[:10]:
            count = judge_analysis["clause_distribution"]["counts"][clause]
            report_lines.append(f"- {clause}: {count} ({pct:.1f}%)")

        if judge_analysis["adjudication_stages"]["counts"]:
            report_lines.extend([
                "",
                "**Adjudication Stages**:",
            ])
            for stage, pct in judge_analysis["adjudication_stages"]["percentages"].items():
                count = judge_analysis["adjudication_stages"]["counts"][stage]
                report_lines.append(f"- {stage}: {count} ({pct:.1f}%)")

        report_lines.append("")

    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[generate_quality_report] Generated report: {output_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate quality report for production datasets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--worker", help="Worker dataset JSONL file")
    ap.add_argument("--judge", help="Judge dataset JSONL file")
    ap.add_argument("--drafter", help="Drafter dataset JSONL file")
    ap.add_argument("--out", required=True, help="Output markdown report file")
    args = ap.parse_args()

    worker_analysis = None
    judge_analysis = None
    drafter_analysis = None

    if args.worker:
        samples = load_jsonl(Path(args.worker))
        worker_analysis = analyze_worker_distribution(samples)
        print(f"[generate_quality_report] Analyzed Worker dataset: {len(samples)} samples")

    if args.judge:
        samples = load_jsonl(Path(args.judge))
        judge_analysis = analyze_judge_distribution(samples)
        print(f"[generate_quality_report] Analyzed Judge dataset: {len(samples)} samples")

    if args.drafter:
        samples = load_jsonl(Path(args.drafter))
        # Drafter uses same schema as Worker
        drafter_analysis = analyze_worker_distribution(samples)
        print(f"[generate_quality_report] Analyzed Drafter dataset: {len(samples)} samples")

    generate_markdown_report(
        worker_analysis=worker_analysis,
        judge_analysis=judge_analysis,
        drafter_analysis=drafter_analysis,
        output_file=Path(args.out),
    )


if __name__ == "__main__":
    main()

