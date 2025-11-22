"""
Randomly sample and quality-check dataset items.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file, skipping headers."""
    samples = []
    if not file_path.exists():
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                if sample.get("__header__"):
                    continue
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    return samples


def assess_worker_quality(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Assess quality of a Worker sample."""
    issues = []
    strengths = []

    # Check prompt quality
    prompt = sample.get("prompt", "")
    if len(prompt) < 20:
        issues.append("Very short prompt")
    elif len(prompt) > 10000:
        issues.append("Very long prompt")
    else:
        strengths.append("Reasonable prompt length")

    if not prompt.strip():
        issues.append("Empty prompt")

    # Check prompt clarity
    if prompt.count("?") == 0 and prompt.count(".") == 0:
        issues.append("Prompt lacks clear structure")
    else:
        strengths.append("Prompt has clear structure")

    # Check teacher_text quality
    teacher_text = sample.get("teacher_text", "")
    if teacher_text:
        if len(teacher_text) < 20:
            issues.append("Very short teacher_text")
        elif len(teacher_text) > 50000:
            issues.append("Very long teacher_text")
        else:
            strengths.append("Reasonable teacher_text length")

        # Check for code blocks
        if "```" in teacher_text:
            strengths.append("Contains code blocks")

        # Check for tool use
        if "tool" in teacher_text.lower() or "json" in teacher_text.lower():
            strengths.append("Contains tool usage")
    else:
        if sample.get("source") == "synthetic":
            strengths.append("Synthetic sample (no teacher_text expected)")
        else:
            issues.append("Missing teacher_text")

    # Check CAWS compliance
    caws_level = sample.get("caws_level", 0)
    has_caws_context = "caws_context" in sample or "working_spec" in sample
    has_evidence = "evidence_manifest" in sample
    has_provenance = "provenance_chain" in sample

    if caws_level == 2 and not (has_evidence and has_provenance):
        issues.append("caws_level=2 but missing evidence/provenance")
    elif caws_level >= 1 and not has_caws_context:
        issues.append(f"caws_level={caws_level} but missing CAWS context")
    else:
        if caws_level > 0:
            strengths.append(f"CAWS level {caws_level} properly implemented")

    # Check task_type
    task_type = sample.get("task_type", "")
    if task_type not in ["plain_kd", "tool_use", "caws_tool", "long_context"]:
        issues.append(f"Invalid task_type: {task_type}")
    else:
        strengths.append(f"Valid task_type: {task_type}")

    # Overall assessment
    quality_score = 10
    quality_score -= len(issues) * 2
    quality_score = max(0, min(10, quality_score))

    return {
        "quality_score": quality_score,
        "issues": issues,
        "strengths": strengths,
        "prompt_length": len(prompt),
        "teacher_text_length": len(teacher_text) if teacher_text else 0,
        "caws_level": caws_level,
        "task_type": task_type,
    }


def assess_judge_quality(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Assess quality of a Judge sample."""
    issues = []
    strengths = []

    # Check prompt quality
    prompt = sample.get("prompt", "")
    if len(prompt) < 20:
        issues.append("Very short prompt")
    elif len(prompt) > 20000:
        issues.append("Very long prompt")
    else:
        strengths.append("Reasonable prompt length")

    # Check a and b
    a = sample.get("a", {})
    b = sample.get("b", {})

    a_text = a.get("text", "")
    b_text = b.get("text", "")

    if len(a_text) < 10:
        issues.append("Very short a.text")
    else:
        strengths.append("a.text has reasonable length")

    if len(b_text) < 10:
        issues.append("Very short b.text")
    else:
        strengths.append("b.text has reasonable length")

    # Check clauses
    a_clauses = a.get("clauses", [])
    b_clauses = b.get("clauses", [])

    valid_clauses = {
        "EVIDENCE_COMPLETENESS",
        "BUDGET_ADHERENCE",
        "GATE_INTEGRITY",
        "PROVENANCE_CLARITY",
        "WAIVER_JUSTIFICATION",
    }

    if not a_clauses and not b_clauses:
        issues.append("Both a and b have no clauses")
    else:
        strengths.append("Samples have clause annotations")

    invalid_clauses = []
    for clause in a_clauses + b_clauses:
        if clause not in valid_clauses:
            invalid_clauses.append(clause)

    if invalid_clauses:
        issues.append(f"Invalid clauses: {invalid_clauses}")
    else:
        strengths.append("All clauses are valid CAWS clauses")

    # Check winner
    winner = sample.get("winner", "")
    if winner not in ["a", "b", "tie"]:
        issues.append(f"Invalid winner: {winner}")
    else:
        strengths.append(f"Valid winner: {winner}")

    # Check for meaningful differences
    if a_text == b_text:
        issues.append("a.text and b.text are identical")
    else:
        strengths.append("a and b have different content")

    # Overall assessment
    quality_score = 10
    quality_score -= len(issues) * 2
    quality_score = max(0, min(10, quality_score))

    return {
        "quality_score": quality_score,
        "issues": issues,
        "strengths": strengths,
        "prompt_length": len(prompt),
        "a_text_length": len(a_text),
        "b_text_length": len(b_text),
        "a_clauses_count": len(a_clauses),
        "b_clauses_count": len(b_clauses),
        "winner": winner,
    }


def assess_drafter_quality(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Assess quality of a Drafter sample."""
    issues = []
    strengths = []

    # Check prompt quality
    prompt = sample.get("prompt", "")
    if len(prompt) < 20:
        issues.append("Very short prompt")
    else:
        strengths.append("Reasonable prompt length")

    # Check teacher_text length (Drafter should be short)
    teacher_text = sample.get("teacher_text", "")
    text_length = len(teacher_text)

    if text_length == 0:
        issues.append("Empty teacher_text")
    elif text_length > 2000:
        issues.append(f"Too long for Drafter: {text_length} chars")
    elif text_length < 50:
        issues.append("Very short teacher_text")
    else:
        strengths.append(
            f"Appropriate length for Drafter: {text_length} chars")

    # Check length bucket
    supervision = sample.get("supervision", {})
    length_bucket = supervision.get("target_length_bucket")
    if length_bucket not in [200, 400, 800]:
        issues.append(f"Invalid length bucket: {length_bucket}")
    else:
        strengths.append(f"Valid length bucket: {length_bucket}")

    # Overall assessment
    quality_score = 10
    quality_score -= len(issues) * 2
    quality_score = max(0, min(10, quality_score))

    return {
        "quality_score": quality_score,
        "issues": issues,
        "strengths": strengths,
        "prompt_length": len(prompt),
        "text_length": text_length,
        "length_bucket": length_bucket,
    }


def print_sample(sample: Dict[str, Any], assessment: Dict[str, Any], sample_type: str):
    """Print sample with quality assessment."""
    print(f"\n{'='*80}")
    print(f"SAMPLE TYPE: {sample_type.upper()}")
    print(f"QUALITY SCORE: {assessment['quality_score']}/10")
    print(f"{'='*80}")

    if assessment['issues']:
        print(f"\n❌ ISSUES ({len(assessment['issues'])}):")
        for issue in assessment['issues']:
            print(f"  • {issue}")

    if assessment['strengths']:
        print(f"\n✅ STRENGTHS ({len(assessment['strengths'])}):")
        for strength in assessment['strengths']:
            print(f"  • {strength}")

    print(f"\n📋 SAMPLE DATA:")
    print(f"  ID: {sample.get('id', 'N/A')}")

    if sample_type == "worker":
        print(f"  Task Type: {sample.get('task_type', 'N/A')}")
        print(f"  CAWS Level: {sample.get('caws_level', 'N/A')}")
        print(f"  Source: {sample.get('source', 'N/A')}")
        print(f"\n  Prompt ({assessment['prompt_length']} chars):")
        prompt = sample.get("prompt", "")
        print(f"    {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        teacher_text = sample.get("teacher_text", "")
        if teacher_text:
            print(
                f"\n  Teacher Text ({assessment['teacher_text_length']} chars):")
            print(
                f"    {teacher_text[:300]}{'...' if len(teacher_text) > 300 else ''}")
        else:
            print(f"\n  Teacher Text: (not present - synthetic sample)")

    elif sample_type == "judge":
        print(f"  Winner: {sample.get('winner', 'N/A')}")
        print(f"\n  Prompt ({assessment['prompt_length']} chars):")
        prompt = sample.get("prompt", "")
        print(f"    {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        a = sample.get("a", {})
        b = sample.get("b", {})
        print(
            f"\n  A ({assessment['a_text_length']} chars, {assessment['a_clauses_count']} clauses):")
        print(f"    Clauses: {a.get('clauses', [])}")
        print(
            f"    Text: {a.get('text', '')[:200]}{'...' if len(a.get('text', '')) > 200 else ''}")

        print(
            f"\n  B ({assessment['b_text_length']} chars, {assessment['b_clauses_count']} clauses):")
        print(f"    Clauses: {b.get('clauses', [])}")
        print(
            f"    Text: {b.get('text', '')[:200]}{'...' if len(b.get('text', '')) > 200 else ''}")

    elif sample_type == "drafter":
        print(f"  Task Type: {sample.get('task_type', 'N/A')}")
        print(f"  Length Bucket: {assessment.get('length_bucket', 'N/A')}")
        print(f"\n  Prompt ({assessment['prompt_length']} chars):")
        prompt = sample.get("prompt", "")
        print(f"    {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        teacher_text = sample.get("teacher_text", "")
        print(f"\n  Teacher Text ({assessment['text_length']} chars):")
        print(
            f"    {teacher_text[:300]}{'...' if len(teacher_text) > 300 else ''}")


def main():
    ap = argparse.ArgumentParser(
        description="Randomly sample and quality-check dataset items",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--worker",
        nargs="+",
        help="Worker dataset files to sample from",
    )
    ap.add_argument(
        "--judge",
        nargs="+",
        help="Judge dataset files to sample from",
    )
    ap.add_argument(
        "--drafter",
        nargs="+",
        help="Drafter dataset files to sample from",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of samples per dataset type (default: 3)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Default to standard datasets if none specified
    if not (args.worker or args.judge or args.drafter):
        args.worker = [
            "data/worker_combined.jsonl",
            "data/caws_tool_examples.jsonl",
        ]
        args.judge = [
            "data/judge/train.jsonl",
            "data/judge/caws_scenarios.jsonl",
        ]
        args.drafter = [
            "data/drafter/drafter_dataset_final.jsonl",
        ]

    # Sample Worker datasets
    if args.worker:
        print(f"\n{'='*80}")
        print("WORKER DATASET SAMPLES")
        print(f"{'='*80}")

        all_worker_samples = []
        for worker_file in args.worker:
            samples = load_jsonl(Path(worker_file))
            all_worker_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {worker_file}")

        if all_worker_samples:
            sampled = random.sample(all_worker_samples, min(
                args.count, len(all_worker_samples)))
            for sample in sampled:
                assessment = assess_worker_quality(sample)
                print_sample(sample, assessment, "worker")

    # Sample Judge datasets
    if args.judge:
        print(f"\n{'='*80}")
        print("JUDGE DATASET SAMPLES")
        print(f"{'='*80}")

        all_judge_samples = []
        for judge_file in args.judge:
            samples = load_jsonl(Path(judge_file))
            all_judge_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {judge_file}")

        if all_judge_samples:
            sampled = random.sample(all_judge_samples, min(
                args.count, len(all_judge_samples)))
            for sample in sampled:
                assessment = assess_judge_quality(sample)
                print_sample(sample, assessment, "judge")

    # Sample Drafter datasets
    if args.drafter:
        print(f"\n{'='*80}")
        print("DRAFTER DATASET SAMPLES")
        print(f"{'='*80}")

        all_drafter_samples = []
        for drafter_file in args.drafter:
            samples = load_jsonl(Path(drafter_file))
            all_drafter_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {drafter_file}")

        if all_drafter_samples:
            sampled = random.sample(all_drafter_samples, min(
                args.count, len(all_drafter_samples)))
            for sample in sampled:
                assessment = assess_drafter_quality(sample)
                print_sample(sample, assessment, "drafter")

    print(f"\n{'='*80}")
    print("SAMPLING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

