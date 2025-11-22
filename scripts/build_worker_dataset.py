"""
Build Worker dataset by combining multiple sources with explicit task type distribution.

Enforces target distributions for task types and CAWS levels, subsamples or oversamples
as needed, and generates a distribution report.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[build_worker_dataset] WARN: File not found: {file_path}")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                # Skip header lines
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(
                    f"[build_worker_dataset] WARN: Invalid JSON at line {line_num} in {file_path}: {e}")
                continue

    return samples


def ensure_sample_fields(sample: Dict[str, Any], default_role: str = "worker") -> Dict[str, Any]:
    """Ensure sample has required fields with defaults."""
    if "id" not in sample:
        sample["id"] = f"worker-{hash(str(sample)) % 1000000:06d}"
    if "role" not in sample:
        sample["role"] = default_role
    if "task_type" not in sample:
        # Infer from existing fields
        if sample.get("tool_name_ids") or sample.get("gold_json_text_ids"):
            sample["task_type"] = "tool_use"
        elif len(sample.get("prompt", "")) > 8000:
            sample["task_type"] = "long_context"
        else:
            sample["task_type"] = "plain_kd"
    if "caws_level" not in sample:
        # Infer from CAWS fields
        if sample.get("evidence_manifest") and sample.get("provenance_chain"):
            sample["caws_level"] = 2
        elif sample.get("caws_context"):
            sample["caws_level"] = 1
        else:
            sample["caws_level"] = 0
    if "source" not in sample:
        sample["source"] = "teacher_kd"

    return sample


def parse_task_mix(task_mix_str: str) -> Dict[str, float]:
    """Parse task mix string like 'plain_kd:0.35 tool_use:0.35 caws_tool:0.2 long_context:0.1'."""
    task_mix = {}
    for part in task_mix_str.split():
        if ":" in part:
            task_type, ratio_str = part.split(":", 1)
            try:
                ratio = float(ratio_str)
                task_mix[task_type] = ratio
            except ValueError:
                print(
                    f"[build_worker_dataset] WARN: Invalid ratio '{ratio_str}' for task type '{task_type}'")

    # Normalize ratios to sum to 1.0
    total = sum(task_mix.values())
    if total > 0:
        task_mix = {k: v / total for k, v in task_mix.items()}

    return task_mix


def enforce_distribution(
    samples: List[Dict[str, Any]],
    target_size: int,
    task_mix: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Enforce task type distribution by subsampling or oversampling.

    Args:
        samples: List of samples with task_type field
        target_size: Target total number of samples
        task_mix: Target distribution {task_type: ratio}

    Returns:
        List of samples with enforced distribution
    """
    # Group samples by task_type
    by_task_type = defaultdict(list)
    for sample in samples:
        task_type = sample.get("task_type", "plain_kd")
        by_task_type[task_type].append(sample)

    # Calculate target counts per task type
    target_counts = {}
    for task_type, ratio in task_mix.items():
        target_counts[task_type] = int(target_size * ratio)

    # Ensure we have enough samples
    total_available = sum(len(by_task_type.get(task_type, []))
                          for task_type in task_mix.keys())
    if total_available < target_size:
        print(
            f"[build_worker_dataset] WARN: Only {total_available} samples available, but target is {target_size}")
        # Scale down target_counts proportionally
        scale = total_available / target_size
        target_counts = {k: int(v * scale) for k, v in target_counts.items()}
        target_size = total_available

    # Sample from each task type
    result = []
    for task_type, target_count in target_counts.items():
        available = by_task_type.get(task_type, [])
        if len(available) >= target_count:
            # Subsample
            selected = random.sample(available, target_count)
        else:
            # Use all available, then oversample if needed
            selected = available.copy()
            if len(selected) < target_count:
                # Oversample by repeating
                needed = target_count - len(selected)
                selected.extend(random.choices(available, k=needed))
                print(
                    f"[build_worker_dataset] WARN: Oversampled {task_type}: {len(available)} -> {target_count}")

        result.extend(selected)

    # Shuffle final result
    random.shuffle(result)

    return result


def generate_distribution_report(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate distribution report for samples."""
    task_type_counts = Counter(s.get("task_type", "unknown") for s in samples)
    caws_level_counts = Counter(s.get("caws_level", 0) for s in samples)
    source_counts = Counter(s.get("source", "unknown") for s in samples)

    total = len(samples)

    # Calculate percentages
    task_type_pct = {k: (v / total * 100) if total >
                     0 else 0 for k, v in task_type_counts.items()}
    caws_level_pct = {k: (v / total * 100) if total >
                      0 else 0 for k, v in caws_level_counts.items()}
    source_pct = {k: (v / total * 100) if total >
                  0 else 0 for k, v in source_counts.items()}

    # Count samples with CAWS context
    caws_samples = sum(1 for s in samples if s.get("caws_level", 0) > 0)
    caws_pct = (caws_samples / total * 100) if total > 0 else 0

    # Count tool-use samples
    tool_use_samples = sum(1 for s in samples if s.get(
        "task_type") in ["tool_use", "caws_tool"])
    tool_use_pct = (tool_use_samples / total * 100) if total > 0 else 0

    # Count long-context samples
    long_context_samples = sum(
        1 for s in samples if s.get("task_type") == "long_context")
    long_context_pct = (long_context_samples / total * 100) if total > 0 else 0

    return {
        "total_samples": total,
        "task_type_distribution": {
            "counts": dict(task_type_counts),
            "percentages": task_type_pct,
        },
        "caws_level_distribution": {
            "counts": dict(caws_level_counts),
            "percentages": caws_level_pct,
        },
        "source_distribution": {
            "counts": dict(source_counts),
            "percentages": source_pct,
        },
        "caws_coverage": {
            "samples_with_caws": caws_samples,
            "percentage": caws_pct,
        },
        "tool_use_coverage": {
            "samples_with_tools": tool_use_samples,
            "percentage": tool_use_pct,
        },
        "long_context_coverage": {
            "samples_long_context": long_context_samples,
            "percentage": long_context_pct,
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Build Worker dataset with explicit task type distribution",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--kd-plain",
        help="Input JSONL file with plain KD samples",
    )
    ap.add_argument(
        "--kd-tool",
        help="Input JSONL file with tool-use KD samples",
    )
    ap.add_argument(
        "--kd-caws",
        help="Input JSONL file with CAWS KD samples",
    )
    ap.add_argument(
        "--caws-tool",
        help="Input JSONL file with CAWS tool examples",
    )
    ap.add_argument(
        "--contextual",
        help="Input JSONL file with contextual prompts",
    )
    ap.add_argument(
        "--target-size",
        type=int,
        default=2000,
        help="Target total number of samples (default: 2000)",
    )
    ap.add_argument(
        "--task-mix",
        type=str,
        default="plain_kd:0.35 tool_use:0.35 caws_tool:0.2 long_context:0.1",
        help="Target task type distribution (default: 'plain_kd:0.35 tool_use:0.35 caws_tool:0.2 long_context:0.1')",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Generate distribution report only (don't write output file)",
    )
    args = ap.parse_args()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)

    # Parse task mix
    task_mix = parse_task_mix(args.task_mix)
    print(f"[build_worker_dataset] Target task mix: {task_mix}")

    # Load all input files
    all_samples = []

    if args.kd_plain:
        samples = load_jsonl(Path(args.kd_plain))
        print(
            f"[build_worker_dataset] Loaded {len(samples)} samples from {args.kd_plain}")
        all_samples.extend(samples)

    if args.kd_tool:
        samples = load_jsonl(Path(args.kd_tool))
        print(
            f"[build_worker_dataset] Loaded {len(samples)} samples from {args.kd_tool}")
        all_samples.extend(samples)

    if args.kd_caws:
        samples = load_jsonl(Path(args.kd_caws))
        print(
            f"[build_worker_dataset] Loaded {len(samples)} samples from {args.kd_caws}")
        all_samples.extend(samples)

    if args.caws_tool:
        samples = load_jsonl(Path(args.caws_tool))
        print(
            f"[build_worker_dataset] Loaded {len(samples)} samples from {args.caws_tool}")
        all_samples.extend(samples)

    if args.contextual:
        samples = load_jsonl(Path(args.contextual))
        print(
            f"[build_worker_dataset] Loaded {len(samples)} samples from {args.contextual}")
        all_samples.extend(samples)

    print(f"[build_worker_dataset] Total samples loaded: {len(all_samples)}")

    # Ensure all samples have required fields
    for sample in all_samples:
        ensure_sample_fields(sample)

    # Enforce distribution
    final_samples = enforce_distribution(
        all_samples, args.target_size, task_mix)
    print(f"[build_worker_dataset] Final sample count: {len(final_samples)}")

    # Generate distribution report
    report = generate_distribution_report(final_samples)

    print("\n[build_worker_dataset] Distribution Report:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  Task type distribution:")
    for task_type, pct in report["task_type_distribution"]["percentages"].items():
        count = report["task_type_distribution"]["counts"][task_type]
        print(f"    {task_type}: {count} ({pct:.1f}%)")
    print(f"  CAWS level distribution:")
    for level, pct in report["caws_level_distribution"]["percentages"].items():
        count = report["caws_level_distribution"]["counts"][level]
        print(f"    Level {level}: {count} ({pct:.1f}%)")
    print(
        f"  CAWS coverage: {report['caws_coverage']['samples_with_caws']} ({report['caws_coverage']['percentage']:.1f}%)")
    print(
        f"  Tool use coverage: {report['tool_use_coverage']['samples_with_tools']} ({report['tool_use_coverage']['percentage']:.1f}%)")
    print(
        f"  Long context coverage: {report['long_context_coverage']['samples_long_context']} ({report['long_context_coverage']['percentage']:.1f}%)")

    if args.report_only:
        print("\n[build_worker_dataset] Report-only mode: not writing output file")
        return

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header with distribution report
        header = {
            "__header__": True,
            "dataset_type": "worker_combined",
            "total_samples": len(final_samples),
            "distribution_report": report,
            "task_mix": task_mix,
            "target_size": args.target_size,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Write samples
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[build_worker_dataset] Output written to: {output_path}")


if __name__ == "__main__":
    main()





