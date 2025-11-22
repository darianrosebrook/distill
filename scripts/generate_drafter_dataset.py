"""
Generate Drafter dataset as thin specialization on Worker data.

Filters Worker datasets for short responses (≤2k tokens), adds length bucket labels,
and creates draft segments from Worker examples. Reuses Worker ontology.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[generate_drafter_dataset] ERROR: File not found: {file_path}")
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
                    f"[generate_drafter_dataset] WARN: Invalid JSON at line {line_num}: {e}")
                continue

    return samples


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from character count."""
    return len(text) // chars_per_token


def get_length_bucket(token_count: int) -> int:
    """Get length bucket (200, 400, or 800 tokens)."""
    if token_count <= 200:
        return 200
    elif token_count <= 400:
        return 400
    elif token_count <= 800:
        return 800
    else:
        return 800  # Cap at 800 for drafter


def create_draft_segment(text: str, max_tokens: int = 400) -> str:
    """
    Create draft segment by taking early portion of text.

    Args:
        text: Full text
        max_tokens: Maximum tokens for draft segment

    Returns:
        Draft segment (early portion of text)
    """
    # Estimate tokens
    estimated_tokens = estimate_tokens(text)

    if estimated_tokens <= max_tokens:
        return text

    # Take first portion
    chars_to_take = max_tokens * 4  # Rough estimate: 4 chars per token
    draft = text[:chars_to_take]

    # Try to end at a sentence boundary
    last_period = draft.rfind(".")
    last_newline = draft.rfind("\n")
    if last_period > chars_to_take * 0.8:  # If period is in last 20%
        draft = draft[:last_period + 1]
    elif last_newline > chars_to_take * 0.8:
        draft = draft[:last_newline]

    return draft


def filter_and_process_samples(
    samples: List[Dict[str, Any]],
    max_length: int = 2048,
    create_drafts: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter samples for drafter requirements and add drafter-specific fields.

    Args:
        samples: List of Worker samples
        max_length: Maximum token length for drafter samples
        create_drafts: If True, create draft segments from full responses

    Returns:
        Filtered and processed samples
    """
    drafter_samples = []

    for sample in samples:
        teacher_text = sample.get("teacher_text", "")
        if not teacher_text:
            continue

        # Estimate token count for filtering and length bucket assignment
        token_count = estimate_tokens(teacher_text)
        
        # Filter by token length (Drafter should be short for fast generation)
        if token_count > max_length:
            continue

        # Determine task_type (prefer plain_kd and tool_use, avoid long_context)
        task_type = sample.get("task_type", "plain_kd")
        if task_type == "long_context":
            # Skip long-context samples for drafter
            continue

        # Create drafter sample
        drafter_sample = {
            "id": sample.get("id", f"drafter-{hash(str(sample)) % 1000000:06d}"),
            "role": "drafter",
            "task_type": task_type,
            "caws_level": sample.get("caws_level", 0),
            "source": sample.get("source", "teacher_kd"),
            "prompt": sample.get("prompt", ""),
            "teacher_text": teacher_text,
        }

        # Add length bucket
        length_bucket = get_length_bucket(token_count)
        drafter_sample["supervision"] = {
            "target_length_bucket": length_bucket,
        }

        # Create draft segment if requested
        if create_drafts:
            draft_segment = create_draft_segment(
                teacher_text, max_tokens=length_bucket)
            drafter_sample["draft_segment"] = draft_segment

        # Copy process-step supervision if present
        if "tool_name_ids" in sample:
            drafter_sample["tool_name_ids"] = sample["tool_name_ids"]
        if "gold_json_text_ids" in sample:
            drafter_sample["gold_json_text_ids"] = sample["gold_json_text_ids"]
        if "integration_mask" in sample:
            drafter_sample["integration_mask"] = sample["integration_mask"]

        # Copy CAWS context if present
        if sample.get("caws_context"):
            drafter_sample["caws_context"] = sample["caws_context"]
        if sample.get("evidence_manifest"):
            drafter_sample["evidence_manifest"] = sample["evidence_manifest"]
        if sample.get("provenance_chain"):
            drafter_sample["provenance_chain"] = sample["provenance_chain"]

        # Copy metadata
        if sample.get("metadata"):
            drafter_sample["metadata"] = sample["metadata"]

        drafter_samples.append(drafter_sample)

    return drafter_samples


def generate_distribution_report(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate distribution report for drafter samples."""
    length_bucket_counts = Counter(
        s.get("supervision", {}).get("target_length_bucket", 0) for s in samples
    )
    task_type_counts = Counter(s.get("task_type", "unknown") for s in samples)
    caws_level_counts = Counter(s.get("caws_level", 0) for s in samples)

    total = len(samples)

    return {
        "total_samples": total,
        "length_bucket_distribution": {
            "counts": dict(length_bucket_counts),
            "percentages": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in length_bucket_counts.items()
            },
        },
        "task_type_distribution": {
            "counts": dict(task_type_counts),
            "percentages": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in task_type_counts.items()
            },
        },
        "caws_level_distribution": {
            "counts": dict(caws_level_counts),
            "percentages": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in caws_level_counts.items()
            },
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate Drafter dataset as thin specialization on Worker data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Input Worker dataset JSONL file")
    ap.add_argument("--out", required=True,
                    help="Output Drafter dataset JSONL file")
    ap.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length for drafter samples (default: 2048)",
    )
    ap.add_argument(
        "--length-buckets",
        type=str,
        default="200,400,800",
        help="Comma-separated list of length buckets (default: '200,400,800')",
    )
    ap.add_argument(
        "--create-drafts",
        action="store_true",
        default=True,
        help="Create draft segments from full responses (default: True)",
    )
    ap.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of samples to generate (default: use all filtered samples)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = ap.parse_args()

    # Set seed
    if args.seed is not None:
        random.seed(args.seed)

    # Load input samples
    print(f"[generate_drafter_dataset] Loading samples from {args.input}")
    samples = load_jsonl(Path(args.input))
    print(f"[generate_drafter_dataset] Loaded {len(samples)} samples")

    # Filter and process
    print(
        f"[generate_drafter_dataset] Filtering samples (max_length={args.max_length})...")
    drafter_samples = filter_and_process_samples(
        samples,
        max_length=args.max_length,
        create_drafts=args.create_drafts,
    )
    print(
        f"[generate_drafter_dataset] Filtered to {len(drafter_samples)} samples")

    # Limit total if requested
    if args.total and len(drafter_samples) > args.total:
        drafter_samples = random.sample(drafter_samples, args.total)
        print(f"[generate_drafter_dataset] Limited to {args.total} samples")

    # Generate distribution report
    report = generate_distribution_report(drafter_samples)

    print("\n[generate_drafter_dataset] Distribution Report:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  Length bucket distribution:")
    for bucket, pct in report["length_bucket_distribution"]["percentages"].items():
        count = report["length_bucket_distribution"]["counts"][bucket]
        print(f"    {bucket} tokens: {count} ({pct:.1f}%)")
    print(f"  Task type distribution:")
    for task_type, pct in report["task_type_distribution"]["percentages"].items():
        count = report["task_type_distribution"]["counts"][task_type]
        print(f"    {task_type}: {count} ({pct:.1f}%)")

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        header = {
            "__header__": True,
            "dataset_type": "drafter",
            "total_samples": len(drafter_samples),
            "distribution_report": report,
            "max_length": args.max_length,
            "length_buckets": [int(b) for b in args.length_buckets.split(",")],
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Write samples
        for sample in drafter_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[generate_drafter_dataset] Output written to: {output_path}")


if __name__ == "__main__":
    main()
