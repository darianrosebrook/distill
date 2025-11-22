"""
Merge backfilled reasoning samples back into the main dataset.

Replaces samples in the main dataset that match IDs from the backfilled dataset,
ensuring all samples have teacher_reasoning content.

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_samples_by_id(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load samples from JSONL file, indexed by ID."""
    samples_by_id = {}
    if not file_path.exists():
        print(f"[merge_backfilled_reasoning] WARN: File not found: {file_path}")
        return samples_by_id

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                sample_id = sample.get("id")
                if sample_id:
                    samples_by_id[sample_id] = sample
            except json.JSONDecodeError as e:
                print(f"[merge_backfilled_reasoning] WARN: Invalid JSON at line {line_num} in {file_path}: {e}")
                continue

    return samples_by_id


def merge_backfilled_reasoning(
    main_dataset: Path,
    backfilled_dataset: Path,
    output_file: Path,
) -> None:
    """Merge backfilled reasoning samples back into main dataset."""
    print(f"[merge_backfilled_reasoning] Loading main dataset: {main_dataset}")
    main_samples_by_id = load_samples_by_id(main_dataset)
    print(f"[merge_backfilled_reasoning] Loaded {len(main_samples_by_id)} samples from main dataset")

    print(f"[merge_backfilled_reasoning] Loading backfilled dataset: {backfilled_dataset}")
    backfilled_samples_by_id = load_samples_by_id(backfilled_dataset)
    print(f"[merge_backfilled_reasoning] Loaded {len(backfilled_samples_by_id)} samples from backfilled dataset")

    # Replace samples in main dataset with backfilled versions
    replaced_count = 0
    for sample_id, backfilled_sample in backfilled_samples_by_id.items():
        if sample_id in main_samples_by_id:
            main_samples_by_id[sample_id] = backfilled_sample
            replaced_count += 1
        else:
            print(f"[merge_backfilled_reasoning] WARN: Sample {sample_id} in backfilled dataset not found in main dataset")

    print(f"[merge_backfilled_reasoning] Replaced {replaced_count} samples with backfilled versions")

    # Write merged dataset
    print(f"[merge_backfilled_reasoning] Writing merged dataset to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        # Write samples in original order (if we can preserve it) or sorted by ID
        for sample_id in sorted(main_samples_by_id.keys()):
            sample = main_samples_by_id[sample_id]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Verify reasoning coverage
    samples_with_reasoning = sum(
        1 for s in main_samples_by_id.values()
        if s.get("teacher_reasoning") and s["teacher_reasoning"]
    )
    print(f"[merge_backfilled_reasoning] Verification:")
    print(f"  Total samples: {len(main_samples_by_id)}")
    print(f"  Samples with reasoning: {samples_with_reasoning}")
    print(f"  Samples without reasoning: {len(main_samples_by_id) - samples_with_reasoning}")
    print(f"[merge_backfilled_reasoning] Merge complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge backfilled reasoning samples back into main dataset"
    )
    parser.add_argument(
        "--main-dataset",
        type=Path,
        required=True,
        help="Path to main dataset JSONL file",
    )
    parser.add_argument(
        "--backfilled-dataset",
        type=Path,
        required=True,
        help="Path to backfilled dataset JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output merged dataset JSONL file",
    )
    args = parser.parse_args()

    merge_backfilled_reasoning(
        args.main_dataset,
        args.backfilled_dataset,
        args.output,
    )


if __name__ == "__main__":
    main()


