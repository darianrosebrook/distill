"""
Combine multiple Judge datasets and split into train/validation sets.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[combine_and_split] WARN: File not found: {file_path}")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[combine_and_split] WARN: Invalid JSON: {e}")
                continue

    return samples


def combine_and_split(
    input_files: List[Path],
    train_output: Path,
    val_output: Path,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> None:
    """Combine datasets and split into train/val."""
    if seed is not None:
        random.seed(seed)

    # Load all samples
    all_samples = []
    for input_file in input_files:
        samples = load_jsonl(input_file)
        print(
            f"[combine_and_split] Loaded {len(samples)} samples from {input_file.name}")
        all_samples.extend(samples)

    print(f"[combine_and_split] Total samples: {len(all_samples)}")

    # Shuffle
    random.shuffle(all_samples)

    # Split
    val_size = int(len(all_samples) * val_ratio)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(
        f"[combine_and_split] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Write train
    train_output.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write val
    val_output.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[combine_and_split] Wrote train to {train_output}")
    print(f"[combine_and_split] Wrote val to {val_output}")


def main():
    ap = argparse.ArgumentParser(
        description="Combine Judge datasets and split into train/val",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Input JSONL files to combine")
    ap.add_argument("--train-out", required=True,
                    help="Output train JSONL file")
    ap.add_argument("--val-out", required=True, help="Output val JSONL file")
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = ap.parse_args()

    input_files = [Path(f) for f in args.inputs]
    train_output = Path(args.train_out)
    val_output = Path(args.val_out)

    combine_and_split(
        input_files=input_files,
        train_output=train_output,
        val_output=val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
