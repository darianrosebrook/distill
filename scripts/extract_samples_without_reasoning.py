"""
Extract samples without teacher_reasoning from KD dataset.

This script extracts samples that are missing the teacher_reasoning field,
saving them to a separate file for backfilling.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def extract_samples_without_reasoning(
    input_file: Path,
    output_file: Path,
) -> int:
    """
    Extract samples that are missing teacher_reasoning field.

    Returns:
        Number of samples extracted.
    """
    samples_without_reasoning: List[Dict[str, Any]] = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON at line {line_num}: {e}")
                continue

            # Check if teacher_reasoning is missing or None
            if "teacher_reasoning" not in sample or sample.get("teacher_reasoning") is None:
                samples_without_reasoning.append(sample)

    # Write extracted samples to output file
    if samples_without_reasoning:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples_without_reasoning:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(samples_without_reasoning)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract samples without teacher_reasoning from KD dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input KD dataset JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for samples without reasoning",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file does not exist: {args.input}")
        return 1

    print(f"Extracting samples without teacher_reasoning from {args.input}...")
    count = extract_samples_without_reasoning(args.input, args.output)

    print(f"Extracted {count} samples without teacher_reasoning to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

