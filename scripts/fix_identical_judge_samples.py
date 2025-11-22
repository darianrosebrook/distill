"""
Fix Judge samples with identical a.text and b.text.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
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


def save_jsonl(samples: List[Dict[str, Any]], file_path: Path):
    """Save samples to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def fix_identical_samples(samples: List[Dict[str, Any]], other_samples: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Fix samples with identical a.text and b.text.

    Strategy:
    1. For samples with identical text, try to find a replacement b.text from other samples
    2. If no replacement found, modify b.text slightly to create distinction
    3. Update winner to "tie" if texts are truly identical
    """
    fixed_samples = []
    other_texts = []

    # Collect texts from other samples for replacement
    if other_samples:
        for sample in other_samples:
            a_text = sample.get("a", {}).get("text", "")
            b_text = sample.get("b", {}).get("text", "")
            if a_text:
                other_texts.append(a_text)
            if b_text:
                other_texts.append(b_text)

    for sample in samples:
        a_text = sample.get("a", {}).get("text", "")
        b_text = sample.get("b", {}).get("text", "")

        # Check if texts are identical
        if a_text == b_text and a_text:
            # Try to find a replacement b.text
            replacement_found = False

            if other_texts:
                # Find a text that's different but similar in length
                target_length = len(a_text)
                candidates = [
                    t for t in other_texts
                    if t != a_text and abs(len(t) - target_length) < target_length * 0.5
                ]

                if candidates:
                    new_b_text = random.choice(candidates)
                    sample["b"]["text"] = new_b_text
                    # Update winner based on clauses
                    a_clauses = sample.get("a", {}).get("clauses", [])
                    b_clauses = sample.get("b", {}).get("clauses", [])
                    if len(a_clauses) > len(b_clauses):
                        sample["winner"] = "a"
                    elif len(b_clauses) > len(a_clauses):
                        sample["winner"] = "b"
                    else:
                        sample["winner"] = "tie"
                    replacement_found = True

            if not replacement_found:
                # If no replacement, mark as tie and add note
                sample["winner"] = "tie"
                # Optionally, we could modify b.text slightly, but that might introduce noise
                # For now, just mark as tie which is acceptable

        fixed_samples.append(sample)

    return fixed_samples


def main():
    ap = argparse.ArgumentParser(
        description="Fix Judge samples with identical a.text and b.text",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input JSONL file to fix",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    ap.add_argument(
        "--other-sources",
        nargs="+",
        help="Other JSONL files to use for finding replacement texts",
    )
    ap.add_argument(
        "--remove-identical",
        action="store_true",
        help="Remove identical samples instead of fixing them",
    )
    args = ap.parse_args()

    # Load input samples
    print(f"[fix_identical_judge_samples] Loading samples from {args.input}")
    samples = load_jsonl(Path(args.input))
    print(f"[fix_identical_judge_samples] Loaded {len(samples)} samples")

    # Count identical samples
    identical_count = 0
    for sample in samples:
        a_text = sample.get("a", {}).get("text", "")
        b_text = sample.get("b", {}).get("text", "")
        if a_text == b_text and a_text:
            identical_count += 1

    print(
        f"[fix_identical_judge_samples] Found {identical_count} samples with identical text")

    if args.remove_identical:
        # Remove identical samples
        fixed_samples = [
            s for s in samples
            if not (s.get("a", {}).get("text", "") == s.get("b", {}).get("text", "") and s.get("a", {}).get("text", ""))
        ]
        print(
            f"[fix_identical_judge_samples] Removed {len(samples) - len(fixed_samples)} identical samples")
    else:
        # Load other sources for replacement
        other_samples = []
        if args.other_sources:
            for source_file in args.other_sources:
                other_samples.extend(load_jsonl(Path(source_file)))
            print(
                f"[fix_identical_judge_samples] Loaded {len(other_samples)} samples from other sources")

        # Fix identical samples
        fixed_samples = fix_identical_samples(samples, other_samples)
        print(
            f"[fix_identical_judge_samples] Fixed {identical_count} identical samples")

    # Save fixed samples
    save_jsonl(fixed_samples, Path(args.output))
    print(
        f"[fix_identical_judge_samples] Saved {len(fixed_samples)} samples to {args.output}")

    # Verify fix
    remaining_identical = 0
    for sample in fixed_samples:
        a_text = sample.get("a", {}).get("text", "")
        b_text = sample.get("b", {}).get("text", "")
        if a_text == b_text and a_text:
            remaining_identical += 1

    if remaining_identical > 0:
        print(
            f"[fix_identical_judge_samples] WARN: {remaining_identical} identical samples remain")
    else:
        print(f"[fix_identical_judge_samples] ✅ All identical samples fixed!")


if __name__ == "__main__":
    main()

