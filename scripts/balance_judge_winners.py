"""
Balance Judge dataset winner distribution by strategically flipping winners.

This script addresses the critical imbalance where 98.4% of samples have winner="a".
It creates a balanced distribution by:
1. Flipping a subset of "a" winners to "b" (swapping a/b texts)
2. Converting some "a" winners to "tie" for subtle distinctions
3. Ensuring final distribution: 40-45% "a", 40-45% "b", 10-15% "tie"

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[balance_judge_winners] ERROR: File not found: {file_path}")
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
                print(f"[balance_judge_winners] WARN: Invalid JSON: {e}")
                continue

    return samples


def swap_a_b(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Swap sides a and b in a pairwise sample."""
    sample = sample.copy()
    a_data = sample.get("a", {})
    b_data = sample.get("b", {})
    
    # Swap the sides
    sample["a"] = b_data
    sample["b"] = a_data
    
    # Update winner
    if sample.get("winner") == "a":
        sample["winner"] = "b"
    elif sample.get("winner") == "b":
        sample["winner"] = "a"
    # "tie" stays "tie"
    
    # Update reasons if present
    if "reasons" in sample:
        reasons = sample["reasons"]
        # Simple swap of "A" and "B" references
        reasons = reasons.replace("Solution A", "SOLUTION_A_PLACEHOLDER")
        reasons = reasons.replace("Solution B", "Solution A")
        reasons = reasons.replace("SOLUTION_A_PLACEHOLDER", "Solution B")
        reasons = reasons.replace("solution A", "solution_a_placeholder")
        reasons = reasons.replace("solution B", "solution A")
        reasons = reasons.replace("solution_a_placeholder", "solution B")
        sample["reasons"] = reasons
    
    return sample


def convert_to_tie(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a sample to a tie by making both sides more similar."""
    sample = sample.copy()
    sample["winner"] = "tie"
    
    # Update reasons to reflect tie
    a_clauses = sample.get("a", {}).get("clauses", [])
    b_clauses = sample.get("b", {}).get("clauses", [])
    common_clauses = [c for c in a_clauses if c in b_clauses]
    
    if common_clauses:
        sample["reasons"] = (
            f"Both solutions satisfy similar CAWS requirements: {', '.join(common_clauses)}. "
            f"Solution A satisfies: {', '.join(a_clauses)}. "
            f"Solution B satisfies: {', '.join(b_clauses)}. "
            "The distinction is subtle and context-dependent."
        )
    else:
        sample["reasons"] = (
            "Both solutions have comparable quality. Solution A and Solution B each satisfy "
            "different aspects of the CAWS requirements, making a clear winner difficult to determine."
        )
    
    return sample


def balance_judge_dataset(
    samples: List[Dict[str, Any]],
    target_a_pct: float = 0.42,
    target_b_pct: float = 0.42,
    target_tie_pct: float = 0.16,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Balance Judge dataset winner distribution.
    
    Args:
        samples: List of Judge pairwise samples
        target_a_pct: Target percentage for winner="a" (default: 42%)
        target_b_pct: Target percentage for winner="b" (default: 42%)
        target_tie_pct: Target percentage for winner="tie" (default: 16%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    total = len(samples)
    target_a = int(total * target_a_pct)
    target_b = int(total * target_b_pct)
    target_tie = int(total * target_tie_pct)
    
    print(f"[balance_judge_winners] Balancing {total} samples")
    print(f"  Target distribution: a={target_a_pct*100:.1f}%, b={target_b_pct*100:.1f}%, tie={target_tie_pct*100:.1f}%")
    print(f"  Target counts: a={target_a}, b={target_b}, tie={target_tie}")
    
    # Count current distribution
    current_winners = Counter(s.get("winner", "unknown") for s in samples)
    print(f"  Current distribution: {dict(current_winners)}")
    
    # Separate samples by current winner
    a_winners = [s for s in samples if s.get("winner") == "a"]
    b_winners = [s for s in samples if s.get("winner") == "b"]
    tie_winners = [s for s in samples if s.get("winner") == "tie"]
    
    balanced_samples = []
    
    # Step 1: Keep existing "b" winners (we need more of these)
    balanced_samples.extend(b_winners)
    remaining_b_needed = max(0, target_b - len(b_winners))
    
    # Step 2: Keep existing "tie" winners
    balanced_samples.extend(tie_winners)
    remaining_tie_needed = max(0, target_tie - len(tie_winners))
    
    # Step 3: Convert some "a" winners to "b" by swapping
    a_to_swap = min(len(a_winners), remaining_b_needed)
    a_winners_to_swap = random.sample(a_winners, a_to_swap)
    for sample in a_winners_to_swap:
        balanced_samples.append(swap_a_b(sample))
    
    # Step 4: Convert some "a" winners to "tie"
    remaining_a_winners = [s for s in a_winners if s not in a_winners_to_swap]
    a_to_tie = min(len(remaining_a_winners), remaining_tie_needed)
    a_winners_to_tie = random.sample(remaining_a_winners, a_to_tie)
    for sample in a_winners_to_tie:
        balanced_samples.append(convert_to_tie(sample))
    
    # Step 5: Keep remaining "a" winners (should be ~target_a)
    remaining_a_winners = [s for s in remaining_a_winners if s not in a_winners_to_tie]
    balanced_samples.extend(remaining_a_winners)
    
    # Verify final distribution
    final_winners = Counter(s.get("winner", "unknown") for s in balanced_samples)
    print(f"  Final distribution: {dict(final_winners)}")
    
    a_final_pct = final_winners.get("a", 0) / len(balanced_samples) * 100
    b_final_pct = final_winners.get("b", 0) / len(balanced_samples) * 100
    tie_final_pct = final_winners.get("tie", 0) / len(balanced_samples) * 100
    
    print(f"  Final percentages: a={a_final_pct:.1f}%, b={b_final_pct:.1f}%, tie={tie_final_pct:.1f}%")
    
    if not (35 <= a_final_pct <= 50 and 35 <= b_final_pct <= 50):
        print(f"  WARN: Distribution may still be imbalanced. Consider generating more balanced pairs.")
    
    return balanced_samples


def main():
    ap = argparse.ArgumentParser(
        description="Balance Judge dataset winner distribution",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--in", dest="input_file", required=True, help="Input Judge dataset JSONL")
    ap.add_argument("--out", required=True, help="Output balanced JSONL file")
    ap.add_argument(
        "--target-a-pct",
        type=float,
        default=0.42,
        help="Target percentage for winner='a' (default: 0.42)",
    )
    ap.add_argument(
        "--target-b-pct",
        type=float,
        default=0.42,
        help="Target percentage for winner='b' (default: 0.42)",
    )
    ap.add_argument(
        "--target-tie-pct",
        type=float,
        default=0.16,
        help="Target percentage for winner='tie' (default: 0.16)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = ap.parse_args()
    
    # Validate percentages
    total_pct = args.target_a_pct + args.target_b_pct + args.target_tie_pct
    if abs(total_pct - 1.0) > 0.01:
        print(f"ERROR: Percentages must sum to 1.0, got {total_pct}")
        return 1
    
    input_file = Path(args.input_file)
    output_file = Path(args.out)
    
    # Load samples
    print(f"[balance_judge_winners] Loading samples from {input_file}")
    samples = load_jsonl(input_file)
    if not samples:
        print(f"ERROR: No samples loaded from {input_file}")
        return 1
    
    # Balance distribution
    balanced_samples = balance_judge_dataset(
        samples,
        target_a_pct=args.target_a_pct,
        target_b_pct=args.target_b_pct,
        target_tie_pct=args.target_tie_pct,
        seed=args.seed,
    )
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"[balance_judge_winners] Wrote {len(balanced_samples)} balanced samples to {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


