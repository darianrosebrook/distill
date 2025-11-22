"""
Build pairwise Judge examples from real Worker outputs.

Takes Worker outputs and creates pairwise comparisons for Judge training.
This generates realistic pairs from actual model behavior.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from arbiter.judge_training.generate_from_worker_outputs import infer_clauses_from_output


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[build_pairs_from_worker] WARN: File not found: {file_path}")
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
                print(f"[build_pairs_from_worker] WARN: Invalid JSON: {e}")
                continue

    return samples


def create_pair_from_worker_outputs(
    a_output: Dict[str, Any],
    b_output: Dict[str, Any],
    working_spec: Optional[Dict[str, Any]] = None,
    pair_id: str = None,
) -> Optional[Dict[str, Any]]:
    """Create a pairwise sample from two Worker outputs."""
    a_text = a_output.get("teacher_text") or a_output.get("text", "")
    b_text = b_output.get("teacher_text") or b_output.get("text", "")

    # Skip if texts are identical
    if a_text == b_text:
        return None

    # Skip if one is substring of the other (>90% similarity)
    if a_text in b_text or b_text in a_text:
        if abs(len(a_text) - len(b_text)) < max(len(a_text), len(b_text)) * 0.1:
            return None

    # Determine winner (prefer longer, more complete responses)
    # In practice, this would be determined by actual CAWS evaluation
    if len(a_text) > len(b_text) * 1.2:
        winner = "a"
    elif len(b_text) > len(a_text) * 1.2:
        winner = "b"
    else:
        # Check CAWS compliance
        a_clauses = infer_clauses_from_output(a_output)
        b_clauses = infer_clauses_from_output(b_output)
        if len(a_clauses) > len(b_clauses):
            winner = "a"
        elif len(b_clauses) > len(a_clauses):
            winner = "b"
        else:
            winner = "a"  # Default to a

    # Infer clauses
    a_clauses = infer_clauses_from_output(a_output)
    b_clauses = infer_clauses_from_output(b_output)

    # Generate reasons
    if winner == "a":
        reasons = f"Solution A is preferred because it satisfies: {', '.join(a_clauses)}"
        if b_clauses:
            reasons += f", while Solution B only satisfies: {', '.join(b_clauses)}"
    else:
        reasons = f"Solution B is preferred because it satisfies: {', '.join(b_clauses)}"
        if a_clauses:
            reasons += f", while Solution A only satisfies: {', '.join(a_clauses)}"

    # Create pair
    pair = {
        "id": pair_id or f"worker-pair-{hash(str(a_output.get('id', '')) + str(b_output.get('id', ''))) % 1000000:06d}",
        "prompt": a_output.get("prompt", b_output.get("prompt", "")),
        "working_spec": working_spec or a_output.get("caws_context", {}).get("working_spec", {}),
        "a": {
            "text": a_text,
            "clauses": a_clauses,
        },
        "b": {
            "text": b_text,
            "clauses": b_clauses,
        },
        "winner": winner,
        "reasons": reasons,
    }

    # Copy optional fields
    if a_output.get("change_diff"):
        pair["a"]["change_diff"] = a_output["change_diff"]
    if a_output.get("rationale"):
        pair["a"]["rationale"] = a_output["rationale"]
    if a_output.get("evidence_manifest"):
        pair["a"]["evidence_manifest"] = a_output["evidence_manifest"]
    if a_output.get("provenance_chain"):
        pair["a"]["provenance_chain"] = a_output["provenance_chain"]

    if b_output.get("change_diff"):
        pair["b"]["change_diff"] = b_output["change_diff"]
    if b_output.get("rationale"):
        pair["b"]["rationale"] = b_output["rationale"]
    if b_output.get("evidence_manifest"):
        pair["b"]["evidence_manifest"] = b_output["evidence_manifest"]
    if b_output.get("provenance_chain"):
        pair["b"]["provenance_chain"] = b_output["provenance_chain"]

    return pair


def build_pairs(
    worker_outputs: List[Dict[str, Any]],
    total_pairs: int,
    working_spec: Optional[Dict[str, Any]] = None,
    max_attempts: int = 100,
) -> List[Dict[str, Any]]:
    """Build pairwise samples from Worker outputs."""
    pairs = []
    seen_pairs = set()

    # Filter valid outputs (must have text)
    valid_outputs = [o for o in worker_outputs if o.get("teacher_text") or o.get("text")]

    if len(valid_outputs) < 2:
        print(f"[build_pairs_from_worker] WARN: Need at least 2 valid outputs, got {len(valid_outputs)}")
        return pairs

    attempts = 0
    while len(pairs) < total_pairs and attempts < max_attempts * total_pairs:
        attempts += 1

        # Sample two different outputs
        a, b = random.sample(valid_outputs, 2)

        # Create pair
        pair = create_pair_from_worker_outputs(a, b, working_spec)
        if not pair:
            continue

        # Check for duplicates
        pair_key = (pair["a"]["text"][:100], pair["b"]["text"][:100])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        pairs.append(pair)

    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Build pairwise Judge examples from Worker outputs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input Worker dataset JSONL file")
    ap.add_argument("--out", required=True, help="Output Judge pairs JSONL file")
    ap.add_argument(
        "--total",
        type=int,
        default=2000,
        help="Total number of pairs to generate (default: 2000)",
    )
    ap.add_argument(
        "--caws-spec-id",
        help="CAWS spec ID to use for working spec",
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

    # Load Worker outputs
    worker_outputs = load_jsonl(Path(args.input))
    print(f"[build_pairs_from_worker] Loaded {len(worker_outputs)} Worker outputs")

    # Extract working spec if available
    working_spec = None
    if args.caws_spec_id:
        try:
            from training.caws_context import extract_caws_context_dict
            working_spec_dict = extract_caws_context_dict(".", spec_id=args.caws_spec_id)
            working_spec = {
                "id": working_spec_dict.get("spec_id", "unknown"),
                "title": working_spec_dict.get("title", "Unknown"),
                "risk_tier": working_spec_dict.get("risk_tier", 2),
                "budget": working_spec_dict.get("budget", {}),
                "scope": working_spec_dict.get("scope", {}),
            }
        except Exception as e:
            print(f"[build_pairs_from_worker] WARN: Failed to extract CAWS context: {e}")

    # Build pairs
    pairs = build_pairs(worker_outputs, args.total, working_spec)
    print(f"[build_pairs_from_worker] Generated {len(pairs)} pairs")

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[build_pairs_from_worker] Wrote pairs to {output_path}")


if __name__ == "__main__":
    main()


