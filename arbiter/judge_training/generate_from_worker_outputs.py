"""
Generate pairwise Judge examples from real Worker outputs.

Compares Worker vs Worker or Worker vs Teacher outputs to create
realistic pairwise training examples for Judge.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Actual CAWS clause names (used in training)
CAWS_CLAUSES = [
    "EVIDENCE_COMPLETENESS",
    "BUDGET_ADHERENCE",
    "GATE_INTEGRITY",
    "PROVENANCE_CLARITY",
    "WAIVER_JUSTIFICATION",
]


def infer_clauses_from_output(output: Dict[str, Any]) -> List[str]:
    """Infer CAWS clauses from Worker output metadata."""
    clauses = []

    # Check CAWS level
    caws_level = output.get("caws_level", 0)
    if caws_level >= 1:
        clauses.append("EVIDENCE_COMPLETENESS")
    if caws_level >= 2:
        clauses.append("PROVENANCE_CLARITY")

    # Check for evidence manifest
    if output.get("evidence_manifest"):
        clauses.append("EVIDENCE_COMPLETENESS")

    # Check for provenance chain
    if output.get("provenance_chain"):
        clauses.append("PROVENANCE_CLARITY")

    # Check task type for gate integrity
    task_type = output.get("task_type", "")
    if task_type in ["tool_use", "caws_tool"]:
        clauses.append("GATE_INTEGRITY")

    # Default clause if none inferred
    if not clauses:
        clauses = ["EVIDENCE_COMPLETENESS"]

    # Remove duplicates while preserving order
    return list(dict.fromkeys(clauses))


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(
            f"[generate_from_worker_outputs] ERROR: File not found: {file_path}")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(
                    f"[generate_from_worker_outputs] WARN: Invalid JSON at line {line_num}: {e}")
                continue

    return samples


def create_pair_from_worker_outputs(
    worker_outputs: List[Dict[str, Any]],
    teacher_outputs: Optional[List[Dict[str, Any]]] = None,
    working_spec: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Create pairwise examples from Worker outputs.

    Args:
        worker_outputs: List of Worker output samples
        teacher_outputs: Optional list of Teacher/Oracle outputs for comparison
        working_spec: Optional working spec for context

    Returns:
        List of pairwise examples
    """
    pairs = []

    # Strategy 1: Worker vs Worker (compare different Worker outputs for same prompt)
    prompt_to_outputs = {}
    for output in worker_outputs:
        prompt = output.get("prompt", "")
        if prompt:
            if prompt not in prompt_to_outputs:
                prompt_to_outputs[prompt] = []
            prompt_to_outputs[prompt].append(output)

    # Create pairs from same-prompt outputs
    for prompt, outputs in prompt_to_outputs.items():
        if len(outputs) >= 2:
            # Filter outputs with valid text
            valid_outputs = [
                o for o in outputs
                if (o.get("teacher_text") or o.get("text")) and len((o.get("teacher_text") or o.get("text", ""))) > 10
            ]

            if len(valid_outputs) < 2:
                continue

            # Try multiple times to find distinct pairs
            max_attempts = min(10, len(valid_outputs) * 2)
            pair_found = False

            for attempt in range(max_attempts):
                # Randomly select two outputs
                a, b = random.sample(valid_outputs, 2)

                # Get text from teacher_text or text field
                a_text = a.get("teacher_text") or a.get("text", "")
                b_text = b.get("teacher_text") or b.get("text", "")

                # Skip if texts are identical (or very similar)
                if a_text == b_text:
                    continue

                # Skip if texts are too similar (more than 90% overlap)
                if len(a_text) > 0 and len(b_text) > 0:
                    # Simple similarity check: if one is substring of other
                    if a_text in b_text or b_text in a_text:
                        if abs(len(a_text) - len(b_text)) < max(len(a_text), len(b_text)) * 0.1:
                            continue

                # Simple heuristic: prefer output with more complete CAWS fields
                a_caws_level = a.get("caws_level", 0)
                b_caws_level = b.get("caws_level", 0)

                # Infer clauses for both outputs
                a_clauses = infer_clauses_from_output(a)
                b_clauses = infer_clauses_from_output(b)

                # Determine winner
                if a_caws_level > b_caws_level:
                    winner = "a"
                elif b_caws_level > a_caws_level:
                    winner = "b"
                else:
                    # Use text length as tiebreaker (longer = more complete)
                    if len(a_text) > len(b_text) * 1.2:
                        winner = "a"
                    elif len(b_text) > len(a_text) * 1.2:
                        winner = "b"
                    else:
                        # Tie or use quality score if available
                        a_score = a.get("teacher_quality_score", 0.5)
                        b_score = b.get("teacher_quality_score", 0.5)
                        if a_score > b_score:
                            winner = "a"
                        elif b_score > a_score:
                            winner = "b"
                        else:
                            winner = "tie"

                pair = {
                    "id": f"worker-pair-{len(pairs)+1:06d}",
                    "prompt": prompt,
                    "working_spec": working_spec,
                    "a": {
                        "text": a_text,
                        "clauses": a_clauses,
                    },
                    "b": {
                        "text": b_text,
                        "clauses": b_clauses,
                    },
                    "winner": winner,
                }

            # Add optional fields if present
            if a.get("evidence_manifest"):
                pair["a"]["evidence_manifest"] = a.get("evidence_manifest")
            if a.get("provenance_chain"):
                pair["a"]["provenance_chain"] = a.get("provenance_chain")
            if b.get("evidence_manifest"):
                pair["b"]["evidence_manifest"] = b.get("evidence_manifest")
            if b.get("provenance_chain"):
                pair["b"]["provenance_chain"] = b.get("provenance_chain")

            pairs.append(pair)

    # Strategy 2: Worker vs Teacher (if teacher outputs available)
    if teacher_outputs:
        # Match by prompt
        teacher_by_prompt = {t.get("prompt", ""): t for t in teacher_outputs}

        # Limit to avoid too many pairs
        for worker_output in worker_outputs[:len(pairs)]:
            prompt = worker_output.get("prompt", "")
            if prompt in teacher_by_prompt:
                teacher_output = teacher_by_prompt[prompt]

                # Infer clauses
                teacher_clauses = infer_clauses_from_output(teacher_output)
                worker_clauses = infer_clauses_from_output(worker_output)

                # Teacher is usually better - add more clauses
                if len(teacher_clauses) < 3:
                    teacher_clauses.extend(
                        ["EVIDENCE_COMPLETENESS", "GATE_INTEGRITY"])
                    teacher_clauses = list(dict.fromkeys(
                        teacher_clauses))  # Remove duplicates

                # Get text from teacher_text or text field, with fallback
                teacher_text = teacher_output.get(
                    "teacher_text") or teacher_output.get("text", "")
                worker_text = worker_output.get(
                    "teacher_text") or worker_output.get("text", "")

                # Skip pairs with empty text
                if not teacher_text or not worker_text:
                    continue

                # Teacher is usually better
                pair = {
                    "id": f"worker-teacher-pair-{len(pairs)+1:06d}",
                    "prompt": prompt,
                    "working_spec": working_spec,
                    "a": {
                        "text": teacher_text,
                        "clauses": teacher_clauses,
                    },
                    "b": {
                        "text": worker_text,
                        "clauses": worker_clauses,
                    },
                    "winner": "a",  # Teacher is usually better
                }

                # Add optional fields if present
                if teacher_output.get("evidence_manifest"):
                    pair["a"]["evidence_manifest"] = teacher_output.get(
                        "evidence_manifest")
                if teacher_output.get("provenance_chain"):
                    pair["a"]["provenance_chain"] = teacher_output.get(
                        "provenance_chain")
                if worker_output.get("evidence_manifest"):
                    pair["b"]["evidence_manifest"] = worker_output.get(
                        "evidence_manifest")
                if worker_output.get("provenance_chain"):
                    pair["b"]["provenance_chain"] = worker_output.get(
                        "provenance_chain")

                pairs.append(pair)

    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Generate pairwise Judge examples from real Worker outputs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--worker-outputs",
        required=True,
        help="Input JSONL file with Worker outputs",
    )
    ap.add_argument(
        "--teacher-outputs",
        help="Optional input JSONL file with Teacher/Oracle outputs for comparison",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL file path",
    )
    ap.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of pairs to generate (default: use all possible pairs)",
    )
    ap.add_argument(
        "--caws-spec-id",
        help="CAWS spec ID to use for working spec extraction",
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

    # Extract working spec if available
    working_spec = None
    if args.caws_spec_id:
        try:
            from training.caws_context import extract_caws_context

            caws_context = extract_caws_context(".", spec_id=args.caws_spec_id)
            if caws_context:
                working_spec = {
                    "id": caws_context.spec_id,
                    "title": caws_context.title,
                    "risk_tier": caws_context.risk_tier,
                    "budget": caws_context.budget,
                    "scope": caws_context.scope,
                }
        except Exception as e:
            print(
                f"[generate_from_worker_outputs] WARN: Failed to extract CAWS context: {e}")

    # Load Worker outputs
    print(
        f"[generate_from_worker_outputs] Loading Worker outputs from {args.worker_outputs}")
    worker_outputs = load_jsonl(Path(args.worker_outputs))
    print(
        f"[generate_from_worker_outputs] Loaded {len(worker_outputs)} Worker outputs")

    # Load Teacher outputs if provided
    teacher_outputs = None
    if args.teacher_outputs:
        print(
            f"[generate_from_worker_outputs] Loading Teacher outputs from {args.teacher_outputs}")
        teacher_outputs = load_jsonl(Path(args.teacher_outputs))
        print(
            f"[generate_from_worker_outputs] Loaded {len(teacher_outputs)} Teacher outputs")

    # Generate pairs
    print(f"[generate_from_worker_outputs] Generating pairwise examples...")
    pairs = create_pair_from_worker_outputs(
        worker_outputs, teacher_outputs, working_spec)
    print(f"[generate_from_worker_outputs] Generated {len(pairs)} pairs")

    # Limit total if requested
    if args.total and len(pairs) > args.total:
        pairs = random.sample(pairs, args.total)
        print(f"[generate_from_worker_outputs] Limited to {args.total} pairs")

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[generate_from_worker_outputs] Output written to: {output_path}")


if __name__ == "__main__":
    main()
