# arbiter/judge_training/build_pairs_from_caws.py
# Transform CAWS adjudication logs -> pairwise judge training lines (v1 simplified)
# @author: @darianrosebrook

"""
Transform CAWS adjudication logs -> pairwise judge training lines.

Supports v1 simplified labels:
- winner: a|b|tie
- reasons: coarse reason codes (EVIDENCE_BETTER, BUDGET_RESPECTED, etc.)

Input JSONL rows (minimal):

{
  "id": "job-123",
  "prompt": "...",
  "candidates": [
      {"text": "...", "clauses": ["EVIDENCE_COMPLETENESS"], "score": 0.78},
      {"text": "...", "clauses": ["BUDGET_ADHERENCE"], "score": 0.55}
  ],
  "winner_index": 0
}
"""

import json
import sys
import argparse
from typing import Dict, Any, List


# v1 simplified reason code mapping
CLAUSE_TO_REASON = {
    "EVIDENCE_COMPLETENESS": "EVIDENCE_BETTER",
    "BUDGET_ADHERENCE": "BUDGET_RESPECTED",
    "GATE_INTEGRITY": "GATE_INTEGRITY",
    "PROVENANCE_CLARITY": "PROVENANCE_CLEAR",
    "WAIVER_JUSTIFICATION": "QUALITY_HIGHER",
}


def map_clauses_to_reasons(clauses: List[str]) -> List[str]:
    """Map CAWS clauses to v1 simplified reason codes."""
    reasons = []
    for clause in clauses:
        reason = CLAUSE_TO_REASON.get(clause)
        if reason and reason not in reasons:
            reasons.append(reason)
    # Add default reasons if none mapped
    if not reasons:
        reasons = ["QUALITY_HIGHER"]
    return reasons


def determine_caws_outcome(a_clauses: List[str], b_clauses: List[str], winner: str) -> str:
    """Determine CAWS outcome (pass/fail/waiver_required) from clauses."""
    if "WAIVER_JUSTIFICATION" in a_clauses or "WAIVER_JUSTIFICATION" in b_clauses:
        return "waiver_required"
    if winner == "tie":
        return "pass"  # Default for ties
    winner_clauses = a_clauses if winner == "a" else b_clauses
    if any(c in winner_clauses for c in ["BUDGET_ADHERENCE", "SCOPE_CORRECT", "GATE_INTEGRITY"]):
        return "pass"
    return "fail"


def main():
    ap = argparse.ArgumentParser(
        description="Transform CAWS adjudication logs to pairwise judge training format (v1)",
    )
    ap.add_argument(
        "input", help="Input JSONL file with CAWS adjudication logs")
    ap.add_argument("output", help="Output JSONL file path")
    ap.add_argument(
        "--include-working-spec",
        action="store_true",
        help="Include working spec from input if available",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as g:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)

                # Skip header lines
                if "__header__" in r:
                    continue

                c = r.get("candidates", [])
                if len(c) < 2:
                    continue

                a, b = c[0], c[1]
                winner_index = r.get("winner_index", 0)

                # Determine winner
                if winner_index == 0:
                    winner = "a"
                elif winner_index == 1:
                    winner = "b"
                else:
                    winner = "tie"

                # Extract clauses
                a_clauses = a.get("clauses", [])
                b_clauses = b.get("clauses", [])

                # Map clauses to reasons based on winner
                if winner == "a":
                    winner_clauses = a_clauses
                elif winner == "b":
                    winner_clauses = b_clauses
                else:  # tie
                    # For ties, combine clauses from both candidates
                    winner_clauses = list(set(a_clauses + b_clauses))

                reasons = map_clauses_to_reasons(winner_clauses)
                outcome = determine_caws_outcome(a_clauses, b_clauses, winner)

                # Build output row in training format
                row = {
                    "id": r.get("id", f"pair-{line_num:06d}"),
                    "prompt": r.get("prompt", ""),
                    "a": {
                        "text": a.get("text", ""),
                        "clauses": a_clauses,
                    },
                    "b": {
                        "text": b.get("text", ""),
                        "clauses": b_clauses,
                    },
                    "winner": winner,
                    "reasons": reasons,
                    "outcome": outcome,
                }

                # Add working spec if requested and available
                if args.include_working_spec and "working_spec" in r:
                    row["working_spec"] = r["working_spec"]

                # Add optional fields if present
                if a.get("change_diff"):
                    row["a"]["change_diff"] = a.get("change_diff")
                if a.get("rationale"):
                    row["a"]["rationale"] = a.get("rationale")
                if a.get("evidence_manifest"):
                    row["a"]["evidence_manifest"] = a.get("evidence_manifest")
                if a.get("provenance_chain"):
                    row["a"]["provenance_chain"] = a.get("provenance_chain")

                if b.get("change_diff"):
                    row["b"]["change_diff"] = b.get("change_diff")
                if b.get("rationale"):
                    row["b"]["rationale"] = b.get("rationale")
                if b.get("evidence_manifest"):
                    row["b"]["evidence_manifest"] = b.get("evidence_manifest")
                if b.get("provenance_chain"):
                    row["b"]["provenance_chain"] = b.get("provenance_chain")

                g.write(json.dumps(row, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(
                    f"[build_pairs_from_caws] WARN: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(
                    f"[build_pairs_from_caws] WARN: Error processing line {line_num}: {e}", file=sys.stderr)
                continue

    print(f"[build_pairs_from_caws] Wrote {args.output}")


if __name__ == "__main__":
    main()
