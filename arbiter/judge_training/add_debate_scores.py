"""
Add debate scores (E, B, G, P components) to existing Judge pairwise samples.

This is a v2 enhancement script that can be run after v1 training to enrich
the dataset with debate scores for multi-task training.

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


# CAWS Debate Scoring Formula: S = 0.4E + 0.3B + 0.2G + 0.1P
DEBATE_WEIGHTS = {
    "E": 0.4,  # Evidence Completeness
    "B": 0.3,  # Budget Adherence
    "G": 0.2,  # Gate Integrity
    "P": 0.1,  # Provenance Clarity
}


def calculate_evidence_score(sample: Dict[str, Any], side: str) -> float:
    """Calculate Evidence Completeness (E) score."""
    side_data = sample.get(side, {})
    evidence_manifest = side_data.get("evidence_manifest", {})
    clauses = side_data.get("clauses", [])

    score = 0.0

    # Check for evidence manifest
    if evidence_manifest:
        claims = evidence_manifest.get("claims", [])
        verification_status = evidence_manifest.get("verification_status", "pending")
        evidence_refs = evidence_manifest.get("evidence_references", [])

        # Base score from claims
        if claims:
            score += 0.3

        # Boost for verified status
        if verification_status == "verified":
            score += 0.4
        elif verification_status == "pending":
            score += 0.2

        # Boost for evidence references
        if evidence_refs:
            score += 0.2

        # Cap at 1.0
        score = min(1.0, score)

    # Check for EVIDENCE_COMPLETENESS clause
    if "EVIDENCE_COMPLETENESS" in clauses:
        score = max(score, 0.7)

    return round(score, 2)


def calculate_budget_score(sample: Dict[str, Any], side: str, working_spec: Dict[str, Any]) -> float:
    """Calculate Budget Adherence (B) score."""
    side_data = sample.get(side, {})
    change_diff = side_data.get("change_diff", "")
    clauses = side_data.get("clauses", [])

    score = 1.0  # Start with full score

    # Check for budget violations in change_diff
    if "EXCEEDS BUDGET" in change_diff.upper() or "BUDGET EXCEEDED" in change_diff.upper():
        score = 0.3

    # Check for BUDGET_ADHERENCE clause
    if "BUDGET_ADHERENCE" in clauses:
        score = 1.0
    elif "WAIVER_JUSTIFICATION" in clauses:
        # Waiver means budget exceeded but justified
        score = 0.6

    # Check working spec budget if available
    if working_spec and "budget" in working_spec:
        budget = working_spec["budget"]
        max_files = budget.get("max_files", 25)
        max_loc = budget.get("max_loc", 1000)

        # Try to extract file/LOC counts from change_diff
        if change_diff:
            # Simple heuristic: look for numbers
            import re
            file_match = re.search(r"(\d+)\s+files?", change_diff, re.IGNORECASE)
            loc_match = re.search(r"(\d+)\s+LOC", change_diff, re.IGNORECASE)

            if file_match:
                files_changed = int(file_match.group(1))
                if files_changed > max_files:
                    score = min(score, 0.4)

            if loc_match:
                loc_changed = int(loc_match.group(1))
                if loc_changed > max_loc:
                    score = min(score, 0.4)

    return round(score, 2)


def calculate_gate_score(side_data: Dict[str, Any]) -> float:
    """Calculate Gate Integrity (G) score."""
    evidence_manifest = side_data.get("evidence_manifest", {})
    clauses = side_data.get("clauses", [])
    text = side_data.get("text", "").lower()

    score = 0.5  # Default neutral score

    # Check for GATE_INTEGRITY clause
    if "GATE_INTEGRITY" in clauses:
        score = 1.0
    elif "WAIVER_JUSTIFICATION" in clauses:
        # Waiver means gates not met but justified
        score = 0.6

    # Check evidence manifest for gate-related claims
    if evidence_manifest:
        claims = evidence_manifest.get("claims", [])
        claim_text = " ".join(claims).lower()

        if any(word in claim_text for word in ["tests pass", "lint clean", "coverage"]):
            score = max(score, 0.8)
        if any(word in claim_text for word in ["tests fail", "lint error", "coverage below"]):
            score = min(score, 0.3)

    # Check text for gate-related content
    if any(phrase in text for phrase in ["all tests pass", "lint clean", "coverage above"]):
        score = max(score, 0.8)
    if any(phrase in text for phrase in ["tests fail", "lint errors", "coverage below"]):
        score = min(score, 0.3)

    return round(score, 2)


def calculate_provenance_score(side_data: Dict[str, Any]) -> float:
    """Calculate Provenance Clarity (P) score."""
    provenance_chain = side_data.get("provenance_chain", {})
    clauses = side_data.get("clauses", [])

    score = 0.0

    # Check for provenance chain
    if provenance_chain:
        steps = provenance_chain.get("steps", [])
        audit_trail = provenance_chain.get("audit_trail", "")

        # Base score from steps
        if steps:
            score += 0.4
            # More steps = better provenance
            if len(steps) >= 3:
                score += 0.3
            elif len(steps) >= 2:
                score += 0.2

        # Boost for audit trail
        if audit_trail:
            score += 0.2

        # Cap at 1.0
        score = min(1.0, score)

    # Check for PROVENANCE_CLARITY clause
    if "PROVENANCE_CLARITY" in clauses:
        score = max(score, 0.7)

    return round(score, 2)


def calculate_total_score(e: float, b: float, g: float, p: float) -> float:
    """Calculate total debate score using formula: S = 0.4E + 0.3B + 0.2G + 0.1P"""
    total = (
        DEBATE_WEIGHTS["E"] * e +
        DEBATE_WEIGHTS["B"] * b +
        DEBATE_WEIGHTS["G"] * g +
        DEBATE_WEIGHTS["P"] * p
    )
    return round(total, 2)


def add_debate_scores_to_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Add debate scores to a single sample."""
    working_spec = sample.get("working_spec", {})

    # Calculate scores for side A
    e_a = calculate_evidence_score(sample, "a")
    b_a = calculate_budget_score(sample, "a", working_spec)
    g_a = calculate_gate_score(sample.get("a", {}))
    p_a = calculate_provenance_score(sample.get("a", {}))
    total_a = calculate_total_score(e_a, b_a, g_a, p_a)

    # Calculate scores for side B
    e_b = calculate_evidence_score(sample, "b")
    b_b = calculate_budget_score(sample, "b", working_spec)
    g_b = calculate_gate_score(sample.get("b", {}))
    p_b = calculate_provenance_score(sample.get("b", {}))
    total_b = calculate_total_score(e_b, b_b, g_b, p_b)

    # Add debate scores
    sample["debate_scores"] = {
        "a": {
            "E": e_a,
            "B": b_a,
            "G": g_a,
            "P": p_a,
            "total": total_a,
        },
        "b": {
            "E": e_b,
            "B": b_b,
            "G": g_b,
            "P": p_b,
            "total": total_b,
        },
    }

    return sample


def main():
    ap = argparse.ArgumentParser(
        description="Add debate scores to Judge pairwise samples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--in", dest="input_file", required=True, help="Input Judge dataset JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    args = ap.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.out)

    # Load samples
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[add_debate_scores] WARN: Invalid JSON: {e}")
                continue

    print(f"[add_debate_scores] Loaded {len(samples)} samples from {input_file}")

    # Add debate scores
    enriched_samples = []
    for sample in samples:
        enriched = add_debate_scores_to_sample(sample)
        enriched_samples.append(enriched)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in enriched_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[add_debate_scores] Wrote {len(enriched_samples)} enriched samples to {output_file}")


if __name__ == "__main__":
    main()


