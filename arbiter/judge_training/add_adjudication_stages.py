"""
Add adjudication cycle stage labels to existing Judge pairwise samples.

This is a v2 enhancement script that classifies samples into one of the 5 stages:
- pleading: Worker submits change with rationale and evidence manifest
- examination: Arbiter checks CAWS budgets and structural diffs
- deliberation: Arbiter runs verifier tests and collects gate metrics
- verdict: Arbiter issues PASS / FAIL / WAIVER_REQUIRED
- publication: Arbiter commits verdict + provenance to git

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def classify_stage(sample: Dict[str, Any]) -> str:
    """Classify a sample into one of the 5 adjudication cycle stages."""
    prompt = sample.get("prompt", "").lower()
    a_data = sample.get("a", {})
    b_data = sample.get("b", {})

    # Check for explicit stage indicators in prompt
    if "pleading" in prompt or "submits change" in prompt or "evidence manifest" in prompt:
        return "pleading"
    if "examination" in prompt or "examine" in prompt or "budget and scope" in prompt:
        return "examination"
    if "deliberation" in prompt or "deliberate" in prompt or "gate integrity" in prompt:
        return "deliberation"
    if "verdict" in prompt or "pass" in prompt.lower() or "fail" in prompt.lower():
        return "verdict"
    if "publication" in prompt or "publish" in prompt or "git" in prompt.lower():
        return "publication"

    # Classify based on content structure
    # Pleading: Has evidence manifest and rationale
    has_evidence_a = bool(a_data.get("evidence_manifest"))
    has_rationale_a = bool(a_data.get("rationale"))
    has_evidence_b = bool(b_data.get("evidence_manifest"))
    has_rationale_b = bool(b_data.get("rationale"))

    if (has_evidence_a or has_evidence_b) and (has_rationale_a or has_rationale_b):
        return "pleading"

    # Examination: Has change_diff and budget/scope checks
    has_diff_a = bool(a_data.get("change_diff"))
    has_diff_b = bool(b_data.get("change_diff"))
    working_spec = sample.get("working_spec")
    has_budget = bool(working_spec and working_spec.get("budget"))

    if (has_diff_a or has_diff_b) and has_budget:
        # Check if prompt mentions budget/scope
        if "budget" in prompt or "scope" in prompt:
            return "examination"

    # Deliberation: Has test/lint/coverage results in evidence manifest
    for side_data in [a_data, b_data]:
        evidence = side_data.get("evidence_manifest", {})
        claims = evidence.get("claims", [])
        claim_text = " ".join(claims).lower()
        if any(word in claim_text for word in ["test", "lint", "coverage", "gate"]):
            return "deliberation"

    # Verdict: Has clear PASS/FAIL/WAIVER in text
    for side_data in [a_data, b_data]:
        text = side_data.get("text", "").lower()
        if any(phrase in text for phrase in ["pass:", "fail:", "waiver_required", "verdict"]):
            return "verdict"

    # Publication: Has complete provenance chain with git/audit trail
    for side_data in [a_data, b_data]:
        provenance = side_data.get("provenance_chain", {})
        steps = provenance.get("steps", [])
        audit_trail = provenance.get("audit_trail", "")
        if len(steps) >= 4 and audit_trail:
            # Check if audit trail mentions git or publication
            if "git" in audit_trail.lower() or "commit" in audit_trail.lower():
                return "publication"

    # Default to verdict if unclear (most common stage)
    return "verdict"


def main():
    ap = argparse.ArgumentParser(
        description="Add adjudication cycle stage labels to Judge pairwise samples",
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
                print(f"[add_adjudication_stages] WARN: Invalid JSON: {e}")
                continue

    print(f"[add_adjudication_stages] Loaded {len(samples)} samples from {input_file}")

    # Classify and add stages
    enriched_samples = []
    stage_counts = {}
    for sample in samples:
        stage = classify_stage(sample)
        sample["adjudication_stage"] = stage
        enriched_samples.append(sample)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    # Print distribution
    print("\n[add_adjudication_stages] Stage distribution:")
    for stage, count in sorted(stage_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {stage}: {count} ({pct:.1f}%)")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in enriched_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[add_adjudication_stages] Wrote {len(enriched_samples)} enriched samples to {output_file}")


if __name__ == "__main__":
    main()


