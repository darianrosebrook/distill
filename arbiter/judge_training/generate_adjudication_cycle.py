"""
Generate CAWS Adjudication Cycle examples for Judge training.

v1.5: Generates examples with minimal v1 schema:
- winner: a|b|tie
- reasons: free-text rationale or short label list
- clauses: CAWS-relevant tags (BUDGET_RESPECTED, SCOPE_VIOLATION, etc.)

For v2 (future): will add debate scores and explicit stage labels.

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from training.caws_context import extract_caws_context, extract_caws_context_dict


# Actual CAWS clause names (used in training)
CAWS_CLAUSES = [
    "EVIDENCE_COMPLETENESS",
    "BUDGET_ADHERENCE",
    "GATE_INTEGRITY",
    "PROVENANCE_CLARITY",
    "WAIVER_JUSTIFICATION",
]


def generate_pleading_stage_pair(working_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate pleading stage example: complete vs incomplete evidence."""
    return {
        "prompt": "Worker submits change with rationale and evidence manifest.",
        "a": {
            "text": "Change submitted with complete evidence manifest and clear rationale.",
            "change_diff": "Modified src/main.py (+50 LOC)",
            "rationale": "Implements feature X as specified in requirements.",
            "evidence_manifest": {
                "claims": ["Feature X implemented", "Tests pass", "Budget respected"],
                "verification_status": "verified",
                "evidence_references": ["file://src/main.py", "file://tests/test_main.py"],
            },
            "provenance_chain": {
                "steps": [
                    {"step": 1, "action": "read", "target": "requirements.md"},
                    {"step": 2, "action": "implement", "target": "src/main.py"},
                    {"step": 3, "action": "test", "target": "tests/test_main.py"},
                ],
                "audit_trail": "Complete provenance chain",
            },
        },
        "b": {
            "text": "Change submitted but evidence manifest is incomplete.",
            "change_diff": "Modified src/main.py (+50 LOC)",
            "rationale": "Implements feature X.",
            "evidence_manifest": {
                "claims": ["Feature X implemented"],
                "verification_status": "pending",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [
                    {"step": 1, "action": "implement", "target": "src/main.py"},
                ],
                "audit_trail": "",
            },
        },
        "winner": "a",
        "a_clauses": ["EVIDENCE_COMPLETENESS", "PROVENANCE_CLARITY", "GATE_INTEGRITY"],
        "b_clauses": ["EVIDENCE_COMPLETENESS"],
    }


def generate_examination_stage_pair(working_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate examination stage example: budget and scope compliance."""
    max_files = working_spec["budget"]["max_files"]
    max_loc = working_spec["budget"]["max_loc"]

    return {
        "prompt": f"Examine change for CAWS budget and scope compliance. Budget: {max_files} files, {max_loc} LOC.",
        "a": {
            "text": f"Change respects budget ({max_files - 5} files, {max_loc - 100} LOC) and scope boundaries.",
            "change_diff": f"Modified {max_files - 5} files, {max_loc - 100} LOC (within budget)",
            "rationale": "All changes within scope and budget.",
            "evidence_manifest": {
                "claims": ["Budget respected", "Scope boundaries maintained"],
                "verification_status": "verified",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "b": {
            "text": f"Change exceeds budget ({max_files + 10} files, {max_loc + 200} LOC) and modifies out-of-scope files.",
            "change_diff": f"Modified {max_files + 10} files, {max_loc + 200} LOC (EXCEEDS BUDGET)",
            "rationale": "Needed to modify additional files.",
            "evidence_manifest": {
                "claims": [],
                "verification_status": "pending",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "winner": "a",
        "a_clauses": ["BUDGET_ADHERENCE", "GATE_INTEGRITY"],
        "b_clauses": ["WAIVER_JUSTIFICATION"],
    }


def generate_deliberation_stage_pair(working_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deliberation stage example: gate integrity (tests, lint, coverage)."""
    return {
        "prompt": "Deliberate on gate integrity: tests, lint, coverage.",
        "a": {
            "text": "All tests pass, lint clean, coverage above threshold.",
            "change_diff": "Modified src/main.py",
            "rationale": "Change passes all quality gates.",
            "evidence_manifest": {
                "claims": ["Tests pass", "Lint clean", "Coverage 85%"],
                "verification_status": "verified",
                "evidence_references": ["test://results.json", "lint://results.json"],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "b": {
            "text": "Tests fail, lint errors present, coverage below threshold.",
            "change_diff": "Modified src/main.py",
            "rationale": "Change needs more work.",
            "evidence_manifest": {
                "claims": ["Tests fail", "Lint errors", "Coverage 60%"],
                "verification_status": "rejected",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "winner": "a",
        "a_clauses": ["GATE_INTEGRITY", "EVIDENCE_COMPLETENESS"],
        "b_clauses": [],
    }


def generate_verdict_stage_pair(working_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate verdict stage example: clear pass/fail/waiver cases."""
    return {
        "prompt": "Issue CAWS verdict: PASS / FAIL / WAIVER_REQUIRED.",
        "a": {
            "text": "PASS: All CAWS requirements met, no waiver needed.",
            "change_diff": "Modified src/main.py",
            "rationale": "Change fully complies with CAWS.",
            "evidence_manifest": {
                "claims": ["All requirements met"],
                "verification_status": "verified",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "b": {
            "text": "FAIL: Budget exceeded without waiver justification.",
            "change_diff": "Modified src/main.py (EXCEEDS BUDGET)",
            "rationale": "Change exceeds budget.",
            "evidence_manifest": {
                "claims": [],
                "verification_status": "rejected",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [],
                "audit_trail": "",
            },
        },
        "winner": "a",
        "a_clauses": ["BUDGET_ADHERENCE", "GATE_INTEGRITY", "EVIDENCE_COMPLETENESS"],
        "b_clauses": ["WAIVER_JUSTIFICATION"],
    }


def _generate_reasons(a_clauses: List[str], b_clauses: List[str], winner: str) -> str:
    """Generate free-text reasons for winner determination."""
    if winner == "a":
        reasons = f"Solution A is preferred because it satisfies: {', '.join(a_clauses)}"
        if b_clauses:
            reasons += f", while Solution B only satisfies: {', '.join(b_clauses)}"
    elif winner == "b":
        reasons = f"Solution B is preferred because it satisfies: {', '.join(b_clauses)}"
        if a_clauses:
            reasons += f", while Solution A only satisfies: {', '.join(a_clauses)}"
    else:
        reasons = f"Both solutions are equivalent. A satisfies: {', '.join(a_clauses)}, B satisfies: {', '.join(b_clauses)}"
    return reasons


def generate_publication_stage_pair(working_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate publication stage example: provenance chain completeness."""
    return {
        "prompt": "Publish verdict with complete provenance chain to git.",
        "a": {
            "text": "Verdict published with complete provenance chain and git trailer.",
            "change_diff": "Modified src/main.py",
            "rationale": "Published with full audit trail.",
            "evidence_manifest": {
                "claims": [],
                "verification_status": "verified",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [
                    {"step": 1, "action": "pleading", "target": "change.diff"},
                    {"step": 2, "action": "examination", "target": "budget_check"},
                    {"step": 3, "action": "deliberation", "target": "gate_check"},
                    {"step": 4, "action": "verdict", "target": "PASS"},
                    {"step": 5, "action": "publication", "target": "git_commit"},
                ],
                "audit_trail": "CAWS-VERDICT-ID: abc123",
            },
        },
        "b": {
            "text": "Verdict published but provenance chain is incomplete.",
            "change_diff": "Modified src/main.py",
            "rationale": "Published verdict.",
            "evidence_manifest": {
                "claims": [],
                "verification_status": "verified",
                "evidence_references": [],
            },
            "provenance_chain": {
                "steps": [
                    {"step": 1, "action": "verdict", "target": "PASS"},
                ],
                "audit_trail": "",
            },
        },
        "winner": "a",
        "a_clauses": ["PROVENANCE_CLARITY", "EVIDENCE_COMPLETENESS", "GATE_INTEGRITY"],
        "b_clauses": ["PROVENANCE_CLARITY"],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate CAWS Adjudication Cycle examples for Judge v1.5 training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Total number of samples to generate (default: 2000)",
    )
    ap.add_argument(
        "--samples-per-stage",
        type=int,
        default=None,
        help="Samples per stage (default: samples / 5)",
    )
    ap.add_argument(
        "--caws-spec-id",
        help="CAWS spec ID to use for context extraction",
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

    # Extract CAWS context if available
    caws_context = None
    if args.caws_spec_id:
        try:
            caws_context = extract_caws_context(".", spec_id=args.caws_spec_id)
        except Exception as e:
            print(
                f"[generate_adjudication_cycle] WARN: Failed to extract CAWS context: {e}")

    # Generate working spec
    if caws_context:
        working_spec = {
            "id": caws_context.spec_id,
            "title": caws_context.title,
            "risk_tier": caws_context.risk_tier,
            "budget": caws_context.budget,
            "scope": caws_context.scope,
        }
    else:
        working_spec = {
            "id": "FEAT-0001",
            "title": "Feature Implementation",
            "risk_tier": 2,
            "budget": {"max_files": 25, "max_loc": 1000},
            "scope": {"in": ["src/"], "out": ["third_party/"]},
        }

    # Stage generators
    stage_generators = {
        "pleading": generate_pleading_stage_pair,
        "examination": generate_examination_stage_pair,
        "deliberation": generate_deliberation_stage_pair,
        "verdict": generate_verdict_stage_pair,
        "publication": generate_publication_stage_pair,
    }

    # Calculate samples per stage
    samples_per_stage = args.samples_per_stage or (
        args.samples // len(stage_generators))

    # Generate samples
    samples = []
    for stage_name, generator in stage_generators.items():
        for i in range(samples_per_stage):
            scenario = generator(working_spec)

            # Add clauses to a and b
            scenario["a"]["clauses"] = scenario.get("a_clauses", [])
            scenario["b"]["clauses"] = scenario.get("b_clauses", [])

            # Generate reasons based on clauses and winner
            reasons = _generate_reasons(scenario["a"]["clauses"], scenario["b"]["clauses"], scenario["winner"])

            sample = {
                "id": f"judge-cycle-{stage_name}-{i+1:06d}",
                "prompt": scenario["prompt"],
                "working_spec": working_spec,
                "a": {
                    "text": scenario["a"]["text"],
                    "clauses": scenario["a"]["clauses"],
                },
                "b": {
                    "text": scenario["b"]["text"],
                    "clauses": scenario["b"]["clauses"],
                },
                "winner": scenario["winner"],
                "reasons": reasons,
            }

            # Add optional fields if present
            if scenario["a"].get("change_diff"):
                sample["a"]["change_diff"] = scenario["a"]["change_diff"]
            if scenario["a"].get("rationale"):
                sample["a"]["rationale"] = scenario["a"]["rationale"]
            if scenario["a"].get("evidence_manifest"):
                sample["a"]["evidence_manifest"] = scenario["a"]["evidence_manifest"]
            if scenario["a"].get("provenance_chain"):
                sample["a"]["provenance_chain"] = scenario["a"]["provenance_chain"]

            if scenario["b"].get("change_diff"):
                sample["b"]["change_diff"] = scenario["b"]["change_diff"]
            if scenario["b"].get("rationale"):
                sample["b"]["rationale"] = scenario["b"]["rationale"]
            if scenario["b"].get("evidence_manifest"):
                sample["b"]["evidence_manifest"] = scenario["b"]["evidence_manifest"]
            if scenario["b"].get("provenance_chain"):
                sample["b"]["provenance_chain"] = scenario["b"]["provenance_chain"]

            samples.append(sample)

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"[generate_adjudication_cycle] Generated {len(samples)} pairwise samples")
    print(f"  Samples per stage: {samples_per_stage}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
