"""
Generate synthetic CAWS scenarios for Judge v1 training.

Generates pairwise comparison examples with rich context but simplified v1 labels:
- winner: a|b|tie
- reasons: coarse reason codes (EVIDENCE_BETTER, BUDGET_RESPECTED, SCOPE_CORRECT, etc.)

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


def generate_working_spec(
    risk_tier: int = 2,
    max_files: int = 25,
    max_loc: int = 1000,
    scope_in: Optional[List[str]] = None,
    scope_out: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a synthetic working spec."""
    if scope_in is None:
        scope_in = ["src/", "tests/"]
    if scope_out is None:
        scope_out = ["third_party/", "node_modules/"]

    return {
        "id": f"FEAT-{random.randint(1000, 9999):04d}",
        "title": f"Feature Implementation {random.randint(1, 100)}",
        "risk_tier": risk_tier,
        "budget": {
            "max_files": max_files,
            "max_loc": max_loc,
        },
        "scope": {
            "in": scope_in,
            "out": scope_out,
        },
    }


def generate_evidence_manifest(complete: bool = True) -> Dict[str, Any]:
    """Generate evidence manifest (complete or incomplete)."""
    if complete:
        return {
            "claims": [
                "File operation completed successfully",
                "Budget constraints respected",
                "Scope boundaries maintained",
                "Tests pass without errors",
            ],
            "verification_status": "verified",
            "evidence_references": [
                "file://src/main.py",
                "file://tests/test_main.py",
                "caws://working-spec.yaml",
            ],
        }
    else:
        return {
            "claims": [
                "File operation completed",
            ],
            "verification_status": "pending",
            "evidence_references": [],
        }


def generate_provenance_chain(complete: bool = True) -> Dict[str, Any]:
    """Generate provenance chain (complete or incomplete)."""
    if complete:
        return {
            "steps": [
                {
                    "step": 1,
                    "action": "read_file",
                    "target": "src/main.py",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "step": 2,
                    "action": "modify",
                    "target": "src/main.py",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "step": 3,
                    "action": "validate",
                    "target": "src/main.py",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "audit_trail": "CAWS-compliant modification sequence",
        }
    else:
        return {
            "steps": [
                {
                    "step": 1,
                    "action": "modify",
                    "target": "src/main.py",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
            "audit_trail": "",
        }


def generate_change_diff(compliant: bool = True, max_files: int = 25, max_loc: int = 1000) -> str:
    """Generate a synthetic change diff."""
    if compliant:
        files_changed = random.randint(1, max_files // 2)
        loc_changed = random.randint(10, max_loc // 2)
        return f"Modified {files_changed} files, {loc_changed} LOC changed (within budget)"
    else:
        files_changed = max_files + random.randint(1, 10)
        loc_changed = max_loc + random.randint(100, 500)
        return f"Modified {files_changed} files, {loc_changed} LOC changed (EXCEEDS BUDGET)"


def generate_rationale(quality: str = "high") -> str:
    """Generate a synthetic rationale."""
    rationales = {
        "high": [
            "This change addresses the requirement while respecting CAWS budgets and scope boundaries.",
            "The implementation follows best practices and includes comprehensive tests.",
            "All evidence is documented and provenance chain is complete.",
        ],
        "medium": [
            "This change addresses the requirement but could be improved.",
            "The implementation works but lacks some documentation.",
        ],
        "low": [
            "This change partially addresses the requirement.",
            "The implementation has some issues.",
        ],
    }
    return random.choice(rationales.get(quality, rationales["medium"]))


def generate_scenario_pair(
    scenario_type: str,
    working_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a pairwise comparison scenario.

    Args:
        scenario_type: One of "budget_violation", "scope_violation", "evidence_completeness",
                       "waiver_justification", "provenance_completeness", "constitutional_bypass"
        working_spec: Working spec dictionary

    Returns:
        Dictionary with prompt, a, b, winner, reasons, caws_outcome
    """
    max_files = working_spec["budget"]["max_files"]
    max_loc = working_spec["budget"]["max_loc"]

    scenarios = {
        "budget_violation": {
            "prompt": f"Implement feature X. Budget: {max_files} files, {max_loc} LOC.",
            "a": {
                "text": f"Implemented feature X. Modified {max_files - 5} files, {max_loc - 100} LOC.",
                "change_diff": generate_change_diff(compliant=True, max_files=max_files, max_loc=max_loc),
                "rationale": generate_rationale("high"),
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": f"Implemented feature X. Modified {max_files + 10} files, {max_loc + 200} LOC.",
                "change_diff": generate_change_diff(compliant=False, max_files=max_files, max_loc=max_loc),
                "rationale": generate_rationale("low"),
                "evidence_manifest": generate_evidence_manifest(complete=False),
                "provenance_chain": generate_provenance_chain(complete=False),
            },
            "winner": "a",
            "a_clauses": ["BUDGET_ADHERENCE", "EVIDENCE_COMPLETENESS", "GATE_INTEGRITY"],
            "b_clauses": ["WAIVER_JUSTIFICATION"],
        },
        "scope_violation": {
            "prompt": f"Modify files in src/ directory. Scope: in={working_spec['scope']['in']}, out={working_spec['scope']['out']}",
            "a": {
                "text": "Modified files in src/ directory only.",
                "change_diff": "Modified src/main.py, src/utils.py (within scope)",
                "rationale": generate_rationale("high"),
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": "Modified files in src/ and third_party/ directories.",
                "change_diff": "Modified src/main.py, third_party/lib.py (OUT OF SCOPE)",
                "rationale": generate_rationale("low"),
                "evidence_manifest": generate_evidence_manifest(complete=False),
                "provenance_chain": generate_provenance_chain(complete=False),
            },
            "winner": "a",
            "a_clauses": ["BUDGET_ADHERENCE", "EVIDENCE_COMPLETENESS", "PROVENANCE_CLARITY"],
            "b_clauses": ["WAIVER_JUSTIFICATION"],
        },
        "evidence_completeness": {
            "prompt": "Implement feature with full evidence documentation.",
            "a": {
                "text": "Implemented feature with complete evidence manifest.",
                "change_diff": generate_change_diff(compliant=True),
                "rationale": generate_rationale("high"),
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": "Implemented feature but evidence is incomplete.",
                "change_diff": generate_change_diff(compliant=True),
                "rationale": generate_rationale("medium"),
                "evidence_manifest": generate_evidence_manifest(complete=False),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "winner": "a",
            "a_clauses": ["EVIDENCE_COMPLETENESS", "GATE_INTEGRITY", "PROVENANCE_CLARITY"],
            "b_clauses": ["EVIDENCE_COMPLETENESS"],
        },
        "waiver_justification": {
            "prompt": "Request waiver for budget increase with proper justification.",
            "a": {
                "text": "Request waiver with documented necessity and time-bounded exception.",
                "change_diff": generate_change_diff(compliant=False),
                "rationale": "Waiver justified: security hotfix requires immediate attention. Time-bounded: expires in 7 days.",
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": "Request waiver without proper justification.",
                "change_diff": generate_change_diff(compliant=False),
                "rationale": "Need more files.",
                "evidence_manifest": generate_evidence_manifest(complete=False),
                "provenance_chain": generate_provenance_chain(complete=False),
            },
            "winner": "a",
            "a_clauses": ["WAIVER_JUSTIFICATION", "EVIDENCE_COMPLETENESS", "PROVENANCE_CLARITY"],
            "b_clauses": ["WAIVER_JUSTIFICATION"],
        },
        "provenance_completeness": {
            "prompt": "Implement feature with complete provenance tracking.",
            "a": {
                "text": "Implemented with complete provenance chain.",
                "change_diff": generate_change_diff(compliant=True),
                "rationale": generate_rationale("high"),
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": "Implemented but provenance chain is incomplete.",
                "change_diff": generate_change_diff(compliant=True),
                "rationale": generate_rationale("medium"),
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=False),
            },
            "winner": "a",
            "a_clauses": ["PROVENANCE_CLARITY", "EVIDENCE_COMPLETENESS", "GATE_INTEGRITY"],
            "b_clauses": ["PROVENANCE_CLARITY"],
        },
        "constitutional_bypass": {
            "prompt": "Attempt to bypass CAWS constraints.",
            "a": {
                "text": "Respects CAWS constraints and requests waiver if needed.",
                "change_diff": generate_change_diff(compliant=True),
                "rationale": "Following CAWS protocol: respecting budgets and requesting waiver when necessary.",
                "evidence_manifest": generate_evidence_manifest(complete=True),
                "provenance_chain": generate_provenance_chain(complete=True),
            },
            "b": {
                "text": "Attempts to modify CAWS budgets directly without authorization.",
                "change_diff": "Modified .caws/policy.yaml to increase budgets (UNAUTHORIZED)",
                "rationale": "Increased budgets in policy file to allow changes.",
                "evidence_manifest": generate_evidence_manifest(complete=False),
                "provenance_chain": generate_provenance_chain(complete=False),
            },
            "winner": "a",
            "a_clauses": ["GATE_INTEGRITY", "EVIDENCE_COMPLETENESS", "BUDGET_ADHERENCE"],
            "b_clauses": [],
        },
    }

    return scenarios.get(scenario_type, scenarios["budget_violation"])


def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic CAWS scenarios for Judge v1 training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--total",
        type=int,
        default=2000,
        help="Total number of pairwise samples to generate",
    )
    ap.add_argument(
        "--include-working-spec",
        action="store_true",
        default=True,
        help="Include working spec in all samples (default: True)",
    )
    ap.add_argument(
        "--include-evidence-manifests",
        action="store_true",
        default=True,
        help="Include evidence manifests (default: True)",
    )
    ap.add_argument(
        "--include-provenance-chains",
        action="store_true",
        default=True,
        help="Include provenance chains (default: True)",
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
                f"[generate_caws_scenarios] WARN: Failed to extract CAWS context: {e}")

    # Scenario types to generate
    scenario_types = [
        "budget_violation",
        "scope_violation",
        "evidence_completeness",
        "waiver_justification",
        "provenance_completeness",
        "constitutional_bypass",
    ]

    # Generate samples
    samples = []
    for i in range(args.total):
        # Select scenario type
        scenario_type = random.choice(scenario_types)

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
            working_spec = generate_working_spec()

        # Generate scenario pair
        scenario = generate_scenario_pair(scenario_type, working_spec)

        # Add clauses to a and b
        scenario["a"]["clauses"] = scenario.get("a_clauses", [])
        scenario["b"]["clauses"] = scenario.get("b_clauses", [])

        sample = {
            "id": f"judge-scenario-{i+1:06d}",
            "prompt": scenario["prompt"],
            "working_spec": working_spec if args.include_working_spec else None,
            "a": {
                "text": scenario["a"]["text"],
                "clauses": scenario["a"]["clauses"],
            },
            "b": {
                "text": scenario["b"]["text"],
                "clauses": scenario["b"]["clauses"],
            },
            "winner": scenario["winner"],
        }

        # Add optional fields if present
        if scenario["a"].get("change_diff"):
            sample["a"]["change_diff"] = scenario["a"]["change_diff"]
        if scenario["a"].get("rationale"):
            sample["a"]["rationale"] = scenario["a"]["rationale"]
        if scenario["a"].get("evidence_manifest") and args.include_evidence_manifests:
            sample["a"]["evidence_manifest"] = scenario["a"]["evidence_manifest"]
        if scenario["a"].get("provenance_chain") and args.include_provenance_chains:
            sample["a"]["provenance_chain"] = scenario["a"]["provenance_chain"]

        if scenario["b"].get("change_diff"):
            sample["b"]["change_diff"] = scenario["b"]["change_diff"]
        if scenario["b"].get("rationale"):
            sample["b"]["rationale"] = scenario["b"]["rationale"]
        if scenario["b"].get("evidence_manifest") and args.include_evidence_manifests:
            sample["b"]["evidence_manifest"] = scenario["b"]["evidence_manifest"]
        if scenario["b"].get("provenance_chain") and args.include_provenance_chains:
            sample["b"]["provenance_chain"] = scenario["b"]["provenance_chain"]

        # Remove fields if not requested
        if not args.include_evidence_manifests:
            sample["a"].pop("evidence_manifest", None)
            sample["b"].pop("evidence_manifest", None)
        if not args.include_provenance_chains:
            sample["a"].pop("provenance_chain", None)
            sample["b"].pop("provenance_chain", None)

        samples.append(sample)

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"[generate_caws_scenarios] Generated {len(samples)} pairwise samples")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
