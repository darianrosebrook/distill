"""
CAWS Context Extraction Utilities for Training Dataset Generation

This module provides utilities for extracting CAWS (Coding-Agent Working Standard)
context from working specifications and policy files. CAWS context is used to
augment prompts during dataset generation, ensuring models learn to produce
CAWS-compliant outputs.

Based on patterns from CAWS_CONTEXT_USAGE_REPORT.md

Author: @darianrosebrook
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class CAWSContext:
    """Structured CAWS context for prompt augmentation"""
    spec_id: str
    title: str
    risk_tier: int
    mode: str
    budget: Dict[str, int]
    scope: Dict[str, List[str]]
    quality: Dict[str, Any]
    acceptance_summary: List[str]
    invariants: List[str]


def extract_caws_context(
    project_root: str = ".",
    spec_id: Optional[str] = None,
) -> Optional[CAWSContext]:
    """
    Extract minimal CAWS context for prompt augmentation.

    Priority order (matching CAWS spec resolution):
    1. Feature-specific spec: `.caws/specs/<spec-id>.yaml` (multi-agent safe)
    2. Legacy fallback: `.caws/working-spec.yaml`

    Reference: CAWS_CONTEXT_USAGE_REPORT.md lines 200-300

    Args:
        project_root: Root directory of the project
        spec_id: Optional spec ID to load specific spec

    Returns:
        CAWSContext object or None if no spec found
    """
    project_path = Path(project_root)
    caws_dir = project_path / ".caws"

    if not caws_dir.exists():
        return None

    # Priority 1: Feature-specific spec
    if spec_id:
        spec_path = caws_dir / "specs" / f"{spec_id}.yaml"
        if spec_path.exists():
            return _load_spec_context(spec_path, project_path)

    # Priority 2: Legacy working-spec.yaml
    legacy_spec = caws_dir / "working-spec.yaml"
    if legacy_spec.exists():
        return _load_spec_context(legacy_spec, project_path)

    return None


def _load_spec_context(spec_path: Path, project_root: Path) -> CAWSContext:
    """
    Load and format spec context.

    Args:
        spec_path: Path to working spec YAML file
        project_root: Root directory of the project

    Returns:
        CAWSContext object
    """
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)

    # Load policy for budget derivation
    policy_path = project_root / ".caws" / "policy.yaml"
    policy = {}
    if policy_path.exists():
        with open(policy_path, 'r') as f:
            policy = yaml.safe_load(f)

    # Derive budget from policy
    risk_tier = spec.get("risk_tier", 2)
    tier_budget = policy.get("risk_tiers", {}).get(str(risk_tier), {})

    # Default budget if not in policy
    if not tier_budget:
        tier_budget = {
            "max_files": 25,
            "max_loc": 1000,
            "coverage_threshold": 80,
            "mutation_threshold": 50,
        }

    # Extract acceptance criteria summary
    acceptance_criteria = spec.get("acceptance", [])
    acceptance_summary = []
    for ac in acceptance_criteria:
        if isinstance(ac, dict):
            summary = f"{ac.get('id', 'A')}: {ac.get('given', '')} → {ac.get('when', '')} → {ac.get('then', '')}"
            acceptance_summary.append(summary)
        elif isinstance(ac, str):
            acceptance_summary.append(ac)

    # Extract invariants
    invariants = spec.get("invariants", [])
    if isinstance(invariants, str):
        invariants = [invariants]

    return CAWSContext(
        spec_id=spec.get("id", "unknown"),
        title=spec.get("title", "Unknown"),
        risk_tier=risk_tier,
        mode=spec.get("mode", "feature"),
        budget={
            "max_files": tier_budget.get("max_files", 25),
            "max_loc": tier_budget.get("max_loc", 1000),
        },
        scope=spec.get("scope", {"in": [], "out": []}),
        quality={
            "coverage_threshold": tier_budget.get("coverage_threshold", 80),
            "mutation_threshold": tier_budget.get("mutation_threshold", 50),
        },
        acceptance_summary=acceptance_summary,
        invariants=invariants,
    )


def format_caws_compact(working_spec: Dict[str, Any]) -> str:
    """
    Format CAWS as compact JSON metadata (≤ 30 tokens target).

    PRIORITY 4: Token-light CAWS format for efficiency.

    Args:
        working_spec: Working specification dict or CAWSContext

    Returns:
        Compact JSON string with minimal CAWS metadata
    """
    # Import here to avoid circular dependency
    from training.prompt_templates import format_caws_compact as _format_compact
    return _format_compact(working_spec)


def format_caws_context_for_prompt(context: Optional[CAWSContext], compact: bool = False) -> str:
    """
    Format CAWS context as markdown for prompt augmentation.

    Reference: CAWS_CONTEXT_USAGE_REPORT.md lines 400-500

    Args:
        context: CAWSContext object or None
        compact: If True, use compact JSON format (≤ 30 tokens). If False, use verbose markdown.

    Returns:
        Formatted string (empty if context is None)
    """
    if not context:
        return ""

    if compact:
        # PRIORITY 4: Use compact JSON format
        return format_caws_compact(context)

    lines = [
        "## CAWS Context",
        f"**Spec ID**: {context.spec_id}",
        f"**Title**: {context.title}",
        f"**Risk Tier**: {context.risk_tier}",
        f"**Mode**: {context.mode}",
        "",
        "### Budget Constraints",
        f"- Max Files: {context.budget.get('max_files')}",
        f"- Max LOC: {context.budget.get('max_loc')}",
        "",
        "### Scope Boundaries",
    ]

    # Scope In
    scope_in = context.scope.get("in", [])
    if scope_in:
        lines.append(f"- In Scope: {', '.join(scope_in[:5])}")
        if len(scope_in) > 5:
            lines.append(f"  (and {len(scope_in) - 5} more)")
    else:
        lines.append("- In Scope: (not specified)")

    # Scope Out
    scope_out = context.scope.get("out", [])
    if scope_out:
        lines.append(f"- Out of Scope: {', '.join(scope_out[:5])}")
        if len(scope_out) > 5:
            lines.append(f"  (and {len(scope_out) - 5} more)")
    else:
        lines.append("- Out of Scope: (not specified)")

    lines.extend([
        "",
        "### Quality Gates",
        f"- Coverage Threshold: {context.quality.get('coverage_threshold')}%",
        f"- Mutation Threshold: {context.quality.get('mutation_threshold')}%",
    ])

    # Acceptance Criteria
    if context.acceptance_summary:
        lines.extend([
            "",
            "### Acceptance Criteria",
        ])
        for ac in context.acceptance_summary[:5]:
            lines.append(f"- {ac}")
        if len(context.acceptance_summary) > 5:
            lines.append(f"  (and {len(context.acceptance_summary) - 5} more)")

    # Invariants
    if context.invariants:
        lines.extend([
            "",
            "### Invariants",
        ])
        for inv in context.invariants[:5]:
            lines.append(f"- {inv}")
        if len(context.invariants) > 5:
            lines.append(f"  (and {len(context.invariants) - 5} more)")

    return "\n".join(lines)


def extract_caws_context_dict(
    project_root: str = ".",
    spec_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract CAWS context as dictionary (for JSON serialization).

    Args:
        project_root: Root directory of the project
        spec_id: Optional spec ID to load specific spec

    Returns:
        Dictionary representation of CAWS context or None
    """
    context = extract_caws_context(project_root, spec_id)
    if not context:
        return None

    return {
        "spec_id": context.spec_id,
        "title": context.title,
        "risk_tier": context.risk_tier,
        "mode": context.mode,
        "budget": context.budget,
        "scope": context.scope,
        "quality": context.quality,
        "acceptance_summary": context.acceptance_summary,
        "invariants": context.invariants,
    }


@lru_cache(maxsize=10)
def _load_policy_cached(policy_path: str) -> Dict[str, Any]:
    """
    Cached policy loader.

    Args:
        policy_path: Path to policy.yaml file

    Returns:
        Policy dictionary
    """
    if not os.path.exists(policy_path):
        return {}

    with open(policy_path, 'r') as f:
        return yaml.safe_load(f) or {}


def derive_budget(
    spec: Dict[str, Any],
    project_root: str = ".",
) -> Dict[str, Any]:
    """
    Derive effective budget from spec and policy, applying waivers if present.

    Reference: CAWS_CONTEXT_USAGE_REPORT.md lines 76-117

    Args:
        spec: Working spec dictionary
        project_root: Root directory of the project

    Returns:
        Dictionary with baseline, effective, waivers_applied, derived_at
    """
    project_path = Path(project_root)
    policy_path = project_path / ".caws" / "policy.yaml"

    # Load policy (with caching)
    policy = _load_policy_cached(str(policy_path))

    # Extract tier budget
    risk_tier = spec.get("risk_tier", 2)
    tier_budget = policy.get("risk_tiers", {}).get(str(risk_tier), {})

    # Baseline budget
    baseline = {
        "max_files": tier_budget.get("max_files", 25),
        "max_loc": tier_budget.get("max_loc", 1000),
    }

    # Effective budget (starts as baseline)
    effective = baseline.copy()

    # Apply waivers if present
    waivers_applied = []
    waiver_ids = spec.get("waiver_ids", [])
    if isinstance(waiver_ids, str):
        waiver_ids = [waiver_ids]

    for waiver_id in waiver_ids:
        waiver_path = project_path / ".caws" / "waivers" / f"{waiver_id}.yaml"
        if waiver_path.exists():
            with open(waiver_path, 'r') as f:
                waiver = yaml.safe_load(f)

            if waiver.get("status") == "active":
                delta = waiver.get("delta", {})
                if delta:
                    effective["max_files"] += delta.get("max_files", 0)
                    effective["max_loc"] += delta.get("max_loc", 0)
                    waivers_applied.append(waiver_id)

    return {
        "baseline": baseline,
        "effective": effective,
        "waivers_applied": waivers_applied,
        "derived_at": str(Path(project_root).absolute()),
    }


# Example usage
if __name__ == "__main__":
    # Test extraction
    context = extract_caws_context(".")
    if context:
        print("CAWS Context Extracted:")
        print(f"  Spec ID: {context.spec_id}")
        print(f"  Title: {context.title}")
        print(f"  Risk Tier: {context.risk_tier}")
        print(f"  Budget: {context.budget}")
        print(f"  Scope In: {context.scope.get('in', [])}")
        print("\nFormatted for Prompt:")
        print(format_caws_context_for_prompt(context))
    else:
        print("No CAWS context found in current directory")
