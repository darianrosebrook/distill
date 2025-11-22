# training/data_gap_filling.py
"""Safe gap-filling library – training/data_gap_filling.py

This is the structural/safe side (no “hallucinating” semantics). It:

Applies rule objects to each record

Focuses on adding well-typed, conservative defaults for CAWS/Arbiter fields

Can be extended later to plug in predictive models
example: python -m training.data_gap_filling data/kd_mix_1500.jsonl data/kd_mix_1500_structural.jsonl --role worker
Later, when you add predictive models, you can wrap them in GapFillRule instances as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

Json = Dict[str, Any]
FillFn = Callable[[Json], Any]


@dataclass
class GapFillRule:
    """Single rule for filling or normalizing a field.

    The rule function receives the full record and returns the filled value.
    """

    name: str
    field: str
    fn: FillFn
    apply_if_missing_only: bool = True
    description: str = ""

    def apply(self, record: Json, track_metadata: bool = True) -> bool:
        """Apply rule to record. Returns True if field was filled/modified."""
        if self.apply_if_missing_only and self.field in record and record[self.field] is not None:
            return False
        
        old_value = record.get(self.field)
        record[self.field] = self.fn(record)
        was_filled = old_value != record[self.field]
        
        if track_metadata and was_filled:
            if "gap_fill_metadata" not in record:
                record["gap_fill_metadata"] = {}
            record["gap_fill_metadata"][self.field] = {
                "method": "structural",
                "rule": self.name,
                "description": self.description,
            }
        
        return was_filled


class DataGapFiller:
    """Applies a set of gap-filling rules to dataset records."""

    def __init__(self, rules: List[GapFillRule], track_metadata: bool = True) -> None:
        self.rules = rules
        self.track_metadata = track_metadata

    def fill_record(self, record: Json) -> Json:
        """Fill record with all rules, then enforce field dependencies."""
        for rule in self.rules:
            rule.apply(record, track_metadata=self.track_metadata)
        
        # Enforce field dependencies after all rules applied
        _enforce_field_dependencies(record)
        
        return record

    def iter_fill(self, records: Iterable[Json]) -> Iterable[Json]:
        for rec in records:
            yield self.fill_record(rec)


# ---- CAWS / Arbiter aware rules ----------------------------------------


def _normalize_working_spec(ws: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize working_spec, filling missing subfields."""
    if not isinstance(ws, dict):
        ws = {}
    
    budget = ws.get("budget")
    if not isinstance(budget, dict):
        budget = {"max_files": 0, "max_loc": 0}
    else:
        budget = {
            "max_files": budget.get("max_files", 0),
            "max_loc": budget.get("max_loc", 0),
        }
    
    scope = ws.get("scope")
    if not isinstance(scope, dict):
        scope = {"in": [], "out": []}
    else:
        scope = {
            "in": scope.get("in", []) if isinstance(scope.get("in"), list) else [],
            "out": scope.get("out", []) if isinstance(scope.get("out"), list) else [],
        }
    
    return {
        "id": ws.get("id", "UNKNOWN"),
        "title": ws.get("title", ""),
        "risk_tier": ws.get("risk_tier", 0),
        "budget": budget,
        "scope": scope,
    }


def _ensure_caws_context(record: Json) -> Any:
    """Fill a minimal caws_context shell if missing, or normalize existing partial data.

    We do *not* invent real budgets or scopes here; that's a semantic step.
    This gives your model structural regularity without asserting false facts.
    """
    existing = record.get("caws_context")
    if isinstance(existing, dict):
        # Normalize partial CAWS context
        working_spec = existing.get("working_spec") or record.get("working_spec") or {}
        return {
            "working_spec": _normalize_working_spec(working_spec),
        }

    working_spec = record.get("working_spec") or {}
    return {
        "working_spec": _normalize_working_spec(working_spec),
    }


def _ensure_evidence_manifest(record: Json) -> Any:
    """Ensure evidence_manifest exists with all required subfields."""
    existing = record.get("evidence_manifest")
    if isinstance(existing, dict):
        # Normalize partial manifest
        return {
            "claims": existing.get("claims", []) if isinstance(existing.get("claims"), list) else [],
            "verification_status": existing.get("verification_status", "UNKNOWN"),
            "evidence_references": existing.get("evidence_references", []) if isinstance(existing.get("evidence_references"), list) else [],
        }

    return {
        "claims": [],
        "verification_status": "UNKNOWN",
        "evidence_references": [],
    }


def _ensure_provenance_chain(record: Json) -> Any:
    """Ensure provenance_chain exists with all required subfields."""
    existing = record.get("provenance_chain")
    if isinstance(existing, dict):
        # Normalize partial chain
        return {
            "steps": existing.get("steps", []) if isinstance(existing.get("steps"), list) else [],
            "audit_trail": existing.get("audit_trail", ""),
        }

    return {
        "steps": [],
        "audit_trail": "",
    }


def _ensure_worker_metadata(record: Json) -> Any:
    """Normalize Worker metadata for KD-style records."""
    existing = record.get("metadata")
    if not isinstance(existing, dict):
        existing = {}

    role = existing.get("role") or record.get("role") or "worker"
    task_type = existing.get("task_type") or record.get(
        "task_type") or "plain_kd"
    caws_level = existing.get("caws_level") or record.get(
        "caws_level") or "none"

    return {
        **existing,
        "role": role,
        "task_type": task_type,
        "caws_level": caws_level,
    }


def _ensure_judge_debate_scores(record: Json) -> Any:
    """Guarantee debate_scores exists with well-typed numeric defaults."""
    existing = record.get("debate_scores")
    if isinstance(existing, dict):
        return existing

    def empty_side() -> Dict[str, float]:
        return {"E": 0.0, "B": 0.0, "G": 0.0, "P": 0.0, "total": 0.0}

    return {
        "a": empty_side(),
        "b": empty_side(),
    }


def _ensure_judge_winner(record: Json) -> Any:
    """Ensure we have a sane winner label. Default to 'tie' if missing."""
    winner = record.get("winner")
    if winner in ("a", "b", "tie"):
        return winner
    return "tie"


def _ensure_process_supervision(record: Json) -> Any:
    """Ensure process supervision fields exist for tool-use tasks.
    
    Per DATASET_STANDARDS_V1.md: if task_type == "tool_use", these fields must exist.
    """
    task_type = record.get("task_type") or record.get("metadata", {}).get("task_type")
    
    # Only create process supervision for tool-use tasks
    if task_type != "tool_use":
        # Return existing if present, otherwise None (will be skipped by rule)
        return record.get("process_supervision")
    
    existing = record.get("process_supervision")
    if isinstance(existing, dict):
        # Normalize partial supervision
        return {
            "tool_name_ids": existing.get("tool_name_ids", []) if isinstance(existing.get("tool_name_ids"), list) else [],
            "gold_json_text_ids": existing.get("gold_json_text_ids", []) if isinstance(existing.get("gold_json_text_ids"), list) else [],
            "integration_mask": existing.get("integration_mask", []) if isinstance(existing.get("integration_mask"), list) else [],
        }
    
    # Create empty arrays for tool-use tasks (will be populated by teacher generation)
    return {
        "tool_name_ids": [],
        "gold_json_text_ids": [],
        "integration_mask": [],
    }


def _enforce_field_dependencies(record: Json) -> None:
    """Enforce field dependencies per DATASET_STANDARDS_V1.md invariants.
    
    - If task_type == "tool_use": process_supervision must exist
    - If caws_level != "none": caws_context.working_spec must be populated
    """
    task_type = record.get("task_type") or record.get("metadata", {}).get("task_type")
    
    # Dependency: tool_use -> process_supervision
    if task_type == "tool_use":
        if "process_supervision" not in record or record["process_supervision"] is None:
            record["process_supervision"] = _ensure_process_supervision(record)
            if "gap_fill_metadata" not in record:
                record["gap_fill_metadata"] = {}
            record["gap_fill_metadata"]["process_supervision"] = {
                "method": "structural",
                "rule": "enforce_field_dependencies",
                "description": "Added process_supervision for tool_use task",
            }
    
    # Dependency: caws_level != "none" -> caws_context must be populated
    caws_level = record.get("caws_level") or record.get("metadata", {}).get("caws_level")
    if caws_level and caws_level != "none" and caws_level != 0:
        caws_context = record.get("caws_context")
        if not isinstance(caws_context, dict) or not caws_context.get("working_spec"):
            # Ensure CAWS context exists with non-default values
            record["caws_context"] = _ensure_caws_context(record)
            if "gap_fill_metadata" not in record:
                record["gap_fill_metadata"] = {}
            record["gap_fill_metadata"]["caws_context"] = {
                "method": "structural",
                "rule": "enforce_field_dependencies",
                "description": "Added caws_context for non-none caws_level",
            }


def make_default_worker_rules() -> List[GapFillRule]:
    """Default structural rules for Worker datasets."""
    return [
        GapFillRule(
            name="ensure_caws_context",
            field="caws_context",
            fn=_ensure_caws_context,
            description="Ensure caws_context.working_spec exists with sane defaults.",
        ),
        GapFillRule(
            name="ensure_evidence_manifest",
            field="evidence_manifest",
            fn=_ensure_evidence_manifest,
            description="Guarantee evidence_manifest shell exists.",
        ),
        GapFillRule(
            name="ensure_provenance_chain",
            field="provenance_chain",
            fn=_ensure_provenance_chain,
            description="Guarantee provenance_chain shell exists.",
        ),
        GapFillRule(
            name="ensure_worker_metadata",
            field="metadata",
            fn=_ensure_worker_metadata,
            description="Normalize Worker metadata (role, task_type, caws_level).",
        ),
        GapFillRule(
            name="ensure_process_supervision",
            field="process_supervision",
            fn=_ensure_process_supervision,
            apply_if_missing_only=True,
            description="Ensure process_supervision fields exist for tool-use tasks.",
        ),
    ]


def make_default_judge_rules() -> List[GapFillRule]:
    """Default structural rules for Judge datasets."""
    return [
        GapFillRule(
            name="ensure_caws_context_from_working_spec",
            field="caws_context",
            fn=_ensure_caws_context,
            description="Backfill caws_context from working_spec if missing.",
        ),
        GapFillRule(
            name="ensure_judge_debate_scores",
            field="debate_scores",
            fn=_ensure_judge_debate_scores,
            description="Ensure debate_scores for both sides.",
        ),
        GapFillRule(
            name="ensure_judge_winner",
            field="winner",
            fn=_ensure_judge_winner,
            apply_if_missing_only=True,
            description="Default winner to 'tie' if missing/invalid.",
        ),
    ]


# ---- Simple JSONL driver (optional CLI) --------------------------------


def fill_jsonl(
    in_path: str,
    out_path: str,
    role: str = "worker",
) -> None:
    import json
    from pathlib import Path

    if role == "worker":
        rules = make_default_worker_rules()
    elif role == "judge":
        rules = make_default_judge_rules()
    else:
        raise ValueError(f"Unsupported role for gap filling: {role}")

    filler = DataGapFiller(rules=rules)

    in_p = Path(in_path)
    out_p = Path(out_path)

    with in_p.open() as fin, out_p.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                # Preserve header lines as-is
                fout.write(line + "\n")
                continue
            obj = json.loads(line)
            filled = filler.fill_record(obj)
            fout.write(json.dumps(filled, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply conservative gap-filling rules to JSONL datasets."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument(
        "--role",
        choices=["worker", "judge"],
        default="worker",
        help="Dataset role; selects which ruleset to apply",
    )

    args = parser.parse_args(argv)
    fill_jsonl(args.input, args.output, role=args.role)


if __name__ == "__main__":
    main()
