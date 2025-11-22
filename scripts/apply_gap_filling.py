#!/usr/bin/env python3
"""
Apply structural (and optionally predictive) gap filling to a JSONL dataset.

Usage (structural only):

    python -m scripts.apply_gap_filling \
        data/kd_mix_1500.jsonl \
        data/kd_mix_1500_v1_structural.jsonl \
        --role worker \
        --structural

Usage (structural + predictive, once models exist):

    python -m scripts.apply_gap_filling \
        data/kd_mix_legacy.jsonl \
        data/kd_mix_legacy_upgraded.jsonl \
        --role worker \
        --structural \
        --predictive \
        --predictive-config configs/data_gap_models.worker.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Structural gap filling rules
from training.data_gap_filling import (
    DataGapFiller,
    make_default_worker_rules,
    make_default_judge_rules,
)


Json = Dict[str, Any]


# --- Predictive layer integration point --------------------------------------


class PredictiveGapFiller:
    """
    Optional: apply predictive models to fill selected fields.

    This is intentionally thin. It’s a hook around whatever you implement in
    e.g. `training.data_gap_models`. For now you can stub it out or make it a
    no-op until you have models trained.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path
        self._models_loaded = False

        if config_path is not None:
            self._load_models(config_path)

    def _load_models(self, config_path: Path) -> None:
        """
        Load model artifacts / config. Replace with your own implementation.
        """
        # Example pattern (you can replace this with real logic later):
        #
        #   from training.data_gap_models import load_models_from_config
        #   self.models = load_models_from_config(config_path)
        #
        # For now this is a placeholder.
        self._models_loaded = True
        # self.models = ...

    def apply(self, record: Json) -> Json:
        """
        Apply predictive models to a single record.

        Must be safe to call repeatedly; should only modify fields explicitly
        designated as model-fillable (e.g. task_type, adjudication_stage, etc.).
        """
        if not self._models_loaded:
            # No-op if models are not loaded.
            return record

        # Pseudocode sketch for later:
        #
        #   record = fill_task_type(record, self.models.task_type_model)
        #   record = fill_adjudication_stage(record, self.models.stage_model)
        #   record = annotate_gap_fill_metadata(record, changes)
        #
        return record


# --- Stats / bookkeeping -----------------------------------------------------


@dataclass
class GapFillStats:
    total_records: int = 0
    header_lines: int = 0
    structural_applied: int = 0
    predictive_applied: int = 0
    errors: int = 0
    skipped_empty_lines: int = 0

    # You can extend this with per-field counters later if you want

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "header_lines": self.header_lines,
            "structural_applied": self.structural_applied,
            "predictive_applied": self.predictive_applied,
            "errors": self.errors,
            "skipped_empty_lines": self.skipped_empty_lines,
        }


# --- Core logic --------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterable[str]:
    with path.open() as f:
        for line in f:
            yield line.rstrip("\n")


def apply_gap_filling(
    in_path: Path,
    out_path: Path,
    role: str,
    structural: bool,
    predictive: bool,
    predictive_config: Optional[Path] = None,
    max_records: Optional[int] = None,
    dry_run: bool = False,
) -> GapFillStats:
    stats = GapFillStats()

    # Structural filler
    if role == "worker":
        structural_filler = DataGapFiller(make_default_worker_rules())
    elif role == "judge":
        structural_filler = DataGapFiller(make_default_judge_rules())
    else:
        raise ValueError(
            f"Unsupported role: {role!r} (expected 'worker' or 'judge')")

    # Predictive filler (optional)
    predictive_filler: Optional[PredictiveGapFiller] = None
    if predictive:
        predictive_filler = PredictiveGapFiller(config_path=predictive_config)

    writer = None if dry_run else out_path.open("w")

    try:
        for raw_line in iter_jsonl(in_path):
            if not raw_line.strip():
                stats.skipped_empty_lines += 1
                continue

            # Preserve header lines literally
            if raw_line.lstrip().startswith('{"__header__"'):
                if writer is not None:
                    writer.write(raw_line + "\n")
                stats.header_lines += 1
                continue

            try:
                record: Json = json.loads(raw_line)
            except json.JSONDecodeError:
                stats.errors += 1
                continue

            stats.total_records += 1

            # Structural gap filling
            if structural:
                record = structural_filler.fill_record(record)
                stats.structural_applied += 1

            # Predictive gap filling
            if predictive and predictive_filler is not None:
                before = json.dumps(record, sort_keys=True)
                record = predictive_filler.apply(record)
                after = json.dumps(record, sort_keys=True)
                if after != before:
                    stats.predictive_applied += 1

            if writer is not None:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if max_records is not None and stats.total_records >= max_records:
                break

    finally:
        if writer is not None:
            writer.close()

    return stats


# --- CLI ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply structural and predictive gap filling to JSONL datasets."
    )
    p.add_argument("input", type=Path, help="Input JSONL file")
    p.add_argument("output", type=Path, help="Output JSONL file")

    p.add_argument(
        "--role",
        choices=["worker", "judge"],
        required=True,
        help="Dataset role; selects which rule set to apply.",
    )

    p.add_argument(
        "--structural",
        action="store_true",
        help="Apply structural gap filling (CAWS shells, metadata, etc.).",
    )
    p.add_argument(
        "--predictive",
        action="store_true",
        help="Apply predictive gap filling (requires models; currently a no-op stub).",
    )
    p.add_argument(
        "--predictive-config",
        type=Path,
        default=None,
        help="Optional config for predictive models (to be defined in training.data_gap_models).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on number of records to process (for smoke testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output file; just compute stats.",
    )

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stats = apply_gap_filling(
        in_path=args.input,
        out_path=args.output,
        role=args.role,
        structural=args.structural,
        predictive=args.predictive,
        predictive_config=args.predictive_config,
        max_records=args.max_records,
        dry_run=args.dry_run,
    )

    print("=== Gap Filling Summary ===")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output} (dry_run={args.dry_run})")
    print(f"Role:    {args.role}")
    print(
        f"Options: structural={args.structural}, predictive={args.predictive}")
    print()
    for k, v in stats.as_dict().items():
        print(f"{k:22s}: {v}")


if __name__ == "__main__":
    main()
