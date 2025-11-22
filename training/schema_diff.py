# training/schema_diff.py
# python -m training.schema_diff data/kd_mix_1500.jsonl data/worker_combined_v2.jsonl > docs/schema_diff_worker.md
"""Schema differencer – training/schema_diff.py

A small utility that:

Infers a structural schema from JSONL samples (types, requiredness, examples)

Compares two inferred schemas (old vs new)

Prints a human-readable diff (and can be imported programmatically)"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


Json = Dict[str, Any]


def _type_tag(value: Any) -> str:
    """Coarse type tagging; you can refine later if needed."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


@dataclass
class FieldSummary:
    name: str
    types: Set[str] = field(default_factory=set)
    present_count: int = 0
    non_null_count: int = 0
    example_values: List[Any] = field(default_factory=list)

    def update(self, value_present: bool, value: Any) -> None:
        if value_present:
            self.present_count += 1
            if value is not None:
                self.non_null_count += 1
                self.types.add(_type_tag(value))
                # Keep up to 3 distinct examples
                if len(self.example_values) < 3:
                    self.example_values.append(value)

    def required_fraction(self, total_samples: int) -> float:
        return self.present_count / total_samples if total_samples else 0.0


@dataclass
class SchemaSummary:
    fields: Dict[str, FieldSummary]
    num_samples: int


@dataclass
class FieldDiff:
    name: str
    in_old: bool
    in_new: bool
    types_old: Optional[Set[str]]
    types_new: Optional[Set[str]]
    required_old: Optional[float]
    required_new: Optional[float]


@dataclass
class SchemaDiff:
    added_fields: List[FieldDiff] = field(default_factory=list)
    removed_fields: List[FieldDiff] = field(default_factory=list)
    changed_fields: List[FieldDiff] = field(default_factory=list)


def _load_jsonl(path: Path, max_samples: Optional[int] = None) -> Iterable[Json]:
    count = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                # Skip metadata headers if present
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def infer_schema_from_jsonl(path: Path, max_samples: int = 2000) -> SchemaSummary:
    fields: Dict[str, FieldSummary] = {}
    num_samples = 0

    for sample in _load_jsonl(path, max_samples=max_samples):
        num_samples += 1
        if not isinstance(sample, dict):
            continue

        # Collect key set per record
        keys = set(sample.keys())
        for key in keys:
            if key not in fields:
                fields[key] = FieldSummary(name=key)
            value = sample.get(key, None)
            fields[key].update(True, value)

        # Also track keys that are globally known but missing in this record
        for key, summary in fields.items():
            if key not in keys:
                summary.update(False, None)

    return SchemaSummary(fields=fields, num_samples=num_samples)


def compare_summaries(old: SchemaSummary, new: SchemaSummary) -> SchemaDiff:
    diff = SchemaDiff()
    old_keys = set(old.fields.keys())
    new_keys = set(new.fields.keys())

    for name in sorted(new_keys - old_keys):
        fs_new = new.fields[name]
        diff.added_fields.append(
            FieldDiff(
                name=name,
                in_old=False,
                in_new=True,
                types_old=None,
                types_new=set(fs_new.types),
                required_old=None,
                required_new=fs_new.required_fraction(new.num_samples),
            )
        )

    for name in sorted(old_keys - new_keys):
        fs_old = old.fields[name]
        diff.removed_fields.append(
            FieldDiff(
                name=name,
                in_old=True,
                in_new=False,
                types_old=set(fs_old.types),
                types_new=None,
                required_old=fs_old.required_fraction(old.num_samples),
                required_new=None,
            )
        )

    for name in sorted(old_keys & new_keys):
        fs_old = old.fields[name]
        fs_new = new.fields[name]
        types_old = set(fs_old.types)
        types_new = set(fs_new.types)
        req_old = fs_old.required_fraction(old.num_samples)
        req_new = fs_new.required_fraction(new.num_samples)

        if types_old != types_new or abs(req_old - req_new) > 0.05:
            diff.changed_fields.append(
                FieldDiff(
                    name=name,
                    in_old=True,
                    in_new=True,
                    types_old=types_old,
                    types_new=types_new,
                    required_old=req_old,
                    required_new=req_new,
                )
            )

    return diff


def _format_fielddiff(fd: FieldDiff) -> str:
    def fmt_types(ts: Optional[Set[str]]) -> str:
        if ts is None:
            return "-"
        return ", ".join(sorted(ts)) if ts else "(none)"

    def fmt_req(val: Optional[float]) -> str:
        if val is None:
            return "-"
        return f"{val:.2f}"

    return (
        f"- `{fd.name}` "
        f"(old: types=[{fmt_types(fd.types_old)}], req={fmt_req(fd.required_old)}; "
        f"new: types=[{fmt_types(fd.types_new)}], req={fmt_req(fd.required_new)})"
    )


def print_markdown_diff(old_path: Path, new_path: Path) -> None:
    old_schema = infer_schema_from_jsonl(old_path)
    new_schema = infer_schema_from_jsonl(new_path)
    diff = compare_summaries(old_schema, new_schema)

    print(f"# Schema Diff\n")
    print(f"- Old: `{old_path}` ({old_schema.num_samples} samples)")
    print(f"- New: `{new_path}` ({new_schema.num_samples} samples)\n")

    if diff.added_fields:
        print("## Added fields\n")
        for fd in diff.added_fields:
            print(_format_fielddiff(fd))
        print()

    if diff.removed_fields:
        print("## Removed fields\n")
        for fd in diff.removed_fields:
            print(_format_fielddiff(fd))
        print()

    if diff.changed_fields:
        print("## Changed fields\n")
        for fd in diff.changed_fields:
            print(_format_fielddiff(fd))
        print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Infer and compare JSONL schemas (old vs new)."
    )
    parser.add_argument("old", type=Path, help="Path to old JSONL dataset")
    parser.add_argument("new", type=Path, help="Path to new JSONL dataset")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max samples to inspect per dataset (default: 2000)",
    )
    args = parser.parse_args(argv)

    print_markdown_diff(args.old, args.new)


if __name__ == "__main__":
    main()
