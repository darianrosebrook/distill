"""
Merge multiple JSONL datasets into a single unified dataset.

Supports deduplication, schema validation, and field normalization.

Author: @darianrosebrook
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


def compute_sample_hash(sample: Dict[str, Any]) -> str:
    """Compute hash for deduplication based on prompt + teacher_text."""
    prompt = sample.get("prompt", "")
    teacher_text = sample.get("teacher_text", "")
    # Use first 500 chars of each to avoid issues with very long texts
    prompt_short = prompt[:500] if prompt else ""
    teacher_short = teacher_text[:500] if teacher_text else ""
    combined = f"{prompt_short}\n{teacher_short}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def normalize_sample(sample: Dict[str, Any], default_role: str = "worker") -> Dict[str, Any]:
    """Normalize sample to ensure required fields exist."""
    normalized = sample.copy()

    # Ensure id exists
    if "id" not in normalized or not normalized["id"]:
        normalized["id"] = f"merged-{hash(str(sample)) % 1000000:06d}"

    # Ensure role exists
    if "role" not in normalized:
        normalized["role"] = default_role

    # Ensure task_type exists (infer if missing)
    if "task_type" not in normalized:
        if normalized.get("tool_name_ids") or normalized.get("gold_json_text_ids"):
            normalized["task_type"] = "tool_use"
        elif len(normalized.get("prompt", "")) > 8000:
            normalized["task_type"] = "long_context"
        else:
            normalized["task_type"] = "plain_kd"

    # Ensure caws_level exists (infer if missing)
    if "caws_level" not in normalized:
        if normalized.get("evidence_manifest") and normalized.get("provenance_chain"):
            normalized["caws_level"] = 2
        elif normalized.get("caws_context"):
            normalized["caws_level"] = 1
        else:
            normalized["caws_level"] = 0

    # Ensure source exists
    if "source" not in normalized:
        normalized["source"] = "teacher_kd"

    return normalized


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"[merge_datasets] WARN: File not found: {file_path}")
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
                print(f"[merge_datasets] WARN: Invalid JSON at line {line_num} in {file_path}: {e}")
                continue

    return samples


def merge_datasets(
    input_files: List[Path],
    output_file: Path,
    deduplicate: bool = True,
    default_role: str = "worker",
) -> None:
    """Merge multiple JSONL datasets into one."""
    all_samples = []
    seen_hashes: Set[str] = set()

    print(f"[merge_datasets] Merging {len(input_files)} datasets...")

    for input_file in input_files:
        samples = load_jsonl(input_file)
        print(f"[merge_datasets] Loaded {len(samples)} samples from {input_file.name}")

        for sample in samples:
            normalized = normalize_sample(sample, default_role=default_role)

            if deduplicate:
                sample_hash = compute_sample_hash(normalized)
                if sample_hash in seen_hashes:
                    continue
                seen_hashes.add(sample_hash)

            all_samples.append(normalized)

    print(f"[merge_datasets] Total unique samples: {len(all_samples)}")

    # Write merged dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[merge_datasets] Wrote {len(all_samples)} samples to {output_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple JSONL datasets into a single unified dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files to merge")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication (keep all samples even if duplicates)",
    )
    ap.add_argument(
        "--default-role",
        default="worker",
        choices=["worker", "judge", "drafter"],
        help="Default role for samples without role field",
    )
    args = ap.parse_args()

    input_files = [Path(f) for f in args.inputs]
    output_file = Path(args.out)

    merge_datasets(
        input_files=input_files,
        output_file=output_file,
        deduplicate=not args.no_deduplicate,
        default_role=args.default_role,
    )


if __name__ == "__main__":
    main()

