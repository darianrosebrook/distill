"""
Merge all datasets into production-ready files.

Orchestrates the merging of Worker, Judge, and Drafter datasets according to
the dataset remediation plan, with validation and reporting.

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from scripts.merge_datasets import merge_datasets, load_jsonl, normalize_sample


def get_dataset_stats(file_path: Path) -> Dict[str, Any]:
    """Get statistics about a dataset file."""
    if not file_path.exists():
        return {"exists": False, "count": 0}

    samples = load_jsonl(file_path)
    # Long context: check total length (prompt + teacher_text) > 8000 chars
    # or if task_type indicates long_context
    long_context_count = sum(
        1 for s in samples
        if (
            s.get("task_type") == "long_context"
            or (len(s.get("prompt", "")) + len(s.get("teacher_text", ""))) > 8000
        )
    )
    
    stats = {
        "exists": True,
        "count": len(samples),
        "with_reasoning": sum(1 for s in samples if s.get("teacher_reasoning")),
        "with_caws": sum(1 for s in samples if s.get("caws_context") or s.get("caws_level", 0) > 0),
        "with_process_supervision": sum(
            1 for s in samples
            if s.get("tool_name_ids") or s.get("gold_json_text_ids") or s.get("integration_mask")
        ),
        "long_context": long_context_count,
    }
    return stats


def merge_worker_datasets(
    output_file: Path,
    kd_mix: Optional[Path] = None,
    contextual: Optional[Path] = None,
    caws_tool_examples: Optional[Path] = None,
    long_context: Optional[Path] = None,
    long_context_additional: Optional[Path] = None,
    deduplicate: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Merge Worker datasets into production file."""
    print("\n" + "=" * 80)
    print("MERGING WORKER DATASETS")
    print("=" * 80)

    # Default file paths
    if kd_mix is None:
        kd_mix = Path("data/kd_mix_production.jsonl")
    if contextual is None:
        contextual = Path("data/contextual_final.jsonl")
    if caws_tool_examples is None:
        caws_tool_examples = Path("data/caws_tool_examples_filled.jsonl")
    if long_context is None:
        long_context = Path("data/kd_long_context.jsonl")
    if long_context_additional is None:
        long_context_additional = Path("data/kd_long_context_additional.jsonl")

    # Collect input files that exist
    input_files = []
    file_stats = {}

    for file_path, name in [
        (kd_mix, "KD Mix Production"),
        (contextual, "Contextual Final"),
        (caws_tool_examples, "CAWS Tool Examples"),
        (long_context, "Long Context"),
        (long_context_additional, "Long Context Additional"),
    ]:
        stats = get_dataset_stats(file_path)
        file_stats[name] = stats
        if stats["exists"] and stats["count"] > 0:
            input_files.append(file_path)
            print(f"  {name}: {stats['count']} samples")
        else:
            print(f"  {name}: Not found or empty (skipping)")

    if not input_files:
        print("\nERROR: No input files found!")
        return {"success": False, "error": "No input files found"}

    print(f"\nMerging {len(input_files)} datasets...")

    if dry_run:
        print("\n[DRY RUN] Would merge the following files:")
        total_samples = 0
        for f in input_files:
            stats = get_dataset_stats(f)
            total_samples += stats["count"]
            print(f"  - {f.name}: {stats['count']} samples")
        
        print(f"\n[DRY RUN] Total samples before deduplication: {total_samples}")
        if deduplicate:
            print("[DRY RUN] Deduplication would be applied (estimated 5-15% reduction)")
            estimated_final = int(total_samples * 0.90)  # Conservative estimate
            print(f"[DRY RUN] Estimated final count: ~{estimated_final} samples")
        else:
            print("[DRY RUN] Deduplication disabled")
            print(f"[DRY RUN] Final count would be: {total_samples} samples")
        
        print(f"\n[DRY RUN] Would write to: {output_file}")
        print("[DRY RUN] No files were actually written.")
        
        # Estimate final stats based on input files
        estimated_stats = {
            "count": int(total_samples * 0.90) if deduplicate else total_samples,
            "with_reasoning": sum(
                get_dataset_stats(f)["with_reasoning"] for f in input_files
            ),
            "with_caws": sum(
                get_dataset_stats(f)["with_caws"] for f in input_files
            ),
            "with_process_supervision": sum(
                get_dataset_stats(f)["with_process_supervision"] for f in input_files
            ),
            "long_context": sum(
                get_dataset_stats(f)["long_context"] for f in input_files
            ),
        }
        final_stats = estimated_stats
    else:
        # Merge datasets
        merge_datasets(
            input_files=input_files,
            output_file=output_file,
            deduplicate=deduplicate,
            default_role="worker",
        )

        # Get final stats
        final_stats = get_dataset_stats(output_file)
    print("\n" + "-" * 80)
    print("WORKER DATASET MERGE COMPLETE")
    print("-" * 80)
    print(f"Output file: {output_file}")
    print(f"Total samples: {final_stats['count']}")
    print(f"  With reasoning: {final_stats['with_reasoning']}")
    print(f"  With CAWS context: {final_stats['with_caws']}")
    print(f"  With process supervision: {final_stats['with_process_supervision']}")
    print(f"  Long context (>8k tokens): {final_stats['long_context']}")

    return {
        "success": True,
        "output_file": str(output_file),
        "total_samples": final_stats["count"],
        "input_stats": file_stats,
        "final_stats": final_stats,
    }


def merge_judge_datasets(
    train_output: Path,
    val_output: Path,
    train_files: Optional[List[Path]] = None,
    val_files: Optional[List[Path]] = None,
    deduplicate: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Merge Judge datasets into train/val production files."""
    print("\n" + "=" * 80)
    print("MERGING JUDGE DATASETS")
    print("=" * 80)

    # If production files already exist and are sufficient, use them
    if train_output.exists() and val_output.exists():
        train_stats = get_dataset_stats(train_output)
        val_stats = get_dataset_stats(val_output)
        total = train_stats["count"] + val_stats["count"]

        if total >= 5000:
            print(f"Production files already exist with {total} samples:")
            print(f"  Train: {train_stats['count']} samples")
            print(f"  Val: {val_stats['count']} samples")
            print("Skipping merge (files already meet requirements)")
            return {
                "success": True,
                "train_file": str(train_output),
                "val_file": str(val_output),
                "train_samples": train_stats["count"],
                "val_samples": val_stats["count"],
                "total_samples": total,
                "skipped": True,
            }

    # Default file paths if not provided
    if train_files is None:
        train_files = [
            Path("data/judge/train_production.jsonl"),
            Path("data/judge/adjudication_cycle.jsonl"),
            Path("data/judge/caws_scenarios.jsonl"),
        ]
    if val_files is None:
        val_files = [
            Path("data/judge/val_production.jsonl"),
        ]

    # Collect existing files
    train_inputs = [f for f in train_files if f.exists()]
    val_inputs = [f for f in val_files if f.exists()]

    if dry_run:
        if train_inputs:
            print(f"\n[DRY RUN] Would merge {len(train_inputs)} train datasets...")
            train_total = 0
            for f in train_inputs:
                stats = get_dataset_stats(f)
                train_total += stats["count"]
                print(f"  {f.name}: {stats['count']} samples")
            print(f"[DRY RUN] Train total before deduplication: {train_total}")
            print(f"[DRY RUN] Would write to: {train_output}")
        else:
            print("\n[DRY RUN] No train input files found")

        if val_inputs:
            print(f"\n[DRY RUN] Would merge {len(val_inputs)} val datasets...")
            val_total = 0
            for f in val_inputs:
                stats = get_dataset_stats(f)
                val_total += stats["count"]
                print(f"  {f.name}: {stats['count']} samples")
            print(f"[DRY RUN] Val total before deduplication: {val_total}")
            print(f"[DRY RUN] Would write to: {val_output}")
        else:
            print("\n[DRY RUN] No val input files found")
        
        print("[DRY RUN] No files were actually written.")
        
        train_stats = {"count": int(sum(get_dataset_stats(f)["count"] for f in train_inputs) * 0.95) if deduplicate else sum(get_dataset_stats(f)["count"] for f in train_inputs)}
        val_stats = {"count": int(sum(get_dataset_stats(f)["count"] for f in val_inputs) * 0.95) if deduplicate else sum(get_dataset_stats(f)["count"] for f in val_inputs)}
    else:
        if train_inputs:
            print(f"\nMerging {len(train_inputs)} train datasets...")
            for f in train_inputs:
                stats = get_dataset_stats(f)
                print(f"  {f.name}: {stats['count']} samples")
            merge_datasets(
                input_files=train_inputs,
                output_file=train_output,
                deduplicate=deduplicate,
                default_role="judge",
            )

        if val_inputs:
            print(f"\nMerging {len(val_inputs)} val datasets...")
            for f in val_inputs:
                stats = get_dataset_stats(f)
                print(f"  {f.name}: {stats['count']} samples")
            merge_datasets(
                input_files=val_inputs,
                output_file=val_output,
                deduplicate=deduplicate,
                default_role="judge",
            )

        train_stats = get_dataset_stats(train_output)
        val_stats = get_dataset_stats(val_output)

    print("\n" + "-" * 80)
    print("JUDGE DATASET MERGE COMPLETE")
    print("-" * 80)
    print(f"Train file: {train_output}")
    print(f"  Samples: {train_stats['count']}")
    print(f"Val file: {val_output}")
    print(f"  Samples: {val_stats['count']}")
    print(f"Total: {train_stats['count'] + val_stats['count']}")

    return {
        "success": True,
        "train_file": str(train_output),
        "val_file": str(val_output),
        "train_samples": train_stats["count"],
        "val_samples": val_stats["count"],
        "total_samples": train_stats["count"] + val_stats["count"],
    }


def merge_drafter_datasets(
    output_file: Path,
    input_files: Optional[List[Path]] = None,
    deduplicate: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Merge Drafter datasets into production file."""
    print("\n" + "=" * 80)
    print("MERGING DRAFTER DATASETS")
    print("=" * 80)

    # Default file paths if not provided
    if input_files is None:
        input_files = [
            Path("data/drafter/drafter_from_kd_mix.jsonl"),
            Path("data/drafter/drafter_dataset_fixed.jsonl"),
        ]

    # Collect existing files
    existing_files = [f for f in input_files if f.exists()]

    if not existing_files:
        print("No Drafter input files found. Skipping merge.")
        return {"success": False, "error": "No input files found"}

    if dry_run:
        print(f"\n[DRY RUN] Would merge {len(existing_files)} datasets...")
        total_samples = 0
        for f in existing_files:
            stats = get_dataset_stats(f)
            total_samples += stats["count"]
            print(f"  {f.name}: {stats['count']} samples")
        
        print(f"[DRY RUN] Total samples before deduplication: {total_samples}")
        if deduplicate:
            print("[DRY RUN] Deduplication would be applied (estimated 5-15% reduction)")
            estimated_final = int(total_samples * 0.90)
            print(f"[DRY RUN] Estimated final count: ~{estimated_final} samples")
        else:
            print(f"[DRY RUN] Final count would be: {total_samples} samples")
        
        print(f"\n[DRY RUN] Would write to: {output_file}")
        print("[DRY RUN] No files were actually written.")
        
        final_stats = {"count": int(total_samples * 0.90) if deduplicate else total_samples}
    else:
        print(f"\nMerging {len(existing_files)} datasets...")
        for f in existing_files:
            stats = get_dataset_stats(f)
            print(f"  {f.name}: {stats['count']} samples")

        merge_datasets(
            input_files=existing_files,
            output_file=output_file,
            deduplicate=deduplicate,
            default_role="drafter",
        )

        final_stats = get_dataset_stats(output_file)
    print("\n" + "-" * 80)
    print("DRAFTER DATASET MERGE COMPLETE")
    print("-" * 80)
    print(f"Output file: {output_file}")
    print(f"Total samples: {final_stats['count']}")

    return {
        "success": True,
        "output_file": str(output_file),
        "total_samples": final_stats["count"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge all datasets into production-ready files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--worker-out",
        type=Path,
        default=Path("data/worker_production.jsonl"),
        help="Output path for Worker dataset (default: data/worker_production.jsonl)",
    )
    parser.add_argument(
        "--judge-train-out",
        type=Path,
        default=Path("data/judge/train_production.jsonl"),
        help="Output path for Judge train dataset (default: data/judge/train_production.jsonl)",
    )
    parser.add_argument(
        "--judge-val-out",
        type=Path,
        default=Path("data/judge/val_production.jsonl"),
        help="Output path for Judge val dataset (default: data/judge/val_production.jsonl)",
    )
    parser.add_argument(
        "--drafter-out",
        type=Path,
        default=Path("data/drafter/drafter_production.jsonl"),
        help="Output path for Drafter dataset (default: data/drafter/drafter_production.jsonl)",
    )
    parser.add_argument(
        "--skip-worker",
        action="store_true",
        help="Skip Worker dataset merge",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Judge dataset merge",
    )
    parser.add_argument(
        "--skip-drafter",
        action="store_true",
        help="Skip Drafter dataset merge",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually writing files",
    )
    args = parser.parse_args()

    results = {}

    # Merge Worker datasets
    if not args.skip_worker:
        results["worker"] = merge_worker_datasets(
            output_file=args.worker_out,
            deduplicate=not args.no_deduplicate,
            dry_run=args.dry_run,
        )

    # Merge Judge datasets
    if not args.skip_judge:
        results["judge"] = merge_judge_datasets(
            train_output=args.judge_train_out,
            val_output=args.judge_val_out,
            deduplicate=not args.no_deduplicate,
            dry_run=args.dry_run,
        )

    # Merge Drafter datasets
    if not args.skip_drafter:
        results["drafter"] = merge_drafter_datasets(
            output_file=args.drafter_out,
            deduplicate=not args.no_deduplicate,
            dry_run=args.dry_run,
        )

    # Print summary
    print("\n" + "=" * 80)
    if args.dry_run:
        print("PRODUCTION DATASET MERGE SUMMARY (DRY RUN)")
    else:
        print("PRODUCTION DATASET MERGE SUMMARY")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    if args.dry_run:
        print("Mode: DRY RUN (no files were written)")
    print()

    if "worker" in results:
        w = results["worker"]
        if w.get("success"):
            print(f"Worker: {w.get('total_samples', 0)} samples -> {w.get('output_file')}")
        else:
            print(f"Worker: FAILED - {w.get('error')}")

    if "judge" in results:
        j = results["judge"]
        if j.get("success"):
            print(f"Judge: {j.get('total_samples', 0)} samples")
            print(f"  Train: {j.get('train_samples', 0)} samples -> {j.get('train_file')}")
            print(f"  Val: {j.get('val_samples', 0)} samples -> {j.get('val_file')}")
        else:
            print(f"Judge: FAILED - {j.get('error')}")

    if "drafter" in results:
        d = results["drafter"]
        if d.get("success"):
            print(f"Drafter: {d.get('total_samples', 0)} samples -> {d.get('output_file')}")
        else:
            print(f"Drafter: FAILED - {d.get('error')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

