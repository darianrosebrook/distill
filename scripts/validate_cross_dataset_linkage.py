#!/usr/bin/env python3
"""
Validate cross-dataset consistency between Worker and Judge datasets.

This script ensures:
- Worker outputs can be converted to Judge inputs
- Working spec IDs appear consistently across datasets
- Provenance is shared correctly
- Schema consistency between related samples

Author: @darianrosebrook
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"  ❌ ERROR: File not found: {file_path}")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                continue
            try:
                sample = json.loads(line)
                sample["_line_num"] = line_num
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  ❌ ERROR: Invalid JSON at line {line_num}: {e}")
                return []

    return samples


def extract_working_spec_id(sample: Dict[str, Any]) -> Optional[str]:
    """Extract working spec ID from a sample."""
    # Try multiple paths
    ws_id = sample.get("working_spec", {}).get("id") if isinstance(sample.get("working_spec"), dict) else None
    if ws_id:
        return ws_id
    
    caws_context = sample.get("caws_context")
    if isinstance(caws_context, dict):
        ws = caws_context.get("working_spec")
        if isinstance(ws, dict):
            ws_id = ws.get("id")
            if ws_id:
                return ws_id
    
    return None


def extract_provenance_hash(sample: Dict[str, Any]) -> Optional[str]:
    """Extract provenance hash from a sample."""
    provenance = sample.get("provenance")
    if isinstance(provenance, dict):
        return provenance.get("hash")
    return None


def validate_worker_judge_linkage(
    worker_samples: List[Dict[str, Any]],
    judge_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate linkage between Worker and Judge datasets.
    
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Extract working spec IDs from both datasets
    worker_ws_ids: Set[str] = set()
    judge_ws_ids: Set[str] = set()
    
    worker_ws_to_samples: Dict[str, List[int]] = defaultdict(list)
    judge_ws_to_samples: Dict[str, List[int]] = defaultdict(list)
    
    for idx, worker_sample in enumerate(worker_samples):
        ws_id = extract_working_spec_id(worker_sample)
        if ws_id:
            worker_ws_ids.add(ws_id)
            worker_ws_to_samples[ws_id].append(idx)
    
    for idx, judge_sample in enumerate(judge_samples):
        ws_id = extract_working_spec_id(judge_sample)
        if ws_id:
            judge_ws_ids.add(ws_id)
            judge_ws_to_samples[ws_id].append(idx)
    
    # Check for shared working spec IDs
    shared_ws_ids = worker_ws_ids & judge_ws_ids
    worker_only_ws_ids = worker_ws_ids - judge_ws_ids
    judge_only_ws_ids = judge_ws_ids - worker_ws_ids
    
    results["stats"]["worker_ws_ids"] = len(worker_ws_ids)
    results["stats"]["judge_ws_ids"] = len(judge_ws_ids)
    results["stats"]["shared_ws_ids"] = len(shared_ws_ids)
    results["stats"]["worker_only_ws_ids"] = len(worker_only_ws_ids)
    results["stats"]["judge_only_ws_ids"] = len(judge_only_ws_ids)
    
    # Warn if no shared IDs (might be okay if datasets are independent)
    if len(shared_ws_ids) == 0:
        results["warnings"].append(
            "No shared working spec IDs between Worker and Judge datasets. "
            "This might be expected if datasets are independent."
        )
    
    # Check provenance hash consistency
    worker_hashes: Set[str] = set()
    judge_hashes: Set[str] = set()
    
    for worker_sample in worker_samples:
        hash_val = extract_provenance_hash(worker_sample)
        if hash_val:
            worker_hashes.add(hash_val)
    
    for judge_sample in judge_samples:
        hash_val = extract_provenance_hash(judge_sample)
        if hash_val:
            judge_hashes.add(hash_val)
    
    shared_hashes = worker_hashes & judge_hashes
    results["stats"]["shared_provenance_hashes"] = len(shared_hashes)
    
    # Validate schema consistency for shared working specs
    schema_errors = []
    for ws_id in shared_ws_ids:
        # Get samples with this working spec ID
        worker_idxs = worker_ws_to_samples[ws_id]
        judge_idxs = judge_ws_to_samples[ws_id]
        
        # Check that working spec structure is consistent
        worker_sample = worker_samples[worker_idxs[0]]
        judge_sample = judge_samples[judge_idxs[0]]
        
        worker_ws = (
            worker_sample.get("caws_context", {}).get("working_spec")
            if isinstance(worker_sample.get("caws_context"), dict)
            else worker_sample.get("working_spec")
        )
        judge_ws = judge_sample.get("working_spec")
        
        if isinstance(worker_ws, dict) and isinstance(judge_ws, dict):
            # Check key fields match
            for field in ["id", "risk_tier"]:
                worker_val = worker_ws.get(field)
                judge_val = judge_ws.get(field)
                if worker_val != judge_val:
                    schema_errors.append(
                        f"Working spec {ws_id}: {field} mismatch "
                        f"(Worker: {worker_val}, Judge: {judge_val})"
                    )
    
    results["errors"].extend(schema_errors)
    
    # Check that Judge samples can reference Worker outputs
    # (This is a heuristic check - Judge samples might reference Worker outputs
    #  in their "a" or "b" text fields)
    worker_output_refs = 0
    for judge_sample in judge_samples:
        a_text = judge_sample.get("a", {}).get("text", "") if isinstance(judge_sample.get("a"), dict) else ""
        b_text = judge_sample.get("b", {}).get("text", "") if isinstance(judge_sample.get("b"), dict) else ""
        
        # Check if text references worker-like patterns
        # (This is a simple heuristic)
        if "worker" in a_text.lower() or "worker" in b_text.lower():
            worker_output_refs += 1
    
    results["stats"]["judge_samples_with_worker_refs"] = worker_output_refs
    
    return results


def print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results."""
    print("\n" + "=" * 80)
    print("CROSS-DATASET LINKAGE VALIDATION")
    print("=" * 80)
    
    print("\nStatistics:")
    for key, value in results["stats"].items():
        print(f"  {key}: {value}")
    
    if results["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    if results["errors"]:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results["errors"][:20]:  # Show first 20
            print(f"  - {error}")
        if len(results["errors"]) > 20:
            print(f"  ... and {len(results['errors']) - 20} more errors")
    
    if not results["errors"]:
        print("\n✅ VALIDATION PASSED")
    else:
        print("\n❌ VALIDATION FAILED")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate cross-dataset consistency between Worker and Judge datasets"
    )
    parser.add_argument(
        "--worker",
        type=Path,
        required=True,
        help="Path to Worker dataset JSONL file",
    )
    parser.add_argument(
        "--judge",
        type=Path,
        required=True,
        help="Path to Judge dataset JSONL file",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with error code if warnings are present",
    )
    
    args = parser.parse_args()
    
    if not args.worker.exists():
        print(f"❌ ERROR: Worker dataset not found: {args.worker}")
        return 1
    
    if not args.judge.exists():
        print(f"❌ ERROR: Judge dataset not found: {args.judge}")
        return 1
    
    print(f"Loading Worker dataset: {args.worker}")
    worker_samples = load_jsonl(args.worker)
    print(f"  Loaded {len(worker_samples)} samples")
    
    print(f"Loading Judge dataset: {args.judge}")
    judge_samples = load_jsonl(args.judge)
    print(f"  Loaded {len(judge_samples)} samples")
    
    results = validate_worker_judge_linkage(worker_samples, judge_samples)
    print_validation_results(results)
    
    if results["errors"]:
        return 1
    
    if args.fail_on_warnings and results["warnings"]:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


