"""Validate that sharded evaluation produces observationally equivalent results to non-sharded evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import summarization for re-summarizing concatenated results
try:
    from eval.reports.summarize import summarize_results
except ImportError:
    summarize_results = None  # Fallback if not available


def stable_shard(sample_id: str, num_shards: int) -> int:
    """Derive shard assignment from stable sample_id hash."""
    h = hashlib.sha256(sample_id.encode('utf-8')).digest()
    val = int.from_bytes(h[:8], 'little')
    return val % num_shards


def synthesize_sample_id(item: Dict[str, Any]) -> str:
    """Synthesize stable sample_id from row if missing."""
    row_json = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(row_json.encode('utf-8')).hexdigest()


def get_sample_id(rec: Dict[str, Any]) -> str:
    """Extract or synthesize sample_id from result record."""
    sample_id = rec.get("sample_id")
    if not sample_id:
        # Try to synthesize from prompt or other stable fields
        stable_fields = {
            "prompt": rec.get("prompt", ""),
            "tool_trace": rec.get("tool_trace", []),
        }
        stable_json = json.dumps(
            stable_fields, sort_keys=True, ensure_ascii=False)
        sample_id = hashlib.sha256(stable_json.encode('utf-8')).hexdigest()
    return str(sample_id)


def canonicalize_args(args: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
    """
    Canonicalize tool arguments using broker's normalization logic.

    Matches the normalization used by ToolBroker for consistent comparison.

    Args:
        args: Tool arguments dict
        tool_name: Tool name (for tool-specific defaults)
    """
    if not isinstance(args, dict):
        return {}

    # Remove None-valued keys
    args = {k: v for k, v in args.items() if v is not None}

    # Normalize query fields (lowercase, collapse whitespace)
    for qk in ("q", "query"):
        if qk in args and isinstance(args[qk], str):
            args[qk] = re.sub(r"\s+", " ", args[qk].strip().lower())

    # Default top_k for web.search*
    if tool_name in ("web.search", "web.search_async") and "top_k" not in args:
        args["top_k"] = 3

    # Sort keys for stable comparison
    return dict(sorted(args.items()))


def canonicalize_grounding(grounding: Any) -> Any:
    """Canonicalize grounding information."""
    if isinstance(grounding, dict):
        return dict(sorted(grounding.items()))
    return grounding


def canonicalize_spans(spans: List[List[int]]) -> List[List[int]]:
    """Canonicalize integration spans."""
    if not spans:
        return []
    # Sort spans by start position, then end position
    return sorted(spans, key=lambda s: (s[0] if len(s) > 0 else 0, s[1] if len(s) > 1 else 0))


def norm_output(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize output record for per-example comparison.

    Canonicalizes exactly like the scorer does, allowing benign differences.
    """
    tool_trace = rec.get("tool_trace", [])

    # Extract tool calls with normalized arguments
    normalized_tools = []
    for tc in tool_trace:
        tool_name = tc.get("name", "")
        args = tc.get("arguments", {})
        norm_args = canonicalize_args(args, tool_name=tool_name)

        normalized_tools.append({
            "name": tool_name,
            "arguments": norm_args,
            "result": tc.get("result", {}),
        })

    # Sort tools by name, then arguments (for deterministic ordering)
    normalized_tools.sort(key=lambda tc: (
        tc.get("name", ""),
        json.dumps(tc.get("arguments", {}), sort_keys=True)
    ))

    # Extract integration spans from model_output
    model_output = rec.get("model_output", "")
    integration_spans = []
    if model_output:
        for m in re.finditer(r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)', model_output, flags=re.UNICODE):
            integration_spans.append([m.start(1), m.end(1)])

    return {
        "sample_id": get_sample_id(rec),
        "tool_trace": normalized_tools,
        "integration_spans": canonicalize_spans(integration_spans),
        "scores": rec.get("scores", {}),
    }


def close(a: float, b: float, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Check if two floats are close using dual tolerance (absolute + relative)."""
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def get_deterministic_env() -> Dict[str, str]:
    """Get deterministic environment variables for subprocess execution."""
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    return env


def run_eval(
    dataset_path: str,
    model: str,
    runner: str,
    fixtures_dir: str,
    output_results: str,
    output_report: str,
    temp_dir: str,
    num_shards: int = 1,
    shard_index: int = 0,
    seed: int = 42,
    prompt_wrapper: Optional[str] = None,
) -> None:
    """
    Run evaluation with deterministic environment.

    Args:
        dataset_path: Path to input dataset
        model: Model path or name
        runner: Runner type (hf_local or openai_http)
        fixtures_dir: Fixtures directory
        output_results: Path to write results JSONL
        output_report: Path to write report JSON
        num_shards: Number of shards (1 for baseline)
        shard_index: Shard index (0 for baseline)
        seed: Random seed
        temp_dir: Temporary directory for intermediate files
        prompt_wrapper: Optional prompt wrapper template path
    """
    env = get_deterministic_env()

    cmd = [
        sys.executable, "-m", "eval.cli",
        "--runner", runner,
        "--model", model,
        "--in", dataset_path,
        "--out", output_results,
        "--report", output_report,
        "--fixtures", fixtures_dir,
        "--num-shards", str(num_shards),
        "--shard-index", str(shard_index),
        "--seed", str(seed),
        "--temperature", "0.0",
        "--min-eligible-for-gates", "15",
        "--fail-on-fingerprint-mismatch",
    ]

    if prompt_wrapper:
        cmd.extend(["--prompt-wrapper", prompt_wrapper])

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed (shard {shard_index}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def load_report(path: str) -> Dict[str, Any]:
    """Load report JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fingerprints_from_report(report: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract all fingerprints from report header and first result.

    Fingerprints may be in header (dataset, registry, tokenizer) or in results (runner, model).
    """
    header = report.get("header", {})
    config = header.get("config", {})

    fingerprints = {
        "dataset_sha256": header.get("dataset_sha256"),
        "tool_registry_sha256": header.get("tool_registry_sha256"),
        "tokenizer_fingerprint": header.get("tokenizer_fingerprint"),
    }

    # Extract prompt_wrapper_sha256 from runner fingerprint if present
    if results:
        first_result = results[0]
        runner_fp = first_result.get("runner_fingerprint", {})
        model_fp = first_result.get("model_fingerprint", {})

        fingerprints["runner_fingerprint"] = runner_fp
        fingerprints["model_fingerprint"] = model_fp

        # Extract prompt_wrapper_sha256 from runner fingerprint
        if isinstance(runner_fp, dict):
            fingerprints["prompt_wrapper_sha256"] = runner_fp.get(
                "prompt_wrapper_sha256")

    return fingerprints


def compare_fingerprints(
    baseline_report: Dict[str, Any],
    baseline_results: List[Dict[str, Any]],
    shard_reports: List[Dict[str, Any]],
    shard_results_list: List[List[Dict[str, Any]]],
) -> Tuple[bool, List[str]]:
    """
    Compare fingerprints across baseline and all shards.

    Returns:
        (all_match, mismatches) tuple
    """
    baseline_fps = extract_fingerprints_from_report(
        baseline_report, baseline_results)
    mismatches = []

    fingerprint_fields = [
        "dataset_sha256",
        "tool_registry_sha256",
        "tokenizer_fingerprint",
        "prompt_wrapper_sha256",
        "runner_fingerprint",
        "model_fingerprint",
    ]

    for shard_idx, (shard_report, shard_results) in enumerate(zip(shard_reports, shard_results_list)):
        shard_fps = extract_fingerprints_from_report(
            shard_report, shard_results)

        for field in fingerprint_fields:
            baseline_val = baseline_fps.get(field)
            shard_val = shard_fps.get(field)

            # Skip if both are None
            if baseline_val is None and shard_val is None:
                continue

            # Handle dict comparison (e.g., tokenizer_fingerprint, runner_fingerprint)
            if isinstance(baseline_val, dict) and isinstance(shard_val, dict):
                if baseline_val != shard_val:
                    mismatches.append(
                        f"Shard {shard_idx}: {field} mismatch\n"
                        f"  Baseline: {json.dumps(baseline_val, sort_keys=True)}\n"
                        f"  Shard:    {json.dumps(shard_val, sort_keys=True)}"
                    )
            elif baseline_val != shard_val:
                mismatches.append(
                    f"Shard {shard_idx}: {field} mismatch\n"
                    f"  Baseline: {baseline_val}\n"
                    f"  Shard:    {shard_val}"
                )

    return len(mismatches) == 0, mismatches


def validate_shard_completeness(
    baseline_results: List[Dict[str, Any]],
    shard_results_list: List[List[Dict[str, Any]]],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate shard completeness and uniqueness.

    Returns:
        (is_valid, diagnostics) tuple
    """
    baseline_ids = {get_sample_id(r) for r in baseline_results}

    shard_ids_list = []
    all_shard_ids = set()
    duplicates = []

    for shard_idx, shard_results in enumerate(shard_results_list):
        shard_ids = {get_sample_id(r) for r in shard_results}
        shard_ids_list.append(shard_ids)

        # Check for duplicates within this shard
        shard_id_counts = Counter(get_sample_id(r) for r in shard_results)
        shard_dupes = [sid for sid,
                       count in shard_id_counts.items() if count > 1]
        if shard_dupes:
            duplicates.extend([(shard_idx, sid) for sid in shard_dupes])

        all_shard_ids.update(shard_ids)

    # Check for duplicates across shards
    for i in range(len(shard_ids_list)):
        for j in range(i + 1, len(shard_ids_list)):
            intersection = shard_ids_list[i] & shard_ids_list[j]
            if intersection:
                duplicates.extend([(i, j, sid) for sid in intersection])

    missing = baseline_ids - all_shard_ids
    extra = all_shard_ids - baseline_ids

    diagnostics = {
        "baseline_count": len(baseline_ids),
        "shard_counts": [len(ids) for ids in shard_ids_list],
        "total_shard_count": len(all_shard_ids),
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra)),
        "duplicates": duplicates,
        "coverage_ok": len(missing) == 0 and len(extra) == 0 and len(duplicates) == 0,
    }

    is_valid = diagnostics["coverage_ok"]
    return is_valid, diagnostics


def compare_per_example(
    baseline_results: List[Dict[str, Any]],
    sharded_results: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Compare per-example normalized outputs.

    Returns:
        (mismatch_count, mismatch_details) tuple
    """
    baseline_by_id = {get_sample_id(r): norm_output(r)
                      for r in baseline_results}
    sharded_by_id = {get_sample_id(r): norm_output(r) for r in sharded_results}

    mismatches = []

    # Check all baseline samples exist in sharded
    for sample_id, baseline_norm in baseline_by_id.items():
        if sample_id not in sharded_by_id:
            mismatches.append({
                "sample_id": sample_id,
                "reason": "missing_in_sharded",
            })
            continue

        sharded_norm = sharded_by_id[sample_id]

        # Compare normalized outputs
        if baseline_norm != sharded_norm:
            mismatches.append({
                "sample_id": sample_id,
                "reason": "output_mismatch",
                "baseline": baseline_norm,
                "sharded": sharded_norm,
            })

    # Check for extra samples in sharded
    for sample_id in sharded_by_id:
        if sample_id not in baseline_by_id:
            mismatches.append({
                "sample_id": sample_id,
                "reason": "extra_in_sharded",
            })

    return len(mismatches), mismatches


def compare_metrics(
    baseline_report: Dict[str, Any],
    sharded_report: Dict[str, Any],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare metrics between baseline and sharded reports.

    Returns:
        (all_match, metric_diffs) tuple
    """
    baseline_summary = baseline_report.get("summary", {})
    sharded_summary = sharded_report.get("summary", {})

    # Metric fields with (field_name, exact_match_required)
    # Note: Some fields may not exist in all reports
    # We compare them if present, but don't fail if missing
    FIELDS = [
        ("avg_integration_f1_macro_lax", False),
        ("avg_integration_f1_macro_strict", False),
        ("avg_integration_f1_micro_lax", False),
        ("avg_integration_f1_micro_strict", False),
        ("num_eligible", True),  # exact (count)
        ("controls_with_integration", True),  # exact (must be 0)
        ("privacy_ok_rate", False),
        ("integration_span_count_histogram", True),  # exact match (dict)
        ("multi_call_parity_rate", False),
        ("json_args_valid_rate", False),
    ]

    metric_diffs = {}
    all_match = True

    for field, exact in FIELDS:
        baseline_val = baseline_summary.get(field)
        sharded_val = sharded_summary.get(field)

        if baseline_val is None or sharded_val is None:
            if baseline_val != sharded_val:
                all_match = False
                metric_diffs[field] = {
                    "baseline": baseline_val,
                    "sharded": sharded_val,
                    "delta": None,
                    "reason": "missing",
                }
            continue

        if exact:
            # For exact matches, handle dict/list comparison
            if isinstance(baseline_val, dict) and isinstance(sharded_val, dict):
                match = (baseline_val == sharded_val)
            elif isinstance(baseline_val, list) and isinstance(sharded_val, list):
                match = (baseline_val == sharded_val)
            else:
                match = (baseline_val == sharded_val)
        else:
            if isinstance(baseline_val, (int, float)) and isinstance(sharded_val, (int, float)):
                match = close(float(baseline_val), float(
                    sharded_val), atol=atol, rtol=rtol)
            else:
                match = (baseline_val == sharded_val)

        if not match:
            all_match = False
            delta = None
            if isinstance(baseline_val, (int, float)) and isinstance(sharded_val, (int, float)):
                delta = float(sharded_val) - float(baseline_val)

            metric_diffs[field] = {
                "baseline": baseline_val,
                "sharded": sharded_val,
                "delta": delta,
                "exact": exact,
            }

    # Compare per_tool_deltas
    baseline_per_tool = baseline_summary.get("per_tool_deltas", {})
    sharded_per_tool = sharded_summary.get("per_tool_deltas", {})

    if baseline_per_tool != sharded_per_tool:
        all_match = False
        metric_diffs["per_tool_deltas"] = {
            "baseline": baseline_per_tool,
            "sharded": sharded_per_tool,
            "delta": None,
            "reason": "dict_mismatch",
        }

    return all_match, metric_diffs


def main() -> int:
    """Main validation entry point."""
    ap = argparse.ArgumentParser(
        description="Validate sharding determinism for evaluation harness"
    )
    ap.add_argument("--dataset", required=True,
                    help="Path to evaluation dataset JSONL")
    ap.add_argument("--model", required=True, help="Model path or name")
    ap.add_argument("--runner", required=True,
                    choices=["hf_local", "openai_http"])
    ap.add_argument("--fixtures", required=True, help="Fixtures directory")
    ap.add_argument("--num-shards", type=int, default=4,
                    help="Number of shards (default: 4)")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="Absolute tolerance (default: 1e-6)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="Relative tolerance (default: 1e-6)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--temp-dir", default=None,
                    help="Temporary directory (default: system temp)")
    ap.add_argument("--output", default="eval/reports/sharding_validation.json",
                    help="Output validation report path")
    ap.add_argument("--prompt-wrapper", default=None,
                    help="Optional prompt wrapper template path")

    args = ap.parse_args()

    # Setup temp directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="sharding_validation_"))

    try:
        # 1. Run baseline evaluation (no sharding)
        print(f"[VALIDATION] Running baseline evaluation...")
        baseline_results_path = temp_dir / "results.baseline.jsonl"
        baseline_report_path = temp_dir / "report.baseline.json"

        run_eval(
            dataset_path=args.dataset,
            model=args.model,
            runner=args.runner,
            fixtures_dir=args.fixtures,
            output_results=str(baseline_results_path),
            output_report=str(baseline_report_path),
            temp_dir=str(temp_dir),
            num_shards=1,
            shard_index=0,
            seed=args.seed,
            prompt_wrapper=args.prompt_wrapper,
        )

        baseline_results = load_jsonl(str(baseline_results_path))
        baseline_report = load_report(str(baseline_report_path))

        print(f"[VALIDATION] Baseline: {len(baseline_results)} results")

        # 2. Run sharded evaluation
        print(
            f"[VALIDATION] Running sharded evaluation ({args.num_shards} shards)...")
        shard_results_list = []
        shard_reports_list = []

        for shard_idx in range(args.num_shards):
            shard_results_path = temp_dir / f"results.shard_{shard_idx}.jsonl"
            shard_report_path = temp_dir / f"report.shard_{shard_idx}.json"

            run_eval(
                dataset_path=args.dataset,
                model=args.model,
                runner=args.runner,
                fixtures_dir=args.fixtures,
                output_results=str(shard_results_path),
                output_report=str(shard_report_path),
                temp_dir=str(temp_dir),
                num_shards=args.num_shards,
                shard_index=shard_idx,
                seed=args.seed,
                prompt_wrapper=args.prompt_wrapper,
            )

            shard_results = load_jsonl(str(shard_results_path))
            shard_report = load_report(str(shard_report_path))

            shard_results_list.append(shard_results)
            shard_reports_list.append(shard_report)

            print(
                f"[VALIDATION] Shard {shard_idx}: {len(shard_results)} results")

        # 3. Concatenate and sort shard results
        print(f"[VALIDATION] Concatenating and sorting shard results...")
        all_sharded_results = []
        for shard_results in shard_results_list:
            all_sharded_results.extend(shard_results)

        # Sort by sample_id (stable sort)
        all_sharded_results.sort(key=lambda r: (
            get_sample_id(r), r.get("tool_trace", [])))

        # 4. Re-summarize concatenated results
        print(f"[VALIDATION] Re-summarizing concatenated results...")
        if summarize_results is None:
            print(
                "[VALIDATION] WARNING: summarize_results not available, skipping metric comparison")
            sharded_report = None
        else:
            baseline_header = baseline_report.get("header", {})
            dataset_sha256 = baseline_header.get("dataset_sha256", "")
            tool_registry_sha256 = baseline_header.get("tool_registry_sha256")
            tokenizer_fingerprint = baseline_header.get(
                "tokenizer_fingerprint")

            sharded_report = summarize_results(
                results=all_sharded_results,
                report_version="1.0.0",
                dataset_header=None,  # Use from baseline
                dataset_sha256=dataset_sha256,
                tool_registry_sha256=tool_registry_sha256,
                tokenizer_fingerprint=tokenizer_fingerprint,
                config={
                    "runner": args.runner,
                    "model": args.model,
                    "seed": args.seed,
                    "temperature": 0.0,
                    "num_shards": args.num_shards,
                    "shard_index": None,  # Merged
                },
                wall_time_sec=0.0,  # Not compared
                gates_overrides={"min_eligible_for_gates": 15},
            )

        # 5. Validate fingerprints
        print(f"[VALIDATION] Validating fingerprints...")
        fingerprints_ok, fingerprint_mismatches = compare_fingerprints(
            baseline_report, baseline_results, shard_reports_list, shard_results_list
        )

        if not fingerprints_ok:
            print("[VALIDATION] FAILED: Fingerprint mismatches detected")
            for mismatch in fingerprint_mismatches:
                print(f"  {mismatch}")
            return 1

        print("[VALIDATION] Fingerprints match")

        # 6. Validate shard completeness
        print(f"[VALIDATION] Validating shard completeness and uniqueness...")
        completeness_ok, completeness_diag = validate_shard_completeness(
            baseline_results, shard_results_list
        )

        if not completeness_ok:
            print("[VALIDATION] FAILED: Shard completeness issues")
            print(f"  Missing: {len(completeness_diag['missing'])} samples")
            print(f"  Extra: {len(completeness_diag['extra'])} samples")
            print(f"  Duplicates: {len(completeness_diag['duplicates'])}")
            return 1

        print("[VALIDATION] Shard completeness OK")

        # 7. Compare per-example outputs
        print(f"[VALIDATION] Comparing per-example outputs...")
        mismatch_count, mismatch_details = compare_per_example(
            baseline_results, all_sharded_results
        )

        if mismatch_count > 0:
            print(
                f"[VALIDATION] FAILED: {mismatch_count} per-example mismatches")
            for detail in mismatch_details[:10]:  # Show first 10
                print(f"  Sample {detail['sample_id']}: {detail['reason']}")
            if len(mismatch_details) > 10:
                print(f"  ... and {len(mismatch_details) - 10} more")
            return 1

        print("[VALIDATION] Per-example outputs match")

        # 8. Compare metrics
        if sharded_report is None:
            print(
                "[VALIDATION] WARNING: Skipping metric comparison (summarize_results unavailable)")
            metrics_ok = True
            metric_diffs = {}
        else:
            print(f"[VALIDATION] Comparing metrics...")
            metrics_ok, metric_diffs = compare_metrics(
                baseline_report, sharded_report, atol=args.atol, rtol=args.rtol
            )

            if not metrics_ok:
                print("[VALIDATION] FAILED: Metric differences detected")
                for field, diff in metric_diffs.items():
                    print(
                        f"  {field}: baseline={diff['baseline']}, sharded={diff['sharded']}, delta={diff.get('delta')}")
            else:
                print("[VALIDATION] Metrics match")

        # 9. Check governance (gates_ok)
        gates_ok = baseline_report.get("gates_ok", True)
        for shard_idx, shard_report in enumerate(shard_reports_list):
            if not shard_report.get("gates_ok", True):
                print(f"[VALIDATION] FAILED: Shard {shard_idx} gates failed")
                gates_ok = False

        # 10. Generate validation report
        # Extract fingerprints for report
        baseline_fps = extract_fingerprints_from_report(
            baseline_report, baseline_results)

        validation_report = {
            "ok": fingerprints_ok and completeness_ok and (mismatch_count == 0) and metrics_ok and gates_ok,
            "num_shards": args.num_shards,
            "dataset_sha256": baseline_fps.get("dataset_sha256"),
            "runner_fingerprint": baseline_fps.get("runner_fingerprint"),
            "model_fingerprint": baseline_fps.get("model_fingerprint"),
            "per_example_mismatches": mismatch_count,
            "metric_diffs": metric_diffs,
            "shard_sizes": completeness_diag["shard_counts"],
            "duplicates": completeness_diag["duplicates"],
            "missing": completeness_diag["missing"],
            "fingerprints_match": fingerprints_ok,
            "coverage_ok": completeness_ok,
            "metrics_match": metrics_ok,
            "gates_ok": gates_ok,
        }

        # Write validation report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)

        print(f"[VALIDATION] Report written to: {output_path}")

        if validation_report["ok"]:
            print("[VALIDATION] PASSED: All checks passed")
            return 0
        else:
            print("[VALIDATION] FAILED: One or more checks failed")
            return 1

    finally:
        # Cleanup temp directory if we created it
        if args.temp_dir is None and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    sys.exit(main())
