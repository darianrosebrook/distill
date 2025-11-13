"""Training callback for automatic checkpoint evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import time


def run_eval_for_checkpoint(
    ckpt_dir: str,
    dataset_path: str = "data/contextual_final.jsonl",
    fixtures_dir: str = "eval/tool_broker/fixtures",
    out_dir: str = "eval/results",
    report_path: str = "eval/reports/latest.json",
    model_max_tokens: int = 8192,
    num_shards: int = 4,
    min_eligible_for_gates: int = 15,
    strict_gate_on_lax: bool = True,
    raise_on_gate_failure: bool = True,
) -> dict:
    """
    Run evaluation harness on a checkpoint with sharding.

    Args:
        ckpt_dir: Path to checkpoint directory or model
        dataset_path: Path to evaluation dataset JSONL
        fixtures_dir: Directory containing tool fixtures
        out_dir: Output directory for results
        report_path: Path to write merged report
        model_max_tokens: Maximum tokens for generation
        num_shards: Number of shards for parallel evaluation
        min_eligible_for_gates: Minimum eligible samples for gates
        strict_gate_on_lax: Whether to use strict gates (default: lax)
        raise_on_gate_failure: Raise exception if gates fail

    Returns:
        Merged report dictionary

    Raises:
        RuntimeError: If gates fail and raise_on_gate_failure=True
        FileNotFoundError: If checkpoint or dataset not found
    """
    # Validate inputs
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs("eval/reports", exist_ok=True)

    # Sharded eval for throughput
    shard_reports = []
    for shard_idx in range(num_shards):
        shard_report = f"{report_path}.shard_{shard_idx}.json"
        cmd = [
            "python",
            "-m",
            "eval.cli",
            "--runner",
            "hf_local",
            "--model",
            ckpt_dir,
            "--in",
            dataset_path,
            "--out",
            f"{out_dir}/preds_shard_{shard_idx}.jsonl",
            "--report",
            shard_report,
            "--fixtures",
            fixtures_dir,
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_idx),
            "--seed",
            "42",
            "--temperature",
            "0.0",
            "--max-tokens",
            str(model_max_tokens),
            "--min-eligible-for-gates",
            str(min_eligible_for_gates),
            "--fail-on-fingerprint-mismatch",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Evaluation failed for shard {shard_idx}: {e.stderr}") from e

        if os.path.exists(shard_report):
            shard_reports.append(shard_report)
        else:
            raise RuntimeError(f"Shard report not created: {shard_report}")

    # Merge shard reports
    reports = []
    for shard_report in shard_reports:
        with open(shard_report, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    merged = {
        "reports": reports,
        "merged_at": time.time(),
        "checkpoint": ckpt_dir,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Gate: fail CI / caller if any shard is gates_ok=false
    gates_ok = all(r.get("gates_ok", False) for r in reports)
    if not gates_ok and raise_on_gate_failure:
        failed_shards = [i for i, r in enumerate(reports) if not r.get("gates_ok", False)]
        raise RuntimeError(
            f"Evaluation gates failed for checkpoint: {ckpt_dir} (failed shards: {failed_shards})"
        )

    # Append to eval history
    history_path = "eval/reports/history.ndjson"
    with open(history_path, "a", encoding="utf-8") as f:
        for r in reports:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "ckpt": ckpt_dir,
                        "summary": r.get("summary", {}),
                        "gates_ok": r.get("gates_ok", False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return merged
