#!/usr/bin/env python3
"""Extract minimal signals for gap analysis.

Author: @darianrosebrook
"""
import json
import os
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def sha256_file(path: str) -> Optional[str]:
    """Compute SHA256 of file."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def run_cmd(cmd: str) -> str:
    """Run command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr.strip()}"


def main() -> None:
    """Extract minimal signals."""
    signals: Dict[str, Any] = {}

    # 1. Latest eval report (merged)
    latest_report = "eval/reports/latest.json"
    if os.path.exists(latest_report):
        with open(latest_report, "r", encoding="utf-8") as f:
            signals["latest_eval_report"] = json.load(f)
    else:
        signals["latest_eval_report"] = None
        signals["latest_eval_report_note"] = "File does not exist. Run eval to generate."

    # 2. Dataset verification report
    verify_report = "eval/reports/verify_dataset.json"
    if os.path.exists(verify_report):
        with open(verify_report, "r", encoding="utf-8") as f:
            signals["dataset_verification"] = json.load(f)
    else:
        # Try alternative location
        alt_verify = "data/contextual_verification_report.json"
        if os.path.exists(alt_verify):
            with open(alt_verify, "r", encoding="utf-8") as f:
                signals["dataset_verification"] = json.load(f)
        else:
            signals["dataset_verification"] = None
            signals["dataset_verification_note"] = "Run 'make contextual-verify' to generate"

    # 3. Fixture coverage summary (from latest report or smoke)
    smoke_report = "eval/reports/smoke.json"
    if os.path.exists(smoke_report):
        with open(smoke_report, "r", encoding="utf-8") as f:
            smoke_data = json.load(f)
            signals["smoke_report"] = {
                "gates_ok": smoke_data.get("gates_ok"),
                "summary": smoke_data.get("summary", {}),
                "header": smoke_data.get("header", {}),
            }
    elif signals.get("latest_eval_report"):
        # Extract from latest report
        latest = signals["latest_eval_report"]
        signals["fixture_coverage"] = {
            "gates_ok": latest.get("gates_ok"),
            "summary": latest.get("summary", {}),
            "header": latest.get("header", {}),
        }
    else:
        signals["fixture_coverage"] = None
        signals["fixture_coverage_note"] = "No eval reports available"

    # 4. CoreML conversion logs (check for recent logs)
    signals["coreml_logs"] = {}

    # Check for worker CoreML artifacts
    worker_mlpackage = "coreml/artifacts/worker/model.mlpackage"
    if os.path.exists(worker_mlpackage):
        signals["coreml_logs"]["worker_exists"] = True
        signals["coreml_logs"]["worker_path"] = worker_mlpackage
    else:
        signals["coreml_logs"]["worker_exists"] = False
        signals["coreml_logs"]["worker_note"] = "Run 'make coreml-worker' to generate"

    # Check for judge CoreML artifacts
    judge_mlpackage = "coreml/artifacts/judge/model.mlpackage"
    if os.path.exists(judge_mlpackage):
        signals["coreml_logs"]["judge_exists"] = True
        signals["coreml_logs"]["judge_path"] = judge_mlpackage
    else:
        signals["coreml_logs"]["judge_exists"] = False
        signals["coreml_logs"]["judge_note"] = "Run 'make judge_coreml' to generate"

    # Dataset sample (10 rows)
    dataset_path = "data/contextual_final.jsonl"
    if os.path.exists(dataset_path):
        signals["dataset_sample"] = []
        signals["dataset_sha256"] = sha256_file(dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                if line.strip():
                    try:
                        signals["dataset_sample"].append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Fixture stats
    fixtures_dir = "eval/tool_broker/fixtures"
    if os.path.exists(fixtures_dir):
        fixture_files = [
            f for f in os.listdir(fixtures_dir)
            if f.endswith(".jsonl")
        ]
        signals["fixtures"] = {
            "count": len(fixture_files),
            "files": fixture_files,
        }

    # Prompt wrapper fingerprint
    prompt_wrapper = "eval/prompt_wrappers/minimal_system_user.j2"
    if os.path.exists(prompt_wrapper):
        signals["prompt_wrapper_sha256"] = sha256_file(prompt_wrapper)
        signals["prompt_wrapper_path"] = prompt_wrapper

    print(json.dumps(signals, indent=2))


if __name__ == "__main__":
    main()
