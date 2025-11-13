#!/usr/bin/env python3
"""Generate readiness report for gap analysis.

Author: @darianrosebrook
"""

import json
import os
import hashlib
import platform
import subprocess
import glob
from typing import Any, Dict, Optional


def sha256p(path: str) -> Optional[str]:
    """Compute SHA256 of file."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def maybe(cmd: str) -> str:
    """Run command and return output or error message."""
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.strip()}"


def main() -> None:
    """Generate readiness report."""
    report: Dict[str, Any] = {
        "host": {
            "python": maybe("python -V"),
            "machine": platform.platform(),
            "cpu": platform.processor(),
            "arch": platform.machine(),
        },
        "files": {},
        "eval": {},
        "ci": {},
        "coreml": {},
        "git": {},
    }

    # Eval reports
    for p in [
        "eval/reports/latest.json",
        "eval/reports/verify_dataset.json",
        "eval/reports/smoke.json",
    ]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    report["eval"][os.path.basename(p)] = json.load(f)
            except Exception as e:
                report["eval"][os.path.basename(p)] = f"ERROR reading: {e}"

    # History
    hist = "eval/reports/history.ndjson"
    if os.path.exists(hist):
        try:
            with open(hist, "r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
            report["eval"]["history_tail"] = [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            report["eval"]["history_error"] = str(e)

    # Fixtures
    fx = glob.glob("eval/tool_broker/fixtures/*")
    report["files"]["fixtures_count"] = len(fx)
    report["files"]["fixtures"] = [os.path.basename(f) for f in fx[:50]]

    # Configs
    for p in [
        "configs/worker_9b.yaml",
        "configs/judge_4b.yaml",
        "configs/drafter_4b.yaml",
    ]:
        if os.path.exists(p):
            report["files"][p] = sha256p(p)

    # CoreML artifacts
    cm = glob.glob("coreml/artifacts/**/*", recursive=True)
    report["coreml"]["artifacts"] = [
        p for p in cm if os.path.isfile(p) and not p.endswith(".DS_Store")
    ][:100]
    report["coreml"]["artifact_count"] = len(cm)

    # Dataset files
    dataset_files = glob.glob("data/contextual*.jsonl")
    report["files"]["dataset_files"] = [os.path.basename(f) for f in dataset_files]
    report["files"]["dataset_count"] = len(dataset_files)

    # Git
    report["git"] = {
        "status_short": maybe("git status -sb"),
        "last_commit": maybe("git log -1 --pretty=oneline"),
        "branch": maybe("git branch --show-current"),
    }

    # CI workflows
    ci_workflows = glob.glob(".github/workflows/eval*.yml")
    report["ci"]["workflows"] = [os.path.basename(w) for w in ci_workflows]
    report["ci"]["workflow_count"] = len(ci_workflows)

    # Prompt wrappers
    prompt_wrappers = glob.glob("eval/prompt_wrappers/*")
    report["files"]["prompt_wrappers"] = [
        os.path.basename(p) for p in prompt_wrappers if os.path.isfile(p)
    ]

    # Tool registry
    registry_files = glob.glob("tools/*.py")
    report["files"]["tool_registry_files"] = [os.path.basename(f) for f in registry_files]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
