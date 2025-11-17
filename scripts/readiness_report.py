#!/usr/bin/env python3
"""Generate comprehensive readiness report for gap analysis.

Author: @darianrosebrook
"""

import json
import os
import hashlib
import platform
import shlex
import subprocess
import glob
from pathlib import Path
from typing import Any, Dict, Optional, List
import sys


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
    """Run command and return output or error message.

    SECURITY: Uses shlex.split() to safely parse command string.
    Commands are hardcoded system commands (not user input).
    """
    try:
        cmd_parts = shlex.split(cmd)
        out = subprocess.check_output(
            cmd_parts, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.strip()}"


def check_dependencies() -> Dict[str, Any]:
    """Check key package versions."""
    deps = {}
    packages = [
        "torch",
        "transformers",
        "coremltools",
        "onnx",
        "onnxruntime",
        "accelerate",
        "datasets",
        "tokenizers",
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            deps[pkg] = {"version": version, "available": True}
        except ImportError:
            deps[pkg] = {"version": None, "available": False}

    return deps


def check_model_checkpoints() -> Dict[str, Any]:
    """Validate checkpoint presence and integrity."""
    checkpoints = {
        "worker": {},
        "judge": {},
        "drafter": {},
    }

    # Try to import checkpoint loading, but don't fail if not available
    try:
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint_loading_available = True
    except (ImportError, ModuleNotFoundError):
        checkpoint_loading_available = False

    # Worker checkpoints
    worker_paths = [
        "models/student/checkpoints/latest.pt",
        "models/student/checkpoints/process_supervised_latest.pt",
    ]
    for path in worker_paths:
        if os.path.exists(path):
            if checkpoint_loading_available:
                try:
                    ckpt = safe_load_checkpoint(path, map_location="cpu")
                    checkpoints["worker"][path] = {
                        "exists": True,
                        "valid": True,
                        "step": ckpt.get("step", None),
                        "has_config": "config" in ckpt,
                        "has_model_arch": "model_arch" in ckpt,
                        "size_bytes": os.path.getsize(path),
                    }
                except Exception as e:
                    checkpoints["worker"][path] = {
                        "exists": True,
                        "valid": False,
                        "error": str(e),
                    }
            else:
                # Can't validate without torch, but file exists
                checkpoints["worker"][path] = {
                    "exists": True,
                    "valid": None,
                    "size_bytes": os.path.getsize(path),
                    "note": "torch not available for validation",
                }
        else:
            checkpoints["worker"][path] = {"exists": False}

    # Judge checkpoints
    judge_paths = [
        "arbiter/judge_training/artifacts/judge.pt",
    ]
    for path in judge_paths:
        if os.path.exists(path):
            if checkpoint_loading_available:
                try:
                    ckpt = safe_load_checkpoint(path, map_location="cpu")
                    checkpoints["judge"][path] = {
                        "exists": True,
                        "valid": True,
                        "step": ckpt.get("step", None),
                        "has_config": "config" in ckpt,
                        "size_bytes": os.path.getsize(path),
                    }
                except Exception as e:
                    checkpoints["judge"][path] = {
                        "exists": True,
                        "valid": False,
                        "error": str(e),
                    }
            else:
                checkpoints["judge"][path] = {
                    "exists": True,
                    "valid": None,
                    "size_bytes": os.path.getsize(path),
                    "note": "torch not available for validation",
                }
        else:
            checkpoints["judge"][path] = {"exists": False}

    # Drafter checkpoints (if any)
    drafter_paths = glob.glob("models/drafter/checkpoints/*.pt")
    for path in drafter_paths[:5]:  # Limit to first 5
        if os.path.exists(path):
            if checkpoint_loading_available:
                try:
                    ckpt = safe_load_checkpoint(path, map_location="cpu")
                    checkpoints["drafter"][path] = {
                        "exists": True,
                        "valid": True,
                        "step": ckpt.get("step", None),
                        "size_bytes": os.path.getsize(path),
                    }
                except Exception as e:
                    checkpoints["drafter"][path] = {
                        "exists": True,
                        "valid": False,
                        "error": str(e),
                    }
            else:
                checkpoints["drafter"][path] = {
                    "exists": True,
                    "valid": None,
                    "size_bytes": os.path.getsize(path),
                    "note": "torch not available for validation",
                }

    return checkpoints


def check_configs() -> Dict[str, Any]:
    """Verify config files exist and are valid."""
    configs = {}
    required_configs = [
        "configs/worker_9b.yaml",
        "configs/judge_4b.yaml",
        "configs/judge_training.yaml",
        "configs/kd_recipe.yaml",
        "configs/drafter_4b.yaml",
    ]

    # Try to import yaml, but don't fail if not available
    try:
        import yaml
        yaml_available = True
    except ImportError:
        yaml_available = False

    for config_path in required_configs:
        if os.path.exists(config_path):
            if yaml_available:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                    configs[config_path] = {
                        "exists": True,
                        "valid": True,
                        "sha256": sha256p(config_path),
                        "has_arch": "arch" in config_data if config_data else False,
                    }
                except Exception as e:
                    configs[config_path] = {
                        "exists": True,
                        "valid": False,
                        "error": str(e),
                    }
            else:
                # YAML not available, just check file exists and get hash
                configs[config_path] = {
                    "exists": True,
                    "valid": None,  # Unknown without yaml
                    "sha256": sha256p(config_path),
                    "note": "yaml module not available for validation",
                }
        else:
            configs[config_path] = {"exists": False}

    return configs


def check_datasets() -> Dict[str, Any]:
    """Validate dataset files and structure."""
    datasets = {
        "kd_mix": {},
        "contextual": {},
    }

    # KD dataset
    kd_path = "data/kd_mix.jsonl"
    if os.path.exists(kd_path):
        try:
            line_count = 0
            with open(kd_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        line_count += 1
            datasets["kd_mix"] = {
                "exists": True,
                "valid": True,
                "sample_count": line_count,
                "size_bytes": os.path.getsize(kd_path),
                "sha256": sha256p(kd_path),
            }
        except Exception as e:
            datasets["kd_mix"] = {
                "exists": True,
                "valid": False,
                "error": str(e),
            }
    else:
        datasets["kd_mix"] = {"exists": False}

    # Contextual datasets
    contextual_files = glob.glob("data/contextual*.jsonl")
    datasets["contextual"]["files"] = []
    datasets["contextual"]["total_samples"] = 0

    for file_path in contextual_files[:10]:  # Limit to first 10
        try:
            line_count = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        line_count += 1
            datasets["contextual"]["files"].append({
                "path": os.path.basename(file_path),
                "valid": True,
                "sample_count": line_count,
                "size_bytes": os.path.getsize(file_path),
            })
            datasets["contextual"]["total_samples"] += line_count
        except Exception as e:
            datasets["contextual"]["files"].append({
                "path": os.path.basename(file_path),
                "valid": False,
                "error": str(e),
            })

    datasets["contextual"]["file_count"] = len(contextual_files)

    return datasets


def check_artifacts() -> Dict[str, Any]:
    """Check export/conversion artifacts."""
    artifacts = {
        "coreml": {},
        "onnx": {},
        "pytorch_export": {},
    }

    # CoreML artifacts
    coreml_dirs = [
        "coreml/artifacts/worker",
        "coreml/artifacts/judge",
        "coreml/artifacts/toy",
    ]
    for dir_path in coreml_dirs:
        if os.path.exists(dir_path):
            mlpackages = glob.glob(
                f"{dir_path}/**/*.mlpackage", recursive=True)
            artifacts["coreml"][dir_path] = {
                "exists": True,
                "mlpackage_count": len(mlpackages),
                "mlpackages": [os.path.basename(p) for p in mlpackages[:5]],
            }
        else:
            artifacts["coreml"][dir_path] = {"exists": False}

    # ONNX artifacts
    onnx_dirs = [
        "artifacts/onnx",
        "arbiter/judge_training/artifacts/onnx",
    ]
    for dir_path in onnx_dirs:
        if os.path.exists(dir_path):
            onnx_files = glob.glob(f"{dir_path}/**/*.onnx", recursive=True)
            artifacts["onnx"][dir_path] = {
                "exists": True,
                "file_count": len(onnx_files),
                "files": [os.path.basename(f) for f in onnx_files[:10]],
            }
        else:
            artifacts["onnx"][dir_path] = {"exists": False}

    # PyTorch exports
    export_dirs = [
        "models/student/exported",
        "arbiter/judge_training/artifacts/exported",
    ]
    for dir_path in export_dirs:
        if os.path.exists(dir_path):
            pt_files = glob.glob(f"{dir_path}/**/*.pt", recursive=True)
            artifacts["pytorch_export"][dir_path] = {
                "exists": True,
                "file_count": len(pt_files),
                "files": [os.path.basename(f) for f in pt_files[:10]],
            }
        else:
            artifacts["pytorch_export"][dir_path] = {"exists": False}

    return artifacts


def check_eval_infrastructure() -> Dict[str, Any]:
    """Validate evaluation setup."""
    eval_data = {
        "reports": {},
        "fixtures": {},
        "wrappers": {},
    }

    # Evaluation reports
    report_files = [
        "eval/reports/latest.json",
        "eval/reports/verify_dataset.json",
        "eval/reports/smoke.json",
    ]
    for report_path in report_files:
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                eval_data["reports"][os.path.basename(report_path)] = {
                    "exists": True,
                    "valid": True,
                    "has_summary": "summary" in report_data,
                    "has_header": "header" in report_data,
                }
                # Extract fixture hit rate if available
                if "summary" in report_data:
                    summary = report_data["summary"]
                    if "fixture_hit_rate" in summary:
                        eval_data["reports"][os.path.basename(
                            report_path)]["fixture_hit_rate"] = summary["fixture_hit_rate"]
            except Exception as e:
                eval_data["reports"][os.path.basename(report_path)] = {
                    "exists": True,
                    "valid": False,
                    "error": str(e),
                }
        else:
            eval_data["reports"][os.path.basename(report_path)] = {
                "exists": False}

    # Fixtures
    fixture_dir = "eval/tool_broker/fixtures"
    if os.path.exists(fixture_dir):
        fixtures = glob.glob(f"{fixture_dir}/*")
        eval_data["fixtures"] = {
            "exists": True,
            "count": len([f for f in fixtures if os.path.isfile(f)]),
            "files": [os.path.basename(f) for f in fixtures if os.path.isfile(f)][:20],
        }
    else:
        eval_data["fixtures"] = {"exists": False}

    # Prompt wrappers
    wrapper_dir = "eval/prompt_wrappers"
    if os.path.exists(wrapper_dir):
        wrappers = glob.glob(f"{wrapper_dir}/*")
        eval_data["wrappers"] = {
            "exists": True,
            "count": len([w for w in wrappers if os.path.isfile(w)]),
            "files": [os.path.basename(w) for w in wrappers if os.path.isfile(w)],
        }
    else:
        eval_data["wrappers"] = {"exists": False}

    return eval_data


def validate_critical_paths() -> Dict[str, Any]:
    """Check end-to-end workflow readiness."""
    paths = {
        "training": {},
        "export": {},
        "eval": {},
        "judge": {},
    }

    # Training path: Dataset → Config → Training script → Checkpoint
    paths["training"] = {
        "dataset": os.path.exists("data/kd_mix.jsonl"),
        "config": os.path.exists("configs/worker_9b.yaml") and os.path.exists("configs/kd_recipe.yaml"),
        "training_script": os.path.exists("training/distill_kd.py"),
        "checkpoint": os.path.exists("models/student/checkpoints/latest.pt"),
        "ready": False,
    }
    paths["training"]["ready"] = all([
        paths["training"]["dataset"],
        paths["training"]["config"],
        paths["training"]["training_script"],
    ])

    # Export path: Checkpoint → PyTorch export → CoreML conversion
    paths["export"] = {
        "checkpoint": os.path.exists("models/student/checkpoints/latest.pt"),
        "pytorch_export": os.path.exists("models/student/exported"),
        "coreml_artifact": os.path.exists("coreml/artifacts/worker"),
        "export_script": os.path.exists("conversion/export_pytorch.py"),
        "conversion_script": os.path.exists("conversion/convert_coreml.py"),
        "ready": False,
    }
    paths["export"]["ready"] = all([
        paths["export"]["checkpoint"],
        paths["export"]["export_script"],
        paths["export"]["conversion_script"],
    ])

    # Evaluation path: Model → Dataset → Runner → Report
    paths["eval"] = {
        "model": os.path.exists("models/student/checkpoints/latest.pt") or os.path.exists("coreml/artifacts/worker"),
        "dataset": os.path.exists("data/contextual_final.jsonl") or os.path.exists("data/contextual*.jsonl"),
        "runner": os.path.exists("eval/runners/hf_local.py") or os.path.exists("eval/runners/openai_http.py"),
        "report": os.path.exists("eval/reports/latest.json"),
        "ready": False,
    }
    paths["eval"]["ready"] = all([
        paths["eval"]["model"],
        paths["eval"]["dataset"],
        paths["eval"]["runner"],
    ])

    # Judge path: Judge training → Export → CoreML
    paths["judge"] = {
        "training_config": os.path.exists("configs/judge_training.yaml"),
        "training_script": os.path.exists("arbiter/judge_training/train.py"),
        "checkpoint": os.path.exists("arbiter/judge_training/artifacts/judge.pt"),
        "export_script": os.path.exists("arbiter/judge_training/export_onnx.py"),
        "coreml_artifact": os.path.exists("coreml/artifacts/judge"),
        "ready": False,
    }
    paths["judge"]["ready"] = all([
        paths["judge"]["training_config"],
        paths["judge"]["training_script"],
    ])

    return paths


def calculate_priorities(report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate prioritized action list."""
    blockers: List[Dict[str, Any]] = []
    high_priority: List[Dict[str, Any]] = []
    medium_priority: List[Dict[str, Any]] = []
    low_priority: List[Dict[str, Any]] = []

    # Check dependencies
    deps = report.get("dependencies", {})
    missing_deps = [pkg for pkg, info in deps.items(
    ) if not info.get("available", False)]
    if missing_deps:
        blockers.append({
            "category": "dependencies",
            "issue": f"Missing critical dependencies: {', '.join(missing_deps)}",
            "action": "Run: pip install -e .",
        })

    # Check configs
    configs = report.get("configs", {})
    missing_configs = [path for path,
                       info in configs.items() if not info.get("exists", False)]
    if missing_configs:
        blockers.append({
            "category": "configs",
            "issue": f"Missing required configs: {', '.join(missing_configs)}",
            "action": "Create missing configuration files",
        })

    # Check training path
    paths = report.get("critical_paths", {})
    training_path = paths.get("training", {})
    if not training_path.get("dataset", False):
        high_priority.append({
            "category": "training",
            "issue": "Missing KD dataset (data/kd_mix.jsonl)",
            "action": "Generate dataset: python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher <endpoint>",
        })
    if not training_path.get("checkpoint", False) and training_path.get("ready", False):
        high_priority.append({
            "category": "training",
            "issue": "No trained checkpoint found",
            "action": "Train model: make worker",
        })

    # Check export path
    export_path = paths.get("export", {})
    if export_path.get("checkpoint", False) and not export_path.get("pytorch_export", False):
        high_priority.append({
            "category": "export",
            "issue": "Checkpoint exists but not exported to PyTorch",
            "action": "Export: make pytorch-worker",
        })
    if export_path.get("pytorch_export", False) and not export_path.get("coreml_artifact", False):
        high_priority.append({
            "category": "export",
            "issue": "PyTorch export exists but not converted to CoreML",
            "action": "Convert: make coreml-worker",
        })

    # Check evaluation
    eval_path = paths.get("eval", {})
    eval_infra = report.get("eval", {})
    fixtures = eval_infra.get("fixtures", {})
    if fixtures.get("count", 0) < 5:
        medium_priority.append({
            "category": "eval",
            "issue": f"Low fixture coverage ({fixtures.get('count', 0)} fixtures)",
            "action": "Add more fixtures to eval/tool_broker/fixtures/",
        })

    # Check artifacts
    artifacts = report.get("artifacts", {})
    coreml = artifacts.get("coreml", {})
    if not any(info.get("exists", False) for info in coreml.values()):
        medium_priority.append({
            "category": "artifacts",
            "issue": "No CoreML artifacts found",
            "action": "Generate CoreML artifacts for deployment",
        })

    return {
        "blockers": blockers,
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority,
    }


def generate_summary(report: Dict[str, Any], priorities: Dict[str, Any]) -> str:
    """Create human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("Readiness Report Summary")
    lines.append("=" * 70)
    lines.append("")

    # Host info
    host = report.get("host", {})
    lines.append(f"Host: {host.get('machine', 'unknown')}")
    lines.append(f"Python: {host.get('python', 'unknown')}")
    lines.append(f"Architecture: {host.get('arch', 'unknown')}")
    lines.append("")

    # Critical paths status
    paths = report.get("critical_paths", {})
    lines.append("Critical Paths:")
    for path_name, path_info in paths.items():
        status = "READY" if path_info.get("ready", False) else "NOT READY"
        lines.append(f"  {path_name.upper()}: {status}")
    lines.append("")

    # Blockers
    if priorities.get("blockers"):
        lines.append("BLOCKERS (must fix before proceeding):")
        for i, blocker in enumerate(priorities["blockers"], 1):
            lines.append(f"  {i}. [{blocker['category']}] {blocker['issue']}")
            lines.append(f"     Action: {blocker['action']}")
        lines.append("")

    # High priority
    if priorities.get("high_priority"):
        lines.append("HIGH PRIORITY:")
        for i, item in enumerate(priorities["high_priority"], 1):
            lines.append(f"  {i}. [{item['category']}] {item['issue']}")
            lines.append(f"     Action: {item['action']}")
        lines.append("")

    # Medium priority
    if priorities.get("medium_priority"):
        lines.append("MEDIUM PRIORITY:")
        for i, item in enumerate(priorities["medium_priority"], 1):
            lines.append(f"  {i}. [{item['category']}] {item['issue']}")
            lines.append(f"     Action: {item['action']}")
        lines.append("")

    # Low priority
    if priorities.get("low_priority"):
        lines.append("LOW PRIORITY:")
        for i, item in enumerate(priorities["low_priority"], 1):
            lines.append(f"  {i}. [{item['category']}] {item['issue']}")
            lines.append(f"     Action: {item['action']}")
        lines.append("")

    # Next steps
    lines.append("=" * 70)
    lines.append("Recommended Next Steps:")
    lines.append("=" * 70)

    if priorities.get("blockers"):
        lines.append("1. Fix all blockers first")
        if priorities.get("high_priority"):
            lines.append("2. Address high priority items")
            if priorities.get("medium_priority"):
                lines.append("3. Work on medium priority items")
    elif priorities.get("high_priority"):
        lines.append("1. Address high priority items")
        if priorities.get("medium_priority"):
            lines.append("2. Work on medium priority items")
    elif priorities.get("medium_priority"):
        lines.append("1. Work on medium priority items")
    else:
        lines.append("Project appears ready! Consider optimization tasks.")

    return "\n".join(lines)


def main() -> None:
    """Generate comprehensive readiness report."""
    report: Dict[str, Any] = {
        "host": {
            "python": maybe("python -V"),
            "machine": platform.platform(),
            "cpu": platform.processor(),
            "arch": platform.machine(),
        },
        "dependencies": check_dependencies(),
        "models": check_model_checkpoints(),
        "configs": check_configs(),
        "datasets": check_datasets(),
        "artifacts": check_artifacts(),
        "eval": check_eval_infrastructure(),
        "critical_paths": validate_critical_paths(),
        "git": {
            "status_short": maybe("git status -sb"),
            "last_commit": maybe("git log -1 --pretty=oneline"),
            "branch": maybe("git branch --show-current"),
        },
    }

    # Calculate priorities
    priorities = calculate_priorities(report)
    report["blockers"] = priorities["blockers"]
    report["priorities"] = priorities

    # Generate and print JSON report
    json_output = json.dumps(report, indent=2)
    print(json_output)

    # Also print human-readable summary
    print("\n")
    print(generate_summary(report, priorities))


if __name__ == "__main__":
    main()
