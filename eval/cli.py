# eval/cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List

# Runners & scoring
from eval.runners.base import Runner

# Import runners conditionally to avoid requiring all dependencies for smoke testing
try:
    from eval.runners.openai_http import OpenAIHTTPRunner
    OPENAI_RUNNER_AVAILABLE = True
except ImportError:
    OPENAI_RUNNER_AVAILABLE = False

try:
    from eval.runners.hf_local import HFLocalRunner
    HF_RUNNER_AVAILABLE = True
except ImportError:
    HF_RUNNER_AVAILABLE = False

try:
    from eval.runners.orchestrator import OrchestratorRunner
    ORCHESTRATOR_RUNNER_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_RUNNER_AVAILABLE = False

from eval.tool_broker.broker import ToolBroker
from eval.scoring.scorer import score_item  # wraps your verify_* logic
from eval.reports.summarize import summarize_results  # macro/micro, deltas, gates
from tools.schema_registry import ToolSchemaRegistry

class MockRunner(Runner):
    """Mock runner for smoke testing without requiring a real model."""

    def generate(self, prompt: str, tools: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Return mock generation result."""
        return {
            "text": f"[MOCK] Response to: {prompt[:50]}...",
            "finish_reason": "stop",
            "tool_calls": [],
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10}
        }

    def fingerprint(self) -> Dict[str, Any]:
        """Return mock fingerprint."""
        return {"runner": "mock", "model": "mock-smoke-test"}


RUNNERS = {
    "mock": MockRunner,  # For smoke testing - always available
}

if OPENAI_RUNNER_AVAILABLE:
    RUNNERS["openai_http"] = OpenAIHTTPRunner

if HF_RUNNER_AVAILABLE:
    RUNNERS["hf_local"] = HFLocalRunner

if ORCHESTRATOR_RUNNER_AVAILABLE:
    RUNNERS["orchestrator"] = OrchestratorRunner


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file_excluding_header(path: str) -> str:
    """Compute dataset SHA256 excluding the first header line (if any)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        first = f.readline()
        try:
            obj = json.loads(first)
            has_header = isinstance(obj, dict) and obj.get(
                "__header__") is True
        except Exception:
            has_header = False
        if not has_header:
            h.update(first)
        for chunk in f:
            h.update(chunk)
    return h.hexdigest()


def stable_shard(sample_id: str, num_shards: int) -> int:
    """
    Derive shard assignment from stable sample_id hash.

    Uses SHA256 hash of sample_id to ensure shard membership remains
    identical across dataset reorders, unlike modulo-based partitioning.

    Args:
        sample_id: Stable sample identifier
        num_shards: Number of shards

    Returns:
        Shard index (0 to num_shards-1)
    """
    h = hashlib.sha256(sample_id.encode('utf-8')).digest()
    # Use first 8 bytes as little-endian to reduce modulo bias
    val = int.from_bytes(h[:8], 'little')
    return val % num_shards


def select_shard(items: List[Dict[str, Any]], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    """
    Select items for a specific shard using stable hash partitioning.

    Uses stable hash of sample_id (or synthesized ID from row) to ensure
    shard membership remains consistent across dataset reorders.

    Args:
        items: List of dataset items
        shard_index: Target shard index (0 to num_shards-1)
        num_shards: Total number of shards

    Returns:
        List of items assigned to this shard
    """
    if num_shards <= 1:
        return items

    result = []
    for item in items:
        # Extract sample_id from metadata, or synthesize from row
        sample_id = None
        if isinstance(item, dict):
            meta = item.get("metadata", {})
            sample_id = meta.get("sample_id")

            # If no sample_id, synthesize stable ID from row
            if not sample_id:
                row_json = json.dumps(item, sort_keys=True, ensure_ascii=False)
                sample_id = hashlib.sha256(
                    row_json.encode('utf-8')).hexdigest()

        if sample_id:
            assigned_shard = stable_shard(sample_id, num_shards)
            if assigned_shard == shard_index:
                result.append(item)

    return result


def main() -> None:
    ap = argparse.ArgumentParser("Tool-Integration Evaluation Harness")
    ap.add_argument("--runner", required=True, choices=RUNNERS.keys())
    ap.add_argument("--model", required=True,
                    help="Model name or local checkpoint path")
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input dataset JSONL (verified)")
    ap.add_argument("--out", required=True, help="Output results JSONL")
    ap.add_argument("--report", required=True, help="Summary report JSON")
    ap.add_argument("--fixtures", required=True,
                    help="Fixtures directory for ToolBroker")
    ap.add_argument("--prompt-wrapper", default=None,
                    help="Path to prompt wrapper template (Jinja2 or string.Template)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--min-eligible-for-gates", type=int, default=15)
    ap.add_argument("--fail-on-fingerprint-mismatch", action="store_true")
    ap.add_argument("--no-fail-on-fingerprint-mismatch",
                    dest="fail_on_fingerprint_mismatch", action="store_false")
    ap.set_defaults(fail_on_fingerprint_mismatch=True)
    ap.add_argument("--determinism-mode", action="store_true",
                    help="Determinism mode: temp=0, top_p=1, no retries, fail on any retry")
    ap.add_argument("--baseline-report", type=str, default=None,
                    help="Path to baseline report for speed gate comparison")
    ap.add_argument("--baseline-dir", type=str, default="eval/baselines",
                    help="Directory for latent efficiency baseline artifacts")
    ap.add_argument("--save-baseline", action="store_true",
                    help="Save current run as baseline (for direct CoT)")
    ap.add_argument("--workload-type", type=str, default="interactive",
                    choices=["interactive", "offline"],
                    help="Workload type: 'interactive' (batch=1) or 'offline' (batch 2-4)")
    args = ap.parse_args()

    # Load dataset
    raw_items = list(read_jsonl(args.inp))
    header = raw_items[0] if raw_items and isinstance(
        raw_items[0], dict) and raw_items[0].get("__header__") else None
    items = raw_items[1:] if header else raw_items

    # Fingerprints
    dataset_sha = sha256_file_excluding_header(args.inp)
    registry_sha = (header or {}).get("tool_registry_sha256")
    tokenizer_fp = (header or {}).get("tokenizer_fingerprint")
    integration_span_cap = (header or {}).get("integration_span_cap", 3)

    # Load hardware profile for batch policy
    hardware_profile = None
    batch_policy = None
    try:
        from eval.hw_profile import load_profiles, match_profile
        from coreml.runtime.batch_policy import BatchPolicy
        from pathlib import Path

        profiles = load_profiles(Path("configs/hardware_profiles.yaml"))
        profile = match_profile(profiles)
        hardware_profile = {"key": profile.key, "config": profile.config}

        batch_policy = BatchPolicy(hardware_profile=hardware_profile)
        selected_batch = batch_policy.select_batch_size(
            workload_type=args.workload_type)
        print(
            f"[eval/cli] Batch policy: workload_type={args.workload_type}, batch_size={selected_batch}")
    except ImportError:
        # Skip batch policy if dependencies not available (for smoke testing)
        pass
    except Exception as e:
        print(f"[eval/cli] WARN: Failed to initialize batch policy: {e}")

    # Load runtime configs if advanced features enabled
    runtime_config = None
    eval_latent = os.getenv("EVAL_LATENT", "0") == "1"
    eval_code_mode = os.getenv("EVAL_CODE_MODE", "0") == "1"
    
    if eval_latent or eval_code_mode or args.runner == "orchestrator":
        try:
            from runtime.config import RuntimeConfig
            from pathlib import Path
            try:
                import yaml
                YAML_AVAILABLE = True
            except ImportError:
                YAML_AVAILABLE = False
            
            # Load config from files if they exist
            latent_config_path = Path("eval/configs/latent.yaml")
            code_mode_config_path = Path("eval/configs/code_mode.yaml")
            
            # Start with defaults
            runtime_config = RuntimeConfig.from_env()
            
            # Override with config files if they exist
            if eval_latent and latent_config_path.exists() and YAML_AVAILABLE:
                with open(latent_config_path, 'r') as f:
                    latent_config = yaml.safe_load(f)
                    gates = latent_config.get("gates", {})
                    efficiency = gates.get("efficiency", {})
                    runtime_config.latent_mode_enabled = True
                    runtime_config.max_refinement_loops = efficiency.get("max_loop_increase", 5) or 5
                    runtime_config.curriculum_probability = 1.0  # Full curriculum for evaluation
            elif eval_latent and latent_config_path.exists() and not YAML_AVAILABLE:
                print("[eval/cli] WARN: YAML not available, skipping latent config loading")

            if eval_code_mode and code_mode_config_path.exists() and YAML_AVAILABLE:
                with open(code_mode_config_path, 'r') as f:
                    code_mode_config = yaml.safe_load(f)
                    gates = code_mode_config.get("gates", {})
                    runtime_config.latent_mode_enabled = False  # Code mode doesn't use latent
            elif eval_code_mode and code_mode_config_path.exists() and not YAML_AVAILABLE:
                print("[eval/cli] WARN: YAML not available, skipping code mode config loading")
                    # Code mode settings would go here if needed
            
            print(f"[eval/cli] Runtime config loaded: latent={runtime_config.latent_mode_enabled}, halt={runtime_config.halt_head_enabled}")
        except Exception as e:
            print(f"[eval/cli] WARN: Failed to load runtime config: {e}")
            runtime_config = None

    # Init runner & broker
    RunnerCls = RUNNERS[args.runner]

    # Determinism mode: enforce temp=0, top_p=1, no retries
    if args.determinism_mode:
        runner_kwargs = {
            "model": args.model,
            "seed": args.seed,
            "temperature": 0.0,  # Force temp=0
            "max_tokens": args.max_tokens,
            "top_p": 1.0,  # Force top_p=1
            "determinism_mode": True,  # Signal to disable retries
        }
    else:
        runner_kwargs = {
            "model": args.model,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }

    if args.prompt_wrapper:
        runner_kwargs["prompt_wrapper"] = args.prompt_wrapper
    
    # Add runtime config for orchestrator runner
    if args.runner == "orchestrator" and runtime_config:
        runner_kwargs["runtime_config"] = runtime_config
        runner_kwargs["use_refinement"] = eval_latent  # Use refinement for latent evaluation
    
    runner = RunnerCls(**runner_kwargs)
    broker = ToolBroker(args.fixtures)

    # Shard
    items = select_shard(items, args.shard_index, args.num_shards)

    # If dataset header exists, optionally fail on fingerprint mismatch
    if header and args.fail_on_fingerprint_mismatch:
        # Minimal verification example; extend as needed
        if "dataset_sha256" in header and header["dataset_sha256"] != dataset_sha:
            print("[EVAL] Fingerprint mismatch: dataset_sha256", file=sys.stderr)
            sys.exit(1)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for item in items:
        prompt = item.get("prompt") or ""
        # Extract tools from metadata if available, otherwise empty list
        meta = item.get("metadata") or {}
        # Build tools list from expected call_sequence (for runner context)
        tools = []
        expected_calls = meta.get("call_sequence", [])
        if expected_calls:
            # Build tool schemas from registry for runner
            reg = ToolSchemaRegistry()
            for call in expected_calls:
                tool_name = call.get("name", "")
                schema = reg.get(tool_name)
                if schema:
                    tools.append({
                        "name": tool_name,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {}),
                    })

        # 1) Generate with tools enabled (runner emits tool_trace without results)
        gen = runner.generate(
            prompt=prompt,
            tools=tools,
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # 2) Broker deterministic tool results
        tool_trace = []
        for call in gen.get("tool_trace", []):
            name = call.get("name")
            args_obj = call.get("arguments", {})
            result = broker.call(name, args_obj)
            tool_trace.append(
                {"name": name, "arguments": args_obj, "result": result})

        # 3) Score using verifier-parity scorer
        scores = score_item(
            item=item,
            model_output=gen.get("model_output", ""),
            tool_trace=tool_trace,
            integration_span_cap=integration_span_cap,
            latent_spans_used=gen.get("latent_spans_used", 0),
            refinement_loops=gen.get("refinement_loops", 1),
            halt_logits=gen.get("halt_logits"),
        )

        results.append({
            "sample_id": meta.get("sample_id"),
            "prompt": prompt,
            "model_output": gen.get("model_output", ""),
            "tool_trace": tool_trace,
            "scores": scores,
            "runner_fingerprint": runner.fingerprint(),
            "model_fingerprint": runner.model_fingerprint(),
            "decoding": {"seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens},
        })

    # Write per-item results
    write_jsonl(args.out, results)

    # Load speed metrics if available (from CoreML speed report)
    speed_metrics = None
    hardware = None
    speed_report_path = os.path.join(
        os.path.dirname(args.report), "speed_coreml.json")
    if os.path.exists(speed_report_path):
        try:
            with open(speed_report_path, 'r') as f:
                speed_report = json.load(f)
                speed_metrics = speed_report.get("speed_metrics")
                hardware = speed_report.get("hardware")
                # Include ANE residency if available
                if "ane_residency" in speed_report:
                    if speed_metrics is None:
                        speed_metrics = {}
                    speed_metrics["ane_residency"] = speed_report["ane_residency"]
        except Exception as e:
            print(f"[eval/cli] WARN: Failed to load speed metrics: {e}")

    # Summarize (macro/micro F1 lax & strict, gates, deltas, histograms)
    report = summarize_results(
        results=results,
        report_version="1.0.0",
        dataset_header=header,
        dataset_sha256=dataset_sha,
        tool_registry_sha256=registry_sha,
        tokenizer_fingerprint=tokenizer_fp,
        config={
            "runner": args.runner,
            "model": args.model,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "min_eligible_for_gates": args.min_eligible_for_gates,
            "fail_on_fingerprint_mismatch": args.fail_on_fingerprint_mismatch,
            "determinism_mode": args.determinism_mode,
        },
        wall_time_sec=time.time() - t0,
        gates_overrides={
            "min_eligible_for_gates": args.min_eligible_for_gates},
        speed_metrics=speed_metrics,
        hardware=hardware,
        baseline_report_path=args.baseline_report,
    )

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Fail if gates fail
    if not report.get("gates_ok", True):
        print("[EVAL] Gates FAILED (see report).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
