# eval/cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

# Runners & scoring
from eval.runners.openai_http import OpenAIHTTPRunner
from eval.runners.hf_local import HFLocalRunner
from eval.tool_broker.broker import ToolBroker
from eval.scoring.scorer import score_item  # wraps your verify_* logic
from eval.reports.summarize import summarize_results  # macro/micro, deltas, gates
from tools.schema_registry import ToolSchemaRegistry

RUNNERS = {
    "openai_http": OpenAIHTTPRunner,
    "hf_local": HFLocalRunner,
}


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


def select_shard(items: List[Dict[str, Any]], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return items
    return [it for i, it in enumerate(items) if i % num_shards == shard_index]


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

    # Init runner & broker
    RunnerCls = RUNNERS[args.runner]
    runner_kwargs = {
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.prompt_wrapper:
        runner_kwargs["prompt_wrapper"] = args.prompt_wrapper
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
        },
        wall_time_sec=time.time() - t0,
        gates_overrides={
            "min_eligible_for_gates": args.min_eligible_for_gates},
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
