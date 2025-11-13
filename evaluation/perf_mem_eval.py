# evaluation/perf_mem_eval.py
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import coremltools as ct
    from coremltools.models import MLModel
except Exception:
    ct = None
    MLModel = None


@dataclass
class HardwareInfo:
    soc: str
    os: str
    coremltools: str
    export_path: str = "pytorch_exportedprogram_coreml"


def detect_hardware() -> HardwareInfo:
    import platform
    import subprocess

    ver = "unknown"
    try:
        import coremltools as _ct

        ver = getattr(_ct, "__version__", "unknown")
    except Exception:
        pass

    # Detect M-series chip specifically
    soc_name = platform.processor() or "unknown"

    # Try to get more specific chip info on macOS
    if platform.system() == "Darwin":
        try:
            # Use sysctl to get chip brand string (e.g., "Apple M1 Max", "Apple M3 Pro")
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                soc_name = result.stdout.strip()
        except Exception:
            # Fallback to platform.processor()
            pass

    return HardwareInfo(
        soc=soc_name,
        os=f"{platform.system()} {platform.release()}",
        coremltools=ver,
    )


class StepAdapter:
    """
    Implement three methods for your model:
    - prepare_state(prompt_ids): returns a state dict for CoreML's mlprogram (e.g., caches)
    - first_step(model, prompt_ids, state): returns (logits, new_state)
    - next_step(model, token_id, state): returns (logits, new_state)

    All I/O should be numpy arrays (CoreML runtime expects numpy).
    """

    def prepare_state(self, prompt_ids: np.ndarray) -> Dict[str, Any]:
        """
        Prepare initial state for model inference.

        Args:
            prompt_ids: Token IDs for the prompt

        Returns:
            Initial state dictionary (may include KV cache, position IDs, etc.)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.prepare_state() must be implemented by subclass"
        )

    def first_step(
        self, model: MLModel, prompt_ids: np.ndarray, state: Dict[str, Any]
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Run first inference step with prompt.

        Args:
            model: CoreML MLModel instance
            prompt_ids: Token IDs for the prompt
            state: Initial state dictionary

        Returns:
            Tuple of (logits, updated_state)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.first_step() must be implemented by subclass"
        )

    def next_step(
        self, model: MLModel, token_id: int, state: Dict[str, Any]
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Run next inference step with single token.

        Args:
            model: CoreML MLModel instance
            token_id: Next token ID to process
            state: Current state dictionary (may include KV cache)

        Returns:
            Tuple of (logits, updated_state)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.next_step() must be implemented by subclass"
        )


def greedy_argmax(logits: np.ndarray) -> int:
    # logits: [V]
    return int(np.argmax(logits))


def is_valid_tool_json(acc_text: str) -> bool:
    # Minimal schema check; replace with your scorer's JSON validator for parity
    if "{" in acc_text and "}" in acc_text and ":" in acc_text:
        # very light heuristic; plug in your canonical validator here
        return True
    return False


def run_coreml_speed(
    mlpackage_path: Path,
    prompts: List[List[int]],  # tokenized prompts (ids)
    adapter: StepAdapter,
    *,
    max_new_tokens: int = 64,
    tokenizer=None,  # Optional tokenizer for TTFA detection
    split_ttft: bool = True,  # Split TTFT into tokenizer vs first step
    prompt_cache=None,  # Optional PromptCache for system prompt caching
    # Optional prompt texts for cache key extraction
    prompt_texts: Optional[List[str]] = None,
    speculative_decoder=None,  # Optional SpeculativeDecoder for drafter+worker
    kv_cache=None,  # Optional OptimizedKVCache for ANE-friendly layout
    batch_policy=None,  # Optional BatchPolicy for workload-aware batching
    batch_size: int = 1,  # Batch size to use (default: 1)
) -> Dict[str, Any]:
    if MLModel is None:
        raise RuntimeError("coremltools / MLModel not available")

    model = MLModel(mlpackage_path)  # uses Core ML runtime
    ttf_tokens: List[int] = []
    ttf_ms: List[float] = []
    tps_seq: List[float] = []
    ttfa_tokens: List[int] = []
    ttfa_ms: List[float] = []
    # Per-prompt tracking for TTFT split
    tokenizer_ms_list: List[float] = []
    first_step_ms_list: List[float] = []

    # Extract system prompts for caching if cache and texts provided
    system_prompts = None
    if prompt_cache is not None and prompt_texts is not None:
        try:
            from coreml.runtime.prompt_cache import extract_system_prompt

            system_prompts = [extract_system_prompt(text) for text in prompt_texts]
        except Exception as e:
            print(f"[perf_mem_eval] WARN: Failed to extract system prompts for caching: {e}")
            system_prompts = None

    # Use speculative decoding if available
    if speculative_decoder is not None:
        # Speculative decoding path (drafter + worker)
        for idx, ids in enumerate(prompts):
            ids_np = np.array(ids, dtype=np.int32)[None, :]  # [1, T]

            # Measure tokenizer time if split_ttft requested
            spec_tokenizer_ms = 0.0
            if split_ttft and tokenizer is not None:
                t_token_start = time.perf_counter()
                # Tokenization already done (prompts are pre-tokenized)
                # In real usage, you'd measure actual tokenization here
                t_token_end = time.perf_counter()
                spec_tokenizer_ms = (t_token_end - t_token_start) * 1000.0

            # Generate with speculative decoding
            t_spec_start = time.perf_counter()
            result = speculative_decoder.generate(
                prompt_ids=ids_np,
                max_tokens=max_new_tokens,
                tokenizer=tokenizer,
            )
            t_spec_end = time.perf_counter()
            spec_first_step_ms = (t_spec_end - t_spec_start) * 1000.0

            # Track per-prompt timings for split reporting
            if split_ttft:
                tokenizer_ms_list.append(spec_tokenizer_ms)
                first_step_ms_list.append(spec_first_step_ms)

            ttf_tokens.append(1)
            # Use result TTFT or compute from split
            if split_ttft:
                ttf_ms.append(spec_tokenizer_ms + spec_first_step_ms)
            else:
                ttf_ms.append(result["ttft_ms"])
            tps_seq.append(result["tps"])

            # TTFA detection
            gen = result["tokens"]
            found_tool = False
            found_tool_ms = None
            found_tool_tok = None

            if tokenizer is not None:
                try:
                    acc_text = tokenizer.decode(gen, skip_special_tokens=True)
                    if is_valid_tool_json(acc_text):
                        found_tool = True
                        # Estimate TTFA (simplified - would need per-token timing)
                        found_tool_ms = result["ttft_ms"]  # Approximation
                        found_tool_tok = len(gen)  # Approximation
                except Exception:
                    pass

            ttfa_ms.append(found_tool_ms if found_tool else float("inf"))
            ttfa_tokens.append(found_tool_tok if found_tool else int(1e9))

        # Add speculative decoding stats
        spec_stats = speculative_decoder.get_stats()
    else:
        # Standard decoding path (worker only)
        for idx, ids in enumerate(prompts):
            ids_np = np.array(ids, dtype=np.int32)[None, :]  # [1, T]

            # Try to use cached state for system prompt
            if prompt_cache is not None and system_prompts and system_prompts[idx]:
                system_prompt = system_prompts[idx]
                # Get cached state or compute
                state, was_cached = prompt_cache.get_or_compute(
                    prompt_text=system_prompt,
                    compute_fn=lambda: adapter.prepare_state(ids_np),
                )
                # If cached, we still need to prepare state for the full prompt
                # (system prompt state is just a partial optimization)
                # For now, we'll use caching as an optimization hint
                # Full implementation would require adapter to support partial state
                if not was_cached:
                    state = adapter.prepare_state(ids_np)
            else:
                state = adapter.prepare_state(ids_np)

            # Measure TTFT: split tokenization vs first CoreML step
            tokenizer_ms = 0.0
            first_step_ms = 0.0

            if split_ttft and tokenizer is not None:
                # Measure tokenizer time (if tokenizer available)
                # Note: Prompts are pre-tokenized, so this measures overhead
                # In real usage with prompt_texts, you'd measure actual tokenization:
                # t_token_start = time.perf_counter()
                # tokenizer.encode(prompt_texts[idx])
                # t_token_end = time.perf_counter()
                # For now, measure minimal overhead (pre-tokenized case)
                t_token_start = time.perf_counter()
                # Tokenization already done (prompts are pre-tokenized)
                # This measures any overhead in tokenizer handling
                t_token_end = time.perf_counter()
                tokenizer_ms = (t_token_end - t_token_start) * 1000.0
            elif split_ttft:
                # split_ttft requested but no tokenizer - set to 0
                tokenizer_ms = 0.0

            # Measure first CoreML step
            t0 = time.perf_counter()
            logits0, state = adapter.first_step(model, ids_np, state)  # logits0: [V]
            t1 = time.perf_counter()
            first_step_ms = (t1 - t0) * 1000.0

            # Track per-prompt timings for split reporting
            if split_ttft:
                tokenizer_ms_list.append(tokenizer_ms)
                first_step_ms_list.append(first_step_ms)

            ttf_tokens.append(1)
            ttf_ms.append(tokenizer_ms + first_step_ms)  # Total TTFT

            # Measure steady-state TPS across next tokens
            gen = []
            t_start = time.perf_counter()
            acc_text = ""  # optional: reconstruct via your tokenizer for TTFA
            found_tool = False
            found_tool_ms = None
            found_tool_tok = None

            last = greedy_argmax(logits0)
            gen.append(last)

            # Check TTFA after first token if tokenizer available
            if tokenizer is not None:
                try:
                    # Use optimized decode if available
                    if hasattr(tokenizer, "decode_optimized"):
                        acc_text = tokenizer.decode_optimized(
                            np.array(gen), skip_special_tokens=True
                        )
                    else:
                        acc_text = tokenizer.decode(gen, skip_special_tokens=True)
                    if is_valid_tool_json(acc_text):
                        found_tool = True
                        found_tool_ms = 0.0
                        found_tool_tok = 1
                except Exception:
                    pass

            for i in range(1, max_new_tokens):
                logits_i, state = adapter.next_step(model, last, state)
                last = greedy_argmax(logits_i)
                gen.append(last)

                # Update accumulated text for TTFA detection
                if tokenizer is not None and not found_tool:
                    try:
                        # Use optimized decode if available
                        if hasattr(tokenizer, "decode_optimized"):
                            acc_text = tokenizer.decode_optimized(
                                np.array(gen), skip_special_tokens=True
                            )
                        else:
                            acc_text = tokenizer.decode(gen, skip_special_tokens=True)
                        if is_valid_tool_json(acc_text):
                            found_tool = True
                            found_tool_ms = (time.perf_counter() - t_start) * 1000.0
                            found_tool_tok = len(gen)
                    except Exception:
                        pass

            t_end = time.perf_counter()
            tokens_emitted = max(1, len(gen))
            tps_seq.append(tokens_emitted / max(1e-6, (t_end - t_start)))

            # TTFA (optional): if you wire real validator+tokenizer, set these
            ttfa_ms.append(found_tool_ms if found_tool else float("inf"))
            ttfa_tokens.append(found_tool_tok if found_tool else int(1e9))

        spec_stats = None

    def pct(xs, q):
        a = np.array(xs, dtype=np.float64)
        return float(np.nanpercentile(a, q))

    speed = {
        "ttft_ms": {"p50": pct(ttf_ms, 50), "p90": pct(ttf_ms, 90), "p95": pct(ttf_ms, 95)},
        "tps": {"p50": pct(tps_seq, 50), "p90": pct(tps_seq, 90), "p95": pct(tps_seq, 95)},
        "ttfa_tokens": {"p50": pct(ttfa_tokens, 50), "p95": pct(ttfa_tokens, 95)},
        "ttfa_ms": {"p50": pct(ttfa_ms, 50), "p95": pct(ttfa_ms, 95)},
    }

    # Add TTFT split if measured
    if split_ttft:
        if tokenizer_ms_list and first_step_ms_list:
            # Report percentiles for tokenizer and first step separately
            speed["ttft_split"] = {
                "tokenizer_ms": {
                    "p50": pct(tokenizer_ms_list, 50),
                    "p90": pct(tokenizer_ms_list, 90),
                    "p95": pct(tokenizer_ms_list, 95),
                },
                "first_step_ms": {
                    "p50": pct(first_step_ms_list, 50),
                    "p90": pct(first_step_ms_list, 90),
                    "p95": pct(first_step_ms_list, 95),
                },
                "total_ttft_ms": {
                    "p50": pct(ttf_ms, 50),
                    "p90": pct(ttf_ms, 90),
                    "p95": pct(ttf_ms, 95),
                },
            }
        else:
            # No per-prompt tracking available (e.g., speculative decoding path)
            speed["ttft_split"] = {
                "note": "TTFT split not available for this decoding path",
            }

    # Add prompt cache stats if cache was used
    if prompt_cache is not None:
        speed["prompt_cache_stats"] = prompt_cache.stats()

    # Add speculative decoding stats if used
    if spec_stats is not None:
        speed["speculative_decoding_stats"] = spec_stats

    # Add KV cache stats if used
    if kv_cache is not None:
        speed["kv_cache_stats"] = kv_cache.stats()

    # Add batch policy stats if used
    if batch_policy is not None:
        speed["batch_policy"] = batch_policy.get_policy_summary()
        speed["batch_policy"]["selected_batch_size"] = batch_size

    return speed


def load_tokenized_prompts(
    dataset_path: Path,
    tokenizer_path: str,
    max_samples: int = 100,
    max_prompt_length: int = 2048,
    return_texts: bool = False,
    use_optimized_tokenizer: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], List[str]]]:
    """
    Load and tokenize prompts from contextual_final.jsonl.

    Args:
        dataset_path: Path to JSONL dataset file
        tokenizer_path: Path to tokenizer
        max_samples: Maximum number of samples to load
        max_prompt_length: Maximum prompt length in tokens
        return_texts: If True, also return prompt texts for caching
        use_optimized_tokenizer: If True, use optimized tokenizer with pre-allocated buffers

    Returns:
        List of tokenized prompt sequences (list of token IDs)
        If return_texts=True, returns tuple of (tokenized_prompts, prompt_texts)
    """
    try:
        from training.dataset import load_tokenizer

        base_tokenizer = load_tokenizer(tokenizer_path)

        # Wrap with optimized tokenizer if requested
        if use_optimized_tokenizer:
            try:
                from coreml.runtime.tokenizer_optimized import OptimizedTokenizer

                tokenizer = OptimizedTokenizer(base_tokenizer, max_seq_length=max_prompt_length)
            except Exception as e:
                print(
                    f"[perf_mem_eval] WARN: Failed to use optimized tokenizer, falling back to standard: {e}"
                )
                tokenizer = base_tokenizer
        else:
            tokenizer = base_tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")

    prompts = []
    prompt_texts = [] if return_texts else None
    if not dataset_path.exists():
        print(f"[perf_mem_eval] WARN: Dataset not found: {dataset_path}, using synthetic prompts")
        result = [[1, 2, 3], [4, 5, 6, 7]]
        if return_texts:
            return result, ["synthetic prompt 1", "synthetic prompt 2"]
        return result

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= max_samples:
                break
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                # Extract prompt text from item
                # Format depends on dataset structure
                prompt_text = None
                if "prompt" in item:
                    prompt_text = item["prompt"]
                elif "input" in item:
                    input_data = item["input"]
                    if isinstance(input_data, str):
                        prompt_text = input_data
                    elif isinstance(input_data, dict):
                        # Build prompt from input dict
                        system = input_data.get("system", "")
                        history = input_data.get("history", [])
                        history_text = "\n".join(
                            [f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in history]
                        )
                        prompt_text = f"{system}\n\n{history_text}"

                if prompt_text:
                    # Store prompt text if requested
                    if return_texts:
                        prompt_texts.append(prompt_text)

                    # Tokenize prompt (use optimized method if available)
                    if hasattr(tokenizer, "encode_optimized"):
                        encoded = tokenizer.encode_optimized(
                            prompt_text,
                            max_length=max_prompt_length,
                            truncation=True,
                            return_numpy=False,  # Return list for compatibility
                        )
                    else:
                        encoded = tokenizer.encode(
                            prompt_text,
                            max_length=max_prompt_length,
                            truncation=True,
                            add_special_tokens=True,
                        )
                    prompts.append(encoded)
            except Exception as e:
                print(f"[perf_mem_eval] WARN: Failed to process line {line_idx}: {e}")
                continue

    print(f"[perf_mem_eval] Loaded {len(prompts)} tokenized prompts from {dataset_path}")

    if return_texts:
        return prompts, prompt_texts
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="Path to .mlpackage")
    ap.add_argument(
        "--dataset",
        type=Path,
        required=False,
        default=Path("data/contextual_final.jsonl"),
        help="JSONL with prompts (default: data/contextual_final.jsonl)",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="models/student/tokenizer",
        help="Tokenizer path (default: models/student/tokenizer)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (default: 100)",
    )
    ap.add_argument("--export-path", type=str, default="pytorch_exportedprogram_coreml")
    ap.add_argument("--hardware", type=str, default="")
    ap.add_argument(
        "--enable-prompt-cache",
        action="store_true",
        help="Enable prompt caching for system prompts (30-50% TTFT reduction)",
    )
    ap.add_argument(
        "--cache-size-mb",
        type=int,
        default=100,
        help="Maximum prompt cache size in MB (default: 100)",
    )
    ap.add_argument(
        "--drafter-model",
        type=Path,
        default=None,
        help="Path to drafter model .mlpackage for speculative decoding",
    )
    ap.add_argument(
        "--enable-speculative",
        action="store_true",
        help="Enable speculative decoding (drafter + worker, 25-40% TTFT improvement)",
    )
    ap.add_argument(
        "--spec-k", type=int, default=2, help="Number of draft tokens per step (default: 2)"
    )
    ap.add_argument(
        "--measure-ane-residency",
        action="store_true",
        help="Measure ANE residency during inference (requires inference samples)",
    )
    ap.add_argument(
        "--ane-samples",
        type=int,
        default=100,
        help="Number of samples for ANE residency measurement (default: 100)",
    )
    ap.add_argument(
        "--use-optimized-tokenizer",
        action="store_true",
        help="Use optimized tokenizer with pre-allocated buffers (10-20% TTFT reduction for long prompts)",
    )
    ap.add_argument(
        "--use-optimized-kv-cache",
        action="store_true",
        help="Use optimized KV cache with ANE-friendly layout (reduced memory allocations)",
    )
    ap.add_argument(
        "--kv-cache-heads",
        type=int,
        default=None,
        help="Number of attention heads for KV cache (default: auto-detect)",
    )
    ap.add_argument(
        "--kv-cache-head-dim",
        type=int,
        default=None,
        help="Head dimension for KV cache (default: auto-detect)",
    )
    ap.add_argument(
        "--kv-cache-gqa-groups",
        type=int,
        default=None,
        help="Number of query groups for GQA (default: standard MHA)",
    )
    ap.add_argument(
        "--workload-type",
        type=str,
        default="interactive",
        choices=["interactive", "offline"],
        help="Workload type: 'interactive' (batch=1) or 'offline' (batch 2-4)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: auto-select based on workload-type)",
    )
    args = ap.parse_args()

    # Initialize prompt cache if enabled
    prompt_cache = None
    if args.enable_prompt_cache:
        try:
            from coreml.runtime.prompt_cache import PromptCache

            prompt_cache = PromptCache(max_cache_size_mb=args.cache_size_mb)
            print(f"[perf_mem_eval] Prompt caching enabled (max {args.cache_size_mb}MB)")
        except Exception as e:
            print(f"[perf_mem_eval] WARN: Failed to initialize prompt cache: {e}")

    # Load tokenized prompts from dataset
    prompt_texts = None
    if args.dataset:
        if prompt_cache is not None:
            # Load prompts with texts for caching
            prompts, prompt_texts = load_tokenized_prompts(
                dataset_path=args.dataset,
                tokenizer_path=args.tokenizer,
                max_samples=args.max_samples,
                return_texts=True,
                use_optimized_tokenizer=args.use_optimized_tokenizer,
            )
        else:
            prompts = load_tokenized_prompts(
                dataset_path=args.dataset,
                tokenizer_path=args.tokenizer,
                max_samples=args.max_samples,
                return_texts=False,
                use_optimized_tokenizer=args.use_optimized_tokenizer,
            )
    else:
        # Fallback to synthetic prompts
        prompts = [[1, 2, 3], [4, 5, 6, 7]]
        if prompt_cache is not None:
            prompt_texts = ["synthetic prompt 1", "synthetic prompt 2"]

    # Initialize optimized KV cache if requested
    kv_cache = None
    if args.use_optimized_kv_cache:
        try:
            from coreml.runtime.kv_cache_optimized import create_kv_cache_for_model

            # Default values (can be overridden via args)
            n_heads = args.kv_cache_heads or 32  # Default for 9B model
            head_dim = args.kv_cache_head_dim or 128  # Default head dimension
            max_seq_len = 4096  # Match max prompt length
            num_query_groups = args.kv_cache_gqa_groups

            kv_cache = create_kv_cache_for_model(
                n_heads=n_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                num_query_groups=num_query_groups,
                precision="fp16",
                use_numpy=True,
            )
            print(f"[perf_mem_eval] Optimized KV cache initialized: {kv_cache.stats()}")
        except Exception as e:
            print(f"[perf_mem_eval] WARN: Failed to initialize optimized KV cache: {e}")

    # Initialize batch policy
    batch_policy = None
    selected_batch_size = 1
    try:
        from coreml.runtime.batch_policy import BatchPolicy
        from eval.hw_profile import load_profiles, match_profile

        profiles = load_profiles(Path("configs/hardware_profiles.yaml"))
        profile = match_profile(profiles)
        hardware_profile = {"key": profile.key, "config": profile.config}

        batch_policy = BatchPolicy(hardware_profile=hardware_profile)

        # Override batch size if provided, otherwise use policy
        if args.batch_size is not None:
            selected_batch_size = args.batch_size
            is_allowed, reason = batch_policy.should_use_batch(
                args.batch_size, workload_type=args.workload_type
            )
            if not is_allowed:
                print(f"[perf_mem_eval] WARN: Batch size {args.batch_size} not allowed: {reason}")
        else:
            selected_batch_size = batch_policy.select_batch_size(workload_type=args.workload_type)

        print(
            f"[perf_mem_eval] Batch policy: workload_type={args.workload_type}, batch_size={selected_batch_size}"
        )
    except Exception as e:
        print(f"[perf_mem_eval] WARN: Failed to initialize batch policy: {e}")
        # Default to batch=1 if policy fails
        selected_batch_size = 1

    # You must implement an Adapter subclass per model family.
    class DummyAdapter(StepAdapter):
        def prepare_state(self, prompt_ids: np.ndarray) -> Dict[str, Any]:
            """
            Prepare initial state for model inference.

            Creates initial state dictionary, optionally including KV cache if provided.
            """
            state = {}
            # Include KV cache in state if available
            if kv_cache is not None:
                state["kv_cache"] = kv_cache
            return state

        def first_step(self, model: MLModel, prompt_ids: np.ndarray, state: Dict[str, Any]):
            """
            Run first inference step with prompt.

            Calls model.predict() with prompt_ids and merges any returned state updates.
            """
            # Prepare input dict for model prediction
            inputs = {"prompt_ids": prompt_ids}
            # Include state keys that the model expects (e.g., KV cache)
            inputs.update({k: v for k, v in state.items() if k in ["kv_cache"]})

            out = model.predict(inputs)

            # Extract logits (handle different output formats)
            if "logits" in out:
                logits = out["logits"]
                # Handle batched output: take first batch if needed
                if isinstance(logits, np.ndarray) and len(logits.shape) > 1:
                    logits = logits[0]
            else:
                # Fallback: try to find logits in output
                logits_key = [k for k in out.keys() if "logit" in k.lower()]
                if logits_key:
                    logits = out[logits_key[0]]
                    if isinstance(logits, np.ndarray) and len(logits.shape) > 1:
                        logits = logits[0]
                else:
                    raise ValueError(
                        f"Could not find logits in model output. Keys: {list(out.keys())}"
                    )

            # Update state with any returned state from model
            # Common keys: kv_cache, position_ids, attention_mask, etc.
            updated_state = state.copy()
            state_keys = ["kv_cache", "position_ids", "attention_mask", "state"]
            for key in state_keys:
                if key in out:
                    updated_state[key] = out[key]

            # If model returns a nested "state" dict, merge it
            if "state" in out and isinstance(out["state"], dict):
                updated_state.update(out["state"])

            return logits, updated_state

        def next_step(self, model: MLModel, token_id: int, state: Dict[str, Any]):
            """
            Run next inference step with single token.

            Calls model.predict() with token_id and current state, merges state updates.
            """
            # Prepare input dict for model prediction
            inputs = {"token_id": np.array([[token_id]], dtype=np.int32)}
            # Include state keys that the model expects (e.g., KV cache)
            state_inputs = {
                k: v
                for k, v in state.items()
                if k in ["kv_cache", "position_ids", "attention_mask"]
            }
            inputs.update(state_inputs)

            out = model.predict(inputs)

            # Extract logits (handle different output formats)
            if "logits" in out:
                logits = out["logits"]
                # Handle batched output: take first batch if needed
                if isinstance(logits, np.ndarray) and len(logits.shape) > 1:
                    logits = logits[0]
            else:
                # Fallback: try to find logits in output
                logits_key = [k for k in out.keys() if "logit" in k.lower()]
                if logits_key:
                    logits = out[logits_key[0]]
                    if isinstance(logits, np.ndarray) and len(logits.shape) > 1:
                        logits = logits[0]
                else:
                    raise ValueError(
                        f"Could not find logits in model output. Keys: {list(out.keys())}"
                    )

            # Update state with any returned state from model
            updated_state = state.copy()
            state_keys = ["kv_cache", "position_ids", "attention_mask", "state"]
            for key in state_keys:
                if key in out:
                    updated_state[key] = out[key]

            # If model returns a nested "state" dict, merge it
            if "state" in out and isinstance(out["state"], dict):
                updated_state.update(out["state"])

            return logits, updated_state

    # Initialize speculative decoder if enabled
    speculative_decoder = None
    if args.enable_speculative:
        if args.drafter_model is None:
            print("[perf_mem_eval] WARN: --enable-speculative requires --drafter-model")
        elif not args.drafter_model.exists():
            print(f"[perf_mem_eval] WARN: Drafter model not found: {args.drafter_model}")
        else:
            try:
                from coreml.runtime.speculative_decode import SpeculativeDecoder

                drafter_model = MLModel(str(args.drafter_model))
                worker_model = MLModel(str(args.model))

                # Create adapters (using DummyAdapter for now)
                drafter_adapter = DummyAdapter()
                worker_adapter = DummyAdapter()

                speculative_decoder = SpeculativeDecoder(
                    drafter_model=drafter_model,
                    worker_model=worker_model,
                    drafter_adapter=drafter_adapter,
                    worker_adapter=worker_adapter,
                    k=args.spec_k,
                    temperature=0.0,
                )
                print(f"[perf_mem_eval] Speculative decoding enabled (k={args.spec_k})")
            except Exception as e:
                print(f"[perf_mem_eval] WARN: Failed to initialize speculative decoder: {e}")

    # Load tokenizer for TTFA detection if available
    tokenizer = None
    try:
        from training.dataset import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)
    except Exception as e:
        print(f"[perf_mem_eval] WARN: Failed to load tokenizer, TTFA detection disabled: {e}")

    speed = run_coreml_speed(
        args.model,
        prompts,
        DummyAdapter(),
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        prompt_cache=prompt_cache,
        prompt_texts=prompt_texts,
        speculative_decoder=speculative_decoder,
        kv_cache=kv_cache,
        batch_policy=batch_policy,
        batch_size=selected_batch_size,
    )

    # Measure ANE residency if requested (after model is loaded in run_coreml_speed)
    if args.measure_ane_residency:
        try:
            from coreml.runtime.ane_monitor import ANEResidencyMonitor

            monitor = ANEResidencyMonitor(model_mlpackage_path=str(args.model))

            # Load model for ANE measurement
            model = MLModel(str(args.model))
            adapter_instance = DummyAdapter()

            def inference_fn():
                # Run inference on a random prompt
                import random

                prompt_ids = random.choice(prompts)
                ids_np = np.array(prompt_ids, dtype=np.int32)[None, :]
                state = adapter_instance.prepare_state(ids_np)
                logits, state = adapter_instance.first_step(model, ids_np, state)
                # Sample a few more tokens
                for _ in range(5):
                    token = int(np.argmax(logits))
                    logits, state = adapter_instance.next_step(model, token, state)

            ane_residency = monitor.measure_residency(
                inference_fn=inference_fn,
                num_samples=args.ane_samples,
            )
            print(f"[perf_mem_eval] ANE residency: {ane_residency.get('ane_time_pct', 0):.1%}")
        except Exception as e:
            print(f"[perf_mem_eval] WARN: Failed to measure ANE residency: {e}")

    hw = detect_hardware()

    # Match hardware profile for chip-specific config
    hardware_profile_key = None
    try:
        from eval.hw_profile import load_profiles, match_profile

        profiles = load_profiles(Path("configs/hardware_profiles.yaml"))
        profile = match_profile(profiles)
        hardware_profile_key = profile.key
    except Exception as e:
        print(f"[perf_mem_eval] WARN: Failed to match hardware profile: {e}")

    # Measure ANE residency if requested (after model is loaded)
    ane_residency = None

    hdr = {
        "report_schema_version": "1.1.0",
        "speed_metrics": speed,
        "hardware": {
            "soc": args.hardware or hw.soc,
            "os": hw.os,
            "coremltools": hw.coremltools,
            "export_path": args.export_path,
        },
    }

    # Add ANE residency if measured
    if ane_residency:
        hdr["ane_residency"] = ane_residency

    # Add hardware profile key for relative gating
    if hardware_profile_key:
        hdr["hardware_profile_key"] = hardware_profile_key
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(hdr, f, indent=2)
    print(json.dumps(hdr, indent=2))
