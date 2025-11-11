# conversion/export_onnx.py
# @author: @darianrosebrook

import json
import os
import typer
import torch
import torch.nn as nn
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

app = typer.Typer()


class DecodeWrapper(nn.Module):
    """Wrapper for decoder-only export with KV cache inputs/outputs."""
    def __init__(self, model: StudentLM, n_layers: int, n_kv_heads: int, d_head: int):
        super().__init__()
        self.model = model
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

    def forward(self, input_ids: torch.Tensor, *kv_caches) -> tuple:
        """Decode forward with KV cache inputs.
        
        Args:
            input_ids: [B, 1] single token
            kv_caches: n_layers * 2 tensors (k_cache, v_cache) each [B, Hk, T_cache, Dh]
                      Empty caches (T_cache=0) should be provided as empty tensors
        """
        kv_list = []
        for i in range(self.n_layers):
            k_idx = i * 2
            v_idx = i * 2 + 1
            if k_idx < len(kv_caches) and v_idx < len(kv_caches):
                k_cache = kv_caches[k_idx]
                v_cache = kv_caches[v_idx]
                # Check if cache is empty (T_cache=0)
                if k_cache.shape[2] == 0:
                    kv_list.append(None)
                else:
                    kv_list.append((k_cache, v_cache))
            else:
                kv_list.append(None)
        
        logits, updated_caches = self.model.forward_decode(input_ids, kv_list, pos=0)
        
        # Flatten updated caches for ONNX export
        outputs = [logits]
        for k_cache, v_cache in updated_caches:
            outputs.extend([k_cache, v_cache])
        return tuple(outputs)


@app.command()
def main(config: str = "conversion/shape_sets.json", mode: str = "both"):
    """Export ONNX models for prefill and/or decode modes.
    
    Args:
        config: Path to shape_sets.json
        mode: Export mode - "prefill", "decode", or "both" (default)
    """
    # Read sequence lengths from shape_sets.json
    try:
        shape_data = json.load(open(config, "r"))
        seqs = [item["seq"] for item in shape_data if "seq" in item]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Fallback to default sequence lengths
        seqs = [2048, 4096, 8192]

    model = StudentLM(ModelCfg())
    model.eval()

    os.makedirs("coreml", exist_ok=True)
    os.makedirs("artifacts/onnx", exist_ok=True)

    cfg = model.cfg

    # Export prefill models (full sequence)
    if mode in ["prefill", "both"]:
        for T in seqs:
            dummy = torch.zeros((1, T), dtype=torch.int32)
            path = f"artifacts/onnx/student_prefill_T{T}.onnx"
            torch.onnx.export(
                model,
                (dummy, None),
                path,
                input_names=["input_ids", "attn_mask"],
                output_names=["logits"],
                opset_version=19,
                do_constant_folding=True,
                dynamic_axes=None,  # enumerated shapes only
            )
            print(f"Exported prefill model: {path}")

    # Export decode models (single token with KV cache)
    if mode in ["decode", "both"]:
        decode_model = DecodeWrapper(model, cfg.n_layers, cfg.n_kv_heads, cfg.d_head)
        decode_model.eval()

        # Create dummy KV cache inputs for export
        # Each layer has k_cache [B, Hk, T_cache, Dh] and v_cache [B, Hk, T_cache, Dh]
        # Use a reasonable cache size for export (can be dynamic in runtime)
        # Start with empty cache (T_cache=0) - ONNX will infer dynamic dimension
        cache_len = 1  # Start with minimal cache for export shape inference
        
        dummy_input_ids = torch.zeros((1, 1), dtype=torch.int32)
        dummy_kv_inputs = []
        
        for _ in range(cfg.n_layers):
            # k_cache: [B, Hk, T_cache, Dh] - T_cache can be dynamic
            dummy_k_cache = torch.zeros((1, cfg.n_kv_heads, cache_len, cfg.d_head), dtype=torch.float16)
            # v_cache: [B, Hk, T_cache, Dh]
            dummy_v_cache = torch.zeros((1, cfg.n_kv_heads, cache_len, cfg.d_head), dtype=torch.float16)
            dummy_kv_inputs.extend([dummy_k_cache, dummy_v_cache])

        # Input names: input_ids + per-layer k_cache/v_cache
        input_names = ["input_ids"]
        for i in range(cfg.n_layers):
            input_names.extend([f"k_cache_{i}", f"v_cache_{i}"])

        # Output names: logits + updated per-layer k_cache/v_cache
        output_names = ["logits"]
        for i in range(cfg.n_layers):
            output_names.extend([f"k_cache_out_{i}", f"v_cache_out_{i}"])

        path = f"artifacts/onnx/student_decode.onnx"
        torch.onnx.export(
            decode_model,
            (dummy_input_ids, *dummy_kv_inputs),
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                **{f"k_cache_{i}": {2: "cache_len"} for i in range(cfg.n_layers)},
                **{f"v_cache_{i}": {2: "cache_len"} for i in range(cfg.n_layers)},
            },
        )
        print(f"Exported decode model: {path}")


if __name__ == "__main__":
    app()
