"""
Export PyTorch student model to TorchScript with prefill/decoder split.

This is the production export path - PyTorch → CoreML (not ONNX).

Usage:
  python -m conversion.export_pytorch --checkpoint models/student/checkpoints/latest.pt --out models/student/exported/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from infra.version_gate import check_export_versions


class PrefillWrapper(nn.Module):
    """Wrapper for prefill export (full sequence, no KV cache)."""

    def __init__(self, model: StudentLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        outputs = self.model(
            input_ids, attn_mask, return_halt_logits=getattr(self.model, "use_halt_head", False)
        )
        if isinstance(outputs, tuple):
            return outputs  # (logits, halt_logits) or (logits,)
        else:
            return (outputs,)  # Wrap single tensor in tuple


class DecodeWrapper(nn.Module):
    """Wrapper for decode export (single token with KV cache)."""

    def __init__(self, model: StudentLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, *kv_caches) -> tuple:
        """Decode forward with KV cache inputs.

        Args:
            input_ids: [B, 1] single token
            kv_caches: n_layers * 2 tensors (k_cache, v_cache) each [B, Hk, T_cache, Dh]
        """
        kv_list = []
        n_layers = self.model.cfg.n_layers
        for i in range(n_layers):
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

        outputs = self.model.forward_decode(
            input_ids,
            kv_list,
            pos=0,
            return_halt_logits=getattr(self.model, "use_halt_head", False),
        )

        if isinstance(outputs, tuple) and len(outputs) == 3:
            # (logits, updated_caches, halt_logits)
            logits, updated_caches, halt_logits = outputs
            # Flatten updated caches for export
            export_outputs = [logits, halt_logits]
            for k_cache, v_cache in updated_caches:
                export_outputs.extend([k_cache, v_cache])
            return tuple(export_outputs)
        else:
            # (logits, updated_caches)
            logits, updated_caches = outputs
            # Flatten updated caches for export
            export_outputs = [logits]
            for k_cache, v_cache in updated_caches:
                export_outputs.extend([k_cache, v_cache])
            return tuple(export_outputs)


def export_prefill(
    model: StudentLM, example_input: torch.Tensor, output_path: Path, enumerated_T: list
):
    """Export prefill model (full sequence)."""
    wrapper = PrefillWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        # Trace with just input_ids (attn_mask is optional)
        # Note: If wrapper returns dict, TorchScript will preserve structure
        traced = torch.jit.trace(wrapper, example_input)
        traced.eval()

        # If wrapper returns dict, we need to handle it differently
        # For now, keep returning tensor directly - CoreML will name it
        # The output name stabilization happens in convert_coreml.py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"[export_pytorch] Saved prefill model: {output_path}")

    # Create contract for prefill
    outputs = [{"name": "logits", "dtype": "float16", "shape": ["B", "T", "V"]}]

    # Add halt logits if model supports it
    if getattr(model, "use_halt_head", False):
        outputs.append({"name": "halt_logits", "dtype": "float16", "shape": ["B", "2"]})

    contract = {
        "inputs": [
            {"name": "input_ids", "dtype": "int32", "shape": ["B", "T"]},
            {"name": "attn_mask", "dtype": "int32", "shape": ["B", "T"], "optional": True},
        ],
        "outputs": outputs,
        "mode": "prefill",
        "enumerated_T": enumerated_T,
        "use_halt_head": getattr(model, "use_halt_head", False),
    }
    contract_path = output_path.parent / f"{output_path.stem}_contract.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"[export_pytorch] Saved contract: {contract_path}")

    return traced


def export_decode(model: StudentLM, output_path: Path, n_layers: int, n_kv_heads: int, d_head: int):
    """Export decode model (single token with KV cache)."""
    wrapper = DecodeWrapper(model)
    wrapper.eval()

    # Create example inputs: single token + empty KV caches
    example_input_ids = torch.zeros((1, 1), dtype=torch.int32)
    example_kv_inputs = []
    for _ in range(n_layers):
        k_cache = torch.zeros((1, n_kv_heads, 0, d_head), dtype=torch.float16)  # Empty cache
        v_cache = torch.zeros((1, n_kv_heads, 0, d_head), dtype=torch.float16)
        example_kv_inputs.extend([k_cache, v_cache])

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_input_ids, *example_kv_inputs))
        traced.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"[export_pytorch] Saved decode model: {output_path}")

    # Create contract for decode
    outputs = [
        {"name": "logits", "dtype": "float16", "shape": ["B", 1, "V"]},
    ]

    kv_cache_start_idx = 1  # After logits
    # Add halt logits if model supports it
    if getattr(model, "use_halt_head", False):
        outputs.append({"name": "halt_logits", "dtype": "float16", "shape": ["B", "2"]})
        kv_cache_start_idx = 2  # After logits and halt_logits

    contract = {
        "inputs": [
            {"name": "input_ids", "dtype": "int32", "shape": ["B", 1]},
        ],
        "outputs": outputs,
        "mode": "decode",
        "kv_cache_outputs": n_layers * 2,  # k_cache and v_cache per layer
        "kv_cache_start_idx": kv_cache_start_idx,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "use_halt_head": getattr(model, "use_halt_head", False),
    }
    # Add KV cache inputs to contract
    for i in range(n_layers):
        contract["inputs"].extend(
            [
                {
                    "name": f"k_cache_{i}",
                    "dtype": "float16",
                    "shape": ["B", n_kv_heads, "T_cache", d_head],
                },
                {
                    "name": f"v_cache_{i}",
                    "dtype": "float16",
                    "shape": ["B", n_kv_heads, "T_cache", d_head],
                },
            ]
        )

    contract_path = output_path.parent / f"{output_path.stem}_contract.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"[export_pytorch] Saved contract: {contract_path}")

    return traced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path (.pt)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--mode",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Export mode (default: both)",
    )
    ap.add_argument(
        "--seq", type=int, default=2048, help="Example sequence length for prefill tracing"
    )
    ap.add_argument(
        "--enumerated-T",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192, 16384],
        help="Enumerated sequence lengths",
    )
    ap.add_argument(
        "--toy", action="store_true", help="Handle toy checkpoint schema (normalize arch config)"
    )
    args = ap.parse_args()

    # Version compatibility check - skip for toy models (they're for testing)
    if not args.toy:
        try:
            check_export_versions()
        except RuntimeError as e:
            print(f"[export_pytorch] ERROR: Version check failed: {e}")
            print("\nTo fix this issue:")
            print("1. Install Python 3.11: brew install python@3.11")
            print("2. Use Python 3.11 for export: python3.11 -m conversion.export_pytorch ...")
            print(
                "3. For toy models only, use --toy flag to bypass: python -m conversion.export_pytorch --toy ..."
            )
            print("\nSee docs/DEPLOYMENT.md for detailed environment setup instructions.")
            sys.exit(1)
    else:
        print("[export_pytorch] ⚠️  Skipping version check for toy model (testing mode)")

    # Load model safely with structure validation
    from training.safe_checkpoint_loading import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(args.checkpoint, map_location="cpu")

    # Load config from checkpoint - required for correct architecture
    if "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {args.checkpoint} missing 'config' field. "
            "Cannot determine model architecture. Please use a checkpoint saved with config."
        )

    config_data = checkpoint["config"]
    arch_cfg = config_data.get("arch", {})

    if not arch_cfg:
        raise ValueError(
            f"Checkpoint {args.checkpoint} config missing 'arch' section. "
            "Cannot determine model architecture."
        )

    # Handle toy checkpoint schema
    if args.toy:
        # Normalize toy arch config to production export spec
        export_spec = {
            "d_model": arch_cfg.get("hidden_size") or arch_cfg.get("d_model", 128),
            "n_layers": arch_cfg.get("n_layers", 2),
            "n_heads": arch_cfg.get("n_heads", 4),
            "n_kv_heads": arch_cfg.get("n_kv_heads", 2),
            "d_head": arch_cfg.get("d_head")
            or (arch_cfg.get("hidden_size", 128) // arch_cfg.get("n_heads", 4)),
            "vocab_size": arch_cfg.get("vocab_size", 512),
            "rope_theta": arch_cfg.get("rope_theta", 10000.0),
            "rope_scaling": arch_cfg.get("rope_scaling", "dynamic"),
            "dropout": arch_cfg.get("dropout", 0.0),
        }
        # Use export_spec values
        cfg = ModelCfg(
            d_model=export_spec["d_model"],
            n_layers=export_spec["n_layers"],
            n_heads=export_spec["n_heads"],
            n_kv_heads=export_spec["n_kv_heads"],
            d_head=export_spec["d_head"],
            vocab_size=export_spec["vocab_size"],
            rope_theta=export_spec["rope_theta"],
            rope_scaling=export_spec["rope_scaling"],
            dropout=export_spec["dropout"],
        )
    else:
        # Production checkpoint schema - require all essential fields
        required_fields = ["d_model", "n_layers", "n_heads", "n_kv_heads", "d_head", "vocab_size"]
        missing_fields = [f for f in required_fields if f not in arch_cfg]
        if missing_fields:
            raise ValueError(
                f"Checkpoint config missing required architecture fields: {missing_fields}. "
                f"Found fields: {list(arch_cfg.keys())}"
            )

        cfg = ModelCfg(
            d_model=arch_cfg["d_model"],
            n_layers=arch_cfg["n_layers"],
            n_heads=arch_cfg["n_heads"],
            n_kv_heads=arch_cfg["n_kv_heads"],
            d_head=arch_cfg["d_head"],
            vocab_size=arch_cfg["vocab_size"],
            rope_theta=arch_cfg.get("rope_theta", 10000.0),
            rope_scaling=arch_cfg.get("rope_scaling", "dynamic"),
            dropout=arch_cfg.get("dropout", 0.0),
        )

    # Also load model flags from config if available
    use_self_evaluation = config_data.get("use_self_evaluation", False)
    use_halt_head = config_data.get("use_halt_head", False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = StudentLM(cfg, use_self_evaluation=use_self_evaluation, use_halt_head=use_halt_head)

    # Load state dict with strict=True to catch mismatches
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load model state dict. Architecture mismatch detected. "
            f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, "
            f"n_kv_heads={cfg.n_kv_heads}, d_head={cfg.d_head}, vocab_size={cfg.vocab_size}. "
            f"Original error: {e}"
        )
    model.eval()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export prefill models
    if args.mode in ["prefill", "both"]:
        for T in args.enumerated_T:
            example_input = torch.zeros((1, T), dtype=torch.int32)
            prefill_path = output_dir / f"student_prefill_T{T}.pt"
            export_prefill(model, example_input, prefill_path, args.enumerated_T)

    # Export decode model
    if args.mode in ["decode", "both"]:
        decode_path = output_dir / "student_decode.pt"
        export_decode(model, decode_path, cfg.n_layers, cfg.n_kv_heads, cfg.d_head)

    print(f"[export_pytorch] ✅ Export complete: {output_dir}")


if __name__ == "__main__":
    main()
