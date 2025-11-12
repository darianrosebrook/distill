"""
Export PyTorch student model to TorchScript with prefill/decoder split.

This is the production export path - PyTorch → CoreML (not ONNX).

Usage:
  python -m conversion.export_pytorch --checkpoint models/student/checkpoints/latest.pt --out models/student/exported/
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


class PrefillWrapper(nn.Module):
    """Wrapper for prefill export (full sequence, no KV cache)."""
    def __init__(self, model: StudentLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(input_ids, attn_mask)


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
        
        logits, updated_caches = self.model.forward_decode(input_ids, kv_list, pos=0)
        
        # Flatten updated caches for export
        outputs = [logits]
        for k_cache, v_cache in updated_caches:
            outputs.extend([k_cache, v_cache])
        return tuple(outputs)


def export_prefill(model: StudentLM, example_input: torch.Tensor, output_path: Path, enumerated_T: list):
    """Export prefill model (full sequence)."""
    wrapper = PrefillWrapper(model)
    wrapper.eval()
    
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_input, None))
        traced.eval()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"[export_pytorch] Saved prefill model: {output_path}")
    
    # Create contract for prefill
    contract = {
        "inputs": [
            {"name": "input_ids", "dtype": "int32", "shape": ["B", "T"]},
            {"name": "attn_mask", "dtype": "int32", "shape": ["B", "T"], "optional": True}
        ],
        "outputs": [
            {"name": "logits", "dtype": "float16", "shape": ["B", "T", "V"]}
        ],
        "mode": "prefill",
        "enumerated_T": enumerated_T,
    }
    contract_path = output_path.parent / f"{output_path.stem}_contract.json"
    with open(contract_path, 'w') as f:
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
    contract = {
        "inputs": [
            {"name": "input_ids", "dtype": "int32", "shape": ["B", 1]},
        ],
        "outputs": [
            {"name": "logits", "dtype": "float16", "shape": ["B", 1, "V"]},
        ],
        "mode": "decode",
        "kv_cache_outputs": n_layers * 2,  # k_cache and v_cache per layer
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
    }
    # Add KV cache inputs to contract
    for i in range(n_layers):
        contract["inputs"].extend([
            {"name": f"k_cache_{i}", "dtype": "float16", "shape": ["B", n_kv_heads, "T_cache", d_head]},
            {"name": f"v_cache_{i}", "dtype": "float16", "shape": ["B", n_kv_heads, "T_cache", d_head]},
        ])
    
    contract_path = output_path.parent / f"{output_path.stem}_contract.json"
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"[export_pytorch] Saved contract: {contract_path}")
    
    return traced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Model checkpoint path (.pt)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--mode', choices=['prefill', 'decode', 'both'], default='both',
                    help='Export mode (default: both)')
    ap.add_argument('--seq', type=int, default=2048, help='Example sequence length for prefill tracing')
    ap.add_argument('--enumerated-T', nargs='+', type=int, default=[2048, 4096, 8192, 16384],
                    help='Enumerated sequence lengths')
    args = ap.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        cfg = ModelCfg()  # TODO: Load from checkpoint if available
    else:
        state_dict = checkpoint
        cfg = ModelCfg()
    
    model = StudentLM(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export prefill models
    if args.mode in ['prefill', 'both']:
        for T in args.enumerated_T:
            example_input = torch.zeros((1, T), dtype=torch.int32)
            prefill_path = output_dir / f"student_prefill_T{T}.pt"
            export_prefill(model, example_input, prefill_path, args.enumerated_T)
    
    # Export decode model
    if args.mode in ['decode', 'both']:
        decode_path = output_dir / "student_decode.pt"
        export_decode(model, decode_path, cfg.n_layers, cfg.n_kv_heads, cfg.d_head)
    
    print(f"[export_pytorch] ✅ Export complete: {output_dir}")


if __name__ == '__main__':
    main()








