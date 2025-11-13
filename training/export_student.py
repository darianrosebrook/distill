"""
Export student model to TorchScript/ExportedProgram for CoreML conversion.

This is the production export path - PyTorch → CoreML (not ONNX).

Usage:
  python -m training.export_student --checkpoint models/student/checkpoints/latest.pt --out models/student/exported/
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


def export_torchscript(model: nn.Module, example_input: torch.Tensor, output_path: Path):
    """Export model as TorchScript."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.eval()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"[export_student] Saved TorchScript: {output_path}")
    return traced


def export_exported_program(model: nn.Module, example_input: torch.Tensor, output_path: Path):
    """Export model as torch.export ExportedProgram."""
    try:
        exported = torch.export.export(model, (example_input,))
    except AttributeError:
        raise RuntimeError("torch.export not available. Use PyTorch 2.0+ or export as TorchScript.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        torch.save(exported, f)
    print(f"[export_student] Saved ExportedProgram: {output_path}")
    return exported


def create_contract(cfg: ModelCfg, enumerated_T: list, output_path: Path):
    """Create contract.json with model specification."""
    contract = {
        "inputs": [
            {
                "name": "input_ids",
                "dtype": "int32",
                "shape": ["B", "T"]
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "dtype": "float16",
                "shape": ["B", "T", "V"]
            }
        ],
        "kv_precision": "fp16",
        "enumerated_T": enumerated_T,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "d_head": cfg.d_head,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "vocab_size": cfg.vocab_size,
    }
    
    contract_path = output_path / "contract.json"
    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"[export_student] Saved contract: {contract_path}")
    return contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Model checkpoint path (.pt)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--format', choices=['torchscript', 'exported', 'both'], default='both',
                    help='Export format (default: both)')
    ap.add_argument('--seq', type=int, default=2048, help='Example sequence length for tracing')
    ap.add_argument('--enumerated-T', nargs='+', type=int, default=[2048, 4096, 8192, 16384],
                    help='Enumerated sequence lengths for contract')
    args = ap.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Try to load config from checkpoint
    cfg = None
    model_arch = None
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        arch_cfg = config_data.get('arch', {})
        cfg = ModelCfg(
            d_model=arch_cfg.get('d_model', 4096),
            n_layers=arch_cfg.get('n_layers', 32),
            n_heads=arch_cfg.get('n_heads', 32),
            n_kv_heads=arch_cfg.get('n_kv_heads', 8),
            d_head=arch_cfg.get('d_head', 128),
            vocab_size=arch_cfg.get('vocab_size', 32000),
            rope_theta=arch_cfg.get('rope_theta', 10000.0),
            rope_scaling=arch_cfg.get('rope_scaling', 'dynamic'),
            dropout=arch_cfg.get('dropout', 0.0),
        )

        # Load model architecture flags
        if 'model_arch' in checkpoint:
            model_arch = checkpoint['model_arch']
            print(f"[export_student] Loaded model arch from checkpoint: {model_arch}")

        print("[export_student] Loaded config from checkpoint")
    
    if cfg is None:
        # Fallback to default config
        cfg = ModelCfg()
        print("[export_student] WARN: No config in checkpoint, using defaults")
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model with architecture flags
    use_halt_head = model_arch.get('use_halt_head', False) if model_arch else False
    use_self_evaluation = model_arch.get('use_self_evaluation', False) if model_arch else False

    model = StudentLM(cfg, use_halt_head=use_halt_head, use_self_evaluation=use_self_evaluation)
    model.load_state_dict(state_dict, strict=False)

    if use_halt_head:
        print("[export_student] Model loaded with halt head support")
    if use_self_evaluation:
        print("[export_student] Model loaded with self-evaluation support")
    model.eval()
    
    # Create example input
    example_input = torch.zeros((1, args.seq), dtype=torch.int32)
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export models
    if args.format in ['torchscript', 'both']:
        export_torchscript(model, example_input, output_dir / "student_fp16.pt")
    
    if args.format in ['exported', 'both']:
        try:
            export_exported_program(model, example_input, output_dir / "student_exported.pt")
        except RuntimeError as e:
            print(f"[export_student] WARN: {e}")
            print("[export_student] Falling back to TorchScript only")
    
    # Create contract
    create_contract(cfg, args.enumerated_T, output_dir)
    
    print(f"[export_student] ✅ Export complete: {output_dir}")


if __name__ == '__main__':
    main()

