"""
Generate PyTorch and CoreML probes for parity comparison.

Usage:
  python -m coreml.probes.generate_probes --pt-model models/toy_block.pt --ml-model coreml/artifacts/toy_block/model.mlpackage --out coreml/probes/toy_block
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from coreml.probes.fx_probe_points import Probe, attach


def generate_pytorch_probes(model_path: str, example_input: torch.Tensor, output_path: str):
    """
    Generate PyTorch probes from a model (TorchScript or regular PyTorch).

    Attempts to load as regular PyTorch model first to support hook-based
    intermediate layer extraction. Falls back to TorchScript if needed.
    """
    model = None
    is_torchscript = False

    # Try to load as regular PyTorch model first (supports hooks)
    try:
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            # Checkpoint format - need model architecture
            # Try to infer from checkpoint config
            from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

            cfg = None
            if "config" in checkpoint:
                config_data = checkpoint["config"]
                arch_cfg = config_data.get("arch", {})
                cfg = ModelCfg(
                    d_model=arch_cfg.get("d_model", 4096),
                    n_layers=arch_cfg.get("n_layers", 32),
                    n_heads=arch_cfg.get("n_heads", 32),
                    n_kv_heads=arch_cfg.get("n_kv_heads", 8),
                    d_head=arch_cfg.get("d_head", 128),
                    vocab_size=arch_cfg.get("vocab_size", 32000),
                )
            else:
                # Use defaults if config not available
                cfg = ModelCfg()

            model = StudentLM(cfg)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print("[generate_probes] Loaded as regular PyTorch model (supports hooks)")
        elif isinstance(checkpoint, dict) and any(
            k.startswith("blocks.") or k.startswith("embed") for k in checkpoint.keys()
        ):
            # State dict format - try to load with default config
            from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

            cfg = ModelCfg()
            model = StudentLM(cfg)
            model.load_state_dict(checkpoint)
            model.eval()
            print(
                "[generate_probes] Loaded as regular PyTorch model from state dict (supports hooks)"
            )
        else:
            raise ValueError("Unknown checkpoint format")
    except Exception as e:
        # Fall back to TorchScript
        print(f"[generate_probes] Could not load as regular PyTorch model: {e}")
        print("[generate_probes] Falling back to TorchScript (hooks not supported)")
        try:
            model = torch.jit.load(model_path)
            model.eval()
            is_torchscript = True
        except Exception as e2:
            raise RuntimeError(f"Failed to load model as PyTorch or TorchScript: {e2}")

    probe = Probe()

    if is_torchscript:
        # TorchScript doesn't support hooks - only save final output
        with torch.no_grad():
            output = model(example_input)
        probe.data["output"] = output.detach().float().cpu().numpy()
        print("[generate_probes] TorchScript model: saved final output only (hooks not supported)")
    else:
        # Regular PyTorch model - attach hooks for intermediate layer extraction
        try:
            # Try to attach probes at key points
            probe = attach(model, probe)

            # Run forward pass to capture intermediate outputs
            with torch.no_grad():
                output = model(example_input)

            # Also save final output
            if isinstance(output, tuple):
                probe.data["output"] = output[0].detach().float().cpu().numpy()
            else:
                probe.data["output"] = output.detach().float().cpu().numpy()

            print(
                f"[generate_probes] Captured {len(probe.data)} probe points: {list(probe.data.keys())}"
            )

            # Clean up hooks (after data is captured)
            for hook in probe.hooks:
                hook.remove()
            probe.hooks.clear()
        except Exception as e:
            # If attach fails (model structure doesn't match), fall back to output only
            print(f"[generate_probes] WARN: Could not attach hooks: {e}")
            print("[generate_probes] Falling back to final output only")
            with torch.no_grad():
                output = model(example_input)
            if isinstance(output, tuple):
                probe.data["output"] = output[0].detach().float().cpu().numpy()
            else:
                probe.data["output"] = output.detach().float().cpu().numpy()

    probe.dump(output_path)
    print(f"[generate_probes] Saved PyTorch probes: {output_path}")
    return probe


def generate_coreml_probes(mlpackage_path: str, example_input: np.ndarray, output_path: str):
    """Generate CoreML probes from an mlpackage."""
    import coremltools as ct

    ml = ct.models.MLModel(mlpackage_path)

    # Get input name from model spec
    spec = ml.get_spec()
    input_name = spec.description.input[0].name

    # Convert input to dict format
    input_dict = {input_name: example_input.astype(np.float32)}

    output = ml.predict(input_dict)

    # Save output as probe
    probe_data = {}
    for key, value in output.items():
        probe_data[key] = np.array(value).astype(np.float32)

    np.savez_compressed(output_path, **probe_data)
    print(f"[generate_probes] Saved CoreML probes: {output_path}")
    return probe_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-model", required=True, help="PyTorch TorchScript model path")
    ap.add_argument("--ml-model", required=True, help="CoreML mlpackage path")
    ap.add_argument("--out", required=True, help="Output directory for probes")
    ap.add_argument("--seq", type=int, default=128, help="Sequence length")
    ap.add_argument("--dmodel", type=int, default=64, help="Model dimension")
    args = ap.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate example input
    example_input = torch.randn((1, args.seq, args.dmodel), dtype=torch.float32)
    example_input_np = example_input.numpy()

    # Generate PyTorch probes
    pt_probe_path = output_dir / "toy_block_pt.npz"
    generate_pytorch_probes(args.pt_model, example_input, str(pt_probe_path))

    # Generate CoreML probes
    ml_probe_path = output_dir / "toy_block_ml.npz"
    generate_coreml_probes(args.ml_model, example_input_np, str(ml_probe_path))

    print(f"[generate_probes] ✅ Probe generation complete: {output_dir}")


if __name__ == "__main__":
    main()
