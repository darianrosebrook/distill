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
    """Generate PyTorch probes from a TorchScript model."""
    model = torch.jit.load(model_path)
    model.eval()
    
    probe = Probe()
    # For a simple block, attach probes at key points
    # Note: TorchScript doesn't support hooks, so we'll need to modify the model
    # For now, just run forward and save the output
    with torch.no_grad():
        output = model(example_input)
    
    # Save output as probe
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
    ap.add_argument('--pt-model', required=True, help='PyTorch TorchScript model path')
    ap.add_argument('--ml-model', required=True, help='CoreML mlpackage path')
    ap.add_argument('--out', required=True, help='Output directory for probes')
    ap.add_argument('--seq', type=int, default=128, help='Sequence length')
    ap.add_argument('--dmodel', type=int, default=64, help='Model dimension')
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


if __name__ == '__main__':
    main()

