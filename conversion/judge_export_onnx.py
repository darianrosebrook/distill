# conversion/judge_export_onnx.py
# Judge-specific ONNX export with enumerated short shapes
# @author: @darianrosebrook

import json
import os
import typer
import torch
from pathlib import Path
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

try:
    import yaml
except ImportError:
    yaml = None

app = typer.Typer()


def load_judge_config(config_path: str) -> ModelCfg:
    """Load judge model configuration from YAML file."""
    if yaml is None:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Judge config not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    arch_cfg = cfg.get("arch", {})

    # Extract model configuration from arch section
    judge_cfg = ModelCfg(
        d_model=arch_cfg.get("d_model", 2048),
        n_layers=arch_cfg.get("n_layers", 24),
        n_heads=arch_cfg.get("n_heads", 16),
        n_kv_heads=arch_cfg.get("n_kv_heads", 4),
        d_head=arch_cfg.get("d_head", 128),
        vocab_size=arch_cfg.get("vocab_size", 32000),
        rope_theta=arch_cfg.get("rope_theta", 10000.0),
        rope_scaling=arch_cfg.get("rope_scaling", "dynamic"),
        dropout=arch_cfg.get("dropout", 0.0),
    )

    return judge_cfg


@app.command()
def main(config: str = "conversion/shape_sets.json", judge_config: str = "configs/judge_4b.yaml"):
    """Export Judge model to ONNX with enumerated short shapes.

    Args:
        config: Path to shape_sets.json (for sequence lengths)
        judge_config: Path to judge model config YAML
    """
    # Load judge configuration from YAML - required for correct architecture
    judge_cfg = load_judge_config(judge_config)
    print(f"[judge_export_onnx] Loaded judge config from: {judge_config}")
    print(
        f"[judge_export_onnx] Model config: d_model={judge_cfg.d_model}, n_layers={judge_cfg.n_layers}, "
        f"n_heads={judge_cfg.n_heads}, n_kv_heads={judge_cfg.n_kv_heads}, "
        f"d_head={judge_cfg.d_head}, vocab_size={judge_cfg.vocab_size}")

    # Judge uses short enumerated shapes: 512, 1024, 2048
    # (Judge reads summaries/claims, not full transcripts)
    # Try to get from config, fallback to defaults
    try:
        with open(config, 'r') as f:
            shape_data = json.load(f)
            judge_seqs = shape_data.get("judge_sequences", [512, 1024, 2048])
    except Exception:
        # Fallback to defaults
        judge_seqs = [512, 1024, 2048]

    model = StudentLM(judge_cfg)
    model.eval()

    os.makedirs("artifacts/onnx/judge", exist_ok=True)

    for T in judge_seqs:
        dummy = torch.zeros((1, T), dtype=torch.int32)
        path = f"artifacts/onnx/judge/judge_T{T}.onnx"
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
        print(f"Exported judge model: {path}")


if __name__ == "__main__":
    app()
