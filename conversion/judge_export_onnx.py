# conversion/judge_export_onnx.py
# Judge-specific ONNX export with enumerated short shapes
# @author: @darianrosebrook

import json
import os
import typer
import torch
import torch.nn as nn
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

app = typer.Typer()


@app.command()
def main(config: str = "conversion/shape_sets.json", judge_config: str = "configs/judge_4b.yaml"):
    """Export Judge model to ONNX with enumerated short shapes.
    
    Args:
        config: Path to shape_sets.json (for sequence lengths)
        judge_config: Path to judge model config
    """
    # Judge uses short enumerated shapes: 512, 1024, 2048
    # (Judge reads summaries/claims, not full transcripts)
    judge_seqs = [512, 1024, 2048]
    
    # PLACEHOLDER: Load judge-specific config
    # Judge config should specify smaller model size (3-4B or 7B)
    judge_cfg = ModelCfg(
        d_model=2048,  # Smaller for 3-4B model
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_head=128,
        vocab_size=32000,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0
    )
    
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

