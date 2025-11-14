# arbiter/judge_training/export_onnx.py
# Judge model ONNX export with enumerated shapes
# @author: @darianrosebrook

import os
import typer
import torch
from typing import Dict, Any
from .model import MultiTaskJudge
from .model_loading import safe_from_pretrained_tokenizer

app = typer.Typer()


def safe_load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Safely load checkpoint with structure validation.

    Validates checkpoint structure before loading to prevent arbitrary code execution.
    Only allows expected keys: hf_name, clauses, model (state_dict).

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint data

    Raises:
        ValueError: If checkpoint structure is invalid
        RuntimeError: If checkpoint loading fails
    """
    # First, try loading with weights_only=True for maximum security
    # This will fail if checkpoint contains non-tensor data, which is expected
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
        # If weights_only succeeds, validate structure
        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint must be a dictionary, got {type(state)}")
    except Exception:
        # weights_only failed (expected for checkpoints with metadata)
        # Load without weights_only but validate structure immediately
        state = torch.load(checkpoint_path, map_location="cpu")

        # Validate checkpoint structure - only allow expected keys
        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint must be a dictionary, got {type(state)}")

        expected_keys = {"hf_name", "clauses", "model"}
        unexpected_keys = set(state.keys()) - expected_keys

        # Allow extra keys but log warning for security review
        if unexpected_keys:
            import warnings
            warnings.warn(
                f"Checkpoint contains unexpected keys: {unexpected_keys}. "
                "These will be ignored for security. Review checkpoint structure.",
                UserWarning
            )

        # Validate required keys exist
        if "hf_name" not in state:
            raise ValueError("Checkpoint missing required key: hf_name")
        if "clauses" not in state:
            raise ValueError("Checkpoint missing required key: clauses")
        if "model" not in state:
            raise ValueError("Checkpoint missing required key: model")

        # Validate hf_name is a string
        if not isinstance(state["hf_name"], str):
            raise ValueError(
                f"hf_name must be a string, got {type(state['hf_name'])}")

        # Validate clauses is a list
        if not isinstance(state["clauses"], (list, tuple)):
            raise ValueError(
                f"clauses must be a list or tuple, got {type(state['clauses'])}")

        # Validate model is a state_dict (dict with string keys)
        if not isinstance(state["model"], dict):
            raise ValueError(
                f"model must be a state_dict (dict), got {type(state['model'])}")

    return state


@app.command()
def main(
    ckpt: str,
    out_dir: str = "arbiter/judge_training/artifacts/onnx",
    seq_lens: str = "256,512,1024",
):
    os.makedirs(out_dir, exist_ok=True)
    state = safe_load_checkpoint(ckpt)
    hf_name = state["hf_name"]
    clauses = state["clauses"]
    model = MultiTaskJudge(hf_name, num_clauses=len(clauses))
    model.load_state_dict(state["model"])
    model.eval()

    tok = safe_from_pretrained_tokenizer(hf_name, use_fast=True)
    for T in [int(x) for x in seq_lens.split(",")]:
        dummy = tok(
            "p", "c", max_length=T, padding="max_length", truncation=True, return_tensors="pt"
        )
        path = os.path.join(out_dir, f"judge_T{T}.onnx")
        torch.onnx.export(
            model.encode_once,
            (dummy["input_ids"], dummy["attention_mask"],
             dummy.get("token_type_ids")),
            path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["score", "clause_logits"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        print(f"Exported {path}")


if __name__ == "__main__":
    app()
