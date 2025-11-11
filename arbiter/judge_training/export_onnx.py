# arbiter/judge_training/export_onnx.py
# Judge model ONNX export with enumerated shapes
# @author: @darianrosebrook

import os
import typer
import torch
from transformers import AutoTokenizer
from .model import MultiTaskJudge

app = typer.Typer()


@app.command()
def main(ckpt: str, out_dir: str = "arbiter/judge_training/artifacts/onnx", seq_lens: str = "256,512,1024"):
    os.makedirs(out_dir, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu")
    hf_name = state["hf_name"]
    clauses = state["clauses"]
    model = MultiTaskJudge(hf_name, num_clauses=len(clauses))
    model.load_state_dict(state["model"])
    model.eval()

    tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    for T in [int(x) for x in seq_lens.split(",")]:
        dummy = tok("p", "c", max_length=T, padding="max_length", truncation=True, return_tensors="pt")
        path = os.path.join(out_dir, f"judge_T{T}.onnx")
        torch.onnx.export(
            model.encode_once,
            (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids")),
            path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["score", "clause_logits"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=None
        )
        print(f"Exported {path}")


if __name__ == "__main__":
    app()

