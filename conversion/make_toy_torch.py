"""
Builds a tiny PyTorch model to exercise PyTorch→CoreML conversion.

Model: Embedding → Linear → RMSNorm → SwiGLU → Linear (attention stub)

Usage:
  python -m conversion.make_toy_torch --seq 128 --vocab 256 --dmodel 64 --out models/toy_torch.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        return self.w2(torch.nn.functional.silu(a) * b)


class ToyTransformer(nn.Module):
    """Tiny transformer-like model for smoke testing PyTorch→CoreML."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm1 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_model * 2)
        self.norm2 = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.embed(input_ids)  # [B, T, D]
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128, help="Sequence length")
    ap.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    ap.add_argument("--dmodel", type=int, default=64, help="Model dimension")
    ap.add_argument(
        "--out", type=str, default="models/toy_torch.pt", help="Output TorchScript path"
    )
    args = ap.parse_args()

    model = ToyTransformer(vocab_size=args.vocab, d_model=args.dmodel)
    model.eval()

    # Create example input
    example_input = torch.zeros((1, args.seq), dtype=torch.int32)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.eval()

    # Save TorchScript
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(
        f"[make_toy_torch] Saved TorchScript model: {args.out} (T={args.seq}, V={args.vocab}, D={args.dmodel})"
    )


if __name__ == "__main__":
    main()
