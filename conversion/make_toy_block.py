"""
Builds a single transformer block for parity testing.

This creates a real transformer block (QKV proj + attention + MLP) that can be
converted to CoreML for parity_full testing without requiring a full model.

Usage:
  python -m conversion.make_toy_block --dmodel 64 --nheads 4 --out models/toy_block.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        return self.w2(F.silu(a) * b)


class MultiHeadAttention(nn.Module):
    """Simplified attention for parity testing (no GQA, no RoPE)."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        
        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        k = self.wk(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        v = self.wv(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        
        # Attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,T]
        attn = F.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, v)  # [B,H,T,Dh]
        
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.wo(y)


class TransformerBlock(nn.Module):
    """Single transformer block for parity testing."""
    def __init__(self, d_model: int, n_heads: int, mlp_hidden_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(d_model * mlp_hidden_mult))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dmodel', type=int, default=64, help='Model dimension')
    ap.add_argument('--nheads', type=int, default=4, help='Number of attention heads')
    ap.add_argument('--seq', type=int, default=128, help='Sequence length')
    ap.add_argument('--out', type=str, default='models/toy_block.pt', help='Output TorchScript path')
    args = ap.parse_args()

    model = TransformerBlock(d_model=args.dmodel, n_heads=args.nheads)
    model.eval()

    # Create example input
    example_input = torch.randn((1, args.seq, args.dmodel), dtype=torch.float32)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.eval()

    # Save TorchScript
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.out)
    print(f"[make_toy_block] Saved transformer block: {args.out} (d={args.dmodel}, heads={args.nheads}, T={args.seq})")


if __name__ == '__main__':
    main()








