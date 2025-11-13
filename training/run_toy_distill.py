"""
Run toy distillation training for end-to-end pipeline testing.

Trains a tiny student model using deterministic teacher logits.
Saves checkpoint with unified schema compatible with export_pytorch.py --toy.

Usage:
    python -m training.run_toy_distill --in toy_kd.jsonl --out toy.ckpt --epochs 2
"""
from training.utils import sha256_state_dict
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import KDDataset, collate_kd_batch
from training.losses import combined_kd_loss
from training.teacher_stub_toy import teacher_logits


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]  # Short SHA
    except Exception:
        return "unknown"


# Import shared SHA256 utility


def main():
    ap = argparse.ArgumentParser(description="Toy distillation training")
    ap.add_argument('--in', '--input', dest='input_path', required=True,
                    help='Input KD dataset JSONL path')
    ap.add_argument('--out', '--output', dest='output_path', required=True,
                    help='Output checkpoint path')
    ap.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    ap.add_argument('--mps', type=int, default=0, choices=[0, 1],
                    help='Use MPS (Metal Performance Shaders) if available (0=no, 1=yes)')
    ap.add_argument('--micro-batch-size', type=int,
                    default=4, help='Micro batch size')
    ap.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    ap.add_argument('--vocab-size', type=int,
                    default=512, help='Vocabulary size')
    ap.add_argument('--d-model', type=int, default=128, help='Model dimension')
    ap.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    ap.add_argument('--n-heads', type=int, default=4,
                    help='Number of attention heads')
    ap.add_argument('--n-kv-heads', type=int, default=2,
                    help='Number of KV heads (GQA)')
    ap.add_argument('--max-seq-len', type=int, default=256,
                    help='Maximum sequence length')
    ap.add_argument('--tokenizer', type=str, default='models/student/tokenizer',
                    help='Tokenizer path')
    args = ap.parse_args()

    # Setup device
    if args.mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"[run_toy_distill] Starting toy distillation training")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Micro batch size: {args.micro_batch_size}")

    # Create model config
    cfg = ModelCfg(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        vocab_size=args.vocab_size,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )

    # Create model
    model = StudentLM(cfg).to(device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create dataset
    try:
        dataset = KDDataset(
            jsonl_path=args.input_path,
            tokenizer_path=args.tokenizer,
            max_seq_length=args.max_seq_len,
            teacher_logits_available=False,  # We'll generate on-the-fly
        )
    except Exception as e:
        print(f"[run_toy_distill] ERROR: Failed to load dataset: {e}")
        print(f"  Tokenizer path: {args.tokenizer}")
        print(f"  Dataset path: {args.input_path}")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_kd_batch,
        num_workers=0,  # Avoid multiprocessing issues in toy tests
    )

    # Training loop
    total_samples = len(dataset)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs

    print(f"[run_toy_distill] Dataset: {total_samples} samples")
    print(f"[run_toy_distill] Steps per epoch: {steps_per_epoch}")
    print(f"[run_toy_distill] Total steps: {total_steps}")

    step = 0
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Clamp token IDs to fit toy vocab size (tokenizer may have larger vocab)
            pre_violate = (input_ids.ge(args.vocab_size)
                           | input_ids.lt(0)).any().item()
            input_ids = torch.clamp(input_ids, 0, args.vocab_size - 1)
            labels = torch.clamp(labels, 0, args.vocab_size - 1)
            if pre_violate and step % 50 == 0:
                print(
                    f"[run_toy_distill] ⚠ clamped token ids to vocab_size={args.vocab_size}", flush=True)

            # Forward pass
            student_logits = model(input_ids)

            # Generate teacher logits on-the-fly
            teacher_logits_tensor = teacher_logits(
                input_ids, vocab_size=args.vocab_size).to(device)

            # Compute loss
            loss_dict = combined_kd_loss(
                student_logits=student_logits.float(),
                teacher_logits=teacher_logits_tensor,
                teacher_targets=teacher_logits_tensor.argmax(dim=-1),
                ground_truth_targets=labels.to(device),
                kl_weight=0.5,
                ce_teacher_weight=0.3,
                ce_ground_truth_weight=0.2,
                kd_temperature=2.0,
            )

            loss = loss_dict["total"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0:
                print(
                    f"[run_toy_distill] Step {step}/{total_steps}: loss={loss.item():.4f}")

    # Save checkpoint with unified schema
    model.eval()
    state_dict = model.state_dict()
    state_sha256 = sha256_state_dict(state_dict)
    git_sha = get_git_sha()

    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
            "arch": {
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "n_kv_heads": args.n_kv_heads,
                "d_model": args.d_model,
                "d_head": cfg.d_head,
                # Alias for d_model (used by --toy flag)
                "hidden_size": args.d_model,
                "vocab_size": args.vocab_size,
                "rope_theta": cfg.rope_theta,
                "rope_scaling": cfg.rope_scaling,
                "dropout": cfg.dropout,
                "gqa": args.n_heads // args.n_kv_heads if args.n_kv_heads > 0 else 1,
                "max_seq_len": args.max_seq_len,
            },
            "tokenizer": {
                "type": "toy_bpe",
                "vocab_size": args.vocab_size,
            },
            "dtype": "fp32",
        },
        "meta": {
            "trainer": "toy-distill",
            "epoch": args.epochs,
            "step": step,
            "git_sha": git_sha,
            "sha256_state": state_sha256,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"[run_toy_distill] ✅ Training complete")
    print(f"  Checkpoint saved: {output_path}")
    print(f"  Model SHA256: {state_sha256[:16]}...")
    print(f"  Git SHA: {git_sha}")


if __name__ == '__main__':
    main()
