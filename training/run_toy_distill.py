"""
Run toy distillation training for end-to-end pipeline testing.

Trains a tiny student model using deterministic teacher logits.
Saves checkpoint with unified schema compatible with export_pytorch.py --toy.

Usage:
    python -m training.run_toy_distill --in toy_kd.jsonl --out toy.ckpt --epochs 2
"""

from training.utils import sha256_state_dict
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import KDDataset, collate_kd_batch
from training.losses import combined_kd_loss
from training.teacher_stub_toy import teacher_logits
from training.teacher_stub_toy import eight_ball_teacher_logits


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8]  # Short SHA
    except Exception:
        return "unknown"


# Import shared SHA256 utility


def main():
    ap = argparse.ArgumentParser(description="Toy distillation training")
    ap.add_argument(
        "--in", "--input", dest="input_path", required=True, help="Input KD dataset JSONL path"
    )
    ap.add_argument(
        "--out", "--output", dest="output_path", required=True, help="Output checkpoint path"
    )
    ap.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    ap.add_argument(
        "--mps",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use MPS (Metal Performance Shaders) if available (0=no, 1=yes)",
    )
    ap.add_argument("--micro-batch-size", type=int, default=4, help="Micro batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--vocab-size", type=int, default=512, help="Vocabulary size")
    ap.add_argument("--d-model", type=int, default=128, help="Model dimension")
    ap.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    ap.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    ap.add_argument("--n-kv-heads", type=int, default=2, help="Number of KV heads (GQA)")
    ap.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")
    ap.add_argument(
        "--tokenizer", type=str, default="models/student/tokenizer", help="Tokenizer path"
    )
    ap.add_argument(
        "--eight-ball",
        dest="eight_ball",
        action="store_true",
        help="Train an 8-ball model that gives mystical fortune-telling responses",
    )
    ap.add_argument(
        "--binary-classifier",
        dest="binary_classifier",
        action="store_true",
        help="Train a binary classifier that outputs YES/NO decisions",
    )
    ap.add_argument(
        "--ternary-classifier",
        dest="ternary_classifier",
        action="store_true",
        help="Train a ternary classifier that outputs YES/NO/UNCERTAIN decisions",
    )
    args = ap.parse_args()

    # Auto-organize outputs into toys directories based on model type
    if args.eight_ball:
        toy_type = "8ball"
    elif args.binary_classifier:
        toy_type = "binary"
    elif args.ternary_classifier:
        toy_type = "ternary"
    else:
        toy_type = "pipeline"

    # If output path doesn't already include toys/, prepend it
    output_path_str = args.output_path
    if not output_path_str.startswith("toys/"):
        output_path_str = f"toys/{toy_type}/{output_path_str}"
    args.output_path = output_path_str

    # Setup device
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("[run_toy_distill] Starting toy distillation training")
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

    # Load tokenizer
    try:
        from training.dataset import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)
        print(f"[run_toy_distill] Loaded tokenizer from: {args.tokenizer}")

        # Validate vocabulary size alignment
        tokenizer_vocab_size = (
            len(tokenizer)
            if hasattr(tokenizer, "__len__")
            else getattr(tokenizer, "vocab_size", None)
        )
        if tokenizer_vocab_size is None:
            print("[run_toy_distill] WARNING: Cannot determine tokenizer vocab size")
        else:
            print(f"[run_toy_distill] Tokenizer vocab size: {tokenizer_vocab_size}")
            if tokenizer_vocab_size > args.vocab_size * 2:
                print("[run_toy_distill] ⚠️  VOCAB MISMATCH WARNING:")
                print(f"  Model vocab_size: {args.vocab_size}")
                print(f"  Tokenizer vocab_size: {tokenizer_vocab_size}")
                print(
                    f"  This will cause {100 * (1 - args.vocab_size / tokenizer_vocab_size):.1f}% of tokens to be clamped!"
                )
                print(
                    f"  Recommendation: Use --vocab-size {tokenizer_vocab_size} or retrain tokenizer"
                )
                # Non-interactive mode: just warn, don't abort (for automated testing)
                print("  Continuing with clamping (non-interactive mode)")
            elif tokenizer_vocab_size != args.vocab_size:
                print(
                    f"[run_toy_distill] ⚠️  Vocab size mismatch: model={args.vocab_size}, tokenizer={tokenizer_vocab_size}"
                )
                print("  Tokens will be clamped to fit model vocab_size")
    except Exception as e:
        print(f"[run_toy_distill] ERROR: Failed to load tokenizer: {e}")
        sys.exit(1)

    # Create dataset
    try:
        # Check if dataset might have teacher_logits (for 8-ball or future ndjson support)
        # For now, assume 8-ball datasets may have them
        teacher_logits_available = args.eight_ball

        dataset = KDDataset(
            jsonl_path=args.input_path,
            tokenizer_path=args.tokenizer,
            max_seq_length=args.max_seq_len,
            teacher_logits_available=teacher_logits_available,
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

    # Create learning rate scheduler with warmup
    warmup_steps = min(10, steps_per_epoch // 4)  # Warmup for first 10 steps or 25% of epoch
    total_training_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / max(1, total_training_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping parameters
    best_loss = float("inf")
    patience = 5  # Stop if no improvement for 5 steps
    patience_counter = 0
    early_stop = False

    step = 0
    for epoch in range(args.epochs):
        if early_stop:
            break
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Clamp token IDs to fit toy vocab size (tokenizer may have larger vocab)
            pre_violate = (input_ids.ge(args.vocab_size) | input_ids.lt(0)).any().item()
            input_ids = torch.clamp(input_ids, 0, args.vocab_size - 1)
            labels = torch.clamp(labels, 0, args.vocab_size - 1)
            if pre_violate and step % 50 == 0:
                print(
                    f"[run_toy_distill] ⚠ clamped token ids to vocab_size={args.vocab_size}",
                    flush=True,
                )

            # Forward pass
            student_logits = model(input_ids)

            # Get teacher logits: prefer pre-computed from dataset, fallback to stub
            if "teacher_logits" in batch and batch["teacher_logits"] is not None:
                # Use pre-computed teacher logits from ndjson dataset
                teacher_logits_tensor = batch["teacher_logits"].to(device)
                if batch_idx == 0 and epoch == 0:
                    print("[run_toy_distill] Using pre-computed teacher logits from dataset")
            else:
                # Generate teacher logits on-the-fly with improved stub
                if args.ternary_classifier:
                    teacher_logits_tensor = eight_ball_teacher_logits(
                        input_ids, vocab_size=args.vocab_size, tokenizer=tokenizer
                    ).to(device)
                    if batch_idx == 0 and epoch == 0:
                        print("[run_toy_distill] Generated ternary classifier teacher logits")
                elif args.binary_classifier:
                    teacher_logits_tensor = eight_ball_teacher_logits(
                        input_ids, vocab_size=args.vocab_size, tokenizer=tokenizer
                    ).to(device)
                    if batch_idx == 0 and epoch == 0:
                        print("[run_toy_distill] Generated binary classifier teacher logits")
                elif args.eight_ball:
                    teacher_logits_tensor = eight_ball_teacher_logits(
                        input_ids, vocab_size=args.vocab_size, tokenizer=tokenizer
                    ).to(device)
                    if batch_idx == 0 and epoch == 0:
                        print("[run_toy_distill] Generated 8-ball teacher logits with tokenizer")
                else:
                    teacher_logits_tensor = teacher_logits(
                        input_ids, vocab_size=args.vocab_size
                    ).to(device)
                    if batch_idx == 0 and epoch == 0:
                        print("[run_toy_distill] Generated toy teacher logits")

            # Compute loss with optimized distillation weights
            # For classification training (8-ball, binary, or ternary), focus on answer positions
            if (args.eight_ball or args.binary_classifier or args.ternary_classifier) and teacher_logits_tensor is None:
                # Create loss mask that heavily weights mystical answer positions
                loss_mask = torch.ones_like(labels, dtype=torch.bool, device=device)

                # For each sample in batch, identify mystical answer positions
                for batch_idx_in_batch in range(labels.shape[0]):
                    sample_data = batch["raw_data"][batch_idx_in_batch]

                    # Get mystical answer and find its position in the sequence
                    mystical_answer = sample_data["metadata"]["mystical_answer"]
                    full_text = sample_data["prompt"] + " " + sample_data["teacher_text"]

                    # Tokenize and find answer positions
                    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                    answer_tokens = tokenizer.encode(mystical_answer, add_special_tokens=False)

                    # Find where answer appears in the sequence
                    for start_pos in range(len(full_tokens) - len(answer_tokens) + 1):
                        if full_tokens[start_pos : start_pos + len(answer_tokens)] == answer_tokens:
                            # Mark these positions with higher weight (True = normal weight)
                            # We'll use loss weighting instead of masking
                            break

                    # For now, weight all positions equally but this could be improved
                    # to give higher weight to mystical answer positions

                loss_dict = combined_kd_loss(
                    student_logits=student_logits.float(),
                    teacher_logits=None,
                    teacher_targets=None,
                    ground_truth_targets=labels.to(device),
                    loss_mask=loss_mask,
                    kl_weight=0.0,  # No KL loss without teacher
                    ce_teacher_weight=0.0,  # No teacher CE
                    ce_ground_truth_weight=1.0,  # Focus on ground truth
                )
            else:
                # Standard KD loss
                loss_dict = combined_kd_loss(
                    student_logits=student_logits.float(),
                    teacher_logits=teacher_logits_tensor,
                    teacher_targets=teacher_logits_tensor.argmax(dim=-1)
                    if teacher_logits_tensor is not None
                    else None,
                    ground_truth_targets=labels.to(device),
                    kl_weight=0.6,  # Increased for better teacher alignment
                    ce_teacher_weight=0.2,  # Reduced to balance with KL
                    ce_ground_truth_weight=0.2,  # Keep ground truth supervision
                    kd_temperature=1.5,  # Lower temperature for sharper distillation
                )

            loss = loss_dict["total"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Update learning rate

            step += 1

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[run_toy_distill] Step {step}/{total_steps}: loss={loss.item():.4f}, lr={current_lr:.2e}"
                )

            # Early stopping check
            current_loss = loss.item()
            if current_loss < best_loss - 0.001:  # Minimum improvement threshold
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"[run_toy_distill] Early stopping at step {step} (no improvement for {patience} steps)"
                )
                early_stop = True
                break

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
            "model_type": "8-ball" if args.eight_ball else "toy",
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print("[run_toy_distill] ✅ Training complete")
    print(f"  Checkpoint saved: {output_path}")
    print(f"  Model SHA256: {state_sha256[:16]}...")
    print(f"  Git SHA: {git_sha}")


if __name__ == "__main__":
    main()
