# arbiter/judge_training/pairwise_ranking.py
# Pairwise ranking distillation for Judge model
# @author: @darianrosebrook

import json
import os
import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

try:
    import yaml
except ImportError:
    yaml = None

from .dataset import PairwiseJudgeDataset, JudgeConfig
from .model import MultiTaskJudge

app = typer.Typer()


@dataclass
class PairwiseExample:
    """Pairwise comparison example for judge training."""

    prompt: str
    output_a: str
    output_b: str
    preference: int  # 0 = A preferred, 1 = B preferred, -1 = tie
    caws_clauses: List[str]  # CAWS clauses relevant to this comparison
    evidence_quality_a: float
    evidence_quality_b: float


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss for judge model training."""

    def __init__(self, temperature: float = 1.0, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self, scores_a: torch.Tensor, scores_b: torch.Tensor, preference: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Args:
            scores_a: [B] or [B, 1] scores for output A
            scores_b: [B] or [B, 1] scores for output B
            preference: [B] preference labels (0=A preferred, 1=B preferred, -1=tie)

        Returns:
            Scalar loss value
        """
        # Ensure scores are [B] shape
        if scores_a.dim() > 1:
            scores_a = scores_a.squeeze(-1)
        if scores_b.dim() > 1:
            scores_b = scores_b.squeeze(-1)

        # Compute ranking loss
        # For preference=0 (A preferred): maximize score_a - score_b (margin loss)
        # For preference=1 (B preferred): maximize score_b - score_a (margin loss)
        # For preference=-1 (tie): minimize |score_a - score_b|
        loss = torch.zeros_like(preference, dtype=torch.float32)

        mask_a = preference == 0
        mask_b = preference == 1
        mask_tie = preference == -1

        if mask_a.any():
            # A preferred: score_a should be > score_b + margin
            diff = scores_b[mask_a] - scores_a[mask_a] + self.margin
            loss[mask_a] = torch.clamp(diff, min=0.0)

        if mask_b.any():
            # B preferred: score_b should be > score_a + margin
            diff = scores_a[mask_b] - scores_b[mask_b] + self.margin
            loss[mask_b] = torch.clamp(diff, min=0.0)

        if mask_tie.any():
            # Tie: minimize difference between scores
            diff = torch.abs(scores_a[mask_tie] - scores_b[mask_tie])
            loss[mask_tie] = diff

        return loss.mean()


class ClauseLabelingLoss(nn.Module):
    """Multi-label classification loss for CAWS clause mapping."""

    def __init__(self):
        super().__init__()

    def forward(self, clause_logits: torch.Tensor, clause_labels: torch.Tensor) -> torch.Tensor:
        """Compute clause labeling loss.

        Args:
            clause_logits: [B, N_clauses] logits for each CAWS clause
            clause_labels: [B, N_clauses] binary labels (1=relevant, 0=not)

        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy_with_logits(clause_logits, clause_labels)


@app.command()
def main(config: str = typer.Argument(...)):
    """Train judge model with pairwise ranking and clause labeling.

    Args:
        config: Path to judge training config YAML/JSON
    """
    if yaml is None:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Extract configuration
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})
    clauses = cfg.get("clauses", [])

    if not clauses:
        raise ValueError("Config must specify 'clauses' list")

    # Setup paths
    train_jsonl = paths_cfg.get("train_jsonl")
    val_jsonl = paths_cfg.get("val_jsonl")
    out_dir = paths_cfg.get("out_dir", "arbiter/judge_training/artifacts")

    if not train_jsonl:
        raise ValueError("Config must specify 'paths.train_jsonl'")
    if not val_jsonl:
        raise ValueError("Config must specify 'paths.val_jsonl'")

    os.makedirs(out_dir, exist_ok=True)

    # Create dataset config
    jc = JudgeConfig(
        hf_name=model_cfg.get("hf_name", "microsoft/deberta-v3-small"),
        max_len=model_cfg.get("max_len", 512),
        clauses=clauses,
    )

    # Load datasets
    print(f"[pairwise_ranking] Loading training dataset: {train_jsonl}")
    train_ds = PairwiseJudgeDataset(train_jsonl, jc)
    print(f"[pairwise_ranking] Training samples: {len(train_ds)}")

    print(f"[pairwise_ranking] Loading validation dataset: {val_jsonl}")
    val_ds = PairwiseJudgeDataset(val_jsonl, jc)
    print(f"[pairwise_ranking] Validation samples: {len(val_ds)}")

    # Create model
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[pairwise_ranking] Device: {device}")

    model = MultiTaskJudge(jc.hf_name, num_clauses=len(jc.clauses))
    model.to(device)
    print(f"[pairwise_ranking] Model: {jc.hf_name}, clauses: {len(jc.clauses)}")

    # Create data loaders
    batch_size = train_cfg.get("batch_size", 8)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Setup optimizer
    lr = train_cfg.get("lr", 1e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup learning rate schedule
    total_steps = train_cfg.get("total_steps", 10000)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Setup loss functions
    ranking_loss_fn = PairwiseRankingLoss(
        temperature=train_cfg.get("temperature", 1.0), margin=train_cfg.get("margin", 0.2)
    )
    clause_loss_fn = ClauseLabelingLoss()
    clause_weight = train_cfg.get("clause_loss_weight", 0.5)

    # Training loop
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("val_every", 1000)
    save_every = train_cfg.get("save_every", 2000)

    print("[pairwise_ranking] Starting training:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")

    global_step = 0
    model.train()

    while global_step < total_steps:
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)

            # Move batch to device
            batch_a = {k: v.to(device) for k, v in batch["a"].items()}
            batch_b = {k: v.to(device) for k, v in batch["b"].items()}
            ya = batch["ya"].to(device)
            yb = batch["yb"].to(device)
            target = batch["target"].to(device).float()

            # Forward pass
            (sa, ca), (sb, cb) = model(batch_a, batch_b)

            # Compute pairwise ranking loss
            # Convert target format: -1=B>A, 0=tie, 1=A>B
            # To preference format: 0=A preferred, 1=B preferred, -1=tie
            preference = torch.zeros_like(target, dtype=torch.long)
            preference[target > 0] = 0  # A preferred
            preference[target < 0] = 1  # B preferred
            preference[target == 0] = -1  # Tie

            # Compute ranking loss using PairwiseRankingLoss
            # PairwiseRankingLoss expects logits_a and logits_b
            # We have scores sa and sb, which are already scalar outputs
            # Reshape to [B, 1] for compatibility with loss function
            sa_expanded = sa.unsqueeze(-1) if sa.dim() == 1 else sa
            sb_expanded = sb.unsqueeze(-1) if sb.dim() == 1 else sb

            rank_loss = ranking_loss_fn(sa_expanded, sb_expanded, preference)

            # Compute clause labeling loss
            clause_loss_a = clause_loss_fn(ca, ya)
            clause_loss_b = clause_loss_fn(cb, yb)
            clause_loss = 0.5 * (clause_loss_a + clause_loss_b)

            # Total loss
            total_loss = rank_loss + clause_weight * clause_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % log_every == 0:
                log_dict = {
                    "step": global_step,
                    "loss": total_loss.item(),
                    "rank_loss": rank_loss.item(),
                    "clause_loss": clause_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
                print(json.dumps(log_dict))

            # Validation
            if global_step % val_every == 0:
                val_metrics = evaluate(
                    model, val_dl, device, ranking_loss_fn, clause_loss_fn, clause_weight
                )
                print(json.dumps({"step": global_step, "val": val_metrics}))
                model.train()

            # Checkpointing
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(out_dir, f"judge_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "hf_name": jc.hf_name,
                        "clauses": jc.clauses,
                        "config": cfg,
                    },
                    checkpoint_path,
                )
                print(f"[pairwise_ranking] Saved checkpoint: {checkpoint_path}")

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    # Final checkpoint
    final_path = os.path.join(out_dir, "judge.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "hf_name": jc.hf_name,
            "clauses": jc.clauses,
            "config": cfg,
        },
        final_path,
    )
    print(f"[pairwise_ranking] Training complete. Saved final model: {final_path}")


@torch.no_grad()
def evaluate(
    model: MultiTaskJudge,
    val_dl: DataLoader,
    device: torch.device,
    ranking_loss_fn: PairwiseRankingLoss,
    clause_loss_fn: ClauseLabelingLoss,
    clause_weight: float,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    total_rank_loss = 0.0
    total_clause_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch in val_dl:
        batch_a = {k: v.to(device) for k, v in batch["a"].items()}
        batch_b = {k: v.to(device) for k, v in batch["b"].items()}
        ya = batch["ya"].to(device)
        yb = batch["yb"].to(device)
        target = batch["target"].to(device).float()

        # Forward pass
        (sa, ca), (sb, cb) = model(batch_a, batch_b)

        # Compute losses
        preference = torch.zeros_like(target, dtype=torch.long)
        preference[target > 0] = 0
        preference[target < 0] = 1
        preference[target == 0] = -1

        sa_expanded = sa.unsqueeze(-1) if sa.dim() == 1 else sa
        sb_expanded = sb.unsqueeze(-1) if sb.dim() == 1 else sb

        rank_loss = ranking_loss_fn(sa_expanded, sb_expanded, preference)
        clause_loss_a = clause_loss_fn(ca, ya)
        clause_loss_b = clause_loss_fn(cb, yb)
        clause_loss = 0.5 * (clause_loss_a + clause_loss_b)
        batch_loss = rank_loss + clause_weight * clause_loss

        total_rank_loss += rank_loss.item()
        total_clause_loss += clause_loss.item()
        total_loss += batch_loss.item()
        num_batches += 1

        # Compute pairwise accuracy (ignoring ties)
        winners = torch.where(sa > sb, 1, -1)
        mask = target != 0
        correct += ((winners == target.to(winners.dtype)) & mask).sum().item()
        total += mask.sum().item()

    return {
        "rank_loss": total_rank_loss / max(1, num_batches),
        "clause_loss": total_clause_loss / max(1, num_batches),
        "total_loss": total_loss / max(1, num_batches),
        "pairwise_acc": correct / max(1, total),
        "n_samples": total,
    }


if __name__ == "__main__":
    app()
