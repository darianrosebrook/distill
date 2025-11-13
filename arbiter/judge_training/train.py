# arbiter/judge_training/train.py
# Judge model training with pairwise ranking and clause labeling
# @author: @darianrosebrook

import os
import json
import typer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .dataset import PairwiseJudgeDataset, JudgeConfig
from .model import MultiTaskJudge

app = typer.Typer()


@app.command()
def main(config: str = "configs/judge_training.yaml"):
    import yaml
    cfg = yaml.safe_load(open(config))
    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)

    jc = JudgeConfig(
        hf_name=cfg["model"]["hf_name"],
        max_len=cfg["model"]["max_len"],
        clauses=cfg["clauses"],
    )
    train_ds = PairwiseJudgeDataset(cfg["paths"]["train_jsonl"], jc)
    val_ds = PairwiseJudgeDataset(cfg["paths"]["val_jsonl"], jc)

    model = MultiTaskJudge(jc.hf_name, num_clauses=len(jc.clauses))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    bs = cfg["train"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["total_steps"]
    warmup = cfg["train"]["warmup_steps"]
    sch = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    bce = nn.BCEWithLogitsLoss()
    rank_margin = cfg["train"]["margin"]
    clause_w = cfg["train"]["clause_loss_weight"]

    def step(batch):
        (sa, ca), (sb, cb) = model(
            {k: v.to(device) for k, v in batch["a"].items()},
            {k: v.to(device) for k, v in batch["b"].items()}
        )
        # Pairwise ranking loss
        target = batch["target"].to(device).float()
        pos = torch.where(target > 0, sa, sb)
        neg = torch.where(target > 0, sb, sa)
        valid = (target != 0).float()
        margin_loss = torch.clamp(rank_margin - (pos - neg), min=0.0) * valid
        rank_loss = margin_loss.mean()
        # Clause BCE (both sides)
        ya = batch["ya"].to(device)
        yb = batch["yb"].to(device)
        clause_loss = 0.5 * (bce(ca, ya) + bce(cb, yb))
        return rank_loss + clause_w * clause_loss, {
            "rank_loss": rank_loss.item(),
            "clause_loss": clause_loss.item(),
        }

    global_step = 0
    model.train()
    while global_step < total_steps:
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            loss, logs = step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            global_step += 1
            if global_step % cfg["train"]["log_every"] == 0:
                print(json.dumps({"step": global_step, "loss": loss.item(), **logs}))
            if global_step % cfg["train"]["val_every"] == 0:
                evaluate(model, val_dl, device)
            if global_step >= total_steps:
                break

    out = os.path.join(cfg["paths"]["out_dir"], "judge.pt")
    torch.save({"model": model.state_dict(), "hf_name": jc.hf_name, "clauses": jc.clauses}, out)
    print(f"saved {out}")


@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    correct = 0
    total = 0
    for b in val_dl:
        (sa, ca), (sb, cb) = model(
            {k: v.to(device) for k, v in b["a"].items()},
            {k: v.to(device) for k, v in b["b"].items()}
        )
        # Pairwise acc ignoring ties
        winners = torch.where(sa > sb, 1, -1)
        mask = (b["target"] != 0)
        correct += ((winners == b["target"].to(winners.dtype)) & mask).sum().item()
        total += mask.sum().item()
    acc = correct / max(1, total)
    print(json.dumps({"val_pairwise_acc": round(acc, 4), "n": total}))
    model.train()


if __name__ == "__main__":
    app()

