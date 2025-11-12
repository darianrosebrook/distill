"""
Training script for post-tool integration stage.

Trains model to continue reasoning after receiving tool results.
"""
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset_post_tool import PostToolDataset, collate_post_tool_batch
from training.tracing import TrainingTracer, create_tracer_from_config
from training.losses import combined_kd_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create student model from config."""
    arch_cfg = cfg.get("arch", {})
    
    model_cfg = ModelCfg(
        d_model=arch_cfg.get("d_model", 4096),
        n_layers=arch_cfg.get("n_layers", 32),
        n_heads=arch_cfg.get("n_heads", 32),
        n_kv_heads=arch_cfg.get("n_kv_heads", 8),
        d_head=arch_cfg.get("d_head", 128),
        vocab_size=arch_cfg.get("vocab_size", 32000),
        rope_theta=arch_cfg.get("rope_theta", 10000.0),
        rope_scaling=arch_cfg.get("rope_scaling", "dynamic"),
        dropout=arch_cfg.get("dropout", 0.0),
    )
    
    model = StudentLM(model_cfg)
    
    # Load checkpoint if specified
    init_cfg = cfg.get("init", {})
    base_checkpoint = init_cfg.get("base_checkpoint")
    if base_checkpoint and Path(base_checkpoint).exists():
        print(f"[distill_post_tool] Loading checkpoint: {base_checkpoint}")
        checkpoint = torch.load(base_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"[distill_post_tool] Checkpoint loaded")
    
    model = model.to(device)
    return model


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        logits = model(input_ids, attention_mask)  # [B, T, V]
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        total_loss = ce_loss
    
    # Backward pass
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
    
    return {
        "total": float(total_loss.item()),
        "ce": float(ce_loss.item()),
    }


def main():
    ap = argparse.ArgumentParser(description="Post-tool integration training")
    ap.add_argument('--config', required=True, help='Config file')
    ap.add_argument('--data', required=True, help='Path to post_tool.jsonl')
    ap.add_argument('--output-dir', default='models/student/checkpoints', help='Output directory')
    ap.add_argument('--resume', help='Resume from checkpoint')
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[distill_post_tool] Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_model(cfg, device)
    
    # Create optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )
    
    # Setup FP16 scaler
    train_cfg = cfg.get("train", {})
    use_fp16 = train_cfg.get("fp16", False)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device.type == 'cuda' else None
    
    # Create dataset
    tokenizer_path = cfg.get("io", {}).get("tokenizer_path", "models/student/tokenizer")
    dataset = PostToolDataset(
        data_path=args.data,
        tokenizer_path=tokenizer_path,
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("micro_batch_size", 2),
        shuffle=True,
        collate_fn=collate_post_tool_batch,
        num_workers=2,
    )
    
    # Initialize tracer
    tracer = create_tracer_from_config(cfg, run_name="post_tool")
    tracer.log_hparams({
        "stage": "post_tool",
        "data_path": args.data,
        "lr": opt_cfg.get("lr", 2e-4),
        "batch_size": train_cfg.get("micro_batch_size", 2),
    })
    
    # Training loop
    total_steps = train_cfg.get("steps", 50000)
    save_every = train_cfg.get("save_every", 1000)
    log_every = train_cfg.get("log_every", 50)
    
    step = 0
    for epoch in range(100):
        for batch in dataloader:
            if step >= total_steps:
                break
            
            loss_dict = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=device,
            )
            
            step += 1
            
            # Logging
            if step % log_every == 0:
                tracer.log_metrics(step=step, metrics=loss_dict, prefix="train/")
            
            # Checkpointing
            if step % save_every == 0:
                checkpoint_path = output_dir / f"post_tool_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_dict.get("total", 0.0),
                    "config": cfg,
                }, checkpoint_path)
                print(f"[distill_post_tool] Saved checkpoint: {checkpoint_path}")
        
        if step >= total_steps:
            break
    
    # Final checkpoint
    final_path = output_dir / "post_tool_latest.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_dict.get("total", 0.0),
        "config": cfg,
    }, final_path)
    
    tracer.close()
    print(f"[distill_post_tool] ✅ Training complete: {final_path}")


if __name__ == '__main__':
    main()


