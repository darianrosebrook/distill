"""
Knowledge distillation training script.

Trains student model from teacher using KD losses:
- KL divergence (soft targets)
- Cross-entropy on teacher predictions
- Cross-entropy on ground truth

Usage:
    python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml
"""
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.losses import combined_kd_loss
from training.dataset import KDDataset, collate_kd_batch
from training.tracing import TrainingTracer, create_tracer_from_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(configs: list) -> Dict[str, Any]:
    """Merge multiple config files."""
    merged = {}
    for config_path in configs:
        config = load_config(config_path)
        # Deep merge
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


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
        print(f"[distill_kd] Loading checkpoint: {base_checkpoint}")
        checkpoint = torch.load(base_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"[distill_kd] Checkpoint loaded")
    
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_cfg = cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 2e-4)
    betas = opt_cfg.get("betas", [0.9, 0.95])
    weight_decay = opt_cfg.get("weight_decay", 0.1)
    
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


def get_sequence_length(step: int, seq_lengths: list, curriculum_schedule: Optional[list] = None) -> int:
    """
    Get current sequence length based on curriculum learning.
    
    Args:
        step: Current training step
        seq_lengths: List of available sequence lengths
        curriculum_schedule: List of step boundaries for each length
        
    Returns:
        Current sequence length to use
    """
    if curriculum_schedule is None:
        # Default: use longest available
        return max(seq_lengths)
    
    # Find which length to use based on step
    for i, boundary in enumerate(curriculum_schedule):
        if step < boundary:
            return seq_lengths[min(i, len(seq_lengths) - 1)]
    
    return seq_lengths[-1]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: Path,
    config: Dict[str, Any],
):
    """Save training checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict (unwrap DDP if needed)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }
    
    # Save latest
    latest_path = output_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save numbered checkpoint
    checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    print(f"[distill_kd] Saved checkpoint: {checkpoint_path}")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    device: torch.device,
    grad_accum_steps: int = 1,
    grad_accum_counter: int = 0,
) -> Dict[str, float]:
    """
    Single training step.
    
    Returns:
        Dictionary with loss values
    """
    model.train()
    
    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    teacher_target_ids = batch.get("teacher_target_ids")
    if teacher_target_ids is not None:
        teacher_target_ids = teacher_target_ids.to(device)
    
    teacher_logits = batch.get("teacher_logits")
    if teacher_logits is not None:
        teacher_logits = teacher_logits.to(device)
    
    # Forward pass
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        # Get student logits
        student_logits = model(input_ids, attention_mask)  # [B, T, V]
        
        # Compute loss
        kd_cfg = cfg.get("distillation", {})
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_target_ids,
            ground_truth_targets=labels,
            kl_weight=kd_cfg.get("kl_weight", 0.5),
            ce_teacher_weight=kd_cfg.get("ce_teacher_weight", 0.3),
            ce_ground_truth_weight=kd_cfg.get("ce_ground_truth_weight", 0.2),
            kd_temperature=cfg.get("kd", {}).get("kd_temperature", 2.0),
        )
        
        loss = loss_dict["total"]
        loss = loss / grad_accum_steps  # Scale for gradient accumulation
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Update weights (if gradient accumulation complete)
    if (grad_accum_counter + 1) % grad_accum_steps == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0))
            optimizer.step()
        optimizer.zero_grad()
    
    # Convert to float for logging
    loss_dict_float = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
    return loss_dict_float


def main():
    ap = argparse.ArgumentParser(description="Knowledge distillation training")
    ap.add_argument('--config', nargs='+', required=True, help='Config file(s) to load')
    ap.add_argument('--resume', help='Resume from checkpoint path')
    ap.add_argument('--output-dir', default='models/student/checkpoints', help='Output directory for checkpoints')
    ap.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    args = ap.parse_args()
    
    # Setup distributed training if needed
    if args.local_rank >= 0:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        is_main_process = args.local_rank == 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
    
    # Load configs
    cfg = merge_configs(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = create_model(cfg, device)
    
    # Setup distributed model if needed
    if args.local_rank >= 0:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create optimizer
    optimizer = create_optimizer(model, cfg)
    
    # Setup FP16 scaler
    train_cfg = cfg.get("train", {})
    use_fp16 = train_cfg.get("fp16", False)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device.type == 'cuda' else None
    
    # Setup gradient checkpointing
    if train_cfg.get("grad_checkpointing", False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"[distill_kd] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'step' in checkpoint:
            start_step = checkpoint['step']
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Create dataset
    io_cfg = cfg.get("io", {})
    tokenizer_path = io_cfg.get("tokenizer_path")
    if not tokenizer_path:
        # Fallback: try configs/worker_9b.yaml tokenizer path
        tokenizer_path = cfg.get("tokenizer", {}).get("path", "models/student/tokenizer")
    
    if not tokenizer_path:
        raise ValueError("tokenizer_path must be specified in config (io.tokenizer_path)")
    
    train_shards = io_cfg.get("train_shards", ["data/kd_mix.jsonl"])
    
    # Use first sequence length for initial dataset
    seq_lengths = train_cfg.get("seq_lengths", [4096])
    current_seq_len = get_sequence_length(start_step, seq_lengths, cfg.get("curriculum", {}).get("schedule"))
    
    dataset = KDDataset(
        jsonl_path=train_shards[0],
        tokenizer_path=tokenizer_path,
        max_seq_length=current_seq_len,
        teacher_logits_available=cfg.get("kd", {}).get("teacher_logits_available", False),
    )
    
    # Create dataloader
    micro_batch_size = train_cfg.get("micro_batch_size", 2)
    grad_accum = train_cfg.get("grad_accum", 16)
    effective_batch_size = micro_batch_size * grad_accum
    
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collate_kd_batch,
        num_workers=2,
        pin_memory=device.type == 'cuda',
    )
    
    # Training loop
    total_steps = train_cfg.get("steps", 200000)
    save_every = train_cfg.get("save_every", 2000)
    log_every = train_cfg.get("log_every", 50)
    
    # Initialize training tracer
    run_name = f"worker_9b_kd_{start_step}"
    tracer = create_tracer_from_config(cfg, run_name=run_name) if is_main_process else None
    
    # Log hyperparameters
    if tracer:
        opt_cfg = cfg.get("optimizer", {})
        dist_cfg = cfg.get("distillation", {})
        tracer.log_hparams({
            "lr": opt_cfg.get("lr", 2e-4),
            "batch_size": effective_batch_size,
            "micro_batch_size": micro_batch_size,
            "grad_accum": grad_accum,
            "fp16": use_fp16,
            "seq_lengths": str(seq_lengths),
            "total_steps": total_steps,
            "device": str(device),
            "kl_weight": dist_cfg.get("kl_weight", 0.5),
            "ce_teacher_weight": dist_cfg.get("ce_teacher_weight", 0.3),
            "ce_ground_truth_weight": dist_cfg.get("ce_ground_truth_weight", 0.2),
        })
    
    print(f"[distill_kd] Starting training:")
    print(f"  Device: {device}")
    print(f"  Total steps: {total_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Micro batch size: {micro_batch_size}, Gradient accumulation: {grad_accum}")
    print(f"  FP16: {use_fp16}")
    print(f"  Sequence lengths: {seq_lengths}")
    if tracer:
        print(f"  Tracing enabled: TensorBoard={tracer.use_tensorboard}, WandB={tracer.use_wandb}")
    
    step = start_step
    grad_accum_counter = 0
    
    # Iterate over dataset multiple times if needed
    while step < total_steps:
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break
            
            # Update sequence length based on curriculum
            current_seq_len = get_sequence_length(step, seq_lengths, cfg.get("curriculum", {}).get("schedule"))
            # Note: For simplicity, we use the dataset's max_seq_length.
            # In production, you'd want to dynamically filter/truncate batches.
            
            # Training step
            loss_dict = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=device,
                grad_accum_steps=grad_accum,
                grad_accum_counter=grad_accum_counter,
            )
            
            grad_accum_counter = (grad_accum_counter + 1) % grad_accum
            step += 1
            
            # Logging
            if step % log_every == 0 and is_main_process:
                if tracer:
                    # Log to tracer (includes console, TensorBoard, WandB, JSON)
                    tracer.log_metrics(step=step, metrics=loss_dict, prefix="train/")
                    
                    # Also log learning rate if available
                    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                        lr = optimizer.param_groups[0].get('lr', 0.0)
                        tracer.log_metrics(step=step, metrics={"learning_rate": lr}, prefix="train/")
                else:
                    # Fallback to console logging
                    loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    print(f"[distill_kd] Step {step}/{total_steps}: {loss_str}")
            
            # Checkpointing
            if step % save_every == 0 and is_main_process:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=loss_dict.get("total", 0.0),
                    output_dir=output_dir,
                    config=cfg,
                )
    
    # Final checkpoint
    if is_main_process:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=loss_dict.get("total", 0.0),
            output_dir=output_dir,
            config=cfg,
        )
        
        # Close tracer and save summary
        if tracer:
            tracer.close()
            print(f"[distill_kd] Training logs: {tracer.log_dir}")
        
        print(f"[distill_kd] ✅ Training complete: {output_dir}")


if __name__ == '__main__':
    main()
