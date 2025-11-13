"""
Process supervision training for tool-use.

Extends KD training with process supervision losses:
- JSON validity loss
- Tool selection loss

Usage:
    python -m training.distill_process --checkpoint models/student/checkpoints/latest.pt --config configs/worker_9b.yaml configs/process_supervision.yaml
"""
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.losses import combined_kd_loss
from training.process_losses import process_supervision_loss
from training.dataset import KDDataset, collate_kd_batch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(configs: list) -> Dict[str, Any]:
    """Merge multiple config files."""
    merged = {}
    for config_path in configs:
        config = load_config(config_path)
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config from checkpoint
    cfg = None
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        arch_cfg = config_data.get('arch', {})
        cfg = ModelCfg(
            d_model=arch_cfg.get('d_model', 4096),
            n_layers=arch_cfg.get('n_layers', 32),
            n_heads=arch_cfg.get('n_heads', 32),
            n_kv_heads=arch_cfg.get('n_kv_heads', 8),
            d_head=arch_cfg.get('d_head', 128),
            vocab_size=arch_cfg.get('vocab_size', 32000),
            rope_theta=arch_cfg.get('rope_theta', 10000.0),
            rope_scaling=arch_cfg.get('rope_scaling', 'dynamic'),
            dropout=arch_cfg.get('dropout', 0.0),
        )
    
    if cfg is None:
        cfg = ModelCfg()
    
    model = StudentLM(cfg)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    return model


def generate_text_from_logits(logits: torch.Tensor, tokenizer, max_new_tokens: int = 512) -> List[str]:
    """
    Generate text from logits using greedy decoding.
    
    Args:
        logits: [B, T, V] model logits
        tokenizer: Tokenizer for decoding
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of generated text strings
    """
    # Get predicted token IDs (greedy)
    pred_token_ids = logits.argmax(dim=-1)  # [B, T]
    
    texts = []
    for i in range(pred_token_ids.size(0)):
        tokens = pred_token_ids[i].cpu().tolist()
        # Decode tokens to text
        try:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
        except Exception:
            texts.append("")
    
    return texts


def train_step_process(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    device: torch.device,
    tokenizer,
    proc_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Training step with process supervision.
    
    Returns:
        Dictionary with loss values
    """
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    teacher_target_ids = batch.get("teacher_target_ids")
    if teacher_target_ids is not None:
        teacher_target_ids = teacher_target_ids.to(device)
    
    teacher_logits = batch.get("teacher_logits")
    if teacher_logits is not None:
        teacher_logits = teacher_logits.to(device)
    
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        # Forward pass
        student_logits = model(input_ids, attention_mask)
        
        # Standard KD loss
        kd_cfg = cfg.get("distillation", {})
        kd_loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_target_ids,
            ground_truth_targets=labels,
            kl_weight=kd_cfg.get("kl_weight", 0.5),
            ce_teacher_weight=kd_cfg.get("ce_teacher_weight", 0.3),
            ce_ground_truth_weight=kd_cfg.get("ce_ground_truth_weight", 0.2),
            kd_temperature=cfg.get("kd", {}).get("kd_temperature", 2.0),
        )
        
        # Process supervision loss
        # Generate text from logits for validation
        generated_texts = generate_text_from_logits(student_logits, tokenizer)
        
        # Extract target tool names from batch metadata (if available)
        target_tool_names = None
        if "tool_name_ids" in batch and "tool_name_mask" in batch:
            tool_name_ids = batch["tool_name_ids"]
            tool_name_mask = batch["tool_name_mask"]
            if tool_name_ids.numel() > 0:
                # Decode tool name IDs back to text for loss computation
                # Handle batched tool names - decode each sample's tool name
                batch_size = tool_name_ids.size(0) if tool_name_ids.dim() > 1 else 1
                decoded_tool_names = []
                
                for i in range(batch_size):
                    if tool_name_ids.dim() > 1:
                        sample_tool_ids = tool_name_ids[i]
                        sample_mask = tool_name_mask[i]
                    else:
                        sample_tool_ids = tool_name_ids
                        sample_mask = tool_name_mask
                    
                    # Extract valid tokens using mask
                    if sample_mask.dtype == torch.bool:
                        valid_tokens = sample_tool_ids[sample_mask]
                    else:
                        # Handle integer mask (1s and 0s)
                        valid_tokens = sample_tool_ids[sample_mask.bool()]
                    
                    if valid_tokens.numel() > 0:
                        target_tool_name = tokenizer.decode(valid_tokens.tolist(), skip_special_tokens=True)
                        # Clean up tool name (remove quotes if present)
                        target_tool_name = target_tool_name.strip('"\'')
                        decoded_tool_names.append(target_tool_name)
                    else:
                        decoded_tool_names.append(None)
                
                # Only use non-None tool names
                if any(name is not None for name in decoded_tool_names):
                    target_tool_names = [name for name in decoded_tool_names if name is not None]
                    if len(target_tool_names) < batch_size:
                        # Pad with None if needed
                        target_tool_names.extend([None] * (batch_size - len(target_tool_names)))
        
        tool_names = proc_cfg.get("tool_names", [])
        
        # Get token IDs from batch if available
        tool_name_ids = batch.get("tool_name_ids")
        tool_name_mask = batch.get("tool_name_mask")
        gold_json_text_ids = batch.get("gold_json_text_ids")
        mask_valid_json_tokens = batch.get("mask_valid_json_tokens")
        
        # Move to device if present
        if tool_name_ids is not None:
            tool_name_ids = tool_name_ids.to(device)
        if tool_name_mask is not None:
            tool_name_mask = tool_name_mask.to(device)
        if gold_json_text_ids is not None:
            gold_json_text_ids = gold_json_text_ids.to(device)
        if mask_valid_json_tokens is not None:
            mask_valid_json_tokens = mask_valid_json_tokens.to(device)
        
        proc_loss_dict = process_supervision_loss(
            logits=student_logits,
            generated_texts=generated_texts,  # Keep for backward compatibility
            target_tool_names=target_tool_names,
            tool_names=tool_names,
            tokenizer=tokenizer,
            json_validity_weight=proc_cfg.get("loss_json_validity_weight", 0.3),
            tool_select_weight=proc_cfg.get("loss_tool_select_weight", 0.7),
            # Pass token ID-based arguments
            tool_name_ids=tool_name_ids,
            tool_name_mask=tool_name_mask,
            gold_json_text_ids=gold_json_text_ids,
            mask_valid_json_tokens=mask_valid_json_tokens,
        )
        
        # Combine losses
        kd_weight = cfg.get("distillation", {}).get("process_supervision_weight", 0.7)
        proc_weight = 1.0 - kd_weight
        
        total_loss = kd_weight * kd_loss_dict["total"] + proc_weight * proc_loss_dict["total"]
        
        loss_dict = {
            **{f"kd_{k}": v for k, v in kd_loss_dict.items()},
            **{f"proc_{k}": v for k, v in proc_loss_dict.items()},
            "total": total_loss,
        }
    
    # Backward pass
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0))
        optimizer.step()
    
    optimizer.zero_grad()
    
    # Convert to float
    loss_dict_float = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
    return loss_dict_float


def main():
    ap = argparse.ArgumentParser(description="Process supervision training")
    ap.add_argument('--checkpoint', required=True, help='Checkpoint to continue from')
    ap.add_argument('--config', nargs='+', required=True, help='Config file(s)')
    ap.add_argument('--output-dir', default='models/student/checkpoints', help='Output directory')
    ap.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    ap.add_argument('--save-every', type=int, default=1000, help='Save checkpoint every N steps')
    ap.add_argument('--log-every', type=int, default=50, help='Log every N steps')
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    cfg = merge_configs(args.config)
    proc_cfg = cfg.get("process_supervision", {})
    
    # Load model
    print(f"[distill_process] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Create optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )
    
    # Setup FP16
    train_cfg = cfg.get("train", {})
    use_fp16 = train_cfg.get("fp16", False)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device.type == 'cuda' else None
    
    # Load tokenizer
    if not HF_TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers required for process supervision")
    
    io_cfg = cfg.get("io", {})
    tokenizer_path = io_cfg.get("tokenizer_path", "models/student/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create dataset
    train_shards = io_cfg.get("train_shards", ["data/kd_mix.jsonl"])
    seq_lengths = train_cfg.get("seq_lengths", [4096])
    
    dataset = KDDataset(
        jsonl_path=train_shards[0],
        tokenizer_path=tokenizer_path,
        max_seq_length=seq_lengths[0],
        teacher_logits_available=cfg.get("kd", {}).get("teacher_logits_available", False),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("micro_batch_size", 2),
        shuffle=True,
        collate_fn=collate_kd_batch,
        num_workers=2,
        pin_memory=device.type == 'cuda',
    )
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[distill_process] Starting process supervision training:")
    print(f"  Device: {device}")
    print(f"  Steps: {args.steps}")
    print(f"  JSON validity weight: {proc_cfg.get('loss_json_validity_weight', 0.3)}")
    print(f"  Tool selection weight: {proc_cfg.get('loss_tool_select_weight', 0.7)}")
    
    step = 0
    for epoch in range(100):  # Max epochs
        for batch in dataloader:
            if step >= args.steps:
                break
            
            loss_dict = train_step_process(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                device=device,
                tokenizer=tokenizer,
                proc_cfg=proc_cfg,
            )
            
            step += 1
            
            if step % args.log_every == 0:
                loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                print(f"[distill_process] Step {step}/{args.steps}: {loss_str}")
            
            if step % args.save_every == 0:
                checkpoint_path = output_dir / f"process_supervised_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_dict.get("total", 0.0),
                    "config": cfg,
                }, checkpoint_path)
                print(f"[distill_process] Saved checkpoint: {checkpoint_path}")
        
        if step >= args.steps:
            break
    
    # Final checkpoint
    final_path = output_dir / "process_supervised_latest.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_dict.get("total", 0.0),
        "config": cfg,
    }, final_path)
    print(f"[distill_process] ✅ Training complete: {final_path}")


if __name__ == '__main__':
    main()
