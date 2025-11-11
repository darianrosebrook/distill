# training/quant_qat_int8.py
# QAT (Quantization-Aware Training) modules for INT8 quantization.
# @author: @darianrosebrook

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg, MHA_GQA, SwiGLU
from training.losses import combined_kd_loss
from training.dataset import KDDataset, collate_kd_batch


class MinMaxObserver(nn.Module):
    """Min-max observer for per-channel quantization."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("min_val", torch.zeros(num_channels))
        self.register_buffer("max_val", torch.zeros(num_channels))
        self.register_buffer("num_observations", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C, ...] - observe per-channel min/max
        if x.dim() >= 2:
            # For weights: [C_out, C_in] or [C]
            # For activations: [B, C, ...]
            dims_to_reduce = list(range(x.dim()))
            if x.dim() >= 2:
                dims_to_reduce.remove(0)  # Keep channel dimension
            
            x_min = x.amin(dim=dims_to_reduce, keepdim=False)
            x_max = x.amax(dim=dims_to_reduce, keepdim=False)
            
            if self.num_observations == 0:
                self.min_val.copy_(x_min)
                self.max_val.copy_(x_max)
            else:
                self.min_val.copy_(torch.minimum(self.min_val, x_min))
                self.max_val.copy_(torch.maximum(self.max_val, x_max))
            
            self.num_observations += 1
        
        return x

    def get_scale_zero_point(self, num_bits: int = 8, signed: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero point for quantization."""
        qmin = -(2 ** (num_bits - 1)) if signed else 0
        qmax = (2 ** (num_bits - 1)) - 1 if signed else (2 ** num_bits) - 1
        
        scale = (self.max_val - self.min_val) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        
        zero_point = qmin - self.min_val / scale
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        return scale, zero_point


class FakeQuantize(nn.Module):
    """Fake quantization for QAT."""
    def __init__(self, observer: MinMaxObserver, num_bits: int = 8, signed: bool = False):
        super().__init__()
        self.observer = observer
        self.num_bits = num_bits
        self.signed = signed
        self.register_buffer("scale", torch.ones(observer.min_val.shape))
        self.register_buffer("zero_point", torch.zeros(observer.min_val.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update observer
            self.observer(x)
            # Update scale/zero_point
            self.scale, self.zero_point = self.observer.get_scale_zero_point(self.num_bits, self.signed)
        
        # Quantize
        qmin = -(2 ** (self.num_bits - 1)) if self.signed else 0
        qmax = (2 ** (self.num_bits - 1)) - 1 if self.signed else (2 ** self.num_bits) - 1
        
        # Per-channel quantization
        if self.scale.dim() > 0:
            # Expand scale/zero_point to match x shape
            scale_expanded = self.scale.view(*([-1] + [1] * (x.dim() - 1)))
            zp_expanded = self.zero_point.view(*([-1] + [1] * (x.dim() - 1)))
        else:
            scale_expanded = self.scale
            zp_expanded = self.zero_point
        
        x_q = torch.round(x / scale_expanded + zp_expanded)
        x_q = torch.clamp(x_q, qmin, qmax)
        
        # Dequantize
        x_dq = (x_q - zp_expanded) * scale_expanded
        
        return x_dq


class QuantizedLinear(nn.Module):
    """Quantized linear layer with per-channel weight quantization."""
    def __init__(self, linear: nn.Linear, weight_bits: int = 8, act_bits: int = 8):
        super().__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        
        # Weight observer (per output channel)
        self.weight_observer = MinMaxObserver(linear.out_features)
        self.weight_fake_quant = FakeQuantize(self.weight_observer, weight_bits, signed=True)
        
        # Activation observer (per channel)
        self.act_observer = MinMaxObserver(linear.in_features)
        self.act_fake_quant = FakeQuantize(self.act_observer, act_bits, signed=False)
        
        # Store original weight
        self.register_parameter("weight", nn.Parameter(linear.weight.data.clone()))
        if linear.bias is not None:
            self.register_parameter("bias", nn.Parameter(linear.bias.data.clone()))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations
        x_q = self.act_fake_quant(x)
        
        # Quantize weights
        w_q = self.weight_fake_quant(self.weight)
        
        # Fake quantized matmul
        return F.linear(x_q, w_q, self.bias)


class QuantizedAttention(nn.Module):
    """Quantized attention with optional pre-softmax clamping."""
    def __init__(self, attn_module: MHA_GQA, weight_bits: int = 8, act_bits: int = 8, 
                 clamp_pre_softmax: bool = True):
        super().__init__()
        self.clamp_pre_softmax = clamp_pre_softmax
        
        # Quantize Q/K/V/O projections (preserves weights)
        self.wq_quant = QuantizedLinear(attn_module.wq, weight_bits, act_bits)
        self.wk_quant = QuantizedLinear(attn_module.wk, weight_bits, act_bits)
        self.wv_quant = QuantizedLinear(attn_module.wv, weight_bits, act_bits)
        self.wo_quant = QuantizedLinear(attn_module.wo, weight_bits, act_bits)
        
        # Store other attributes
        self.n_heads = attn_module.n_heads
        self.n_kv_heads = attn_module.n_kv_heads
        self.d_head = attn_module.d_head
        self.head_groups = attn_module.head_groups
        self.rope = attn_module.rope
        self.attn_dropout = attn_module.attn_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        import math
        b, t, d = x.shape
        
        # Quantized Q/K/V projections
        q = self.wq_quant(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk_quant(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv_quant(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        # RoPE on Q/K
        q, k = self.rope.apply(q, k)

        # GQA expansion
        if self.head_groups > 1:
            k = k.repeat_interleave(self.head_groups, dim=1)
            v = v.repeat_interleave(self.head_groups, dim=1)

        # Attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Optional clamping before softmax
        if self.clamp_pre_softmax:
            # Clamp to reasonable range for INT8
            attn_scores = torch.clamp(attn_scores, min=-128.0, max=127.0)
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.d_head)
        
        return self.wo_quant(y)


def quantize_linear(linear: nn.Linear, weight_bits: int = 8, act_bits: int = 8) -> QuantizedLinear:
    """Replace a Linear layer with QuantizedLinear."""
    return QuantizedLinear(linear, weight_bits=weight_bits, act_bits=act_bits)


def quantize_attention(attn: MHA_GQA, weight_bits: int = 8, act_bits: int = 8,
                       clamp_pre_softmax: bool = True) -> QuantizedAttention:
    """Replace MHA_GQA with QuantizedAttention."""
    return QuantizedAttention(attn, weight_bits=weight_bits, act_bits=act_bits,
                              clamp_pre_softmax=clamp_pre_softmax)


def quantize_swiglu(swiglu: SwiGLU, weight_bits: int = 8, act_bits: int = 8) -> nn.Module:
    """Replace SwiGLU layers with quantized versions."""
    # Create new SwiGLU with quantized linear layers
    quantized = SwiGLU(swiglu.w1.in_features, swiglu.w1.out_features)
    quantized.w1 = quantize_linear(swiglu.w1, weight_bits, act_bits)
    quantized.w3 = quantize_linear(swiglu.w3, weight_bits, act_bits)
    quantized.w2 = quantize_linear(swiglu.w2, weight_bits, act_bits)
    return quantized


def quantize_model(model: StudentLM, weight_bits: int = 8, act_bits: int = 8,
                   fake_quant_in_attention: bool = True, clamp_pre_softmax: bool = True) -> StudentLM:
    """Convert model to QAT version with fake quantization.
    
    Recursively replaces:
    - Linear layers → QuantizedLinear
    - MHA_GQA → QuantizedAttention (if fake_quant_in_attention)
    - SwiGLU → Quantized SwiGLU
    """
    model = model  # Work in-place on the model
    
    # Quantize embedding (optional, usually kept FP16)
    # model.embed stays FP16 for now
    
    # Quantize blocks
    for block in model.blocks:
        # Quantize attention
        if fake_quant_in_attention:
            block.attn = quantize_attention(block.attn, weight_bits, act_bits, clamp_pre_softmax)
        else:
            # Still quantize linear layers in attention
            block.attn.wq = quantize_linear(block.attn.wq, weight_bits, act_bits)
            block.attn.wk = quantize_linear(block.attn.wk, weight_bits, act_bits)
            block.attn.wv = quantize_linear(block.attn.wv, weight_bits, act_bits)
            block.attn.wo = quantize_linear(block.attn.wo, weight_bits, act_bits)
        
        # Quantize MLP (SwiGLU)
        block.mlp = quantize_swiglu(block.mlp, weight_bits, act_bits)
    
    # Quantize output head
    model.lm_head = quantize_linear(model.lm_head, weight_bits, act_bits)
    
    return model


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> StudentLM:
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


def main():
    ap = argparse.ArgumentParser(description="QAT training with INT8 quantization")
    ap.add_argument('--checkpoint', required=True, help='Checkpoint to continue from')
    ap.add_argument('--config', nargs='+', required=True, help='Config file(s)')
    ap.add_argument('--output-dir', default='models/student/checkpoints', help='Output directory')
    ap.add_argument('--steps', type=int, default=30000, help='Number of training steps')
    ap.add_argument('--save-every', type=int, default=2000, help='Save checkpoint every N steps')
    ap.add_argument('--log-every', type=int, default=100, help='Log every N steps')
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    cfg = {}
    for config_path in args.config:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                cfg.update(yaml.safe_load(f))
            else:
                cfg.update(json.load(f))
    
    qat_cfg = cfg.get("qat", {})
    if not qat_cfg.get("enable", True):
        print("[quant_qat_int8] QAT disabled in config, exiting")
        return
    
    weight_bits = qat_cfg.get("weight_bits", 8)
    act_bits = qat_cfg.get("act_bits", 8)
    fake_quant_in_attention = qat_cfg.get("fake_quant_in_attention", True)
    clamp_pre_softmax = qat_cfg.get("clamp_pre_softmax", True)
    
    print(f"[quant_qat_int8] QAT config:")
    print(f"  weight_bits={weight_bits}, act_bits={act_bits}")
    print(f"  fake_quant_in_attention={fake_quant_in_attention}")
    print(f"  clamp_pre_softmax={clamp_pre_softmax}")
    
    # Load model
    print(f"[quant_qat_int8] Loading model from: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Apply quantization
    print("[quant_qat_int8] Applying quantization...")
    quantized_model = quantize_model(
        model,
        weight_bits=weight_bits,
        act_bits=act_bits,
        fake_quant_in_attention=fake_quant_in_attention,
        clamp_pre_softmax=clamp_pre_softmax
    )
    print("[quant_qat_int8] Quantization applied")
    
    # Setup optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        quantized_model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),  # Lower LR for QAT
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )
    
    # Setup FP16
    train_cfg = cfg.get("train", {})
    use_fp16 = train_cfg.get("fp16", False)
    scaler = GradScaler() if use_fp16 and device.type == 'cuda' else None
    
    # Create dataset
    io_cfg = cfg.get("io", {})
    train_shards = io_cfg.get("train_shards", ["data/kd_mix.jsonl"])
    tokenizer_path = io_cfg.get("tokenizer_path", "models/student/tokenizer")
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
    
    print(f"[quant_qat_int8] Starting QAT training:")
    print(f"  Device: {device}")
    print(f"  Steps: {args.steps}")
    
    quantized_model.train()
    step = 0
    
    for epoch in range(100):  # Max epochs
        for batch in dataloader:
            if step >= args.steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch.get("teacher_logits")
            if teacher_logits is not None:
                teacher_logits = teacher_logits.to(device)
            
            with autocast(enabled=scaler is not None):
                # Forward pass
                student_logits = quantized_model(input_ids, attention_mask)
                
                # KD loss
                kd_cfg = cfg.get("distillation", {})
                loss_dict = combined_kd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    teacher_targets=None,
                    ground_truth_targets=labels,
                    kl_weight=kd_cfg.get("kl_weight", 0.5),
                    ce_teacher_weight=kd_cfg.get("ce_teacher_weight", 0.3),
                    ce_ground_truth_weight=kd_cfg.get("ce_ground_truth_weight", 0.2),
                    kd_temperature=cfg.get("kd", {}).get("kd_temperature", 2.0),
                )
                
                loss = loss_dict["total"]
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), opt_cfg.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), opt_cfg.get("grad_clip", 1.0))
                optimizer.step()
            
            optimizer.zero_grad()
            step += 1
            
            if step % args.log_every == 0:
                loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                print(f"[quant_qat_int8] Step {step}/{args.steps}: {loss_str}")
            
            if step % args.save_every == 0:
                checkpoint_path = output_dir / f"qat_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": quantized_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_dict.get("total", 0.0),
                    "config": cfg,
                    "qat_config": qat_cfg,
                }, checkpoint_path)
                print(f"[quant_qat_int8] Saved checkpoint: {checkpoint_path}")
        
        if step >= args.steps:
            break
    
    # Final checkpoint
    final_path = output_dir / "qat_latest.pt"
    torch.save({
        "step": step,
        "model_state_dict": quantized_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_dict.get("total", 0.0),
        "config": cfg,
        "qat_config": qat_cfg,
    }, final_path)
    print(f"[quant_qat_int8] ✅ QAT training complete: {final_path}")


if __name__ == "__main__":
    main()
