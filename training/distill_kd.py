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
from training.losses import (
    combined_kd_loss,
    caws_compliance_loss,
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
)
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

    # Enable self-evaluation head if configured
    use_self_evaluation = cfg.get("distillation", {}).get(
        "use_self_evaluation", False)
    model = StudentLM(model_cfg, use_self_evaluation=use_self_evaluation)

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

    # Initialize projection layers for intermediate layer matching if needed
    kd_cfg = cfg.get("distillation", {})
    if kd_cfg.get("use_intermediate_layers", False):
        layer_mapping = kd_cfg.get("layer_mapping", {})
        if layer_mapping:
            # Get teacher model dimensions from config
            teacher_d_model = cfg.get("teacher", {}).get(
                "d_model", model_cfg.d_model)

            if teacher_d_model != model_cfg.d_model:
                projection_layers = create_projection_layers(
                    student_d_model=model_cfg.d_model,
                    teacher_d_model=teacher_d_model,
                    layer_mapping=layer_mapping,
                    device=device,
                )
                # Store projection layers as model attribute for later use
                model.projection_layers = projection_layers
                print(
                    f"[distill_kd] Initialized {len(projection_layers)} projection layers for intermediate layer matching")

    return model


def create_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config, including projection layers if present."""
    opt_cfg = cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 2e-4)
    betas = opt_cfg.get("betas", [0.9, 0.95])
    weight_decay = opt_cfg.get("weight_decay", 0.1)

    # Collect all parameters (model + projection layers if present)
    params = list(model.parameters())

    # Add projection layers to optimizer if they exist
    if hasattr(model, 'projection_layers') and model.projection_layers:
        projection_params = [layer.parameters()
                             for layer in model.projection_layers.values()]
        # Flatten nested parameter lists
        for layer_params in projection_params:
            params.extend(list(layer_params))
        print(
            f"[distill_kd] Added {len(model.projection_layers)} projection layers to optimizer")

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
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
    model_state = model.module.state_dict() if isinstance(
        model, DDP) else model.state_dict()

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
    current_step: int = 0,
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

    # CoT-free validation: Fail if reasoning_content detected
    teacher_reasoning_content = batch.get("teacher_reasoning_content")
    if teacher_reasoning_content is not None:
        if isinstance(teacher_reasoning_content, list):
            teacher_reasoning_content = next(
                (rc for rc in teacher_reasoning_content if rc is not None), None)
        if teacher_reasoning_content:
            raise ValueError(
                "CoT-free training: teacher_reasoning_content detected in batch. "
                "Training on reasoning_content violates ToS. Use process-step supervision instead."
            )

    # Forward pass
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        kd_cfg = cfg.get("distillation", {})

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Intermediate Layer Matching
        # ====================================================================
        # Enable intermediate layer matching if configured
        use_intermediate_layers = kd_cfg.get("use_intermediate_layers", False)
        return_hidden_states = use_intermediate_layers

        # Get student logits (and optionally hidden states)
        if return_hidden_states:
            student_logits, student_hidden_states = model(
                input_ids,
                attention_mask,
                return_hidden_states=True
            )  # [B, T, V], List[[B, T, D]]
        else:
            student_logits = model(input_ids, attention_mask)  # [B, T, V]
            student_hidden_states = None

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Self-Evaluation Head
        # ====================================================================
        # Enable self-evaluation if configured
        use_self_eval = kd_cfg.get("use_self_evaluation", False)
        return_eval_score = use_self_eval

        if return_eval_score:
            # Re-run forward pass if we need eval score but didn't get it above
            if not return_hidden_states:
                student_logits, eval_score = model(
                    input_ids,
                    attention_mask,
                    return_eval_score=True
                )  # [B, T, V], [B, 1]
            else:
                # Get eval score separately (model doesn't support both flags simultaneously)
                _, eval_score = model(
                    input_ids,
                    attention_mask,
                    return_eval_score=True
                )  # [B, 1]
        else:
            eval_score = None

        # Compute loss

        # Get tokenizer for process-step supervision, CAWS compliance, and claim extraction
        tokenizer = None
        needs_tokenizer = (
            kd_cfg.get("w_tool", 0.0) > 0 or
            kd_cfg.get("w_args", 0.0) > 0 or
            kd_cfg.get("use_caws_compliance", False) or
            kd_cfg.get("use_claim_extraction", False)
        )
        if needs_tokenizer:
            # Try to get tokenizer from model or config
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            elif hasattr(model, 'module') and hasattr(model.module, 'tokenizer'):
                tokenizer = model.module.tokenizer
            elif 'tokenizer_path' in cfg:
                from training.dataset import load_tokenizer
                tokenizer = load_tokenizer(cfg['tokenizer_path'])

        # PRIORITY 2: Entropy-based scheduling (data-driven, replaces linear schedules)
        use_entropy_scheduling = kd_cfg.get("use_entropy_scheduling", False)

        if use_entropy_scheduling and teacher_logits is not None:
            # Compute entropy from teacher logits and derive temperature/weights
            from training.losses import entropy_weighting

            current_temperature, entropy_weights = entropy_weighting(
                teacher_logits=teacher_logits,
                min_entropy=kd_cfg.get("min_entropy", 2.0),
                max_entropy=kd_cfg.get("max_entropy", 8.0),
                min_temp=kd_cfg.get("min_temperature", 1.5),
                max_temp=kd_cfg.get("max_temperature", 3.0),
                base_kl_weight=kd_cfg.get("kl_weight", 0.4),
                base_ce_teacher_weight=kd_cfg.get("ce_teacher_weight", 0.2),
                base_ce_gt_weight=kd_cfg.get("ce_ground_truth_weight", 0.2),
            )

            kl_weight = entropy_weights["kl_weight"]
            ce_teacher_weight = entropy_weights["ce_teacher_weight"]
            ce_ground_truth_weight = entropy_weights["ce_ground_truth_weight"]
            entropy_value = entropy_weights.get("entropy", 0.0)

            # Log entropy for monitoring
            if current_step % 100 == 0:
                print(f"[distill_kd] Entropy: {entropy_value:.3f}, Temp: {current_temperature:.3f}, "
                      f"KL: {kl_weight:.3f}, CE_GT: {ce_ground_truth_weight:.3f}")
        else:
            # Fallback to linear schedules if entropy scheduling disabled or teacher_logits unavailable
            # Calculate adaptive temperature if enabled
            current_temperature = cfg.get("kd", {}).get("kd_temperature", 2.0)
            if kd_cfg.get("use_temperature_schedule", False):
                from training.losses import adaptive_temperature
                total_steps = cfg.get("train", {}).get("total_steps", 100000)
                # Use current_step for scheduling (actual training step, not grad accum counter)
                current_temperature = adaptive_temperature(
                    step=current_step,
                    total_steps=total_steps,
                    base_temp=kd_cfg.get("base_temperature", 2.0),
                    min_temp=kd_cfg.get("min_temperature", 1.5),
                    max_temp=kd_cfg.get("max_temperature", 3.0),
                )

            # Calculate adaptive loss weights if enabled
            kl_weight = kd_cfg.get("kl_weight", 0.4)
            ce_teacher_weight = kd_cfg.get("ce_teacher_weight", 0.2)
            ce_ground_truth_weight = kd_cfg.get("ce_ground_truth_weight", 0.2)

            if kd_cfg.get("use_weight_schedule", False):
                from training.losses import loss_weight_schedule
                total_steps = cfg.get("train", {}).get("total_steps", 100000)
                # Use current_step for scheduling (actual training step, not grad accum counter)
                weights = loss_weight_schedule(
                    step=current_step,
                    total_steps=total_steps,
                    early_teacher_weight=kd_cfg.get(
                        "early_teacher_weight", 0.7),
                    late_teacher_weight=kd_cfg.get("late_teacher_weight", 0.3),
                    early_gt_weight=kd_cfg.get("early_gt_weight", 0.3),
                    late_gt_weight=kd_cfg.get("late_gt_weight", 0.7),
                )
                kl_weight = weights["kl_weight"]
                ce_teacher_weight = weights["ce_teacher_weight"]
                ce_ground_truth_weight = weights["ce_ground_truth_weight"]

        # Process-step supervision weights (replaces ce_reasoning_weight)
        w_tool = kd_cfg.get("w_tool", 0.15)
        w_args = kd_cfg.get("w_args", 0.15)
        w_integr = kd_cfg.get("w_integr", 0.10)

        # Extract process-step supervision targets from batch
        tool_name_ids = batch.get("tool_name_ids")
        tool_name_mask = batch.get("tool_name_mask")
        gold_json_text_ids = batch.get("gold_json_text_ids")
        mask_valid_json_tokens = batch.get("mask_valid_json_tokens")
        tool_result_fields = batch.get("tool_result_fields")
        integration_mask = batch.get("integration_mask")

        # Move to device if present
        if tool_name_ids is not None:
            tool_name_ids = tool_name_ids.to(device)
        if tool_name_mask is not None:
            tool_name_mask = tool_name_mask.to(device)
        if gold_json_text_ids is not None:
            gold_json_text_ids = gold_json_text_ids.to(device)
        if mask_valid_json_tokens is not None:
            mask_valid_json_tokens = mask_valid_json_tokens.to(device)
        if tool_result_fields is not None:
            tool_result_fields = tool_result_fields.to(device)
        if integration_mask is not None:
            integration_mask = integration_mask.to(device)

        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_target_ids,
            ground_truth_targets=labels,
            # Process-step supervision targets (replaces teacher_reasoning_content)
            tool_name_ids=tool_name_ids,
            tool_name_mask=tool_name_mask,
            gold_json_text_ids=gold_json_text_ids,
            mask_valid_json_tokens=mask_valid_json_tokens,
            tool_result_fields=tool_result_fields,
            integration_mask=integration_mask,
            # Loss weights
            kl_weight=kl_weight,
            ce_teacher_weight=ce_teacher_weight,
            w_tool=w_tool,
            w_args=w_args,
            w_integr=w_integr,
            ce_ground_truth_weight=ce_ground_truth_weight,
            kd_temperature=current_temperature,
        )

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Intermediate Layer Loss
        # ====================================================================
        if use_intermediate_layers and student_hidden_states is not None:
            # Get teacher hidden states from batch (if available)
            # NOTE: Teacher hidden states must be extracted during dataset generation
            # or via a teacher model wrapper that exposes hidden states
            teacher_hidden_states = batch.get("teacher_hidden_states")

            if teacher_hidden_states is not None:
                # Move to device if tensor
                if isinstance(teacher_hidden_states, list):
                    teacher_hidden_states = [
                        h.to(device) if isinstance(h, torch.Tensor) else h
                        for h in teacher_hidden_states
                    ]
                elif isinstance(teacher_hidden_states, torch.Tensor):
                    teacher_hidden_states = teacher_hidden_states.to(device)

                # Layer mapping: student_layer_idx -> teacher_layer_idx
                # Example: Map student layers [0, 8, 16, 24] to teacher layers [0, 16, 32, 48]
                layer_mapping = kd_cfg.get("layer_mapping", {})
                if not layer_mapping:
                    # Default: Map evenly spaced student layers to teacher layers
                    # Assuming teacher has ~2x more layers than student
                    n_student_layers = len(
                        student_hidden_states) - 1  # Exclude embedding
                    n_teacher_layers = len(
                        teacher_hidden_states) - 1 if isinstance(teacher_hidden_states, list) else 0
                    if n_teacher_layers > 0:
                        # Map every 4th student layer to corresponding teacher layer
                        layer_mapping = {
                            i: int(i * (n_teacher_layers / n_student_layers))
                            for i in range(0, n_student_layers, 4)
                        }

                # Use pre-initialized projection layers from model (if available)
                # or create them on-the-fly if dimensions differ
                projection_layers = None
                if hasattr(model, 'projection_layers') and model.projection_layers:
                    # Use pre-initialized projection layers
                    projection_layers = model.projection_layers
                elif layer_mapping:
                    student_d_model = student_hidden_states[0].size(-1)
                    teacher_d_model = teacher_hidden_states[0].size(-1) if isinstance(
                        teacher_hidden_states, list) else student_d_model

                    if student_d_model != teacher_d_model:
                        # Create projection layers on-the-fly (fallback)
                        # NOTE: These won't be in optimizer, so they won't be trained
                        # Better to initialize them during model creation
                        projection_layers = create_projection_layers(
                            student_d_model=student_d_model,
                            teacher_d_model=teacher_d_model,
                            layer_mapping=layer_mapping,
                            device=device,
                        )
                        print("[distill_kd] WARN: Created projection layers on-the-fly. "
                              "Consider initializing them during model creation for proper training.")

                # Compute intermediate layer loss
                intermediate_loss = intermediate_layer_loss(
                    student_hidden_states=student_hidden_states,
                    teacher_hidden_states=teacher_hidden_states,
                    layer_mapping=layer_mapping,
                    projection_layers=projection_layers,
                )

                # Add to loss dict with configurable weight
                intermediate_weight = kd_cfg.get(
                    "intermediate_layer_weight", 0.1)
                loss_dict["intermediate_layer"] = intermediate_loss
                loss_dict["total"] = loss_dict["total"] + \
                    intermediate_weight * intermediate_loss

        # ====================================================================
        # PRIORITY 3: JSON Repair Loop + Metric
        # ====================================================================
        # Check JSON validity and repair needs for tool-use batches
        use_json_repair_check = kd_cfg.get("use_json_repair_check", False)
        json_repair_weight = kd_cfg.get("json_repair_weight", 0.05)

        if use_json_repair_check and json_repair_weight > 0:
            # Only check on tool-use batches (when process-step targets present)
            # This avoids expensive text generation on every batch
            if tool_name_ids is not None or gold_json_text_ids is not None:
                # Generate text from student logits for repair checking
                # Use greedy decoding for efficiency
                if tokenizer is None:
                    # Try to get tokenizer
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                    elif hasattr(model, 'module') and hasattr(model.module, 'tokenizer'):
                        tokenizer = model.module.tokenizer
                    elif 'tokenizer_path' in cfg:
                        from training.dataset import load_tokenizer
                        tokenizer = load_tokenizer(cfg['tokenizer_path'])

                if tokenizer is not None:
                    try:
                        from training.json_repair import check_json_repair_needed, batch_check_json_repair

                        # Generate text from logits (greedy decoding)
                        # Only generate for the response portion (after prompt)
                        pred_token_ids = student_logits.argmax(
                            dim=-1)  # [B, T]
                        generated_texts = []

                        for i in range(pred_token_ids.size(0)):
                            # Decode tokens to text
                            tokens = pred_token_ids[i].cpu().tolist()
                            try:
                                text = tokenizer.decode(
                                    tokens, skip_special_tokens=True)
                                generated_texts.append(text)
                            except:
                                generated_texts.append("")

                        # Check repair needs for batch
                        repair_metrics = batch_check_json_repair(
                            generated_texts, use_jsonrepair=True)

                        # Compute repair loss per sample
                        from training.losses import json_repair_loss
                        repair_losses = []
                        for text in generated_texts:
                            _, needs_repair = check_json_repair_needed(
                                text, use_jsonrepair=True)
                            repair_loss = json_repair_loss(needs_repair)
                            repair_losses.append(repair_loss)

                        if repair_losses:
                            # Average repair loss over batch
                            batch_repair_loss = torch.stack(
                                repair_losses).mean()
                            loss_dict["json_repair"] = batch_repair_loss
                            loss_dict["total"] = loss_dict["total"] + \
                                json_repair_weight * batch_repair_loss

                            # Log repair metrics periodically
                            if current_step % 100 == 0:
                                print(f"[distill_kd] JSON repair: rate={repair_metrics['repair_rate']:.3f}, "
                                      f"valid={repair_metrics['valid_json_count']}/{repair_metrics['total']}")
                    except Exception as e:
                        # Don't fail training if repair check fails
                        if current_step % 100 == 0:
                            print(
                                f"[distill_kd] WARN: JSON repair check failed: {e}")

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Self-Evaluation Loss
        # ====================================================================
        if use_self_eval and eval_score is not None:
            # Get teacher quality score from batch
            # NOTE: Teacher quality scores should be computed during dataset generation
            # or via a quality scoring mechanism (e.g., human eval, automated metrics)
            teacher_quality_score = batch.get("teacher_quality_score")

            if teacher_quality_score is not None:
                # Handle different formats
                if isinstance(teacher_quality_score, (int, float)):
                    teacher_quality = float(teacher_quality_score)
                elif isinstance(teacher_quality_score, torch.Tensor):
                    teacher_quality = teacher_quality_score.to(device)
                else:
                    teacher_quality = None

                if teacher_quality is not None:
                    # Compute self-evaluation loss
                    eval_loss = self_evaluation_loss(
                        student_eval_score=eval_score,
                        teacher_quality_score=teacher_quality,
                    )

                    # Add to loss dict with configurable weight
                    eval_weight = kd_cfg.get("self_evaluation_weight", 0.05)
                    loss_dict["self_evaluation"] = eval_loss
                    loss_dict["total"] = loss_dict["total"] + \
                        eval_weight * eval_loss

        # ====================================================================
        # PRIORITY 5: CAWS Structure Scoring
        # ====================================================================
        # Compute CAWS structure scores and add structure loss
        use_caws_structure = kd_cfg.get("use_caws_structure", False)
        caws_structure_weight = kd_cfg.get("caws_structure_weight", 0.05)
        
        if use_caws_structure and caws_structure_weight > 0:
            # Only check on batches with text outputs (when teacher_text available)
            teacher_text = batch.get("teacher_text")
            if teacher_text is not None:
                # Generate text from student logits for structure comparison
                if tokenizer is None:
                    # Try to get tokenizer
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                    elif hasattr(model, 'module') and hasattr(model.module, 'tokenizer'):
                        tokenizer = model.module.tokenizer
                    elif 'tokenizer_path' in cfg:
                        from training.dataset import load_tokenizer
                        tokenizer = load_tokenizer(cfg['tokenizer_path'])
                
                if tokenizer is not None:
                    try:
                        from training.caws_structure import caws_structure_score
                        from training.losses import caws_structure_loss
                        
                        # Generate text from student logits (greedy decoding)
                        pred_token_ids = student_logits.argmax(dim=-1)  # [B, T]
                        student_texts = []
                        
                        for i in range(pred_token_ids.size(0)):
                            tokens = pred_token_ids[i].cpu().tolist()
                            try:
                                text = tokenizer.decode(tokens, skip_special_tokens=True)
                                student_texts.append(text)
                            except:
                                student_texts.append("")
                        
                        # Compute structure scores
                        teacher_text_normalized = teacher_text
                        if isinstance(teacher_text, list):
                            teacher_text_normalized = teacher_text[0] if teacher_text else ""
                        elif not isinstance(teacher_text, str):
                            teacher_text_normalized = str(teacher_text)
                        
                        teacher_structure_score = caws_structure_score(teacher_text_normalized)
                        
                        # Compute structure loss for each student output
                        structure_losses = []
                        for student_text in student_texts:
                            student_structure_score = caws_structure_score(student_text)
                            struct_loss = caws_structure_loss(
                                teacher_score=teacher_structure_score,
                                student_score=student_structure_score
                            )
                            structure_losses.append(struct_loss)
                        
                        if structure_losses:
                            # Average structure loss over batch
                            batch_structure_loss = torch.stack(structure_losses).mean()
                            loss_dict["caws_structure"] = batch_structure_loss
                            loss_dict["total"] = loss_dict["total"] + caws_structure_weight * batch_structure_loss
                            
                            # Log structure scores periodically
                            if current_step % 100 == 0:
                                avg_student_score = sum(caws_structure_score(st) for st in student_texts) / len(student_texts) if student_texts else 0.0
                                print(f"[distill_kd] CAWS structure: teacher={teacher_structure_score:.3f}, "
                                      f"student_avg={avg_student_score:.3f}, loss={batch_structure_loss.item():.3f}")
                    except Exception as e:
                        # Don't fail training if structure check fails
                        if current_step % 100 == 0:
                            print(f"[distill_kd] WARN: CAWS structure check failed: {e}")

        # CAWS compliance loss (optional, config-driven)
        if kd_cfg.get("use_caws_compliance", False):
            # Get teacher text from batch if available
            teacher_text = batch.get("teacher_text")
            if teacher_text is None:
                # Try to get from batch metadata
                teacher_text = batch.get("metadata", {}).get(
                    "teacher_text") if isinstance(batch.get("metadata"), dict) else None

            if teacher_text:
                # Generate student output for compliance check
                # Use argmax decoding (greedy) for efficiency
                student_output_ids = student_logits.argmax(dim=-1)  # [B, T]

                # Decode student output (only first sequence in batch for efficiency)
                if tokenizer:
                    try:
                        student_output = tokenizer.decode(
                            student_output_ids[0].cpu().tolist(),
                            skip_special_tokens=True
                        )

                        # Get claim extractor if available (optional)
                        claim_extractor = None
                        if "claim_extractor" in cfg:
                            # Claim extractor would be passed via config or initialized elsewhere
                            # For now, use None (will fall back to structure check)
                            pass

                        # Compute CAWS compliance loss
                        compliance_loss = caws_compliance_loss(
                            student_output=student_output,
                            teacher_output=teacher_text if isinstance(
                                teacher_text, str) else teacher_text[0] if isinstance(teacher_text, list) else "",
                            claim_extractor=claim_extractor,
                        )

                        # Add to loss with configurable weight
                        compliance_weight = kd_cfg.get(
                            "caws_compliance_weight", 0.05)
                        compliance_loss_scaled = compliance_loss.to(
                            device) * compliance_weight
                        loss_dict["caws_compliance"] = compliance_loss_scaled
                        loss_dict["total"] = loss_dict["total"] + \
                            compliance_loss_scaled
                    except Exception as e:
                        # If decoding fails, skip compliance loss
                        print(
                            f"[distill_kd] WARN: Failed to compute CAWS compliance loss: {e}")

        # ====================================================================
        # PRIORITY 4 INTEGRATION: Claim Extraction Loss
        # ====================================================================
        if kd_cfg.get("use_claim_extraction", False):
            # Get teacher text from batch
            teacher_text = batch.get("teacher_text")
            if teacher_text is None:
                # Try to get from batch metadata
                teacher_text = batch.get("metadata", {}).get(
                    "teacher_text") if isinstance(batch.get("metadata"), dict) else None

            if teacher_text and tokenizer:
                # Generate student output for claim extraction comparison
                # Use argmax decoding (greedy) for efficiency
                student_output_ids = student_logits.argmax(dim=-1)  # [B, T]

                # Decode student output (only first sequence in batch for efficiency)
                try:
                    student_output = tokenizer.decode(
                        student_output_ids[0].cpu().tolist(),
                        skip_special_tokens=True
                    )

                    # Get claim extractor if available (optional, will create default if None)
                    claim_extractor = None
                    if "claim_extractor" in cfg:
                        # Claim extractor would be passed via config or initialized elsewhere
                        # For now, use None (will create SimpleClaimExtractor in loss function)
                        pass

                    # Normalize teacher text format
                    teacher_text_normalized = teacher_text
                    if isinstance(teacher_text, list):
                        teacher_text_normalized = teacher_text[0] if teacher_text else ""
                    elif not isinstance(teacher_text, str):
                        teacher_text_normalized = str(teacher_text)

                    # Compute claim extraction loss
                    from training.losses import claim_extraction_loss
                    claim_loss = claim_extraction_loss(
                        student_output=student_output,
                        teacher_output=teacher_text_normalized,
                        claim_extractor=claim_extractor,
                        min_claim_ratio=kd_cfg.get("min_claim_ratio", 0.5),
                        min_success_rate_ratio=kd_cfg.get(
                            "min_success_rate_ratio", 0.7),
                    )

                    # Add to loss with configurable weight
                    claim_weight = kd_cfg.get("claim_extraction_weight", 0.1)
                    claim_loss_scaled = claim_loss.to(device) * claim_weight
                    loss_dict["claim_extraction"] = claim_loss_scaled
                    loss_dict["total"] = loss_dict["total"] + claim_loss_scaled

                    # Also log claim extraction metrics for monitoring
                    if kd_cfg.get("log_claim_metrics", True):
                        from training.claim_extraction import compute_claim_extraction_metrics
                        claim_metrics = compute_claim_extraction_metrics(
                            student_output=student_output,
                            teacher_output=teacher_text_normalized,
                            claim_extractor=claim_extractor,
                        )
                        # Add metrics to loss_dict for logging (as floats, not tensors)
                        loss_dict["claim_count_student"] = float(
                            claim_metrics.get("student_claim_count", 0))
                        loss_dict["claim_count_teacher"] = float(
                            claim_metrics.get("teacher_claim_count", 0))
                        loss_dict["claim_ratio"] = float(
                            claim_metrics.get("claim_ratio", 0.0))
                        loss_dict["success_rate_student"] = float(
                            claim_metrics.get("student_success_rate", 0.0))
                        loss_dict["success_rate_teacher"] = float(
                            claim_metrics.get("teacher_success_rate", 0.0))
                        loss_dict["success_rate_ratio"] = float(
                            claim_metrics.get("success_rate_ratio", 0.0))
                except Exception as e:
                    # If decoding or claim extraction fails, skip claim extraction loss
                    print(
                        f"[distill_kd] WARN: Failed to compute claim extraction loss: {e}")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get(
                "optimizer", {}).get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get(
                "optimizer", {}).get("grad_clip", 1.0))
            optimizer.step()
        optimizer.zero_grad()

    # Convert to float for logging
    loss_dict_float = {k: float(v.item()) if isinstance(
        v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
    return loss_dict_float


def main():
    ap = argparse.ArgumentParser(description="Knowledge distillation training")
    ap.add_argument('--config', nargs='+', required=True,
                    help='Config file(s) to load')
    ap.add_argument('--resume', help='Resume from checkpoint path')
    ap.add_argument('--output-dir', default='models/student/checkpoints',
                    help='Output directory for checkpoints')
    ap.add_argument('--local-rank', type=int, default=-1,
                    help='Local rank for distributed training')
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
        tokenizer_path = cfg.get("tokenizer", {}).get(
            "path", "models/student/tokenizer")

    if not tokenizer_path:
        raise ValueError(
            "tokenizer_path must be specified in config (io.tokenizer_path)")

    train_shards = io_cfg.get("train_shards", ["data/kd_mix.jsonl"])

    # Use first sequence length for initial dataset
    seq_lengths = train_cfg.get("seq_lengths", [4096])
    current_seq_len = get_sequence_length(
        start_step, seq_lengths, cfg.get("curriculum", {}).get("schedule"))

    dataset = KDDataset(
        jsonl_path=train_shards[0],
        tokenizer_path=tokenizer_path,
        max_seq_length=current_seq_len,
        teacher_logits_available=cfg.get("kd", {}).get(
            "teacher_logits_available", False),
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
    tracer = create_tracer_from_config(
        cfg, run_name=run_name) if is_main_process else None

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
    print(
        f"  Micro batch size: {micro_batch_size}, Gradient accumulation: {grad_accum}")
    print(f"  FP16: {use_fp16}")
    print(f"  Sequence lengths: {seq_lengths}")
    if tracer:
        print(
            f"  Tracing enabled: TensorBoard={tracer.use_tensorboard}, WandB={tracer.use_wandb}")

    step = start_step
    grad_accum_counter = 0

    # Iterate over dataset multiple times if needed
    while step < total_steps:
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break

            # Update sequence length based on curriculum
            current_seq_len = get_sequence_length(
                step, seq_lengths, cfg.get("curriculum", {}).get("schedule"))
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
                current_step=step,
            )

            grad_accum_counter = (grad_accum_counter + 1) % grad_accum
            step += 1

            # Logging
            if step % log_every == 0 and is_main_process:
                if tracer:
                    # Log to tracer (includes console, TensorBoard, WandB, JSON)
                    tracer.log_metrics(
                        step=step, metrics=loss_dict, prefix="train/")

                    # Also log learning rate if available
                    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                        lr = optimizer.param_groups[0].get('lr', 0.0)
                        tracer.log_metrics(step=step, metrics={
                                           "learning_rate": lr}, prefix="train/")
                else:
                    # Fallback to console logging
                    loss_str = ", ".join(
                        [f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    print(
                        f"[distill_kd] Step {step}/{total_steps}: {loss_str}")

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
