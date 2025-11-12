"""
Knowledge distillation loss functions.

Implements:
- KL divergence loss (soft targets from teacher)
- Cross-entropy loss on teacher predictions
- Combined KD loss with configurable weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher distributions.
    
    Args:
        student_logits: [B, T, V] or [B*T, V] student logits
        teacher_logits: [B, T, V] or [B*T, V] teacher logits
        temperature: Temperature for softmax (default: 1.0)
        reduction: "mean", "sum", or "none"
        
    Returns:
        KL divergence loss
    """
    # Flatten if needed
    if student_logits.dim() == 3:
        student_logits = student_logits.view(-1, student_logits.size(-1))
    if teacher_logits.dim() == 3:
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
    
    # Apply temperature scaling
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
    # = sum(P_teacher * log(P_teacher)) - sum(P_teacher * log(P_student))
    kl = F.kl_div(student_probs, teacher_probs, reduction='none', log_target=False)
    kl = kl.sum(dim=-1)  # Sum over vocabulary
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def cross_entropy_on_teacher(
    student_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute cross-entropy loss where targets are teacher's predicted tokens.
    
    Args:
        student_logits: [B, T, V] student logits
        teacher_targets: [B, T] teacher's predicted token IDs
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Cross-entropy loss
    """
    # Flatten for cross-entropy
    logits_flat = student_logits.view(-1, student_logits.size(-1))
    targets_flat = teacher_targets.view(-1)
    
    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def combined_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    teacher_targets: Optional[torch.Tensor],
    ground_truth_targets: Optional[torch.Tensor],
    kl_weight: float = 0.5,
    ce_teacher_weight: float = 0.3,
    ce_ground_truth_weight: float = 0.2,
    kd_temperature: float = 2.0,
    ignore_index: int = -100
) -> dict:
    """
    Compute combined knowledge distillation loss.
    
    Args:
        student_logits: [B, T, V] student model logits
        teacher_logits: [B, T, V] teacher model logits (optional)
        teacher_targets: [B, T] teacher's predicted tokens (optional)
        ground_truth_targets: [B, T] ground truth token IDs (optional)
        kl_weight: Weight for KL divergence loss
        ce_teacher_weight: Weight for CE on teacher predictions
        ce_ground_truth_weight: Weight for CE on ground truth
        kd_temperature: Temperature for KD softmax
        ignore_index: Index to ignore in CE losses
        
    Returns:
        Dictionary with individual losses and total loss
    """
    losses = {}
    total_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    
    # KL divergence loss (if teacher logits available)
    if teacher_logits is not None and kl_weight > 0:
        kl_loss = kl_divergence(student_logits, teacher_logits, temperature=kd_temperature)
        losses["kl_div"] = kl_loss
        total_loss = total_loss + kl_weight * kl_loss
    
    # Cross-entropy on teacher predictions
    if teacher_targets is not None and ce_teacher_weight > 0:
        ce_teacher = cross_entropy_on_teacher(student_logits, teacher_targets, ignore_index=ignore_index)
        losses["ce_teacher"] = ce_teacher
        total_loss = total_loss + ce_teacher_weight * ce_teacher
    
    # Cross-entropy on ground truth (standard language modeling)
    if ground_truth_targets is not None and ce_ground_truth_weight > 0:
        logits_flat = student_logits.view(-1, student_logits.size(-1))
        targets_flat = ground_truth_targets.view(-1)
        ce_gt = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
        losses["ce_ground_truth"] = ce_gt
        total_loss = total_loss + ce_ground_truth_weight * ce_gt
    
    losses["total"] = total_loss
    return losses
