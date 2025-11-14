"""
Knowledge distillation loss functions.

Implements:
- KL divergence loss (soft targets from teacher)
- Cross-entropy loss on teacher predictions
- Process-step supervision losses (tool name, JSON args, integration)
- Combined KD loss with configurable weights
- Temperature scheduling
- Loss weight scheduling
- Intermediate layer matching
- Self-evaluation loss
- Length-aware KD loss (hinged, completeness-aware)
- Early tool call loss (gated + ramped)
- Halt head loss (for learned halting)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import re


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
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
    kl = F.kl_div(student_probs, teacher_probs, reduction="none", log_target=False)
    kl = kl.sum(dim=-1)  # Sum over vocabulary

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def cross_entropy_on_teacher(
    student_logits: torch.Tensor, teacher_targets: torch.Tensor, ignore_index: int = -100
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


def tool_name_loss(
    student_logits: torch.Tensor,
    tool_name_ids: torch.Tensor,
    tool_name_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss over tool name token span.

    Supervises tool selection decision without training on reasoning prose.
    Locates tool name tokens in the sequence and computes loss at those positions.

    Args:
        student_logits: [B, T, V] student logits
        tool_name_ids: [B, T_tool] tool name token IDs from teacher
        tool_name_mask: [B, T_tool] mask for tool name tokens (1 = valid, 0 = ignore)
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss on tool name tokens only
    """
    # Find positions where tool name tokens appear in the sequence
    # For simplicity, assume tool_name_ids are aligned with sequence positions
    tool_length = min(tool_name_ids.size(1), student_logits.size(1))
    tool_logits = student_logits[:, :tool_length, :].contiguous()
    tool_targets = tool_name_ids[:, :tool_length].contiguous()
    tool_mask = tool_name_mask[:, :tool_length]

    # Apply mask: only compute loss on valid tool name tokens
    tool_targets = torch.where(
        tool_mask.bool(), tool_targets, torch.full_like(tool_targets, ignore_index)
    )

    # Flatten
    logits_flat = tool_logits.view(-1, tool_logits.size(-1))
    targets_flat = tool_targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def json_argument_loss(
    student_logits: torch.Tensor,
    gold_json_text_ids: torch.Tensor,
    mask_valid_json_tokens: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss only on JSON structural tokens (not narrative prose).

    Supervises JSON argument structure without training on reasoning prose.

    Args:
        student_logits: [B, T, V] student logits
        gold_json_text_ids: [B, T_json] gold JSON token IDs from teacher
        mask_valid_json_tokens: [B, T_json] mask indicating valid JSON tokens (1) vs prose (0)
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss on JSON tokens only
    """
    # Extract logits at JSON positions
    json_length = min(gold_json_text_ids.size(1), student_logits.size(1))
    json_logits = student_logits[:, :json_length, :].contiguous()
    json_targets = gold_json_text_ids[:, :json_length].contiguous()
    json_mask = mask_valid_json_tokens[:, :json_length]

    # Apply mask: only compute loss on valid JSON tokens
    json_targets = torch.where(
        json_mask.bool(), json_targets, torch.full_like(json_targets, ignore_index)
    )

    # Flatten
    logits_flat = json_logits.view(-1, json_logits.size(-1))
    targets_flat = json_targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def integration_copy_loss(
    student_logits: torch.Tensor,
    tool_result_fields: torch.Tensor,
    integration_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Pointer-like loss for copying cited fields from tool result.

    Encourages student to integrate tool results similar to teacher.

    Args:
        student_logits: [B, T, V] student logits
        tool_result_fields: [B, T_int] token IDs of tool result fields to copy
        integration_mask: [B, T_int] mask indicating integration span positions
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss on integration spans
    """
    # Extract logits at integration positions
    int_length = min(tool_result_fields.size(1), student_logits.size(1))
    int_logits = student_logits[:, :int_length, :].contiguous()
    int_targets = tool_result_fields[:, :int_length].contiguous()
    int_mask = integration_mask[:, :int_length]

    # Apply mask: only compute loss on integration spans
    int_targets = torch.where(
        int_mask.bool(), int_targets, torch.full_like(int_targets, ignore_index)
    )

    # Flatten
    logits_flat = int_logits.view(-1, int_logits.size(-1))
    targets_flat = int_targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def halt_head_loss(
    halt_logits: torch.Tensor,
    halt_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for halt head predictions.

    Supervises learned halting: predicts when refinement should stop.

    Args:
        halt_logits: [B, 2] halt head logits [continue, halt]
        halt_targets: [B] target class (0 = continue, 1 = halt)

    Returns:
        Cross-entropy loss
    """
    return F.cross_entropy(halt_logits, halt_targets)


def intermediate_layer_loss(
    student_hidden_states: List[torch.Tensor],
    teacher_hidden_states: List[torch.Tensor],
    layer_mapping: Dict[int, int],
    projection_layers: Optional[List[nn.Module]] = None,
) -> torch.Tensor:
    """
    MSE loss between intermediate hidden states of student and teacher.

    Args:
        student_hidden_states: List of [B, T, D_s] hidden states per layer
        teacher_hidden_states: List of [B, T, D_t] hidden states per layer
        layer_mapping: Dict mapping student layer indices to teacher layer indices
        projection_layers: Optional list of projection layers to align dimensions

    Returns:
        MSE loss averaged over matched layers
    """
    losses = []
    for student_idx, teacher_idx in layer_mapping.items():
        if student_idx >= len(student_hidden_states) or teacher_idx >= len(teacher_hidden_states):
            continue

        student_h = student_hidden_states[student_idx]  # [B, T, D_s]
        teacher_h = teacher_hidden_states[teacher_idx]  # [B, T, D_t]

        # Project student hidden state if needed
        if projection_layers and student_idx < len(projection_layers):
            student_h = projection_layers[student_idx](student_h)

        # Ensure same shape
        min_seq_len = min(student_h.size(1), teacher_h.size(1))
        student_h = student_h[:, :min_seq_len, :]
        teacher_h = teacher_h[:, :min_seq_len, :]

        # MSE loss
        mse = F.mse_loss(student_h, teacher_h)
        losses.append(mse)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=student_hidden_states[0].device, requires_grad=True)


def self_evaluation_loss(
    student_eval_score: torch.Tensor,
    teacher_quality_score: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss between student self-evaluation score and teacher quality score.

    Args:
        student_eval_score: [B, 1] student self-evaluation score (0-1)
        teacher_quality_score: [B] teacher quality score (0-1)

    Returns:
        MSE loss
    """
    teacher_score = teacher_quality_score.unsqueeze(1)  # [B, 1]
    return F.mse_loss(student_eval_score, teacher_score)


def create_projection_layers(
    student_d_model: int,
    teacher_d_model: int,
    layer_mapping: Dict[int, int],
    device: torch.device,
) -> List[nn.Module]:
    """
    Create projection layers to align student and teacher hidden dimensions.

    Args:
        student_d_model: Student model dimension
        teacher_d_model: Teacher model dimension
        layer_mapping: Dict mapping student layer indices to teacher layer indices
        device: Device to place layers on

    Returns:
        List of projection layers (one per student layer in mapping)
    """
    max_student_layer = max(layer_mapping.keys()) if layer_mapping else 0
    projection_layers = []

    for i in range(max_student_layer + 1):
        if i in layer_mapping:
            # Create projection layer: student_dim -> teacher_dim
            proj = nn.Linear(student_d_model, teacher_d_model, bias=False)
            proj = proj.to(device)
            projection_layers.append(proj)
        else:
            projection_layers.append(None)

    return projection_layers


def length_aware_kd_loss(
    student_length: int,
    teacher_length: int,
    required_fields_present: torch.Tensor,
    max_allowed_ratio: float = 1.2,
) -> torch.Tensor:
    """
    Penalize student length only when it exceeds teacher by more than max_allowed_ratio
    AND required fields are missing.

    Hinged loss: only penalizes when student is both too long AND incomplete.

    Args:
        student_length: Student sequence length
        teacher_length: Teacher sequence length
        required_fields_present: [B] boolean tensor indicating if all required fields present
        max_allowed_ratio: Maximum allowed length ratio (default: 1.2 = 20% longer)

    Returns:
        Length penalty loss (scalar tensor)
    """
    if teacher_length == 0:
        return torch.tensor(0.0, requires_grad=True)

    length_ratio = student_length / teacher_length
    exceeds_ratio = length_ratio > max_allowed_ratio

    # Only penalize if exceeds ratio AND fields missing
    # required_fields_present is [B], take mean to get batch-level penalty
    fields_missing = (~required_fields_present).float().mean()

    if exceeds_ratio and fields_missing > 0:
        # Penalty proportional to excess length and missing fields
        excess = length_ratio - max_allowed_ratio
        penalty = excess * fields_missing
        return torch.tensor(penalty, requires_grad=True)
    else:
        return torch.tensor(0.0, requires_grad=True)


def early_tool_call_loss(
    student_tool_call_position: int,
    teacher_tool_call_position: int,
    sequence_length: int,
    ramp_start: int = 1000,
    ramp_end: int = 5000,
) -> torch.Tensor:
    """
    Gated + ramped loss encouraging early tool calls.

    Penalizes student for calling tools later than teacher, with ramping schedule.

    Args:
        student_tool_call_position: Position of student's first tool call (-1 if none)
        teacher_tool_call_position: Position of teacher's first tool call (-1 if none)
        sequence_length: Total sequence length
        ramp_start: Step to start ramping loss weight
        ramp_end: Step to reach full loss weight

    Returns:
        Early tool call penalty loss
    """
    # If teacher didn't call tools, no penalty
    if teacher_tool_call_position < 0:
        return torch.tensor(0.0, requires_grad=True)

    # If student didn't call tools, penalize heavily
    if student_tool_call_position < 0:
        return torch.tensor(1.0, requires_grad=True)

    # Normalize positions to [0, 1]
    student_norm = student_tool_call_position / sequence_length
    teacher_norm = teacher_tool_call_position / sequence_length

    # Penalty: difference in normalized positions
    diff = student_norm - teacher_norm

    # Only penalize if student is later
    if diff > 0:
        return torch.tensor(float(diff), requires_grad=True)
    else:
        return torch.tensor(0.0, requires_grad=True)


def curriculum_temperature(epoch: int, total_epochs: int) -> float:
    """
    Curriculum learning temperature schedule.

    Starts with higher temperature (more exploration) and decreases over time.
    """
    # Linear schedule from 2.0 to 1.0
    return 2.0 - (epoch / total_epochs)


def combined_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    teacher_targets: Optional[torch.Tensor],
    ground_truth_targets: Optional[torch.Tensor],
    # Process-step supervision targets (replaces reasoning_content)
    tool_name_ids: Optional[torch.Tensor] = None,
    tool_name_mask: Optional[torch.Tensor] = None,
    gold_json_text_ids: Optional[torch.Tensor] = None,
    mask_valid_json_tokens: Optional[torch.Tensor] = None,
    tool_result_fields: Optional[torch.Tensor] = None,
    integration_mask: Optional[torch.Tensor] = None,
    # Loss weights
    kl_weight: float = 0.4,
    ce_teacher_weight: float = 0.2,
    w_tool: float = 0.15,
    w_args: float = 0.15,
    w_integr: float = 0.10,
    ce_ground_truth_weight: float = 0.2,
    kd_temperature: float = 2.0,
    ignore_index: int = -100,
    # Code-mode preference loss (optional)
    code_mode_loss: Optional[torch.Tensor] = None,
    code_mode_weight: float = 0.0,
    # Latent curriculum loss mask (optional)
    loss_mask: Optional[torch.Tensor] = None,
    # Halt head loss (optional)
    halt_logits: Optional[torch.Tensor] = None,
    halt_targets: Optional[torch.Tensor] = None,
    halt_weight: float = 0.0,
) -> dict:
    """
    Compute combined knowledge distillation loss with process-step supervision.

    Process-step supervision replaces reasoning_content loss to avoid ToS violations
    and focus on decision quality rather than narrative prose.

    Args:
        student_logits: [B, T, V] student model logits
        teacher_logits: [B, T, V] teacher model logits (optional)
        teacher_targets: [B, T] teacher's predicted tokens (optional)
        ground_truth_targets: [B, T] ground truth token IDs (optional)
        tool_name_ids: [B, T_tool] tool name token IDs from teacher (optional)
        tool_name_mask: [B, T_tool] mask for tool name tokens (optional)
        gold_json_text_ids: [B, T_json] gold JSON token IDs from teacher (optional)
        mask_valid_json_tokens: [B, T_json] mask for valid JSON tokens (optional)
        tool_result_fields: [B, T_int] tool result field token IDs (optional)
        integration_mask: [B, T_int] mask for integration spans (optional)
        kl_weight: Weight for KL divergence loss
        ce_teacher_weight: Weight for CE on teacher predictions
        w_tool: Weight for tool name loss
        w_args: Weight for JSON argument loss
        w_integr: Weight for integration copy loss
        ce_ground_truth_weight: Weight for CE on ground truth
        kd_temperature: Temperature for KD softmax
        ignore_index: Index to ignore in CE losses
        code_mode_loss: Optional code-mode preference loss tensor
        code_mode_weight: Weight for code-mode loss
        loss_mask: Optional loss mask for latent curriculum [B, T]
        halt_logits: Optional halt head logits [B, 2]
        halt_targets: Optional halt targets [B] (0=continue, 1=halt)
        halt_weight: Weight for halt head loss

    Returns:
        Dictionary with individual losses and total loss
    """
    # Cast student logits to FP32 for loss computation (model may output FP16)
    student_logits = student_logits.float()

    losses = {}
    total_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

    # KL divergence loss (if teacher logits available)
    if teacher_logits is not None and kl_weight > 0:
        kl_loss = kl_divergence(student_logits, teacher_logits, temperature=kd_temperature)
        losses["kl_div"] = kl_loss
        total_loss = total_loss + kl_weight * kl_loss

    # Cross-entropy on teacher predictions
    if teacher_targets is not None and ce_teacher_weight > 0:
        # Apply loss mask if available (masks latent slots)
        if loss_mask is not None:
            # Expand mask to match teacher_targets shape
            if (
                loss_mask.shape[0] == teacher_targets.shape[0]
                and loss_mask.shape[1] == teacher_targets.shape[1]
            ):
                # Mask out latent slots: set to ignore_index where mask is False
                teacher_targets_masked = torch.where(
                    loss_mask, teacher_targets, torch.full_like(teacher_targets, ignore_index)
                )
            else:
                teacher_targets_masked = teacher_targets
        else:
            teacher_targets_masked = teacher_targets

        ce_teacher = cross_entropy_on_teacher(
            student_logits, teacher_targets_masked, ignore_index=ignore_index
        )
        losses["ce_teacher"] = ce_teacher
        total_loss = total_loss + ce_teacher_weight * ce_teacher

    # Process-step supervision losses (replaces reasoning_content)
    if tool_name_ids is not None and w_tool > 0:
        tool_loss = tool_name_loss(
            student_logits, tool_name_ids, tool_name_mask, ignore_index=ignore_index
        )
        losses["tool_name"] = tool_loss
        total_loss = total_loss + w_tool * tool_loss

    if gold_json_text_ids is not None and w_args > 0:
        json_loss = json_argument_loss(
            student_logits, gold_json_text_ids, mask_valid_json_tokens, ignore_index=ignore_index
        )
        losses["json_args"] = json_loss
        total_loss = total_loss + w_args * json_loss

    if tool_result_fields is not None and w_integr > 0:
        integr_loss = integration_copy_loss(
            student_logits, tool_result_fields, integration_mask, ignore_index=ignore_index
        )
        losses["integration"] = integr_loss
        total_loss = total_loss + w_integr * integr_loss

    # Cross-entropy on ground truth (standard language modeling)
    if ground_truth_targets is not None and ce_ground_truth_weight > 0:
        # Apply loss mask if available (masks latent slots)
        if loss_mask is not None:
            # Expand mask to match ground_truth_targets shape
            if (
                loss_mask.shape[0] == ground_truth_targets.shape[0]
                and loss_mask.shape[1] == ground_truth_targets.shape[1]
            ):
                # Mask out latent slots: set to ignore_index where mask is False
                ground_truth_masked = torch.where(
                    loss_mask,
                    ground_truth_targets,
                    torch.full_like(ground_truth_targets, ignore_index),
                )
            else:
                ground_truth_masked = ground_truth_targets
        else:
            ground_truth_masked = ground_truth_targets

        logits_flat = student_logits.view(-1, student_logits.size(-1))
        targets_flat = ground_truth_masked.view(-1)
        ce_gt = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
        losses["ce_ground_truth"] = ce_gt
        total_loss = total_loss + ce_ground_truth_weight * ce_gt

    # Code-mode preference loss (optional)
    if code_mode_loss is not None and code_mode_weight > 0:
        losses["code_mode_pref"] = code_mode_loss
        total_loss = total_loss + code_mode_weight * code_mode_loss

    # Halt head loss (optional)
    if halt_logits is not None and halt_targets is not None and halt_weight > 0:
        halt_loss = halt_head_loss(halt_logits, halt_targets)
        losses["halt_head"] = halt_loss
        total_loss = total_loss + halt_weight * halt_loss

    losses["total"] = total_loss

    # Assert loss finiteness (catch NaNs/Infs early)
    if not torch.isfinite(total_loss).all():
        nan_count = torch.isnan(total_loss).sum().item()
        inf_count = torch.isinf(total_loss).sum().item()
        raise RuntimeError(
            f"Total loss is not finite in combined_kd_loss: "
            f"NaN count={nan_count}, Inf count={inf_count}. "
            f"Individual losses: {[(k, v.item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()]}"
        )

    return losses


class CodeModePreferenceLoss(nn.Module):
    """
    Differentiable loss function that encourages TypeScript API orchestration
    over direct tool calls for eligible scenarios.

    Uses token-level log-probability shaping instead of decoded text to provide
    gradients. Operates on span targets from the data generator.

    Reference: code-mode-latent-reasoning.md Milestone 1

    Eligibility rules:
    - min_tools ≥ 2 OR
    - min_intermediate_chars ≥ 10k OR
    - pii_tags_present = true
    """

    def __init__(
        self,
        eligibility_rules: Dict[str, Any],
        reward: Dict[str, bool],
        vocab_ids: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize code-mode preference loss.

        Args:
            eligibility_rules: Dict with min_tools, min_intermediate_chars, pii_patterns
            reward: Dict with prefer_ts_api_over_direct_tool, penalize_tool_result_roundtrip
            vocab_ids: Dict mapping marker names to token IDs (e.g., {'import': 1234, 'from': 5678})
            weights: Dict with 'pos' and 'neg' weights for positive/negative terms
        """
        super().__init__()
        self.rules = eligibility_rules
        self.reward_cfg = reward
        self.min_tools = eligibility_rules.get("min_tools", 2)
        self.min_intermediate_chars = eligibility_rules.get("min_intermediate_chars", 10000)
        self.pii_patterns = eligibility_rules.get("pii_patterns", [])
        self.vocab_ids = vocab_ids or {}
        self.weights = weights or {"pos": 1.0, "neg": 1.0}

    def _compute_eligibility_mask(
        self, batch_meta: List[Dict[str, Any]], batch_size: int
    ) -> torch.Tensor:
        """
        Compute eligibility mask for code-mode preference.

        Args:
            batch_meta: List of batch metadata dicts
            batch_size: Size of batch

        Returns:
            Boolean tensor [batch_size] indicating eligibility
        """
        eligibility_mask = torch.zeros(batch_size, dtype=torch.bool)

        for i, meta in enumerate(batch_meta):
            tool_count = meta.get("tool_count", 0)
            intermediate_sizes = meta.get("intermediate_sizes", [])
            pii_tags_present = meta.get("pii_tags_present", False)

            eligible = (
                tool_count >= self.min_tools
                or (intermediate_sizes and max(intermediate_sizes) >= self.min_intermediate_chars)
                or pii_tags_present
            )

            eligibility_mask[i] = eligible

        return eligibility_mask

    def forward(
        self,
        student_logits: torch.Tensor,
        span_targets: Optional[Dict[str, torch.Tensor]] = None,
        batch_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """
        Compute code-mode preference loss.

        Args:
            student_logits: [B, T, V] student logits
            span_targets: Optional dict with TS API span targets
            batch_meta: Optional list of batch metadata dicts with eligibility info

        Returns:
            Loss tensor (scalar)
        """
        if batch_meta is None or len(batch_meta) == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # Check eligibility for the batch
        batch_size = len(batch_meta)
        eligibility_mask = self._compute_eligibility_mask(batch_meta, batch_size)

        # If no items in batch are eligible, return zero loss
        if not eligibility_mask.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)


def caws_compliance_loss(
    student_output: str,
    teacher_output: str,
    claim_extractor=None,
) -> torch.Tensor:
    """
    CAWS compliance loss for self-evaluation training.

    Evaluates how well the student output complies with CAWS requirements:
    - Budget constraints (tier limits for loops and latent spans)
    - Quality requirements (supported claims, proper reasoning)
    - Feature usage appropriateness

    Args:
        student_output: The model's generated text output
        teacher_output: The teacher's reference output
        claim_extractor: Optional claim extractor for claim evaluation

    Returns:
        Loss tensor where lower values indicate better CAWS compliance
    """
    loss_components = []

    # 1. Evaluate budget compliance (latent spans, loop count)
    budget_penalty = _evaluate_budget_compliance(student_output)
    loss_components.append(budget_penalty)

    # 2. Evaluate quality compliance (claim support, reasoning structure)
    quality_penalty = _evaluate_quality_compliance(student_output, teacher_output, claim_extractor)
    loss_components.append(quality_penalty)

    # 3. Evaluate feature usage compliance (appropriate use of code-mode, latent reasoning)
    feature_penalty = _evaluate_feature_usage_compliance(student_output)
    loss_components.append(feature_penalty)

    # Combine penalties (sum with equal weights for now)
    total_loss = sum(loss_components)

    # Ensure loss is non-negative and requires gradients for training
    return torch.clamp(torch.tensor(float(total_loss), requires_grad=True), min=0.0)


def caws_structure_loss(teacher_score: float, student_score: float) -> torch.Tensor:
    """
    CAWS structure loss for distillation training.

    Penalizes when student structure score is lower than teacher score.
    No penalty when student score is equal or higher (student is better).

    Args:
        teacher_score: Teacher's CAWS structure score (0.0 to 1.0)
        student_score: Student's CAWS structure score (0.0 to 1.0)

    Returns:
        Loss tensor where higher values indicate worse structure compliance
    """
    if student_score >= teacher_score:
        # Student is as good or better - no penalty
        return torch.tensor(0.0, requires_grad=True)

    # Penalize the difference (teacher - student)
    loss_value = teacher_score - student_score
    return torch.clamp(torch.tensor(float(loss_value), requires_grad=True), min=0.0)


def entropy_weighting(
    teacher_logits: torch.Tensor,
    min_entropy: float = 2.0,
    max_entropy: float = 8.0,
    min_temp: float = 1.5,
    max_temp: float = 3.0,
) -> tuple[float, Dict[str, float]]:
    """
    Compute entropy-based temperature and weighting for distillation.

    Higher entropy in teacher predictions → higher temperature, higher KL weight
    Lower entropy in teacher predictions → lower temperature, higher CE_GT weight

    Args:
        teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
        min_entropy: Minimum expected entropy
        max_entropy: Maximum expected entropy
        min_temp: Minimum temperature to apply
        max_temp: Maximum temperature to apply

    Returns:
        Tuple of (temperature, weights_dict) where weights_dict contains:
        - "entropy": Computed entropy value
        - "kl_weight": Weight for KL divergence loss
        - "ce_teacher_weight": Weight for cross-entropy on teacher predictions
        - "ce_ground_truth_weight": Weight for cross-entropy on ground truth
    """
    # Compute entropy of teacher predictions
    # entropy = -sum(p * log(p)) where p = softmax(logits)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-10), dim=-1)
    avg_entropy = torch.mean(entropy).item()

    # Map entropy to temperature
    # entropy ∈ [min_entropy, max_entropy] → temp ∈ [min_temp, max_temp]
    entropy_ratio = (avg_entropy - min_entropy) / (max_entropy - min_entropy)
    entropy_ratio = torch.clamp(torch.tensor(entropy_ratio), 0.0, 1.0).item()

    temperature = min_temp + entropy_ratio * (max_temp - min_temp)

    # Map entropy to weights (normalized to sum to 1.0)
    # High entropy → favor KL and CE_teacher (teacher alignment)
    # Low entropy → favor CE_ground_truth (ground truth alignment)
    kl_weight = entropy_ratio * 0.5  # 0 to 0.5
    ce_teacher_weight = entropy_ratio * 0.5  # 0 to 0.5
    ce_gt_weight = 1.0 - entropy_ratio  # 1.0 to 0

    weights = {
        "entropy": avg_entropy,
        "kl_weight": kl_weight,
        "ce_teacher_weight": ce_teacher_weight,
        "ce_ground_truth_weight": ce_gt_weight,
    }

    return temperature, weights


def _evaluate_budget_compliance(student_output: str) -> float:
    """
    Evaluate if output complies with CAWS budget constraints.

    Penalizes:
    - Excessive latent spans (>3 for tier 3)
    - Excessive refinement loops
    - Over-budget feature usage

    Returns:
        Penalty score (0.0 = compliant, >0.0 = violation)
    """
    penalty = 0.0

    # Count latent spans (<bot>...</bot> pairs)
    bot_count = student_output.count("<bot>")
    eot_count = student_output.count("<eot>")

    # Penalize mismatched latent spans
    if bot_count != eot_count:
        penalty += 1.0

    # Penalize excessive latent spans (beyond tier 3 limits)
    max_allowed_latent = 3  # Tier 3 max
    if bot_count > max_allowed_latent:
        penalty += (bot_count - max_allowed_latent) * 0.5

    # Penalize excessive refinement loops (rough heuristic: multiple "Step X:" patterns)
    step_patterns = len(re.findall(r"Step \d+:", student_output, re.IGNORECASE))
    max_allowed_steps = 5  # Reasonable limit
    if step_patterns > max_allowed_steps:
        penalty += (step_patterns - max_allowed_steps) * 0.2

    return penalty


def _evaluate_quality_compliance(
    student_output: str, teacher_output: str, claim_extractor=None
) -> float:
    """
    Evaluate quality compliance of student output.

    Penalizes:
    - Unsupported claims
    - Contradictory information
    - Poor reasoning structure

    Returns:
        Penalty score (0.0 = high quality, >0.0 = quality issues)
    """
    penalty = 0.0

    # Use claim extractor if available
    if claim_extractor:
        try:
            student_claims = claim_extractor.extract_claims(student_output)
            teacher_claims = claim_extractor.extract_claims(teacher_output)

            # Penalize unsupported claims
            unsupported_claims = 0
            for claim in student_claims:
                if not _claim_supported_by_teacher(claim, teacher_claims):
                    unsupported_claims += 1

            penalty += unsupported_claims * 0.3
        except Exception:
            # Fallback to heuristic evaluation if claim extraction fails
            penalty += 0.1

    # Heuristic quality checks
    # Penalize very short outputs (likely incomplete reasoning)
    if len(student_output.strip()) < 50:
        penalty += 1.0

    # Penalize outputs with contradiction markers
    contradiction_indicators = ["however", "but", "contrary", "despite", "although"]
    contradiction_count = sum(
        1 for word in contradiction_indicators if word.lower() in student_output.lower()
    )
    penalty += contradiction_count * 0.1

    # Penalize outputs that don't have clear conclusion/answer
    if not re.search(r"(answer|conclusion|result|therefore)", student_output, re.IGNORECASE):
        penalty += 0.2

    return penalty


def _evaluate_feature_usage_compliance(student_output: str) -> float:
    """
    Evaluate appropriate usage of CAWS features.

    Penalizes:
    - Code-mode when not needed (no API calls)
    - Latent reasoning when direct reasoning would suffice
    - Inappropriate feature combinations

    Returns:
        Penalty score (0.0 = appropriate usage, >0.0 = inappropriate usage)
    """
    penalty = 0.0

    # Check for code-mode indicators when no API usage
    code_indicators = ["import", "from", "callMCPTool", "await", "function", "const", "let"]
    has_code_patterns = any(indicator in student_output for indicator in code_indicators)

    # Check for actual tool usage (rough heuristic)
    tool_usage_indicators = ["google_drive", "salesforce", "api", "tool", "call"]
    has_tool_usage = any(
        indicator.lower() in student_output.lower() for indicator in tool_usage_indicators
    )

    # Penalize code-mode syntax without tool usage (inappropriate code-mode)
    if has_code_patterns and not has_tool_usage:
        penalty += 0.5

    # Penalize excessive latent spans for simple tasks
    latent_span_count = student_output.count("<bot>")
    output_length = len(student_output)

    # For short outputs, penalize heavy latent usage
    if output_length < 200 and latent_span_count > 2:
        penalty += 0.3

    # Penalize missing tool integration when tools are mentioned
    if has_tool_usage and not has_code_patterns and "<bot>" not in student_output:
        # Should probably use some form of integration (code-mode or latent)
        penalty += 0.2

    return penalty


def _claim_supported_by_teacher(student_claim: Any, teacher_claims: List[Any]) -> bool:
    """
    Check if a student claim is supported by teacher claims.

    Uses keyword-based similarity matching between student and teacher claims.
    Future enhancement could include semantic similarity and entailment checking.
    """
    if not teacher_claims:
        return False

    # Extract claim text (handle both ExtractedClaim objects and strings)
    if hasattr(student_claim, "statement"):
        student_text = student_claim.statement.lower()
    else:
        student_text = str(student_claim).lower()

    # Simple keyword matching approach
    student_words = set(re.findall(r"\b\w+\b", student_text))

    for teacher_claim in teacher_claims:
        # Extract teacher claim text
        if hasattr(teacher_claim, "statement"):
            teacher_text = teacher_claim.statement.lower()
        else:
            teacher_text = str(teacher_claim).lower()

        teacher_words = set(re.findall(r"\b\w+\b", teacher_text))

        # Check for significant overlap (Jaccard similarity > 0.3)
        intersection = student_words & teacher_words
        union = student_words | teacher_words

        if union and len(intersection) / len(union) > 0.3:
            return True

        # Check for substring containment
        if student_text in teacher_text or teacher_text in student_text:
            return True

    return False
