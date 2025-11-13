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
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple


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
    kl = F.kl_div(student_probs, teacher_probs,
                  reduction='none', log_target=False)
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


def tool_name_loss(
    student_logits: torch.Tensor,
    tool_name_ids: torch.Tensor,
    tool_name_mask: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Cross-entropy loss over tool name token span.

    Supervises tool selection decision without training on reasoning prose.
    Locates tool name tokens in the sequence and computes loss at those positions.

    Args:
        student_logits: [B, T, V] student logits
        tool_name_ids: [B, T_tool] tool name token IDs from teacher
        tool_name_mask: [B, T_tool] mask indicating valid tool name tokens
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss on tool name tokens
    """
    device = student_logits.device
    batch_size = student_logits.size(0)
    seq_len = student_logits.size(1)

    # Handle different tensor shapes
    if tool_name_ids.dim() == 1:
        tool_name_ids = tool_name_ids.unsqueeze(0)
    if tool_name_mask.dim() == 1:
        tool_name_mask = tool_name_mask.unsqueeze(0)

    losses = []

    for i in range(min(batch_size, tool_name_ids.size(0))):
        sample_tool_ids = tool_name_ids[i]
        sample_mask = tool_name_mask[i]

        # Extract valid tool name tokens
        if sample_mask.dtype == torch.bool:
            valid_tool_tokens = sample_tool_ids[sample_mask]
        else:
            valid_tool_tokens = sample_tool_ids[sample_mask.bool()]

        if valid_tool_tokens.numel() == 0:
            continue

        # Convert to list for sequence matching
        target_tokens = valid_tool_tokens.cpu().tolist()

        # Get predicted token IDs from logits to locate tool name position
        pred_token_ids = student_logits[i].argmax(dim=-1).cpu().tolist()

        # Find position where tool name tokens appear in predicted sequence
        tool_name_pos = None
        for j in range(seq_len - len(target_tokens) + 1):
            # Check if target tokens match at this position
            if pred_token_ids[j:j+len(target_tokens)] == target_tokens:
                tool_name_pos = j
                break

        # If tool name found, compute loss at those positions
        if tool_name_pos is not None:
            # Compute loss for each token in tool name
            for k, target_token_id in enumerate(target_tokens):
                pos = tool_name_pos + k
                if pos < seq_len:
                    logits_slice = student_logits[i,
                                                  pos, :].unsqueeze(0)  # [1, V]
                    targets = torch.tensor(
                        [target_token_id], device=device, dtype=torch.long)

                    if not logits_slice.is_contiguous():
                        logits_slice = logits_slice.contiguous()

                    ce_loss = F.cross_entropy(
                        logits_slice,
                        targets,
                        ignore_index=ignore_index
                    )
                    losses.append(ce_loss)
            continue

        # Fallback: if tool name not found, search in latter half of sequence
        # Tool calls typically appear after the prompt
        if seq_len > 10:
            start_pos = seq_len // 2
            best_loss = None

            for pos in range(start_pos, min(start_pos + 20, seq_len)):
                # Compute loss for first token of tool name
                target_token_id = target_tokens[0]
                logits_slice = student_logits[i, pos, :].unsqueeze(0)  # [1, V]
                targets = torch.tensor(
                    [target_token_id], device=device, dtype=torch.long)

                if not logits_slice.is_contiguous():
                    logits_slice = logits_slice.contiguous()

                ce_loss = F.cross_entropy(
                    logits_slice,
                    targets,
                    ignore_index=ignore_index,
                    reduction='none'
                )

                if best_loss is None or ce_loss.item() < best_loss.item():
                    best_loss = ce_loss

            if best_loss is not None:
                losses.append(best_loss)
        else:
            # For short sequences, use middle position
            pos = seq_len // 2
            target_token_id = target_tokens[0]
            logits_slice = student_logits[i, pos, :].unsqueeze(0)  # [1, V]
            targets = torch.tensor(
                [target_token_id], device=device, dtype=torch.long)

            if not logits_slice.is_contiguous():
                logits_slice = logits_slice.contiguous()

            ce_loss = F.cross_entropy(
                logits_slice,
                targets,
                ignore_index=ignore_index
            )
            losses.append(ce_loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def json_argument_loss(
    student_logits: torch.Tensor,
    gold_json_text_ids: torch.Tensor,
    mask_valid_json_tokens: torch.Tensor,
    ignore_index: int = -100
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
        json_mask.bool(),
        json_targets,
        torch.full_like(json_targets, ignore_index)
    )

    # Flatten
    logits_flat = json_logits.view(-1, json_logits.size(-1))
    targets_flat = json_targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def integration_copy_loss(
    student_logits: torch.Tensor,
    tool_result_fields: torch.Tensor,
    integration_mask: torch.Tensor,
    ignore_index: int = -100
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
        int_mask.bool(),
        int_targets,
        torch.full_like(int_targets, ignore_index)
    )

    # Flatten
    logits_flat = int_logits.view(-1, int_logits.size(-1))
    targets_flat = int_targets.view(-1)

    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def _has_structured_content(text: str) -> bool:
    """
    Check if text has structured content (code blocks, lists, etc.).

    Used for CAWS compliance heuristic when claim extractor is not available.

    Args:
        text: Text to check

    Returns:
        True if text appears to have structured content
    """
    # Check first 500 characters for structure indicators
    preview = text[:500] if len(text) > 500 else text

    return (
        "```" in preview or  # Code blocks
        "- " in preview[:100] or  # Bullet list
        "1. " in preview[:100] or  # Numbered list
        "{" in preview and "}" in preview or  # JSON-like
        "[" in preview and "]" in preview or  # Array-like
        "##" in preview or  # Markdown headers
        "###" in preview
    )


@dataclass
class LengthAwareKDDiagnostics:
    """Diagnostics for length-aware KD loss."""
    median_len_teacher: float
    median_len_student: float
    median_rel_excess: float
    frac_penalized: float


@dataclass
class EarlyToolDiag:
    """Diagnostics for early tool call loss."""
    frac_should_use: float
    frac_target_available: float
    mean_json_prior_nll0: float


def length_aware_kd_loss(
    student_attn_mask: torch.Tensor,         # [B, T] {0,1}
    teacher_attn_mask: torch.Tensor,         # [B, T] {0,1}
    # [B] bool or {0,1}; True => student covered all required args/evidence
    required_fields_present: torch.Tensor,
    *,
    # penalty starts when student > (1 + hinge)*teacher
    hinge: float = 0.15,
    slope: float = 1.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Penalize *extra* student length only when required fields are NOT fully present.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 1

    - Relative excess = max(0, (Ls-Lt)/max(Lt,1))
    - Hinge: only penalize the portion above `hinge`
    - Mask out examples where `required_fields_present` is True (completeness exemption)

    Args:
        student_attn_mask: [B, T] attention mask for student sequences ({0,1})
        teacher_attn_mask: [B, T] attention mask for teacher sequences ({0,1})
        required_fields_present: [B] boolean tensor; True if student has all required fields
        hinge: Relative excess threshold before penalty starts (default: 0.15)
        slope: Multiplier for penalty (default: 1.0)
        reduction: "mean", "sum", or "none"

    Returns:
        Tuple of (loss, diagnostics_dict)
    """
    assert student_attn_mask.ndim == 2 and teacher_attn_mask.ndim == 2
    device = student_attn_mask.device
    B = student_attn_mask.size(0)

    Ls = student_attn_mask.sum(dim=1).to(torch.float32)  # [B]
    Lt = torch.clamp(teacher_attn_mask.sum(
        dim=1).to(torch.float32), min=1.0)  # [B]

    rel_excess = torch.clamp((Ls - Lt) / Lt, min=0.0)  # [B], >=0
    over = torch.clamp(rel_excess - hinge, min=0.0)    # hinge
    penalize = (~required_fields_present.bool()).to(
        over.dtype)  # 1 if missing fields

    per_example = slope * over * penalize  # [B]

    if reduction == "mean":
        loss = per_example.mean()
    elif reduction == "sum":
        loss = per_example.sum()
    else:
        loss = per_example  # no reduction

    with torch.no_grad():
        diags = LengthAwareKDDiagnostics(
            median_len_teacher=float(torch.median(Lt).item()),
            median_len_student=float(torch.median(Ls).item()),
            median_rel_excess=float(torch.median(rel_excess).item()),
            frac_penalized=float((penalize > 0).float().mean().item()),
        )
    return loss, {
        "len_kd.median_len_teacher": diags.median_len_teacher,
        "len_kd.median_len_student": diags.median_len_student,
        "len_kd.median_rel_excess": diags.median_rel_excess,
        "len_kd.frac_penalized": diags.frac_penalized,
    }


def early_tool_call_loss(
    # [B, T, V] student logits for current step window
    logits: torch.Tensor,
    # [B, T] ids aligned with logits for teacher targets (if available)
    input_ids: torch.Tensor,
    tool_should_be_used: torch.Tensor,         # [B] bool
    *,
    # HuggingFace-like tokenizer with .convert_ids_to_tokens
    tokenizer,
    # [B, N] or None; -100 ignored
    teacher_prefix_ids: Optional[torch.Tensor] = None,
    N: int = 25,
    # small prior to bias into JSON envelope when no teacher prefix
    json_prior_weight: float = 0.02,
    ce_weight: float = 0.2,              # CE weight when teacher prefix is available
    ramp_t: float = 1.0,                 # 0..1 ramp multiplier applied to both terms
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Encourage a valid tool JSON within the first N tokens *only when* the teacher indicates a tool is warranted.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 1

    Modes:
    - If `teacher_prefix_ids` provided: cross-entropy on first N tokens (ignore_index=-100).
    - Otherwise: apply a light JSON-envelope prior at position 0 to bias toward '{'/'['/'"' starts.

    Safety:
    - Loss masked out when tool_should_be_used == False (prevents hallucinated tools).
    - Keep magnitudes small; the main task loss dominates.

    Args:
        logits: [B, T, V] student logits for current step window
        input_ids: [B, T] token IDs aligned with logits (for teacher targets if available)
        tool_should_be_used: [B] boolean tensor; True when teacher indicates tool should be used
        tokenizer: HuggingFace-like tokenizer with .convert_ids_to_tokens method
        teacher_prefix_ids: Optional [B, N] teacher prefix token IDs (-100 ignored)
        N: Number of tokens to consider for early tool call (default: 25)
        json_prior_weight: Weight for JSON-envelope prior when no teacher prefix (default: 0.02)
        ce_weight: Weight for cross-entropy when teacher prefix available (default: 0.2)
        ramp_t: Ramp multiplier (0..1) applied to both terms (default: 1.0)

    Returns:
        Tuple of (loss, diagnostics_dict)
    """
    assert logits.ndim == 3 and input_ids.ndim == 2
    B, T, V = logits.shape
    N_eff = min(N, T)

    should = tool_should_be_used.bool()  # [B]
    any_should = should.any()

    loss = logits.new_zeros(())
    total_weight = 0.0

    # 1) Teacher-guided CE on first N tokens (if available)
    ce_term = logits.new_zeros(())
    ce_applied = False
    if teacher_prefix_ids is not None:
        # Align shapes
        assert teacher_prefix_ids.ndim == 2 and teacher_prefix_ids.size(
            1) >= N_eff
        targets = teacher_prefix_ids[:, :N_eff].to(logits.device)  # [B, N_eff]
        # Mask examples where tool isn't needed
        mask_b = should.unsqueeze(1).expand(-1, N_eff)  # [B, N_eff]
        targets = torch.where(mask_b, targets, torch.full_like(
            targets, -100))  # use ignore_index=-100

        ce = F.cross_entropy(
            logits[:, :N_eff, :].reshape(-1, V),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        ce = ce.reshape(B, N_eff)
        # Only apply to examples where tool should be used
        ce_masked = ce * mask_b.float()
        ce_term = ce_masked.sum() / (mask_b.sum().float() + 1e-8)
        ce_applied = True

    # 2) JSON-envelope prior (when no teacher prefix or as fallback)
    json_prior_term = logits.new_zeros(())
    json_prior_applied = False
    if not ce_applied or json_prior_weight > 0:
        # Get token IDs for JSON start tokens: '{', '[', '"'
        json_start_tokens = []
        for tok_str in ["{", "[", '"']:
            try:
                # Try convert_tokens_to_ids first (standard HuggingFace method)
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    tok_id = tokenizer.convert_tokens_to_ids(tok_str)
                else:
                    # Fallback: use encode and take first token
                    encoded = tokenizer.encode(
                        tok_str, add_special_tokens=False)
                    tok_id = encoded[0] if encoded else None

                if tok_id is not None and tok_id < V:
                    json_start_tokens.append(tok_id)
            except (KeyError, AttributeError, IndexError):
                pass

        if json_start_tokens and any_should:
            # Apply prior at position 0 for examples where tool should be used
            pos0_logits = logits[:, 0, :]  # [B, V]
            pos0_logprobs = F.log_softmax(pos0_logits, dim=-1)  # [B, V]

            # Sum logprobs for JSON start tokens
            json_logprob = pos0_logprobs[:, json_start_tokens].logsumexp(
                dim=-1)  # [B]
            # Negative logprob (higher = better, so negate for loss)
            json_prior_term = -json_logprob * should.float()
            json_prior_term = json_prior_term.sum() / (should.sum().float() + 1e-8)
            json_prior_applied = True

    # Combine terms with ramp
    if ce_applied:
        loss = loss + ramp_t * ce_weight * ce_term
        total_weight += ce_weight
    if json_prior_applied:
        loss = loss + ramp_t * json_prior_weight * json_prior_term
        total_weight += json_prior_weight

    # Normalize by total weight if both terms applied
    if total_weight > 0 and (ce_applied and json_prior_applied):
        loss = loss / total_weight

    # Diagnostics
    with torch.no_grad():
        diags = EarlyToolDiag(
            frac_should_use=float(should.float().mean().item()),
            frac_target_available=float(ce_applied),
            mean_json_prior_nll0=float(
                json_prior_term.item()) if json_prior_applied else 0.0,
        )

    return loss, {
        "early_tool.frac_should_use": diags.frac_should_use,
        "early_tool.frac_target_available": diags.frac_target_available,
        "early_tool.mean_json_prior_nll0": diags.mean_json_prior_nll0,
    }


def create_projection_layers(
    student_d_model: int,
    teacher_d_model: int,
    layer_mapping: Dict[int, int],
    device: Optional[torch.device] = None,
) -> Dict[int, nn.Module]:
    """
    Create projection layers for intermediate layer matching.

    Args:
        student_d_model: Hidden dimension of student model
        teacher_d_model: Hidden dimension of teacher model
        layer_mapping: Dictionary mapping student layer index to teacher layer index
        device: Optional device to place layers on

    Returns:
        Dictionary mapping student layer index to projection layer
    """
    projection_layers = {}
    for student_layer_idx in layer_mapping.keys():
        projection = nn.Linear(student_d_model, teacher_d_model, bias=False)
        if device:
            projection = projection.to(device)
        projection_layers[student_layer_idx] = projection
    return projection_layers


def intermediate_layer_loss(
    student_hidden_states: List[torch.Tensor],
    teacher_hidden_states: List[torch.Tensor],
    layer_mapping: Dict[int, int],  # student_layer -> teacher_layer
    projection_layers: Optional[Dict[int, nn.Module]] = None,
) -> torch.Tensor:
    """
    Match intermediate hidden states between teacher and student.

    Reference: DISTILLATION_BEST_PRACTICES_2025.md lines 90-114

    Args:
        student_hidden_states: List of [B, T, D] hidden states from student model
        teacher_hidden_states: List of [B, T, D] hidden states from teacher model
        layer_mapping: Dictionary mapping student layer index to teacher layer index
        projection_layers: Optional dictionary of projection layers for dimension matching

    Returns:
        Average MSE loss across matched layers
    """
    losses = []

    for student_layer_idx, teacher_layer_idx in layer_mapping.items():
        if student_layer_idx >= len(student_hidden_states):
            continue
        if teacher_layer_idx >= len(teacher_hidden_states):
            continue

        student_h = student_hidden_states[student_layer_idx]
        teacher_h = teacher_hidden_states[teacher_layer_idx]

        # Project to same dimension if needed
        if student_h.size(-1) != teacher_h.size(-1):
            if projection_layers and student_layer_idx in projection_layers:
                # Use provided projection layer
                student_h = projection_layers[student_layer_idx](student_h)
            else:
                # Simple linear projection (fallback - should be provided)
                # Note: This creates a new layer each time, which is inefficient
                # Better to provide projection_layers from model initialization
                projection = nn.Linear(
                    student_h.size(-1), teacher_h.size(-1)).to(student_h.device)
                student_h = projection(student_h)

        # Ensure same sequence length (take minimum)
        min_seq_len = min(student_h.size(1), teacher_h.size(1))
        student_h = student_h[:, :min_seq_len, :]
        teacher_h = teacher_h[:, :min_seq_len, :]

        # MSE loss on hidden states
        loss = F.mse_loss(student_h, teacher_h)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=student_hidden_states[0].device)

    return sum(losses) / len(losses)


def self_evaluation_loss(
    student_eval_score: torch.Tensor,
    teacher_quality_score: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Train self-evaluation head to predict output quality.

    Reference: DISTILLATION_BEST_PRACTICES_2025.md lines 158-160

    Args:
        student_eval_score: [B, 1] or [B] self-evaluation scores from student
        teacher_quality_score: Scalar or tensor target quality score (0-1)

    Returns:
        MSE loss between student prediction and teacher quality score
    """
    if isinstance(teacher_quality_score, (int, float)):
        target = torch.tensor(
            float(teacher_quality_score),
            device=student_eval_score.device,
            dtype=student_eval_score.dtype
        )
    else:
        target = teacher_quality_score.to(student_eval_score.device)

    # Ensure same shape
    student_score = student_eval_score.squeeze()
    if target.dim() == 0:
        # Expand to batch size if needed
        target = target.expand(student_score.size(0))

    return F.mse_loss(student_score, target)


def claim_extraction_loss(
    student_output: str,
    teacher_output: str,
    claim_extractor: Optional[Any] = None,
    min_claim_ratio: float = 0.5,
    min_success_rate_ratio: float = 0.7,
) -> torch.Tensor:
    """
    Loss function penalizing outputs that fail claim extraction.

    Reference: CLAIM_EXTRACTION_SKEPTICISM_GUARD_RAILS.md lines 1231-1309

    Args:
        student_output: Student model output text
        teacher_output: Teacher model output text
        claim_extractor: Optional claim extractor instance
        min_claim_ratio: Minimum ratio of student claims to teacher claims (default 0.5)
        min_success_rate_ratio: Minimum ratio of student success rate to teacher (default 0.7)

    Returns:
        Loss tensor (higher = worse claim extraction performance)
    """
    if claim_extractor is None:
        from training.claim_extraction import SimpleClaimExtractor, compute_claim_extraction_metrics
        claim_extractor = SimpleClaimExtractor()
        metrics = compute_claim_extraction_metrics(
            student_output, teacher_output, claim_extractor)
    else:
        from training.claim_extraction import compute_claim_extraction_metrics
        metrics = compute_claim_extraction_metrics(
            student_output, teacher_output, claim_extractor)

    # Penalize if student has too few claims relative to teacher
    claim_ratio = metrics.get("claim_ratio", 0.0)
    if claim_ratio < min_claim_ratio:
        claim_penalty = 1.0 - (claim_ratio / min_claim_ratio)
    else:
        claim_penalty = 0.0

    # Penalize if student success rate is too low relative to teacher
    success_rate_ratio = metrics.get("success_rate_ratio", 0.0)
    if success_rate_ratio < min_success_rate_ratio:
        success_penalty = 1.0 - (success_rate_ratio / min_success_rate_ratio)
    else:
        success_penalty = 0.0

    # Combined loss (weighted average)
    total_loss = 0.6 * claim_penalty + 0.4 * success_penalty

    return torch.tensor(total_loss, dtype=torch.float32)


def caws_compliance_loss(
    student_output: str,
    teacher_output: str,
    claim_extractor: Optional[Any] = None,
) -> torch.Tensor:
    """
    Distill CAWS compliance patterns from teacher.

    Penalizes outputs that fail claim extraction.
    Rewards outputs with high claim extraction success rate.

    Reference: DISTILLATION_FOR_ARBITER_STACK.md lines 102-128
    CLAIM_EXTRACTION_SKEPTICISM_GUARD_RAILS.md lines 1116-1346

    Args:
        student_output: Student model generated text
        teacher_output: Teacher model generated text
        claim_extractor: Optional claim extractor instance (if available)

    Returns:
        Loss tensor (0.0 = compliant, >0.0 = non-compliant)
    """
    if claim_extractor:
        # Use claim extractor if available
        try:
            teacher_claims = claim_extractor.extract(teacher_output)
            student_claims = claim_extractor.extract(student_output)

            if len(teacher_claims) > 0 and len(student_claims) == 0:
                # High penalty: Teacher has claims but student doesn't
                return torch.tensor(1.0)

            if len(teacher_claims) == 0:
                # No claims to verify, use structure check
                teacher_structured = _has_structured_content(teacher_output)
                student_structured = _has_structured_content(student_output)

                if teacher_structured and not student_structured:
                    return torch.tensor(0.5)

                return torch.tensor(0.0)

            # Compute claim similarity (simple heuristic)
            # More sophisticated: Use claim verification pipeline
            teacher_claim_count = len(teacher_claims)
            student_claim_count = len(student_claims)

            # Penalize if student has significantly fewer claims
            if student_claim_count < teacher_claim_count * 0.5:
                return torch.tensor(0.8)

            # Reward similarity (inverse loss)
            similarity = min(student_claim_count / teacher_claim_count, 1.0)
            return torch.tensor(1.0 - similarity)

        except Exception:
            # Fallback to structure check if extractor fails
            pass

    # Fallback: Simple structure check
    teacher_structured = _has_structured_content(teacher_output)
    student_structured = _has_structured_content(student_output)

    if teacher_structured and not student_structured:
        # Medium penalty: Teacher has structure but student doesn't
        return torch.tensor(0.5)

    # No penalty if both have structure or both lack structure
    return torch.tensor(0.0)


def adaptive_temperature(
    step: int,
    total_steps: int,
    base_temp: float = 2.0,
    min_temp: float = 1.5,
    max_temp: float = 3.0,
) -> float:
    """
    Adaptive temperature based on training progress.

    Early: Higher temp for exploration
    Late: Lower temp for fine-tuning

    Args:
        step: Current training step
        total_steps: Total training steps
        base_temp: Base temperature (default: 2.0)
        min_temp: Minimum temperature (default: 1.5)
        max_temp: Maximum temperature (default: 3.0)

    Returns:
        Current temperature value
    """
    progress = step / total_steps if total_steps > 0 else 0.0

    # Linear decay from max_temp to min_temp
    temperature = max_temp * (1 - progress) + min_temp * progress

    return temperature


def curriculum_temperature(epoch: int, total_epochs: int) -> float:
    """
    Curriculum learning temperature schedule.

    Early epochs: High temp for exploration
    Mid epochs: Medium temp
    Late epochs: Low temp for precision
    """
    progress = epoch / total_epochs if total_epochs > 0 else 0.0

    if progress < 0.3:  # First 30%: Exploration
        return 3.0
    elif progress < 0.7:  # Middle 40%: Balanced
        return 2.0
    else:  # Last 30%: Precision
        return 1.5


def loss_weight_schedule(
    step: int,
    total_steps: int,
    early_teacher_weight: float = 0.7,
    late_teacher_weight: float = 0.3,
    early_gt_weight: float = 0.3,
    late_gt_weight: float = 0.7,
) -> Dict[str, float]:
    """
    Schedule loss weights during training.

    Early: Higher weight on teacher (learn distribution)
    Late: Higher weight on ground truth (fine-tune correctness)

    Returns:
        Dictionary with kl_weight, ce_teacher_weight, ce_ground_truth_weight
    """
    progress = step / total_steps if total_steps > 0 else 0.0

    # Linear interpolation
    teacher_weight = early_teacher_weight * \
        (1 - progress) + late_teacher_weight * progress
    gt_weight = early_gt_weight * (1 - progress) + late_gt_weight * progress

    # Normalize to sum to 1.0
    total = teacher_weight + gt_weight
    if total > 0:
        teacher_weight /= total
        gt_weight /= total

    return {
        "kl_weight": teacher_weight * 0.6,  # 60% of teacher weight to KL
        "ce_teacher_weight": teacher_weight * 0.4,  # 40% of teacher weight to CE
        "ce_ground_truth_weight": gt_weight,
    }


def entropy_weighting(
    teacher_logits: torch.Tensor,
    min_entropy: float = 2.0,
    max_entropy: float = 8.0,
    min_temp: float = 1.5,
    max_temp: float = 3.0,
    base_kl_weight: float = 0.4,
    base_ce_teacher_weight: float = 0.2,
    base_ce_gt_weight: float = 0.2,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute entropy-based temperature and weights from teacher logits.

    High entropy → high temperature (explore), high KL weight (learn distribution)
    Low entropy → low temperature (exploit), high CE_GT weight (learn correctness)

    Args:
        teacher_logits: [B, T, V] teacher model logits
        min_entropy: Minimum entropy threshold (default: 2.0)
        max_entropy: Maximum entropy threshold (default: 8.0)
        min_temp: Minimum temperature (default: 1.5)
        max_temp: Maximum temperature (default: 3.0)
        base_kl_weight: Base KL weight (default: 0.4)
        base_ce_teacher_weight: Base CE teacher weight (default: 0.2)
        base_ce_gt_weight: Base CE ground truth weight (default: 0.2)

    Returns:
        (temperature, weights_dict) where weights_dict contains:
        - kl_weight: Weight for KL divergence loss
        - ce_teacher_weight: Weight for CE on teacher predictions
        - ce_ground_truth_weight: Weight for CE on ground truth
    """
    # Compute entropy: H = -sum(p * log(p)) over vocabulary
    # teacher_logits: [B, T, V]
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)  # [B, T, V]
    probs = teacher_logprobs.exp()  # [B, T, V]

    # Entropy per position: H = -sum(p * log(p))
    entropy_per_pos = -(probs * teacher_logprobs).sum(dim=-1)  # [B, T]

    # Mean entropy over batch and sequence
    entropy = entropy_per_pos.mean().item()

    # Normalize entropy to [0, 1] range
    # Clamp entropy to [min_entropy, max_entropy] range
    entropy_clamped = max(min_entropy, min(max_entropy, entropy))
    normalized = (entropy_clamped - min_entropy) / (max_entropy - min_entropy)

    # Map normalized entropy to temperature
    # High entropy (high uncertainty) → high temp (explore)
    # Low entropy (low uncertainty) → low temp (exploit)
    temperature = min_temp + normalized * (max_temp - min_temp)

    # Map normalized entropy to weights
    # High entropy → high KL weight (learn distribution)
    # Low entropy → high CE_GT weight (learn correctness)
    # Inverse relationship: high entropy → high KL, low CE_GT
    # [base_kl_weight, base_kl_weight + 0.2]
    kl_weight = base_kl_weight + normalized * 0.2
    # [base_ce_gt_weight + 0.2, base_ce_gt_weight]
    ce_gt_weight = base_ce_gt_weight + (1 - normalized) * 0.2
    ce_teacher_weight = base_ce_teacher_weight  # Keep constant

    # Normalize weights to maintain relative proportions
    total_weight = kl_weight + ce_teacher_weight + ce_gt_weight
    if total_weight > 0:
        kl_weight /= total_weight
        ce_teacher_weight /= total_weight
        ce_gt_weight /= total_weight

    return temperature, {
        "kl_weight": kl_weight,
        "ce_teacher_weight": ce_teacher_weight,
        "ce_ground_truth_weight": ce_gt_weight,
        "entropy": entropy,  # Include for logging
    }


def json_repair_loss(required_repair: bool) -> torch.Tensor:
    """
    Binary loss penalizing sequences that required JSON repair.

    Args:
        required_repair: True if JSON repair was needed, False otherwise

    Returns:
        Loss tensor: 1.0 if repair needed, 0.0 otherwise
    """
    return torch.tensor(1.0 if required_repair else 0.0, dtype=torch.float32, requires_grad=True)


def caws_structure_loss(teacher_score: float, student_score: float) -> torch.Tensor:
    """
    Loss based on CAWS structure score difference.

    Only penalizes if student score is lower than teacher score.
    Encourages student to match teacher's structure quality.

    Args:
        teacher_score: CAWS structure score from teacher output (0.0-1.0)
        student_score: CAWS structure score from student output (0.0-1.0)

    Returns:
        Loss tensor: max(0.0, teacher_score - student_score)
    """
    diff = teacher_score - student_score
    # Only penalize if student < teacher
    loss_value = max(0.0, diff)
    return torch.tensor(loss_value, dtype=torch.float32, requires_grad=True)


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

    Returns:
        Dictionary with individual losses and total loss
    """
    losses = {}
    total_loss = torch.tensor(
        0.0, device=student_logits.device, dtype=student_logits.dtype)

    # KL divergence loss (if teacher logits available)
    if teacher_logits is not None and kl_weight > 0:
        kl_loss = kl_divergence(
            student_logits, teacher_logits, temperature=kd_temperature)
        losses["kl_div"] = kl_loss
        total_loss = total_loss + kl_weight * kl_loss

    # Cross-entropy on teacher predictions
    if teacher_targets is not None and ce_teacher_weight > 0:
        ce_teacher = cross_entropy_on_teacher(
            student_logits, teacher_targets, ignore_index=ignore_index)
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
        logits_flat = student_logits.view(-1, student_logits.size(-1))
        targets_flat = ground_truth_targets.view(-1)
        ce_gt = F.cross_entropy(logits_flat, targets_flat,
                                ignore_index=ignore_index)
        losses["ce_ground_truth"] = ce_gt
        total_loss = total_loss + ce_ground_truth_weight * ce_gt

    losses["total"] = total_loss
    return losses
