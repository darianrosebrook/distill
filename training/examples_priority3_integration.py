"""
Priority 3 Integration Examples

This file demonstrates how to integrate intermediate layer matching and
self-evaluation head features into training scripts.

These examples show:
1. How to enable intermediate layer matching
2. How to enable self-evaluation head
3. How to extract teacher hidden states (if using local teacher model)
4. How to compute quality scores for teacher outputs

Author: @darianrosebrook
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from models.student.architectures.gqa_transformer import StudentLM
from training.losses import (
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
)


# ============================================================================
# Example 1: Intermediate Layer Matching with Local Teacher Model
# ============================================================================

def extract_teacher_hidden_states(
    teacher_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Extract hidden states from a local teacher model.

    This assumes the teacher model has a similar structure to StudentLM
    and supports return_hidden_states=True.

    Args:
        teacher_model: Teacher model instance
        input_ids: [B, T] input token IDs
        attention_mask: Optional attention mask

    Returns:
        List of hidden states per layer
    """
    # If teacher model supports return_hidden_states
    if hasattr(teacher_model, 'forward'):
        try:
            _, teacher_hidden_states = teacher_model(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True
            )
            return teacher_hidden_states
        except TypeError:
            # Teacher model doesn't support return_hidden_states
            pass

    # Fallback: Hook-based extraction
    hidden_states = []

    def hook_fn(module, input, output):
        # Store hidden state after this layer
        if isinstance(output, tuple):
            hidden_states.append(output[0])
        else:
            hidden_states.append(output)

    # Register hooks on transformer blocks
    hooks = []
    for name, module in teacher_model.named_modules():
        if 'block' in name.lower() or 'layer' in name.lower():
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = teacher_model(input_ids, attention_mask=attention_mask)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return hidden_states


def example_intermediate_layer_matching(
    student_model: StudentLM,
    teacher_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Example: Compute intermediate layer matching loss.

    This shows how to:
    1. Extract hidden states from both student and teacher
    2. Create projection layers if dimensions differ
    3. Compute intermediate layer loss
    """
    kd_cfg = config.get("distillation", {})

    # 1. Extract student hidden states
    student_logits, student_hidden_states = student_model(
        input_ids,
        attention_mask=attention_mask,
        return_hidden_states=True
    )

    # 2. Extract teacher hidden states
    teacher_hidden_states = extract_teacher_hidden_states(
        teacher_model,
        input_ids,
        attention_mask=attention_mask
    )

    # 3. Define layer mapping
    # Example: Map student layers [0, 8, 16, 24] to teacher layers [0, 16, 32, 48]
    layer_mapping = kd_cfg.get("layer_mapping", {
        0: 0,   # Student layer 0 -> Teacher layer 0
        8: 16,  # Student layer 8 -> Teacher layer 16
        16: 32,  # Student layer 16 -> Teacher layer 32
        24: 48,  # Student layer 24 -> Teacher layer 48
    })

    # 4. Create projection layers if dimensions differ
    projection_layers = None
    if student_hidden_states and teacher_hidden_states:
        student_d_model = student_hidden_states[0].size(-1)
        teacher_d_model = teacher_hidden_states[0].size(-1)

        if student_d_model != teacher_d_model:
            projection_layers = create_projection_layers(
                student_d_model=student_d_model,
                teacher_d_model=teacher_d_model,
                layer_mapping=layer_mapping,
                device=device,
            )
            # NOTE: Add projection_layers to optimizer parameters!
            # optimizer.add_param_group({'params': list(projection_layers.values())})

    # 5. Compute intermediate layer loss
    intermediate_loss = intermediate_layer_loss(
        student_hidden_states=student_hidden_states,
        teacher_hidden_states=teacher_hidden_states,
        layer_mapping=layer_mapping,
        projection_layers=projection_layers,
    )

    return intermediate_loss


# ============================================================================
# Example 2: Self-Evaluation Head Usage
# ============================================================================

def compute_teacher_quality_score(
    teacher_output: str,
    ground_truth: Optional[str] = None,
    method: str = "heuristic",
) -> float:
    """
    Compute quality score for teacher output.

    This is a placeholder - in practice, you would use:
    - Human evaluation scores
    - Automated metrics (BLEU, ROUGE, etc.)
    - Model-based evaluation
    - Task-specific metrics

    Args:
        teacher_output: Teacher model generated text
        ground_truth: Optional ground truth text for comparison
        method: Scoring method ("heuristic", "bleu", "rouge", etc.)

    Returns:
        Quality score between 0.0 and 1.0
    """
    if method == "heuristic":
        # Simple heuristic: Check for structure and length
        score = 0.5  # Base score

        # Reward structured content
        if "```" in teacher_output or "- " in teacher_output[:100]:
            score += 0.2

        # Reward reasonable length (not too short, not too long)
        length = len(teacher_output.split())
        if 50 <= length <= 500:
            score += 0.2
        elif length < 10:
            score -= 0.3

        # Reward if ground truth provided and similar
        if ground_truth:
            # Simple word overlap
            teacher_words = set(teacher_output.lower().split())
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                overlap = len(teacher_words & gt_words) / len(gt_words)
                score += overlap * 0.1

        return max(0.0, min(1.0, score))

    elif method == "bleu":
        # BLEU score computation
        if not ground_truth:
            return 0.0  # Cannot compute BLEU without ground truth

        try:
            # Try to use nltk if available
            from nltk.translate.bleu_score import sentence_bleu
            reference = [ground_truth.split()]
            candidate = teacher_output.split()
            bleu_score = sentence_bleu(reference, candidate)
            return float(bleu_score)
        except ImportError:
            # Fallback: Simple n-gram overlap approximation
            # This is a simplified BLEU approximation without nltk
            reference_tokens = ground_truth.lower().split()
            candidate_tokens = teacher_output.lower().split()

            if not reference_tokens or not candidate_tokens:
                return 0.0

            # Unigram precision (simplified BLEU-1)
            reference_unigrams = set(reference_tokens)
            candidate_unigrams = set(candidate_tokens)

            matches = len(reference_unigrams & candidate_unigrams)
            precision = matches / \
                len(candidate_unigrams) if candidate_unigrams else 0.0

            # Brevity penalty (simplified)
            brevity_penalty = min(1.0, len(
                candidate_tokens) / len(reference_tokens)) if reference_tokens else 0.0

            # Simplified BLEU score (unigram precision with brevity penalty)
            bleu_approx = precision * brevity_penalty
            return float(bleu_approx)

    else:
        return 0.5


def example_self_evaluation_training(
    student_model: StudentLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_output: str,
    ground_truth: Optional[str] = None,
) -> torch.Tensor:
    """
    Example: Train self-evaluation head.

    This shows how to:
    1. Get evaluation score from student model
    2. Compute teacher quality score
    3. Compute self-evaluation loss
    """
    # 1. Get student evaluation score
    student_logits, eval_score = student_model(
        input_ids,
        attention_mask=attention_mask,
        return_eval_score=True
    )

    # 2. Compute teacher quality score
    teacher_quality = compute_teacher_quality_score(
        teacher_output=teacher_output,
        ground_truth=ground_truth,
        method="heuristic"
    )

    # 3. Compute self-evaluation loss
    eval_loss = self_evaluation_loss(
        student_eval_score=eval_score,
        teacher_quality_score=teacher_quality,
    )

    return eval_loss


# ============================================================================
# Example 3: Combined Usage in Training Loop
# ============================================================================

def example_combined_training_step(
    student_model: StudentLM,
    teacher_model: Optional[nn.Module],
    batch: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Example: Combined training step with both intermediate layers and self-evaluation.

    This shows how to integrate both features into a single training step.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    kd_cfg = config.get("distillation", {})

    losses = {}

    # Forward pass with both features enabled
    use_intermediate = kd_cfg.get("use_intermediate_layers", False)
    use_self_eval = kd_cfg.get("use_self_evaluation", False)

    # Get student outputs
    if use_intermediate and use_self_eval:
        # Need two forward passes (model doesn't support both simultaneously)
        student_logits, student_hidden_states = student_model(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True
        )
        _, eval_score = student_model(
            input_ids,
            attention_mask=attention_mask,
            return_eval_score=True
        )
    elif use_intermediate:
        student_logits, student_hidden_states = student_model(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True
        )
        eval_score = None
    elif use_self_eval:
        student_logits, eval_score = student_model(
            input_ids,
            attention_mask=attention_mask,
            return_eval_score=True
        )
        student_hidden_states = None
    else:
        student_model(
            input_ids, attention_mask=attention_mask)
        student_hidden_states = None
        eval_score = None

    # Compute intermediate layer loss
    if use_intermediate and student_hidden_states is not None and teacher_model is not None:
        teacher_hidden_states = extract_teacher_hidden_states(
            teacher_model,
            input_ids,
            attention_mask=attention_mask
        )

        layer_mapping = kd_cfg.get("layer_mapping", {})
        projection_layers = None

        if layer_mapping:
            student_d_model = student_hidden_states[0].size(-1)
            teacher_d_model = teacher_hidden_states[0].size(-1)

            if student_d_model != teacher_d_model:
                projection_layers = create_projection_layers(
                    student_d_model=student_d_model,
                    teacher_d_model=teacher_d_model,
                    layer_mapping=layer_mapping,
                    device=device,
                )

        intermediate_loss = intermediate_layer_loss(
            student_hidden_states=student_hidden_states,
            teacher_hidden_states=teacher_hidden_states,
            layer_mapping=layer_mapping,
            projection_layers=projection_layers,
        )

        losses["intermediate_layer"] = intermediate_loss

    # Compute self-evaluation loss
    if use_self_eval and eval_score is not None:
        teacher_output = batch.get("teacher_text", "")
        ground_truth = batch.get("ground_truth_text")

        teacher_quality = compute_teacher_quality_score(
            teacher_output=teacher_output,
            ground_truth=ground_truth,
        )

        eval_loss = self_evaluation_loss(
            student_eval_score=eval_score,
            teacher_quality_score=teacher_quality,
        )

        losses["self_evaluation"] = eval_loss

    return losses


# ============================================================================
# Example 4: Configuration Example
# ============================================================================

EXAMPLE_CONFIG = {
    "distillation": {
        # Enable intermediate layer matching
        "use_intermediate_layers": True,
        "intermediate_layer_weight": 0.1,
        "layer_mapping": {
            0: 0,   # Student layer 0 -> Teacher layer 0
            8: 16,  # Student layer 8 -> Teacher layer 16
            16: 32,  # Student layer 16 -> Teacher layer 32
            24: 48,  # Student layer 24 -> Teacher layer 48
        },

        # Enable self-evaluation head
        "use_self_evaluation": True,
        "self_evaluation_weight": 0.05,

        # Other KD settings
        "kl_weight": 0.4,
        "ce_teacher_weight": 0.2,
        "ce_ground_truth_weight": 0.2,
        "kd_temperature": 2.0,
    },
    "arch": {
        "d_model": 3584,
        "n_layers": 32,
        # ... other architecture settings
    },
}
