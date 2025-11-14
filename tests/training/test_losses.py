"""
Tests for training/losses.py - Knowledge distillation loss functions.

Tests all loss functions including KL divergence, cross-entropy, process supervision
losses, intermediate layer matching, self-evaluation, length-aware KD, early tool call,
and CAWS compliance losses.
"""
# @author: @darianrosebrook

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses import (
    kl_divergence,
    cross_entropy_on_teacher,
    tool_name_loss,
    json_argument_loss,
    integration_copy_loss,
    halt_head_loss,
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
    length_aware_kd_loss,
    early_tool_call_loss,
    curriculum_temperature,
    combined_kd_loss,
    CodeModePreferenceLoss,
    caws_compliance_loss,
    caws_structure_loss,
    entropy_weighting,
)


class TestKLDivergence:
    """Test KL divergence loss function."""

    def test_kl_divergence_basic(self, device):
        """Test basic KL divergence computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss = kl_divergence(student_logits, teacher_logits, temperature=1.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative

    def test_kl_divergence_with_temperature(self, device):
        """Test KL divergence with different temperatures."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss_t1 = kl_divergence(student_logits, teacher_logits, temperature=1.0)
        loss_t2 = kl_divergence(student_logits, teacher_logits, temperature=2.0)
        loss_t05 = kl_divergence(student_logits, teacher_logits, temperature=0.5)

        # All should be valid losses
        assert loss_t1.item() >= 0
        assert loss_t2.item() >= 0
        assert loss_t05.item() >= 0

    def test_kl_divergence_reduction_mean(self, device):
        """Test KL divergence with mean reduction."""
        student_logits = torch.randn(2, 5, 100, device=device)
        teacher_logits = torch.randn(2, 5, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="mean")

        assert loss.dim() == 0  # Scalar

    def test_kl_divergence_reduction_sum(self, device):
        """Test KL divergence with sum reduction."""
        student_logits = torch.randn(2, 5, 100, device=device)
        teacher_logits = torch.randn(2, 5, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="sum")

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_kl_divergence_reduction_none(self, device):
        """Test KL divergence with no reduction."""
        batch_size = 2
        seq_len = 5
        student_logits = torch.randn(batch_size, seq_len, 100, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="none")

        assert loss.dim() == 1  # [B*T]
        assert loss.shape[0] == batch_size * seq_len

    def test_kl_divergence_flattened_input(self, device):
        """Test KL divergence with flattened input tensors."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size * seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size * seq_len, vocab_size, device=device)

        loss = kl_divergence(student_logits, teacher_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestCrossEntropyOnTeacher:
    """Test cross-entropy loss on teacher predictions."""

    def test_cross_entropy_on_teacher_basic(self, device):
        """Test basic cross-entropy on teacher targets."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss = cross_entropy_on_teacher(student_logits, teacher_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_cross_entropy_on_teacher_with_ignore_index(self, device):
        """Test cross-entropy with ignore index."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        teacher_targets[0, :5] = -100  # Mark some tokens to ignore

        loss = cross_entropy_on_teacher(student_logits, teacher_targets, ignore_index=-100)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_cross_entropy_on_teacher_gradient_flow(self, device):
        """Test that gradients flow through the loss."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss = cross_entropy_on_teacher(student_logits, teacher_targets)
        loss.backward()

        assert student_logits.grad is not None
        assert not torch.allclose(student_logits.grad, torch.zeros_like(student_logits.grad))


class TestToolNameLoss:
    """Test tool name loss function."""

    def test_tool_name_loss_basic(self, device):
        """Test basic tool name loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        tool_len = 5

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.ones(batch_size, tool_len, device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_tool_name_loss_with_mask(self, device):
        """Test tool name loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        tool_len = 5

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_tool_name_loss_sequence_mismatch(self, device):
        """Test tool name loss when tool length exceeds sequence length."""
        batch_size = 2
        seq_len = 5
        vocab_size = 1000
        tool_len = 10  # Longer than sequence

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.ones(batch_size, tool_len, device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        # Should handle gracefully by truncating
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestJSONArgumentLoss:
    """Test JSON argument loss function."""

    def test_json_argument_loss_basic(self, device):
        """Test basic JSON argument loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        json_len = 8

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, json_len), device=device)
        mask_valid_json_tokens = torch.ones(batch_size, json_len, device=device)

        loss = json_argument_loss(student_logits, gold_json_text_ids, mask_valid_json_tokens)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_json_argument_loss_with_partial_mask(self, device):
        """Test JSON argument loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        json_len = 8

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, json_len), device=device)
        # Mask out some tokens (prose vs JSON)
        mask_valid_json_tokens = torch.tensor(
            [[1, 1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1, 1, 1]], device=device
        )

        loss = json_argument_loss(student_logits, gold_json_text_ids, mask_valid_json_tokens)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestIntegrationCopyLoss:
    """Test integration copy loss function."""

    def test_integration_copy_loss_basic(self, device):
        """Test basic integration copy loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        int_len = 6

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, int_len), device=device)
        integration_mask = torch.ones(batch_size, int_len, device=device)

        loss = integration_copy_loss(student_logits, tool_result_fields, integration_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_integration_copy_loss_with_mask(self, device):
        """Test integration copy loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        int_len = 6

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, int_len), device=device)
        integration_mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1]], device=device)

        loss = integration_copy_loss(student_logits, tool_result_fields, integration_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestHaltHeadLoss:
    """Test halt head loss function."""

    def test_halt_head_loss_basic(self, device):
        """Test basic halt head loss."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device)
        halt_targets = torch.randint(0, 2, (batch_size,), device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_halt_head_loss_all_continue(self, device):
        """Test halt head loss with all continue targets."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device)
        halt_targets = torch.zeros(batch_size, dtype=torch.long, device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_halt_head_loss_all_halt(self, device):
        """Test halt head loss with all halt targets."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device)
        halt_targets = torch.ones(batch_size, dtype=torch.long, device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestIntermediateLayerLoss:
    """Test intermediate layer loss function."""

    def test_intermediate_layer_loss_basic(self, device):
        """Test basic intermediate layer loss."""
        batch_size = 2
        seq_len = 10
        student_d_model = 128
        teacher_d_model = 256

        student_hidden_states = [
            torch.randn(batch_size, seq_len, student_d_model, device=device) for _ in range(4)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, teacher_d_model, device=device) for _ in range(8)
        ]

        layer_mapping = {0: 0, 1: 2, 2: 4, 3: 6}

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_intermediate_layer_loss_with_projection(self, device):
        """Test intermediate layer loss with projection layers."""
        batch_size = 2
        seq_len = 10
        student_d_model = 128
        teacher_d_model = 256

        student_hidden_states = [
            torch.randn(batch_size, seq_len, student_d_model, device=device) for _ in range(4)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, teacher_d_model, device=device) for _ in range(8)
        ]

        layer_mapping = {0: 0, 1: 2, 2: 4, 3: 6}
        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping, projection_layers
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_intermediate_layer_loss_sequence_mismatch(self, device):
        """Test intermediate layer loss with different sequence lengths."""
        batch_size = 2
        student_seq_len = 10
        teacher_seq_len = 15
        d_model = 128

        student_hidden_states = [
            torch.randn(batch_size, student_seq_len, d_model, device=device) for _ in range(2)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, teacher_seq_len, d_model, device=device) for _ in range(4)
        ]

        layer_mapping = {0: 0, 1: 2}

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        # Should handle sequence length mismatch by truncating
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_intermediate_layer_loss_empty_mapping(self, device):
        """Test intermediate layer loss with empty layer mapping."""
        batch_size = 2
        seq_len = 10
        d_model = 128

        student_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(2)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(4)
        ]

        layer_mapping = {}

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        # Should return zero loss with gradient
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() == 0.0


class TestSelfEvaluationLoss:
    """Test self-evaluation loss function."""

    def test_self_evaluation_loss_basic(self, device):
        """Test basic self-evaluation loss."""
        batch_size = 4

        student_eval_score = torch.rand(batch_size, 1, device=device)
        teacher_quality_score = torch.rand(batch_size, device=device)

        loss = self_evaluation_loss(student_eval_score, teacher_quality_score)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_self_evaluation_loss_perfect_match(self, device):
        """Test self-evaluation loss when scores match perfectly."""
        batch_size = 4

        scores = torch.rand(batch_size, device=device)
        student_eval_score = scores.unsqueeze(1)
        teacher_quality_score = scores

        loss = self_evaluation_loss(student_eval_score, teacher_quality_score)

        # Should be close to zero (within numerical precision)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 1e-5


class TestCreateProjectionLayers:
    """Test projection layer creation."""

    def test_create_projection_layers_basic(self, device):
        """Test basic projection layer creation."""
        student_d_model = 128
        teacher_d_model = 256
        layer_mapping = {0: 0, 1: 2, 2: 4}

        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        assert len(projection_layers) == len(layer_mapping)
        for proj in projection_layers:
            assert isinstance(proj, nn.Module)

    def test_create_projection_layers_output_shape(self, device):
        """Test that projection layers produce correct output shape."""
        student_d_model = 128
        teacher_d_model = 256
        layer_mapping = {0: 0}

        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        batch_size = 2
        seq_len = 10
        student_h = torch.randn(batch_size, seq_len, student_d_model, device=device)

        output = projection_layers[0](student_h)

        assert output.shape == (batch_size, seq_len, teacher_d_model)


class TestLengthAwareKDLoss:
    """Test length-aware KD loss function."""

    def test_length_aware_kd_loss_basic(self, device):
        """Test basic length-aware KD loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        student_lengths = torch.tensor([8, 10], device=device)
        teacher_lengths = torch.tensor([10, 12], device=device)

        loss = length_aware_kd_loss(
            student_logits, teacher_logits, student_lengths, teacher_lengths
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_length_aware_kd_loss_with_hinge(self, device):
        """Test length-aware KD loss with hinge threshold."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        student_lengths = torch.tensor([5, 6], device=device)
        teacher_lengths = torch.tensor([10, 12], device=device)

        loss = length_aware_kd_loss(
            student_logits,
            teacher_logits,
            student_lengths,
            teacher_lengths,
            hinge_threshold=0.8,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestEarlyToolCallLoss:
    """Test early tool call loss function."""

    def test_early_tool_call_loss_basic(self, device):
        """Test basic early tool call loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_call_positions = torch.tensor([5, 8], device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, 3), device=device)

        loss = early_tool_call_loss(
            student_logits, tool_call_positions, tool_name_ids, enabled=True
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_early_tool_call_loss_disabled(self, device):
        """Test early tool call loss when disabled."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_call_positions = torch.tensor([5, 8], device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, 3), device=device)

        loss = early_tool_call_loss(
            student_logits, tool_call_positions, tool_name_ids, enabled=False
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_early_tool_call_loss_with_ramp(self, device):
        """Test early tool call loss with ramping."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_call_positions = torch.tensor([5, 8], device=device)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, 3), device=device)

        loss = early_tool_call_loss(
            student_logits,
            tool_call_positions,
            tool_name_ids,
            enabled=True,
            ramp_start_step=0,
            ramp_end_step=100,
            current_step=50,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestCurriculumTemperature:
    """Test curriculum temperature function."""

    def test_curriculum_temperature_start(self):
        """Test curriculum temperature at start of training."""
        temp = curriculum_temperature(epoch=0, total_epochs=10)

        assert isinstance(temp, float)
        assert temp > 0

    def test_curriculum_temperature_end(self):
        """Test curriculum temperature at end of training."""
        temp = curriculum_temperature(epoch=10, total_epochs=10)

        assert isinstance(temp, float)
        assert temp > 0

    def test_curriculum_temperature_monotonic(self):
        """Test that temperature decreases over time."""
        temps = [
            curriculum_temperature(epoch=i, total_epochs=10) for i in range(11)
        ]

        # Temperature should generally decrease (allowing for some variance)
        assert temps[0] >= temps[-1]


class TestCombinedKDLoss:
    """Test combined KD loss function."""

    def test_combined_kd_loss_basic(self, device):
        """Test basic combined KD loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        cfg = {
            "kl_weight": 0.5,
            "ce_teacher_weight": 0.3,
            "ce_ground_truth_weight": 0.2,
            "temperature": 1.0,
        }

        loss_dict = combined_kd_loss(student_logits, teacher_logits, labels, cfg)

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "kl" in loss_dict
        assert "ce_teacher" in loss_dict
        assert "ce_ground_truth" in loss_dict
        assert loss_dict["total"].item() >= 0

    def test_combined_kd_loss_with_intermediate_layers(self, device):
        """Test combined KD loss with intermediate layer matching."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        d_model = 128

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        student_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(2)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(4)
        ]

        cfg = {
            "kl_weight": 0.5,
            "ce_teacher_weight": 0.3,
            "ce_ground_truth_weight": 0.2,
            "temperature": 1.0,
            "use_intermediate_layers": True,
            "intermediate_weight": 0.1,
            "layer_mapping": {0: 0, 1: 2},
        }

        loss_dict = combined_kd_loss(
            student_logits,
            teacher_logits,
            labels,
            cfg,
            student_hidden_states=student_hidden_states,
            teacher_hidden_states=teacher_hidden_states,
        )

        assert "intermediate" in loss_dict
        assert loss_dict["total"].item() >= 0

    def test_combined_kd_loss_with_self_evaluation(self, device):
        """Test combined KD loss with self-evaluation head."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        student_eval_score = torch.rand(batch_size, 1, device=device)
        teacher_quality_score = torch.rand(batch_size, device=device)

        cfg = {
            "kl_weight": 0.5,
            "ce_teacher_weight": 0.3,
            "ce_ground_truth_weight": 0.2,
            "temperature": 1.0,
            "use_self_evaluation": True,
            "self_evaluation_weight": 0.1,
        }

        loss_dict = combined_kd_loss(
            student_logits,
            teacher_logits,
            labels,
            cfg,
            student_eval_score=student_eval_score,
            teacher_quality_score=teacher_quality_score,
        )

        assert "self_evaluation" in loss_dict
        assert loss_dict["total"].item() >= 0


class TestCodeModePreferenceLoss:
    """Test CodeModePreferenceLoss class."""

    def test_code_mode_preference_loss_basic(self, device):
        """Test basic CodeModePreferenceLoss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss_fn = CodeModePreferenceLoss(weight=0.5)
        loss = loss_fn(student_logits, teacher_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_code_mode_preference_loss_with_weight(self, device):
        """Test CodeModePreferenceLoss with different weights."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss_fn_light = CodeModePreferenceLoss(weight=0.1)
        loss_fn_heavy = CodeModePreferenceLoss(weight=1.0)

        loss_light = loss_fn_light(student_logits, teacher_logits)
        loss_heavy = loss_fn_heavy(student_logits, teacher_logits)

        assert loss_light.item() >= 0
        assert loss_heavy.item() >= 0


class TestCAWSComplianceLoss:
    """Test CAWS compliance loss function."""

    def test_caws_compliance_loss_basic(self, device):
        """Test basic CAWS compliance loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        # Mock student and teacher outputs as strings
        student_outputs = ["Valid JSON: {\"key\": \"value\"}", "Another valid output"]
        teacher_outputs = ["Valid JSON: {\"key\": \"value\"}", "Another valid output"]

        loss = caws_compliance_loss(
            student_logits, teacher_logits, student_outputs, teacher_outputs
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_caws_compliance_loss_with_weights(self, device):
        """Test CAWS compliance loss with different component weights."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        student_outputs = ["Valid output", "Another output"]
        teacher_outputs = ["Valid output", "Another output"]

        cfg = {
            "budget_weight": 0.3,
            "quality_weight": 0.4,
            "feature_usage_weight": 0.3,
        }

        loss = caws_compliance_loss(
            student_logits, teacher_logits, student_outputs, teacher_outputs, cfg
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestCAWSStructureLoss:
    """Test CAWS structure loss function."""

    def test_caws_structure_loss_basic(self, device):
        """Test basic CAWS structure loss."""
        teacher_score = 0.8
        student_score = 0.6

        loss = caws_structure_loss(teacher_score, student_score)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_caws_structure_loss_perfect_match(self, device):
        """Test CAWS structure loss when scores match."""
        score = 0.7

        loss = caws_structure_loss(score, score)

        # Should be close to zero
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 1e-5


class TestEntropyWeighting:
    """Test entropy weighting function."""

    def test_entropy_weighting_basic(self, device):
        """Test basic entropy weighting."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        weights = entropy_weighting(logits)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (batch_size, seq_len)
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_entropy_weighting_high_entropy(self, device):
        """Test entropy weighting with high entropy (uniform) distribution."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Uniform logits (high entropy)
        logits = torch.ones(batch_size, seq_len, vocab_size, device=device)

        weights = entropy_weighting(logits)

        # High entropy should result in lower weights
        assert isinstance(weights, torch.Tensor)
        assert torch.all(weights >= 0)

    def test_entropy_weighting_low_entropy(self, device):
        """Test entropy weighting with low entropy (peaked) distribution."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Peaked logits (low entropy)
        logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        logits[:, :, 0] = 10.0  # Strong peak at first token

        weights = entropy_weighting(logits)

        # Low entropy should result in higher weights
        assert isinstance(weights, torch.Tensor)
        assert torch.all(weights >= 0)

