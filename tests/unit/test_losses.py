"""
Unit tests for training loss functions.
"""

import torch
import torch.nn.functional as F

from training.losses import (
    kl_divergence,
    combined_kd_loss,
    cross_entropy_on_teacher,
    tool_name_loss,
    json_argument_loss,
    integration_copy_loss,
    halt_head_loss,
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
    curriculum_temperature,
)


class TestKLDivergence:
    """Tests for KL divergence loss."""

    def test_kl_divergence_basic(self):
        """Test basic KL divergence computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        loss = kl_divergence(student_logits, teacher_logits, temperature=1.0)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "KL divergence should be non-negative"

    def test_kl_divergence_temperature(self):
        """Test temperature scaling affects loss."""
        batch_size = 2
        vocab_size = 10

        student_logits = torch.randn(batch_size, vocab_size)
        teacher_logits = torch.randn(batch_size, vocab_size)

        loss_t1 = kl_divergence(student_logits, teacher_logits, temperature=1.0)
        loss_t2 = kl_divergence(student_logits, teacher_logits, temperature=2.0)

        # Different temperatures should give different losses
        assert not torch.isclose(loss_t1, loss_t2), (
            "Different temperatures should give different losses"
        )

    def test_kl_divergence_reduction(self):
        """Test different reduction modes."""
        batch_size = 2
        vocab_size = 10

        student_logits = torch.randn(batch_size, vocab_size)
        teacher_logits = torch.randn(batch_size, vocab_size)

        loss_mean = kl_divergence(student_logits, teacher_logits, reduction="mean")
        loss_sum = kl_divergence(student_logits, teacher_logits, reduction="sum")

        assert loss_mean.dim() == 0
        assert loss_sum.dim() == 0
        assert loss_sum.item() > loss_mean.item(), "Sum should be larger than mean"

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        batch_size = 2
        vocab_size = 10

        logits = torch.randn(batch_size, vocab_size)

        loss = kl_divergence(logits, logits, temperature=1.0)

        # Should be close to zero (within numerical precision)
        assert loss.item() < 1e-5, (
            f"KL divergence should be ~0 for identical distributions, got {loss.item()}"
        )


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss (using PyTorch F.cross_entropy)."""

    def test_cross_entropy_basic(self):
        """Test basic cross-entropy computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Cross-entropy should be non-negative"

    def test_cross_entropy_ignore_index(self):
        """Test ignore_index functionality."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        ignore_index = -100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 0] = ignore_index  # Mark one token to ignore

        loss = F.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=ignore_index
        )

        assert loss.dim() == 0
        assert loss.item() >= 0


class TestCombinedKDLoss:
    """Tests for combined KD loss."""

    def test_combined_kd_loss_basic(self):
        """Test basic combined KD loss computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        ground_truth_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        metrics = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=None,
            ground_truth_targets=ground_truth_labels,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
        )

        assert "total" in metrics
        assert metrics["total"].dim() == 0, "Loss should be scalar"
        assert metrics["total"].item() >= 0, "Loss should be non-negative"
        assert "kl_div" in metrics or "ce_ground_truth" in metrics

    def test_combined_kd_loss_weights(self):
        """Test that different weights produce different losses."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        ground_truth_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        metrics1 = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=None,
            ground_truth_targets=ground_truth_labels,
            kl_weight=0.9,
            ce_teacher_weight=0.05,
            ce_ground_truth_weight=0.05,
            kd_temperature=2.0,
        )

        metrics2 = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=None,
            ground_truth_targets=ground_truth_labels,
            kl_weight=0.1,
            ce_teacher_weight=0.45,
            ce_ground_truth_weight=0.45,
            kd_temperature=2.0,
        )

        # Different weights should produce different losses
        assert not torch.isclose(metrics1["total"], metrics2["total"]), (
            "Different weights should produce different losses"
        )

    def test_combined_kd_loss_gradient_flow(self):
        """Test that gradients flow through combined loss."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        ground_truth_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        metrics = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=None,
            ground_truth_targets=ground_truth_labels,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
        )

        metrics["total"].backward()

        assert student_logits.grad is not None, "Gradients should flow to student_logits"
        assert not torch.allclose(student_logits.grad, torch.zeros_like(student_logits.grad)), (
            "Gradients should be non-zero"
        )


class TestCrossEntropyOnTeacher:
    """Tests for cross-entropy on teacher targets."""

    def test_cross_entropy_on_teacher_basic(self):
        """Test basic cross-entropy on teacher targets."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = cross_entropy_on_teacher(student_logits, teacher_targets)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_cross_entropy_on_teacher_with_ignore_index(self):
        """Test cross-entropy with ignore_index."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        teacher_targets[0, 0] = -100  # Ignore this token

        loss = cross_entropy_on_teacher(student_logits, teacher_targets, ignore_index=-100)

        assert loss.dim() == 0, "Loss should be scalar"


class TestToolNameLoss:
    """Tests for tool name loss."""

    def test_tool_name_loss_basic(self):
        """Test basic tool name loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tool_name_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_tool_name_loss_with_mask(self):
        """Test tool name loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tool_name_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        tool_name_mask[:, :5] = True  # Only first 5 tokens

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert loss.dim() == 0, "Loss should be scalar"


class TestJSONArgumentLoss:
    """Tests for JSON argument loss."""

    def test_json_argument_loss_basic(self):
        """Test basic JSON argument loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_valid_json_tokens = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss = json_argument_loss(
            student_logits, gold_json_text_ids, mask_valid_json_tokens
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestIntegrationCopyLoss:
    """Tests for integration copy loss."""

    def test_integration_copy_loss_basic(self):
        """Test basic integration copy loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, seq_len))
        integration_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss = integration_copy_loss(
            student_logits, tool_result_fields, integration_mask
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestHaltHeadLoss:
    """Tests for halt head loss."""

    def test_halt_head_loss_basic(self):
        """Test basic halt head loss."""
        batch_size = 2
        seq_len = 10

        halt_logits = torch.randn(batch_size, seq_len)
        halt_targets = torch.randint(0, 2, (batch_size, seq_len)).float()

        loss = halt_head_loss(halt_logits, halt_targets)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestIntermediateLayerLoss:
    """Tests for intermediate layer loss."""

    def test_intermediate_layer_loss_basic(self):
        """Test basic intermediate layer loss."""
        batch_size = 2
        seq_len = 10
        d_model = 128

        student_hidden_states = [
            torch.randn(batch_size, seq_len, d_model),
            torch.randn(batch_size, seq_len, d_model),
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, d_model),
            torch.randn(batch_size, seq_len, d_model),
        ]
        layer_mapping = {0: 0, 1: 1}  # Map student layer i to teacher layer i

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestSelfEvaluationLoss:
    """Tests for self-evaluation loss."""

    def test_self_evaluation_loss_basic(self):
        """Test basic self-evaluation loss."""
        batch_size = 2

        student_eval_score = torch.rand(batch_size, 1)  # [B, 1]
        teacher_quality_score = torch.rand(batch_size)  # [B]

        loss = self_evaluation_loss(student_eval_score, teacher_quality_score)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestCreateProjectionLayers:
    """Tests for create_projection_layers function."""

    def test_create_projection_layers_basic(self):
        """Test creating projection layers."""
        student_d_model = 128
        teacher_d_model = 256
        layer_mapping = {0: 0, 1: 2}  # Map student layer 0 to teacher 0, student 1 to teacher 2

        device = torch.device("cpu")
        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        assert len(projection_layers) == len(layer_mapping)
        assert isinstance(projection_layers[0], torch.nn.Module)


class TestCurriculumTemperature:
    """Tests for curriculum temperature function."""

    def test_curriculum_temperature_start(self):
        """Test curriculum temperature at start (epoch 0)."""
        temp = curriculum_temperature(epoch=0, total_epochs=10)
        assert temp == 2.0, "Should start at 2.0"

    def test_curriculum_temperature_end(self):
        """Test curriculum temperature at end."""
        temp = curriculum_temperature(epoch=10, total_epochs=10)
        assert temp == 1.0, "Should end at 1.0"

    def test_curriculum_temperature_midpoint(self):
        """Test curriculum temperature at midpoint."""
        temp = curriculum_temperature(epoch=5, total_epochs=10)
        assert temp == 1.5, "Should be 1.5 at midpoint"
