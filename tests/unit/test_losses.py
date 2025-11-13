"""
Unit tests for training loss functions.
"""

import torch
import torch.nn.functional as F

from training.losses import (
    kl_divergence,
    combined_kd_loss,
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
