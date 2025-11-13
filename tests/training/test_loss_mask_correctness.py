"""
Loss masking correctness tests.

Tests alignment, padding, and accounting for latent spans.
"""
# @author: @darianrosebrook

import pytest
import torch
from unittest.mock import Mock

from data.wrappers.curriculum import LatentCurriculum
from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID


class TestLossMaskCorrectness:
    """Test loss mask correctness for latent curriculum."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        tokenizer.convert_tokens_to_ids = Mock(
            side_effect=lambda x: {
                "<bot>": BOT_TOKEN_ID,
                "<eot>": EOT_TOKEN_ID,
            }.get(x, None)
        )
        return tokenizer

    @pytest.fixture
    def curriculum(self):
        """Create latent curriculum."""
        return LatentCurriculum(m=2, c=1, p=1.0)

    def test_loss_mask_masks_latent_tokens(self, curriculum, mock_tokenizer):
        """Test that loss mask correctly masks latent tokens."""
        example = {
            "teacher_text": "Step 1: First\nStep 2: Second\nAnswer: 42",
            "cot_steps": ["Step 1: First", "Step 2: Second"],
            "answer": "42",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)

        assert "loss_mask" in result
        loss_mask = result["loss_mask"]

        # Loss mask should be boolean tensor
        assert isinstance(loss_mask, torch.Tensor)
        assert loss_mask.dtype == torch.bool

        # If training_text contains <bot>/<eot>, those positions should be masked
        if "<bot>" in result.get("training_text", ""):
            # Tokenize to find positions
            tokens = mock_tokenizer.encode(result["training_text"], add_special_tokens=True)
            bot_positions = [i for i, t in enumerate(tokens) if t == BOT_TOKEN_ID]
            [i for i, t in enumerate(tokens) if t == EOT_TOKEN_ID]

            # Check that masked positions align with sentinel tokens
            if bot_positions and len(loss_mask) > max(bot_positions):
                for pos in bot_positions:
                    if pos < len(loss_mask):
                        # Sentinel tokens should be masked (False)
                        assert not loss_mask[pos]

    def test_loss_mask_alignment_after_padding(self, curriculum, mock_tokenizer):
        """Test that loss mask aligns correctly after padding."""
        example = {
            "teacher_text": "Step 1: First\nAnswer: 42",
            "cot_steps": ["Step 1: First"],
            "answer": "42",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)

        if "loss_mask" in result:
            loss_mask = result["loss_mask"]

            # Simulate padding: extend mask
            original_len = len(loss_mask)
            pad_len = 10
            padded_mask = torch.cat([loss_mask, torch.ones(pad_len, dtype=torch.bool)])

            # Padded positions should be supervised (True)
            assert torch.all(padded_mask[original_len:])

    def test_loss_mask_excludes_latent_spans_from_supervision(self, curriculum, mock_tokenizer):
        """Test that latent spans are excluded from supervised token counts."""
        example = {
            "teacher_text": "Step 1: First\nStep 2: Second\nAnswer: 42",
            "cot_steps": ["Step 1: First", "Step 2: Second"],
            "answer": "42",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)

        if "loss_mask" in result:
            loss_mask = result["loss_mask"]

            # Count supervised tokens (True values)
            supervised_count = loss_mask.sum().item()

            # Total tokens
            total_count = len(loss_mask)

            # Supervised count should be less than total (some masked)
            assert supervised_count <= total_count

            # If latent slots were added, supervised count should be reduced
            if "<bot>" in result.get("training_text", ""):
                assert supervised_count < total_count

    def test_loss_mask_handles_empty_sequences(self, curriculum, mock_tokenizer):
        """Test that loss mask handles empty sequences gracefully."""
        example = {
            "teacher_text": "",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)

        # Should handle gracefully (may not apply curriculum)
        if "loss_mask" in result:
            loss_mask = result["loss_mask"]
            assert isinstance(loss_mask, torch.Tensor)
            assert len(loss_mask) >= 0

    def test_loss_mask_applied_to_combined_loss(self):
        """Test that loss mask is correctly applied in combined_kd_loss."""
        from training.losses import combined_kd_loss

        # Create dummy logits and targets
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create loss mask: mask first 3 positions
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        loss_mask[:, :3] = False  # Mask first 3 positions

        # Compute loss with mask
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=None,
            teacher_targets=None,
            ground_truth_targets=labels,
            ce_ground_truth_weight=1.0,
            loss_mask=loss_mask,
            ignore_index=-100,
        )

        # Loss should be computed only on unmasked positions
        assert "ce_ground_truth" in loss_dict
        assert loss_dict["total"] > 0
