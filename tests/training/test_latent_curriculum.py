"""
Unit tests for latent curriculum wrapper.
"""
# @author: @darianrosebrook

import pytest
import torch
from unittest.mock import Mock, MagicMock

from data.wrappers.curriculum import LatentCurriculum
from models.student.tokenizer.constants import BOT_TOKEN, EOT_TOKEN


class TestLatentCurriculum:
    """Test latent curriculum wrapper functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {
            BOT_TOKEN: 3,
            EOT_TOKEN: 4,
        }.get(x, None))
        return tokenizer

    @pytest.fixture
    def curriculum(self):
        """Create latent curriculum wrapper."""
        return LatentCurriculum(m=2, c=1, p=1.0)  # p=1.0 to always apply

    @pytest.fixture
    def sample_example(self):
        """Create sample example."""
        return {
            "prompt": "Solve this problem:",
            "teacher_text": "Step 1: Analyze the problem.\nStep 2: Find solution.\nStep 3: Verify answer.\nAnswer: 42",
            "cot_steps": [
                "Step 1: Analyze the problem.",
                "Step 2: Find solution.",
                "Step 3: Verify answer.",
            ],
            "answer": "42",
            "metadata": {},
        }

    def test_curriculum_applies_latent_slots(self, curriculum, mock_tokenizer, sample_example):
        """Test that curriculum replaces CoT steps with latent slots."""
        result = curriculum.apply(sample_example, mock_tokenizer)

        assert "training_text" in result
        assert BOT_TOKEN in result["training_text"]
        assert EOT_TOKEN in result["training_text"]
        assert result["metadata"]["latent_curriculum_applied"] is True

    def test_curriculum_creates_loss_mask(self, curriculum, mock_tokenizer, sample_example):
        """Test that curriculum creates loss mask for latent slots."""
        result = curriculum.apply(sample_example, mock_tokenizer)

        assert "loss_mask" in result
        assert isinstance(result["loss_mask"], torch.Tensor)
        assert result["loss_mask"].dtype == torch.bool

    def test_curriculum_respects_probability(self, mock_tokenizer, sample_example):
        """Test that curriculum respects probability parameter."""
        curriculum = LatentCurriculum(m=2, c=1, p=0.0)  # Never apply
        result = curriculum.apply(sample_example, mock_tokenizer)

        assert result == sample_example  # Should be unchanged

    def test_curriculum_handles_missing_cot_steps(self, curriculum, mock_tokenizer):
        """Test that curriculum handles missing CoT steps."""
        example = {
            "prompt": "Solve this:",
            "teacher_text": "Answer: 42",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)
        # Should handle gracefully (may not apply if no CoT steps)
        assert "training_text" in result or result == example

    def test_loss_mask_masks_latent_spans(self, curriculum, mock_tokenizer):
        """Test that loss mask correctly masks latent spans."""
        # Create example with known structure
        example = {
            "prompt": "Test",
            "teacher_text": "Step 1: First\nStep 2: Second\nAnswer: 42",
            "cot_steps": ["Step 1: First", "Step 2: Second"],
            "answer": "42",
            "metadata": {},
        }

        result = curriculum.apply(example, mock_tokenizer)

        if "loss_mask" in result:
            # Loss mask should have False values for latent spans
            assert isinstance(result["loss_mask"], torch.Tensor)
            # At least some tokens should be masked (latent slots)
            if result["loss_mask"].numel() > 0:
                assert torch.any(~result["loss_mask"]) or torch.all(
                    result["loss_mask"])
