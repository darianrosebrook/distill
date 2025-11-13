"""
End-to-end tests for latent reasoning.

Tests:
- Training with latent curriculum
- Inference with latent spans
- CAWS budget enforcement
"""
# @author: @darianrosebrook

import pytest
import torch
import os
from unittest.mock import Mock, patch

from data.wrappers.curriculum import LatentCurriculum
from runtime.engine.loop import LatentModeEngine
from runtime.orchestration.refine import RefinementController, CAWSBudgetTier


class TestLatentReasoningE2E:
    """End-to-end tests for latent reasoning."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.forward_hidden = Mock(return_value=torch.randn(1, 10, 128))
        model.forward_decode = Mock(return_value=(torch.randn(1, 1, 1000), []))
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="test output")
        tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {
            "<bot>": 3,
            "<eot>": 4,
        }.get(x, None))
        return tokenizer
    
    def test_training_with_latent_curriculum(self, mock_tokenizer):
        """Test that training pipeline integrates latent curriculum."""
        curriculum = LatentCurriculum(m=2, c=1, p=1.0)
        
        example = {
            "prompt": "Solve:",
            "teacher_text": "Step 1: First\nStep 2: Second\nAnswer: 42",
            "cot_steps": ["Step 1: First", "Step 2: Second"],
            "answer": "42",
            "metadata": {},
        }
        
        result = curriculum.apply(example, mock_tokenizer)
        
        assert "training_text" in result
        assert "loss_mask" in result
        assert result["metadata"]["latent_curriculum_applied"] is True
    
    def test_inference_with_latent_spans(self, mock_model, mock_tokenizer):
        """Test that inference handles latent spans correctly."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )
        
        input_ids = torch.tensor([[1, 3, 100, 101, 4, 200]])  # With sentinels
        
        result = engine.generate_with_latent_mode(
            input_ids,
            max_new_tokens=10,
        )
        
        assert "tokens" in result
        assert "mode_transitions" in result
        assert len(result["mode_transitions"]) >= 2
    
    def test_caws_budget_enforcement(self):
        """Test that CAWS budget limits are enforced."""
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_2,
            halt_head_enabled=False,
        )
        
        current_output = {"text": "test"}
        
        # Should halt after max loops
        controller.loop_count = controller.max_loops - 1
        should_halt, metadata = controller.should_halt(
            current_output, judge_score=0.5
        )
        
        assert should_halt is True
        assert metadata["halt_reason"] == "max_loops_reached"
    
    def test_latent_spans_respect_caws_tier(self):
        """Test that latent spans respect CAWS tier limits."""
        controller_t1 = RefinementController(caws_tier=CAWSBudgetTier.TIER_1)
        assert controller_t1.max_latent_spans == 0
        
        controller_t2 = RefinementController(caws_tier=CAWSBudgetTier.TIER_2)
        assert controller_t2.max_latent_spans == 1
        
        controller_t3 = RefinementController(caws_tier=CAWSBudgetTier.TIER_3)
        assert controller_t3.max_latent_spans == 3

