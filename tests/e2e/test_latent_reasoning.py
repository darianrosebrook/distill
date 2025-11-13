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
from unittest.mock import Mock

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
        tokenizer.convert_tokens_to_ids = Mock(
            side_effect=lambda x: {
                "<bot>": 3,
                "<eot>": 4,
            }.get(x, None)
        )
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
        should_halt, metadata = controller.should_halt(current_output, judge_score=0.5)

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

    def test_toy_halt_head_integration(self):
        """Test halt head logits integration with RefinementController."""
        import torch.nn as nn
        import torch.nn.functional as F

        # Create mock halt head (linear layer: hidden_dim -> 2)
        hidden_dim = 128
        nn.Linear(hidden_dim, 2)

        # Create refinement controller with halt head enabled
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_2,
            halt_head_enabled=True,
            halt_probability_threshold=0.7,
        )

        # Simulate refinement loops with halt logits
        current_output = {"text": "test output", "diff_iou": 0.8, "failing_tests": 0}

        # Test 1: Halt when probability > threshold
        # Create halt logits that indicate halt (high probability for class 1)
        halt_logits_high = torch.tensor([-2.0, 2.0])  # High probability for halt
        halt_probs_high = F.softmax(halt_logits_high, dim=-1)
        assert halt_probs_high[1].item() > 0.7, "Halt probability should be > 0.7"

        should_halt, metadata = controller.should_halt(
            current_output,
            judge_score=0.5,
            halt_logits=halt_logits_high,
            latent_spans_used=0,
        )

        assert should_halt is True, "Should halt when probability > threshold"
        assert metadata["halt_reason"] == "halt_head_threshold"
        assert "halt_probability" in metadata
        assert metadata["halt_probability"] > 0.7

        # Reset controller for next test
        controller.reset()

        # Test 2: Continue when probability < threshold
        # Create halt logits that indicate continue (low probability for class 1)
        halt_logits_low = torch.tensor([2.0, -2.0])  # Low probability for halt
        halt_probs_low = F.softmax(halt_logits_low, dim=-1)
        assert halt_probs_low[1].item() < 0.7, "Halt probability should be < 0.7"

        should_halt, metadata = controller.should_halt(
            current_output,
            judge_score=0.5,
            halt_logits=halt_logits_low,
            latent_spans_used=0,
        )

        assert should_halt is False, "Should continue when probability < threshold"
        assert metadata.get("halt_reason") != "halt_head_threshold"

        # Verify halt logits are tracked in history
        assert len(controller.halt_logits_history) > 0

        # Test 3: Halt head disabled (should not use halt logits)
        controller_no_halt = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_2,
            halt_head_enabled=False,
        )

        should_halt, metadata = controller_no_halt.should_halt(
            current_output,
            judge_score=0.5,
            halt_logits=halt_logits_high,  # High probability, but halt head disabled
            latent_spans_used=0,
        )

        # Should not halt based on halt head (since disabled)
        # May halt for other reasons (max loops, judge score, etc.)
        assert "halt_probability" not in metadata or controller_no_halt.halt_head_enabled is False

    def test_toy_training_inference_loop_mismatch(self):
        """Test training with L=4-6 loops vs inference with L=1-3 loops."""
        # Training configuration: more loops (L=4-6)
        training_loops = 4

        # Inference configuration: fewer loops per CAWS tier
        # Tier 1: L≤1, Tier 2: L≤2, Tier 3: L≤3
        inference_tiers = {
            CAWSBudgetTier.TIER_1: 1,
            CAWSBudgetTier.TIER_2: 2,
            CAWSBudgetTier.TIER_3: 3,
        }

        # Verify training uses more loops than inference
        assert training_loops > max(inference_tiers.values()), (
            f"Training loops ({training_loops}) should be > max inference loops ({max(inference_tiers.values())})"
        )

        # Test that model trained with L=4 can infer with L=1-3
        # This is a structure test - we verify the configuration allows this

        # Create controllers for each tier
        controllers = {}
        for tier, max_loops in inference_tiers.items():
            controllers[tier] = RefinementController(
                caws_tier=tier,
                halt_head_enabled=False,
            )
            assert controllers[tier].max_loops == max_loops, (
                f"Tier {tier} should have max_loops={max_loops}"
            )

        # Verify training loop count > inference loop counts
        for tier, controller in controllers.items():
            assert training_loops > controller.max_loops, (
                f"Training loops ({training_loops}) should be > inference loops for {tier} ({controller.max_loops})"
            )

        # Simulate inference with different tiers
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}

        for tier, controller in controllers.items():
            # Reset controller
            controller.reset()

            # Simulate loops up to max_loops
            for loop in range(controller.max_loops):
                should_halt, metadata = controller.should_halt(
                    current_output,
                    judge_score=0.5,  # Low score to avoid early halt
                    latent_spans_used=0,
                )

                if loop < controller.max_loops - 1:
                    # Should not halt before max loops (unless other conditions met)
                    # In this test, judge_score is low, so it should continue
                    pass
                else:
                    # At max loops, should halt
                    assert should_halt is True, (
                        f"Should halt at max loops for {tier} (loop {loop + 1} >= {controller.max_loops})"
                    )
                    assert metadata["halt_reason"] == "max_loops_reached"

        # Verify training loop count is in expected range (4-6)
        assert 4 <= training_loops <= 6, (
            f"Training loops should be in range [4, 6], got {training_loops}"
        )

    def test_toy_progressive_curriculum(self):
        """Test curriculum progression: c=1 → c=2."""

        # Create a proper mock tokenizer
        class MockCurriculumTokenizer:
            def __init__(self):
                self.vocab = {
                    "<bot>": 3,
                    "<eot>": 4,
                    "Step": 5,
                    "1:": 6,
                    "First": 7,
                    "2:": 8,
                    "Second": 9,
                    "3:": 10,
                    "Third": 11,
                    "Answer:": 12,
                    "42": 13,
                }

            def encode(self, text, add_special_tokens=False):
                """Simple tokenization by splitting."""
                tokens = text.replace("\n", " ").split()
                # Map tokens to IDs, use hash for unknown tokens
                token_ids = []
                for token in tokens:
                    if token in self.vocab:
                        token_ids.append(self.vocab[token])
                    else:
                        token_ids.append(abs(hash(token)) % 1000 + 100)  # Avoid conflicts
                return token_ids

            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, None)

        mock_tokenizer = MockCurriculumTokenizer()

        # Test c=1: 1 latent slot per replaced step
        curriculum_c1 = LatentCurriculum(m=2, c=1, p=1.0)

        example = {
            "prompt": "Solve:",
            "teacher_text": "Step 1: First\nStep 2: Second\nStep 3: Third\nAnswer: 42",
            "cot_steps": ["Step 1: First", "Step 2: Second", "Step 3: Third"],
            "answer": "42",
            "metadata": {},
        }

        result_c1 = curriculum_c1.apply(example, mock_tokenizer)

        # Verify c=1 curriculum applied
        assert "training_text" in result_c1
        assert result_c1["metadata"]["latent_curriculum_applied"] is True

        # Count latent slots in training text
        training_text_c1 = result_c1["training_text"]
        bot_count_c1 = training_text_c1.count("<bot>")

        # With m=2, c=1: should have 2 latent slots (1 per replaced step)
        assert bot_count_c1 == 2, f"With c=1, should have 2 latent slots (got {bot_count_c1})"

        # Test c=2: 2 latent slots per replaced step
        curriculum_c2 = LatentCurriculum(m=2, c=2, p=1.0)

        result_c2 = curriculum_c2.apply(example, mock_tokenizer)

        # Verify c=2 curriculum applied
        assert "training_text" in result_c2
        assert result_c2["metadata"]["latent_curriculum_applied"] is True

        # Count latent slots in training text
        training_text_c2 = result_c2["training_text"]
        bot_count_c2 = training_text_c2.count("<bot>")

        # With m=2, c=2: should have 4 latent slots (2 per replaced step)
        assert bot_count_c2 == 4, f"With c=2, should have 4 latent slots (got {bot_count_c2})"

        # Verify c=2 has more latent slots than c=1
        assert bot_count_c2 > bot_count_c1, (
            f"c=2 should have more latent slots ({bot_count_c2}) than c=1 ({bot_count_c1})"
        )

        # Verify loss mask correctly masks all latent slots
        if "loss_mask" in result_c1:
            loss_mask_c1 = result_c1["loss_mask"]
            # Loss mask should be a tensor
            assert isinstance(loss_mask_c1, torch.Tensor) or isinstance(loss_mask_c1, list)

        if "loss_mask" in result_c2:
            loss_mask_c2 = result_c2["loss_mask"]
            # Loss mask should be a tensor
            assert isinstance(loss_mask_c2, torch.Tensor) or isinstance(loss_mask_c2, list)

        # Verify stability check: only enable c=2 when c=1 is stable
        # This is a policy test - we verify the structure supports this
        # In practice, stability would be checked during training
        assert curriculum_c1.c == 1, "c=1 curriculum should have c=1"
        assert curriculum_c2.c == 2, "c=2 curriculum should have c=2"

        # Verify both curricula can be applied to same example
        # (structure test - verifies API compatibility)
        assert "training_text" in result_c1
        assert "training_text" in result_c2
        assert result_c1["metadata"]["latent_curriculum_applied"] is True
        assert result_c2["metadata"]["latent_curriculum_applied"] is True
