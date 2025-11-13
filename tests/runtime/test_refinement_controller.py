"""
Unit tests for refinement controller.
"""
# @author: @darianrosebrook

import pytest
import torch

from runtime.orchestration.refine import RefinementController, CAWSBudgetTier


class TestRefinementController:
    """Test refinement controller functionality."""

    @pytest.fixture
    def controller(self):
        """Create refinement controller."""
        return RefinementController(
            judge_score_threshold=0.8,
            halt_probability_threshold=0.7,
            caws_tier=CAWSBudgetTier.TIER_2,
            halt_head_enabled=True,
        )

    def test_should_halt_on_max_loops(self, controller):
        """Test that controller halts when max loops reached."""
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}

        # Exceed max loops
        controller.loop_count = controller.max_loops - 1

        should_halt, metadata = controller.should_halt(current_output, judge_score=0.9)

        assert should_halt is True
        assert metadata["halt_reason"] == "max_loops_reached"

    def test_should_halt_on_judge_score_and_delta_shrinking(self, controller):
        """Test that controller halts when judge score ≥ τ and deltas shrink."""
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}
        controller.previous_output = {"text": "test", "diff_iou": 0.8, "failing_tests": 1}

        should_halt, metadata = controller.should_halt(current_output, judge_score=0.85)

        assert should_halt is True
        assert metadata["halt_reason"] == "judge_score_and_delta_shrinking"

    def test_should_halt_on_halt_head_threshold(self, controller):
        """Test that controller halts when halt head probability exceeds threshold."""
        halt_logits = torch.tensor([0.0, 5.0])  # High halt probability
        current_output = {"text": "test"}

        should_halt, metadata = controller.should_halt(
            current_output, judge_score=0.5, halt_logits=halt_logits
        )

        assert should_halt is True
        assert metadata["halt_reason"] == "halt_head_threshold"
        assert metadata["halt_probability"] > controller.halt_probability_threshold

    def test_should_continue_when_conditions_not_met(self, controller):
        """Test that controller continues when halting conditions not met."""
        current_output = {"text": "test", "diff_iou": 0.7, "failing_tests": 2}

        should_halt, metadata = controller.should_halt(current_output, judge_score=0.5)

        assert should_halt is False
        assert metadata["halt_reason"] is None

    def test_tier_limits_enforced(self):
        """Test that CAWS tier limits are enforced."""
        # Tier 1: max loops = 1
        controller_t1 = RefinementController(caws_tier=CAWSBudgetTier.TIER_1)
        assert controller_t1.max_loops == 1
        assert controller_t1.max_latent_spans == 0

        # Tier 2: max loops = 2
        controller_t2 = RefinementController(caws_tier=CAWSBudgetTier.TIER_2)
        assert controller_t2.max_loops == 2
        assert controller_t2.max_latent_spans == 1

        # Tier 3: max loops = 3
        controller_t3 = RefinementController(caws_tier=CAWSBudgetTier.TIER_3)
        assert controller_t3.max_loops == 3
        assert controller_t3.max_latent_spans == 3

    def test_reset_clears_state(self, controller):
        """Test that reset clears controller state."""
        controller.loop_count = 5
        controller.previous_output = {"text": "test"}
        controller.latent_span_count = 3

        controller.reset()

        assert controller.loop_count == 0
        assert controller.previous_output is None
        assert controller.latent_span_count == 0
