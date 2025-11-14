"""
CAWS budget enforcement tests.

Tests tier limits, hard caps, and budget breach handling.
"""
# @author: @darianrosebrook

import pytest
from unittest.mock import Mock

from runtime.orchestration.refine import RefinementController, CAWSBudgetTier


class TestCAWSBudgetEnforcement:
    """Test CAWS budget tier enforcement."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.use_halt_head = False
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return Mock()

    @pytest.fixture
    def mock_judge(self):
        """Create mock judge."""
        judge = Mock()
        judge.score = Mock(return_value={"score": 0.5})
        return judge

    def test_tier_1_limits(self, mock_model, mock_tokenizer):
        """Test Tier-1 limits: latent=0, loops≤1."""
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_1,
        )

        assert controller.max_latent_spans == 0
        assert controller.max_loops == 1

        # Should halt after 1 loop
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}
        should_halt, metadata = controller.should_halt(current_output, judge_score=0.9)

        assert should_halt is True
        assert controller.loop_count <= 1

    def test_tier_2_limits(self, mock_model, mock_tokenizer):
        """Test Tier-2 limits: latent≤1, loops≤2."""
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_2,
        )

        assert controller.max_latent_spans == 1
        assert controller.max_loops == 2

        # Should halt after 2 loops
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}
        # First loop - should continue
        should_halt_1, _ = controller.should_halt(current_output, judge_score=0.5)
        assert should_halt_1 is False
        # Second loop - should halt (max loops reached)
        should_halt_2, metadata = controller.should_halt(current_output, judge_score=0.5)
        assert should_halt_2 is True
        assert controller.loop_count <= 2

    def test_tier_3_limits(self, mock_model, mock_tokenizer):
        """Test Tier-3 limits: latent≤3, loops≤3."""
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_3,
        )

        assert controller.max_latent_spans == 3
        assert controller.max_loops == 3

        # Should halt after 3 loops
        current_output = {"text": "test", "diff_iou": 0.9, "failing_tests": 0}
        # First and second loops - should continue
        should_halt_1, _ = controller.should_halt(current_output, judge_score=0.5)
        assert should_halt_1 is False
        should_halt_2, _ = controller.should_halt(current_output, judge_score=0.5)
        assert should_halt_2 is False
        # Third loop - should halt (max loops reached)
        should_halt_3, metadata = controller.should_halt(current_output, judge_score=0.5)
        assert should_halt_3 is True
        assert controller.loop_count <= 3

    def test_budget_breach_forces_halt(self, mock_model, mock_tokenizer, mock_judge):
        """Test that budget breach forces halt even if Judge score is low."""
        controller = RefinementController(
            caws_tier=CAWSBudgetTier.TIER_1,
        )

        # Low Judge score (would normally continue, but tier limit forces halt)
        current_output = {"text": "test", "diff_iou": 0.3, "failing_tests": 5}
        should_halt, metadata = controller.should_halt(current_output, judge_score=0.3)

        # Should still halt at max loops (hard cap) even with low judge score
        assert should_halt is True
        assert metadata["halt_reason"] == "max_loops_reached"
        assert controller.loop_count <= 1

    def test_latent_spans_respect_tier(self, mock_model, mock_tokenizer):
        """Test that latent spans respect tier limits."""
        from runtime.engine.loop import LatentModeEngine

        # Tier-1: no latent spans
        engine_t1 = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_spans=0,  # Tier-1 limit
        )

        # Tier-2: ≤1 latent span
        engine_t2 = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_spans=1,  # Tier-2 limit
        )

        # Tier-3: ≤3 latent spans
        engine_t3 = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_spans=3,  # Tier-3 limit
        )

        assert engine_t1.max_latent_spans == 0
        assert engine_t2.max_latent_spans == 1
        assert engine_t3.max_latent_spans == 3
