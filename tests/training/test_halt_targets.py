"""
Tests for training/halt_targets.py - Halt head target derivation.

Tests HaltHeadTargets class and halt target derivation from curriculum,
Judge scores, and combined signals.
"""
# @author: @darianrosebrook

import torch

from training.halt_targets import HaltHeadTargets, create_halt_targets_batch


class TestHaltHeadTargets:
    """Test HaltHeadTargets class functionality."""

    def test_initialization_default(self):
        """Test HaltHeadTargets initialization with default parameters."""
        halt_targets = HaltHeadTargets()

        assert halt_targets.judge_score_threshold == 0.8
        assert halt_targets.delta_shrinking_threshold == 0.05
        assert halt_targets.caws_tier == "Tier-2"
        assert halt_targets.warmup_steps == 1000

    def test_initialization_custom(self):
        """Test HaltHeadTargets initialization with custom parameters."""
        halt_targets = HaltHeadTargets(
            judge_score_threshold=0.9,
            delta_shrinking_threshold=0.1,
            caws_tier="Tier-1",
            warmup_steps=500,
        )

        assert halt_targets.judge_score_threshold == 0.9
        assert halt_targets.delta_shrinking_threshold == 0.1
        assert halt_targets.caws_tier == "Tier-1"
        assert halt_targets.warmup_steps == 500

    def test_derive_from_curriculum_at_max_loops(self):
        """Test curriculum derivation when at max loops."""
        halt_targets = HaltHeadTargets()

        # At max loops, must halt
        target = halt_targets.derive_from_curriculum(
            loop_index=9, max_loops=10, curriculum_stage=None
        )

        assert target == 1  # Must halt

    def test_derive_from_curriculum_before_max_loops(self):
        """Test curriculum derivation before max loops."""
        halt_targets = HaltHeadTargets()

        # Before max loops, continue
        target = halt_targets.derive_from_curriculum(
            loop_index=5, max_loops=10, curriculum_stage=None
        )

        assert target == 0  # Continue

    def test_derive_from_curriculum_stage_0(self):
        """Test curriculum derivation with stage 0 (early stage)."""
        halt_targets = HaltHeadTargets()

        # Stage 0: allow all loops except last
        target = halt_targets.derive_from_curriculum(
            loop_index=8, max_loops=10, curriculum_stage=0
        )

        assert target == 0  # Continue (not at max yet)

        # At max, must halt
        target = halt_targets.derive_from_curriculum(
            loop_index=9, max_loops=10, curriculum_stage=0
        )

        assert target == 1  # Must halt

    def test_derive_from_curriculum_stage_1(self):
        """Test curriculum derivation with stage 1 (mid stage)."""
        halt_targets = HaltHeadTargets()

        # Stage 1: halt at 75% of max
        max_loops = 10
        int(max_loops * 0.75)  # 7

        # Before threshold, continue
        target = halt_targets.derive_from_curriculum(
            loop_index=6, max_loops=max_loops, curriculum_stage=1
        )

        assert target == 0  # Continue

        # At threshold, halt
        target = halt_targets.derive_from_curriculum(
            loop_index=7, max_loops=max_loops, curriculum_stage=1
        )

        assert target == 1  # Halt

    def test_derive_from_curriculum_stage_2_plus(self):
        """Test curriculum derivation with stage 2+ (late stage)."""
        halt_targets = HaltHeadTargets()

        # Stage 2+: halt at 50% of max
        max_loops = 10
        int(max_loops * 0.5)  # 5

        # Before threshold, continue
        target = halt_targets.derive_from_curriculum(
            loop_index=4, max_loops=max_loops, curriculum_stage=2
        )

        assert target == 0  # Continue

        # At threshold, halt
        target = halt_targets.derive_from_curriculum(
            loop_index=5, max_loops=max_loops, curriculum_stage=2
        )

        assert target == 1  # Halt

        # Stage 3 should also use 50% threshold
        target = halt_targets.derive_from_curriculum(
            loop_index=5, max_loops=max_loops, curriculum_stage=3
        )

        assert target == 1  # Halt

    def test_derive_from_judge_above_threshold_no_prev(self):
        """Test Judge derivation when score above threshold, no previous score."""
        halt_targets = HaltHeadTargets(judge_score_threshold=0.8)

        # Score above threshold, but no previous score
        target = halt_targets.derive_from_judge(judge_score=0.9, prev_score=None, loop_index=0)

        assert target == 0  # Continue to see if it improves

    def test_derive_from_judge_above_threshold_delta_shrinking(self):
        """Test Judge derivation when score above threshold with delta shrinking."""
        halt_targets = HaltHeadTargets(
            judge_score_threshold=0.8, delta_shrinking_threshold=0.05
        )

        # Score above threshold, but delta is small (shrinking)
        prev_score = 0.88
        current_score = 0.90
        current_score - prev_score  # 0.02 < 0.05

        target = halt_targets.derive_from_judge(
            judge_score=current_score, prev_score=prev_score, loop_index=5
        )

        assert target == 1  # Halt (delta shrinking)

    def test_derive_from_judge_above_threshold_large_delta(self):
        """Test Judge derivation when score above threshold with large delta."""
        halt_targets = HaltHeadTargets(
            judge_score_threshold=0.8, delta_shrinking_threshold=0.05
        )

        # Score above threshold, delta is large (still improving)
        prev_score = 0.75
        current_score = 0.90
        current_score - prev_score  # 0.15 > 0.05

        target = halt_targets.derive_from_judge(
            judge_score=current_score, prev_score=prev_score, loop_index=5
        )

        assert target == 0  # Continue (still improving)

    def test_derive_from_judge_below_threshold(self):
        """Test Judge derivation when score below threshold."""
        halt_targets = HaltHeadTargets(judge_score_threshold=0.8)

        # Score below threshold
        target = halt_targets.derive_from_judge(judge_score=0.5, prev_score=None, loop_index=0)

        assert target == 0  # Continue

    def test_derive_from_judge_at_threshold(self):
        """Test Judge derivation when score exactly at threshold."""
        halt_targets = HaltHeadTargets(judge_score_threshold=0.8)

        # Score at threshold
        target = halt_targets.derive_from_judge(judge_score=0.8, prev_score=None, loop_index=0)

        assert target == 0  # Continue (no previous score to compare)

    def test_derive_from_combined_max_loops(self):
        """Test combined derivation with max loops (hard cap)."""
        halt_targets = HaltHeadTargets()

        # Hard cap: must halt at max loops
        target = halt_targets.derive_from_combined(
            loop_index=9,
            max_loops=10,
            judge_score=0.9,
            prev_score=0.8,
            curriculum_stage=0,
        )

        assert target == 1  # Must halt

    def test_derive_from_combined_judge_halts(self):
        """Test combined derivation when Judge signals halt."""
        halt_targets = HaltHeadTargets(
            judge_score_threshold=0.8, delta_shrinking_threshold=0.05
        )

        # Judge signals halt (delta shrinking)
        target = halt_targets.derive_from_combined(
            loop_index=5,
            max_loops=10,
            judge_score=0.9,
            prev_score=0.88,  # Small delta
            curriculum_stage=0,
        )

        assert target == 1  # Halt (Judge signal)

    def test_derive_from_combined_curriculum_fallback(self):
        """Test combined derivation falls back to curriculum when no Judge signal."""
        halt_targets = HaltHeadTargets()

        # No Judge signal, use curriculum
        target = halt_targets.derive_from_combined(
            loop_index=5,
            max_loops=10,
            judge_score=None,
            prev_score=None,
            curriculum_stage=2,  # Stage 2: halt at 50%
        )

        assert target == 1  # Halt (curriculum stage 2)

    def test_derive_from_combined_no_signals(self):
        """Test combined derivation with no signals (default continue)."""
        halt_targets = HaltHeadTargets()

        # No signals, before max loops
        target = halt_targets.derive_from_combined(
            loop_index=3,
            max_loops=10,
            judge_score=None,
            prev_score=None,
            curriculum_stage=None,
        )

        assert target == 0  # Continue (default)

    def test_should_apply_loss_before_warmup(self):
        """Test should_apply_loss before warmup steps."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        # Before warmup
        should_apply = halt_targets.should_apply_loss(current_step=500)

        assert should_apply is False

    def test_should_apply_loss_at_warmup(self):
        """Test should_apply_loss at warmup steps."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        # At warmup
        should_apply = halt_targets.should_apply_loss(current_step=1000)

        assert should_apply is True

    def test_should_apply_loss_after_warmup(self):
        """Test should_apply_loss after warmup steps."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        # After warmup
        should_apply = halt_targets.should_apply_loss(current_step=1500)

        assert should_apply is True

    def test_should_apply_loss_zero_warmup(self):
        """Test should_apply_loss with zero warmup steps."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        # Should apply immediately
        should_apply = halt_targets.should_apply_loss(current_step=0)

        assert should_apply is True


class TestCreateHaltTargetsBatch:
    """Test create_halt_targets_batch function."""

    def test_create_batch_before_warmup(self):
        """Test batch creation before warmup (should return None)."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        batch_metadata = [
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": 0.9,
                "prev_score": 0.8,
                "curriculum_stage": 0,
            }
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=500)

        assert result is None

    def test_create_batch_after_warmup(self):
        """Test batch creation after warmup."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        batch_metadata = [
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": 0.9,
                "prev_score": 0.88,  # Small delta
                "curriculum_stage": 0,
            },
            {
                "loop_index": 3,
                "max_loops": 10,
                "judge_score": 0.5,
                "prev_score": None,
                "curriculum_stage": 0,
            },
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=1500)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)
        assert result.dtype == torch.long

        # First sample should halt (Judge signal)
        assert result[0].item() == 1

        # Second sample should continue (low score)
        assert result[1].item() == 0

    def test_create_batch_missing_metadata(self):
        """Test batch creation with missing metadata fields."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        batch_metadata = [
            {
                # Missing loop_index, max_loops, etc.
            },
            {
                "loop_index": 5,
                # Missing max_loops
            },
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=100)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    def test_create_batch_empty_metadata(self):
        """Test batch creation with empty metadata list."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        batch_metadata = []

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=100)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (0,)

    def test_create_batch_curriculum_stage_variations(self):
        """Test batch creation with different curriculum stages."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        batch_metadata = [
            {
                "loop_index": 4,
                "max_loops": 10,
                "judge_score": None,
                "prev_score": None,
                "curriculum_stage": 0,  # Stage 0: continue
            },
            {
                "loop_index": 7,
                "max_loops": 10,
                "judge_score": None,
                "prev_score": None,
                "curriculum_stage": 1,  # Stage 1: halt at 75%
            },
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": None,
                "prev_score": None,
                "curriculum_stage": 2,  # Stage 2: halt at 50%
            },
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=100)

        assert result is not None
        assert result.shape == (3,)

        # Stage 0: continue (loop_index 4 < 9)
        assert result[0].item() == 0

        # Stage 1: halt (loop_index 7 >= 7.5 -> 7)
        assert result[1].item() == 1

        # Stage 2: halt (loop_index 5 >= 5)
        assert result[2].item() == 1

    def test_create_batch_judge_score_variations(self):
        """Test batch creation with different Judge score scenarios."""
        halt_targets = HaltHeadTargets(
            warmup_steps=0, judge_score_threshold=0.8, delta_shrinking_threshold=0.05
        )

        batch_metadata = [
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": 0.9,
                "prev_score": 0.88,  # Small delta (shrinking)
                "curriculum_stage": 0,
            },
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": 0.9,
                "prev_score": 0.75,  # Large delta (improving)
                "curriculum_stage": 0,
            },
            {
                "loop_index": 5,
                "max_loops": 10,
                "judge_score": 0.5,  # Below threshold
                "prev_score": None,
                "curriculum_stage": 0,
            },
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=100)

        assert result is not None
        assert result.shape == (3,)

        # Delta shrinking: halt
        assert result[0].item() == 1

        # Large delta: continue
        assert result[1].item() == 0

        # Below threshold: continue
        assert result[2].item() == 0

    def test_create_batch_max_loops_edge_cases(self):
        """Test batch creation with max loops edge cases."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        batch_metadata = [
            {
                "loop_index": 9,
                "max_loops": 10,
                "judge_score": None,
                "prev_score": None,
                "curriculum_stage": None,
            },
            {
                "loop_index": 10,
                "max_loops": 10,
                "judge_score": None,
                "prev_score": None,
                "curriculum_stage": None,
            },
        ]

        result = create_halt_targets_batch(batch_metadata, halt_targets, current_step=100)

        assert result is not None
        assert result.shape == (2,)

        # At max-1: halt
        assert result[0].item() == 1

        # At max: halt
        assert result[1].item() == 1


class TestHaltTargetsIntegration:
    """Test integration scenarios for halt targets."""

    def test_training_progression_curriculum(self):
        """Test halt targets across training progression with curriculum."""
        halt_targets = HaltHeadTargets(warmup_steps=0)

        max_loops = 10
        curriculum_stage = 2  # Late stage: halt at 50%

        targets = []
        for loop_index in range(max_loops):
            target = halt_targets.derive_from_combined(
                loop_index=loop_index,
                max_loops=max_loops,
                judge_score=None,
                prev_score=None,
                curriculum_stage=curriculum_stage,
            )
            targets.append(target)

        # Should continue until 50% (loop 5), then halt
        assert targets[:5] == [0, 0, 0, 0, 0]  # Continue
        assert targets[5:] == [1, 1, 1, 1, 1]  # Halt

    def test_training_progression_judge(self):
        """Test halt targets across training progression with Judge scores."""
        halt_targets = HaltHeadTargets(
            warmup_steps=0, judge_score_threshold=0.8, delta_shrinking_threshold=0.05
        )

        max_loops = 10
        judge_scores = [0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92]

        targets = []
        prev_score = None
        for loop_index, judge_score in enumerate(judge_scores):
            target = halt_targets.derive_from_combined(
                loop_index=loop_index,
                max_loops=max_loops,
                judge_score=judge_score,
                prev_score=prev_score,
                curriculum_stage=None,
            )
            targets.append(target)
            prev_score = judge_score

        # Should continue until delta shrinking detected
        # At loop 5: 0.88 -> 0.90 (delta 0.02 < 0.05) -> halt
        assert targets[5] == 1  # Halt when delta shrinking

    def test_warmup_schedule(self):
        """Test warmup schedule behavior."""
        halt_targets = HaltHeadTargets(warmup_steps=1000)

        batch_metadata = [{"loop_index": 5, "max_loops": 10}]

        # Before warmup
        result_before = create_halt_targets_batch(batch_metadata, halt_targets, current_step=500)
        assert result_before is None

        # At warmup
        result_at = create_halt_targets_batch(batch_metadata, halt_targets, current_step=1000)
        assert result_at is not None

        # After warmup
        result_after = create_halt_targets_batch(batch_metadata, halt_targets, current_step=1500)
        assert result_after is not None








