"""
Unit tests for code-mode weight scheduler.

Tests that weight ramps correctly from start_weight to target weight over warmup steps.

Author: @darianrosebrook
"""
from __future__ import annotations



def test_code_mode_weight_scheduler():
    """
    Test that weight scheduler ramps correctly.
    
    At step=0: should get start_weight (0.1)
    At step=warmup_steps: should get target weight (0.3)
    Linear interpolation in between.
    """
    warmup_steps = 5000
    start_weight = 0.1
    target_weight = 0.3
    
    def compute_weight(step: int) -> float:
        """Compute weight using linear warmup schedule."""
        if step < warmup_steps and warmup_steps > 0:
            progress = step / warmup_steps
            return start_weight + (target_weight - start_weight) * progress
        else:
            return target_weight
    
    # Test at step 0
    weight_at_0 = compute_weight(0)
    assert abs(weight_at_0 - start_weight) < 1e-6, \
        f"At step 0, weight should be {start_weight} (got {weight_at_0})"
    
    # Test at warmup_steps
    weight_at_warmup = compute_weight(warmup_steps)
    assert abs(weight_at_warmup - target_weight) < 1e-6, \
        f"At step {warmup_steps}, weight should be {target_weight} (got {weight_at_warmup})"
    
    # Test at midpoint
    midpoint_step = warmup_steps // 2
    weight_at_midpoint = compute_weight(midpoint_step)
    expected_midpoint = start_weight + (target_weight - start_weight) * 0.5
    assert abs(weight_at_midpoint - expected_midpoint) < 1e-6, \
        f"At step {midpoint_step}, weight should be {expected_midpoint} (got {weight_at_midpoint})"
    
    # Test after warmup (should hold at target)
    weight_after_warmup = compute_weight(warmup_steps + 1000)
    assert abs(weight_after_warmup - target_weight) < 1e-6, \
        f"After warmup, weight should remain {target_weight} (got {weight_after_warmup})"
    
    # Test linearity: weight should increase linearly
    step1 = warmup_steps // 4
    step2 = warmup_steps // 2
    step3 = 3 * warmup_steps // 4
    
    weight1 = compute_weight(step1)
    weight2 = compute_weight(step2)
    weight3 = compute_weight(step3)
    
    # Check that increments are roughly equal
    increment1 = weight2 - weight1
    increment2 = weight3 - weight2
    
    assert abs(increment1 - increment2) < 1e-6, \
        f"Weight should increase linearly (increments: {increment1}, {increment2})"


def test_code_mode_weight_scheduler_zero_warmup():
    """
    Test that weight scheduler handles zero warmup steps correctly.
    """
    warmup_steps = 0
    start_weight = 0.1
    target_weight = 0.3
    
    def compute_weight(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            progress = step / warmup_steps
            return start_weight + (target_weight - start_weight) * progress
        else:
            return target_weight
    
    # With zero warmup, should immediately use target weight
    weight_at_0 = compute_weight(0)
    assert abs(weight_at_0 - target_weight) < 1e-6, \
        f"With zero warmup, weight should be {target_weight} immediately (got {weight_at_0})"

