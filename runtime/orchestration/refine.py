"""
Outer-loop refinement controller with learned halting and CAWS budget enforcement.

Implements:
- Judge score threshold (τ) check
- Delta shrinking detection (diff-IoU↑, failing tests↓)
- Halt probability from learned halt head
- Hard-cap by CAWS tier
- Loop count tracking and logging
"""
# @author: @darianrosebrook

import os
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np


class CAWSBudgetTier(Enum):
    """CAWS budget tiers with different limits."""
    TIER_1 = "tier_1"  # latent=0, loops≤1
    TIER_2 = "tier_2"  # latent≤1, loops≤2
    TIER_3 = "tier_3"  # latent≤3, loops≤3


class RefinementController:
    """
    Controller for outer-loop refinement with learned halting.
    
    Halts when:
    - Judge score ≥ τ AND deltas shrink
    - OR halt probability from learned head > threshold
    - OR hard-cap by CAWS tier reached
    
    Otherwise continues up to max_loops.
    """
    
    def __init__(
        self,
        judge_score_threshold: float = 0.8,
        halt_probability_threshold: float = 0.7,
        caws_tier: CAWSBudgetTier = CAWSBudgetTier.TIER_2,
        max_latent_spans: Optional[int] = None,
        max_loops: Optional[int] = None,
        halt_head_enabled: bool = False,
    ):
        """
        Initialize refinement controller.
        
        Args:
            judge_score_threshold: Minimum judge score (τ) to consider halting
            halt_probability_threshold: Minimum halt probability from learned head
            caws_tier: CAWS budget tier (determines max_loops and max_latent)
            max_latent_spans: Override max latent spans (if None, uses tier default)
            max_loops: Override max loops (if None, uses tier default)
            halt_head_enabled: Whether to use learned halt head (HALT_HEAD env var)
        """
        self.judge_score_threshold = judge_score_threshold
        self.halt_probability_threshold = halt_probability_threshold
        self.caws_tier = caws_tier
        self.halt_head_enabled = halt_head_enabled or (os.getenv("HALT_HEAD", "0") == "1")
        
        # Set tier-based limits
        if max_latent_spans is None:
            max_latent_spans = self._get_tier_max_latent(caws_tier)
        if max_loops is None:
            max_loops = self._get_tier_max_loops(caws_tier)
        
        self.max_latent_spans = max_latent_spans
        self.max_loops = max_loops
        
        # Override with environment variables if set
        env_max_loops = os.getenv("MAX_LOOPS")
        if env_max_loops:
            self.max_loops = int(env_max_loops)
        
        # State tracking
        self.loop_count = 0
        self.previous_output = None
        self.previous_score = None
        self.latent_span_count = 0
        self.halt_logits_history = []
        
    def _get_tier_max_latent(self, tier: CAWSBudgetTier) -> int:
        """Get max latent spans for tier."""
        tier_limits = {
            CAWSBudgetTier.TIER_1: 0,
            CAWSBudgetTier.TIER_2: 1,
            CAWSBudgetTier.TIER_3: 3,
        }
        return tier_limits.get(tier, 1)
    
    def _get_tier_max_loops(self, tier: CAWSBudgetTier) -> int:
        """Get max loops for tier."""
        tier_limits = {
            CAWSBudgetTier.TIER_1: 1,
            CAWSBudgetTier.TIER_2: 2,
            CAWSBudgetTier.TIER_3: 3,
        }
        return tier_limits.get(tier, 2)
    
    def should_halt(
        self,
        current_output: Dict[str, Any],
        judge_score: float,
        halt_logits: Optional[torch.Tensor] = None,
        latent_spans_used: int = 0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if refinement should halt.
        
        Args:
            current_output: Current model output dict
            judge_score: Judge score for current output (0-1)
            halt_logits: Halt head logits [2] (if available)
            latent_spans_used: Number of latent spans used in current iteration
        
        Returns:
            Tuple of (should_halt: bool, metadata: dict)
        """
        self.loop_count += 1
        self.latent_span_count += latent_spans_used
        
        metadata = {
            "loop_count": self.loop_count,
            "judge_score": judge_score,
            "latent_spans_used": latent_spans_used,
            "total_latent_spans": self.latent_span_count,
            "halt_reason": None,
        }
        
        # Check hard-cap by CAWS tier
        if self.loop_count >= self.max_loops:
            metadata["halt_reason"] = "max_loops_reached"
            return True, metadata
        
        if self.latent_span_count > self.max_latent_spans:
            metadata["halt_reason"] = "max_latent_spans_reached"
            return True, metadata
        
        # Check halt head probability (if enabled)
        if self.halt_head_enabled and halt_logits is not None:
            halt_probs = torch.softmax(halt_logits, dim=-1)
            halt_prob = halt_probs[1].item()  # Probability of halting
            self.halt_logits_history.append(halt_prob)
            metadata["halt_probability"] = halt_prob
            
            if halt_prob > self.halt_probability_threshold:
                metadata["halt_reason"] = "halt_head_threshold"
                return True, metadata
        
        # Check judge score threshold AND delta shrinking
        if judge_score >= self.judge_score_threshold:
            # Check if deltas are shrinking
            delta_shrinking = self._check_delta_shrinking(current_output)
            metadata["delta_shrinking"] = delta_shrinking
            
            if delta_shrinking:
                metadata["halt_reason"] = "judge_score_and_delta_shrinking"
                return True, metadata
        
        # Continue refinement
        self.previous_output = current_output
        self.previous_score = judge_score
        return False, metadata
    
    def _check_delta_shrinking(self, current_output: Dict[str, Any]) -> bool:
        """
        Check if deltas are shrinking (convergence indicator).
        
        Metrics:
        - diff-IoU↑ (improving similarity)
        - failing tests↓ (fewer failures)
        
        Args:
            current_output: Current model output
        
        Returns:
            True if deltas are shrinking (converging)
        """
        if self.previous_output is None:
            return False
        
        # Extract metrics from outputs
        current_diff_iou = current_output.get("diff_iou", 0.0)
        current_failing_tests = current_output.get("failing_tests", 0)
        
        prev_diff_iou = self.previous_output.get("diff_iou", 0.0)
        prev_failing_tests = self.previous_output.get("failing_tests", 0)
        
        # Delta shrinking: diff-IoU increasing AND failing tests decreasing
        diff_iou_improving = current_diff_iou >= prev_diff_iou
        failing_tests_decreasing = current_failing_tests <= prev_failing_tests
        
        return diff_iou_improving and failing_tests_decreasing
    
    def reset(self):
        """Reset controller state for new refinement session."""
        self.loop_count = 0
        self.previous_output = None
        self.previous_score = None
        self.latent_span_count = 0
        self.halt_logits_history = []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current controller metadata."""
        return {
            "loop_count": self.loop_count,
            "max_loops": self.max_loops,
            "latent_span_count": self.latent_span_count,
            "max_latent_spans": self.max_latent_spans,
            "caws_tier": self.caws_tier.value,
            "halt_head_enabled": self.halt_head_enabled,
            "halt_logits_history": self.halt_logits_history,
        }


def update_inputs(previous_output: Dict[str, Any], feedback: Optional[str] = None) -> Dict[str, Any]:
    """
    Update inputs for next refinement iteration.
    
    Args:
        previous_output: Previous model output
        feedback: Optional feedback string
    
    Returns:
        Updated input dict for next iteration
    """
    # Extract relevant information from previous output
    updated_inputs = {
        "previous_output": previous_output.get("text", ""),
        "previous_score": previous_output.get("judge_score", 0.0),
    }
    
    if feedback:
        updated_inputs["feedback"] = feedback
    
    # Add error information if available
    if "errors" in previous_output:
        updated_inputs["errors"] = previous_output["errors"]
    
    # Add test results if available
    if "test_results" in previous_output:
        updated_inputs["test_results"] = previous_output["test_results"]
    
    return updated_inputs

