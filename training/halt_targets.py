"""
Halt head target derivation for learned halting.

Derives supervision targets from:
- Curriculum stage boundaries
- Judge score thresholds
- Delta shrinking detection
- CAWS tier limits
"""
# @author: @darianrosebrook

from typing import Dict, Any, Optional, List
import torch



class HaltHeadTargets:
    """
    Derives halt head supervision targets from curriculum/Judge signals.
    
    Targets indicate when refinement should stop:
    - 0 = continue (refinement should continue)
    - 1 = halt (refinement should stop)
    """
    
    def __init__(
        self,
        judge_score_threshold: float = 0.8,
        delta_shrinking_threshold: float = 0.05,  # 5% improvement threshold
        caws_tier: str = "Tier-2",
        warmup_steps: int = 1000,
    ):
        """
        Initialize halt head target derivation.
        
        Args:
            judge_score_threshold: Judge score threshold for halting (tau)
            delta_shrinking_threshold: Minimum improvement for delta shrinking
            caws_tier: CAWS tier for loop limits
            warmup_steps: Number of warmup steps before applying halt loss
        """
        self.judge_score_threshold = judge_score_threshold
        self.delta_shrinking_threshold = delta_shrinking_threshold
        self.caws_tier = caws_tier
        self.warmup_steps = warmup_steps
    
    def derive_from_curriculum(
        self,
        loop_index: int,
        max_loops: int,
        curriculum_stage: Optional[int] = None,
    ) -> int:
        """
        Derive halt target from curriculum stage boundaries.
        
        Early curriculum stages: allow more loops
        Later stages: encourage earlier halting
        
        Args:
            loop_index: Current loop index (0-indexed)
            max_loops: Maximum loops allowed for tier
            curriculum_stage: Optional curriculum stage (0=early, higher=later)
        
        Returns:
            Target: 0 (continue) or 1 (halt)
        """
        # If at max loops, must halt
        if loop_index >= max_loops - 1:
            return 1
        
        # Curriculum-based halting: later stages encourage earlier halting
        if curriculum_stage is not None:
            # Stage 0: allow all loops
            # Stage 1: halt at 75% of max
            # Stage 2+: halt at 50% of max
            if curriculum_stage == 0:
                return 0 if loop_index < max_loops - 1 else 1
            elif curriculum_stage == 1:
                halt_threshold = int(max_loops * 0.75)
                return 1 if loop_index >= halt_threshold else 0
            else:
                halt_threshold = int(max_loops * 0.5)
                return 1 if loop_index >= halt_threshold else 0
        
        # Default: allow all loops except last
        return 0 if loop_index < max_loops - 1 else 1
    
    def derive_from_judge(
        self,
        judge_score: float,
        prev_score: Optional[float] = None,
        loop_index: int = 0,
    ) -> int:
        """
        Derive halt target from Judge score and delta shrinking.
        
        Args:
            judge_score: Current Judge score (0-1)
            prev_score: Previous Judge score (for delta shrinking)
            loop_index: Current loop index
        
        Returns:
            Target: 0 (continue) or 1 (halt)
        """
        # If score exceeds threshold, consider halting
        if judge_score >= self.judge_score_threshold:
            # Check delta shrinking: if score improved significantly, halt
            if prev_score is not None:
                score_delta = judge_score - prev_score
                if score_delta < self.delta_shrinking_threshold:
                    # Score improvement is small (deltas shrinking), halt
                    return 1
            else:
                # First loop with high score, continue to see if it improves
                return 0
        
        # Score below threshold, continue
        return 0
    
    def derive_from_combined(
        self,
        loop_index: int,
        max_loops: int,
        judge_score: Optional[float] = None,
        prev_score: Optional[float] = None,
        curriculum_stage: Optional[int] = None,
    ) -> int:
        """
        Derive halt target from combined signals (curriculum + Judge).
        
        Priority:
        1. CAWS tier limit (hard cap)
        2. Judge score + delta shrinking
        3. Curriculum stage boundaries
        
        Args:
            loop_index: Current loop index
            max_loops: Maximum loops for tier
            judge_score: Optional current Judge score
            prev_score: Optional previous Judge score
            curriculum_stage: Optional curriculum stage
        
        Returns:
            Target: 0 (continue) or 1 (halt)
        """
        # Hard cap: must halt at max loops
        if loop_index >= max_loops - 1:
            return 1
        
        # Judge-based halting (if available)
        if judge_score is not None:
            judge_target = self.derive_from_judge(judge_score, prev_score, loop_index)
            if judge_target == 1:
                return 1
        
        # Curriculum-based halting (fallback)
        return self.derive_from_curriculum(loop_index, max_loops, curriculum_stage)
    
    def should_apply_loss(
        self,
        current_step: int,
    ) -> bool:
        """
        Check if halt loss should be applied (warmup schedule).
        
        Args:
            current_step: Current training step
        
        Returns:
            True if loss should be applied, False during warmup
        """
        return current_step >= self.warmup_steps


def create_halt_targets_batch(
    batch_metadata: List[Dict[str, Any]],
    halt_targets: HaltHeadTargets,
    current_step: int,
) -> Optional[torch.Tensor]:
    """
    Create halt targets batch from batch metadata.
    
    Args:
        batch_metadata: List of metadata dicts per sample with:
            - loop_index: int
            - max_loops: int
            - judge_score: Optional[float]
            - prev_score: Optional[float]
            - curriculum_stage: Optional[int]
        halt_targets: HaltHeadTargets instance
        current_step: Current training step
    
    Returns:
        [B] tensor of halt targets (0=continue, 1=halt) or None if warmup
    """
    if not halt_targets.should_apply_loss(current_step):
        return None
    
    targets = []
    for meta in batch_metadata:
        target = halt_targets.derive_from_combined(
            loop_index=meta.get("loop_index", 0),
            max_loops=meta.get("max_loops", 2),
            judge_score=meta.get("judge_score"),
            prev_score=meta.get("prev_score"),
            curriculum_stage=meta.get("curriculum_stage"),
        )
        targets.append(target)
    
    return torch.tensor(targets, dtype=torch.long)

