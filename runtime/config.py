"""
Runtime configuration system for coordinating inference features.

Centralizes configuration for:
- Latent mode settings
- Halt head parameters
- CAWS budget enforcement
- Curriculum settings
- Efficiency tracking
"""

import os
from typing import Dict, Any, Union
from pathlib import Path
import json

from runtime.orchestration.refine import CAWSBudgetTier


class RuntimeConfig:
    """
    Runtime configuration that coordinates all inference features.

    This replaces scattered environment variables with a centralized config.
    """

    def __init__(
        self,
        # Core features
        latent_mode_enabled: bool = False,
        halt_head_enabled: bool = False,
        # CAWS budget settings
        caws_tier: CAWSBudgetTier = CAWSBudgetTier.TIER_2,
        max_refinement_loops: int = 5,
        # Halt head parameters
        judge_score_threshold: float = 0.8,
        halt_probability_threshold: float = 0.7,
        # Latent mode parameters
        curriculum_probability: float = 1.0,
        curriculum_slots: int = 1,
        max_latent_spans: int = 10,
        max_latent_length: int = 100,
        # Generation parameters
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        # Efficiency tracking
        enable_efficiency_tracking: bool = True,
    ):
        self.latent_mode_enabled = latent_mode_enabled
        self.halt_head_enabled = halt_head_enabled
        self.caws_tier = caws_tier
        self.max_refinement_loops = max_refinement_loops
        self.judge_score_threshold = judge_score_threshold
        self.halt_probability_threshold = halt_probability_threshold
        self.curriculum_probability = curriculum_probability
        self.curriculum_slots = curriculum_slots
        self.max_latent_spans = max_latent_spans
        self.max_latent_length = max_latent_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_efficiency_tracking = enable_efficiency_tracking

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Create config from environment variables."""
        return cls(
            latent_mode_enabled=os.getenv("LATENT_MODE", "0") == "1",
            halt_head_enabled=os.getenv("HALT_HEAD", "0") == "1",
            caws_tier=cls._parse_caws_tier(os.getenv("CAWS_TIER", "tier_2")),
            max_refinement_loops=int(os.getenv("MAX_REFINEMENT_LOOPS", "5")),
            judge_score_threshold=float(os.getenv("JUDGE_THRESHOLD", "0.8")),
            halt_probability_threshold=float(os.getenv("HALT_THRESHOLD", "0.7")),
            curriculum_probability=float(os.getenv("CURRICULUM_P", "1.0")),
            curriculum_slots=int(os.getenv("CURRICULUM_SLOTS", "1")),
            max_latent_spans=int(os.getenv("MAX_LATENT_SPANS", "10")),
            max_latent_length=int(os.getenv("MAX_LATENT_LENGTH", "100")),
            temperature=float(os.getenv("TEMPERATURE", "1.0")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
            enable_efficiency_tracking=os.getenv("EFFICIENCY_TRACKING", "1") == "1",
        )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "RuntimeConfig":
        """Load config from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(
            latent_mode_enabled=data.get("latent_mode_enabled", False),
            halt_head_enabled=data.get("halt_head_enabled", False),
            caws_tier=cls._parse_caws_tier(data.get("caws_tier", "tier_2")),
            max_refinement_loops=data.get("max_refinement_loops", 5),
            judge_score_threshold=data.get("judge_score_threshold", 0.8),
            halt_probability_threshold=data.get("halt_probability_threshold", 0.7),
            curriculum_probability=data.get("curriculum_probability", 1.0),
            curriculum_slots=data.get("curriculum_slots", 1),
            max_latent_spans=data.get("max_latent_spans", 10),
            max_latent_length=data.get("max_latent_length", 100),
            temperature=data.get("temperature", 1.0),
            max_new_tokens=data.get("max_new_tokens", 256),
            enable_efficiency_tracking=data.get("enable_efficiency_tracking", True),
        )

    @classmethod
    def _parse_caws_tier(cls, tier_str: str) -> CAWSBudgetTier:
        """Parse CAWS tier string to enum."""
        tier_map = {
            "tier_1": CAWSBudgetTier.TIER_1,
            "tier_2": CAWSBudgetTier.TIER_2,
            "tier_3": CAWSBudgetTier.TIER_3,
        }
        return tier_map.get(tier_str, CAWSBudgetTier.TIER_2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "latent_mode_enabled": self.latent_mode_enabled,
            "halt_head_enabled": self.halt_head_enabled,
            "caws_tier": self.caws_tier.value,
            "max_refinement_loops": self.max_refinement_loops,
            "judge_score_threshold": self.judge_score_threshold,
            "halt_probability_threshold": self.halt_probability_threshold,
            "curriculum_probability": self.curriculum_probability,
            "curriculum_slots": self.curriculum_slots,
            "max_latent_spans": self.max_latent_spans,
            "max_latent_length": self.max_latent_length,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "enable_efficiency_tracking": self.enable_efficiency_tracking,
        }

    def save(self, config_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "latent_mode": self.latent_mode_enabled,
            "halt_head": self.halt_head_enabled,
            "efficiency_tracking": self.enable_efficiency_tracking,
        }
        return feature_map.get(feature, False)

    def get_caws_limits(self) -> Dict[str, int]:
        """Get CAWS tier limits."""
        limits = {
            CAWSBudgetTier.TIER_1: {"max_loops": 1, "max_latent_spans": 0},
            CAWSBudgetTier.TIER_2: {"max_loops": 2, "max_latent_spans": 1},
            CAWSBudgetTier.TIER_3: {"max_loops": 3, "max_latent_spans": 3},
        }
        return limits.get(self.caws_tier, limits[CAWSBudgetTier.TIER_2])

    def __str__(self) -> str:
        """String representation of config."""
        features = []
        if self.latent_mode_enabled:
            features.append(f"latent(c={self.curriculum_slots}, p={self.curriculum_probability})")
        if self.halt_head_enabled:
            features.append(
                f"halt(τ={self.judge_score_threshold}, p={self.halt_probability_threshold})"
            )

        feature_str = ", ".join(features) if features else "standard"
        return f"RuntimeConfig({feature_str}, tier={self.caws_tier.value}, loops≤{self.max_refinement_loops})"
