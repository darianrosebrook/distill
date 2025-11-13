"""
Feature flags and configuration management.

Provides centralized feature flag management with environment variable support.
"""

import os
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
from enum import Enum


class FeatureFlag(Enum):
    """Available feature flags."""

    DISTILLATION = "distillation"
    CODE_MODE = "code_mode"
    LATENT_REASONING = "latent_reasoning"
    QUANTIZATION = "quantization"
    SELF_EVALUATION = "self_evaluation"
    HALT_HEAD = "halt_head"
    PROGRESSIVE_CURRICULUM = "progressive_curriculum"
    CAWS_COMPLIANCE = "caws_compliance"
    PERFORMANCE_PROFILING = "performance_profiling"
    STRUCTURED_LOGGING = "structured_logging"


@dataclass
class FeatureConfig:
    """Configuration for a feature flag."""

    enabled: bool
    env_var: str
    description: str
    dependencies: Set[FeatureFlag]
    conflicts: Set[FeatureFlag]


class FeatureManager:
    """Manages feature flags and their interactions."""

    def __init__(self):
        """Initialize feature manager with all available features."""
        self.features = self._initialize_features()
        self._validate_feature_dependencies()

    def _initialize_features(self) -> Dict[FeatureFlag, FeatureConfig]:
        """Initialize all feature configurations."""
        return {
            FeatureFlag.DISTILLATION: FeatureConfig(
                enabled=False,
                env_var="ENABLE_DISTILLATION",
                description="Knowledge distillation from teacher model",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.CODE_MODE: FeatureConfig(
                enabled=False,
                env_var="TRAIN_CODE_MODE",
                description="TypeScript API orchestration loss",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.LATENT_REASONING: FeatureConfig(
                enabled=False,
                env_var="TRAIN_LATENT",
                description="Latent reasoning with curriculum learning",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.QUANTIZATION: FeatureConfig(
                enabled=False,
                env_var="ENABLE_QUANTIZATION",
                description="Quantization-aware training",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.SELF_EVALUATION: FeatureConfig(
                enabled=False,
                env_var="ENABLE_SELF_EVALUATION",
                description="Self-evaluation training loop",
                dependencies={FeatureFlag.CAWS_COMPLIANCE},
                conflicts=set(),
            ),
            FeatureFlag.HALT_HEAD: FeatureConfig(
                enabled=False,
                env_var="ENABLE_HALT_HEAD",
                description="Learned halting head for inference",
                dependencies={FeatureFlag.LATENT_REASONING},
                conflicts=set(),
            ),
            FeatureFlag.PROGRESSIVE_CURRICULUM: FeatureConfig(
                enabled=False,
                env_var="ENABLE_PROGRESSIVE_CURRICULUM",
                description="Progressive curriculum learning",
                dependencies={FeatureFlag.LATENT_REASONING},
                conflicts=set(),
            ),
            FeatureFlag.CAWS_COMPLIANCE: FeatureConfig(
                enabled=False,
                env_var="ENABLE_CAWS_COMPLIANCE",
                description="CAWS compliance checking and enforcement",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.PERFORMANCE_PROFILING: FeatureConfig(
                enabled=False,
                env_var="ENABLE_PERFORMANCE_PROFILING",
                description="Detailed performance monitoring and profiling",
                dependencies=set(),
                conflicts=set(),
            ),
            FeatureFlag.STRUCTURED_LOGGING: FeatureConfig(
                enabled=False,
                env_var="ENABLE_STRUCTURED_LOGGING",
                description="JSON-structured logging output",
                dependencies=set(),
                conflicts=set(),
            ),
        }

    def _validate_feature_dependencies(self) -> None:
        """Validate that feature dependencies are properly configured."""
        for feature, config in self.features.items():
            # Check that dependencies exist
            for dep in config.dependencies:
                if dep not in self.features:
                    raise ValueError(
                        f"Feature {feature.value} depends on unknown feature {dep.value}"
                    )

            # Check for circular dependencies (basic check)
            for dep in config.dependencies:
                if feature in self.features[dep].dependencies:
                    raise ValueError(
                        f"Circular dependency detected between {feature.value} and {dep.value}"
                    )

    def load_from_environment(self) -> None:
        """Load feature flags from environment variables."""
        for feature, config in self.features.items():
            env_value = os.getenv(config.env_var, "").lower()
            if env_value in ("1", "true", "yes", "on"):
                config.enabled = True
            elif env_value in ("0", "false", "no", "off"):
                config.enabled = False
            # Otherwise keep default

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load feature flags from configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        feature_config = config.get("features", {})

        for feature, feature_cfg in self.features.items():
            if feature.value in feature_config:
                feature_cfg.enabled = bool(feature_config[feature.value])

    def enable_feature(self, feature: FeatureFlag) -> None:
        """Enable a specific feature flag.

        Args:
            feature: Feature flag to enable

        Raises:
            ValueError: If dependencies are not satisfied
        """
        if feature not in self.features:
            raise ValueError(f"Unknown feature: {feature.value}")

        # Check dependencies
        for dep in self.features[feature].dependencies:
            if not self.features[dep].enabled:
                raise ValueError(
                    f"Cannot enable {feature.value}: dependency {dep.value} not enabled"
                )

        # Check conflicts
        for conflict in self.features[feature].conflicts:
            if self.features[conflict].enabled:
                raise ValueError(f"Cannot enable {feature.value}: conflicts with {conflict.value}")

        self.features[feature].enabled = True

    def disable_feature(self, feature: FeatureFlag) -> None:
        """Disable a specific feature flag.

        Args:
            feature: Feature flag to disable
        """
        if feature in self.features:
            self.features[feature].enabled = False

            # Disable dependent features
            for other_feature, config in self.features.items():
                if feature in config.dependencies:
                    print(
                        f"WARNING: Disabling {other_feature.value} due to dependency on {feature.value}"
                    )
                    config.enabled = False

    def is_enabled(self, feature: FeatureFlag) -> bool:
        """Check if a feature is enabled.

        Args:
            feature: Feature flag to check

        Returns:
            True if feature is enabled
        """
        return self.features.get(feature, FeatureConfig(False, "", "", set(), set())).enabled

    def get_enabled_features(self) -> Set[FeatureFlag]:
        """Get all enabled features.

        Returns:
            Set of enabled feature flags
        """
        return {feature for feature, config in self.features.items() if config.enabled}

    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all features.

        Returns:
            Dictionary mapping feature names to their status
        """
        status = {}
        for feature, config in self.features.items():
            status[feature.value] = {
                "enabled": config.enabled,
                "description": config.description,
                "env_var": config.env_var,
                "dependencies": [f.value for f in config.dependencies],
                "conflicts": [f.value for f in config.conflicts],
            }
        return status

    def validate_configuration(self) -> List[str]:
        """Validate current feature configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for feature, config in self.features.items():
            if config.enabled:
                # Check dependencies
                for dep in config.dependencies:
                    if not self.features[dep].enabled:
                        errors.append(f"Feature {feature.value} requires {dep.value} to be enabled")

                # Check conflicts
                for conflict in config.conflicts:
                    if self.features[conflict].enabled:
                        errors.append(f"Feature {feature.value} conflicts with {conflict.value}")

        return errors

    def apply_to_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply feature flags to training configuration.

        Args:
            config: Base configuration dictionary

        Returns:
            Configuration with features applied
        """
        updated_config = config.copy()

        # Apply distillation settings
        if self.is_enabled(FeatureFlag.DISTILLATION):
            distillation_cfg = updated_config.setdefault("distillation", {})
            distillation_cfg["enabled"] = True

        # Apply code mode settings
        if self.is_enabled(FeatureFlag.CODE_MODE):
            distill_cfg = updated_config.setdefault("distill", {})
            code_mode_cfg = distill_cfg.setdefault("code_mode", {})
            code_mode_cfg["enabled"] = True

        # Apply latent reasoning settings
        if self.is_enabled(FeatureFlag.LATENT_REASONING):
            latent_cfg = updated_config.setdefault("latent", {})
            latent_cfg["enabled"] = True

        # Apply quantization settings
        if self.is_enabled(FeatureFlag.QUANTIZATION):
            quant_cfg = updated_config.setdefault("quant", {})
            quant_cfg["enabled"] = True

        # Apply halt head settings
        if self.is_enabled(FeatureFlag.HALT_HEAD):
            # This would require model architecture changes
            print("WARNING: Halt head requires model architecture modifications")

        return updated_config


# Global feature manager instance
feature_manager = FeatureManager()


def initialize_features(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize feature flags from environment and config.

    Args:
        config: Optional configuration dictionary
    """
    # Load from environment first
    feature_manager.load_from_environment()

    # Then override with config if provided
    if config:
        feature_manager.load_from_config(config)

    # Validate configuration
    errors = feature_manager.validate_configuration()
    if errors:
        print("Feature configuration errors:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise ValueError("Invalid feature configuration")

    # Log enabled features
    enabled = feature_manager.get_enabled_features()
    if enabled:
        print("Enabled features:")
        for feature in enabled:
            print(f"  ✅ {feature.value}")
    else:
        print("No features enabled")


if __name__ == "__main__":
    # Example usage
    initialize_features()

    status = feature_manager.get_feature_status()
    print("\nFeature status:")
    for name, info in status.items():
        status_str = "✅" if info["enabled"] else "❌"
        print(f"  {status_str} {name}: {info['description']}")
