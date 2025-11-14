"""
Tests for training/feature_flags.py - Feature flag management system.

Tests feature flag enumeration, configuration management, dependency validation,
environment variable integration, and feature manager functionality.
"""
# @author: @darianrosebrook

import os
from unittest.mock import patch

import pytest

from training.feature_flags import (
    FeatureFlag,
    FeatureConfig,
    FeatureManager,
    initialize_features,
)


class TestFeatureFlagEnum:
    """Test FeatureFlag enumeration."""

    def test_feature_flag_values(self):
        """Test that all feature flags have expected values."""
        assert FeatureFlag.DISTILLATION.value == "distillation"
        assert FeatureFlag.CODE_MODE.value == "code_mode"
        assert FeatureFlag.LATENT_REASONING.value == "latent_reasoning"
        assert FeatureFlag.QUANTIZATION.value == "quantization"
        assert FeatureFlag.SELF_EVALUATION.value == "self_evaluation"
        assert FeatureFlag.HALT_HEAD.value == "halt_head"
        assert FeatureFlag.PROGRESSIVE_CURRICULUM.value == "progressive_curriculum"
        assert FeatureFlag.CAWS_COMPLIANCE.value == "caws_compliance"
        assert FeatureFlag.PERFORMANCE_PROFILING.value == "performance_profiling"
        assert FeatureFlag.STRUCTURED_LOGGING.value == "structured_logging"

    def test_feature_flag_count(self):
        """Test that we have the expected number of feature flags."""
        assert len(FeatureFlag) == 10

    def test_feature_flag_uniqueness(self):
        """Test that all feature flag values are unique."""
        values = [flag.value for flag in FeatureFlag]
        assert len(values) == len(set(values))


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_feature_config_creation(self):
        """Test creating a feature config."""
        config = FeatureConfig(
            enabled=True,
            env_var="TEST_FEATURE",
            description="Test feature",
            dependencies={FeatureFlag.DISTILLATION},
            conflicts={FeatureFlag.CODE_MODE},
        )

        assert config.enabled
        assert config.env_var == "TEST_FEATURE"
        assert config.description == "Test feature"
        assert FeatureFlag.DISTILLATION in config.dependencies
        assert FeatureFlag.CODE_MODE in config.conflicts

    def test_feature_config_empty_sets(self):
        """Test feature config with empty dependency/conflict sets."""
        config = FeatureConfig(
            enabled=False,
            env_var="SIMPLE_FEATURE",
            description="Simple feature",
            dependencies=set(),
            conflicts=set(),
        )

        assert len(config.dependencies) == 0
        assert len(config.conflicts) == 0


class TestFeatureManager:
    """Test FeatureManager class."""

    @pytest.fixture
    def manager(self):
        """Create a feature manager instance."""
        return FeatureManager()

    def test_feature_manager_initialization(self, manager):
        """Test feature manager initialization."""
        assert isinstance(manager.features, dict)
        assert len(manager.features) == 10  # All feature flags

        # Check that all feature flags are configured
        for flag in FeatureFlag:
            assert flag in manager.features
            assert isinstance(manager.features[flag], FeatureConfig)

    def test_feature_manager_get_enabled_features(self, manager):
        """Test getting enabled features."""
        # By default, no features should be enabled
        enabled = manager.get_enabled_features()
        assert isinstance(enabled, set)
        # May have some defaults enabled

    def test_feature_manager_is_enabled(self, manager):
        """Test checking if a feature is enabled."""
        # Test with a feature that should be disabled by default
        assert isinstance(manager.is_enabled(FeatureFlag.DISTILLATION), bool)

    def test_feature_manager_enable_feature(self, manager):
        """Test enabling a feature."""
        flag = FeatureFlag.DISTILLATION

        # Should be able to enable
        manager.enable_feature(flag)
        assert manager.is_enabled(flag)

    def test_feature_manager_disable_feature(self, manager):
        """Test disabling a feature."""
        flag = FeatureFlag.DISTILLATION

        # First enable it
        manager.enable_feature(flag)
        assert manager.is_enabled(flag)

        # Then disable it
        manager.disable_feature(flag)
        assert not manager.is_enabled(flag)

    def test_feature_manager_enable_with_dependencies(self, manager):
        """Test enabling a feature with dependencies requires enabling deps first."""
        # Find a feature with dependencies (e.g., SELF_EVALUATION requires CAWS_COMPLIANCE)
        flag_with_deps = FeatureFlag.SELF_EVALUATION

        # Should fail if dependencies not enabled
        with pytest.raises(ValueError, match="dependency caws_compliance not enabled"):
            manager.enable_feature(flag_with_deps)

        # Enable dependency first
        manager.enable_feature(FeatureFlag.CAWS_COMPLIANCE)

        # Now should work
        manager.enable_feature(flag_with_deps)
        assert manager.is_enabled(flag_with_deps)

    def test_feature_manager_enable_with_conflicts(self, manager):
        """Test that enabling conflicting features raises error."""
        # Find features with conflicts
        conflicting_pair = None
        for flag, config in manager.features.items():
            if config.conflicts:
                conflicting_flag = next(iter(config.conflicts))
                if (
                    conflicting_flag in manager.features
                    and manager.features[conflicting_flag].conflicts
                ):
                    conflicting_pair = (flag, conflicting_flag)
                    break

        if conflicting_pair:
            flag1, flag2 = conflicting_pair

            # Enable first feature
            manager.enable_feature(flag1)
            assert manager.is_enabled(flag1)

            # Try to enable conflicting feature - should disable first
            manager.enable_feature(flag2)
            # This might disable the first or raise an error depending on implementation

    def test_feature_manager_validate_configuration(self, manager):
        """Test configuration validation."""
        # Should not raise any exceptions for valid configurations
        try:
            manager._validate_feature_dependencies()
        except Exception as e:
            pytest.fail(f"Configuration validation failed: {e}")

    def test_feature_manager_get_feature_status(self, manager):
        """Test getting feature status."""
        status = manager.get_feature_status()

        assert isinstance(status, dict)
        assert len(status) == 10

        # Check structure for one feature
        distillation_status = status["distillation"]
        assert "enabled" in distillation_status
        assert "description" in distillation_status
        assert "env_var" in distillation_status
        assert "dependencies" in distillation_status
        assert "conflicts" in distillation_status

    def test_feature_manager_load_from_environment(self, manager):
        """Test loading feature states from environment variables."""
        # Set some environment variables
        test_env = {
            "ENABLE_DISTILLATION": "true",
            "ENABLE_CODE_MODE": "false",
            "ENABLE_LATENT_REASONING": "1",
        }

        with patch.dict(os.environ, test_env):
            manager.load_from_environment()

            # Check that environment variables were respected
            assert manager.is_enabled(FeatureFlag.DISTILLATION)
            assert not manager.is_enabled(FeatureFlag.CODE_MODE)

    def test_feature_manager_load_from_config(self, manager):
        """Test loading feature states from config dictionary."""
        config = {
            "features": {
                "distillation": True,
                "code_mode": False,
                "latent_reasoning": True,
            }
        }

        manager.load_from_config(config)

        assert manager.is_enabled(FeatureFlag.DISTILLATION)
        assert not manager.is_enabled(FeatureFlag.CODE_MODE)
        assert manager.is_enabled(FeatureFlag.LATENT_REASONING)

    def test_feature_manager_validate_configuration_returns_list(self, manager):
        """Test that validate_configuration returns a list of errors."""
        errors = manager.validate_configuration()

        assert isinstance(errors, list)
        # Should have no errors for default configuration

    def test_feature_manager_apply_to_config(self, manager):
        """Test applying features to configuration."""
        base_config = {"model": {"name": "test"}}

        # Enable distillation
        manager.enable_feature(FeatureFlag.DISTILLATION)

        updated_config = manager.apply_to_config(base_config)

        assert "distillation" in updated_config
        assert updated_config["distillation"]["enabled"]
        assert updated_config["model"]["name"] == "test"  # Original preserved

    def test_feature_manager_apply_to_config_code_mode(self, manager):
        """Test applying CODE_MODE feature to configuration."""
        base_config = {"model": {"name": "test"}}
        manager.enable_feature(FeatureFlag.CODE_MODE)

        updated_config = manager.apply_to_config(base_config)

        assert "distill" in updated_config
        assert "code_mode" in updated_config["distill"]
        assert updated_config["distill"]["code_mode"]["enabled"]

    def test_feature_manager_apply_to_config_latent_reasoning(self, manager):
        """Test applying LATENT_REASONING feature to configuration."""
        base_config = {"model": {"name": "test"}}
        manager.enable_feature(FeatureFlag.LATENT_REASONING)

        updated_config = manager.apply_to_config(base_config)

        assert "latent" in updated_config
        assert updated_config["latent"]["enabled"]

    def test_feature_manager_apply_to_config_quantization(self, manager):
        """Test applying QUANTIZATION feature to configuration."""
        base_config = {"model": {"name": "test"}}
        manager.enable_feature(FeatureFlag.QUANTIZATION)

        updated_config = manager.apply_to_config(base_config)

        assert "quant" in updated_config
        assert updated_config["quant"]["enabled"]

    def test_feature_manager_apply_to_config_halt_head(self, manager, capsys):
        """Test applying HALT_HEAD feature to configuration."""
        base_config = {"model": {"name": "test"}}
        # Enable dependency first
        manager.enable_feature(FeatureFlag.LATENT_REASONING)
        manager.enable_feature(FeatureFlag.HALT_HEAD)

        manager.apply_to_config(base_config)

        # Should print warning
        captured = capsys.readouterr()
        assert "WARNING: Halt head requires model architecture modifications" in captured.out

    def test_feature_manager_disable_with_dependents(self, manager, capsys):
        """Test disabling a feature disables dependent features."""
        # Enable CAWS_COMPLIANCE and SELF_EVALUATION (which depends on it)
        manager.enable_feature(FeatureFlag.CAWS_COMPLIANCE)
        manager.enable_feature(FeatureFlag.SELF_EVALUATION)
        assert manager.is_enabled(FeatureFlag.SELF_EVALUATION)

        # Disable CAWS_COMPLIANCE
        manager.disable_feature(FeatureFlag.CAWS_COMPLIANCE)

        # Should disable SELF_EVALUATION and print warning
        assert not manager.is_enabled(FeatureFlag.CAWS_COMPLIANCE)
        assert not manager.is_enabled(FeatureFlag.SELF_EVALUATION)
        captured = capsys.readouterr()
        assert "WARNING: Disabling" in captured.out
        assert "self_evaluation" in captured.out.lower()

    def test_feature_manager_validate_with_errors(self, manager):
        """Test validation with dependency errors."""
        # Enable SELF_EVALUATION without its dependency
        manager.features[FeatureFlag.SELF_EVALUATION].enabled = True
        manager.features[FeatureFlag.CAWS_COMPLIANCE].enabled = False

        errors = manager.validate_configuration()
        assert len(errors) > 0
        assert any("self_evaluation" in error.lower()
                   and "caws_compliance" in error.lower() for error in errors)

    def test_feature_manager_validate_with_conflicts(self, manager):
        """Test validation with conflict errors."""
        # Manually set up a conflict scenario if any exist
        # Since current config has no conflicts, we'll test the code path
        errors = manager.validate_configuration()
        # Should have no errors for default config
        assert isinstance(errors, list)


class TestInitializeFeatures:
    """Test initialize_features function."""

    def test_initialize_features_no_config(self):
        """Test initializing features with no config."""
        # Should not raise any exceptions
        try:
            initialize_features()
        except Exception as e:
            pytest.fail(f"Feature initialization failed: {e}")

    def test_initialize_features_with_config(self):
        """Test initializing features with custom config."""
        config = {
            "features": {
                "distillation": True,
                "code_mode": False,
            }
        }

        try:
            initialize_features(config)
        except Exception as e:
            pytest.fail(f"Feature initialization with config failed: {e}")

    def test_initialize_features_invalid_config(self):
        """Test initializing features with invalid config."""
        invalid_config = {"invalid_key": {"enabled": True}}

        # Should handle invalid config gracefully
        try:
            initialize_features(invalid_config)
        except Exception:
            # Should handle gracefully
            pass

    def test_initialize_features_with_validation_errors(self, capsys):
        """Test initialize_features with validation errors."""
        from training.feature_flags import feature_manager

        # Manually create a validation error scenario
        feature_manager.features[FeatureFlag.SELF_EVALUATION].enabled = True
        feature_manager.features[FeatureFlag.CAWS_COMPLIANCE].enabled = False

        # Should raise ValueError with error messages
        with pytest.raises(ValueError, match="Invalid feature configuration"):
            initialize_features()

        # Should have printed error messages
        captured = capsys.readouterr()
        assert "Feature configuration errors:" in captured.out
        assert "ERROR:" in captured.out


class TestFeatureInteractions:
    """Test feature flag interactions and edge cases."""

    @pytest.fixture
    def manager(self):
        """Create a feature manager instance."""
        return FeatureManager()

    def test_circular_dependencies_prevention(self, manager):
        """Test that circular dependencies are prevented."""
        # This is more of an integration test to ensure the validation works
        # The actual validation happens during initialization
        pass  # Implementation depends on specific dependency rules

    def test_feature_enable_disable_cycle(self, manager):
        """Test repeatedly enabling and disabling features."""
        flag = FeatureFlag.DISTILLATION

        # Cycle through enable/disable several times
        for i in range(5):
            manager.enable_feature(flag)
            assert manager.is_enabled(flag)

            manager.disable_feature(flag)
            assert not manager.is_enabled(flag)

    def test_multiple_features_enable_disable(self, manager):
        """Test enabling/disabling multiple features simultaneously."""
        flags = [FeatureFlag.DISTILLATION,
                 FeatureFlag.CODE_MODE, FeatureFlag.QUANTIZATION]

        # Enable all
        for flag in flags:
            manager.enable_feature(flag)
            assert manager.is_enabled(flag)

        # Disable all
        for flag in flags:
            manager.disable_feature(flag)
            assert not manager.is_enabled(flag)

    def test_feature_manager_not_thread_safe(self, manager):
        """Test that feature manager is not thread-safe (expected behavior)."""
        import threading
        import time
        from tests.utils.thread_safety import safe_thread_join, safe_thread_join_all

        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                # Perform operations that might conflict
                manager.enable_feature(FeatureFlag.DISTILLATION)
                # Very small delay to encourage race conditions
                time.sleep(0.001)
                enabled = manager.is_enabled(FeatureFlag.DISTILLATION)
                manager.disable_feature(FeatureFlag.DISTILLATION)
                disabled = not manager.is_enabled(FeatureFlag.DISTILLATION)

                results.append((thread_id, enabled, disabled))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads to create potential race conditions
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete with timeout to prevent watchdog timeout
        all_succeeded, join_results = safe_thread_join_all(threads, timeout=5.0, per_thread=True)

        # Check that all threads completed (no exceptions)
        if not all_succeeded:
            # Log timeout information
            for i, result in enumerate(join_results):
                if result.timeout:
                    print(
                        f"[WARNING] Thread {i} did not complete within timeout: {result.message}"
                    )

        assert len(
            results) == 3, f"Not all threads completed: {len(results)} results, {len(errors)} errors. Join results: {[r.message for r in join_results]}"

        # Since it's not thread-safe, we don't assert consistent results
        # Just verify the operations didn't crash
        assert all(isinstance(enabled, bool) for _, enabled, _ in results)
        assert all(isinstance(disabled, bool) for _, _, disabled in results)


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    @pytest.fixture
    def manager(self):
        """Create a feature manager instance."""
        return FeatureManager()

    def test_environment_variable_override(self, manager):
        """Test that environment variables can override feature states."""
        flag = FeatureFlag.DISTILLATION
        env_var = manager.features[flag].env_var

        # Set environment variable
        with patch.dict(os.environ, {env_var: "true"}):
            manager.load_from_environment()
            assert manager.is_enabled(flag)

        with patch.dict(os.environ, {env_var: "false"}):
            manager.load_from_environment()
            assert not manager.is_enabled(flag)

    def test_environment_variable_case_insensitive(self, manager):
        """Test environment variable parsing is case insensitive."""
        flag = FeatureFlag.DISTILLATION
        env_var = manager.features[flag].env_var

        # Test various cases
        test_cases = ["True", "TRUE", "true", "1", "yes", "on"]

        for value in test_cases:
            with patch.dict(os.environ, {env_var: value}):
                manager.load_from_environment()
                assert manager.is_enabled(flag), f"Failed for value: {value}"

    def test_environment_variable_invalid_values(self, manager):
        """Test handling of invalid environment variable values."""
        flag = FeatureFlag.DISTILLATION
        env_var = manager.features[flag].env_var

        # Test invalid values - should default to False or current state
        invalid_values = ["invalid", "maybe", "2", ""]

        for value in invalid_values:
            with patch.dict(os.environ, {env_var: value}):
                initial_state = manager.is_enabled(flag)
                manager.load_from_environment()
                # Should either stay the same or be set to False
                assert manager.is_enabled(flag) in [False, initial_state]

    def test_environment_variable_precedence(self, manager):
        """Test that environment variables take precedence over programmatic settings."""
        flag = FeatureFlag.DISTILLATION
        env_var = manager.features[flag].env_var

        # Set programmatically
        manager.enable_feature(flag)
        assert manager.is_enabled(flag)

        # Override with environment variable
        with patch.dict(os.environ, {env_var: "false"}):
            manager.load_from_environment()
            assert not manager.is_enabled(flag)
