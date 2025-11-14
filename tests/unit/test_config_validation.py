"""
Tests for training/config_validation.py - Configuration validation and schema enforcement.

Tests JSON schema validation, config file validation, default config creation,
and configuration merging functionality.
"""
# @author: @darianrosebrook

import tempfile
from pathlib import Path

import pytest
import yaml

from training.config_validation import (
    validate_training_config,
    validate_config_file,
    create_default_config,
    save_config_template,
    merge_configs,
    TRAINING_CONFIG_SCHEMA,
)


class TestConfigSchemaValidation:
    """Test JSON schema validation functionality."""

    def test_training_config_schema_structure(self):
        """Test that the schema has the expected structure."""
        assert "type" in TRAINING_CONFIG_SCHEMA
        assert TRAINING_CONFIG_SCHEMA["type"] == "object"
        assert "properties" in TRAINING_CONFIG_SCHEMA

        # Check required sections
        properties = TRAINING_CONFIG_SCHEMA["properties"]
        assert "model" in properties
        assert "training" in properties
        assert "optimizer" in properties
        assert "distillation" in properties

    def test_validate_training_config_valid(self):
        """Test validation of valid training configuration."""
        valid_config = {
            "model": {
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "vocab_size": 1000,
                "rope_theta": 10000.0,
                "rope_scaling": "dynamic",
                "dropout": 0.1,
            },
            "training": {
                "steps": 1000,
                "batch_size": 4,
                "seq_length": 512,
                "grad_accum_steps": 1,
                "lr": 1e-4,
                "warmup_steps": 100,
                "save_every": 500,
                "val_every": 100,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "distillation": {
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.7,
            },
        }

        errors = validate_training_config(valid_config)
        assert len(errors) == 0

    def test_validate_training_config_missing_required(self):
        """Test validation fails with missing required fields."""
        invalid_config = {
            "model": {
                # Missing required fields
            },
            "training": {
                # Missing required fields
            },
        }

        errors = validate_training_config(invalid_config)
        assert len(errors) > 0
        assert any("d_model" in error.lower() or "n_layers" in error.lower() for error in errors)

    def test_validate_training_config_invalid_types(self):
        """Test validation fails with invalid data types."""
        invalid_config = {
            "model": {
                "d_model": -1,  # Invalid negative value
                "n_layers": 2,
                "n_heads": 4,
                "vocab_size": 1000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 4,
            },
        }

        errors = validate_training_config(invalid_config)
        assert len(errors) > 0
        assert any("minimum" in error or "d_model" in error for error in errors)

    def test_validate_training_config_invalid_enum(self):
        """Test validation fails with invalid enum values."""
        invalid_config = {
            "model": {
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "vocab_size": 1000,
                "rope_scaling": "invalid_scaling",  # Invalid enum value
            },
            "training": {
                "steps": 1000,
                "batch_size": 4,
            },
        }

        errors = validate_training_config(invalid_config)
        assert len(errors) > 0

    def test_validate_config_file_valid_yaml(self):
        """Test validation of valid YAML config file."""
        config_content = """
        model:
          d_model: 128
          n_layers: 2
          n_heads: 4
          vocab_size: 1000
        training:
          steps: 1000
          batch_size: 4
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            errors = validate_config_file(config_path)
            assert len(errors) == 0
        finally:
            config_path.unlink()

    def test_validate_config_file_invalid_yaml(self):
        """Test validation of invalid YAML config file."""
        config_content = """
        model:
          d_model: -1  # Invalid negative value
        training:
          steps: 1000
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            errors = validate_config_file(config_path)
            assert len(errors) > 0
        finally:
            config_path.unlink()

    def test_validate_config_file_nonexistent(self):
        """Test validation of nonexistent config file."""
        nonexistent_path = Path("nonexistent_config.yaml")

        with pytest.raises(FileNotFoundError):
            validate_config_file(nonexistent_path)


class TestDefaultConfigCreation:
    """Test default configuration creation."""

    def test_create_default_config_structure(self):
        """Test that default config has expected structure."""
        config = create_default_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config
        assert "optimizer" in config
        assert "distillation" in config

        # Check model section
        model_config = config["model"]
        assert "d_model" in model_config
        assert "n_layers" in model_config
        assert "vocab_size" in model_config

        # Check training section
        training_config = config["training"]
        assert "steps" in training_config
        assert "batch_size" in training_config
        assert "lr" in training_config

    def test_create_default_config_values(self):
        """Test that default config has reasonable default values."""
        config = create_default_config()

        # Model defaults
        assert config["model"]["d_model"] > 0
        assert config["model"]["n_layers"] > 0
        assert config["model"]["vocab_size"] > 0

        # Training defaults
        assert config["training"]["steps"] > 0
        assert config["training"]["batch_size"] > 0
        assert config["training"]["lr"] > 0

        # Optimizer defaults
        assert "type" in config["optimizer"]
        assert config["optimizer"]["lr"] > 0

    def test_save_config_template(self, tmp_path):
        """Test saving config template to file."""
        template_path = tmp_path / "config_template.yaml"

        save_config_template(template_path)

        assert template_path.exists()

        # Verify it's valid YAML and contains expected structure
        with open(template_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        assert "model" in loaded_config
        assert "training" in loaded_config
        assert "optimizer" in loaded_config


class TestConfigMerging:
    """Test configuration merging functionality."""

    def test_merge_configs_basic(self):
        """Test basic config merging."""
        config1_content = """
        model:
          d_model: 128
          vocab_size: 1000
        """
        config2_content = """
        training:
          steps: 1000
          batch_size: 4
        """
        config3_content = """
        optimizer:
          lr: 0.0001
        """

        files = []
        try:
            for content in [config1_content, config2_content, config3_content]:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    f.write(content)
                    files.append(f.name)

            merged = merge_configs(files)

            assert merged["model"]["d_model"] == 128
            assert merged["training"]["steps"] == 1000
            assert merged["optimizer"]["lr"] == 0.0001
        finally:
            for f in files:
                Path(f).unlink()

    def test_merge_configs_overrides(self):
        """Test that later configs override earlier ones."""
        config1_content = """
        model:
          d_model: 128
        """
        config2_content = """
        model:
          d_model: 256
          n_layers: 4
        """

        files = []
        try:
            for content in [config1_content, config2_content]:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    f.write(content)
                    files.append(f.name)

            merged = merge_configs(files)

            assert merged["model"]["d_model"] == 256  # Overridden
            assert merged["model"]["n_layers"] == 4  # Added
        finally:
            for f in files:
                Path(f).unlink()

    def test_merge_configs_empty_list(self):
        """Test merging with empty config list."""
        merged = merge_configs([])
        assert merged == {}

    def test_merge_configs_invalid_file(self):
        """Test merging with invalid config file."""
        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            invalid_file = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                merge_configs([invalid_file])
        finally:
            Path(invalid_file).unlink()


class TestDeepMerge:
    """Test deep merge functionality."""

    def test_deep_merge_nested_dicts(self):
        """Test deep merging of nested dictionaries."""
        from training.config_validation import _deep_merge

        base = {"model": {"d_model": 128, "nested": {"a": 1, "b": 2}}, "training": {"steps": 100}}

        update = {
            "model": {
                "n_layers": 4,
                "nested": {"b": 3, "c": 4},  # Override b, add c
            },
            "optimizer": {"lr": 1e-4},
        }

        merged = _deep_merge(base, update)

        assert merged["model"]["d_model"] == 128  # Preserved
        assert merged["model"]["n_layers"] == 4  # Added
        assert merged["model"]["nested"]["a"] == 1  # Preserved
        assert merged["model"]["nested"]["b"] == 3  # Overridden
        assert merged["model"]["nested"]["c"] == 4  # Added
        assert merged["training"]["steps"] == 100  # Preserved
        assert merged["optimizer"]["lr"] == 1e-4  # Added

    def test_deep_merge_non_dict_values(self):
        """Test deep merge with non-dict values."""
        from training.config_validation import _deep_merge

        base = {"value": 1, "list": [1, 2]}
        update = {"value": 2, "list": [3, 4]}

        merged = _deep_merge(base, update)

        assert merged["value"] == 2  # Overridden
        assert merged["list"] == [3, 4]  # Overridden (not merged)


class TestConfigFileValidationEdgeCases:
    """Test edge cases and error paths in config file validation."""

    def test_validate_config_file_json_format(self, tmp_path):
        """Test validate_config_file with JSON format file."""
        import json
        config_file = tmp_path / "config.json"
        valid_config = create_default_config()
        with open(config_file, "w") as f:
            json.dump(valid_config, f)

        errors = validate_config_file(config_file)
        assert errors == []

    def test_validate_config_file_yaml_parse_error(self, tmp_path):
        """Test validate_config_file with invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "Failed to parse configuration file" in errors[0]

    def test_validate_config_file_json_parse_error(self, tmp_path):
        """Test validate_config_file with invalid JSON."""
        import json
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write('{"invalid": json}')

        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "Failed to parse configuration file" in errors[0]

    def test_validate_config_file_unexpected_error(self, tmp_path, monkeypatch):
        """Test validate_config_file with unexpected error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("valid: yaml")

        # Mock open to raise an unexpected error
        def mock_open(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        monkeypatch.setattr("builtins.open", mock_open)

        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "Unexpected error validating configuration" in errors[0]

    def test_validate_config_file_with_validation_errors(self, tmp_path):
        """Test validate_config_file with config that has validation errors."""
        config_file = tmp_path / "invalid_config.yaml"
        invalid_config = {"invalid": "config", "missing_required": True}
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        errors = validate_config_file(config_file)
        assert len(errors) > 0  # Should have validation errors


class TestSaveConfigTemplate:
    """Test save_config_template function."""

    def test_save_config_template_yaml(self, tmp_path):
        """Test save_config_template saves YAML format."""
        template_path = tmp_path / "template.yaml"
        save_config_template(template_path)

        assert template_path.exists()
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
        assert "model" in config
        assert "training" in config

    def test_save_config_template_json(self, tmp_path):
        """Test save_config_template saves JSON format."""
        import json
        template_path = tmp_path / "template.json"
        save_config_template(template_path)

        assert template_path.exists()
        with open(template_path, "r") as f:
            config = json.load(f)
        assert "model" in config
        assert "training" in config


class TestMergeConfigsEdgeCases:
    """Test edge cases in merge_configs function."""

    def test_merge_configs_file_not_found(self):
        """Test merge_configs raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            merge_configs(["nonexistent.yaml"])

    def test_merge_configs_json_format(self, tmp_path):
        """Test merge_configs with JSON format files."""
        import json
        config1 = tmp_path / "config1.json"
        config2 = tmp_path / "config2.json"

        with open(config1, "w") as f:
            json.dump({"model": {"d_model": 128}}, f)
        with open(config2, "w") as f:
            json.dump({"model": {"n_layers": 4}}, f)

        merged = merge_configs([str(config1), str(config2)])
        assert merged["model"]["d_model"] == 128
        assert merged["model"]["n_layers"] == 4

    def test_merge_configs_validate_final_valid(self, tmp_path):
        """Test merge_configs with validate_final=True and valid config."""
        config1 = tmp_path / "config1.yaml"
        config2 = tmp_path / "config2.yaml"

        base_config = create_default_config()
        with open(config1, "w") as f:
            yaml.dump(base_config, f)

        update_config = {"training": {"steps": 500}}
        with open(config2, "w") as f:
            yaml.dump(update_config, f)

        merged = merge_configs([str(config1), str(config2)], validate_final=True)
        assert merged["training"]["steps"] == 500

    def test_merge_configs_validate_final_invalid(self, tmp_path):
        """Test merge_configs with validate_final=True and invalid config."""
        config_file = tmp_path / "invalid.yaml"
        invalid_config = {"invalid": "config"}
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match="Invalid merged configuration"):
            merge_configs([str(config_file)], validate_final=True)
