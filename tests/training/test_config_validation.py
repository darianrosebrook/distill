"""
Tests for training/config_validation.py - Configuration validation and schema enforcement.

Tests JSON schema validation, configuration file validation, default config creation,
and configuration merging.
"""
# @author: @darianrosebrook

import json
import pytest
import yaml
from pathlib import Path
from training.config_validation import (
    validate_training_config,
    validate_config_file,
    create_default_config,
    save_config_template,
    merge_configs,
    TRAINING_CONFIG_SCHEMA,
)


class TestValidateTrainingConfig:
    """Test validate_training_config function."""

    def test_validate_training_config_valid(self):
        """Test validating valid training configuration."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert errors == []

    def test_validate_training_config_missing_required_model(self):
        """Test validating config missing required model fields."""
        config = {
            "model": {
                "d_model": 512,
                # Missing n_layers, n_heads, vocab_size
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0
        assert any("n_layers" in error.lower() or "required" in error.lower() for error in errors)

    def test_validate_training_config_missing_required_training(self):
        """Test validating config missing required training fields."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                # Missing steps, batch_size
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0

    def test_validate_training_config_invalid_model_type(self):
        """Test validating config with invalid model field type."""
        config = {
            "model": {
                "d_model": "invalid",  # Should be integer
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0

    def test_validate_training_config_invalid_training_type(self):
        """Test validating config with invalid training field type."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": "invalid",  # Should be integer
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0

    def test_validate_training_config_invalid_enum(self):
        """Test validating config with invalid enum value."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
                "rope_scaling": "invalid",  # Should be "dynamic", "linear", or "yarn"
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0

    def test_validate_training_config_invalid_range(self):
        """Test validating config with value outside valid range."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
                "dropout": 1.5,  # Should be between 0 and 1
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        errors = validate_training_config(config)
        assert len(errors) > 0

    def test_validate_training_config_optional_fields(self):
        """Test validating config with optional fields."""
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 2e-4,
            },
            "distillation": {
                "enabled": True,
                "temperature": 2.0,
            },
        }
        errors = validate_training_config(config)
        assert errors == []


class TestValidateConfigFile:
    """Test validate_config_file function."""

    def test_validate_config_file_valid_yaml(self, tmp_path):
        """Test validating valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        errors = validate_config_file(config_file)
        assert errors == []

    def test_validate_config_file_valid_json(self, tmp_path):
        """Test validating valid JSON config file."""
        config_file = tmp_path / "config.json"
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config_file, "w") as f:
            json.dump(config, f)

        errors = validate_config_file(config_file)
        assert errors == []

    def test_validate_config_file_not_found(self, tmp_path):
        """Test validating non-existent config file."""
        config_file = tmp_path / "nonexistent.yaml"
        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "not found" in errors[0].lower()

    def test_validate_config_file_invalid_yaml(self, tmp_path):
        """Test validating invalid YAML config file."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "parse" in errors[0].lower() or "yaml" in errors[0].lower()

    def test_validate_config_file_invalid_json(self, tmp_path):
        """Test validating invalid JSON config file."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write('{"invalid": json}')  # Missing quotes around json

        errors = validate_config_file(config_file)
        assert len(errors) > 0
        assert "parse" in errors[0].lower() or "json" in errors[0].lower()

    def test_validate_config_file_invalid_schema(self, tmp_path):
        """Test validating config file with invalid schema."""
        config_file = tmp_path / "invalid_schema.yaml"
        config = {
            "model": {
                "d_model": "invalid",  # Should be integer
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        errors = validate_config_file(config_file)
        assert len(errors) > 0


class TestCreateDefaultConfig:
    """Test create_default_config function."""

    def test_create_default_config_structure(self):
        """Test that default config has correct structure."""
        config = create_default_config()

        assert "model" in config
        assert "training" in config
        assert isinstance(config["model"], dict)
        assert isinstance(config["training"], dict)

    def test_create_default_config_model_fields(self):
        """Test that default config has required model fields."""
        config = create_default_config()

        assert "d_model" in config["model"]
        assert "n_layers" in config["model"]
        assert "n_heads" in config["model"]
        assert "vocab_size" in config["model"]
        assert isinstance(config["model"]["d_model"], int)
        assert isinstance(config["model"]["n_layers"], int)

    def test_create_default_config_training_fields(self):
        """Test that default config has required training fields."""
        config = create_default_config()

        assert "steps" in config["training"]
        assert "batch_size" in config["training"]
        assert isinstance(config["training"]["steps"], int)
        assert isinstance(config["training"]["batch_size"], int)

    def test_create_default_config_validates(self):
        """Test that default config passes validation."""
        config = create_default_config()
        errors = validate_training_config(config)
        assert errors == []

    def test_create_default_config_optional_fields(self):
        """Test that default config includes optional fields."""
        config = create_default_config()

        assert "optimizer" in config
        assert "distillation" in config
        assert "latent" in config
        assert "quant" in config


class TestSaveConfigTemplate:
    """Test save_config_template function."""

    def test_save_config_template_yaml(self, tmp_path):
        """Test saving config template as YAML."""
        template_path = tmp_path / "template.yaml"
        save_config_template(template_path)

        assert template_path.exists()

        with open(template_path, "r") as f:
            config = yaml.safe_load(f)

        assert "model" in config
        assert "training" in config

    def test_save_config_template_json(self, tmp_path):
        """Test saving config template as JSON."""
        template_path = tmp_path / "template.json"
        save_config_template(template_path)

        assert template_path.exists()

        with open(template_path, "r") as f:
            config = json.load(f)

        assert "model" in config
        assert "training" in config

    def test_save_config_template_creates_directory(self, tmp_path):
        """Test that template creation creates parent directories."""
        template_path = tmp_path / "nested" / "deep" / "template.yaml"
        save_config_template(template_path)

        assert template_path.exists()
        assert template_path.parent.exists()

    def test_save_config_template_validates(self, tmp_path):
        """Test that saved template passes validation."""
        template_path = tmp_path / "template.yaml"
        save_config_template(template_path)

        errors = validate_config_file(template_path)
        assert errors == []


class TestMergeConfigs:
    """Test merge_configs function."""

    def test_merge_configs_single_file(self, tmp_path):
        """Test merging single config file."""
        config_file = tmp_path / "config1.yaml"
        config = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        merged = merge_configs([str(config_file)])
        assert merged["model"]["d_model"] == 512
        assert merged["training"]["steps"] == 1000

    def test_merge_configs_multiple_files(self, tmp_path):
        """Test merging multiple config files."""
        config1_file = tmp_path / "config1.yaml"
        config1 = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "vocab_size": 32000,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config1_file, "w") as f:
            yaml.dump(config1, f)

        config2_file = tmp_path / "config2.yaml"
        config2 = {
            "training": {
                "steps": 2000,  # Override
                "lr": 1e-4,  # New field
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
            },
        }
        with open(config2_file, "w") as f:
            yaml.dump(config2, f)

        merged = merge_configs([str(config1_file), str(config2_file)])
        assert merged["model"]["d_model"] == 512  # From config1
        assert merged["training"]["steps"] == 2000  # Overridden by config2
        assert merged["training"]["lr"] == 1e-4  # From config2
        assert merged["optimizer"]["type"] == "adamw"  # From config2

    def test_merge_configs_deep_merge(self, tmp_path):
        """Test that merging performs deep merge."""
        config1_file = tmp_path / "config1.yaml"
        config1 = {
            "model": {
                "d_model": 512,
                "n_layers": 8,
            },
            "training": {
                "steps": 1000,
                "batch_size": 8,
            },
        }
        with open(config1_file, "w") as f:
            yaml.dump(config1, f)

        config2_file = tmp_path / "config2.yaml"
        config2 = {
            "model": {
                "n_heads": 8,  # Add to existing model section
            },
            "training": {
                "steps": 2000,  # Override
            },
        }
        with open(config2_file, "w") as f:
            yaml.dump(config2, f)

        merged = merge_configs([str(config1_file), str(config2_file)])
        assert merged["model"]["d_model"] == 512  # From config1
        assert merged["model"]["n_layers"] == 8  # From config1
        assert merged["model"]["n_heads"] == 8  # From config2
        assert merged["training"]["steps"] == 2000  # Overridden by config2
        assert merged["training"]["batch_size"] == 8  # From config1

    def test_merge_configs_file_not_found(self, tmp_path):
        """Test merging with non-existent file."""
        with pytest.raises(ValueError, match="not found"):
            merge_configs([str(tmp_path / "nonexistent.yaml")])

    def test_merge_configs_invalid_file(self, tmp_path):
        """Test merging with invalid config file."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content")

        with pytest.raises(ValueError, match="Invalid configuration"):
            merge_configs([str(invalid_file)])

    def test_merge_configs_empty_list(self):
        """Test merging with empty file list."""
        with pytest.raises(ValueError):
            merge_configs([])

