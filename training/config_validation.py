"""
Configuration validation and schema enforcement.

Provides JSON schema validation for training configurations.
"""
import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List
import yaml


# JSON Schema for training configuration validation
TRAINING_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "d_model": {"type": "integer", "minimum": 1},
                "n_layers": {"type": "integer", "minimum": 1},
                "n_heads": {"type": "integer", "minimum": 1},
                "n_kv_heads": {"type": "integer", "minimum": 1},
                "d_head": {"type": "integer", "minimum": 1},
                "vocab_size": {"type": "integer", "minimum": 1},
                "rope_theta": {"type": "number", "minimum": 0},
                "rope_scaling": {"type": "string", "enum": ["dynamic", "linear", "yarn"]},
                "dropout": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["d_model", "n_layers", "n_heads", "vocab_size"]
        },
        "training": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "minimum": 1},
                "batch_size": {"type": "integer", "minimum": 1},
                "seq_length": {"type": "integer", "minimum": 1},
                "grad_accum_steps": {"type": "integer", "minimum": 1},
                "lr": {"type": "number", "minimum": 0},
                "warmup_steps": {"type": "integer", "minimum": 0},
                "save_every": {"type": "integer", "minimum": 1},
                "val_every": {"type": "integer", "minimum": 1},
            },
            "required": ["steps", "batch_size"]
        },
        "optimizer": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adam", "adamw", "sgd"]},
                "lr": {"type": "number", "minimum": 0},
                "betas": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "weight_decay": {"type": "number", "minimum": 0},
            }
        },
        "distillation": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "temperature": {"type": "number", "minimum": 0},
                "kl_weight": {"type": "number", "minimum": 0},
                "ce_weight": {"type": "number", "minimum": 0},
            }
        },
        "latent": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "m": {"type": "integer", "minimum": 1},
                "c": {"type": "integer", "minimum": 1},
                "p": {"type": "number", "minimum": 0, "maximum": 1},
            }
        },
        "quant": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "start_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "lr_multiplier": {"type": "number", "minimum": 0},
            }
        },
    },
    "required": ["model", "training"]
}


def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """Validate training configuration against schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    try:
        jsonschema.validate(config, TRAINING_CONFIG_SCHEMA)
        return []
    except jsonschema.ValidationError as e:
        return [f"Configuration validation error: {e.message}"]
    except jsonschema.SchemaError as e:
        return [f"Schema validation error: {e.message}"]


def validate_config_file(config_path: Path) -> List[str]:
    """Validate configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        List of validation error messages (empty if valid)
    """
    if not config_path.exists():
        return [f"Configuration file not found: {config_path}"]

    try:
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Validate configuration
        errors = validate_training_config(config)

        if not errors:
            print(f"✅ Configuration validation passed: {config_path}")
        else:
            print(f"❌ Configuration validation failed: {config_path}")
            for error in errors:
                print(f"   {error}")

        return errors

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        return [f"Failed to parse configuration file: {e}"]
    except Exception as e:
        return [f"Unexpected error validating configuration: {e}"]


def create_default_config() -> Dict[str, Any]:
    """Create a default training configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "d_model": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "d_head": 128,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "rope_scaling": "dynamic",
            "dropout": 0.0,
        },
        "training": {
            "steps": 100000,
            "batch_size": 8,
            "seq_length": 4096,
            "grad_accum_steps": 1,
            "lr": 2e-4,
            "warmup_steps": 1000,
            "save_every": 1000,
            "val_every": 1000,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 2e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        },
        "distillation": {
            "enabled": True,
            "temperature": 2.0,
            "kl_weight": 0.5,
            "ce_weight": 0.5,
        },
        "latent": {
            "enabled": False,
            "m": 2,
            "c": 1,
            "p": 0.5,
        },
        "quant": {
            "enabled": False,
            "start_fraction": 0.8,
            "lr_multiplier": 0.1,
        },
    }


def save_config_template(output_path: Path) -> None:
    """Save a configuration template file.

    Args:
        output_path: Path to save the template
    """
    config = create_default_config()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, indent=2)

    print(f"Configuration template saved to: {output_path}")


def merge_configs(config_files: List[str]) -> Dict[str, Any]:
    """Merge multiple configuration files.

    Args:
        config_files: List of configuration file paths

    Returns:
        Merged configuration dictionary

    Raises:
        ValueError: If configuration files are invalid or conflicting
    """
    merged_config = {}

    for config_file in config_files:
        config_path = Path(config_file)

        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        # Validate individual config
        errors = validate_config_file(config_path)
        if errors:
            raise ValueError(
                f"Invalid configuration in {config_path}: {'; '.join(errors)}")

        # Load and merge
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Deep merge
        merged_config = _deep_merge(merged_config, config)

    return merged_config


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        update: Dictionary to merge into base

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration validation")
    parser.add_argument("config_file", help="Configuration file to validate")
    parser.add_argument("--template", action="store_true",
                        help="Generate template config")

    args = parser.parse_args()

    if args.template:
        template_path = Path("config_template.yaml")
        save_config_template(template_path)
    else:
        config_path = Path(args.config_file)
        errors = validate_config_file(config_path)

        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            exit(1)
        else:
            print("Configuration is valid!")
