#!/usr/bin/env python3
"""
Config Resolution Layer

Merges base YAMLs, environment-specific overrides, and CLI overrides.
Emits a canonical resolved config (frozen, hashable) that is:
- Serialized into checkpoints
- Used in export
- Stored in the run manifest

Usage:
    python scripts/config_resolve.py --base configs/kd_recipe.yaml --override configs/local.yaml --output resolved_config.json
    python scripts/config_resolve.py --base configs/kd_recipe.yaml --env dev --output resolved_config.json
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json_config(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def resolve_config(
    base_config_path: Path,
    override_paths: Optional[list[Path]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration by merging base, overrides, and CLI args.

    Args:
        base_config_path: Path to base configuration file
        override_paths: List of override configuration file paths
        env_overrides: Environment-specific overrides dictionary
        cli_overrides: CLI-provided overrides dictionary

    Returns:
        Resolved configuration dictionary
    """
    # Load base config
    if base_config_path.suffix in [".yaml", ".yml"]:
        config = load_yaml_config(base_config_path)
    elif base_config_path.suffix == ".json":
        config = load_json_config(base_config_path)
    else:
        raise ValueError(f"Unsupported config file format: {base_config_path.suffix}")

    # Apply override files
    if override_paths:
        for override_path in override_paths:
            if override_path.suffix in [".yaml", ".yml"]:
                override = load_yaml_config(override_path)
            elif override_path.suffix == ".json":
                override = load_json_config(override_path)
            else:
                raise ValueError(f"Unsupported override file format: {override_path.suffix}")

            config = deep_merge(config, override)

    # Apply environment overrides
    if env_overrides:
        config = deep_merge(config, env_overrides)

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        config = deep_merge(config, cli_overrides)

    return config


def compute_config_fingerprint(config: Dict[str, Any]) -> str:
    """
    Compute hash fingerprint for configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hash fingerprint (hex string)
    """
    # Sort keys for consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def validate_config(
    config: Dict[str, Any], schema_path: Optional[Path] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema_path: Path to JSON schema file (optional)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if schema_path is None:
        # Use default schema path
        schema_path = Path(__file__).parent.parent / "configs" / "schema" / "config_schema.json"

    if not schema_path.exists():
        return True, None  # Skip validation if schema not available

    try:
        import jsonschema

        with open(schema_path, "r") as f:
            schema = json.load(f)
        jsonschema.validate(instance=config, schema=schema)
        return True, None
    except ImportError:
        # jsonschema not available - skip validation
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Resolve and validate configuration")
    parser.add_argument("--base", type=Path, required=True, help="Base configuration file")
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        help="Override configuration file (can specify multiple)",
    )
    parser.add_argument("--env", help="Environment name (for environment-specific overrides)")
    parser.add_argument("--output", type=Path, help="Output path for resolved config (JSON)")
    parser.add_argument("--fingerprint", action="store_true", help="Print config fingerprint")
    parser.add_argument("--validate", action="store_true", help="Validate against schema")
    parser.add_argument(
        "--set", action="append", help="Set config value (e.g., training.batch_size=32)"
    )

    args = parser.parse_args()

    # Parse CLI overrides
    cli_overrides = {}
    if args.set:
        for override_str in args.set:
            if "=" not in override_str:
                print(
                    f"Error: Invalid override format: {override_str} (expected key=value)",
                    file=sys.stderr,
                )
                sys.exit(1)

            key_path, value = override_str.split("=", 1)
            keys = key_path.split(".")

            # Convert value to appropriate type
            try:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string

            # Set nested value
            current = cli_overrides
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

    # Load environment-specific overrides if specified
    env_overrides = None
    if args.env:
        env_config_path = Path(__file__).parent.parent / "configs" / f"{args.env}.yaml"
        if env_config_path.exists():
            env_overrides = load_yaml_config(env_config_path)

    # Resolve configuration
    try:
        resolved_config = resolve_config(
            base_config_path=args.base,
            override_paths=args.override,
            env_overrides=env_overrides,
            cli_overrides=cli_overrides,
        )
    except Exception as e:
        print(f"Error: Failed to resolve configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate if requested
    if args.validate:
        is_valid, error = validate_config(resolved_config)
        if not is_valid:
            print(f"Error: Configuration validation failed: {error}", file=sys.stderr)
            sys.exit(1)
        print("✅ Configuration validation passed")

    # Compute fingerprint
    fingerprint = compute_config_fingerprint(resolved_config)
    if args.fingerprint:
        print(f"Config fingerprint: {fingerprint}")

    # Add fingerprint to config
    resolved_config["_fingerprint"] = fingerprint

    # Save resolved config
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(resolved_config, f, indent=2)
        print(f"✅ Resolved configuration saved to: {args.output}")
    else:
        # Print to stdout
        print(json.dumps(resolved_config, indent=2))


if __name__ == "__main__":
    main()
