"""
Per-tool schema registry for tighter validation.

Provides tool-specific JSON schemas for constrained decoding and validation.
Each tool has its own schema with required arguments and type constraints.

Usage:
    from tools.schema_registry import ToolSchemaRegistry

    registry = ToolSchemaRegistry()

    # Get schema for a specific tool
    schema = registry.get_schema("web.search")

    # Validate tool call against schema
    is_valid = registry.validate_tool_call("web.search", {"name": "web.search", "arguments": {"q": "test"}})

    # Get all registered tools
    tools = registry.list_tools()
"""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class ToolSchemaRegistry:
    """
    Registry for tool-specific JSON schemas.

    Provides tighter validation than generic tool call schema by enforcing
    tool-specific argument requirements and types.
    """

    def __init__(self, schemas_dir: Optional[Path] = None):
        """
        Initialize schema registry.

        Args:
            schemas_dir: Directory containing tool schema files (default: tools/schemas/)
        """
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent / "schemas"

        self.schemas_dir = schemas_dir
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load all tool schemas from schemas directory."""
        if not self.schemas_dir.exists():
            self.schemas_dir.mkdir(parents=True, exist_ok=True)
            return

        # Load JSON schema files
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, "r") as f:
                    schema = json.load(f)
                    tool_name = schema.get("tool_name") or schema_file.stem
                    self._schemas[tool_name] = schema
            except Exception as e:
                print(f"[ToolSchemaRegistry] WARN: Failed to load schema {schema_file}: {e}")

        # If no schemas found, initialize with default schemas
        if not self._schemas:
            self._init_default_schemas()

    def _init_default_schemas(self):
        """Initialize registry with default tool schemas."""
        default_schemas = {
            "web.search": {
                "tool_name": "web.search",
                "type": "object",
                "required": ["q"],
                "properties": {
                    "q": {"type": "string"},
                    "recency": {"type": "integer", "minimum": 0, "maximum": 365},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "site": {
                        "type": "string",
                        "enum": ["example.com", "example.org", "example.net"],
                    },
                },
            },
            "web.open": {
                "tool_name": "web.open",
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
            "read_file": {
                "tool_name": "read_file",
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string", "enum": ["utf-8", "latin-1"]},
                },
                "oneOf": [{"required": ["path"]}, {"required": ["file_id"]}],
            },
            "repo.read": {
                "tool_name": "repo.read",
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string", "enum": ["repo.read"]},
                    "arguments": {
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"},
                            "grep": {
                                "type": "string",
                                "description": "Optional grep pattern to filter lines",
                            },
                            "lines": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Optional line numbers to read",
                            },
                        },
                    },
                },
            },
            "file.write": {
                "tool_name": "file.write",
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string", "enum": ["file.write"]},
                    "arguments": {
                        "type": "object",
                        "required": ["path", "content"],
                        "properties": {
                            "path": {"type": "string", "description": "File path to write"},
                            "content": {"type": "string", "description": "Content to write"},
                            "append": {
                                "type": "boolean",
                                "description": "Append to file instead of overwrite",
                                "default": False,
                            },
                        },
                    },
                },
            },
            "code.execute": {
                "tool_name": "code.execute",
                "type": "object",
                "required": ["language", "code"],
                "properties": {
                    "language": {"type": "string", "enum": ["python", "bash", "javascript"]},
                    "code": {"type": "string"},
                    "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 10000},
                },
            },
        }

        self._schemas.update(default_schemas)

    def get_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool (e.g., "web.search")

        Returns:
            Tool schema dict, or None if not found
        """
        return self._schemas.get(tool_name)

    def get(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Alias for get_schema() for compatibility with external code.

        Args:
            tool_name: Name of the tool (e.g., "web.search")

        Returns:
            Tool schema dict, or None if not found
        """
        return self.get_schema(tool_name)

    def all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered schemas.

        Returns:
            Dictionary mapping tool names to schemas
        """
        return dict(self._schemas)

    def register_schema(self, tool_name: str, schema: Dict[str, Any]):
        """
        Register a new tool schema.

        Args:
            tool_name: Name of the tool
            schema: JSON schema dict
        """
        schema["tool_name"] = tool_name
        self._schemas[tool_name] = schema

    def validate_tool_call(
        self, tool_name: str, tool_call: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call against its schema.

        Args:
            tool_name: Expected tool name
            tool_call: Tool call dict with "name" and "arguments"

        Returns:
            (is_valid, error_message) tuple
        """
        schema = self.get_schema(tool_name)
        if not schema:
            # Fall back to generic validation
            return self._validate_generic(tool_call)

        # Validate against schema
        from coreml.runtime.constrained_decode import SchemaValidator

        validator = SchemaValidator(schema)
        is_valid, error = validator.validate(tool_call)

        if not is_valid:
            return False, error

        # Additional checks: tool name must match
        if tool_call.get("name") != tool_name:
            return (
                False,
                f"Tool name mismatch: expected '{tool_name}', got '{tool_call.get('name')}'",
            )

        return True, None

    def _validate_generic(self, tool_call: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate against generic tool call schema."""
        if not isinstance(tool_call, dict):
            return False, "Tool call must be a dictionary"

        if "name" not in tool_call:
            return False, "Missing required field: name"

        if "arguments" not in tool_call:
            return False, "Missing required field: arguments"

        if not isinstance(tool_call["name"], str):
            return False, "Field 'name' must be a string"

        if not isinstance(tool_call["arguments"], dict):
            return False, "Field 'arguments' must be an object"

        return True, None

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._schemas.keys())

    def get_generic_schema(self) -> Dict[str, Any]:
        """
        Get generic tool call schema (fallback when tool-specific schema not available).

        Returns:
            Generic tool call schema
        """
        return {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object", "additionalProperties": True},
            },
        }

    def save_schema(self, tool_name: str, output_file: Optional[Path] = None):
        """
        Save a tool schema to a JSON file.

        Args:
            tool_name: Name of the tool
            output_file: Output file path (default: schemas_dir/{tool_name}.json)
        """
        schema = self.get_schema(tool_name)
        if not schema:
            raise ValueError(f"Schema not found for tool: {tool_name}")

        if output_file is None:
            output_file = self.schemas_dir / f"{tool_name}.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)


# Global registry instance
_registry: Optional[ToolSchemaRegistry] = None


def get_registry() -> ToolSchemaRegistry:
    """Get global schema registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolSchemaRegistry()
    return _registry


def validate_args(schema: dict, args: dict) -> tuple[bool, list[str]]:
    """
    Validate arguments against a schema.

    Args:
        schema: Schema dict with "required" and "properties"
        args: Arguments dict to validate

    Returns:
        (is_valid, list_of_errors) tuple
    """
    errs = []
    if not isinstance(args, dict):
        return False, ["args_not_dict"]
    for k in schema.get("required", []):
        if k not in args:
            errs.append(f"missing:{k}")
    props = schema.get("properties", {})
    for k, v in args.items():
        spec = props.get(k, {})
        t = spec.get("type")
        if t == "string" and not isinstance(v, str):
            errs.append(f"type:{k}")
        if t == "integer" and not isinstance(v, int):
            errs.append(f"type:{k}")
        if t == "number" and not isinstance(v, (int, float)):
            errs.append(f"type:{k}")
        if "enum" in spec and v not in spec["enum"]:
            errs.append(f"enum:{k}")
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if isinstance(v, (int, float)):
            if lo is not None and v < lo:
                errs.append(f"min:{k}")
            if hi is not None and v > hi:
                errs.append(f"max:{k}")
    return (len(errs) == 0), errs
