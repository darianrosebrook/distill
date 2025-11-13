"""
Unit tests for tool schema registry.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.schema_registry import ToolSchemaRegistry, get_registry


class TestToolSchemaRegistry:
    """Test tool schema registry."""

    def test_init(self):
        """Test registry initialization."""
        registry = ToolSchemaRegistry()

        assert registry.schemas_dir is not None
        assert isinstance(registry._schemas, dict)
        assert len(registry._schemas) > 0  # Should have default schemas

    def test_get_schema_existing(self):
        """Test getting existing schema."""
        registry = ToolSchemaRegistry()

        schema = registry.get_schema("web.search")

        assert schema is not None
        assert schema["tool_name"] == "web.search"
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_get_schema_nonexistent(self):
        """Test getting non-existent schema."""
        registry = ToolSchemaRegistry()

        schema = registry.get_schema("nonexistent.tool")

        assert schema is None

    def test_register_schema(self):
        """Test registering a new schema."""
        registry = ToolSchemaRegistry()

        new_schema = {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "enum": ["test.tool"]},
                "arguments": {"type": "object"},
            },
        }

        registry.register_schema("test.tool", new_schema)

        retrieved = registry.get_schema("test.tool")
        assert retrieved is not None
        assert retrieved["tool_name"] == "test.tool"

    def test_validate_tool_call_valid(self):
        """Test validating a valid tool call."""
        registry = ToolSchemaRegistry()

        tool_call = {"name": "web.search", "arguments": {"q": "test query"}}

        is_valid, error = registry.validate_tool_call("web.search", tool_call)

        assert is_valid is True
        assert error is None

    def test_validate_tool_call_missing_required(self):
        """Test validating tool call with missing required field."""
        registry = ToolSchemaRegistry()

        tool_call = {
            "name": "web.search",
            "arguments": {},  # Missing "q"
        }

        is_valid, error = registry.validate_tool_call("web.search", tool_call)

        assert is_valid is False
        assert error is not None
        assert "required" in error.lower() or "missing" in error.lower()

    def test_validate_tool_call_name_mismatch(self):
        """Test validating tool call with name mismatch."""
        registry = ToolSchemaRegistry()

        tool_call = {"name": "wrong.tool", "arguments": {"q": "test"}}

        is_valid, error = registry.validate_tool_call("web.search", tool_call)

        assert is_valid is False
        assert "mismatch" in error.lower() or "name" in error.lower()

    def test_validate_generic_fallback(self):
        """Test generic validation fallback."""
        registry = ToolSchemaRegistry()

        tool_call = {"name": "unknown.tool", "arguments": {"key": "value"}}

        is_valid, error = registry.validate_tool_call("unknown.tool", tool_call)

        # Should use generic validation
        assert is_valid is True
        assert error is None

    def test_list_tools(self):
        """Test listing all registered tools."""
        registry = ToolSchemaRegistry()

        tools = registry.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "web.search" in tools

    def test_get_generic_schema(self):
        """Test getting generic schema."""
        registry = ToolSchemaRegistry()

        schema = registry.get_generic_schema()

        assert schema["type"] == "object"
        assert "name" in schema["required"]
        assert "arguments" in schema["required"]

    def test_save_schema(self):
        """Test saving schema to file."""
        with TemporaryDirectory() as tmpdir:
            schemas_dir = Path(tmpdir) / "schemas"
            registry = ToolSchemaRegistry(schemas_dir=schemas_dir)

            # Register a test schema
            test_schema = {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string", "enum": ["test.save"]},
                    "arguments": {"type": "object"},
                },
            }
            registry.register_schema("test.save", test_schema)

            # Save to file
            output_file = schemas_dir / "test.save.json"
            registry.save_schema("test.save", output_file)

            # Verify file exists and contains correct schema
            assert output_file.exists()
            with open(output_file, "r") as f:
                saved_schema = json.load(f)
                assert saved_schema["tool_name"] == "test.save"

    def test_load_schemas_from_directory(self):
        """Test loading schemas from directory."""
        with TemporaryDirectory() as tmpdir:
            schemas_dir = Path(tmpdir) / "schemas"
            schemas_dir.mkdir()

            # Create a test schema file
            test_schema = {
                "tool_name": "test.load",
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string", "enum": ["test.load"]},
                    "arguments": {"type": "object"},
                },
            }

            schema_file = schemas_dir / "test.load.json"
            with open(schema_file, "w") as f:
                json.dump(test_schema, f)

            # Load registry
            registry = ToolSchemaRegistry(schemas_dir=schemas_dir)

            # Verify schema was loaded
            loaded_schema = registry.get_schema("test.load")
            assert loaded_schema is not None
            assert loaded_schema["tool_name"] == "test.load"


class TestGetRegistry:
    """Test global registry function."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_registry_functionality(self):
        """Test that global registry works."""
        registry = get_registry()

        assert isinstance(registry, ToolSchemaRegistry)
        assert len(registry.list_tools()) > 0
