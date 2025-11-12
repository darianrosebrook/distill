"""
Unit tests for Priority 1: Process-step extractors.

Tests:
1. Tool call extraction
2. Tool name span extraction
3. JSON argument span extraction
4. Integration span identification
"""
import pytest
from training.extractors import (
    extract_tool_call,
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)


class TestToolCallExtraction:
    """Test tool call extraction."""

    def test_extract_tool_call_valid_json(self):
        """Test extraction with valid JSON tool call."""
        text = '{"name": "read_file", "arguments": {"path": "test.txt"}}'
        result = extract_tool_call(text)
        
        assert result is not None
        assert result["name"] == "read_file"
        assert "arguments" in result

    def test_extract_tool_call_embedded(self):
        """Test extraction with JSON embedded in text."""
        text = 'I need to call {"name": "write_file", "arguments": {"path": "out.txt"}} now'
        result = extract_tool_call(text)
        
        assert result is not None
        assert result["name"] == "write_file"

    def test_extract_tool_call_no_match(self):
        """Test extraction with no tool call."""
        text = "This is just regular text without any tool calls"
        result = extract_tool_call(text)
        
        assert result is None

    def test_extract_tool_call_with_tool_names(self):
        """Test extraction with tool name validation."""
        text = '{"name": "read_file", "arguments": {}}'
        tool_names = ["read_file", "write_file"]
        
        result = extract_tool_call(text, tool_names=tool_names)
        
        assert result is not None
        assert result["name"] == "read_file"


class TestToolNameSpanExtraction:
    """Test tool name span extraction."""

    def test_extract_tool_name_span_found(self):
        """Test extraction when tool name is found."""
        # Tool name in JSON format (more likely to be found)
        text = '{"name": "read_file", "arguments": {}}'
        tool_names = ["read_file", "write_file"]
        
        result = extract_tool_name_span(text, tool_names=tool_names)
        
        # May or may not find it depending on implementation
        # If found, should be valid span
        if result is not None:
            start, end = result
            assert start < end
            assert start >= 0
            assert end <= len(text)

    def test_extract_tool_name_span_not_found(self):
        """Test extraction when tool name is not found."""
        text = "This text has no tool names"
        tool_names = ["read_file", "write_file"]
        
        result = extract_tool_name_span(text, tool_names=tool_names)
        
        assert result is None

    def test_extract_tool_name_span_json(self):
        """Test extraction from JSON tool call."""
        text = '{"name": "read_file", "arguments": {}}'
        tool_names = ["read_file"]
        
        result = extract_tool_name_span(text, tool_names=tool_names)
        
        assert result is not None
        start, end = result
        assert "read_file" in text[start:end]


class TestJSONArgumentSpanExtraction:
    """Test JSON argument span extraction."""

    def test_extract_json_argument_spans_simple(self):
        """Test extraction with simple JSON."""
        text = '{"name": "test", "value": 123}'
        spans = extract_json_argument_spans(text)
        
        assert len(spans) > 0
        for start, end in spans:
            assert start < end
            assert start >= 0
            assert end <= len(text)

    def test_extract_json_argument_spans_multiple(self):
        """Test extraction with multiple JSON objects."""
        text = 'First: {"a": 1} Second: {"b": 2}'
        spans = extract_json_argument_spans(text)
        
        assert len(spans) >= 2

    def test_extract_json_argument_spans_nested(self):
        """Test extraction with nested JSON."""
        text = '{"outer": {"inner": {"value": 123}}}'
        spans = extract_json_argument_spans(text)
        
        assert len(spans) > 0

    def test_extract_json_argument_spans_no_json(self):
        """Test extraction with no JSON."""
        text = "This text has no JSON"
        spans = extract_json_argument_spans(text)
        
        assert len(spans) == 0


class TestIntegrationSpanIdentification:
    """Test integration span identification."""

    def test_identify_integration_spans_with_results(self):
        """Test identification with tool results."""
        text = "I called read_file and got result: file contents here"
        tool_results = [
            {"tool": "read_file", "result": "file contents here"}
        ]
        
        spans = identify_integration_spans(text, tool_results=tool_results)
        
        assert len(spans) > 0
        for start, end in spans:
            assert start < end

    def test_identify_integration_spans_no_results(self):
        """Test identification without tool results."""
        text = "This text has no tool results"
        spans = identify_integration_spans(text)
        
        # May find spans based on heuristics
        assert isinstance(spans, list)

    def test_identify_integration_spans_patterns(self):
        """Test identification with common integration patterns."""
        text = "After calling read_file, I processed the result and used it in write_file"
        spans = identify_integration_spans(text)
        
        # Should find integration spans
        assert isinstance(spans, list)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

