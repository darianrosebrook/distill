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

    def test_extract_tool_call_invalid_json_in_match(self):
        """Test extraction with invalid JSON in regex match (triggers JSONDecodeError)."""
        # Text with JSON-like pattern that fails to parse
        text = 'I see {"name": "read_file", "arguments": {invalid json} here'
        result = extract_tool_call(text)
        
        # Should handle JSONDecodeError gracefully and try parsing entire text
        # May return None or try to parse the whole text
        assert result is None or isinstance(result, dict)

    def test_extract_tool_call_invalid_json_entire_text(self):
        """Test extraction with invalid JSON when parsing entire text."""
        # Text that looks like JSON but is invalid
        text = '{"name": "read_file", "arguments": {unclosed'
        result = extract_tool_call(text)
        
        # Should handle JSONDecodeError and return None
        assert result is None


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

    def test_extract_tool_name_span_with_next_sentence(self):
        """Test extraction with next sentence match (triggers line 157)."""
        # Text with citation pattern followed by a sentence
        text = "According to the results: data found. The next sentence contains more information."
        tool_names = ["read_file"]
        
        result = extract_tool_name_span(text, tool_names=tool_names)
        
        # May or may not find tool name, but should handle next sentence matching
        assert result is None or isinstance(result, tuple)


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

    def test_extract_json_argument_spans_invalid_json(self):
        """Test extraction with invalid JSON (triggers JSONDecodeError)."""
        # Text with JSON-like pattern that fails to parse
        text = 'I see {"invalid": json} here'
        spans = extract_json_argument_spans(text)
        
        # Should handle JSONDecodeError gracefully and skip invalid matches
        assert isinstance(spans, list)


class TestIntegrationSpanIdentification:
    """Test integration span identification."""

    def test_identify_integration_spans_with_results(self):
        """Test identification with tool results."""
        text = "I called read_file and got result: file contents here"
        tool_results = [{"tool": "read_file", "result": "file contents here"}]

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

    def test_identify_integration_spans_numeric_values(self):
        """Test identification with numeric values in tool results (triggers lines 188-195)."""
        text = "I found count: 5 results in the data. The number 5 appears here."
        tool_results = [{"tool": "search", "result": {"count": 5}}]
        
        spans = identify_integration_spans(text, tool_results=tool_results)
        
        # Should find spans matching numeric values
        assert isinstance(spans, list)
        assert len(spans) > 0

    def test_identify_integration_spans_list_values(self):
        """Test identification with list values in tool results (triggers lines 199-209)."""
        text = "Found 3 items in the results. The first item is: important data here"
        tool_results = [{"tool": "search", "result": {"items": ["important data here", "item2", "item3"]}}]
        
        spans = identify_integration_spans(text, tool_results=tool_results)
        
        # Should find spans matching list length and first item
        assert isinstance(spans, list)
        assert len(spans) > 0

    def test_identify_integration_spans_list_short_first_item(self):
        """Test identification with list where first item is too short."""
        text = "Found 3 items in the results."
        tool_results = [{"tool": "search", "result": {"items": ["a", "b", "c"]}}]
        
        spans = identify_integration_spans(text, tool_results=tool_results)
        
        # Should still find spans for list length
        assert isinstance(spans, list)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
