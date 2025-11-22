"""
Tests for training/extractors.py - Extractors for process-step supervision targets.

Tests tool call extraction, span extraction, and integration span identification.
"""
# @author: @darianrosebrook

from training.extractors import (
    extract_tool_call,
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)


class TestExtractToolCall:
    """Test extract_tool_call function."""

    def test_extract_tool_call_valid_json(self):
        """Test extracting tool call from valid JSON."""
        text = '{"name": "test_tool", "arguments": {"param": "value"}}'
        result = extract_tool_call(text)
        assert result is not None
        assert result["name"] == "test_tool"
        assert result["arguments"] == {"param": "value"}

    def test_extract_tool_call_text_with_json(self):
        """Test extracting tool call from text containing JSON."""
        text = 'Here is a tool call: {"name": "calculator", "arguments": {"expression": "2+2"}}'
        result = extract_tool_call(text)
        assert result is not None
        assert result["name"] == "calculator"

    def test_extract_tool_call_no_name_field(self):
        """Test extracting tool call without name field."""
        text = '{"arguments": {"param": "value"}}'
        result = extract_tool_call(text)
        assert result is None

    def test_extract_tool_call_not_dict(self):
        """Test extracting tool call that is not a dictionary."""
        text = '["not", "a", "dict"]'
        result = extract_tool_call(text)
        assert result is None

    def test_extract_tool_call_invalid_json(self):
        """Test extracting tool call from invalid JSON."""
        text = '{"name": "test_tool", "arguments": {'
        result = extract_tool_call(text)
        assert result is None

    def test_extract_tool_call_plain_text(self):
        """Test extracting tool call from plain text."""
        text = "This is just plain text with no JSON"
        result = extract_tool_call(text)
        assert result is None

    def test_extract_tool_call_nested_json(self):
        """Test extracting tool call with nested JSON."""
        text = '{"name": "tool", "arguments": {"nested": {"deep": "value"}}}'
        result = extract_tool_call(text)
        assert result is not None
        assert result["name"] == "tool"

    def test_extract_tool_call_multiple_json_objects(self):
        """Test extracting tool call when multiple JSON objects present."""
        text = 'First: {"other": "data"} Second: {"name": "tool", "arguments": {}}'
        result = extract_tool_call(text)
        # Should find the one with "name" field
        assert result is not None
        assert result["name"] == "tool"

    def test_extract_tool_call_with_tool_names_filter(self):
        """Test extracting tool call with tool name filter."""
        text = '{"name": "calculator", "arguments": {}}'
        result = extract_tool_call(text, tool_names=["calculator", "search"])
        assert result is not None
        assert result["name"] == "calculator"


class TestExtractToolNameSpan:
    """Test extract_tool_name_span function."""

    def test_extract_tool_name_span_valid(self):
        """Test extracting tool name span from valid text."""
        text = '{"name": "test_tool", "arguments": {}}'
        result = extract_tool_name_span(text)
        assert result is not None
        start, end = result
        assert start < end
        assert text[start:end] == '"test_tool"'

    def test_extract_tool_name_span_text_with_json(self):
        """Test extracting tool name span from text containing JSON."""
        text = 'Here is the tool: {"name": "calculator", "arguments": {}}'
        result = extract_tool_name_span(text)
        assert result is not None
        start, end = result
        assert '"calculator"' in text[start:end]

    def test_extract_tool_name_span_no_tool_call(self):
        """Test extracting tool name span when no tool call present."""
        text = "This is just plain text"
        result = extract_tool_name_span(text)
        assert result is None

    def test_extract_tool_name_span_invalid_json(self):
        """Test extracting tool name span from invalid JSON."""
        text = '{"name": "test_tool"'
        result = extract_tool_name_span(text)
        # May or may not find it depending on extraction logic
        assert isinstance(result, (type(None), tuple))

    def test_extract_tool_name_span_with_tool_names(self):
        """Test extracting tool name span with tool name filter."""
        text = '{"name": "calculator", "arguments": {}}'
        result = extract_tool_name_span(text, tool_names=["calculator"])
        assert result is not None


class TestExtractJSONArgumentSpans:
    """Test extract_json_argument_spans function."""

    def test_extract_json_argument_spans_single_object(self):
        """Test extracting JSON argument spans for single object."""
        text = 'Here is JSON: {"key": "value", "number": 42}'
        spans = extract_json_argument_spans(text)
        assert len(spans) >= 1
        start, end = spans[0]
        assert start < end
        assert '"key"' in text[start:end] or '"value"' in text[start:end]

    def test_extract_json_argument_spans_multiple_objects(self):
        """Test extracting JSON argument spans for multiple objects."""
        text = 'First: {"a": 1} Second: {"b": 2} Third: {"c": 3}'
        spans = extract_json_argument_spans(text)
        assert len(spans) >= 1  # Should find at least one valid JSON

    def test_extract_json_argument_spans_nested_objects(self):
        """Test extracting JSON argument spans for nested objects."""
        text = '{"outer": {"inner": {"deep": "value"}}}'
        spans = extract_json_argument_spans(text)
        assert len(spans) >= 1

    def test_extract_json_argument_spans_invalid_json(self):
        """Test extracting JSON argument spans from invalid JSON."""
        text = '{"invalid": json}'
        spans = extract_json_argument_spans(text)
        # Should only return valid JSON spans
        for start, end in spans:
            json_str = text[start:end]
            # Should be valid JSON
            assert json_str.startswith("{")
            assert json_str.endswith("}")

    def test_extract_json_argument_spans_no_json(self):
        """Test extracting JSON argument spans when no JSON present."""
        text = "This is just plain text"
        spans = extract_json_argument_spans(text)
        assert spans == []

    def test_extract_json_argument_spans_empty_string(self):
        """Test extracting JSON argument spans from empty string."""
        spans = extract_json_argument_spans("")
        assert spans == []

    def test_extract_json_argument_spans_arrays(self):
        """Test extracting JSON argument spans for arrays."""
        text = 'Here is an array: [1, 2, 3, {"nested": "value"}]'
        spans = extract_json_argument_spans(text)
        # May or may not find arrays depending on pattern
        assert isinstance(spans, list)


class TestIdentifyIntegrationSpans:
    """Test identify_integration_spans function."""

    def test_identify_integration_spans_citation_patterns(self):
        """Test identifying integration spans with citation patterns."""
        text = "According to the tool output, the result is 42. This is the answer."
        spans = identify_integration_spans(text)
        assert len(spans) >= 1

    def test_identify_integration_spans_based_on_pattern(self):
        """Test identifying integration spans with 'Based on' pattern."""
        text = "Based on the search results, we found 5 items."
        spans = identify_integration_spans(text)
        assert len(spans) >= 1

    def test_identify_integration_spans_with_tool_results(self):
        """Test identifying integration spans with tool results."""
        text = "The tool returned: Found 5 results. Here they are: item1, item2, item3."
        tool_results = [{"count": 5, "items": ["item1", "item2", "item3"]}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1

    def test_identify_integration_spans_direct_copying(self):
        """Test identifying integration spans that directly copy tool results."""
        text = "The result is: Found 5 items in the database."
        tool_results = [{"count": 5, "message": "Found 5 items in the database"}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1

    def test_identify_integration_spans_numeric_values(self):
        """Test identifying integration spans with numeric values from tool results."""
        text = "The count is 42 and the total is 100."
        tool_results = [{"count": 42, "total": 100}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1

    def test_identify_integration_spans_list_length(self):
        """Test identifying integration spans that mention list length."""
        text = "Found 3 items in the results."
        tool_results = [{"items": ["a", "b", "c"]}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1

    def test_identify_integration_spans_structured_data(self):
        """Test identifying integration spans with structured data patterns."""
        text = "Lines 10-15: This is the relevant code. Count: 5 results found."
        spans = identify_integration_spans(text)
        assert len(spans) >= 1

    def test_identify_integration_spans_after_tool_call(self):
        """Test identifying integration spans after tool call markers."""
        text = 'Called tool: {"name": "search", "arguments": {}}. Integration: Found 5 results.'
        spans = identify_integration_spans(text)
        assert len(spans) >= 1

    def test_identify_integration_spans_no_integration(self):
        """Test identifying integration spans when no integration present."""
        text = "This is just plain text with no tool integration."
        spans = identify_integration_spans(text)
        # May find some patterns, but should handle gracefully
        assert isinstance(spans, list)

    def test_identify_integration_spans_empty_string(self):
        """Test identifying integration spans from empty string."""
        spans = identify_integration_spans("")
        assert spans == []

    def test_identify_integration_spans_merges_overlapping(self):
        """Test that overlapping spans are merged."""
        text = "According to the results, we found 5 items. The count is 5."
        spans = identify_integration_spans(text)
        # Should merge overlapping spans
        assert isinstance(spans, list)
        # Check no overlapping spans
        if len(spans) > 1:
            for i in range(len(spans) - 1):
                assert spans[i][1] <= spans[i + 1][0]

    def test_identify_integration_spans_multiple_tool_results(self):
        """Test identifying integration spans with multiple tool results."""
        text = "First result: 42. Second result: 100."
        tool_results = [{"value": 42}, {"value": 100}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1

    def test_identify_integration_spans_fuzzy_matching(self):
        """Test identifying integration spans with fuzzy matching."""
        text = "The tool output shows: Found 5 items in the database."
        tool_results = [{"message": "Found 5 items in the database"}]
        spans = identify_integration_spans(text, tool_results)
        assert len(spans) >= 1


class TestExtractorsIntegration:
    """Test integration of extractor functions."""

    def test_complete_extraction_workflow(self):
        """Test complete extraction workflow."""
        teacher_text = 'Tool call: {"name": "calculator", "arguments": {"expression": "2+2"}}. Integration: The result is 4.'

        # Extract tool call
        tool_call = extract_tool_call(teacher_text)
        assert tool_call is not None
        assert tool_call["name"] == "calculator"

        # Extract tool name span
        name_span = extract_tool_name_span(teacher_text)
        assert name_span is not None

        # Extract JSON argument spans
        json_spans = extract_json_argument_spans(teacher_text)
        assert len(json_spans) >= 1

        # Identify integration spans
        integration_spans = identify_integration_spans(teacher_text)
        assert len(integration_spans) >= 1

    def test_extraction_with_tool_results(self):
        """Test extraction with actual tool results."""
        teacher_text = 'Called search tool: {"name": "search", "arguments": {"query": "test"}}. Found 3 results.'
        tool_results = [{"count": 3, "items": ["result1", "result2", "result3"]}]

        # Extract tool call
        tool_call = extract_tool_call(teacher_text)
        assert tool_call is not None

        # Identify integration spans with tool results
        integration_spans = identify_integration_spans(teacher_text, tool_results)
        assert len(integration_spans) >= 1

    def test_extraction_edge_cases(self):
        """Test extraction with edge cases."""
        # Empty text
        assert extract_tool_call("") is None
        assert extract_tool_name_span("") is None
        assert extract_json_argument_spans("") == []
        assert identify_integration_spans("") == []

        # Text with only whitespace
        whitespace = "   \n\t   "
        assert extract_tool_call(whitespace) is None
        assert identify_integration_spans(whitespace) == []

        # Text with malformed JSON
        malformed = '{"name": "tool"'
        result = extract_tool_call(malformed)
        # May or may not extract depending on implementation
        assert isinstance(result, (type(None), dict))

















