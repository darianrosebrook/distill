"""
Tests for training/input_validation.py - Input validation and sanitization utilities.

Tests data validation, sanitization, security checks, and error handling
for training data inputs.
"""
# @author: @darianrosebrook

import pytest
from pathlib import Path
from training.input_validation import (
    ValidationError,
    InputValidator,
    validate_training_data,
    validate_tool_trace,
    validator,
)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating a ValidationError."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestInputValidator:
    """Test InputValidator class."""

    @pytest.fixture
    def strict_validator(self):
        """Create validator in strict mode."""
        return InputValidator(strict_mode=True)

    @pytest.fixture
    def lenient_validator(self):
        """Create validator in lenient mode."""
        return InputValidator(strict_mode=False)

    def test_validator_initialization(self, strict_validator):
        """Test validator initialization."""
        assert strict_validator.strict_mode
        assert len(strict_validator.compiled_patterns) > 0

    def test_validate_text_input_valid(self, strict_validator):
        """Test validating valid text input."""
        text = "This is valid text input"
        result = strict_validator.validate_text_input(text, "test_field")
        assert result == text

    def test_validate_text_input_non_string(self, strict_validator):
        """Test validating non-string input raises error."""
        with pytest.raises(ValidationError):
            strict_validator.validate_text_input(123, "test_field")

    def test_validate_text_input_non_string_lenient(self, lenient_validator):
        """Test validating non-string input in lenient mode."""
        result = lenient_validator.validate_text_input(123, "test_field")
        assert result == "123"

    def test_validate_text_input_too_long(self, strict_validator):
        """Test validating text that exceeds max length."""
        long_text = "x" * (strict_validator.MAX_PROMPT_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            strict_validator.validate_text_input(long_text, "prompt")

    def test_validate_text_input_too_long_lenient(self, lenient_validator):
        """Test validating text that exceeds max length in lenient mode."""
        long_text = "x" * (lenient_validator.MAX_PROMPT_LENGTH + 1)
        result = lenient_validator.validate_text_input(long_text, "prompt")
        assert len(result) == lenient_validator.MAX_PROMPT_LENGTH

    def test_validate_text_input_suspicious_script_tag(self, strict_validator):
        """Test detecting suspicious script tags."""
        malicious_text = '<script>alert("xss")</script>'
        with pytest.raises(ValidationError, match="contains suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

    def test_validate_text_input_suspicious_javascript_url(self, strict_validator):
        """Test detecting suspicious JavaScript URLs."""
        malicious_text = 'Click here: <a href="javascript:alert(1)">link</a>'
        with pytest.raises(ValidationError, match="contains suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

    def test_validate_text_input_suspicious_event_handler(self, strict_validator):
        """Test detecting suspicious event handlers."""
        malicious_text = '<div onclick="alert(1)">Click me</div>'
        with pytest.raises(ValidationError, match="contains suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

    def test_validate_text_input_none_strict(self, strict_validator):
        """Test validating None input in strict mode."""
        with pytest.raises(ValidationError, match="cannot be None"):
            strict_validator.validate_text_input(None, "test_field")

    def test_validate_text_input_none_lenient(self, lenient_validator, capsys):
        """Test validating None input in lenient mode (line 81-86)."""
        result = lenient_validator.validate_text_input(None, "test_field")
        assert result == ""
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_validate_text_input_suspicious_lenient(self, lenient_validator, capsys):
        """Test suspicious pattern in lenient mode (line 114)."""
        malicious_text = '<script>alert("xss")</script>'
        result = lenient_validator.validate_text_input(malicious_text, "test_field")
        # Should return the text but print warning
        assert result == malicious_text
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_contains_suspicious_patterns_non_string(self, strict_validator):
        """Test _contains_suspicious_patterns with non-string input (line 59-60)."""
        result = strict_validator._contains_suspicious_patterns(123)
        assert not result

    def test_contains_suspicious_patterns_matches(self, strict_validator):
        """Test _contains_suspicious_patterns with matching pattern (line 62-65)."""
        result = strict_validator._contains_suspicious_patterns('<script>alert("xss")</script>')
        assert result

    def test_contains_suspicious_patterns_no_match(self, strict_validator):
        """Test _contains_suspicious_patterns with no matching pattern (line 65)."""
        result = strict_validator._contains_suspicious_patterns("Clean text with no suspicious content")
        assert not result

    def test_validate_numeric_input_valid(self, strict_validator):
        """Test validating valid numeric input."""
        result = strict_validator.validate_numeric_input(42, "test_num", min_val=0, max_val=100)
        assert result == 42

    def test_validate_numeric_input_string_number(self, strict_validator):
        """Test validating string number."""
        result = strict_validator.validate_numeric_input("42", "test_num", min_val=0, max_val=100)
        assert result == 42.0

    def test_validate_numeric_input_below_min(self, strict_validator):
        """Test validating number below minimum."""
        with pytest.raises(ValidationError, match="must be >="):
            strict_validator.validate_numeric_input(5, "test_num", min_val=10, max_val=100)

    def test_validate_numeric_input_above_max(self, strict_validator):
        """Test validating number above maximum."""
        with pytest.raises(ValidationError, match="must be <="):
            strict_validator.validate_numeric_input(150, "test_num", min_val=0, max_val=100)

    def test_validate_numeric_input_invalid_type(self, strict_validator):
        """Test validating non-numeric input."""
        with pytest.raises(ValidationError, match="must be numeric"):
            strict_validator.validate_numeric_input("not a number", "test_num")

    def test_validate_tool_call_valid(self, strict_validator):
        """Test validating valid tool call."""
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}
        result = strict_validator.validate_tool_call(tool_call)
        assert result["name"] == "test_tool"
        assert result["arguments"] == {"param": "value"}

    def test_validate_tool_call_not_dict(self, strict_validator):
        """Test validating tool call that is not a dictionary."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_tool_call("not a dict")

    def test_validate_tool_call_missing_name(self, strict_validator):
        """Test validating tool call missing name field."""
        tool_call = {"arguments": {"param": "value"}}
        with pytest.raises(ValidationError, match="missing required field"):
            strict_validator.validate_tool_call(tool_call)

    def test_validate_tool_call_missing_arguments(self, strict_validator):
        """Test validating tool call missing arguments field."""
        tool_call = {"name": "test_tool"}
        with pytest.raises(ValidationError, match="missing required field"):
            strict_validator.validate_tool_call(tool_call)

    def test_validate_tool_call_invalid_name_format(self, strict_validator):
        """Test validating tool call with invalid name format."""
        tool_call = {"name": "123invalid", "arguments": {}}
        with pytest.raises(ValidationError, match="Invalid tool name format"):
            strict_validator.validate_tool_call(tool_call)

    def test_validate_tools_valid(self, strict_validator):
        """Test validating valid tools list (line 253-283)."""
        tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]
        result = strict_validator.validate_tools(tools)
        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"

    def test_validate_tools_not_list(self, strict_validator):
        """Test validating tools that is not a list."""
        with pytest.raises(ValidationError, match="must be a list"):
            strict_validator.validate_tools("not a list")

    def test_validate_tools_too_many(self, strict_validator):
        """Test validating too many tools."""
        tools = [{"name": f"tool{i}", "description": f"Tool {i}"} for i in range(strict_validator.MAX_TOOL_COUNT + 1)]
        with pytest.raises(ValidationError, match="too many tools"):
            strict_validator.validate_tools(tools)

    def test_validate_tools_invalid_tool_type(self, strict_validator):
        """Test validating tools with invalid tool type."""
        tools = [{"name": "tool1", "description": "Tool 1"}, "not a dict"]
        with pytest.raises(ValidationError, match="Tool at index 1"):
            strict_validator.validate_tools(tools)

    def test_validate_tools_missing_name(self, strict_validator):
        """Test validating tools with missing name field."""
        tools = [{"description": "Tool without name"}]
        with pytest.raises(ValidationError, match="missing required field"):
            strict_validator.validate_tools(tools)

    def test_validate_tools_missing_description(self, strict_validator):
        """Test validating tools with missing description field."""
        tools = [{"name": "tool1"}]
        with pytest.raises(ValidationError, match="missing required field"):
            strict_validator.validate_tools(tools)

    def test_validate_structured_data_valid(self, strict_validator):
        """Test validating structured data (line 171-207)."""
        data = {"prompt": "Test prompt", "response": "Test response"}
        result = strict_validator.validate_structured_data(data)
        assert result["prompt"] == "Test prompt"
        assert result["response"] == "Test response"

    def test_validate_structured_data_not_dict(self, strict_validator):
        """Test validating structured data that is not a dict."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_structured_data("not a dict")

    def test_validate_structured_data_missing_required(self, strict_validator):
        """Test validating structured data missing required field."""
        data = {"prompt": "Test prompt"}
        with pytest.raises(ValidationError, match="missing required field"):
            strict_validator.validate_structured_data(data)

    def test_validate_structured_data_custom_required_fields(self, strict_validator):
        """Test validating structured data with custom required fields."""
        data = {"custom_field": "value"}
        result = strict_validator.validate_structured_data(data, required_fields=["custom_field"])
        assert result["custom_field"] == "value"

    def test_validate_structured_data_invalid_required_type(self, strict_validator):
        """Test validating structured data with invalid required field type."""
        data = {"prompt": 123, "response": "Test response"}
        with pytest.raises(ValidationError, match="invalid type"):
            strict_validator.validate_structured_data(data)

    def test_validate_structured_data_nested_dict(self, strict_validator):
        """Test validating structured data with nested dict."""
        data = {"prompt": "Test", "response": "Test", "metadata": {"key": "value"}}
        result = strict_validator.validate_structured_data(data)
        assert result["metadata"] == {"key": "value"}

    def test_validate_structured_data_primitive_types(self, strict_validator):
        """Test validating structured data with primitive types."""
        data = {
            "prompt": "Test",
            "response": "Test",
            "count": 42,
            "score": 0.95,
            "active": True,
            "tags": None,
            "items": [1, 2, 3],
        }
        result = strict_validator.validate_structured_data(data)
        assert result["count"] == 42
        assert result["score"] == 0.95
        assert result["active"]
        assert result["tags"] is None
        assert result["items"] == [1, 2, 3]

    def test_validate_structured_data_invalid_type(self, strict_validator):
        """Test validating structured data with invalid type."""
        class CustomClass:
            pass
        data = {"prompt": "Test", "response": "Test", "custom": CustomClass()}
        with pytest.raises(ValidationError, match="invalid type"):
            strict_validator.validate_structured_data(data)

    def test_validate_structured_data_optional_string_field(self, strict_validator):
        """Test validating structured data with optional string field (line 195)."""
        data = {"prompt": "Test", "response": "Test", "optional_field": "optional value"}
        result = strict_validator.validate_structured_data(data)
        assert result["optional_field"] == "optional value"

    def test_validate_tool_call_arguments_not_dict(self, strict_validator):
        """Test validating tool call with non-dict arguments."""
        tool_call = {"name": "test_tool", "arguments": "not a dict"}
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_tool_call(tool_call)

    def test_validate_training_example_valid(self, strict_validator):
        """Test validating valid training example."""
        example = {
            "prompt": "What is 2+2?",
            "teacher_text": "The answer is 4",
            "answer": "4",
            "cot_steps": ["Step 1", "Step 2"],
            "metadata": {"tool_count": 2},
        }
        result = strict_validator.validate_training_example(example)
        assert "prompt" in result
        assert "teacher_text" in result
        assert "answer" in result

    def test_validate_training_example_not_dict(self, strict_validator):
        """Test validating training example that is not a dictionary."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_training_example("not a dict")

    def test_validate_training_example_invalid_cot_steps(self, strict_validator):
        """Test validating training example with invalid cot_steps (line 334)."""
        example = {"prompt": "Test prompt", "response": "Test response", "cot_steps": "not a list"}
        with pytest.raises(ValidationError, match="must be a list"):
            strict_validator.validate_training_example(example)

    def test_validate_training_example_missing_prompt(self, strict_validator):
        """Test validating training example missing prompt (line 302)."""
        example = {"response": "Test response"}
        with pytest.raises(ValidationError, match="missing required field: prompt"):
            strict_validator.validate_training_example(example)

    def test_validate_training_example_with_answer_alias(self, strict_validator):
        """Test validating training example with answer alias (line 316)."""
        example = {"prompt": "Test prompt", "answer": "Test answer"}
        result = strict_validator.validate_training_example(example)
        assert "response" in result
        assert result["response"] == "Test answer"

    def test_validate_training_example_missing_response_and_answer(self, strict_validator):
        """Test validating training example missing both response and answer (line 322)."""
        example = {"prompt": "Test prompt"}
        with pytest.raises(ValidationError, match="missing required field: response"):
            strict_validator.validate_training_example(example)

    def test_validate_metadata_valid(self, strict_validator):
        """Test validating valid metadata."""
        metadata = {
            "tool_count": 5,
            "intermediate_sizes": [10, 20, 30],
            "pii_tags_present": False,
            "eligible_for_code_mode": True,
        }
        result = strict_validator.validate_metadata(metadata)
        assert result["tool_count"] == 5
        assert result["intermediate_sizes"] == [10, 20, 30]
        assert not result["pii_tags_present"]

    def test_validate_metadata_not_dict(self, strict_validator):
        """Test validating metadata that is not a dictionary."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_metadata("not a dict")

    def test_validate_metadata_invalid_tool_count(self, strict_validator):
        """Test validating metadata with invalid tool_count."""
        metadata = {"tool_count": -1}
        with pytest.raises(ValidationError, match="must be >="):
            strict_validator.validate_metadata(metadata)

    def test_validate_metadata_tool_count_too_high(self, strict_validator):
        """Test validating metadata with tool_count exceeding max."""
        metadata = {"tool_count": strict_validator.MAX_TOOL_COUNT + 1}
        with pytest.raises(ValidationError, match="must be <="):
            strict_validator.validate_metadata(metadata)

    def test_validate_metadata_invalid_intermediate_sizes(self, strict_validator):
        """Test validating metadata with invalid intermediate_sizes."""
        metadata = {"intermediate_sizes": "not a list"}
        with pytest.raises(ValidationError, match="must be a list"):
            strict_validator.validate_metadata(metadata)

    def test_validate_metadata_invalid_boolean_field(self, strict_validator):
        """Test validating metadata with invalid boolean field."""
        metadata = {"pii_tags_present": "not a boolean"}
        with pytest.raises(ValidationError, match="must be a boolean"):
            strict_validator.validate_metadata(metadata)

    def test_validate_batch_valid(self, strict_validator):
        """Test validating valid batch."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "labels": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
        }
        result = strict_validator.validate_batch(batch)
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_validate_batch_not_dict(self, strict_validator):
        """Test validating batch that is not a dictionary."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            strict_validator.validate_batch("not a dict")

    def test_validate_batch_not_tensor(self, strict_validator):
        """Test validating batch with non-tensor field."""
        batch = {"input_ids": "not a tensor"}
        with pytest.raises(ValidationError, match="must be a tensor"):
            strict_validator.validate_batch(batch)

    def test_validate_batch_nan_values(self, strict_validator):
        """Test validating batch with NaN values."""
        import torch

        batch = {"input_ids": torch.tensor([[1.0, float("nan"), 3.0]])}
        with pytest.raises(ValidationError, match="contains NaN values"):
            strict_validator.validate_batch(batch)

    def test_validate_batch_inf_values(self, strict_validator):
        """Test validating batch with infinite values."""
        import torch

        batch = {"input_ids": torch.tensor([[1.0, float("inf"), 3.0]])}
        with pytest.raises(ValidationError, match="contains infinite values"):
            strict_validator.validate_batch(batch)

    def test_sanitize_json_string_valid(self, strict_validator):
        """Test sanitizing valid JSON string."""
        json_str = '{"key": "value"}'
        result = strict_validator.sanitize_json_string(json_str)
        assert result == json_str

    def test_sanitize_json_string_null_bytes(self, strict_validator):
        """Test sanitizing JSON string with null bytes."""
        json_str = '{"key": "value\x00"}'
        result = strict_validator.sanitize_json_string(json_str)
        assert "\x00" not in result

    def test_sanitize_json_string_too_long(self, strict_validator):
        """Test sanitizing JSON string that is too long (line 446-447)."""
        long_json = '{"key": "' + "x" * 2000000 + '"}'
        result = strict_validator.sanitize_json_string(long_json)
        assert len(result) <= 1000000
        # Test exact boundary: exactly 1000000 chars should pass through
        exact_boundary = "x" * 1000000
        result_exact = strict_validator.sanitize_json_string(exact_boundary)
        assert len(result_exact) == 1000000
        # Test one over boundary: should be truncated
        over_boundary = "x" * 1000001
        result_over = strict_validator.sanitize_json_string(over_boundary)
        assert len(result_over) == 1000000

    def test_validate_file_path_valid(self, strict_validator, tmp_path):
        """Test validating valid file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = strict_validator.validate_file_path(test_file, must_exist=True)
        # Function returns str, not Path (line 497)
        assert isinstance(result, str)
        assert Path(result).exists()

    def test_validate_file_path_not_exist(self, strict_validator, tmp_path):
        """Test validating file path that doesn't exist."""
        non_existent = tmp_path / "nonexistent.txt"
        with pytest.raises(ValidationError, match="does not exist"):
            strict_validator.validate_file_path(non_existent, must_exist=True)

    def test_validate_file_path_not_exist_optional(self, strict_validator, tmp_path):
        """Test validating file path that doesn't exist (optional)."""
        non_existent = tmp_path / "nonexistent.txt"
        result = strict_validator.validate_file_path(non_existent, must_exist=False)
        assert isinstance(result, str)

    def test_validate_file_path_permission_error(self, strict_validator, tmp_path, monkeypatch):
        """Test validating file path with permission error (line 479-480)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        def mock_exists(path):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("pathlib.Path.exists", mock_exists)
        with pytest.raises(ValidationError, match="cannot access file"):
            strict_validator.validate_file_path(test_file)

    def test_validate_file_path_stat_permission_error(self, strict_validator, tmp_path, monkeypatch):
        """Test validating file path with stat permission error (line 495)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        def mock_stat(self):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(Path, "stat", mock_stat)
        with pytest.raises(ValidationError, match="cannot access file"):
            strict_validator.validate_file_path(test_file)

    def test_validate_file_path_invalid_path(self, strict_validator, monkeypatch):
        """Test validating file path with invalid path (line 469-470)."""
        def mock_relative_to(self, other):
            raise ValueError("Invalid path")

        monkeypatch.setattr(Path, "relative_to", mock_relative_to)
        with pytest.raises(ValidationError, match="Invalid path"):
            strict_validator.validate_file_path("/invalid/../path")

    def test_validate_file_path_dangerous_extension(self, strict_validator, tmp_path):
        """Test validating file path with dangerous extension."""
        dangerous_file = tmp_path / "malicious.exe"
        dangerous_file.write_text("malicious content")
        with pytest.raises(ValidationError, match="Dangerous file extension"):
            strict_validator.validate_file_path(dangerous_file)

    def test_validate_file_path_too_large(self, strict_validator, tmp_path):
        """Test validating file path with file that is too large."""
        large_file = tmp_path / "large.txt"
        # Create a file larger than MAX_FILE_SIZE_MB
        large_content = "x" * (strict_validator.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        large_file.write_text(large_content)
        with pytest.raises(ValidationError, match="too large"):
            strict_validator.validate_file_path(large_file)


class TestValidateTrainingData:
    """Test validate_training_data function."""

    def test_validate_training_data_valid(self):
        """Test validating valid training data."""
        data = [
            {"prompt": "Test 1", "answer": "Answer 1"},
            {"prompt": "Test 2", "answer": "Answer 2"},
        ]
        result = validate_training_data(data)
        assert len(result) == 2
        assert result[0]["prompt"] == "Test 1"

    def test_validate_training_data_invalid_example(self):
        """Test validating training data with invalid example."""
        data = [
            {"prompt": "Test 1", "answer": "Answer 1"},
            {"prompt": '<script>alert("xss")</script>', "answer": "Answer 2"},
        ]
        with pytest.raises(ValidationError, match="Example 1"):
            validate_training_data(data)

    def test_validate_training_data_not_list(self):
        """Test validating training data that is not a list (line 517)."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_training_data("not a list")

    def test_validate_training_data_empty_list(self):
        """Test validating empty training data."""
        result = validate_training_data([])
        assert result == []


class TestValidateToolTrace:
    """Test validate_tool_trace function."""

    def test_validate_tool_trace_valid(self):
        """Test validating valid tool trace."""
        # validate_tool_trace expects tool_name, tool_input, tool_output (not name/arguments)
        trace = [
            {"tool_name": "tool1", "tool_input": "input1", "tool_output": "output1"},
            {"tool_name": "tool2", "tool_input": "input2", "tool_output": "output2"},
        ]
        result = validate_tool_trace(trace)
        assert len(result) == 2
        assert result[0]["tool_name"] == "tool1"

    def test_validate_tool_trace_not_list(self):
        """Test validating tool trace that is not a list."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_tool_trace("not a list")

    def test_validate_tool_trace_invalid_tool_call(self):
        """Test validating tool trace with invalid tool call."""
        trace = [
            {"tool_name": "tool1", "tool_input": "input1", "tool_output": "output1"},
            {"invalid": "tool call"},
        ]
        with pytest.raises(ValidationError, match="Tool trace entry 1"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_not_dict(self):
        """Test validating tool trace entry that is not a dict (line 551)."""
        trace = ["not a dict"]
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_missing_tool_name(self):
        """Test validating tool trace missing tool_name (line 557)."""
        trace = [{"tool_input": "input", "tool_output": "output"}]
        with pytest.raises(ValidationError, match="missing required field: tool_name"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_missing_tool_input(self):
        """Test validating tool trace missing tool_input (line 565)."""
        trace = [{"tool_name": "tool1", "tool_output": "output"}]
        with pytest.raises(ValidationError, match="missing required field: tool_input"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_missing_tool_output(self):
        """Test validating tool trace missing tool_output (line 576)."""
        trace = [{"tool_name": "tool1", "tool_input": "input"}]
        with pytest.raises(ValidationError, match="missing required field: tool_output"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_non_string_input(self):
        """Test validating tool trace with non-string tool_input."""
        trace = [{"tool_name": "tool1", "tool_input": {"json": "object"}, "tool_output": "output"}]
        result = validate_tool_trace(trace)
        assert result[0]["tool_input"] == {"json": "object"}

    def test_validate_tool_trace_non_string_output(self):
        """Test validating tool trace with non-string tool_output (line 578)."""
        trace = [{"tool_name": "tool1", "tool_input": "input", "tool_output": {"json": "object"}}]
        result = validate_tool_trace(trace)
        assert result[0]["tool_output"] == {"json": "object"}
        # Test with string output to ensure isinstance check works (line 578)
        trace_str = [{"tool_name": "tool1", "tool_input": "input", "tool_output": "string output"}]
        result_str = validate_tool_trace(trace_str)
        assert result_str[0]["tool_output"] == "string output"

    def test_validate_tool_trace_preserves_other_fields(self):
        """Test validating tool trace preserves other fields (line 586-588)."""
        trace = [{
            "tool_name": "tool1",
            "tool_input": "input",
            "tool_output": "output",
            "timestamp": "2024-01-01",
            "duration": 0.5,
        }]
        result = validate_tool_trace(trace)
        assert result[0]["timestamp"] == "2024-01-01"
        assert result[0]["duration"] == 0.5

    def test_validate_tool_trace_empty_list(self):
        """Test validating empty tool trace."""
        result = validate_tool_trace([])
        assert result == []


class TestGlobalValidator:
    """Test global validator instance."""

    def test_global_validator_exists(self):
        """Test that global validator instance exists."""
        assert validator is not None
        assert isinstance(validator, InputValidator)

    def test_global_validator_strict_mode(self):
        """Test that global validator is in strict mode."""
        assert validator.strict_mode


class TestInputValidatorAdditional:
    """Additional tests for InputValidator to cover missing lines."""

    @pytest.fixture
    def strict_validator(self):
        """Create validator in strict mode."""
        return InputValidator(strict_mode=True)

    def test_validate_training_example_missing_response_after_check(self, strict_validator):
        """Test validate_training_example when response/answer check fails (line 322)."""
        # This tests the else branch when neither response nor answer is present
        # after the initial check (line 304-306 checks, then 321-323 else branch)
        example = {"prompt": "Test prompt"}
        # Remove both response and answer to trigger line 322
        with pytest.raises(ValidationError, match="missing required field: response"):
            strict_validator.validate_training_example(example)

    def test_validate_file_path_stat_permission_error_handling(self, strict_validator, tmp_path, monkeypatch):
        """Test validate_file_path with stat permission error (line 495)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Mock stat() to raise PermissionError when called on the file
        def mock_stat(self):
            if self == test_file:
                raise PermissionError("Permission denied")
            # For other paths, use original stat
            return Path.stat(self)

        monkeypatch.setattr(Path, "stat", mock_stat)
        
        with pytest.raises(ValidationError, match="cannot access file"):
            strict_validator.validate_file_path(test_file, must_exist=True)

    def test_validate_training_example_with_both_response_and_answer(self, strict_validator):
        """Test validate_training_example when both response and answer are present (line 318, 340-342)."""
        # Test that when both response and answer are present, answer is still validated (line 318)
        example = {
            "prompt": "Test prompt",
            "response": "Test response",
            "answer": "Test answer",  # Both present - answer should still be validated
        }
        result = strict_validator.validate_training_example(example)
        assert "response" in result
        assert "answer" in result
        assert result["response"] == "Test response"
        assert result["answer"] == "Test answer"
        # Test that answer validation happens even when response is present (line 318 check)
        example_with_suspicious_answer = {
            "prompt": "Test prompt",
            "response": "Test response",
            "answer": '<script>alert("xss")</script>',  # Suspicious answer should be caught
        }
        with pytest.raises(ValidationError, match="contains suspicious content"):
            strict_validator.validate_training_example(example_with_suspicious_answer)

    def test_validate_file_path_directory_not_file(self, strict_validator, tmp_path):
        """Test validate_file_path when path exists but is a directory (not a file) - line 485."""
        # Create a directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        # Directory exists but is not a file, so file size check is skipped
        result = strict_validator.validate_file_path(test_dir, must_exist=True)
        assert isinstance(result, str)
        # Should succeed (directories don't go through file size check)

    def test_validate_file_path_path_is_not_file(self, strict_validator, tmp_path):
        """Test validate_file_path when path exists but is_file() returns False - line 485."""
        # Path exists but is not a file - should skip file size check
        test_dir = tmp_path / "subdir"
        test_dir.mkdir()
        
        result = strict_validator.validate_file_path(test_dir, must_exist=True)
        assert isinstance(result, str)
        # Should not raise error for directory (file size check only for files)

    def test_validate_batch_tensor_without_isnan(self, strict_validator):
        """Test validate_batch with tensor that doesn't have isnan() method - line 423."""
        # Create a mock tensor-like object without isnan() method
        class MockTensor:
            shape = (2, 128)
            def isnan(self):
                raise AttributeError("isnan not available")
        
        batch = {"input_ids": MockTensor()}
        # Should not raise error if isnan() is not available
        result = strict_validator.validate_batch(batch)
        assert "input_ids" in result

    def test_validate_batch_tensor_without_isinf(self, strict_validator):
        """Test validate_batch with tensor that doesn't have isinf() method - line 426."""
        # Create a mock tensor-like object without isinf() method
        class MockTensor:
            shape = (2, 128)
            def isnan(self):
                return type('MockResult', (), {'any': lambda self: False})()
            def isinf(self):
                raise AttributeError("isinf not available")
        
        batch = {"input_ids": MockTensor()}
        # Should not raise error if isinf() is not available
        result = strict_validator.validate_batch(batch)
        assert "input_ids" in result

    def test_validate_file_path_relative_to_success(self, strict_validator, tmp_path):
        """Test validate_file_path when relative_to() succeeds (normal case) - line 468."""
        # Normal path should succeed relative_to check
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        result = strict_validator.validate_file_path(test_file, must_exist=True)
        assert isinstance(result, str)
        assert Path(result).exists()

    def test_validate_metadata_eligible_for_code_mode(self, strict_validator):
        """Test validate_metadata with eligible_for_code_mode field - line 388-393."""
        metadata = {
            "tool_count": 2,
            "eligible_for_code_mode": True,
        }
        result = strict_validator.validate_metadata(metadata)
        assert result["eligible_for_code_mode"] is True
        
        # Test invalid type
        invalid_metadata = {"eligible_for_code_mode": "not a boolean"}
        with pytest.raises(ValidationError, match="must be a boolean"):
            strict_validator.validate_metadata(invalid_metadata)

    def test_validate_training_example_cot_steps_with_suspicious_content(self, strict_validator):
        """Test validate_training_example when cot_steps contains suspicious content."""
        example = {
            "prompt": "Test prompt",
            "response": "Test response",
            "cot_steps": ["Step 1", '<script>alert("xss")</script>'],  # Suspicious content in cot_step
        }
        with pytest.raises(ValidationError, match="contains suspicious content"):
            strict_validator.validate_training_example(example)

    def test_validate_batch_partial_tensor_fields(self, strict_validator):
        """Test validate_batch with only some tensor fields present."""
        import torch
        
        # Only input_ids present (not labels or attention_mask)
        batch = {"input_ids": torch.randint(0, 1000, (2, 128))}
        result = strict_validator.validate_batch(batch)
        assert "input_ids" in result
        assert "labels" not in result
        assert "attention_mask" not in result

    def test_validate_batch_all_tensor_fields(self, strict_validator):
        """Test validate_batch with all tensor fields present."""
        import torch
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "labels": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128, dtype=torch.bool),
        }
        result = strict_validator.validate_batch(batch)
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_sanitize_json_string_exactly_at_boundary(self, strict_validator):
        """Test sanitize_json_string at exactly 1000000 character boundary - line 446."""
        # Exactly at boundary should pass through unchanged
        exact_boundary = "x" * 1000000
        result = strict_validator.sanitize_json_string(exact_boundary)
        assert len(result) == 1000000
        assert result == exact_boundary

    def test_validate_metadata_intermediate_sizes_empty_list(self, strict_validator):
        """Test validate_metadata with empty intermediate_sizes list."""
        metadata = {"intermediate_sizes": []}
        result = strict_validator.validate_metadata(metadata)
        assert result["intermediate_sizes"] == []

    def test_validate_metadata_intermediate_sizes_multiple(self, strict_validator):
        """Test validate_metadata with multiple intermediate sizes."""
        metadata = {"intermediate_sizes": [100, 200, 300]}
        result = strict_validator.validate_metadata(metadata)
        assert result["intermediate_sizes"] == [100, 200, 300]

    def test_validate_metadata_preserves_unknown_fields(self, strict_validator):
        """Test that validate_metadata preserves unknown fields - line 366."""
        metadata = {
            "tool_count": 2,
            "unknown_field": "should be preserved",
            "another_unknown": 42,
        }
        result = strict_validator.validate_metadata(metadata)
        assert "unknown_field" in result
        assert "another_unknown" in result
        assert result["unknown_field"] == "should be preserved"
        assert result["another_unknown"] == 42


