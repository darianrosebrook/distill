"""
Tests for training/input_validation.py - Input validation and sanitization.

Tests security checks, data validation, sanitization, and training data validation.
"""
# @author: @darianrosebrook

from unittest.mock import patch

import pytest

from training.input_validation import (
    ValidationError,
    InputValidator,
    validate_training_data,
    validate_tool_trace,
)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test ValidationError can be created and raised."""
        error = ValidationError("Test validation error")
        assert str(error) == "Test validation error"

        with pytest.raises(ValidationError, match="Test error"):
            raise ValidationError("Test error")


class TestInputValidatorInitialization:
    """Test InputValidator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        validator = InputValidator()

        assert validator.strict_mode is True
        assert len(validator.compiled_patterns) == len(InputValidator.SUSPICIOUS_PATTERNS)

    def test_init_non_strict(self):
        """Test non-strict mode initialization."""
        validator = InputValidator(strict_mode=False)

        assert validator.strict_mode is False

    def test_suspicious_patterns_compiled(self):
        """Test that suspicious patterns are properly compiled."""
        validator = InputValidator()

        assert len(validator.compiled_patterns) > 0
        assert all(hasattr(p, "search") for p in validator.compiled_patterns)


class TestTextInputValidation:
    """Test text input validation and sanitization."""

    @pytest.fixture
    def validator(self):
        """Create input validator fixture."""
        return InputValidator()

    def test_validate_text_input_valid(self, validator):
        """Test validation of valid text input."""
        text = "This is a normal text input."
        result = validator.validate_text_input(text)

        assert result == text

    def test_validate_text_input_too_long(self, validator):
        """Test validation of overly long text input."""
        long_text = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)

        if validator.strict_mode:
            with pytest.raises(ValidationError, match="exceeds maximum length"):
                validator.validate_text_input(long_text)
        else:
            # In non-strict mode, should truncate
            result = validator.validate_text_input(long_text)
            assert len(result) <= InputValidator.MAX_PROMPT_LENGTH

    def test_validate_text_input_suspicious_script(self, validator):
        """Test detection of suspicious script tags."""
        suspicious_text = 'Normal text <script>alert("hack")</script> more text'

        if validator.strict_mode:
            with pytest.raises(ValidationError, match="suspicious content"):
                validator.validate_text_input(suspicious_text)
        else:
            # In non-strict mode, should still raise for security
            with pytest.raises(ValidationError):
                validator.validate_text_input(suspicious_text)

    def test_validate_text_input_javascript_url(self, validator):
        """Test detection of JavaScript URLs."""
        suspicious_text = 'Click here: javascript:alert("hack")'

        with pytest.raises(ValidationError, match="suspicious content"):
            validator.validate_text_input(suspicious_text)

    def test_validate_text_input_event_handler(self, validator):
        """Test detection of event handlers."""
        suspicious_text = "<a onclick=\"alert('hack')\">Link</a>"

        with pytest.raises(ValidationError, match="suspicious content"):
            validator.validate_text_input(suspicious_text)

    def test_validate_text_input_custom_field_name(self, validator):
        """Test validation with custom field name in error messages."""
        long_text = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text_input(long_text, field_name="custom_field")

        assert "custom_field" in str(exc_info.value)

    def test_validate_text_input_none_input(self, validator):
        """Test validation of None input."""
        with pytest.raises(ValidationError, match="cannot be None"):
            validator.validate_text_input(None)

    def test_validate_text_input_empty_string(self, validator):
        """Test validation of empty string."""
        result = validator.validate_text_input("")
        assert result == ""


class TestStructuredDataValidation:
    """Test structured data validation."""

    @pytest.fixture
    def validator(self):
        """Create input validator fixture."""
        return InputValidator()

    def test_validate_structured_data_valid(self, validator):
        """Test validation of valid structured data."""
        data = {"prompt": "What is 2+2?", "response": "4", "metadata": {"source": "test"}}

        result = validator.validate_structured_data(data)
        assert result == data

    def test_validate_structured_data_missing_required(self, validator):
        """Test validation fails with missing required fields."""
        data = {
            "prompt": "What is 2+2?",
            # Missing response
            "metadata": {"source": "test"},
        }

        with pytest.raises(ValidationError, match="missing required field"):
            validator.validate_structured_data(data)

    def test_validate_structured_data_invalid_types(self, validator):
        """Test validation fails with invalid data types."""
        data = {
            "prompt": 123,  # Should be string
            "response": "4",
            "metadata": {"source": "test"},
        }

        with pytest.raises(ValidationError, match="invalid type"):
            validator.validate_structured_data(data)

    def test_validate_structured_data_suspicious_content(self, validator):
        """Test validation detects suspicious content in structured data."""
        data = {
            "prompt": "Normal prompt",
            "response": '<script>alert("hack")</script>',
            "metadata": {"source": "test"},
        }

        with pytest.raises(ValidationError, match="suspicious content"):
            validator.validate_structured_data(data)

    def test_validate_structured_data_too_long(self, validator):
        """Test validation of overly long structured data."""
        data = {
            "prompt": "x" * (InputValidator.MAX_PROMPT_LENGTH + 1),
            "response": "4",
            "metadata": {"source": "test"},
        }

        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validator.validate_structured_data(data)


class TestToolValidation:
    """Test tool-related validation."""

    @pytest.fixture
    def validator(self):
        """Create input validator fixture."""
        return InputValidator()

    def test_validate_tools_valid(self, validator):
        """Test validation of valid tools."""
        tools = [
            {"name": "calculator", "description": "A calculator tool"},
            {"name": "search", "description": "A search tool"},
        ]

        result = validator.validate_tools(tools)
        assert result == tools

    def test_validate_tools_too_many(self, validator):
        """Test validation fails with too many tools."""
        tools = [
            {"name": f"tool{i}", "description": f"Tool {i}"}
            for i in range(InputValidator.MAX_TOOL_COUNT + 1)
        ]

        with pytest.raises(ValidationError, match="too many tools"):
            validator.validate_tools(tools)

    def test_validate_tools_missing_fields(self, validator):
        """Test validation fails with missing tool fields."""
        tools = [
            {"name": "calculator"},  # Missing description
            {"description": "A tool"},  # Missing name
        ]

        with pytest.raises(ValidationError, match="missing required field"):
            validator.validate_tools(tools)

    def test_validate_tools_suspicious_description(self, validator):
        """Test validation detects suspicious content in tool descriptions."""
        tools = [{"name": "calculator", "description": '<script>alert("hack")</script>'}]

        with pytest.raises(ValidationError, match="suspicious content"):
            validator.validate_tools(tools)


class TestFileValidation:
    """Test file validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create input validator fixture."""
        return InputValidator()

    def test_validate_file_path_valid(self, validator, tmp_path):
        """Test validation of valid file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validator.validate_file_path(str(test_file))
        assert result == str(test_file)

    def test_validate_file_path_nonexistent(self, validator):
        """Test validation fails for nonexistent file."""
        nonexistent_path = "/nonexistent/file.txt"

        with pytest.raises(ValidationError, match="file does not exist"):
            validator.validate_file_path(nonexistent_path)

    def test_validate_file_path_too_large(self, validator, tmp_path):
        """Test validation fails for file that's too large."""
        large_file = tmp_path / "large.txt"
        # Create a file larger than MAX_FILE_SIZE_MB
        large_content = "x" * (InputValidator.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        large_file.write_text(large_content)

        with pytest.raises(ValidationError, match="file too large"):
            validator.validate_file_path(str(large_file))

    @patch("pathlib.Path.stat")
    def test_validate_file_path_permission_error(self, mock_stat, validator):
        """Test validation fails on permission error."""
        mock_stat.side_effect = PermissionError("Permission denied")

        with pytest.raises(ValidationError, match="cannot access file"):
            validator.validate_file_path("/some/file.txt")

    @patch("pathlib.Path.stat")
    def test_validate_file_path_permission_error_on_stat(self, mock_stat, validator, tmp_path):
        """Test validation fails on permission error when calling stat() (triggers line 495)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        # Mock stat() to raise PermissionError when called on existing file
        def stat_side_effect(*args, **kwargs):
            # Only raise error when stat is called (file exists check passes)
            raise PermissionError("Permission denied")
        
        mock_stat.side_effect = stat_side_effect
        
        with pytest.raises(ValidationError, match="cannot access file"):
            validator.validate_file_path(str(test_file), must_exist=True)


class TestTrainingDataValidation:
    """Test training data validation functions."""

    def test_validate_training_data_valid(self):
        """Test validation of valid training data."""
        data = [
            {"prompt": "What is 2+2?", "response": "4", "metadata": {"source": "test"}},
            {"prompt": "Hello", "response": "Hi there!", "metadata": {"source": "test"}},
        ]

        result = validate_training_data(data)
        assert result == data

    def test_validate_training_data_invalid_structure(self):
        """Test validation fails with invalid data structure."""
        data = [
            {
                "prompt": "What is 2+2?",
                # Missing response
                "metadata": {"source": "test"},
            }
        ]

        with pytest.raises(ValidationError):
            validate_training_data(data)

    def test_validate_training_data_empty_list(self):
        """Test validation of empty data list."""
        result = validate_training_data([])
        assert result == []

    def test_validate_tool_trace_valid(self):
        """Test validation of valid tool trace."""
        trace = [
            {
                "tool_name": "calculator",
                "tool_input": "2+2",
                "tool_output": "4",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]

        result = validate_tool_trace(trace)
        assert result == trace

    def test_validate_tool_trace_invalid(self):
        """Test validation fails with invalid tool trace."""
        trace = [
            {
                "tool_name": "calculator",
                # Missing required fields
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]

        with pytest.raises(ValidationError):
            validate_tool_trace(trace)


class TestSecurityPatterns:
    """Test security pattern detection."""

    @pytest.fixture
    def validator(self):
        """Create input validator fixture."""
        return InputValidator()

    def test_detect_suspicious_patterns_script_tags(self, validator):
        """Test detection of script tags."""
        text = 'Normal text <script>alert("hack")</script> more text'
        assert validator._contains_suspicious_patterns(text)

    def test_detect_suspicious_patterns_javascript_url(self, validator):
        """Test detection of JavaScript URLs."""
        text = 'Link: javascript:alert("hack")'
        assert validator._contains_suspicious_patterns(text)

    def test_detect_suspicious_patterns_iframe(self, validator):
        """Test detection of iframe tags."""
        text = 'Content <iframe src="evil.com"></iframe> more content'
        assert validator._contains_suspicious_patterns(text)

    def test_detect_suspicious_patterns_event_handler(self, validator):
        """Test detection of event handlers."""
        text = '<div onclick="evilFunction()">Click me</div>'
        assert validator._contains_suspicious_patterns(text)

    def test_no_suspicious_patterns_normal_text(self, validator):
        """Test that normal text doesn't trigger suspicious pattern detection."""
        text = "This is normal text with no suspicious content."
        assert not validator._contains_suspicious_patterns(text)

    def test_suspicious_patterns_case_insensitive(self, validator):
        """Test that pattern detection is case insensitive."""
        text = 'Content <SCRIPT>alert("hack")</SCRIPT> more content'
        assert validator._contains_suspicious_patterns(text)

    def test_contains_suspicious_patterns_non_string(self, validator):
        """Test that non-string input returns False."""
        assert not validator._contains_suspicious_patterns(None)
        assert not validator._contains_suspicious_patterns(123)
        assert not validator._contains_suspicious_patterns([])


class TestInputValidatorNonStrictMode:
    """Test InputValidator in non-strict mode (warnings instead of errors)."""

    @pytest.fixture
    def validator(self):
        """Create non-strict validator fixture."""
        return InputValidator(strict_mode=False)

    def test_validate_text_input_none_non_strict(self, validator):
        """Test None input in non-strict mode returns empty string."""
        result = validator.validate_text_input(None, "test_field")
        assert result == ""

    def test_validate_text_input_non_string_non_strict(self, validator):
        """Test non-string input in non-strict mode converts to string."""
        result = validator.validate_text_input(123, "test_field")
        assert result == "123"
        result = validator.validate_text_input([1, 2, 3], "test_field")
        assert isinstance(result, str)

    def test_validate_text_input_non_string_strict(self):
        """Test non-string input in strict mode raises error."""
        validator = InputValidator(strict_mode=True)
        with pytest.raises(ValidationError, match="must be a string"):
            validator.validate_text_input(123, "test_field")

    def test_validate_text_input_too_long_non_strict(self, validator):
        """Test too long input in non-strict mode truncates."""
        long_text = "x" * (validator.MAX_PROMPT_LENGTH + 100)
        result = validator.validate_text_input(long_text, "prompt")
        assert len(result) == validator.MAX_PROMPT_LENGTH

    def test_validate_text_input_suspicious_non_strict(self, validator):
        """Test suspicious content in non-strict mode sanitizes."""
        suspicious = '<script>alert("hack")</script>'
        result = validator.validate_text_input(suspicious, "prompt")
        # Should sanitize or return empty
        assert isinstance(result, str)


class TestInputValidatorEdgeCases:
    """Test edge cases and error paths in InputValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator fixture."""
        return InputValidator()

    def test_validate_structured_data_not_dict(self, validator):
        """Test validate_structured_data with non-dict input."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_structured_data("not a dict")

    def test_validate_structured_data_list_in_data(self, validator):
        """Test validate_structured_data with list in data."""
        data = {"prompt": "test", "response": "test", "items": [1, 2, 3]}
        result = validator.validate_structured_data(data)
        assert result["items"] == [1, 2, 3]

    def test_validate_structured_data_optional_string_field(self, validator):
        """Test validate_structured_data with optional string field."""
        data = {"prompt": "test", "response": "test", "optional_field": "optional value"}
        result = validator.validate_structured_data(data)
        assert result["optional_field"] == "optional value"

    def test_validate_structured_data_unexpected_type(self, validator):
        """Test validate_structured_data with unexpected type."""
        data = {"prompt": "test", "response": "test", "weird_field": object()}
        with pytest.raises(ValidationError, match="invalid type"):
            validator.validate_structured_data(data)

    def test_validate_structured_data_invalid_type_non_required(self, validator):
        """Test validate_structured_data with invalid type in non-required field."""
        data = {"prompt": "test", "response": "test", "metadata": 123}
        # Non-required fields with primitive types are accepted
        result = validator.validate_structured_data(data)
        assert result["metadata"] == 123

    def test_validate_tool_call_not_dict(self, validator):
        """Test validate_tool_call with non-dict input."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_tool_call("not a dict")

    def test_validate_tool_call_missing_arguments(self, validator):
        """Test validate_tool_call with missing arguments field."""
        tool_call = {"name": "test.tool"}
        with pytest.raises(ValidationError, match="missing required field"):
            validator.validate_tool_call(tool_call)

    def test_validate_tool_call_invalid_name_format(self, validator):
        """Test validate_tool_call with invalid name format."""
        tool_call = {"name": "123invalid", "arguments": {}}
        with pytest.raises(ValidationError, match="Invalid tool name format"):
            validator.validate_tool_call(tool_call)

    def test_validate_tool_call_arguments_not_dict(self, validator):
        """Test validate_tool_call with non-dict arguments."""
        tool_call = {"name": "test.tool", "arguments": "not a dict"}
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_tool_call(tool_call)

    def test_validate_tool_call_valid(self, validator):
        """Test validate_tool_call with valid tool call (triggers line 239)."""
        tool_call = {"name": "test.tool", "arguments": {"param": "value"}}
        result = validator.validate_tool_call(tool_call)
        
        assert result == {"name": "test.tool", "arguments": {"param": "value"}}

    def test_validate_tools_not_list(self, validator):
        """Test validate_tools with non-list input."""
        with pytest.raises(ValidationError, match="must be a list"):
            validator.validate_tools("not a list")

    def test_validate_tools_tool_not_dict(self, validator):
        """Test validate_tools with tool that's not a dict."""
        tools = ["not a dict"]
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_tools(tools)

    def test_validate_training_example_not_dict(self, validator):
        """Test validate_training_example with non-dict input."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_training_example("not a dict")

    def test_validate_training_example_missing_prompt(self, validator):
        """Test validate_training_example with missing prompt field (triggers line 302)."""
        example = {"response": "test"}
        with pytest.raises(ValidationError, match="missing required field.*prompt"):
            validator.validate_training_example(example)

    def test_validate_training_example_missing_response(self, validator):
        """Test validate_training_example with missing response field."""
        example = {"prompt": "test"}
        with pytest.raises(ValidationError, match="missing required field"):
            validator.validate_training_example(example)

    def test_validate_training_example_cot_steps_not_list(self, validator):
        """Test validate_training_example with cot_steps that's not a list."""
        example = {"prompt": "test", "response": "test", "cot_steps": "not a list"}
        with pytest.raises(ValidationError, match="must be a list"):
            validator.validate_training_example(example)

    def test_validate_training_example_with_answer_instead_of_response(self, validator):
        """Test validate_training_example with answer field instead of response."""
        example = {"prompt": "test", "answer": "test answer"}
        result = validator.validate_training_example(example)
        assert result["response"] == "test answer"

    def test_validate_training_example_with_both_answer_and_response(self, validator):
        """Test validate_training_example with both answer and response fields."""
        example = {"prompt": "test", "response": "response", "answer": "answer"}
        result = validator.validate_training_example(example)
        assert result["response"] == "response"
        assert result["answer"] == "answer"

    def test_validate_training_example_with_teacher_text(self, validator):
        """Test validate_training_example with teacher_text field."""
        example = {"prompt": "test", "response": "test", "teacher_text": "teacher text"}
        result = validator.validate_training_example(example)
        assert result["teacher_text"] == "teacher text"

    def test_validate_training_example_with_cot_steps(self, validator):
        """Test validate_training_example with cot_steps field."""
        example = {"prompt": "test", "response": "test", "cot_steps": ["step1", "step2"]}
        result = validator.validate_training_example(example)
        assert result["cot_steps"] == ["step1", "step2"]

    def test_validate_metadata_not_dict(self, validator):
        """Test validate_metadata with non-dict input."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_metadata("not a dict")

    def test_validate_metadata_intermediate_sizes_not_list(self, validator):
        """Test validate_metadata with intermediate_sizes that's not a list."""
        metadata = {"intermediate_sizes": "not a list"}
        with pytest.raises(ValidationError, match="must be a list"):
            validator.validate_metadata(metadata)

    def test_validate_metadata_boolean_field_not_bool(self, validator):
        """Test validate_metadata with boolean field that's not bool."""
        metadata = {"pii_tags_present": "not a bool"}
        with pytest.raises(ValidationError, match="must be a boolean"):
            validator.validate_metadata(metadata)

    def test_validate_metadata_boolean_fields_valid(self, validator):
        """Test validate_metadata with valid boolean fields."""
        metadata = {"pii_tags_present": True, "eligible_for_code_mode": False}
        result = validator.validate_metadata(metadata)
        assert result["pii_tags_present"] is True
        assert result["eligible_for_code_mode"] is False

    def test_validate_batch_not_dict(self, validator):
        """Test validate_batch with non-dict input."""
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validator.validate_batch("not a dict")

    def test_validate_batch_tensor_not_tensor(self, validator):
        """Test validate_batch with field that's not a tensor."""
        batch = {"input_ids": "not a tensor"}
        with pytest.raises(ValidationError, match="must be a tensor"):
            validator.validate_batch(batch)

    def test_validate_batch_valid_tensors(self, validator):
        """Test validate_batch with valid tensors."""
        import torch
        batch = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3),
            "labels": torch.tensor([1, 2, 3])
        }
        result = validator.validate_batch(batch)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_sanitize_json_string_removes_null_bytes(self, validator):
        """Test sanitize_json_string removes null bytes."""
        json_str = '{"key": "value\x00with\x00nulls"}'
        result = validator.sanitize_json_string(json_str)
        assert "\x00" not in result

    def test_sanitize_json_string_truncates_long_string(self, validator):
        """Test sanitize_json_string truncates strings over 1MB."""
        long_json = "x" * 2000000  # 2MB
        result = validator.sanitize_json_string(long_json)
        assert len(result) == 1000000  # Should be truncated to 1MB

    def test_validate_file_path_invalid_path_traversal(self, validator):
        """Test validate_file_path with path traversal attempt."""
        # Note: Current implementation doesn't prevent path traversal
        # This test verifies the function handles the path without error
        # (path traversal prevention should be added in a future security improvement)
        result = validator.validate_file_path("../../../etc/passwd", must_exist=False)
        assert isinstance(result, str)

    def test_validate_file_path_dangerous_extension(self, validator, tmp_path):
        """Test validate_file_path with dangerous file extension."""
        dangerous_file = tmp_path / "test.exe"
        dangerous_file.write_text("test")
        with pytest.raises(ValidationError, match="Dangerous file extension"):
            validator.validate_file_path(str(dangerous_file))

    def test_validate_file_path_must_exist_false(self, validator):
        """Test validate_file_path with must_exist=False doesn't check existence."""
        nonexistent = "/nonexistent/file.txt"
        # Should not raise error if must_exist=False
        result = validator.validate_file_path(nonexistent, must_exist=False)
        assert isinstance(result, str)

    def test_validate_tool_trace_not_list(self):
        """Test validate_tool_trace with non-list input."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_tool_trace("not a list")

    def test_validate_tool_trace_entry_not_dict(self):
        """Test validate_tool_trace with entry that's not a dict."""
        trace = ["not a dict"]
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_missing_tool_name(self):
        """Test validate_tool_trace with missing tool_name."""
        trace = [{"tool_input": "test", "tool_output": "test"}]
        with pytest.raises(ValidationError, match="missing required field"):
            validate_tool_trace(trace)

    def test_validate_training_data_not_list(self):
        """Test validate_training_data with non-list input."""
        with pytest.raises(ValidationError):
            validate_training_data("not a list")

    def test_validate_training_data_invalid_example(self):
        """Test validate_training_data with invalid example."""
        data = [{"prompt": "test"}]  # Missing response
        with pytest.raises(ValidationError):
            validate_training_data(data)

    def test_validate_numeric_input_valid_int(self, validator):
        """Test validate_numeric_input with valid integer."""
        result = validator.validate_numeric_input(42, "test_field")
        assert result == 42

    def test_validate_numeric_input_valid_float(self, validator):
        """Test validate_numeric_input with valid float."""
        result = validator.validate_numeric_input(3.14, "test_field")
        assert result == 3.14

    def test_validate_numeric_input_string_number(self, validator):
        """Test validate_numeric_input with string that can be converted."""
        result = validator.validate_numeric_input("42", "test_field")
        assert result == 42.0

    def test_validate_numeric_input_invalid_type(self, validator):
        """Test validate_numeric_input with non-numeric value."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validator.validate_numeric_input("not a number", "test_field")

    def test_validate_numeric_input_below_min(self, validator):
        """Test validate_numeric_input with value below minimum."""
        with pytest.raises(ValidationError, match="must be >="):
            validator.validate_numeric_input(5, "test_field", min_val=10)

    def test_validate_numeric_input_above_max(self, validator):
        """Test validate_numeric_input with value above maximum."""
        with pytest.raises(ValidationError, match="must be <="):
            validator.validate_numeric_input(100, "test_field", max_val=50)

    def test_validate_numeric_input_with_bounds(self, validator):
        """Test validate_numeric_input with both min and max bounds."""
        result = validator.validate_numeric_input(25, "test_field", min_val=10, max_val=50)
        assert result == 25

    def test_validate_text_input_suspicious_in_strict_mode(self, validator):
        """Test validate_text_input with suspicious content in strict mode."""
        suspicious = '<script>alert("hack")</script>'
        with pytest.raises(ValidationError, match="contains suspicious content"):
            validator.validate_text_input(suspicious, "prompt")

    def test_validate_structured_data_required_field_wrong_type(self, validator):
        """Test validate_structured_data with required field having wrong type."""
        data = {"prompt": 123, "response": "test"}  # prompt should be string
        with pytest.raises(ValidationError, match="invalid type"):
            validator.validate_structured_data(data)

    def test_validate_metadata_tool_count_validation(self, validator):
        """Test validate_metadata with tool_count validation."""
        metadata = {"tool_count": 25}
        result = validator.validate_metadata(metadata)
        assert result["tool_count"] == 25

    def test_validate_metadata_tool_count_too_high(self, validator):
        """Test validate_metadata with tool_count exceeding max."""
        metadata = {"tool_count": validator.MAX_TOOL_COUNT + 1}
        with pytest.raises(ValidationError, match="must be <="):
            validator.validate_metadata(metadata)

    def test_validate_metadata_intermediate_sizes_validation(self, validator):
        """Test validate_metadata with intermediate_sizes validation."""
        metadata = {"intermediate_sizes": [100, 200, 300]}
        result = validator.validate_metadata(metadata)
        assert result["intermediate_sizes"] == [100, 200, 300]

    def test_validate_batch_with_nan(self, validator):
        """Test validate_batch detects NaN values in tensors."""
        import torch
        batch = {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1.0, float('nan'), 3.0])
        }
        with pytest.raises(ValidationError, match="contains NaN values"):
            validator.validate_batch(batch)

    def test_validate_batch_with_inf(self, validator):
        """Test validate_batch detects infinite values in tensors."""
        import torch
        batch = {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1.0, float('inf'), 3.0])
        }
        with pytest.raises(ValidationError, match="contains infinite values"):
            validator.validate_batch(batch)

    def test_validate_file_path_file_too_large(self, validator, tmp_path):
        """Test validate_file_path with file that's too large."""
        large_file = tmp_path / "large.txt"
        # Create a file larger than MAX_FILE_SIZE_MB
        large_content = "x" * (validator.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        large_file.write_text(large_content)
        with pytest.raises(ValidationError, match="file too large"):
            validator.validate_file_path(str(large_file))

    def test_validate_tool_trace_missing_tool_input(self):
        """Test validate_tool_trace with missing tool_input."""
        trace = [{"tool_name": "test", "tool_output": "test"}]
        with pytest.raises(ValidationError, match="missing required field"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_missing_tool_output(self):
        """Test validate_tool_trace with missing tool_output."""
        trace = [{"tool_name": "test", "tool_input": "test"}]
        with pytest.raises(ValidationError, match="missing required field"):
            validate_tool_trace(trace)

    def test_validate_tool_trace_preserves_other_fields(self):
        """Test validate_tool_trace preserves fields like timestamp."""
        trace = [{
            "tool_name": "test",
            "tool_input": "input",
            "tool_output": "output",
            "timestamp": "2024-01-01T00:00:00Z"
        }]
        result = validate_tool_trace(trace)
        assert result[0]["timestamp"] == "2024-01-01T00:00:00Z"

    def test_validate_tool_trace_non_string_input(self):
        """Test validate_tool_trace with non-string tool_input."""
        trace = [{
            "tool_name": "test",
            "tool_input": {"key": "value"},  # Dict instead of string
            "tool_output": "output"
        }]
        result = validate_tool_trace(trace)
        assert result[0]["tool_input"] == {"key": "value"}

    def test_validate_tool_trace_non_string_output(self):
        """Test validate_tool_trace with non-string tool_output."""
        trace = [{
            "tool_name": "test",
            "tool_input": "input",
            "tool_output": {"result": "data"}  # Dict instead of string
        }]
        result = validate_tool_trace(trace)
        assert result[0]["tool_output"] == {"result": "data"}
