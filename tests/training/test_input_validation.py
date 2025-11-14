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
        assert strict_validator.strict_mode == True
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
        with pytest.raises(ValidationError, match="too long"):
            strict_validator.validate_text_input(long_text, "prompt")

    def test_validate_text_input_too_long_lenient(self, lenient_validator):
        """Test validating text that exceeds max length in lenient mode."""
        long_text = "x" * (lenient_validator.MAX_PROMPT_LENGTH + 1)
        result = lenient_validator.validate_text_input(long_text, "prompt")
        assert len(result) == lenient_validator.MAX_PROMPT_LENGTH

    def test_validate_text_input_suspicious_script_tag(self, strict_validator):
        """Test detecting suspicious script tags."""
        malicious_text = '<script>alert("xss")</script>'
        with pytest.raises(ValidationError, match="Suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

    def test_validate_text_input_suspicious_javascript_url(self, strict_validator):
        """Test detecting suspicious JavaScript URLs."""
        malicious_text = 'Click here: <a href="javascript:alert(1)">link</a>'
        with pytest.raises(ValidationError, match="Suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

    def test_validate_text_input_suspicious_event_handler(self, strict_validator):
        """Test detecting suspicious event handlers."""
        malicious_text = '<div onclick="alert(1)">Click me</div>'
        with pytest.raises(ValidationError, match="Suspicious content"):
            strict_validator.validate_text_input(malicious_text, "test_field")

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
        """Test validating training example with invalid cot_steps."""
        example = {"prompt": "Test", "cot_steps": "not a list"}
        with pytest.raises(ValidationError, match="must be a list"):
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
        assert result["pii_tags_present"] == False

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
        """Test sanitizing JSON string that is too long."""
        long_json = '{"key": "' + "x" * 2000000 + '"}'
        result = strict_validator.sanitize_json_string(long_json)
        assert len(result) <= 1000000

    def test_validate_file_path_valid(self, strict_validator, tmp_path):
        """Test validating valid file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = strict_validator.validate_file_path(test_file, must_exist=True)
        assert isinstance(result, Path)
        assert result.exists()

    def test_validate_file_path_not_exist(self, strict_validator, tmp_path):
        """Test validating file path that doesn't exist."""
        non_existent = tmp_path / "nonexistent.txt"
        with pytest.raises(ValidationError, match="does not exist"):
            strict_validator.validate_file_path(non_existent, must_exist=True)

    def test_validate_file_path_not_exist_optional(self, strict_validator, tmp_path):
        """Test validating file path that doesn't exist (optional)."""
        non_existent = tmp_path / "nonexistent.txt"
        result = strict_validator.validate_file_path(non_existent, must_exist=False)
        assert isinstance(result, Path)

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

    def test_validate_training_data_empty_list(self):
        """Test validating empty training data."""
        result = validate_training_data([])
        assert result == []


class TestValidateToolTrace:
    """Test validate_tool_trace function."""

    def test_validate_tool_trace_valid(self):
        """Test validating valid tool trace."""
        trace = [
            {"name": "tool1", "arguments": {"param1": "value1"}},
            {"name": "tool2", "arguments": {"param2": "value2"}},
        ]
        result = validate_tool_trace(trace)
        assert len(result) == 2
        assert result[0]["name"] == "tool1"

    def test_validate_tool_trace_not_list(self):
        """Test validating tool trace that is not a list."""
        with pytest.raises(ValidationError, match="must be a list"):
            validate_tool_trace("not a list")

    def test_validate_tool_trace_invalid_tool_call(self):
        """Test validating tool trace with invalid tool call."""
        trace = [
            {"name": "tool1", "arguments": {"param1": "value1"}},
            {"invalid": "tool call"},
        ]
        with pytest.raises(ValidationError, match="Tool call 1"):
            validate_tool_trace(trace)

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
        assert validator.strict_mode == True

