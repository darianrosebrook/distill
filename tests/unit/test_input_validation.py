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
