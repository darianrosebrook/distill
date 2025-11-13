"""
Input validation and sanitization utilities.

Provides data validation, sanitization, and security checks for training data.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class InputValidator:
    """Validate and sanitize training inputs."""

    # Security patterns to detect
    SUSPICIOUS_PATTERNS = [
        r"<\s*script[^>]*>.*?</\s*script\s*>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>.*?</iframe>",  # Iframes
        r"<object[^>]*>.*?</object>",  # Object/embed tags
        r"<embed[^>]*>",  # Embed tags
    ]

    # Maximum lengths for various inputs
    MAX_PROMPT_LENGTH = 50000
    MAX_RESPONSE_LENGTH = 100000
    MAX_TOOL_COUNT = 50
    MAX_FILE_SIZE_MB = 100

    def __init__(self, strict_mode: bool = True):
        """Initialize input validator.

        Args:
            strict_mode: If True, raise errors on validation failures
        """
        self.strict_mode = strict_mode
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.SUSPICIOUS_PATTERNS
        ]

    def validate_text_input(self, text: str, field_name: str = "text") -> str:
        """Validate and sanitize text input.

        Args:
            text: Input text to validate
            field_name: Name of the field for error messages

        Returns:
            Sanitized text

        Raises:
            ValidationError: If validation fails and strict_mode is True
        """
        if not isinstance(text, str):
            error_msg = f"{field_name} must be a string, got {type(text)}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return str(text)

        # Check length
        if len(text) > getattr(self, f"MAX_{field_name.upper()}_LENGTH", self.MAX_PROMPT_LENGTH):
            error_msg = f"{field_name} too long: {len(text)} characters"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                text = text[
                    : getattr(self, f"MAX_{field_name.upper()}_LENGTH", self.MAX_PROMPT_LENGTH)
                ]

        # Check for suspicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                error_msg = f"Suspicious content detected in {field_name}"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                else:
                    print(f"WARNING: {error_msg}")

        return text

    def validate_numeric_input(
        self,
        value: Any,
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Union[int, float]:
        """Validate numeric input.

        Args:
            value: Value to validate
            field_name: Field name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated numeric value

        Raises:
            ValidationError: If validation fails
        """
        try:
            numeric_value = float(value) if not isinstance(value, (int, float)) else value
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be numeric, got {type(value)}")

        if min_val is not None and numeric_value < min_val:
            raise ValidationError(f"{field_name} must be >= {min_val}, got {numeric_value}")

        if max_val is not None and numeric_value > max_val:
            raise ValidationError(f"{field_name} must be <= {max_val}, got {numeric_value}")

        return numeric_value

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool call structure.

        Args:
            tool_call: Tool call dictionary to validate

        Returns:
            Validated tool call

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tool_call, dict):
            raise ValidationError("Tool call must be a dictionary")

        required_fields = ["name", "arguments"]
        for field in required_fields:
            if field not in tool_call:
                raise ValidationError(f"Tool call missing required field: {field}")

        # Validate name
        name = self.validate_text_input(tool_call["name"], "tool_name")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", name):
            raise ValidationError(f"Invalid tool name format: {name}")

        # Validate arguments
        if not isinstance(tool_call["arguments"], dict):
            raise ValidationError("Tool arguments must be a dictionary")

        return {"name": name, "arguments": tool_call["arguments"]}

    def validate_training_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete training example.

        Args:
            example: Training example dictionary

        Returns:
            Validated and sanitized example

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(example, dict):
            raise ValidationError("Training example must be a dictionary")

        validated = {}

        # Validate prompt
        if "prompt" in example:
            validated["prompt"] = self.validate_text_input(example["prompt"], "prompt")

        # Validate teacher_text
        if "teacher_text" in example:
            validated["teacher_text"] = self.validate_text_input(
                example["teacher_text"], "teacher_text"
            )

        # Validate cot_steps
        if "cot_steps" in example:
            if not isinstance(example["cot_steps"], list):
                raise ValidationError("cot_steps must be a list")
            validated["cot_steps"] = [
                self.validate_text_input(step, "cot_step") for step in example["cot_steps"]
            ]

        # Validate answer
        if "answer" in example:
            validated["answer"] = self.validate_text_input(example["answer"], "answer")

        # Validate metadata
        if "metadata" in example:
            validated["metadata"] = self.validate_metadata(example["metadata"])

        return validated

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata dictionary.

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            Validated metadata

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        validated = {}

        # Validate tool_count
        if "tool_count" in metadata:
            validated["tool_count"] = int(
                self.validate_numeric_input(
                    metadata["tool_count"], "tool_count", min_val=0, max_val=self.MAX_TOOL_COUNT
                )
            )

        # Validate intermediate_sizes
        if "intermediate_sizes" in metadata:
            if not isinstance(metadata["intermediate_sizes"], list):
                raise ValidationError("intermediate_sizes must be a list")

            validated["intermediate_sizes"] = [
                int(self.validate_numeric_input(size, "intermediate_size", min_val=0))
                for size in metadata["intermediate_sizes"]
            ]

        # Validate boolean flags
        boolean_fields = ["pii_tags_present", "eligible_for_code_mode"]
        for field in boolean_fields:
            if field in metadata:
                if not isinstance(metadata[field], bool):
                    raise ValidationError(f"{field} must be a boolean")
                validated[field] = metadata[field]

        return validated

    def validate_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a training batch.

        Args:
            batch: Batch dictionary to validate

        Returns:
            Validated batch

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(batch, dict):
            raise ValidationError("Batch must be a dictionary")

        validated = {}

        # Validate tensor shapes and types
        tensor_fields = ["input_ids", "labels", "attention_mask"]
        for field in tensor_fields:
            if field in batch:
                tensor = batch[field]
                if not hasattr(tensor, "shape"):
                    raise ValidationError(f"{field} must be a tensor")

                # Check for NaN/Inf values
                if hasattr(tensor, "isnan") and tensor.isnan().any():
                    raise ValidationError(f"{field} contains NaN values")

                if hasattr(tensor, "isinf") and tensor.isinf().any():
                    raise ValidationError(f"{field} contains infinite values")

                validated[field] = tensor

        return validated

    def sanitize_json_string(self, json_str: str) -> str:
        """Sanitize JSON string for safe parsing.

        Args:
            json_str: JSON string to sanitize

        Returns:
            Sanitized JSON string
        """
        # Remove any null bytes
        json_str = json_str.replace("\x00", "")

        # Limit length
        if len(json_str) > 1000000:  # 1MB limit
            json_str = json_str[:1000000]

        return json_str

    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate file path for security.

        Args:
            file_path: File path to validate
            must_exist: Whether file must exist

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        path = Path(file_path).resolve()

        # Check for directory traversal
        try:
            path.relative_to(path.parent)
        except ValueError:
            raise ValidationError(f"Invalid path: {file_path}")

        # Check file extension (basic security)
        dangerous_extensions = {".exe", ".bat", ".cmd", ".scr", ".pif", ".com"}
        if path.suffix.lower() in dangerous_extensions:
            raise ValidationError(f"Dangerous file extension: {path.suffix}")

        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")

        if path.exists() and path.is_file():
            # Check file size
            size_mb = path.stat().st_size / 1024 / 1024
            if size_mb > self.MAX_FILE_SIZE_MB:
                raise ValidationError(
                    f"File too large: {size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB"
                )

        return path


# Global validator instance
validator = InputValidator(strict_mode=True)


def validate_training_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and sanitize training data.

    Args:
        data: List of training examples

    Returns:
        Validated and sanitized data

    Raises:
        ValidationError: If validation fails
    """
    validated_data = []

    for i, example in enumerate(data):
        try:
            validated_example = validator.validate_training_example(example)
            validated_data.append(validated_example)
        except ValidationError as e:
            raise ValidationError(f"Example {i}: {e}")

    return validated_data


def validate_tool_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate tool execution trace.

    Args:
        trace: Tool execution trace

    Returns:
        Validated trace

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(trace, list):
        raise ValidationError("Tool trace must be a list")

    validated_trace = []

    for i, tool_call in enumerate(trace):
        try:
            validated_call = validator.validate_tool_call(tool_call)
            validated_trace.append(validated_call)
        except ValidationError as e:
            raise ValidationError(f"Tool call {i}: {e}")

    return validated_trace


if __name__ == "__main__":
    # Example usage and testing
    validator = InputValidator(strict_mode=True)

    # Test text validation
    try:
        clean_text = validator.validate_text_input("Hello world", "test_field")
        print(f"✅ Text validation passed: {clean_text}")
    except ValidationError as e:
        print(f"❌ Text validation failed: {e}")

    # Test numeric validation
    try:
        valid_num = validator.validate_numeric_input(42, "test_num", min_val=0, max_val=100)
        print(f"✅ Numeric validation passed: {valid_num}")
    except ValidationError as e:
        print(f"❌ Numeric validation failed: {e}")

    # Test tool call validation
    try:
        valid_tool = validator.validate_tool_call(
            {"name": "test_tool", "arguments": {"param": "value"}}
        )
        print(f"✅ Tool call validation passed: {valid_tool}")
    except ValidationError as e:
        print(f"❌ Tool call validation failed: {e}")
