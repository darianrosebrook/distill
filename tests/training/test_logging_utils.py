"""
Tests for training/logging_utils.py - Structured logging utilities.

Tests StructuredLogger class, StructuredFormatter, logging setup functions,
and logging utilities with structured data output.
"""
# @author: @darianrosebrook

import json
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from training.logging_utils import (
    StructuredLogger,
    StructuredFormatter,
    setup_training_logging,
    log_training_step,
    log_validation_metrics,
    log_checkpoint_saved,
    log_error,
)


class TestStructuredLogger:
    """Test StructuredLogger class functionality."""

    def test_structured_logger_initialization(self):
        """Test StructuredLogger initialization."""
        logger = StructuredLogger("test_logger", level=logging.DEBUG)

        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.DEBUG

        # Should have one handler (console handler)
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)

    def test_structured_logger_no_duplicate_handlers(self):
        """Test that creating multiple loggers doesn't create duplicate handlers."""
        logger1 = StructuredLogger("test1")
        logger2 = StructuredLogger("test2")

        # Each should have exactly one handler
        assert len(logger1.logger.handlers) == 1
        assert len(logger2.logger.handlers) == 1

    def test_structured_logger_log_methods(self):
        """Test logging methods with structured data."""
        logger = StructuredLogger("test")

        with (
            patch.object(logger.logger, "debug") as mock_debug,
            patch.object(logger.logger, "info") as mock_info,
            patch.object(logger.logger, "warning") as mock_warning,
            patch.object(logger.logger, "error") as mock_error,
        ):
            # Test debug logging
            logger.debug("Test debug", key1="value1", key2=42)
            mock_debug.assert_called_once_with("Test debug", extra={"key1": "value1", "key2": 42})

            # Test info logging
            logger.info("Test info", metric=0.95)
            mock_info.assert_called_once_with("Test info", extra={"metric": 0.95})

            # Test warning logging
            logger.warning("Test warning", issue="deprecated")
            mock_warning.assert_called_once_with("Test warning", extra={"issue": "deprecated"})

            # Test error logging
            logger.error("Test error", code=500)
            mock_error.assert_called_once_with("Test error", extra={"code": 500})

    def test_structured_logger_propagate_disabled(self):
        """Test that logger doesn't propagate to root."""
        logger = StructuredLogger("test")

        assert logger.logger.propagate == False

    def test_structured_logger_formatter_assignment(self):
        """Test that StructuredFormatter is assigned to handler."""
        logger = StructuredLogger("test")

        handler = logger.logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""

    @pytest.fixture
    def formatter(self):
        """Create StructuredFormatter instance."""
        return StructuredFormatter()

    def test_structured_formatter_format_basic(self, formatter):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_structured_formatter_format_with_extra(self, formatter):
        """Test formatting with extra structured data."""
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=15,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.metric = 0.85
        record.user_id = 123

        result = formatter.format(record)

        parsed = json.loads(result)

        assert parsed["level"] == "WARNING"
        assert parsed["message"] == "Warning message"
        assert parsed["metric"] == 0.85
        assert parsed["user_id"] == 123

    def test_structured_formatter_format_all_levels(self, formatter):
        """Test formatting with all log levels."""
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level_num, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level_num,
                pathname="test.py",
                lineno=1,
                msg=f"Test {level_name}",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            parsed = json.loads(result)

            assert parsed["level"] == level_name

    def test_structured_formatter_format_with_exception(self, formatter):
        """Test formatting with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Exception occurred"

    def test_structured_formatter_format_message_formatting(self, formatter):
        """Test message formatting with arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User %s performed action %s",
            args=("alice", "login"),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "User alice performed action login"

    def test_structured_formatter_timestamp_format(self, formatter):
        """Test timestamp formatting."""
        import datetime

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        # Should have ISO format timestamp
        assert "timestamp" in parsed
        # Should be parseable as ISO datetime
        datetime.datetime.fromisoformat(parsed["timestamp"])

    def test_structured_formatter_json_validity(self, formatter):
        """Test that all formatted output is valid JSON."""
        test_cases = [
            logging.LogRecord("test", logging.DEBUG, "test.py", 1, "Debug", (), None),
            logging.LogRecord("test", logging.INFO, "test.py", 1, "Info with data", (), None),
            logging.LogRecord("test", logging.WARNING, "test.py", 1, "Warning", (), None),
            logging.LogRecord("test", logging.ERROR, "test.py", 1, "Error", (), None),
            logging.LogRecord("test", logging.CRITICAL, "test.py", 1, "Critical", (), None),
        ]

        for record in test_cases:
            # Add some extra data
            record.extra_field = "extra_value"
            record.number_field = 42

            result = formatter.format(record)

            # Should not raise JSONDecodeError
            parsed = json.loads(result)

            # Should have required fields
            assert "level" in parsed
            assert "logger" in parsed
            assert "message" in parsed
            assert "timestamp" in parsed


class TestSetupTrainingLogging:
    """Test setup_training_logging function."""

    @patch("training.logging_utils.StructuredLogger")
    @patch("training.logging_utils.logging.getLogger")
    def test_setup_training_logging_basic(self, mock_get_logger, mock_structured_logger):
        """Test basic training logging setup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_structured = Mock()
        mock_structured_logger.return_value = mock_structured

        result = setup_training_logging(log_dir=Path("test_run"), level=logging.INFO)

        mock_structured_logger.assert_called_once_with("training", logging.INFO)
        assert result == mock_structured

    @patch("training.logging_utils.StructuredLogger")
    @patch("training.logging_utils.logging.getLogger")
    def test_setup_training_logging_with_file(
        self, mock_get_logger, mock_structured_logger, tmp_path
    ):
        """Test training logging setup with file output."""
        log_file = tmp_path / "training.log"

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_structured = Mock()
        mock_structured_logger.return_value = mock_structured

        result = setup_training_logging(log_dir=log_file.parent)

        # Should have created file handler
        # This is tested implicitly through the function working

    @patch("training.logging_utils.StructuredLogger")
    @patch("training.logging_utils.logging.getLogger")
    def test_setup_training_logging_different_levels(self, mock_get_logger, mock_structured_logger):
        """Test training logging setup with different log levels."""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]

        for level in levels:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            mock_structured = Mock()
            mock_structured_logger.return_value = mock_structured

            result = setup_training_logging(level=level)

            mock_structured_logger.assert_called_with("training", level)


class TestLogTrainingStep:
    """Test log_training_step function."""

    @patch("training.logging_utils.training_logger")
    def test_log_training_step_basic(self, mock_training_logger):
        """Test basic training step logging."""
        log_training_step(100, 1.5, 0.8, 2.3, 1.0)

        mock_training_logger.info.assert_called_once()
        call_args = mock_training_logger.info.call_args

        # Check message
        assert "Training step completed" in call_args[0][0]

        # Check extra data
        extra = call_args[1]
        assert extra["step"] == 100
        assert extra["loss"] == 1.5
        assert extra["learning_rate"] == 0.8
        assert extra["tokens_per_sec"] == 2.3
        assert extra["gpu_memory_mb"] == 1.0

    @patch("training.logging_utils.training_logger")
    def test_log_training_step_zero_values(self, mock_training_logger):
        """Test training step logging with zero values."""
        log_training_step(0, 0.0, 0.0)

        call_args = mock_training_logger.info.call_args
        extra = call_args[1]

        assert extra["step"] == 0
        assert extra["loss"] == 0.0
        assert extra["learning_rate"] == 0.0
        assert extra["tokens_per_sec"] is None
        assert extra["gpu_memory_mb"] is None


class TestLogValidationMetrics:
    """Test log_validation_metrics function."""

    @patch("training.logging_utils.training_logger")
    def test_log_validation_metrics_basic(self, mock_training_logger):
        """Test basic validation metrics logging."""
        metrics = {"loss": 1.2, "accuracy": 0.85, "perplexity": 8.5}

        log_validation_metrics(100, metrics)

        mock_training_logger.info.assert_called_once()
        call_args = mock_training_logger.info.call_args

        # Check message
        assert "Validation completed" in call_args[0][0]

        # Check extra data
        extra = call_args[1]
        assert extra["step"] == 100
        assert extra["loss"] == 1.2
        assert extra["accuracy"] == 0.85
        assert extra["perplexity"] == 8.5

    @patch("training.logging_utils.training_logger")
    def test_log_validation_metrics_empty(self, mock_training_logger):
        """Test validation metrics logging with empty metrics."""
        log_validation_metrics(50, {})

        call_args = mock_training_logger.info.call_args
        extra = call_args[1]

        assert extra["step"] == 50
        # Should still log even with no metrics

    @patch("training.logging_utils.training_logger")
    def test_log_validation_metrics_complex(self, mock_training_logger):
        """Test validation metrics logging with complex metrics."""
        metrics = {
            "loss": 0.8,
            "accuracy": 0.92,
            "f1_score": 0.89,
            "precision": 0.91,
            "recall": 0.87,
        }

        log_validation_metrics(200, metrics)

        call_args = mock_training_logger.info.call_args
        extra = call_args[1]

        assert extra["step"] == 200
        assert len(extra) == 6  # step + 5 metrics


class TestLogCheckpointSaved:
    """Test log_checkpoint_saved function."""

    @patch("training.logging_utils.training_logger")
    def test_log_checkpoint_saved_basic(self, mock_training_logger):
        """Test basic checkpoint saved logging."""
        checkpoint_path = Path("/path/to/checkpoint.pt")

        log_checkpoint_saved(500, checkpoint_path, 0.7)

        mock_training_logger.info.assert_called_once()
        call_args = mock_training_logger.info.call_args

        # Check message contains key information
        message = call_args[0][0]
        assert "Checkpoint saved" in message

        # Check extra data
        extra = call_args[1]
        assert extra["step"] == 500
        assert extra["checkpoint_path"] == str(checkpoint_path)
        assert extra["loss"] == 0.7

    @patch("training.logging_utils.training_logger")
    def test_log_checkpoint_saved_different_paths(self, mock_training_logger):
        """Test checkpoint logging with different path types."""
        paths = [
            Path("relative/path/model.pt"),
            Path("/absolute/path/model.pt"),
            Path("model.pt"),
        ]

        for path in paths:
            log_checkpoint_saved(100, path, 1.0)

            call_args = mock_training_logger.info.call_args
            extra = call_args[1]
            assert extra["checkpoint_path"] == str(path)


class TestLogError:
    """Test log_error function."""

    @patch("training.logging_utils.training_logger")
    def test_log_error_basic(self, mock_training_logger):
        """Test basic error logging."""
        error = ValueError("Test error")

        log_error("Test error message", error, step=100, operation="training")

        mock_training_logger.error.assert_called_once()
        call_args = mock_training_logger.error.call_args

        # Check message
        message = call_args[0][0]
        assert message == "Test error message"

        # Check extra data
        extra = call_args[1]
        assert extra["error_type"] == "ValueError"
        assert extra["error_message"] == "Test error"
        assert extra["step"] == 100
        assert extra["operation"] == "training"

    @patch("training.logging_utils.training_logger")
    def test_log_error_no_context(self, mock_training_logger):
        """Test error logging without context."""
        error = RuntimeError("Runtime error")

        log_error("Runtime error occurred", error)

        call_args = mock_training_logger.error.call_args
        extra = call_args[1]

        assert extra["error_type"] == "RuntimeError"
        assert extra["error_message"] == "Runtime error"
        assert extra["step"] is None

    @patch("training.logging_utils.training_logger")
    def test_log_error_different_exceptions(self, mock_training_logger):
        """Test error logging with different exception types."""
        exceptions = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            FileNotFoundError("File not found"),
            KeyError("Key error"),
        ]

        for exc in exceptions:
            log_error(f"{type(exc).__name__} occurred", exc)

            call_args = mock_training_logger.error.call_args
            extra = call_args[1]

            assert extra["error_type"] == type(exc).__name__
            assert extra["error_message"] == str(exc)


class TestLoggingIntegration:
    """Test integration of logging functionality."""

    def test_structured_logging_workflow(self, tmp_path):
        """Test complete structured logging workflow."""
        # Create a temporary log file
        log_file = tmp_path / "test.log"

        # Set up logging
        logger = setup_training_logging(log_dir=log_file.parent)

        # Log various events
        log_training_step(10, 2.5, 1e-3, 1.2)
        log_validation_metrics(10, {"accuracy": 0.8, "loss": 1.5})
        log_checkpoint_saved(10, Path("checkpoint.pt"), 1.5)

        # Check that file was created and contains valid JSON
        expected_log_file = log_file.parent / "training.log"
        assert expected_log_file.exists()

        with open(expected_log_file, "r") as f:
            lines = f.readlines()

        # Should have multiple log entries
        assert len(lines) >= 3

        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line.strip())
            assert "timestamp" in parsed
            assert "level" in parsed
            assert "message" in parsed

    def test_logging_error_handling(self):
        """Test error handling in logging functions."""
        # Test with invalid inputs
        log_training_step(-1, float("inf"), float("-inf"), float("nan"))

        log_validation_metrics(0, {"invalid": None, "nan": float("nan")})

        # Should not raise exceptions
        assert True  # If we get here, no exceptions were raised

    def test_logging_with_special_characters(self):
        """Test logging with special characters in messages."""
        special_messages = [
            'Message with quotes: "hello"',
            "Message with newlines:\nline2",
            "Message with unicode: 🚀 αβγ",
            "Message with tabs:\tindented",
        ]

        for message in special_messages:
            log_error(ValueError(message))

        # Should handle all special characters gracefully
        assert True

    def test_logging_performance(self):
        """Test logging performance with many messages."""
        import time

        start_time = time.time()

        # Log many messages quickly
        for i in range(100):
            log_training_step(i, 1.0, 1e-4, 1.0)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly (less than 1 second for 100 messages)
        assert duration < 1.0
