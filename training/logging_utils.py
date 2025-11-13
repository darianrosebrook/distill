"""
Structured logging utilities for training and inference.

Provides consistent logging format, levels, and structured data output.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json


class StructuredLogger:
    """Structured logger with JSON output and consistent formatting."""

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize logger.

        Args:
            name: Logger name (usually __name__)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Use structured formatter
        formatter = StructuredFormatter()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.propagate = False  # Don't propagate to root logger

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        self.logger.critical(message, extra=kwargs)

    def add_file_handler(self, log_file: Path, level: int = logging.INFO) -> None:
        """Add file handler for persistent logging.

        Args:
            log_file: Path to log file
            level: Logging level for file handler
        """
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = StructuredFormatter()
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract structured data from record
        structured_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add any extra fields from the record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                ]:
                    structured_data[key] = value

        return json.dumps(structured_data, default=str)


# Global logger instance for training
training_logger = StructuredLogger("training")


def setup_training_logging(
    log_dir: Optional[Path] = None, level: int = logging.INFO
) -> StructuredLogger:
    """Setup training logging with optional file output.

    Args:
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    global training_logger
    training_logger = StructuredLogger("training", level)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        training_logger.add_file_handler(log_file, level)

    return training_logger


def log_training_step(
    step: int,
    loss: float,
    lr: float,
    tokens_per_sec: Optional[float] = None,
    gpu_memory_mb: Optional[float] = None,
) -> None:
    """Log training step metrics.

    Args:
        step: Current training step
        loss: Current loss value
        lr: Current learning rate
        tokens_per_sec: Tokens processed per second
        gpu_memory_mb: GPU memory usage in MB
    """
    training_logger.info(
        "Training step completed",
        step=step,
        loss=loss,
        learning_rate=lr,
        tokens_per_sec=tokens_per_sec,
        gpu_memory_mb=gpu_memory_mb,
    )


def log_validation_metrics(step: int, metrics: Dict[str, Any]) -> None:
    """Log validation metrics.

    Args:
        step: Current training step
        metrics: Validation metrics dictionary
    """
    training_logger.info("Validation completed", step=step, **metrics)


def log_checkpoint_saved(step: int, checkpoint_path: Path, loss: float) -> None:
    """Log checkpoint saving.

    Args:
        step: Training step when checkpoint was saved
        checkpoint_path: Path to saved checkpoint
        loss: Current loss value
    """
    training_logger.info(
        "Checkpoint saved",
        step=step,
        checkpoint_path=str(checkpoint_path),
        loss=loss,
    )


def log_error(
    message: str, error: Optional[Exception] = None, step: Optional[int] = None, **kwargs
) -> None:
    """Log error with context.

    Args:
        message: Error message
        error: Exception object
        step: Current training step
        **kwargs: Additional context
    """
    error_context = {
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "step": step,
        **kwargs,
    }

    training_logger.error(message, **error_context)
