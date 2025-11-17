"""
Pytest configuration and fixtures for training tests.

Provides reusable fixtures for test setup and mock objects.

Author: @darianrosebrook
"""

import pytest
from unittest.mock import Mock, patch
from .conftest_mock_utils import create_mock_tokenizer_subscriptable


@pytest.fixture
def mock_tokenizer():
    """Provide a properly configured mock tokenizer for tests."""
    return create_mock_tokenizer_subscriptable()


@pytest.fixture
def mock_safe_tokenizer_loader(mock_tokenizer):
    """Provide a mock safe_from_pretrained_tokenizer function."""
    with patch("training.safe_model_loading.safe_from_pretrained_tokenizer") as mock:
        mock.return_value = mock_tokenizer
        yield mock


@pytest.fixture
def mock_load_tokenizer(mock_tokenizer):
    """Provide a mock load_tokenizer function."""
    with patch("training.dataset_answer_generation.load_tokenizer") as mock:
        mock.return_value = mock_tokenizer
        yield mock


@pytest.fixture
def mock_load_tokenizer_post_tool(mock_tokenizer):
    """Provide a mock load_tokenizer function for post_tool tests."""
    with patch("training.dataset_post_tool.load_tokenizer") as mock:
        mock.return_value = mock_tokenizer
        yield mock


@pytest.fixture
def mock_load_tokenizer_tool_select(mock_tokenizer):
    """Provide a mock load_tokenizer function for tool_select tests."""
    with patch("training.dataset_tool_select.load_tokenizer") as mock:
        mock.return_value = mock_tokenizer
        yield mock


@pytest.fixture
def hf_tokenizer_available():
    """Patch HF_TOKENIZER_AVAILABLE to True."""
    with patch("training.dataset.HF_TOKENIZER_AVAILABLE", True):
        with patch("training.dataset_answer_generation.HF_TOKENIZER_AVAILABLE", True):
            with patch("training.dataset_post_tool.HF_TOKENIZER_AVAILABLE", True):
                with patch("training.dataset_tool_select.HF_TOKENIZER_AVAILABLE", True):
                    yield True

