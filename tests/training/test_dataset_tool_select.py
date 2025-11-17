"""
Unit tests for training/dataset_tool_select.py

Tests ToolSelectDataset for tool selection and argument synthesis training.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from training.dataset_tool_select import ToolSelectDataset
from .conftest_mock_utils import create_mock_tokenizer_subscriptable


class TestToolSelectDataset:
    """Test ToolSelectDataset functionality."""

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_init_basic(self, mock_load_tokenizer):
        """Test basic ToolSelectDataset initialization."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create test data
        test_data = [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search", "description": "Search the web"}],
                    "history": [{"role": "user", "content": "Search for python"}]
                },
                "target": {
                    "name": "search",
                    "arguments": {"query": "python"}
                }
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(
                data_path=temp_path,
                tokenizer_path="test_tokenizer",
                max_seq_length=512,
                max_target_length=128
            )

            assert len(dataset) == 1
            assert dataset.data_path == Path(temp_path)
            assert dataset.tokenizer == mock_tokenizer
            assert dataset.max_seq_length == 512
            assert dataset.max_target_length == 128
            assert len(dataset.examples) == 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_multiple_examples(self, mock_load_tokenizer):
        """Test ToolSelectDataset with multiple examples."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search"}],
                    "history": [{"role": "user", "content": "Search for python"}]
                },
                "target": {"name": "search", "arguments": {"query": "python"}}
            },
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "calculate"}],
                    "history": [{"role": "user", "content": "Calculate 2+2"}]
                },
                "target": {"name": "calculate", "arguments": {"expression": "2+2"}}
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 2

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_getitem_basic(self, mock_load_tokenizer):
        """Test basic __getitem__ functionality."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search", "description": "Search the web"}],
                "history": [{"role": "user", "content": "Search for python"}]
            },
            "target": {
                "name": "search",
                "arguments": {"query": "python"}
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Check basic structure
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["attention_mask"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_complex_tools(self, mock_load_tokenizer):
        """Test with complex tool definitions."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant with multiple tools.",
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"}
                            }
                        }
                    },
                    {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string", "description": "Math expression"}
                            }
                        }
                    }
                ],
                "history": [
                    {"role": "user", "content": "Search for python and calculate 2+2"}
                ]
            },
            "target": {
                "name": "search",
                "arguments": {"query": "python"}
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle complex tool definitions
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_empty_arguments(self, mock_load_tokenizer):
        """Test with tool that has no arguments."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "ping", "description": "Check connectivity"}],
                "history": [{"role": "user", "content": "Ping the server"}]
            },
            "target": {
                "name": "ping",
                "arguments": {}  # No arguments
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_truncation(self, mock_load_tokenizer):
        """Test input and target truncation."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        # Mock very long sequences
        long_input_tokens = list(range(1000))  # Very long input
        long_target_tokens = list(range(500))  # Very long target
        mock_tokenizer.encode.side_effect = [
            long_input_tokens, long_target_tokens]
        mock_tokenizer.encode_plus.return_value = {
            "input_ids": long_input_tokens[:512],  # Truncated
            "attention_mask": [1] * 512
        }
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [{"role": "user", "content": "Very long query " * 100}]
            },
            "target": {
                "name": "search",
                "arguments": {"query": "very long query " * 50}
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(
                temp_path, "test_tokenizer",
                max_seq_length=512, max_target_length=128
            )
            item = dataset[0]

            # Should be truncated to max lengths
            assert len(item["input_ids"]) <= 512
            assert len(item["attention_mask"]) <= 512
            assert len(item["labels"]) <= 128

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_empty_history(self, mock_load_tokenizer):
        """Test with empty conversation history."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": []  # Empty history
            },
            "target": {
                "name": "search",
                "arguments": {"query": "test"}
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should still work with empty history
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_only_valid_data(self, mock_load_tokenizer):
        """Test loading only valid data."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Only valid data (ToolSelectDataset doesn't skip malformed JSON)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Valid
            f.write('{"input": {"system": "test", "tools": [{"name": "search"}], "history": []}, "target": {"name": "search", "arguments": {}}}\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            # Should load the valid line
            assert len(dataset) == 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_tool_select.load_tokenizer")
    def test_tool_select_dataset_large_dataset(self, mock_load_tokenizer):
        """Test with larger dataset."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create 50 examples
        test_data = []
        for i in range(50):
            test_data.append({
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": f"tool_{i}"}],
                    "history": [{"role": "user", "content": f"Use tool {i}"}]
                },
                "target": {
                    "name": f"tool_{i}",
                    "arguments": {"param": f"value_{i}"}
                }
            })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = ToolSelectDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 50

            # Test random access
            item = dataset[25]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()
