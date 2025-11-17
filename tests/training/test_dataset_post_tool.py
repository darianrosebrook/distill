"""
Unit tests for training/dataset_post_tool.py

Tests PostToolDataset for post-tool answer generation training.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from training.dataset_post_tool import PostToolDataset
from .conftest_mock_utils import create_mock_tokenizer_subscriptable


class TestPostToolDataset:
    """Test PostToolDataset functionality."""

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_init_basic(self, mock_load_tokenizer):
        """Test basic PostToolDataset initialization."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create test data
        test_data = [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search", "description": "Search the web"}],
                    "history": [
                        {"role": "user", "content": "Search for python"},
                        {"role": "assistant", "content": "I'll search for python.", "tool_calls": [
                            {"name": "search", "arguments": {"query": "python"}}]},
                        {"role": "tool", "content": "Python is a programming language created by Guido van Rossum."}
                    ]
                },
                "target": {
                    "text": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991."
                }
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(
                data_path=temp_path,
                tokenizer_path="test_tokenizer",
                max_seq_length=512
            )

            assert len(dataset) == 1
            assert dataset.data_path == Path(temp_path)
            assert dataset.tokenizer == mock_tokenizer
            assert dataset.max_seq_length == 512
            assert len(dataset.examples) == 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_multiple_examples(self, mock_load_tokenizer):
        """Test PostToolDataset with multiple examples."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "calculate"}],
                    "history": [
                        {"role": "user", "content": "Calculate 15 * 7"},
                        {"role": "assistant", "content": "Let me calculate that for you.", "tool_calls": [
                            {"name": "calculate", "arguments": {"expression": "15*7"}}]},
                        {"role": "tool", "content": "Result: 105"}
                    ]
                },
                "target": {
                    "text": "15 multiplied by 7 equals 105."}
            },
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search"}],
                    "history": [
                        {"role": "user", "content": "What is the capital of France?"},
                        {"role": "assistant", "content": "I'll look that up for you.", "tool_calls": [
                            {"name": "search", "arguments": {"query": "capital of France"}}]},
                        {"role": "tool", "content": "Paris is the capital and most populous city of France."}
                    ]
                },
                "target": {
                    "text": "The capital of France is Paris."}
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 2

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_getitem_basic(self, mock_load_tokenizer):
        """Test basic __getitem__ functionality."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant with multiple tools.",
                "tools": [
                    {"name": "search", "description": "Search the web"},
                    {"name": "calculate", "description": "Perform calculations"},
                    {"name": "format", "description": "Format results"}
                ],
                "history": [
                    {"role": "user", "content": "Search for python and calculate pi"},
                    {"role": "assistant", "content": "I'll search and calculate.", "tool_calls": [
                        {"name": "search", "arguments": {
                            "query": "python programming"}},
                        {"name": "calculate", "arguments": {
                            "expression": "3.14159"}}
                    ]},
                    {"role": "tool", "name": "search",
                        "content": "Python is a programming language"},
                    {"role": "tool", "name": "calculate",
                        "content": "Result: 3.14159"},
                    {"role": "assistant",
                        "content": "Now I'll format the combined result."},
                    {"role": "tool", "name": "format",
                        "content": "Python: programming language\\nPi: 3.14159"}
                ]
            },
            "target": {
                    "text": "Python is a programming language, and the value of π (pi) is approximately 3.14159."
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle complex tool interactions
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_tool_errors(self, mock_load_tokenizer):
        """Test with tool execution errors in history."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [
                    {"role": "user", "content": "Search for nonexistent item"},
                    {"role": "assistant", "content": "Let me search for that.", "tool_calls": [
                        {"name": "search", "arguments": {"query": "nonexistent"}}]},
                    {"role": "tool", "content": "Error: No results found for 'nonexistent'"}
                ]
            },
            "target": {
                    "text": "I'm sorry, but I couldn't find any information about 'nonexistent'. Could you please try a different search term?"
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle tool errors gracefully
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_truncation(self, mock_load_tokenizer):
        """Test input and target truncation."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [{"role": "user", "content": "Very long question " * 100}]
            },
            "target": {
                "text": "Very long answer " * 50
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(
                temp_path, "test_tokenizer",
                max_seq_length=512
            )
            item = dataset[0]

            # Should be truncated to max lengths
            assert len(item["input_ids"]) <= 512
            assert len(item["attention_mask"]) <= 512

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_empty_tools(self, mock_load_tokenizer):
        """Test with empty tools list (shouldn't happen but test robustness)."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [],  # No tools (unusual but test robustness)
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            },
            "target": {
                    "text": "Hello! How can I help you?"
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should work even with no tools
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_string_target(self, mock_load_tokenizer):
        """Test with string target instead of dict."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [{"role": "user", "content": "What is AI?"}]
            },
            "target": {"text": "Artificial Intelligence is a field of computer science that focuses on creating intelligent machines."}
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle dict target
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_only_valid_data(self, mock_load_tokenizer):
        """Test loading only valid data."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Only valid data (PostToolDataset doesn't skip malformed JSON)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Valid
            f.write('{"input": {"system": "test", "tools": [{"name": "search"}], "history": [{"role": "user", "content": "hi"}]}, "target": {"text": "hello"}}\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            # Should load the valid line
            assert len(dataset) == 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_post_tool.load_tokenizer")
    def test_post_tool_dataset_large_dataset(self, mock_load_tokenizer):
        """Test with larger dataset."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create 25 examples
        test_data = []
        for i in range(25):
            test_data.append({
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "tool"}],
                    "history": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Processing {i}", "tool_calls": [
                            {"name": "tool", "arguments": {"param": f"value_{i}"}}]},
                        {"role": "tool", "content": f"Result {i}"}
                    ]
                },
                "target": {
                    "text": f"Answer {i}"
                }
            })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = PostToolDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 25

            # Test random access
            item = dataset[12]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()
