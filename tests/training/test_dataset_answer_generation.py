"""
Unit tests for training/dataset_answer_generation.py

Tests AnswerGenerationDataset for answer generation training.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from training.dataset_answer_generation import AnswerGenerationDataset
from .conftest_mock_utils import create_mock_tokenizer_subscriptable


class TestAnswerGenerationDataset:
    """Test AnswerGenerationDataset functionality."""

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_init_basic(self, mock_load_tokenizer):
        """Test basic AnswerGenerationDataset initialization."""
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
                        {"role": "assistant", "content": "I found information about Python.", "tool_calls": [
                            {"name": "search", "arguments": {"query": "python"}}]},
                        {"role": "tool", "content": "Python is a programming language."}
                    ]
                },
                "target": {
                    "text": "Python is a high-level programming language known for its simplicity and readability."
                }
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(
                data_path=temp_path,
                tokenizer_path="test_tokenizer",
                max_seq_length=512,
            )

            assert len(dataset) == 1
            assert dataset.data_path == Path(temp_path)
            assert dataset.tokenizer == mock_tokenizer
            assert dataset.max_seq_length == 512
            assert len(dataset.examples) == 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_multiple_examples(self, mock_load_tokenizer):
        """Test AnswerGenerationDataset with multiple examples."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "calculate"}],
                    "history": [
                        {"role": "user", "content": "Calculate 2+2"},
                        {"role": "assistant", "content": "Let me calculate that.", "tool_calls": [
                            {"name": "calculate", "arguments": {"expression": "2+2"}}]},
                        {"role": "tool", "content": "Result: 4"}
                    ]
                },
                "target": {"text": "2 + 2 equals 4."}
            },
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search"}],
                    "history": [
                        {"role": "user", "content": "What's the weather?"},
                        {"role": "assistant", "content": "I'll check the weather.", "tool_calls": [
                            {"name": "search", "arguments": {"query": "current weather"}}]},
                        {"role": "tool", "content": "Sunny, 75°F"}
                    ]
                },
                "target": {"text": "The weather is sunny with a temperature of 75°F."}
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 2

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_getitem_basic(self, mock_load_tokenizer):
        """Test basic __getitem__ functionality."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [
                    {"role": "user", "content": "Search for python"},
                    {"role": "assistant", "content": "Searching...", "tool_calls": [
                        {"name": "search", "arguments": {"query": "python"}}]},
                    {"role": "tool", "content": "Python info here"}
                ]
            },
            "target": {
                "text": "Python is a programming language."
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
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

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_complex_history(self, mock_load_tokenizer):
        """Test with complex conversation history."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant with multiple tools.",
                "tools": [
                    {"name": "search", "description": "Search the web"},
                    {"name": "calculate", "description": "Do math"},
                    {"name": "format", "description": "Format text"}
                ],
                "history": [
                    {"role": "user", "content": "Search for python and calculate 2+2"},
                    {"role": "assistant", "content": "I'll search and calculate.", "tool_calls": [
                        {"name": "search", "arguments": {"query": "python"}},
                        {"name": "calculate", "arguments": {"expression": "2+2"}}
                    ]},
                    {"role": "tool", "name": "search",
                        "content": "Python is a language"},
                    {"role": "tool", "name": "calculate", "content": "Result: 4"},
                    {"role": "assistant", "content": "Now I'll format the result."},
                    {"role": "tool", "name": "format",
                        "content": "Formatted: Python is a language. 2+2 = 4"}
                ]
            },
            "target": {
                "text": "Based on the search results and calculation, Python is a programming language and 2+2 equals 4."
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle complex history
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_no_tools(self, mock_load_tokenizer):
        """Test with no tools available."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [],  # No tools
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            },
            "target": {
                "text": "Hello! How can I help you today?"
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should work without tools
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_truncation(self, mock_load_tokenizer):
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
            dataset = AnswerGenerationDataset(
                temp_path, "test_tokenizer",
                max_seq_length=512
            )
            item = dataset[0]

            # Should be truncated to max lengths
            assert len(item["input_ids"]) <= 512
            assert len(item["attention_mask"]) <= 512

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_minimal_history(self, mock_load_tokenizer):
        """Test with minimal conversation history."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [
                    {"role": "user", "content": "Hello"}
                    # No assistant/tool responses
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
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle minimal history
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_answer_only(self, mock_load_tokenizer):
        """Test with answer-only target (no additional fields)."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        test_data = [{
            "input": {
                "system": "You are a helpful assistant.",
                "tools": [{"name": "search"}],
                "history": [{"role": "user", "content": "What is AI?"}]
            },
            "target": {
                "text": "Artificial Intelligence is a field of computer science."
            }
        }]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(test_data[0], f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should handle dict target
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_malformed_data(self, mock_load_tokenizer):
        """Test handling of malformed data."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mix of valid and invalid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"invalid": json}\n')  # Invalid JSON
            # Valid
            f.write(
                '{"input": {"system": "test", "tools": [{"name": "search"}], "history": [{"role": "user", "content": "hi"}]}, "target": {"text": "hello"}}\n')
            temp_path = f.name

        try:
            # Should handle malformed JSON gracefully
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            # Should load the valid line
            assert len(dataset) >= 1

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset_answer_generation.load_tokenizer")
    def test_answer_generation_dataset_large_dataset(self, mock_load_tokenizer):
        """Test with larger dataset."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create 30 examples
        test_data = []
        for i in range(30):
            test_data.append({
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "tool"}],
                    "history": [{"role": "user", "content": f"Question {i}"}]
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
            dataset = AnswerGenerationDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 30

            # Test random access
            item = dataset[15]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()
