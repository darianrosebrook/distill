"""
Tests for training dataset modules - answer generation, post-tool, and tool selection.

Tests data loading, tokenization, batch collation, and error handling for all three
dataset types used in different training stages.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch

from training.dataset_answer_generation import (
    AnswerGenerationDataset,
    collate_answer_generation_batch,
)
from training.dataset_post_tool import (
    PostToolDataset,
    collate_post_tool_batch,
)
from training.dataset_tool_select import (
    ToolSelectDataset,
    collate_tool_select_batch,
)


class TestAnswerGenerationDataset:
    """Test answer generation dataset functionality."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for answer generation."""
        return [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [
                        {
                            "name": "calculator",
                            "schema": {
                                "type": "object",
                                "properties": {"expression": {"type": "string"}},
                            },
                        }
                    ],
                    "history": [
                        {"role": "user", "content": "What is 2+2?"},
                        {
                            "role": "assistant",
                            "content": "I need to calculate 2+2 using the calculator tool.",
                        },
                    ],
                    "tool_result": {"result": "4"},
                },
                "target": {"text": "The answer is 4."},
            },
            {
                "input": {
                    "system": "You are a coding assistant.",
                    "tools": [],
                    "history": [
                        {"role": "user", "content": "Write a function to add two numbers."}
                    ],
                },
                "target": {"text": "def add(a, b): return a + b"},
            },
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Mock tokenize methods to return tensors
        def mock_tokenize(text, **kwargs):
            # Simple mock: return tensor with length based on text
            seq_len = len(text.split()) + 2  # Add some padding
            vocab_size = 1000
            return {
                "input_ids": torch.randint(1, vocab_size, (1, seq_len)),
                "attention_mask": torch.ones(1, seq_len),
            }

        # Use side_effect to make the mock callable properly
        tokenizer.side_effect = mock_tokenize
        return tokenizer

    def test_dataset_initialization(self, sample_data, mock_tokenizer, tmp_path):
        """Test dataset initialization."""
        # Create temporary data file
        data_file = tmp_path / "answer_gen.jsonl"
        with open(data_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(
                data_path=str(data_file), tokenizer_path="dummy_path", max_seq_length=512
            )

            assert len(dataset) == 2
            assert dataset.max_seq_length == 512

    def test_dataset_getitem_basic(self, sample_data, mock_tokenizer, tmp_path):
        """Test basic __getitem__ functionality."""
        data_file = tmp_path / "answer_gen.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(sample_data[0]) + "\n")

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")

            item = dataset[0]

            # Check required keys
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert "target_text" in item

            # Check tensor shapes
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["attention_mask"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)

            # Check target text
            assert item["target_text"] == "The answer is 4."

    def test_dataset_prompt_formatting(self, mock_tokenizer, tmp_path):
        """Test prompt formatting logic."""
        data = {
            "input": {
                "system": "You are helpful.",
                "tools": [{"name": "calc", "schema": {"type": "object"}}],
                "history": [{"role": "user", "content": "Hi"}],
                "tool_result": {"result": "42"},
            },
            "target": {"text": "Answer"},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")

            # Check that prompt contains expected elements
            item = dataset[0]

            # Should have processed the data
            assert item is not None

    def test_dataset_empty_tools(self, mock_tokenizer, tmp_path):
        """Test dataset with empty tools list."""
        data = {
            "input": {
                "system": "You are helpful.",
                "tools": [],
                "history": [{"role": "user", "content": "Hi"}],
            },
            "target": {"text": "Answer"},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")

            item = dataset[0]
            assert item is not None

    def test_dataset_missing_fields(self, mock_tokenizer, tmp_path):
        """Test dataset handles missing optional fields."""
        data = {
            "input": {
                # Missing system, tools, history, tool_result
            },
            "target": {"text": "Answer"},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")

            item = dataset[0]
            assert item["target_text"] == "Answer"

    def test_dataset_file_not_found(self, mock_tokenizer):
        """Test dataset raises error for missing file."""
        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            with pytest.raises(FileNotFoundError):
                AnswerGenerationDataset("nonexistent.jsonl", "dummy_path")

    def test_collate_answer_generation_batch(self):
        """Test batch collation."""
        batch = [
            {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 100, (10,)),
                "target_text": "Answer 1",
            },
            {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 100, (10,)),
                "target_text": "Answer 2",
            },
        ]

        result = collate_answer_generation_batch(batch)

        # Check batched tensors
        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"].shape == (2, 10)
        assert result["labels"].shape == (2, 10)

        # Check metadata
        assert result["target_texts"] == ["Answer 1", "Answer 2"]


class TestPostToolDataset:
    """Test post-tool dataset functionality."""

    @pytest.fixture
    def post_tool_data(self):
        """Sample data for post-tool dataset."""
        return [
            {
                "input": {
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "search", "schema": {"type": "object"}}],
                    "history": [
                        {"role": "user", "content": "Search for Python docs"},
                        {
                            "role": "assistant",
                            "content": "I need to search for Python documentation.",
                        },
                    ],
                    "tool_result": {"results": ["Python docs found"]},
                },
                "target": {"text": "Based on the search results, here's what I found..."},
            }
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0

        def mock_tokenize(text, **kwargs):
            seq_len = len(text.split()) + 2
            vocab_size = 1000
            return {
                "input_ids": torch.randint(1, vocab_size, (1, seq_len)),
                "attention_mask": torch.ones(1, seq_len),
            }

        # Use side_effect to make the mock callable properly
        tokenizer.side_effect = mock_tokenize
        return tokenizer

    def test_dataset_initialization(self, post_tool_data, mock_tokenizer, tmp_path):
        """Test post-tool dataset initialization."""
        data_file = tmp_path / "post_tool.jsonl"
        with open(data_file, "w") as f:
            for item in post_tool_data:
                f.write(json.dumps(item) + "\n")

        with patch("training.dataset_post_tool.load_tokenizer", return_value=mock_tokenizer):
            dataset = PostToolDataset(str(data_file), "dummy_path", max_seq_length=2048)

            assert len(dataset) == 1
            assert dataset.max_seq_length == 2048

    def test_dataset_getitem(self, post_tool_data, mock_tokenizer, tmp_path):
        """Test post-tool dataset __getitem__."""
        data_file = tmp_path / "post_tool.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(post_tool_data[0]) + "\n")

        with patch("training.dataset_post_tool.load_tokenizer", return_value=mock_tokenizer):
            dataset = PostToolDataset(str(data_file), "dummy_path")

            item = dataset[0]

            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert "target_text" in item

            assert item["target_text"] == "Based on the search results, here's what I found..."

    def test_dataset_tool_result_formatting(self, mock_tokenizer, tmp_path):
        """Test tool result formatting."""
        data = {
            "input": {
                "system": "Test system",
                "tools": [],
                "history": [{"role": "user", "content": "test"}],
                "tool_result": {"status": "success", "data": [1, 2, 3]},
            },
            "target": {"text": "Response"},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch("training.dataset_post_tool.load_tokenizer", return_value=mock_tokenizer):
            dataset = PostToolDataset(str(data_file), "dummy_path")

            item = dataset[0]
            assert item is not None

    def test_collate_post_tool_batch(self):
        """Test post-tool batch collation."""
        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.randint(0, 100, (20,)),
                "target_text": "Response 1",
            },
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.randint(0, 100, (20,)),
                "target_text": "Response 2",
            },
        ]

        result = collate_post_tool_batch(batch)

        assert result["input_ids"].shape == (2, 20)
        assert result["attention_mask"].shape == (2, 20)
        assert result["labels"].shape == (2, 20)
        assert result["target_texts"] == ["Response 1", "Response 2"]


class TestToolSelectDataset:
    """Test tool selection dataset functionality."""

    @pytest.fixture
    def tool_select_data(self):
        """Sample data for tool selection."""
        return [
            {
                "input": {
                    "system": "You are a helpful assistant with tools.",
                    "tools": [
                        {
                            "name": "calculator",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "expression": {
                                        "type": "string",
                                        "description": "Math expression to evaluate",
                                    }
                                },
                            },
                        },
                        {
                            "name": "search",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query"}
                                },
                            },
                        },
                    ],
                    "history": [{"role": "user", "content": "What is 15 * 7?"}],
                },
                "target": {"name": "calculator", "arguments": {"expression": "15 * 7"}},
            }
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0

        def mock_tokenize(text, **kwargs):
            seq_len = len(text.split()) + 2
            vocab_size = 1000
            return {
                "input_ids": torch.randint(1, vocab_size, (1, seq_len)),
                "attention_mask": torch.ones(1, seq_len),
            }

        # Use side_effect to make the mock callable properly
        tokenizer.side_effect = mock_tokenize
        return tokenizer

    def test_dataset_initialization(self, tool_select_data, mock_tokenizer, tmp_path):
        """Test tool selection dataset initialization."""
        data_file = tmp_path / "tool_select.jsonl"
        with open(data_file, "w") as f:
            for item in tool_select_data:
                f.write(json.dumps(item) + "\n")

        with patch("training.dataset_tool_select.load_tokenizer", return_value=mock_tokenizer):
            dataset = ToolSelectDataset(
                str(data_file), "dummy_path", max_seq_length=1024, max_target_length=256
            )

            assert len(dataset) == 1
            assert dataset.max_seq_length == 1024
            assert dataset.max_target_length == 256

    def test_dataset_getitem(self, tool_select_data, mock_tokenizer, tmp_path):
        """Test tool selection dataset __getitem__."""
        data_file = tmp_path / "tool_select.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(tool_select_data[0]) + "\n")

        with patch("training.dataset_tool_select.load_tokenizer", return_value=mock_tokenizer):
            dataset = ToolSelectDataset(str(data_file), "dummy_path")

            item = dataset[0]

            # Check required keys
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert "tool_name" in item
            assert "tool_arguments" in item
            assert "tool_call_json" in item

            # Check values
            assert item["tool_name"] == "calculator"
            assert item["tool_arguments"] == {"expression": "15 * 7"}
            assert isinstance(item["tool_call_json"], str)

    def test_dataset_tool_formatting(self, mock_tokenizer, tmp_path):
        """Test tool manifest formatting."""
        data = {
            "input": {
                "system": "Test system",
                "tools": [
                    {
                        "name": "test_tool",
                        "schema": {"type": "object", "properties": {"arg": {"type": "string"}}},
                    }
                ],
                "history": [{"role": "user", "content": "test"}],
            },
            "target": {"name": "test_tool", "arguments": {"arg": "value"}},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch("training.dataset_tool_select.load_tokenizer", return_value=mock_tokenizer):
            dataset = ToolSelectDataset(str(data_file), "dummy_path")

            item = dataset[0]

            # Should have formatted the tool correctly
            assert item["tool_name"] == "test_tool"
            assert item["tool_arguments"]["arg"] == "value"

    def test_dataset_empty_tools(self, mock_tokenizer, tmp_path):
        """Test dataset with empty tools list."""
        data = {
            "input": {
                "system": "Test system",
                "tools": [],
                "history": [{"role": "user", "content": "test"}],
            },
            "target": {"name": "manual_response", "arguments": {}},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch("training.dataset_tool_select.load_tokenizer", return_value=mock_tokenizer):
            dataset = ToolSelectDataset(str(data_file), "dummy_path")

            item = dataset[0]
            assert item["tool_name"] == "manual_response"

    def test_collate_tool_select_batch(self):
        """Test tool selection batch collation."""
        batch = [
            {
                "input_ids": torch.randint(0, 100, (50,)),
                "attention_mask": torch.ones(50),
                "labels": torch.randint(0, 100, (25,)),
                "tool_name": "calculator",
                "tool_arguments": {"expr": "2+2"},
                "tool_call_json": '{"name": "calculator", "arguments": {"expr": "2+2"}}',
            },
            {
                "input_ids": torch.randint(0, 100, (50,)),
                "attention_mask": torch.ones(50),
                "labels": torch.randint(0, 100, (25,)),
                "tool_name": "search",
                "tool_arguments": {"query": "python"},
                "tool_call_json": '{"name": "search", "arguments": {"query": "python"}}',
            },
        ]

        result = collate_tool_select_batch(batch)

        # Check tensor shapes
        assert result["input_ids"].shape == (2, 50)
        assert result["attention_mask"].shape == (2, 50)
        assert result["labels"].shape == (2, 25)

        # Check metadata
        assert result["tool_names"] == ["calculator", "search"]
        assert len(result["tool_arguments"]) == 2
        assert len(result["tool_call_jsons"]) == 2

    def test_dataset_json_formatting(self, mock_tokenizer, tmp_path):
        """Test JSON formatting for tool calls."""
        data = {
            "input": {
                "system": "Test",
                "tools": [],
                "history": [{"role": "user", "content": "test"}],
            },
            "target": {"name": "test_tool", "arguments": {"key": "value", "number": 42}},
        }

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps(data) + "\n")

        with patch("training.dataset_tool_select.load_tokenizer", return_value=mock_tokenizer):
            dataset = ToolSelectDataset(str(data_file), "dummy_path")

            item = dataset[0]

            # Check JSON formatting
            tool_call = json.loads(item["tool_call_json"])
            assert tool_call["name"] == "test_tool"
            assert tool_call["arguments"]["key"] == "value"
            assert tool_call["arguments"]["number"] == 42


class TestDatasetErrorHandling:
    """Test error handling across all dataset types."""

    def test_answer_generation_invalid_json(self, mock_tokenizer, tmp_path):
        """Test answer generation dataset with invalid JSON."""
        data_file = tmp_path / "invalid.jsonl"
        with open(data_file, "w") as f:
            f.write("invalid json line\n")
            f.write('{"valid": "json"}\n')

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")

            # Should skip invalid JSON and load valid one
            assert len(dataset) == 1

    def test_datasets_empty_file(self, mock_tokenizer, tmp_path):
        """Test datasets with empty file."""
        data_file = tmp_path / "empty.jsonl"
        data_file.touch()  # Create empty file

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")
            assert len(dataset) == 0

    def test_datasets_whitespace_only_lines(self, mock_tokenizer, tmp_path):
        """Test datasets handle whitespace-only lines."""
        data_file = tmp_path / "whitespace.jsonl"
        with open(data_file, "w") as f:
            f.write("\n")
            f.write("   \n")
            f.write("\t\t\n")
            f.write('{"valid": "data"}\n')

        with patch(
            "training.dataset_answer_generation.load_tokenizer", return_value=mock_tokenizer
        ):
            dataset = AnswerGenerationDataset(str(data_file), "dummy_path")
            assert len(dataset) == 1
