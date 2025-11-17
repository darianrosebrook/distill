"""
Unit tests for training/dataset.py

Tests KD dataset loading, tokenization, and batch collation functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from training.dataset import (
    load_tokenizer,
    KDDataset,
    collate_kd_batch,
)
from .conftest_mock_utils import (
    create_mock_tokenizer_subscriptable,
    create_mock_encoded_output,
)


class TestLoadTokenizer:
    """Test tokenizer loading functionality."""

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_load_tokenizer_success(self, mock_safe_tokenizer):
        """Test successful tokenizer loading."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_safe_tokenizer.return_value = mock_tokenizer

        result = load_tokenizer("test_tokenizer_path")

        assert result == mock_tokenizer
        mock_safe_tokenizer.assert_called_once_with("test_tokenizer_path")
        assert result.pad_token == "[EOS]"  # Should set pad_token to eos_token

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_load_tokenizer_with_existing_pad_token(self, mock_safe_tokenizer):
        """Test tokenizer loading when pad_token already exists."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_safe_tokenizer.return_value = mock_tokenizer

        result = load_tokenizer("test_tokenizer_path")

        assert result == mock_tokenizer
        # Should not override existing pad_token
        assert result.pad_token == "[PAD]"

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", False)
    def test_load_tokenizer_transformers_not_available(self):
        """Test error when transformers library is not available."""
        with pytest.raises(RuntimeError, match="transformers library required"):
            load_tokenizer("test_tokenizer_path")


class TestKDDataset:
    """Test KDDataset class functionality."""

    def _setup_mock_tokenizer(self, mock_safe_tokenizer):
        """Set up mock tokenizer for tests - non-decorated helper."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_safe_tokenizer.return_value = mock_tokenizer
        return mock_tokenizer

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_basic(self, mock_safe_tokenizer):
        """Test basic KDDataset initialization."""
        mock_tokenizer = self._setup_mock_tokenizer(mock_safe_tokenizer)

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                "prompt": "test prompt",
                "teacher_text": "test teacher text"
            }, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(
                jsonl_path=temp_path,
                tokenizer_path="test_tokenizer",
                max_seq_length=512,
            )

            assert dataset.jsonl_path == Path(temp_path)
            assert dataset.tokenizer_path == "test_tokenizer"
            assert dataset.max_seq_length == 512
            assert dataset.teacher_logits_available is False
            assert dataset.latent_curriculum is None
            assert dataset.tokenizer == mock_tokenizer
            assert len(dataset.samples) == 1
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_with_teacher_logits(self, mock_safe_tokenizer):
        """Test KDDataset initialization with teacher logits available."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                "prompt": "test prompt",
                "teacher_text": "test teacher text",
                "teacher_logits": [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
            }, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(
                jsonl_path=temp_path,
                tokenizer_path="test_tokenizer",
                teacher_logits_available=True,
            )

            assert dataset.teacher_logits_available is True
            assert len(dataset.samples) == 1
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", False)
    def test_kd_dataset_init_no_transformers(self):
        """Test error when transformers not available."""
        with pytest.raises(RuntimeError, match="transformers library required"):
            KDDataset("dummy.jsonl", "dummy_tokenizer")

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_load_data_complex_sample(self, mock_safe_tokenizer):
        """Test loading complex sample with all supervision targets."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        complex_sample = {
            "prompt": "test prompt",
            "teacher_text": "test teacher text",
            "tool_name_ids": [100, 200, 300],
            "tool_name_mask": [1, 1, 0],
            "gold_json_text_ids": [400, 500],
            "mask_valid_json_tokens": [1, 0],
            "tool_result_fields": [600, 700, 800],
            "integration_mask": [1, 1, 1, 0, 0],
            "teacher_logits": [[0.1, 0.9], [0.2, 0.8]],
            "metadata": {"source": "test", "quality": 0.95}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(complex_sample, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            assert len(dataset.samples) == 1
            sample = dataset.samples[0]

            # Check all fields are preserved
            assert sample["prompt"] == "test prompt"
            assert sample["teacher_text"] == "test teacher text"
            assert sample["tool_name_ids"] == [100, 200, 300]
            assert sample["tool_name_mask"] == [1, 1, 0]
            assert sample["gold_json_text_ids"] == [400, 500]
            assert sample["mask_valid_json_tokens"] == [1, 0]
            assert sample["tool_result_fields"] == [600, 700, 800]
            assert sample["integration_mask"] == [1, 1, 1, 0, 0]
            assert sample["teacher_logits"] == [[0.1, 0.9], [0.2, 0.8]]
            assert sample["metadata"] == {"source": "test", "quality": 0.95}
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_len(self, mock_safe_tokenizer):
        """Test dataset length reporting."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        # Create file with 3 samples
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(3):
                json.dump({
                    "prompt": f"prompt {i}",
                    "teacher_text": f"teacher {i}"
                }, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")
            assert len(dataset) == 3
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_basic(self, mock_safe_tokenizer):
        """Test basic __getitem__ functionality."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                "prompt": "test prompt",
                "teacher_text": "test teacher text"
            }, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            item = dataset[0]

            # Check basic tensor outputs
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert "teacher_target_ids" in item

            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["attention_mask"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)
            assert isinstance(item["teacher_target_ids"], torch.Tensor)

            # Check shapes
            # Mock tokenizer returns 5 tokens
            assert item["input_ids"].shape == (5,)
            assert item["attention_mask"].shape == (5,)
            assert item["labels"].shape == (5,)
            assert item["teacher_target_ids"].shape == (5,)
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_supervision_targets(self, mock_safe_tokenizer):
        """Test __getitem__ with process-step supervision targets."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample = {
            "prompt": "test prompt",
            "teacher_text": "test teacher text",
            "tool_name_ids": [100, 200, 300],
            "tool_name_mask": [1, 1, 0],
            "gold_json_text_ids": [400, 500],
            "mask_valid_json_tokens": [1, 0],
            "tool_result_fields": [600, 700, 800],
            "integration_mask": [1, 1, 1, 0, 0],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            item = dataset[0]

            # Check supervision targets are included as tensors
            assert "tool_name_ids" in item
            assert "tool_name_mask" in item
            assert "gold_json_text_ids" in item
            assert "mask_valid_json_tokens" in item
            assert "tool_result_fields" in item
            assert "integration_mask" in item

            assert isinstance(item["tool_name_ids"], torch.Tensor)
            assert isinstance(item["tool_name_mask"], torch.Tensor)
            assert isinstance(item["gold_json_text_ids"], torch.Tensor)
            assert isinstance(item["mask_valid_json_tokens"], torch.Tensor)
            assert isinstance(item["tool_result_fields"], torch.Tensor)
            assert isinstance(item["integration_mask"], torch.Tensor)

            # Check values are preserved
            assert torch.equal(item["tool_name_ids"],
                               torch.tensor([100, 200, 300]))
            assert torch.equal(item["tool_name_mask"], torch.tensor([1, 1, 0]))
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_teacher_logits(self, mock_safe_tokenizer):
        """Test __getitem__ with teacher logits."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample = {
            "prompt": "test prompt",
            "teacher_text": "test teacher text",
            "teacher_logits": [[0.1, 0.9], [0.2, 0.8]],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer",
                                teacher_logits_available=True)

            item = dataset[0]

            assert "teacher_logits" in item
            assert isinstance(item["teacher_logits"], torch.Tensor)
            expected = torch.tensor(
                [[0.1, 0.9], [0.2, 0.8]], dtype=torch.float32)
            assert torch.allclose(item["teacher_logits"], expected)
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_sequence_truncation(self, mock_safe_tokenizer):
        """Test sequence truncation when exceeding max_seq_length."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        # Return a long sequence
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        mock_tokenizer.encode_plus.return_value = {
            "input_ids": list(range(100)),
            "attention_mask": [1] * 100,
        }
        mock_safe_tokenizer.return_value = mock_tokenizer

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                "prompt": "test prompt",
                "teacher_text": "test teacher text"
            }, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer", max_seq_length=50)

            item = dataset[0]

            # Should be truncated to max_seq_length
            assert item["input_ids"].shape[0] == 50
            assert item["attention_mask"].shape[0] == 50
            assert item["labels"].shape[0] == 50
            assert item["teacher_target_ids"].shape[0] == 50
        finally:
            Path(temp_path).unlink()


class TestCollateKDBatch:
    """Test batch collation functionality."""

    def test_collate_kd_batch_basic(self):
        """Test basic batch collation."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "teacher_target_ids": torch.tensor([4, 5, 6]),
            },
            {
                "input_ids": torch.tensor([7, 8, 9, 10]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([7, 8, 9, 10]),
                "teacher_target_ids": torch.tensor([11, 12, 13, 14]),
            }
        ]

        result = collate_kd_batch(batch)

        # Check basic fields are collated and padded
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert "teacher_target_ids" in result

        # Check shapes - should be padded to max length (4)
        assert result["input_ids"].shape == (2, 4)  # [batch_size, max_len]
        assert result["attention_mask"].shape == (2, 4)
        assert result["labels"].shape == (2, 4)
        assert result["teacher_target_ids"].shape == (2, 4)

        # Check padding values (assuming pad_token_id = 0 for simplicity in test)
        # First sequence should be [1, 2, 3, 0], second should be [7, 8, 9, 10]

    def test_collate_kd_batch_with_supervision_targets(self):
        """Test batch collation with process-step supervision targets."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "teacher_target_ids": torch.tensor([4, 5, 6]),
                "tool_name_ids": torch.tensor([100, 200]),
                "tool_name_mask": torch.tensor([1, 0]),
                "gold_json_text_ids": torch.tensor([300, 400, 500]),
                "mask_valid_json_tokens": torch.tensor([1, 1, 0]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([7, 8]),
                "teacher_target_ids": torch.tensor([9, 10]),
                "tool_name_ids": torch.tensor([600]),
                "tool_name_mask": torch.tensor([1]),
                "gold_json_text_ids": torch.tensor([700, 800]),
                "mask_valid_json_tokens": torch.tensor([1, 0]),
                "tool_result_fields": torch.tensor([900, 1000]),
                "integration_mask": torch.tensor([1, 0]),
            }
        ]

        result = collate_kd_batch(batch)

        # Check all supervision targets are included
        assert "tool_name_ids" in result
        assert "tool_name_mask" in result
        assert "gold_json_text_ids" in result
        assert "mask_valid_json_tokens" in result
        assert "tool_result_fields" in result
        assert "integration_mask" in result

        # Check shapes are properly padded
        assert result["tool_name_ids"].shape == (2, 2)  # max_len = 2
        assert result["gold_json_text_ids"].shape == (2, 3)  # max_len = 3

    def test_collate_kd_batch_with_teacher_logits(self):
        """Test batch collation with teacher logits."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
                "teacher_target_ids": torch.tensor([3, 4]),
                "teacher_logits": torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "teacher_target_ids": torch.tensor([7, 8]),
                "teacher_logits": torch.tensor([[0.3, 0.7], [0.4, 0.6]]),
            }
        ]

        result = collate_kd_batch(batch)

        assert "teacher_logits" in result
        assert result["teacher_logits"].shape == (
            2, 2, 2)  # [batch_size, seq_len, vocab_size]

    def test_collate_kd_batch_with_loss_mask(self):
        """Test batch collation with latent curriculum loss mask."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "teacher_target_ids": torch.tensor([4, 5, 6]),
                "loss_mask": torch.tensor([1.0, 0.5, 0.0]),
            }
        ]

        result = collate_kd_batch(batch)

        assert "loss_mask" in result
        assert result["loss_mask"].shape == (1, 3)

    def test_collate_kd_batch_empty_batch(self):
        """Test batch collation with empty batch."""
        batch = []

        # This would typically not happen in practice, but test edge case
        with pytest.raises(ValueError):  # max() on empty sequence
            collate_kd_batch(batch)

    def test_collate_kd_batch_single_item(self):
        """Test batch collation with single item."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "teacher_target_ids": torch.tensor([4, 5, 6]),
            }
        ]

        result = collate_kd_batch(batch)

        # Should still work with single item
        assert result["input_ids"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)

    def test_collate_kd_batch_variable_lengths(self):
        """Test batch collation with highly variable sequence lengths."""
        batch = [
            {
                "input_ids": torch.tensor([1]),  # Very short
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([1]),
                "teacher_target_ids": torch.tensor([2]),
            },
            {
                # Much longer
                "input_ids": torch.tensor([3, 4, 5, 6, 7, 8, 9, 10]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
                "labels": torch.tensor([3, 4, 5, 6, 7, 8, 9, 10]),
                "teacher_target_ids": torch.tensor([11, 12, 13, 14, 15, 16, 17, 18]),
            }
        ]

        result = collate_kd_batch(batch)

        # Should pad to max length (8)
        assert result["input_ids"].shape == (2, 8)
        assert result["attention_mask"].shape == (2, 8)
        assert result["labels"].shape == (2, 8)
        assert result["teacher_target_ids"].shape == (2, 8)

        # Check first sequence is padded: [1, 0, 0, 0, 0, 0, 0, 0]
        assert result["input_ids"][0, 0] == 1
        # Assuming pad_token_id = 0
        assert torch.all(result["input_ids"][0, 1:] == 0)


class TestKDDatasetEnhanced:
    """Enhanced tests for KDDataset to improve coverage."""

    def _setup_mock_tokenizer(self, mock_safe_tokenizer):
        """Set up mock tokenizer for tests - non-decorated helper."""
        mock_tokenizer = create_mock_tokenizer_subscriptable()
        mock_safe_tokenizer.return_value = mock_tokenizer
        return mock_tokenizer

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_fingerprint_extraction(self, mock_safe_tokenizer):
        """Test dataset fingerprint extraction from header."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        header_data = {
            "dataset_fingerprint": "abc123",
            "created_at": "2024-01-01",
            "version": "1.0"
        }

        sample_data = {
            "prompt": "test prompt",
            "teacher_text": "test teacher"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write header first
            json.dump({"__header__": header_data}, f)
            f.write('\n')
            # Write sample data
            json.dump(sample_data, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            assert dataset.dataset_fingerprint == "abc123"
            assert dataset.dataset_header == header_data
            assert len(dataset.samples) == 1
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_no_fingerprint(self, mock_safe_tokenizer):
        """Test dataset without fingerprint."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample_data = {
            "prompt": "test prompt",
            "teacher_text": "test teacher"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_data, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            assert dataset.dataset_fingerprint is None
            assert dataset.dataset_header is None
            assert len(dataset.samples) == 1
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_empty_file(self, mock_safe_tokenizer):
        """Test dataset with empty file."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")
            assert len(dataset.samples) == 0
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_malformed_json(self, mock_safe_tokenizer):
        """Test dataset with malformed JSON."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"incomplete": json}\n')  # Malformed JSON
            f.write('{"prompt": "test", "teacher_text": "response", "valid": "json"}\n')  # Valid JSON with required fields
            temp_path = f.name

        try:
            # Should handle malformed JSON gracefully and load valid lines
            dataset = KDDataset(temp_path, "test_tokenizer")
            assert len(dataset.samples) == 1  # Only the valid line
            assert dataset.samples[0]["valid"] == "json"
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_tokenization_with_special_tokens(self, mock_safe_tokenizer):
        """Test tokenization handling with special tokens."""
        mock_tokenizer = self._setup_mock_tokenizer(mock_safe_tokenizer)
        # encode should be a Mock, not just return_value
        mock_tokenizer.encode = Mock(return_value=[101, 2057, 102])  # [CLS] hello [SEP]

        sample_data = {
            "prompt": "hello world",
            "teacher_text": "greeting response"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_data, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer", max_seq_length=10)
            item = dataset[0]

            # Check tokenization was called
            assert mock_tokenizer.encode.called
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_truncation_edge_cases(self, mock_safe_tokenizer):
        """Test truncation with edge cases."""
        mock_tokenizer = self._setup_mock_tokenizer(mock_safe_tokenizer)
        # Mock very long sequence
        long_tokens = list(range(100))  # 100 tokens
        mock_tokenizer.encode.return_value = long_tokens

        sample_data = {
            "prompt": "very long prompt " * 20,  # Long text
            "teacher_text": "short response"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_data, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer", max_seq_length=50)
            item = dataset[0]

            # Should be truncated to max_seq_length
            assert len(item["input_ids"]) <= 50
            assert len(item["attention_mask"]) <= 50
            assert len(item["labels"]) <= 50
        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_supervision_target_processing(self, mock_safe_tokenizer):
        """Test processing of various supervision targets."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        complex_sample = {
            "prompt": "test prompt",
            "teacher_text": "test teacher",
            "tool_name_ids": [100, 200, 300, 400],
            "tool_name_mask": [1, 1, 0, 0],
            "gold_json_text_ids": [500, 600],
            "mask_valid_json_tokens": [1, 0],
            "tool_result_fields": [700, 800, 900],
            "integration_mask": [1, 0, 1, 0, 1],
            "teacher_logits": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(complex_sample, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer", teacher_logits_available=True)
            item = dataset[0]

            # Check all supervision targets are present
            assert "tool_name_ids" in item
            assert "tool_name_mask" in item
            assert "gold_json_text_ids" in item
            assert "mask_valid_json_tokens" in item
            assert "tool_result_fields" in item
            assert "integration_mask" in item
            assert "teacher_logits" in item

            # Check tensor shapes/types
            assert isinstance(item["tool_name_ids"], torch.Tensor)
            assert isinstance(item["tool_name_mask"], torch.Tensor)
            assert isinstance(item["teacher_logits"], torch.Tensor)

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_missing_supervision_targets(self, mock_safe_tokenizer):
        """Test handling of missing supervision targets."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        minimal_sample = {
            "prompt": "test prompt",
            "teacher_text": "test teacher"
            # No supervision targets
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(minimal_sample, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Should not have supervision targets
            assert "tool_name_ids" not in item
            assert "tool_name_mask" not in item
            assert "gold_json_text_ids" not in item
            assert "teacher_logits" not in item

            # But should have basic fields
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_teacher_logits_tensor_conversion(self, mock_safe_tokenizer):
        """Test teacher logits tensor conversion."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample_with_logits = {
            "prompt": "test prompt",
            "teacher_text": "test teacher",
            "teacher_logits": [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]  # 2x3 logits
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_with_logits, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer",
                                teacher_logits_available=True)
            item = dataset[0]

            assert "teacher_logits" in item
            assert isinstance(item["teacher_logits"], torch.Tensor)
            assert item["teacher_logits"].shape == (
                2, 3)  # Should preserve original shape

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_teacher_logits_not_available(self, mock_safe_tokenizer):
        """Test when teacher_logits_available=False but logits present."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample_with_logits = {
            "prompt": "test prompt",
            "teacher_text": "test teacher",
            "teacher_logits": [[0.1, 0.9], [0.2, 0.8]]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_with_logits, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer",
                                teacher_logits_available=False)
            item = dataset[0]

            # Should not include teacher_logits when not available
            assert "teacher_logits" not in item

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_metadata_preservation(self, mock_safe_tokenizer):
        """Test metadata preservation in dataset."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        sample_with_metadata = {
            "prompt": "test prompt",
            "teacher_text": "test teacher",
            "metadata": {
                "source": "synthetic",
                "quality_score": 0.95,
                "tags": ["test", "validation"],
                "nested": {"key": "value"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_with_metadata, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")
            item = dataset[0]

            # Metadata should be preserved
            assert "metadata" in item
            assert item["metadata"]["source"] == "synthetic"
            assert item["metadata"]["quality_score"] == 0.95
            assert item["metadata"]["nested"]["key"] == "value"

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_tokenization_error_handling(self, mock_safe_tokenizer):
        """Test error handling during tokenization."""
        mock_tokenizer = self._setup_mock_tokenizer(mock_safe_tokenizer)
        # Make encode raise an exception on all calls
        mock_tokenizer.encode = Mock(side_effect=Exception("Tokenization failed"))

        sample_data = {
            "prompt": "test prompt",
            "teacher_text": "test teacher"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(sample_data, f)
            f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            # Should raise exception when tokenization fails
            with pytest.raises((Exception, RuntimeError)):
                dataset[0]

        finally:
            Path(temp_path).unlink()

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_large_dataset_handling(self, mock_safe_tokenizer):
        """Test handling of larger datasets."""
        self._setup_mock_tokenizer(mock_safe_tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Create 100 samples
            for i in range(100):
                sample = {
                    "prompt": f"prompt {i}",
                    "teacher_text": f"teacher {i}"
                }
                json.dump(sample, f)
                f.write('\n')
            temp_path = f.name

        try:
            dataset = KDDataset(temp_path, "test_tokenizer")

            assert len(dataset) == 100

            # Test random access
            item = dataset[50]
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item

        finally:
            Path(temp_path).unlink()

    def test_collate_kd_batch_attention_mask_padding(self):
        """Test attention mask padding in collate function."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([4, 5, 6, 7, 8]),
            }
        ]

        result = collate_kd_batch(batch)

        # Should pad to max length (5)
        assert result["input_ids"].shape == (2, 5)
        assert result["attention_mask"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)

        # Attention mask should be 1 for real tokens, 0 for padding
        assert result["attention_mask"][0, 0] == 1
        assert result["attention_mask"][0, 1] == 1
        assert result["attention_mask"][0, 2] == 1
        assert result["attention_mask"][0, 3] == 0  # Padding
        assert result["attention_mask"][0, 4] == 0  # Padding

    def test_collate_kd_batch_supervision_target_padding(self):
        """Test supervision target padding."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "tool_name_ids": torch.tensor([100, 200]),
                "tool_name_mask": torch.tensor([1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([4, 5]),
                "tool_name_ids": torch.tensor([300, 400, 500]),
                "tool_name_mask": torch.tensor([1, 1, 0]),
            }
        ]

        result = collate_kd_batch(batch)

        # Supervision targets should be padded to max length (3)
        assert result["tool_name_ids"].shape == (2, 3)
        assert result["tool_name_mask"].shape == (2, 3)

        # First item should have padding
        assert result["tool_name_ids"][0, 2] == 0  # Padding token
        assert result["tool_name_mask"][0, 2] == 0  # Padding mask

    def test_collate_kd_batch_teacher_logits_handling(self):
        """Test teacher logits handling in collate function."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
                "teacher_logits": torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([3, 4]),
                "teacher_logits": torch.tensor([[0.3, 0.7], [0.4, 0.6]]),
            }
        ]

        result = collate_kd_batch(batch)

        # Teacher logits should be stacked, not padded
        assert result["teacher_logits"].shape == (
            2, 2, 2)  # [batch, seq, vocab]

    def test_collate_kd_batch_missing_fields(self):
        """Test collate function with missing fields."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([3, 4]),
                "teacher_logits": torch.tensor([[0.1, 0.9]]),
            }
        ]

        result = collate_kd_batch(batch)

        # Should handle missing teacher_logits in first item
        assert "teacher_logits" in result
        # First item should have None/empty for teacher_logits
        assert result["teacher_logits"][0] is None or len(
            result["teacher_logits"][0]) == 0
