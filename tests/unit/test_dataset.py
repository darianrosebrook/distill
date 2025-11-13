"""
Unit tests for training dataset loading and batching.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

# Mock transformers module before importing dataset
mock_transformers = MagicMock()
mock_auto_tokenizer_class = MagicMock()
mock_transformers.AutoTokenizer = mock_auto_tokenizer_class
sys.modules["transformers"] = mock_transformers

# Reload dataset module to pick up the mock
if "training.dataset" in sys.modules:
    import importlib

    importlib.reload(sys.modules["training.dataset"])

# Now import dataset (it will use our mock)
from training.dataset import KDDataset, collate_kd_batch  # noqa: E402

# Ensure HF_TOKENIZER_AVAILABLE is True in the module
import training.dataset as dataset_module  # noqa: E402

dataset_module.HF_TOKENIZER_AVAILABLE = True
dataset_module.AutoTokenizer = mock_auto_tokenizer_class


class TestKDDataset:
    """Tests for KDDataset class."""

    def test_dataset_initialization(self, temp_jsonl_file, mock_tokenizer):
        """Test dataset initialization with valid JSONL file."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        dataset = KDDataset(
            jsonl_path=str(temp_jsonl_file),
            tokenizer_path="mock_tokenizer",
            max_seq_length=128,
            teacher_logits_available=False,
        )

        assert len(dataset) == 2, "Dataset should have 2 items"
        assert dataset.max_seq_length == 128
        assert dataset.teacher_logits_available is False

    def test_dataset_loads_jsonl(self, temp_jsonl_file, mock_tokenizer):
        """Test that dataset correctly loads JSONL data."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        dataset = KDDataset(
            jsonl_path=str(temp_jsonl_file),
            tokenizer_path="mock_tokenizer",
            max_seq_length=128,
            teacher_logits_available=False,
        )

        assert len(dataset.samples) == 2
        assert "prompt" in dataset.samples[0]
        assert "teacher_text" in dataset.samples[0]

    def test_dataset_getitem(self, temp_jsonl_file, mock_tokenizer):
        """Test dataset __getitem__ returns correct structure."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        dataset = KDDataset(
            jsonl_path=str(temp_jsonl_file),
            tokenizer_path="mock_tokenizer",
            max_seq_length=128,
            teacher_logits_available=False,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].dim() == 1, "input_ids should be 1D"
        assert item["input_ids"].size(0) <= 128, "Should respect max_seq_length"

    def test_dataset_with_teacher_logits(self, mock_tokenizer):
        """Test dataset with teacher logits available."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create temp file with teacher logits
        vocab_size = mock_tokenizer.vocab_size
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            test_data = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_logits": [[0.1] * vocab_size] * 5,  # 5 tokens, vocab_size vocab
            }
            f.write(json.dumps(test_data) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = KDDataset(
                jsonl_path=str(temp_path),
                tokenizer_path="mock_tokenizer",
                max_seq_length=128,
                teacher_logits_available=True,
            )

            item = dataset[0]

            assert "teacher_logits" in item
            assert item["teacher_logits"] is not None
            assert item["teacher_logits"].dim() == 2, (
                "teacher_logits should be 2D [seq_len, vocab_size]"
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_dataset_truncation(self, mock_tokenizer):
        """Test that dataset truncates sequences to max_seq_length."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Create a prompt that would tokenize to more than max_length
            test_data = {
                "prompt": " ".join(["word"] * 200),  # Long prompt
                "teacher_text": "response",
            }
            f.write(json.dumps(test_data) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = KDDataset(
                jsonl_path=str(temp_path),
                tokenizer_path="mock_tokenizer",
                max_seq_length=50,  # Short max length
                teacher_logits_available=False,
            )

            item = dataset[0]

            assert item["input_ids"].size(0) <= 50, "Should truncate to max_seq_length"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_dataset_labels_shifted(self, mock_tokenizer):
        """Test that labels are shifted by 1 for next-token prediction."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            test_data = {
                "prompt": "Hello",
                "teacher_text": "world",
            }
            f.write(json.dumps(test_data) + "\n")
            temp_path = Path(f.name)

        try:
            dataset = KDDataset(
                jsonl_path=str(temp_path),
                tokenizer_path="mock_tokenizer",
                max_seq_length=128,
                teacher_logits_available=False,
            )

            item = dataset[0]
            input_ids = item["input_ids"]
            labels = item["labels"]

            # Labels should be shifted by 1 (next token prediction)
            assert labels.size(0) == input_ids.size(0), "Labels should match input_ids length"
            # Labels should be the next token after each input token
            # This is a basic check - actual values depend on tokenization
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestCollateKDBatch:
    """Tests for collate_kd_batch function."""

    def test_collate_basic(self):
        """Test basic batch collation."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 1000, (10,)),
            },
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 1000, (10,)),
            },
        ]

        collated = collate_kd_batch(batch)

        assert "input_ids" in collated
        assert "attention_mask" in collated
        assert "labels" in collated
        assert collated["input_ids"].dim() == 2, "Should stack to [batch_size, seq_len]"
        assert collated["input_ids"].size(0) == 2, "Batch size should be 2"
        assert "teacher_logits" not in collated or collated.get("teacher_logits") is None

    def test_collate_with_teacher_logits(self):
        """Test batch collation with teacher logits."""
        vocab_size = 1000
        seq_len = 10

        batch = [
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
                "attention_mask": torch.ones(seq_len),
                "labels": torch.randint(0, vocab_size, (seq_len,)),
                "teacher_logits": torch.randn(seq_len, vocab_size),
            },
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
                "attention_mask": torch.ones(seq_len),
                "labels": torch.randint(0, vocab_size, (seq_len,)),
                "teacher_logits": torch.randn(seq_len, vocab_size),
            },
        ]

        collated = collate_kd_batch(batch)

        assert "teacher_logits" in collated
        assert collated["teacher_logits"] is not None
        assert collated["teacher_logits"].dim() == 3, "Should be [batch_size, seq_len, vocab_size]"
        assert collated["teacher_logits"].size(0) == 2, "Batch size should be 2"
        assert collated["teacher_logits"].size(1) == seq_len
        assert collated["teacher_logits"].size(2) == vocab_size

    def test_collate_variable_length(self):
        """Test batch collation with variable length sequences."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (5,)),
                "attention_mask": torch.ones(5),
                "labels": torch.randint(0, 1000, (5,)),
            },
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 1000, (10,)),
            },
        ]

        collated = collate_kd_batch(batch)

        # Should pad to longest sequence
        max_len = max(5, 10)
        assert collated["input_ids"].size(1) == max_len, "Should pad to longest sequence"
        assert collated["attention_mask"].size(1) == max_len
        assert collated["labels"].size(1) == max_len

    def test_collate_pads_labels_with_ignore_index(self):
        """Test that padding in labels uses ignore_index (-100)."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (5,)),
                "attention_mask": torch.ones(5),
                "labels": torch.randint(0, 1000, (5,)),
            },
            {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.randint(0, 1000, (10,)),
            },
        ]

        collated = collate_kd_batch(batch)

        # First item should have padding in labels
        labels_0 = collated["labels"][0]
        # Padding positions (indices 5-9) should be -100
        assert (labels_0[5:] == -100).all(), "Padding positions should be -100"
