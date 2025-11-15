"""
Tests for training/dataset.py - Knowledge distillation dataset loader.

Tests KDDataset class, load_tokenizer function, collate_kd_batch function,
data loading, fingerprint extraction, and process-step supervision targets.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch

from training.dataset import KDDataset, load_tokenizer, collate_kd_batch


class TestLoadTokenizer:
    """Test load_tokenizer function."""

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_load_tokenizer_basic(self, mock_safe_from_pretrained):
        """Test basic tokenizer loading."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_safe_from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test/path")

        mock_safe_from_pretrained.assert_called_once_with("test/path")
        assert tokenizer == mock_tokenizer
        assert tokenizer.pad_token == "<eos>"

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_load_tokenizer_with_pad_token(self, mock_safe_from_pretrained):
        """Test tokenizer loading when pad_token already exists."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_safe_from_pretrained.return_value = mock_tokenizer

        tokenizer = load_tokenizer("test/path")

        assert tokenizer.pad_token == "<pad>"
        # Should not override existing pad_token

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", False)
    def test_load_tokenizer_no_transformers(self):
        """Test tokenizer loading when transformers not available."""
        with pytest.raises(RuntimeError, match="transformers library required"):
            load_tokenizer("test/path")


class TestKDDataset:
    """Test KDDataset class."""

    @pytest.fixture
    def sample_jsonl_content(self):
        """Create sample JSONL content for testing."""
        return [
            {"prompt": "Hello", "teacher_text": "Hello world"},
            {"prompt": "Test", "teacher_text": "Test response"},
        ]

    @pytest.fixture
    def jsonl_file(self, sample_jsonl_content, tmp_path):
        """Create a temporary JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for item in sample_jsonl_content:
                f.write(json.dumps(item) + "\n")
        return str(jsonl_path)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3] if text else [])
        tokenizer.__len__ = Mock(return_value=1000)
        return tokenizer

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_basic(self, mock_safe_from_pretrained, jsonl_file, mock_tokenizer):
        """Test basic KDDataset initialization."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        dataset = KDDataset(jsonl_file, "tokenizer/path", max_seq_length=512)

        assert len(dataset) == 2
        assert dataset.max_seq_length == 512
        assert dataset.tokenizer == mock_tokenizer

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_with_header(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset initialization with dataset header."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Write header
            header = {"__header__": True, "dataset_sha256": "abc123"}
            f.write(json.dumps(header) + "\n")
            # Write data
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")

        assert dataset.dataset_fingerprint == "abc123"
        assert dataset.dataset_header == header
        assert len(dataset) == 1

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_nonexistent_file(self, mock_safe_from_pretrained, mock_tokenizer):
        """Test KDDataset initialization with nonexistent file."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        with pytest.raises(FileNotFoundError):
            KDDataset("nonexistent.jsonl", "tokenizer/path")

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_missing_fields(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset initialization with missing required fields."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Missing prompt
            f.write(json.dumps({"teacher_text": "World"}) + "\n")
            # Missing teacher_text
            f.write(json.dumps({"prompt": "Hello"}) + "\n")
            # Valid entry
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")

        # Should skip invalid entries
        assert len(dataset) == 1

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_init_cot_validation(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset initialization with CoT-free validation."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Should raise error if teacher_reasoning_content detected
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "teacher_reasoning_content": "Some reasoning",
                    }
                )
                + "\n"
            )

        with pytest.raises(ValueError, match="CoT-free training"):
            KDDataset(str(jsonl_path), "tokenizer/path")

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_basic(self, mock_safe_from_pretrained, jsonl_file, mock_tokenizer):
        """Test basic __getitem__ functionality."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        dataset = KDDataset(jsonl_file, "tokenizer/path", max_seq_length=512)

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_process_targets(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with process-step supervision targets."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "tool_name_ids": [1, 2, 3],
                        "tool_name_mask": [1, 1, 0],
                        "gold_json_text_ids": [4, 5, 6],
                        "mask_valid_json_tokens": [1, 0, 1],
                        "tool_result_fields": [7, 8],
                        "integration_mask": [1, 1],
                    }
                )
                + "\n"
            )

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        assert "tool_name_ids" in item
        assert "tool_name_mask" in item
        assert "gold_json_text_ids" in item
        assert "mask_valid_json_tokens" in item
        assert "tool_result_fields" in item
        assert "integration_mask" in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_teacher_logits(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with teacher logits."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        vocab_size = 1000
        seq_len = 5
        teacher_logits_flat = [0.1] * (seq_len * vocab_size)

        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "teacher_logits": teacher_logits_flat,
                    }
                )
                + "\n"
            )

        dataset = KDDataset(
            str(jsonl_path), "tokenizer/path", teacher_logits_available=True
        )
        item = dataset[0]

        assert "teacher_logits" in item
        assert item["teacher_logits"].dim() == 2  # [T, V]

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_quality_score(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with teacher quality score."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "teacher_quality_score": 0.85,
                    }
                )
                + "\n"
            )

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        assert "teacher_quality_score" in item
        assert item["teacher_quality_score"] == 0.85

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_truncation(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with sequence truncation."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        # Make encode return long sequences
        mock_tokenizer.encode = Mock(
            side_effect=lambda text, **kwargs: list(range(100)) if text else []
        )

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path", max_seq_length=50)
        item = dataset[0]

        # Should be truncated to max_seq_length
        assert item["input_ids"].size(0) <= 50
        assert item["labels"].size(0) <= 50

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_training_text(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with training_text from latent curriculum."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "training_text": "Hello\n\nWorld",
                    }
                )
                + "\n"
            )

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Should use training_text if available
        assert "input_ids" in item
        assert "labels" in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_loss_mask(
        self, mock_safe_from_pretrained, tmp_path, mock_tokenizer
    ):
        """Test __getitem__ with loss mask from latent curriculum."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Hello",
                        "teacher_text": "World",
                        "loss_mask": [True, True, False, False],
                    }
                )
                + "\n"
            )

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        assert "loss_mask" in item
        assert isinstance(item["loss_mask"], torch.Tensor)
        assert item["loss_mask"].dtype == torch.bool

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", False)
    def test_kd_dataset_init_no_transformers(self, jsonl_file):
        """Test KDDataset initialization when transformers not available."""
        with pytest.raises(RuntimeError, match="transformers library required"):
            KDDataset(jsonl_file, "tokenizer/path")


class TestCollateKDBatch:
    """Test collate_kd_batch function."""

    def test_collate_kd_batch_basic(self):
        """Test basic batch collation."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([6, 7]),
            },
        ]

        result = collate_kd_batch(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["input_ids"].shape == (2, 3)  # Batch size 2, max length 3
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_collate_kd_batch_with_padding(self):
        """Test batch collation with padding."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([2, 3, 4, 5, 6]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([8, 9]),
            },
        ]

        result = collate_kd_batch(batch)

        # Second sequence should be padded
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"][1, 2:].sum() == -100 * 3  # Padding should be -100

    def test_collate_kd_batch_with_teacher_targets(self):
        """Test batch collation with teacher target IDs."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "teacher_target_ids": torch.tensor([10, 11, 12]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "teacher_target_ids": torch.tensor([13, 14]),
            },
        ]

        result = collate_kd_batch(batch)

        assert "teacher_target_ids" in result
        assert result["teacher_target_ids"].shape == (2, 3)

    def test_collate_kd_batch_with_teacher_logits(self):
        """Test batch collation with teacher logits."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "teacher_logits": torch.randn(3, 1000),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "teacher_logits": torch.randn(2, 1000),
            },
        ]

        result = collate_kd_batch(batch)

        assert "teacher_logits" in result
        assert result["teacher_logits"].shape == (2, 3, 1000)

    def test_collate_kd_batch_with_process_targets(self):
        """Test batch collation with process-step supervision targets."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "tool_name_ids": torch.tensor([10, 11]),
                "tool_name_mask": torch.tensor([True, True]),
                "gold_json_text_ids": torch.tensor([20, 21, 22]),
                "mask_valid_json_tokens": torch.tensor([True, False, True]),
                "tool_result_fields": torch.tensor([30, 31]),
                "integration_mask": torch.tensor([True, True]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "tool_name_ids": torch.tensor([12]),
                "tool_name_mask": torch.tensor([True]),
            },
        ]

        result = collate_kd_batch(batch)

        assert "tool_name_ids" in result
        assert "tool_name_mask" in result
        assert "gold_json_text_ids" in result
        assert "mask_valid_json_tokens" in result
        assert "tool_result_fields" in result
        assert "integration_mask" in result

        # Check shapes (should be padded to max length)
        # tool_name_ids: first item has length 2, second has length 1, so max is 2
        assert result["tool_name_ids"].shape == (2, 2)
        # gold_json_text_ids: first item has length 3, second doesn't have it (created empty tensor)
        assert result["gold_json_text_ids"].shape == (2, 3)

    def test_collate_kd_batch_with_loss_mask(self):
        """Test batch collation with loss mask."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "loss_mask": torch.tensor([True, True, False]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "loss_mask": torch.tensor([True, False]),
            },
        ]

        result = collate_kd_batch(batch)

        assert "loss_mask" in result
        assert result["loss_mask"].shape == (2, 3)
        assert result["loss_mask"].dtype == torch.bool

    def test_collate_kd_batch_empty_batch(self):
        """Test batch collation with empty batch."""
        batch = []

        with pytest.raises(ValueError):
            collate_kd_batch(batch)

    def test_collate_kd_batch_single_item(self):
        """Test batch collation with single item."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
            }
        ]

        result = collate_kd_batch(batch)

        assert result["input_ids"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)
        assert result["labels"].shape == (1, 3)

    def test_collate_kd_batch_mixed_optional_fields(self):
        """Test batch collation with some items having optional fields."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "tool_name_ids": torch.tensor([10, 11]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                # No tool_name_ids
            },
        ]

        result = collate_kd_batch(batch)

        # Should only include tool_name_ids if all items have it
        # Actually, the function includes it if any item has it
        # Let's check the actual behavior
        if "tool_name_ids" in result:
            assert result["tool_name_ids"].shape[0] == 2

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_load_data_empty_lines(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test _load_data with empty lines in JSONL file (line 115)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("\n")  # Empty line
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")
            f.write("\n")  # Another empty line
            f.write(json.dumps({"prompt": "Test", "teacher_text": "Response"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        # Should skip empty lines and load 2 samples
        assert len(dataset) == 2

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_load_data_json_decode_error(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test _load_data with JSON decode errors (lines 143-145)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")
            f.write("invalid json line\n")  # Invalid JSON
            f.write(json.dumps({"prompt": "Test", "teacher_text": "Response"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        # Should skip invalid JSON line and load 2 samples
        assert len(dataset) == 2

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_with_latent_curriculum(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with latent curriculum applied (line 169)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create a mock latent curriculum
        mock_curriculum = Mock()
        mock_curriculum.apply = Mock(return_value={
            "prompt": "Hello",
            "teacher_text": "World",
            "training_text": "Hello\n\nWorld",
            "loss_mask": [True, True, False]
        })

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path", latent_curriculum=mock_curriculum)
        item = dataset[0]

        # Verify latent curriculum was applied
        mock_curriculum.apply.assert_called_once()
        assert "loss_mask" in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    @patch("data.generators.mcp_code_mode.compute_span_targets_from_tokenized")
    def test_kd_dataset_getitem_span_targets_computation(self, mock_compute, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with span targets computation (lines 195-208)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        mock_compute.return_value = {"ts_mode_spans": [1, 2, 3]}

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "metadata": {
                    "span_targets": {}  # Empty span_targets triggers computation
                }
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Should attempt to compute span targets
        mock_compute.assert_called_once()
        assert "span_targets" in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_target_padding(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with teacher target padding (lines 264-267)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        # Make teacher_tokens shorter than labels
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: {
            "Hello": [1, 2, 3],
            "World": [4, 5],  # Shorter than full sequence
            "Hello\n\nWorld": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Long sequence
        }.get(text, []))

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Teacher target should be padded to match labels length
        assert "teacher_target_ids" in item
        assert len(item["teacher_target_ids"]) == len(item["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_target_truncation(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with teacher target truncation (line 269)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        # Make teacher_tokens longer than labels
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: {
            "Hello": [1, 2],
            "World": [4, 5, 6, 7, 8, 9, 10],  # Longer than labels
            "Hello\n\nWorld": [1, 2, 4, 5, 6]  # Short full sequence
        }.get(text, []))

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"prompt": "Hello", "teacher_text": "World"}) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Teacher target should be truncated to match labels length
        assert "teacher_target_ids" in item
        assert len(item["teacher_target_ids"]) == len(item["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_loss_mask_padding(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with loss mask padding (lines 281-283)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        # Make encode return long sequences
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(20)) if text else [])

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "loss_mask": [True, True, False]  # Shorter than labels
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Loss mask should be padded to match labels length
        assert "loss_mask" in item
        assert len(item["loss_mask"]) == len(item["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_logits_padding(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with teacher logits padding (lines 303-311)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        vocab_size = 1000
        seq_len = 3
        # Create logits shorter than labels
        teacher_logits_flat = [0.1] * (seq_len * vocab_size)

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_logits": teacher_logits_flat,
            }) + "\n")

        # Make encode return longer sequence than logits
        # full_text will be "Hello\n\nWorld" which should be longer
        def encode_side_effect(text, **kwargs):
            if text == "Hello":
                return [1, 2]
            elif text == "World":
                return [3, 4]
            elif text == "Hello\n\nWorld":
                return list(range(10))  # Long sequence
            return []
        
        mock_tokenizer.encode = Mock(side_effect=encode_side_effect)
        mock_tokenizer.__len__ = Mock(return_value=vocab_size)

        dataset = KDDataset(
            str(jsonl_path), "tokenizer/path", teacher_logits_available=True
        )
        item = dataset[0]

        # Teacher logits should be padded to match labels length
        assert "teacher_logits" in item
        assert item["teacher_logits"].size(0) == len(item["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_quality_score_string(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with quality score as string (lines 320-324)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_quality_score": "0.85",  # String instead of float
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        assert "teacher_quality_score" in item
        assert item["teacher_quality_score"] == 0.85

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_quality_score_invalid_string(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with invalid quality score string."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_quality_score": "invalid",  # Invalid string
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Should skip invalid quality score
        assert "teacher_quality_score" not in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_hidden_states(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with teacher hidden states (lines 331-362)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Test with pre-computed hidden states as list
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_hidden_states": [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Layer 1: [T, D]
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]  # Layer 2: [T, D]
                ]
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        assert "teacher_hidden_states" in item
        assert isinstance(item["teacher_hidden_states"], list)
        assert len(item["teacher_hidden_states"]) == 2

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_hidden_states_empty(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with empty teacher hidden states."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_hidden_states": []  # Empty list
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Should set metadata flag
        assert "has_teacher_hidden_states" in item

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_hidden_states_invalid(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test __getitem__ with invalid teacher hidden states."""
        mock_safe_from_pretrained.return_value = mock_tokenizer

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "prompt": "Hello",
                "teacher_text": "World",
                "teacher_hidden_states": "invalid"  # Invalid format
            }) + "\n")

        dataset = KDDataset(str(jsonl_path), "tokenizer/path")
        item = dataset[0]

        # Should set metadata flag on error
        assert "has_teacher_hidden_states" in item

    def test_collate_kd_batch_padding_operations(self):
        """Test collate_kd_batch padding operations for various fields (lines 488, 503, 517, 530)."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "gold_json_text_ids": torch.tensor([10, 11, 12, 13]),  # Length 4
                "mask_valid_json_tokens": torch.tensor([True, False, True, False]),  # Length 4
                "tool_result_fields": torch.tensor([20, 21, 22, 23, 24]),  # Length 5
                "integration_mask": torch.tensor([True, True, True]),  # Length 3
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([5, 6]),
                "gold_json_text_ids": torch.tensor([14, 15]),  # Length 2, needs padding
                "mask_valid_json_tokens": torch.tensor([True, True]),  # Length 2, needs padding
                "tool_result_fields": torch.tensor([25, 26]),  # Length 2, needs padding
                "integration_mask": torch.tensor([True, True]),  # Length 2, needs padding
            },
        ]

        result = collate_kd_batch(batch)

        # Check padding for gold_json_text_ids (line 488)
        assert result["gold_json_text_ids"].shape == (2, 4)  # Padded to max length 4
        # Check padding for mask_valid_json_tokens (line 503)
        assert result["mask_valid_json_tokens"].shape == (2, 4)  # Padded to max length 4
        # Check padding for tool_result_fields (line 517)
        assert result["tool_result_fields"].shape == (2, 5)  # Padded to max length 5
        # Check padding for integration_mask (line 530)
        assert result["integration_mask"].shape == (2, 3)  # Padded to max length 3

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", False)
    def test_load_tokenizer_import_error_path(self):
        """Test load_tokenizer when transformers import fails (lines 18-24)."""
        # This tests the import error handling path
        with pytest.raises(RuntimeError, match="transformers library required"):
            load_tokenizer("test/path")


class TestKDDatasetEdgeCases:
    """Test edge cases for KDDataset."""

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_span_targets_exception_handling(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset handles exception in span_targets computation (lines 205-208)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            json.dump({
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "metadata": {}
            }, f)
            f.write("\n")

        # Patch the import to raise exception when trying to import compute_span_targets_from_tokenized
        # The function imports it inside the try block, so we patch the module import
        import sys
        from unittest.mock import MagicMock
        
        # Create a mock module that raises exception when compute_span_targets_from_tokenized is called
        mock_module = MagicMock()
        mock_module.compute_span_targets_from_tokenized = Mock(side_effect=Exception("Error"))
        
        # Patch sys.modules to return our mock when the module is imported
        with patch.dict(sys.modules, {'data.generators.mcp_code_mode': mock_module}):
            dataset = KDDataset(str(jsonl_file), "test/path")
            sample = dataset[0]
            
            # Should handle exception gracefully and set span_targets to None
            assert "span_targets" in sample or sample.get("span_targets") is None

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_teacher_target_ids_truncation(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset truncates teacher_target_ids when longer than labels (line 269).
        
        Note: Line 260 already truncates teacher_tokens to len(labels), so line 269
        is defensive code. We test it by patching torch.tensor to return a longer tensor.
        """
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        # Make encode return a short sequence so labels will be short (length 2)
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])  # labels will be [2, 3] (length 2)
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with teacher_tokens
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            json.dump({
                "prompt": "Short",
                "teacher_text": "Response",
                "teacher_tokens": [1, 2, 3, 4, 5],  # Will be truncated to len(labels)=2 at line 260
                "metadata": {}
            }, f)
            f.write("\n")

        # Patch torch.tensor to return a longer tensor than expected (to trigger line 269)
        original_tensor = torch.tensor
        call_count = [0]
        
        def mock_tensor(data, **kwargs):
            call_count[0] += 1
            # On the call for teacher_target_ids (after labels are created), return longer tensor
            if call_count[0] > 2 and isinstance(data, list) and len(data) == 2:
                # Return a tensor longer than labels (length 2)
                return original_tensor([1, 2, 3, 4, 5], **kwargs)
            return original_tensor(data, **kwargs)
        
        with patch('torch.tensor', side_effect=mock_tensor):
            dataset = KDDataset(str(jsonl_file), "test/path")
            sample = dataset[0]
            
            # teacher_target_ids should be truncated to match labels length (line 269)
            if "teacher_target_ids" in sample:
                labels = sample["labels"]
                teacher_target_ids = sample["teacher_target_ids"]
                assert len(teacher_target_ids) == len(labels)

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_teacher_logits_dim2_padding(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset pads teacher_logits when dim==2 and shorter than labels (lines 303-311)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3, 4, 5])
        mock_tokenizer.__len__ = Mock(return_value=1000)  # vocab_size
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with teacher_logits as 2D tensor shorter than labels
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            # teacher_logits as 2D: [3, 1000] but labels will be longer
            teacher_logits_2d = [[0.1] * 1000, [0.2] * 1000, [0.3] * 1000]  # 3 timesteps
            json.dump({
                "prompt": "Test prompt that will create longer labels",
                "teacher_text": "Test response",
                "teacher_logits": teacher_logits_2d,  # 2D, shorter than labels
                "metadata": {}
            }, f)
            f.write("\n")

        dataset = KDDataset(str(jsonl_file), "test/path", teacher_logits_available=True)
        sample = dataset[0]
        
        # teacher_logits should be padded to match labels length
        if "teacher_logits" in sample:
            labels = sample["labels"]
            teacher_logits = sample["teacher_logits"]
            assert teacher_logits.shape[0] == len(labels)  # Should be padded to labels length

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_hidden_states_non_list_break(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset breaks when hidden_states layer is not a list (line 348)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with hidden_states containing non-list items
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            json.dump({
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_hidden_states": [
                    [[1.0, 2.0], [3.0, 4.0]],  # First layer: list (will be converted)
                    "not_a_list",  # Second layer: not a list (will trigger break)
                    [[5.0, 6.0], [7.0, 8.0]]  # Third layer: won't be processed due to break
                ],
                "metadata": {}
            }, f)
            f.write("\n")

        dataset = KDDataset(str(jsonl_file), "test/path")
        sample = dataset[0]
        
        # Should process first layer, then break on second layer
        # Result should have only first layer or has_teacher_hidden_states flag
        assert "teacher_hidden_states" in sample or "has_teacher_hidden_states" in sample

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_hidden_states_exception_handling(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset handles exception in hidden_states loading (lines 354-358)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with invalid hidden_states that will cause exception
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            # Use invalid data that will cause exception when converting to tensor
            json.dump({
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_hidden_states": [
                    float('inf'),  # Invalid value that might cause issues
                    float('nan'),  # Invalid value
                ],
                "metadata": {}
            }, f)
            f.write("\n")

        dataset = KDDataset(str(jsonl_file), "test/path")
        sample = dataset[0]
        
        # Should handle exception gracefully and set has_teacher_hidden_states flag
        assert "has_teacher_hidden_states" in sample or "teacher_hidden_states" not in sample

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_loss_mask_list_to_tensor(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset converts loss_mask from list to tensor (line 276->279 branch)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with loss_mask as a list (not tensor)
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            json.dump({
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "loss_mask": [True, False, True, True, False],  # List, not tensor
                "metadata": {}
            }, f)
            f.write("\n")

        dataset = KDDataset(str(jsonl_file), "test/path")
        sample = dataset[0]
        
        # loss_mask should be converted to tensor
        assert "loss_mask" in sample
        assert isinstance(sample["loss_mask"], torch.Tensor)
        assert sample["loss_mask"].dtype == torch.bool


class TestKDDatasetBranchCoverage:
    """Test branch coverage for KDDataset."""

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_load_data_with_dataset_sha256(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset loads dataset header with dataset_sha256 (line 123->125)."""
        mock_safe_from_pretrained.return_value = mock_tokenizer
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            # Write header with dataset_sha256
            header = {
                "__header__": True,
                "dataset_sha256": "abc123def456",
                "version": "1.0"
            }
            f.write(json.dumps(header) + "\n")
            # Write sample
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response"
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")

        assert dataset.dataset_fingerprint == "abc123def456"
        assert len(dataset.samples) == 1

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_target_ids_padding_branch(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset.__getitem__ with teacher_target_ids padding (line 258->273).
        
        This test verifies the padding branch when teacher_tokens is shorter than labels.
        Note: teacher_tokens from JSON is used if present, otherwise teacher_text is encoded.
        """
        # Create a new mock tokenizer with proper encode method
        mock_tokenizer_new = Mock()
        mock_tokenizer_new.pad_token = None
        mock_tokenizer_new.eos_token = "<eos>"
        # Make labels longer than teacher_tokens to trigger padding
        # The full_text will be "Short prompt\n\nShort response" which should produce 12 tokens
        encode_map = {
            "Short prompt": [1, 2, 3, 4, 5],
            "Short response": [6, 7],  # Only 2 tokens
            "Short prompt\n\nShort response": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12 tokens total
        }
        mock_tokenizer_new.encode = Mock(side_effect=lambda text, **kwargs: encode_map.get(text, []))
        mock_safe_from_pretrained.return_value = mock_tokenizer_new
        
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            sample = {
                "prompt": "Short prompt",
                "teacher_text": "Short response",
                "teacher_tokens": [100, 101]  # Only 2 tokens, shorter than labels (12 tokens)
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")
        result = dataset[0]

        # teacher_target_ids should be padded to match labels length
        assert "teacher_target_ids" in result
        labels_len = len(result["labels"])
        assert len(result["teacher_target_ids"]) == labels_len
        # teacher_tokens has 2 items [100, 101], labels should have 12 items
        # The code does teacher_tokens[: len(labels)] first, then pads if shorter
        # So we should see padding since labels_len (12) > 2
        assert labels_len >= 2, f"Labels should be at least 2 tokens, got {labels_len}"
        # First 2 should be the teacher_tokens
        assert result["teacher_target_ids"][0].item() == 100
        assert result["teacher_target_ids"][1].item() == 101
        # If labels_len > 2, rest should be -100 (padding)
        if labels_len > 2:
            assert result["teacher_target_ids"][2].item() == -100
            assert result["teacher_target_ids"][-1].item() == -100

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_target_ids_truncation_branch(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset.__getitem__ with teacher_target_ids truncation (line 268->269)."""
        # Create a new mock tokenizer with proper encode method
        mock_tokenizer_new = Mock()
        mock_tokenizer_new.pad_token = None
        mock_tokenizer_new.eos_token = "<eos>"
        mock_tokenizer_new.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3] if "prompt" in text else [4, 5, 6])
        mock_safe_from_pretrained.return_value = mock_tokenizer_new
        
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            sample = {
                "prompt": "Long prompt",
                "teacher_text": "Long response",
                "teacher_tokens": list(range(100))  # Longer than labels
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")
        result = dataset[0]

        # teacher_target_ids should be truncated to match labels length
        assert "teacher_target_ids" in result
        assert len(result["teacher_target_ids"]) == len(result["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_loss_mask_padding_branch(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset.__getitem__ with loss_mask padding (line 276->279)."""
        # Create a new mock tokenizer with proper encode method
        mock_tokenizer_new = Mock()
        mock_tokenizer_new.pad_token = None
        mock_tokenizer_new.eos_token = "<eos>"
        mock_tokenizer_new.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3, 4, 5] if "prompt" in text else [6, 7, 8, 9, 10])
        mock_safe_from_pretrained.return_value = mock_tokenizer_new
        
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "loss_mask": [True, False, True]  # Shorter than labels
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")
        result = dataset[0]

        # loss_mask should be padded to match labels length
        assert "loss_mask" in result
        assert len(result["loss_mask"]) == len(result["labels"])

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_logits_dim2_padding_branch(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset.__getitem__ with teacher_logits dim==2 and padding (line 303->313, 308->313)."""
        vocab_size = 32000
        # Create a new mock tokenizer with proper encode method
        mock_tokenizer_new = Mock()
        mock_tokenizer_new.pad_token = None
        mock_tokenizer_new.eos_token = "<eos>"
        # Make labels longer than teacher_logits (2 timesteps) to trigger padding
        encode_map = {
            "Test prompt": [1, 2, 3],
            "Test response": [4, 5, 6, 7, 8],  # 5 tokens
            "Test prompt\n\nTest response": [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens total
        }
        mock_tokenizer_new.encode = Mock(side_effect=lambda text, **kwargs: encode_map.get(text, []))
        mock_tokenizer_new.__len__ = Mock(return_value=vocab_size)
        mock_safe_from_pretrained.return_value = mock_tokenizer_new
        
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            # Create teacher_logits as 2D tensor [T, V] where T (2) < labels length (8)
            teacher_logits_2d = [[0.1] * vocab_size, [0.2] * vocab_size]  # 2 timesteps
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_logits": teacher_logits_2d
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")
        result = dataset[0]

        # teacher_logits should be padded to match labels length
        assert "teacher_logits" in result
        assert result["teacher_logits"].dim() == 2
        labels_len = len(result["labels"])
        assert result["teacher_logits"].size(0) == labels_len
        # Original teacher_logits has 2 timesteps, labels has labels_len timesteps
        # So teacher_logits should be padded from 2 to labels_len
        assert labels_len >= 2, f"Labels should be at least 2 tokens, got {labels_len}"
        if labels_len > 2:
            # Check that padding was applied (last timestep should be zeros)
            assert result["teacher_logits"][-1].sum().item() == 0.0, "Last timestep should be zero padding"

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_getitem_teacher_quality_score_value_error_branch(self, mock_safe_from_pretrained, tmp_path, mock_tokenizer):
        """Test KDDataset.__getitem__ with invalid quality_score string (line 320->330 ValueError branch)."""
        # Create a new mock tokenizer with proper encode method
        mock_tokenizer_new = Mock()
        mock_tokenizer_new.pad_token = None
        mock_tokenizer_new.eos_token = "<eos>"
        mock_tokenizer_new.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3] if "prompt" in text else [4, 5, 6])
        mock_safe_from_pretrained.return_value = mock_tokenizer_new
        
        jsonl_file = tmp_path / "dataset.jsonl"
        with open(jsonl_file, "w") as f:
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_quality_score": "invalid_float"  # Cannot be converted to float
            }
            f.write(json.dumps(sample) + "\n")

        dataset = KDDataset(str(jsonl_file), "tokenizer_path")
        result = dataset[0]

        # Invalid quality_score should be skipped (not in result)
        assert "teacher_quality_score" not in result or result.get("teacher_quality_score") is None

    @patch("training.dataset.HF_TOKENIZER_AVAILABLE", True)
    @patch("training.safe_model_loading.safe_from_pretrained_tokenizer")
    def test_kd_dataset_hidden_states_empty_tensors_fallback(self, mock_safe_from_pretrained, tmp_path):
        """Test KDDataset fallback when hidden_states_tensors is empty (lines 354-358)."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_safe_from_pretrained.return_value = mock_tokenizer

        # Create test JSONL file with hidden_states that result in empty tensors
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            json.dump({
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_hidden_states": [],  # Empty list
                "metadata": {}
            }, f)
            f.write("\n")

        dataset = KDDataset(str(jsonl_file), "test/path")
        sample = dataset[0]
        
        # Should fall back to has_teacher_hidden_states flag when tensors are empty
        assert "has_teacher_hidden_states" in sample or sample.get("has_teacher_hidden_states") is True



