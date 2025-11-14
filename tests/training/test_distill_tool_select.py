"""
Tests for training/distill_tool_select.py - Tool selection training script.

Tests model creation, training step, configuration loading, and constrained decoding integration.
"""
# @author: @darianrosebrook

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from training.distill_tool_select import load_config, create_model, train_step


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_valid_yaml(self):
        """Test loading a valid YAML config file."""
        config_content = """
        arch:
          vocab_size: 1000
          d_model: 128
        train:
          steps: 100
        process_supervision:
          json_validity_weight: 0.3
          tool_select_weight: 0.7
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["arch"]["vocab_size"] == 1000
            assert config["train"]["steps"] == 100
            assert config["process_supervision"]["json_validity_weight"] == 0.3
        finally:
            Path(config_path).unlink()

    def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestCreateModel:
    """Test create_model function."""

    @pytest.fixture
    def basic_config(self):
        """Basic model configuration for testing."""
        return {
            "arch": {
                "vocab_size": 1000,
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_head": 32,
                "rope_theta": 10000.0,
                "rope_scaling": "none",
                "dropout": 0.0,
            }
        }

    def test_create_model_basic(self, basic_config, device):
        """Test basic model creation."""
        model = create_model(basic_config, device)

        assert isinstance(model, nn.Module)
        assert model.training  # Model should be in training mode by default

    def test_create_model_with_checkpoint(self, basic_config, device, tmp_path):
        """Test model creation with checkpoint loading."""
        # Create a dummy checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        dummy_model = nn.Linear(10, 5)
        torch.save({"model_state_dict": dummy_model.state_dict()}, checkpoint_path)

        config_with_checkpoint = basic_config.copy()
        config_with_checkpoint["init"] = {"base_checkpoint": str(checkpoint_path)}

        # This will fail because model architectures don't match, but should attempt loading
        model = create_model(config_with_checkpoint, device)

        assert isinstance(model, nn.Module)

    def test_create_model_default_values(self, device):
        """Test model creation with minimal config (uses defaults)."""
        minimal_config = {"arch": {}}

        model = create_model(minimal_config, device)

        assert isinstance(model, nn.Module)


class TestTrainStep:
    """Test train_step function."""

    @pytest.fixture
    def training_config(self):
        """Config for training step tests."""
        return {
            "arch": {"vocab_size": 1000},
            "process_supervision": {
                "json_validity_weight": 0.3,
                "tool_select_weight": 0.7,
            },
            "io": {"tokenizer_path": "models/student/tokenizer"},
        }

    @pytest.fixture
    def sample_batch(self):
        """Create sample training batch."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "tool_names": ["tool1", "tool2"],
        }

    @pytest.fixture
    def mock_model(self, small_model):
        """Use small model from conftest."""
        return small_model

    @pytest.fixture
    def simple_optimizer(self, small_model):
        """Create simple optimizer for real model."""
        return AdamW(small_model.parameters(), lr=1e-3)

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_basic(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test basic training step."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1", "arguments": {}}')
        mock_load_tokenizer.return_value = mock_tokenizer

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            decoder=None,
        )

        assert isinstance(result, dict)
        assert "total" in result
        assert "ce" in result
        assert "json_validity" in result
        assert "tool_select" in result
        assert isinstance(result["total"], float)
        assert result["total"] >= 0

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_with_decoder(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with constrained decoder."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1", "arguments": {}}')
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock constrained decoder
        mock_decoder = Mock()
        mock_decoder_state = Mock()
        mock_decoder_state.complete = True
        mock_decoder.start = Mock(return_value=mock_decoder_state)
        mock_decoder.allowed_token_mask = Mock(return_value=torch.ones(1000, dtype=torch.bool))
        mock_decoder.push = Mock(return_value=mock_decoder_state)
        mock_decoder.finalize = Mock()

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            decoder=mock_decoder,
        )

        assert isinstance(result, dict)
        assert "total" in result
        # Decoder should be used
        mock_decoder.start.assert_called()

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_with_fp16(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with FP16 scaler."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1", "arguments": {}}')
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create FP16 scaler (only works on CUDA)
        scaler = None
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=scaler,
            cfg=training_config,
            device=device,
            decoder=None,
        )

        assert isinstance(result, dict)
        assert "total" in result

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_zero_weights(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, device
    ):
        """Test training step with zero process supervision weights."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer

        config_zero_weights = {
            "arch": {"vocab_size": 1000},
            "process_supervision": {
                "json_validity_weight": 0.0,
                "tool_select_weight": 0.0,
            },
            "io": {"tokenizer_path": "models/student/tokenizer"},
        }

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=config_zero_weights,
            device=device,
            decoder=None,
        )

        # Should only have CE loss
        assert isinstance(result, dict)
        assert "total" in result
        assert result["json_validity"] == 0.0
        assert result["tool_select"] == 0.0

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_no_tool_names(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step without tool names in batch."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1", "arguments": {}}')
        mock_load_tokenizer.return_value = mock_tokenizer

        # Remove tool_names from batch
        sample_batch_no_tools = {k: v for k, v in sample_batch.items() if k != "tool_names"}

        # Move batch to device
        for k, v in sample_batch_no_tools.items():
            if isinstance(v, torch.Tensor):
                sample_batch_no_tools[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch_no_tools,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            decoder=None,
        )

        # Should still work, tool selection loss will be zero
        assert isinstance(result, dict)
        assert "total" in result

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_decoder_incomplete(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with decoder that produces incomplete JSON."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1"')  # Incomplete JSON
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock constrained decoder with incomplete state
        mock_decoder = Mock()
        mock_decoder_state = Mock()
        mock_decoder_state.complete = False
        mock_decoder.start = Mock(return_value=mock_decoder_state)
        mock_decoder.allowed_token_mask = Mock(return_value=torch.ones(1000, dtype=torch.bool))
        mock_decoder.push = Mock(return_value=mock_decoder_state)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            decoder=mock_decoder,
        )

        # Should handle incomplete JSON gracefully
        assert isinstance(result, dict)
        assert "total" in result

    @patch("training.distill_tool_select.load_tokenizer")
    def test_train_step_decoder_invalid(
        self, mock_load_tokenizer, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with decoder that produces invalid JSON."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool1"')  # Invalid JSON
        mock_load_tokenizer.return_value = mock_tokenizer

        # Mock constrained decoder that raises ValueError on finalize
        mock_decoder = Mock()
        mock_decoder_state = Mock()
        mock_decoder_state.complete = True
        mock_decoder.start = Mock(return_value=mock_decoder_state)
        mock_decoder.allowed_token_mask = Mock(return_value=torch.ones(1000, dtype=torch.bool))
        mock_decoder.push = Mock(return_value=mock_decoder_state)
        mock_decoder.finalize = Mock(side_effect=ValueError("Invalid JSON"))

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            decoder=mock_decoder,
        )

        # Should handle invalid JSON gracefully
        assert isinstance(result, dict)
        assert "total" in result







