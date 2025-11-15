"""
Tests for training/distill_answer_generation.py - Answer generation training script.

Tests model creation, training steps, and configuration loading.
"""
# @author: @darianrosebrook

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import yaml

from training.distill_answer_generation import (
    load_config,
    create_model,
    train_step,
)


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "arch": {"d_model": 512, "n_layers": 4},
            "optimizer": {"lr": 0.001},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(str(config_file))
        assert result == config_data
        assert result["arch"]["d_model"] == 512

    def test_load_config_nonexistent_file(self):
        """Test loading a nonexistent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestCreateModel:
    """Test create_model function."""

    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_default_config(self, mock_student_lm):
        """Test creating model with default config values."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model

        cfg = {}
        device = torch.device("cpu")

        result = create_model(cfg, device)

        assert result == mock_model
        mock_student_lm.assert_called_once()
        mock_model.to.assert_called_once_with(device)

    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_custom_config(self, mock_student_lm):
        """Test creating model with custom config values."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model

        cfg = {
            "arch": {
                "d_model": 1024,
                "n_layers": 8,
                "n_heads": 16,
                "n_kv_heads": 4,
                "d_head": 64,
                "vocab_size": 50000,
                "rope_theta": 20000.0,
                "rope_scaling": "linear",
                "dropout": 0.1,
            }
        }
        device = torch.device("cpu")

        result = create_model(cfg, device)

        assert result == mock_model
        mock_student_lm.assert_called_once()
        # Verify ModelCfg was created with correct values
        call_args = mock_student_lm.call_args[0][0]
        assert call_args.d_model == 1024
        assert call_args.n_layers == 8

    @patch("training.safe_checkpoint_loading.safe_load_checkpoint")
    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_with_checkpoint(self, mock_student_lm, mock_load_checkpoint, tmp_path):
        """Test creating model and loading checkpoint."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.load_state_dict.return_value = None

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.touch()

        mock_checkpoint = {"model_state_dict": {"layer1.weight": torch.randn(10, 10)}}
        mock_load_checkpoint.return_value = mock_checkpoint

        cfg = {
            "init": {"base_checkpoint": str(checkpoint_path)},
        }
        device = torch.device("cpu")

        result = create_model(cfg, device)

        assert result == mock_model
        mock_load_checkpoint.assert_called_once()
        mock_model.load_state_dict.assert_called_once()

    @patch("training.safe_checkpoint_loading.safe_load_checkpoint")
    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_checkpoint_no_model_state_dict(self, mock_student_lm, mock_load_checkpoint, tmp_path):
        """Test loading checkpoint without model_state_dict key."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.load_state_dict.return_value = None

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.touch()

        mock_checkpoint = {"layer1.weight": torch.randn(10, 10)}
        mock_load_checkpoint.return_value = mock_checkpoint

        cfg = {
            "init": {"base_checkpoint": str(checkpoint_path)},
        }
        device = torch.device("cpu")

        result = create_model(cfg, device)

        assert result == mock_model
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint, strict=False)


class TestTrainStep:
    """Test train_step function."""

    def test_train_step_basic(self):
        """Test basic training step without scaler."""
        # Create a simple model that returns logits
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 1000)
            
            def forward(self, input_ids, attention_mask):
                # Return logits of shape [B, T, V]
                batch_size, seq_len = input_ids.shape
                return self.linear(torch.randn(batch_size, seq_len, 1000))
        
        model = SimpleModel()
        model.train()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 1000, (2, 10)),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        device = torch.device("cpu")
        cfg = {}

        result = train_step(model, batch, optimizer, None, cfg, device)

        assert "total" in result
        assert "ce" in result
        assert isinstance(result["total"], float)
        assert isinstance(result["ce"], float)
        assert result["total"] > 0
        assert result["ce"] > 0

    def test_train_step_with_scaler(self):
        """Test training step with FP16 scaler (lines 95-97)."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 1000)
            
            def forward(self, input_ids, attention_mask):
                batch_size, seq_len = input_ids.shape
                return self.linear(torch.randn(batch_size, seq_len, 1000))
        
        model = SimpleModel()
        model.train()

        # Use CPU but create a mock scaler to test the scaler path
        scaler = Mock()
        scaled_loss = Mock()
        scaled_loss.backward = Mock()
        scaler.scale.return_value = scaled_loss
        scaler.step = Mock()
        scaler.update = Mock()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 1000, (2, 10)),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        device = torch.device("cpu")
        cfg = {}

        result = train_step(model, batch, optimizer, scaler, cfg, device)

        assert "total" in result
        assert "ce" in result
        assert isinstance(result["total"], float)
        assert isinstance(result["ce"], float)
        # Verify scaler methods were called
        scaler.scale.assert_called_once()
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once()

    def test_train_step_with_ignore_index(self):
        """Test training step with ignore_index in labels."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1000, 1000)
            
            def forward(self, input_ids, attention_mask):
                batch_size, seq_len = input_ids.shape
                return self.linear(torch.randn(batch_size, seq_len, 1000))
        
        model = SimpleModel()
        model.train()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.full((2, 10), -100),  # All ignored
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        device = torch.device("cpu")
        cfg = {}

        result = train_step(model, batch, optimizer, None, cfg, device)

        assert "total" in result
        assert "ce" in result
        # Loss should be very small or zero when all labels are ignored
        assert isinstance(result["total"], float)

    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_checkpoint_path_nonexistent(self, mock_student_lm):
        """Test create_model when checkpoint path doesn't exist (line 52)."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        cfg = {
            "init": {"base_checkpoint": "nonexistent_checkpoint.pt"},
        }
        device = torch.device("cpu")
        
        result = create_model(cfg, device)
        
        assert result == mock_model
        # Should not try to load checkpoint if path doesn't exist
        mock_model.load_state_dict.assert_not_called()

    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_no_checkpoint_config(self, mock_student_lm):
        """Test create_model when no checkpoint config is provided."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        cfg = {}  # No init config
        device = torch.device("cpu")
        
        result = create_model(cfg, device)
        
        assert result == mock_model
        mock_model.load_state_dict.assert_not_called()

    @patch("training.safe_checkpoint_loading.safe_load_checkpoint")
    @patch("training.distill_answer_generation.StudentLM")
    def test_create_model_checkpoint_no_model_state_dict(self, mock_student_lm, mock_load_checkpoint, tmp_path):
        """Test loading checkpoint without model_state_dict key (lines 58-59)."""
        mock_model = Mock()
        mock_student_lm.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.load_state_dict.return_value = None

        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.touch()

        mock_checkpoint = {"layer1.weight": torch.randn(10, 10)}  # No model_state_dict key
        mock_load_checkpoint.return_value = mock_checkpoint

        cfg = {
            "init": {"base_checkpoint": str(checkpoint_path)},
        }
        device = torch.device("cpu")

        result = create_model(cfg, device)

        assert result == mock_model
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint, strict=False)

