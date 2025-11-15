"""
Unit tests for training/safe_checkpoint_loading.py

Tests safe checkpoint loading with validation and security measures.
"""

import pytest
import torch
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from training.safe_checkpoint_loading import safe_load_checkpoint


class TestSafeLoadCheckpoint:
    """Test safe checkpoint loading functionality."""

    def test_safe_load_checkpoint_success_weights_only(self):
        """Test successful loading with weights_only=True."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "config": {"hidden_size": 128},
            "step": 100,
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            # First call succeeds with weights_only=True
            mock_load.return_value = checkpoint_data

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
            # Should call torch.load with weights_only=True first
            mock_load.assert_called_with(
                Path("test.pth"),
                map_location="cpu",
                weights_only=True
            )

    def test_safe_load_checkpoint_fallback_to_weights_only_false(self):
        """Test fallback to weights_only=False when weights_only=True fails."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "config": {"hidden_size": 128},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "meta": {"version": "1.0"},
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            # weights_only=True fails, weights_only=False succeeds
            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
            # Should call torch.load twice
            assert mock_load.call_count == 2
            # Second call should be with weights_only=False
            mock_load.assert_any_call(
                Path("test.pth"),
                map_location="cpu",
                weights_only=False
            )

    def test_safe_load_checkpoint_file_not_found(self):
        """Test error when checkpoint file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
                safe_load_checkpoint("nonexistent.pth")

    def test_safe_load_checkpoint_invalid_structure_weights_only(self):
        """Test error when checkpoint loaded with weights_only=True is not a dict."""
        with patch("torch.load", return_value="not a dict"), \
             patch("pathlib.Path.exists", return_value=True):

            with pytest.raises(ValueError, match="Checkpoint must be a dictionary"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_invalid_structure_fallback(self):
        """Test error when checkpoint loaded in fallback is not a dict."""
        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), "not a dict"]

            with pytest.raises(ValueError, match="Checkpoint must be a dictionary"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_load_failure(self):
        """Test error when both loading attempts fail."""
        with patch("torch.load", side_effect=Exception("Load failed")), \
             patch("pathlib.Path.exists", return_value=True):

            with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_unexpected_keys_warning(self):
        """Test warning for unexpected keys in checkpoint."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "unexpected_key": "value",  # This should trigger warning
            "another_unexpected": 123,
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True), \
             warnings.catch_warnings(record=True) as w:

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
            # Should have warning about unexpected keys
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "unexpected keys" in str(w[0].message)

    def test_safe_load_checkpoint_missing_required_keys(self):
        """Test error when required keys are missing."""
        checkpoint_data = {
            "config": {"hidden_size": 128},
            # Missing required "model_state_dict"
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            with pytest.raises(ValueError, match="missing required keys"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_custom_required_keys(self):
        """Test with custom required keys."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "custom_required": "value",
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint(
                "test.pth",
                required_keys={"custom_required"}
            )

            assert result == checkpoint_data

    def test_safe_load_checkpoint_custom_expected_keys(self):
        """Test with custom expected keys."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "custom_expected": "value",
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True), \
             warnings.catch_warnings(record=True) as w:

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint(
                "test.pth",
                expected_keys={"model_state_dict", "custom_expected"}
            )

            assert result == checkpoint_data
            # No warning since custom_expected is expected
            assert len(w) == 0

    def test_safe_load_checkpoint_invalid_model_state_dict(self):
        """Test error when model_state_dict is not a dict."""
        checkpoint_data = {
            "model_state_dict": "not a dict",  # Invalid
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            with pytest.raises(ValueError, match="model_state_dict must be a dictionary"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_invalid_config(self):
        """Test error when config is not a dict."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "config": "not a dict",  # Invalid
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            with pytest.raises(ValueError, match="config must be a dictionary"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_invalid_model_arch(self):
        """Test error when model_arch is not a dict."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "model_arch": "not a dict",  # Invalid
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            with pytest.raises(ValueError, match="model_arch must be a dictionary"):
                safe_load_checkpoint("test.pth")

    def test_safe_load_checkpoint_none_config_arch_allowed(self):
        """Test that None values for config and model_arch are allowed."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
            "config": None,  # None is allowed
            "model_arch": None,  # None is allowed
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint("test.pth")
            assert result == checkpoint_data

    def test_safe_load_checkpoint_map_location_parameter(self):
        """Test that map_location parameter is passed correctly."""
        checkpoint_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.return_value = checkpoint_data

            result = safe_load_checkpoint("test.pth", map_location="cuda:0")

            assert result == checkpoint_data
            # Should call with specified map_location
            mock_load.assert_called_with(
                Path("test.pth"),
                map_location="cuda:0",
                weights_only=True
            )

    def test_safe_load_checkpoint_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        checkpoint_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.return_value = checkpoint_data

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
            # Should work with string path (converted to Path internally)

    def test_safe_load_checkpoint_pathlib_path_input(self):
        """Test that Path objects are handled correctly."""
        checkpoint_data = {"model_state_dict": {"layer.weight": torch.randn(10, 5)}}
        path_obj = Path("test.pth")

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.return_value = checkpoint_data

            result = safe_load_checkpoint(path_obj)

            assert result == checkpoint_data
            # Path should be used directly (not converted again)
            mock_load.assert_called_with(
                path_obj,
                map_location="cpu",
                weights_only=True
            )

    def test_safe_load_checkpoint_comprehensive_validation(self):
        """Test comprehensive validation of a complete checkpoint."""
        checkpoint_data = {
            "model_state_dict": {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10),
                "layer2.weight": torch.randn(5, 10),
            },
            "config": {
                "hidden_size": 128,
                "num_layers": 2,
                "vocab_size": 1000,
            },
            "model_arch": {
                "type": "transformer",
                "attention": "multi-head",
                "feedforward": "mlp",
            },
            "step": 1000,
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [{"lr": 0.001}],
            },
            "loss": 2.5,
            "meta": {
                "version": "1.0",
                "timestamp": "2024-01-01",
            },
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
            # Should pass all validations without errors or warnings

    def test_safe_load_checkpoint_minimal_valid_checkpoint(self):
        """Test loading of minimal valid checkpoint."""
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 5)},
        }

        with patch("torch.load") as mock_load, \
             patch("pathlib.Path.exists", return_value=True):

            mock_load.side_effect = [Exception("weights_only failed"), checkpoint_data]

            result = safe_load_checkpoint("test.pth")

            assert result == checkpoint_data
