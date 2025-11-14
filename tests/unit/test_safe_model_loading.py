"""
Tests for training/safe_model_loading.py - Safe model loading with revision pinning.

Tests revision pinning, config loading, and safe model/tokenizer loading functions.
"""
# @author: @darianrosebrook

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import warnings
import yaml

from training.safe_model_loading import (
    get_model_revision,
    safe_from_pretrained_tokenizer,
    safe_from_pretrained_model,
    safe_from_pretrained_causal_lm,
)


class TestGetModelRevision:
    """Tests for get_model_revision function."""

    def test_get_model_revision_not_configured(self):
        """Test get_model_revision when model is not in config."""
        # Mock config file to not exist
        with patch("pathlib.Path.exists", return_value=False):
            revision = get_model_revision("test-model")
            assert revision is None

    def test_get_model_revision_dict_format(self, tmp_path):
        """Test get_model_revision with dict format in config."""
        config = {
            "model_revisions": {
                "test-model": {
                    "revision": "abc123"
                }
            }
        }

        # Mock the file operations to return our config
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("yaml.safe_load", return_value=config):
                    revision = get_model_revision("test-model")
                    assert revision == "abc123"

    def test_get_model_revision_string_format(self):
        """Test get_model_revision with legacy string format in config."""
        config = {
            "model_revisions": {
                "test-model": "abc123"  # Legacy format: just string
            }
        }

        # Mock the file operations to return our config
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    revision = get_model_revision("test-model")
                    assert revision == "abc123"

    def test_get_model_revision_config_load_error(self):
        """Test get_model_revision handles config loading errors gracefully."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=IOError("File error")):
                revision = get_model_revision("test-model")
                # Should return None on error
                assert revision is None

    def test_get_model_revision_yaml_error(self):
        """Test get_model_revision handles YAML parsing errors."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                    revision = get_model_revision("test-model")
                    # Should return None on YAML error
                    assert revision is None


class TestSafeFromPretrainedTokenizer:
    """Tests for safe_from_pretrained_tokenizer function."""

    @patch("training.safe_model_loading.AutoTokenizer")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_tokenizer_with_explicit_revision(self, mock_get_revision, mock_tokenizer_class):
        """Test safe_from_pretrained_tokenizer with explicit revision."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = safe_from_pretrained_tokenizer("test-model", revision="abc123")

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="abc123",
            use_fast=True,
            trust_remote_code=False,
        )
        assert result == mock_tokenizer
        mock_get_revision.assert_not_called()

    @patch("training.safe_model_loading.AutoTokenizer")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_tokenizer_with_config_revision(self, mock_get_revision, mock_tokenizer_class):
        """Test safe_from_pretrained_tokenizer using revision from config."""
        mock_get_revision.return_value = "config-revision"
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = safe_from_pretrained_tokenizer("test-model")

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="config-revision",
            use_fast=True,
            trust_remote_code=False,
        )
        assert result == mock_tokenizer

    @patch("training.safe_model_loading.AutoTokenizer")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_tokenizer_fallback_to_main(self, mock_get_revision, mock_tokenizer_class):
        """Test safe_from_pretrained_tokenizer falls back to 'main' when no revision."""
        mock_get_revision.return_value = None
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_from_pretrained_tokenizer("test-model")

        # Should warn about using 'main'
        assert len(w) == 1
        assert "not in revision map" in str(w[0].message)
        
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="main",
            use_fast=True,
            trust_remote_code=False,
        )
        assert result == mock_tokenizer

    def test_safe_from_pretrained_tokenizer_invalid_model_name(self):
        """Test safe_from_pretrained_tokenizer with invalid model_name."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_tokenizer("")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_tokenizer(None)

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_tokenizer(123)

    @patch("training.safe_model_loading.AutoTokenizer")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_tokenizer_with_kwargs(self, mock_get_revision, mock_tokenizer_class):
        """Test safe_from_pretrained_tokenizer passes through kwargs."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = safe_from_pretrained_tokenizer(
            "test-model",
            revision="abc123",
            use_fast=False,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="float16"
        )

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="abc123",
            use_fast=False,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="float16",
        )
        assert result == mock_tokenizer


class TestSafeFromPretrainedModel:
    """Tests for safe_from_pretrained_model function."""

    @patch("training.safe_model_loading.AutoModel")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_model_with_explicit_revision(self, mock_get_revision, mock_model_class):
        """Test safe_from_pretrained_model with explicit revision."""
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        result = safe_from_pretrained_model("test-model", revision="abc123")

        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="abc123",
            trust_remote_code=False,
        )
        assert result == mock_model
        mock_get_revision.assert_not_called()

    @patch("training.safe_model_loading.AutoModel")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_model_fallback_to_main(self, mock_get_revision, mock_model_class):
        """Test safe_from_pretrained_model falls back to 'main' when no revision."""
        mock_get_revision.return_value = None
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_from_pretrained_model("test-model")

        # Should warn about using 'main'
        assert len(w) == 1
        assert "not in revision map" in str(w[0].message)
        
        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="main",
            trust_remote_code=False,
        )
        assert result == mock_model

    def test_safe_from_pretrained_model_invalid_model_name(self):
        """Test safe_from_pretrained_model with invalid model_name."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_model("")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_model(None)

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_model(123)


class TestSafeFromPretrainedCausalLM:
    """Tests for safe_from_pretrained_causal_lm function."""

    @patch("training.safe_model_loading.AutoModelForCausalLM")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_causal_lm_with_explicit_revision(self, mock_get_revision, mock_model_class):
        """Test safe_from_pretrained_causal_lm with explicit revision."""
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        result = safe_from_pretrained_causal_lm("test-model", revision="abc123")

        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="abc123",
            trust_remote_code=False,
        )
        assert result == mock_model
        mock_get_revision.assert_not_called()

    @patch("training.safe_model_loading.AutoModelForCausalLM")
    @patch("training.safe_model_loading.get_model_revision")
    def test_safe_from_pretrained_causal_lm_fallback_to_main(self, mock_get_revision, mock_model_class):
        """Test safe_from_pretrained_causal_lm falls back to 'main' when no revision."""
        mock_get_revision.return_value = None
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_from_pretrained_causal_lm("test-model")

        # Should warn about using 'main'
        assert len(w) == 1
        assert "not in revision map" in str(w[0].message)
        
        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            revision="main",
            trust_remote_code=False,
        )
        assert result == mock_model

    def test_safe_from_pretrained_causal_lm_invalid_model_name(self):
        """Test safe_from_pretrained_causal_lm with invalid model_name."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_causal_lm("")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_causal_lm(None)

        with pytest.raises(ValueError, match="must be a non-empty string"):
            safe_from_pretrained_causal_lm(123)

