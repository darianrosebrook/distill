"""
Tests for training/tokenizer_migration.py - Tokenizer/vocab migration utilities.

Tests token ID verification, embedding resizing, and special token handling.
"""
# @author: @darianrosebrook

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
import torch.nn as nn
from training.tokenizer_migration import (
    verify_token_ids,
    resize_model_embeddings,
)


class TestVerifyTokenIDs:
    """Test verify_token_ids function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {"<|bot|>": 1, "<|eot|>": 2}.get(x, -1))
        return tokenizer

    def test_verify_token_ids_matching(self, mock_tokenizer):
        """Test verification when token IDs match."""
        # Mock the constants to match
        with patch("training.tokenizer_migration.BOT_TOKEN_ID", 1), \
             patch("training.tokenizer_migration.EOT_TOKEN_ID", 2):
            result = verify_token_ids(mock_tokenizer)

            assert result["ids_match"] == True
            assert len(result["errors"]) == 0
            assert result["bot_token_id"] == 1
            assert result["eot_token_id"] == 2

    def test_verify_token_ids_mismatch(self, mock_tokenizer):
        """Test verification when token IDs don't match."""
        # Mock the constants to not match
        with patch("training.tokenizer_migration.BOT_TOKEN_ID", 10), \
             patch("training.tokenizer_migration.EOT_TOKEN_ID", 20):
            result = verify_token_ids(mock_tokenizer)

            assert result["ids_match"] == False
            assert len(result["errors"]) == 2
            assert "BOT token ID mismatch" in result["errors"][0]
            assert "EOT token ID mismatch" in result["errors"][1]

    def test_verify_token_ids_no_convert_method(self):
        """Test verification when tokenizer has no convert_tokens_to_ids."""
        tokenizer = Mock()
        del tokenizer.convert_tokens_to_ids

        result = verify_token_ids(tokenizer)

        assert result["bot_token_id"] is None
        assert result["eot_token_id"] is None
        assert result["ids_match"] == False


class TestResizeModelEmbeddings:
    """Test resize_model_embeddings function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with embeddings."""
        model = Mock(spec=nn.Module)
        model.embedding = Mock(spec=nn.Embedding)
        model.embedding.num_embeddings = 1000
        model.embedding.embedding_dim = 512
        model.lm_head = Mock(spec=nn.Linear)
        model.lm_head.out_features = 1000
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.vocab_size = 2000
        return tokenizer

    def test_resize_model_embeddings_basic(self, mock_model, mock_tokenizer):
        """Test basic embedding resizing."""
        resized_model, metadata = resize_model_embeddings(mock_model, mock_tokenizer)

        assert metadata["new_vocab_size"] == 2000
        assert metadata["original_vocab_size"] == 1000
        assert metadata["embedding_resized"] == True

    def test_resize_model_embeddings_with_new_vocab_size(self, mock_model, mock_tokenizer):
        """Test resizing with explicit new vocab size."""
        resized_model, metadata = resize_model_embeddings(
            mock_model, mock_tokenizer, new_vocab_size=1500
        )

        assert metadata["new_vocab_size"] == 1500

    def test_resize_model_embeddings_tokenizer_len(self):
        """Test resizing with tokenizer that uses __len__."""
        model = Mock(spec=nn.Module)
        model.embedding = Mock(spec=nn.Embedding)
        model.embedding.num_embeddings = 1000

        tokenizer = Mock()
        del tokenizer.vocab_size
        tokenizer.__len__ = Mock(return_value=2000)

        resized_model, metadata = resize_model_embeddings(model, tokenizer)

        assert metadata["new_vocab_size"] == 2000

    def test_resize_model_embeddings_no_vocab_size(self):
        """Test resizing when tokenizer has no vocab size."""
        model = Mock(spec=nn.Module)
        model.embedding = Mock(spec=nn.Embedding)
        model.embedding.num_embeddings = 1000

        tokenizer = Mock()
        del tokenizer.vocab_size
        del tokenizer.__len__

        with pytest.raises(ValueError, match="Cannot determine tokenizer vocabulary size"):
            resize_model_embeddings(model, tokenizer)

