"""
Unit tests for latent mode detection and processing.
"""
# @author: @darianrosebrook

import pytest
import torch
from unittest.mock import Mock

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID
from runtime.engine.loop import LatentModeEngine


class TestLatentModeEngine:
    """Test latent mode engine functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with forward_hidden method."""
        model = Mock()
        model.forward_hidden = Mock(return_value=torch.randn(1, 10, 128))
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        return tokenizer

    @pytest.fixture
    def engine(self, mock_model, mock_tokenizer):
        """Create latent mode engine."""
        return LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_length=10,
            max_latent_spans=5,
        )

    def test_bot_token_switches_to_latent_mode(self, engine):
        """Test that <bot> token switches to latent mode."""
        hidden_state = torch.randn(1, 5, 128)
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            BOT_TOKEN_ID, hidden_state, None, 0
        )

        assert engine.current_mode == "latent"
        assert metadata["mode_changed"] is True
        assert metadata["transition"] == "language -> latent"
        assert token_id is None  # No token generated in latent mode

    def test_eot_token_switches_to_language_mode(self, engine):
        """Test that <eot> token switches back to language mode."""
        # First enter latent mode
        hidden_state = torch.randn(1, 5, 128)
        engine.process_token(BOT_TOKEN_ID, hidden_state, None, 0)

        # Then exit latent mode
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            EOT_TOKEN_ID, hidden_state, None, 1
        )

        assert engine.current_mode == "language"
        assert metadata["mode_changed"] is True
        assert metadata["transition"] == "latent -> language"

    def test_latent_mode_processes_hidden_state(self, engine, mock_model):
        """Test that latent mode processes hidden state without generating tokens."""
        hidden_state = torch.randn(1, 5, 128)

        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, hidden_state, None, 0)

        # Process token in latent mode
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            100, hidden_state, None, 1
        )

        assert token_id is None
        assert updated_hidden is not None
        assert mock_model.forward_hidden.called

    def test_max_latent_length_safety_check(self, engine):
        """Test that max latent length triggers safety exit."""
        hidden_state = torch.randn(1, 5, 128)
        engine.max_latent_length = 3

        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, hidden_state, None, 0)

        # Process tokens until max length
        for i in range(5):
            engine.process_token(100 + i, hidden_state, None, i + 1)

        # Should have exited latent mode
        assert engine.current_mode == "language"
        assert len(engine.errors) > 0

    def test_unmatched_bot_token_error(self, engine):
        """Test that unmatched <bot> token generates error."""
        hidden_state = torch.randn(1, 5, 128)

        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, hidden_state, None, 0)

        # Try to enter latent mode again (unmatched <bot>)
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            BOT_TOKEN_ID, hidden_state, None, 1
        )

        assert "unmatched_bot" in metadata.get("error", "")
        assert len(engine.errors) > 0

    def test_unmatched_eot_token_error(self, engine):
        """Test that unmatched <eot> token generates error."""
        # Try to exit latent mode without entering (unmatched <eot>)
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            EOT_TOKEN_ID, None, None, 0
        )

        assert "unmatched_eot" in metadata.get("error", "")
        assert len(engine.errors) > 0

    def test_max_latent_spans_limit(self, engine):
        """Test that max latent spans limit is enforced."""
        hidden_state = torch.randn(1, 5, 128)
        engine.max_latent_spans = 2

        # Enter latent mode multiple times
        for i in range(5):
            # Exit latent mode before each attempt (except first)
            if i > 0 and engine.current_mode == "latent":
                # Exit latent mode by processing EOT
                engine.process_token(EOT_TOKEN_ID, hidden_state, None, i * 2 - 1)
            
            token_id, updated_hidden, kv_caches, metadata = engine.process_token(
                BOT_TOKEN_ID, hidden_state, None, i * 2
            )
            if i < 2:
                # First two should succeed
                assert engine.current_mode == "latent"
            else:
                # After max, should be rejected
                assert (
                    metadata.get("error") == "max_latent_spans" or engine.current_mode == "language"
                )

    def test_latent_mode_disabled(self, mock_model, mock_tokenizer):
        """Test that latent mode is disabled when flag is False."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=False,
        )

        hidden_state = torch.randn(1, 5, 128)
        token_id, updated_hidden, kv_caches, metadata = engine.process_token(
            BOT_TOKEN_ID, hidden_state, None, 0
        )

        # Should process normally (no latent mode)
        assert token_id == BOT_TOKEN_ID
        assert engine.current_mode == "language"
