"""
Smoke test for latent mode sentinel parsing.

Verifies that runner parses sentinels correctly (≤4 min CPU time).
"""
# @author: @darianrosebrook

import pytest
import torch
import time
from unittest.mock import Mock

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID
from runtime.engine.loop import LatentModeEngine


class TestLatentSmoke:
    """Smoke tests for latent mode functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.forward_hidden = Mock(return_value=torch.randn(1, 5, 128))
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        return tokenizer

    def test_smoke_sentinel_parsing(self, mock_model, mock_tokenizer):
        """Smoke test: verify sentinel tokens are parsed correctly."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )

        # Create sequence with sentinels: <bot> ... <eot>
        tokens = [BOT_TOKEN_ID, 100, 101, 102, EOT_TOKEN_ID, 200, 201]
        hidden_state = torch.randn(1, len(tokens), 128)

        start_time = time.time()

        # Process tokens
        for i, token_id in enumerate(tokens):
            engine.process_token(token_id, hidden_state[:, i : i + 1, :], None, i)

        elapsed = time.time() - start_time

        # Should complete quickly (smoke test)
        assert elapsed < 1.0  # Should be much faster than 4 min

        # Verify mode transitions
        assert len(engine.mode_transitions) >= 2  # At least enter and exit

        # Verify final mode is language
        assert engine.current_mode == "language"

    def test_smoke_multiple_spans(self, mock_model, mock_tokenizer):
        """Smoke test: verify multiple latent spans are handled."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )

        # Create sequence with multiple spans
        tokens = [
            BOT_TOKEN_ID,
            100,
            EOT_TOKEN_ID,
            200,
            BOT_TOKEN_ID,
            300,
            301,
            EOT_TOKEN_ID,
            400,
        ]
        hidden_state = torch.randn(1, len(tokens), 128)

        start_time = time.time()

        for i, token_id in enumerate(tokens):
            engine.process_token(token_id, hidden_state[:, i : i + 1, :], None, i)

        elapsed = time.time() - start_time

        assert elapsed < 1.0
        assert len(engine.latent_span_lengths) == 2
        assert engine.current_mode == "language"
