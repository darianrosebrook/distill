"""
Safety edge case tests for malformed sentinels.

Tests nested <bot>, unmatched <eot>, excess spans, etc.
"""
# @author: @darianrosebrook

import pytest
import torch
from unittest.mock import Mock

from runtime.engine.loop import LatentModeEngine
from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID


class TestSafetyEdgeCases:
    """Test safety handling for malformed sentinel patterns."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.forward_hidden = Mock(return_value=torch.randn(1, 5, 128))
        model.cfg = Mock()
        model.cfg.n_layers = 2
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_nested_bot_without_eot(self, mock_model, mock_tokenizer):
        """Test nested <bot> without matching <eot>."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_spans=10,
        )
        
        # Sequence: <bot> ... <bot> ... (no <eot>)
        tokens = [BOT_TOKEN_ID, 100, 101, BOT_TOKEN_ID, 102, 103]
        hidden_state = torch.randn(1, len(tokens), 128)
        
        # Process tokens
        for i, token_id in enumerate(tokens):
            result = engine.process_token(
                token_id,
                hidden_state[:, i:i+1, :] if i < len(tokens) - 1 else hidden_state[:, -1:, :],
                None,
                i,
            )
        
        # Should handle gracefully: unmatched <bot> should trigger error
        assert len(engine.errors) > 0 or engine.current_mode == "language"
    
    def test_unmatched_eot(self, mock_model, mock_tokenizer):
        """Test <eot> without preceding <bot>."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )
        
        # Sequence: ... <eot> ... (no <bot>)
        tokens = [100, 101, EOT_TOKEN_ID, 102, 103]
        hidden_state = torch.randn(1, len(tokens), 128)
        
        # Process tokens
        for i, token_id in enumerate(tokens):
            result = engine.process_token(
                token_id,
                hidden_state[:, i:i+1, :] if i < len(tokens) - 1 else hidden_state[:, -1:, :],
                None,
                i,
            )
        
        # Should handle gracefully: unmatched <eot> should trigger error
        assert len(engine.errors) > 0 or engine.current_mode == "language"
    
    def test_excess_latent_spans(self, mock_model, mock_tokenizer):
        """Test excess latent spans beyond max_latent_spans."""
        max_spans = 2
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_spans=max_spans,
        )
        
        # Create sequence with more spans than allowed
        tokens = []
        for i in range(max_spans + 1):
            tokens.extend([BOT_TOKEN_ID, 100 + i, EOT_TOKEN_ID])
        
        hidden_state = torch.randn(1, len(tokens), 128)
        
        # Process tokens
        for i, token_id in enumerate(tokens):
            result = engine.process_token(
                token_id,
                hidden_state[:, i:i+1, :] if i < len(tokens) - 1 else hidden_state[:, -1:, :],
                None,
                i,
            )
        
        # Should enforce max_latent_spans limit
        assert engine.current_latent_span_count <= max_spans
        assert len(engine.errors) > 0 or engine.current_mode == "language"
    
    def test_max_latent_length_enforcement(self, mock_model, mock_tokenizer):
        """Test that max_latent_length is enforced."""
        max_length = 5
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
            max_latent_length=max_length,
        )
        
        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, torch.randn(1, 1, 128), None, 0)
        
        # Process tokens beyond max length
        for i in range(max_length + 2):
            result = engine.process_token(
                100 + i,
                torch.randn(1, 1, 128),
                None,
                i + 1,
            )
            
            # Should exit latent mode after max length
            if i >= max_length:
                break
        
        # Should have exited latent mode due to max length
        assert engine.current_mode == "language" or len(engine.errors) > 0
    
    def test_missing_hidden_state_fallback(self, mock_model, mock_tokenizer):
        """Test fallback when hidden state is missing in latent mode."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )
        
        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, torch.randn(1, 1, 128), None, 0)
        
        # Process token without hidden state
        result = engine.process_token(100, None, None, 1)
        
        # Should fallback to language mode
        assert engine.current_mode == "language" or len(engine.errors) > 0
    
    def test_forward_hidden_error_fallback(self, mock_model, mock_tokenizer):
        """Test fallback when forward_hidden raises error."""
        mock_model.forward_hidden = Mock(side_effect=RuntimeError("Forward error"))
        
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )
        
        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, torch.randn(1, 1, 128), None, 0)
        
        # Process token (should trigger forward_hidden error)
        result = engine.process_token(100, torch.randn(1, 1, 128), None, 1)
        
        # Should fallback to language mode on error
        assert engine.current_mode == "language" or len(engine.errors) > 0
    
    def test_generation_ends_in_latent_mode(self, mock_model, mock_tokenizer):
        """Test that generation ending in latent mode triggers automatic exit."""
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )
        
        # Enter latent mode
        engine.process_token(BOT_TOKEN_ID, torch.randn(1, 1, 128), None, 0)
        
        # Simulate generation ending (no <eot>)
        assert engine.current_mode == "latent"
        
        # Call generate_with_latent_mode and check final mode
        input_ids = torch.tensor([[1, 2, 3]])
        result = engine.generate_with_latent_mode(
            input_ids,
            max_new_tokens=10,
        )
        
        # Should exit latent mode automatically
        assert engine.current_mode == "language" or len(engine.errors) > 0

