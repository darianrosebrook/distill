"""
Unit tests for halt head functionality.
"""
# @author: @darianrosebrook

import pytest
import torch

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


class TestHaltHead:
    """Test halt head functionality."""
    
    @pytest.fixture
    def model_cfg(self):
        """Create model config."""
        return ModelCfg(
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=32,
            vocab_size=1000,
        )
    
    @pytest.fixture
    def model_with_halt(self, model_cfg):
        """Create model with halt head."""
        return StudentLM(cfg=model_cfg, use_halt_head=True)
    
    @pytest.fixture
    def model_without_halt(self, model_cfg):
        """Create model without halt head."""
        return StudentLM(cfg=model_cfg, use_halt_head=False)
    
    def test_halt_head_initialized(self, model_with_halt):
        """Test that halt head is initialized when use_halt_head=True."""
        assert model_with_halt.use_halt_head is True
        assert hasattr(model_with_halt, "halt_head")
        assert model_with_halt.halt_head is not None
    
    def test_halt_head_not_initialized(self, model_without_halt):
        """Test that halt head is not initialized when use_halt_head=False."""
        assert model_without_halt.use_halt_head is False
        assert not hasattr(model_without_halt, "halt_head") or model_with_halt.halt_head is None
    
    def test_forward_returns_halt_logits(self, model_with_halt):
        """Test that forward returns halt logits when requested."""
        input_ids = torch.randint(0, 1000, (1, 10))
        
        result = model_with_halt(input_ids, return_halt_logits=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        logits, halt_logits = result
        assert halt_logits.shape == (1, 2)  # [B, 2]
    
    def test_forward_hidden_processes_without_lm_head(self, model_with_halt):
        """Test that forward_hidden processes hidden state without LM head."""
        hidden_state = torch.randn(1, 10, 128)
        
        result = model_with_halt.forward_hidden(hidden_state)
        
        assert result.shape == hidden_state.shape
        assert result.shape[-1] == 128  # Same hidden dimension
    
    def test_halt_head_output_shape(self, model_with_halt):
        """Test that halt head outputs correct shape."""
        input_ids = torch.randint(0, 1000, (2, 5))  # Batch size 2
        
        result = model_with_halt(input_ids, return_halt_logits=True)
        logits, halt_logits = result
        
        assert halt_logits.shape == (2, 2)  # [B, 2]
    
    def test_halt_logits_are_differentiable(self, model_with_halt):
        """Test that halt logits are differentiable."""
        input_ids = torch.randint(0, 1000, (1, 10))
        
        result = model_with_halt(input_ids, return_halt_logits=True)
        logits, halt_logits = result
        
        # Check that halt logits require grad
        assert halt_logits.requires_grad or not model_with_halt.training

