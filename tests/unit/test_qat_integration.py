"""
Unit tests for QAT integration into training loop.

Tests QAT enablement logic, model quantization, and stability checks.
"""
import torch
import pytest
from unittest.mock import Mock, patch, MagicMock

from training.distill_kd import (
    should_enable_qat,
    apply_qat_to_model,
    check_qat_stability,
)


class TestShouldEnableQAT:
    """Tests for QAT enablement logic."""
    
    def test_qat_disabled(self):
        """Test that QAT is disabled when config says so."""
        qat_cfg = {"enabled": False}
        assert should_enable_qat(step=1000, total_steps=1000, qat_cfg=qat_cfg) is False
    
    def test_qat_enabled_at_start_fraction(self):
        """Test that QAT enables at configured start fraction."""
        qat_cfg = {"enabled": True, "start_fraction": 0.8}
        total_steps = 1000
        
        # Before start fraction
        assert should_enable_qat(step=799, total_steps=total_steps, qat_cfg=qat_cfg) is False
        
        # At start fraction
        assert should_enable_qat(step=800, total_steps=total_steps, qat_cfg=qat_cfg) is True
        
        # After start fraction
        assert should_enable_qat(step=900, total_steps=total_steps, qat_cfg=qat_cfg) is True
    
    def test_qat_default_start_fraction(self):
        """Test default start fraction (0.8 = last 20%)."""
        qat_cfg = {"enabled": True}  # No start_fraction specified
        total_steps = 1000
        
        # Default should be 0.8
        assert should_enable_qat(step=799, total_steps=total_steps, qat_cfg=qat_cfg) is False
        assert should_enable_qat(step=800, total_steps=total_steps, qat_cfg=qat_cfg) is True


class TestApplyQATToModel:
    """Tests for QAT model application."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for QAT testing."""
        model = Mock()
        model.blocks = [Mock() for _ in range(2)]
        model.lm_head = Mock()
        return model
    
    def test_qat_not_available(self, mock_model, device):
        """Test that QAT raises error when not available."""
        with patch('training.distill_kd.QAT_AVAILABLE', False):
            qat_cfg = {"enabled": True, "weight_bits": 8, "act_bits": 8}
            with pytest.raises(RuntimeError, match="QAT not available"):
                apply_qat_to_model(mock_model, qat_cfg, device)
    
    def test_qat_config_parameters(self, mock_model, device):
        """Test that QAT uses config parameters."""
        with patch('training.distill_kd.QAT_AVAILABLE', True):
            with patch('training.distill_kd.quantize_model') as mock_quantize:
                mock_quantize.return_value = mock_model
                
                qat_cfg = {
                    "enabled": True,
                    "weight_bits": 8,
                    "act_bits": 8,
                    "fake_quant_in_attention": True,
                    "clamp_pre_softmax": True,
                }
                
                result = apply_qat_to_model(mock_model, qat_cfg, device)
                
                mock_quantize.assert_called_once()
                call_kwargs = mock_quantize.call_args[1]
                assert call_kwargs["weight_bits"] == 8
                assert call_kwargs["act_bits"] == 8
                assert call_kwargs["fake_quant_in_attention"] is True
                assert call_kwargs["clamp_pre_softmax"] is True


class TestCheckQATStability:
    """Tests for QAT stability checks."""
    
    @pytest.fixture
    def mock_model(self, device):
        """Create a mock model."""
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        
        def forward_mock(input_ids, attention_mask=None):
            return torch.randn(input_ids.shape[0], input_ids.shape[1], 32000)
        
        model.side_effect = forward_mock
        return model
    
    def test_stability_check_no_nan(self, mock_model, device):
        """Test stability check when no NaNs."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        
        # Mock model to return valid logits
        mock_model.return_value = torch.randn(2, 10, 32000)
        
        metrics = check_qat_stability(mock_model, batch, device)
        
        assert metrics["qat_stability.has_nan"] == 0.0
        assert "qat_stability.cosine_sim" in metrics
    
    def test_stability_check_with_nan(self, mock_model, device):
        """Test stability check when NaNs detected."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }
        
        # Mock model to return NaN logits
        nan_logits = torch.randn(2, 10, 32000)
        nan_logits[0, 0, 0] = float('nan')
        mock_model.return_value = nan_logits
        
        metrics = check_qat_stability(mock_model, batch, device)
        
        assert metrics["qat_stability.has_nan"] == 1.0
    
    def test_stability_check_error_handling(self, mock_model, device):
        """Test stability check error handling."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }
        
        # Mock model to raise exception
        mock_model.side_effect = RuntimeError("Model error")
        
        metrics = check_qat_stability(mock_model, batch, device)
        
        assert metrics["qat_stability.has_nan"] == 1.0
        assert "qat_stability.error" in metrics
    
    def test_stability_check_model_mode(self, mock_model, device):
        """Test that model is set to eval then train."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
        }
        mock_model.return_value = torch.randn(2, 10, 32000)
        
        check_qat_stability(mock_model, batch, device)
        
        # Model should be set to eval, then train
        mock_model.eval.assert_called_once()
        mock_model.train.assert_called_once()

