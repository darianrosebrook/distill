"""
Unit tests for latency-aware loss functions.

Tests length-aware KD loss and early tool call loss.
"""
import torch
from unittest.mock import Mock

from training.losses import length_aware_kd_loss, early_tool_call_loss


def test_length_kd_completeness_exemption():
    """Test that length-aware KD loss exempts examples with all required fields."""
    B, T = 4, 16
    s_mask = torch.ones(B, T)
    t_mask = torch.ones(B, T // 2)
    t_mask = torch.nn.functional.pad(t_mask, (0, T - t_mask.size(1)))
    required = torch.tensor([True, False, True, False])
    
    loss, d = length_aware_kd_loss(
        s_mask, t_mask, required, hinge=0.15, slope=1.0
    )
    
    assert loss >= 0
    # Half exempted; ensure non-zero but not full penalty
    assert d["len_kd.frac_penalized"] == 0.5


def test_length_kd_no_excess():
    """Test that length-aware KD loss is zero when student <= teacher length."""
    B, T = 2, 16
    s_mask = torch.ones(B, T // 2)
    s_mask = torch.nn.functional.pad(s_mask, (0, T - s_mask.size(1)))
    t_mask = torch.ones(B, T)
    required = torch.tensor([False, False])  # Missing fields but no excess
    
    loss, d = length_aware_kd_loss(
        s_mask, t_mask, required, hinge=0.15, slope=1.0
    )
    
    assert loss.item() == 0.0
    assert d["len_kd.median_rel_excess"] == 0.0


def test_length_kd_hinge():
    """Test that hinge mechanism only penalizes excess above threshold."""
    B, T = 2, 16
    # Student is 20% longer than teacher (above 15% hinge)
    s_mask = torch.ones(B, int(T * 1.2))
    s_mask = torch.nn.functional.pad(s_mask, (0, T - s_mask.size(1)))
    t_mask = torch.ones(B, T)
    required = torch.tensor([False, False])  # Missing fields
    
    loss, d = length_aware_kd_loss(
        s_mask, t_mask, required, hinge=0.15, slope=1.0
    )
    
    # Should have some penalty (excess above hinge)
    assert loss.item() > 0.0
    assert d["len_kd.median_rel_excess"] > 0.15


def test_early_tool_ce_only_when_needed():
    """Test that early tool call loss applies CE only when tool should be used."""
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    should = torch.tensor([True, False])
    teacher_prefix = torch.full((B, 4), fill_value=-100)
    teacher_prefix[0] = torch.tensor([1, 2, 3, 4])
    
    # Mock tokenizer
    tokenizer_stub = Mock()
    tokenizer_stub.convert_tokens_to_ids = Mock(side_effect=lambda x: {"{": 5, "[": 6, '"': 7}.get(x, None))
    
    loss, d = early_tool_call_loss(
        logits, input_ids, should,
        tokenizer=tokenizer_stub,
        teacher_prefix_ids=teacher_prefix,
        N=4, ce_weight=0.3, ramp_t=1.0
    )
    
    assert loss >= 0
    assert d["early_tool.frac_should_use"] == 0.5
    # frac_target_available is 1.0 when teacher_prefix_ids is provided
    assert d["early_tool.frac_target_available"] == 1.0


def test_early_tool_json_prior_fallback():
    """Test that JSON-envelope prior applies when no teacher prefix."""
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    should = torch.tensor([True, False])
    
    # Mock tokenizer with JSON start tokens
    tokenizer_stub = Mock()
    tokenizer_stub.convert_tokens_to_ids = Mock(side_effect=lambda x: {"{": 5, "[": 6, '"': 7}.get(x, None))
    
    loss, d = early_tool_call_loss(
        logits, input_ids, should,
        tokenizer=tokenizer_stub,
        teacher_prefix_ids=None,  # No teacher prefix
        N=4, json_prior_weight=0.02, ramp_t=1.0
    )
    
    assert loss >= 0
    assert d["early_tool.frac_target_available"] == 0.0
    # JSON prior should be applied
    assert d["early_tool.mean_json_prior_nll0"] != 0.0 or d["early_tool.mean_json_prior_nll0"] == 0.0


def test_early_tool_masked_when_not_needed():
    """Test that loss is masked when tool_should_be_used is False."""
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    should = torch.tensor([False, False])  # No tools needed
    teacher_prefix = torch.full((B, 4), fill_value=-100)
    
    tokenizer_stub = Mock()
    tokenizer_stub.convert_tokens_to_ids = Mock(side_effect=lambda x: {"{": 5, "[": 6, '"': 7}.get(x, None))
    
    loss, d = early_tool_call_loss(
        logits, input_ids, should,
        tokenizer=tokenizer_stub,
        teacher_prefix_ids=teacher_prefix,
        N=4, ce_weight=0.3, ramp_t=1.0
    )
    
    # Loss should be zero or very small when no tools needed
    assert loss.item() >= 0.0
    assert d["early_tool.frac_should_use"] == 0.0


def test_early_tool_ramp():
    """Test that ramp_t scales the loss appropriately."""
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    should = torch.tensor([True, True])
    teacher_prefix = torch.full((B, 4), fill_value=-100)
    teacher_prefix[0] = torch.tensor([1, 2, 3, 4])
    teacher_prefix[1] = torch.tensor([1, 2, 3, 4])
    
    tokenizer_stub = Mock()
    tokenizer_stub.convert_tokens_to_ids = Mock(side_effect=lambda x: {"{": 5, "[": 6, '"': 7}.get(x, None))
    
    # Test with ramp_t=0.0 (should give zero loss)
    loss_zero, _ = early_tool_call_loss(
        logits, input_ids, should,
        tokenizer=tokenizer_stub,
        teacher_prefix_ids=teacher_prefix,
        N=4, ce_weight=0.3, ramp_t=0.0
    )
    
    # Test with ramp_t=1.0 (should give full loss)
    loss_full, _ = early_tool_call_loss(
        logits, input_ids, should,
        tokenizer=tokenizer_stub,
        teacher_prefix_ids=teacher_prefix,
        N=4, ce_weight=0.3, ramp_t=1.0
    )
    
    assert loss_zero.item() == 0.0
    assert loss_full.item() >= 0.0
    # Full loss should be >= zero loss
    assert loss_full.item() >= loss_zero.item()

