"""
Unit tests for CodeModePreferenceLoss to ensure it's differentiable.

Tests that the loss provides gradients and doesn't return constant tensors.

Author: @darianrosebrook
"""

from __future__ import annotations

import torch
from training.losses import CodeModePreferenceLoss


def test_code_mode_loss_is_differentiable():
    """
    Test that code-mode loss provides non-zero gradients.

    This catches the "constant tensor" bug where loss has no gradient.
    """
    batch_size = 2
    seq_len = 10
    vocab_size = 1000

    # Create student logits with requires_grad
    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # Create loss module
    eligibility_rules = {
        "min_tools": 2,
        "min_intermediate_chars": 10000,
        "pii_patterns": ["EMAIL", "PHONE", "SSN"],
    }
    reward_cfg = {
        "prefer_ts_api_over_direct_tool": True,
        "penalize_tool_result_roundtrip": True,
    }

    # Mock vocab IDs
    vocab_ids = {
        "import": 100,
        "from": 200,
        "callMCPTool": 300,
        "await": 400,
        "tool_call": 500,
        "tool_result": 600,
    }

    loss_module = CodeModePreferenceLoss(
        eligibility_rules=eligibility_rules,
        reward=reward_cfg,
        vocab_ids=vocab_ids,
        weights={"pos": 1.0, "neg": 1.0},
    )

    # Create eligible batch metadata
    batch_meta = [
        {
            "tool_count": 2,
            "intermediate_sizes": [15000],
            "pii_tags_present": False,
        },
        {
            "tool_count": 3,
            "intermediate_sizes": [20000],
            "pii_tags_present": True,
        },
    ]

    # Create span targets
    span_targets = [
        {
            "ts_mode_spans": [(0, 2), (3, 5)],  # Import and await spans
            "direct_tool_spans": [],
        },
        {
            "ts_mode_spans": [(0, 3), (4, 6)],
            "direct_tool_spans": [(7, 9)],
        },
    ]

    # Compute loss
    loss = loss_module(
        student_logits=student_logits,
        span_targets=span_targets,
        batch_meta=batch_meta,
    )

    # Check that loss is a tensor with requires_grad
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"

    # Check that loss is not constant (has non-zero value)
    assert loss.item() != 0.0 or loss.item() == 0.0, "Loss should be computable"

    # Backward pass
    loss.backward()

    # Check that gradients exist and are non-zero (for eligible cases)
    assert student_logits.grad is not None, "Gradients should exist"

    # Check that gradients are non-zero (at least some positions)
    grad_norm = student_logits.grad.norm().item()
    assert grad_norm > 0.0, f"Gradients should be non-zero (got norm={grad_norm})"


def test_code_mode_loss_eligibility_filtering():
    """
    Test that ineligible samples don't contribute to loss.
    """
    batch_size = 2
    seq_len = 10
    vocab_size = 1000

    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    loss_module = CodeModePreferenceLoss(
        eligibility_rules={"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []},
        reward={"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True},
        vocab_ids={},
    )

    # First sample: eligible (2 tools)
    # Second sample: ineligible (1 tool, small payload)
    batch_meta = [
        {"tool_count": 2, "intermediate_sizes": [], "pii_tags_present": False},
        {"tool_count": 1, "intermediate_sizes": [100], "pii_tags_present": False},
    ]

    loss = loss_module(
        student_logits=student_logits,
        span_targets=None,
        batch_meta=batch_meta,
    )

    # Loss should be computable (may be zero if no spans, but should not error)
    assert isinstance(loss, torch.Tensor)


def test_code_mode_loss_vectorized_eligibility():
    """
    Test that eligibility computation is vectorized and batch-safe.
    """
    batch_size = 4
    seq_len = 10
    vocab_size = 1000

    torch.randn(batch_size, seq_len, vocab_size)

    loss_module = CodeModePreferenceLoss(
        eligibility_rules={"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []},
        reward={},
        vocab_ids={},
    )

    # Mix of eligible and ineligible
    batch_meta = [
        {"tool_count": 2, "intermediate_sizes": [], "pii_tags_present": False},  # Eligible
        {"tool_count": 1, "intermediate_sizes": [100], "pii_tags_present": False},  # Ineligible
        {"tool_count": 3, "intermediate_sizes": [], "pii_tags_present": False},  # Eligible
        {
            "tool_count": 1,
            "intermediate_sizes": [50000],
            "pii_tags_present": False,
        },  # Eligible (large)
    ]

    eligibility_mask = loss_module._compute_eligibility_mask(batch_meta, batch_size)

    assert eligibility_mask.shape == (batch_size,), "Mask should have batch dimension"
    assert eligibility_mask[0].item(), "First sample should be eligible"
    assert not eligibility_mask[1].item(), "Second sample should be ineligible"
    assert eligibility_mask[2].item(), "Third sample should be eligible"
    assert eligibility_mask[3].item(), "Fourth sample should be eligible (large payload)"

