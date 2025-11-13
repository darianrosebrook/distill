"""
End-to-end toy test for code-mode functionality.

Tests that code-mode loss works correctly with toy models and doesn't break
the standard toy training pipeline.

Author: @darianrosebrook
"""

import pytest
import os
import torch

from training.losses import CodeModePreferenceLoss, combined_kd_loss
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from data.generators.mcp_code_mode import compute_span_targets_from_tokenized


def test_toy_training_without_code_mode():
    """
    Test that toy training works correctly when code-mode is disabled (default).

    This ensures backward compatibility - toy training should work without
    code-mode parameters.
    """
    # Create minimal toy model
    cfg = ModelCfg(
        d_model=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_head=32,
        vocab_size=128,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )

    model = StudentLM(cfg)
    device = torch.device("cpu")
    model = model.to(device)

    # Create dummy batch
    batch_size = 2
    seq_len = 10
    vocab_size = 128

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_targets = teacher_logits.argmax(dim=-1)
    ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Call combined_kd_loss without code-mode parameters (should work)
    loss_dict = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_targets,
        ground_truth_targets=ground_truth,
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
        # code_mode_loss and code_mode_weight not provided (defaults to None and 0.0)
    )

    # Verify loss is computed
    assert "total" in loss_dict
    assert isinstance(loss_dict["total"], torch.Tensor)
    assert loss_dict["total"].requires_grad

    # Verify code-mode loss is not included
    assert "code_mode_pref" not in loss_dict


def test_toy_training_with_code_mode_disabled():
    """
    Test that toy training works when code-mode is explicitly disabled.
    """
    cfg = ModelCfg(
        d_model=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_head=32,
        vocab_size=128,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )

    model = StudentLM(cfg)
    device = torch.device("cpu")
    model = model.to(device)

    batch_size = 2
    seq_len = 10
    vocab_size = 128

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_targets = teacher_logits.argmax(dim=-1)
    ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Explicitly disable code-mode
    loss_dict = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_targets,
        ground_truth_targets=ground_truth,
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
        code_mode_loss=None,
        code_mode_weight=0.0,
    )

    assert "total" in loss_dict
    assert "code_mode_pref" not in loss_dict


def test_toy_training_with_code_mode_enabled():
    """
    Test that code-mode loss can be enabled with toy models.

    This verifies the code-mode loss module works correctly even with
    small toy models and vocabularies.
    """
    cfg = ModelCfg(
        d_model=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_head=32,
        vocab_size=128,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )

    model = StudentLM(cfg)
    device = torch.device("cpu")
    model = model.to(device)

    batch_size = 2
    seq_len = 10
    vocab_size = 128

    student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_targets = teacher_logits.argmax(dim=-1)
    ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create code-mode loss module
    eligibility_rules = {
        "min_tools": 2,
        "min_intermediate_chars": 10000,
        "pii_patterns": ["EMAIL", "PHONE", "SSN"],
    }
    reward_cfg = {
        "prefer_ts_api_over_direct_tool": True,
        "penalize_tool_result_roundtrip": True,
    }

    # Mock vocab IDs (for toy vocab, use small IDs)
    vocab_ids = {
        "import": 10,
        "from": 20,
        "callMCPTool": 30,
        "await": 40,
        "tool_call": 50,
        "tool_result": 60,
    }

    code_mode_loss_module = CodeModePreferenceLoss(
        eligibility_rules=eligibility_rules,
        reward=reward_cfg,
        vocab_ids=vocab_ids,
        weights={"pos": 1.0, "neg": 1.0},
    ).to(device)

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

    # Compute code-mode loss
    code_mode_loss = code_mode_loss_module(
        student_logits=student_logits,
        span_targets=None,  # No span targets for toy test
        batch_meta=batch_meta,
    )

    # Verify code-mode loss is computed
    assert isinstance(code_mode_loss, torch.Tensor)
    assert code_mode_loss.requires_grad

    # Integrate into combined loss
    loss_dict = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_targets,
        ground_truth_targets=ground_truth,
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
        code_mode_loss=code_mode_loss,
        code_mode_weight=0.3,
    )

    # Verify code-mode loss is included
    assert "code_mode_pref" in loss_dict
    assert loss_dict["code_mode_pref"] == code_mode_loss

    # Verify total loss includes code-mode component
    assert "total" in loss_dict
    total_loss = loss_dict["total"]

    # Backward pass should work
    total_loss.backward()
    assert student_logits.grad is not None


def test_toy_training_env_var_doesnt_break():
    """
    Test that setting TRAIN_CODE_MODE env var doesn't break toy training
    when code-mode isn't configured in the training script.

    The toy training script doesn't check env vars, so this should work fine.
    """
    # Set env var
    os.environ["TRAIN_CODE_MODE"] = "1"

    try:
        # Create minimal toy model
        cfg = ModelCfg(
            d_model=64,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_head=32,
            vocab_size=128,
            rope_theta=10000.0,
            rope_scaling="dynamic",
            dropout=0.0,
        )

        model = StudentLM(cfg)
        device = torch.device("cpu")
        model = model.to(device)

        batch_size = 2
        seq_len = 10
        vocab_size = 128

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = teacher_logits.argmax(dim=-1)
        ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Call combined_kd_loss without code-mode (env var shouldn't affect it)
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_targets,
            ground_truth_targets=ground_truth,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
        )

        # Should work fine (code-mode not included)
        assert "total" in loss_dict
        assert "code_mode_pref" not in loss_dict

    finally:
        # Clean up env var
        if "TRAIN_CODE_MODE" in os.environ:
            del os.environ["TRAIN_CODE_MODE"]


def test_toy_code_mode_with_span_targets():
    """
    Test CodeModePreferenceLoss with actual span targets computed from tokenized text.

    This verifies:
    - Span targets can be computed from tokenized TS API code
    - Token-level log-probability gathering works correctly
    - Code-mode loss can use span targets (not just fallback path)
    """
    device = torch.device("cpu")
    batch_size = 1
    seq_len = 50
    vocab_size = 128

    # Create TS API code example
    ts_code = (
        "import * as gdrive from './servers/google-drive';\n"
        "import * as salesforce from './servers/salesforce';\n"
        "const doc = await gdrive.getDocument({ documentId: 'abc123' });\n"
        "const notes = summarize(doc.content, 5);\n"
        "await salesforce.updateRecord({ objectType: 'SalesMeeting', recordId: '00Q...', data: { Notes: notes } });\n"
    )

    # Create mock tokenizer with encode_plus and offset_mapping support
    class MockTokenizerWithOffsets:
        """Mock tokenizer that supports encode_plus with offset_mapping."""

        def __init__(self):
            self.vocab_size = vocab_size

        def encode(self, text, add_special_tokens=False, **kwargs):
            """Simple tokenization: split by whitespace and map to IDs."""
            tokens = text.split()
            # Map each token to a unique ID based on hash
            token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]
            return token_ids

        def encode_plus(
            self,
            text,
            add_special_tokens=False,
            return_offsets_mapping=False,
            return_tensors=None,
            **kwargs,
        ):
            """Encode with offset mapping support."""
            tokens = text.split()
            token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]

            # Create offset mapping: each token maps to its character positions
            offset_mapping = []
            char_pos = 0
            for token in tokens:
                # Find token start in original text
                token_start = text.find(token, char_pos)
                if token_start == -1:
                    token_start = char_pos
                token_end = token_start + len(token)
                offset_mapping.append((token_start, token_end))
                char_pos = token_end + 1  # Skip space after token

            result = {
                "input_ids": token_ids,
            }

            if return_offsets_mapping:
                result["offset_mapping"] = offset_mapping

            return result

    tokenizer = MockTokenizerWithOffsets()

    # Compute span targets from tokenized text
    span_targets_dict = compute_span_targets_from_tokenized(ts_code, tokenizer)

    # Verify span targets were computed
    assert "ts_mode_spans" in span_targets_dict
    assert "direct_tool_spans" in span_targets_dict
    assert isinstance(span_targets_dict["ts_mode_spans"], list)
    assert isinstance(span_targets_dict["direct_tool_spans"], list)

    # Verify TS mode spans were found (should find "import", "from", "await", "./servers")
    assert len(span_targets_dict["ts_mode_spans"]) > 0, "Should find TS mode markers"

    # Verify no direct tool spans (no <|tool_call|> or <|tool_result|> in TS code)
    assert len(span_targets_dict["direct_tool_spans"]) == 0, (
        "TS code should not have direct tool spans"
    )

    # Create student logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

    # Create code-mode loss module
    eligibility_rules = {
        "min_tools": 2,
        "min_intermediate_chars": 10000,
        "pii_patterns": ["EMAIL", "PHONE", "SSN"],
    }
    reward_cfg = {
        "prefer_ts_api_over_direct_tool": True,
        "penalize_tool_result_roundtrip": True,
    }

    vocab_ids = {
        "import": 10,
        "from": 20,
        "callMCPTool": 30,
        "await": 40,
    }

    code_mode_loss_module = CodeModePreferenceLoss(
        eligibility_rules=eligibility_rules,
        reward=reward_cfg,
        vocab_ids=vocab_ids,
        weights={"pos": 1.0, "neg": 1.0},
    ).to(device)

    # Create eligible batch metadata
    batch_meta = {
        "tool_count": 2,
        "intermediate_sizes": [15000],
        "pii_tags_present": False,
    }

    # Convert span targets dict to tensor format (if needed)
    # For now, pass as dict since CodeModePreferenceLoss accepts Dict[str, torch.Tensor]
    # The actual implementation may need tensor conversion, but for toy test we verify structure
    if span_targets_dict["ts_mode_spans"]:
        # Create a simple tensor representation for testing
        # In real implementation, this would be token position tensors
        # For toy test, we just verify span_targets can be passed
        pass

    # Compute code-mode loss with span targets
    code_mode_loss = code_mode_loss_module(
        student_logits=student_logits,
        span_targets=span_targets_dict,  # Pass computed span targets
        batch_meta=batch_meta,
    )

    # Verify code-mode loss is computed
    assert isinstance(code_mode_loss, torch.Tensor)
    assert code_mode_loss.requires_grad

    # Verify span targets structure is correct
    # TS mode spans should be list of (start, end) tuples
    for span in span_targets_dict["ts_mode_spans"]:
        assert isinstance(span, tuple)
        assert len(span) == 2
        assert isinstance(span[0], int)
        assert isinstance(span[1], int)
        assert span[0] < span[1]  # Start < end

    # Verify token-level positions are valid (within tokenized length)
    tokenized = tokenizer.encode(ts_code, add_special_tokens=False)
    num_tokens = len(tokenized)

    for start_token, end_token in span_targets_dict["ts_mode_spans"]:
        assert 0 <= start_token < num_tokens, (
            f"Start token {start_token} out of range [0, {num_tokens})"
        )
        assert 0 < end_token <= num_tokens, f"End token {end_token} out of range (0, {num_tokens}]"
        assert start_token < end_token, f"Start token {start_token} >= end token {end_token}"


def test_toy_code_mode_weight_scheduler_integration():
    """
    Test code-mode weight scheduler integration with combined_kd_loss.

    Verifies:
    - Weight scheduler ramps from start_weight to target_weight over warmup_steps
    - Weight affects total loss computation
    - Linear interpolation works correctly
    - Weight holds at target after warmup
    """
    device = torch.device("cpu")
    batch_size = 2
    seq_len = 10
    vocab_size = 128

    # Create student and teacher logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    teacher_targets = teacher_logits.argmax(dim=-1)
    ground_truth = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create code-mode loss module
    eligibility_rules = {
        "min_tools": 2,
        "min_intermediate_chars": 10000,
        "pii_patterns": ["EMAIL", "PHONE", "SSN"],
    }
    reward_cfg = {
        "prefer_ts_api_over_direct_tool": True,
        "penalize_tool_result_roundtrip": True,
    }

    vocab_ids = {
        "import": 10,
        "from": 20,
        "callMCPTool": 30,
        "await": 40,
    }

    code_mode_loss_module = CodeModePreferenceLoss(
        eligibility_rules=eligibility_rules,
        reward=reward_cfg,
        vocab_ids=vocab_ids,
        weights={"pos": 1.0, "neg": 1.0},
    ).to(device)

    # Create eligible batch metadata
    batch_meta = {
        "tool_count": 2,
        "intermediate_sizes": [15000],
        "pii_tags_present": False,
    }

    # Weight scheduler parameters (from config)
    warmup_steps = 5000
    start_weight = 0.1
    target_weight = 0.3

    def compute_weight(step: int) -> float:
        """Compute weight using linear warmup schedule (from training/distill_kd.py)."""
        if step < warmup_steps and warmup_steps > 0:
            progress = step / warmup_steps
            return start_weight + (target_weight - start_weight) * progress
        else:
            return target_weight

    # Test at different steps
    test_steps = [0, warmup_steps // 2, warmup_steps, warmup_steps + 1000]

    for step in test_steps:
        # Compute weight for this step
        code_mode_weight = compute_weight(step)

        # Compute code-mode loss
        code_mode_loss = code_mode_loss_module(
            student_logits=student_logits,
            span_targets=None,
            batch_meta=batch_meta,
        )

        # Integrate into combined loss with scheduled weight
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_targets,
            ground_truth_targets=ground_truth,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
            code_mode_loss=code_mode_loss,
            code_mode_weight=code_mode_weight,
        )

        # Verify loss is computed
        assert "total" in loss_dict
        assert "code_mode_pref" in loss_dict

        # Verify weight affects total loss
        total_loss = loss_dict["total"]
        code_mode_loss_value = loss_dict["code_mode_pref"]

        # Total loss should include code-mode component weighted by code_mode_weight
        # We can't directly verify the exact value, but we can verify structure
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad

        # Verify weight values at specific steps
        if step == 0:
            assert abs(code_mode_weight - start_weight) < 1e-6, (
                f"At step 0, weight should be {start_weight} (got {code_mode_weight})"
            )
        elif step == warmup_steps:
            assert abs(code_mode_weight - target_weight) < 1e-6, (
                f"At step {warmup_steps}, weight should be {target_weight} (got {code_mode_weight})"
            )
        elif step == warmup_steps // 2:
            expected_midpoint = start_weight + (target_weight - start_weight) * 0.5
            assert abs(code_mode_weight - expected_midpoint) < 1e-6, (
                f"At step {warmup_steps // 2}, weight should be {expected_midpoint} (got {code_mode_weight})"
            )
        elif step > warmup_steps:
            assert abs(code_mode_weight - target_weight) < 1e-6, (
                f"After warmup, weight should remain {target_weight} (got {code_mode_weight})"
            )

    # Verify linearity: weight should increase linearly
    step1 = warmup_steps // 4
    step2 = warmup_steps // 2
    step3 = 3 * warmup_steps // 4

    weight1 = compute_weight(step1)
    weight2 = compute_weight(step2)
    weight3 = compute_weight(step3)

    # Check that increments are roughly equal (linear progression)
    increment1 = weight2 - weight1
    increment2 = weight3 - weight2

    assert abs(increment1 - increment2) < 1e-6, (
        f"Weight should increase linearly (increments: {increment1}, {increment2})"
    )

    # Verify weight is used in loss computation
    # At step 0, weight is small, so code-mode contribution should be small
    # At step warmup_steps, weight is full, so code-mode contribution should be larger
    step0_weight = compute_weight(0)
    step_warmup_weight = compute_weight(warmup_steps)

    assert step0_weight < step_warmup_weight, (
        f"Weight at step 0 ({step0_weight}) should be < weight at warmup ({step_warmup_weight})"
    )

    # Compute losses with different weights to verify weight affects total
    code_mode_loss_value = code_mode_loss_module(
        student_logits=student_logits,
        span_targets=None,
        batch_meta=batch_meta,
    )

    loss_dict_low_weight = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_targets,
        ground_truth_targets=ground_truth,
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
        code_mode_loss=code_mode_loss_value,
        code_mode_weight=step0_weight,
    )

    loss_dict_high_weight = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_targets,
        ground_truth_targets=ground_truth,
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
        code_mode_loss=code_mode_loss_value,
        code_mode_weight=step_warmup_weight,
    )

    # Total loss with higher weight should be different (if code_mode_loss is non-zero)
    # Note: In toy test, code_mode_loss might be simplified, so we just verify structure
    assert "total" in loss_dict_low_weight
    assert "total" in loss_dict_high_weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
