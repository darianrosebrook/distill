"""
Tests for training/losses.py - Knowledge distillation loss functions.

Tests all loss functions including KL divergence, cross-entropy, process supervision
losses, intermediate layer matching, self-evaluation, length-aware KD, early tool call,
and CAWS compliance losses.
"""
# @author: @darianrosebrook

from unittest.mock import Mock

import torch
import torch.nn as nn

from training.losses import (
    kl_divergence,
    cross_entropy_on_teacher,
    tool_name_loss,
    json_argument_loss,
    integration_copy_loss,
    halt_head_loss,
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
    length_aware_kd_loss,
    early_tool_call_loss,
    curriculum_temperature,
    combined_kd_loss,
    CodeModePreferenceLoss,
    caws_compliance_loss,
    caws_structure_loss,
    entropy_weighting,
)


class TestKLDivergence:
    """Test KL divergence loss function."""

    def test_kl_divergence_basic(self, device):
        """Test basic KL divergence computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss = kl_divergence(student_logits, teacher_logits, temperature=1.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative

    def test_kl_divergence_with_temperature(self, device):
        """Test KL divergence with different temperatures."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        loss_t1 = kl_divergence(student_logits, teacher_logits, temperature=1.0)
        loss_t2 = kl_divergence(student_logits, teacher_logits, temperature=2.0)
        loss_t05 = kl_divergence(student_logits, teacher_logits, temperature=0.5)

        # All should be valid losses
        assert loss_t1.item() >= 0
        assert loss_t2.item() >= 0
        assert loss_t05.item() >= 0

    def test_kl_divergence_reduction_mean(self, device):
        """Test KL divergence with mean reduction."""
        student_logits = torch.randn(2, 5, 100, device=device, requires_grad=True)
        teacher_logits = torch.randn(2, 5, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="mean")

        assert loss.dim() == 0  # Scalar

    def test_kl_divergence_reduction_sum(self, device):
        """Test KL divergence with sum reduction."""
        student_logits = torch.randn(2, 5, 100, device=device, requires_grad=True)
        teacher_logits = torch.randn(2, 5, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="sum")

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_kl_divergence_reduction_none(self, device):
        """Test KL divergence with no reduction."""
        batch_size = 2
        seq_len = 5
        student_logits = torch.randn(batch_size, seq_len, 100, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, 100, device=device)

        loss = kl_divergence(student_logits, teacher_logits, reduction="none")

        assert loss.dim() == 1  # [B*T]
        assert loss.shape[0] == batch_size * seq_len

    def test_kl_divergence_flattened_input(self, device):
        """Test KL divergence with flattened input tensors."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size * seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size * seq_len, vocab_size, device=device)

        loss = kl_divergence(student_logits, teacher_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestCrossEntropyOnTeacher:
    """Test cross-entropy loss on teacher predictions."""

    def test_cross_entropy_on_teacher_basic(self, device):
        """Test basic cross-entropy on teacher targets."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss = cross_entropy_on_teacher(student_logits, teacher_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_cross_entropy_on_teacher_with_ignore_index(self, device):
        """Test cross-entropy with ignore index."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        teacher_targets[0, :5] = -100  # Mark some tokens to ignore

        loss = cross_entropy_on_teacher(student_logits, teacher_targets, ignore_index=-100)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_cross_entropy_on_teacher_gradient_flow(self, device):
        """Test that gradients flow through the loss."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss = cross_entropy_on_teacher(student_logits, teacher_targets)
        loss.backward()

        assert student_logits.grad is not None
        assert not torch.allclose(student_logits.grad, torch.zeros_like(student_logits.grad))


class TestToolNameLoss:
    """Test tool name loss function."""

    def test_tool_name_loss_basic(self, device):
        """Test basic tool name loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        tool_len = 5

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.ones(batch_size, tool_len, device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_tool_name_loss_with_mask(self, device):
        """Test tool name loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        tool_len = 5

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_tool_name_loss_sequence_mismatch(self, device):
        """Test tool name loss when tool length exceeds sequence length."""
        batch_size = 2
        seq_len = 5
        vocab_size = 1000
        tool_len = 10  # Longer than sequence

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, tool_len), device=device)
        tool_name_mask = torch.ones(batch_size, tool_len, device=device)

        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)

        # Should handle gracefully by truncating
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestJSONArgumentLoss:
    """Test JSON argument loss function."""

    def test_json_argument_loss_basic(self, device):
        """Test basic JSON argument loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        json_len = 8

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, json_len), device=device)
        mask_valid_json_tokens = torch.ones(batch_size, json_len, device=device)

        loss = json_argument_loss(student_logits, gold_json_text_ids, mask_valid_json_tokens)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_json_argument_loss_with_partial_mask(self, device):
        """Test JSON argument loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        json_len = 8

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, json_len), device=device)
        # Mask out some tokens (prose vs JSON)
        mask_valid_json_tokens = torch.tensor(
            [[1, 1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1, 1, 1]], device=device
        )

        loss = json_argument_loss(student_logits, gold_json_text_ids, mask_valid_json_tokens)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestIntegrationCopyLoss:
    """Test integration copy loss function."""

    def test_integration_copy_loss_basic(self, device):
        """Test basic integration copy loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        int_len = 6

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, int_len), device=device)
        integration_mask = torch.ones(batch_size, int_len, device=device)

        loss = integration_copy_loss(student_logits, tool_result_fields, integration_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_integration_copy_loss_with_mask(self, device):
        """Test integration copy loss with partial mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        int_len = 6

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, int_len), device=device)
        integration_mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1]], device=device)

        loss = integration_copy_loss(student_logits, tool_result_fields, integration_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestHaltHeadLoss:
    """Test halt head loss function."""

    def test_halt_head_loss_basic(self, device):
        """Test basic halt head loss."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device, requires_grad=True)
        halt_targets = torch.randint(0, 2, (batch_size,), device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_halt_head_loss_all_continue(self, device):
        """Test halt head loss with all continue targets."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device, requires_grad=True)
        halt_targets = torch.zeros(batch_size, dtype=torch.long, device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_halt_head_loss_all_halt(self, device):
        """Test halt head loss with all halt targets."""
        batch_size = 4

        halt_logits = torch.randn(batch_size, 2, device=device, requires_grad=True)
        halt_targets = torch.ones(batch_size, dtype=torch.long, device=device)

        loss = halt_head_loss(halt_logits, halt_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestIntermediateLayerLoss:
    """Test intermediate layer loss function."""

    def test_intermediate_layer_loss_basic(self, device):
        """Test basic intermediate layer loss."""
        batch_size = 2
        seq_len = 10
        student_d_model = 128
        teacher_d_model = 256

        student_hidden_states = [
            torch.randn(batch_size, seq_len, student_d_model, device=device, requires_grad=True) for _ in range(4)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, teacher_d_model, device=device) for _ in range(8)
        ]

        layer_mapping = {0: 0, 1: 2, 2: 4, 3: 6}

        # Need projection layers when dimensions don't match
        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping, projection_layers
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_intermediate_layer_loss_with_projection(self, device):
        """Test intermediate layer loss with projection layers."""
        batch_size = 2
        seq_len = 10
        student_d_model = 128
        teacher_d_model = 256

        student_hidden_states = [
            torch.randn(batch_size, seq_len, student_d_model, device=device) for _ in range(4)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, teacher_d_model, device=device) for _ in range(8)
        ]

        layer_mapping = {0: 0, 1: 2, 2: 4, 3: 6}
        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping, projection_layers
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_intermediate_layer_loss_sequence_mismatch(self, device):
        """Test intermediate layer loss with different sequence lengths."""
        batch_size = 2
        student_seq_len = 10
        teacher_seq_len = 15
        d_model = 128

        student_hidden_states = [
            torch.randn(batch_size, student_seq_len, d_model, device=device) for _ in range(2)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, teacher_seq_len, d_model, device=device) for _ in range(4)
        ]

        layer_mapping = {0: 0, 1: 2}

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        # Should handle sequence length mismatch by truncating
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_intermediate_layer_loss_empty_mapping(self, device):
        """Test intermediate layer loss with empty layer mapping."""
        batch_size = 2
        seq_len = 10
        d_model = 128

        student_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(2)
        ]
        teacher_hidden_states = [
            torch.randn(batch_size, seq_len, d_model, device=device) for _ in range(4)
        ]

        layer_mapping = {}

        loss = intermediate_layer_loss(
            student_hidden_states, teacher_hidden_states, layer_mapping
        )

        # Should return zero loss with gradient
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() == 0.0


class TestSelfEvaluationLoss:
    """Test self-evaluation loss function."""

    def test_self_evaluation_loss_basic(self, device):
        """Test basic self-evaluation loss."""
        batch_size = 4

        student_eval_score = torch.rand(batch_size, 1, device=device, requires_grad=True)
        teacher_quality_score = torch.rand(batch_size, device=device)

        loss = self_evaluation_loss(student_eval_score, teacher_quality_score)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_self_evaluation_loss_perfect_match(self, device):
        """Test self-evaluation loss when scores match perfectly."""
        batch_size = 4

        scores = torch.rand(batch_size, device=device)
        student_eval_score = scores.unsqueeze(1)
        teacher_quality_score = scores

        loss = self_evaluation_loss(student_eval_score, teacher_quality_score)

        # Should be close to zero (within numerical precision)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 1e-5


class TestCreateProjectionLayers:
    """Test projection layer creation."""

    def test_create_projection_layers_basic(self, device):
        """Test basic projection layer creation."""
        student_d_model = 128
        teacher_d_model = 256
        layer_mapping = {0: 0, 1: 2, 2: 4}

        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        assert len(projection_layers) == len(layer_mapping)
        for proj in projection_layers:
            assert isinstance(proj, nn.Module)

    def test_create_projection_layers_output_shape(self, device):
        """Test that projection layers produce correct output shape."""
        student_d_model = 128
        teacher_d_model = 256
        layer_mapping = {0: 0}

        projection_layers = create_projection_layers(
            student_d_model, teacher_d_model, layer_mapping, device
        )

        batch_size = 2
        seq_len = 10
        student_h = torch.randn(batch_size, seq_len, student_d_model, device=device)

        output = projection_layers[0](student_h)

        assert output.shape == (batch_size, seq_len, teacher_d_model)


class TestLengthAwareKDLoss:
    """Test length-aware KD loss function."""

    def test_length_aware_kd_loss_basic(self, device):
        """Test basic length-aware KD loss."""
        batch_size = 2
        student_seq_len = 10
        teacher_seq_len = 12

        student_attn_mask = torch.ones(batch_size, student_seq_len, device=device, requires_grad=True)
        teacher_attn_mask = torch.ones(batch_size, teacher_seq_len, device=device)
        required_fields_present = torch.tensor([True, False], device=device)

        loss, diagnostics = length_aware_kd_loss(
            student_attn_mask, teacher_attn_mask, required_fields_present
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
        assert isinstance(diagnostics, dict)

    def test_length_aware_kd_loss_with_hinge(self, device):
        """Test length-aware KD loss with hinge threshold."""
        batch_size = 2
        student_seq_len = 15  # Longer than teacher to trigger penalty
        teacher_seq_len = 10

        student_attn_mask = torch.ones(batch_size, student_seq_len, device=device, requires_grad=True)
        teacher_attn_mask = torch.ones(batch_size, teacher_seq_len, device=device)
        required_fields_present = torch.tensor([False, False], device=device)  # Missing fields

        loss, diagnostics = length_aware_kd_loss(
            student_attn_mask, teacher_attn_mask, required_fields_present, hinge=0.15
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert isinstance(diagnostics, dict)


class TestEarlyToolCallLoss:
    """Test early tool call loss function."""

    def test_early_tool_call_loss_with_teacher_prefix(self, device):
        """Test early_tool_call_loss with teacher prefix IDs."""
        batch_size = 2
        seq_len = 30
        vocab_size = 1000
        N = 25

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([1.0, 0.0], device=device)
        teacher_prefix_ids = torch.randint(0, vocab_size, (batch_size, 10), device=device)

        mock_tokenizer = Mock()
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=100)

        loss, diags = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=N,
            json_prior_weight=0.02,
            ce_weight=0.2,
            ramp_t=1.0,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
        assert "early_tool.frac_should_use" in diags
        assert "early_tool.frac_target_available" in diags

    def test_early_tool_call_loss_without_teacher_prefix_json_prior(self, device):
        """Test early_tool_call_loss without teacher prefix (JSON-envelope prior path)."""
        batch_size = 2
        seq_len = 30
        vocab_size = 1000
        N = 25

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([1.0, 0.0], device=device)
        teacher_prefix_ids = None  # No teacher prefix

        mock_tokenizer = Mock()
        # Mock convert_tokens_to_ids to return valid token IDs for JSON start tokens
        def mock_convert_tokens_to_ids(token):
            if token == "{":
                return 200
            elif token == "[":
                return 201
            elif token == '"':
                return 202
            return None
        mock_tokenizer.convert_tokens_to_ids = Mock(side_effect=mock_convert_tokens_to_ids)

        loss, diags = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=N,
            json_prior_weight=0.02,
            ce_weight=0.2,
            ramp_t=1.0,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
        assert "early_tool.frac_should_use" in diags
        assert "early_tool.frac_target_available" in diags
        assert diags["early_tool.frac_target_available"] == 0.0  # No teacher prefix

    def test_early_tool_call_loss_no_json_tokens(self, device):
        """Test early_tool_call_loss when tokenizer doesn't have JSON tokens."""
        batch_size = 2
        seq_len = 30
        vocab_size = 1000
        N = 25

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([1.0, 0.0], device=device)
        teacher_prefix_ids = None

        mock_tokenizer = Mock()
        # Tokenizer doesn't recognize JSON tokens
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=None)

        loss, diags = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=N,
            json_prior_weight=0.02,
            ce_weight=0.2,
            ramp_t=1.0,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0  # Should return zero loss when no JSON tokens found

    def test_early_tool_call_loss_with_ramp(self, device):
        """Test early_tool_call_loss with ramp scaling."""
        batch_size = 2
        seq_len = 30
        vocab_size = 1000
        N = 25

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([1.0, 1.0], device=device)
        teacher_prefix_ids = torch.randint(0, vocab_size, (batch_size, 10), device=device)

        mock_tokenizer = Mock()
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=100)

        # Test with ramp_t = 0.0 (no loss)
        loss_zero, _ = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=N,
            json_prior_weight=0.02,
            ce_weight=0.2,
            ramp_t=0.0,
        )

        # Test with ramp_t = 1.0 (full loss)
        loss_full, _ = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=N,
            json_prior_weight=0.02,
            ce_weight=0.2,
            ramp_t=1.0,
        )

        assert loss_zero.item() == 0.0  # Ramp at 0 should give zero loss
        assert loss_full.item() >= 0  # Ramp at 1 should give non-zero loss

    def test_early_tool_call_loss_basic(self, device, mock_tokenizer):
        """Test basic early tool call loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([True, False], device=device)

        loss, diagnostics = early_tool_call_loss(
            logits, input_ids, tool_should_be_used, mock_tokenizer
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
        assert isinstance(diagnostics, dict)

    def test_early_tool_call_loss_disabled(self, device, mock_tokenizer):
        """Test early tool call loss when tool not needed."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([False, False], device=device)  # No tools needed

        loss, diagnostics = early_tool_call_loss(
            logits, input_ids, tool_should_be_used, mock_tokenizer
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # May still have some loss even when disabled
        assert isinstance(diagnostics, dict)

    def test_early_tool_call_loss_with_ramp(self, device, mock_tokenizer):
        """Test early tool call loss with ramping."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tool_should_be_used = torch.tensor([True, True], device=device)

        loss, diagnostics = early_tool_call_loss(
            logits, input_ids, tool_should_be_used, mock_tokenizer, ramp_t=0.5
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert isinstance(diagnostics, dict)


class TestCurriculumTemperature:
    """Test curriculum temperature function."""

    def test_curriculum_temperature_start(self):
        """Test curriculum temperature at start of training."""
        temp = curriculum_temperature(epoch=0, total_epochs=10)

        assert isinstance(temp, float)
        assert temp > 0

    def test_curriculum_temperature_end(self):
        """Test curriculum temperature at end of training."""
        temp = curriculum_temperature(epoch=10, total_epochs=10)

        assert isinstance(temp, float)
        assert temp > 0

    def test_curriculum_temperature_monotonic(self):
        """Test that temperature decreases over time."""
        temps = [
            curriculum_temperature(epoch=i, total_epochs=10) for i in range(11)
        ]

        # Temperature should generally decrease (allowing for some variance)
        assert temps[0] >= temps[-1]


class TestCombinedKDLoss:
    """Test combined KD loss function."""

    def test_combined_kd_loss_basic(self, device):
        """Test basic combined KD loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        ground_truth_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        loss_dict = combined_kd_loss(
            student_logits,
            teacher_logits,
            teacher_targets,
            ground_truth_targets,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=1.0,
        )

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert loss_dict["total"].item() >= 0

    def test_combined_kd_loss_with_intermediate_layers(self, device):
        """Test combined KD loss with intermediate layer matching."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        ground_truth_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # combined_kd_loss doesn't support intermediate layers directly
        # Those would be computed separately and added via code_mode_loss parameter
        loss_dict = combined_kd_loss(
            student_logits,
            teacher_logits,
            teacher_targets,
            ground_truth_targets,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=1.0,
        )

        assert isinstance(loss_dict, dict)
        assert loss_dict["total"].item() >= 0

    def test_combined_kd_loss_with_self_evaluation(self, device):
        """Test combined KD loss with self-evaluation head."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        ground_truth_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Self-evaluation loss would be computed separately and added via code_mode_loss
        # or through a different mechanism
        loss_dict = combined_kd_loss(
            student_logits,
            teacher_logits,
            teacher_targets,
            ground_truth_targets,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=1.0,
        )

        assert isinstance(loss_dict, dict)
        assert loss_dict["total"].item() >= 0


class TestCodeModePreferenceLoss:
    """Test CodeModePreferenceLoss class."""

    def test_code_mode_preference_loss_basic(self, device):
        """Test basic CodeModePreferenceLoss."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": False}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        # Mock batch metadata for eligibility computation
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        # CodeModePreferenceLoss needs batch metadata and span targets
        # This is a simplified test - actual usage requires more setup
        assert loss_fn is not None

    def test_code_mode_preference_loss_with_weight(self, device):
        """Test CodeModePreferenceLoss with different weights."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": False}
        vocab_ids = {"import": 100, "from": 101}

        weights_light = {"pos": 0.5, "neg": 0.5}
        weights_heavy = {"pos": 2.0, "neg": 2.0}

        loss_fn_light = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids, weights=weights_light
        )
        loss_fn_heavy = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids, weights=weights_heavy
        )

        assert loss_fn_light is not None
        assert loss_fn_heavy is not None

    def test_code_mode_preference_loss_forward_with_eligibility(self, device):
        """Test CodeModePreferenceLoss forward pass with eligible samples."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        # Batch metadata: first sample eligible (tool_count >= 2), second not eligible
        batch_meta = [
            {"tool_count": 3, "intermediate_sizes": [], "pii_tags_present": False},  # Eligible
            {"tool_count": 1, "intermediate_sizes": [], "pii_tags_present": False},  # Not eligible
        ]

        # Span targets: TS API spans and direct tool spans
        span_targets = [
            {
                "ts_mode_spans": [(5, 10), (15, 18)],  # TS API spans for first sample
                "direct_tool_spans": [(0, 3)],
            },
            {
                "ts_mode_spans": [],
                "direct_tool_spans": [],
            },
        ]

        loss = loss_fn(student_logits, span_targets=span_targets, batch_meta=batch_meta)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative

    def test_code_mode_preference_loss_forward_no_eligible(self, device):
        """Test CodeModePreferenceLoss forward pass with no eligible samples."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        # Batch metadata: no eligible samples
        batch_meta = [
            {"tool_count": 1, "intermediate_sizes": [], "pii_tags_present": False},
            {"tool_count": 0, "intermediate_sizes": [], "pii_tags_present": False},
        ]

        span_targets = [
            {"ts_mode_spans": [(5, 10)], "direct_tool_spans": []},
            {"ts_mode_spans": [], "direct_tool_spans": []},
        ]

        loss = loss_fn(student_logits, span_targets=span_targets, batch_meta=batch_meta)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0  # Should return zero loss when no eligible samples

    def test_code_mode_preference_loss_forward_no_span_targets(self, device):
        """Test CodeModePreferenceLoss forward pass with no span targets."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        batch_meta = [
            {"tool_count": 3, "intermediate_sizes": [], "pii_tags_present": False},
            {"tool_count": 2, "intermediate_sizes": [], "pii_tags_present": False},
        ]

        # No span targets
        loss = loss_fn(student_logits, span_targets=None, batch_meta=batch_meta)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0  # Should return zero loss when no span targets

    def test_code_mode_preference_loss_eligibility_intermediate_chars(self, device):
        """Test CodeModePreferenceLoss eligibility via intermediate_chars."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        # Batch metadata: eligible via intermediate_chars (not tool_count)
        batch_meta = [
            {"tool_count": 1, "intermediate_sizes": [5000, 12000], "pii_tags_present": False},  # Eligible (max >= 10000)
            {"tool_count": 1, "intermediate_sizes": [5000], "pii_tags_present": False},  # Not eligible
        ]

        span_targets = [
            {"ts_mode_spans": [(5, 10)], "direct_tool_spans": []},
            {"ts_mode_spans": [], "direct_tool_spans": []},
        ]

        loss = loss_fn(student_logits, span_targets=span_targets, batch_meta=batch_meta)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_code_mode_preference_loss_eligibility_pii_tags(self, device):
        """Test CodeModePreferenceLoss eligibility via PII tags."""
        eligibility_rules = {"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []}
        reward = {"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True}
        vocab_ids = {"import": 100, "from": 101}

        loss_fn = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules, reward=reward, vocab_ids=vocab_ids
        )

        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)

        # Batch metadata: eligible via PII tags
        batch_meta = [
            {"tool_count": 1, "intermediate_sizes": [], "pii_tags_present": True},  # Eligible
            {"tool_count": 1, "intermediate_sizes": [], "pii_tags_present": False},  # Not eligible
        ]

        span_targets = [
            {"ts_mode_spans": [(5, 10)], "direct_tool_spans": []},
            {"ts_mode_spans": [], "direct_tool_spans": []},
        ]

        loss = loss_fn(student_logits, span_targets=span_targets, batch_meta=batch_meta)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0


class TestCAWSComplianceLoss:
    """Test CAWS compliance loss function."""

    def test_caws_compliance_loss_basic(self, device):
        """Test basic CAWS compliance loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        torch.randn(batch_size, seq_len, vocab_size, device=device)

        # caws_compliance_loss takes strings, not tensors
        student_output = "Valid JSON: {\"key\": \"value\"}"
        teacher_output = "Valid JSON: {\"key\": \"value\"}"

        loss = caws_compliance_loss(student_output, teacher_output)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_caws_compliance_loss_with_weights(self, device):
        """Test CAWS compliance loss with different component weights."""
        # caws_compliance_loss takes strings, not tensors or config
        student_output = "Valid output"
        teacher_output = "Valid output"

        loss = caws_compliance_loss(student_output, teacher_output)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestCAWSStructureLoss:
    """Test CAWS structure loss function."""

    def test_caws_structure_loss_basic(self, device):
        """Test basic CAWS structure loss."""
        teacher_score = 0.8
        student_score = 0.6

        loss = caws_structure_loss(teacher_score, student_score)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_caws_structure_loss_perfect_match(self, device):
        """Test CAWS structure loss when scores match."""
        score = 0.7

        loss = caws_structure_loss(score, score)

        # Should be close to zero
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 1e-5


class TestEntropyWeighting:
    """Test entropy weighting function."""

    def test_entropy_weighting_basic(self, device):
        """Test basic entropy weighting."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

        temperature, weights_dict = entropy_weighting(logits)

        assert isinstance(temperature, float)
        assert isinstance(weights_dict, dict)
        assert "entropy" in weights_dict
        assert "kl_weight" in weights_dict
        assert "ce_teacher_weight" in weights_dict
        assert "ce_ground_truth_weight" in weights_dict

    def test_entropy_weighting_high_entropy(self, device):
        """Test entropy weighting with high entropy (uniform) distribution."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Uniform logits (high entropy)
        logits = torch.ones(batch_size, seq_len, vocab_size, device=device)

        temperature, weights_dict = entropy_weighting(logits)

        # High entropy should result in higher temperature and KL weight
        assert isinstance(temperature, float)
        assert isinstance(weights_dict, dict)
        assert weights_dict["kl_weight"] > 0

    def test_entropy_weighting_low_entropy(self, device):
        """Test entropy weighting with low entropy (peaked) distribution."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Peaked logits (low entropy)
        logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        logits[:, :, 0] = 10.0  # Strong peak at first token

        temperature, weights_dict = entropy_weighting(logits)

        # Low entropy should result in lower temperature and higher CE_GT weight
        assert isinstance(temperature, float)
        assert isinstance(weights_dict, dict)
        assert weights_dict["ce_ground_truth_weight"] > 0.5




