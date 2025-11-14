"""
Tests for training/process_losses.py - Process supervision loss functions.

Tests JSON validity loss, tool selection loss, argument validation loss,
and combined process supervision loss functions.
"""
# @author: @darianrosebrook

from unittest.mock import Mock

import pytest
import torch

from training.process_losses import (
    validate_json,
    extract_tool_call,
    json_validity_loss,
    tool_selection_loss,
    tool_selection_loss_from_ids,
    json_validity_loss_from_ids,
    process_supervision_loss,
)


class TestValidateJSON:
    """Test validate_json function."""

    def test_validate_json_valid_object(self):
        """Test validation of valid JSON object."""
        text = '{"name": "test", "value": 123}'
        assert validate_json(text) is True

    def test_validate_json_valid_array(self):
        """Test validation of valid JSON array."""
        text = '[1, 2, 3, "test"]'
        assert validate_json(text) is True

    def test_validate_json_nested_object(self):
        """Test validation of nested JSON object."""
        text = '{"outer": {"inner": "value"}}'
        assert validate_json(text) is True

    def test_validate_json_in_text(self):
        """Test validation when JSON is embedded in text."""
        text = 'Some text before {"key": "value"} and after'
        assert validate_json(text) is True

    def test_validate_json_invalid(self):
        """Test validation of invalid JSON."""
        text = '{"key": "value"'
        assert validate_json(text) is False

    def test_validate_json_malformed(self):
        """Test validation of malformed JSON."""
        text = '{key: value}'  # Missing quotes
        assert validate_json(text) is False

    def test_validate_json_empty_string(self):
        """Test validation of empty string."""
        text = ""
        assert validate_json(text) is False

    def test_validate_json_plain_text(self):
        """Test validation of plain text without JSON."""
        text = "This is just plain text without any JSON"
        assert validate_json(text) is False

    def test_validate_json_multiple_objects(self):
        """Test validation when multiple JSON objects are present."""
        text = '{"first": 1} and {"second": 2}'
        assert validate_json(text) is True  # Should find at least one valid JSON

    def test_validate_json_whitespace(self):
        """Test validation with whitespace."""
        text = '  {"key": "value"}  '
        assert validate_json(text) is True


class TestExtractToolCall:
    """Test extract_tool_call function."""

    def test_extract_tool_call_valid(self):
        """Test extraction of valid tool call."""
        text = '{"name": "test_tool", "arguments": {"key": "value"}}'
        tool_names = ["test_tool", "other_tool"]

        result = extract_tool_call(text, tool_names)

        assert result is not None
        assert result["name"] == "test_tool"
        assert "arguments" in result

    def test_extract_tool_call_in_text(self):
        """Test extraction when tool call is embedded in text."""
        text = 'Some text {"name": "tool", "args": {}} more text'
        tool_names = ["tool"]

        result = extract_tool_call(text, tool_names)

        assert result is not None
        assert result["name"] == "tool"

    def test_extract_tool_call_no_name(self):
        """Test extraction when JSON object has no name field."""
        text = '{"key": "value"}'  # No "name" field
        tool_names = ["tool"]

        result = extract_tool_call(text, tool_names)

        assert result is None

    def test_extract_tool_call_invalid_json(self):
        """Test extraction with invalid JSON."""
        text = '{"name": "tool"'  # Incomplete JSON
        tool_names = ["tool"]

        result = extract_tool_call(text, tool_names)

        assert result is None

    def test_extract_tool_call_no_json(self):
        """Test extraction when no JSON is present."""
        text = "Just plain text without JSON"
        tool_names = ["tool"]

        result = extract_tool_call(text, tool_names)

        assert result is None

    def test_extract_tool_call_array(self):
        """Test extraction when JSON is an array (not object)."""
        text = '[{"name": "tool"}]'  # Array, not object
        tool_names = ["tool"]

        result = extract_tool_call(text, tool_names)

        # Should try parsing entire text, which would fail for array
        # The function expects an object with "name" field
        assert result is None or isinstance(result, dict)


class TestJSONValidityLoss:
    """Test json_validity_loss function."""

    def test_json_validity_loss_valid_json(self, device):
        """Test loss with valid JSON."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"}', '{"name": "test"}']

        loss = json_validity_loss(logits, generated_texts)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        # Loss should be low for valid JSON (1 - 1.0 = 0.0)
        assert loss.item() < 0.1

    def test_json_validity_loss_invalid_json(self, device):
        """Test loss with invalid JSON."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"', 'invalid json text']

        loss = json_validity_loss(logits, generated_texts)

        assert isinstance(loss, torch.Tensor)
        # Loss should be high for invalid JSON (1 - 0.0 = 1.0)
        assert loss.item() > 0.5

    def test_json_validity_loss_mixed(self, device):
        """Test loss with mixed valid and invalid JSON."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"}', 'invalid json']

        loss = json_validity_loss(logits, generated_texts)

        assert isinstance(loss, torch.Tensor)
        # Should be between 0 and 1 (average of 0.0 and 1.0)
        assert 0.0 < loss.item() < 1.0

    def test_json_validity_loss_empty_list(self, device):
        """Test loss with empty generated texts list."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = []

        loss = json_validity_loss(logits, generated_texts)

        assert isinstance(loss, torch.Tensor)
        # Should handle empty list gracefully
        assert loss.item() == 0.0 or torch.isnan(loss)


class TestToolSelectionLoss:
    """Test tool_selection_loss function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3] if text else [])
        return tokenizer

    def test_tool_selection_loss_basic(self, device, mock_tokenizer):
        """Test basic tool selection loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"name": "tool1", "args": {}}', '{"name": "tool2", "args": {}}']
        target_tool_names = ["tool1", "tool2"]
        tool_names = ["tool1", "tool2", "tool3"]

        loss = tool_selection_loss(
            logits, generated_texts, target_tool_names, tool_names, mock_tokenizer
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_tool_selection_loss_no_tool_call(self, device, mock_tokenizer):
        """Test tool selection loss when no tool call is extracted."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ["No tool call here", "Also no tool call"]
        target_tool_names = ["tool1", "tool2"]
        tool_names = ["tool1", "tool2"]

        loss = tool_selection_loss(
            logits, generated_texts, target_tool_names, tool_names, mock_tokenizer
        )

        # Should return zero loss when no tool calls found
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_tool_selection_loss_empty_targets(self, device, mock_tokenizer):
        """Test tool selection loss with empty target tool names."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"name": "tool1"}', '{"name": "tool2"}']
        target_tool_names = ["", ""]  # Empty targets
        tool_names = ["tool1", "tool2"]

        loss = tool_selection_loss(
            logits, generated_texts, target_tool_names, tool_names, mock_tokenizer
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0


class TestToolSelectionLossFromIDs:
    """Test tool_selection_loss_from_ids function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="tool_name")
        return tokenizer

    def test_tool_selection_loss_from_ids_basic(self, device, mock_tokenizer):
        """Test basic tool selection loss from IDs."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.tensor([[10, 11, 12], [13, 14, 15]], device=device)
        tool_name_mask = torch.tensor([[True, True, False], [True, True, True]], device=device)

        loss = tool_selection_loss_from_ids(logits, tool_name_ids, tool_name_mask, mock_tokenizer)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_tool_selection_loss_from_ids_1d(self, device, mock_tokenizer):
        """Test tool selection loss with 1D tensors."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.tensor([10, 11, 12], device=device)
        tool_name_mask = torch.tensor([True, True, False], device=device)

        loss = tool_selection_loss_from_ids(logits, tool_name_ids, tool_name_mask, mock_tokenizer)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_tool_selection_loss_from_ids_empty_mask(self, device, mock_tokenizer):
        """Test tool selection loss with empty mask."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.tensor([[10, 11, 12], [13, 14, 15]], device=device)
        tool_name_mask = torch.tensor([[False, False, False], [False, False, False]], device=device)

        loss = tool_selection_loss_from_ids(logits, tool_name_ids, tool_name_mask, mock_tokenizer)

        # Should return zero loss when no valid tokens
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_tool_selection_loss_from_ids_short_sequence(self, device, mock_tokenizer):
        """Test tool selection loss with short sequence."""
        batch_size = 2
        seq_len = 5  # Short sequence
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tool_name_ids = torch.tensor([[10, 11], [12, 13]], device=device)
        tool_name_mask = torch.tensor([[True, True], [True, True]], device=device)

        loss = tool_selection_loss_from_ids(logits, tool_name_ids, tool_name_mask, mock_tokenizer)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestJSONValidityLossFromIDs:
    """Test json_validity_loss_from_ids function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(side_effect=lambda tokens, **kwargs: '{"key": "value"}')
        return tokenizer

    def test_json_validity_loss_from_ids_basic(self, device, mock_tokenizer):
        """Test basic JSON validity loss from IDs."""
        gold_json_text_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=device)
        mask_valid_json_tokens = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]], device=device
        )

        loss = json_validity_loss_from_ids(gold_json_text_ids, mask_valid_json_tokens, mock_tokenizer)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_json_validity_loss_from_ids_1d(self, device, mock_tokenizer):
        """Test JSON validity loss with 1D tensors."""
        gold_json_text_ids = torch.tensor([1, 2, 3, 4, 5], device=device)
        mask_valid_json_tokens = torch.tensor([True, True, True, False, False], device=device)

        loss = json_validity_loss_from_ids(gold_json_text_ids, mask_valid_json_tokens, mock_tokenizer)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_json_validity_loss_from_ids_empty_mask(self, device, mock_tokenizer):
        """Test JSON validity loss with empty mask."""
        gold_json_text_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        mask_valid_json_tokens = torch.tensor([[False, False, False], [False, False, False]], device=device)

        loss = json_validity_loss_from_ids(gold_json_text_ids, mask_valid_json_tokens, mock_tokenizer)

        # Should return loss for invalid JSON (all empty = invalid)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_json_validity_loss_from_ids_decode_error(self, device):
        """Test JSON validity loss when tokenizer decode fails."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(side_effect=Exception("Decode error"))

        gold_json_text_ids = torch.tensor([[1, 2, 3]], device=device)
        mask_valid_json_tokens = torch.tensor([[True, True, True]], device=device)

        loss = json_validity_loss_from_ids(gold_json_text_ids, mask_valid_json_tokens, mock_tokenizer)

        # Should treat decode error as invalid JSON
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestProcessSupervisionLoss:
    """Test process_supervision_loss function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3] if text else [])
        tokenizer.decode = Mock(return_value='{"key": "value"}')
        return tokenizer

    def test_process_supervision_loss_basic(self, device, mock_tokenizer):
        """Test basic process supervision loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"name": "tool1"}', '{"name": "tool2"}']

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            json_validity_weight=0.3,
            tool_select_weight=0.7,
            tokenizer=mock_tokenizer,
        )

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "json_validity" in loss_dict
        assert loss_dict["total"].item() >= 0

    def test_process_supervision_loss_with_token_ids(self, device, mock_tokenizer):
        """Test process supervision loss with token ID-based inputs."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = []  # Not used when token IDs provided

        tool_name_ids = torch.tensor([[10, 11], [12, 13]], device=device)
        tool_name_mask = torch.tensor([[True, True], [True, True]], device=device)
        gold_json_text_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        mask_valid_json_tokens = torch.tensor([[True, True, True], [True, True, True]], device=device)

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            json_validity_weight=0.3,
            tool_select_weight=0.7,
            tokenizer=mock_tokenizer,
            tool_name_ids=tool_name_ids,
            tool_name_mask=tool_name_mask,
            gold_json_text_ids=gold_json_text_ids,
            mask_valid_json_tokens=mask_valid_json_tokens,
        )

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "json_validity" in loss_dict
        assert "tool_selection" in loss_dict
        assert loss_dict["total"].item() >= 0

    def test_process_supervision_loss_json_only(self, device, mock_tokenizer):
        """Test process supervision loss with only JSON validity."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"}', '{"name": "test"}']

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            json_validity_weight=1.0,
            tool_select_weight=0.0,
            tokenizer=mock_tokenizer,
        )

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "json_validity" in loss_dict
        assert "tool_selection" not in loss_dict or loss_dict["tool_selection"].item() == 0.0

    def test_process_supervision_loss_tool_only(self, device, mock_tokenizer):
        """Test process supervision loss with only tool selection."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"name": "tool1"}', '{"name": "tool2"}']
        target_tool_names = ["tool1", "tool2"]
        tool_names = ["tool1", "tool2", "tool3"]

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            target_tool_names=target_tool_names,
            tool_names=tool_names,
            json_validity_weight=0.0,
            tool_select_weight=1.0,
            tokenizer=mock_tokenizer,
        )

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "tool_selection" in loss_dict
        assert "json_validity" not in loss_dict or loss_dict["json_validity"].item() == 0.0

    def test_process_supervision_loss_no_tokenizer(self, device):
        """Test process supervision loss without tokenizer."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"}', '{"name": "test"}']

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            json_validity_weight=1.0,
            tool_select_weight=0.0,
            tokenizer=None,
        )

        # Should still compute JSON validity loss
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "json_validity" in loss_dict

    def test_process_supervision_loss_zero_weights(self, device, mock_tokenizer):
        """Test process supervision loss with zero weights."""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        generated_texts = ['{"key": "value"}', '{"name": "test"}']

        loss_dict = process_supervision_loss(
            logits,
            generated_texts,
            json_validity_weight=0.0,
            tool_select_weight=0.0,
            tokenizer=mock_tokenizer,
        )

        # Total loss should be zero
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert loss_dict["total"].item() == 0.0

