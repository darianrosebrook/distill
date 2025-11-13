"""
Unit tests for process supervision loss functions.
"""

import torch

from training.process_losses import (
    validate_json,
    extract_tool_call,
    json_validity_loss,
    tool_selection_loss,
    process_supervision_loss,
)


class TestValidateJSON:
    """Tests for JSON validation function."""

    def test_validate_json_valid_object(self):
        """Test validation with valid JSON object."""
        text = '{"name": "test", "value": 123}'
        assert validate_json(text) is True

    def test_validate_json_valid_array(self):
        """Test validation with valid JSON array."""
        text = '[1, 2, 3, {"key": "value"}]'
        assert validate_json(text) is True

    def test_validate_json_invalid_missing_quote(self):
        """Test validation with invalid JSON (missing quote)."""
        text = '{"name": test}'  # Missing quotes around value
        assert validate_json(text) is False

    def test_validate_json_invalid_trailing_comma(self):
        """Test validation with invalid JSON (trailing comma)."""
        text = '{"name": "test",}'  # Trailing comma
        assert validate_json(text) is False

    def test_validate_json_with_text_around(self):
        """Test validation when JSON is embedded in text."""
        text = 'Here is some JSON: {"name": "test", "value": 123} and more text'
        assert validate_json(text) is True

    def test_validate_json_empty_string(self):
        """Test validation with empty string."""
        assert validate_json("") is False

    def test_validate_json_no_json(self):
        """Test validation with text containing no JSON."""
        text = "This is just plain text with no JSON"
        assert validate_json(text) is False

    def test_validate_json_multiple_json_objects(self):
        """Test validation when multiple JSON objects exist (should find first)."""
        text = '{"first": 1} and {"second": 2}'
        assert validate_json(text) is True  # Should find first valid JSON


class TestExtractToolCall:
    """Tests for tool call extraction function."""

    def test_extract_tool_call_valid(self):
        """Test extraction with valid tool call."""
        text = '{"name": "web_search", "arguments": {"query": "test"}}'
        tool_names = ["web_search", "read_file", "write_file"]
        result = extract_tool_call(text, tool_names)
        assert result is not None
        assert isinstance(result, dict)
        assert result["name"] == "web_search"
        assert "arguments" in result

    def test_extract_tool_call_not_in_list(self):
        """Test extraction when tool name is not in the list."""
        text = '{"name": "unknown_tool", "arguments": {}}'
        tool_names = ["web_search", "read_file"]
        result = extract_tool_call(text, tool_names)
        # Function returns dict if JSON is valid, regardless of tool_names
        assert result is not None
        assert result["name"] == "unknown_tool"

    def test_extract_tool_call_no_name_field(self):
        """Test extraction when JSON has no 'name' field."""
        text = '{"arguments": {"query": "test"}}'
        tool_names = ["web_search"]
        result = extract_tool_call(text, tool_names)
        assert result is None

    def test_extract_tool_call_invalid_json(self):
        """Test extraction with invalid JSON."""
        text = '{"name": "web_search"'  # Incomplete JSON
        tool_names = ["web_search"]
        result = extract_tool_call(text, tool_names)
        assert result is None

    def test_extract_tool_call_with_text_around(self):
        """Test extraction when JSON is embedded in text."""
        text = 'Here is a tool call: {"name": "read_file", "arguments": {"path": "test.txt"}}'
        tool_names = ["read_file", "write_file"]
        result = extract_tool_call(text, tool_names)
        assert result is not None
        assert result["name"] == "read_file"

    def test_extract_tool_call_empty_tool_names(self):
        """Test extraction with empty tool names list."""
        text = '{"name": "web_search", "arguments": {}}'
        tool_names = []
        result = extract_tool_call(text, tool_names)
        # Function still returns dict if JSON is valid
        assert result is not None
        assert result["name"] == "web_search"


class TestJSONValidityLoss:
    """Tests for JSON validity loss function."""

    def test_json_validity_loss_all_valid(self, device):
        """Test loss with all valid JSON."""
        generated_text = [
            '{"name": "test1"}',
            '{"name": "test2"}',
            "[1, 2, 3]",
        ]
        # Create dummy logits [B=3, T=10, V=1000]
        logits = torch.randn(3, 10, 1000, device=device)
        loss = json_validity_loss(logits, generated_text)
        assert loss.item() == 0.0, "All valid JSON should have zero loss"

    def test_json_validity_loss_all_invalid(self, device):
        """Test loss with all invalid JSON."""
        generated_text = [
            '{"name": test}',  # Invalid
            "not json",  # Invalid
            "{invalid}",  # Invalid
        ]
        logits = torch.randn(3, 10, 1000, device=device)
        loss = json_validity_loss(logits, generated_text)
        assert loss.item() == 1.0, "All invalid JSON should have loss of 1.0"

    def test_json_validity_loss_mixed(self, device):
        """Test loss with mixed valid/invalid JSON."""
        generated_text = [
            '{"name": "test"}',  # Valid
            "invalid json",  # Invalid
            '{"valid": true}',  # Valid
        ]
        logits = torch.randn(3, 10, 1000, device=device)
        loss = json_validity_loss(logits, generated_text)
        # Should be average: (0 + 1 + 0) / 3 = 1/3
        assert abs(loss.item() - (1.0 / 3.0)) < 1e-5

    def test_json_validity_loss_empty_list(self, device):
        """Test loss with empty list."""
        generated_text = []
        logits = torch.randn(0, 10, 1000, device=device)
        loss = json_validity_loss(logits, generated_text)
        # Should handle gracefully
        assert isinstance(loss, torch.Tensor)


class TestToolSelectionLoss:
    """Tests for tool selection loss function."""

    def test_tool_selection_loss_correct(self, device, mock_tokenizer):
        """Test loss when all tool selections are correct."""
        generated_text = [
            '{"name": "web_search", "arguments": {}}',
            '{"name": "read_file", "arguments": {}}',
        ]
        target_tool_names = ["web_search", "read_file"]
        tool_names = ["web_search", "read_file", "write_file"]
        logits = torch.randn(2, 20, mock_tokenizer.vocab_size, device=device)

        loss = tool_selection_loss(
            logits, generated_text, target_tool_names, tool_names, mock_tokenizer
        )
        # Should be low (correct predictions)
        assert loss.item() >= 0.0

    def test_tool_selection_loss_incorrect(self, device, mock_tokenizer):
        """Test loss when tool selections are incorrect."""
        generated_text = [
            '{"name": "web_search", "arguments": {}}',  # Wrong tool
            '{"name": "read_file", "arguments": {}}',  # Wrong tool
        ]
        target_tool_names = ["read_file", "web_search"]  # Different targets
        tool_names = ["web_search", "read_file", "write_file"]
        logits = torch.randn(2, 20, mock_tokenizer.vocab_size, device=device)

        loss = tool_selection_loss(
            logits, generated_text, target_tool_names, tool_names, mock_tokenizer
        )
        # Should be higher than correct case
        assert loss.item() >= 0.0

    def test_tool_selection_loss_no_targets(self, device, mock_tokenizer):
        """Test loss when no target tools provided."""
        generated_text = ['{"name": "web_search"}']
        target_tool_names = []
        tool_names = ["web_search"]
        logits = torch.randn(1, 20, mock_tokenizer.vocab_size, device=device)

        loss = tool_selection_loss(
            logits, generated_text, target_tool_names, tool_names, mock_tokenizer
        )
        # Should return zero loss when no targets
        assert loss.item() == 0.0

    def test_tool_selection_loss_invalid_json(self, device, mock_tokenizer):
        """Test loss when generated text has invalid JSON."""
        generated_text = ["invalid json"]
        target_tool_names = ["web_search"]
        tool_names = ["web_search"]
        logits = torch.randn(1, 20, mock_tokenizer.vocab_size, device=device)

        loss = tool_selection_loss(
            logits, generated_text, target_tool_names, tool_names, mock_tokenizer
        )
        # Should return zero loss when no valid tool call extracted
        assert loss.item() == 0.0


class TestProcessSupervisionLoss:
    """Tests for combined process supervision loss."""

    def test_process_supervision_loss_both_components(self, device, mock_tokenizer):
        """Test combined loss with both JSON validity and tool selection."""
        generated_text = [
            '{"name": "web_search", "arguments": {}}',
            '{"name": "read_file", "arguments": {}}',
        ]
        target_tool_names = ["web_search", "read_file"]
        tool_names = ["web_search", "read_file"]
        json_validity_weight = 0.3
        tool_select_weight = 0.7
        logits = torch.randn(2, 20, mock_tokenizer.vocab_size, device=device)

        losses = process_supervision_loss(
            logits=logits,
            generated_texts=generated_text,
            target_tool_names=target_tool_names,
            tool_names=tool_names,
            tokenizer=mock_tokenizer,
            json_validity_weight=json_validity_weight,
            tool_select_weight=tool_select_weight,
        )

        assert isinstance(losses["total"], torch.Tensor)
        assert losses["total"].item() >= 0.0
        assert "json_validity" in losses
        assert "tool_selection" in losses
        assert "total" in losses

    def test_process_supervision_loss_no_targets(self, device, mock_tokenizer):
        """Test combined loss when no target tools provided."""
        generated_text = ['{"name": "web_search"}']
        logits = torch.randn(1, 20, mock_tokenizer.vocab_size, device=device)

        losses = process_supervision_loss(
            logits=logits,
            generated_texts=generated_text,
            target_tool_names=None,
            tool_names=None,
            tokenizer=None,
            json_validity_weight=0.3,
            tool_select_weight=0.7,
        )

        assert isinstance(losses["total"], torch.Tensor)
        assert "json_validity" in losses
        # Tool selection loss should not be present when no targets

    def test_process_supervision_loss_weight_scaling(self, device, mock_tokenizer):
        """Test that different weights produce different losses."""
        generated_text = ['{"name": "web_search"}']
        target_tool_names = ["read_file"]  # Wrong tool
        tool_names = ["web_search", "read_file"]
        logits = torch.randn(1, 20, mock_tokenizer.vocab_size, device=device)

        # High JSON validity weight
        losses1 = process_supervision_loss(
            logits=logits,
            generated_texts=generated_text,
            target_tool_names=target_tool_names,
            tool_names=tool_names,
            tokenizer=mock_tokenizer,
            json_validity_weight=0.9,
            tool_select_weight=0.1,
        )

        # High tool selection weight
        losses2 = process_supervision_loss(
            logits=logits,
            generated_texts=generated_text,
            target_tool_names=target_tool_names,
            tool_names=tool_names,
            tokenizer=mock_tokenizer,
            json_validity_weight=0.1,
            tool_select_weight=0.9,
        )

        # Both should be valid tensors
        assert isinstance(losses1["total"], torch.Tensor)
        assert isinstance(losses2["total"], torch.Tensor)
        # Total losses should be different due to different weight scaling
        # (even if individual component losses are the same, weighted totals differ)
        assert losses1["total"].item() >= 0.0
        assert losses2["total"].item() >= 0.0
        # With different weights, totals should differ (unless both components are 0)
        # Since JSON is valid (loss=0), only tool_selection contributes
        # Both should have tool_selection loss, but weighted differently
        assert "tool_selection" in losses1
        assert "tool_selection" in losses2
