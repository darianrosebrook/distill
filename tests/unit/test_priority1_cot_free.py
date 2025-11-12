"""
Unit tests for Priority 1: CoT-free validation and process-step supervision.

Tests:
1. CoT-free validation raises error when reasoning_content detected
2. Process-step loss functions work correctly
3. Extractors module functions work
"""
import pytest
import torch
import json
from pathlib import Path
import tempfile

from training.dataset import KDDataset
from training.losses import (
    tool_name_loss,
    json_argument_loss,
    integration_copy_loss,
    combined_kd_loss,
)
from training.extractors import (
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)


class TestCoTFreeValidation:
    """Test CoT-free validation in dataset loading."""

    def test_dataset_rejects_reasoning_content(self):
        """Test that dataset raises ValueError when reasoning_content detected."""
        # Create temporary JSONL file with reasoning_content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                "teacher_reasoning_content": "This is reasoning content that should be rejected",
            }
            f.write(json.dumps(sample) + "\n")
            temp_path = f.name

        try:
            # Should raise ValueError
            with pytest.raises(ValueError, match="CoT-free training.*teacher_reasoning_content"):
                dataset = KDDataset(
                    jsonl_path=temp_path,
                    tokenizer_path="gpt2",  # Use small tokenizer for testing
                )
        finally:
            Path(temp_path).unlink()

    def test_dataset_accepts_valid_sample(self):
        """Test that dataset accepts samples without reasoning_content."""
        # Create temporary JSONL file without reasoning_content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "prompt": "Test prompt",
                "teacher_text": "Test response",
                # No teacher_reasoning_content
            }
            f.write(json.dumps(sample) + "\n")
            temp_path = f.name

        try:
            # Should not raise error
            dataset = KDDataset(
                jsonl_path=temp_path,
                tokenizer_path="gpt2",
            )
            assert len(dataset) == 1
        finally:
            Path(temp_path).unlink()

    def test_dataset_accepts_process_step_targets(self):
        """Test that dataset accepts process-step supervision targets."""
        # Create temporary JSONL file with process-step targets
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "prompt": "Test prompt",
                "teacher_text": '{"name": "test_tool", "arguments": {}}',
                "tool_name_ids": [1, 2, 3],
                "tool_name_mask": [True, True, True],
                "gold_json_text_ids": [4, 5, 6],
                "mask_valid_json_tokens": [True, True, True],
            }
            f.write(json.dumps(sample) + "\n")
            temp_path = f.name

        try:
            # Should not raise error
            dataset = KDDataset(
                jsonl_path=temp_path,
                tokenizer_path="gpt2",
            )
            assert len(dataset) == 1
            
            # Check that process-step targets are in sample
            sample = dataset[0]
            assert "tool_name_ids" in sample
            assert "tool_name_mask" in sample
        finally:
            Path(temp_path).unlink()


class TestProcessStepLosses:
    """Test process-step loss functions."""

    def test_tool_name_loss(self):
        """Test tool_name_loss function."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, 5))  # 5 tokens for tool name
        tool_name_mask = torch.ones(batch_size, 5, dtype=torch.bool)
        
        loss = tool_name_loss(student_logits, tool_name_ids, tool_name_mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_json_argument_loss(self):
        """Test json_argument_loss function."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, 8))
        mask_valid_json_tokens = torch.ones(batch_size, 8, dtype=torch.bool)
        
        loss = json_argument_loss(
            student_logits, gold_json_text_ids, mask_valid_json_tokens
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_integration_copy_loss(self):
        """Test integration_copy_loss function."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, 6))
        integration_mask = torch.ones(batch_size, 6, dtype=torch.bool)
        
        loss = integration_copy_loss(
            student_logits, tool_result_fields, integration_mask
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_combined_kd_loss_with_process_step(self):
        """Test combined_kd_loss with process-step supervision."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        ground_truth_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Process-step targets
        tool_name_ids = torch.randint(0, vocab_size, (batch_size, 5))
        tool_name_mask = torch.ones(batch_size, 5, dtype=torch.bool)
        gold_json_text_ids = torch.randint(0, vocab_size, (batch_size, 8))
        mask_valid_json_tokens = torch.ones(batch_size, 8, dtype=torch.bool)
        tool_result_fields = torch.randint(0, vocab_size, (batch_size, 6))
        integration_mask = torch.ones(batch_size, 6, dtype=torch.bool)
        
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_targets,
            ground_truth_targets=ground_truth_targets,
            tool_name_ids=tool_name_ids,
            tool_name_mask=tool_name_mask,
            gold_json_text_ids=gold_json_text_ids,
            mask_valid_json_tokens=mask_valid_json_tokens,
            tool_result_fields=tool_result_fields,
            integration_mask=integration_mask,
            w_tool=0.15,
            w_args=0.15,
            w_integr=0.10,
        )
        
        assert "total" in loss_dict
        assert "tool_name" in loss_dict
        assert "json_args" in loss_dict
        assert "integration" in loss_dict
        assert loss_dict["total"].item() >= 0
        assert loss_dict["total"].requires_grad

    def test_combined_kd_loss_without_process_step(self):
        """Test combined_kd_loss works without process-step targets."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        ground_truth_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # No process-step targets
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_targets,
            ground_truth_targets=ground_truth_targets,
            w_tool=0.0,  # Disable process-step losses
            w_args=0.0,
            w_integr=0.0,
        )
        
        assert "total" in loss_dict
        assert "tool_name" not in loss_dict  # Should not be computed
        assert "json_args" not in loss_dict
        assert "integration" not in loss_dict
        assert loss_dict["total"].item() >= 0


class TestExtractors:
    """Test extractor functions."""

    def test_extract_tool_name_span(self):
        """Test extract_tool_name_span function."""
        text = 'Here is a tool call: {"name": "web_search", "arguments": {"query": "test"}}'
        tool_names = ["web_search", "read_file"]
        
        span = extract_tool_name_span(text, tool_names)
        
        assert span is not None
        start, end = span
        assert start < end
        assert text[start:end] == '"web_search"'

    def test_extract_json_argument_spans(self):
        """Test extract_json_argument_spans function."""
        text = 'Tool call: {"name": "test", "args": {"key": "value"}} and more text'
        
        spans = extract_json_argument_spans(text)
        
        assert len(spans) > 0
        for start, end in spans:
            assert start < end
            json_str = text[start:end]
            # Should be valid JSON
            import json
            json.loads(json_str)

    def test_identify_integration_spans(self):
        """Test identify_integration_spans function."""
        text = "According to the search results, Python is a programming language."
        tool_results = [{"content": "Python is a programming language"}]
        
        spans = identify_integration_spans(text, tool_results)
        
        # Should find citation pattern
        assert len(spans) > 0
        for start, end in spans:
            assert start < end
            assert start < len(text)
            assert end <= len(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

