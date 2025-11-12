"""
Unit tests for enumerated shape training functionality.

Tests shape sampling and batch truncation for enumerated shape training.
"""
import torch
import pytest
from unittest.mock import Mock, patch

from training.distill_kd import sample_enumerated_shape, truncate_batch_to_shape


class TestSampleEnumeratedShape:
    """Tests for enumerated shape sampling."""
    
    def test_default_production_mix_4_shapes(self):
        """Test default production mix for 4 shapes."""
        seq_lengths = [512, 1024, 2048, 4096]
        
        # Sample multiple times to verify distribution
        samples = [sample_enumerated_shape(seq_lengths, step=i) for i in range(100)]
        
        # Verify all samples are valid shapes
        assert all(s in seq_lengths for s in samples)
        
        # Verify distribution approximates production mix (0.5:0.3:0.15:0.05)
        # Smallest shape (4096) should be least common
        count_4096 = sum(1 for s in samples if s == 4096)
        assert count_4096 < 20  # Should be ~5% of 100 samples
    
    def test_custom_shape_probs(self):
        """Test custom probability distribution."""
        seq_lengths = [512, 1024, 2048]
        shape_probs = [0.7, 0.2, 0.1]
        
        samples = [sample_enumerated_shape(seq_lengths, shape_probs=shape_probs, step=i) 
                   for i in range(100)]
        
        # Verify all samples are valid
        assert all(s in seq_lengths for s in samples)
        
        # Verify 512 is most common (~70%)
        count_512 = sum(1 for s in samples if s == 512)
        assert count_512 > 60  # Should be ~70% of 100 samples
    
    def test_periodic_upweight_rare(self):
        """Test that rare shapes are periodically upweighted."""
        seq_lengths = [512, 1024, 2048, 4096]
        
        # Sample at step 0 (should upweight)
        sample_at_0 = sample_enumerated_shape(seq_lengths, step=0, periodic_upweight_rare=True)
        
        # Sample at step 1 (should not upweight)
        sample_at_1 = sample_enumerated_shape(seq_lengths, step=1, periodic_upweight_rare=True)
        
        # Both should be valid
        assert sample_at_0 in seq_lengths
        assert sample_at_1 in seq_lengths
    
    def test_no_periodic_upweight(self):
        """Test that periodic upweighting can be disabled."""
        seq_lengths = [512, 1024]
        
        # Should work without periodic upweighting
        sample = sample_enumerated_shape(seq_lengths, step=100, periodic_upweight_rare=False)
        assert sample in seq_lengths


class TestTruncateBatchToShape:
    """Tests for batch truncation to target shape."""
    
    def test_truncate_sequence_keys(self):
        """Test truncation of sequence dimension keys."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 100)),
            "attention_mask": torch.ones(2, 100),
            "labels": torch.randint(0, 1000, (2, 100)),
            "teacher_target_ids": torch.randint(0, 1000, (2, 100)),
        }
        
        truncated = truncate_batch_to_shape(batch, target_length=50)
        
        assert truncated["input_ids"].shape[1] == 50
        assert truncated["attention_mask"].shape[1] == 50
        assert truncated["labels"].shape[1] == 50
        assert truncated["teacher_target_ids"].shape[1] == 50
    
    def test_truncate_sequence_vocab_keys(self):
        """Test truncation of sequence+vocab dimension keys."""
        batch = {
            "teacher_logits": torch.randn(2, 100, 32000),
        }
        
        truncated = truncate_batch_to_shape(batch, target_length=50)
        
        assert truncated["teacher_logits"].shape[1] == 50
        assert truncated["teacher_logits"].shape[2] == 32000  # Vocab dimension preserved
    
    def test_no_truncation_when_shorter(self):
        """Test that shorter sequences are not padded."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 30)),
        }
        
        truncated = truncate_batch_to_shape(batch, target_length=50)
        
        # Should remain 30 (no padding, just truncation)
        assert truncated["input_ids"].shape[1] == 30
    
    def test_preserve_metadata_keys(self):
        """Test that metadata keys are preserved unchanged."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 100)),
            "metadata": {"some": "data", "count": 42},
        }
        
        truncated = truncate_batch_to_shape(batch, target_length=50)
        
        assert truncated["metadata"] == batch["metadata"]
        assert truncated["input_ids"].shape[1] == 50
    
    def test_all_sequence_keys_truncated(self):
        """Test that all sequence dimension keys are truncated."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 100)),
            "attention_mask": torch.ones(2, 100),
            "labels": torch.randint(0, 1000, (2, 100)),
            "teacher_target_ids": torch.randint(0, 1000, (2, 100)),
            "tool_name_ids": torch.randint(0, 1000, (2, 100)),
            "tool_name_mask": torch.ones(2, 100, dtype=torch.bool),
            "gold_json_text_ids": torch.randint(0, 1000, (2, 100)),
            "mask_valid_json_tokens": torch.ones(2, 100, dtype=torch.bool),
            "tool_result_fields": torch.randint(0, 1000, (2, 100)),
            "integration_mask": torch.ones(2, 100, dtype=torch.bool),
            "teacher_attention_mask": torch.ones(2, 100),
        }
        
        truncated = truncate_batch_to_shape(batch, target_length=50)
        
        for key in ["input_ids", "attention_mask", "labels", "teacher_target_ids",
                   "tool_name_ids", "tool_name_mask", "gold_json_text_ids",
                   "mask_valid_json_tokens", "tool_result_fields", "integration_mask",
                   "teacher_attention_mask"]:
            assert truncated[key].shape[1] == 50, f"{key} not truncated correctly"

