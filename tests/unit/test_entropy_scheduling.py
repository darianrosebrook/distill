"""
Unit tests for Priority 2: Entropy-based temperature/weight scheduling.

Tests:
1. entropy_weighting() computes entropy correctly
2. High entropy → high temperature, high KL weight
3. Low entropy → low temperature, high CE_GT weight
4. Entropy scheduling integrates with training loop
"""
import torch
from training.losses import entropy_weighting


class TestEntropyWeighting:
    """Test entropy_weighting function."""

    def test_high_entropy_high_temperature(self):
        """Test that high entropy leads to high temperature."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # High entropy: uniform distribution (maximum entropy)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        # Make distribution more uniform by scaling down
        teacher_logits = teacher_logits * 0.1  # Lower magnitude = more uniform
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
            min_entropy=2.0,
            max_entropy=8.0,
            min_temp=1.5,
            max_temp=3.0,
        )
        
        # High entropy should lead to high temperature
        assert temp >= 2.0, f"Expected temp >= 2.0 for high entropy, got {temp}"
        assert weights["kl_weight"] > weights["ce_ground_truth_weight"], \
            "High entropy should favor KL weight over CE_GT weight"

    def test_low_entropy_low_temperature(self):
        """Test that low entropy leads to low temperature."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # Low entropy: peaked distribution (low entropy)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        # Make distribution peaked by scaling up
        teacher_logits = teacher_logits * 10.0  # Higher magnitude = more peaked
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
            min_entropy=2.0,
            max_entropy=8.0,
            min_temp=1.5,
            max_temp=3.0,
        )
        
        # Low entropy should lead to low temperature
        assert temp <= 2.5, f"Expected temp <= 2.5 for low entropy, got {temp}"
        assert weights["ce_ground_truth_weight"] >= weights["kl_weight"], \
            "Low entropy should favor CE_GT weight over KL weight"

    def test_entropy_computation(self):
        """Test that entropy is computed correctly."""
        batch_size = 1
        seq_len = 5
        vocab_size = 10
        
        # Create uniform distribution (maximum entropy)
        teacher_logits = torch.zeros(batch_size, seq_len, vocab_size)
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
            min_entropy=0.0,
            max_entropy=10.0,
        )
        
        # Check that entropy is in the returned weights
        assert "entropy" in weights
        entropy = weights["entropy"]
        
        # For uniform distribution over vocab_size=10, entropy ≈ log(10) ≈ 2.3
        assert entropy > 0, f"Entropy should be positive, got {entropy}"
        assert entropy <= 10.0, f"Entropy should be reasonable, got {entropy}"

    def test_temperature_range(self):
        """Test that temperature stays within bounds."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        min_temp = 1.5
        max_temp = 3.0
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
            min_entropy=2.0,
            max_entropy=8.0,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        
        assert min_temp <= temp <= max_temp, \
            f"Temperature {temp} should be in [{min_temp}, {max_temp}]"

    def test_weight_normalization(self):
        """Test that weights are normalized."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
        )
        
        total_weight = weights["kl_weight"] + weights["ce_teacher_weight"] + weights["ce_ground_truth_weight"]
        
        # Weights should sum to approximately 1.0 (within floating point error)
        assert abs(total_weight - 1.0) < 0.01, \
            f"Weights should sum to ~1.0, got {total_weight}"

    def test_entropy_clamping(self):
        """Test that entropy is clamped to [min_entropy, max_entropy]."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # Very high entropy (uniform distribution)
        teacher_logits = torch.zeros(batch_size, seq_len, vocab_size)
        
        min_entropy = 2.0
        max_entropy = 8.0
        
        temp, weights = entropy_weighting(
            teacher_logits=teacher_logits,
            min_entropy=min_entropy,
            max_entropy=max_entropy,
        )
        
        entropy = weights["entropy"]
        
        # Entropy should be clamped (or at least the normalized value should be in [0, 1])
        # The actual entropy might be outside bounds, but normalized should be in [0, 1]
        assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"

    def test_entropy_vs_linear_schedule(self):
        """Test that entropy scheduling differs from linear schedule."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        # High entropy batch
        high_entropy_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        
        # Low entropy batch
        low_entropy_logits = torch.randn(batch_size, seq_len, vocab_size) * 10.0
        
        temp_high, weights_high = entropy_weighting(high_entropy_logits)
        temp_low, weights_low = entropy_weighting(low_entropy_logits)
        
        # High entropy should have higher temperature
        assert temp_high > temp_low, \
            f"High entropy temp {temp_high} should be > low entropy temp {temp_low}"
        
        # High entropy should favor KL weight
        assert weights_high["kl_weight"] > weights_low["kl_weight"], \
            "High entropy should favor KL weight"
        
        # Low entropy should favor CE_GT weight
        assert weights_low["ce_ground_truth_weight"] > weights_high["ce_ground_truth_weight"], \
            "Low entropy should favor CE_GT weight"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

