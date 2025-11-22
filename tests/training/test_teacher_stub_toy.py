"""
Tests for training/teacher_stub_toy.py - Deterministic teacher stub for toy distillation.

Tests deterministic teacher logits generation for toy training.
"""
# @author: @darianrosebrook

import torch
from training.teacher_stub_toy import (
    teacher_logits,
    eight_ball_teacher_logits,
)


class TestTeacherLogits:
    """Test teacher_logits function."""

    def test_teacher_logits_shape(self):
        """Test that teacher logits have correct shape."""
        batch_size = 2
        seq_len = 10
        vocab_size = 512
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = teacher_logits(token_ids, vocab_size=vocab_size)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_teacher_logits_deterministic(self):
        """Test that teacher logits are deterministic."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        logits1 = teacher_logits(token_ids, vocab_size=vocab_size)
        logits2 = teacher_logits(token_ids, vocab_size=vocab_size)

        torch.testing.assert_close(logits1, logits2)

    def test_teacher_logits_device_preservation(self):
        """Test that logits are on same device as input."""
        if torch.cuda.is_available():
            token_ids = torch.randint(0, 100, (1, 5), device="cuda")
            vocab_size = 512

            logits = teacher_logits(token_ids, vocab_size=vocab_size)

            assert logits.device == token_ids.device

    def test_teacher_logits_dtype(self):
        """Test that logits have correct dtype."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        logits = teacher_logits(token_ids, vocab_size=vocab_size)

        assert logits.dtype == torch.float32

    def test_teacher_logits_hot_tokens(self):
        """Test that hot tokens have higher logits."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        logits = teacher_logits(token_ids, vocab_size=vocab_size)

        # Hot tokens should have higher logits
        hot_tokens = [5, 17, 33, 7, 8]
        for hot_token in hot_tokens:
            if hot_token < vocab_size:
                # Check that hot token has relatively high logits
                assert logits[0, 0, hot_token] > 0

    def test_teacher_logits_different_batch_sizes(self):
        """Test teacher logits with different batch sizes."""
        vocab_size = 512

        for batch_size in [1, 2, 4]:
            token_ids = torch.randint(0, 100, (batch_size, 10))
            logits = teacher_logits(token_ids, vocab_size=vocab_size)
            assert logits.shape[0] == batch_size

    def test_teacher_logits_different_seq_lengths(self):
        """Test teacher logits with different sequence lengths."""
        vocab_size = 512

        for seq_len in [5, 10, 20]:
            token_ids = torch.randint(0, 100, (2, seq_len))
            logits = teacher_logits(token_ids, vocab_size=vocab_size)
            assert logits.shape[1] == seq_len

    def test_teacher_logits_different_vocab_sizes(self):
        """Test teacher logits with different vocab sizes."""
        token_ids = torch.randint(0, 100, (1, 5))

        for vocab_size in [128, 256, 512, 1024]:
            logits = teacher_logits(token_ids, vocab_size=vocab_size)
            assert logits.shape[2] == vocab_size


class TestEightBallTeacherLogits:
    """Test eight_ball_teacher_logits function."""

    def test_eight_ball_teacher_logits_shape(self):
        """Test that 8-ball teacher logits have correct shape."""
        batch_size = 2
        seq_len = 10
        vocab_size = 512
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_eight_ball_teacher_logits_deterministic(self):
        """Test that 8-ball teacher logits are deterministic."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        logits1 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)
        logits2 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        torch.testing.assert_close(logits1, logits2)

    def test_eight_ball_teacher_logits_ternary_classifier(self):
        """Test 8-ball logits with ternary classifier tokens."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 500  # Large enough for ternary tokens (400-402)

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Ternary classifier tokens should have boosted logits
        ternary_tokens = [400, 401, 402]
        for token_id in ternary_tokens:
            if token_id < vocab_size:
                assert logits[0, -1, token_id] > 0  # Final position should have boost

    def test_eight_ball_teacher_logits_binary_classifier(self):
        """Test 8-ball logits with binary classifier tokens."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 350  # Large enough for binary tokens (300-301) but not ternary

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Binary classifier tokens should have boosted logits
        binary_tokens = [300, 301]
        for token_id in binary_tokens:
            if token_id < vocab_size:
                assert logits[0, -1, token_id] > 0

    def test_eight_ball_teacher_logits_eight_ball_tokens(self):
        """Test 8-ball logits with 8-ball token IDs."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 250  # Large enough for 8-ball tokens (200-219) but not classifiers

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # 8-ball tokens should have boosted logits
        eight_ball_tokens = list(range(200, 220))
        for token_id in eight_ball_tokens:
            if token_id < vocab_size:
                assert logits[0, -1, token_id] > 0

    def test_eight_ball_teacher_logits_small_vocab(self):
        """Test 8-ball logits with small vocab size."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 50  # Too small for special tokens

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Should still return valid logits
        assert logits.shape == (1, 5, vocab_size)

    def test_eight_ball_teacher_logits_device_preservation(self):
        """Test that 8-ball logits are on same device as input."""
        if torch.cuda.is_available():
            token_ids = torch.randint(0, 100, (1, 5), device="cuda")
            vocab_size = 512

            logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

            assert logits.device == token_ids.device

    def test_eight_ball_teacher_logits_dtype(self):
        """Test that 8-ball logits have correct dtype."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        assert logits.dtype == torch.float32

    def test_eight_ball_teacher_logits_final_position_boost(self):
        """Test that final position has boosted logits for answer tokens."""
        token_ids = torch.randint(0, 100, (2, 10))
        vocab_size = 500

        logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Final position should have different logits than other positions
        # (due to answer position boost)
        ternary_tokens = [400, 401, 402]
        for token_id in ternary_tokens:
            if token_id < vocab_size:
                # Final position should have higher logits
                final_logit = logits[0, -1, token_id]
                middle_logit = logits[0, 5, token_id]
                assert final_logit >= middle_logit

    def test_eight_ball_teacher_logits_question_hash_determinism(self):
        """Test that question hash produces deterministic answers."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 500

        logits1 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)
        logits2 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Should be deterministic
        torch.testing.assert_close(logits1, logits2)

    def test_eight_ball_teacher_logits_different_batch_sizes(self):
        """Test 8-ball logits with different batch sizes."""
        vocab_size = 500

        for batch_size in [1, 2, 4]:
            token_ids = torch.randint(0, 100, (batch_size, 10))
            logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)
            assert logits.shape[0] == batch_size


class TestTeacherStubIntegration:
    """Test integration of teacher stub functions."""

    def test_teacher_logits_vs_eight_ball_different_seeds(self):
        """Test that regular and 8-ball logits use different seeds."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        regular_logits = teacher_logits(token_ids, vocab_size=vocab_size)
        eight_ball_logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        # Should be different due to different seeds
        assert not torch.allclose(regular_logits, eight_ball_logits)

    def test_both_functions_same_input_shape(self):
        """Test that both functions produce same shape for same input."""
        token_ids = torch.randint(0, 100, (2, 10))
        vocab_size = 512

        regular_logits = teacher_logits(token_ids, vocab_size=vocab_size)
        eight_ball_logits = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)

        assert regular_logits.shape == eight_ball_logits.shape

    def test_deterministic_across_calls(self):
        """Test that both functions are deterministic across multiple calls."""
        token_ids = torch.randint(0, 100, (1, 5))
        vocab_size = 512

        # Regular logits
        logits1 = teacher_logits(token_ids, vocab_size=vocab_size)
        logits2 = teacher_logits(token_ids, vocab_size=vocab_size)
        torch.testing.assert_close(logits1, logits2)

        # 8-ball logits
        logits3 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)
        logits4 = eight_ball_teacher_logits(token_ids, vocab_size=vocab_size)
        torch.testing.assert_close(logits3, logits4)

















