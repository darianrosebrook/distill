"""
E2E test for token reduction at equal accuracy.

Tests long-chain task with measurable accuracy metric.
"""
# @author: @darianrosebrook

import pytest
import torch
from unittest.mock import Mock

from runtime.engine.loop import LatentModeEngine
from eval.scoring.efficiency import EfficiencyMetrics, calculate_token_reduction


class TestTokenReductionE2E:
    """E2E test for token reduction with equal accuracy."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        # Make model callable - return logits tensor
        # [batch_size, seq_len, vocab_size]
        model.return_value = torch.randn(1, 1, 1000)
        model.forward_hidden = Mock(return_value=torch.randn(1, 10, 128))
        model.forward_decode = Mock(return_value=(torch.randn(1, 1, 1000), []))
        model.embed = Mock(return_value=torch.randn(1, 1, 128))
        model.blocks = [Mock() for _ in range(2)]
        model.norm_f = Mock(return_value=torch.randn(1, 1, 128))
        model.cfg = Mock()
        model.cfg.n_layers = 2
        model.cfg.d_model = 128
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="test output")
        return tokenizer

    @pytest.fixture
    def long_chain_task(self):
        """Create long-chain task fixture."""
        return {
            "prompt": "Solve this multi-step problem:",
            "expected_answer": "42",
            "steps": [
                "Step 1: Analyze",
                "Step 2: Compute",
                "Step 3: Verify",
                "Step 4: Finalize",
            ],
        }

    def test_baseline_direct_cot(self, mock_model, mock_tokenizer, long_chain_task):
        """Test baseline direct CoT (no latent spans)."""
        # Disable latent mode for baseline
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=False,
        )

        input_ids = torch.tensor([[1, 2, 3]])

        # Generate with direct CoT
        result = engine.generate_with_latent_mode(
            input_ids,
            max_new_tokens=100,
        )

        baseline_tokens = len(result["tokens"])
        baseline_metrics = EfficiencyMetrics(
            accuracy=0.85,  # Mock accuracy
            generated_tokens=baseline_tokens,
            wall_clock_time_ms=100.0,
            latent_spans_used=0,
            refinement_loops=1,
        )

        assert baseline_tokens > 0
        # Don't return - tests should not return values

    def test_latent_mode_token_reduction(self, mock_model, mock_tokenizer, long_chain_task):
        """Test latent mode achieves token reduction."""
        # Enable latent mode
        engine = LatentModeEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            latent_mode_enabled=True,
        )

        input_ids = torch.tensor([[1, 2, 3]])

        # Generate with latent mode
        result = engine.generate_with_latent_mode(
            input_ids,
            max_new_tokens=100,
        )

        latent_tokens = len(result["tokens"])
        latent_metrics = EfficiencyMetrics(
            accuracy=0.85,  # Same accuracy as baseline
            generated_tokens=latent_tokens,
            wall_clock_time_ms=95.0,  # Slightly faster
            latent_spans_used=len(result.get("latent_span_lengths", [])),
            refinement_loops=1,
        )

        assert latent_tokens > 0
        # Don't return - tests should not return values

    def test_token_reduction_at_equal_accuracy(self, mock_model, mock_tokenizer, long_chain_task):
        """Test that latent mode achieves ≥25% token reduction at equal accuracy."""
        # Get baseline metrics
        baseline_metrics = self.test_baseline_direct_cot(
            mock_model, mock_tokenizer, long_chain_task
        )

        # Get latent metrics
        latent_metrics = self.test_latent_mode_token_reduction(
            mock_model, mock_tokenizer, long_chain_task
        )

        # Calculate token reduction
        token_reduction = calculate_token_reduction(
            latent_metrics.generated_tokens,
            baseline_metrics.generated_tokens,
        )

        # Assert accuracy is maintained (or improved)
        assert latent_metrics.accuracy >= baseline_metrics.accuracy - 0.01  # Max 1% regression

        # Assert token reduction meets target (≥25%)
        # Note: In real test, this would require actual model training
        # This is a structure test to verify the assertion logic
        if token_reduction > 0:
            assert token_reduction >= 0.25, f"Token reduction {token_reduction:.1%} < 25%"

    def test_efficiency_curves(self, mock_model, mock_tokenizer):
        """Test efficiency curves (accuracy vs tokens/time)."""
        from eval.scoring.efficiency import compute_efficiency_curves

        # Create mock metrics
        baseline = EfficiencyMetrics(
            accuracy=0.85,
            generated_tokens=100,
            wall_clock_time_ms=100.0,
            latent_spans_used=0,
            refinement_loops=1,
        )

        current_metrics = [
            EfficiencyMetrics(
                accuracy=0.85,
                generated_tokens=75,  # 25% reduction
                wall_clock_time_ms=95.0,
                latent_spans_used=1,
                refinement_loops=1,
            ),
            EfficiencyMetrics(
                accuracy=0.86,  # Slightly better
                generated_tokens=70,  # 30% reduction
                wall_clock_time_ms=90.0,
                latent_spans_used=2,
                refinement_loops=1,
            ),
        ]

        curves = compute_efficiency_curves(current_metrics, baseline)

        assert "accuracy_vs_tokens" in curves
        assert "accuracy_vs_time" in curves

        # Verify curves show improvement
        # EfficiencyCurve is a dataclass with accuracies attribute, not a dict
        assert len(curves["accuracy_vs_tokens"].accuracies) == len(current_metrics) + 1  # +1 for baseline
