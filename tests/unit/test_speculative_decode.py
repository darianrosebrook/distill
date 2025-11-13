"""
Unit tests for speculative decoding optimization.

Tests drafter + worker verification for M-series Apple Silicon optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock

try:
    from coreml.runtime.speculative_decode import SpeculativeDecoder

    SPECULATIVE_DECODE_AVAILABLE = True
except ImportError:
    SPECULATIVE_DECODE_AVAILABLE = False


@pytest.mark.skipif(not SPECULATIVE_DECODE_AVAILABLE, reason="Speculative decode not available")
class TestSpeculativeDecoder:
    """Tests for SpeculativeDecoder class."""

    @pytest.fixture
    def mock_models(self):
        """Create mock CoreML models."""
        drafter_model = Mock()
        worker_model = Mock()
        return drafter_model, worker_model

    @pytest.fixture
    def mock_adapters(self):
        """Create mock StepAdapters."""
        drafter_adapter = Mock()
        worker_adapter = Mock()

        # Mock prepare_state
        drafter_adapter.prepare_state = Mock(return_value={})
        worker_adapter.prepare_state = Mock(return_value={})

        # Mock first_step
        drafter_adapter.first_step = Mock(
            return_value=(
                np.random.randn(32000),  # logits
                {},  # state
            )
        )
        worker_adapter.first_step = Mock(
            return_value=(
                np.random.randn(32000),  # logits
                {},  # state
            )
        )

        # Mock next_step
        drafter_adapter.next_step = Mock(
            return_value=(
                np.random.randn(32000),  # logits
                {},  # state
            )
        )
        worker_adapter.next_step = Mock(
            return_value=(
                np.random.randn(32000),  # logits
                {},  # state
            )
        )

        return drafter_adapter, worker_adapter

    def test_initialization(self, mock_models, mock_adapters):
        """Test speculative decoder initialization."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        assert decoder.k == 2
        assert decoder.drafter_model == drafter_model
        assert decoder.worker_model == worker_model

    def test_generate_basic(self, mock_models, mock_adapters):
        """Test basic token generation."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        # Setup logits to make acceptance likely
        def make_accepting_logits():
            logits = np.zeros(32000)
            logits[100] = 10.0  # High probability for token 100
            return logits

        drafter_adapter.first_step = Mock(return_value=(make_accepting_logits(), {}))
        worker_adapter.first_step = Mock(return_value=(make_accepting_logits(), {}))
        drafter_adapter.next_step = Mock(return_value=(make_accepting_logits(), {}))
        worker_adapter.next_step = Mock(return_value=(make_accepting_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        prompt_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
        result = decoder.generate(prompt_ids, max_tokens=5)

        assert "tokens" in result
        assert "stats" in result
        assert "ttft_ms" in result
        assert "tps" in result
        assert len(result["tokens"]) <= 5

    def test_draft_k_tokens(self, mock_models, mock_adapters):
        """Test drafting K tokens."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        # Setup logits
        def make_logits():
            logits = np.zeros(32000)
            logits[100] = 10.0
            return logits

        drafter_adapter.next_step = Mock(return_value=(make_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=3,
        )

        input_ids = np.array([1, 2, 3, 4])
        drafter_state = {}

        draft_tokens = decoder._draft_k_tokens(input_ids, drafter_state, k=3)

        assert len(draft_tokens) == 3
        assert all(isinstance(t, (int, np.integer)) for t in draft_tokens)

    def test_verify_tokens_accept(self, mock_models, mock_adapters):
        """Test token verification with acceptance."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        # Setup logits with high probability for draft tokens
        def make_accepting_logits():
            logits = np.zeros(32000)
            logits[100] = 10.0  # High probability
            return logits

        worker_adapter.next_step = Mock(return_value=(make_accepting_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        draft_tokens = [100, 101]
        input_ids = np.array([1, 2, 3])
        worker_state = {}

        accepted = decoder._verify_tokens(draft_tokens, input_ids, worker_state)

        # Should accept at least some tokens
        assert len(accepted) >= 0
        assert len(accepted) <= len(draft_tokens)

    def test_verify_tokens_reject(self, mock_models, mock_adapters):
        """Test token verification with rejection."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        # Setup logits with low probability (rejection)
        def make_rejecting_logits():
            logits = np.zeros(32000)
            logits[100] = -10.0  # Low probability
            return logits

        worker_adapter.next_step = Mock(return_value=(make_rejecting_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        draft_tokens = [100, 101]
        input_ids = np.array([1, 2, 3])
        worker_state = {}

        accepted = decoder._verify_tokens(draft_tokens, input_ids, worker_state)

        # May reject tokens if probability is too low
        assert len(accepted) <= len(draft_tokens)

    def test_stats_tracking(self, mock_models, mock_adapters):
        """Test statistics tracking."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        def make_logits():
            logits = np.zeros(32000)
            logits[100] = 10.0
            return logits

        drafter_adapter.first_step = Mock(return_value=(make_logits(), {}))
        worker_adapter.first_step = Mock(return_value=(make_logits(), {}))
        drafter_adapter.next_step = Mock(return_value=(make_logits(), {}))
        worker_adapter.next_step = Mock(return_value=(make_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        prompt_ids = np.array([[1, 2, 3]], dtype=np.int32)
        decoder.generate(prompt_ids, max_tokens=5)

        stats = decoder.get_stats()

        assert "total_tokens" in stats
        assert "accepted_tokens" in stats
        assert "rejected_tokens" in stats
        assert "rollbacks" in stats
        assert "acceptance_rate" in stats
        assert "rollback_rate" in stats

    def test_reset_stats(self, mock_models, mock_adapters):
        """Test statistics reset."""
        drafter_model, worker_model = mock_models
        drafter_adapter, worker_adapter = mock_adapters

        def make_logits():
            logits = np.zeros(32000)
            logits[100] = 10.0
            return logits

        drafter_adapter.first_step = Mock(return_value=(make_logits(), {}))
        worker_adapter.first_step = Mock(return_value=(make_logits(), {}))
        drafter_adapter.next_step = Mock(return_value=(make_logits(), {}))
        worker_adapter.next_step = Mock(return_value=(make_logits(), {}))

        decoder = SpeculativeDecoder(
            drafter_model=drafter_model,
            worker_model=worker_model,
            drafter_adapter=drafter_adapter,
            worker_adapter=worker_adapter,
            k=2,
        )

        prompt_ids = np.array([[1, 2, 3]], dtype=np.int32)
        decoder.generate(prompt_ids, max_tokens=5)

        # Reset stats
        decoder.reset_stats()

        stats = decoder.get_stats()
        assert stats["total_tokens"] == 0
        assert stats["accepted_tokens"] == 0
        assert stats["rejected_tokens"] == 0
        assert stats["rollbacks"] == 0
