"""
Unit tests for speed metrics measurement.

Tests TTFT/TPS/TTFA measurement and aggregation.
"""

import time
import torch
import pytest
from unittest.mock import Mock

from training.speed_metrics import (
    measure_proxy,
    aggregate_speed_metrics,
    is_valid_tool_json,
)


class TestMeasureProxy:
    """Tests for proxy speed metrics measurement."""

    @pytest.fixture
    def mock_model(self, device):
        """Create a mock model."""
        model = Mock()
        model.eval = Mock()

        def forward_mock(input_ids, attention_mask=None):
            # Simulate forward pass time
            time.sleep(0.001)  # 1ms delay
            return torch.randn(input_ids.shape[0], input_ids.shape[1], 32000)

        # Set up forward_decode to return proper tuple for decode path testing
        def forward_decode_mock(token_ids, kv_cache, pos):
            time.sleep(0.001)  # 1ms delay
            logits = torch.randn(1, 1, 32000)
            return logits, kv_cache  # Return tuple as expected

        model.forward_decode = forward_decode_mock
        model.side_effect = forward_mock
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value='{"name": "test_tool"}')
        return tokenizer

    def test_ttft_measurement(self, mock_model, mock_tokenizer, device):
        """Test TTFT measurement."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=5,
        )

        assert "ttft_ms" in metrics
        assert metrics["ttft_ms"] > 0.0
        assert metrics["ttft_ms"] < 1000.0  # Should be reasonable

    def test_tps_measurement(self, mock_model, mock_tokenizer, device):
        """Test TPS measurement."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=10,
        )

        assert "tps" in metrics
        assert metrics["tps"] > 0.0
        assert "tokens_generated" in metrics

    def test_ttfa_measurement(self, mock_model, mock_tokenizer, device):
        """Test TTFA measurement when valid tool JSON found."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        # Mock tokenizer to return valid tool JSON
        mock_tokenizer.decode.return_value = '{"name": "web_search", "arguments": {"q": "test"}}'

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=10,
        )

        assert "ttfa_tokens" in metrics
        assert "ttfa_ms" in metrics

    def test_ttfa_no_tool(self, mock_model, mock_tokenizer, device):
        """Test TTFA when no tool JSON found."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        # Mock tokenizer to return non-tool text
        mock_tokenizer.decode.return_value = "This is just regular text."

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=10,
        )

        assert metrics["ttfa_tokens"] == float("inf")
        assert metrics["ttfa_ms"] == float("inf")

    def test_no_tokenizer(self, mock_model, device):
        """Test measurement without tokenizer (TTFA should be inf)."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=None,
            device=device,
            max_new_tokens=10,
        )

        assert metrics["ttfa_tokens"] == float("inf")
        assert metrics["ttfa_ms"] == float("inf")

    def test_model_without_forward_decode(self, device):
        """Test measurement with model that doesn't have forward_decode (fallback path)."""
        # Create a model without forward_decode method
        class ModelWithoutDecode:
            def eval(self):
                pass

            def __call__(self, input_ids, attention_mask=None):
                time.sleep(0.001)
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 32000)

        model = ModelWithoutDecode()

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "test_tool"}')

        metrics = measure_proxy(
            model=model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=5,
        )

        assert "ttft_ms" in metrics
        assert "tps" in metrics
        assert metrics["tps"] > 0.0

    def test_tokenizer_decode_exception(self, mock_model, device):
        """Test TTFA measurement when tokenizer.decode raises exception."""
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
        }

        # Mock tokenizer that raises exception
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(side_effect=Exception("Decode error"))

        metrics = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=10,
        )

        # Should handle exception gracefully, TTFA should be inf
        assert metrics["ttfa_tokens"] == float("inf")
        assert metrics["ttfa_ms"] == float("inf")


class TestAggregateSpeedMetrics:
    """Tests for speed metrics aggregation."""

    def test_aggregate_empty_list(self):
        """Test aggregation with empty list."""
        result = aggregate_speed_metrics([])

        assert result["ttft_ms"]["p50"] == 0.0
        assert result["tps"]["p50"] == 0.0

    def test_aggregate_single_metric(self):
        """Test aggregation with single metric."""
        metrics_list = [
            {
                "ttft_ms": 100.0,
                "tps": 50.0,
                "ttfa_tokens": 10.0,
                "ttfa_ms": 200.0,
            }
        ]

        result = aggregate_speed_metrics(metrics_list)

        assert result["ttft_ms"]["p50"] == 100.0
        assert result["tps"]["p50"] == 50.0
        assert result["ttfa_tokens"]["p50"] == 10.0

    def test_aggregate_multiple_metrics(self):
        """Test aggregation with multiple metrics."""
        metrics_list = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": 10.0, "ttfa_ms": 200.0},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": 15.0, "ttfa_ms": 250.0},
            {"ttft_ms": 200.0, "tps": 70.0, "ttfa_tokens": 20.0, "ttfa_ms": 300.0},
        ]

        result = aggregate_speed_metrics(metrics_list)

        assert result["ttft_ms"]["p50"] == 150.0
        # numpy interpolates: 100 + 0.95*(200-100) = 195
        assert result["ttft_ms"]["p95"] == 195.0
        assert result["tps"]["p50"] == 60.0

    def test_aggregate_filters_inf(self):
        """Test that inf values are filtered from TTFA aggregation."""
        metrics_list = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": 10.0, "ttfa_ms": 200.0},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": float("inf"), "ttfa_ms": float("inf")},
        ]

        result = aggregate_speed_metrics(metrics_list)

        # Should only aggregate the finite value
        assert result["ttfa_tokens"]["p50"] == 10.0
        assert result["ttfa_ms"]["p50"] == 200.0

    def test_aggregate_all_inf_values(self):
        """Test aggregation when all values are inf (should return 0.0)."""
        import numpy as np
        metrics_list = [
            {"ttft_ms": float("inf"), "tps": float("inf"), "ttfa_tokens": float("inf"), "ttfa_ms": float("inf")},
            {"ttft_ms": float("inf"), "tps": float("inf"), "ttfa_tokens": float("inf"), "ttfa_ms": float("inf")},
        ]

        result = aggregate_speed_metrics(metrics_list)

        # When all values are inf, pct should return 0.0
        assert result["ttft_ms"]["p50"] == 0.0
        assert result["tps"]["p50"] == 0.0
        # TTFA should still be inf since they're handled separately
        assert result["ttfa_tokens"]["p50"] == float("inf")
        assert result["ttfa_ms"]["p50"] == float("inf")


class TestIsValidToolJSON:
    """Tests for tool JSON validation."""

    def test_valid_tool_json(self):
        """Test detection of valid tool JSON."""
        assert is_valid_tool_json('{"name": "web_search", "arguments": {"q": "test"}}')
        assert is_valid_tool_json('{"tool": "read_file", "path": "test.txt"}')

    def test_invalid_tool_json(self):
        """Test detection of invalid tool JSON."""
        assert not is_valid_tool_json("Just regular text")
        assert not is_valid_tool_json('{"not": "a tool"}')
        assert not is_valid_tool_json("")

    def test_partial_json(self):
        """Test that partial JSON is detected."""
        # Should detect JSON structure even if incomplete
        # Missing closing brace but has structure
        assert is_valid_tool_json('{"name": "test"')
