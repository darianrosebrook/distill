"""
Unit tests for training/speed_metrics.py

Tests speed metrics measurement functionality including proxy measurements,
tool JSON validation, and metrics aggregation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from training.speed_metrics import (
    measure_proxy,
    is_valid_tool_json,
    aggregate_speed_metrics,
)


class TestIsValidToolJson:
    """Test tool JSON validation functionality."""

    def test_is_valid_tool_json_valid_tool_call(self):
        """Test detection of valid tool call JSON."""
        text = '{"name": "search", "arguments": {"query": "test"}}'
        assert is_valid_tool_json(text) is True

    def test_is_valid_tool_json_valid_tool_name(self):
        """Test detection when text contains 'tool' keyword."""
        text = '{"tool": "calculator", "params": [1, 2, 3]}'
        assert is_valid_tool_json(text) is True

    def test_is_valid_tool_json_partial_json(self):
        """Test detection of partial JSON (missing closing brace)."""
        text = '{"name": "search", "arguments": {"query": "test"'
        assert is_valid_tool_json(text) is True

    def test_is_valid_tool_json_invalid_no_colon(self):
        """Test rejection when no colon present."""
        text = '{"name" "search"}'
        assert is_valid_tool_json(text) is False

    def test_is_valid_tool_json_invalid_no_brace(self):
        """Test rejection when no opening brace."""
        text = '"name": "search"'
        assert is_valid_tool_json(text) is False

    def test_is_valid_tool_json_invalid_no_tool_keywords(self):
        """Test rejection when no tool-related keywords."""
        text = '{"user": "john", "age": 30}'
        assert is_valid_tool_json(text) is False

    def test_is_valid_tool_json_empty_text(self):
        """Test empty text."""
        assert is_valid_tool_json("") is False

    def test_is_valid_tool_json_plain_text(self):
        """Test plain text without JSON structure."""
        text = "This is just plain text without any JSON."
        assert is_valid_tool_json(text) is False

    def test_is_valid_tool_json_malformed_json(self):
        """Test malformed JSON structure."""
        text = '{"name": "search", "arguments": }'
        # Still passes because it has colon and is within braces (heuristic)
        assert is_valid_tool_json(text) is True


class TestMeasureProxy:
    """Test proxy speed metrics measurement."""

    @patch("training.speed_metrics.time")
    def test_measure_proxy_basic_functionality(self, mock_time, device):
        """Test basic proxy measurement functionality."""
        # Mock time to return predictable values
        mock_time.perf_counter.side_effect = [0.0, 0.1, 0.2, 0.3]  # t0, t1, t_start, t_end

        # Create mock model
        mock_model = Mock()
        mock_model.eval.return_value = None

        # Mock logits: shape [B=1, T=5, V=1000]
        logits = torch.randn(1, 5, 1000, device=device)
        mock_model.return_value = logits

        # Mock forward_decode for KV cache path - return tuple of (logits, kv_cache)
        kv_cache_result = torch.randn(1, 1, 1000, device=device)
        mock_model.forward_decode.return_value = (kv_cache_result, Mock())

        # Create test batch
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 5), device=device),
            "attention_mask": torch.ones(1, 5, dtype=torch.long, device=device),
        }

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = '{"name": "test"}'

        result = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=4,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "ttft_ms" in result
        assert "tps" in result
        assert "ttfa_tokens" in result
        assert "ttfa_ms" in result
        assert "tokens_generated" in result

        # Verify reasonable values
        assert result["ttft_ms"] == 100.0  # (0.1 - 0.0) * 1000
        assert result["tps"] > 0  # Should be positive
        assert result["ttfa_tokens"] == 4  # Detected valid JSON after 4 tokens generated
        assert result["ttfa_ms"] == 300.0  # (0.3 - 0.0) * 1000
        assert result["tokens_generated"] == 4  # Generated 4 tokens

    @patch("training.speed_metrics.time")
    def test_measure_proxy_without_kv_cache(self, mock_time, device):
        """Test proxy measurement without KV cache (fallback path)."""
        mock_time.perf_counter.side_effect = [0.0, 0.1, 0.2, 0.3]

        # Create mock model without forward_decode
        mock_model = Mock()
        mock_model.eval.return_value = None

        # Remove forward_decode to force fallback path
        del mock_model.forward_decode

        # Mock logits for initial and subsequent calls
        initial_logits = torch.randn(1, 5, 1000, device=device)
        subsequent_logits = torch.randn(1, 6, 1000, device=device)
        mock_model.side_effect = [initial_logits, subsequent_logits]

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 5), device=device),
        }

        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "plain text"

        result = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=2,
        )

        # Should use fallback path and not detect valid JSON
        assert result["ttfa_tokens"] == float("inf")
        assert result["ttfa_ms"] == float("inf")

    def test_measure_proxy_no_tokenizer(self, device):
        """Test proxy measurement without tokenizer."""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.return_value = torch.randn(1, 5, 1000, device=device)
        # Remove forward_decode to force fallback path
        del mock_model.forward_decode

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 5), device=device),
        }

        result = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=None,
            device=device,
        )

        # Should skip TTFA measurement
        assert result["ttfa_tokens"] == float("inf")
        assert result["ttfa_ms"] == float("inf")

    def test_measure_proxy_tokenizer_decode_error(self, device):
        """Test proxy measurement when tokenizer decode fails."""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.return_value = torch.randn(1, 5, 1000, device=device)
        # Remove forward_decode to force fallback path
        del mock_model.forward_decode

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 5), device=device),
        }

        mock_tokenizer = Mock()
        mock_tokenizer.decode.side_effect = Exception("Decode error")

        result = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
        )

        # Should handle decode error gracefully
        assert result["ttfa_tokens"] == float("inf")
        assert result["ttfa_ms"] == float("inf")

    def test_measure_proxy_no_attention_mask(self, device):
        """Test proxy measurement without attention mask."""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.return_value = torch.randn(1, 5, 1000, device=device)
        # Remove forward_decode to force fallback path
        del mock_model.forward_decode

        batch = {
            "input_ids": torch.randint(0, 1000, (1, 5), device=device),
            # No attention_mask
        }

        result = measure_proxy(
            model=mock_model,
            batch=batch,
            tokenizer=None,
            device=device,
        )

        # Should work without attention mask
        assert isinstance(result, dict)
        assert result["ttft_ms"] > 0


class TestAggregateSpeedMetrics:
    """Test speed metrics aggregation functionality."""

    def test_aggregate_speed_metrics_empty_list(self):
        """Test aggregation with empty metrics list."""
        result = aggregate_speed_metrics([])
        expected = {
            "ttft_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0},
            "tps": {"p50": 0.0, "p90": 0.0, "p95": 0.0},
            "ttfa_tokens": {"p50": 0.0, "p95": 0.0},
            "ttfa_ms": {"p50": 0.0, "p95": 0.0},
        }
        assert result == expected

    def test_aggregate_speed_metrics_single_item(self):
        """Test aggregation with single metrics item."""
        metrics = [{
            "ttft_ms": 100.0,
            "tps": 50.0,
            "ttfa_tokens": 5.0,
            "ttfa_ms": 200.0,
            "tokens_generated": 10,
        }]

        result = aggregate_speed_metrics(metrics)

        # Single item should return same values for all percentiles
        assert result["ttft_ms"]["p50"] == 100.0
        assert result["ttft_ms"]["p90"] == 100.0
        assert result["ttft_ms"]["p95"] == 100.0

        assert result["tps"]["p50"] == 50.0
        assert result["tps"]["p90"] == 50.0
        assert result["tps"]["p95"] == 50.0

        assert result["ttfa_tokens"]["p50"] == 5.0
        assert result["ttfa_tokens"]["p95"] == 5.0

        assert result["ttfa_ms"]["p50"] == 200.0
        assert result["ttfa_ms"]["p95"] == 200.0

    def test_aggregate_speed_metrics_multiple_items(self):
        """Test aggregation with multiple metrics items."""
        metrics = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": 5.0, "ttfa_ms": 200.0, "tokens_generated": 10},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": 7.0, "ttfa_ms": 300.0, "tokens_generated": 12},
            {"ttft_ms": 120.0, "tps": 55.0, "ttfa_tokens": 6.0, "ttfa_ms": 250.0, "tokens_generated": 11},
        ]

        result = aggregate_speed_metrics(metrics)

        # Check that percentiles are calculated correctly
        assert result["ttft_ms"]["p50"] == 120.0  # Median
        assert result["ttft_ms"]["p90"] > 120.0   # 90th percentile
        assert result["ttft_ms"]["p95"] > result["ttft_ms"]["p90"]  # 95th percentile

        assert result["tps"]["p50"] == 55.0
        assert result["tps"]["p90"] > 55.0
        assert result["tps"]["p95"] > result["tps"]["p90"]

    def test_aggregate_speed_metrics_with_inf_values(self):
        """Test aggregation when some TTFA values are infinite."""
        metrics = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": 5.0, "ttfa_ms": 200.0, "tokens_generated": 10},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": float("inf"), "ttfa_ms": float("inf"), "tokens_generated": 12},
            {"ttft_ms": 120.0, "tps": 55.0, "ttfa_tokens": 6.0, "ttfa_ms": 250.0, "tokens_generated": 11},
        ]

        result = aggregate_speed_metrics(metrics)

        # TTFA metrics should filter out inf values
        assert 5.0 <= result["ttfa_tokens"]["p50"] <= 6.0  # Median of [5.0, 6.0]
        assert result["ttfa_tokens"]["p95"] >= 5.9   # 95th percentile of [5.0, 6.0] (approx 5.95)

        assert 200.0 <= result["ttfa_ms"]["p50"] <= 250.0  # Median of [200.0, 250.0]
        assert result["ttfa_ms"]["p95"] >= 240.0   # 95th percentile of [200.0, 250.0] (approx 245)

    def test_aggregate_speed_metrics_all_inf_ttfa(self):
        """Test aggregation when all TTFA values are infinite."""
        metrics = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": float("inf"), "ttfa_ms": float("inf"), "tokens_generated": 10},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": float("inf"), "ttfa_ms": float("inf"), "tokens_generated": 12},
        ]

        result = aggregate_speed_metrics(metrics)

        # Should return inf when no finite values
        assert result["ttfa_tokens"]["p50"] == float("inf")
        assert result["ttfa_tokens"]["p95"] == float("inf")
        assert result["ttfa_ms"]["p50"] == float("inf")
        assert result["ttfa_ms"]["p95"] == float("inf")

    def test_aggregate_speed_metrics_varied_performance(self):
        """Test aggregation with varied performance characteristics."""
        # Create metrics with different performance profiles
        metrics = [
            # Fast model
            {"ttft_ms": 50.0, "tps": 100.0, "ttfa_tokens": 3.0, "ttfa_ms": 100.0, "tokens_generated": 10},
            # Medium model
            {"ttft_ms": 100.0, "tps": 75.0, "ttfa_tokens": 5.0, "ttfa_ms": 200.0, "tokens_generated": 10},
            # Slow model
            {"ttft_ms": 200.0, "tps": 50.0, "ttfa_tokens": 8.0, "ttfa_ms": 400.0, "tokens_generated": 10},
        ]

        result = aggregate_speed_metrics(metrics)

        # Verify percentiles reflect the performance distribution
        assert 50.0 <= result["ttft_ms"]["p50"] <= 200.0
        assert 50.0 <= result["tps"]["p50"] <= 100.0
        assert 3.0 <= result["ttfa_tokens"]["p50"] <= 8.0
        assert 100.0 <= result["ttfa_ms"]["p50"] <= 400.0

        # 95th percentile should be at or near the highest value
        assert result["ttft_ms"]["p95"] >= 150.0
        assert result["tps"]["p95"] >= 75.0
        assert result["ttfa_tokens"]["p95"] >= 5.0
        assert result["ttfa_ms"]["p95"] >= 200.0
