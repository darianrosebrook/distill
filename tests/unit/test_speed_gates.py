"""
Unit tests for speed gates evaluation.

Tests speed gate evaluation logic and baseline comparison.
"""

import json

from eval.scoring.scorer import (
    evaluate_speed_gates,
    load_baseline_speed_metrics,
)


class TestEvaluateSpeedGates:
    """Tests for speed gates evaluation."""

    def test_no_baseline_skips_gates(self):
        """Test that gates pass when no baseline provided."""
        current_metrics = {
            "ttft_ms": {"p50": 300.0, "p95": 500.0},
            "tps": {"p50": 25.0, "p95": 20.0},
            "ttfa_tokens": {"p95": 30.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=None,
            hardware_match=True,
        )

        assert result["gates_passed"] is True
        assert result["ttft_regression"]["skipped"] is True
        assert result["ttft_regression"]["reason"] == "no_baseline"

    def test_hardware_mismatch_skips_gates(self):
        """Test that gates are skipped when hardware doesn't match."""
        current_metrics = {
            "ttft_ms": {"p50": 300.0, "p95": 500.0},
            "tps": {"p50": 25.0, "p95": 20.0},
            "ttfa_tokens": {"p95": 30.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=False,
        )

        assert result["gates_passed"] is True
        assert result["ttft_regression"]["skipped"] is True
        assert result["ttft_regression"]["reason"] == "hardware_mismatch"

    def test_ttft_regression_pass(self):
        """Test TTFT regression check when within threshold."""
        current_metrics = {
            "ttft_ms": {"p50": 210.0, "p95": 420.0},  # 5% regression (within threshold)
            "tps": {"p50": 30.0, "p95": 25.0},
            "ttfa_tokens": {"p95": 25.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
            regression_threshold=0.05,
        )

        assert result["gates_passed"] is True
        assert result["ttft_regression"]["p50"]["passed"] is True
        assert result["ttft_regression"]["p95"]["passed"] is True

    def test_ttft_regression_fail(self):
        """Test TTFT regression check when exceeds threshold."""
        current_metrics = {
            "ttft_ms": {"p50": 220.0, "p95": 450.0},  # 10% and 12.5% regression (exceeds 5%)
            "tps": {"p50": 30.0, "p95": 25.0},
            "ttfa_tokens": {"p95": 25.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
            regression_threshold=0.05,
        )

        assert result["gates_passed"] is False
        assert result["ttft_regression"]["p50"]["passed"] is False
        assert result["ttft_regression"]["p95"]["passed"] is False
        assert len(result["errors"]) > 0

    def test_tps_regression_pass(self):
        """Test TPS regression check when within threshold."""
        current_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 28.5, "p95": 23.75},  # 5% regression (within threshold)
            "ttfa_tokens": {"p95": 25.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
            regression_threshold=0.05,
        )

        assert result["gates_passed"] is True
        assert result["tps_regression"]["p50"]["passed"] is True

    def test_tps_regression_fail(self):
        """Test TPS regression check when exceeds threshold."""
        current_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 27.0, "p95": 22.0},  # 10% and 12% regression (exceeds 5%)
            "ttfa_tokens": {"p95": 25.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
            regression_threshold=0.05,
        )

        assert result["gates_passed"] is False
        assert result["tps_regression"]["p50"]["passed"] is False
        assert len(result["errors"]) > 0

    def test_ttfa_gate_pass(self):
        """Test TTFA gate when p95 tokens <= 25."""
        current_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
            "ttfa_tokens": {"p95": 20.0},  # Within threshold
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
        )

        assert result["gates_passed"] is True
        assert result["ttfa_gate"]["passed"] is True

    def test_ttfa_gate_fail(self):
        """Test TTFA gate when p95 tokens > 25."""
        current_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
            "ttfa_tokens": {"p95": 30.0},  # Exceeds threshold
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
        )

        assert result["gates_passed"] is False
        assert result["ttfa_gate"]["passed"] is False
        assert "TTFA gate failed" in result["errors"][0]

    def test_all_gates_pass(self):
        """Test that all gates pass when metrics are good."""
        current_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
            "ttfa_tokens": {"p95": 20.0},
        }

        baseline_metrics = {
            "ttft_ms": {"p50": 200.0, "p95": 400.0},
            "tps": {"p50": 30.0, "p95": 25.0},
        }

        result = evaluate_speed_gates(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            hardware_match=True,
        )

        assert result["gates_passed"] is True
        assert len(result["errors"]) == 0


class TestLoadBaselineSpeedMetrics:
    """Tests for baseline speed metrics loading."""

    def test_load_baseline_success(self, tmp_path):
        """Test successful baseline loading."""
        report_path = tmp_path / "baseline.json"
        report_data = {
            "speed_metrics": {
                "ttft_ms": {"p50": 200.0, "p95": 400.0},
                "tps": {"p50": 30.0, "p95": 25.0},
            }
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f)

        baseline = load_baseline_speed_metrics(report_path)

        assert baseline is not None
        assert baseline["ttft_ms"]["p50"] == 200.0

    def test_load_baseline_not_found(self, tmp_path):
        """Test baseline loading when file doesn't exist."""
        report_path = tmp_path / "nonexistent.json"

        baseline = load_baseline_speed_metrics(report_path)

        assert baseline is None

    def test_load_baseline_invalid_json(self, tmp_path):
        """Test baseline loading with invalid JSON."""
        report_path = tmp_path / "invalid.json"

        with open(report_path, "w") as f:
            f.write("invalid json content")

        baseline = load_baseline_speed_metrics(report_path)

        assert baseline is None
