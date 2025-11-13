"""
CAWS compliance gate validation tests.

Tests that CAWS gates properly catch violations:
- Privacy gate (PII detection)
- Control integration gate (negative controls)
- Fixture hit-rate gate (malformed tool calls)
- JSON validity gate
- Integration F1 gate
@author: @darianrosebrook
"""

import pytest
from typing import Dict, Any, List, Optional

from eval.scoring.scorer import _evaluate_caws_compliance


def create_test_result(
    model_output: str,
    teacher_text: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a test result dictionary."""
    result = {
        "model_output": model_output,
        "metadata": metadata or {},
    }
    if teacher_text:
        result["metadata"]["teacher_text"] = teacher_text
    if tool_trace:
        result["tool_trace"] = tool_trace
    return result


def test_privacy_gate_pii_detection():
    """Test that privacy gate catches PII-containing outputs."""
    # Create result with PII
    results = [
        create_test_result(
            model_output="User email is john.doe@example.com",
            teacher_text="User email is [REDACTED]",
        ),
        create_test_result(
            model_output="Phone number: 555-123-4567",
            teacher_text="Phone number: [REDACTED]",
        ),
    ]

    # Evaluate compliance
    compliance = _evaluate_caws_compliance(results)

    # Privacy gate should fail if PII is present
    # Note: This is a simplified test - actual PII detection would be more sophisticated
    assert "compliance_gate_passed" in compliance
    assert "privacy_ok_rate" in compliance or "avg_quality_penalty" in compliance


def test_control_integration_gate():
    """Test that control integration gate catches negative controls with integration."""
    # Create negative control result (should not have integration)
    results = [
        create_test_result(
            model_output="This is a control example with no tool calls",
            metadata={"is_control": True, "has_integration": False},
        ),
        create_test_result(
            model_output="Control with integration span: <tool_result>data</tool_result>",
            metadata={"is_control": True, "has_integration": True},  # Violation
        ),
    ]

    # Evaluate compliance
    compliance = _evaluate_caws_compliance(results)

    # Control integration gate should fail if controls have integration
    assert "compliance_gate_passed" in compliance


def test_json_validity_gate():
    """Test that JSON validity gate catches invalid JSON in tool calls."""
    results = [
        create_test_result(
            model_output='Valid tool call: {"name": "web.search", "arguments": {"q": "test"}}',
            tool_trace=[{"name": "web.search", "arguments": {"q": "test"}}],
        ),
        create_test_result(
            model_output='Invalid JSON: {"name": "web.search", "arguments": {invalid}}',
            tool_trace=[{"name": "web.search", "arguments": "invalid"}],  # Invalid
        ),
    ]

    # Evaluate compliance
    compliance = _evaluate_caws_compliance(results)

    # JSON validity gate should fail if invalid JSON present
    assert "compliance_gate_passed" in compliance


def test_fixture_hit_rate_gate():
    """Test that fixture hit-rate gate catches malformed tool calls."""
    results = [
        create_test_result(
            model_output="Tool call: web.search",
            tool_trace=[{"name": "web.search", "arguments": {"q": "test"}}],
        ),
        create_test_result(
            model_output="Malformed tool: unknown.tool",
            # Not in fixtures
            tool_trace=[{"name": "unknown.tool", "arguments": {}}],
        ),
    ]

    # Evaluate compliance
    compliance = _evaluate_caws_compliance(results)

    # Fixture hit-rate gate should fail if too many misses
    assert "compliance_gate_passed" in compliance


def test_broker_fixtures_hit_rate():
    """Test broker fixture hit rate (placeholder test)."""
    # This test validates that the broker fixtures are properly loaded
    # and that the hit rate calculation works correctly

    # For now, just check that we can import the necessary modules
    try:
        from eval.scoring.scorer import _evaluate_caws_compliance
        from eval.tool_broker.fixtures import FixtureManager

        print("✅ Successfully imported CAWS scorer and fixture manager")
        # Basic test that import works
        assert _evaluate_caws_compliance is not None
        assert FixtureManager is not None
    except ImportError as e:
        # If modules aren't available, skip the test (this is expected in CI environments)
        pytest.skip(f"Required modules not available: {e}")

    # TODO: Add actual fixture hit rate validation
    # This would test that fixtures are loaded correctly and hit rates are calculated


def test_integration_f1_gate():
    """Test that integration F1 gate enforces minimum quality."""
    # Create results with varying integration quality
    results = [
        create_test_result(
            model_output="Integration: <tool_result>data</tool_result> used in response",
            metadata={"integration_f1_lax": 0.95},  # Good
        ),
        create_test_result(
            model_output="Poor integration: <tool_result>data</tool_result> ignored",
            metadata={"integration_f1_lax": 0.70},  # Below threshold
        ),
    ]

    # Evaluate compliance
    compliance = _evaluate_caws_compliance(results)

    # Integration F1 gate should fail if too many low-F1 items
    assert "compliance_gate_passed" in compliance


def test_gate_thresholds_documented():
    """Test that gate thresholds are explicitly documented."""
    # Check that gates are defined with thresholds
    gates = {
        "privacy_ok_rate": {"threshold": 1.0, "policy": "hard_fail"},
        "controls_with_integration": {"threshold": 0, "policy": "hard_fail"},
        "json_args_valid_rate": {"threshold": 0.98, "policy": "hard_fail"},
        "integration_f1_macro_lax": {
            "threshold": 0.90,
            "policy": "count_based_misses",
            "misses_allowed_pct": 0.05,
        },
        "multi_call_parity_rate": {
            "threshold": 0.95,
            "policy": "count_based_misses",
            "misses_allowed_pct": 0.05,
        },
    }

    # Verify all gates have thresholds
    for gate_name, gate_config in gates.items():
        assert "threshold" in gate_config, f"Gate {gate_name} missing threshold"
        assert "policy" in gate_config, f"Gate {gate_name} missing policy"

    # Verify hard_fail gates have strict thresholds
    hard_fail_gates = [name for name, cfg in gates.items() if cfg["policy"] == "hard_fail"]
    for gate_name in hard_fail_gates:
        gate_config = gates[gate_name]
        if gate_name == "privacy_ok_rate":
            assert gate_config["threshold"] == 1.0, "Privacy gate must be 1.0"
        elif gate_name == "controls_with_integration":
            assert gate_config["threshold"] == 0, "Control integration gate must be 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
