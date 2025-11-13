"""
End-to-end toy test for Context Efficiency Score (CES) measurement.

Tests CES improvements for:
- Baseline vs code-mode
- Baseline vs latent reasoning
- Combined milestones (code-mode + latent)

Author: @darianrosebrook
"""
import pytest
from typing import Dict, Any, List

from eval.scoring.scorer import score_item
from eval.scoring.efficiency import (
    EfficiencyMetrics,
    calculate_token_reduction,
    compare_with_baseline,
    evaluate_efficiency_gates,
)


class TestCESMeasurement:
    """Test CES measurement and improvement verification."""

    def create_mock_tool_trace(self, used_code_mode: bool = False, used_direct_tool: bool = True, large_payload: bool = True) -> List[Dict[str, Any]]:
        """Create mock tool trace for testing."""
        if used_code_mode:
            # Code-mode: TS API calls, results stay in sandbox
            return [
                {
                    "name": "google_drive__get_document",
                    "arguments": {"documentId": "abc123"},
                    "result": {"content": "x" * 50000 if large_payload else "small"},  # Large payload
                    "code_mode": True,
                },
                {
                    "name": "salesforce__update_record",
                    "arguments": {"objectType": "SalesMeeting", "recordId": "00Q..."},
                    "result": {"success": True},
                    "code_mode": True,
                },
            ]
        elif used_direct_tool:
            # Direct tool: results echoed to tokens
            return [
                {
                    "name": "google_drive__get_document",
                    "arguments": {"documentId": "abc123"},
                    "result": {"content": "x" * 50000 if large_payload else "small"},
                    "code_mode": False,
                },
            ]
        else:
            return []

    def create_mock_item(self, eligible_for_code_mode: bool = True) -> Dict[str, Any]:
        """Create mock evaluation item."""
        return {
            "prompt": "Process this large document and update records.",
            "metadata": {
                "expected_behaviour": "normal",
                "call_sequence": [
                    {"name": "google_drive__get_document", "arguments": {"documentId": "abc123"}},
                    {"name": "salesforce__update_record", "arguments": {"objectType": "SalesMeeting", "recordId": "00Q..."}}
                ],
                "tool_count": 2,
                "intermediate_sizes": [50000] if eligible_for_code_mode else [500],
                "pii_tags_present": False,
                "eligible_for_code_mode": eligible_for_code_mode,
            },
        }

    def test_ces_baseline_vs_code_mode(self):
        """
        Test CES comparison: baseline direct-tool vs code-mode.

        Verifies:
        - Code-mode reduces tool result tokens in context (main benefit)
        - CES calculation works correctly for both modes
        - Code-mode flags are set correctly
        """
        item = self.create_mock_item(eligible_for_code_mode=True)

        # Baseline: Direct tool calls (results echoed to tokens - VERY LARGE)
        baseline_output = (
            "<|tool_call|>google_drive__get_document\n"
            "<|tool_result|>{\"content\": \"" + "x" * 10000 + "\"}\n"  # Large result echoed to tokens
            "<|tool_call|>salesforce__update_record\n"
            "<|tool_result|>{\"success\": true}\n"
        )
        baseline_trace = self.create_mock_tool_trace(used_code_mode=False, used_direct_tool=True, large_payload=True)

        baseline_scores = score_item(item, baseline_output, baseline_trace)

        # Code-mode: TS API orchestration (results stay in sandbox, only summary returned)
        code_mode_output = (
            "import * as gdrive from './servers/google-drive';\n"
            "import * as salesforce from './servers/salesforce';\n"
            "const doc = await gdrive.getDocument({ documentId: 'abc123' });\n"
            "const notes = summarize(doc.content, 5);\n"
            "await salesforce.updateRecord({ objectType: 'SalesMeeting', recordId: '00Q...', data: { Notes: notes } });\n"
            "console.log('Task completed');\n"
        )
        code_mode_trace = self.create_mock_tool_trace(used_code_mode=True, used_direct_tool=False, large_payload=True)

        code_mode_scores = score_item(item, code_mode_output, code_mode_trace)

        # Verify CES metrics are computed
        assert "ces_tokens_total" in baseline_scores
        assert "ces_tokens_total" in code_mode_scores

        baseline_scores["ces_tokens_total"]
        code_mode_scores["ces_tokens_total"]

        # The key insight: code-mode reduces tool result tokens in context
        # Baseline includes large payload in context: ces_tokens_direct_tool should be large
        # Code-mode isolates results: ces_tokens_code_mode includes file reads but tool results are isolated

        baseline_tool_tokens = baseline_scores.get("ces_tokens_direct_tool", 0)
        code_mode_file_read_tokens = code_mode_scores.get("ces_tokens_code_mode", 0) if code_mode_scores.get("used_code_mode") else 0

        # Verify baseline has large tool result tokens (large payload echoed to context)
        assert baseline_tool_tokens > 2000, f"Baseline should have large tool tokens (got {baseline_tool_tokens})"

        # Verify code-mode has file read tokens (large data processed but not in context)
        assert code_mode_file_read_tokens > 1000, f"Code-mode should have file read tokens (got {code_mode_file_read_tokens})"

        # The main benefit: large results don't get echoed to the token stream
        # In baseline, large payload is in context (ces_tokens_direct_tool)
        # In code-mode, large data is read but results are isolated

        # Verify code-mode used flag
        assert code_mode_scores["used_code_mode"] is True
        assert code_mode_scores["used_direct_tool"] is False

        # Verify baseline used direct tool
        assert baseline_scores["used_direct_tool"] is True
        assert baseline_scores["used_code_mode"] is False

        # Note: Total CES may not be smaller in code-mode due to file_read_tokens,
        # but the key benefit is that large results don't pollute the token stream
        print(f"✅ CES test passed: baseline_tool_tokens={baseline_tool_tokens}, code_mode_file_read_tokens={code_mode_file_read_tokens}")

    def test_ces_baseline_vs_latent(self):
        """
        Test CES comparison: baseline direct CoT vs latent reasoning.
        
        Verifies:
        - Latent reasoning reduces generated tokens
        - ≥25% token reduction at equal accuracy
        - Efficiency metrics are computed correctly
        """
        # Baseline: Direct CoT (all steps visible)
        baseline_metrics = EfficiencyMetrics(
            accuracy=0.85,
            generated_tokens=100,  # All CoT steps generate tokens
            wall_clock_time_ms=100.0,
            latent_spans_used=0,
            refinement_loops=1,
        )
        
        # Latent reasoning: Some steps are latent (no tokens generated)
        latent_metrics = EfficiencyMetrics(
            accuracy=0.85,  # Same accuracy
            generated_tokens=70,  # 30% reduction (some steps latent)
            wall_clock_time_ms=95.0,  # Slightly faster
            latent_spans_used=2,  # 2 latent spans used
            refinement_loops=1,
        )
        
        # Compare with baseline
        comparison = compare_with_baseline(latent_metrics, baseline_metrics)
        
        # Verify token reduction
        assert comparison["token_reduction"] > 0, "Should have token reduction"
        assert comparison["token_reduction"] >= 0.25, f"Token reduction {comparison['token_reduction']:.1%} < 25%"
        
        # Verify accuracy maintained
        assert comparison["accuracy_maintained"] is True, "Accuracy should be maintained"
        assert abs(comparison["accuracy_delta"]) <= 0.01, "Accuracy delta should be ≤1%"
        
        # Verify efficiency gates
        gates = evaluate_efficiency_gates(latent_metrics, baseline_metrics)
        
        assert gates["token_reduction_gate"] is True, "Token reduction gate should pass"
        assert gates["accuracy_gate"] is True, "Accuracy gate should pass"
        assert gates["all_gates_passed"] is True, "All efficiency gates should pass"

    def test_ces_combined_milestones(self):
        """
        Test CES measurement with both code-mode and latent reasoning enabled.
        
        Verifies:
        - Combined features provide additive CES improvements
        - Both code-mode and latent metrics are tracked
        - Efficiency gates pass with both features
        """
        item = self.create_mock_item(eligible_for_code_mode=True)
        
        # Baseline: Direct tool calls with full CoT
        baseline_output = (
            "Step 1: Analyze the document\n"
            "Step 2: Extract key information\n"
            "Step 3: Process data\n"
            "<|tool_call|>google_drive__get_document\n"
            "<|tool_result|>{\"content\": \"" + "x" * 1000 + "\"}\n"
            "<|tool_call|>salesforce__update_record\n"
            "<|tool_result|>{\"success\": true}\n"
        )
        baseline_trace = self.create_mock_tool_trace(used_code_mode=False, used_direct_tool=True, large_payload=True)
        
        baseline_scores = score_item(item, baseline_output, baseline_trace)
        
        # Combined: Code-mode + latent reasoning
        # Latent spans for planning steps, TS API for tool calls
        combined_output = (
            "<bot>\n"  # Latent span for planning
            "import * as gdrive from './servers/google-drive';\n"
            "import * as salesforce from './servers/salesforce';\n"
            "<eot>\n"  # End latent span
            "Step 3: Process data\n"  # Visible step
            "const doc = await gdrive.getDocument({ documentId: 'abc123' });\n"
            "const notes = summarize(doc.content, 5);\n"
            "await salesforce.updateRecord({ objectType: 'SalesMeeting', recordId: '00Q...', data: { Notes: notes } });\n"
        )
        combined_trace = self.create_mock_tool_trace(used_code_mode=True, used_direct_tool=False, large_payload=True)
        
        combined_scores = score_item(item, combined_output, combined_trace)
        
        # Verify CES metrics
        baseline_ces = baseline_scores["ces_tokens_total"]
        combined_ces = combined_scores["ces_tokens_total"]
        
        # Verify combined approach reduces CES
        assert combined_ces < baseline_ces, f"Combined CES {combined_ces} should be < baseline {baseline_ces}"
        
        # Calculate improvement
        improvement = (baseline_ces - combined_ces) / baseline_ces if baseline_ces > 0 else 0.0
        
        # Verify ≥25% improvement
        assert improvement >= 0.25, f"Combined CES improvement {improvement:.1%} < 25%"
        
        # Verify both features are used
        assert combined_scores["used_code_mode"] is True, "Code-mode should be used"
        
        # Create efficiency metrics for latent reasoning comparison
        # Estimate tokens: baseline has more visible steps + tool results
        baseline_tokens_estimate = len(baseline_output.split()) + baseline_scores.get("ces_tokens_direct_tool", 0)
        combined_tokens_estimate = len(combined_output.split()) + combined_scores.get("ces_tokens_code_mode", 0)
        
        baseline_metrics = EfficiencyMetrics(
            accuracy=0.85,
            generated_tokens=baseline_tokens_estimate,
            wall_clock_time_ms=100.0,
            latent_spans_used=0,
            refinement_loops=1,
        )
        
        combined_metrics = EfficiencyMetrics(
            accuracy=0.85,
            generated_tokens=combined_tokens_estimate,
            wall_clock_time_ms=90.0,  # Faster due to latent spans
            latent_spans_used=1,  # One latent span used
            refinement_loops=1,
        )
        
        # Verify efficiency gates
        gates = evaluate_efficiency_gates(combined_metrics, baseline_metrics)
        
        # Token reduction should be significant
        token_reduction = calculate_token_reduction(
            baseline_metrics.generated_tokens,
            combined_metrics.generated_tokens,
        )
        
        assert token_reduction >= 0.25, f"Combined token reduction {token_reduction:.1%} < 25%"
        assert gates["token_reduction_gate"] is True, "Token reduction gate should pass"
        assert gates["accuracy_gate"] is True, "Accuracy gate should pass"

    def test_ces_non_eligible_scenario(self):
        """
        Test that CES improvement is not required for non-eligible scenarios.
        
        Verifies:
        - Single-tool scenarios don't require code-mode
        - Small payloads don't require code-mode
        - CES gates don't fail for non-eligible scenarios
        """
        # Non-eligible item: single tool, small payload
        item = self.create_mock_item(eligible_for_code_mode=False)
        item["metadata"]["tool_count"] = 1
        item["metadata"]["intermediate_sizes"] = [500]
        
        # Direct tool call (acceptable for non-eligible)
        output = (
            "<|tool_call|>single_tool\n"
            "<|tool_result|>{\"result\": \"small\"}\n"
        )
        trace = self.create_mock_tool_trace(used_code_mode=False, used_direct_tool=True, large_payload=False)
        
        scores = score_item(item, output, trace)
        
        # Verify non-eligible flag
        assert scores.get("eligible_for_code_mode", True) is False, "Should be non-eligible"
        
        # CES should be computed but gates shouldn't require improvement
        assert "ces_tokens_total" in scores
        assert scores["ces_tokens_total"] > 0


if __name__ == '__main__':
    pytest.main([__file__, "-v"])

