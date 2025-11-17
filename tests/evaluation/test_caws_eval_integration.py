"""
Integration tests for evaluation/caws_eval.py - Real CAWS compliance evaluation.

Tests that actually exercise the CAWS compliance logic using real spec files,
change diffs, and evaluation data instead of just mocking everything.
"""
# @author: @darianrosebrook

import json

import pytest

from evaluation.caws_eval import (
    validate_budget_adherence,
    validate_gate_integrity,
    validate_provenance_clarity,
    evaluate_caws_compliance,
    _load_working_spec,
    _load_json_file,
)


class TestCawsEvalIntegration:
    """Integration tests that actually exercise real CAWS evaluation logic."""

    def test_validate_budget_adherence_real_diff(self):
        """Test validate_budget_adherence with real git diff content."""
        # Create a realistic git diff
        diff_content = """diff --git a/file1.py b/file1.py
index 1234567..abcdef0 100644
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,5 @@
 def old_function():
-    pass
+    print("modified")
+    return True

+def new_function():
+    return "added"
diff --git a/file2.py b/file2.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/file2.py
@@ -0,0 +1,10 @@
+class NewClass:
+    def method(self):
+        return "new file"
"""

        result = validate_budget_adherence(
            diff_content, max_loc=100, max_files=5)

        # Should detect the changes
        assert result["within_budget"] is True
        assert result["lines_added"] > 0  # Should detect additions
        assert result["lines_removed"] > 0  # Should detect removal
        assert result["files_changed_count"] == 2  # Two files changed
        assert result["total_loc"] > 0

    def test_validate_budget_adherence_empty_diff(self):
        """Test validate_budget_adherence with empty diff."""
        result = validate_budget_adherence("", max_loc=100, max_files=5)

        assert result["within_budget"] is True
        assert result["lines_added"] == 0
        assert result["lines_removed"] == 0
        # Default when no files detected
        assert result["files_changed_count"] == 1
        assert result["total_loc"] == 0

    def test_validate_budget_adherence_over_budget(self):
        """Test validate_budget_adherence when exceeding budget."""
        # Create a diff that exceeds the limits
        large_diff = "diff --git a/big.py b/big.py\n"
        large_diff += "--- a/big.py\n+++ b/big.py\n"
        for i in range(150):  # Exceed 100 LOC limit
            large_diff += f"+line {i}\n"

        result = validate_budget_adherence(
            large_diff, max_loc=100, max_files=5)

        assert result["within_budget"] is False
        assert result["lines_added"] >= 150
        assert result["total_loc"] >= 150

    def test_validate_gate_integrity_passing_results(self):
        """Test validate_gate_integrity with passing test/lint/coverage results."""
        test_results = {
            "passed": 95,
            "failed": 0
        }

        lint_results = {
            "errors": 0,
            "warnings": 2
        }

        coverage_results = {
            "line_percent": 85.0,
            "branch_percent": 80.0
        }

        result = validate_gate_integrity(
            test_results, lint_results, coverage_results)

        # Should pass with good results
        assert result["all_gates_pass"] is True
        assert result["tests_pass"] is True
        assert result["lint_pass"] is True
        assert result["coverage_pass"] is True

    def test_validate_gate_integrity_failing_results(self):
        """Test validate_gate_integrity with failing results."""
        test_results = {
            "passed": 50,  # Low pass count
            "failed": 50
        }

        lint_results = {
            "errors": 5,  # Has errors
            "warnings": 10
        }

        coverage_results = {
            "line_percent": 40.0,  # Very low coverage
            "branch_percent": 30.0
        }

        result = validate_gate_integrity(
            test_results, lint_results, coverage_results)

        assert result["all_gates_pass"] is False  # Should fail
        assert result["tests_pass"] is False
        assert result["lint_pass"] is False
        assert result["coverage_pass"] is False

    def test_validate_provenance_clarity_good_evidence(self):
        """Test validate_provenance_clarity with good rationale and evidence."""
        rationale = "Adding new feature X to improve performance by 20%. Implementation includes comprehensive tests and maintains backward compatibility."

        evidence_manifest = {
            "test_results": {"coverage": 85.0, "tests_passed": 95},
            "benchmarks": {"performance_improvement": 18.5},
            "compatibility": {"breaking_changes": False}
        }

        result = validate_provenance_clarity(
            rationale, evidence_manifest, change_diff=True)

        # Should pass with good rationale
        assert result["overall_clarity"] is True
        assert result["rationale_present"] is True
        assert result["evidence_present"] is True
        assert result["alignment_score"] == 1.0

    def test_validate_provenance_clarity_poor_evidence(self):
        """Test validate_provenance_clarity with poor rationale and missing evidence."""
        rationale = "fix"  # Too brief

        evidence_manifest = {}  # Empty evidence

        result = validate_provenance_clarity(
            rationale, evidence_manifest, change_diff=False)

        assert result["overall_clarity"] is False  # Should fail
        # "fix" is present but too brief
        assert result["rationale_present"] is True
        assert result["evidence_present"] is False  # Empty evidence

    def test_evaluate_caws_compliance_full_integration(self, tmp_path):
        """Test evaluate_caws_compliance with full integration."""
        # Create a working spec file
        working_spec = {
            "budgets": {
                "max_loc": 1000,
                "max_files": 10
            }
        }

        spec_file = tmp_path / "working_spec.yaml"
        try:
            import yaml
            with open(spec_file, "w") as f:
                yaml.safe_dump(working_spec, f)
        except ImportError:
            # Fallback to JSON if yaml not available
            spec_file = tmp_path / "working_spec.json"
            with open(spec_file, "w") as f:
                json.dump(working_spec, f)

        # Good test results
        test_results = {"passed": 98, "failed": 2}
        lint_results = {"errors": 0, "warnings": 1}
        coverage_results = {"line_percent": 88.0, "branch_percent": 82.0}

        result = evaluate_caws_compliance(
            change_id="TEST-123",
            working_spec=working_spec,
            change_diff="small change",
            rationale="Adding new feature with comprehensive tests",
            test_results=test_results,
            lint_results=lint_results,
            coverage_results=coverage_results
        )

        assert result["change_id"] == "TEST-123"
        assert result["verdict"] == "APPROVED"  # Should pass
        assert result["overall_compliance"] is True
        assert "caws_compliance" in result
        assert result["budget_adherence"]["within_budget"] is True
        assert result["gate_integrity"]["all_gates_pass"] is True

    def test_evaluate_caws_compliance_budget_violation(self):
        """Test evaluate_caws_compliance when budget is violated."""
        # Large change that violates budget (1500 lines added)
        large_diff = "\n".join(["+" + str(i)
                               for i in range(1500)])  # 1500 lines added

        result = evaluate_caws_compliance(
            change_id="BIG-CHANGE",
            change_diff=large_diff,
            max_loc=1000,
            max_files=10,
            rationale="Large refactoring",
            test_results={"passed": 95, "failed": 5},
            lint_results={"errors": 0, "warnings": 0},
            coverage_results={"line_percent": 85.0}
        )

        # Should be rejected due to budget
        assert result["verdict"] == "REJECTED"
        assert result["overall_compliance"] is False
        assert result["budget_adherence"]["within_budget"] is False

    def test_load_working_spec_json_file(self, tmp_path):
        """Test _load_working_spec with JSON file."""
        spec_data = {
            "budgets": {"max_loc": 500, "max_files": 5},
            "gates": {"require_tests": True}
        }

        spec_file = tmp_path / "spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        result = _load_working_spec(str(spec_file))

        assert result == spec_data

    def test_load_working_spec_yaml_file(self, tmp_path):
        """Test _load_working_spec with YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        spec_data = {
            "budgets": {"max_loc": 800, "max_files": 8},
            "gates": {"require_coverage": True}
        }

        spec_file = tmp_path / "spec.yaml"
        with open(spec_file, "w") as f:
            yaml.safe_dump(spec_data, f)

        result = _load_working_spec(str(spec_file))

        assert result == spec_data

    def test_load_json_file_real_file(self, tmp_path):
        """Test _load_json_file with real JSON file."""
        test_data = {
            "evidence": {
                "test_coverage": 87.5,
                "performance_impact": "neutral"
            }
        }

        json_file = tmp_path / "evidence.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)

        result = _load_json_file(str(json_file))

        assert result == test_data

    def test_load_json_file_nonexistent(self, tmp_path):
        """Test _load_json_file with nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.json"

        result = _load_json_file(str(nonexistent))

        assert result is None

    def test_evaluate_caws_compliance_evidence_file(self, tmp_path):
        """Test evaluate_caws_compliance with evidence file."""
        # Create evidence file
        evidence_data = {
            "test_results": {"coverage": 90.0},
            "benchmarks": {"improvement": 15.0}
        }

        evidence_file = tmp_path / "evidence.json"
        with open(evidence_file, "w") as f:
            json.dump(evidence_data, f)

        result = evaluate_caws_compliance(
            change_id="EVIDENCE-TEST",
            evidence=str(evidence_file),
            rationale="Feature with evidence",
            test_results={"passed": 48, "failed": 2},
            lint_results={"errors": 0, "warnings": 1},
            coverage_results={"line_percent": 85.0}
        )

        assert result["change_id"] == "EVIDENCE-TEST"
        # Should pass with good evidence and results

    def test_budget_adherence_complex_diff(self):
        """Test budget adherence with complex diff containing multiple hunks."""
        # Create a complex diff with multiple files and hunks
        complex_diff = """diff --git a/src/main.py b/src/main.py
index 1234567..abcdef0 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,9 @@ def main():
     # Old code
-    result = process_data(data)
+    # New improved code
+    result = process_data(data)
+    log_result(result)
+
     return result
"""

        result = validate_budget_adherence(
            complex_diff, max_loc=50, max_files=3)

        # Should detect the complex changes (1 unique file, removes a/b prefixes)
        assert result["files_changed_count"] == 1
        assert result["lines_added"] > 10
        assert result["total_loc"] > 0

        # Should be within budget for this test (50 LOC, 3 files limit)
        assert result["within_budget"] is True

    def test_provenance_clarity_edge_cases(self):
        """Test validate_provenance_clarity with edge cases."""
        # Test with None rationale
        result = validate_provenance_clarity(None, {}, False)
        assert result["overall_clarity"] is False
        assert result["rationale_score"] == 0.0

        # Test with empty rationale
        result = validate_provenance_clarity("", {}, True)
        assert result["overall_clarity"] is False

        # Test with very long rationale (should still work)
        long_rationale = "This is a very detailed explanation. " * 50
        result = validate_provenance_clarity(
            long_rationale, {"evidence": "present"}, True)
        # Should score well due to length and evidence
        assert result["rationale_score"] > 0.5
