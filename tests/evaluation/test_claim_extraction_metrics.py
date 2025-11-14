"""
Tests for evaluation/claim_extraction_metrics.py - Claim extraction evaluation metrics.

Tests claim extraction evaluation, student-teacher comparison metrics,
and claim extraction quality assessment.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import pytest

from evaluation.claim_extraction_metrics import (
    ClaimExtractionEvalResult,
    ClaimExtractionEvaluator,
)


class TestClaimExtractionEvalResult:
    """Test ClaimExtractionEvalResult dataclass."""

    def test_claim_extraction_eval_result_creation(self):
        """Test creating ClaimExtractionEvalResult instance."""
        result = ClaimExtractionEvalResult(
            student_claim_count=15,
            teacher_claim_count=12,
            student_success_rate=0.8,
            teacher_success_rate=0.9,
            claim_ratio=1.25,
            success_rate_ratio=0.89,
            claim_extraction_loss=0.15,
        )

        assert result.student_claim_count == 15
        assert result.teacher_claim_count == 12
        assert result.student_success_rate == 0.8
        assert result.teacher_success_rate == 0.9
        assert result.claim_ratio == 1.25
        assert result.success_rate_ratio == 0.89
        assert result.claim_extraction_loss == 0.15

    def test_claim_extraction_eval_result_default_values(self):
        """Test ClaimExtractionEvalResult with default values."""
        result = ClaimExtractionEvalResult(
            student_claim_count=10,
            teacher_claim_count=10,
            student_success_rate=0.5,
            teacher_success_rate=0.5,
            claim_ratio=1.0,
            success_rate_ratio=1.0,
            claim_extraction_loss=0.0,
        )

        assert result.claim_ratio == 1.0  # Student/Teacher ratio
        assert result.success_rate_ratio == 1.0
        assert result.claim_extraction_loss == 0.0

    def test_claim_extraction_eval_result_calculated_fields(self):
        """Test that calculated fields are properly computed."""
        # Test claim ratio calculation (student/teacher)
        result = ClaimExtractionEvalResult(
            student_claim_count=20,
            teacher_claim_count=10,
            student_success_rate=0.8,
            teacher_success_rate=0.9,
            claim_ratio=2.0,  # 20/10
            success_rate_ratio=0.89,  # 0.8/0.9
            claim_extraction_loss=0.11,  # |2.0 - 0.89| or some loss metric
        )

        assert result.claim_ratio == 2.0
        assert result.success_rate_ratio == 0.89
        assert result.claim_extraction_loss == 0.11


class TestClaimExtractionEvaluator:
    """Test ClaimExtractionEvaluator class."""

    @pytest.fixture
    def mock_extractor(self):
        """Create mock claim extractor."""
        extractor = Mock()
        extractor.extract_claims = Mock(
            return_value=[
                {"text": "claim1", "confidence": 0.9},
                {"text": "claim2", "confidence": 0.8},
            ]
        )
        return extractor

    @pytest.fixture
    def evaluator(self, mock_extractor):
        """Create ClaimExtractionEvaluator instance."""
        return ClaimExtractionEvaluator(mock_extractor)

    @pytest.fixture
    def default_evaluator(self):
        """Create evaluator with default extractor."""
        with patch("evaluation.claim_extraction_metrics.SimpleClaimExtractor") as mock_class:
            mock_instance = Mock()
            mock_instance.extract_claims = Mock(
                return_value=[
                    {"text": "default_claim1", "confidence": 0.85},
                    {"text": "default_claim2", "confidence": 0.75},
                ]
            )
            mock_class.return_value = mock_instance

            evaluator = ClaimExtractionEvaluator()
            return evaluator

    def test_evaluator_initialization_with_extractor(self, mock_extractor):
        """Test evaluator initialization with custom extractor."""
        evaluator = ClaimExtractionEvaluator(mock_extractor)

        assert evaluator.extractor == mock_extractor

    def test_evaluator_initialization_default(self):
        """Test evaluator initialization with default extractor."""
        with patch("evaluation.claim_extraction_metrics.SimpleClaimExtractor") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            evaluator = ClaimExtractionEvaluator()

            mock_class.assert_called_once()
            assert evaluator.extractor == mock_instance

    @patch("evaluation.claim_extraction_metrics.compute_claim_extraction_metrics")
    def test_evaluate_success(self, mock_compute_metrics, evaluator):
        """Test successful evaluation."""
        student_outputs = [
            "The sky is blue because of light scattering.",
            "Water boils at 100 degrees Celsius at sea level.",
        ]
        teacher_outputs = [
            "Rayleigh scattering makes the sky appear blue.",
            "The boiling point of water is 100°C at 1 atm pressure.",
        ]

        # Mock the metric computation
        mock_compute_metrics.return_value = {
            "student_claim_count": 4,
            "teacher_claim_count": 3,
            "student_success_rate": 0.75,
            "teacher_success_rate": 0.85,
            "claim_ratio": 1.33,
            "success_rate_ratio": 0.88,
            "claim_extraction_loss": 0.45,
        }

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert isinstance(result, ClaimExtractionEvalResult)
        assert result.student_claim_count == 4
        assert result.teacher_claim_count == 3
        assert result.student_success_rate == pytest.approx(0.75, abs=0.01)
        assert result.teacher_success_rate == pytest.approx(0.85, abs=0.01)
        assert result.claim_ratio == pytest.approx(1.33, abs=0.01)
        assert result.success_rate_ratio == pytest.approx(0.88, abs=0.01)
        assert result.claim_extraction_loss == pytest.approx(0.45, abs=0.01)

        # Verify compute_claim_extraction_metrics was called
        mock_compute_metrics.assert_called_once_with(
            student_outputs, teacher_outputs, evaluator.extractor
        )

    def test_evaluate_empty_outputs(self, evaluator):
        """Test evaluation with empty output lists."""
        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "student_claim_count": 0,
                "teacher_claim_count": 0,
                "student_success_rate": 0.0,
                "teacher_success_rate": 0.0,
                "claim_ratio": 0.0,
                "success_rate_ratio": 0.0,
                "claim_extraction_loss": 0.0,
            }

            result = evaluator.evaluate([], [])

            assert result.student_claim_count == 0
            assert result.teacher_claim_count == 0
            assert result.claim_ratio == 0.0

    def test_evaluate_single_output(self, evaluator):
        """Test evaluation with single outputs."""
        student_outputs = ["Single student claim about science."]
        teacher_outputs = ["Single teacher explanation of physics."]

        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "student_claim_count": 1,
                "teacher_claim_count": 2,
                "student_success_rate": 1.0,
                "teacher_success_rate": 0.5,
                "claim_ratio": 0.5,
                "success_rate_ratio": 2.0,
                "claim_extraction_loss": 1.5,
            }

            result = evaluator.evaluate(student_outputs, teacher_outputs)

            assert result.student_claim_count == 1
            assert result.teacher_claim_count == 2
            assert result.claim_ratio == 0.5

    def test_evaluate_mismatched_lengths(self, evaluator):
        """Test evaluation with mismatched student/teacher output lengths."""
        student_outputs = ["Student output 1", "Student output 2", "Student output 3"]
        teacher_outputs = ["Teacher output 1", "Teacher output 2"]  # Different length

        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "student_claim_count": 5,
                "teacher_claim_count": 3,
                "student_success_rate": 0.6,
                "teacher_success_rate": 0.8,
                "claim_ratio": 1.67,
                "success_rate_ratio": 0.75,
                "claim_extraction_loss": 0.92,
            }

            result = evaluator.evaluate(student_outputs, teacher_outputs)

            # Should still work despite length mismatch
            assert result.student_claim_count == 5
            assert result.teacher_claim_count == 3
            # Use approximate comparison for floating point values
            assert result.claim_ratio == pytest.approx(1.67, abs=0.01)
            assert result.success_rate_ratio == pytest.approx(0.75, abs=0.01)
            assert result.claim_extraction_loss == pytest.approx(0.92, abs=0.01)

    def test_evaluate_with_none_outputs(self, evaluator):
        """Test evaluation handling None values in outputs."""
        student_outputs = ["Valid output", None, "Another valid output"]
        teacher_outputs = ["Teacher output 1", "Teacher output 2", None]

        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "student_claim_count": 3,
                "teacher_claim_count": 2,
                "student_success_rate": 0.7,
                "teacher_success_rate": 0.9,
                "claim_ratio": 1.5,
                "success_rate_ratio": 0.78,
                "claim_extraction_loss": 0.72,
            }

            result = evaluator.evaluate(student_outputs, teacher_outputs)

            # Should handle None values gracefully
            assert isinstance(result, ClaimExtractionEvalResult)

    @patch("evaluation.claim_extraction_metrics.compute_claim_extraction_metrics")
    def test_evaluate_extractor_failure(self, mock_compute, evaluator):
        """Test evaluation when claim extraction fails."""
        mock_compute.side_effect = Exception("Extraction failed")

        student_outputs = ["Test output"]
        teacher_outputs = ["Reference output"]

        with pytest.raises(Exception):
            evaluator.evaluate(student_outputs, teacher_outputs)


class TestClaimExtractionMetricsIntegration:
    """Test integration of claim extraction metrics."""

    def test_complete_evaluation_workflow(self):
        """Test complete claim extraction evaluation workflow."""
        with (
            patch(
                "evaluation.claim_extraction_metrics.SimpleClaimExtractor"
            ) as mock_extractor_class,
            patch(
                "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
            ) as mock_compute,
        ):
            # Mock extractor
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor

            # Mock computation results
            mock_compute.return_value = {
                "student_claim_count": 8,
                "teacher_claim_count": 6,
                "student_success_rate": 0.75,
                "teacher_success_rate": 0.83,
                "claim_ratio": 1.33,
                "success_rate_ratio": 0.90,
                "claim_extraction_loss": 0.43,
            }

            # Create evaluator and run evaluation
            evaluator = ClaimExtractionEvaluator()

            student_outputs = [
                "Photosynthesis converts light energy to chemical energy in plants.",
                "The Earth orbits the Sun once every 365.25 days.",
                "Water molecules consist of two hydrogen atoms and one oxygen atom.",
            ]

            teacher_outputs = [
                "Plants use photosynthesis to convert light energy into chemical energy stored in glucose.",
                "Earth's orbital period around the Sun is approximately 365.25 days.",
                "A water molecule (H2O) contains two hydrogen atoms bonded to one oxygen atom.",
            ]

            result = evaluator.evaluate(student_outputs, teacher_outputs)

            # Verify results
            assert result.student_claim_count == 8
            assert result.teacher_claim_count == 6
            assert result.student_success_rate == 0.75
            assert result.teacher_success_rate == 0.83
            assert result.claim_ratio == 1.33
            assert result.success_rate_ratio == 0.90
            assert result.claim_extraction_loss == 0.43

    def test_evaluation_with_realistic_outputs(self):
        """Test evaluation with realistic student/teacher output pairs."""
        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.return_value = {
                "student_claim_count": 5,
                "teacher_claim_count": 4,
                "student_success_rate": 0.8,
                "teacher_success_rate": 0.9,
                "claim_ratio": 1.25,
                "success_rate_ratio": 0.89,
                "claim_extraction_loss": 0.36,
            }

            evaluator = ClaimExtractionEvaluator()

            # Realistic educational content
            student_outputs = [
                "Gravity is a force that pulls objects toward Earth.",
                "The mitochondria is the powerhouse of the cell.",
                "Light travels faster than sound.",
            ]

            teacher_outputs = [
                "Gravity is the force that attracts objects with mass toward the center of Earth.",
                "Mitochondria are often called the powerhouse of the cell because they generate ATP.",
                "Electromagnetic waves, including light, travel at 299,792,458 m/s, while sound waves travel at approximately 343 m/s at sea level.",
            ]

            result = evaluator.evaluate(student_outputs, teacher_outputs)

            # Verify the evaluation completed
            assert isinstance(result, ClaimExtractionEvalResult)
            assert (
                result.student_claim_count >= result.teacher_claim_count
            )  # Student might extract more claims
            assert 0.0 <= result.student_success_rate <= 1.0
            assert 0.0 <= result.teacher_success_rate <= 1.0
            assert result.claim_ratio > 0.0
            assert result.success_rate_ratio > 0.0

    def test_metrics_calculation_edge_cases(self):
        """Test metrics calculation with edge cases."""
        test_cases = [
            # Empty results
            {
                "student_claim_count": 0,
                "teacher_claim_count": 0,
                "student_success_rate": 0.0,
                "teacher_success_rate": 0.0,
                "claim_ratio": 0.0,
                "success_rate_ratio": 0.0,
                "claim_extraction_loss": 0.0,
            },
            # Perfect student performance
            {
                "student_claim_count": 10,
                "teacher_claim_count": 10,
                "student_success_rate": 1.0,
                "teacher_success_rate": 1.0,
                "claim_ratio": 1.0,
                "success_rate_ratio": 1.0,
                "claim_extraction_loss": 0.0,
            },
            # Student underperforming
            {
                "student_claim_count": 3,
                "teacher_claim_count": 8,
                "student_success_rate": 0.5,
                "teacher_success_rate": 0.9,
                "claim_ratio": 0.375,
                "success_rate_ratio": 0.556,
                "claim_extraction_loss": 0.181,
            },
        ]

        evaluator = ClaimExtractionEvaluator()

        for i, expected_metrics in enumerate(test_cases):
            with patch(
                "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
            ) as mock_compute:
                mock_compute.return_value = expected_metrics

                result = evaluator.evaluate([f"Test output {i}"], [f"Reference {i}"])

                assert result.student_claim_count == expected_metrics["student_claim_count"]
                assert result.teacher_claim_count == expected_metrics["teacher_claim_count"]
                assert result.student_success_rate == expected_metrics["student_success_rate"]
                assert result.teacher_success_rate == expected_metrics["teacher_success_rate"]
                assert abs(result.claim_ratio - expected_metrics["claim_ratio"]) < 0.001
                assert (
                    abs(result.success_rate_ratio - expected_metrics["success_rate_ratio"]) < 0.001
                )
                assert (
                    abs(result.claim_extraction_loss - expected_metrics["claim_extraction_loss"])
                    < 0.001
                )

    def test_evaluator_reuse(self):
        """Test that evaluator can be reused for multiple evaluations."""
        with patch(
            "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
        ) as mock_compute:
            mock_compute.side_effect = [
                # First evaluation
                {
                    "student_claim_count": 5,
                    "teacher_claim_count": 4,
                    "student_success_rate": 0.8,
                    "teacher_success_rate": 0.9,
                    "claim_ratio": 1.25,
                    "success_rate_ratio": 0.89,
                    "claim_extraction_loss": 0.36,
                },
                # Second evaluation
                {
                    "student_claim_count": 7,
                    "teacher_claim_count": 5,
                    "student_success_rate": 0.75,
                    "teacher_success_rate": 0.85,
                    "claim_ratio": 1.4,
                    "success_rate_ratio": 0.88,
                    "claim_extraction_loss": 0.52,
                },
            ]

            evaluator = ClaimExtractionEvaluator()

            # First evaluation
            result1 = evaluator.evaluate(["First test"], ["First reference"])
            assert result1.student_claim_count == 5

            # Second evaluation
            result2 = evaluator.evaluate(["Second test"], ["Second reference"])
            assert result2.student_claim_count == 7

            # Verify both calls were made
            assert mock_compute.call_count == 2

    def test_evaluator_with_different_extractors(self):
        """Test evaluator with different claim extractors."""
        # Test with different mock extractors
        extractors = [
            Mock(extract_claims=lambda x: [{"text": "claim1", "confidence": 0.9}]),
            Mock(
                extract_claims=lambda x: [
                    {"text": "claim2", "confidence": 0.8},
                    {"text": "claim3", "confidence": 0.7},
                ]
            ),
            Mock(extract_claims=lambda x: []),  # No claims extracted
        ]

        for extractor in extractors:
            evaluator = ClaimExtractionEvaluator(extractor)

            with patch(
                "evaluation.claim_extraction_metrics.compute_claim_extraction_metrics"
            ) as mock_compute:
                mock_compute.return_value = {
                    "student_claim_count": len(extractor.extract_claims("dummy")),
                    "teacher_claim_count": 2,
                    "student_success_rate": 0.5,
                    "teacher_success_rate": 0.7,
                    "claim_ratio": 0.5,
                    "success_rate_ratio": 0.71,
                    "claim_extraction_loss": 0.21,
                }

                result = evaluator.evaluate(["test"], ["reference"])

                assert isinstance(result, ClaimExtractionEvalResult)
                assert result.student_claim_count == len(extractor.extract_claims("dummy"))
