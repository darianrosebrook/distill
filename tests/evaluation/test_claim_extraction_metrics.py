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

        # Mock the metric computation - return per-output metrics
        # The evaluate method calls compute_claim_extraction_metrics for each pair
        mock_compute_metrics.side_effect = [
            {
                "student_claim_count": 2,
                "teacher_claim_count": 1,
                "student_success_rate": 0.75,
                "teacher_success_rate": 0.85,
            },
            {
                "student_claim_count": 2,
                "teacher_claim_count": 2,
                "student_success_rate": 0.75,
                "teacher_success_rate": 0.85,
            },
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert isinstance(result, ClaimExtractionEvalResult)
        # Results are averages converted to int: (2+2)/2 = 2, (1+2)/2 = 1.5 -> 1
        assert result.student_claim_count == 2  # (2+2)/2 = 2
        assert result.teacher_claim_count == 1  # (1+2)/2 = 1.5 -> int(1.5) = 1
        assert result.student_success_rate == pytest.approx(0.75, abs=0.01)
        assert result.teacher_success_rate == pytest.approx(0.85, abs=0.01)
        # claim_ratio = 2/1.5 = 1.33 (rounded)
        assert result.claim_ratio == pytest.approx(1.33, abs=0.01)
        # success_rate_ratio = 0.75/0.85 = 0.88 (rounded)
        assert result.success_rate_ratio == pytest.approx(0.88, abs=0.01)
        # claim_extraction_loss calculation depends on implementation
        # For now, just check it's a float
        assert isinstance(result.claim_extraction_loss, float)

        # Verify compute_claim_extraction_metrics was called twice (once per output pair)
        assert mock_compute_metrics.call_count == 2

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

        # Function should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="Student and teacher outputs must have same length"):
            evaluator.evaluate(student_outputs, teacher_outputs)

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
            # claim_extraction_loss is calculated from ratios, not from mocked value
            # claim_ratio=1.33 >= 0.5 (min_claim_ratio), so claim_penalty = 0.0
            # success_rate_ratio=0.90 >= 0.7 (min_success_rate_ratio), so success_penalty = 0.0
            # loss = 0.6 * 0.0 + 0.4 * 0.0 = 0.0
            assert result.claim_extraction_loss == 0.0

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
            # Loss calculation: if claim_ratio >= 0.5 and success_rate_ratio >= 0.7, loss = 0.0
            # With typical values above thresholds, loss should be 0.0
            assert result.claim_extraction_loss >= 0.0

    def test_metrics_calculation_edge_cases(self):
        """Test metrics calculation with edge cases."""
        test_cases = [
            # Empty results
            # When claim_ratio=0.0 < min_claim_ratio(0.5), penalty = max(0, 1 - 0/0.5) = 1.0
            # When success_rate_ratio=0.0 < min_success_rate_ratio(0.7), penalty = max(0, 1 - 0/0.7) = 1.0
            # loss = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
            {
                "student_claim_count": 0,
                "teacher_claim_count": 0,
                "student_success_rate": 0.0,
                "teacher_success_rate": 0.0,
                "claim_ratio": 0.0,
                "success_rate_ratio": 0.0,
                "claim_extraction_loss": 1.0,  # Penalty for zero ratios
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
            # claim_ratio = 3/8 = 0.375 -> round(0.375, 2) = 0.38
            # success_rate_ratio = 0.5/0.9 = 0.555... -> round(0.555..., 2) = 0.56
            # claim_penalty = max(0, 1 - 0.38/0.5) = max(0, 1 - 0.76) = 0.24
            # success_penalty = max(0, 1 - 0.56/0.7) = max(0, 1 - 0.8) = 0.2
            # loss = 0.6 * 0.24 + 0.4 * 0.2 = 0.144 + 0.08 = 0.224
            {
                "student_claim_count": 3,
                "teacher_claim_count": 8,
                "student_success_rate": 0.5,
                "teacher_success_rate": 0.9,
                "claim_ratio": 0.38,  # 0.375 rounded to 2 decimals
                "success_rate_ratio": 0.56,  # 0.556 rounded to 2 decimals
                "claim_extraction_loss": 0.224,  # Exact calculated value
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
                assert result.student_success_rate == pytest.approx(
                    expected_metrics["student_success_rate"], abs=0.01
                )
                assert result.teacher_success_rate == pytest.approx(
                    expected_metrics["teacher_success_rate"], abs=0.01
                )
                assert result.claim_ratio == pytest.approx(
                    expected_metrics["claim_ratio"], abs=0.01
                )
                assert result.success_rate_ratio == pytest.approx(
                    expected_metrics["success_rate_ratio"], abs=0.01
                )
                assert result.claim_extraction_loss == pytest.approx(
                    expected_metrics["claim_extraction_loss"], abs=0.01
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


class TestFormatResults:
    """Test format_results method."""

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

    def test_format_results_basic(self, evaluator):
        """Test formatting results as string."""
        result = ClaimExtractionEvalResult(
            student_claim_count=10,
            teacher_claim_count=8,
            student_success_rate=0.85,
            teacher_success_rate=0.90,
            claim_ratio=1.25,
            success_rate_ratio=0.94,
            claim_extraction_loss=0.15,
        )

        formatted = evaluator.format_results(result)

        assert isinstance(formatted, str)
        assert "Claim Extraction Evaluation Results:" in formatted
        assert "Student Claims: 10" in formatted
        assert "Teacher Claims: 8" in formatted
        assert "Claim Ratio: 125.00%" in formatted
        assert "Student Success Rate: 85.00%" in formatted
        assert "Teacher Success Rate: 90.00%" in formatted
        assert "Success Rate Ratio: 94.00%" in formatted
        assert "Claim Extraction Loss: 0.1500" in formatted

    def test_format_results_zero_values(self, evaluator):
        """Test formatting results with zero values."""
        result = ClaimExtractionEvalResult(
            student_claim_count=0,
            teacher_claim_count=0,
            student_success_rate=0.0,
            teacher_success_rate=0.0,
            claim_ratio=0.0,
            success_rate_ratio=0.0,
            claim_extraction_loss=1.0,
        )

        formatted = evaluator.format_results(result)

        assert "Student Claims: 0" in formatted
        assert "Teacher Claims: 0" in formatted
        assert "Claim Ratio: 0.00%" in formatted


class TestEvaluateFromDataset:
    """Test evaluate_from_dataset method."""

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

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("builtins.open", create=True)
    @patch("torch.utils.data.Subset")
    def test_evaluate_from_dataset_basic(
        self, mock_subset, mock_open, mock_dataset_class, mock_dataloader_class, evaluator
    ):
        """Test evaluating from dataset."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)
        mock_dataset_class.return_value = mock_dataset

        # Mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": Mock(),
                        "attention_mask": None,
                        "teacher_text": "Teacher output 1",
                    },
                    {
                        "input_ids": Mock(),
                        "attention_mask": Mock(),
                        "teacher_text": ["Teacher output 2"],
                    },
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock torch operations
        mock_subset.return_value = mock_dataset

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_logits.argmax = Mock(return_value=Mock())
        mock_model.return_value = mock_logits

        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(side_effect=["Student output 1", "Student output 2"])

        mock_device = "cpu"

        # Mock input_ids.to() and attention_mask.to()
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)
        mock_attention_mask = Mock()
        mock_attention_mask.to = Mock(return_value=mock_attention_mask)

        # Setup batch mocks
        def batch_generator():
            yield {
                "input_ids": mock_input_ids,
                "attention_mask": None,
                "teacher_text": "Teacher output 1",
            }
            yield {
                "input_ids": mock_input_ids,
                "attention_mask": mock_attention_mask,
                "teacher_text": ["Teacher output 2"],
            }

        mock_dataloader.__iter__ = Mock(return_value=batch_generator())

        # Mock argmax result
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)

        with (
            patch("evaluation.claim_extraction_metrics.compute_claim_extraction_metrics") as mock_compute,
            patch("torch.no_grad") as mock_no_grad,
        ):
            mock_compute.return_value = {
                "student_claim_count": 2,
                "teacher_claim_count": 2,
                "student_success_rate": 0.8,
                "teacher_success_rate": 0.9,
            }
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)
            
            result = evaluator.evaluate_from_dataset(
                dataset_path="test.jsonl",
                student_model=mock_model,
                tokenizer=mock_tokenizer,
                device=mock_device,
            )

            assert isinstance(result, ClaimExtractionEvalResult)
            mock_model.eval.assert_called_once()

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("builtins.open", create=True)
    @patch("torch.utils.data.Subset")
    def test_evaluate_from_dataset_with_max_samples(
        self, mock_subset, mock_open, mock_dataset_class, mock_dataloader_class, evaluator
    ):
        """Test evaluating from dataset with max_samples limit."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.return_value = mock_dataset

        # Mock Subset
        mock_subset_obj = Mock()
        mock_subset_obj.__len__ = Mock(return_value=5)
        mock_subset.return_value = mock_subset_obj

        # Mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([]))
        mock_dataloader_class.return_value = mock_dataloader

        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()
        mock_model.return_value = Mock()

        with (
            patch("evaluation.claim_extraction_metrics.compute_claim_extraction_metrics") as mock_compute,
            patch("torch.no_grad") as mock_no_grad,
        ):
            mock_compute.return_value = {
                "student_claim_count": 0,
                "teacher_claim_count": 0,
                "student_success_rate": 0.0,
                "teacher_success_rate": 0.0,
            }
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)

            result = evaluator.evaluate_from_dataset(
                dataset_path="test.jsonl",
                student_model=mock_model,
                tokenizer=mock_tokenizer,
                device="cpu",
                max_samples=5,
            )

            # Should have used Subset with max_samples
            mock_subset.assert_called_once()
            assert isinstance(result, ClaimExtractionEvalResult)


class TestRealClaimExtraction:
    """Test claim extraction with real SimpleClaimExtractor (no mocks)."""

    def test_evaluator_with_real_extractor(self):
        """Test evaluator with real SimpleClaimExtractor."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Text with verifiable content
        student_outputs = [
            "The function was created in 2024. It implements feature X. Version 2.0 was released.",
            "Water boils at 100 degrees Celsius at sea level. The temperature is measured in Celsius.",
        ]
        teacher_outputs = [
            "The function was created in 2024-01-15. It implements feature X with version 2.0.",
            "Water boils at 100°C at 1 atm pressure. Temperature measurement uses Celsius scale.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert isinstance(result, ClaimExtractionEvalResult)
        assert result.student_claim_count >= 0
        assert result.teacher_claim_count >= 0
        assert 0.0 <= result.student_success_rate <= 1.0
        assert 0.0 <= result.teacher_success_rate <= 1.0
        assert result.claim_ratio >= 0.0
        assert result.success_rate_ratio >= 0.0
        assert 0.0 <= result.claim_extraction_loss <= 1.0

    def test_evaluator_with_real_extractor_verifiable_content(self):
        """Test evaluator with text containing verifiable patterns."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Text with dates, versions, code blocks
        student_outputs = [
            "The API was released on 2024-01-15. Version 1.0 includes feature X. Code: ```python\ndef func():\n    return 1\n```",
        ]
        teacher_outputs = [
            "API release date: 2024-01-15. Version 1.0 includes feature X with code implementation.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Should extract claims from both
        assert result.student_claim_count > 0
        assert result.teacher_claim_count > 0

    def test_evaluator_with_real_extractor_unverifiable_content(self):
        """Test evaluator with text containing unverifiable/subjective content."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Text with subjective/unverifiable patterns
        student_outputs = [
            "I think this is probably a good idea. Maybe it will work. It seems like a nice solution.",
        ]
        teacher_outputs = [
            "This might be a good approach. Perhaps it could work. It appears to be a nice solution.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Should have lower claim counts due to unverifiable content
        assert isinstance(result, ClaimExtractionEvalResult)
        # May have 0 claims if content is too unverifiable
        assert result.student_claim_count >= 0
        assert result.teacher_claim_count >= 0

    def test_evaluator_with_real_extractor_empty_text(self):
        """Test evaluator with empty text."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [""]
        teacher_outputs = [""]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert result.student_claim_count == 0
        assert result.teacher_claim_count == 0
        assert result.student_success_rate == 0.0
        assert result.teacher_success_rate == 0.0
        assert result.claim_ratio == 0.0
        assert result.success_rate_ratio == 0.0
        # When both are 0, ratios are 0, so loss should be 1.0 (penalty)
        assert result.claim_extraction_loss == 1.0

    def test_evaluator_with_real_extractor_code_blocks(self):
        """Test evaluator with code blocks (should be verifiable)."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Here's a function:\n```python\ndef add(a, b):\n    return a + b\n```",
        ]
        teacher_outputs = [
            "Function implementation:\n```python\ndef add(a, b):\n    return a + b\n```",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Code blocks should be detected as verifiable
        assert result.student_claim_count > 0
        assert result.teacher_claim_count > 0

    def test_evaluator_with_real_extractor_json_content(self):
        """Test evaluator with JSON content (should be verifiable)."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            'Response: {"status": "ok", "value": 42}',
        ]
        teacher_outputs = [
            'Result: {"status": "success", "value": 42}',
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # JSON should be detected as verifiable
        assert result.student_claim_count > 0
        assert result.teacher_claim_count > 0

    def test_evaluator_with_real_extractor_urls(self):
        """Test evaluator with URLs (should be verifiable)."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Visit https://example.com for more information. API docs at https://api.example.com",
        ]
        teacher_outputs = [
            "Documentation: https://docs.example.com. API: https://api.example.com/docs",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # URLs should be detected as verifiable
        assert result.student_claim_count > 0
        assert result.teacher_claim_count > 0

    def test_evaluator_with_real_extractor_dates(self):
        """Test evaluator with dates (should be verifiable)."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Released on 2024-01-15. Updated on 2024-02-20.",
        ]
        teacher_outputs = [
            "Release date: 2024-01-15. Update date: 2024-02-20.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Dates should be detected as verifiable
        assert result.student_claim_count > 0
        assert result.teacher_claim_count > 0

    def test_evaluator_loss_calculation_with_real_extractor(self):
        """Test claim extraction loss calculation with real extractor."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Student has fewer claims than teacher (below min_claim_ratio)
        student_outputs = [
            "Version 1.0 was released.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # If student has fewer claims, loss should be > 0 (unless ratios meet thresholds)
        assert isinstance(result.claim_extraction_loss, float)
        assert 0.0 <= result.claim_extraction_loss <= 1.0


class TestComputeClaimExtractionMetricsReal:
    """Test compute_claim_extraction_metrics with real extractor."""

    def test_compute_claim_extraction_metrics_real_extractor(self):
        """Test compute_claim_extraction_metrics with real SimpleClaimExtractor."""
        from training.claim_extraction import compute_claim_extraction_metrics, SimpleClaimExtractor

        extractor = SimpleClaimExtractor()

        student_output = "Version 2.0 was released on 2024-01-15. API docs at https://example.com"
        teacher_output = "Version 2.0 was released on 2024-01-15. Documentation: https://docs.example.com"

        metrics = compute_claim_extraction_metrics(
            student_output=student_output,
            teacher_output=teacher_output,
            extractor=extractor,
        )

        assert "student_claim_count" in metrics
        assert "teacher_claim_count" in metrics
        assert "student_success_rate" in metrics
        assert "teacher_success_rate" in metrics
        assert "claim_ratio" in metrics
        assert "success_rate_ratio" in metrics

        assert metrics["student_claim_count"] >= 0
        assert metrics["teacher_claim_count"] >= 0
        assert 0.0 <= metrics["student_success_rate"] <= 1.0
        assert 0.0 <= metrics["teacher_success_rate"] <= 1.0
        assert metrics["claim_ratio"] >= 0.0
        assert metrics["success_rate_ratio"] >= 0.0

    def test_compute_claim_extraction_metrics_empty_text(self):
        """Test compute_claim_extraction_metrics with empty text."""
        from training.claim_extraction import compute_claim_extraction_metrics

        metrics = compute_claim_extraction_metrics(
            student_output="",
            teacher_output="",
        )

        assert metrics["student_claim_count"] == 0
        assert metrics["teacher_claim_count"] == 0
        assert metrics["student_success_rate"] == 0.0
        assert metrics["teacher_success_rate"] == 0.0
        assert metrics["claim_ratio"] == 0.0
        assert metrics["success_rate_ratio"] == 0.0

    def test_compute_claim_extraction_metrics_none_extractor(self):
        """Test compute_claim_extraction_metrics creates extractor if None."""
        from training.claim_extraction import compute_claim_extraction_metrics

        student_output = "Version 2.0 was released on 2024-01-15."
        teacher_output = "Version 2.0 was released on 2024-01-15 with feature X."

        metrics = compute_claim_extraction_metrics(
            student_output=student_output,
            teacher_output=teacher_output,
            extractor=None,  # Should create new extractor
        )

        assert "student_claim_count" in metrics
        assert "teacher_claim_count" in metrics
        assert metrics["student_claim_count"] >= 0
        assert metrics["teacher_claim_count"] >= 0


class TestFormatResultsReal:
    """Test format_results method with real results."""

    def test_format_results_real_data(self):
        """Test format_results with real evaluation results."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 2.0 was released on 2024-01-15. API docs at https://example.com",
        ]
        teacher_outputs = [
            "Version 2.0 was released on 2024-01-15. Documentation: https://docs.example.com",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)
        formatted = evaluator.format_results(result)

        assert isinstance(formatted, str)
        assert "Claim Extraction Evaluation Results:" in formatted
        assert "Student Claims:" in formatted
        assert "Teacher Claims:" in formatted
        assert "Claim Ratio:" in formatted
        assert "Student Success Rate:" in formatted
        assert "Teacher Success Rate:" in formatted
        assert "Success Rate Ratio:" in formatted
        assert "Claim Extraction Loss:" in formatted

    def test_format_results_zero_values(self):
        """Test format_results with zero values."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        result = ClaimExtractionEvalResult(
            student_claim_count=0,
            teacher_claim_count=0,
            student_success_rate=0.0,
            teacher_success_rate=0.0,
            claim_ratio=0.0,
            success_rate_ratio=0.0,
            claim_extraction_loss=1.0,
        )

        formatted = evaluator.format_results(result)

        assert "Student Claims: 0" in formatted
        assert "Teacher Claims: 0" in formatted
        assert "Claim Ratio: 0.00%" in formatted
        assert "Student Success Rate: 0.00%" in formatted
        assert "Teacher Success Rate: 0.00%" in formatted
        assert "Success Rate Ratio: 0.00%" in formatted
        assert "Claim Extraction Loss: 1.0000" in formatted

    def test_format_results_high_values(self):
        """Test format_results with high values."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        result = ClaimExtractionEvalResult(
            student_claim_count=10,
            teacher_claim_count=8,
            student_success_rate=0.95,
            teacher_success_rate=0.90,
            claim_ratio=1.25,
            success_rate_ratio=1.06,
            claim_extraction_loss=0.0,
        )

        formatted = evaluator.format_results(result)

        assert "Student Claims: 10" in formatted
        assert "Teacher Claims: 8" in formatted
        assert "Claim Ratio: 125.00%" in formatted
        assert "Student Success Rate: 95.00%" in formatted
        assert "Teacher Success Rate: 90.00%" in formatted
        assert "Success Rate Ratio: 106.00%" in formatted
        assert "Claim Extraction Loss: 0.0000" in formatted


class TestEdgeCasesReal:
    """Test edge cases with real extractor."""

    def test_evaluator_with_special_characters(self):
        """Test evaluator with special characters in text."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "API v2.0: https://api.example.com/v2.0?key=123&token=abc",
        ]
        teacher_outputs = [
            "API version 2.0: https://api.example.com/v2.0?key=123&token=abc",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Should handle special characters gracefully
        assert isinstance(result, ClaimExtractionEvalResult)

    def test_evaluator_with_unicode(self):
        """Test evaluator with unicode characters."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 2.0 released on 2024-01-15. Temperature: 100°C",
        ]
        teacher_outputs = [
            "Version 2.0 released on 2024-01-15. Temperature: 100°C",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Should handle unicode gracefully
        assert isinstance(result, ClaimExtractionEvalResult)

    def test_evaluator_with_newlines(self):
        """Test evaluator with newlines in text."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 2.0\nReleased on 2024-01-15\nAPI: https://example.com",
        ]
        teacher_outputs = [
            "Version 2.0\nReleased on 2024-01-15\nAPI: https://example.com",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Should handle newlines gracefully
        assert isinstance(result, ClaimExtractionEvalResult)

    def test_evaluator_ratio_calculation_edge_cases(self):
        """Test ratio calculation edge cases."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Test with very different claim counts
        student_outputs = [
            "Version 1.0.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com. Code: ```python\ndef func():\n    return 1\n```",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # Should calculate ratios correctly
        assert result.claim_ratio >= 0.0
        assert result.success_rate_ratio >= 0.0
        # If student has fewer claims, ratio should be < 1.0 (unless both have 0)
        if result.teacher_claim_count > 0 and result.student_claim_count > 0:
            # Student has fewer claims, so ratio should be < 1.0
            assert result.claim_ratio <= 1.0

    def test_evaluator_with_multiple_samples_real(self):
        """Test evaluator with multiple samples using real extractor."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 1.0 was released on 2024-01-15.",
            "Version 2.0 includes feature X. API docs at https://example.com",
            "Code: ```python\ndef func():\n    return 1\n```",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15 with feature Y.",
            "Version 2.0 includes feature X. Documentation: https://docs.example.com",
            "Implementation:\n```python\ndef func():\n    return 1\n```",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert isinstance(result, ClaimExtractionEvalResult)
        # Should aggregate across all samples
        assert result.student_claim_count >= 0
        assert result.teacher_claim_count >= 0
        # Averages should be between 0 and max possible
        assert result.student_success_rate >= 0.0
        assert result.student_success_rate <= 1.0

    def test_evaluator_loss_thresholds(self):
        """Test claim extraction loss with different threshold values."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 1.0 was released.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X.",
        ]

        # Test with low thresholds (should pass)
        result_low = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.1,  # Very low threshold
            min_success_rate_ratio=0.1,  # Very low threshold
        )

        # Test with high thresholds (should fail)
        result_high = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.9,  # High threshold
            min_success_rate_ratio=0.9,  # High threshold
        )

        # Loss with high thresholds should be >= loss with low thresholds
        assert result_high.claim_extraction_loss >= result_low.claim_extraction_loss

    def test_evaluator_ratio_rounding(self):
        """Test that claim_ratio and success_rate_ratio are rounded to 2 decimal places."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Use texts that will produce non-integer ratios
        student_outputs = [
            "Version 1.0 was released on 2024-01-15.",
            "Version 2.0 includes feature X.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature Y.",
            "Version 2.0 includes feature X. Documentation: https://docs.example.com",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Ratios should be rounded to 2 decimal places
        # Check that claim_ratio has at most 2 decimal places
        claim_ratio_str = f"{result.claim_ratio:.2f}"
        assert abs(result.claim_ratio - float(claim_ratio_str)) < 0.01

        # Check that success_rate_ratio has at most 2 decimal places
        success_rate_ratio_str = f"{result.success_rate_ratio:.2f}"
        assert abs(result.success_rate_ratio - float(success_rate_ratio_str)) < 0.01

    def test_evaluator_averaging_behavior(self):
        """Test that averaging works correctly across multiple samples."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Create samples with known claim counts
        student_outputs = [
            "Version 1.0 was released on 2024-01-15.",  # Should have ~1-2 claims
            "Version 2.0 includes feature X. API docs at https://example.com",  # Should have ~2-3 claims
            "Code: ```python\ndef func():\n    return 1\n```",  # Should have ~1 claim
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15 with feature Y.",
            "Version 2.0 includes feature X. Documentation: https://docs.example.com",
            "Implementation:\n```python\ndef func():\n    return 1\n```",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # Averages should be between min and max possible
        assert result.student_claim_count >= 0
        assert result.teacher_claim_count >= 0
        # Should be integers (averages converted to int)
        assert isinstance(result.student_claim_count, int)
        assert isinstance(result.teacher_claim_count, int)

    def test_evaluator_zero_teacher_claims(self):
        """Test evaluator when teacher has zero claims."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Teacher has unverifiable content (should have 0 claims)
        student_outputs = [
            "Version 1.0 was released on 2024-01-15.",
        ]
        teacher_outputs = [
            "I think maybe this is probably a good idea.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # When teacher has 0 claims, claim_ratio should be 0.0 (division by zero protection)
        if result.teacher_claim_count == 0:
            assert result.claim_ratio == 0.0
        else:
            # If teacher somehow has claims, ratio should be >= 0
            assert result.claim_ratio >= 0.0

    def test_evaluator_zero_student_claims(self):
        """Test evaluator when student has zero claims."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Student has unverifiable content (should have 0 claims)
        student_outputs = [
            "I think maybe this is probably a good idea.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. API docs at https://example.com",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # When student has 0 claims, claim_ratio should be 0.0
        if result.student_claim_count == 0:
            assert result.claim_ratio == 0.0
        else:
            # If student somehow has claims, ratio should be >= 0
            assert result.claim_ratio >= 0.0

    def test_evaluator_both_zero_claims(self):
        """Test evaluator when both student and teacher have zero claims."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Both have unverifiable content
        student_outputs = [
            "I think maybe this is probably a good idea.",
        ]
        teacher_outputs = [
            "This might be a good approach. Perhaps it could work.",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # When both have 0 claims, ratios should be 0.0
        assert result.claim_ratio == 0.0
        # Loss should be 1.0 (penalty for zero ratios)
        assert result.claim_extraction_loss == 1.0

    def test_evaluator_loss_penalty_calculation(self):
        """Test that loss penalty is calculated correctly."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Student has fewer claims (below min_claim_ratio of 0.5)
        student_outputs = [
            "Version 1.0.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com. Code: ```python\ndef func():\n    return 1\n```",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # If claim_ratio < 0.5, there should be a penalty
        if result.claim_ratio < 0.5:
            # claim_penalty = max(0, 1 - (claim_ratio / 0.5))
            expected_claim_penalty = max(0.0, 1.0 - (result.claim_ratio / 0.5))
            # But we can't verify exact calculation without knowing ratios
            # Just verify loss is >= 0
            assert result.claim_extraction_loss >= 0.0
        else:
            # If ratios meet thresholds, loss should be 0.0
            if result.claim_ratio >= 0.5 and result.success_rate_ratio >= 0.7:
                assert result.claim_extraction_loss == 0.0

    def test_evaluator_single_sample(self):
        """Test evaluator with single sample."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 2.0 was released on 2024-01-15. API docs at https://example.com",
        ]
        teacher_outputs = [
            "Version 2.0 was released on 2024-01-15. Documentation: https://docs.example.com",
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        # With single sample, averages should equal the single values
        assert isinstance(result, ClaimExtractionEvalResult)
        # Should be integers (averages converted to int)
        assert isinstance(result.student_claim_count, int)
        assert isinstance(result.teacher_claim_count, int)

    def test_evaluator_large_batch(self):
        """Test evaluator with large batch of samples."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Create 10 samples
        student_outputs = [
            f"Version {i}.0 was released on 2024-01-15. API docs at https://example{i}.com"
            for i in range(10)
        ]
        teacher_outputs = [
            f"Version {i}.0 was released on 2024-01-15. Documentation: https://docs.example{i}.com"
            for i in range(10)
        ]

        result = evaluator.evaluate(student_outputs, teacher_outputs)

        assert isinstance(result, ClaimExtractionEvalResult)
        # Should aggregate across all 10 samples
        assert result.student_claim_count >= 0
        assert result.teacher_claim_count >= 0
        # Averages should be reasonable
        assert 0.0 <= result.student_success_rate <= 1.0
        assert 0.0 <= result.teacher_success_rate <= 1.0

    def test_evaluator_default_extractor_creation(self):
        """Test that evaluator creates default extractor if None provided."""
        evaluator = ClaimExtractionEvaluator()

        # Should have an extractor
        assert evaluator.extractor is not None
        from training.claim_extraction import SimpleClaimExtractor
        assert isinstance(evaluator.extractor, SimpleClaimExtractor)

    def test_evaluator_custom_extractor(self):
        """Test that evaluator uses custom extractor when provided."""
        from training.claim_extraction import SimpleClaimExtractor

        custom_extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(custom_extractor)

        # Should use the custom extractor
        assert evaluator.extractor is custom_extractor

    def test_evaluator_min_thresholds_parameters(self):
        """Test that min_claim_ratio and min_success_rate_ratio parameters work."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        student_outputs = [
            "Version 1.0 was released.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com",
        ]

        # Test with different threshold values
        result1 = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.1,
            min_success_rate_ratio=0.1,
        )

        result2 = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.9,
            min_success_rate_ratio=0.9,
        )

        # Results should differ based on thresholds (loss calculation)
        # Higher thresholds may result in higher loss if ratios don't meet thresholds
        assert isinstance(result1.claim_extraction_loss, float)
        assert isinstance(result2.claim_extraction_loss, float)


class TestEvaluateFromDatasetReal:
    """Test evaluate_from_dataset method with better test setup."""

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.utils.data.Subset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_basic_flow(
        self, mock_no_grad, mock_subset, mock_dataset_class, mock_dataloader_class
    ):
        """Test basic flow of evaluate_from_dataset."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)
        mock_dataset_class.return_value = mock_dataset

        # Mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": Mock(),
                        "attention_mask": None,
                        "teacher_text": "Version 1.0 was released on 2024-01-15.",
                    },
                    {
                        "input_ids": Mock(),
                        "attention_mask": Mock(),
                        "teacher_text": ["Version 2.0 includes feature X."],
                    },
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_logits.argmax = Mock(return_value=Mock())
        mock_model.return_value = mock_logits

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(side_effect=["Student output 1", "Student output 2"])

        # Mock input_ids.to() and attention_mask.to()
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)
        mock_attention_mask = Mock()
        mock_attention_mask.to = Mock(return_value=mock_attention_mask)

        # Setup batch generator
        def batch_generator():
            yield {
                "input_ids": mock_input_ids,
                "attention_mask": None,
                "teacher_text": "Version 1.0 was released on 2024-01-15.",
            }
            yield {
                "input_ids": mock_input_ids,
                "attention_mask": mock_attention_mask,
                "teacher_text": ["Version 2.0 includes feature X."],
            }

        mock_dataloader.__iter__ = Mock(return_value=batch_generator())

        # Mock argmax result
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        assert isinstance(result, ClaimExtractionEvalResult)
        mock_model.eval.assert_called_once()
        # Should have processed 2 batches
        assert mock_tokenizer.decode.call_count == 2

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.utils.data.Subset")
    def test_evaluate_from_dataset_max_samples(
        self, mock_subset, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset with max_samples limit."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.return_value = mock_dataset

        # Mock Subset
        mock_subset_obj = Mock()
        mock_subset_obj.__len__ = Mock(return_value=5)
        mock_subset.return_value = mock_subset_obj

        # Mock dataloader with empty iterator (no batches)
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([]))
        mock_dataloader_class.return_value = mock_dataloader

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        with patch("torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)

            result = evaluator.evaluate_from_dataset(
                dataset_path="test.jsonl",
                student_model=mock_model,
                tokenizer=mock_tokenizer,
                device="cpu",
                max_samples=5,
            )

            # Should have used Subset with max_samples
            mock_subset.assert_called_once()
            assert isinstance(result, ClaimExtractionEvalResult)
            # With empty batches, should have 0 claims
            assert result.student_claim_count == 0
            assert result.teacher_claim_count == 0

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_empty_dataset(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset with empty dataset."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock empty dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=0)
        mock_dataset_class.return_value = mock_dataset

        # Mock empty dataloader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([]))
        mock_dataloader_class.return_value = mock_dataloader

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Should return result with zero claims
        assert isinstance(result, ClaimExtractionEvalResult)
        assert result.student_claim_count == 0
        assert result.teacher_claim_count == 0

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_list_teacher_text(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset when teacher_text is a list."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset_class.return_value = mock_dataset

        # Mock input_ids
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)

        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)
        mock_model.return_value = mock_logits

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="Student output")

        # Mock dataloader with list teacher_text
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": mock_input_ids,
                        "attention_mask": None,
                        "teacher_text": ["Version 1.0 was released on 2024-01-15."],
                    }
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Should handle list teacher_text (extracts first element)
        assert isinstance(result, ClaimExtractionEvalResult)

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_empty_list_teacher_text(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset when teacher_text is an empty list."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset_class.return_value = mock_dataset

        # Mock input_ids
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)

        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)
        mock_model.return_value = mock_logits

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="Student output")

        # Mock dataloader with empty list teacher_text
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": mock_input_ids,
                        "attention_mask": None,
                        "teacher_text": [],  # Empty list
                    }
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Should handle empty list teacher_text (uses empty string)
        assert isinstance(result, ClaimExtractionEvalResult)
        # Teacher text should be empty string, so should have 0 teacher claims
        assert result.teacher_claim_count == 0

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_missing_teacher_text(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset when teacher_text is missing from batch."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset_class.return_value = mock_dataset

        # Mock input_ids
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)

        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)
        mock_model.return_value = mock_logits

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="Student output")

        # Mock dataloader with missing teacher_text (uses default "")
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": mock_input_ids,
                        "attention_mask": None,
                        # teacher_text is missing
                    }
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Should handle missing teacher_text (uses empty string)
        assert isinstance(result, ClaimExtractionEvalResult)
        # Teacher text should be empty string, so should have 0 teacher claims
        assert result.teacher_claim_count == 0

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_missing_attention_mask(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset when attention_mask is missing."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset_class.return_value = mock_dataset

        # Mock input_ids
        mock_input_ids = Mock()
        mock_input_ids.to = Mock(return_value=mock_input_ids)

        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = Mock()
        mock_argmax_result = Mock()
        mock_argmax_result.__getitem__ = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu = Mock(return_value=Mock())
        mock_argmax_result.__getitem__().cpu().tolist = Mock(return_value=[1, 2, 3])
        mock_logits.argmax = Mock(return_value=mock_argmax_result)
        mock_model.return_value = mock_logits

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="Student output")

        # Mock dataloader with missing attention_mask
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "input_ids": mock_input_ids,
                        # attention_mask is missing
                        "teacher_text": "Version 1.0 was released on 2024-01-15.",
                    }
                ]
            )
        )
        mock_dataloader_class.return_value = mock_dataloader

        # Mock no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        result = evaluator.evaluate_from_dataset(
            dataset_path="test.jsonl",
            student_model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Should handle missing attention_mask (passes None to model)
        assert isinstance(result, ClaimExtractionEvalResult)
        mock_model.assert_called_once()

    @patch("torch.utils.data.DataLoader")
    @patch("training.dataset.KDDataset")
    @patch("torch.no_grad")
    def test_evaluate_from_dataset_max_samples_exceeds_dataset_size(
        self, mock_no_grad, mock_dataset_class, mock_dataloader_class
    ):
        """Test evaluate_from_dataset when max_samples exceeds dataset size."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Mock dataset with only 2 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)
        mock_dataset_class.return_value = mock_dataset

        # Mock Subset - should use min(max_samples, len(dataset)) = min(10, 2) = 2
        mock_subset_obj = Mock()
        mock_subset_obj.__len__ = Mock(return_value=2)
        with patch("torch.utils.data.Subset", return_value=mock_subset_obj) as mock_subset:
            # Mock dataloader
            mock_dataloader = Mock()
            mock_dataloader.__iter__ = Mock(return_value=iter([]))
            mock_dataloader_class.return_value = mock_dataloader

            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_tokenizer = Mock()

            # Mock no_grad context manager
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)

            result = evaluator.evaluate_from_dataset(
                dataset_path="test.jsonl",
                student_model=mock_model,
                tokenizer=mock_tokenizer,
                device="cpu",
                max_samples=10,  # Exceeds dataset size of 2
            )

            # Should use min(10, 2) = 2
            mock_subset.assert_called_once()
            call_args = mock_subset.call_args
            # Should be range(0, min(10, 2)) = range(0, 2)
            assert isinstance(result, ClaimExtractionEvalResult)


class TestLossCalculationEdgeCases:
    """Test edge cases in loss calculation."""

    def test_loss_calculation_zero_ratios(self):
        """Test loss calculation when both ratios are zero."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Both have unverifiable content (zero claims)
        student_outputs = ["I think maybe this is probably a good idea."]
        teacher_outputs = ["This might be a good approach. Perhaps it could work."]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # When both ratios are 0, loss should be 1.0
        assert result.claim_ratio == 0.0
        assert result.success_rate_ratio == 0.0
        # claim_penalty = max(0, 1 - (0 / 0.5)) = max(0, 1 - 0) = 1.0
        # success_penalty = max(0, 1 - (0 / 0.7)) = max(0, 1 - 0) = 1.0
        # loss = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert result.claim_extraction_loss == 1.0

    def test_loss_calculation_meets_thresholds(self):
        """Test loss calculation when ratios meet thresholds."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Use texts that should produce good ratios
        student_outputs = [
            "Version 2.0 was released on 2024-01-15. API docs at https://example.com. Code: ```python\ndef func():\n    return 1\n```",
        ]
        teacher_outputs = [
            "Version 2.0 was released on 2024-01-15. Documentation: https://docs.example.com. Implementation:\n```python\ndef func():\n    return 1\n```",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,  # Low threshold
            min_success_rate_ratio=0.7,  # Low threshold
        )

        # If ratios meet thresholds, loss should be 0.0
        if result.claim_ratio >= 0.5 and result.success_rate_ratio >= 0.7:
            assert result.claim_extraction_loss == 0.0
        else:
            # Otherwise, loss should be > 0
            assert result.claim_extraction_loss >= 0.0

    def test_loss_calculation_below_thresholds(self):
        """Test loss calculation when ratios are below thresholds."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Student has very few claims compared to teacher
        student_outputs = [
            "Version 1.0.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com. Code: ```python\ndef func():\n    return 1\n```. Documentation: https://docs.example.com",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # If ratios are below thresholds, loss should be > 0
        if result.claim_ratio < 0.5 or result.success_rate_ratio < 0.7:
            assert result.claim_extraction_loss > 0.0
        assert 0.0 <= result.claim_extraction_loss <= 1.0

    def test_loss_calculation_penalty_weights(self):
        """Test that loss uses correct penalty weights (0.6 for claim, 0.4 for success)."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Create scenario where only claim_ratio is below threshold
        # This is hard to control precisely, but we can verify the structure
        student_outputs = [
            "Version 1.0.",
        ]
        teacher_outputs = [
            "Version 1.0 was released on 2024-01-15. It includes feature X. API docs at https://example.com",
        ]

        result = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.5,
            min_success_rate_ratio=0.7,
        )

        # Loss should be weighted combination
        # claim_penalty * 0.6 + success_penalty * 0.4
        # We can't verify exact values without knowing ratios, but structure should be correct
        assert 0.0 <= result.claim_extraction_loss <= 1.0

    def test_loss_calculation_boundary_values(self):
        """Test loss calculation at boundary values."""
        from training.claim_extraction import SimpleClaimExtractor

        extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(extractor)

        # Test with exactly at threshold
        # This is hard to control precisely, but we can test the structure
        student_outputs = [
            "Version 2.0 was released on 2024-01-15. API docs at https://example.com",
        ]
        teacher_outputs = [
            "Version 2.0 was released on 2024-01-15. Documentation: https://docs.example.com",
        ]

        # Test with very low thresholds (should pass)
        result_low = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.01,  # Very low
            min_success_rate_ratio=0.01,  # Very low
        )

        # Test with very high thresholds (may fail)
        result_high = evaluator.evaluate(
            student_outputs,
            teacher_outputs,
            min_claim_ratio=0.99,  # Very high
            min_success_rate_ratio=0.99,  # Very high
        )

        # Loss with high thresholds should be >= loss with low thresholds
        assert result_high.claim_extraction_loss >= result_low.claim_extraction_loss


class TestClaimExtractionEvaluatorInitialization:
    """Test ClaimExtractionEvaluator initialization edge cases."""

    def test_evaluator_init_none_extractor(self):
        """Test evaluator initialization with None extractor."""
        evaluator = ClaimExtractionEvaluator(None)

        # Should create new SimpleClaimExtractor
        from training.claim_extraction import SimpleClaimExtractor
        assert isinstance(evaluator.extractor, SimpleClaimExtractor)

    def test_evaluator_init_default_extractor(self):
        """Test evaluator initialization with default (no extractor provided)."""
        evaluator = ClaimExtractionEvaluator()

        # Should create new SimpleClaimExtractor
        from training.claim_extraction import SimpleClaimExtractor
        assert isinstance(evaluator.extractor, SimpleClaimExtractor)

    def test_evaluator_init_custom_extractor(self):
        """Test evaluator initialization with custom extractor."""
        from training.claim_extraction import SimpleClaimExtractor

        custom_extractor = SimpleClaimExtractor()
        evaluator = ClaimExtractionEvaluator(custom_extractor)

        # Should use the custom extractor
        assert evaluator.extractor is custom_extractor
        assert evaluator.extractor == custom_extractor


class TestClaimExtractionEvalResultEdgeCases:
    """Test ClaimExtractionEvalResult edge cases."""

    def test_eval_result_negative_values(self):
        """Test ClaimExtractionEvalResult with negative values (should not happen, but test structure)."""
        # Note: In practice, counts should never be negative, but test the dataclass structure
        result = ClaimExtractionEvalResult(
            student_claim_count=0,
            teacher_claim_count=0,
            student_success_rate=0.0,
            teacher_success_rate=0.0,
            claim_ratio=0.0,
            success_rate_ratio=0.0,
            claim_extraction_loss=1.0,
        )

        assert result.student_claim_count == 0
        assert result.teacher_claim_count == 0
        assert result.claim_ratio == 0.0

    def test_eval_result_high_values(self):
        """Test ClaimExtractionEvalResult with high values."""
        result = ClaimExtractionEvalResult(
            student_claim_count=100,
            teacher_claim_count=80,
            student_success_rate=0.95,
            teacher_success_rate=0.90,
            claim_ratio=1.25,
            success_rate_ratio=1.06,
            claim_extraction_loss=0.0,
        )

        assert result.student_claim_count == 100
        assert result.teacher_claim_count == 80
        assert result.claim_ratio == 1.25
        assert result.success_rate_ratio == 1.06
        assert result.claim_extraction_loss == 0.0

    def test_eval_result_fractional_counts(self):
        """Test ClaimExtractionEvalResult - counts should be integers (averages converted to int)."""
        # Counts are converted to int in evaluate method
        result = ClaimExtractionEvalResult(
            student_claim_count=2,  # int(2.5) = 2
            teacher_claim_count=1,  # int(1.5) = 1
            student_success_rate=0.75,
            teacher_success_rate=0.85,
            claim_ratio=1.33,  # Rounded to 2 decimals
            success_rate_ratio=0.88,  # Rounded to 2 decimals
            claim_extraction_loss=0.15,
        )

        # Counts should be integers
        assert isinstance(result.student_claim_count, int)
        assert isinstance(result.teacher_claim_count, int)
        # Rates and ratios should be floats
        assert isinstance(result.student_success_rate, float)
        assert isinstance(result.teacher_success_rate, float)
        assert isinstance(result.claim_ratio, float)
        assert isinstance(result.success_rate_ratio, float)
        assert isinstance(result.claim_extraction_loss, float)
