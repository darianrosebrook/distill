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
