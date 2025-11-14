"""
Tests for evaluation/pipeline_preservation_eval.py - Pipeline preservation evaluation.

Tests comparison of classification model predictions across pipeline stages to verify
class distribution is preserved through conversion and deployment.
"""
# @author: @darianrosebrook

import importlib
import json
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch

from evaluation.classification_eval import (
    ClassificationConfig,
    PredictionResult,
)

# Import the module using importlib
pipeline_preservation_eval_module = importlib.import_module(
    "evaluation.pipeline_preservation_eval")

# Import main function
main = pipeline_preservation_eval_module.main


class TestPipelinePreservationEvalMain:
    """Test main function for pipeline preservation evaluation."""

    @pytest.fixture
    def mock_config(self):
        """Create mock classification config."""
        config = Mock(spec=ClassificationConfig)
        config.name = "test_config"
        config.class_names = ["class1", "class2", "class3"]
        return config

    @pytest.fixture
    def mock_pytorch_results(self):
        """Create mock PyTorch prediction results."""
        return [
            PredictionResult(
                question="Test question 1?",
                predicted_class_id=1,
                predicted_class_name="class2",
                class_probabilities=torch.tensor([0.1, 0.7, 0.2]),
            ),
            PredictionResult(
                question="Test question 2?",
                predicted_class_id=0,
                predicted_class_name="class1",
                class_probabilities=torch.tensor([0.8, 0.1, 0.1]),
            ),
        ]

    @pytest.fixture
    def mock_coreml_results(self):
        """Create mock CoreML prediction results."""
        return [
            PredictionResult(
                question="Test question 1?",
                predicted_class_id=1,
                predicted_class_name="class2",
                class_probabilities=torch.tensor([0.1, 0.7, 0.2]),
            ),
            PredictionResult(
                question="Test question 2?",
                predicted_class_id=0,
                predicted_class_name="class1",
                class_probabilities=torch.tensor([0.8, 0.1, 0.1]),
            ),
        ]

    @pytest.fixture
    def mock_ollama_results(self):
        """Create mock Ollama prediction results."""
        return [
            PredictionResult(
                question="Test question 1?",
                predicted_class_id=1,
                predicted_class_name="class2",
            ),
            PredictionResult(
                question="Test question 2?",
                predicted_class_id=0,
                predicted_class_name="class1",
            ),
        ]

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_coreml_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_ollama_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_pytorch_only(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_ollama,
        mock_eval_coreml,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        tmp_path,
    ):
        """Test main function with PyTorch model only."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "model.pt",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),  # Add required output-dir argument
        ]):
            try:
                main()
            except SystemExit:
                pass

        mock_load_config.assert_called_once_with("test.config")
        mock_eval_pytorch.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_coreml_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_pytorch_and_coreml_comparison(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_coreml,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        mock_coreml_results,
        tmp_path,
    ):
        """Test main function with PyTorch and CoreML comparison."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results
        mock_eval_coreml.return_value = mock_coreml_results

        mock_metrics = Mock()
        mock_metrics.exact_match_rate = 1.0
        mock_metrics.mean_l2_drift = 0.001
        mock_metrics.mean_kl_divergence = 0.0005
        mock_compare.return_value = mock_metrics

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "pytorch.pt",
            "--coreml-model", "coreml.mlmodel",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        mock_eval_pytorch.assert_called_once()
        mock_eval_coreml.assert_called_once()
        mock_compare.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_ollama_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_pytorch_and_ollama_comparison(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_ollama,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        mock_ollama_results,
        tmp_path,
    ):
        """Test main function with PyTorch and Ollama comparison."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results
        mock_eval_ollama.return_value = mock_ollama_results

        mock_metrics = Mock()
        mock_metrics.exact_match_rate = 0.9
        mock_metrics.mean_l2_drift = None
        mock_metrics.mean_kl_divergence = None
        mock_compare.return_value = mock_metrics

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "pytorch.pt",
            "--ollama-model", "ollama_model",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        mock_eval_pytorch.assert_called_once()
        mock_eval_ollama.assert_called_once()
        mock_compare.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_coreml_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_ollama_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_all_backends_comparison(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_ollama,
        mock_eval_coreml,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        mock_coreml_results,
        mock_ollama_results,
        tmp_path,
    ):
        """Test main function with all backends (PyTorch, CoreML, Ollama)."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results
        mock_eval_coreml.return_value = mock_coreml_results
        mock_eval_ollama.return_value = mock_ollama_results

        mock_metrics = Mock()
        mock_metrics.exact_match_rate = 1.0
        mock_metrics.mean_l2_drift = 0.001
        mock_metrics.mean_kl_divergence = 0.0005
        mock_compare.return_value = mock_metrics

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "pytorch.pt",
            "--coreml-model", "coreml.mlmodel",
            "--ollama-model", "ollama_model",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        mock_eval_pytorch.assert_called_once()
        mock_eval_coreml.assert_called_once()
        mock_eval_ollama.assert_called_once()
        # Should compare both CoreML and Ollama
        assert mock_compare.call_count == 2

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("builtins.print")
    def test_main_config_load_error(self, mock_print, mock_load_config):
        """Test main function handles config loading errors."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")

        with patch("sys.argv", ["pipeline_preservation_eval.py", "--tokenizer", "tokenizer", "--config", "missing.config"]):
            with pytest.raises((SystemExit, FileNotFoundError)):
                main()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_with_eval_questions_file(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_load_config,
        mock_config,
        tmp_path,
    ):
        """Test main function loads questions from eval-questions file."""
        mock_load_config.return_value = mock_config

        questions_file = tmp_path / "questions.json"
        questions_data = {"questions": ["Question 1?", "Question 2?"]}
        with open(questions_file, "w") as f:
            json.dump(questions_data, f)

        with (
            patch("sys.argv", [
                "pipeline_preservation_eval.py",
                "--pytorch-model", "model.pt",
                "--tokenizer", "tokenizer",
                "--config", "test.config",
                "--eval-questions", str(questions_file),
                "--output-dir", str(tmp_path / "output"),
            ]),
            patch(
                "evaluation.pipeline_preservation_eval.evaluate_pytorch_model", return_value=[]),
        ):
            mock_file_handle = MagicMock()
            mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
            mock_file_handle.__exit__ = Mock(return_value=None)
            mock_file_handle.write = Mock()
            mock_file_handle.read = Mock(
                return_value=json.dumps(questions_data))
            mock_open.return_value = mock_file_handle

            try:
                main()
            except SystemExit:
                pass

        mock_load_config.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    @patch("importlib.util.find_spec")
    @patch("importlib.util.module_from_spec")
    def test_main_with_default_questions_from_module(
        self,
        mock_module_from_spec,
        mock_find_spec,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_load_config,
        mock_config,
        tmp_path,
    ):
        """Test main function finds default questions from config module."""
        mock_load_config.return_value = mock_config

        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_find_spec.return_value = mock_spec

        mock_module = Mock()
        mock_module.get_eight_ball_questions = Mock(
            return_value=["Question 1?", "Question 2?"])
        mock_module_from_spec.return_value = mock_module

        with (
            patch("sys.argv", [
                "pipeline_preservation_eval.py",
                "--pytorch-model", "model.pt",
                "--tokenizer", "tokenizer",
                "--config", "evaluation.toy.eight_ball.EIGHT_BALL_CONFIG",
                "--output-dir", str(tmp_path / "output"),
            ]),
            patch(
                "evaluation.pipeline_preservation_eval.evaluate_pytorch_model", return_value=[]),
        ):
            mock_file_handle = MagicMock()
            mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
            mock_file_handle.__exit__ = Mock(return_value=None)
            mock_file_handle.write = Mock()
            mock_open.return_value = mock_file_handle

            try:
                main()
            except SystemExit:
                pass

        mock_load_config.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_fallback_to_default_questions(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_load_config,
        mock_config,
        tmp_path,
    ):
        """Test main function falls back to default questions when module not found."""
        mock_load_config.return_value = mock_config

        with (
            patch("sys.argv", [
                "pipeline_preservation_eval.py",
                "--pytorch-model", "model.pt",
                "--tokenizer", "tokenizer",
                "--config", "test.config",
                "--output-dir", str(tmp_path / "output"),
            ]),
            patch(
                "evaluation.pipeline_preservation_eval.evaluate_pytorch_model", return_value=[]),
            patch("importlib.util.find_spec", return_value=None),
        ):
            mock_file_handle = MagicMock()
            mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
            mock_file_handle.__exit__ = Mock(return_value=None)
            mock_file_handle.write = Mock()
            mock_open.return_value = mock_file_handle

            try:
                main()
            except SystemExit:
                pass

        # Should fall back to default questions
        mock_load_config.assert_called_once()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_output_dir_creation(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        tmp_path,
    ):
        """Test main function creates output directory."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results

        output_dir = tmp_path / "custom_output"
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "model.pt",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(output_dir),
        ]):
            try:
                main()
            except SystemExit:
                pass

        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_saves_pytorch_predictions(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        tmp_path,
    ):
        """Test main function saves PyTorch predictions to JSON."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "model.pt",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        # Should write PyTorch predictions JSON
        mock_open.assert_called()
        mock_file_handle.write.assert_called()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_coreml_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_saves_comparison_metrics(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_coreml,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        mock_pytorch_results,
        mock_coreml_results,
        tmp_path,
    ):
        """Test main function saves comparison metrics."""
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = mock_pytorch_results
        mock_eval_coreml.return_value = mock_coreml_results

        mock_metrics = Mock()
        mock_metrics.exact_match_rate = 0.95
        mock_metrics.mean_l2_drift = 0.002
        mock_metrics.mean_kl_divergence = 0.001
        mock_compare.return_value = mock_metrics

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "pytorch.pt",
            "--coreml-model", "coreml.mlmodel",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        # Should save comparison metrics
        mock_file_handle.write.assert_called()

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_main_handles_none_probabilities(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_pytorch,
        mock_load_config,
        mock_config,
        tmp_path,
    ):
        """Test main function handles predictions without class probabilities."""
        results_no_probs = [
            PredictionResult(
                question="Test question?",
                predicted_class_id=1,
                predicted_class_name="class2",
                class_probabilities=None,
            ),
        ]
        mock_load_config.return_value = mock_config
        mock_eval_pytorch.return_value = results_no_probs

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "model.pt",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        # Should handle None probabilities gracefully
        mock_file_handle.write.assert_called()


class TestPipelinePreservationEvalIntegration:
    """Test integration scenarios for pipeline preservation evaluation."""

    @patch("evaluation.pipeline_preservation_eval.load_classification_config")
    @patch("evaluation.pipeline_preservation_eval.evaluate_pytorch_model")
    @patch("evaluation.pipeline_preservation_eval.evaluate_coreml_model")
    @patch("evaluation.pipeline_preservation_eval.compare_predictions")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    def test_pipeline_comparison_workflow(
        self,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_compare,
        mock_eval_coreml,
        mock_eval_pytorch,
        mock_load_config,
        tmp_path,
    ):
        """Test complete pipeline comparison workflow."""
        mock_config = Mock(spec=ClassificationConfig)
        mock_config.name = "test"
        mock_config.class_names = ["A", "B"]
        mock_load_config.return_value = mock_config

        pytorch_results = [
            PredictionResult(question="Q1?", predicted_class_id=0,
                             predicted_class_name="A"),
        ]
        coreml_results = [
            PredictionResult(question="Q1?", predicted_class_id=0,
                             predicted_class_name="A"),
        ]

        mock_eval_pytorch.return_value = pytorch_results
        mock_eval_coreml.return_value = coreml_results

        mock_metrics = Mock()
        mock_metrics.exact_match_rate = 1.0
        mock_metrics.mean_l2_drift = None
        mock_metrics.mean_kl_divergence = None
        mock_compare.return_value = mock_metrics

        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle

        with patch("sys.argv", [
            "pipeline_preservation_eval.py",
            "--pytorch-model", "pytorch.pt",
            "--coreml-model", "coreml.mlmodel",
            "--tokenizer", "tokenizer",
            "--config", "test.config",
            "--output-dir", str(tmp_path / "output"),
        ]):
            try:
                main()
            except SystemExit:
                pass

        # Verify complete workflow executed
        mock_eval_pytorch.assert_called_once()
        mock_eval_coreml.assert_called_once()
        mock_compare.assert_called_once()
