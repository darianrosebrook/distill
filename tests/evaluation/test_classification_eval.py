"""
Tests for evaluation/classification_eval.py - Classification model evaluation framework.

Tests classification evaluation treating models as N-class classifiers,
prediction comparison, and evaluation metrics using configurable frameworks.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import numpy as np
import torch
import torch.nn as nn

# Import the module using importlib
import importlib

classification_eval_module = importlib.import_module("evaluation.classification_eval")

# Import classes and functions from the module
ClassificationConfig = classification_eval_module.ClassificationConfig
PredictionResult = classification_eval_module.PredictionResult
EvaluationMetrics = classification_eval_module.EvaluationMetrics
load_classification_config = classification_eval_module.load_classification_config
evaluate_pytorch_model = classification_eval_module.evaluate_pytorch_model
evaluate_coreml_model = classification_eval_module.evaluate_coreml_model
evaluate_ollama_model = classification_eval_module.evaluate_ollama_model
compare_predictions = classification_eval_module.compare_predictions
main = classification_eval_module.main


class TestClassificationConfig:
    """Test ClassificationConfig dataclass."""

    def test_classification_config_creation(self):
        """Test creating ClassificationConfig instance."""
        config = ClassificationConfig(
            name="test_task",
            class_names=["class_a", "class_b", "class_c"],
            token_ids=[100, 200, 300],
            id_to_name={100: "class_a", 200: "class_b", 300: "class_c"},
            name_to_id={"class_a": 100, "class_b": 200, "class_c": 300},
        )

        assert config.name == "test_task"
        assert config.class_names == ["class_a", "class_b", "class_c"]
        assert config.token_ids == [100, 200, 300]
        assert config.id_to_name[100] == "class_a"
        assert config.name_to_id["class_b"] == 200

    def test_classification_config_validation(self):
        """Test that config has consistent mappings."""
        # Valid config
        config = ClassificationConfig(
            name="valid",
            class_names=["A", "B"],
            token_ids=[1, 2],
            id_to_name={1: "A", 2: "B"},
            name_to_id={"A": 1, "B": 2},
        )

        # Check consistency
        assert len(config.class_names) == len(config.token_ids)
        assert len(config.id_to_name) == len(config.name_to_id)
        assert set(config.class_names) == set(config.name_to_id.keys())
        assert set(config.token_ids) == set(config.id_to_name.keys())


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating PredictionResult instance."""
        result = PredictionResult(
            question="What is the answer?",
            predicted_class_id=100,
            predicted_class_name="class_a",
            class_probabilities=np.array([0.1, 0.8, 0.1]),
        )

        assert result.question == "What is the answer?"
        assert result.predicted_class_id == 100
        assert result.predicted_class_name == "class_a"
        assert np.array_equal(result.class_probabilities, np.array([0.1, 0.8, 0.1]))

    def test_prediction_result_without_probabilities(self):
        """Test PredictionResult without class probabilities."""
        result = PredictionResult(
            question="Test question?", predicted_class_id=200, predicted_class_name="class_b"
        )

        assert result.question == "Test question?"
        assert result.predicted_class_id == 200
        assert result.predicted_class_name == "class_b"
        assert result.class_probabilities is None


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_evaluation_metrics_creation(self):
        """Test creating EvaluationMetrics instance."""
        metrics = EvaluationMetrics(
            total_questions=100,
            exact_match_rate=0.85,
            mean_l2_drift=0.12,
            class_distribution=[20, 30, 25, 25],  # Should sum to total_questions
            prediction_confidence=0.78,
        )

        assert metrics.total_questions == 100
        assert metrics.exact_match_rate == 0.85
        assert metrics.mean_l2_drift == 0.12
        assert sum(metrics.class_distribution) == 100
        assert metrics.prediction_confidence == 0.78

    def test_evaluation_metrics_minimal(self):
        """Test EvaluationMetrics with minimal required fields."""
        metrics = EvaluationMetrics(total_questions=50, exact_match_rate=0.90)

        assert metrics.total_questions == 50
        assert metrics.exact_match_rate == 0.90
        assert metrics.mean_l2_drift is None
        assert metrics.class_distribution is None
        assert metrics.prediction_confidence is None


class TestLoadClassificationConfig:
    """Test load_classification_config function."""

    def test_load_classification_config_success(self, tmp_path):
        """Test successful config loading."""
        config_data = {
            "name": "test_classification",
            "class_names": ["positive", "negative", "neutral"],
            "token_ids": [1000, 2000, 3000],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = load_classification_config(str(config_file))

        assert isinstance(result, ClassificationConfig)
        assert result.name == "test_classification"
        assert result.class_names == ["positive", "negative", "neutral"]
        assert result.token_ids == [1000, 2000, 3000]

        # Check derived mappings
        assert result.id_to_name[1000] == "positive"
        assert result.name_to_id["negative"] == 2000

    def test_load_classification_config_file_not_found(self):
        """Test config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_classification_config("nonexistent.json")

    def test_load_classification_config_invalid_json(self, tmp_path):
        """Test config loading with invalid JSON."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_classification_config(str(config_file))

    def test_load_classification_config_missing_fields(self, tmp_path):
        """Test config loading with missing required fields."""
        config_data = {"name": "test"}  # Missing class_names and token_ids

        config_file = tmp_path / "incomplete.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(KeyError):
            load_classification_config(str(config_file))


class TestEvaluatePyTorchModel:
    """Test evaluate_pytorch_model function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create mock ClassificationConfig."""
        return ClassificationConfig(
            name="test",
            class_names=["A", "B", "C"],
            token_ids=[100, 200, 300],
            id_to_name={100: "A", 200: "B", 300: "C"},
            name_to_id={"A": 100, "B": 200, "C": 300},
        )

    def test_evaluate_pytorch_model_success(self, mock_model, mock_tokenizer, mock_config):
        """Test successful PyTorch model evaluation."""
        questions = ["What is it?", "Which one?"]

        # Mock model outputs with logits favoring token 200 (class B)
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(2, 10, 1000)
        mock_outputs.logits[:, -1, 200] = 10.0  # High logit for token 200

        with (
            patch("evaluation.classification_eval.AutoTokenizer", return_value=mock_tokenizer),
            patch("torch.no_grad"),
            patch.object(mock_model, "__call__", return_value=mock_outputs),
        ):
            results = evaluate_pytorch_model("dummy_model", questions, mock_config)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

        # Check first result
        result1 = results[0]
        assert result1.question == "What is it?"
        assert result1.predicted_class_id == 200
        assert result1.predicted_class_name == "B"
        assert result1.class_probabilities is not None

    def test_evaluate_pytorch_model_empty_questions(self, mock_model, mock_config):
        """Test evaluation with empty questions list."""
        with patch("evaluation.classification_eval.AutoTokenizer"), patch("torch.no_grad"):
            results = evaluate_pytorch_model("dummy_model", [], mock_config)

        assert results == []

    def test_evaluate_pytorch_model_tokenizer_failure(self, mock_config):
        """Test evaluation when tokenizer loading fails."""
        questions = ["Test question"]

        with patch(
            "evaluation.classification_eval.AutoTokenizer", side_effect=Exception("Tokenizer error")
        ):
            with pytest.raises(Exception):
                evaluate_pytorch_model("dummy_model", questions, mock_config)


class TestEvaluateCoreMLModel:
    """Test evaluate_coreml_model function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ClassificationConfig."""
        return ClassificationConfig(
            name="test",
            class_names=["X", "Y"],
            token_ids=[500, 600],
            id_to_name={500: "X", 600: "Y"},
            name_to_id={"X": 500, "Y": 600},
        )

    def test_evaluate_coreml_model_success(self, mock_config, tmp_path):
        """Test successful CoreML model evaluation."""
        questions = ["First?", "Second?"]

        # Mock CoreML prediction results
        mock_predictions = [
            {"output": [0.3, 0.7]},  # Probabilities favoring class Y (600)
            {"output": [0.8, 0.2]},  # Probabilities favoring class X (500)
        ]

        with (
            patch("evaluation.classification_eval.ctk.load") as mock_load,
            patch("builtins.open"),
            patch("json.load") as mock_json_load,
        ):
            mock_coreml_model = Mock()
            mock_load.return_value = mock_coreml_model
            mock_json_load.return_value = {"input_ids": [1, 2, 3]}

            # Mock the prediction method
            mock_coreml_model.predict = Mock(side_effect=mock_predictions)

            results = evaluate_coreml_model("dummy_model.mlpackage", questions, mock_config)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

        # Check results
        assert results[0].predicted_class_name == "Y"  # Higher prob for second class
        assert results[1].predicted_class_name == "X"  # Higher prob for first class

    def test_evaluate_coreml_model_file_not_found(self, mock_config):
        """Test CoreML evaluation with missing model file."""
        questions = ["Test question"]

        with patch("evaluation.classification_eval.ctk.load", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                evaluate_coreml_model("nonexistent.mlpackage", questions, mock_config)


class TestEvaluateOllamaModel:
    """Test evaluate_ollama_model function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ClassificationConfig."""
        return ClassificationConfig(
            name="test",
            class_names=["Yes", "No"],
            token_ids=[700, 800],
            id_to_name={700: "Yes", 800: "No"},
            name_to_id={"Yes": 700, "No": 800},
        )

    def test_evaluate_ollama_model_success(self, mock_config):
        """Test successful Ollama model evaluation."""
        questions = ["Will it work?", "Should I proceed?"]

        # Mock subprocess output
        mock_output = """[
            {"predicted_class_id": 700, "predicted_class_name": "Yes", "class_probabilities": [0.9, 0.1]},
            {"predicted_class_id": 800, "predicted_class_name": "No", "class_probabilities": [0.2, 0.8]}
        ]"""

        with (
            patch("subprocess.run") as mock_run,
            patch(
                "json.loads",
                return_value=[
                    {
                        "predicted_class_id": 700,
                        "predicted_class_name": "Yes",
                        "class_probabilities": [0.9, 0.1],
                    },
                    {
                        "predicted_class_id": 800,
                        "predicted_class_name": "No",
                        "class_probabilities": [0.2, 0.8],
                    },
                ],
            ),
        ):
            mock_process = Mock()
            mock_process.stdout = mock_output
            mock_process.stderr = ""
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            results = evaluate_ollama_model("test_model", questions, mock_config)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)
        assert results[0].predicted_class_name == "Yes"
        assert results[1].predicted_class_name == "No"

    def test_evaluate_ollama_model_subprocess_failure(self, mock_config):
        """Test Ollama evaluation when subprocess fails."""
        questions = ["Test question"]

        with patch("subprocess.run") as mock_run:
            mock_process = Mock()
            mock_process.returncode = 1
            mock_process.stderr = "Ollama error"
            mock_run.return_value = mock_process

            with pytest.raises(Exception):
                evaluate_ollama_model("test_model", questions, mock_config)

    def test_evaluate_ollama_model_invalid_json(self, mock_config):
        """Test Ollama evaluation with invalid JSON output."""
        questions = ["Test question"]

        with (
            patch("subprocess.run") as mock_run,
            patch("json.loads", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)),
        ):
            mock_process = Mock()
            mock_process.stdout = "invalid json"
            mock_process.stderr = ""
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            with pytest.raises(json.JSONDecodeError):
                evaluate_ollama_model("test_model", questions, mock_config)


class TestComparePredictions:
    """Test compare_predictions function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ClassificationConfig."""
        return ClassificationConfig(
            name="test",
            class_names=["A", "B", "C"],
            token_ids=[100, 200, 300],
            id_to_name={100: "A", 200: "B", 300: "C"},
            name_to_id={"A": 100, "B": 200, "C": 300},
        )

    def test_compare_predictions_identical(self, mock_config):
        """Test comparing identical predictions."""
        predictions1 = [
            PredictionResult("Q1", 100, "A", np.array([0.8, 0.1, 0.1])),
            PredictionResult("Q2", 200, "B", np.array([0.2, 0.7, 0.1])),
        ]
        predictions2 = [
            PredictionResult("Q1", 100, "A", np.array([0.8, 0.1, 0.1])),
            PredictionResult("Q2", 200, "B", np.array([0.2, 0.7, 0.1])),
        ]

        metrics = compare_predictions(predictions1, predictions2, mock_config)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.total_questions == 2
        assert metrics.exact_match_rate == 1.0
        assert metrics.class_distribution == [1, 1, 0]  # A:1, B:1, C:0

    def test_compare_predictions_different(self, mock_config):
        """Test comparing different predictions."""
        predictions1 = [
            PredictionResult("Q1", 100, "A", np.array([0.8, 0.1, 0.1])),
            PredictionResult("Q2", 200, "B", np.array([0.2, 0.7, 0.1])),
        ]
        predictions2 = [
            PredictionResult("Q1", 200, "B", np.array([0.3, 0.6, 0.1])),  # Different prediction
            PredictionResult("Q2", 200, "B", np.array([0.2, 0.7, 0.1])),  # Same prediction
        ]

        metrics = compare_predictions(predictions1, predictions2, mock_config)

        assert metrics.total_questions == 2
        assert metrics.exact_match_rate == 0.5  # 1 out of 2 match
        assert metrics.class_distribution == [0, 2, 0]  # A:0, B:2, C:0 from predictions2

    def test_compare_predictions_empty_lists(self, mock_config):
        """Test comparing empty prediction lists."""
        metrics = compare_predictions([], [], mock_config)

        assert metrics.total_questions == 0
        assert metrics.exact_match_rate == 0.0

    def test_compare_predictions_different_lengths(self, mock_config):
        """Test comparing predictions with different lengths."""
        predictions1 = [PredictionResult("Q1", 100, "A")]
        predictions2 = [PredictionResult("Q1", 100, "A"), PredictionResult("Q2", 200, "B")]

        with pytest.raises(ValueError, match="Reference and candidate must have same length"):
            compare_predictions(predictions1, predictions2, mock_config)

    def test_compare_predictions_with_probabilities(self, mock_config):
        """Test comparing predictions with class probabilities."""
        predictions1 = [
            PredictionResult("Q1", 100, "A", np.array([0.8, 0.1, 0.1])),
            PredictionResult("Q2", 200, "B", np.array([0.2, 0.7, 0.1])),
        ]
        predictions2 = [
            PredictionResult("Q1", 100, "A", np.array([0.7, 0.2, 0.1])),  # Close probabilities
            PredictionResult("Q2", 300, "C", np.array([0.1, 0.1, 0.8])),  # Different prediction
        ]

        metrics = compare_predictions(predictions1, predictions2, mock_config)

        assert metrics.total_questions == 2
        assert metrics.exact_match_rate == 0.5  # Only first matches
        assert metrics.mean_l2_drift is not None  # Should calculate drift

    def test_compare_predictions_without_probabilities(self, mock_config):
        """Test comparing predictions without class probabilities."""
        predictions1 = [PredictionResult("Q1", 100, "A"), PredictionResult("Q2", 200, "B")]
        predictions2 = [PredictionResult("Q1", 100, "A"), PredictionResult("Q2", 300, "C")]

        metrics = compare_predictions(predictions1, predictions2, mock_config)

        assert metrics.total_questions == 2
        assert metrics.exact_match_rate == 0.5
        assert metrics.mean_l2_drift is None  # No probabilities to compare


class TestMainFunction:
    """Test main function."""

    @patch("evaluation.classification_eval.evaluate_pytorch_model")
    @patch("evaluation.classification_eval.evaluate_coreml_model")
    @patch("evaluation.classification_eval.evaluate_ollama_model")
    @patch("evaluation.classification_eval.compare_predictions")
    @patch("evaluation.classification_eval.load_classification_config")
    @patch("evaluation.classification_eval.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_pytorch_evaluation(
        self,
        mock_print,
        mock_parser_class,
        mock_load_config,
        mock_compare,
        mock_eval_ollama,
        mock_eval_coreml,
        mock_eval_pytorch,
    ):
        """Test main function with PyTorch model evaluation."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model_path = "model.pt"
        mock_args.config_path = "config.json"
        mock_args.backend = "pytorch"
        mock_args.compare_with = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock config
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Mock evaluation
        mock_predictions = [Mock(), Mock()]
        mock_eval_pytorch.return_value = mock_predictions

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_eval_pytorch.assert_called_once()

    @patch("evaluation.classification_eval.evaluate_coreml_model")
    @patch("evaluation.classification_eval.load_classification_config")
    @patch("evaluation.classification_eval.argparse.ArgumentParser")
    def test_main_coreml_evaluation(self, mock_parser_class, mock_load_config, mock_eval_coreml):
        """Test main function with CoreML model evaluation."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model_path = "model.mlpackage"
        mock_args.config_path = "config.json"
        mock_args.backend = "coreml"
        mock_args.compare_with = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock config and evaluation
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        mock_eval_coreml.return_value = [Mock()]

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_eval_coreml.assert_called_once()

    @patch("evaluation.classification_eval.load_classification_config")
    @patch("evaluation.classification_eval.argparse.ArgumentParser")
    def test_main_invalid_backend(self, mock_parser_class, mock_load_config):
        """Test main function with invalid backend."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "invalid_backend"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_config.return_value = Mock()

        with pytest.raises(SystemExit):
            main()

    @patch("evaluation.classification_eval.load_classification_config")
    @patch("evaluation.classification_eval.argparse.ArgumentParser")
    def test_main_config_not_found(self, mock_parser_class, mock_load_config):
        """Test main function with missing config file."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config_path = "nonexistent.json"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_config.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(SystemExit):
            main()


class TestClassificationEvalIntegration:
    """Test integration of classification evaluation components."""

    def test_complete_evaluation_workflow(self, tmp_path):
        """Test complete classification evaluation workflow."""
        # Create test config
        config_data = {
            "name": "binary_classification",
            "class_names": ["positive", "negative"],
            "token_ids": [1000, 2000],
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Load config
        config = load_classification_config(str(config_file))
        assert config.name == "binary_classification"

        # Test with mock model evaluation
        with patch("evaluation.classification_eval.evaluate_pytorch_model") as mock_eval:
            mock_predictions = [
                PredictionResult("Is this good?", 1000, "positive", np.array([0.9, 0.1])),
                PredictionResult("Is this bad?", 2000, "negative", np.array([0.2, 0.8])),
            ]
            mock_eval.return_value = mock_predictions

            results = evaluate_pytorch_model("dummy_model", ["Q1", "Q2"], config)

            assert len(results) == 2
            assert results[0].predicted_class_name == "positive"
            assert results[1].predicted_class_name == "negative"

    def test_evaluation_metrics_calculation(self, tmp_path):
        """Test evaluation metrics calculation."""
        # Create test config
        config_data = {"name": "test", "class_names": ["A", "B"], "token_ids": [1, 2]}

        config_file = tmp_path / "metrics_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = load_classification_config(str(config_file))

        # Create test predictions
        predictions1 = [
            PredictionResult("Q1", 1, "A", np.array([0.8, 0.2])),
            PredictionResult("Q2", 1, "A", np.array([0.7, 0.3])),
        ]

        predictions2 = [
            PredictionResult("Q1", 1, "A", np.array([0.9, 0.1])),  # Close match
            PredictionResult("Q2", 2, "B", np.array([0.4, 0.6])),  # Different prediction
        ]

        metrics = compare_predictions(predictions1, predictions2, config)

        assert metrics.total_questions == 2
        assert metrics.exact_match_rate == 0.5  # 1 out of 2 match
        assert metrics.class_distribution == [1, 1]  # A:1, B:1 from predictions2
        assert isinstance(metrics.mean_l2_drift, float)

    def test_config_validation(self):
        """Test classification config validation."""
        # Valid config
        config = ClassificationConfig(
            name="valid",
            class_names=["X", "Y", "Z"],
            token_ids=[10, 20, 30],
            id_to_name={10: "X", 20: "Y", 30: "Z"},
            name_to_id={"X": 10, "Y": 20, "Z": 30},
        )

        # Verify mappings are consistent
        assert len(config.class_names) == len(config.token_ids)
        assert all(
            config.id_to_name[tid] == name
            for tid, name in zip(config.token_ids, config.class_names)
        )
        assert all(
            config.name_to_id[name] == tid
            for name, tid in zip(config.class_names, config.token_ids)
        )

    def test_prediction_result_validation(self):
        """Test prediction result validation."""
        # Valid result
        result = PredictionResult(
            question="Test?",
            predicted_class_id=100,
            predicted_class_name="valid_class",
            class_probabilities=np.array([0.5, 0.3, 0.2]),
        )

        assert result.question
        assert result.predicted_class_id > 0
        assert result.predicted_class_name
        if result.class_probabilities is not None:
            assert abs(np.sum(result.class_probabilities) - 1.0) < 0.1  # Should roughly sum to 1
