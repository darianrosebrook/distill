"""
Tests for evaluation/8ball_eval.py - 8-Ball model evaluation framework.

Tests 8-ball evaluation treating model as 20-class classifier,
prediction comparison, and evaluation metrics.
"""
# @author: @darianrosebrook

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

import importlib

eightball_eval_module = importlib.import_module("evaluation.8ball_eval")

# Import the constants and functions from the module
EIGHT_BALL_ANSWERS = eightball_eval_module.EIGHT_BALL_ANSWERS
EIGHT_BALL_TOKEN_IDS = eightball_eval_module.EIGHT_BALL_TOKEN_IDS
ID_TO_ANSWER = eightball_eval_module.ID_TO_ANSWER
PredictionResult = eightball_eval_module.PredictionResult
EvaluationMetrics = eightball_eval_module.EvaluationMetrics
load_eval_questions = eightball_eval_module.load_eval_questions
evaluate_pytorch_model = eightball_eval_module.evaluate_pytorch_model
evaluate_coreml_model = eightball_eval_module.evaluate_coreml_model
evaluate_ollama_model = eightball_eval_module.evaluate_ollama_model
compare_predictions = eightball_eval_module.compare_predictions
main = eightball_eval_module.main


class TestEightBallConstants:
    """Test 8-ball constants and mappings."""

    def test_eight_ball_answers_count(self):
        """Test that we have exactly 20 answers."""
        assert len(EIGHT_BALL_ANSWERS) == 20

    def test_eight_ball_answers_content(self):
        """Test that answers are reasonable 8-ball responses."""
        assert "It is certain" in EIGHT_BALL_ANSWERS
        assert "Very doubtful" in EIGHT_BALL_ANSWERS
        assert "Reply hazy, try again" in EIGHT_BALL_ANSWERS
        assert "Don't count on it" in EIGHT_BALL_ANSWERS

    def test_eight_ball_token_ids_range(self):
        """Test token ID range."""
        assert len(EIGHT_BALL_TOKEN_IDS) == 20
        assert EIGHT_BALL_TOKEN_IDS[0] == 200
        assert EIGHT_BALL_TOKEN_IDS[-1] == 219
        assert EIGHT_BALL_TOKEN_IDS == list(range(200, 220))

    def test_id_to_answer_mapping(self):
        """Test token ID to answer mapping."""
        assert len(ID_TO_ANSWER) == 20
        assert ID_TO_ANSWER[200] == "It is certain"
        assert ID_TO_ANSWER[219] == "Very doubtful"
        assert ID_TO_ANSWER[209] == "Signs point to yes"  # Index 9
        assert ID_TO_ANSWER[210] == "Reply hazy, try again"  # Index 10

    def test_id_to_answer_coverage(self):
        """Test that all token IDs are mapped."""
        for token_id in EIGHT_BALL_TOKEN_IDS:
            assert token_id in ID_TO_ANSWER
            assert ID_TO_ANSWER[token_id] in EIGHT_BALL_ANSWERS


class TestDataClasses:
    """Test dataclasses for prediction results and metrics."""

    def test_prediction_result_creation(self):
        """Test PredictionResult dataclass creation."""
        result = PredictionResult(
            question="Will I win the lottery?",
            predicted_token=205,
            predicted_answer="Outlook good",
            confidence=0.85,
            is_correct=True,
        )

        assert result.question == "Will I win the lottery?"
        assert result.predicted_token == 205
        assert result.predicted_answer == "Outlook good"
        assert result.confidence == 0.85
        assert result.is_correct == True

    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics dataclass creation."""
        metrics = EvaluationMetrics(
            total_predictions=100,
            correct_predictions=85,
            accuracy=0.85,
            token_distribution=[5] * 20,  # Equal distribution
            answer_distribution={"Yes": 50, "No": 30, "Maybe": 20},
        )

        assert metrics.total_predictions == 100
        assert metrics.correct_predictions == 85
        assert metrics.accuracy == 0.85
        assert len(metrics.token_distribution) == 20
        assert sum(metrics.token_distribution) == 100
        assert metrics.answer_distribution["Yes"] == 50


class TestLoadEvalQuestions:
    """Test loading evaluation questions."""

    def test_load_eval_questions_json_file(self, tmp_path):
        """Test loading questions from JSON file."""
        questions = ["Will it rain tomorrow?", "Should I buy stocks?", "Will I find true love?"]

        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(questions, f)

        result = load_eval_questions(json_file)

        assert result == questions

    def test_load_eval_questions_text_file(self, tmp_path):
        """Test loading questions from text file (one per line)."""
        questions = ["Will it rain tomorrow?", "Should I buy stocks?", "Will I find true love?"]

        text_file = tmp_path / "questions.txt"
        with open(text_file, "w") as f:
            f.write("\n".join(questions))

        result = load_eval_questions(text_file)

        assert result == questions

    def test_load_eval_questions_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_eval_questions(Path("nonexistent.json"))

    def test_load_eval_questions_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_eval_questions(json_file)


class TestEvaluatePyTorchModel:
    """Test PyTorch model evaluation."""

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
        tokenizer.encode = Mock(return_value=[1, 2, 3])  # Sample tokens
        return tokenizer

    def test_evaluate_pytorch_model_success(self, mock_model, mock_tokenizer):
        """Test successful PyTorch model evaluation."""
        questions = ["Will it rain?", "Should I invest?"]

        # Mock model outputs with token predictions
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(2, 10, 1000)  # [batch, seq, vocab]

        # Set high logits for 8-ball token range (200-219)
        mock_outputs.logits[:, -1, 205] = 10.0  # "Outlook good"
        mock_outputs.logits[:, -1, 210] = 8.0  # "Signs point to yes"

        with (
            patch("evaluation.eightball_eval.AutoTokenizer", return_value=mock_tokenizer),
            patch("torch.no_grad"),
            patch.object(mock_model, "__call__", return_value=mock_outputs),
        ):
            results = evaluate_pytorch_model("dummy_model", questions)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

        # Check first result
        result1 = results[0]
        assert result1.question == "Will it rain?"
        assert result1.predicted_token in EIGHT_BALL_TOKEN_IDS
        assert result1.predicted_answer in EIGHT_BALL_ANSWERS
        assert isinstance(result1.confidence, float)
        assert 0.0 <= result1.confidence <= 1.0

    def test_evaluate_pytorch_model_empty_questions(self, mock_model):
        """Test evaluation with empty questions list."""
        with patch("evaluation.eightball_eval.AutoTokenizer"), patch("torch.no_grad"):
            results = evaluate_pytorch_model("dummy_model", [])

        assert results == []

    def test_evaluate_pytorch_model_tokenizer_failure(self):
        """Test evaluation when tokenizer loading fails."""
        questions = ["Test question"]

        with patch(
            "evaluation.eightball_eval.AutoTokenizer", side_effect=Exception("Tokenizer error")
        ):
            with pytest.raises(Exception):
                evaluate_pytorch_model("dummy_model", questions)


class TestEvaluateCoreMLModel:
    """Test CoreML model evaluation."""

    @pytest.fixture
    def mock_coreml_model(self):
        """Create mock CoreML model."""
        model = Mock()
        # Mock CoreML model interface
        return model

    def test_evaluate_coreml_model_success(self, mock_coreml_model):
        """Test successful CoreML model evaluation."""
        questions = ["Will it work?", "Should I proceed?"]

        # Mock CoreML prediction results
        mock_predictions = [
            {"output": [0.1, 0.9, 0.8]},  # Mock logits
            {"output": [0.7, 0.2, 0.6]},
        ]

        with (
            patch("evaluation.eightball_eval.ctk.load") as mock_load,
            patch("builtins.open"),
            patch("json.load") as mock_json_load,
        ):
            mock_load.return_value = mock_coreml_model
            mock_json_load.return_value = {"input_ids": [1, 2, 3]}

            # Mock the prediction method
            mock_coreml_model.predict = Mock(side_effect=mock_predictions)

            with patch("evaluation.eightball_eval.load_eval_questions", return_value=questions):
                results = evaluate_coreml_model("dummy_model.mlpackage", questions)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_evaluate_coreml_model_file_not_found(self):
        """Test CoreML evaluation with missing model file."""
        questions = ["Test question"]

        with patch("evaluation.eightball_eval.ctk.load", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                evaluate_coreml_model("nonexistent.mlpackage", questions)


class TestEvaluateOllamaModel:
    """Test Ollama model evaluation."""

    def test_evaluate_ollama_model_success(self):
        """Test successful Ollama model evaluation."""
        questions = ["Will it succeed?", "Should I continue?"]

        # Mock subprocess output
        mock_output = """[
            {"predicted_token": 205, "predicted_answer": "Outlook good", "confidence": 0.85},
            {"predicted_token": 210, "predicted_answer": "Signs point to yes", "confidence": 0.72}
        ]"""

        with (
            patch("subprocess.run") as mock_run,
            patch(
                "json.loads",
                return_value=[
                    {
                        "predicted_token": 205,
                        "predicted_answer": "Outlook good",
                        "confidence": 0.85,
                    },
                    {
                        "predicted_token": 210,
                        "predicted_answer": "Signs point to yes",
                        "confidence": 0.72,
                    },
                ],
            ),
        ):
            mock_process = Mock()
            mock_process.stdout = mock_output
            mock_process.stderr = ""
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            results = evaluate_ollama_model("test_model", questions)

        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_evaluate_ollama_model_subprocess_failure(self):
        """Test Ollama evaluation when subprocess fails."""
        questions = ["Test question"]

        with patch("subprocess.run") as mock_run:
            mock_process = Mock()
            mock_process.returncode = 1
            mock_process.stderr = "Ollama error"
            mock_process.stdout = ""  # Add stdout to make Mock iterable
            mock_run.return_value = mock_process

            # The function should raise Exception for subprocess failures
            with pytest.raises(Exception, match="Error evaluating question"):
                evaluate_ollama_model("test_model", questions)

    def test_evaluate_ollama_model_invalid_json(self):
        """Test Ollama evaluation with invalid JSON output."""
        questions = ["Test question"]

        with patch("subprocess.run") as mock_run:
            mock_process = Mock()
            mock_process.stdout = "invalid json"  # Not valid JSON
            mock_process.stderr = ""
            mock_process.returncode = 0
            mock_run.return_value = mock_process

            # The function converts JSONDecodeError to Exception
            with pytest.raises(Exception, match="Invalid JSON"):
                evaluate_ollama_model("test_model", questions)


class TestComparePredictions:
    """Test prediction comparison functionality."""

    def test_compare_predictions_identical(self):
        """Test comparing identical predictions."""
        predictions1 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 205, "Outlook good", 0.8, False),
        ]
        predictions2 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 205, "Outlook good", 0.8, False),
        ]

        metrics = compare_predictions(predictions1, predictions2)

        assert metrics.total_predictions == 2
        assert metrics.correct_predictions == 2
        assert metrics.accuracy == 1.0
        assert len(metrics.token_distribution) == 20

    def test_compare_predictions_different(self):
        """Test comparing different predictions."""
        predictions1 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 205, "Outlook good", 0.8, False),
        ]
        predictions2 = [
            PredictionResult("Q1", 201, "It is decidedly so", 0.7, False),
            PredictionResult("Q2", 210, "Signs point to yes", 0.6, True),
        ]

        metrics = compare_predictions(predictions1, predictions2)

        assert metrics.total_predictions == 2
        assert metrics.correct_predictions == 0
        assert metrics.accuracy == 0.0

    def test_compare_predictions_empty_lists(self):
        """Test comparing empty prediction lists."""
        metrics = compare_predictions([], [])

        assert metrics.total_predictions == 0
        assert metrics.correct_predictions == 0
        assert metrics.accuracy == 0.0

    def test_compare_predictions_different_lengths(self):
        """Test comparing predictions with different lengths."""
        predictions1 = [PredictionResult("Q1", 200, "It is certain", 0.9, True)]
        predictions2 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 205, "Outlook good", 0.8, False),
        ]

        with pytest.raises(ValueError, match="Reference and candidate must have same length"):
            compare_predictions(predictions1, predictions2)

    def test_compare_predictions_token_distribution(self):
        """Test token distribution calculation."""
        predictions1 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 200, "It is certain", 0.8, True),
            PredictionResult("Q3", 205, "Outlook good", 0.7, False),
        ]
        predictions2 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 201, "It is decidedly so", 0.8, False),
            PredictionResult("Q3", 205, "Outlook good", 0.7, True),
        ]

        metrics = compare_predictions(predictions1, predictions2)

        # Should count correct predictions (2 out of 3)
        assert metrics.correct_predictions == 2
        assert metrics.accuracy == 2.0 / 3.0

        # Token distribution should reflect the predictions
        assert len(metrics.token_distribution) == 20


class TestMainFunction:
    """Test main function."""

    @patch("evaluation.eightball_eval.evaluate_pytorch_model")
    @patch("evaluation.eightball_eval.evaluate_coreml_model")
    @patch("evaluation.eightball_eval.evaluate_ollama_model")
    @patch("evaluation.eightball_eval.compare_predictions")
    @patch("evaluation.eightball_eval.load_eval_questions")
    @patch("evaluation.eightball_eval.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_pytorch_evaluation(
        self,
        mock_print,
        mock_parser_class,
        mock_load_questions,
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
        mock_args.eval_file = "questions.json"
        mock_args.backend = "pytorch"
        mock_args.compare_with = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock data
        questions = ["Q1", "Q2"]
        predictions = [Mock(), Mock()]

        mock_load_questions.return_value = questions
        mock_eval_pytorch.return_value = predictions

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_eval_pytorch.assert_called_once_with("model.pt", questions)

    @patch("evaluation.eightball_eval.evaluate_coreml_model")
    @patch("evaluation.eightball_eval.evaluate_ollama_model")
    @patch("evaluation.eightball_eval.load_eval_questions")
    @patch("evaluation.eightball_eval.argparse.ArgumentParser")
    def test_main_coreml_evaluation(
        self, mock_parser_class, mock_load_questions, mock_eval_ollama, mock_eval_coreml
    ):
        """Test main function with CoreML model evaluation."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model_path = "model.mlpackage"
        mock_args.eval_file = "questions.json"
        mock_args.backend = "coreml"
        mock_args.compare_with = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock data
        questions = ["Q1", "Q2"]
        predictions = [Mock(), Mock()]

        mock_load_questions.return_value = questions
        mock_eval_coreml.return_value = predictions

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_eval_coreml.assert_called_once_with("model.mlpackage", questions)

    @patch("evaluation.eightball_eval.evaluate_ollama_model")
    @patch("evaluation.eightball_eval.load_eval_questions")
    @patch("evaluation.eightball_eval.argparse.ArgumentParser")
    def test_main_ollama_evaluation(self, mock_parser_class, mock_load_questions, mock_eval_ollama):
        """Test main function with Ollama model evaluation."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model_path = "llama2"
        mock_args.eval_file = "questions.json"
        mock_args.backend = "ollama"
        mock_args.compare_with = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock data
        questions = ["Q1", "Q2"]
        predictions = [Mock(), Mock()]

        mock_load_questions.return_value = questions
        mock_eval_ollama.return_value = predictions

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_eval_ollama.assert_called_once_with("llama2", questions)

    @patch("evaluation.eightball_eval.argparse.ArgumentParser")
    def test_main_invalid_backend(self, mock_parser_class):
        """Test main function with invalid backend."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "invalid_backend"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        with pytest.raises(SystemExit):
            main()

    @patch("evaluation.eightball_eval.load_eval_questions")
    @patch("evaluation.eightball_eval.argparse.ArgumentParser")
    def test_main_missing_eval_file(self, mock_parser_class, mock_load_questions):
        """Test main function with missing evaluation file."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.eval_file = "nonexistent.json"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_questions.side_effect = FileNotFoundError

        with pytest.raises(SystemExit):
            main()


class TestEightBallIntegration:
    """Test integration of 8-ball evaluation components."""

    def test_eight_ball_answer_consistency(self):
        """Test that answers and token mappings are consistent."""
        # Every answer should have a corresponding token ID
        for answer in EIGHT_BALL_ANSWERS:
            # Find the token ID for this answer
            token_ids = [tid for tid, ans in ID_TO_ANSWER.items() if ans == answer]
            assert len(token_ids) == 1, f"Answer '{answer}' should have exactly one token ID"

            token_id = token_ids[0]
            assert token_id in EIGHT_BALL_TOKEN_IDS

    def test_prediction_result_validation(self):
        """Test that prediction results are properly validated."""
        # Valid result
        valid_result = PredictionResult(
            question="Test?",
            predicted_token=200,
            predicted_answer="It is certain",
            confidence=0.8,
            is_correct=True,
        )

        assert valid_result.predicted_token in EIGHT_BALL_TOKEN_IDS
        assert valid_result.predicted_answer in EIGHT_BALL_ANSWERS
        assert 0.0 <= valid_result.confidence <= 1.0

    def test_evaluation_metrics_calculation(self):
        """Test that evaluation metrics are calculated correctly."""
        # Create test predictions
        predictions1 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 201, "It is decidedly so", 0.8, True),
            PredictionResult("Q3", 205, "Outlook good", 0.7, False),
        ]

        predictions2 = [
            PredictionResult("Q1", 200, "It is certain", 0.9, True),
            PredictionResult("Q2", 205, "Outlook good", 0.6, False),
            PredictionResult("Q3", 205, "Outlook good", 0.7, True),
        ]

        metrics = compare_predictions(predictions1, predictions2)

        # Check basic metrics
        assert metrics.total_predictions == 3
        assert metrics.correct_predictions == 2  # Q1 and Q3 match
        assert abs(metrics.accuracy - (2.0 / 3.0)) < 0.001

        # Check distributions
        assert len(metrics.token_distribution) == 20
        assert sum(metrics.token_distribution) == 3  # Total predictions

        # Check answer distribution
        assert "It is certain" in metrics.answer_distribution
        assert "Outlook good" in metrics.answer_distribution

    def test_evaluation_workflow(self, tmp_path):
        """Test complete evaluation workflow."""
        # Create test questions file
        questions = ["Will it work?", "Should I proceed?", "Will I succeed?"]
        questions_file = tmp_path / "test_questions.json"

        with open(questions_file, "w") as f:
            json.dump(questions, f)

        # Test loading questions
        loaded_questions = load_eval_questions(questions_file)
        assert loaded_questions == questions

        # Test with mock model evaluation
        with patch("evaluation.eightball_eval.evaluate_pytorch_model") as mock_eval:
            mock_predictions = [
                PredictionResult(q, 200 + i, EIGHT_BALL_ANSWERS[i], 0.8, True)
                for i, q in enumerate(questions)
            ]
            mock_eval.return_value = mock_predictions

            results = evaluate_pytorch_model("dummy_model", questions)

            assert len(results) == len(questions)
            for i, result in enumerate(results):
                assert result.question == questions[i]
                assert result.predicted_token in EIGHT_BALL_TOKEN_IDS
                assert result.predicted_answer in EIGHT_BALL_ANSWERS
