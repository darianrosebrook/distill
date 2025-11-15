"""
Integration tests for evaluation/8ball_eval.py - Real 8-Ball model evaluation.

Tests that actually exercise the evaluation logic using controlled inputs,
providing meaningful coverage instead of just mocking everything.
"""
# @author: @darianrosebrook

import json
import numpy as np

import pytest

from evaluation.eightball_eval import (
    load_eval_questions,
    PredictionResult,
    EIGHT_BALL_TOKEN_IDS,
    ID_TO_ANSWER,
)


class TestEightBallEvaluationIntegration:
    """Integration tests that actually exercise real evaluation logic."""

    def test_load_eval_questions_integration(self, tmp_path):
        """Test load_eval_questions with real file operations."""
        questions = [
            "Will it rain tomorrow?",
            "Should I buy stocks?",
            "Will I find true love?"
        ]

        # Test JSON format
        json_file = tmp_path / "questions.json"
        with open(json_file, "w") as f:
            json.dump(questions, f)

        result = load_eval_questions(json_file)
        assert result == questions

        # Test text format
        text_file = tmp_path / "questions.txt"
        with open(text_file, "w") as f:
            f.write("\n".join(questions))

        result = load_eval_questions(text_file)
        assert result == questions

    def test_prediction_result_class_probabilities(self):
        """Test that PredictionResult properly computes class probabilities."""
        # Create a result manually to test the class probabilities logic
        logits = np.random.randn(1000)
        logits[200:220] += 2.0  # Boost 8-ball range
        logits[205] += 3.0  # Boost specific token

        # Simulate what happens in the evaluation
        eight_ball_logits = logits[200:220]  # Extract 8-ball range
        probs = np.exp(eight_ball_logits) / np.exp(eight_ball_logits).sum()
        predicted_token_idx = np.argmax(probs)
        predicted_token = 200 + predicted_token_idx
        confidence = probs[predicted_token_idx]

        # Create PredictionResult
        result = PredictionResult(
            question="Test?",
            predicted_token=predicted_token,
            confidence=confidence,
            class_probabilities=probs
        )

        assert result.predicted_token == predicted_token
        assert result.predicted_answer == ID_TO_ANSWER[predicted_token]
        assert abs(result.confidence - confidence) < 1e-6
        assert result.class_probabilities.shape == (20,)

    def test_prediction_result_token_compatibility(self):
        """Test PredictionResult token/class_id compatibility."""
        # Test that predicted_token and predicted_class_id work interchangeably
        result1 = PredictionResult(
            question="Test?",
            predicted_token=205,
            confidence=0.8
        )

        result2 = PredictionResult(
            question="Test?",
            predicted_class_id=205,
            confidence=0.8
        )

        # Both should have the same final values
        assert result1.predicted_token == 205
        assert result1.predicted_class_id == 205
        assert result2.predicted_token == 205
        assert result2.predicted_class_id == 205

        # Both should resolve to the same answer
        assert result1.predicted_answer == result2.predicted_answer == "Outlook good"

    def test_eight_ball_token_range_coverage(self):
        """Test that all 8-ball tokens are properly mapped."""
        assert len(EIGHT_BALL_TOKEN_IDS) == 20
        assert len(ID_TO_ANSWER) == 20

        # Verify the range
        assert min(EIGHT_BALL_TOKEN_IDS) == 200
        assert max(EIGHT_BALL_TOKEN_IDS) == 219

        # Verify all tokens have answers
        for token_id in EIGHT_BALL_TOKEN_IDS:
            assert token_id in ID_TO_ANSWER
            assert ID_TO_ANSWER[token_id] is not None

        # Test some specific mappings
        assert ID_TO_ANSWER[200] == "It is certain"
        assert ID_TO_ANSWER[205] == "As I see it, yes"
        assert ID_TO_ANSWER[219] == "Very doubtful"

    def test_load_eval_questions_error_cases(self, tmp_path):
        """Test error cases for load_eval_questions."""
        # Test nonexistent file
        nonexistent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_eval_questions(nonexistent)

        # Test invalid JSON
        invalid_json = tmp_path / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("invalid json content {")

        with pytest.raises(json.JSONDecodeError):
            load_eval_questions(invalid_json)

    def test_prediction_result_edge_cases(self):
        """Test PredictionResult edge cases."""
        # Test with unknown token ID
        result = PredictionResult(
            question="Test?",
            predicted_token=999,  # Not in 8-ball range
            confidence=0.5
        )
        assert result.predicted_answer == "Unknown"

        # Test with no token/id provided
        result = PredictionResult(question="Test?", confidence=0.5)
        assert result.predicted_token is None
        assert result.predicted_class_id is None
        assert result.predicted_answer == "Unknown"

    def test_load_eval_questions_empty_file(self, tmp_path):
        """Test load_eval_questions with empty files."""
        # Empty JSON array
        empty_json = tmp_path / "empty.json"
        with open(empty_json, "w") as f:
            json.dump([], f)

        result = load_eval_questions(empty_json)
        assert result == []

        # Empty text file
        empty_text = tmp_path / "empty.txt"
        with open(empty_text, "w") as f:
            f.write("")

        result = load_eval_questions(empty_text)
        assert result == []
