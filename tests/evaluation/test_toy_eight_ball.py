"""
Tests for evaluation/toy/eight_ball.py - 8-Ball toy model evaluation configuration.

Tests 8-ball answer mappings, configuration, and question loading.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.toy.eight_ball import (
    EIGHT_BALL_ANSWERS,
    EIGHT_BALL_TOKEN_START,
    EIGHT_BALL_TOKEN_END,
    EIGHT_BALL_TOKEN_IDS,
    ID_TO_ANSWER,
    ANSWER_TO_ID,
    EIGHT_BALL_CONFIG,
    ClassificationConfig,
    load_eval_questions,
    get_eight_ball_questions,
)


class TestEightBallConstants:
    """Test 8-ball constants and mappings."""

    def test_eight_ball_answers_count(self):
        """Test that we have 20 8-ball answers."""
        assert len(EIGHT_BALL_ANSWERS) == 20

    def test_eight_ball_token_range(self):
        """Test token ID range."""
        assert EIGHT_BALL_TOKEN_START == 200
        assert EIGHT_BALL_TOKEN_END == 219
        assert len(EIGHT_BALL_TOKEN_IDS) == 20

    def test_id_to_answer_mapping(self):
        """Test ID to answer mapping."""
        assert len(ID_TO_ANSWER) == 20
        assert ID_TO_ANSWER[200] == "It is certain"
        assert ID_TO_ANSWER[219] == "Very doubtful"

    def test_answer_to_id_mapping(self):
        """Test answer to ID mapping."""
        assert len(ANSWER_TO_ID) == 20
        assert ANSWER_TO_ID["It is certain"] == 200
        assert ANSWER_TO_ID["Very doubtful"] == 219

    def test_mapping_consistency(self):
        """Test that mappings are consistent."""
        for token_id in EIGHT_BALL_TOKEN_IDS:
            answer = ID_TO_ANSWER[token_id]
            assert ANSWER_TO_ID[answer] == token_id

    def test_all_answers_unique(self):
        """Test that all answers are unique."""
        assert len(EIGHT_BALL_ANSWERS) == len(set(EIGHT_BALL_ANSWERS))


class TestClassificationConfig:
    """Test ClassificationConfig dataclass."""

    def test_eight_ball_config_creation(self):
        """Test EIGHT_BALL_CONFIG creation."""
        assert EIGHT_BALL_CONFIG.name == "8-ball"
        assert len(EIGHT_BALL_CONFIG.class_names) == 20
        assert len(EIGHT_BALL_CONFIG.token_ids) == 20
        assert len(EIGHT_BALL_CONFIG.id_to_name) == 20
        assert len(EIGHT_BALL_CONFIG.name_to_id) == 20

    def test_eight_ball_config_mappings(self):
        """Test that config mappings match constants."""
        assert EIGHT_BALL_CONFIG.id_to_name == ID_TO_ANSWER
        assert EIGHT_BALL_CONFIG.name_to_id == ANSWER_TO_ID
        assert EIGHT_BALL_CONFIG.class_names == EIGHT_BALL_ANSWERS
        assert EIGHT_BALL_CONFIG.token_ids == EIGHT_BALL_TOKEN_IDS

    def test_classification_config_creation(self):
        """Test creating a custom ClassificationConfig."""
        config = ClassificationConfig(
            name="test",
            class_names=["A", "B"],
            token_ids=[100, 101],
            id_to_name={100: "A", 101: "B"},
            name_to_id={"A": 100, "B": 101},
        )
        assert config.name == "test"
        assert len(config.class_names) == 2


class TestLoadEvalQuestions:
    """Test load_eval_questions function."""

    def test_load_eval_questions_existing_file(self, tmp_path):
        """Test loading questions from existing file."""
        eval_file = tmp_path / "questions.json"
        questions = ["Question 1", "Question 2", "Question 3"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = load_eval_questions(eval_file)
        assert loaded == questions

    def test_load_eval_questions_nonexistent_file(self, tmp_path):
        """Test loading questions when file doesn't exist (creates default)."""
        eval_file = tmp_path / "nonexistent.json"
        assert not eval_file.exists()

        loaded = load_eval_questions(eval_file)
        assert len(loaded) == 20
        assert eval_file.exists()

        # Verify file was created with default questions
        with open(eval_file) as f:
            data = json.load(f)
        assert "questions" in data
        assert len(data["questions"]) == 20

    def test_load_eval_questions_empty_questions(self, tmp_path):
        """Test loading file with empty questions list."""
        eval_file = tmp_path / "empty.json"
        with open(eval_file, "w") as f:
            json.dump({"questions": []}, f)

        loaded = load_eval_questions(eval_file)
        assert loaded == []

    def test_load_eval_questions_missing_questions_key(self, tmp_path):
        """Test loading file without questions key."""
        eval_file = tmp_path / "no_questions.json"
        with open(eval_file, "w") as f:
            json.dump({"other": "data"}, f)

        loaded = load_eval_questions(eval_file)
        assert loaded == []


class TestGetEightBallQuestions:
    """Test get_eight_ball_questions function."""

    def test_get_eight_ball_questions_default(self, tmp_path):
        """Test getting questions with default path."""
        default_path = Path("evaluation/8ball_eval_questions.json")
        # This will try to load from default location or create it
        # We'll test with a custom path instead
        pass  # Skip default path test as it may create files in project root

    def test_get_eight_ball_questions_custom_path(self, tmp_path):
        """Test getting questions with custom path."""
        eval_file = tmp_path / "custom_8ball.json"
        questions = ["Custom question 1", "Custom question 2"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = get_eight_ball_questions(eval_file)
        assert loaded == questions

    def test_get_eight_ball_questions_creates_default(self, tmp_path):
        """Test that get_eight_ball_questions creates default if file doesn't exist."""
        eval_file = tmp_path / "new_8ball.json"
        assert not eval_file.exists()

        loaded = get_eight_ball_questions(eval_file)
        assert len(loaded) == 20
        assert eval_file.exists()


class TestEightBallIntegration:
    """Test integration of 8-ball components."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow from config to questions."""
        # Get config
        config = EIGHT_BALL_CONFIG
        assert config.name == "8-ball"

        # Get questions
        eval_file = tmp_path / "workflow_8ball.json"
        questions = get_eight_ball_questions(eval_file)
        assert len(questions) > 0

        # Verify we can map answers
        for token_id in config.token_ids:
            answer = config.id_to_name[token_id]
            assert config.name_to_id[answer] == token_id

    def test_answer_consistency(self):
        """Test that all mappings are consistent."""
        # Test forward mapping
        for token_id, answer in ID_TO_ANSWER.items():
            assert ANSWER_TO_ID[answer] == token_id

        # Test reverse mapping
        for answer, token_id in ANSWER_TO_ID.items():
            assert ID_TO_ANSWER[token_id] == answer

    def test_token_ids_sequential(self):
        """Test that token IDs are sequential."""
        expected_ids = list(range(EIGHT_BALL_TOKEN_START, EIGHT_BALL_TOKEN_END + 1))
        assert EIGHT_BALL_TOKEN_IDS == expected_ids

