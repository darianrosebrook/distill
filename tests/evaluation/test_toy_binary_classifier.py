"""
Tests for evaluation/toy/binary_classifier.py - Binary classifier toy model evaluation.

Tests binary classifier configuration, mappings, and question loading.
"""
# @author: @darianrosebrook

import json


from evaluation.toy.binary_classifier import (
    BINARY_ANSWERS,
    BINARY_TOKEN_START,
    BINARY_TOKEN_END,
    BINARY_TOKEN_IDS,
    ID_TO_BINARY_ANSWER,
    BINARY_ANSWER_TO_ID,
    BINARY_CLASSIFIER_CONFIG,
    load_binary_eval_questions,
    get_binary_questions,
)


class TestBinaryClassifierConstants:
    """Test binary classifier constants and mappings."""

    def test_binary_answers_count(self):
        """Test that we have 2 binary answers."""
        assert len(BINARY_ANSWERS) == 2
        assert "YES" in BINARY_ANSWERS
        assert "NO" in BINARY_ANSWERS

    def test_binary_token_range(self):
        """Test token ID range."""
        assert BINARY_TOKEN_START == 300
        assert BINARY_TOKEN_END == 301
        assert len(BINARY_TOKEN_IDS) == 2
        assert BINARY_TOKEN_IDS == [300, 301]

    def test_id_to_binary_answer_mapping(self):
        """Test ID to answer mapping."""
        assert len(ID_TO_BINARY_ANSWER) == 2
        assert ID_TO_BINARY_ANSWER[300] == "YES"
        assert ID_TO_BINARY_ANSWER[301] == "NO"

    def test_binary_answer_to_id_mapping(self):
        """Test answer to ID mapping."""
        assert len(BINARY_ANSWER_TO_ID) == 2
        assert BINARY_ANSWER_TO_ID["YES"] == 300
        assert BINARY_ANSWER_TO_ID["NO"] == 301

    def test_mapping_consistency(self):
        """Test that mappings are consistent."""
        for token_id in BINARY_TOKEN_IDS:
            answer = ID_TO_BINARY_ANSWER[token_id]
            assert BINARY_ANSWER_TO_ID[answer] == token_id


class TestBinaryClassifierConfig:
    """Test BINARY_CLASSIFIER_CONFIG."""

    def test_binary_classifier_config_creation(self):
        """Test BINARY_CLASSIFIER_CONFIG creation."""
        assert BINARY_CLASSIFIER_CONFIG.name == "binary-classifier"
        assert len(BINARY_CLASSIFIER_CONFIG.class_names) == 2
        assert len(BINARY_CLASSIFIER_CONFIG.token_ids) == 2
        assert len(BINARY_CLASSIFIER_CONFIG.id_to_name) == 2
        assert len(BINARY_CLASSIFIER_CONFIG.name_to_id) == 2

    def test_binary_classifier_config_mappings(self):
        """Test that config mappings match constants."""
        assert BINARY_CLASSIFIER_CONFIG.id_to_name == ID_TO_BINARY_ANSWER
        assert BINARY_CLASSIFIER_CONFIG.name_to_id == BINARY_ANSWER_TO_ID
        assert BINARY_CLASSIFIER_CONFIG.class_names == BINARY_ANSWERS
        assert BINARY_CLASSIFIER_CONFIG.token_ids == BINARY_TOKEN_IDS


class TestLoadBinaryEvalQuestions:
    """Test load_binary_eval_questions function."""

    def test_load_binary_eval_questions_existing_file(self, tmp_path):
        """Test loading questions from existing file."""
        eval_file = tmp_path / "binary_questions.json"
        questions = ["Question 1", "Question 2"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = load_binary_eval_questions(eval_file)
        assert loaded == questions

    def test_load_binary_eval_questions_nonexistent_file(self, tmp_path):
        """Test loading questions when file doesn't exist (creates default)."""
        eval_file = tmp_path / "nonexistent_binary.json"
        assert not eval_file.exists()

        loaded = load_binary_eval_questions(eval_file)
        assert len(loaded) == 10
        assert eval_file.exists()

        # Verify file was created with default questions
        with open(eval_file) as f:
            data = json.load(f)
        assert "questions" in data
        assert len(data["questions"]) == 10

    def test_load_binary_eval_questions_default_questions_format(self, tmp_path):
        """Test that default questions have correct format."""
        eval_file = tmp_path / "default_binary.json"
        loaded = load_binary_eval_questions(eval_file)

        # All questions should contain "EVIDENCE:" and "QUESTION:"
        for question in loaded:
            assert "EVIDENCE:" in question
            assert "QUESTION:" in question
            assert "ANSWER (YES or NO):" in question

    def test_load_binary_eval_questions_empty_questions(self, tmp_path):
        """Test loading file with empty questions list."""
        eval_file = tmp_path / "empty_binary.json"
        with open(eval_file, "w") as f:
            json.dump({"questions": []}, f)

        loaded = load_binary_eval_questions(eval_file)
        assert loaded == []


class TestGetBinaryQuestions:
    """Test get_binary_questions function."""

    def test_get_binary_questions_custom_path(self, tmp_path):
        """Test getting questions with custom path."""
        eval_file = tmp_path / "custom_binary.json"
        questions = ["Custom question 1", "Custom question 2"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = get_binary_questions(eval_file)
        assert loaded == questions

    def test_get_binary_questions_creates_default(self, tmp_path):
        """Test that get_binary_questions creates default if file doesn't exist."""
        eval_file = tmp_path / "new_binary.json"
        assert not eval_file.exists()

        loaded = get_binary_questions(eval_file)
        assert len(loaded) == 10
        assert eval_file.exists()


class TestBinaryClassifierIntegration:
    """Test integration of binary classifier components."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow from config to questions."""
        # Get config
        config = BINARY_CLASSIFIER_CONFIG
        assert config.name == "binary-classifier"

        # Get questions
        eval_file = tmp_path / "workflow_binary.json"
        questions = get_binary_questions(eval_file)
        assert len(questions) > 0

        # Verify we can map answers
        for token_id in config.token_ids:
            answer = config.id_to_name[token_id]
            assert config.name_to_id[answer] == token_id

    def test_answer_consistency(self):
        """Test that all mappings are consistent."""
        # Test forward mapping
        for token_id, answer in ID_TO_BINARY_ANSWER.items():
            assert BINARY_ANSWER_TO_ID[answer] == token_id

        # Test reverse mapping
        for answer, token_id in BINARY_ANSWER_TO_ID.items():
            assert ID_TO_BINARY_ANSWER[token_id] == answer







