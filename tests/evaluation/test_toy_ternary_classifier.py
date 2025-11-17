"""
Tests for evaluation/toy/ternary_classifier.py - Ternary classifier toy model evaluation.

Tests ternary classifier configuration, mappings, and question loading.
"""
# @author: @darianrosebrook

import json


from evaluation.toy.ternary_classifier import (
    TERNARY_ANSWERS,
    TERNARY_TOKEN_START,
    TERNARY_TOKEN_END,
    TERNARY_TOKEN_IDS,
    ID_TO_TERNARY_ANSWER,
    TERNARY_ANSWER_TO_ID,
    TERNARY_CLASSIFIER_CONFIG,
    load_ternary_eval_questions,
    get_ternary_questions,
)


class TestTernaryClassifierConstants:
    """Test ternary classifier constants and mappings."""

    def test_ternary_answers_count(self):
        """Test that we have 3 ternary answers."""
        assert len(TERNARY_ANSWERS) == 3
        assert "YES" in TERNARY_ANSWERS
        assert "NO" in TERNARY_ANSWERS
        assert "UNCERTAIN" in TERNARY_ANSWERS

    def test_ternary_token_range(self):
        """Test token ID range."""
        assert TERNARY_TOKEN_START == 400
        assert TERNARY_TOKEN_END == 402
        assert len(TERNARY_TOKEN_IDS) == 3
        assert TERNARY_TOKEN_IDS == [400, 401, 402]

    def test_id_to_ternary_answer_mapping(self):
        """Test ID to answer mapping."""
        assert len(ID_TO_TERNARY_ANSWER) == 3
        assert ID_TO_TERNARY_ANSWER[400] == "YES"
        assert ID_TO_TERNARY_ANSWER[401] == "NO"
        assert ID_TO_TERNARY_ANSWER[402] == "UNCERTAIN"

    def test_ternary_answer_to_id_mapping(self):
        """Test answer to ID mapping."""
        assert len(TERNARY_ANSWER_TO_ID) == 3
        assert TERNARY_ANSWER_TO_ID["YES"] == 400
        assert TERNARY_ANSWER_TO_ID["NO"] == 401
        assert TERNARY_ANSWER_TO_ID["UNCERTAIN"] == 402

    def test_mapping_consistency(self):
        """Test that mappings are consistent."""
        for token_id in TERNARY_TOKEN_IDS:
            answer = ID_TO_TERNARY_ANSWER[token_id]
            assert TERNARY_ANSWER_TO_ID[answer] == token_id


class TestTernaryClassifierConfig:
    """Test TERNARY_CLASSIFIER_CONFIG."""

    def test_ternary_classifier_config_creation(self):
        """Test TERNARY_CLASSIFIER_CONFIG creation."""
        assert TERNARY_CLASSIFIER_CONFIG.name == "ternary-classifier"
        assert len(TERNARY_CLASSIFIER_CONFIG.class_names) == 3
        assert len(TERNARY_CLASSIFIER_CONFIG.token_ids) == 3
        assert len(TERNARY_CLASSIFIER_CONFIG.id_to_name) == 3
        assert len(TERNARY_CLASSIFIER_CONFIG.name_to_id) == 3

    def test_ternary_classifier_config_mappings(self):
        """Test that config mappings match constants."""
        assert TERNARY_CLASSIFIER_CONFIG.id_to_name == ID_TO_TERNARY_ANSWER
        assert TERNARY_CLASSIFIER_CONFIG.name_to_id == TERNARY_ANSWER_TO_ID
        assert TERNARY_CLASSIFIER_CONFIG.class_names == TERNARY_ANSWERS
        assert TERNARY_CLASSIFIER_CONFIG.token_ids == TERNARY_TOKEN_IDS


class TestLoadTernaryEvalQuestions:
    """Test load_ternary_eval_questions function."""

    def test_load_ternary_eval_questions_existing_file(self, tmp_path):
        """Test loading questions from existing file."""
        eval_file = tmp_path / "ternary_questions.json"
        questions = ["Question 1", "Question 2"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = load_ternary_eval_questions(eval_file)
        assert loaded == questions

    def test_load_ternary_eval_questions_nonexistent_file(self, tmp_path):
        """Test loading questions when file doesn't exist (creates default)."""
        eval_file = tmp_path / "nonexistent_ternary.json"
        assert not eval_file.exists()

        loaded = load_ternary_eval_questions(eval_file)
        assert len(loaded) == 20
        assert eval_file.exists()

        # Verify file was created with default questions
        with open(eval_file) as f:
            data = json.load(f)
        assert "questions" in data
        assert len(data["questions"]) == 20

    def test_load_ternary_eval_questions_default_questions_format(self, tmp_path):
        """Test that default questions have correct format."""
        eval_file = tmp_path / "default_ternary.json"
        loaded = load_ternary_eval_questions(eval_file)

        # All questions should contain "EVIDENCE:" and "QUESTION:"
        for question in loaded:
            assert "EVIDENCE:" in question
            assert "QUESTION:" in question
            assert "ANSWER (YES or NO or UNCERTAIN):" in question

    def test_load_ternary_eval_questions_empty_questions(self, tmp_path):
        """Test loading file with empty questions list."""
        eval_file = tmp_path / "empty_ternary.json"
        with open(eval_file, "w") as f:
            json.dump({"questions": []}, f)

        loaded = load_ternary_eval_questions(eval_file)
        assert loaded == []


class TestGetTernaryQuestions:
    """Test get_ternary_questions function."""

    def test_get_ternary_questions_custom_path(self, tmp_path):
        """Test getting questions with custom path."""
        eval_file = tmp_path / "custom_ternary.json"
        questions = ["Custom question 1", "Custom question 2"]
        with open(eval_file, "w") as f:
            json.dump({"questions": questions}, f)

        loaded = get_ternary_questions(eval_file)
        assert loaded == questions

    def test_get_ternary_questions_creates_default(self, tmp_path):
        """Test that get_ternary_questions creates default if file doesn't exist."""
        eval_file = tmp_path / "new_ternary.json"
        assert not eval_file.exists()

        loaded = get_ternary_questions(eval_file)
        assert len(loaded) == 20
        assert eval_file.exists()


class TestTernaryClassifierIntegration:
    """Test integration of ternary classifier components."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow from config to questions."""
        # Get config
        config = TERNARY_CLASSIFIER_CONFIG
        assert config.name == "ternary-classifier"

        # Get questions
        eval_file = tmp_path / "workflow_ternary.json"
        questions = get_ternary_questions(eval_file)
        assert len(questions) > 0

        # Verify we can map answers
        for token_id in config.token_ids:
            answer = config.id_to_name[token_id]
            assert config.name_to_id[answer] == token_id

    def test_answer_consistency(self):
        """Test that all mappings are consistent."""
        # Test forward mapping
        for token_id, answer in ID_TO_TERNARY_ANSWER.items():
            assert TERNARY_ANSWER_TO_ID[answer] == token_id

        # Test reverse mapping
        for answer, token_id in TERNARY_ANSWER_TO_ID.items():
            assert ID_TO_TERNARY_ANSWER[token_id] == answer

    def test_uncertain_answer(self):
        """Test that UNCERTAIN is a valid answer."""
        assert "UNCERTAIN" in TERNARY_ANSWERS
        assert TERNARY_ANSWER_TO_ID["UNCERTAIN"] == 402
        assert ID_TO_TERNARY_ANSWER[402] == "UNCERTAIN"







