"""
Integration tests for evaluation/classification_eval.py - Real classification evaluation.

Tests that actually exercise the evaluation logic using toy models and real configs,
providing meaningful coverage instead of just mocking everything.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np

import pytest

from evaluation.classification_eval import (
    load_classification_config,
    evaluate_pytorch_model,
    ClassificationConfig,
    PredictionResult,
)


class ToyClassificationModel:
    """Simple toy model that predicts specific token IDs for classification."""

    def __init__(self, vocab_size=1000, target_token_id=300):
        self.vocab_size = vocab_size
        self.target_token_id = target_token_id

    def __call__(self, input_ids):
        """Mock forward pass that favors the target token ID."""
        import numpy as np

        batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1

        # Create logits that favor the target token
        logits = np.random.randn(
            batch_size, self.vocab_size).astype(np.float32)

        # Boost the target token and nearby tokens (for classification range)
        logits[:, self.target_token_id] += 5.0  # Strong boost for target
        # Boost neighboring tokens to simulate classification distribution
        for offset in [-2, -1, 1, 2]:
            token_id = self.target_token_id + offset
            if 0 <= token_id < self.vocab_size:
                logits[:, token_id] += 2.0

        return type('MockOutput', (), {'logits': logits})()

    def eval(self):
        """Mock eval method."""
        pass

    def to(self, device):
        """Mock to method."""
        return self


class TestClassificationEvalIntegration:
    """Integration tests that actually exercise real evaluation logic."""

    @pytest.fixture
    def toy_model(self):
        """Create a simple toy classification model."""
        return ToyClassificationModel(vocab_size=1000, target_token_id=300)

    @pytest.fixture
    def toy_tokenizer(self):
        """Create a mock tokenizer that returns reasonable token IDs."""
        tokenizer = Mock()

        def tokenize_side_effect(text):
            # Simple tokenization: split by spaces and map to IDs
            tokens = text.lower().split()
            # Map common words to reasonable token IDs
            token_map = {
                'what': 100, 'is': 101, 'the': 102, 'weather': 103,
                'like': 104, 'today': 105, 'should': 106, 'i': 107,
                'buy': 108, 'stocks': 109, 'test': 110, 'question': 111
            }
            token_ids = [token_map.get(token, 50) for token in tokens]
            return {
                'input_ids': np.array([token_ids]),
                'attention_mask': np.array([[1] * len(token_ids)])
            }

        tokenizer.side_effect = tokenize_side_effect
        return tokenizer

    def test_load_classification_config_from_json_file(self, tmp_path):
        """Test load_classification_config with real JSON file."""
        config_data = {
            "name": "test_classifier",
            "class_names": ["positive", "negative", "neutral"],
            "token_ids": [300, 301, 302]
        }

        json_file = tmp_path / "config.json"
        with open(json_file, "w") as f:
            json.dump(config_data, f)

        config = load_classification_config(str(json_file))

        assert isinstance(config, ClassificationConfig)
        assert config.name == "test_classifier"
        assert config.class_names == ["positive", "negative", "neutral"]
        assert config.token_ids == [300, 301, 302]
        assert config.id_to_name[300] == "positive"
        assert config.name_to_id["negative"] == 301

    def test_load_classification_config_from_yaml_file(self, tmp_path):
        """Test load_classification_config with real YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        config_data = {
            "name": "yaml_classifier",
            "class_names": ["yes", "no", "maybe"],
            "token_ids": [400, 401, 402]
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.safe_dump(config_data, f)

        config = load_classification_config(str(yaml_file))

        assert isinstance(config, ClassificationConfig)
        assert config.name == "yaml_classifier"
        assert config.class_names == ["yes", "no", "maybe"]

    def test_load_classification_config_from_module_path(self, tmp_path):
        """Test load_classification_config from module path."""
        # Create a temporary module file
        module_content = '''
from evaluation.classification_eval import ClassificationConfig

TEST_CONFIG = ClassificationConfig(
    name="module_test",
    class_names=["hot", "cold", "warm"],
    token_ids=[500, 501, 502],
)
'''

        module_file = tmp_path / "test_module.py"
        with open(module_file, "w") as f:
            f.write(module_content)

        # Add to path temporarily
        import sys
        sys.path.insert(0, str(tmp_path))

        try:
            config = load_classification_config("test_module.TEST_CONFIG")
            assert isinstance(config, ClassificationConfig)
            assert config.name == "module_test"
            assert config.class_names == ["hot", "cold", "warm"]
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_classification_config_error_cases(self, tmp_path):
        """Test error handling in load_classification_config."""
        # Test nonexistent file
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_classification_config(str(nonexistent))

        # Test invalid JSON
        invalid_json = tmp_path / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("invalid json content {")

        with pytest.raises(json.JSONDecodeError):
            load_classification_config(str(invalid_json))

        # Test missing required fields
        incomplete_json = tmp_path / "incomplete.json"
        with open(incomplete_json, "w") as f:
            json.dump({"name": "test"}, f)

        with pytest.raises(KeyError):
            load_classification_config(str(incomplete_json))

    @patch('evaluation.classification_eval.AutoModelForCausalLM')
    @patch('evaluation.classification_eval.AutoTokenizer')
    def test_evaluate_pytorch_model_real_execution(self, mock_tokenizer_class, mock_model_class, toy_model, toy_tokenizer):
        """Test evaluate_pytorch_model with real model execution."""
        # Set up mocks to use our toy implementations
        mock_model_class.from_pretrained.return_value = toy_model
        mock_tokenizer_class.from_pretrained.return_value = toy_tokenizer

        # Create a test config
        config = ClassificationConfig(
            name="test_eval",
            class_names=["positive", "negative", "neutral"],
            token_ids=[300, 301, 302],
        )

        questions = ["What is the weather like today?", "Should I buy stocks?"]

        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            # For this test, just create an empty file since we're mocking the loading

        try:
            results = evaluate_pytorch_model(model_path, questions, config)

            # Verify results structure
            assert len(results) == 2
            assert all(isinstance(r, PredictionResult) for r in results)

            # Check that predictions are in the classification range
            for result in results:
                assert result.predicted_class_id in [300, 301, 302]
                assert result.predicted_class_name in [
                    "positive", "negative", "neutral"]
                assert 0.0 <= result.class_probabilities.sum() <= 1.0  # Should be normalized
                assert result.class_probabilities.shape == (3,)  # 3 classes

        finally:
            Path(model_path).unlink()

    @patch('evaluation.classification_eval.AutoModelForCausalLM')
    @patch('evaluation.classification_eval.AutoTokenizer')
    def test_evaluate_pytorch_model_empty_questions(self, mock_tokenizer_class, mock_model_class):
        """Test evaluate_pytorch_model with empty questions list."""
        mock_model_class.from_pretrained.return_value = Mock()
        mock_tokenizer_class.from_pretrained.return_value = Mock()

        config = ClassificationConfig(
            name="test",
            class_names=["a", "b"],
            token_ids=[100, 101],
        )

        results = evaluate_pytorch_model("dummy_model", [], config)
        assert results == []

    @patch('evaluation.classification_eval.AutoModelForCausalLM')
    @patch('evaluation.classification_eval.AutoTokenizer')
    def test_evaluate_pytorch_model_custom_tokenizer(self, mock_tokenizer_class, mock_model_class, toy_model, toy_tokenizer):
        """Test evaluate_pytorch_model with custom tokenizer path."""
        mock_model_class.from_pretrained.return_value = toy_model
        mock_tokenizer_class.from_pretrained.return_value = toy_tokenizer

        config = ClassificationConfig(
            name="test",
            class_names=["yes", "no"],
            token_ids=[200, 201],
        )

        questions = ["Test question?"]

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as model_f:
            model_path = model_f.name

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tokenizer_f:
            tokenizer_path = tokenizer_f.name

        try:
            results = evaluate_pytorch_model(
                model_path, tokenizer_path, questions, config)

            assert len(results) == 1
            assert isinstance(results[0], PredictionResult)
            assert results[0].predicted_class_id in [200, 201]

        finally:
            Path(model_path).unlink()
            Path(tokenizer_path).unlink()

    def test_classification_config_mappings(self):
        """Test ClassificationConfig id/name mappings."""
        config = ClassificationConfig(
            name="mapping_test",
            class_names=["red", "green", "blue"],
            token_ids=[10, 20, 30],
        )

        # Test id_to_name mapping
        assert config.id_to_name[10] == "red"
        assert config.id_to_name[20] == "green"
        assert config.id_to_name[30] == "blue"

        # Test name_to_id mapping
        assert config.name_to_id["red"] == 10
        assert config.name_to_id["green"] == 20
        assert config.name_to_id["blue"] == 30

    def test_prediction_result_structure(self):
        """Test PredictionResult data structure."""
        result = PredictionResult(
            question="Test question?",
            predicted_class_id=42,
            predicted_class_name="test_class",
            class_probabilities=np.array([0.1, 0.3, 0.6])
        )

        assert result.question == "Test question?"
        assert result.predicted_class_id == 42
        assert result.predicted_class_name == "test_class"
        assert np.allclose(result.class_probabilities, [0.1, 0.3, 0.6])

    @patch('evaluation.classification_eval.AutoModelForCausalLM')
    @patch('evaluation.classification_eval.AutoTokenizer')
    def test_evaluate_pytorch_model_error_handling(self, mock_tokenizer_class, mock_model_class):
        """Test error handling in evaluate_pytorch_model."""
        # Test missing config
        with pytest.raises(TypeError, match="missing required argument: config"):
            evaluate_pytorch_model("model.pt", ["question"])

        # Test transformers not available
        with patch('evaluation.classification_eval.AutoModelForCausalLM', None):
            config = ClassificationConfig(
                name="test",
                class_names=["a"],
                token_ids=[100],
            )
            with pytest.raises(ImportError, match="transformers library required"):
                evaluate_pytorch_model("model.pt", ["question"], config)

    def test_load_classification_config_edge_cases(self, tmp_path):
        """Test edge cases for load_classification_config."""
        # Test file path that looks like module path but doesn't exist
        with pytest.raises(FileNotFoundError):
            load_classification_config("nonexistent.json")

        # Test invalid module path
        with pytest.raises(ValueError):
            load_classification_config("invalid.module.path")
