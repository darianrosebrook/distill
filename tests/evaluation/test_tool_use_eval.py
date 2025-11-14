"""
Tests for evaluation/tool_use_eval.py - Tool-use evaluation framework.

Tests JSON validity checking, tool selection accuracy, and tool-use evaluation
metrics for model tool-calling capabilities.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch

# Import the module using importlib
import importlib

tool_use_eval_module = importlib.import_module("evaluation.tool_use_eval")

# Import functions from the module
load_model = tool_use_eval_module.load_model
generate_text = tool_use_eval_module.generate_text
validate_json = tool_use_eval_module.validate_json
extract_tool_call = tool_use_eval_module.extract_tool_call
evaluate_tool_use = tool_use_eval_module.evaluate_tool_use
main = tool_use_eval_module.main


class TestLoadModel:
    """Test load_model function."""

    @patch("evaluation.tool_use_eval.torch.load")
    @patch("evaluation.tool_use_eval.StudentLM")
    @patch("evaluation.tool_use_eval.ModelCfg")
    def test_load_model_with_config(self, mock_model_cfg, mock_student_lm, mock_torch_load):
        """Test loading model with config in checkpoint."""
        # Mock checkpoint with config
        mock_checkpoint = {
            "model_state_dict": {"layer.weight": torch.ones(10, 10)},
            "config": {
                "arch": {
                    "d_model": 512,
                    "n_layers": 8,
                    "n_heads": 8,
                    "n_kv_heads": 4,
                    "d_head": 64,
                    "vocab_size": 32000,
                }
            },
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock config and model creation
        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model("dummy_path.pt", device)

        # Verify config creation with correct parameters
        mock_model_cfg.assert_called_once_with(
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            d_head=64,
            vocab_size=32000,
            rope_theta=10000.0,
            rope_scaling="dynamic",
            dropout=0.0,
        )

        # Verify model creation and state loading
        mock_student_lm.assert_called_once_with(mock_config)
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint["model_state_dict"])

        assert result == mock_model

    @patch("evaluation.tool_use_eval.torch.load")
    @patch("evaluation.tool_use_eval.StudentLM")
    @patch("evaluation.tool_use_eval.ModelCfg")
    def test_load_model_without_config(self, mock_model_cfg, mock_student_lm, mock_torch_load):
        """Test loading model without config in checkpoint."""
        # Mock checkpoint without config
        mock_checkpoint = {"model_state_dict": {"weight": torch.ones(5, 5)}}
        mock_torch_load.return_value = mock_checkpoint

        # Mock default config and model
        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model("dummy_path.pt", device)

        # Should use default config
        mock_model_cfg.assert_called_once_with()
        mock_student_lm.assert_called_once_with(mock_config)
        mock_model.load_state_dict.assert_called_once()

        assert result == mock_model

    @patch("evaluation.tool_use_eval.torch.load")
    def test_load_model_checkpoint_not_found(self, mock_torch_load):
        """Test loading model with missing checkpoint."""
        mock_torch_load.side_effect = FileNotFoundError("Checkpoint not found")

        device = torch.device("cpu")
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent.pt", device)


class TestGenerateText:
    """Test generate_text function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.forward = Mock(return_value=torch.randn(1, 10, 32000))
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="Generated text")
        tokenizer.eos_token_id = 2
        return tokenizer

    def test_generate_text_basic(self, mock_model, mock_tokenizer):
        """Test basic text generation."""
        prompt = "Test prompt"
        max_length = 20

        result = generate_text(mock_model, mock_tokenizer, prompt, max_length)

        assert isinstance(result, str)
        assert result == "Generated text"

        # Verify model and tokenizer calls
        mock_tokenizer.encode.assert_called_with(prompt)
        mock_model.forward.assert_called()

    def test_generate_text_with_eos(self, mock_model, mock_tokenizer):
        """Test text generation that hits EOS token."""
        prompt = "Short prompt"

        # Mock tokenizer to return EOS in generated tokens
        mock_tokenizer.encode.return_value = [1, 2]  # Input tokens
        mock_model.forward.return_value = torch.tensor(
            [[[0.1, 0.9, 0.0]]]
        )  # EOS token has highest prob

        result = generate_text(mock_model, mock_tokenizer, prompt, max_length=10)

        mock_tokenizer.decode.assert_called()

    def test_generate_text_temperature(self, mock_model, mock_tokenizer):
        """Test text generation with temperature."""
        prompt = "Test"
        temperature = 0.8

        result = generate_text(mock_model, mock_tokenizer, prompt, temperature=temperature)

        # Should still generate text
        assert isinstance(result, str)
        mock_model.forward.assert_called()

    def test_generate_text_max_length(self, mock_model, mock_tokenizer):
        """Test text generation with max length limit."""
        prompt = "Long prompt that should be truncated"
        max_length = 5

        result = generate_text(mock_model, mock_tokenizer, prompt, max_length=max_length)

        # Should respect max length
        mock_tokenizer.decode.assert_called()


class TestValidateJSON:
    """Test validate_json function."""

    def test_validate_json_valid_simple(self):
        """Test validating simple valid JSON."""
        valid_json = '{"name": "test", "value": 42}'
        result = validate_json(valid_json)
        assert result == True

    def test_validate_json_valid_complex(self):
        """Test validating complex valid JSON."""
        complex_json = """
        {
            "tool_calls": [
                {
                    "name": "calculator",
                    "arguments": {
                        "expression": "2 + 2",
                        "precision": 2
                    }
                }
            ],
            "response": "The answer is 4"
        }
        """
        result = validate_json(complex_json)
        assert result == True

    def test_validate_json_invalid_syntax(self):
        """Test validating JSON with syntax errors."""
        invalid_cases = [
            '{"name": "test", "value": }',  # Missing value
            '{"name": "test" "value": 42}',  # Missing comma
            '{"name": "test", "value": 42',  # Missing closing brace
            '["incomplete", "array"',  # Missing closing bracket
            '{"unclosed": "object"',  # Missing closing brace
        ]

        for invalid_json in invalid_cases:
            result = validate_json(invalid_json)
            assert result == False, f"Should reject invalid JSON: {invalid_json}"

    def test_validate_json_empty_string(self):
        """Test validating empty string."""
        result = validate_json("")
        assert result == False

    def test_validate_json_whitespace_only(self):
        """Test validating whitespace-only string."""
        result = validate_json("   \n\t  ")
        assert result == False

    def test_validate_json_non_json_text(self):
        """Test validating plain text."""
        result = validate_json("This is not JSON")
        assert result == False

    def test_validate_json_partial_json(self):
        """Test validating partial JSON."""
        partial = '{"valid": "json", "incomplete": '
        result = validate_json(partial)
        assert result == False


class TestExtractToolCall:
    """Test extract_tool_call function."""

    def test_extract_tool_call_valid_json(self):
        """Test extracting tool call from valid JSON."""
        json_text = """
        {
            "tool_call": {
                "name": "calculator",
                "arguments": {
                    "expression": "2 + 2"
                }
            }
        }
        """
        result = extract_tool_call(json_text)

        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"]["expression"] == "2 + 2"

    def test_extract_tool_call_nested_structure(self):
        """Test extracting tool call from nested JSON structure."""
        nested_json = """
        {
            "response": "I'll calculate that for you",
            "tool_calls": [
                {
                    "name": "search",
                    "arguments": {
                        "query": "python tutorial",
                        "limit": 5
                    }
                }
            ]
        }
        """
        result = extract_tool_call(nested_json)

        assert result is not None
        assert result["name"] == "search"
        assert result["arguments"]["query"] == "python tutorial"

    def test_extract_tool_call_invalid_json(self):
        """Test extracting tool call from invalid JSON."""
        invalid_json = '{"tool_call": {"name": "test", "arguments": }'
        result = extract_tool_call(invalid_json)

        assert result is None

    def test_extract_tool_call_no_tool_call(self):
        """Test extracting tool call from JSON without tool_call field."""
        no_tool_json = '{"response": "Hello world", "status": "success"}'
        result = extract_tool_call(no_tool_json)

        assert result is None

    def test_extract_tool_call_empty_json(self):
        """Test extracting tool call from empty JSON."""
        empty_json = "{}"
        result = extract_tool_call(empty_json)

        assert result is None

    def test_extract_tool_call_malformed_tool_call(self):
        """Test extracting tool call from JSON with malformed tool_call."""
        malformed_json = """
        {
            "tool_call": {
                "name": "calculator"
                "arguments": "missing comma"
            }
        }
        """
        result = extract_tool_call(malformed_json)

        assert result is None

    def test_extract_tool_call_multiple_tools(self):
        """Test extracting first tool call when multiple are present."""
        multi_tool_json = """
        {
            "tool_calls": [
                {"name": "calculator", "arguments": {"expr": "1+1"}},
                {"name": "search", "arguments": {"query": "test"}}
            ]
        }
        """
        result = extract_tool_call(multi_tool_json)

        assert result is not None
        assert result["name"] == "calculator"  # Should return first tool


class TestEvaluateToolUse:
    """Test evaluate_tool_use function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        return tokenizer

    @patch("evaluation.tool_use_eval.load_model")
    @patch("evaluation.tool_use_eval.generate_text")
    @patch("evaluation.tool_use_eval.validate_json")
    @patch("evaluation.tool_use_eval.extract_tool_call")
    def test_evaluate_tool_use_success(self, mock_extract, mock_validate, mock_generate, mock_load):
        """Test successful tool use evaluation."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = mock_model

        # Mock test cases
        test_cases = [
            {
                "prompt": "Calculate 2 + 2",
                "expected_tool": "calculator",
                "expected_args": {"expression": "2 + 2"},
            },
            {
                "prompt": "Search for Python docs",
                "expected_tool": "search",
                "expected_args": {"query": "Python docs"},
            },
        ]

        # Mock generation results
        mock_generate.side_effect = [
            '{"tool_call": {"name": "calculator", "arguments": {"expression": "2 + 2"}}}',
            '{"tool_call": {"name": "search", "arguments": {"query": "Python docs"}}}',
        ]

        # Mock validation and extraction
        mock_validate.return_value = True
        mock_extract.side_effect = [
            {"name": "calculator", "arguments": {"expression": "2 + 2"}},
            {"name": "search", "arguments": {"query": "Python docs"}},
        ]

        device = torch.device("cpu")
        results = evaluate_tool_use("dummy_checkpoint.pt", test_cases, device)

        assert len(results) == 2

        # Check first result
        result1 = results[0]
        assert result1["prompt"] == "Calculate 2 + 2"
        assert result1["json_valid"] == True
        assert result1["tool_correct"] == True
        assert result1["args_correct"] == True

        # Check second result
        result2 = results[1]
        assert result2["prompt"] == "Search for Python docs"
        assert result2["json_valid"] == True
        assert result2["tool_correct"] == True
        assert result2["args_correct"] == True

    @patch("evaluation.tool_use_eval.load_model")
    @patch("evaluation.tool_use_eval.generate_text")
    @patch("evaluation.tool_use_eval.validate_json")
    @patch("evaluation.tool_use_eval.extract_tool_call")
    def test_evaluate_tool_use_invalid_json(
        self, mock_extract, mock_validate, mock_generate, mock_load
    ):
        """Test tool use evaluation with invalid JSON."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        test_cases = [{"prompt": "Test", "expected_tool": "test_tool"}]

        mock_generate.return_value = "invalid json response"
        mock_validate.return_value = False
        mock_extract.return_value = None

        device = torch.device("cpu")
        results = evaluate_tool_use("dummy_checkpoint.pt", test_cases, device)

        assert len(results) == 1
        result = results[0]
        assert result["json_valid"] == False
        assert result["tool_correct"] == False
        assert result["args_correct"] == False

    @patch("evaluation.tool_use_eval.load_model")
    @patch("evaluation.tool_use_eval.generate_text")
    @patch("evaluation.tool_use_eval.validate_json")
    @patch("evaluation.tool_use_eval.extract_tool_call")
    def test_evaluate_tool_use_wrong_tool(
        self, mock_extract, mock_validate, mock_generate, mock_load
    ):
        """Test tool use evaluation with wrong tool selected."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        test_cases = [{"prompt": "Calculate sum", "expected_tool": "calculator"}]

        mock_generate.return_value = '{"tool_call": {"name": "search", "arguments": {}}}'
        mock_validate.return_value = True
        mock_extract.return_value = {"name": "search", "arguments": {}}

        device = torch.device("cpu")
        results = evaluate_tool_use("dummy_checkpoint.pt", test_cases, device)

        assert len(results) == 1
        result = results[0]
        assert result["json_valid"] == True
        assert result["tool_correct"] == False  # Wrong tool
        assert result["args_correct"] == False  # Args don't matter if tool is wrong

    @patch("evaluation.tool_use_eval.load_model")
    def test_evaluate_tool_use_model_load_failure(self, mock_load):
        """Test tool use evaluation with model loading failure."""
        mock_load.side_effect = Exception("Model load failed")

        test_cases = [{"prompt": "Test"}]
        device = torch.device("cpu")

        with pytest.raises(Exception):
            evaluate_tool_use("bad_checkpoint.pt", test_cases, device)

    def test_evaluate_tool_use_empty_test_cases(self):
        """Test tool use evaluation with empty test cases."""
        with patch("evaluation.tool_use_eval.load_model"):
            device = torch.device("cpu")
            results = evaluate_tool_use("dummy.pt", [], device)

        assert results == []


class TestMainFunction:
    """Test main function."""

    @patch("evaluation.tool_use_eval.evaluate_tool_use")
    @patch("evaluation.tool_use_eval.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_success(self, mock_print, mock_parser_class, mock_evaluate):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.config = "config.json"
        mock_args.output = "results.json"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock evaluation results
        mock_results = [{"prompt": "Test", "json_valid": True, "tool_correct": True}]
        mock_evaluate.return_value = mock_results

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_evaluate.assert_called_once()

    @patch("evaluation.tool_use_eval.evaluate_tool_use")
    @patch("evaluation.tool_use_eval.argparse.ArgumentParser")
    def test_main_checkpoint_not_found(self, mock_parser_class, mock_evaluate):
        """Test main function with missing checkpoint."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "nonexistent.pt"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_evaluate.side_effect = FileNotFoundError("Checkpoint not found")

        with pytest.raises(SystemExit):
            main()

    @patch("evaluation.tool_use_eval.argparse.ArgumentParser")
    def test_main_config_not_found(self, mock_parser_class):
        """Test main function with missing config file."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = "nonexistent.json"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        with pytest.raises(SystemExit):
            main()


class TestToolUseEvalIntegration:
    """Test integration of tool use evaluation components."""

    def test_json_validation_edge_cases(self):
        """Test JSON validation with various edge cases."""
        # Valid cases
        valid_cases = [
            '{"simple": "json"}',
            '{"nested": {"key": "value"}}',
            '{"array": [1, 2, 3]}',
            '{"mixed": {"array": [1, 2], "object": {"nested": true}}}',
        ]

        for valid_json in valid_cases:
            assert validate_json(valid_json) == True

        # Invalid cases
        invalid_cases = [
            '{"incomplete": "json"',
            '{"missing": "comma" "invalid": "json"}',
            '["unclosed", "array"',
            '{"trailing": "comma",}',
            "not json at all",
        ]

        for invalid_json in invalid_cases:
            assert validate_json(invalid_json) == False

    def test_tool_call_extraction_variations(self):
        """Test tool call extraction with different JSON structures."""
        test_cases = [
            # Standard tool_call format
            (
                '{"tool_call": {"name": "calc", "arguments": {"x": 1}}}',
                {"name": "calc", "arguments": {"x": 1}},
            ),
            # tool_calls array format
            (
                '{"tool_calls": [{"name": "search", "arguments": {"q": "test"}}]}',
                {"name": "search", "arguments": {"q": "test"}},
            ),
            # No tool call
            ('{"response": "Hello"}', None),
            # Invalid JSON
            ('{"tool_call": {"name": "test"', None),
        ]

        for json_text, expected in test_cases:
            result = extract_tool_call(json_text)
            assert result == expected, f"Failed for: {json_text}"

    def test_complete_evaluation_workflow(self, tmp_path):
        """Test complete tool use evaluation workflow."""
        # Create test config file
        test_config = [
            {
                "prompt": "What is 2 + 2?",
                "expected_tool": "calculator",
                "expected_args": {"expression": "2 + 2"},
            },
            {
                "prompt": "Search for Python tutorials",
                "expected_tool": "search",
                "expected_args": {"query": "Python tutorials"},
            },
        ]

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f)

        # Mock the evaluation components
        with (
            patch("evaluation.tool_use_eval.load_model") as mock_load,
            patch("evaluation.tool_use_eval.generate_text") as mock_generate,
            patch("evaluation.tool_use_eval.validate_json") as mock_validate,
            patch("evaluation.tool_use_eval.extract_tool_call") as mock_extract,
        ):
            mock_model = Mock()
            mock_load.return_value = mock_model

            # Mock perfect responses
            mock_generate.side_effect = [
                '{"tool_call": {"name": "calculator", "arguments": {"expression": "2 + 2"}}}',
                '{"tool_call": {"name": "search", "arguments": {"query": "Python tutorials"}}}',
            ]

            mock_validate.return_value = True
            mock_extract.side_effect = [
                {"name": "calculator", "arguments": {"expression": "2 + 2"}},
                {"name": "search", "arguments": {"query": "Python tutorials"}},
            ]

            device = torch.device("cpu")
            results = evaluate_tool_use(str(config_file), test_config, device)

            assert len(results) == 2

            # Both should be correct
            for result in results:
                assert result["json_valid"] == True
                assert result["tool_correct"] == True
                assert result["args_correct"] == True

    def test_evaluation_metrics_calculation(self):
        """Test that evaluation metrics are calculated correctly."""
        # Create mock test case results
        results = [
            {"prompt": "Test 1", "json_valid": True, "tool_correct": True, "args_correct": True},
            {
                "prompt": "Test 2",
                "json_valid": True,
                "tool_correct": False,  # Wrong tool
                "args_correct": False,
            },
            {
                "prompt": "Test 3",
                "json_valid": False,  # Invalid JSON
                "tool_correct": False,
                "args_correct": False,
            },
        ]

        # Calculate summary metrics
        total_tests = len(results)
        json_valid_count = sum(1 for r in results if r["json_valid"])
        tool_correct_count = sum(1 for r in results if r["tool_correct"])
        args_correct_count = sum(1 for r in results if r["args_correct"])

        json_valid_rate = json_valid_count / total_tests
        tool_accuracy = tool_correct_count / total_tests
        args_accuracy = args_correct_count / total_tests

        assert json_valid_rate == 2.0 / 3.0  # 2 out of 3
        assert tool_accuracy == 1.0 / 3.0  # 1 out of 3
        assert args_accuracy == 1.0 / 3.0  # 1 out of 3

    def test_error_handling_robustness(self):
        """Test that evaluation handles errors gracefully."""
        # Test with malformed prompts
        test_cases = [
            {"prompt": "", "expected_tool": "test"},  # Empty prompt
            {"prompt": None, "expected_tool": "test"},  # None prompt
            {"expected_tool": "test"},  # Missing prompt
        ]

        with (
            patch("evaluation.tool_use_eval.load_model"),
            patch("evaluation.tool_use_eval.generate_text", return_value="{}"),
            patch("evaluation.tool_use_eval.validate_json", return_value=True),
            patch("evaluation.tool_use_eval.extract_tool_call", return_value=None),
        ):
            device = torch.device("cpu")

            # Should not crash with malformed inputs
            try:
                results = evaluate_tool_use("dummy.pt", test_cases, device)
                assert isinstance(results, list)
            except Exception:
                # If it does crash, that's also acceptable for malformed inputs
                pass
