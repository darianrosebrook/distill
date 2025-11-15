"""
Tests for coreml/runtime/generate_coreml.py - CoreML runtime generator.

Tests CoreML model loading, tool call generation, streaming text generation,
constrained decoding, and error handling.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

import pytest
import torch

# Import the module functions
from coreml.runtime.generate_coreml import (
    load_coreml_model,
    generate_tool_call,
    generate_text_streaming,
    main,
)


class TestLoadCoreMLModel:
    """Test CoreML model loading functionality."""

    @patch("coreml.runtime.generate_coreml.ct")
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_load_coreml_model_success(self, mock_ct, tmp_path):
        """Test successful CoreML model loading."""
        # Create a mock .mlpackage file
        mlpackage_path = str(tmp_path / "model.mlpackage")

        mock_model = Mock()
        mock_ct.models.MLModel.return_value = mock_model

        result = load_coreml_model(mlpackage_path)

        assert result == mock_model
        mock_ct.models.MLModel.assert_called_once_with(mlpackage_path)

    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", False)
    def test_load_coreml_model_no_coreml(self):
        """Test loading when CoreML is not available."""
        with pytest.raises(ImportError, match="coremltools not available"):
            load_coreml_model("dummy.mlpackage")

    @patch("coreml.runtime.generate_coreml.ct")
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_load_coreml_model_file_error(self, mock_ct):
        """Test loading with file error."""
        mock_ct.models.MLModel.side_effect = Exception("File not found")

        with pytest.raises(Exception, match="File not found"):
            load_coreml_model("nonexistent.mlpackage")


class TestGenerateToolCall:
    """Test tool call generation functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]], dtype=torch.long))
        tokenizer.decode = Mock(return_value='{"name": "test_tool", "arguments": {"arg": "value"}}')
        return tokenizer

    @pytest.fixture
    def mock_coreml_model(self):
        """Create mock CoreML model."""
        model = Mock()
        model.predict = Mock(return_value={"logits": np.random.randn(1, 3, 1000).astype(np.float32)})
        return model

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.JSONConstrainedDecoder")
    def test_generate_tool_call_coreml_success(
        self, mock_decoder_class, mock_coreml_model, mock_tokenizer
    ):
        """Test successful tool call generation with CoreML model."""
        # Mock constrained decoder
        mock_decoder = Mock()
        mock_decoder.start.return_value = Mock()
        mock_decoder.allowed_token_mask.return_value = np.ones(1000, dtype=bool)
        mock_decoder.push.return_value = Mock(complete=True, buffer='{"name": "test_tool", "arguments": {"arg": "value"}}')
        mock_decoder_class.return_value = mock_decoder

        result = generate_tool_call(
            model=mock_coreml_model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            tools=[{"name": "test_tool"}],
        )

        assert "name" in result
        assert "arguments" in result
        assert result["name"] == "test_tool"

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", False)
    def test_generate_tool_call_pytorch_success(self, mock_tokenizer):
        """Test successful tool call generation with PyTorch model."""
        # Create mock PyTorch model
        model = Mock()
        model.return_value = torch.randn(1, 3, 1000)

        result = generate_tool_call(
            model=model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            tools=[{"name": "test_tool"}],
            device="cpu",
        )

        assert isinstance(result, dict)
        # Should contain parsed tool call from mocked decoder

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", False)
    def test_generate_tool_call_no_torch(self, mock_tokenizer):
        """Test tool call generation when PyTorch is not available."""
        with pytest.raises(ImportError, match="PyTorch required"):
            generate_tool_call(
                model=Mock(),
                tokenizer=mock_tokenizer,
                prompt="Test prompt",
                tools=[{"name": "test_tool"}],
            )

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.JSONConstrainedDecoder")
    def test_generate_tool_call_with_halt_logits(
        self, mock_decoder_class, mock_coreml_model, mock_tokenizer
    ):
        """Test tool call generation with halt logits."""
        # Mock constrained decoder
        mock_decoder = Mock()
        mock_decoder.start.return_value = Mock()
        mock_decoder.allowed_token_mask.return_value = np.ones(1000, dtype=bool)
        mock_decoder.push.return_value = Mock(complete=True, buffer='{"name": "test_tool", "arguments": {"arg": "value"}}')
        mock_decoder_class.return_value = mock_decoder

        # Mock model to return halt logits
        mock_coreml_model.predict.return_value = {
            "logits": np.random.randn(1, 3, 1000).astype(np.float32),
            "halt_logits": np.random.randn(1, 3, 2).astype(np.float32)
        }

        result = generate_tool_call(
            model=mock_coreml_model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            tools=[{"name": "test_tool"}],
            return_halt_logits=True,
        )

        assert "name" in result
        assert "arguments" in result
        assert "halt_logits" in result
        assert isinstance(result["halt_logits"], np.ndarray)


class TestGenerateTextStreaming:
    """Test streaming text generation functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]], dtype=torch.long))
        tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: "generated text")
        tokenizer.eos_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_coreml_model(self):
        """Create mock CoreML model."""
        model = Mock()
        model.predict = Mock(return_value={"logits": np.random.randn(1, 1, 1000).astype(np.float32)})
        return model

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_generate_text_streaming_basic(self, mock_coreml_model, mock_tokenizer):
        """Test basic streaming text generation."""
        def mock_callback(token_text, **kwargs):
            pass

        result = generate_text_streaming(
            model=mock_coreml_model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            max_new_tokens=5,
            stream_callback=mock_callback,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.JSONConstrainedDecoder")
    def test_generate_text_streaming_constrained(self, mock_decoder_class, mock_tokenizer):
        """Test streaming text generation with constrained decoding."""
        # Mock constrained decoder
        mock_decoder = Mock()
        mock_decoder.start.return_value = Mock()
        mock_decoder.allowed_token_mask.return_value = np.ones(1000, dtype=bool)
        mock_decoder.push.return_value = Mock(complete=True, buffer="constrained output")
        mock_decoder_class.return_value = mock_decoder

        # Mock model
        model = Mock()
        model.return_value = torch.randn(1, 1, 1000)

        result = generate_text_streaming(
            model=model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            max_new_tokens=5,
            use_constrained_decoding=True,
            schema={"type": "object"},
        )

        assert isinstance(result, str)

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_generate_text_streaming_with_tools(self, mock_coreml_model, mock_tokenizer):
        """Test streaming text generation with tool integration."""
        result = generate_text_streaming(
            model=mock_coreml_model,
            tokenizer=mock_tokenizer,
            prompt="Test prompt",
            max_new_tokens=5,
            tools=[{"name": "test_tool"}],
        )

        assert isinstance(result, str)


class TestMainFunction:
    """Test main function functionality."""

    @patch("coreml.runtime.generate_coreml.argparse.ArgumentParser")
    @patch("coreml.runtime.generate_coreml.load_coreml_model")
    @patch("coreml.runtime.generate_coreml.generate_tool_call")
    @patch("coreml.runtime.generate_coreml.generate_text_streaming")
    def test_main_tool_call_mode(self, mock_generate_text, mock_generate_tool, mock_load_model, mock_parser_class):
        """Test main function in tool call mode."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = "tool_call"
        mock_args.model = "test.mlpackage"
        mock_args.prompt = "Test prompt"
        mock_args.tools = '[{"name": "test_tool"}]'
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock model and generation
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_generate_tool.return_value = {"name": "test_tool", "arguments": {}}

        main()

        mock_load_model.assert_called_once_with("test.mlpackage")
        mock_generate_tool.assert_called_once()

    @patch("coreml.runtime.generate_coreml.argparse.ArgumentParser")
    @patch("coreml.runtime.generate_coreml.load_coreml_model")
    @patch("coreml.runtime.generate_coreml.generate_text_streaming")
    def test_main_text_generation_mode(self, mock_generate_text, mock_load_model, mock_parser_class):
        """Test main function in text generation mode."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = "text"
        mock_args.model = "test.mlpackage"
        mock_args.prompt = "Test prompt"
        mock_args.max_tokens = 50
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock model and generation
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_generate_text.return_value = "Generated text"

        main()

        mock_load_model.assert_called_once_with("test.mlpackage")
        mock_generate_text.assert_called_once()


class TestErrorHandling:
    """Test error handling in CoreML generation."""

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_generate_tool_call_model_error(self, mock_tokenizer):
        """Test tool call generation with model error."""
        model = Mock()
        model.predict.side_effect = Exception("Model inference failed")

        with pytest.raises(RuntimeError, match="Model inference failed"):
            generate_tool_call(
                model=model,
                tokenizer=mock_tokenizer,
                prompt="Test prompt",
                tools=[{"name": "test_tool"}],
            )

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_generate_text_streaming_tokenizer_error(self, mock_coreml_model):
        """Test text generation with tokenizer error."""
        tokenizer = Mock()
        tokenizer.encode.side_effect = Exception("Tokenizer encoding failed")

        with pytest.raises(RuntimeError, match="Generation failed"):
            generate_text_streaming(
                model=mock_coreml_model,
                tokenizer=tokenizer,
                prompt="Test prompt",
                max_new_tokens=5,
            )


class TestIntegrationScenarios:
    """Test integration scenarios for CoreML generation."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.tensor([[1, 2, 3]], dtype=torch.long))
        tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: "test output")
        tokenizer.eos_token_id = 0
        return tokenizer

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_full_tool_call_workflow(self, mock_tokenizer):
        """Test complete tool call workflow."""
        # Mock PyTorch model for simplicity
        model = Mock()
        model.return_value = torch.randn(1, 1, 1000)

        result = generate_tool_call(
            model=model,
            tokenizer=mock_tokenizer,
            prompt="Please call a tool to help me",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            ],
        )

        assert isinstance(result, dict)

    @patch("coreml.runtime.generate_coreml.TORCH_AVAILABLE", True)
    @patch("coreml.runtime.generate_coreml.COREML_AVAILABLE", True)
    def test_full_text_generation_workflow(self, mock_tokenizer):
        """Test complete text generation workflow."""
        # Mock PyTorch model
        model = Mock()
        model.return_value = torch.randn(1, 1, 1000)

        result = generate_text_streaming(
            model=model,
            tokenizer=mock_tokenizer,
            prompt="Write a short story about",
            max_new_tokens=20,
            temperature=0.8,
        )

        assert isinstance(result, str)
