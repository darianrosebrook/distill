"""
Tests for conversion/make_toy_torch.py - Toy PyTorch model creation.

Tests RMSNorm, SwiGLU, ToyTransformer classes, and main function
for creating toy PyTorch models for CoreML conversion testing.
"""
# @author: @darianrosebrook

import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

# Import the module
import importlib
make_toy_torch_module = importlib.import_module("conversion.make_toy_torch")

RMSNorm = make_toy_torch_module.RMSNorm
SwiGLU = make_toy_torch_module.SwiGLU
ToyTransformer = make_toy_torch_module.ToyTransformer
main = make_toy_torch_module.main


class TestRMSNorm:
    """Test RMSNorm class."""

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        d = 64
        rmsnorm = RMSNorm(d)

        x = torch.randn(2, 10, d)
        output = rmsnorm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_eps(self):
        """Test RMSNorm with different eps values."""
        d = 32
        rmsnorm = RMSNorm(d, eps=1e-5)

        x = torch.randn(1, 5, d)
        output = rmsnorm(x)

        assert output.shape == x.shape


class TestSwiGLU:
    """Test SwiGLU class."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        d_in = 64
        d_hidden = 128
        swiglu = SwiGLU(d_in, d_hidden)

        x = torch.randn(2, 10, d_in)
        output = swiglu(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_swiglu_silu_activation(self):
        """Test SwiGLU uses SiLU activation."""
        d_in = 32
        d_hidden = 64
        swiglu = SwiGLU(d_in, d_hidden)

        x = torch.randn(1, 5, d_in)
        output = swiglu(x)

        # Output should be different from input
        assert not torch.allclose(output, x)


class TestToyTransformer:
    """Test ToyTransformer class."""

    def test_toy_transformer_forward(self):
        """Test ToyTransformer forward pass."""
        vocab_size = 256
        d_model = 64
        model = ToyTransformer(vocab_size=vocab_size, d_model=d_model)

        input_ids = torch.randint(0, vocab_size, (2, 10))
        logits = model(input_ids)

        assert logits.shape == (2, 10, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_toy_transformer_components(self):
        """Test ToyTransformer has all required components."""
        vocab_size = 128
        d_model = 32
        model = ToyTransformer(vocab_size=vocab_size, d_model=d_model)

        assert hasattr(model, "embed")
        assert hasattr(model, "norm1")
        assert hasattr(model, "mlp")
        assert hasattr(model, "norm2")
        assert hasattr(model, "lm_head")

        assert isinstance(model.embed, nn.Embedding)
        assert isinstance(model.norm1, RMSNorm)
        assert isinstance(model.mlp, SwiGLU)
        assert isinstance(model.norm2, RMSNorm)
        assert isinstance(model.lm_head, nn.Linear)

    def test_toy_transformer_different_sizes(self):
        """Test ToyTransformer with different sizes."""
        configs = [
            (64, 32),  # Small
            (256, 64),  # Medium
            (512, 128),  # Large
        ]

        for vocab_size, d_model in configs:
            model = ToyTransformer(vocab_size=vocab_size, d_model=d_model)
            input_ids = torch.randint(0, vocab_size, (1, 5))
            logits = model(input_ids)

            assert logits.shape == (1, 5, vocab_size)

    def test_toy_transformer_forward_flow(self):
        """Test ToyTransformer forward flow through all layers."""
        vocab_size = 128
        d_model = 32
        model = ToyTransformer(vocab_size=vocab_size, d_model=d_model)
        model.eval()

        input_ids = torch.randint(0, vocab_size, (1, 10))

        # Test each step
        x = model.embed(input_ids)
        assert x.shape == (1, 10, d_model)

        x = model.norm1(x)
        assert x.shape == (1, 10, d_model)

        x = model.mlp(x)
        assert x.shape == (1, 10, d_model)

        x = model.norm2(x)
        assert x.shape == (1, 10, d_model)

        logits = model.lm_head(x)
        assert logits.shape == (1, 10, vocab_size)


class TestMainFunction:
    """Test main function."""

    @patch("conversion.make_toy_torch.torch.jit.trace")
    def test_main_success(self, mock_trace, tmp_path):
        """Test successful main function execution."""
        output_path = tmp_path / "toy_torch.pt"

        mock_traced = Mock()
        mock_traced.save = Mock()
        mock_traced.eval = Mock()
        mock_trace.return_value = mock_traced

        with patch("sys.argv", ["make_toy_torch", "--out", str(output_path)]):
            main()

        mock_trace.assert_called_once()
        # Check that save was called with the actual path string
        assert mock_traced.save.called
        assert str(output_path) in str(mock_traced.save.call_args)

    @patch("conversion.make_toy_torch.torch.jit.trace")
    @patch("conversion.make_toy_torch.torch.jit.save")
    def test_main_default_args(self, mock_save, mock_trace, tmp_path):
        """Test main function with default arguments."""
        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        with patch("sys.argv", ["make_toy_torch"]), patch(
            "conversion.make_toy_torch.Path"
        ) as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            main()

            mock_trace.assert_called_once()
            # Check default values were used
            call_args = mock_trace.call_args[0]
            model = call_args[0]
            assert isinstance(model, ToyTransformer)

    @patch("conversion.make_toy_torch.torch.jit.trace")
    @patch("conversion.make_toy_torch.torch.jit.save")
    def test_main_custom_args(self, mock_save, mock_trace, tmp_path):
        """Test main function with custom arguments."""
        output_path = tmp_path / "custom_torch.pt"

        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        with patch(
            "sys.argv",
            [
                "make_toy_torch",
                "--seq",
                "256",
                "--vocab",
                "512",
                "--dmodel",
                "128",
                "--out",
                str(output_path),
            ],
        ), patch("conversion.make_toy_torch.Path") as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            main()

            # Verify model was created with custom args
            call_args = mock_trace.call_args[0]
            model = call_args[0]
            assert isinstance(model, ToyTransformer)
            assert model.embed.num_embeddings == 512  # Custom vocab
            assert model.embed.embedding_dim == 128  # Custom dmodel

    @patch("conversion.make_toy_torch.torch.jit.trace")
    def test_main_trace_failure(self, mock_trace, tmp_path):
        """Test main function with tracing failure."""
        output_path = tmp_path / "toy_torch.pt"
        mock_trace.side_effect = Exception("Tracing failed")

        with patch("sys.argv", ["make_toy_torch", "--out", str(output_path)]):
            with pytest.raises(Exception):
                main()

    def test_main_model_eval_mode(self, tmp_path):
        """Test that model is set to eval mode."""
        output_path = tmp_path / "toy_torch.pt"

        with patch("conversion.make_toy_torch.torch.jit.trace") as mock_trace, patch(
            "conversion.make_toy_torch.torch.jit.save"
        ), patch("conversion.make_toy_torch.Path") as mock_path_class:
            mock_traced = Mock()
            mock_trace.return_value = mock_traced

            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            with patch("sys.argv", ["make_toy_torch", "--out", str(output_path)]):
                main()

                # Verify model was set to eval mode
                call_args = mock_trace.call_args[0]
                model = call_args[0]
                assert not model.training  # Should be in eval mode

    def test_main_input_dtype(self, tmp_path):
        """Test that input is created with correct dtype."""
        output_path = tmp_path / "toy_torch.pt"

        with patch("conversion.make_toy_torch.torch.jit.trace") as mock_trace, patch(
            "conversion.make_toy_torch.torch.jit.save"
        ), patch("conversion.make_toy_torch.Path") as mock_path_class:
            mock_traced = Mock()
            mock_trace.return_value = mock_traced

            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            with patch("sys.argv", ["make_toy_torch", "--out", str(output_path)]):
                main()

                # Verify input dtype
                call_args = mock_trace.call_args[0]
                example_input = call_args[1]
                assert example_input.dtype == torch.int32


class TestIntegration:
    """Test integration of components."""

    def test_full_model_creation(self, tmp_path):
        """Test creating a full toy transformer and tracing it."""
        output_path = tmp_path / "test_torch.pt"

        model = ToyTransformer(vocab_size=256, d_model=64)
        model.eval()

        input_ids = torch.zeros((1, 128), dtype=torch.int32)
        with torch.no_grad():
            traced = torch.jit.trace(model, input_ids)
            traced.eval()

        # Verify traced model works
        output = traced(input_ids)
        assert output.shape == (1, 128, 256)

    def test_model_components_work_together(self):
        """Test all model components work together."""
        vocab_size = 128
        d_model = 32

        model = ToyTransformer(vocab_size=vocab_size, d_model=d_model)
        model.eval()

        input_ids = torch.randint(0, vocab_size, (1, 10))

        # Test forward pass
        logits = model(input_ids)
        assert logits.shape == (1, 10, vocab_size)

        # Test that logits are reasonable
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


