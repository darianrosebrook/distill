"""
Tests for conversion/make_toy_block.py - Toy transformer block creation.

Tests RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock classes,
and main function for creating toy transformer blocks.
"""
# @author: @darianrosebrook

import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

# Import the module
import importlib
make_toy_block_module = importlib.import_module("conversion.make_toy_block")

RMSNorm = make_toy_block_module.RMSNorm
SwiGLU = make_toy_block_module.SwiGLU
MultiHeadAttention = make_toy_block_module.MultiHeadAttention
TransformerBlock = make_toy_block_module.TransformerBlock
main = make_toy_block_module.main


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

    def test_rmsnorm_weight(self):
        """Test RMSNorm weight parameter."""
        d = 16
        rmsnorm = RMSNorm(d)

        assert rmsnorm.weight.shape == (d,)
        assert torch.allclose(rmsnorm.weight, torch.ones(d))


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

    def test_swiglu_dimensions(self):
        """Test SwiGLU with different dimensions."""
        d_in = 32
        d_hidden = 64
        swiglu = SwiGLU(d_in, d_hidden)

        x = torch.randn(1, 5, d_in)
        output = swiglu(x)

        assert output.shape == (1, 5, d_in)


class TestMultiHeadAttention:
    """Test MultiHeadAttention class."""

    def test_mha_forward(self):
        """Test MultiHeadAttention forward pass."""
        d_model = 64
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)

        x = torch.randn(2, 10, d_model)
        output = mha(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mha_different_configs(self):
        """Test MultiHeadAttention with different configurations."""
        configs = [
            (32, 2),  # Small
            (128, 8),  # Medium
            (256, 16),  # Large
        ]

        for d_model, n_heads in configs:
            mha = MultiHeadAttention(d_model, n_heads)
            x = torch.randn(1, 5, d_model)
            output = mha(x)

            assert output.shape == x.shape

    def test_mha_d_head_calculation(self):
        """Test MultiHeadAttention d_head calculation."""
        d_model = 64
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)

        assert mha.d_head == d_model // n_heads
        assert mha.d_head == 16


class TestTransformerBlock:
    """Test TransformerBlock class."""

    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass."""
        d_model = 64
        n_heads = 4
        block = TransformerBlock(d_model, n_heads)

        x = torch.randn(2, 10, d_model)
        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_transformer_block_residual(self):
        """Test TransformerBlock residual connections."""
        d_model = 32
        n_heads = 2
        block = TransformerBlock(d_model, n_heads)
        block.eval()

        x = torch.randn(1, 5, d_model)
        output = block(x)

        # Output should be different from input (due to transformations)
        assert not torch.allclose(output, x)

    def test_transformer_block_mlp_hidden_mult(self):
        """Test TransformerBlock with custom MLP hidden multiplier."""
        d_model = 64
        n_heads = 4
        mlp_hidden_mult = 2.0
        block = TransformerBlock(d_model, n_heads, mlp_hidden_mult=mlp_hidden_mult)

        x = torch.randn(1, 5, d_model)
        output = block(x)

        assert output.shape == x.shape


class TestMainFunction:
    """Test main function."""

    @patch("conversion.make_toy_block.torch.jit.trace")
    def test_main_success(self, mock_trace, tmp_path):
        """Test successful main function execution."""
        output_path = tmp_path / "toy_block.pt"

        mock_traced = Mock()
        mock_traced.save = Mock()
        mock_traced.eval = Mock()
        mock_trace.return_value = mock_traced

        with patch("sys.argv", ["make_toy_block", "--out", str(output_path)]):
            main()

        mock_trace.assert_called_once()
        # Check that save was called with the actual path string
        assert mock_traced.save.called
        assert str(output_path) in str(mock_traced.save.call_args)

    @patch("conversion.make_toy_block.torch.jit.trace")
    @patch("conversion.make_toy_block.torch.jit.save")
    def test_main_default_args(self, mock_save, mock_trace, tmp_path):
        """Test main function with default arguments."""
        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        with patch("sys.argv", ["make_toy_block"]), patch(
            "conversion.make_toy_block.Path"
        ) as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            main()

            mock_trace.assert_called_once()
            # Check default dmodel and nheads were used
            call_args = mock_trace.call_args[0]
            model = call_args[0]
            assert isinstance(model, TransformerBlock)

    @patch("conversion.make_toy_block.torch.jit.trace")
    @patch("conversion.make_toy_block.torch.jit.save")
    def test_main_custom_args(self, mock_save, mock_trace, tmp_path):
        """Test main function with custom arguments."""
        output_path = tmp_path / "custom_block.pt"

        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        with patch(
            "sys.argv",
            [
                "make_toy_block",
                "--dmodel",
                "128",
                "--nheads",
                "8",
                "--seq",
                "256",
                "--out",
                str(output_path),
            ],
        ), patch("conversion.make_toy_block.Path") as mock_path_class:
            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            main()

            # Verify model was created with custom args
            call_args = mock_trace.call_args[0]
            model = call_args[0]
            assert isinstance(model, TransformerBlock)
            assert model.attn.d_model == 128
            assert model.attn.n_heads == 8

    @patch("conversion.make_toy_block.torch.jit.trace")
    def test_main_trace_failure(self, mock_trace, tmp_path):
        """Test main function with tracing failure."""
        output_path = tmp_path / "toy_block.pt"
        mock_trace.side_effect = Exception("Tracing failed")

        with patch("sys.argv", ["make_toy_block", "--out", str(output_path)]):
            with pytest.raises(Exception):
                main()

    def test_main_model_eval_mode(self, tmp_path):
        """Test that model is set to eval mode."""
        output_path = tmp_path / "toy_block.pt"

        with patch("conversion.make_toy_block.torch.jit.trace") as mock_trace, patch(
            "conversion.make_toy_block.torch.jit.save"
        ), patch("conversion.make_toy_block.Path") as mock_path_class:
            mock_traced = Mock()
            mock_trace.return_value = mock_traced

            mock_path = Mock()
            mock_path.parent.mkdir = Mock()
            mock_path_class.return_value = mock_path

            with patch("sys.argv", ["make_toy_block", "--out", str(output_path)]):
                main()

                # Verify model was set to eval mode
                call_args = mock_trace.call_args[0]
                model = call_args[0]
                assert not model.training  # Should be in eval mode


class TestIntegration:
    """Test integration of components."""

    def test_full_block_creation(self, tmp_path):
        """Test creating a full transformer block and tracing it."""
        output_path = tmp_path / "test_block.pt"

        block = TransformerBlock(d_model=64, n_heads=4)
        block.eval()

        x = torch.randn(1, 128, 64)
        with torch.no_grad():
            traced = torch.jit.trace(block, x)
            traced.eval()

        # Verify traced model works
        output = traced(x)
        assert output.shape == x.shape

    def test_block_components(self):
        """Test all block components work together."""
        d_model = 32
        n_heads = 2

        block = TransformerBlock(d_model, n_heads)
        block.eval()

        x = torch.randn(1, 10, d_model)

        # Test each component
        norm1_out = block.norm1(x)
        assert norm1_out.shape == x.shape

        attn_out = block.attn(norm1_out)
        assert attn_out.shape == x.shape

        norm2_out = block.norm2(attn_out + x)
        assert norm2_out.shape == x.shape

        mlp_out = block.mlp(norm2_out)
        assert mlp_out.shape == x.shape

        final_out = mlp_out + attn_out + x
        assert final_out.shape == x.shape


