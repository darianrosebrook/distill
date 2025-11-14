"""
Tests for conversion/export_pytorch.py - PyTorch model export framework.

Tests TorchScript export with prefill/decoder split, model loading from checkpoints,
wrapper classes, and export functionality using toy models.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import pytest
import torch

from conversion.export_pytorch import (
    PrefillWrapper,
    DecodeWrapper,
    export_prefill,
    export_decode,
    main,
)


class TestPrefillWrapper:
    """Test PrefillWrapper class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock StudentLM model."""
        model = Mock()
        model.use_halt_head = False

        # Mock forward method
        def mock_forward(input_ids, attn_mask, return_halt_logits=False):
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            if return_halt_logits:
                halt_logits = torch.randn(batch_size, seq_len, 1)
                return logits, halt_logits
            return logits

        model.__call__ = mock_forward
        return model

    def test_prefill_wrapper_init(self, mock_model):
        """Test PrefillWrapper initialization."""
        wrapper = PrefillWrapper(mock_model)
        assert wrapper.model == mock_model

    def test_prefill_wrapper_forward_without_halt(self, mock_model):
        """Test forward pass without halt logits."""
        mock_model.use_halt_head = False
        wrapper = PrefillWrapper(mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))  # [batch_size, seq_len]
        attn_mask = torch.ones(2, 10)

        result = wrapper(input_ids, attn_mask)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 10, 32000)

    def test_prefill_wrapper_forward_with_halt(self, mock_model):
        """Test forward pass with halt logits."""
        mock_model.use_halt_head = True
        wrapper = PrefillWrapper(mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))
        attn_mask = torch.ones(2, 10)

        result = wrapper(input_ids, attn_mask)

        # Should return tuple with both logits
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 10, 32000)
        assert result[1].shape == (2, 10, 1)

    def test_prefill_wrapper_forward_without_attn_mask(self, mock_model):
        """Test forward pass without attention mask."""
        wrapper = PrefillWrapper(mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))

        result = wrapper(input_ids)

        assert isinstance(result, tuple)
        assert len(result) == 1


class TestDecodeWrapper:
    """Test DecodeWrapper class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock StudentLM model."""
        model = Mock()

        # Mock forward method for decode (single token with KV cache)
        def mock_forward(input_ids, *kv_caches):
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return logits

        model.__call__ = mock_forward
        return model

    def test_decode_wrapper_init(self, mock_model):
        """Test DecodeWrapper initialization."""
        wrapper = DecodeWrapper(mock_model)
        assert wrapper.model == mock_model

    def test_decode_wrapper_forward_single_token(self, mock_model):
        """Test forward pass with single token."""
        wrapper = DecodeWrapper(mock_model)

        input_ids = torch.randint(0, 32000, (2, 1))  # [batch_size, 1]

        result = wrapper(input_ids)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 1, 32000)

    def test_decode_wrapper_forward_with_kv_cache(self, mock_model):
        """Test forward pass with KV cache tensors."""
        wrapper = DecodeWrapper(mock_model)

        input_ids = torch.randint(0, 32000, (2, 1))

        # Mock KV cache tensors (simplified)
        kv_cache_1 = torch.randn(2, 8, 10, 64)  # [batch, n_kv_heads, seq_len, d_head]
        kv_cache_2 = torch.randn(2, 8, 10, 64)

        result = wrapper(input_ids, kv_cache_1, kv_cache_2)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 1, 32000)


class TestExportPrefill:
    """Test prefill export functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock StudentLM model."""
        model = Mock()
        model.use_halt_head = False
        return model

    @patch("conversion.export_pytorch.torch.jit.trace")
    @patch("conversion.export_pytorch.torch.jit.save")
    def test_export_prefill_success(self, mock_save, mock_trace, mock_model, tmp_path):
        """Test successful prefill export."""
        # Mock traced model
        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        output_path = tmp_path / "prefill.pt"

        result = export_prefill(
            model=mock_model, output_path=output_path, seq_length=512, vocab_size=32000
        )

        # Verify tracing was called
        mock_trace.assert_called_once()

        # Verify save was called
        mock_save.assert_called_once_with(mock_traced, output_path)

        assert result == output_path

    @patch("conversion.export_pytorch.torch.jit.trace")
    def test_export_prefill_trace_failure(self, mock_trace, mock_model, tmp_path):
        """Test prefill export with tracing failure."""
        mock_trace.side_effect = Exception("Tracing failed")

        output_path = tmp_path / "prefill.pt"

        with pytest.raises(Exception):
            export_prefill(
                model=mock_model, output_path=output_path, seq_length=512, vocab_size=32000
            )

    @patch("conversion.export_pytorch.torch.jit.trace")
    @patch("conversion.export_pytorch.torch.jit.save")
    def test_export_prefill_with_halt_head(self, mock_save, mock_trace, tmp_path):
        """Test prefill export with halt head."""
        mock_model = Mock()
        mock_model.use_halt_head = True

        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        output_path = tmp_path / "prefill.pt"

        result = export_prefill(
            model=mock_model, output_path=output_path, seq_length=512, vocab_size=32000
        )

        # Should still work with halt head
        assert result == output_path


class TestExportDecode:
    """Test decode export functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock StudentLM model."""
        model = Mock()
        return model

    @patch("conversion.export_pytorch.torch.jit.trace")
    @patch("conversion.export_pytorch.torch.jit.save")
    def test_export_decode_success(self, mock_save, mock_trace, mock_model, tmp_path):
        """Test successful decode export."""
        # Mock traced model
        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        output_path = tmp_path / "decode.pt"

        result = export_decode(
            model=mock_model, output_path=output_path, n_layers=8, n_kv_heads=4, d_head=64
        )

        # Verify tracing was called
        mock_trace.assert_called_once()

        # Verify save was called
        mock_save.assert_called_once_with(mock_traced, output_path)

        assert result == output_path

    @patch("conversion.export_pytorch.torch.jit.trace")
    def test_export_decode_trace_failure(self, mock_trace, mock_model, tmp_path):
        """Test decode export with tracing failure."""
        mock_trace.side_effect = Exception("Tracing failed")

        output_path = tmp_path / "decode.pt"

        with pytest.raises(Exception):
            export_decode(
                model=mock_model, output_path=output_path, n_layers=8, n_kv_heads=4, d_head=64
            )

    @patch("conversion.export_pytorch.torch.jit.trace")
    @patch("conversion.export_pytorch.torch.jit.save")
    def test_export_decode_different_configs(self, mock_save, mock_trace, mock_model, tmp_path):
        """Test decode export with different model configurations."""
        mock_traced = Mock()
        mock_trace.return_value = mock_traced

        # Test with different layer/head configurations
        configs = [
            (6, 6, 128),  # 6 layers, 6 kv heads, 128 d_head
            (12, 2, 64),  # 12 layers, 2 kv heads, 64 d_head
            (24, 8, 32),  # 24 layers, 8 kv heads, 32 d_head
        ]

        for n_layers, n_kv_heads, d_head in configs:
            output_path = tmp_path / f"decode_{n_layers}_{n_kv_heads}_{d_head}.pt"

            result = export_decode(
                model=mock_model,
                output_path=output_path,
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
            )

            assert result == output_path


class TestMainFunction:
    """Test main function."""

    @patch("conversion.export_pytorch.export_prefill")
    @patch("conversion.export_pytorch.export_decode")
    @patch("conversion.export_pytorch.torch.load")
    @patch("conversion.export_pytorch.StudentLM")
    @patch("conversion.export_pytorch.ModelCfg")
    @patch("conversion.export_pytorch.Path")
    @patch("conversion.export_pytorch.argparse.ArgumentParser")
    def test_main_success(
        self,
        mock_parser_class,
        mock_path_class,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_export_decode,
        mock_export_prefill,
    ):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.out = "output_dir"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock Path
        mock_output_dir = Mock()
        mock_output_dir.mkdir = Mock()
        mock_path_class.return_value = mock_output_dir

        # Mock checkpoint loading
        mock_checkpoint = {
            "model_state_dict": {"weight": torch.randn(10, 10)},
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

        # Mock model creation
        mock_model_instance = Mock()
        mock_student_lm.return_value = mock_model_instance

        # Mock config
        mock_config_instance = Mock()
        mock_model_cfg.return_value = mock_config_instance

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        # Verify exports were called
        mock_export_prefill.assert_called_once()
        mock_export_decode.assert_called_once()

    @patch("conversion.export_pytorch.export_prefill")
    @patch("conversion.export_pytorch.export_decode")
    @patch("conversion.export_pytorch.torch.load")
    @patch("conversion.export_pytorch.StudentLM")
    @patch("conversion.export_pytorch.ModelCfg")
    @patch("conversion.export_pytorch.Path")
    @patch("conversion.export_pytorch.argparse.ArgumentParser")
    def test_main_checkpoint_without_config(
        self,
        mock_parser_class,
        mock_path_class,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_export_decode,
        mock_export_prefill,
    ):
        """Test main function with checkpoint without config."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.out = "output_dir"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock Path
        mock_output_dir = Mock()
        mock_output_dir.mkdir = Mock()
        mock_path_class.return_value = mock_output_dir

        # Mock checkpoint without config
        mock_checkpoint = {"model_state_dict": {"weight": torch.randn(10, 10)}}
        mock_torch_load.return_value = mock_checkpoint

        # Mock model creation
        mock_model_instance = Mock()
        mock_student_lm.return_value = mock_model_instance

        # Mock default config
        mock_config_instance = Mock()
        mock_model_cfg.return_value = mock_config_instance

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

    @patch("conversion.export_pytorch.torch.load")
    @patch("conversion.export_pytorch.argparse.ArgumentParser")
    def test_main_checkpoint_not_found(self, mock_parser_class, mock_torch_load):
        """Test main function with missing checkpoint."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "nonexistent.pt"
        mock_args.out = "output_dir"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_torch_load.side_effect = FileNotFoundError("Checkpoint not found")

        with pytest.raises(FileNotFoundError):
            main()


class TestModelConfigurations:
    """Test different model configurations."""

    def test_prefill_wrapper_different_batch_sizes(self):
        """Test PrefillWrapper with different batch sizes."""
        model = Mock()
        model.use_halt_head = False

        def mock_forward(input_ids, attn_mask=None, return_halt_logits=False):
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return logits

        model.__call__ = mock_forward

        wrapper = PrefillWrapper(model)

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        seq_length = 128

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 32000, (batch_size, seq_length))
            attn_mask = torch.ones(batch_size, seq_length)

            result = wrapper(input_ids, attn_mask)
            assert result[0].shape == (batch_size, seq_length, 32000)

    def test_decode_wrapper_different_kv_configs(self):
        """Test DecodeWrapper with different KV cache configurations."""
        model = Mock()

        def mock_forward(input_ids, *kv_caches):
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return logits

        model.__call__ = mock_forward

        wrapper = DecodeWrapper(model)

        # Test with different numbers of KV cache tensors
        kv_configs = [
            [],  # No KV cache
            [torch.randn(2, 4, 10, 64)],  # 1 layer
            [torch.randn(2, 4, 10, 64), torch.randn(2, 4, 10, 64)],  # 2 layers
        ]

        input_ids = torch.randint(0, 32000, (2, 1))

        for kv_caches in kv_configs:
            result = wrapper(input_ids, *kv_caches)
            assert result[0].shape == (2, 1, 32000)


class TestExportEdgeCases:
    """Test export edge cases."""

    def test_export_prefill_minimal_config(self, tmp_path):
        """Test prefill export with minimal configuration."""
        model = Mock()
        model.use_halt_head = False

        output_path = tmp_path / "minimal.pt"

        with (
            patch("conversion.export_pytorch.torch.jit.trace"),
            patch("conversion.export_pytorch.torch.jit.save"),
        ):
            result = export_prefill(
                model=model,
                output_path=output_path,
                seq_length=1,  # Minimal
                vocab_size=100,  # Minimal
            )

            assert result == output_path

    def test_export_decode_minimal_config(self, tmp_path):
        """Test decode export with minimal configuration."""
        model = Mock()

        output_path = tmp_path / "minimal_decode.pt"

        with (
            patch("conversion.export_pytorch.torch.jit.trace"),
            patch("conversion.export_pytorch.torch.jit.save"),
        ):
            result = export_decode(
                model=model,
                output_path=output_path,
                n_layers=1,  # Minimal
                n_kv_heads=1,  # Minimal
                d_head=32,  # Minimal
            )

            assert result == output_path

    def test_wrapper_parameter_validation(self):
        """Test that wrappers handle None inputs appropriately."""
        model = Mock()
        model.use_halt_head = False

        def mock_forward(input_ids, attn_mask=None, return_halt_logits=False):
            return torch.randn(1, 10, 1000)

        model.__call__ = mock_forward

        wrapper = PrefillWrapper(model)

        # Test with None attn_mask
        input_ids = torch.randint(0, 1000, (1, 10))
        result = wrapper(input_ids, None)
        assert result[0].shape == (1, 10, 1000)
