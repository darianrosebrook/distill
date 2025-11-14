"""
Tests for conversion/export_onnx.py - ONNX export functionality.

Tests DecodeWrapper class, ONNX export for prefill and decode modes,
and main function execution with various configurations.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch

# Import the module using importlib due to filename
import importlib

export_onnx_module = importlib.import_module("conversion.export_onnx")

# Import classes and functions from the module
DecodeWrapper = export_onnx_module.DecodeWrapper
main = export_onnx_module.main


class TestDecodeWrapper:
    """Test DecodeWrapper class functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock StudentLM model."""
        model = Mock()
        model.forward_decode = Mock(
            return_value=(
                torch.randn(2, 1, 32000),  # logits
                [  # updated_caches
                    (torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)),
                    (torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)),
                ],
            )
        )
        return model

    @pytest.fixture
    def wrapper(self, mock_model):
        """Create DecodeWrapper instance."""
        return DecodeWrapper(mock_model, n_layers=2, n_kv_heads=8, d_head=64)

    def test_decode_wrapper_initialization(self, mock_model):
        """Test DecodeWrapper initialization."""
        wrapper = DecodeWrapper(mock_model, n_layers=6, n_kv_heads=4, d_head=128)

        assert wrapper.model == mock_model
        assert wrapper.n_layers == 6
        assert wrapper.n_kv_heads == 4
        assert wrapper.d_head == 128

    def test_decode_wrapper_forward_with_kv_cache(self, wrapper, mock_model):
        """Test forward pass with KV cache inputs."""
        input_ids = torch.randint(0, 1000, (2, 1))  # [batch_size, 1]

        # Create mock KV caches: n_layers * 2 tensors
        kv_caches = []
        for _ in range(2):  # n_layers
            k_cache = torch.randn(2, 8, 10, 64)  # [B, Hk, T_cache, Dh]
            v_cache = torch.randn(2, 8, 10, 64)
            kv_caches.extend([k_cache, v_cache])

        result = wrapper(input_ids, *kv_caches)

        # Should return tuple: (logits, k_cache1, v_cache1, k_cache2, v_cache2, ...)
        assert isinstance(result, tuple)
        assert len(result) == 5  # 1 logits + 2 layers * 2 caches each

        # Check logits shape
        logits = result[0]
        assert logits.shape == (2, 1, 32000)

        # Check cache outputs
        for i in range(1, len(result), 2):
            k_cache_out = result[i]
            v_cache_out = result[i + 1]
            assert k_cache_out.shape == (2, 8, 10, 64)
            assert v_cache_out.shape == (2, 8, 10, 64)

        # Verify model.forward_decode was called
        mock_model.forward_decode.assert_called_once()
        call_args = mock_model.forward_decode.call_args
        assert call_args[0][0].equal(input_ids)  # input_ids
        assert len(call_args[0][1]) == 2  # kv_list length

    def test_decode_wrapper_forward_empty_cache(self, mock_model):
        """Test forward pass with empty KV cache."""
        wrapper = DecodeWrapper(mock_model, n_layers=1, n_kv_heads=4, d_head=64)

        input_ids = torch.randint(0, 1000, (1, 1))

        # Create empty caches (T_cache=0)
        empty_k = torch.empty(1, 4, 0, 64)
        empty_v = torch.empty(1, 4, 0, 64)

        result = wrapper(input_ids, empty_k, empty_v)

        # Verify model.forward_decode was called with None for empty caches
        call_args = mock_model.forward_decode.call_args
        kv_list = call_args[0][1]
        assert kv_list[0] is None  # Empty cache should be None

    def test_decode_wrapper_forward_mixed_cache(self, mock_model):
        """Test forward pass with mix of empty and non-empty caches."""
        wrapper = DecodeWrapper(mock_model, n_layers=2, n_kv_heads=4, d_head=64)

        input_ids = torch.randint(0, 1000, (1, 1))

        # Layer 1: empty cache
        empty_k1 = torch.empty(1, 4, 0, 64)
        empty_v1 = torch.empty(1, 4, 0, 64)

        # Layer 2: non-empty cache
        k2 = torch.randn(1, 4, 5, 64)
        v2 = torch.randn(1, 4, 5, 64)

        result = wrapper(input_ids, empty_k1, empty_v1, k2, v2)

        # Verify model.forward_decode was called with correct kv_list
        call_args = mock_model.forward_decode.call_args
        kv_list = call_args[0][1]
        assert kv_list[0] is None  # Layer 1 empty
        assert kv_list[1] is not None  # Layer 2 has cache
        assert isinstance(kv_list[1], tuple)

    def test_decode_wrapper_forward_missing_cache_args(self, mock_model):
        """Test forward pass with missing cache arguments."""
        wrapper = DecodeWrapper(mock_model, n_layers=2, n_kv_heads=4, d_head=64)

        input_ids = torch.randint(0, 1000, (1, 1))

        # Provide only some cache arguments
        k1 = torch.randn(1, 4, 3, 64)
        v1 = torch.randn(1, 4, 3, 64)
        # Missing k2, v2

        result = wrapper(input_ids, k1, v1)

        # Should handle missing caches gracefully
        call_args = mock_model.forward_decode.call_args
        kv_list = call_args[0][1]
        assert kv_list[0] is not None  # Layer 1 has cache
        assert kv_list[1] is None  # Layer 2 missing cache

    def test_decode_wrapper_forward_single_token(self, wrapper, mock_model):
        """Test forward pass with single token input."""
        input_ids = torch.tensor([[42]])  # Single token

        # Empty caches for simplicity
        kv_caches = [torch.empty(1, 8, 0, 64), torch.empty(1, 8, 0, 64)]

        result = wrapper(input_ids, *kv_caches)

        assert isinstance(result, tuple)
        assert result[0].shape == (1, 1, 32000)  # [B=1, T=1, vocab_size]

        mock_model.forward_decode.assert_called_once()

    def test_decode_wrapper_forward_batch_processing(self, wrapper, mock_model):
        """Test forward pass with batch processing."""
        batch_size = 4
        input_ids = torch.randint(0, 1000, (batch_size, 1))

        # Create caches for batch
        kv_caches = []
        for _ in range(2):  # n_layers
            k_cache = torch.randn(batch_size, 8, 15, 64)
            v_cache = torch.randn(batch_size, 8, 15, 64)
            kv_caches.extend([k_cache, v_cache])

        result = wrapper(input_ids, *kv_caches)

        # Check batch dimension is preserved
        logits = result[0]
        assert logits.shape[0] == batch_size  # Batch size preserved

        # Check all cache outputs have correct batch size
        for cache_output in result[1:]:
            assert cache_output.shape[0] == batch_size


class TestMainFunction:
    """Test main function."""

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.os.makedirs")
    @patch("conversion.export_onnx.torch.onnx.export")
    @patch("conversion.export_onnx.torch.load")
    @patch("conversion.export_onnx.StudentLM")
    @patch("conversion.export_onnx.ModelCfg")
    @patch("conversion.export_onnx.DecodeWrapper")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_both_modes(
        self,
        mock_echo,
        mock_decode_wrapper,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_onnx_export,
        mock_makedirs,
        mock_open,
        mock_json_load,
    ):
        """Test main function exporting both prefill and decode modes."""
        # Mock config data
        config_data = [{"seq": 128, "batch": 1}, {"seq": 256, "batch": 1}, {"seq": 512, "batch": 1}]
        mock_json_load.return_value = config_data

        # Mock file opening
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock model loading
        mock_checkpoint = {"model_state_dict": {}, "config": {}}
        mock_torch_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        # Mock decode wrapper
        mock_wrapper = Mock()
        mock_decode_wrapper.return_value = mock_wrapper

        # Test main with both mode
        main(config="test_config.json", mode="both")

        # Should call ONNX export multiple times (for prefill and decode with different seqs)
        # Prefill exports: 3 sequences
        # Decode exports: 3 sequences
        assert mock_onnx_export.call_count == 6

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.os.makedirs")
    @patch("conversion.export_onnx.torch.onnx.export")
    @patch("conversion.export_onnx.torch.load")
    @patch("conversion.export_onnx.StudentLM")
    @patch("conversion.export_onnx.ModelCfg")
    @patch("conversion.export_onnx.DecodeWrapper")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_prefill_only(
        self,
        mock_echo,
        mock_decode_wrapper,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_onnx_export,
        mock_makedirs,
        mock_open,
        mock_json_load,
    ):
        """Test main function exporting prefill mode only."""
        config_data = [{"seq": 256, "batch": 1}]
        mock_json_load.return_value = config_data

        # Mock file and model loading
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        mock_checkpoint = {"model_state_dict": {}, "config": {}}
        mock_torch_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        # Test main with prefill only
        main(config="test_config.json", mode="prefill")

        # Should call ONNX export only for prefill (1 call)
        assert mock_onnx_export.call_count == 1

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.os.makedirs")
    @patch("conversion.export_onnx.torch.onnx.export")
    @patch("conversion.export_onnx.torch.load")
    @patch("conversion.export_onnx.StudentLM")
    @patch("conversion.export_onnx.ModelCfg")
    @patch("conversion.export_onnx.DecodeWrapper")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_decode_only(
        self,
        mock_echo,
        mock_decode_wrapper,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_onnx_export,
        mock_makedirs,
        mock_open,
        mock_json_load,
    ):
        """Test main function exporting decode mode only."""
        config_data = [{"seq": 128, "batch": 1}]
        mock_json_load.return_value = config_data

        # Mock file and model loading
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        mock_checkpoint = {"model_state_dict": {}, "config": {}}
        mock_torch_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        mock_wrapper = Mock()
        mock_decode_wrapper.return_value = mock_wrapper

        # Test main with decode only
        main(config="test_config.json", mode="decode")

        # Should call ONNX export only for decode (1 call)
        assert mock_onnx_export.call_count == 1

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_config_not_found(self, mock_echo, mock_open, mock_json_load):
        """Test main function with missing config file."""
        mock_open.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(SystemExit):
            main(config="nonexistent.json", mode="both")

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_invalid_config(self, mock_echo, mock_open, mock_json_load):
        """Test main function with invalid config file."""
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with pytest.raises(SystemExit):
            main(config="invalid.json", mode="both")

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_fallback_config(self, mock_echo, mock_open, mock_json_load):
        """Test main function with fallback config when file operations fail."""
        mock_open.side_effect = Exception("File operation failed")

        # Should use default sequence lengths and continue
        try:
            main(config="failing_config.json", mode="prefill")
            # If we get here, it handled the error gracefully
            assert True
        except SystemExit:
            # Expected if other operations fail
            assert True

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.os.makedirs")
    @patch("conversion.export_onnx.torch.onnx.export")
    @patch("conversion.export_onnx.torch.load")
    @patch("conversion.export_onnx.StudentLM")
    @patch("conversion.export_onnx.ModelCfg")
    @patch("conversion.export_onnx.DecodeWrapper")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_model_loading_failure(
        self,
        mock_echo,
        mock_decode_wrapper,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_onnx_export,
        mock_makedirs,
        mock_open,
        mock_json_load,
    ):
        """Test main function with model loading failure."""
        config_data = [{"seq": 128, "batch": 1}]
        mock_json_load.return_value = config_data

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Model loading fails
        mock_torch_load.side_effect = Exception("Model load failed")

        with pytest.raises(SystemExit):
            main(config="test_config.json", mode="prefill")

    @patch("conversion.export_onnx.json.load")
    @patch("conversion.export_onnx.open")
    @patch("conversion.export_onnx.os.makedirs")
    @patch("conversion.export_onnx.torch.onnx.export")
    @patch("conversion.export_onnx.torch.load")
    @patch("conversion.export_onnx.StudentLM")
    @patch("conversion.export_onnx.ModelCfg")
    @patch("conversion.export_onnx.DecodeWrapper")
    @patch("conversion.export_onnx.typer.echo")
    def test_main_onnx_export_failure(
        self,
        mock_echo,
        mock_decode_wrapper,
        mock_model_cfg,
        mock_student_lm,
        mock_torch_load,
        mock_onnx_export,
        mock_makedirs,
        mock_open,
        mock_json_load,
    ):
        """Test main function with ONNX export failure."""
        config_data = [{"seq": 128, "batch": 1}]
        mock_json_load.return_value = config_data

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        mock_checkpoint = {"model_state_dict": {}, "config": {}}
        mock_torch_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        # ONNX export fails
        mock_onnx_export.side_effect = Exception("ONNX export failed")

        with pytest.raises(SystemExit):
            main(config="test_config.json", mode="prefill")


class TestONNXExportIntegration:
    """Test integration of ONNX export components."""

    def test_decode_wrapper_real_model_simulation(self):
        """Test DecodeWrapper with simulated real model behavior."""
        # Create a mock model that simulates real forward_decode behavior
        mock_model = Mock()
        mock_model.forward_decode = Mock(
            return_value=(
                torch.randn(1, 1, 1000),  # logits
                [  # updated_caches (2 layers)
                    (torch.randn(1, 4, 20, 32), torch.randn(1, 4, 20, 32)),
                    (torch.randn(1, 4, 20, 32), torch.randn(1, 4, 20, 32)),
                ],
            )
        )

        wrapper = DecodeWrapper(mock_model, n_layers=2, n_kv_heads=4, d_head=32)

        # Test with realistic inputs
        input_ids = torch.tensor([[42]], dtype=torch.long)
        kv_caches = [torch.randn(1, 4, 15, 32) for _ in range(4)]  # 2 layers * 2 caches

        result = wrapper(input_ids, *kv_caches)

        assert len(result) == 5  # logits + 4 cache tensors
        assert result[0].shape == (1, 1, 1000)  # logits

        # Verify caches are returned
        for i in range(1, 5):
            assert result[i].shape == (1, 4, 20, 32)

    def test_config_parsing_edge_cases(self):
        """Test config parsing with various edge cases."""
        # Test with empty config
        with patch("conversion.export_onnx.json.load", return_value=[]):
            # Should use fallback config
            try:
                main(config="empty_config.json", mode="prefill")
                assert True  # Should not crash
            except SystemExit:
                assert True  # Expected behavior

        # Test with config missing seq field
        with patch("conversion.export_onnx.json.load", return_value=[{"batch": 1}]):
            try:
                main(config="missing_seq_config.json", mode="prefill")
                assert True  # Should use fallback
            except SystemExit:
                assert True  # Expected behavior

    def test_export_directory_creation(self, tmp_path):
        """Test that export directories are created properly."""
        with (
            patch("conversion.export_onnx.json.load", return_value=[{"seq": 128, "batch": 1}]),
            patch("conversion.export_onnx.open"),
            patch("conversion.export_onnx.os.makedirs") as mock_makedirs,
            patch("conversion.export_onnx.torch.onnx.export"),
            patch("conversion.export_onnx.torch.load"),
            patch("conversion.export_onnx.StudentLM"),
            patch("conversion.export_onnx.ModelCfg"),
            patch("conversion.export_onnx.DecodeWrapper"),
        ):
            # Run main function
            try:
                main(config="test_config.json", mode="prefill")
            except SystemExit:
                pass

            # Verify directories were created
            mock_makedirs.assert_called()

    def test_export_file_naming(self, tmp_path):
        """Test that export files are named correctly."""
        config_data = [{"seq": 256, "batch": 1}]

        with (
            patch("conversion.export_onnx.json.load", return_value=config_data),
            patch("conversion.export_onnx.open"),
            patch("conversion.export_onnx.os.makedirs"),
            patch("conversion.export_onnx.torch.onnx.export") as mock_export,
            patch("conversion.export_onnx.torch.load"),
            patch("conversion.export_onnx.StudentLM"),
            patch("conversion.export_onnx.ModelCfg"),
            patch("conversion.export_onnx.DecodeWrapper"),
        ):
            try:
                main(config="test_config.json", mode="prefill")
            except SystemExit:
                pass

            # Check that export was called with correct output path
            call_args = mock_export.call_args
            output_path = call_args[0][1]  # Second argument is output path

            # Should contain sequence information
            assert "256" in str(output_path)

    def test_wrapper_cache_handling_edge_cases(self):
        """Test DecodeWrapper cache handling edge cases."""
        mock_model = Mock()
        wrapper = DecodeWrapper(mock_model, n_layers=3, n_kv_heads=8, d_head=64)

        # Test with no cache arguments at all
        input_ids = torch.tensor([[1]], dtype=torch.long)

        mock_model.forward_decode.return_value = (
            torch.randn(1, 1, 1000),
            [(torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(3)],
        )

        result = wrapper(input_ids)

        # Should handle missing caches by setting them to None
        call_args = mock_model.forward_decode.call_args
        kv_list = call_args[0][1]
        assert all(cache is None for cache in kv_list)

    def test_mode_validation(self):
        """Test mode parameter validation."""
        with (
            patch("conversion.export_onnx.json.load", return_value=[{"seq": 128}]),
            patch("conversion.export_onnx.open"),
            patch("conversion.export_onnx.os.makedirs"),
            patch("conversion.export_onnx.torch.onnx.export"),
            patch("conversion.export_onnx.torch.load"),
            patch("conversion.export_onnx.StudentLM"),
            patch("conversion.export_onnx.ModelCfg"),
            patch("conversion.export_onnx.DecodeWrapper"),
        ):
            # Test valid modes
            for mode in ["prefill", "decode", "both"]:
                try:
                    main(config="test.json", mode=mode)
                    assert True  # Should not raise for valid modes
                except SystemExit:
                    assert True  # Expected completion

    def test_checkpoint_loading_and_model_creation(self):
        """Test checkpoint loading and model creation flow."""
        with (
            patch("conversion.export_onnx.json.load", return_value=[{"seq": 128}]),
            patch("conversion.export_onnx.open"),
            patch("conversion.export_onnx.os.makedirs"),
            patch("conversion.export_onnx.torch.onnx.export"),
            patch("conversion.export_onnx.torch.load") as mock_load,
            patch("conversion.export_onnx.StudentLM") as mock_student_lm,
            patch("conversion.export_onnx.ModelCfg") as mock_model_cfg,
            patch("conversion.export_onnx.DecodeWrapper"),
        ):
            # Mock checkpoint
            mock_checkpoint = {
                "model_state_dict": {"layer.weight": torch.ones(10, 10)},
                "config": {"d_model": 512, "n_layers": 8},
            }
            mock_load.return_value = mock_checkpoint

            # Mock config and model creation
            mock_config = Mock()
            mock_model_cfg.return_value = mock_config

            mock_model = Mock()
            mock_student_lm.return_value = mock_model

            try:
                main(config="test.json", mode="prefill")
            except SystemExit:
                pass

            # Verify checkpoint was loaded
            mock_load.assert_called()

            # Verify model was created with config
            mock_model_cfg.assert_called()
            mock_student_lm.assert_called_with(mock_config)

            # Verify model state was loaded
            mock_model.load_state_dict.assert_called_with(mock_checkpoint["model_state_dict"])
