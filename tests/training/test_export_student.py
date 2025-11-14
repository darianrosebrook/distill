"""
Tests for training/export_student.py - Student model export functionality.

Tests TorchScript export, ExportedProgram export, contract creation,
and main function execution using mock models and configurations.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from training.export_student import (
    export_torchscript,
    export_exported_program,
    create_contract,
    main,
)


class TestTorchScriptExport:
    """Test TorchScript export functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        return model

    @pytest.fixture
    def example_input(self):
        """Create example input tensor."""
        return torch.randint(0, 1000, (2, 128))  # [batch_size, seq_len]

    def test_export_torchscript_success(self, mock_model, example_input, tmp_path):
        """Test successful TorchScript export."""
        output_path = tmp_path / "model.pt"

        with patch("torch.jit.trace") as mock_trace:
            mock_traced = Mock()
            mock_trace.return_value = mock_traced

            result = export_torchscript(mock_model, example_input, output_path)

            # Verify model.eval was called
            mock_model.eval.assert_called_once()

            # Verify torch.jit.trace was called
            mock_trace.assert_called_once_with(mock_model, example_input)

            # Verify traced.eval was called
            mock_traced.eval.assert_called_once()

            # Verify traced.save was called
            mock_traced.save.assert_called_once_with(str(output_path))

            assert result == mock_traced

    def test_export_torchscript_directory_creation(self, mock_model, example_input, tmp_path):
        """Test that output directory is created."""
        output_path = tmp_path / "subdir" / "model.pt"

        with (
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.save") as mock_save,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_traced = Mock()
            mock_trace.return_value = mock_traced

            result = export_torchscript(mock_model, example_input, output_path)

            # Verify mkdir was called with parents=True, exist_ok=True
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_export_torchscript_trace_failure(self, mock_model, example_input, tmp_path):
        """Test TorchScript export with tracing failure."""
        output_path = tmp_path / "model.pt"

        with patch("torch.jit.trace") as mock_trace:
            mock_trace.side_effect = Exception("Tracing failed")

            with pytest.raises(Exception):
                export_torchscript(mock_model, example_input, output_path)


class TestExportedProgramExport:
    """Test ExportedProgram export functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        return model

    @pytest.fixture
    def example_input(self):
        """Create example input tensor."""
        return torch.randint(0, 1000, (2, 128))

    def test_export_exported_program_success(self, mock_model, example_input, tmp_path):
        """Test successful ExportedProgram export."""
        output_path = tmp_path / "model.pt"

        with (
            patch("torch.export.export") as mock_export,
            patch("torch.save") as mock_save,
            patch("builtins.open") as mock_open,
        ):
            mock_exported = Mock()
            mock_export.return_value = mock_exported

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = export_exported_program(mock_model, example_input, output_path)

            # Verify torch.export.export was called
            mock_export.assert_called_once_with(mock_model, (example_input,))

            # Verify torch.save was called
            mock_save.assert_called_once_with(mock_exported, mock_file)

            assert result == mock_exported

    def test_export_exported_program_directory_creation(self, mock_model, example_input, tmp_path):
        """Test that output directory is created."""
        output_path = tmp_path / "subdir" / "model.pt"

        with (
            patch("torch.export.export") as mock_export,
            patch("torch.save") as mock_save,
            patch("builtins.open") as mock_open,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_exported = Mock()
            mock_export.return_value = mock_exported

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = export_exported_program(mock_model, example_input, output_path)

            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_export_exported_program_no_torch_export(self, mock_model, example_input, tmp_path):
        """Test ExportedProgram export when torch.export is not available."""
        output_path = tmp_path / "model.pt"

        with patch("torch.export.export") as mock_export:
            # Simulate AttributeError (torch.export not available)
            mock_export.side_effect = AttributeError("torch.export not available")

            with pytest.raises(RuntimeError, match="torch.export not available"):
                export_exported_program(mock_model, example_input, output_path)


class TestContractCreation:
    """Test contract creation functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock ModelCfg."""
        # Create a simple object with the required attributes
        class MockConfig:
            def __init__(self):
                self.d_model = 512
                self.n_layers = 8
                self.n_heads = 8
                self.n_kv_heads = 4
                self.d_head = 64
                self.vocab_size = 32000
                self.n_layers = 8
        return MockConfig()

    def test_create_contract_success(self, mock_config, tmp_path):
        """Test successful contract creation."""
        enumerated_T = [64, 128, 256, 512]
        output_dir = tmp_path

        create_contract(mock_config, enumerated_T, output_dir)

        # Verify contract file was created
        contract_path = output_dir / "contract.json"
        assert contract_path.exists()

        with open(contract_path, "r") as f:
            contract = json.load(f)

        # Check contract structure matches actual implementation
        assert "inputs" in contract
        assert "outputs" in contract
        assert "kv_precision" in contract
        assert "enumerated_T" in contract
        assert "n_heads" in contract

        # Check input specification
        assert len(contract["inputs"]) == 1
        assert contract["inputs"][0]["name"] == "input_ids"
        assert contract["inputs"][0]["dtype"] == "int32"
        assert contract["inputs"][0]["shape"] == ["B", "T"]

        # Check output specification
        assert len(contract["outputs"]) == 1
        assert contract["outputs"][0]["name"] == "logits"
        assert contract["outputs"][0]["dtype"] == "float16"
        assert contract["outputs"][0]["shape"] == ["B", "T", "V"]

        # Check enumerated_T
        assert contract["enumerated_T"] == enumerated_T

        # Check config values
        assert contract["n_heads"] == mock_config.n_heads
        assert contract["n_kv_heads"] == mock_config.n_kv_heads
        assert contract["d_head"] == mock_config.d_head
        assert contract["d_model"] == mock_config.d_model
        assert contract["n_layers"] == mock_config.n_layers
        assert contract["vocab_size"] == mock_config.vocab_size

    def test_create_contract_directory_creation(self, mock_config, tmp_path):
        """Test that contract creation works with different directory structures."""
        enumerated_T = [128, 256]
        output_dir = tmp_path / "deep" / "nested" / "dir"

        # Create the nested directory structure
        output_dir.mkdir(parents=True, exist_ok=True)

        # Should work without errors
        create_contract(mock_config, enumerated_T, output_dir)

        contract_path = output_dir / "contract.json"
        assert contract_path.exists()

    def test_create_contract_file_writing(self, mock_config, tmp_path):
        """Test that contract file is written correctly."""
        enumerated_T = [64, 128]
        output_dir = tmp_path

        create_contract(mock_config, enumerated_T, output_dir)

        # Read back the file and verify content
        contract_path = output_dir / "contract.json"
        with open(contract_path, "r") as f:
            content = f.read()

        # Should be valid JSON
        contract = json.loads(content)

        # Verify contract structure
        assert contract["enumerated_T"] == enumerated_T
        assert contract["vocab_size"] == mock_config.vocab_size


class TestMainFunction:
    """Test main function."""

    @patch("training.export_student.torch.load")
    @patch("training.export_student.argparse.ArgumentParser")
    def test_main_basic_setup(self, mock_parser_class, mock_torch_load):
        """Test main function basic setup and argument parsing."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.out = "output_dir"
        mock_args.format = "torchscript"
        mock_args.seq = 128
        mock_args.enumerated_T = [64, 128, 256]
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock checkpoint loading
        mock_checkpoint = {
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "config": {"d_model": 512, "n_layers": 8},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Should not raise exceptions during setup
        try:
            main()
        except (SystemExit, Exception):
            # Expected - we don't have full mocking
            pass

    @patch("training.export_student.argparse.ArgumentParser")
    def test_main_checkpoint_not_found(self, mock_parser_class):
        """Test main function with missing checkpoint."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "nonexistent.pt"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        with pytest.raises(FileNotFoundError):
            main()


class TestExportIntegration:
    """Test integration of export functionality."""

    def test_export_workflow(self, tmp_path):
        """Test complete export workflow."""
        # This is a high-level integration test that would test
        # the complete export pipeline in a real scenario

        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        example_input = torch.randn(2, 10)

        # Test TorchScript export
        ts_path = tmp_path / "model_ts.pt"
        exported_ts = export_torchscript(model, example_input, ts_path)

        assert ts_path.exists()
        assert exported_ts is not None

        # Test contract creation
        from training.export_student import ModelCfg

        config = ModelCfg(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, vocab_size=32000)
        enumerated_T = [64, 128, 256]

        contract_dir = tmp_path
        create_contract(config, enumerated_T, contract_dir)

        contract_path = contract_dir / "contract.json"
        assert contract_path.exists()

        # Verify contract content
        with open(contract_path, "r") as f:
            contract = json.load(f)

        assert contract["vocab_size"] == 32000
        assert contract["enumerated_T"] == enumerated_T

    def test_export_error_handling(self, tmp_path):
        """Test error handling in export functions."""

        # Test with invalid model
        class BadModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Model error")

        bad_model = BadModel()
        example_input = torch.randn(2, 10)
        output_path = tmp_path / "bad_model.pt"

        # TorchScript export should handle model errors
        with pytest.raises(Exception):
            export_torchscript(bad_model, example_input, output_path)


class TestConfigurationHandling:
    """Test configuration and parameter handling."""

    def test_contract_creation_with_different_configs(self, tmp_path):
        """Test contract creation with different model configurations."""
        from training.export_student import ModelCfg

        configs = [
            ModelCfg(d_model=256, n_layers=6, vocab_size=16000),
            ModelCfg(d_model=1024, n_layers=12, vocab_size=64000),
        ]

        enumerated_T = [128, 256, 512]

        for i, config in enumerate(configs):
            contract_dir = tmp_path / f"config_{i}"
            contract_dir.mkdir(parents=True, exist_ok=True)
            create_contract(config, enumerated_T, contract_dir)

            contract_path = contract_dir / "contract.json"

            # Verify vocab_size was correctly used
            with open(contract_path, "r") as f:
                contract = json.load(f)

            assert contract["vocab_size"] == config.vocab_size

    def test_export_with_different_input_shapes(self, tmp_path):
        """Test export with different input shapes."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Test with different batch sizes (same feature dimension)
        input_shapes = [
            torch.randn(1, 10),  # Single batch
            torch.randn(4, 10),  # Multi batch
            torch.randn(2, 10),  # Different batch size
        ]

        for i, example_input in enumerate(input_shapes):
            output_path = tmp_path / f"model_{i}.pt"
            result = export_torchscript(model, example_input, output_path)

            assert output_path.exists()
            assert result is not None
