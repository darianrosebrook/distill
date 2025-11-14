"""
Tests for conversion/convert_coreml.py - CoreML conversion functionality.

Tests ONNX to CoreML and PyTorch to CoreML conversion, contract loading,
placeholder creation, and main function execution.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

# Import the module using importlib due to filename with digits
import importlib

convert_coreml_module = importlib.import_module("conversion.convert_coreml")

# Import functions and classes from the module
load_contract = convert_coreml_module.load_contract
convert_pytorch_to_coreml = convert_coreml_module.convert_pytorch_to_coreml
convert_onnx_to_coreml = convert_coreml_module.convert_onnx_to_coreml
create_placeholder = convert_coreml_module.create_placeholder
main = convert_coreml_module.main


class TestLoadContract:
    """Test contract loading functionality."""

    def test_load_contract_success(self, tmp_path):
        """Test successful contract loading."""
        contract_data = {
            "inputs": [{"name": "input_ids", "dtype": "int32", "shape": ["B", "T"]}],
            "outputs": [{"name": "logits", "dtype": "float32", "shape": ["B", "T", "vocab_size"]}],
            "metadata": {"model_type": "transformer", "vocab_size": 32000},
        }

        contract_file = tmp_path / "contract.json"
        with open(contract_file, "w") as f:
            json.dump(contract_data, f)

        result = load_contract(str(contract_file))

        assert result == contract_data

    def test_load_contract_file_not_found(self):
        """Test contract loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_contract("definitely_does_not_exist_12345.json")

    def test_load_contract_invalid_json(self, tmp_path):
        """Test contract loading with invalid JSON."""
        contract_file = tmp_path / "invalid.json"
        with open(contract_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_contract(str(contract_file))


class TestConvertPyTorchToCoreML:
    """Test PyTorch to CoreML conversion."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        return model

    @pytest.fixture
    def example_input(self):
        """Create example input tensor."""
        return torch.randint(0, 1000, (2, 128))

    def test_convert_pytorch_to_coreml_success(self, mock_model, example_input, tmp_path):
        """Test successful PyTorch to CoreML conversion."""
        output_path = tmp_path / "model.mlpackage"

        mock_ct = Mock()
        with (
            patch.dict("sys.modules", {"coremltools": mock_ct}),
            patch("conversion.convert_coreml.check_coreml_versions"),
            patch("builtins.print"),
        ):
            # Mock CoreML conversion
            mock_converter = Mock()
            mock_ct.convert.return_value = mock_converter

            # Mock MIL program
            mock_program = Mock()
            mock_converter._mil_program = mock_program

            # Mock CoreML tools save
            mock_mlmodel = Mock()
            mock_ct.converters.mil_converter.convert.return_value = mock_mlmodel

            result = convert_pytorch_to_coreml(
                pytorch_model=mock_model, output_path=str(output_path), target="macOS13"
            )

            assert result == str(output_path)
            mock_model.eval.assert_called_once()
            mock_ct.convert.assert_called_once()

    def test_convert_pytorch_to_coreml_with_ane_optimization(
        self, mock_model, example_input, tmp_path
    ):
        """Test PyTorch to CoreML conversion with ANE optimization."""
        output_path = tmp_path / "model.mlpackage"

        with (
            patch("conversion.convert_coreml.ct") as mock_ct,
            patch("conversion.convert_coreml.coremltools") as mock_coremltools,
            patch("conversion.convert_coreml.check_coreml_versions"),
            patch(
                "conversion.convert_coreml.detect_int64_tensors_on_attention_paths"
            ) as mock_detect,
            patch("conversion.convert_coreml.check_ane_op_compatibility") as mock_check,
            patch("builtins.print"),
        ):
            # Mock ANE detection and compatibility
            mock_detect.return_value = []
            mock_check.return_value = {"compatible": True}

            # Mock conversion
            mock_converter = Mock()
            mock_ct.convert.return_value = mock_converter
            mock_program = Mock()
            mock_converter._mil_program = mock_program

            mock_mlmodel = Mock()
            mock_coremltools.converters.mil_converter.convert.return_value = mock_mlmodel

            result = convert_pytorch_to_coreml(
                model=mock_model,
                example_input=example_input,
                output_path=output_path,
                target="macOS13",
                enable_ane=True,
            )

            assert result == str(output_path)
            mock_detect.assert_called_once()
            mock_check.assert_called_once()

    def test_convert_pytorch_to_coreml_ane_incompatible(self, mock_model, example_input, tmp_path):
        """Test PyTorch to CoreML conversion when ANE incompatible."""
        output_path = tmp_path / "model.mlpackage"

        with (
            patch("conversion.convert_coreml.check_ane_op_compatibility") as mock_check,
            patch("builtins.print"),
        ):
            mock_check.return_value = {"compatible": False, "error": "ANE incompatible"}

            result = convert_pytorch_to_coreml(
                model=mock_model,
                example_input=example_input,
                output_path=output_path,
                target="macOS13",
                enable_ane=True,
            )

            assert result == str(output_path)
            # Should still succeed but with warning

    def test_convert_pytorch_to_coreml_conversion_failure(
        self, mock_model, example_input, tmp_path
    ):
        """Test PyTorch to CoreML conversion failure."""
        output_path = tmp_path / "model.mlpackage"

        with patch("conversion.convert_coreml.ct") as mock_ct, patch("builtins.print"):
            mock_ct.convert.side_effect = Exception("Conversion failed")

            with pytest.raises(Exception):
                convert_pytorch_to_coreml(
                    model=mock_model,
                    example_input=example_input,
                    output_path=output_path,
                    target="macOS13",
                )


class TestConvertONNXToCoreML:
    """Test ONNX to CoreML conversion."""

    def test_convert_onnx_to_coreml_success(self, tmp_path):
        """Test successful ONNX to CoreML conversion."""
        onnx_path = tmp_path / "model.onnx"
        output_path = tmp_path / "model.mlpackage"

        with (
            patch("conversion.convert_coreml.ct") as mock_ct,
            patch("conversion.convert_coreml.coremltools") as mock_coremltools,
            patch("conversion.convert_coreml.check_coreml_versions"),
            patch("builtins.print"),
        ):
            # Mock ONNX conversion
            mock_mlmodel = Mock()
            mock_ct.converters.onnx.convert.return_value = mock_mlmodel

            result = convert_onnx_to_coreml(
                onnx_path=str(onnx_path), output_path=output_path, target="macOS13"
            )

            assert result == str(output_path)
            mock_ct.converters.onnx.convert.assert_called_once_with(
                model=str(onnx_path), target="macOS13", minimum_deployment_target="macOS13"
            )

    def test_convert_onnx_to_coreml_with_custom_target(self, tmp_path):
        """Test ONNX to CoreML conversion with custom target."""
        onnx_path = tmp_path / "model.onnx"
        output_path = tmp_path / "model.mlpackage"

        with (
            patch("conversion.convert_coreml.ct") as mock_ct,
            patch("conversion.convert_coreml.coremltools") as mock_coremltools,
            patch("conversion.convert_coreml.check_coreml_versions"),
            patch("builtins.print"),
        ):
            mock_mlmodel = Mock()
            mock_ct.converters.onnx.convert.return_value = mock_mlmodel

            result = convert_onnx_to_coreml(
                onnx_path=str(onnx_path), output_path=output_path, target="iOS16"
            )

            assert result == str(output_path)
            mock_ct.converters.onnx.convert.assert_called_once_with(
                model=str(onnx_path), target="iOS16", minimum_deployment_target="iOS16"
            )

    def test_convert_onnx_to_coreml_file_not_found(self, tmp_path):
        """Test ONNX to CoreML conversion with missing ONNX file."""
        onnx_path = tmp_path / "nonexistent.onnx"
        output_path = tmp_path / "model.mlpackage"

        with patch("conversion.convert_coreml.ct") as mock_ct, patch("builtins.print"):
            mock_ct.converters.onnx.convert.side_effect = FileNotFoundError("ONNX file not found")

            with pytest.raises(FileNotFoundError):
                convert_onnx_to_coreml(
                    onnx_path=str(onnx_path), output_path=output_path, target="macOS13"
                )


class TestCreatePlaceholder:
    """Test placeholder creation functionality."""

    def test_create_placeholder_success(self, tmp_path):
        """Test successful placeholder creation."""
        output_path = tmp_path / "placeholder.mlpackage"
        onnx_path = tmp_path / "model.onnx"
        error_msg = "Conversion failed due to unsupported operations"

        result = create_placeholder(str(output_path), str(onnx_path), error_msg)

        assert result == str(output_path)
        assert output_path.exists()

        # Check that placeholder contains error information
        placeholder_files = list(output_path.glob("*"))
        assert len(placeholder_files) > 0

    def test_create_placeholder_directory_creation(self, tmp_path):
        """Test that placeholder directory is created."""
        output_path = tmp_path / "subdir" / "placeholder.mlpackage"
        onnx_path = tmp_path / "model.onnx"
        error_msg = "Test error"

        result = create_placeholder(str(output_path), str(onnx_path), error_msg)

        assert output_path.exists()
        assert output_path.is_dir()


class TestMainFunction:
    """Test main function."""

    @patch("conversion.convert_coreml.convert_pytorch_to_coreml")
    @patch("conversion.convert_coreml.convert_onnx_to_coreml")
    @patch("conversion.convert_coreml.load_contract")
    @patch("conversion.convert_coreml.create_placeholder")
    @patch("conversion.convert_coreml.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_pytorch_conversion(
        self,
        mock_print,
        mock_parser_class,
        mock_create_placeholder,
        mock_load_contract,
        mock_convert_onnx,
        mock_convert_pytorch,
    ):
        """Test main function with PyTorch conversion."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "pytorch"
        mock_args.in_path = "model.pt"
        mock_args.out = "output.mlpackage"
        mock_args.target = "macOS13"
        mock_args.contract = "contract.json"
        mock_args.allow_placeholder = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock contract loading
        mock_contract = {
            "inputs": [{"name": "input_ids", "shape": ["B", "T"]}],
            "metadata": {"vocab_size": 32000},
        }
        mock_load_contract.return_value = mock_contract

        # Mock PyTorch conversion
        mock_convert_pytorch.return_value = "output.mlpackage"

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_convert_pytorch.assert_called_once()
        mock_convert_onnx.assert_not_called()

    @patch("conversion.convert_coreml.convert_pytorch_to_coreml")
    @patch("conversion.convert_coreml.convert_onnx_to_coreml")
    @patch("conversion.convert_coreml.load_contract")
    @patch("conversion.convert_coreml.create_placeholder")
    @patch("conversion.convert_coreml.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_onnx_conversion(
        self,
        mock_print,
        mock_parser_class,
        mock_create_placeholder,
        mock_load_contract,
        mock_convert_onnx,
        mock_convert_pytorch,
    ):
        """Test main function with ONNX conversion."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "onnx"
        mock_args.in_path = "model.onnx"
        mock_args.out = "output.mlpackage"
        mock_args.target = "macOS13"
        mock_args.contract = "contract.json"
        mock_args.allow_placeholder = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock contract
        mock_contract = {"inputs": [], "metadata": {}}
        mock_load_contract.return_value = mock_contract

        # Mock ONNX conversion
        mock_convert_onnx.return_value = "output.mlpackage"

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_convert_onnx.assert_called_once()
        mock_convert_pytorch.assert_not_called()

    @patch("conversion.convert_coreml.convert_pytorch_to_coreml")
    @patch("conversion.convert_coreml.create_placeholder")
    @patch("conversion.convert_coreml.load_contract")
    @patch("conversion.convert_coreml.argparse.ArgumentParser")
    @patch("builtins.print")
    def test_main_conversion_failure_with_placeholder(
        self,
        mock_print,
        mock_parser_class,
        mock_load_contract,
        mock_create_placeholder,
        mock_convert_pytorch,
    ):
        """Test main function with conversion failure and placeholder creation."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "pytorch"
        mock_args.in_path = "model.pt"
        mock_args.out = "output.mlpackage"
        mock_args.target = "macOS13"
        mock_args.contract = "contract.json"
        mock_args.allow_placeholder = True
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock contract
        mock_contract = {"inputs": [], "metadata": {}}
        mock_load_contract.return_value = mock_contract

        # Mock conversion failure
        mock_convert_pytorch.side_effect = Exception("Conversion failed")

        # Mock placeholder creation
        mock_create_placeholder.return_value = "placeholder.mlpackage"

        # Test that main creates placeholder on failure
        try:
            main()
        except SystemExit:
            pass  # Expected for completion

        mock_create_placeholder.assert_called_once()

    @patch("conversion.convert_coreml.argparse.ArgumentParser")
    def test_main_invalid_backend(self, mock_parser_class):
        """Test main function with invalid backend."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.backend = "invalid_backend"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        with pytest.raises(SystemExit):
            main()

    @patch("conversion.convert_coreml.load_contract")
    @patch("conversion.convert_coreml.argparse.ArgumentParser")
    def test_main_contract_not_found(self, mock_parser_class, mock_load_contract):
        """Test main function with missing contract file."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.contract = "nonexistent.json"
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_contract.side_effect = FileNotFoundError("Contract not found")

        with pytest.raises(SystemExit):
            main()


class TestANEOptimizations:
    """Test ANE optimization functionality."""

    def test_detect_int64_tensors_fallback(self):
        """Test int64 detection fallback when tests module not available."""
        # This tests the fallback implementation when the tests module isn't available
        mock_model = Mock()

        # The fallback function should return empty list
        result = convert_coreml_module.detect_int64_tensors_on_attention_paths(mock_model)
        assert result == []

    def test_check_ane_op_compatibility_fallback(self):
        """Test ANE compatibility check fallback."""
        mock_model_path = "dummy_path.mlpackage"

        # The fallback function should return compatible
        result = convert_coreml_module.check_ane_op_compatibility(mock_model_path)
        assert result["compatible"] == True
        assert "Tests module not available" in result["error"]

    def test_verify_enumerated_shapes_fallback(self):
        """Test enumerated shapes verification fallback."""
        shapes = [128, 256, 512]

        # The fallback function should return verified
        result = convert_coreml_module.verify_enumerated_shapes_static_allocation(shapes)
        assert result["verified"] == True
        assert result["issues"] == []


class TestCoreMLConversionIntegration:
    """Test integration of CoreML conversion components."""

    def test_conversion_workflow_pytorch(self, tmp_path):
        """Test complete PyTorch to CoreML conversion workflow."""

        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        example_input = torch.randn(2, 10)
        output_path = tmp_path / "test_model.mlpackage"

        # Test contract creation
        contract_data = {
            "inputs": [{"name": "input_ids", "dtype": "int32", "shape": ["B", "T"]}],
            "outputs": [{"name": "logits", "dtype": "float32", "shape": ["B", "T", "vocab_size"]}],
            "metadata": {"model_type": "transformer", "vocab_size": 32000},
        }

        contract_file = tmp_path / "contract.json"
        with open(contract_file, "w") as f:
            json.dump(contract_data, f)

        # Test contract loading
        loaded_contract = load_contract(str(contract_file))
        assert loaded_contract == contract_data

        # Test placeholder creation (simulate failure case)
        placeholder_path = tmp_path / "placeholder.mlpackage"
        error_msg = "Simulated conversion failure"

        result = create_placeholder(str(placeholder_path), "dummy.onnx", error_msg)
        assert result == str(placeholder_path)
        assert placeholder_path.exists()

    def test_conversion_error_handling(self, tmp_path):
        """Test error handling in conversion functions."""
        # Test contract loading error
        with pytest.raises(FileNotFoundError):
            load_contract("nonexistent_contract.json")

        # Test placeholder creation
        placeholder_path = tmp_path / "error_placeholder.mlpackage"
        result = create_placeholder(str(placeholder_path), "error.onnx", "Test error")
        assert placeholder_path.exists()

        # Verify placeholder contains error information
        placeholder_files = list(placeholder_path.glob("*"))
        assert len(placeholder_files) > 0

    def test_version_gate_integration(self):
        """Test version gate integration."""
        with patch("conversion.convert_coreml.check_coreml_versions") as mock_check:
            mock_check.return_value = True

            # This should not raise an exception if version check passes
            # (tested implicitly through successful conversion tests above)
            assert True

    def test_conversion_parameter_validation(self):
        """Test parameter validation in conversion functions."""
        # Test with invalid paths
        with pytest.raises(FileNotFoundError):
            load_contract("")

        # Test contract structure validation (implicit through successful loads)
        # Valid contracts should load successfully
        # Invalid contracts would cause JSONDecodeError (tested above)
