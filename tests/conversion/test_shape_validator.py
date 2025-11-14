"""
Tests for conversion/shape_validator.py - Shape validation functionality.

Tests shape validation utilities including single shape validation,
enumerated shape validation, shape retrieval functions, and result checking.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch

# Import the module using importlib due to filename with digits
import importlib
shape_validator_module = importlib.import_module("conversion.shape_validator")

# Import functions and classes from the module
validate_shape_with_model = shape_validator_module.validate_shape_with_model
validate_enumerated_shapes = shape_validator_module.validate_enumerated_shapes
get_production_shapes = shape_validator_module.get_production_shapes
get_toy_shapes = shape_validator_module.get_toy_shapes
get_primary_shape = shape_validator_module.get_primary_shape
check_shape_validation_results = shape_validator_module.check_shape_validation_results


class TestValidateShapeWithModel:
    """Test single shape validation functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        return model

    def test_validate_shape_with_model_success(self, mock_model):
        """Test successful shape validation."""
        vocab_size = 1000
        shape = 128

        # Mock successful forward pass
        mock_output = torch.randn(1, shape, vocab_size)
        mock_model.return_value = mock_output

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "ok"
        assert result["error"] is None
        assert result["output_shape"] == (1, shape, vocab_size)
        mock_model.eval.assert_called_once()

    def test_validate_shape_with_model_tuple_output(self, mock_model):
        """Test shape validation with tuple output (e.g., with loss)."""
        vocab_size = 1000
        shape = 128

        # Mock tuple output
        mock_logits = torch.randn(1, shape, vocab_size)
        mock_loss = torch.tensor(2.5)
        mock_model.return_value = (mock_logits, mock_loss)

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "ok"
        assert result["output_shape"] == (1, shape, vocab_size)

    def test_validate_shape_with_model_shape_mismatch(self, mock_model):
        """Test shape validation with incorrect output shape."""
        vocab_size = 1000
        shape = 128

        # Mock wrong output shape
        mock_output = torch.randn(1, shape, vocab_size + 100)  # Wrong vocab size
        mock_model.return_value = mock_output

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "error"
        assert "Output shape mismatch" in result["error"]
        assert result["output_shape"] == (1, shape, vocab_size + 100)

    def test_validate_shape_with_model_nan_values(self, mock_model):
        """Test shape validation with NaN values in output."""
        vocab_size = 1000
        shape = 128

        # Mock output with NaN
        mock_output = torch.randn(1, shape, vocab_size)
        mock_output[0, 0, 0] = float('nan')
        mock_model.return_value = mock_output

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "error"
        assert "NaN or Inf values" in result["error"]

    def test_validate_shape_with_model_inf_values(self, mock_model):
        """Test shape validation with Inf values in output."""
        vocab_size = 1000
        shape = 128

        # Mock output with Inf
        mock_output = torch.randn(1, shape, vocab_size)
        mock_output[0, 0, 0] = float('inf')
        mock_model.return_value = mock_output

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "error"
        assert "NaN or Inf values" in result["error"]

    def test_validate_shape_with_model_runtime_error(self, mock_model):
        """Test shape validation with RuntimeError."""
        vocab_size = 1000
        shape = 128

        # Mock RuntimeError (using CPU to avoid CUDA issues)
        mock_model.side_effect = RuntimeError("Some runtime error")

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "error"
        assert result["error"].startswith("RuntimeError:")
        assert "Some runtime error" in result["error"]
        assert result["output_shape"] is None

    def test_validate_shape_with_model_unexpected_error(self, mock_model):
        """Test shape validation with unexpected error."""
        vocab_size = 1000
        shape = 128

        # Mock unexpected error
        mock_model.side_effect = ValueError("Unexpected error")

        result = validate_shape_with_model(mock_model, shape, vocab_size, "cpu")

        assert result["shape"] == shape
        assert result["status"] == "error"
        assert "Unexpected error: Unexpected error" == result["error"]
        assert result["output_shape"] is None


class TestValidateEnumeratedShapes:
    """Test enumerated shape validation functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        return model

    def test_validate_enumerated_shapes_success(self, mock_model):
        """Test successful enumerated shape validation."""
        vocab_size = 1000
        shapes = [64, 128, 256]
        primary_shape = 128

        # Mock successful results for all shapes
        def mock_validate_shape(model, shape, vocab, device):
            return {
                "shape": shape,
                "status": "ok",
                "error": None,
                "output_shape": (1, shape, vocab)
            }

        with patch('conversion.shape_validator.validate_shape_with_model', side_effect=mock_validate_shape):
            result = validate_enumerated_shapes(mock_model, shapes, vocab_size, primary_shape, "cpu")

        assert result["primary_shape"] == primary_shape
        assert result["primary_status"] == "ok"
        assert result["all_ok"] is True
        assert result["primary_ok"] is True
        assert len(result["results"]) == 3

    def test_validate_enumerated_shapes_default_primary(self, mock_model):
        """Test enumerated shape validation with default primary shape."""
        vocab_size = 1000
        shapes = [64, 128, 256]

        def mock_validate_shape(model, shape, vocab, device):
            return {
                "shape": shape,
                "status": "ok",
                "error": None,
                "output_shape": (1, shape, vocab)
            }

        with patch('conversion.shape_validator.validate_shape_with_model', side_effect=mock_validate_shape):
            result = validate_enumerated_shapes(mock_model, shapes, vocab_size, device="cpu")

        assert result["primary_shape"] == 64  # First shape is default primary

    def test_validate_enumerated_shapes_empty_shapes(self, mock_model):
        """Test enumerated shape validation with empty shape list."""
        vocab_size = 1000
        shapes = []

        result = validate_enumerated_shapes(mock_model, shapes, vocab_size, device="cpu")

        assert result["primary_shape"] is None
        assert result["primary_status"] == "error"
        assert result["all_ok"] is False
        assert result["primary_ok"] is False
        assert result["error"] == "No shapes provided"
        assert result["results"] == []

    def test_validate_enumerated_shapes_primary_not_in_shapes(self, mock_model):
        """Test enumerated shape validation with primary shape not in shapes list."""
        vocab_size = 1000
        shapes = [64, 128, 256]
        primary_shape = 512

        result = validate_enumerated_shapes(mock_model, shapes, vocab_size, primary_shape, "cpu")

        assert result["primary_shape"] == primary_shape
        assert result["primary_status"] == "error"
        assert result["all_ok"] is False
        assert result["primary_ok"] is False
        assert result["error"] == "Primary shape 512 not in enumerated_shapes"
        assert result["results"] == []

    def test_validate_enumerated_shapes_partial_failure(self, mock_model):
        """Test enumerated shape validation with some shapes failing."""
        vocab_size = 1000
        shapes = [64, 128, 256]
        primary_shape = 128

        def mock_validate_shape(model, shape, vocab, device):
            if shape == 64:
                return {
                    "shape": shape,
                    "status": "error",
                    "error": "Shape too small",
                    "output_shape": None
                }
            else:
                return {
                    "shape": shape,
                    "status": "ok",
                    "error": None,
                    "output_shape": (1, shape, vocab)
                }

        with patch('conversion.shape_validator.validate_shape_with_model', side_effect=mock_validate_shape):
            result = validate_enumerated_shapes(mock_model, shapes, vocab_size, primary_shape, "cpu")

        assert result["primary_shape"] == primary_shape
        assert result["primary_status"] == "ok"
        assert result["all_ok"] is False  # Not all shapes pass
        assert result["primary_ok"] is True  # Primary shape passes
        assert len(result["results"]) == 3


class TestGetProductionShapes:
    """Test production shape retrieval."""

    def test_get_production_shapes(self):
        """Test production shape retrieval."""
        shapes = get_production_shapes()

        assert shapes == [512, 1024, 2048, 4096]
        assert isinstance(shapes, list)
        assert all(isinstance(s, int) for s in shapes)


class TestGetToyShapes:
    """Test toy shape retrieval."""

    def test_get_toy_shapes(self):
        """Test toy shape retrieval."""
        shapes = get_toy_shapes()

        assert shapes == [64, 128, 256]
        assert isinstance(shapes, list)
        assert all(isinstance(s, int) for s in shapes)


class TestGetPrimaryShape:
    """Test primary shape selection functionality."""

    def test_get_primary_shape_toy_with_128(self):
        """Test primary shape selection for toy models with 128 available."""
        shapes = [64, 128, 256]

        result = get_primary_shape(shapes, is_toy=True)

        assert result == 128

    def test_get_primary_shape_toy_without_128(self):
        """Test primary shape selection for toy models without 128."""
        shapes = [64, 256, 512]

        result = get_primary_shape(shapes, is_toy=True)

        assert result == 64  # First shape fallback

    def test_get_primary_shape_production_with_1024(self):
        """Test primary shape selection for production models with 1024 available."""
        shapes = [512, 1024, 2048]

        result = get_primary_shape(shapes, is_toy=False)

        assert result == 1024

    def test_get_primary_shape_production_without_1024(self):
        """Test primary shape selection for production models without 1024."""
        shapes = [512, 2048, 4096]

        result = get_primary_shape(shapes, is_toy=False)

        assert result == 512  # First shape fallback

    def test_get_primary_shape_empty_shapes_toy(self):
        """Test primary shape selection with empty shapes for toy model."""
        shapes = []

        result = get_primary_shape(shapes, is_toy=True)

        assert result == 128  # Default fallback

    def test_get_primary_shape_empty_shapes_production(self):
        """Test primary shape selection with empty shapes for production model."""
        shapes = []

        result = get_primary_shape(shapes, is_toy=False)

        assert result == 128  # Default fallback


class TestCheckShapeValidationResults:
    """Test shape validation result checking functionality."""

    def test_check_shape_validation_results_all_success(self):
        """Test result checking with all shapes successful."""
        validation_results = {
            "primary_shape": 128,
            "primary_status": "ok",
            "results": [
                {"shape": 64, "status": "ok"},
                {"shape": 128, "status": "ok"},
                {"shape": 256, "status": "ok"}
            ],
            "all_ok": True,
            "primary_ok": True
        }

        success, errors = check_shape_validation_results(validation_results, require_all=False)

        assert success is True
        assert errors == []

    def test_check_shape_validation_results_primary_failure(self):
        """Test result checking with primary shape failure."""
        validation_results = {
            "primary_shape": 128,
            "primary_status": "error",
            "results": [
                {"shape": 64, "status": "ok"},
                {"shape": 128, "status": "error", "error": "Shape validation failed"},
                {"shape": 256, "status": "ok"}
            ],
            "all_ok": False,
            "primary_ok": False
        }

        success, errors = check_shape_validation_results(validation_results, require_all=False)

        assert success is False
        assert len(errors) == 2  # Primary failure + detailed error
        assert "Primary shape 128 validation failed" in errors
        assert "Shape 128: Shape validation failed" in errors

    def test_check_shape_validation_results_require_all_failure(self):
        """Test result checking requiring all shapes to pass."""
        validation_results = {
            "primary_shape": 128,
            "primary_status": "ok",
            "results": [
                {"shape": 64, "status": "ok"},
                {"shape": 128, "status": "ok"},
                {"shape": 256, "status": "error", "error": "Shape too large"}
            ],
            "all_ok": False,
            "primary_ok": True
        }

        success, errors = check_shape_validation_results(validation_results, require_all=True)

        assert success is False
        assert len(errors) == 2  # All required failure + detailed error
        assert "Shape validation failed for shapes: [256]" in errors
        assert "Shape 256: Shape too large" in errors

    def test_check_shape_validation_results_require_all_success(self):
        """Test result checking requiring all shapes to pass when all pass."""
        validation_results = {
            "primary_shape": 128,
            "primary_status": "ok",
            "results": [
                {"shape": 64, "status": "ok"},
                {"shape": 128, "status": "ok"},
                {"shape": 256, "status": "ok"}
            ],
            "all_ok": True,
            "primary_ok": True
        }

        success, errors = check_shape_validation_results(validation_results, require_all=True)

        assert success is True
        assert errors == []

    def test_check_shape_validation_results_no_results(self):
        """Test result checking with missing results."""
        validation_results = {
            "primary_shape": 128,
            "primary_status": "ok",
            "all_ok": True,
            "primary_ok": True
            # Missing "results" key
        }

        success, errors = check_shape_validation_results(validation_results, require_all=False)

        assert success is True
        assert errors == []
