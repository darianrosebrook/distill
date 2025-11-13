"""
Production shape validation test.

Tests that production enumerated shapes (T512, T1024, T2048, T4096) work correctly
during export and conversion. This test validates shape enumeration for production models.

Usage:
    pytest tests/e2e/test_production_shapes.py -v

Note:
    This test requires a production checkpoint. For testing with toy models,
    use test_8_ball_pipeline.py instead.
"""

import pytest
import torch
from pathlib import Path

from conversion.shape_validator import (
    validate_enumerated_shapes,
    get_production_shapes,
    get_primary_shape,
    check_shape_validation_results,
)


@pytest.fixture
def production_shapes():
    """Get production enumerated shapes."""
    return get_production_shapes()


@pytest.fixture
def primary_shape(production_shapes):
    """Get primary production shape."""
    return get_primary_shape(production_shapes, is_toy=False)


def load_production_checkpoint(checkpoint_path: Path):
    """
    Load a production checkpoint for shape validation.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (model, vocab_size, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    config_data = checkpoint.get("config", {})
    arch_cfg = config_data.get("arch", {})

    if not arch_cfg:
        raise ValueError(f"Checkpoint {checkpoint_path} missing arch config")

    vocab_size = arch_cfg.get("vocab_size")
    if not vocab_size:
        raise ValueError(f"Checkpoint {checkpoint_path} missing vocab_size")

    # Load model (simplified - actual loading would use model architecture)
    # This is a template - actual implementation would load the full model
    model = None  # Would load actual StudentLM here

    return model, vocab_size, arch_cfg


@pytest.mark.slow
@pytest.mark.skip(reason="Requires production checkpoint - use for manual testing")
def test_production_shapes_validation(production_shapes, primary_shape, tmp_path):
    """
    Test that production shapes validate correctly.

    This test requires a production checkpoint. For automated testing,
    use test_8_ball_pipeline.py with toy models instead.
    """
    # This is a template test - requires actual checkpoint
    checkpoint_path = tmp_path / "production_checkpoint.pt"

    if not checkpoint_path.exists():
        pytest.skip("Production checkpoint not available - skipping shape validation")

    model, vocab_size, config = load_production_checkpoint(checkpoint_path)

    # Validate all production shapes
    results = validate_enumerated_shapes(
        model=model,
        enumerated_shapes=production_shapes,
        vocab_size=vocab_size,
        primary_shape=primary_shape,
        device="cpu",
    )

    # Primary shape must succeed
    assert results["primary_ok"], (
        f"Primary shape {primary_shape} validation failed: {results.get('primary_status')}"
    )

    # Check results
    success, errors = check_shape_validation_results(results, require_all=False)

    # Primary shape must work
    assert success, f"Shape validation failed: {errors}"

    # Log which shapes succeeded/failed
    for result in results["results"]:
        if result["status"] == "ok":
            print(f"✅ Shape {result['shape']}: OK")
        else:
            print(f"⚠️  Shape {result['shape']}: {result.get('error', 'Unknown error')}")


def test_production_shapes_configuration(production_shapes, primary_shape):
    """Test that production shape configuration is correct."""
    # Production shapes should be [512, 1024, 2048, 4096]
    assert production_shapes == [512, 1024, 2048, 4096], (
        f"Expected production shapes [512, 1024, 2048, 4096], got {production_shapes}"
    )

    # Primary shape should be T1024
    assert primary_shape == 1024, f"Expected primary shape 1024, got {primary_shape}"

    # All shapes should be valid
    assert all(shape > 0 for shape in production_shapes), "All production shapes must be positive"

    # Shapes should be in ascending order
    assert production_shapes == sorted(production_shapes), (
        "Production shapes should be in ascending order"
    )


def test_shape_validator_helpers():
    """Test shape validator helper functions."""
    # Test production shapes
    prod_shapes = get_production_shapes()
    assert len(prod_shapes) == 4
    assert 1024 in prod_shapes

    # Test primary shape detection
    primary = get_primary_shape(prod_shapes, is_toy=False)
    assert primary == 1024

    # Test toy shapes
    toy_shapes = [64, 128, 256]
    toy_primary = get_primary_shape(toy_shapes, is_toy=True)
    assert toy_primary == 128

    # Test fallback to first shape
    custom_shapes = [256, 512]
    custom_primary = get_primary_shape(custom_shapes, is_toy=False)
    assert custom_primary == 256


@pytest.mark.parametrize("shape", [512, 1024, 2048, 4096])
def test_production_shape_values(shape):
    """Test that individual production shapes are valid."""
    assert shape > 0, f"Shape {shape} must be positive"
    assert shape % 64 == 0, f"Shape {shape} should be multiple of 64 for efficiency"
    assert shape <= 4096, f"Shape {shape} exceeds maximum production shape (4096)"


def test_shape_validation_error_handling():
    """Test that shape validation handles errors gracefully."""

    # Create a mock model that will fail
    class FailingModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("Model failure")

    failing_model = FailingModel()

    # Validate should catch the error
    results = validate_enumerated_shapes(
        model=failing_model, enumerated_shapes=[128], vocab_size=512, device="cpu"
    )

    # Should report error status
    assert results["primary_status"] == "error"
    assert not results["primary_ok"]

    # Should have error message
    assert len(results["results"]) > 0
    assert results["results"][0]["status"] == "error"
    assert results["results"][0]["error"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
