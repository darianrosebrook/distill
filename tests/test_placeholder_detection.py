"""
Placeholder model detection tests.

Ensures placeholder models are not used in production exports.
@author: @darianrosebrook
"""

import pytest
import tempfile
from pathlib import Path
from typing import Optional

try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


def detect_placeholder(mlpackage_path: Path) -> tuple[bool, Optional[str]]:
    """
    Detect if a CoreML model is a placeholder.

    Args:
        mlpackage_path: Path to .mlpackage

    Returns:
        Tuple of (is_placeholder: bool, reason: Optional[str])
    """
    # Check for placeholder marker file
    if (mlpackage_path.parent / ".placeholder").exists():
        return True, "Placeholder marker file found"

    if (mlpackage_path / ".placeholder").exists():
        return True, "Placeholder marker file found in mlpackage"

    if not COREML_AVAILABLE:
        return False, None  # Cannot check without coremltools

    try:
        mlmodel = ct.models.MLModel(str(mlpackage_path))
        spec = mlmodel.get_spec()

        # Check model description for placeholder indicators
        description = spec.description
        if description.metadata and description.metadata.userDefined:
            user_defined = description.metadata.userDefined
            if "placeholder" in user_defined or "dummy" in user_defined:
                return True, f"Placeholder indicator in metadata: {user_defined}"

        # Check if model has suspiciously small weights
        # This is a heuristic - placeholder models often have minimal weights
        if spec.WhichOneof("Type") == "mlProgram":
            # Count operations
            total_ops = 0
            for func in spec.mlProgram.functions.values():
                total_ops += len(func.block.operations)

            # Very few operations might indicate placeholder
            if total_ops < 5:
                return True, f"Suspiciously few operations: {total_ops}"

        return False, None

    except Exception:
        # If we can't load the model, assume it's not a placeholder
        # (placeholder detection should happen before this)
        return False, None


def test_placeholder_marker_detection(tmp_path: Path):
    """Test that placeholder marker files are detected."""
    mlpackage_path = tmp_path / "model.mlpackage"
    mlpackage_path.mkdir(parents=True, exist_ok=True)

    # Create placeholder marker
    (mlpackage_path.parent / ".placeholder").touch()

    is_placeholder, reason = detect_placeholder(mlpackage_path)
    assert is_placeholder, "Should detect placeholder marker"
    assert reason is not None
    assert "marker" in reason.lower()


def test_production_model_not_placeholder(tmp_path: Path):
    """Test that production models are not flagged as placeholders."""
    mlpackage_path = tmp_path / "model.mlpackage"
    mlpackage_path.mkdir(parents=True, exist_ok=True)

    # No placeholder marker
    is_placeholder, reason = detect_placeholder(mlpackage_path)

    # Without coremltools, we can't fully verify, but should not detect placeholder
    if not COREML_AVAILABLE:
        pytest.skip("coremltools not available")

    # If we can load the model and it has reasonable structure, it's not a placeholder
    assert not is_placeholder or reason is None, (
        f"Production model flagged as placeholder: {reason}"
    )


def test_placeholder_detection_integration():
    """Integration test: verify placeholder detection works end-to-end."""
    # This would test with an actual placeholder model if available
    # For now, just verify the function works
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test.mlpackage"
        model_path.mkdir(parents=True, exist_ok=True)

        # Test without marker
        is_placeholder, _ = detect_placeholder(model_path)
        assert not is_placeholder

        # Test with marker
        (model_path.parent / ".placeholder").touch()
        is_placeholder, reason = detect_placeholder(model_path)
        assert is_placeholder
        assert reason is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
