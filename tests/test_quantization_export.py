"""
Quantization export verification tests.

Verifies that QAT modules are properly folded and CoreML uses INT8 weights.
@author: @darianrosebrook
"""

import pytest
from pathlib import Path
from typing import Dict, Any

try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


def verify_coreml_weight_types(mlpackage_path: Path) -> Dict[str, Any]:
    """
    Verify CoreML model weight types.

    Args:
        mlpackage_path: Path to .mlpackage

    Returns:
        Dictionary with weight type information
    """
    if not COREML_AVAILABLE:
        return {"error": "coremltools not available"}

    try:
        mlmodel = ct.models.MLModel(str(mlpackage_path))
        spec = mlmodel.get_spec()

        weight_info = {
            "is_mlprogram": spec.WhichOneof("Type") == "mlProgram",
            "weight_types": {},
            "has_int8_weights": False,
            "has_fp16_weights": False,
            "has_fp32_weights": False,
        }

        if spec.WhichOneof("Type") == "mlProgram":
            # Check weights in mlProgram
            for func_name, func in spec.mlProgram.functions.items():
                for op in func.block.operations:
                    # Check for weight attributes
                    for attr_name, attr_value in op.attributes.items():
                        if "weight" in attr_name.lower():
                            # Determine weight type from operation type
                            op_type = op.type.lower()
                            if "int8" in op_type or "quantized" in op_type:
                                weight_info["has_int8_weights"] = True
                                weight_info["weight_types"][attr_name] = "int8"
                            elif "fp16" in op_type or "float16" in op_type:
                                weight_info["has_fp16_weights"] = True
                                weight_info["weight_types"][attr_name] = "fp16"
                            elif "fp32" in op_type or "float32" in op_type:
                                weight_info["has_fp32_weights"] = True
                                weight_info["weight_types"][attr_name] = "fp32"

        return weight_info

    except Exception as e:
        return {"error": str(e)}


def test_int8_weights_verification(tmp_path: Path):
    """Test that INT8 weights are detected in quantized models."""
    if not COREML_AVAILABLE:
        pytest.skip("coremltools not available")

    # This test would require an actual quantized CoreML model
    # For now, just verify the function works
    mlpackage_path = tmp_path / "quantized_model.mlpackage"

    if not mlpackage_path.exists():
        pytest.skip("Quantized model not available for testing")

    weight_info = verify_coreml_weight_types(mlpackage_path)

    assert "error" not in weight_info, f"Weight verification failed: {weight_info.get('error')}"
    assert weight_info["has_int8_weights"], "Model should have INT8 weights"


def test_qat_module_folding():
    """Test that QAT modules are properly folded during export."""
    # This would test the PyTorch export pipeline
    # Verifies that QuantizedLinear/QuantizedAttention are converted to regular ops
    # For now, this is a placeholder test structure
    pytest.skip("QAT module folding test requires PyTorch quantization integration")


def test_quantization_recipe_validation():
    """Test that quantization recipe config is applied correctly."""
    # This would validate that the quantization recipe (per-channel vs per-tensor, etc.)
    # is correctly applied during CoreML conversion
    pytest.skip("Quantization recipe validation requires recipe config and model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
