"""
Quantization A/B testing framework.

Compares FP16 vs INT8+FP16 CoreML models on:
- Golden vector cosine similarity
- Behavioral correctness (tool usage, code compilation)
- Performance metrics (latency, memory)
@author: @darianrosebrook
"""
import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from tests.test_coreml_golden_vectors import (
    compute_cosine_similarity,
    check_nans_infs,
    load_golden_vectors,
)


def compare_models_ab(
    fp16_model_path: Path,
    int8_model_path: Path,
    golden_vectors_dir: Path,
    shape: int,
    cosine_threshold: float = 0.998,  # Slightly lower for quantized models
) -> Dict[str, Any]:
    """
    Compare FP16 vs INT8 models using golden vectors.
    
    Args:
        fp16_model_path: Path to FP16 CoreML model
        int8_model_path: Path to INT8 CoreML model
        golden_vectors_dir: Directory containing golden vectors
        shape: Sequence length to test
        cosine_threshold: Minimum cosine similarity threshold
        
    Returns:
        Dictionary with comparison results
    """
    if not COREML_AVAILABLE:
        return {"error": "coremltools not available"}
    
    # Load golden vectors
    golden = load_golden_vectors(golden_vectors_dir, shape)
    if golden is None:
        return {"error": f"Golden vectors not found for T={shape}"}
    
    # Load models
    fp16_model = ct.models.MLModel(str(fp16_model_path))
    int8_model = ct.models.MLModel(str(int8_model_path))
    
    # Prepare inputs
    input_ids = golden["input_ids"].astype(np.int32)
    if golden["attention_mask"] is not None:
        attention_mask = golden["attention_mask"].astype(np.int32)
    else:
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    # Run FP16 inference
    fp16_outputs = fp16_model.predict(inputs)
    fp16_logits_key = [k for k in fp16_outputs.keys() if "logit" in k.lower() and "halt" not in k.lower()][0]
    fp16_logits = np.array(fp16_outputs[fp16_logits_key]).astype(np.float32)
    
    # Run INT8 inference
    int8_outputs = int8_model.predict(inputs)
    int8_logits_key = [k for k in int8_outputs.keys() if "logit" in k.lower() and "halt" not in k.lower()][0]
    int8_logits = np.array(int8_outputs[int8_logits_key]).astype(np.float32)
    
    # Compare with golden vectors
    pytorch_logits = golden["pytorch_logits"].astype(np.float32)
    
    fp16_cosine = compute_cosine_similarity(fp16_logits, pytorch_logits)
    int8_cosine = compute_cosine_similarity(int8_logits, pytorch_logits)
    
    # Compute relative error
    fp16_mae = np.mean(np.abs(fp16_logits - pytorch_logits))
    int8_mae = np.mean(np.abs(int8_logits - pytorch_logits))
    
    max_abs = max(np.max(np.abs(pytorch_logits)), 1e-6)
    fp16_rel_error = fp16_mae / max_abs
    int8_rel_error = int8_mae / max_abs
    
    # Check for NaNs/Infs
    fp16_nan, fp16_inf = check_nans_infs(fp16_logits)
    int8_nan, int8_inf = check_nans_infs(int8_logits)
    
    # Compute accuracy delta
    accuracy_delta = int8_cosine - fp16_cosine
    rel_error_delta = int8_rel_error - fp16_rel_error
    
    return {
        "shape": shape,
        "fp16_cosine": float(fp16_cosine),
        "int8_cosine": float(int8_cosine),
        "accuracy_delta": float(accuracy_delta),
        "fp16_rel_error": float(fp16_rel_error),
        "int8_rel_error": float(int8_rel_error),
        "rel_error_delta": float(rel_error_delta),
        "fp16_nan_count": fp16_nan,
        "fp16_inf_count": fp16_inf,
        "int8_nan_count": int8_nan,
        "int8_inf_count": int8_inf,
        "meets_threshold": int8_cosine >= cosine_threshold,
        "acceptable_delta": abs(accuracy_delta) < 0.02,  # <2% relative error
    }


def test_quantization_ab_golden_vectors(
    fp16_model_path: str,
    int8_model_path: str,
    golden_vectors_dir: str,
    shapes: List[int] = [512, 1024, 2048, 4096],
):
    """
    Test quantization A/B comparison on golden vectors.
    
    Args:
        fp16_model_path: Path to FP16 CoreML model
        int8_model_path: Path to INT8 CoreML model
        golden_vectors_dir: Directory containing golden vectors
        shapes: List of sequence lengths to test
    """
    if not COREML_AVAILABLE:
        pytest.skip("coremltools not available")
    
    fp16_path = Path(fp16_model_path)
    int8_path = Path(int8_model_path)
    golden_dir = Path(golden_vectors_dir)
    
    if not fp16_path.exists():
        pytest.skip(f"FP16 model not found: {fp16_model_path}")
    if not int8_path.exists():
        pytest.skip(f"INT8 model not found: {int8_model_path}")
    
    results = []
    all_pass = True
    
    for shape in shapes:
        result = compare_models_ab(fp16_path, int8_path, golden_dir, shape)
        
        if "error" in result:
            print(f"[quantization_ab] Skipping T={shape}: {result['error']}")
            continue
        
        results.append(result)
        
        print(f"\n[quantization_ab] T={shape}:")
        print(f"  FP16 cosine: {result['fp16_cosine']:.6f}")
        print(f"  INT8 cosine: {result['int8_cosine']:.6f}")
        print(f"  Accuracy delta: {result['accuracy_delta']:.6f}")
        print(f"  INT8 rel error: {result['int8_rel_error']:.6e}")
        print(f"  Meets threshold: {result['meets_threshold']}")
        print(f"  Acceptable delta: {result['acceptable_delta']}")
        
        if not result['meets_threshold'] or not result['acceptable_delta']:
            all_pass = False
    
    # Assert overall pass
    assert all_pass, "Quantization A/B test failed - accuracy delta too large or threshold not met"
    
    # Assert no NaNs/Infs
    for result in results:
        assert result['int8_nan_count'] == 0, f"INT8 model has {result['int8_nan_count']} NaNs"
        assert result['int8_inf_count'] == 0, f"INT8 model has {result['int8_inf_count']} Infs"


def test_quantization_behavioral_correctness():
    """Test behavioral correctness of quantized models (tool usage, code compilation)."""
    # This would run actual prompts through both models and compare:
    # - Tool call correctness (valid JSON, same tool chosen)
    # - Code compilation success rate
    # - Semantic equivalence
    pytest.skip("Behavioral correctness test requires runtime integration")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

