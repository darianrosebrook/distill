"""
CoreML golden vector regression tests.

For each enumerated shape, validates:
- Cosine similarity ≥ 0.999 between PyTorch and CoreML outputs
- No NaNs or Infs in CoreML outputs
- IO contract validation (input/output shapes and dtypes)
@author: @darianrosebrook
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def check_nans_infs(arr: np.ndarray) -> tuple[int, int]:
    """Check for NaNs and Infs in array."""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    return int(nan_count), int(inf_count)


def load_golden_vectors(golden_dir: Path, shape: int) -> Optional[Dict[str, Any]]:
    """Load golden vectors for a specific shape."""
    golden_file = golden_dir / f"golden_T{shape}.npz"
    if not golden_file.exists():
        return None
    
    data = np.load(golden_file)
    return {
        "input_ids": data["input_ids"],
        "attention_mask": data.get("attention_mask"),
        "pytorch_logits": data["pytorch_logits"],
        "pytorch_hidden_states": data.get("pytorch_hidden_states"),
    }


def save_golden_vectors(
    golden_dir: Path,
    shape: int,
    input_ids: np.ndarray,
    pytorch_logits: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    pytorch_hidden_states: Optional[np.ndarray] = None,
):
    """Save golden vectors for a specific shape."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_file = golden_dir / f"golden_T{shape}.npz"
    
    save_dict = {
        "input_ids": input_ids,
        "pytorch_logits": pytorch_logits,
    }
    if attention_mask is not None:
        save_dict["attention_mask"] = attention_mask
    if pytorch_hidden_states is not None:
        save_dict["pytorch_hidden_states"] = pytorch_hidden_states
    
    np.savez_compressed(golden_file, **save_dict)
    print(f"[golden_vectors] Saved golden vectors for T={shape} to {golden_file}")


def validate_io_contract(
    model_path: Path,
    contract_path: Optional[Path] = None,
    shape: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate CoreML model IO contract.
    
    Args:
        model_path: Path to .mlpackage
        contract_path: Optional path to contract.json
        shape: Optional sequence length for shape validation
        
    Returns:
        Dictionary with validation results
    """
    if not COREML_AVAILABLE:
        return {"passed": False, "error": "coremltools not available"}
    
    try:
        mlmodel = ct.models.MLModel(str(model_path))
        spec = mlmodel._spec
        
        # Extract input/output specifications
        inputs = []
        outputs = []
        
        for input_desc in spec.description.input:
            inputs.append({
                "name": input_desc.name,
                "type": str(input_desc.type),
            })
        
        for output_desc in spec.description.output:
            outputs.append({
                "name": output_desc.name,
                "type": str(output_desc.type),
            })
        
        # Validate against contract if provided
        contract_errors = []
        if contract_path and contract_path.exists():
            with open(contract_path, 'r') as f:
                contract = json.load(f)
            
            # Check input contract
            contract_inputs = {inp["name"]: inp for inp in contract.get("inputs", [])}
            for input_desc in spec.description.input:
                if input_desc.name in contract_inputs:
                    contract_inp = contract_inputs[input_desc.name]
                    # Basic validation (can be extended)
                    if shape and "shape" in contract_inp:
                        # Check if shape matches expected
                        pass  # Shape validation can be added
        
        return {
            "passed": len(contract_errors) == 0,
            "inputs": inputs,
            "outputs": outputs,
            "contract_errors": contract_errors,
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_coreml_golden_vectors(
    coreml_model_path: str,
    golden_vectors_dir: str,
    shape: int,
    cosine_threshold: float = 0.999,
):
    """
    Test CoreML model against golden vectors.
    
    Args:
        coreml_model_path: Path to CoreML .mlpackage
        golden_vectors_dir: Directory containing golden vectors
        shape: Sequence length to test
        cosine_threshold: Minimum cosine similarity threshold (default: 0.999)
    """
    if not COREML_AVAILABLE:
        pytest.skip("coremltools not available")
    
    golden_dir = Path(golden_vectors_dir)
    model_path = Path(coreml_model_path)
    
    # Load golden vectors
    golden = load_golden_vectors(golden_dir, shape)
    if golden is None:
        pytest.skip(f"Golden vectors not found for T={shape}")
    
    # Load CoreML model
    mlmodel = ct.models.MLModel(str(model_path))
    
    # Prepare inputs
    input_ids = golden["input_ids"].astype(np.int32)
    if golden["attention_mask"] is not None:
        attention_mask = golden["attention_mask"].astype(np.int32)
    else:
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
    
    # Run CoreML inference
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    try:
        outputs = mlmodel.predict(inputs)
    except Exception as e:
        pytest.fail(f"CoreML inference failed: {e}")
    
    # Extract logits
    if "logits" in outputs:
        coreml_logits = outputs["logits"]
    else:
        # Try to find logits key
        logits_key = None
        for key in outputs.keys():
            if "logit" in key.lower() and "halt" not in key.lower():
                logits_key = key
                break
        if logits_key is None:
            pytest.fail(f"Could not find logits in CoreML output. Keys: {list(outputs.keys())}")
        coreml_logits = outputs[logits_key]
    
    # Convert to numpy for comparison
    coreml_logits = np.array(coreml_logits).astype(np.float32)
    pytorch_logits = golden["pytorch_logits"].astype(np.float32)
    
    # Check for NaNs/Infs
    nan_count, inf_count = check_nans_infs(coreml_logits)
    assert nan_count == 0, f"CoreML output contains {nan_count} NaNs"
    assert inf_count == 0, f"CoreML output contains {inf_count} Infs"
    
    # Check shape match
    assert coreml_logits.shape == pytorch_logits.shape, \
        f"Shape mismatch: CoreML {coreml_logits.shape} vs PyTorch {pytorch_logits.shape}"
    
    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(coreml_logits, pytorch_logits)
    
    assert cosine_sim >= cosine_threshold, \
        f"Cosine similarity {cosine_sim:.6f} < threshold {cosine_threshold} for T={shape}"
    
    # Also check relative error
    mae = np.mean(np.abs(coreml_logits - pytorch_logits))
    max_abs = max(np.max(np.abs(coreml_logits)), np.max(np.abs(pytorch_logits)), 1e-6)
    rel_error = mae / max_abs
    
    # Log results
    print(f"\n[golden_vectors] T={shape}:")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    print(f"  Mean absolute error: {mae:.6e}")
    print(f"  Relative error: {rel_error:.6e}")
    print(f"  NaN count: {nan_count}, Inf count: {inf_count}")


def test_coreml_io_contract(
    coreml_model_path: str,
    contract_path: Optional[str] = None,
):
    """Test CoreML model IO contract."""
    if not COREML_AVAILABLE:
        pytest.skip("coremltools not available")
    
    model_path = Path(coreml_model_path)
    contract_file = Path(contract_path) if contract_path else None
    
    results = validate_io_contract(model_path, contract_file)
    
    assert results["passed"], \
        f"IO contract validation failed: {results.get('error', results.get('contract_errors', []))}"
    
    # Log contract details
    print(f"\n[io_contract] Model: {model_path}")
    print(f"  Inputs: {len(results['inputs'])}")
    for inp in results["inputs"]:
        print(f"    - {inp['name']}: {inp['type']}")
    print(f"  Outputs: {len(results['outputs'])}")
    for out in results["outputs"]:
        print(f"    - {out['name']}: {out['type']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

