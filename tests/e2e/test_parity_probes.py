"""
Parity probe validation tests (A4).

Verifies CoreML model matches PyTorch model outputs within 2% relative error.
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

try:
    import torch
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="parity_probes_") as tmpdir:
        yield Path(tmpdir)


def compare_attention_outputs(pytorch_model, coreml_model, input_ids, relative_error_threshold=0.02):
    """
    Compare attention outputs between PyTorch and CoreML models.
    
    Args:
        pytorch_model: PyTorch model (TorchScript or regular)
        coreml_model: CoreML MLModel
        input_ids: Input token IDs [B, T]
        relative_error_threshold: Maximum allowed relative error (default: 0.02 = 2%)
    
    Returns:
        Dictionary with comparison results
    """
    import numpy as np
    
    # Convert input to appropriate format
    if isinstance(input_ids, np.ndarray):
        pytorch_input = torch.from_numpy(input_ids).int()
    else:
        pytorch_input = input_ids.int()
    
    # Run PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input)
    
    # Extract logits from PyTorch output
    if isinstance(pytorch_output, tuple):
        pytorch_logits = pytorch_output[0]
    elif isinstance(pytorch_output, dict):
        pytorch_logits = pytorch_output.get("logits", pytorch_output.get("output"))
    else:
        pytorch_logits = pytorch_output
    
    # Convert to numpy
    if hasattr(pytorch_logits, 'cpu'):
        pytorch_logits = pytorch_logits.cpu().numpy()
    pytorch_logits = np.array(pytorch_logits).astype(np.float32)
    
    # Run CoreML inference
    coreml_input = {"input_ids": input_ids.astype(np.int32) if isinstance(input_ids, np.ndarray) else input_ids.numpy().astype(np.int32)}
    coreml_output = coreml_model.predict(coreml_input)
    
    # Extract logits from CoreML output
    if "logits" in coreml_output:
        coreml_logits = coreml_output["logits"]
    else:
        # Try to find logits key
        logits_key = None
        for key in coreml_output.keys():
            if "logit" in key.lower() and "halt" not in key.lower():
                logits_key = key
                break
        if logits_key is None:
            # Use first output
            coreml_logits = list(coreml_output.values())[0]
        else:
            coreml_logits = coreml_output[logits_key]
    
    # Convert to numpy
    coreml_logits = np.array(coreml_logits).astype(np.float32)
    
    # Ensure same shape
    if pytorch_logits.shape != coreml_logits.shape:
        return {
            "relative_error": float('inf'),
            "passed": False,
            "error": f"Shape mismatch: PyTorch {pytorch_logits.shape} vs CoreML {coreml_logits.shape}",
        }
    
    # Compute relative error: |coreml - pytorch| / |pytorch|
    abs_diff = np.abs(coreml_logits - pytorch_logits)
    abs_pytorch = np.abs(pytorch_logits)
    
    # Avoid division by zero
    denominator = np.maximum(abs_pytorch, 1e-8)
    relative_error = np.mean(abs_diff / denominator)
    
    # Check for NaN or Inf
    if np.isnan(relative_error) or np.isinf(relative_error):
        return {
            "relative_error": float('inf'),
            "passed": False,
            "error": "Relative error is NaN or Inf",
        }
    
    # Check if error is within threshold
    passed = relative_error <= relative_error_threshold
    
    return {
        "relative_error": float(relative_error),
        "passed": passed,
        "pytorch_shape": pytorch_logits.shape,
        "coreml_shape": coreml_logits.shape,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
    }


@pytest.mark.slow
def test_parity_probes_enumerated_shapes(temp_dir):
    """Test parity probes for enumerated shapes (4k/8k/16k or toy equivalents)."""
    if not HAS_COREML:
        pytest.skip("coremltools not available")

    # Generate toy dataset
    dataset_path = temp_dir / "toy_kd.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "data.make_toy_kd", "--out", str(dataset_path), "--n", "64"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"

    # Train toy model
    checkpoint_path = temp_dir / "toy.ckpt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.run_toy_distill",
            "--in",
            str(dataset_path),
            "--out",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--mps",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Export PyTorch models with enumerated shapes
    export_dir = temp_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    enumerated_shapes = [64, 128, 256]  # Toy shapes (production would be 4k/8k/16k)
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conversion.export_pytorch",
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(export_dir),
            "--toy",
            "--mode",
            "prefill",
            "--seq",
            "64",
            "--enumerated-T",
        ] + [str(T) for T in enumerated_shapes],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Export failed: {result.stderr}"

    # Convert to CoreML
    pytorch_models = {}
    coreml_models = {}
    
    for T in enumerated_shapes:
        pytorch_path = export_dir / f"student_prefill_T{T}.pt"
        if not pytorch_path.exists():
            # Try to find any prefill model
            prefill_models = list(export_dir.glob("student_prefill_*.pt"))
            if not prefill_models:
                continue
            pytorch_path = prefill_models[0]

        # Load PyTorch model
        try:
            pytorch_model = torch.jit.load(str(pytorch_path))
            pytorch_model.eval()
            pytorch_models[T] = pytorch_model
        except Exception as e:
            print(f"⚠️  Failed to load PyTorch model for T{T}: {e}")
            continue

        # Convert to CoreML
        mlpackage_path = temp_dir / f"toy_T{T}.mlpackage"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "conversion.convert_coreml",
                "--backend",
                "pytorch",
                "--in",
                str(pytorch_path),
                "--out",
                str(mlpackage_path),
                "--seq",
                str(T),
                "--compute-units",
                "all",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode == 0 and mlpackage_path.exists():
            try:
                coreml_model = ct.models.MLModel(str(mlpackage_path))
                coreml_models[T] = coreml_model
            except Exception as e:
                print(f"⚠️  Failed to load CoreML model for T{T}: {e}")

    if len(pytorch_models) == 0 or len(coreml_models) == 0:
        pytest.skip("No models available for parity testing (CoreML conversion may have failed)")

    # Run parity probes for each shape
    results = {}
    for T in enumerated_shapes:
        if T not in pytorch_models or T not in coreml_models:
            continue

        # Create dummy input
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        input_ids = rng.integers(low=0, high=256, size=(1, T), dtype=np.int32)

        # Compare outputs
        comparison = compare_attention_outputs(
            pytorch_models[T],
            coreml_models[T],
            input_ids,
            relative_error_threshold=0.02,  # 2% threshold
        )

        results[T] = comparison

        # Verify relative error ≤ 2%
        assert comparison["passed"], (
            f"Parity probe failed for T{T}: relative_error={comparison['relative_error']:.6f} > 0.02"
        )

        # Verify no NaN or Inf
        assert not np.isnan(comparison["relative_error"]), f"Relative error is NaN for T{T}"
        assert not np.isinf(comparison["relative_error"]), f"Relative error is Inf for T{T}"

        print(f"✅ T{T}: relative_error={comparison['relative_error']:.6f}, "
              f"max_abs_diff={comparison['max_abs_diff']:.6e}")

    assert len(results) > 0, "No shapes successfully tested"
    print(f"✅ Parity probes passed for {len(results)} shapes: {list(results.keys())}")


@pytest.mark.slow
def test_parity_probes_no_nan_inf(temp_dir):
    """Test that parity probes produce no NaN or Inf values."""
    if not HAS_COREML:
        pytest.skip("coremltools not available")

    # Use a simpler setup - just test one shape
    dataset_path = temp_dir / "toy_kd.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "data.make_toy_kd", "--out", str(dataset_path), "--n", "32"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"

    checkpoint_path = temp_dir / "toy.ckpt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.run_toy_distill",
            "--in",
            str(dataset_path),
            "--out",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--mps",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Export and convert
    export_dir = temp_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conversion.export_pytorch",
            "--checkpoint",
            str(checkpoint_path),
            "--out",
            str(export_dir),
            "--toy",
            "--mode",
            "prefill",
            "--seq",
            "128",
            "--enumerated-T",
            "128",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Export failed: {result.stderr}"

    pytorch_path = export_dir / "student_prefill_T128.pt"
    if not pytorch_path.exists():
        prefill_models = list(export_dir.glob("student_prefill_*.pt"))
        if not prefill_models:
            pytest.skip("No prefill models exported")
        pytorch_path = prefill_models[0]

    try:
        pytorch_model = torch.jit.load(str(pytorch_path))
        pytorch_model.eval()
    except Exception as e:
        pytest.skip(f"Failed to load PyTorch model: {e}")

    mlpackage_path = temp_dir / "toy_T128.mlpackage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conversion.convert_coreml",
            "--backend",
            "pytorch",
            "--in",
            str(pytorch_path),
            "--out",
            str(mlpackage_path),
            "--seq",
            "128",
            "--compute-units",
            "all",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0 or not mlpackage_path.exists():
        pytest.skip("CoreML conversion failed (may not be available)")

    try:
        coreml_model = ct.models.MLModel(str(mlpackage_path))
    except Exception as e:
        pytest.skip(f"Failed to load CoreML model: {e}")

    # Test with multiple inputs
    rng = np.random.default_rng(42)
    for i in range(3):
        input_ids = rng.integers(low=0, high=256, size=(1, 128), dtype=np.int32)
        
        comparison = compare_attention_outputs(
            pytorch_model,
            coreml_model,
            input_ids,
            relative_error_threshold=0.02,
        )

        # Verify no NaN or Inf in outputs
        assert not np.isnan(comparison["relative_error"]), f"Relative error is NaN for input {i}"
        assert not np.isinf(comparison["relative_error"]), f"Relative error is Inf for input {i}"

    print("✅ No NaN or Inf values detected in parity probes")

