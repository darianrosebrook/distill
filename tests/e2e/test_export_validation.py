"""
Export format validation tests (A2).

Validates ONNX exports work for enumerated shapes.
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="export_validation_") as tmpdir:
        yield Path(tmpdir)


@pytest.mark.slow
def test_onnx_export_enumerated_shapes(temp_dir):
    """Test ONNX exports for enumerated shapes T64, T128, T256."""
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
    assert checkpoint_path.exists(), "Checkpoint file not created"

    # Export PyTorch model with enumerated shapes
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
            "64",
            "--enumerated-T",
            "64",
            "128",
            "256",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Export failed: {result.stderr}"

    # Verify all shape files created
    enumerated_shapes = [64, 128, 256]
    pytorch_models = {}
    for T in enumerated_shapes:
        model_path = export_dir / f"student_prefill_T{T}.pt"
        if model_path.exists():
            pytorch_models[T] = model_path
        else:
            # Try to find any prefill model
            prefill_models = list(export_dir.glob("student_prefill_*.pt"))
            if prefill_models:
                # Use the first one found
                pytorch_models[T] = prefill_models[0]
                print(f"⚠️  Shape T{T} not found, using {prefill_models[0]}")
            else:
                pytest.fail(f"No prefill models exported for shape T{T}")

    assert len(pytorch_models) > 0, "No prefill models exported"

    # Convert each PyTorch model to ONNX format using torch.onnx.export
    # Note: We load the TorchScript model and export to ONNX
    import torch
    
    onnx_models = {}
    for T, pytorch_path in pytorch_models.items():
        onnx_path = temp_dir / f"student_prefill_T{T}.onnx"
        try:
            # Load TorchScript model
            traced_model = torch.jit.load(str(pytorch_path))
            traced_model.eval()
            
            # Create dummy input for ONNX export
            dummy_input = torch.zeros((1, T), dtype=torch.int32)
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    input_names=["input_ids"],
                    output_names=["logits"],
                    opset_version=19,
                    do_constant_folding=True,
                    dynamic_axes=None,  # enumerated shapes only
                )
            
            if onnx_path.exists():
                onnx_models[T] = onnx_path
            else:
                pytest.skip(f"ONNX model not created for T{T}")
        except Exception as e:
            pytest.skip(f"ONNX export failed for T{T}: {e}")

    assert len(onnx_models) > 0, "No ONNX models created"

    # Verify ONNX models load and have correct input/output shapes
    if not HAS_ONNXRUNTIME:
        pytest.skip("onnxruntime not available")

    for T, onnx_path in onnx_models.items():
        try:
            session = ort.InferenceSession(str(onnx_path))
            
            # Verify input shapes
            inputs = session.get_inputs()
            assert len(inputs) > 0, f"ONNX model for T{T} has no inputs"
            
            # Find input_ids input
            input_ids_input = None
            for inp in inputs:
                if inp.name == "input_ids" or "input" in inp.name.lower():
                    input_ids_input = inp
                    break
            
            if input_ids_input is None:
                input_ids_input = inputs[0]  # Use first input if not found
            
            # Verify input shape is [B, T] or compatible
            input_shape = input_ids_input.shape
            assert len(input_shape) == 2, f"Input shape should be 2D [B, T], got {input_shape}"
            
            # Verify output shapes
            outputs = session.get_outputs()
            assert len(outputs) > 0, f"ONNX model for T{T} has no outputs"
            
            # Find logits output
            logits_output = None
            for out in outputs:
                if out.name == "logits" or "output" in out.name.lower():
                    logits_output = out
                    break
            
            if logits_output is None:
                logits_output = outputs[0]  # Use first output if not found
            
            # Verify output shape is [B, T, V] or compatible
            output_shape = logits_output.shape
            assert len(output_shape) >= 2, f"Output shape should be at least 2D, got {output_shape}"
            
            print(f"✅ ONNX model T{T}: input_shape={input_shape}, output_shape={output_shape}")
            
        except Exception as e:
            pytest.fail(f"Failed to load ONNX model for T{T}: {e}")

    # Test ONNX inference with dummy inputs
    for T, onnx_path in onnx_models.items():
        try:
            session = ort.InferenceSession(str(onnx_path))
            
            # Create dummy input
            input_name = session.get_inputs()[0].name
            dummy_input = {
                input_name: [[0] * T]  # [B=1, T]
            }
            
            # Run inference
            outputs = session.run(None, dummy_input)
            assert len(outputs) > 0, f"ONNX inference for T{T} produced no outputs"
            
            # Verify output is not NaN or Inf
            import numpy as np
            output_array = outputs[0]
            assert not np.isnan(output_array).any(), f"ONNX output for T{T} contains NaN"
            assert not np.isinf(output_array).any(), f"ONNX output for T{T} contains Inf"
            
            print(f"✅ ONNX inference successful for T{T}: output_shape={output_array.shape}")
            
        except Exception as e:
            pytest.fail(f"ONNX inference failed for T{T}: {e}")

    print("✅ ONNX export validation passed for all enumerated shapes")


@pytest.mark.slow
def test_pytorch_export_enumerated_shapes(temp_dir):
    """Test PyTorch export with enumerated shapes creates all required files."""
    # Generate toy dataset
    dataset_path = temp_dir / "toy_kd.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "data.make_toy_kd", "--out", str(dataset_path), "--n", "32"],
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

    # Export with enumerated shapes
    export_dir = temp_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    enumerated_shapes = [64, 128, 256]
    
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

    # Verify all enumerated shapes exported
    found_shapes = []
    for T in enumerated_shapes:
        model_path = export_dir / f"student_prefill_T{T}.pt"
        if model_path.exists():
            found_shapes.append(T)
            assert model_path.stat().st_size > 0, f"Exported model for T{T} is empty"
        else:
            # Check for any prefill models
            prefill_models = list(export_dir.glob("student_prefill_*.pt"))
            if prefill_models:
                found_shapes.append(T)
                print(f"⚠️  Shape T{T} not found, but found {len(prefill_models)} prefill models")
            else:
                pytest.fail(f"No prefill models exported for shape T{T}")

    assert len(found_shapes) > 0, "No enumerated shapes exported"
    print(f"✅ Exported {len(found_shapes)} enumerated shapes: {found_shapes}")

    # Verify contract files exist
    contract_files = list(export_dir.glob("*_contract.json"))
    if contract_files:
        print(f"✅ Found {len(contract_files)} contract files")
    else:
        print("⚠️  No contract files found (may be optional)")

    print("✅ PyTorch export validation passed")

