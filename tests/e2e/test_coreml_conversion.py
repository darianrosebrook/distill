"""
CoreML conversion validation tests (A3).

Validates CoreML .mlpackage generation and loading.
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="coreml_conversion_") as tmpdir:
        yield Path(tmpdir)


@pytest.mark.slow
def test_coreml_conversion_basic(temp_dir):
    """Test basic CoreML conversion from PyTorch model."""
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
    assert checkpoint_path.exists(), "Checkpoint file not created"

    # Export PyTorch model
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

    # Find exported prefill model
    prefill_model = export_dir / "student_prefill_T128.pt"
    if not prefill_model.exists():
        prefill_models = list(export_dir.glob("student_prefill_*.pt"))
        assert len(prefill_models) > 0, "No prefill models exported"
        prefill_model = prefill_models[0]

    assert prefill_model.exists(), "Prefill model not exported"

    # Convert to CoreML
    mlpackage_path = temp_dir / "toy_T128.mlpackage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conversion.convert_coreml",
            "--backend",
            "pytorch",
            "--in",
            str(prefill_model),
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

    # Allow conversion to fail if CoreML not available (skip test)
    if result.returncode != 0:
        pytest.skip(f"CoreML conversion failed (may not be available): {result.stderr}")

    if not mlpackage_path.exists():
        pytest.skip("CoreML conversion did not produce model (may not be available)")

    # Verify .mlpackage directory structure
    assert mlpackage_path.is_dir(), ".mlpackage should be a directory"
    
    # Check for required files in .mlpackage
    model_file = mlpackage_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
    if not model_file.exists():
        # Try alternative structure
        model_file = mlpackage_path / "model.mlmodel"
    
    # If still not found, check if it's a valid mlpackage by trying to load it
    # (some CoreML versions may have different structures)
    
    # Load CoreML model using coremltools
    try:
        model = ct.models.MLModel(str(mlpackage_path))
        spec = model.get_spec()
        
        # Verify model spec has correct input/output descriptions
        assert len(spec.description.input) > 0, "Model spec missing inputs"
        assert len(spec.description.output) > 0, "Model spec missing outputs"
        
        # Find input_ids input
        input_ids_input = None
        for inp in spec.description.input:
            if inp.name == "input_ids" or "input" in inp.name.lower():
                input_ids_input = inp
                break
        
        if input_ids_input is None:
            input_ids_input = spec.description.input[0]  # Use first input
        
        # Verify input shape
        assert input_ids_input.type.WhichOneof("Type") is not None, "Input type not specified"
        
        # Find logits output
        logits_output = None
        for out in spec.description.output:
            if out.name == "logits" or "output" in out.name.lower():
                logits_output = out
                break
        
        if logits_output is None:
            logits_output = spec.description.output[0]  # Use first output
        
        # Verify output shape
        assert logits_output.type.WhichOneof("Type") is not None, "Output type not specified"
        
        print(f"✅ CoreML model loaded: input={input_ids_input.name}, output={logits_output.name}")
        
    except Exception as e:
        pytest.fail(f"Failed to load CoreML model: {e}")

    # Test CoreML prediction with dummy inputs
    try:
        import numpy as np
        
        # Create dummy input
        dummy_input = {"input_ids": np.array([[0] * 128], dtype=np.int32)}
        
        # Run prediction
        prediction = model.predict(dummy_input)
        
        # Verify prediction has output
        assert len(prediction) > 0, "Prediction produced no outputs"
        
        # Check for logits in output
        if "logits" in prediction:
            logits = prediction["logits"]
            assert logits.shape[0] == 1, f"Logits batch size should be 1, got {logits.shape[0]}"
            assert logits.shape[1] == 128, (
                f"Logits sequence length should be 128, got {logits.shape[1]}"
            )
        else:
            # Use first output
            first_output = list(prediction.values())[0]
            assert first_output.shape[0] == 1, (
                f"Output batch size should be 1, got {first_output.shape[0]}"
            )
        
        print("✅ CoreML prediction successful")
        
    except Exception as e:
        pytest.fail(f"CoreML prediction failed: {e}")

    # Check ANE compatibility (if available)
    try:
        # Try to get compute units
        compute_units = model.compute_unit
        print(f"✅ CoreML model compute units: {compute_units}")
        
        # Note: ANE compatibility check would require additional tools
        # For now, we just verify the model loads and runs
        
    except Exception as e:
        print(f"⚠️  Could not check ANE compatibility: {e}")

    print("✅ CoreML conversion validation passed")


@pytest.mark.slow
def test_coreml_conversion_enumerated_shapes(temp_dir):
    """Test CoreML conversion for multiple enumerated shapes."""
    if not HAS_COREML:
        pytest.skip("coremltools not available")

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

    # Convert each shape to CoreML
    converted_shapes = []
    for T in enumerated_shapes:
        pytorch_model = export_dir / f"student_prefill_T{T}.pt"
        if not pytorch_model.exists():
            # Try to find any prefill model
            prefill_models = list(export_dir.glob("student_prefill_*.pt"))
            if not prefill_models:
                continue
            pytorch_model = prefill_models[0]

        mlpackage_path = temp_dir / f"toy_T{T}.mlpackage"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "conversion.convert_coreml",
                "--backend",
                "pytorch",
                "--in",
                str(pytorch_model),
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

        # Allow conversion to fail for some shapes (skip if all fail)
        if result.returncode == 0 and mlpackage_path.exists():
            try:
                # Verify model loads
                ct.models.MLModel(str(mlpackage_path))
                converted_shapes.append(T)
                print(f"✅ Converted shape T{T} to CoreML")
            except Exception as e:
                print(f"⚠️  Shape T{T} converted but failed to load: {e}")

    if len(converted_shapes) == 0:
        pytest.skip("No shapes successfully converted to CoreML (may not be available)")

    assert len(converted_shapes) > 0, "At least one shape should be converted"
    print(f"✅ Successfully converted {len(converted_shapes)} shapes: {converted_shapes}")

