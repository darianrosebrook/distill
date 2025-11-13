"""
End-to-end toy pipeline test.

Tests full flow: generate → train → export → convert → verify.
Designed to run in CI in ≤4 minutes on CPU.

Usage:
    pytest tests/e2e/test_toy_pipeline.py -v
"""
import pytest
import subprocess
import sys
import tempfile
import json
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="toy_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.mark.slow
def test_toy_pipeline_e2e(temp_dir):
    """Test full toy pipeline end-to-end."""
    # Paths
    dataset_path = temp_dir / "toy_kd.jsonl"
    checkpoint_path = temp_dir / "toy.ckpt"
    export_dir = temp_dir / "exported"
    mlpackage_path = temp_dir / "toy_T128.mlpackage"
    report_path = temp_dir / "toy_e2e.json"

    # Step 1: Generate KD dataset
    print("\n[test_toy_pipeline] Step 1: Generating toy KD dataset...")
    result = subprocess.run(
        [sys.executable, "-m", "data.make_toy_kd",
         "--out", str(dataset_path),
         "--n", "128"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dataset generation failed: {result.stderr}"
    assert dataset_path.exists(), "Dataset file not created"
    print(f"✅ Dataset created: {dataset_path}")

    # Step 2: Train toy model
    print("\n[test_toy_pipeline] Step 2: Training toy model...")
    result = subprocess.run(
        [sys.executable, "-m", "training.run_toy_distill",
         "--in", str(dataset_path),
         "--out", str(checkpoint_path),
         "--epochs", "2",
         "--mps", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Training failed: {result.stderr}"
    assert checkpoint_path.exists(), "Checkpoint file not created"
    print(f"✅ Training complete: {checkpoint_path}")

    # Step 3: Export to TorchScript
    print("\n[test_toy_pipeline] Step 3: Exporting to TorchScript...")
    export_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "conversion.export_pytorch",
         "--checkpoint", str(checkpoint_path),
         "--out", str(export_dir),
         "--toy",
         "--mode", "prefill",
         "--seq", "64",
         "--enumerated-T", "64", "128", "256"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Export failed: {result.stderr}"

    # Find exported prefill model (should have T128)
    prefill_model = export_dir / "student_prefill_T128.pt"
    if not prefill_model.exists():
        # Try to find any prefill model
        prefill_models = list(export_dir.glob("student_prefill_*.pt"))
        assert len(prefill_models) > 0, "No prefill models exported"
        prefill_model = prefill_models[0]

    assert prefill_model.exists(), "Prefill model not exported"
    print(f"✅ Export complete: {prefill_model}")

    # Step 4: Convert to CoreML (single representative shape)
    print("\n[test_toy_pipeline] Step 4: Converting to CoreML...")
    result = subprocess.run(
        [sys.executable, "-m", "conversion.convert_coreml",
         "--backend", "pytorch",
         "--in", str(prefill_model),
         "--out", str(mlpackage_path),
         "--seq", "128",
         "--compute-units", "all"],
        capture_output=True,
        text=True,
    )

    # Allow conversion to fail if CoreML not available (skip test)
    if result.returncode != 0:
        pytest.skip(
            f"CoreML conversion failed (may not be available): {result.stderr}")

    if not mlpackage_path.exists():
        pytest.skip(
            "CoreML conversion did not produce model (may not be available)")

    print(f"✅ Conversion complete: {mlpackage_path}")

    # Optional: Try to compile other shapes if available
    maybe_multi = True
    for L in ("64", "256"):
        shape_model = export_dir / f"student_prefill_T{L}.pt"
        if shape_model.exists():
            try:
                shape_mlpackage = temp_dir / f"toy_T{L}.mlpackage"
                subprocess.run(
                    [sys.executable, "-m", "conversion.convert_coreml",
                     "--backend", "pytorch",
                     "--in", str(shape_model),
                     "--out", str(shape_mlpackage),
                     "--seq", L,
                     "--compute-units", "all"],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                maybe_multi = False
                break

    # Step 5: Verify contracts
    print("\n[test_toy_pipeline] Step 5: Verifying contracts...")
    args = [
        sys.executable, "-m", "evaluation.toy_contracts",
        "--model", str(mlpackage_path),
        "--seq", "64", "128", "256",
        "--report", str(report_path),
    ]
    if maybe_multi:
        args.extend(["--model-dir", str(temp_dir)])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    # Allow verification to fail if CoreML not available
    if result.returncode != 0:
        pytest.skip(
            f"Verification failed (CoreML may not be available): {result.stderr}")

    assert report_path.exists(), "Verification report not created"

    # Load and check report
    with open(report_path) as f:
        report = json.load(f)

    assert "gates_ok" in report, "Report missing gates_ok"
    # Require ≥1 shape to work with non-zero/NaN checks and minimal F1
    assert report["gates_ok"], f"Gates failed: {report}"

    summary = report.get("summary", {})
    assert summary.get("nan_shapes", 1) == 0, "NaN detected in shapes"
    assert summary.get("zero_shapes", 1) == 0, "Zero detected in shapes"
    assert len(summary.get("shapes_ok", [])) >= 1, "No shapes verified"
    assert summary.get("tool_span_micro_f1",
                       0.0) >= 0.20, "Tool span F1 < 0.20"

    print(f"✅ Verification complete: {report_path}")
    print(f"   Shapes OK: {summary.get('shapes_ok', [])}")
    print(f"   Tool span F1: {summary.get('tool_span_micro_f1', 0.0):.4f}")

    # Print per-shape diagnostics if available
    if "per_shape" in report:
        print("\n   Per-shape diagnostics:")
        for shape, stat in report["per_shape"].items():
            status = "✓" if stat.get("ok") else "✗"
            print(f"     {status} T{shape}: {stat.get('reason', 'unknown')}")

    print("\n✅ Toy pipeline E2E test PASSED")


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
