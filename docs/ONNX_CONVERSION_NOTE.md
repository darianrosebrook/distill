# ONNX → CoreML Conversion Note

## Current Status

CoreMLTools 9.0's `ct.convert()` function **does not support ONNX models directly**. It only supports:
- TensorFlow (1.x and 2.x)
- PyTorch (TorchScript and ExportedProgram)

## Smoke Test Behavior

The smoke test correctly handles this limitation:
1. Attempts conversion using `ct.convert()` (auto-detection)
2. Falls back to `ct.converters.mil.convert()` (not available for ONNX)
3. Creates a **placeholder** with `.placeholder` marker when conversion fails
4. All downstream scripts detect placeholder and return **SKIP** status
5. Test reports **"PASSED (with SKIPS)"** - SKIPS are visible in CI totals

This is the **expected behavior** for smoke tests - they validate the pipeline structure without requiring full conversion.

## Production Conversion Path

### ✅ Preferred (Ship-grade): PyTorch → CoreML

Since we own the student architecture, the production path is:

1. **Export PyTorch model** as TorchScript or `torch.export` ExportedProgram:
   ```python
   # In training/export_student.py
   model.eval()
   traced = torch.jit.trace(model, example_input)
   # OR
   exported = torch.export.export(model, (example_input,))
   ```

2. **Convert directly to CoreML**:
   ```python
   import coremltools as ct
   mlmodel = ct.convert(
       torch_model_or_exported_program,
       source="pytorch",
       compute_units=ct.ComputeUnit.ALL,
       minimum_deployment_target=ct.target.macOS13,
   )
   ```

**Why this is preferred:**
- Highest parity (no ONNX impedance mismatches)
- Supported front-end (official CoreMLTools path)
- Easier to enforce enumerated shapes and dtypes
- Single source of truth (PyTorch) for training, QAT, and export
- Probe harnesses map cleanly to PyTorch hooks

### ⚠️ Acceptable (Only if you don't control the model)

If you receive an ONNX model from an external source:
- Re-author the network in PyTorch (one-time port) rather than round-tripping ONNX

### ❌ Not Recommended

- **`onnx2pytorch`**: Semantic gaps (control-flow, opset, quant nodes)
- **`onnx2coreml`**: Stale, incomplete op coverage, unmaintained
- **Manual MIL for full models**: Too much bespoke work (fine for tiny fusions only)

## ONNX as Diagnostic Interchange

**ONNX is treated as a diagnostic interchange format** - useful for:
- Probe comparisons (PyTorch → ONNX → CoreML probes)
- Graph visualization and debugging
- Cross-framework validation

**ONNX is NOT the production input to CoreML** - use PyTorch directly.

## Smoke Test Targets

- **`make smoke_torch`**: Real mlpackage from PyTorch toy (proves supported front-end works)
- **`make smoke_toy`**: ONNX placeholder (proves pipeline control-flow and SKIP handling)
- **`make parity_full`**: Requires ORT, Python 3.11, real PyTorch block conversion

## Verification

Run smoke test:
```bash
make smoke_toy
```

Expected output:
- Version check passes ✅
- ONNX model created ✅
- ONNX sanitized ✅
- Placeholder created (SKIP) ✅
- ANE checks SKIP ✅
- Probes SKIP ✅
- Test **PASSED (with SKIPS)** ✅

This confirms the pipeline works correctly even when full conversion is unavailable.
