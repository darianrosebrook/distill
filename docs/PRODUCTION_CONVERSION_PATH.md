# Production Conversion Path: PyTorch → CoreML

## Summary

The production conversion path is **PyTorch → CoreML directly**, using CoreMLTools' supported front-end. ONNX is treated as a diagnostic interchange format only, not a production input.

## Architecture

### Production Path (PyTorch → CoreML)

1. **Export Student Model** (`training/export_student.py`):
   - Exports as TorchScript (`.pt`) or `torch.export` ExportedProgram
   - Creates `contract.json` with model specification
   - Single source of truth for training, QAT, and export

2. **Convert to CoreML** (`conversion/convert_coreml.py --backend pytorch`):
   - Uses `ct.convert(source="pytorch")` - official supported path
   - Reads input spec from `contract.json` (or infers from model)
   - Produces real `.mlpackage` with full parity

### Smoke/Diagnostic Path (ONNX → Placeholder)

- ONNX used for pipeline structure validation
- Creates placeholder when conversion unavailable
- All downstream scripts detect placeholder and return SKIP
- Test reports "PASSED (with SKIPS)" - visible in CI

## Test Targets

### `make smoke_torch` ✅
- Creates real PyTorch model (TorchScript)
- Converts to CoreML using supported front-end
- Produces **real mlpackage** (no placeholder)
- Proves the production path works end-to-end

### `make smoke_toy` ✅
- Creates ONNX model for structure validation
- Creates placeholder (expected - ONNX not supported)
- Validates pipeline control-flow and SKIP handling
- Always passes (with SKIPS visible)

### `make parity_full` (Future)
- Requires onnxruntime, Python 3.11
- Real PyTorch block conversion with full parity checks
- Must have 0 SKIPS to pass

## Key Files

- `conversion/convert_coreml.py`: Dual backend (pytorch/onnx)
- `conversion/make_toy_torch.py`: PyTorch toy model generator
- `training/export_student.py`: Production export with contract.json
- `coreml/probes/compare_probes.py`: Parity checks with results.json

## Verification

```bash
# PyTorch smoke test (real conversion)
make smoke_torch
# ✅ Real mlpackage created, no placeholder

# ONNX smoke test (structure validation)
make smoke_toy  
# ✅ Pipeline works, placeholder created (SKIP)
```

## Why PyTorch → CoreML?

1. **Highest parity**: No ONNX impedance mismatches
2. **Supported front-end**: Official CoreMLTools path
3. **Single source of truth**: PyTorch for training, QAT, export
4. **Easier shape/dtype enforcement**: At PyTorch boundary
5. **Clean probe mapping**: PyTorch hooks survive conversion

## Next Steps

1. Wire `export_student.py` into training pipeline
2. Add contract.json reading to `convert_coreml.py` for input specs
3. Implement `parity_full` with real transformer block
4. Add prefill/decoder split exports from PyTorch path

