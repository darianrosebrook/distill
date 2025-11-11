# Smoke Test Infrastructure Improvements

## Summary

This document describes the improvements made to the smoke test infrastructure based on feedback to ensure clean "toy → production" handoff and eliminate fragile assumptions.

## Key Changes

### 1. Python Version Discipline

**Problem**: Python 3.14 lacks wheels for `onnxruntime`, forcing placeholder logic.

**Solution**:
- Added `infra/version_gate.py` to enforce Python 3.10/3.11
- Version check runs before smoke tests via `make check-versions`
- Clear error messages guide users to correct Python version

### 2. Dependency Management

**Problem**: Unclear dependency requirements, pip version churn.

**Solution**:
- `requirements-core.txt`: Core dependencies (coremltools, onnx, numpy) with pinned versions
- `requirements-ort.txt`: Optional onnxruntime (platform-specific)
- `Makefile` targets:
  - `deps-core`: Install core dependencies
  - `deps-ort`: Auto-detect platform and install appropriate onnxruntime

### 3. ONNX → CoreML Converter

**Problem**: Using unsupported `source="onnx"` API, fallback to non-existent `coremltools.converters.onnx`.

**Solution**:
- Rewrote `conversion/convert_coreml.py` to use public MIL converter API:
  - Primary: `ct.convert(onnx_model, ...)` with auto-detection
  - Fallback: `ct.converters.mil.convert(onnx_model, ...)` if auto-detection fails
- Proper CLI with `--backend`, `--in`, `--out`, `--allow-placeholder` flags
- Clear error messages with actionable suggestions

### 4. Placeholder Semantics

**Problem**: Placeholder models reported as "PASS" instead of "SKIP".

**Solution**:
- Placeholder models now create `.placeholder` marker file
- All downstream scripts check for `.placeholder` and return SKIP status
- Logs clearly indicate "SKIP parity – placeholder model detected"
- CI can detect SKIP vs PASS separately

### 5. ANE Remote Proxy Warnings

**Problem**: Verbose "Failed to load _ML*RemoteProxy" warnings cluttering output.

**Solution**:
- Set `MLTOOLS_VERBOSE=0` by default
- Suppress warnings via `warnings.filterwarnings()`
- Optional `--ane-plan` flag for verbose logging in dev sessions

### 6. Test Flow Split

**Problem**: Single test path mixing smoke and parity requirements.

**Solution**:
- `smoke_toy`: Never requires onnxruntime; uses placeholder if conversion unavailable
- `parity_full`: Requires onnxruntime; fails loud if conversion unavailable
- Clear separation allows CI to gate appropriately

### 7. Version Gates

**Problem**: No upfront validation of environment.

**Solution**:
- `infra/version_gate.py` checks:
  - Python version (3.10 or 3.11)
  - macOS version (warns if < 13)
  - coremltools version (>= 9.0)
  - onnxruntime availability (optional)
- Fail-fast with actionable remediation messages

## Usage

### Smoke Test (No ORT Required)

```bash
make smoke_toy
```

This will:
1. Check Python version
2. Build toy ONNX model
3. Sanitize ONNX
4. Convert to CoreML (creates placeholder if conversion unavailable)
5. Run ANE checks (SKIP if placeholder)
6. Run parity probes (SKIP if placeholder)

### Full Parity Test (Requires ORT)

```bash
make parity_full
```

This will:
1. Check Python version
2. Install onnxruntime (platform-specific)
3. Build toy ONNX model
4. Sanitize ONNX
5. Convert to CoreML (fails loud if conversion unavailable)
6. Run ANE checks
7. Run full parity comparison

### Manual Converter Usage

```bash
# Production conversion (fails loud)
python -m conversion.convert_coreml \
  --backend onnx \
  --in model.onnx \
  --out model.mlpackage

# Smoke test (creates placeholder on failure)
python -m conversion.convert_coreml \
  --backend onnx \
  --in model.onnx \
  --out model.mlpackage \
  --allow-placeholder
```

## Files Changed

- `infra/version_gate.py` (new): Version checking module
- `requirements-core.txt` (new): Core dependencies
- `requirements-ort.txt` (new): Optional onnxruntime
- `conversion/convert_coreml.py`: Complete rewrite with proper MIL API
- `coreml/probes/compare_probes.py`: Check `.placeholder` marker
- `coreml/ane_checks.py`: Check `.placeholder` marker
- `Makefile`: Split smoke vs parity targets, version gates
- `.gitignore`: Add `.placeholder` markers

## Next Steps

1. **Production Conversion**: When ready, implement proper ONNX→MIL conversion pipeline
2. **Pre-softmax Parity**: Add NumPy-based parity checks for early taps (without ORT)
3. **CI Integration**: Wire `smoke_toy` (always green) and `parity_full` (gated) into CI

## Notes

- Placeholder models are marked with `.placeholder` file and `CONVERSION_NOTE.txt`
- All scripts gracefully handle placeholders and return SKIP status
- Version gates prevent running on unsupported Python versions
- Clear separation between smoke (always works) and parity (requires full stack)

