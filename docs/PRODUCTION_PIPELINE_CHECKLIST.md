# Production Pipeline Checklist

Pre-deployment verification checklist for production model pipeline (dataset → training → export → conversion → verification).

## Pre-Export Verification

### Environment Setup

- [ ] Python 3.11 installed and accessible (`python3.11 --version`)
- [ ] PyTorch 2.9.1+ installed (`python -c "import torch; print(torch.__version__)"`)
- [ ] coremltools installed (`python -c "import coremltools; print(coremltools.__version__)"`)
- [ ] Checkpoint file exists and is valid
- [ ] Checkpoint contains `model_arch` metadata

### Model Checkpoint Validation

- [ ] Checkpoint loads without errors
- [ ] Model architecture flags are present (`use_halt_head`, etc.)
- [ ] Model vocabulary size matches tokenizer
- [ ] Model can perform forward pass on sample input
- [ ] Checkpoint step number is recorded

### Shape Enumeration Pre-Check

Before full export, verify model handles expected shapes:

- [ ] T512 shape works (minimum production shape)
- [ ] T1024 shape works (primary production shape)
- [ ] T2048 shape works (if needed)
- [ ] T4096 shape works (if needed)

**Note**: Primary shape (T1024) must work. Secondary shapes may fail but should be logged.

## Export Process

### PyTorch Export

- [ ] Use Python 3.11: `python3.11 -m conversion.export_pytorch ...`
- [ ] Export completes without errors
- [ ] Contract file generated (`*_contract.json`)
- [ ] TorchScript files created (`*.pt`)
- [ ] Shape validation results recorded in contract

### Contract File Verification

- [ ] Contract file is valid JSON
- [ ] Contains `model_arch` flags
- [ ] Contains `outputs` specification
- [ ] Contains `shape_validation` results
- [ ] Primary shape (T1024) validation shows success

### Shape Validation Results

Check contract file for shape validation:

```bash
python -c "import json; print(json.load(open('contract.json'))['shape_validation'])"
```

Expected output:
- T1024: `{"status": "ok", ...}` (must succeed)
- Other shapes: May show `{"status": "error", ...}` (acceptable if primary works)

## CoreML Conversion

### Pre-Conversion Checks

- [ ] TorchScript model loads: `torch.jit.load('model.pt')`
- [ ] Contract file exists and is readable
- [ ] Output directory exists and is writable
- [ ] Sufficient disk space available

### Conversion Process

- [ ] Use Python 3.11: `python3.11 -m conversion.convert_coreml ...`
- [ ] Conversion completes (warnings are acceptable)
- [ ] `.mlpackage` file created
- [ ] Conversion logs reviewed for actual errors (not warnings)

### Post-Conversion Verification

- [ ] CoreML model file exists
- [ ] Model can be loaded: `coremltools.models.MLModel('model.mlpackage')`
- [ ] Output names match contract (`logits`, `halt_logits` if enabled)
- [ ] Model spec shows correct number of outputs

## Contract Verification

### Verification Process

- [ ] Run verification: `python -m conversion.verify_contract ...`
- [ ] Verification report generated
- [ ] Primary shape (T1024) verification passes
- [ ] Secondary shape failures are logged but don't block

### Verification Report Review

- [ ] `gates_ok` status checked
- [ ] T1024 shape shows `ok` status
- [ ] Error messages reviewed for non-primary shapes
- [ ] Any critical errors addressed

## Known Issues and Workarounds

### PyTorch Version Compatibility

**Issue**: PyTorch 2.9.1 shows warnings with coremltools

**Workaround**:
- Warnings are non-fatal - monitor for actual errors
- If conversion fails, consider: `pip install torch==2.8.0`
- Check conversion logs distinguish warnings from errors

### Shape Enumeration Failures

**Issue**: Some shapes (T64/T256 for toy, potentially T512/T4096 for production) may fail

**Workaround**:
- Primary shape must succeed (T128 for toy, T1024 for production)
- Secondary shape failures are logged but don't block export
- Check contract file to see which shapes validated
- If primary shape fails, investigate model architecture

### Python Version Requirements

**Issue**: Export/conversion require Python 3.10/3.11, training uses 3.13+

**Workaround**:
- Use Python 3.11 explicitly for export/conversion steps
- Install via Homebrew: `brew install python@3.11`
- Use `--toy` flag only for toy models (never for production)

### CoreML Conversion Errors

**Issue**: Conversion may fail silently or with unclear errors

**Workaround**:
1. Check conversion logs for detailed error messages
2. Verify PyTorch model loads: `torch.jit.load('model.pt')`
3. Check contract file exists and is valid JSON
4. Verify coremltools installation
5. Try with `--allow-placeholder` flag to see if non-critical

## Post-Deployment Verification

### Model Loading

- [ ] PyTorch model loads in production environment
- [ ] CoreML model loads on target device (if applicable)
- [ ] Model outputs match expected format
- [ ] Inference latency meets requirements

### Runtime Configuration

- [ ] Runtime config file generated
- [ ] Config matches model capabilities
- [ ] Halt head enabled if model supports it
- [ ] Latent mode configured if needed

### Integration Testing

- [ ] Model works with inference orchestrator
- [ ] Tool calling works (if applicable)
- [ ] Halt head outputs correct (if enabled)
- [ ] Latent reasoning works (if enabled)

## Emergency Procedures

### Export Fails

1. Check Python version: `python3.11 --version`
2. Verify checkpoint loads: `torch.load('checkpoint.pt')`
3. Check disk space: `df -h`
4. Review export logs for specific error
5. Try with `--toy` flag only if testing toy model

### Conversion Fails

1. Verify TorchScript model loads: `torch.jit.load('model.pt')`
2. Check contract file exists and is valid
3. Review conversion logs for actual errors
4. Try downgrading PyTorch: `pip install torch==2.8.0`
5. Verify coremltools installation

### Verification Fails

1. Check which shapes failed in contract file
2. Verify primary shape (T1024) succeeded
3. Review verification report for details
4. If primary shape failed, investigate model architecture
5. Secondary shape failures are acceptable if primary works

## Quality Gates

### Must Pass (Blockers)

- [ ] Python 3.11 available for export/conversion
- [ ] Checkpoint loads and validates
- [ ] Export completes successfully
- [ ] Contract file generated and valid
- [ ] Primary shape (T1024) validation succeeds
- [ ] CoreML conversion completes (warnings OK)
- [ ] CoreML model loads successfully

### Should Pass (Warnings)

- [ ] All shape enumerations succeed (T512, T1024, T2048, T4096)
- [ ] No PyTorch compatibility warnings
- [ ] Verification shows all gates OK
- [ ] Secondary shapes validate successfully

### Nice to Have (Optional)

- [ ] All shapes validate without errors
- [ ] Zero warnings in conversion logs
- [ ] Perfect verification report
- [ ] Optimal model size and performance

## Documentation

After successful deployment:

- [ ] Update model registry with checkpoint info
- [ ] Document any workarounds used
- [ ] Record shape validation results
- [ ] Note any compatibility issues encountered
- [ ] Update deployment manifest

## See Also

- [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md) - Full deployment guide
- [`conversion/export_pytorch.py`](../conversion/export_pytorch.py) - Export implementation
- [`conversion/convert_coreml.py`](../conversion/convert_coreml.py) - CoreML conversion
- [`conversion/verify_contract.py`](../conversion/verify_contract.py) - Contract verification

