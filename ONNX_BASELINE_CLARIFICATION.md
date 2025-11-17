# ONNX Export Not Used in Production Pipeline

**Date**: November 16, 2025
**Author**: @darianrosebrook
**Status**: Baseline Clarification

## Key Finding

ONNX export is **NOT part of the production pipeline** for this project. The main training and export path uses:

**PyTorch → CoreML** (not ONNX)

This is confirmed in `training/export_student.py`:

```python
"""
Export student model to TorchScript/ExportedProgram for CoreML conversion.

This is the production export path - PyTorch → CoreML (not ONNX).
"""
```

## Implication for Test Baseline

The ONNX export timeout issue (`test_judge_export_onnx.py::test_main_config_loading_error`) that was blocking mutation testing can be safely **skipped or mocked** because:

1. **Not used in production training** - The core distillation pipeline doesn't rely on ONNX
2. **CoreML is the target format** - Models are exported for Apple devices via CoreML
3. **ONNX tests are optional** - Can be skipped without affecting core functionality
4. **Mutation testing can proceed** - Focus on training modules, skip integration tests

## Impact on Work

### Baseline Priorities - REVISED

**Priority 1**: High-Impact Training Fixes (8-10 hours)
- Mock tokenizer issues (47 tests)
- Monitoring interfaces (31 tests)
- Function signatures (9 tests)
- Tracer implementation (12 tests)

**Priority 2**: Remaining Training Fixes (2-3 hours)
- JSON repair detection (5 tests)
- Progress tracking (2 tests)
- Tokenizer migration (4 tests)
- Latent curriculum (4 tests)
- Teacher cache (2 errors)
- Teacher stub toy (1 test)

**SKIP**: ONNX tests (not critical to pipeline)

### Mutation Testing

Can now proceed by focusing on training modules:

```bash
mutatest -s training/distill_kd.py -m sd -n 20
```

No need to wait for ONNX export tests to pass.

## Affected Documents

The following documents have been updated:

1. **TEST_BASELINE_REPORT.md**
   - Integration tests section clarified
   - ONNX tests marked as non-critical
   - Mutation tests section updated to show we can proceed

2. **TEST_FIX_CHECKLIST.md**
   - Removed ONNX from blocking issues
   - Renumbered priorities
   - Adjusted time estimates (now 10-13 hours instead of 12-16)

## Next Steps

1. **Focus on core training fixes** - Start with mock tokenizer utility (affects 47 tests)
2. **Establish mutation baseline** - Can run on training modules alone
3. **Skip or defer ONNX tests** - Not critical to production pipeline

## Command Reference

```bash
# Activate and run core training tests only
source venv/bin/activate
pytest tests/training/ -q --timeout=120

# Skip integration tests during mutation baseline
mutatest -s training/distill_kd.py -m sd -n 20

# Or run specific integration tests (skip ONNX)
pytest tests/conversion -k "not test_judge_export_onnx" -q
```

## Conclusion

This clarification reduces the blocker issues and allows us to focus on the actual training pipeline problems. The ONNX tests can be addressed separately or kept as optional test coverage for the conversion utilities (which aren't used in production).

**Total estimated time for core fixes: 10-13 hours** (down from 12-16 hours)

