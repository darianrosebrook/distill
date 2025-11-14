# Model Conversion

Model export and conversion utilities for PyTorch → ONNX → CoreML pipeline.

## Overview

This directory contains the production export path for converting trained models to deployment formats:

- **PyTorch Export** (Production): `export_pytorch.py` - Exports to TorchScript/ExportedProgram
- **CoreML Conversion** (Production): `convert_coreml.py` - Converts PyTorch models to CoreML mlpackage
- **ONNX Export** (Debug): `export_onnx.py` - Optional ONNX export for debugging

## Production Path

The recommended production path is:

```
PyTorch Model → export_pytorch.py → CoreML (via convert_coreml.py)
```

**Not recommended**: ONNX intermediate step (use only for debugging).

## Key Files

### `export_pytorch.py`
- Exports trained models to TorchScript/ExportedProgram format
- Supports prefill and decode modes
- Generates contract.json for input specifications
- Handles enumerated shapes (512/1024/2048/4096)

**Usage**:
```bash
python -m conversion.export_pytorch \
  --checkpoint models/student/checkpoints/latest.pt \
  --out models/student/exported/ \
  --mode both \
  --enumerated-T 512 1024 2048
```

### `convert_coreml.py`
- Converts PyTorch models to CoreML mlpackage
- Supports both TorchScript and ExportedProgram
- Reads contract.json for input specifications
- Handles enumerated shapes automatically

**Usage**:
```bash
python -m conversion.convert_coreml \
  --backend pytorch \
  --in models/student/exported/student_fp16.pt \
  --out coreml/artifacts/worker/model.mlpackage \
  --contract models/student/exported/contract.json
```

### `export_onnx.py`
- Optional ONNX export for debugging
- Not used in production path
- Useful for model inspection and debugging

### `judge_export_onnx.py` / `judge_export_coreml.py`
- Judge-specific export utilities
- Handles judge model architecture differences

## Shape Enumeration

Models are exported with enumerated shapes for ANE efficiency:
- Prefill: 512, 1024, 2048, 4096 (configurable)
- Decode: Single token with KV cache

Shape enumeration is specified in `shape_sets.json` or via `--enumerated-T` flag.

## Contract Generation

Each exported model includes a `contract.json` file specifying:
- Input names and shapes
- Output names and shapes
- Enumerated sequence lengths
- Data types

Contracts are used by CoreML conversion to properly specify inputs.

## Requirements

- **PyTorch**: ≥ 2.0 (for ExportedProgram support)
- **CoreMLTools**: ≥ 9.0
- **Python**: 3.10 or 3.11 (not 3.13+)

## See Also

- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Complete pipeline review
- [`coreml/README.md`](../coreml/README.md) - CoreML runtime documentation










