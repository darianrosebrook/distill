# ONNX Models

**Why it exists**: Enables cross-platform model deployment and validation, providing a common format that works across different operating systems and hardware.

**What's in it**: ONNX model files and graph optimization utilities that enable running the same models on Windows, Linux, macOS, and various ML frameworks.

**Key Features**:

- Cross-Platform: Single model format that runs everywhere (Windows/Linux/macOS)
- Graph Surgery: ONNX graph optimization and transformation utilities
- Validation Testing: Lightweight models for testing conversion pipelines
- Framework Agnostic: Works with ONNX Runtime, OpenVINO, TensorRT, etc.

## Overview

The ONNX directory provides cross-platform model artifacts that enable testing and validation of the conversion pipeline without platform-specific dependencies.

## Files

### Model Artifacts

- `toy.onnx` - Minimal toy transformer model for testing
- `toy.sanitized.onnx` - Optimized version with graph surgery applied

### Usage

Toy models are primarily used for:

- Pipeline validation without heavy model dependencies
- ONNX conversion testing
- Cross-platform compatibility verification

## Model Specifications

### Toy Model (`toy.onnx`)

```yaml
Architecture: Transformer
Parameters: ~50K
Layers: 1
Heads: 2
Vocab Size: 512
Max Sequence: 128
Precision: FP32
```

### Sanitized Model (`toy.sanitized.onnx`)

- Graph optimizations applied
- Constant folding
- Dead code elimination
- Shape inference completed

## Conversion Pipeline

ONNX models are created via the conversion pipeline:

1. **PyTorch Model** → `conversion/export_pytorch.py`
2. **ONNX Export** → `conversion/export_onnx.py`
3. **Graph Surgery** → `conversion/onnx_surgery.py`
4. **Validation** → Runtime testing

## Testing

ONNX models are validated using:

```bash
# Compare ONNX vs PyTorch outputs
python -m coreml.probes.compare_probes \
  --onnx onnx/toy.sanitized.onnx \
  --pt models/toy_block.pt \
  --seq 128

# Run inference tests
python -m onnxruntime_inference_test onnx/toy.onnx
```

## See Also

- [`conversion/export_onnx.py`](../conversion/export_onnx.py) - ONNX export utilities
- [`conversion/onnx_surgery.py`](../conversion/onnx_surgery.py) - Graph optimization
- [`coreml/probes/`](../coreml/probes/) - Model validation probes
- [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) - Deployment options
