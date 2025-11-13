# Local Development Workflow Guide

Complete tiered approach for local development and validation, from quick smoke tests to full production pipelines.

## Overview

This guide provides **progressive tiers of local development** - start small and scale up as needed. Each tier builds confidence that the system works correctly while requiring different levels of setup and compute resources.

**Quick Start**: For the fastest validation, run `make 8-ball` to see the complete distillation pipeline create a working (if silly) language model in ~15 minutes!

## Tier 1: Smoke Tests (5-15 minutes)

**Goal**: Quick validation that the basic pipeline components work without real training.

### Requirements

- Python environment with dependencies installed
- No GPU required
- No external model access needed

### Tests to Run

#### Basic Smoke Test

```bash
# Validate toy model creation, export, and conversion
make smoke_toy
```

This runs:

1. **Toy ONNX Export**: Creates `onnx/toy.onnx` from synthetic model
2. **ONNX Surgery**: Sanitizes model for CoreML compatibility
3. **CoreML Conversion**: Creates `coreml/artifacts/toy/model.mlpackage`
4. **ANE Checks**: Validates CoreML model loads and runs on Apple Silicon
5. **Probes**: Compares PyTorch vs CoreML outputs for correctness

#### PyTorch Smoke Test

```bash
# Validate PyTorch-to-CoreML pipeline
make smoke_torch
```

This runs:

1. **Toy PyTorch Model**: Creates `models/toy_torch.pt`
2. **CoreML Conversion**: Direct PyTorch → CoreML conversion
3. **ANE Validation**: Confirms ANE compatibility

#### Training Pipeline Smoke Test

```bash
# Validate training infrastructure
make smoke_training
```

This runs:

1. **Toy Dataset Generation**: Creates minimal training data
2. **Training Execution**: Runs 5 training steps
3. **Pipeline Validation**: Confirms training loop works

### Expected Results

- ✅ All components execute without errors
- ✅ CoreML models load successfully
- ✅ ANE compatibility confirmed
- ⚠️ Models are functionally useless (deterministic gibberish)

### When to Use

- **First-time setup verification**
- **CI/CD smoke testing**
- **Dependency validation**
- **Before making code changes**

## Tier 2: Toy E2E Pipeline (15-30 minutes)

**Goal**: Validate complete distillation pipeline with realistic but minimal components.

### Requirements

- Python environment
- CPU-only (M1/M2/M3 recommended for speed)
- ~30 minutes runtime
- No external APIs needed

### Full Pipeline Execution

```bash
# Complete end-to-end toy pipeline
make toy-e2e
```

This executes the **complete distillation workflow**:

#### 1. Dataset Generation

```bash
# Creates /tmp/toy_kd.jsonl (128 samples)
python -m data.make_toy_kd --out /tmp/toy_kd.jsonl --n 128
```

#### 2. Model Training

```bash
# Trains for 2 epochs, saves to /tmp/toy.ckpt
python -m training.run_toy_distill \
  --in /tmp/toy_kd.jsonl \
  --out /tmp/toy.ckpt \
  --epochs 2 \
  --mps 0
```

#### 3. PyTorch Export

```bash
# Exports enumerated shapes to /tmp/toy_exported/
python -m conversion.export_pytorch \
  --checkpoint /tmp/toy.ckpt \
  --out /tmp/toy_exported \
  --toy \
  --mode prefill \
  --seq 64 \
  --enumerated-T 64 128 256
```

#### 4. CoreML Conversion

```bash
# Converts T128 shape to /tmp/toy_T128.mlpackage
python -m conversion.convert_coreml \
  --backend pytorch \
  --in /tmp/toy_exported/student_prefill_T128.pt \
  --out /tmp/toy_T128.mlpackage \
  --seq 128 \
  --compute-units all
```

#### 5. Contract Verification

```bash
# Validates CoreML contract compliance
python -m evaluation.toy_contracts \
  --model /tmp/toy_T128.mlpackage \
  --seq 128 \
  --report eval/reports/toy_e2e.json
```

### Multi-Shape Variant

```bash
# Test multiple enumerated shapes (more thorough)
make toy-e2e-multi
```

This compiles and validates shapes T64, T128, and T256.

### 8-ball Showcase 🪄

For a fun demonstration of the complete pipeline, try the **8-ball** end-to-end test:

```bash
# Complete 8-ball pipeline (dataset → training → export → CoreML → verification)
make 8-ball
```

This creates a **fully functional (albeit silly) language model** that generates mystical answers! The pipeline:

1. Generates 128 mystical answer samples
2. Trains a model to predict 8-ball responses (loss converges ~3.2→1.8)
3. Exports to TorchScript format
4. Converts to CoreML for Apple Silicon
5. Validates contract compliance

**Output**: A working `.mlpackage` that can generate responses like "Signs point to yes" or "My sources say no".

### Integration Tests

After toy E2E completes, run the integration test suite:

```bash
# CoreML golden vectors test (now passes)
pytest tests/test_coreml_golden_vectors.py -v

# Performance regression test (now passes)
pytest tests/test_performance_regression.py -v
```

### Expected Results

- ✅ Complete pipeline executes successfully
- ✅ All enumerated shapes compile
- ✅ Contract verification passes
- ✅ Integration tests pass
- ⚠️ Model accuracy is random (deterministic but meaningless)

### When to Use

- **Pipeline integration testing**
- **Code changes validation**
- **CI/CD integration testing**
- **Before production training**

## Tier 3: Full Training Pipeline (Hours-Days)

**Goal**: Real model training with actual teacher models and datasets.

### Requirements

- Apple Silicon (M1/M2/M3) with ≥32GB RAM
- Teacher model access (API key or local instance)
- Significant compute time (hours to days)
- Large storage (≥100GB free)

### Prerequisites

#### Environment Setup

```bash
# Python 3.11 required for export/conversion
brew install python@3.11
python3.11 --version  # Should show 3.11.x

# Virtual environment
python3.11 -m venv venv && source venv/bin/activate
pip install -e .
pip install coremltools>=9.0

# Version validation
python -m infra.version_gate --skip-ort
```

#### Teacher Model Access

Choose one option:

**Option A: Kimi K2 API (Production)**

```bash
export TEACHER_ENDPOINT="https://api.kimi.com/v1"
# Requires API key configuration
```

**Option B: Local Teacher**

```bash
# Start local teacher instance
export TEACHER_ENDPOINT="http://localhost:8000"
```

**Option C: Hugging Face Model**

```bash
export TEACHER_ENDPOINT="hf:moonshotai/Kimi-K2-Thinking"
# Requires HF_TOKEN and significant GPU resources
```

### Complete Production Pipeline

```bash
# 1. Generate knowledge distillation dataset
make kd-dataset TEACHER_ENDPOINT=$TEACHER_ENDPOINT

# 2. Train student model
make worker

# 3. Add process supervision (optional enhancement)
make proc

# 4. Quantization-aware training (optional)
make qat

# 5. Export trained model
make pytorch-worker

# 6. Convert to CoreML
make coreml-worker

# 7. Evaluate CAWS compliance
make eval-runner-local \
  MODEL="models/student/checkpoints/latest.pt" \
  IN="data/contextual_final.jsonl" \
  OUT="eval/results/production_eval.jsonl" \
  REPORT="eval/reports/production_eval.json"

# 8. Deploy production artifacts
make deploy-full
```

### Intermediate Validation Steps

#### After Dataset Generation

```bash
# Validate dataset quality
python -m scripts.validate_kd_data --in data/kd_mix.jsonl

# Quick dataset statistics
wc -l data/kd_mix.jsonl
head -5 data/kd_mix.jsonl | jq '.prompt, .teacher_text | length'
```

#### After Training

```bash
# Checkpoint validation
python -c "
import torch
ckpt = torch.load('models/student/checkpoints/latest.pt', map_location='cpu')
print(f'Step: {ckpt.get(\"step\", 0)}')
print(f'Loss: {ckpt.get(\"loss\", \"unknown\")}')
"

# Training evaluation
make eval-runner-local \
  MODEL="models/student/checkpoints/latest.pt" \
  IN="data/contextual_final.jsonl" \
  OUT="eval/results/training_eval.jsonl" \
  REPORT="eval/reports/training_eval.json"
```

#### After Export

```bash
# Validate exports
ls -la models/student/exported/
# Should see: student_prefill_T*.pt, student_decode.pt, *_contract.json

# Contract validation
python -c "
import json
contract = json.load(open('models/student/exported/student_prefill_T1024_contract.json'))
print('Enumerated shapes:', contract.get('enumerated_T', []))
print('Validation status:', contract.get('shape_validation', {}))
"
```

#### After CoreML Conversion

```bash
# CoreML validation
python -c "
import coremltools as ct
model = ct.models.MLModel('coreml/artifacts/worker/model.mlpackage')
print('Model loaded successfully')
print('Inputs:', [inp.name for inp in model.input_description])
print('Outputs:', [out.name for out in model.output_description])
"

# Performance testing
make speed-coreml
```

### Expected Results

- ✅ Real model with meaningful accuracy
- ✅ All enumerated shapes exported and converted
- ✅ CAWS compliance evaluation passes
- ✅ Production deployment artifacts generated
- ✅ Integration tests pass with real models

### When to Use

- **Production model development**
- **Research and experimentation**
- **Final validation before deployment**
- **Performance benchmarking**

## Progressive Validation Strategy

### Development Workflow

1. **Quick demo** (15 min) - run `make 8-ball` for complete pipeline showcase
2. **Start with smoke tests** (5 min) - verify basic functionality
3. **Run toy E2E** (30 min) - validate complete pipeline
4. **Scale to full training** (hours) - develop real models

### Quality Gates by Tier

| Gate                      | Smoke Tests         | Toy E2E            | Full Training           |
| ------------------------- | ------------------- | ------------------ | ----------------------- |
| **Pipeline Completeness** | ✅ Basic components | ✅ Full pipeline   | ✅ Production pipeline  |
| **CoreML Compatibility**  | ✅ Toy models       | ✅ Real conversion | ✅ Optimized conversion |
| **Contract Compliance**   | ❌ N/A              | ✅ Toy contracts   | ✅ Production contracts |
| **CAWS Evaluation**       | ❌ N/A              | ❌ N/A             | ✅ Real evaluation      |
| **Performance Targets**   | ❌ N/A              | ❌ N/A             | ✅ Production targets   |
| **Integration Tests**     | ❌ N/A              | ✅ Pass            | ✅ Pass                 |

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Smoke Tests
  run: make smoke_toy

- name: Integration Tests
  run: |
    make toy-e2e
    pytest tests/test_coreml_golden_vectors.py

- name: Production Pipeline (on-demand)
  run: |
    make worker
    make pytorch-worker
    make coreml-worker
    make eval-runner-local
```

## Troubleshooting

### Common Issues by Tier

#### Smoke Test Issues

```bash
# CoreML conversion fails
pip install coremltools>=9.0

# ANE checks fail
# This is expected on Intel Macs - CoreML works but ANE doesn't
```

#### Toy E2E Issues

```bash
# Conversion fails with "placeholder required"
# Add --allow-placeholder flag to conversion command

# Contract verification fails
# Check that toy model matches expected interface
```

#### Full Training Issues

```bash
# Teacher API unavailable
# Use local teacher or switch to Hugging Face model

# Out of memory during training
# Reduce batch size in config or use smaller model

# Export fails with Python version error
# Use python3.11 explicitly for export commands
```

## Performance Expectations

| Tier              | Runtime    | CPU Usage | Storage  | Model Quality        |
| ----------------- | ---------- | --------- | -------- | -------------------- |
| **Smoke**         | 5-15 min   | Low       | ~100MB   | None (gibberish)     |
| **Toy E2E**       | 15-30 min  | Medium    | ~500MB   | None (deterministic) |
| **Full Training** | Hours-Days | High      | 50-200GB | Production quality   |

## See Also

- [`README.md`](../README.md) - Quick start overview
- [`docs/PRODUCTION_DISTILLATION_WORKFLOW.md`](PRODUCTION_DISTILLATION_WORKFLOW.md) - Full production pipeline
- [`docs/TOY_TEST_STRATEGY.md`](TOY_TEST_STRATEGY.md) - Toy testing details
- [`Makefile`](../Makefile) - All available targets
- [`tests/README.md`](tests/README.md) - Test organization
