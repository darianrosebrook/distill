# Production Distillation Workflow Guide

Complete end-to-end workflow for distillation training → conversion → verification → deployment.

## Overview

The production distillation pipeline consists of 8 phases:

1. **Environment Setup** - Python, dependencies, hardware
2. **Dataset Generation** - Knowledge distillation dataset from teacher
3. **Training** - Student model training with process supervision
4. **Export** - PyTorch TorchScript export (production path)
5. **Conversion** - CoreML conversion for Apple Silicon
6. **Verification** - Contract and shape validation
7. **Evaluation** - CAWS compliance testing
8. **Deployment** - Production packaging

## Phase 1: Environment Setup

### Prerequisites

**Hardware**: Apple Silicon (M1/M2/M3) with ≥32GB RAM
**Python**: 3.10 or 3.11 (NOT 3.13+ for export/conversion)
**Storage**: ≥100GB free space

### Installation

```bash
# Install Python 3.11 (required for export/conversion)
brew install python@3.11

# Clone and setup
git clone <repo>
cd distill
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -e .
pip install coremltools>=9.0

# Verify versions
python3.11 --version  # Should be 3.11.x
python -c "import torch; print(torch.__version__)"  # Should be 2.9.1+
python -c "import coremltools; print(coremltools.__version__)"  # Should be 9.0+
```

### Version Gate Check

```bash
# Verify all components compatible
python -m infra.version_gate --skip-ort
```

## Phase 2: Dataset Generation

### Option A: Production KD Dataset (Recommended)

Generate from Kimi K2 API or local teacher:

```bash
# Generate KD dataset (production scale)
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher https://api.kimi.com/v1 \
    --total 10000 \
    --cache-dir data/logits/

# Verify dataset quality
python -m scripts.validate_kd_data --in data/kd_mix.jsonl
```

### Option B: Contextual Dataset (Research)

For research and evaluation:

```bash
# Generate contextual prompts
make contextual-gen

# Extract process targets
make contextual-extract

# Verify extraction quality
make contextual-verify

# Final dataset ready: data/contextual_final.jsonl
```

### Dataset Validation

```bash
# Comprehensive validation
python -m scripts.verify_contextual_set \
    --in data/contextual_final.jsonl \
    --report data/verification_report.json \
    --tokenizer models/student/tokenizer
```

## Phase 3: Training

### Worker Model Training (Primary)

```bash
# Train 9B GQA worker model
make worker

# Checkpoints saved to: models/student/checkpoints/latest.pt
```

### Optional: Judge Model Training

```bash
# Train constitutional arbiter
make judge

# Checkpoints saved to: arbiter/judge_training/artifacts/judge.pt
```

### Training Validation

```bash
# Run smoke tests
make smoke_training

# Quick evaluation during training
python -m eval.cli \
    --runner hf_local \
    --model models/student/checkpoints/latest.pt \
    --in data/contextual_final.jsonl \
    --out eval/results/training_eval.jsonl \
    --report eval/reports/training_eval.json
```

## Phase 4: Export

### Pre-Export Checks

```bash
# Verify checkpoint integrity
python -c "
import torch
ckpt = torch.load('models/student/checkpoints/latest.pt', map_location='cpu')
print(f'Step: {ckpt.get(\"step\", 0)}')
print(f'Config: {ckpt.get(\"config\", {})}')
"

# Check model architecture
python -c "
import torch
ckpt = torch.load('models/student/checkpoints/latest.pt')
print('Model arch:', ckpt.get('model_arch', {}))
"
```

### PyTorch Export (Production Path)

**CRITICAL**: Use Python 3.11 for export

```bash
# Export with enumerated shapes (production)
python3.11 -m conversion.export_pytorch \
    --checkpoint models/student/checkpoints/latest.pt \
    --out models/student/exported/ \
    --mode both \
    --enumerated-T 512 1024 2048

# Verify exports created
ls -la models/student/exported/
# Should see: student_prefill_T*.pt, student_decode.pt, *_contract.json
```

### Export Validation

```bash
# Check contract files
cat models/student/exported/student_prefill_T1024_contract.json | jq '.shape_validation'

# Verify enumerated shapes
python -c "
import json
contract = json.load(open('models/student/exported/student_prefill_T1024_contract.json'))
shapes = contract.get('enumerated_T', [])
print(f'Enumerated shapes: {shapes}')
for shape in shapes:
    result = contract.get('shape_validation', {}).get(str(shape), {})
    status = result.get('status', 'unknown')
    print(f'T{shape}: {status}')
"
```

## Phase 5: CoreML Conversion

### Pre-Conversion Setup

```bash
# Ensure Python 3.11 active
python3.11 --version

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Conversion Command

```bash
# Convert primary shape (T1024) - production
python3.11 -m conversion.convert_coreml \
    --backend pytorch \
    --in models/student/exported/student_prefill_T1024.pt \
    --out coreml/artifacts/worker/model.mlpackage \
    --contract models/student/exported/student_prefill_T1024_contract.json
```

### Conversion Validation

```bash
# Verify CoreML package
ls -la coreml/artifacts/worker/model.mlpackage/

# Check model loads
python -c "
import coremltools as ct
model = ct.models.MLModel('coreml/artifacts/worker/model.mlpackage')
print('CoreML model loaded successfully')
print('Input names:', [inp.name for inp in model.input_description])
print('Output names:', [out.name for out in model.output_description])
"
```

## Phase 6: Contract Verification

### Run Verification

```bash
# Verify contract compliance
python -m conversion.verify_contract \
    --model coreml/artifacts/worker/model.mlpackage \
    --contract models/student/exported/student_prefill_T1024_contract.json \
    --report models/student/exported/verification_report.json
```

### Verification Results

```bash
# Check verification report
cat models/student/exported/verification_report.json | jq '.gates_ok'

# Review detailed results
python -c "
import json
report = json.load(open('models/student/exported/verification_report.json'))
print('Gates OK:', report.get('gates_ok'))
for gate, result in report.get('gates', {}).items():
    print(f'{gate}: {result}')
"
```

## Phase 7: Evaluation

### CAWS Compliance Testing

```bash
# Full evaluation with CAWS gates
make eval-runner-local \
    MODEL="models/student/checkpoints/latest.pt" \
    IN="data/contextual_final.jsonl" \
    OUT="eval/results/final_eval.jsonl" \
    REPORT="eval/reports/final_eval.json"

# Check CAWS compliance
cat eval/reports/final_eval.json | jq '.summary'
```

### CoreML Performance Testing

```bash
# CoreML speed evaluation
make speed-coreml \
    HARDWARE="$(python -c "import platform; print(platform.processor() or 'unknown')")" \
    EXPORT_PATH="pytorch_exportedprogram_coreml"
```

### Evaluation Validation

```bash
# Verify fixture coverage (critical for CAWS)
make eval-fixture-stats

# Check for regressions
python -m eval.reports.diff_reports \
    eval/reports/baseline.json \
    eval/reports/final_eval.json
```

## Phase 8: Deployment

### Generate Deployment Artifacts

```bash
# Full deployment with all artifacts
make deploy-full

# Or use script directly
python -m scripts.deploy_model \
    --checkpoint models/student/checkpoints/latest.pt \
    --out-dir models/student/deployed/ \
    --export-pytorch \
    --export-coreml \
    --latent-mode \
    --halt-head \
    --caws-tier tier_2
```

### Deployment Validation

```bash
# Verify deployment artifacts
ls -la models/student/deployed/
# Should contain: model_prefill_fp16.pt, model.mlpackage, runtime_config.json, deployment_manifest.json

# Test runtime config
python -c "
import json
config = json.load(open('models/student/deployed/runtime_config.json'))
print('Runtime config:', config)
"

# Test deployment manifest
python -c "
import json
manifest = json.load(open('models/student/deployed/deployment_manifest.json'))
print('Model capabilities:', manifest['capabilities'])
"
```

### Production Inference Test

```bash
# Test production inference
python -m scripts.inference_production \
    --model models/student/deployed/model_prefill_fp16.pt \
    --prompt "Read the file config.json" \
    --output test_output.json

# Verify output
cat test_output.json | jq '.'
```

## Troubleshooting Guide

### Common Issues

#### Python Version Errors

**Error**: `Version check failed: Python 3.13 detected`

**Fix**:
```bash
# Use Python 3.11 for export/conversion
python3.11 -m conversion.export_pytorch --checkpoint model.pt --out exported/

# Verify Python 3.11 installed
brew install python@3.11
python3.11 --version
```

#### CoreML Conversion Fails

**Error**: `Torch version 2.9.1 has not been tested with coremltools`

**Fix**:
- This is non-fatal - conversion usually succeeds
- Monitor logs for actual errors vs warnings
- If conversion fails, try: `pip install torch==2.8.0`

#### Shape Validation Errors

**Error**: `RuntimeError` for certain shapes (T64/T256)

**Fix**:
- Primary shape (T1024) must work
- Secondary shape failures are logged but don't block
- Check contract file for which shapes succeeded

#### Checkpoint Loading Issues

**Error**: `Missing 'config' field in checkpoint`

**Fix**:
- Verify checkpoint was saved with proper metadata
- Check training completed successfully
- Re-export if necessary

### Recovery Procedures

#### Export Fails Mid-Way

```bash
# Check partial exports
ls -la models/student/exported/

# Resume from existing exports
# (Export script will skip already completed shapes)
python3.11 -m conversion.export_pytorch \
    --checkpoint models/student/checkpoints/latest.pt \
    --out models/student/exported/ \
    --enumerated-T 512 1024 2048
```

#### Conversion Fails

```bash
# Try with different compute units
python3.11 -m conversion.convert_coreml \
    --backend pytorch \
    --in model.pt \
    --out model.mlpackage \
    --compute-units cpuonly

# Check conversion logs for specific errors
# Look for: torch import issues, shape mismatches, dtype problems
```

#### Verification Fails

```bash
# Check verification report details
cat verification_report.json | jq '.errors'

# Common fixes:
# - Ensure contract file matches model
# - Verify enumerated shapes in both files
# - Check model output specifications
```

## Quality Gates

### Pre-Deployment Checks

- [ ] Python 3.11 available for export/conversion
- [ ] Checkpoint loads and validates
- [ ] Export completes for primary shape (T1024)
- [ ] CoreML conversion succeeds
- [ ] Contract verification passes
- [ ] CAWS evaluation meets thresholds
- [ ] Fixture hit rate ≥ 95%
- [ ] Deployment artifacts generated
- [ ] Production inference test passes

### Production Readiness

- [ ] All enumerated shapes export successfully (512/1024/2048)
- [ ] CoreML performance meets latency targets
- [ ] Memory usage within limits
- [ ] Evaluation metrics stable across runs
- [ ] Deployment manifest accurate
- [ ] Runtime configuration validated

## Automation

### Makefile Targets

```bash
# Complete production pipeline (recommended)
make worker                    # Train
make pytorch-worker           # Export
make coreml-worker            # Convert
make eval-runner-local        # Evaluate
make deploy-full              # Deploy

# Individual phases
make kd-dataset               # Dataset generation
make contextual-pipeline      # Research dataset
make proc                     # Process supervision
make qat                      # Quantization
make probes                   # CoreML validation
```

### CI/CD Integration

For automated pipelines:

```yaml
# GitHub Actions example
- name: Run Distillation Pipeline
  run: |
    make worker
    make pytorch-worker
    make coreml-worker
    make eval-runner-local
    make deploy-full

- name: Quality Gates
  run: |
    # Verify deployment artifacts
    test -f models/student/deployed/model.mlpackage
    test -f models/student/deployed/runtime_config.json

    # Check evaluation scores
    jq -e '.summary.fixture_hit_rate >= 0.95' eval/reports/final_eval.json
```

## Performance Optimization

### Training Optimizations

```bash
# Use enumerated shapes during training
python -m training.distill_kd \
    --config configs/student_9b_gqa.yaml \
    --config configs/kd_recipe.yaml \
    --enumerated-shapes 512,1024,2048
```

### Inference Optimizations

```bash
# Enable speculative decoding
make speed-coreml-speculative

# Test prompt caching
make speed-coreml-cached

# Measure ANE residency
make speed-coreml-ane
```

## See Also

- [`README.md`](../README.md) - Quick start overview
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) - Detailed deployment guide
- [`docs/PRODUCTION_PIPELINE_CHECKLIST.md`](PRODUCTION_PIPELINE_CHECKLIST.md) - Pre-deployment checklist
- [`Makefile`](../Makefile) - All available targets
- [`scripts/README.md`](scripts/README.md) - Script documentation
