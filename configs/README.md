# Configuration Files

Model and training configuration files.

## Overview

This directory contains YAML configuration files for:
- Model architectures
- Training recipes
- Evaluation suites
- Hardware profiles

## Model Configs

### Worker Model
- `worker_9b.yaml` - Worker model architecture (~9B GQA)

### Judge Model
- `judge_4b.yaml` - Judge model architecture (3-4B)
- `judge_training.yaml` - Judge training configuration

### Drafter Model
- `drafter_4b.yaml` - Drafter model architecture (~4B)

### Legacy Configs
- `student_7b_gqa.yaml` - Legacy 7B config
- `student_8b_gqa.yaml` - Legacy 8B config
- `student_9b_gqa.yaml` - Legacy 9B config

## Training Configs

### Knowledge Distillation
- `kd_recipe.yaml` - Main KD recipe with loss weights and curriculum

### Process Supervision
- `process_supervision.yaml` - Process-step supervision configuration

### Quantization
- `quant_qat_int8.yaml` - Quantization-aware training config

### Checkpoint Management

Training configurations support automatic checkpoint cleanup to prevent disk space issues:

```yaml
training:
  steps: 1500
  save_every: 100
  checkpoint_cleanup:
    max_checkpoints: 3  # Keep most recent N checkpoints (plus milestones)
    min_free_space_gb: 50.0  # Reduce retention if free space drops below this
```

**Configuration Options**:

- `max_checkpoints` (int, default: 3): Maximum number of recent checkpoints to keep in addition to milestone checkpoints
- `min_free_space_gb` (float, default: 50.0): Threshold for automatic retention reduction when disk space is low

**How It Works**:

- Milestone checkpoints (10%, 25%, 50%, 75%, 90%, 100% of training steps) are always preserved
- The most recent N checkpoints are kept (where N = `max_checkpoints`)
- Old checkpoints beyond this retention policy are automatically deleted after each save
- If free disk space drops below `min_free_space_gb`, retention is automatically reduced

See [`training/README.md`](../training/README.md#checkpoint-management) for detailed documentation.

## Evaluation Configs

- `eval_suites.yaml` - Evaluation suite definitions
- `hardware_profiles.yaml` - Hardware profile configurations

## Export Configs

- `export_onnx.yaml` - ONNX export configuration
- `convert_coreml.yaml` - CoreML conversion configuration

## Tool Schema

- `tool_schema.json` - Tool registry schema definition

## Usage

Configs are merged when multiple are specified:
```bash
python -m training.distill_kd \
  --config configs/worker_9b.yaml \
  configs/kd_recipe.yaml
```

Later configs override earlier ones.

## See Also

- [`training/README.md`](../training/README.md) - Training documentation
- [`models/README.md`](../models/README.md) - Model architecture documentation















