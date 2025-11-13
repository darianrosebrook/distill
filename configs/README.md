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

