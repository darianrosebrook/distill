# Training

Knowledge distillation training scripts, loss functions, and training utilities.

## Overview

This directory contains the complete training pipeline for distilling student models from teacher models using:

- **Knowledge Distillation (KD)**: Soft targets and KL divergence
- **Process-Step Supervision**: Tool name, JSON args, integration spans
- **Code-Mode Training**: Latent reasoning and token reduction
- **Quantization-Aware Training (QAT)**: INT8 quantization

## Key Training Scripts

### `distill_kd.py` (Main Training Script)

Primary knowledge distillation training with support for:

- Intermediate layer matching
- Self-evaluation heads
- Halt heads (for learned halting)
- Process-step supervision
- CAWS compliance loss
- Claim extraction loss
- Code-mode and latent reasoning curriculum

**Usage**:

```bash
python -m training.distill_kd \
  --config configs/worker_9b.yaml \
  configs/kd_recipe.yaml
```

### `distill_process.py`

Process-step supervision training (extends KD):

- JSON validity loss
- Tool selection loss
- Integration span supervision

**Usage**:

```bash
python -m training.distill_process \
  --checkpoint models/student/checkpoints/latest.pt \
  --config configs/worker_9b.yaml \
  configs/process_supervision.yaml \
  --steps 10000
```

### `distill_answer_generation.py`

Answer generation stage training:

- Trains model to generate final answers or code after tool execution
- Uses `AnswerGenerationDataset` for loading answer generation data

**Usage**:

```bash
python -m training.distill_answer_generation \
  --config configs/worker_9b.yaml \
  --data data/answer_generation.jsonl
```

### `distill_intermediate.py` (DEPRECATED)

⚠️ **DEPRECATED**: Intermediate layer loss is now integrated into `distill_kd.py`.

Use `distill_kd.py` with `intermediate_layer_weight` in config instead.

## Loss Functions

### `losses.py`

Comprehensive loss function library:

- `combined_kd_loss()` - Main KD loss (KL divergence + CE)
- `tool_name_loss()` - Tool selection supervision
- `json_argument_loss()` - JSON structure supervision
- `intermediate_layer_loss()` - Hidden state matching
- `CodeModePreferenceLoss` - Code-mode curriculum loss
- `caws_compliance_loss()` - CAWS rule compliance
- `claim_extraction_loss()` - Claim extraction supervision

## Dataset Loading

### `dataset.py`

- `KDDataset` - Main dataset loader for JSONL format
- `collate_kd_batch()` - Batch collation with padding
- Process-step supervision target handling
- Latent curriculum support

**Dataset Format**:

```json
{
  "prompt": "...",
  "teacher_text": "...",
  "tool_name_ids": [...],
  "gold_json_text_ids": [...],
  "integration_mask": [...],
  "metadata": {...}
}
```

## Training Utilities

### `tracing.py`

Training metrics and logging:

- TensorBoard integration
- WandB integration (optional)
- JSON log files
- Console logging

### `extractors.py`

Process-step target extraction:

- Tool name span extraction
- JSON argument span extraction
- Integration span identification

### `tokenizer_migration.py`

Tokenizer and model migration utilities:

- Embedding resizing
- Special token initialization
- Token ID verification

### `halt_targets.py`

Halt head supervision target derivation:

- Curriculum-based halting
- Judge score-based halting
- Delta shrinking detection

## Training Stages

1. **Initial KD**: `make worker` or `make kd`
2. **Process Supervision**: `make proc`
3. **Quantization**: `make qat`

## Configuration

Training is configured via YAML files in `configs/`:

- `worker_9b.yaml` - Worker model architecture
- `kd_recipe.yaml` - Knowledge distillation recipe
- `process_supervision.yaml` - Process-step supervision config
- `quant_qat_int8.yaml` - Quantization config

## See Also

- [`docs/DISTILLATION_GUIDE.md`](../docs/DISTILLATION_GUIDE.md) - Complete distillation guide
- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Pipeline review
- [`data/README.md`](../data/README.md) - Dataset format documentation
