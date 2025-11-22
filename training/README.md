# Training

Knowledge distillation training scripts, loss functions, and training utilities.

## Overview

This directory contains the complete training pipeline for distilling student models from teacher models using:

- **Knowledge Distillation (KD)**: Soft targets and KL divergence
- **Process-Step Supervision**: Tool name, JSON args, integration spans
- **Code-Mode Training**: Latent reasoning and token reduction
- **Quantization-Aware Training (QAT)**: INT8 quantization

## Device Support

All training scripts automatically detect and use the best available device:
- **CUDA** (NVIDIA GPUs) - highest priority
- **MPS** (Apple Silicon) - second priority  
- **CPU** - fallback

Device detection is handled by `device_utils.get_training_device()` which is used consistently across all training scripts.

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

## Checkpoint Management

### Automatic Checkpoint Cleanup

The training system automatically manages checkpoint storage to prevent disk space exhaustion. After each checkpoint save, old checkpoints are cleaned up while preserving important recovery points.

#### What Gets Kept

The cleanup system preserves:

1. **Milestone Checkpoints**: Always kept at 10%, 25%, 50%, 75%, 90%, and 100% of training steps
2. **Recent Checkpoints**: The most recent N checkpoints (configurable, default: 3)
3. **Latest Checkpoint**: The `latest.pt` checkpoint is always preserved
4. **Current Checkpoint**: The checkpoint being saved is never deleted

#### Configuration

Configure checkpoint cleanup in your training config file:

```yaml
training:
  steps: 1500
  save_every: 100
  checkpoint_cleanup:
    max_checkpoints: 3  # Keep most recent N checkpoints (plus milestones)
    min_free_space_gb: 50.0  # Reduce retention if free space drops below this
```

**Parameters**:

- `max_checkpoints` (default: 3): Maximum number of recent checkpoints to keep (in addition to milestones)
- `min_free_space_gb` (default: 50.0): If free disk space drops below this threshold, retention is automatically reduced

#### How It Works

1. **After Each Save**: When a checkpoint is saved, the cleanup function runs automatically
2. **Disk Space Check**: System checks available disk space
3. **Retention Adjustment**: If space is low (< `min_free_space_gb`), retention is reduced by 1
4. **Cleanup**: Old checkpoints not in the keep list are deleted
5. **Logging**: Cleanup actions are logged with freed space information

#### Example Behavior

For a 1500-step training run with `max_checkpoints: 3`:

- **Step 100**: Saves checkpoint, keeps it (recent)
- **Step 150**: Saves checkpoint, keeps it (milestone 10%)
- **Step 200**: Saves checkpoint, keeps it (recent)
- **Step 300**: Saves checkpoint, keeps it (recent)
- **Step 375**: Saves checkpoint, keeps it (milestone 25%)
- **Step 400**: Saves checkpoint, deletes step 100 (oldest non-milestone)
- **Step 500**: Saves checkpoint, keeps it (milestone 50%)
- **Step 600**: Saves checkpoint, deletes step 200 (oldest non-milestone)
- And so on...

Milestone checkpoints (150, 375, 750, 1125, 1350, 1500) are always preserved regardless of the `max_checkpoints` setting.

#### Disk Space Management

The system monitors disk space and automatically adjusts behavior:

- **Normal Operation**: Keeps `max_checkpoints` recent checkpoints + milestones
- **Low Space**: Reduces retention by 1 checkpoint if free space < `min_free_space_gb`
- **Critical Space**: More aggressive cleanup if space is critically low

Checkpoint cleanup logs show:
```
[distill_kd] Deleted old checkpoint: checkpoint_step_100.pt (23.00GB)
[distill_kd] Checkpoint cleanup: deleted 1 old checkpoints, freed 23.00GB. Keeping 4 checkpoints.
```

### Manual Checkpoint Management

If you need to manually manage checkpoints:

```bash
# List all checkpoints
ls -lh models/student/checkpoints_*/checkpoint_step_*.pt

# Delete specific checkpoint
rm models/student/checkpoints_*/checkpoint_step_100.pt

# Resume from checkpoint
python -m training.distill_kd \
  --config configs/worker_9b.yaml \
  --resume models/student/checkpoints_*/checkpoint_step_700.pt
```

### Recovery

The training system automatically detects and resumes from the most recent checkpoint:

1. Checks `progress/checkpoints.json` for checkpoint metadata
2. Loads the checkpoint with the highest step number
3. Resumes training from that step

You can also explicitly specify a checkpoint with `--resume`.

## See Also

- [`docs/DISTILLATION_GUIDE.md`](../docs/DISTILLATION_GUIDE.md) - Complete distillation guide
- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Pipeline review
- [`data/README.md`](../data/README.md) - Dataset format documentation
