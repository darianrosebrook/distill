# Data Directory

Dataset files, generated data, and data processing artifacts.

> **⚠️ Important**: Actual dataset files (`.jsonl`, `.json`) are **private** and kept in a separate private git repository. Only infrastructure (scripts, schemas, generators) is tracked in this public repository. See [`DATA_PRIVACY_SETUP.md`](DATA_PRIVACY_SETUP.md) for details.

## Overview

This directory contains:

- **KD Datasets**: Knowledge distillation datasets (`kd_mix.jsonl`) - _private_
- **Contextual Datasets**: Contextual tool-integration datasets - _private_
- **Test Data**: Test datasets and fixtures - _private_
- **Reports**: Dataset verification reports
- **Cache**: Teacher response cache (gitignored)

## Dataset Files

### Knowledge Distillation

- `kd_mix.jsonl` - Main KD dataset from teacher model
- `kd_mix_test.jsonl` - Test KD dataset

### Contextual Datasets

- `contextual_prompts.jsonl` - Generated contextual prompts
- `contextual_extracted.jsonl` - With process-step targets extracted
- `contextual_final.jsonl` - Verified final dataset

### Test Data

- `test_*.jsonl` - Various test datasets
- `judge/*.jsonl` - Judge training datasets

## Directory Structure

```
data/
├── generators/        # Dataset generation utilities
├── wrappers/         # Data wrapper utilities (curriculum, etc.)
├── resources/        # Resource files (TypeScript)
├── reports/          # Verification reports
├── checkpoints_test/ # Test checkpoints (gitignored)
└── kd_cache_test/    # Test cache (gitignored)
```

## Dataset Format

### KD Dataset Format

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

### Contextual Dataset Format

See [`docs/DATASET_CARD_CONTEXTUAL.md`](../docs/DATASET_CARD_CONTEXTUAL.md) for complete schema.

## Generation

Generate datasets using scripts in `scripts/`:

```bash
# KD dataset
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher <endpoint>

# Contextual dataset
python -m scripts.generate_contextual_prompts --out data/contextual_prompts.jsonl
```

## Verification

Verify datasets before training:

```bash
python -m scripts.verify_contextual_set \
  --in data/contextual_extracted.jsonl \
  --report data/reports/verification.json
```

## See Also

- [`docs/CONTEXTUAL_DATASET_GENERATION.md`](../docs/CONTEXTUAL_DATASET_GENERATION.md) - Generation guide
- [`docs/DATASET_CARD_CONTEXTUAL.md`](../docs/DATASET_CARD_CONTEXTUAL.md) - Dataset schema
- [`scripts/README.md`](../scripts/README.md) - Generation scripts

## Version v1.1

**Promoted**: 2025-11-22 01:48:52  
**Dataset**: `worker_production_tools_v1.jsonl`  
**Status**: ⚠️ Check coverage report

