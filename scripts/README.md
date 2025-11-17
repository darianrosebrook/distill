# Scripts

Dataset generation, verification, and utility scripts.

## Overview

This directory contains scripts for:
- **Dataset Generation**: Knowledge distillation dataset creation
- **Process-Step Extraction**: Tool name, JSON, integration span extraction
- **Verification**: Dataset quality and correctness validation
- **Utilities**: Token span alignment, PII redaction, etc.

## Dataset Generation

### `make_kd_mix.py`
Generate knowledge distillation dataset from teacher model:
- Supports HTTP API and HuggingFace backends
- Caching support for cost optimization
- Tier-aware rate limiting

**Usage**:
```bash
python -m scripts.make_kd_mix \
  --out data/kd_mix.jsonl \
  --teacher https://api.kimi.com/v1 \
  --total 1000
```

### `make_kd_mix_hardened.py`
Hardened version with:
- Budget tracking and enforcement
- Checkpoint recovery
- Process-step target extraction
- Tier-aware backoff

### `generate_contextual_prompts.py`
Generate contextual dataset with tool integration:
- Stratified scenario coverage
- Process-step targets
- Multi-locale validation
- Scale tests (1k/10k)

**Usage**:
```bash
python -m scripts.generate_contextual_prompts \
  --out data/contextual_prompts.jsonl \
  --total 60 \
  --seed 42
```

## Process-Step Extraction

### `extract_process_targets.py`
Extract process-step supervision targets:
- Tool name token spans
- JSON argument token spans
- Integration span token IDs
- Token alignment validation

**Usage**:
```bash
python -m scripts.extract_process_targets \
  --in data/contextual_prompts.jsonl \
  --out data/contextual_extracted.jsonl \
  --tokenizer-path models/student/tokenizer
```

## Verification

### `verify_contextual_set.py`
Comprehensive dataset verification:
- Integration F1 (lax/strict)
- Privacy compliance
- Control contamination checks
- Fixture hit rate validation
- Token alignment verification

**Usage**:
```bash
python -m scripts.verify_contextual_set \
  --in data/contextual_extracted.jsonl \
  --report data/contextual_verification_report.json \
  --tokenizer models/student/tokenizer
```

## Utilities

### `util_token_spans.py`
Token span alignment utilities:
- Byte-to-token span conversion
- Text normalization for alignment
- Span validation

### `util_sanitize.py`
PII redaction and URL allowlisting:
- Email/phone/SSN detection
- URL context validation
- Privacy compliance

### `validate_sharding_determinism.py`
Validate sharded evaluation determinism:
- Shard membership consistency
- Per-example equivalence
- Metric comparison with tolerance

## See Also

- [`docs/CONTEXTUAL_DATASET_GENERATION.md`](../docs/CONTEXTUAL_DATASET_GENERATION.md) - Complete generation guide
- [`docs/DATASET_CARD_CONTEXTUAL.md`](../docs/DATASET_CARD_CONTEXTUAL.md) - Dataset schema
- [`data/README.md`](../data/README.md) - Dataset directory documentation











