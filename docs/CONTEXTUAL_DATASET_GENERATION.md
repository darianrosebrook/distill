# Contextual Dataset Generation

This document describes the enhanced contextual dataset generation system for testing process-step supervision losses (`w_tool`, `w_args`, `w_integr`) and CAWS integration.

## Overview

The system generates test datasets with:
- **Stratified coverage** across scenarios × complexity × structure
- **Control cases** (no-tool, decline, retry)
- **Adversarial cases** (range violations, malformed JSON, ambiguity)
- **Multi-lingual support** (5-10% non-English prompts)
- **Long-context samples** (8-12k tokens with distractions)
- **Comprehensive verification** with hard stops

## Components

### 1. Prompt Generator (`scripts/generate_contextual_prompts.py`)

Generates prompts with strict stratification enforcement.

**Usage:**
```bash
python -m scripts.generate_contextual_prompts \
    --out data/test_contextual_prompts.jsonl \
    --total 60 \
    --enforce-stratification
```

**Features:**
- Enforces minimum coverage per (scenario × complexity) cell
- Generates control cases (≥10% of dataset)
- Generates adversarial cases (at least one per type)
- Includes multi-lingual samples (5-10% non-English)
- Adds long-context samples (2-3 samples with 8-12k tokens)
- Uses compact CAWS JSON headers (≤30 tokens)

### 2. Process-Step Target Extractor (`scripts/extract_process_targets.py`)

Extracts process-step supervision targets and adds token spans.

**Usage:**
```bash
python -m scripts.extract_process_targets \
    --in data/test_contextual.jsonl \
    --out data/test_contextual_final.jsonl \
    --tokenizer-path models/student/tokenizer \
    --add-token-spans
```

**Features:**
- Extracts tool name spans (bytes + tokens)
- Extracts JSON argument spans (bytes + tokens)
- Extracts integration spans (bytes + tokens)
- Validates arguments against schema registry
- Extracts integration fields for F1 scoring

### 3. Verification Script (`scripts/verify_contextual_set.py`)

Comprehensive verification with hard stops.

**Usage:**
```bash
python -m scripts.verify_contextual_set \
    --in data/test_contextual_final.jsonl \
    --report data/test_contextual.report.json \
    --tokenizer models/student/tokenizer \
    --check-stratification \
    --check-controls \
    --check-adversarial
```

**Checks:**
- Stratification heatmap (all cells populated)
- Control case validation (no_tool, decline, retry)
- Adversarial case validation (range violations, malformed JSON, ambiguity)
- Token alignment round-trip (≥99.5%)
- Integration F1 scoring (≥0.90)
- CAWS header parsing (≥95%)
- Privacy scan (0 emails, 0 UUIDs, URL allowlist)
- Multi-lingual extraction rate (≥90%)
- Long-context stability (within 3% of short-context)

## Workflow

### Step 1: Generate Prompts

```bash
python -m scripts.generate_contextual_prompts \
    --out data/test_contextual_prompts.jsonl \
    --total 60
```

### Step 2: Generate Teacher Responses

```bash
python -m scripts.make_kd_mix_hardened \
    --out data/test_contextual.jsonl \
    --teacher <teacher_endpoint> \
    --prompts-file data/test_contextual_prompts.jsonl \
    --tokenizer-path models/student/tokenizer \
    --cache-dir data/test_cache/ \
    --checkpoint-dir data/test_checkpoints/ \
    --temperature 1.0 \
    --max-tokens 16384
```

### Step 3: Extract Process-Step Targets

```bash
python -m scripts.extract_process_targets \
    --in data/test_contextual.jsonl \
    --out data/test_contextual_final.jsonl \
    --tokenizer-path models/student/tokenizer \
    --add-token-spans
```

### Step 4: Verify Dataset

```bash
python -m scripts.verify_contextual_set \
    --in data/test_contextual_final.jsonl \
    --report data/test_contextual.report.json \
    --tokenizer models/student/tokenizer \
    --check-stratification \
    --check-controls \
    --check-adversarial
```

## Success Criteria

The verification script enforces these thresholds:

- **Coverage**: All (S×C) cells populated; all structure types present
- **Extraction**: ≥95% full success; JSON validity ≥98%; arg semantics ≥98%
- **Integration**: F1 ≥0.90 for multi-call items
- **Controls**: `no_tool` samples show zero tool spans
- **CAWS**: Header present and parsed for ≥95%; token overhead ≤30
- **Safety**: 0 secret hits; URL allowlist pass; PII redaction pass
- **Stability**: Long-context extraction within 3% of short-context
- **Multilingual**: Extraction ≥90% on non-EN subset
- **Token alignment**: Round-trip check ≥99.5%

## Stratification Matrix

Minimum coverage requirements:

| Scenario \ Complexity | single_call | multi_call | branching_error_recovery |
| --------------------- | ----------: | ---------: | -----------------------: |
| file_ops              |          ≥6 |         ≥4 |                       ≥2 |
| web_search            |          ≥4 |         ≥4 |                       ≥2 |
| code_exec             |          ≥3 |         ≥3 |                       ≥2 |
| multi_step            |           — |         ≥4 |                       ≥2 |

Structure types: `flat_args`, `nested_args`, `arrays`, `enums`, `numeric_ranges`, `optional_keys` (≥1 each per scenario)

## Files Created

- `scripts/generate_contextual_prompts.py` - Enhanced prompt generator
- `scripts/extract_process_targets.py` - Process-step target extractor
- `scripts/verify_contextual_set.py` - Comprehensive verifier
- `scripts/util_sanitize.py` - PII redaction utilities
- `scripts/util_token_spans.py` - Token span alignment utilities
- `tools/schema_registry.py` - Enhanced with `validate_args()` function

## Author

@darianrosebrook


