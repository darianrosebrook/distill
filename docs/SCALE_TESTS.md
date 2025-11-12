# Scale Tests for Contextual Dataset Generation

This document describes how to run scale tests (N=1k, N=10k) to verify long-context stratification and dataset quality at scale.

## Overview

Scale tests verify that:
- Long-context quotas remain stable at larger N (2-5% of items)
- Token length distributions match intended spectrum (10th/50th/90th percentiles)
- Integration span cap enforcement works correctly
- Deterministic sharding produces unique sample_ids
- Performance budgets are met

## Prerequisites

- Tokenizer available at `models/student/tokenizer`
- Sufficient disk space for large datasets (~100MB for 1k, ~1GB for 10k)
- Python environment with all dependencies installed

## Quick Start

### N=1k Scale Test

```bash
# Generate 1k samples
make gen-scale-1k

# Extract and verify
make verify-scale-1k
```

### N=10k Scale Test

```bash
# Generate 10k samples (sharded across 10 shards)
make gen-scale-10k

# Extract and verify
make verify-scale-10k
```

## Manual Commands

### Generate N=1k Dataset

```bash
python -m scripts.generate_contextual_prompts \
    --out data/contextual_prompts_1k.jsonl \
    --total 1000 \
    --seed 42 \
    --num-shards 1 \
    --shard-index 0 \
    --integration-span-cap 3 \
    --tokenizer models/student/tokenizer
```

### Generate N=10k Dataset (Sharded)

For 10k samples, use sharding for parallel generation:

```bash
# Shard 0
python -m scripts.generate_contextual_prompts \
    --out data/contextual_prompts_10k_shard0.jsonl \
    --total 10000 \
    --seed 42 \
    --num-shards 10 \
    --shard-index 0 \
    --integration-span-cap 3 \
    --tokenizer models/student/tokenizer

# Shard 1
python -m scripts.generate_contextual_prompts \
    --out data/contextual_prompts_10k_shard1.jsonl \
    --total 10000 \
    --seed 42 \
    --num-shards 10 \
    --shard-index 1 \
    --integration-span-cap 3 \
    --tokenizer models/student/tokenizer

# ... repeat for shards 2-9 ...

# Concatenate shards (lexicographic by shard_index)
cat data/contextual_prompts_10k_shard*.jsonl > data/contextual_prompts_10k.jsonl
```

### Extract Process Targets

```bash
python -m scripts.extract_process_targets \
    --in data/contextual_prompts_1k.jsonl \
    --out data/contextual_extracted_1k.jsonl \
    --tokenizer-path models/student/tokenizer
```

### Verify Dataset

```bash
python -m scripts.verify_contextual_set \
    --in data/contextual_extracted_1k.jsonl \
    --report data/reports/verify_scale_1k.json \
    --tokenizer models/student/tokenizer \
    --perf-budget-sec-per-100 2.0
```

## Success Criteria

### Long-Context Quota

For N ≥ 1k:
- Long-context items should be **2-5%** of total
- Check `summary.long_context_count` in verification report
- Gate: `long_context_quota` passes (warning-only policy)

### Token Length Distribution

Check `summary.token_length_stats` in verification report:
- **10th percentile**: Should match short-context baseline
- **50th percentile**: Should reflect mix of short and medium contexts
- **90th percentile**: Should include long-context samples (8k+ tokens)

### Integration Span Cap

Check `summary.integration_span_count_histogram`:
- **≥99%** of eligible items should have ≤ cap (default: 3)
- Check `summary.integration_spans_over_cap_count` for offenders
- Gate: Warning if >10% exceed cap

### Deterministic Sharding

For sharded generation:
- **Zero duplicate sample_ids** across shards
- Check `summary.duplicate_sample_ids` (should be empty)
- Gate: Hard fail on any duplicates

### Performance Budget

Check `summary.time_per_100_items`:
- Should be ≤ `--perf-budget-sec-per-100` (default: 2.0)
- Gate: Warning if exceeded, with top regex offenders listed

## Example Report Analysis

After running `verify-scale-1k`, check `data/reports/verify_scale_1k.json`:

```json
{
  "header": {
    "report_version": "1.0.0",
    "num_items": 1000,
    "num_eligible": 850,
    "num_controls": 100,
    "num_negative_controls": 0
  },
  "summary": {
    "long_context_count": 30,
    "long_context_quota": {
      "min_pct": 0.02,
      "max_pct": 0.05,
      "actual_pct": 0.03,
      "ok": true
    },
    "token_length_stats": {
      "p10": 512,
      "p50": 2048,
      "p90": 8500
    },
    "integration_span_count_histogram": {
      "0": 100,
      "1": 400,
      "2": 250,
      "3": 100,
      "4": 0
    },
    "integration_spans_over_cap_count": 0,
    "time_per_100_items": 1.8,
    "duplicate_sample_ids": []
  }
}
```

## Troubleshooting

### Long-Context Quota Too Low

If long-context count is < 2%:
- Check that `--disable-long-context` is not set
- Verify tokenizer is available for token-aware generation
- Check that `total >= 20` (long-context only added for N ≥ 20)

### Integration Span Cap Exceeded

If >10% exceed cap:
- Check `summary.integration_spans_over_cap_items` for patterns
- Consider raising `--integration-span-cap` to 4 for multi-call scenarios
- Review teacher response generation logic

### Performance Budget Exceeded

If `time_per_100_items` > budget:
- Check `summary.top_regex_offenders` for longest Integration spans
- Consider optimizing regex patterns in verifier
- Review token alignment performance

### Duplicate Sample IDs

If duplicates found:
- Verify `--num-shards` and `--shard-index` are set correctly
- Check that `sample_id` generation includes `shard_index`
- Ensure deterministic seed is used

## CI Integration

For CI/CD, add these targets to your pipeline:

```yaml
# .github/workflows/scale-tests.yml
- name: Scale Test N=1k
  run: |
    make gen-scale-1k
    make verify-scale-1k
    # Check report for gates
    python -c "import json; r=json.load(open('data/reports/verify_scale_1k.json')); assert r['summary']['long_context_quota']['ok'], 'Long-context quota failed'"
```

## See Also

- `docs/DATASET_CARD_CONTEXTUAL.md` - Dataset card with full schema
- `docs/CONTEXTUAL_DATASET_GENERATION.md` - Generation workflow
- `scripts/verify_contextual_set.py` - Verification script documentation

