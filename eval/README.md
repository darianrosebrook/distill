# Evaluation Harness

Deterministic evaluation harness for tool-integration behaviors with CAWS gates.

## Overview

This directory contains the complete evaluation infrastructure:
- **Runners**: Model execution (OpenAI-compatible, HuggingFace local)
- **Scoring**: Verifier-parity scoring (strict/lax F1)
- **Tool Broker**: Deterministic fixture replay
- **Reports**: Comprehensive evaluation reports

## Quick Start

### Local HuggingFace Model
```bash
python -m eval.cli \
  --runner hf_local \
  --model /path/to/checkpoint \
  --in data/contextual_final.jsonl \
  --out eval/results/local.jsonl \
  --report eval/reports/latest.json \
  --fixtures eval/tool_broker/fixtures
```

### OpenAI-Compatible Endpoint
```bash
python -m eval.cli \
  --runner openai_http \
  --model gpt-4o \
  --in data/contextual_final.jsonl \
  --out eval/results/4o.jsonl \
  --report eval/reports/latest.json \
  --fixtures eval/tool_broker/fixtures
```

## Directory Structure

```
eval/
├── cli.py              # Main evaluation CLI
├── HARNESS.md          # Complete harness documentation
├── runners/            # Model execution runners
│   ├── base.py        # Base runner interface
│   ├── openai_http.py # OpenAI-compatible API runner
│   └── hf_local.py    # HuggingFace local runner
├── scoring/            # Scoring logic
│   ├── scorer.py      # Main scorer (verifier-parity)
│   ├── baseline.py    # Baseline comparison
│   └── efficiency.py  # Efficiency metrics
├── tool_broker/        # Deterministic tool replay
│   ├── broker.py      # Tool broker implementation
│   └── fixtures/      # Tool response fixtures
├── reports/            # Report generation
│   └── summarize.py   # Report summarization
└── schemas/           # Evaluation schemas
```

## CAWS Gates

Every evaluation enforces constitutional gates:

| Gate                 | Metric | Threshold   | Action |
| -------------------- | ------ | ----------- | ------ |
| Integration F1 (lax) | ≥ 0.90 | Pass        |        |
| Privacy OK Rate      | = 1.0  | Pass        |        |
| Control Integration  | = 0    | Hard fail   |        |
| Fixture Hit-Rate     | ≥ 95 % | Warn / Fail |        |

## Determinism

The harness ensures reproducible evaluation:
- SHA-256 fingerprints for all inputs
- Deterministic fixture replay
- Stable hash partitioning for sharding
- Temperature=0, seed handling

## Fixtures

Tool calls are replayed via fixtures in `tool_broker/fixtures/`:
- `read_file.jsonl` - File reading fixtures
- `web.search.jsonl` - Web search fixtures
- `code.execute.jsonl` - Code execution fixtures

**Add missing fixtures**:
```bash
# Check fixture hit rate
make eval-fixture-stats

# Add fixtures to eval/tool_broker/fixtures/
```

## Sharding

Evaluation can be sharded for parallel execution:
```bash
# 4-way sharding
for i in 0 1 2 3; do
  python -m eval.cli \
    --num-shards 4 \
    --shard-index $i \
    --out eval/results/shard_$i.jsonl &
done
wait
```

Sharding uses stable hash partitioning to ensure deterministic results.

## See Also

- [`HARNESS.md`](HARNESS.md) - Complete harness documentation
- [`docs/ACCEPTANCE_GATES.md`](../docs/ACCEPTANCE_GATES.md) - Quality gates
- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Pipeline review




















