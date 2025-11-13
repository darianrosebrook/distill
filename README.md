# kimi-student

**Multi-Model Distillation and Governance Stack for CAWS-Compliant AI on Apple Silicon**

---

## Overview

**kimi-student** is a multi-model distillation and governance framework that trains small, ANE-optimized models ("students") under the **CAWS** (Coding-Agent Working Spec) standard.

It unifies _teacher–student distillation_, _tool-use dataset generation_, and _deterministic evaluation_ into a single reproducible pipeline—complete with gating, fingerprinting, and CI governance for CoreML deployment on Apple Silicon.

> **Goal:** Reproduce the reasoning quality of large teachers (e.g., Kimi K2 Thinking 32B) in small, local models that can run on-device while remaining verifiably compliant with CAWS constitutional rules.

---

## Architecture at a Glance

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  K2 Teacher  │ → │ KD Dataset   │ → │ Student Train│
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              │
┌────────────────────────────┘
▼
┌────────────────────────────┐
│ Eval Harness (CAWS Gates) │
│ • Tool broker fixtures     │
│ • Verifier-parity scoring  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────┐
│  Reports / CI Dash │
└────────────────────┘
```

---

## Model Portfolio

| Model                  | Params        | Context   | Role                                                | Precision                       | Deployment   |
| ---------------------- | ------------- | --------- | --------------------------------------------------- | ------------------------------- | ------------ |
| **Worker**             | ~9 B (GQA)    | 8 – 16 k  | Code edits · tool-use JSON · long-context retrieval | INT8 weights + FP16 activations | CoreML (ANE) |
| **Judge**              | 3 – 4 B / 7 B | 512 – 2 k | CAWS arbiter · constitutional adjudication          | INT8 + FP16                     | CoreML       |
| **Drafter** (optional) | ~4 B          | ≤ 2 k     | Speculative decoding · sub-second TTFA              | INT8 + FP16                     | CoreML       |

**Why CoreML / Apple Silicon**

CoreML provides near-zero-latency inference on the Apple Neural Engine, enabling real-time arbitration and speculative decoding without network calls.

All exports use enumerated shapes (512/1024/2048) and quantized FP16/INT8 formats for ANE efficiency.

### CoreML Production Environment (Apple Silicon)

For Worker/Judge CoreML exports:

- **Hardware**: Apple Silicon (M1/M2/M3; 32–64 GB recommended)
- **Python**: 3.10 or 3.11 (not 3.13+)
- **CoreMLTools**: ≥ 9.0
- **Export path**: PyTorch ExportedProgram → CoreML (not ONNX)
- **Shapes** (recommended): 512/1024/2048 (optionally 4096)
- **Precision**: INT8 weights, FP16 activations

Commands:

```bash
make pytorch-worker    # export PyTorch model first
make coreml-worker     # production path via PyTorch exporter
```

---

## Quick Start

### 1 · Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2 · Build Knowledge-Distillation Dataset

```bash
# Option A: Kimi K2 API (recommended for M1 Max)
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher https://api.kimi.com/v1

# Option B: Local HTTP teacher
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher http://localhost:8000

# Option C: Hugging Face model (80 GB GPU)
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher hf:moonshotai/Kimi-K2-Thinking
```

> _K2 Thinking (32 B) cannot run locally on M1 Max — see_ `docs/external/KIMI_K2_SETUP.md`.

### 3 · Train the Worker

```bash
make worker
make pytorch-worker    # export PyTorch model (production path)
make coreml-worker     # convert to CoreML (requires PyTorch export)
```

> **Note**: `make onnx-worker` is optional for debug only. The production path is PyTorch ExportedProgram → CoreML (not ONNX).

### 4 · Evaluate a Real Model (not verification-only)

Pick ONE runner:

**Local HF checkpoint (recommended during training)**

```bash
make eval-runner-local \
  MODEL="/path/to/ckpt" \
  IN="data/contextual_final.jsonl" \
  OUT="eval/results/local.jsonl" \
  REPORT="eval/reports/latest.json"
```

**OpenAI-compatible endpoint (for baselines)**

```bash
make eval-runner-openai \
  MODEL="gpt-4o" \
  IN="data/contextual_final.jsonl" \
  OUT="eval/results/4o.jsonl" \
  REPORT="eval/reports/latest.json"
```

> **Note on verification-only mode**
>
> `runner: verification_only` is for dataset QA. It reuses dataset ground truth to compute a proxy F1 and will **always** be insufficient for CAWS gates. Use `hf_local` or `openai_http` for real evaluation.

### 5 · Process Supervision & Quantization

```bash
make probes
make proc     # process supervision
make qat      # quantization-aware training
```

### 6 · Add Governance (Judge Model)

```bash
make judge_train
make judge_onnx
make judge_coreml
make caws-eval
```

---

## Evaluation Harness and CAWS Gates

Every model is verified by a deterministic evaluation loop that enforces **constitutional gates**:

| Gate                 | Metric | Threshold   | Action |
| -------------------- | ------ | ----------- | ------ |
| Integration F1 (lax) | ≥ 0.90 | Pass        |        |
| Privacy OK Rate      | = 1.0  | Pass        |        |
| Control Integration  | = 0    | Hard fail   |        |
| Fixture Hit-Rate     | ≥ 95 % | Warn / Fail |        |

**Reproducibility guarantees**

- SHA-256 fingerprints for dataset · tool registry · tokenizer · model

- Deterministic fixture replay via `ToolBroker`

- Verifier-parity scoring for strict / lax F1

- Gating enforced in CI; regressions halt merge

For details see [`eval/HARNESS.md`](eval/HARNESS.md).

#### Fixture Coverage (Grounding)

This repo enforces fixture replay for tool calls. To pass gates:

- **Fixture hit-rate ≥ 95%** (target ≥ 98%)
- Add missing fixture files under `eval/tool_broker/fixtures/`:
  - `read_file.jsonl`, `web.search.jsonl`, `code.execute.jsonl`, etc.
- Run:

```bash
make eval-fixture-stats
# or
jq '.summary | {fixture_hit_rate, fixture_miss_count}' eval/reports/latest.json
```

#### Token Alignment & Strict F1

Strict scoring requires byte/token span alignment. Enable during generation:

```bash
make contextual-gen
make contextual-extract    # emits integration_spans_bytes and token ids
make contextual-verify
```

The report should show `token_align_ok_rate > 0`. If it is 0.0, strict gates will be inconclusive.

#### Reproducibility & Fingerprints (Gates)

Every report header must include:

- `dataset_sha256`
- `tool_registry_sha256`
- `tokenizer_fingerprint`

CI fails if any are null/unknown:

```bash
jq '.header | {dataset_sha256, tool_registry_sha256, tokenizer_fingerprint}' eval/reports/latest.json
```

---

## Contextual Dataset Generation & Verification

End-to-end pipeline for process-step supervision:

```bash
# Full pipeline: generate → extract → verify
make contextual-pipeline

# Scale tests
make gen-scale-1k verify-scale-1k
make gen-scale-10k verify-scale-10k
```

### Features

- Stratified scenario × complexity × structure coverage

- Process-step targets (tool names + JSON arguments + byte/token offsets)

- Multi-locale validation ("1 234,56" vs "1,234.56")

- Deterministic evaluation and fixture replay

- Scale tests (1 k / 10 k samples)

Docs:

[`docs/DATASET_CARD_CONTEXTUAL.md`](docs/DATASET_CARD_CONTEXTUAL.md) ·

[`docs/CONTEXTUAL_DATASET_GENERATION.md`](docs/CONTEXTUAL_DATASET_GENERATION.md)

---

## CI and Governance Workflows

### Nightly Evaluation

Runs full gated evaluation on latest checkpoint (configure in `.github/workflows/`).

### PR Smoke Tests

Small deterministic slice enforcing fixture coverage and CAWS gates before merge (see `.github/workflows/broker-smoke.yml`).

### Toy End-to-End Pipeline

Lightweight E2E verification that tests the full flow: dataset generation → training → export → CoreML conversion → verification gates. Runs in ≤4 minutes on CPU.

```bash
# Single shape (fast, default)
make toy-e2e

# Multiple enumerated shapes (slower, more thorough)
make toy-e2e-multi
```

**CoreML Enumerated Shapes (Toy Path):**

In practice, CoreML compiles one shape per mlpackage quickly; multi-shape enumerations are possible but increase compile time and often degrade determinism in CI. The toy E2E test therefore compiles **one representative shape** by default (T128), with an optional "multi" target that compiles `T64/T128/T256` as **separate** packages and verifies that **at least one** shape runs end-to-end. Production models should still export enumerated shapes per your main conversion path; the toy path is intentionally minimal.

**Gates:**

- ≥1 enumerated shape compiles and runs end-to-end
- No NaN/zero outputs detected
- Tool span micro-F1 ≥ 0.20 (via deterministic greedy decode)
- Per-shape diagnostics included in report

### Makefile Shortcuts

```bash
make eval-runner-openai    # evaluate with OpenAI-compatible endpoint
make eval-runner-local     # evaluate with local HuggingFace model
make eval-smoke            # smoke test with GPT-4
make toy-e2e               # toy E2E pipeline (single shape)
make toy-e2e-multi         # toy E2E pipeline (multiple shapes)
```

---

## Reproducibility Artifacts

| Directory                     | Contents                 | Fingerprinted By        |
| ----------------------------- | ------------------------ | ----------------------- |
| `models/student/checkpoints/` | Distillation checkpoints | SHA-256                 |
| `artifacts/onnx/`             | ONNX graphs              | Model + tokenizer       |
| `coreml/artifacts/`           | CoreML mlpackages        | Enumerated shapes       |
| `eval/reports/`               | JSON reports             | Dataset · Tool registry |

Each report embeds `dataset_sha256`, `tool_registry_sha256`, and `prompt_wrapper_sha256`.

---

## Extending the System

| Extension                  | How to Add                                       | Docs                      |
| -------------------------- | ------------------------------------------------ | ------------------------- |
| **New Tools**              | Add fixture + schema → update `tool_broker`      | `eval/tool_broker/`       |
| **New Models**             | Add config YAML → implement ONNX/CoreML exporter | `conversion/`             |
| **New Governance Clauses** | Extend `arbiter/schemas` + train Judge           | `arbiter/judge_training/` |

---

## Directory Structure (Top Level)

```
arbiter/         – CAWS governance stack (judge training · schemas)
configs/         – Model configs (worker / judge / drafter)
conversion/      – PyTorch → ONNX → CoreML exporters
scripts/         – Dataset generation & verification
eval/            – Evaluation harness · runners · reports
tests/           – Unit · integration · property · CI smoke
```

---

## Roadmap / Next Focus

1. **Integration Testing & Validation**

   - Verify normalization robustness

   - Validate fingerprint tracking

2. **Fixture Coverage Expansion**

   - Add common tool fixtures (`math.eval`, `web.crawl`, `file.write`)

   - Target ≥ 98 % broker coverage

3. **Runner Parity Validation**

   - Ensure OpenAI and HF runners match (ΔF1 ≤ 0.03)

4. **Performance & Scalability**

   - Parallel evaluation for 10 k+ datasets

5. **Dashboard Publishing**

   - Time-series history tracking

   - Automated report aggregation

---

## Summary

kimi-student now forms a **self-auditing infrastructure for grounded tool-integration learning**.

It delivers **distilled CoreML models** verified through **CAWS-compliant evaluation**, with constitutional gates enforced in CI and reproducible governance for on-device deployment.

> **Next milestones:**

> • Expand fixture library (≤ 2 % miss rate)

> • Automate dashboard publishing

> • Publicly release Worker and Judge CoreML artifacts for benchmarking

---

## Documentation

- **Dataset Card**: [`docs/DATASET_CARD_CONTEXTUAL.md`](docs/DATASET_CARD_CONTEXTUAL.md) - Dataset schema and policies
- **Generation Guide**: [`docs/CONTEXTUAL_DATASET_GENERATION.md`](docs/CONTEXTUAL_DATASET_GENERATION.md) - Full workflow
- **Evaluation Harness**: [`eval/HARNESS.md`](eval/HARNESS.md) - Evaluation harness documentation
- **Scale Tests**: [`docs/SCALE_TESTS.md`](docs/SCALE_TESTS.md) - Scale testing guide
- **CAWS Guide**: [`docs/CAWS_AGENT_GUIDE.md`](docs/CAWS_AGENT_GUIDE.md) - CAWS agent workflow guide
- **Arbiter Theory**: [`arbiter_theory.md`](arbiter_theory.md) - Arbiter stack architecture
- **Distillation Guide**: [`docs/DISTILLATION_GUIDE.md`](docs/DISTILLATION_GUIDE.md) - Distillation guide and rationale
- **Full Docs Index**: [`docs/README.md`](docs/README.md) - Complete documentation index
