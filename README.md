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
# Use ARM-native Python (Homebrew) for Apple Silicon compatibility
/opt/homebrew/bin/python3.11 -m venv venv && source venv/bin/activate
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

**📖 Complete Workflow Guide**: See [`docs/PRODUCTION_DISTILLATION_WORKFLOW.md`](docs/PRODUCTION_DISTILLATION_WORKFLOW.md) for the full end-to-end distillation pipeline.

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

> **📖 Local Development Tiers**: For comprehensive local validation at different scales, see [`docs/LOCAL_DEVELOPMENT_WORKFLOW.md`](docs/LOCAL_DEVELOPMENT_WORKFLOW.md) - from 5-minute smoke tests to full production pipelines.

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

### Toy Test Suite

<p align="center">
  <img src="8-ball.png" alt="8-ball" width="100" height="100">
  <br>
  <em>🎱 Our toy models enable fast pipeline testing without expensive compute! 🎱</em>
</p>

**Comprehensive lightweight testing without large language models or expensive compute** - validates the complete distillation pipeline using deterministic, optimized test doubles. Enables fast iteration during development.

**Two Toy Models for Different Testing Scenarios:**

1. **Toy Baseline Model** - Tests general pipeline functionality and tool span extraction

   - Purpose: Validates core distillation pipeline, tool-calling patterns, and integration points
   - Training data: Simple tool-calling examples (e.g., `tool.call{"name":"sum","args":{"a":1,"b":2}}`)
   - Use case: General pipeline validation, tool span detection, claims extraction testing
   - Command: `make toy-e2e`

2. **8-ball Model** - Tests domain-specific pattern learning with mystical responses
   - Purpose: Validates domain-specific learning, pattern recognition, and response generation
   - Training data: 8-ball style fortune-telling responses (e.g., "It is certain", "Very doubtful")
   - Use case: Testing domain adaptation, pattern learning, response quality validation
   - Command: `make 8-ball`

Both models use the same architecture (~623K parameters) and training pipeline, but test different aspects of the distillation system. The toy baseline focuses on structural correctness (tool spans, JSON parsing), while the 8-ball model focuses on semantic pattern learning (domain-specific responses).

**Key Features:**

- **Fast execution**: Full pipeline in ≤4 minutes on CPU
- **Complete coverage**: All integration points tested
- **Deterministic**: Reproducible results across environments
- **Same interfaces**: Uses same interfaces and validation as real models

**Performance Benchmarks:**

Our toy models deliver **measured inference speeds**:

| Model            | TTFT       | TPS                  | Parameters | Training Data | Training Cost | Inference Cost (1M queries) |
| ---------------- | ---------- | -------------------- | ---------- | ------------- | ------------- | --------------------------- |
| **🎱 8-ball**    | **1.22ms** | **1,090 tokens/sec** | **~623K**  | 128 samples   | **$0.00003**  | **$0.0025**                 |
| **Toy Baseline** | ~5-10ms    | ~500-800 tokens/sec  | ~623K      | 128 samples   | $0.00003      | $0.005                      |

**Key Metrics:**

- **Parameters**: ~623K (tiny - fits in L2 cache!)
- **Training Data**: 128 samples (minimal dataset for testing)
- **Training Cost**: $0.00003 (10 seconds on CPU)
- **Memory Footprint**: ~2.4MB (perfect for edge deployment)
- **Power Efficiency**: CPU-only, no GPU required
- **Latency**: Sub-millisecond responses
- **Throughput**: >1,000 tokens/second on CPU
- **Carbon Footprint**: Near-zero (CPU training, minimal inference)

_Benchmarks measured on CPU. TTFT = Time to First Token, TPS = Tokens per Second. Costs estimated for AWS CPU instances. Training cost is effectively free - less than 1/100th of a penny!_

**🎱 Ollama Integration**: Convert the 8-ball to GGUF format for Ollama deployment:

```bash
# Convert PyTorch checkpoint to GGUF
python convert_to_gguf.py --checkpoint /tmp/8_ball.ckpt --out 8-ball.gguf

# Create Ollama model
ollama create 8-ball -f Modelfile.8-ball

# Run mystical queries
ollama run 8-ball "Will this work?"
```

The conversion script handles: PyTorch → HuggingFace → GGUF → Ollama Modelfile generation.

**✅ Model Validation**: The 8-ball successfully learned its training patterns! It generates appropriate responses with 8-ball phrases like "Concentrate and ask again", "Very doubtful", and "It is decidedly so". This validates that the distillation pipeline can learn domain-specific patterns from teacher outputs.

### Conversion Process Learnings

**Practical lessons from converting our toy model to production format:**

**Vocabulary Expansion:**

- Toy model: 512 tokens → Ollama requirement: 32,002 tokens (+31,490 extra tokens)
- **Scaling implication**: Larger models need careful vocabulary alignment with target ecosystems
- **Cost**: ~6x file size increase (1.1MB → 17.4MB) due to expanded embeddings

**GGUF Conversion Pipeline:**

- PyTorch checkpoint → HuggingFace format → GGUF conversion → Ollama registration
- **Architecture compatibility**: ARM64 compilation issues required specific flags
- **Python version sensitivity**: Python 3.11 required for stable conversion
- **Model structure**: Llama-style transformers convert cleanly; custom architectures may need adaptation

**Performance Scaling:**

- Toy model (623K params): 1.22ms TTFT, 1,090 TPS on CPU
- **Efficiency trend**: Sub-millisecond latency maintained at small scale
- **Memory**: 2.4MB footprint (fits in CPU cache)
- **Throughput**: >1,000 TPS suggests good parallelization potential

**Cost Scaling:**

- Training: $0.00003 (10 seconds CPU)
- Inference: $0.0025 per 1M queries
- **Scaling factor**: Costs scale roughly with parameter count and compute requirements
- **Optimization opportunity**: CPU-only inference keeps costs minimal

**Model Behavior:**

- Learned patterns rather than memorizing exact sequences (normal for generative models)
- Deterministic teacher logits during training produced consistent learning
- Temperature sampling (0.8) in inference adds appropriate randomness

**Production Readiness Checklist:**

- [ ] Verify target ecosystem vocabulary requirements
- [ ] Test conversion pipeline with model architecture
- [ ] Validate performance scaling assumptions
- [ ] Plan for increased storage/compute costs
- [ ] Consider model size vs. performance trade-offs

```bash
# E2E pipeline tests
make toy-e2e               # Single shape (fast, default)
make toy-e2e-multi         # Multiple shapes (thorough)

# Feature-specific tests
pytest tests/ -k toy       # All toy tests
pytest tests/e2e/test_toy_pipeline.py -v
pytest tests/e2e/test_toy_code_mode.py -v
pytest tests/e2e/test_toy_combined_milestones.py -v
```

**Test Categories:**

- **Pipeline Tests**: Dataset → Training → Export → Conversion → Verification
- **Feature Tests**: Code-mode, latent reasoning, combined milestones
- **Efficiency Tests**: Context Efficiency Score (CES) measurement
- **Integration Tests**: Claims framework with model outputs

**CoreML Toy Path:**
Toy tests compile one representative shape by default (T128), with optional multi-shape testing (T64/T128/T256). Production models should export enumerated shapes via the main conversion path.

**Quality Gates:**

- ✅ ≥1 enumerated shape compiles and runs end-to-end
- ✅ No NaN/zero outputs detected
- ✅ Tool span micro-F1 ≥ 0.20 (via deterministic greedy decode)
- ✅ Per-shape diagnostics included in report
- ✅ Feature-specific validation (code-mode eligibility, latent curriculum, etc.)

**See Also:**

- [`docs/TOY_TEST_STRATEGY.md`](docs/TOY_TEST_STRATEGY.md) - Comprehensive toy testing strategy
- [`tests/TOY_TEST_COVERAGE_ANALYSIS.md`](tests/TOY_TEST_COVERAGE_ANALYSIS.md) - Detailed coverage analysis

### Makefile Shortcuts

```bash
make eval-runner-openai    # evaluate with OpenAI-compatible endpoint
make eval-runner-local     # evaluate with local HuggingFace model
make eval-smoke            # smoke test with GPT-4
```

---

## Reproducibility Artifacts

| Directory                                | Contents                       | Fingerprinted By        |
| ---------------------------------------- | ------------------------------ | ----------------------- |
| `models/student/checkpoints/`            | Distillation checkpoints       | SHA-256                 |
| `artifacts/onnx/`                        | Student ONNX graphs (created)  | Model + tokenizer       |
| `arbiter/judge_training/artifacts/onnx/` | Judge ONNX graphs              | Model + tokenizer       |
| `coreml/artifacts/`                      | CoreML mlpackages              | Enumerated shapes       |
| `eval/reports/`                          | JSON evaluation reports        | Dataset · Tool registry |
| `data/`                                  | Generated datasets & artifacts | Content SHA-256         |

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
arbiter/         – CAWS governance stack (judge training · claims · schemas)
build/           – Dataset building and packaging utilities
capture/         – Data capture, normalization, and validation
codemod/         – Code modification and transformation utilities
configs/         – Model configs (worker / judge / drafter / training recipes)
conversion/      – PyTorch → ONNX → CoreML exporters and validators
coreml/          – CoreML-specific tools, runtime, and ANE optimizations
data/            – Datasets, evaluation data, and generated artifacts
docs/            – Documentation and guides
eval/            – Evaluation harness · runners · reports · scoring
evaluation/      – Specialized evaluation utilities (perf, memory, reasoning)
infra/           – Infrastructure utilities and health checks
models/          – Model definitions and pre-built artifacts
onnx/            – ONNX-specific tools and surgery utilities
runtime/         – Runtime orchestration and refinement controllers
schemas/         – JSON schemas and validation definitions
scripts/         – Dataset generation, verification, and utility scripts
tests/           – Unit · integration · e2e · property · CI smoke tests
tools/           – Tool definitions and schema validation
training/        – Training utilities, losses, and distillation
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

kimi-student forms a **self-auditing infrastructure for grounded tool-integration learning**.

It delivers **distilled CoreML models** verified through **CAWS-compliant evaluation**, with constitutional gates enforced in CI and reproducible governance for on-device deployment.

> **Current implementation:**
>
> • Core distillation pipeline functional
> • Basic fixture library implemented
> • CAWS evaluation framework working
>
> **Next milestones:**

> • Expand fixture library (≤ 2 % miss rate)

> • Automate dashboard publishing

> • Publicly release Worker and Judge CoreML artifacts for benchmarking

---

## Documentation

- **Local Development**: [`docs/LOCAL_DEVELOPMENT_WORKFLOW.md`](docs/LOCAL_DEVELOPMENT_WORKFLOW.md) - Progressive local validation tiers
- **Toy Test Strategy**: [`docs/TOY_TEST_STRATEGY.md`](docs/TOY_TEST_STRATEGY.md) - Comprehensive toy testing approach
- **Dataset Card**: [`docs/DATASET_CARD_CONTEXTUAL.md`](docs/DATASET_CARD_CONTEXTUAL.md) - Dataset schema and policies
- **Generation Guide**: [`docs/CONTEXTUAL_DATASET_GENERATION.md`](docs/CONTEXTUAL_DATASET_GENERATION.md) - Full workflow
- **Evaluation Harness**: [`eval/HARNESS.md`](eval/HARNESS.md) - Evaluation harness documentation
- **Scale Tests**: [`docs/SCALE_TESTS.md`](docs/SCALE_TESTS.md) - Scale testing guide
- **CAWS Guide**: [`docs/CAWS_AGENT_GUIDE.md`](docs/CAWS_AGENT_GUIDE.md) - CAWS agent workflow guide
- **Arbiter Theory**: [`docs/ARBITER_THEORY.md`](docs/ARBITER_THEORY.md) - Arbiter stack architecture
- **Distillation Guide**: [`docs/DISTILLATION_GUIDE.md`](docs/DISTILLATION_GUIDE.md) - Distillation guide and rationale
- **Deployment**: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) - Model deployment and runtime configuration
- **Full Docs Index**: [`docs/README.md`](docs/README.md) - Complete documentation index
