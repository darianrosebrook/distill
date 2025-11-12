# kimi-student: Multi-Model Portfolio for CAWS Arbiter Stack

Distilled student models for specialized roles in CAWS-compliant multi-model orchestration, optimized for Apple Silicon (M-series) with ANE acceleration via CoreML.

## Model Portfolio

### Worker (~9B GQA)

- **Role**: Primary generator for code edits, tool-use JSON, long-context retrieval
- **Target**: Best quality/latency envelope on 64GB for 8-16k context
- **Precision**: INT8 weights + FP16 activations
- **Config**: `configs/worker_9b.yaml`

### Judge (3-4B or 7B)

- **Role**: Constitutional arbiter for CAWS compliance
- **Target**: Fast, local decision-making for adjudication cycles
- **Precision**: INT8 weights + FP16 activations
- **Config**: `configs/judge_4b.yaml`
- **Export**: Short enumerated shapes (512/1024/2048) for summaries/claims

### Drafter (~4B, optional)

- **Role**: Speculative decoding for sub-second TTFA
- **Target**: Fast token generation, verified by Worker/Judge
- **Config**: `configs/drafter_4b.yaml`

## Getting Started

### Stage 1: Distillation MVP (Current Focus)

**Goal**: Distill K2-Thinking into ANE-friendly student that runs on M1 Max.

1. Create a Python 3.10+ env and install deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

2. Build KD dataset:

```bash
# Option A: Use Kimi K2 Thinking API (recommended for M1 Max)
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher https://api.kimi.com/v1

# Option B: Use local HTTP endpoint
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher http://localhost:8000

# Option C: Use HuggingFace model (requires GPU with 80GB+ VRAM)
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher hf:moonshotai/Kimi-K2-Thinking
```

**Note**: Kimi K2 Thinking (32B active params) cannot run locally on M1 Max. See `docs/external/KIMI_K2_SETUP.md` for API setup.

3. Train Worker model (8-9B GQA):

```bash
make worker
make onnx-worker
make coreml-worker
```

4. Validate with probes and perf gates:

```bash
make probes
make eval
```

5. Add process supervision and QAT:

```bash
make proc  # Process supervision
make qat   # Quantization-aware training
```

### Stage 2: Governance (After MVP)

**Goal**: Add CAWS governance and runtime enforcement.

6. Train Judge model (after student passes gates):

```bash
make judge_train
make judge_onnx
make judge_coreml
```

7. Evaluate CAWS compliance:

```bash
make caws-eval
```

See `docs/DISTILLATION_GUIDE.md` for detailed distillation guide and rationale.

## Contextual Dataset Generation & Evaluation

The project includes a comprehensive pipeline for generating and evaluating contextual datasets with process-step supervision targets for tool-use training.

### Quick Start

**Generate and verify a contextual dataset:**

```bash
# Full pipeline: generate → extract → verify
make contextual-pipeline

# Or step-by-step:
make contextual-gen          # Generate prompts
make contextual-extract      # Extract process targets
make contextual-verify       # Verify quality
```

**Scale tests (N=1k, N=10k):**

```bash
make gen-scale-1k           # Generate 1k samples
make verify-scale-1k         # Verify 1k dataset

make gen-scale-10k           # Generate 10k samples (sharded)
make verify-scale-10k        # Verify 10k dataset
```

**Evaluate model performance:**

```bash
# OpenAI-compatible endpoint
make eval-runner-openai \
  MODEL="gpt-4" \
  IN="data/contextual_extracted.jsonl" \
  OUT="eval/results.jsonl" \
  REPORT="eval/report.json"

# Local HuggingFace model
make eval-runner-local \
  MODEL="/path/to/model" \
  IN="data/contextual_extracted.jsonl" \
  OUT="eval/results.jsonl" \
  REPORT="eval/report.json"
```

### Features

- **Stratified generation**: Coverage across scenarios × complexity × structure
- **Process-step targets**: Tool names, JSON arguments, and integration spans with byte/token offsets
- **Comprehensive verification**: Integration F1 (lax/strict), token alignment, control contamination checks
- **Deterministic evaluation**: Tool broker replays fixtures (no live network calls)
- **Multi-runner support**: OpenAI HTTP and HuggingFace local runners
- **Scale testing**: N=1k and N=10k with deterministic sharding

### Documentation

- **Dataset Card**: `docs/DATASET_CARD_CONTEXTUAL.md` - Dataset schema and policies
- **Generation Guide**: `docs/CONTEXTUAL_DATASET_GENERATION.md` - Full workflow
- **Evaluation Harness**: `eval/HARNESS.md` - Evaluation harness documentation
- **Scale Tests**: `docs/SCALE_TESTS.md` - Scale testing guide
- **CAWS Guide**: `docs/CAWS_AGENT_GUIDE.md` - CAWS agent workflow guide
- **Arbiter Theory**: `docs/ARBITER_THEORY.md` - Arbiter stack architecture
- **Full Docs Index**: `docs/README.md` - Complete documentation index

## Directory Structure

```
arbiter/
├── judge_training/     # Pairwise ranking, clause labeling
├── claimify/           # 4-stage claim extraction pipeline
├── schemas/            # CAWS verdict, waiver, evidence schemas
├── eval/               # CAWS-specific evaluation metrics
└── swift/              # Swift bridge for CoreML judge runtime

configs/
├── worker_9b.yaml      # Worker model config
├── judge_4b.yaml       # Judge model config
└── drafter_4b.yaml     # Drafter model config (optional)

conversion/
├── export_onnx.py         # Worker ONNX export
├── export_pytorch.py      # PyTorch export (production path)
├── convert_coreml.py      # CoreML conversion (PyTorch/ONNX backends)
├── judge_export_onnx.py  # Judge ONNX export
└── judge_export_coreml.py # Judge CoreML export

scripts/
├── generate_contextual_prompts.py  # Contextual prompt generation
├── extract_process_targets.py      # Process-step target extraction
└── verify_contextual_set.py        # Dataset quality verification

eval/
├── cli.py                  # Evaluation harness CLI
├── runners/                # Model runners (OpenAI HTTP, HF local)
├── tool_broker/           # Deterministic tool result replay
├── scoring/                # Verifier-parity scoring
└── reports/                # Report summarization

tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── property/               # Property-based tests (Hypothesis)
└── ci/                     # CI smoke tests
```

## CAWS Integration

The arbiter stack enforces CAWS governance through:

- **Claimify Pipeline**: 4-stage claim extraction and verification
- **CAWS Schemas**: Verdict, waiver, and evidence manifest schemas
- **Judge Training**: Pairwise ranking + clause labeling for constitutional arbitration
- **CAWS Evaluation**: End-to-end compliance checking

See `arbiter/README.md` for detailed arbiter stack documentation.

## Artifacts

- Model checkpoints: `models/student/checkpoints/`
- ONNX exports: `artifacts/onnx/` (worker) and `arbiter/judge_training/artifacts/onnx/` (judge)
- CoreML models: `coreml/artifacts/` (worker and judge)
