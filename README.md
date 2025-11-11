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

1) Create a Python 3.10+ env and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

2) Build datasets:
```bash
python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher http://localhost:8000
```

3) Train models (recommended order):
```bash
# Judge first (de-risk quickly)
make judge
make onnx-judge
make coreml-judge

# Worker pilot
make worker
make onnx-worker
make coreml-worker

# Optional: Drafter for latency
make drafter
```

4) Evaluate:
```bash
make eval  # Includes CAWS evaluation
make caws-eval  # CAWS-specific evaluation only
```

## Directory Structure

```
arbiter/
├── judge_training/     # Pairwise ranking, clause labeling
├── claimify/           # 4-stage claim extraction pipeline
├── schemas/            # CAWS verdict, waiver, evidence schemas
└── eval/               # CAWS-specific evaluation metrics

configs/
├── worker_9b.yaml      # Worker model config
├── judge_4b.yaml       # Judge model config
└── drafter_4b.yaml     # Drafter model config (optional)

conversion/
├── export_onnx.py      # Worker ONNX export
├── judge_export_onnx.py    # Judge ONNX export
└── judge_export_coreml.py  # Judge CoreML export
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
- ONNX exports: `artifacts/onnx/` (worker) and `artifacts/onnx/judge/` (judge)
- CoreML models: `coreml/`
