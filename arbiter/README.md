# Arbiter Stack for CAWS-Compliant Multi-Model Orchestration

This directory contains the scaffolding for training and evaluating specialized models that work together under CAWS governance.

## Model Portfolio

### Worker (~9B GQA)
- **Role**: Primary generator for code edits, tool-use JSON, long-context retrieval
- **Target**: Best quality/latency envelope on 64GB for 8-16k context
- **Precision**: INT8 weights + FP16 activations
- **Training**: Standard KD + process supervision + long-context curriculum + QAT
- **Config**: `configs/worker_9b.yaml`

### Judge (3-4B or 7B)
- **Role**: Constitutional arbiter for CAWS compliance
- **Target**: Fast, local decision-making for adjudication cycles
- **Precision**: INT8 weights + FP16 activations
- **Training**: Pairwise ranking + clause labeling + constitutional tuning
- **Config**: `configs/judge_4b.yaml`
- **Export**: Short enumerated shapes (512/1024/2048) for summaries/claims

### Drafter (~4B, optional)
- **Role**: Speculative decoding for sub-second TTFA
- **Target**: Fast token generation, verified by Worker/Judge
- **Config**: `configs/drafter_4b.yaml`

## Directory Structure

```
arbiter/
├── judge_training/     # Pairwise ranking, clause labeling training
├── claimify/           # 4-stage claim extraction pipeline
├── schemas/            # CAWS verdict, waiver, evidence schemas
└── eval/               # CAWS-specific evaluation metrics
```

## Training Workflow

1. **Judge First** (de-risk quickly):
   ```bash
   make judge
   make onnx-judge
   make coreml-judge
   ```

2. **Worker Pilot**:
   ```bash
   make worker
   make onnx-worker
   make coreml-worker
   ```

3. **CAWS Evaluation**:
   ```bash
   make caws-eval
   ```

## Claimify Pipeline

The 4-stage claim extraction pipeline (`arbiter/claimify/pipeline.py`):

1. **Contextual Disambiguation**: Resolve ambiguities before extraction
2. **Verifiable Content Qualification**: Filter to objectively checkable content
3. **Atomic Claim Decomposition**: Extract atomic, verifiable claims
4. **CAWS-Compliant Verification**: Verify against evidence manifests

## CAWS Schemas

- `schemas/caws_verdict.schema.json`: Verdict records for adjudication
- `schemas/waiver.schema.json`: Waiver records for exceptional circumstances
- `schemas/evidence_manifest.schema.json`: Evidence manifests for claim verification

## Evaluation Metrics

- **Pairwise Accuracy**: Judge ranking accuracy on A vs B comparisons
- **Clause F1**: CAWS clause mapping F1 score
- **Claim P/R/F1**: Claim extraction and verification precision/recall/F1

See `arbiter/eval/caws_metrics.py` for implementation.

