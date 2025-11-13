# Toy Model Evaluation Patterns

This directory contains toy model configurations and evaluation patterns for testing the distillation pipeline. Toy models enable fast iteration during development while ensuring production-ready integration.

## Overview

Toy models are lightweight test doubles that validate the complete distillation pipeline without expensive compute. They demonstrate different evaluation patterns:

- **Pipeline**: Basic distillation pipeline validation (used in e2e integration tests)
- **8-Ball Classifier**: 20-class classification for mystical fortune-telling
- **Binary Classifier**: YES/NO decisions for "should we proceed?" questions
- **Ternary Classifier**: YES/NO/UNCERTAIN decisions with uncertainty handling

## Pipeline

The pipeline toy validates the basic distillation pipeline without specialized classification logic. Used primarily for end-to-end integration testing (`tests/e2e/test_toy_pipeline.py`).

### Directory Structure

**Location**: `toys/pipeline/`

### Training

```bash
# Generate pipeline dataset (no special flags)
python -m data.make_toy_kd --out pipeline_data.jsonl --n 128

# Train pipeline model (no special flags)
python training/run_toy_distill.py \
    --in pipeline_data.jsonl \
    --out pipeline_model.pt \
    --vocab-size 512
```

### Use Cases

- **E2E Pipeline Testing**: Validates complete distillation flow
- **CI/CD Integration**: Fast validation of pipeline changes
- **Development Testing**: Quick iteration during development

## 8-Ball Classifier

The 8-ball model treats fortune-telling as a **20-class classification task** where the model predicts one of 20 classic 8-ball answers.

### Configuration

**Location**: `evaluation/toy/eight_ball.py`

**Token IDs**: 200-219 (20 answers)

| Token ID | Answer                    |
| -------- | ------------------------- |
| 200      | It is certain             |
| 201      | It is decidedly so        |
| 202      | Without a doubt           |
| 203      | Yes definitely            |
| 204      | You may rely on it        |
| 205      | As I see it, yes          |
| 206      | Most likely               |
| 207      | Outlook good              |
| 208      | Yes                       |
| 209      | Signs point to yes        |
| 210      | Reply hazy, try again     |
| 211      | Ask again later           |
| 212      | Better not tell you now   |
| 213      | Cannot predict now        |
| 214      | Concentrate and ask again |
| 215      | Don't count on it         |
| 216      | My reply is no            |
| 217      | My sources say no         |
| 218      | Outlook not so good       |
| 219      | Very doubtful             |

### Evaluation

```bash
# Single backend evaluation
python evaluation/classification_eval.py \
    --backend ollama \
    --model 8-ball \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG

# Pipeline preservation test
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model ~/8_ball_hf \
    --ollama-model 8-ball \
    --tokenizer ~/8_ball_hf \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG
```

### Training Integration

The teacher stub (`training/teacher_stub_toy.py`) prefers 8-ball answer tokens when `vocab_size >= 220`, making the model learn classification targets.

### Output Formatting

Format Ollama output to human-readable answers:

```bash
ollama run 8-ball "Should I go to the doctor?" | python scripts/format_8ball_output.py
# Output: "Don't count on it"
```

## Binary Classifier

The binary classifier outputs YES/NO decisions for "should we proceed?" questions based on provided evidence.

### Configuration

**Location**: `evaluation/toy/binary_classifier.py`

**Token IDs**: 300 (YES), 301 (NO)

**Answers**: ["YES", "NO"] - should proceed or not

### Data Format

Binary classifier data uses a structured format with evidence and questions:

```
EVIDENCE: <description of situation> QUESTION: Should we proceed? ANSWER (YES or NO):
```

**Target**: "YES" or "NO" based on evidence analysis

### Training

```bash
# Generate binary classifier dataset
python -m data.make_toy_kd --out binary_data.jsonl --n 128 --binary-classifier

# Train binary classifier
python training/run_toy_distill.py \
    --in binary_data.jsonl \
    --out models/binary_classifier.pt \
    --binary-classifier \
    --vocab-size 512
```

### Evaluation

```bash
# Evaluate binary classifier
python evaluation/classification_eval.py \
    --backend pytorch \
    --model models/binary_classifier.pt \
    --tokenizer models/student/tokenizer \
    --config evaluation.toy.binary_classifier.BINARY_CLASSIFIER_CONFIG

# Pipeline preservation test
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model models/binary_classifier.pt \
    --ollama-model binary-classifier \
    --tokenizer models/student/tokenizer \
    --config evaluation.toy.binary_classifier.BINARY_CLASSIFIER_CONFIG
```

### Output Formatting

Format Ollama output to human-readable decisions:

```bash
ollama run binary-classifier "EVIDENCE: All tests pass. QUESTION: Should we proceed? ANSWER (YES or NO):" | python scripts/format_binary_output.py
# Output: "YES - You should proceed"
```

### Use Cases

- **Deployment Gates**: Should we deploy this version?
- **Code Review**: Is this PR ready to merge?
- **Quality Checks**: Does this meet our standards?
- **Risk Assessment**: Should we proceed with this change?

## Ternary Classifier

The ternary classifier outputs YES/NO/UNCERTAIN decisions for "should we proceed?" questions, providing an additional "insufficient evidence" option for cases where the decision cannot be clearly determined.

### Configuration

**Location**: `evaluation/toy/ternary_classifier.py`

**Token IDs**: 400 (YES), 401 (NO), 402 (UNCERTAIN)

**Answers**: ["YES", "NO", "UNCERTAIN"] - should proceed, should not proceed, or insufficient evidence

### Data Format

Ternary classifier data uses a structured format with evidence and questions:

```
EVIDENCE: <description of situation> QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):
```

**Target**: "YES", "NO", or "UNCERTAIN" based on evidence analysis

### Training

```bash
# Generate ternary classifier dataset
python -m data.make_toy_kd --out ternary_data.jsonl --n 128 --ternary-classifier

# Train ternary classifier
python training/run_toy_distill.py \
    --in ternary_data.jsonl \
    --out models/ternary_classifier.pt \
    --ternary-classifier \
    --vocab-size 512
```

### Evaluation

```bash
# Evaluate ternary classifier
python evaluation/classification_eval.py \
    --backend pytorch \
    --model models/ternary_classifier.pt \
    --tokenizer models/student/tokenizer \
    --config evaluation.toy.ternary_classifier.TERNARY_CLASSIFIER_CONFIG

# Pipeline preservation test
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model models/ternary_classifier.pt \
    --ollama-model ternary-classifier \
    --tokenizer models/student/tokenizer \
    --config evaluation.toy.ternary_classifier.TERNARY_CLASSIFIER_CONFIG
```

### Output Formatting

Format Ollama output to human-readable decisions:

```bash
ollama run ternary-classifier "EVIDENCE: All tests pass. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):" | python scripts/format_ternary_output.py
# Output: "YES - You should proceed"
```

### Use Cases

- **High-Stakes Decisions**: Where uncertainty should be explicitly acknowledged
- **Medical/Financial**: Areas requiring clear evidence thresholds
- **Graduated Risk**: Different confidence levels for different actions
- **Regulatory Compliance**: Where "insufficient evidence" is a valid outcome

### Configuration Template

```python
# evaluation/toy/binary_classifier.py
from evaluation.classification_eval import ClassificationConfig

BINARY_ANSWERS = ["revisit", "proceed"]  # or ["no", "yes"]
BINARY_TOKEN_IDS = [300, 301]

BINARY_CONFIG = ClassificationConfig(
    name="binary-classifier",
    class_names=BINARY_ANSWERS,
    token_ids=BINARY_TOKEN_IDS,
    id_to_name={tid: name for tid, name in zip(BINARY_TOKEN_IDS, BINARY_ANSWERS)},
    name_to_id={name: tid for tid, name in zip(BINARY_TOKEN_IDS, BINARY_ANSWERS)},
)
```

### Use Cases

- **Code Review**: "Should this PR be merged?" → "revisit" or "proceed"
- **Task Triage**: "Does this need immediate attention?" → "yes" or "no"
- **Quality Gates**: "Does this pass the bar?" → "revisit" or "proceed"

### Evaluation Pattern

```bash
python evaluation/classification_eval.py \
    --backend pytorch \
    --model /path/to/binary_model \
    --tokenizer /path/to/tokenizer \
    --config evaluation.toy.binary_classifier.BINARY_CONFIG
```

## Adding New Toy Classifiers

1. **Create config** in `evaluation/toy/your_classifier.py` (see `ternary_classifier.py` as example)
2. **Choose token IDs** that don't conflict (avoid 200-219 for 8-ball, 300-301 for binary, 400-402 for ternary)
3. **Define classes** and evaluation questions
4. **Update teacher stub** to prefer your token IDs during training
5. **Test pipeline** preservation

## Implementation Details

### ClassificationConfig

```python
@dataclass
class ClassificationConfig:
    name: str                    # Classifier name
    class_names: List[str]       # Human-readable class names
    token_ids: List[int]         # Token IDs for each class
    id_to_name: Dict[int, str]   # Token ID → class name mapping
    name_to_id: Dict[str, int]   # Class name → token ID mapping
```

### Evaluation Flow

1. **Tokenize input** with model's tokenizer
2. **Run inference** to get logits
3. **Extract class probabilities** from configured token IDs
4. **Predict class** with argmax
5. **Map to human-readable** name via config

### Pipeline Preservation

Compare predictions across stages:

- PyTorch FP32 (reference)
- CoreML export
- GGUF conversion + Ollama

Metrics:

- Exact match rate
- Probability distribution drift (L2/KL)

## Integration with Main Framework

Toy classifiers use the same evaluation infrastructure as production models:

- **General evaluation**: `evaluation/classification_eval.py`
- **Pipeline comparison**: `evaluation/pipeline_preservation_eval.py`
- **Training integration**: Teacher stubs prefer classifier tokens

## Testing the Pipeline

```bash
# Train toy model
make toy-e2e

# Convert to different formats
python scripts/convert_to_gguf.py --checkpoint /tmp/8_ball.ckpt --out 8-ball.gguf
ollama create 8-ball -f Modelfile.8-ball

# Evaluate across backends
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model ~/8_ball_hf \
    --ollama-model 8-ball \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG
```

## See Also

- [`evaluation/README.md`](../evaluation/README.md) - General evaluation framework
- [`training/teacher_stub_toy.py`](../training/teacher_stub_toy.py) - Toy training integration
- [`scripts/format_8ball_output.py`](../scripts/format_8ball_output.py) - Output formatting
