# Classification Model Evaluation Framework

This directory contains general-purpose tools for evaluating **classification models** and verifying that class distributions are preserved through the conversion pipeline.

## Overview

The framework evaluates models as N-way classifiers where:

- **Input**: Natural language text
- **Output**: One of N predefined class labels (mapped to specific token IDs)
- **Evaluation**: Verify that predictions are preserved across conversion stages (PyTorch → CoreML → GGUF → Ollama)

## Architecture

- **Configurable Classification**: Define any classification task via `ClassificationConfig`
- **Multi-Backend Support**: PyTorch, CoreML, and Ollama evaluation
- **Pipeline Preservation**: Compare predictions across conversion stages
- **Extensible**: Easy to add new classification tasks

## Adding a New Classification Task

1. **Create a config module** in `evaluation/toy/` or your preferred location:

```python
from evaluation.classification_eval import ClassificationConfig

# Define your classes
MY_CLASSES = ["class_a", "class_b", "class_c"]
MY_TOKEN_IDS = [300, 301, 302]  # Choose unused token IDs

MY_CONFIG = ClassificationConfig(
    name="my_classifier",
    class_names=MY_CLASSES,
    token_ids=MY_TOKEN_IDS,
    id_to_name={tid: name for tid, name in zip(MY_TOKEN_IDS, MY_CLASSES)},
    name_to_id={name: tid for tid, name in zip(MY_TOKEN_IDS, MY_CLASSES)},
)
```

2. **Add evaluation questions** (optional):

```python
def get_my_questions():
    return [
        "Sample question 1?",
        "Sample question 2?",
        # ...
    ]
```

3. **Use with evaluation tools**:

```bash
# Single backend evaluation
python evaluation/classification_eval.py \
    --backend pytorch \
    --model /path/to/model \
    --tokenizer /path/to/tokenizer \
    --config your.module.MY_CONFIG

# Pipeline comparison
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model /path/to/pytorch \
    --ollama-model my-model \
    --tokenizer /path/to/tokenizer \
    --config your.module.MY_CONFIG
```

## Example: 8-Ball Classifier

The 8-ball model demonstrates the framework with a 20-class classification task:

| Token ID | Answer             |
| -------- | ------------------ |
| 200      | It is certain      |
| 201      | It is decidedly so |
| ...      | ...                |
| 219      | Very doubtful      |

**Configuration**: `evaluation.toy.eight_ball.EIGHT_BALL_CONFIG`

**Usage**:

```bash
python evaluation/classification_eval.py \
    --backend ollama \
    --model 8-ball \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG
```

## Tools

### `classification_eval.py`

Main evaluation script for classification models across different backends:

```bash
# Evaluate PyTorch model
python evaluation/classification_eval.py \
    --backend pytorch \
    --model /path/to/model \
    --tokenizer /path/to/tokenizer \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG \
    --output pytorch_predictions.json

# Evaluate CoreML model
python evaluation/classification_eval.py \
    --backend coreml \
    --model /path/to/model.mlpackage \
    --tokenizer /path/to/tokenizer \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG \
    --output coreml_predictions.json

# Evaluate Ollama model
python evaluation/classification_eval.py \
    --backend ollama \
    --model model-name \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG \
    --output ollama_predictions.json
```

### `pipeline_preservation_eval.py`

Compare predictions across pipeline stages to verify preservation:

```bash
python evaluation/pipeline_preservation_eval.py \
    --pytorch-model /path/to/pytorch \
    --coreml-model /path/to/coreml \
    --ollama-model model-name \
    --tokenizer /path/to/tokenizer \
    --config evaluation.toy.eight_ball.EIGHT_BALL_CONFIG \
    --output-dir /tmp/pipeline_comparison
```

This will:

1. Evaluate each backend
2. Compare predictions against PyTorch (reference)
3. Calculate metrics:
   - Exact match rate
   - Mean L2 drift (probability distribution)
   - Mean KL divergence

### `format_8ball_output.py`

Format Ollama output by mapping token IDs to human-readable answers:

```bash
ollama run 8-ball "Should I go to the doctor?" | python scripts/format_8ball_output.py
```

## Metrics

The evaluation framework tracks:

1. **Exact Match Rate**: Percentage of questions where predicted class ID matches reference
2. **Mean L2 Drift**: Average L2 distance between probability distributions
3. **Mean KL Divergence**: Average KL divergence between probability distributions

## Toy Model Evaluation

For toy-specific evaluation patterns and configurations, see [`toys/README.md`](../toys/README.md).

This includes:

- 8-ball classifier evaluation
- Future binary classifier patterns
- Toy model training and evaluation workflows
