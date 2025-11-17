# Models

Model architectures and tokenizer configurations.

## Overview

This directory contains:
- **Student Model**: GQA transformer architecture
- **Tokenizer**: Tokenizer configuration and special tokens
- **Model Configs**: Architecture configuration utilities

## Student Model

### Architecture: GQA Transformer

The student model uses Grouped Query Attention (GQA) for efficiency:
- **Worker**: ~9B parameters, 8-16k context
- **Judge**: 3-4B or 7B parameters, 512-2k context
- **Drafter**: ~4B parameters, ≤2k context

### Key Files

- `student/architectures/gqa_transformer.py` - Main model implementation
- `student/tokenizer/` - Tokenizer configuration
- `student/tokenizer/constants.py` - Special token constants (BOT, EOT)

## Model Configuration

Models are configured via YAML files in `configs/`:
- `worker_9b.yaml` - Worker model config
- `judge_4b.yaml` - Judge model config
- `drafter_4b.yaml` - Drafter model config

## Tokenizer

Tokenizer is located at `models/student/tokenizer/`:
- HuggingFace format tokenizer
- Special tokens: `<bot>`, `<eot>`
- Token IDs defined in `constants.py`

**Special Tokens**:
- `BOT_TOKEN`: Beginning of tool call
- `EOT_TOKEN`: End of tool call

## See Also

- [`configs/README.md`](../configs/README.md) - Configuration documentation
- [`training/README.md`](../training/README.md) - Training documentation











