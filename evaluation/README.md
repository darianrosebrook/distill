# Evaluation Scripts

High-level evaluation scripts for performance, memory, and CAWS compliance.

## Overview

This directory contains evaluation scripts that build on the `eval/` harness:
- **Performance Evaluation**: Speed and memory benchmarks
- **CAWS Evaluation**: CAWS compliance checking
- **Tool Use Evaluation**: Tool integration evaluation
- **Long Context Evaluation**: Long-context performance

## Key Scripts

- `perf_mem_eval.py` - Performance and memory evaluation
- `caws_eval.py` - CAWS compliance evaluation
- `tool_use_eval.py` - Tool use evaluation
- `long_ctx_eval.py` - Long-context evaluation
- `toy_contracts.py` - Toy model contract validation

## Usage

Run evaluations:
```bash
make eval  # Run all evaluations
```

## See Also

- [`eval/README.md`](../eval/README.md) - Evaluation harness documentation
- [`docs/ACCEPTANCE_GATES.md`](../docs/ACCEPTANCE_GATES.md) - Quality gates

