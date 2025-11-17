# Tools

Tool registry and schema definitions for tool-integration evaluation.

## Overview

This directory contains:
- **Tool Registry**: Central registry of available tools
- **Schema Definitions**: Tool argument schemas
- **Validation**: Tool argument validation utilities

## Key Files

- `schema_registry.py` - Central tool registry
- Tool schemas and validation logic

## Tool Registry

The tool registry defines:
- Available tool names
- Tool argument schemas (JSON Schema)
- Tool validation rules

Tools are used throughout the pipeline:
- Dataset generation (tool call extraction)
- Training (process-step supervision)
- Evaluation (tool call validation)

## See Also

- [`eval/tool_broker/`](../eval/tool_broker/) - Tool broker for deterministic replay
- [`docs/CONTEXTUAL_DATASET_GENERATION.md`](../docs/CONTEXTUAL_DATASET_GENERATION.md) - Tool integration in datasets











