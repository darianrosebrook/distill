# Capture

**Why it exists**: Captures the "ground truth" of tool execution during inference, enabling supervised learning of tool-integrated reasoning patterns.

**What's in it**: HTTP proxy server and trace processing utilities that intercept, normalize, and validate tool calls - the data collection system that makes tool-integrated training possible.

**Key Features**:

- HTTP Proxy: Intercepts and logs all tool API calls during inference
- Trace Normalization: Standardizes diverse tool response formats into consistent schemas
- PII Redaction: Automatically removes sensitive data from captured traces
- Schema Validation: Ensures trace completeness and format compliance

## Overview

The capture directory provides the data collection infrastructure that enables learning from tool-integrated reasoning, capturing the complex interactions between models and external tools.

## Key Components

### `proxy_server.py`

**HTTP proxy server** for capturing tool execution traces:

- **Request Interception**: Captures outgoing tool API calls
- **Response Recording**: Logs tool responses and metadata
- **Trace Formatting**: Converts to standardized trace format
- **Error Handling**: Graceful handling of network failures

**Usage**:

```python
from capture.proxy_server import TraceProxy

proxy = TraceProxy(
    host="localhost",
    port=8080,
    trace_output="traces.jsonl"
)

# Start proxy server
proxy.start()

# Tool calls through proxy are automatically captured
```

### `normalize_trace.py`

**Trace normalization** utilities:

- **Format Standardization**: Converts various trace formats to unified schema
- **Data Cleaning**: Removes sensitive information, normalizes timestamps
- **Metadata Enrichment**: Adds context and provenance information
- **Schema Validation**: Ensures compliance with trace schema

**Supported Formats**:

- Raw API responses
- Tool broker fixtures
- Custom trace formats

### `validators.py`

**Trace validation** and quality assurance:

- **Schema Compliance**: Validates against JSON schemas
- **Data Integrity**: Checks for required fields and data types
- **PII Detection**: Identifies and flags sensitive information
- **Completeness**: Ensures all required trace components are present

## Trace Schema

All traces conform to the standard schema:

```json
{
  "name": "tool_name",
  "arguments": { "param": "value" },
  "result": { "status": "success", "data": "..." },
  "timestamp": "2024-01-01T00:00:00Z",
  "duration_ms": 150,
  "metadata": {
    "source": "proxy_server",
    "version": "1.0"
  }
}
```

## Usage Patterns

### During Inference

```python
# Capture traces during model inference
from capture.proxy_server import TraceProxy
from runtime.engine.loop import InferenceEngine

proxy = TraceProxy(trace_output="inference_traces.jsonl")
proxy.start()

engine = InferenceEngine(model=model, tokenizer=tokenizer)
result = engine.generate(prompt="Use tools to solve this")

proxy.stop()
```

### Dataset Generation

```python
# Generate training data with trace capture
from capture.normalize_trace import TraceNormalizer
from scripts.make_kd_mix import DatasetGenerator

normalizer = TraceNormalizer()
generator = DatasetGenerator(trace_normalizer=normalizer)

dataset = generator.generate_from_prompts(prompts, capture_traces=True)
```

### Validation Pipeline

```python
# Validate captured traces
from capture.validators import TraceValidator

validator = TraceValidator()
results = validator.validate_batch(trace_files)

for result in results:
    if not result.valid:
        print(f"Invalid trace: {result.errors}")
```

## Integration Points

### With Runtime

- **Inference Engine**: `runtime/engine/loop.py` integrates trace capture
- **Tool Broker**: `eval/tool_broker/` provides deterministic replay
- **Orchestration**: `runtime/orchestration/` coordinates trace collection

### With Scripts

- **Dataset Generation**: `scripts/make_kd_mix.py` captures traces during generation
- **Processing**: `scripts/extract_process_targets.py` uses traces for supervision

### With Evaluation

- **Trace Replay**: `eval/tool_broker/fixtures/` uses captured traces
- **Validation**: `eval/scoring/` validates trace compliance

## Development

### Adding New Trace Sources

1. Implement trace capture logic
2. Add normalization for new format
3. Update validators for new schema
4. Add integration tests

### Testing

```bash
# Test trace capture
python -m pytest tests/test_trace_capture.py -v

# Test normalization
python -m pytest tests/test_trace_normalization.py -v

# Test validation
python -m pytest tests/test_trace_validation.py -v
```

## Security Considerations

- **PII Redaction**: Automatically removes sensitive data from traces
- **Access Control**: Proxy server authentication and authorization
- **Data Sanitization**: Cleans traces before storage or transmission
- **Audit Logging**: Comprehensive logging of trace operations

## See Also

- [`schemas/README.md`](../schemas/README.md) - Trace schema definitions
- [`eval/tool_broker/`](../eval/tool_broker/) - Tool broker integration
- [`runtime/engine/loop.py`](../runtime/engine/loop.py) - Inference engine
- [`docs/CONTEXTUAL_DATASET_GENERATION.md`](../docs/CONTEXTUAL_DATASET_GENERATION.md) - Dataset generation guide
- [`docs/`](../docs/) - Main project documentation
