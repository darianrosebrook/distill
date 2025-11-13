# Schemas

**Why it exists**: Prevents data corruption and integration bugs by enforcing consistent data formats across the entire distillation pipeline - from dataset generation to model inference.

**What's in it**: JSON Schema definitions that validate everything from tool traces to evaluation results, ensuring type safety and data integrity without runtime type checking.

**Key Features**:

- Trace Validation: Ensures tool execution traces are complete and well-formed
- API Contracts: Standardized interfaces between pipeline components
- Data Integrity: Catches format errors before they cause training/inference failures
- Cross-Platform: Works across Python, JavaScript, and other environments

## Overview

The schemas directory provides the "type system" for the distillation pipeline, preventing subtle bugs that arise from malformed data or API mismatches.

## Schema Files

### `trace.schema.json`

**Tool trace validation schema** for captured execution traces:

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "arguments": { "type": "object" },
    "result": { "type": "object" },
    "timestamp": { "type": "string", "format": "date-time" },
    "duration_ms": { "type": "number" }
  },
  "required": ["name", "arguments", "result"]
}
```

**Validates**:

- Tool execution traces from `capture/`
- Tool broker fixtures in `eval/tool_broker/`
- Dataset generation tool calls

## Usage

### Validation

```python
import jsonschema
from schemas.trace import TRACE_SCHEMA

# Validate tool trace
jsonschema.validate(trace_data, TRACE_SCHEMA)
```

### In Scripts

Schemas are used throughout the codebase:

- **Dataset Generation**: `scripts/make_kd_mix.py`
- **Tool Validation**: `tools/schema_registry.py`
- **Trace Processing**: `capture/validators.py`
- **Evaluation**: `eval/scoring/`

## Schema Standards

All schemas follow:

- **JSON Schema Draft 2020-12**
- **Descriptive error messages**
- **Comprehensive validation rules**
- **Backwards compatibility**

## Development

### Adding New Schemas

1. Create schema file: `new_feature.schema.json`
2. Add validation logic to relevant modules
3. Update this README
4. Add tests in `tests/test_schema_validation.py`

### Schema Testing

```bash
# Validate all schemas
python -m pytest tests/ -k schema

# Test specific schema
python -c "import jsonschema; jsonschema.validate(data, schema)"
```

## Integration Points

### With Tools

- **Schema Registry**: `tools/schema_registry.py` loads and validates schemas
- **Tool Validation**: Runtime validation of tool arguments/results

### With Evaluation

- **Trace Validation**: `eval/scoring/` validates captured traces
- **Fixture Validation**: Tool broker fixtures conform to schemas

### With Capture

- **Trace Normalization**: `capture/normalize_trace.py` uses schemas
- **Validation**: `capture/validators.py` enforces schema compliance

## See Also

- [`arbiter/schemas/`](../arbiter/schemas/) - CAWS governance schemas
- [`eval/schemas/`](../eval/schemas/) - Evaluation-specific schemas
- [`tools/schema_registry.py`](../tools/schema_registry.py) - Schema loading utilities
- [`capture/validators.py`](../capture/validators.py) - Validation utilities
