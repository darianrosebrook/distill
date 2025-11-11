# JudgeBridge - CoreML Judge Swift Bridge

Swift Package providing a dynamic library and HTTP server for CoreML judge inference.

## Components

- **JudgeBridge Library**: Dynamic library with C FFI for CoreML judge (`encode_once`)
- **JudgeServer**: SwiftNIO HTTP server with `/score` and `/compare` endpoints
- **Calibration**: Optional Platt/temperature calibration + clause thresholds

## Building

```bash
cd arbiter/swift/JudgeBridge
swift build
```

## Running JudgeServer

```bash
swift run JudgeServer \
  --model arbiter/judge_training/artifacts/coreml/judge.mlpackage \
  --seq-len 512 \
  --clauses 5 \
  [--calibration calibration.json] \
  [--host 127.0.0.1] \
  [--port 8088]
```

## API Endpoints

### POST /score
Score a single candidate:
```json
{
  "input_ids_a": [1, 2, 3, ...],
  "attention_mask_a": [1, 1, 1, ...],
  "token_type_ids_a": [0, 0, 0, ...]
}
```

### POST /compare
Compare two candidates:
```json
{
  "input_ids_a": [...],
  "attention_mask_a": [...],
  "token_type_ids_a": [...],
  "input_ids_b": [...],
  "attention_mask_b": [...],
  "token_type_ids_b": [...]
}
```

### GET /health
Health check endpoint.

## C FFI Usage

See `Sources/JudgeBridge/include/JudgeBridge.h` for C API:

```c
void* handle = judge_create(model_path, seq_len, n_clauses, calibration_path);
float score;
float clause_probs[5];
judge_encode_once(handle, input_ids, attention_mask, token_type_ids, length, &score, clause_probs, 5);
judge_destroy(handle);
```

