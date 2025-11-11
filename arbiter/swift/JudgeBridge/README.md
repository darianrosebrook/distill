# JudgeBridge - CoreML Judge Swift Bridge

Swift Package providing a dynamic library and HTTP server for CoreML judge inference.

## Components

- **JudgeBridge Library**: Dynamic library with C FFI for CoreML judge (`encode_once`)
- **JudgeServer**: SwiftNIO HTTP server with `/score` and `/compare` endpoints
- **Calibration**: Optional Platt/temperature calibration + clause thresholds

## Building

From the Swift package root:

```bash
cd arbiter/swift/JudgeBridge
swift build -c release
```

Outputs:
- Dynamic library: `.build/release/libJudgeBridge.dylib`
- Server executable: `.build/release/JudgeServer`

## Running JudgeServer

```bash
./.build/release/JudgeServer \
  --model arbiter/judge_training/artifacts/coreml/judge.mlpackage \
  --seq-len 512 \
  --clauses 5 \
  [--calibration calibration.json] \
  [--host 127.0.0.1] \
  [--port 8088]
```

### Calibration JSON Format

Optional calibration file (`calibration.json`):

```json
{
  "score_platt": {
    "a": 1.2,
    "b": -0.5
  },
  "clause_temperature": 1.5,
  "clause_thresholds": [0.7, 0.6, 0.8, 0.65, 0.75]
}
```

## HTTP API

### POST /score

Score a single candidate (arrays are pre-tokenized int32):

```bash
curl -X POST localhost:8088 \
  -H 'content-type: application/json' \
  -d '{
    "input_ids_a": [101, 102, 103, ...],
    "attention_mask_a": [1, 1, 1, ...],
    "token_type_ids_a": [0, 0, 0, ...]
  }'
```

Response:
```json
{
  "score": 0.85,
  "clause_probs": [0.9, 0.7, 0.8, 0.6, 0.75]
}
```

### POST /compare

Compare two candidates:

```bash
curl -X POST localhost:8088 \
  -H 'content-type: application/json' \
  -d '{
    "input_ids_a": [...],
    "attention_mask_a": [...],
    "token_type_ids_a": [...],
    "input_ids_b": [...],
    "attention_mask_b": [...],
    "token_type_ids_b": [...]
  }'
```

Response:
```json
{
  "verdict": "A",
  "A": {
    "score": 0.85,
    "clause_probs": [0.9, 0.7, 0.8, 0.6, 0.75]
  },
  "B": {
    "score": 0.62,
    "clause_probs": [0.5, 0.6, 0.4, 0.5, 0.55]
  }
}
```

### GET /health

Health check endpoint:

```bash
curl http://localhost:8088/health
```

Response:
```json
{"ok": true}
```

## C FFI Usage

Include the header and link against the dynamic library:

```c
#include "JudgeBridge.h"

void* h = judge_create("judge.mlpackage", 512, 5, "calibration.json");
if (!h) {
    // handle error
    return;
}

int32_t input_ids[512] = {101, 102, ...};
int32_t attention_mask[512] = {1, 1, ...};
float score;
float clause_probs[5];

int32_t rc = judge_encode_once(
    h,
    input_ids,
    attention_mask,
    NULL,  // token_type_ids (NULL = zeros)
    512,   // length
    &score,
    clause_probs,
    5      // clause_probs length
);

if (rc == 0) {
    // success: score and clause_probs populated
} else {
    // error: rc < 0
}

judge_destroy(h);
```

## Rust FFI Usage

```rust
#[link(name = "JudgeBridge")]
extern "C" {
    fn judge_create(
        model: *const i8,
        seq_len: i32,
        n_clauses: i32,
        calib: *const i8
    ) -> *mut std::ffi::c_void;
    
    fn judge_destroy(h: *mut std::ffi::c_void);
    
    fn judge_encode_once(
        h: *mut std::ffi::c_void,
        ids: *const i32,
        mask: *const i32,
        tt: *const i32,
        len: i32,
        out_score: *mut f32,
        out_probs: *mut f32,
        probs_len: i32
    ) -> i32;
}

// Usage example
let model_path = CString::new("judge.mlpackage").unwrap();
let handle = unsafe {
    judge_create(
        model_path.as_ptr(),
        512,
        5,
        std::ptr::null()
    )
};

if handle.is_null() {
    panic!("Failed to create judge");
}

let mut score: f32 = 0.0;
let mut probs: [f32; 5] = [0.0; 5];

let rc = unsafe {
    judge_encode_once(
        handle,
        input_ids.as_ptr(),
        attention_mask.as_ptr(),
        std::ptr::null(),
        512,
        &mut score,
        probs.as_mut_ptr(),
        5
    )
};

if rc == 0 {
    // success
} else {
    // error
}

unsafe { judge_destroy(handle); }
```

## Important Notes

- **Fixed Sequence Lengths**: The bridge expects enumerated fixed sequence lengths (e.g., 512/1024). Build and load the matching `.mlpackage` for your target sequence length.

- **Tokenization**: Tokenization remains on your side; pass `[1,T]` `int32` arrays. The server expects pre-tokenized inputs.

- **Model Format**: Uses CoreML MLProgram format (iOS 16+ / macOS 13+).

## Next Steps / Future Enhancements

- **Text Tokenization**: Wire a fast local tokenizer (SentencePiece/BPE) in Swift to accept raw text inputs
- **C++ Wrapper**: Add a tiny C++ wrapper with RAII semantics
- **Rust Crate**: Create a Rust crate with safe bindings plus an async client for the HTTP server

## File Structure

```
JudgeBridge/
├── Package.swift
├── Sources/
│   ├── JudgeBridge/
│   │   ├── Calibration.swift
│   │   ├── CoreMLJudgeBridge.swift
│   │   ├── FFI.swift
│   │   └── include/
│   │       └── JudgeBridge.h
│   └── JudgeServer/
│       └── main.swift
└── README.md
```
