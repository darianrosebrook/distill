// Sources/JudgeBridge/include/JudgeBridge.h
// C FFI header for CoreML Judge Bridge
// @author: @darianrosebrook

#ifndef JUDGEBRIDGE_H
#define JUDGEBRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns opaque handle or NULL on error. Caller must call judge_destroy.
void* judge_create(const char* model_path, int32_t seq_len, int32_t n_clauses, const char* calibration_path);

void judge_destroy(void* handle);

// Returns 0 on success; negative on error.
int32_t judge_encode_once(
    void* handle,
    const int32_t* input_ids,
    const int32_t* attention_mask,
    const int32_t* token_type_ids,  // may be NULL to use zeros
    int32_t length,
    float* out_score,
    float* out_clause_probs,  // array length = out_clause_len
    int32_t out_clause_len
);

#ifdef __cplusplus
}
#endif

#endif

