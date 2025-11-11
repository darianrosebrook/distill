# TASK_RUNNERS.md (wiring plan)

## Web/File Search & Research

1. **Retrieve** with Embedder → BM25 hybrid.
2. **Rerank** top-50→top-5 with cross-encoder.
3. **Extract** spans with QA head (MiniLM/DeBERTa-small).
4. **Synthesize** with StudentLM; attach citations; pass to **Pairwise Arbiter** for self-competition (N=2–3 drafts).

## File Editing & Coding

1. **Plan** (StudentLM-Coder emits plan-of-edit under CAWS budgets).
2. **Edit** using AST guards (tree-sitter) + constrained patches.
3. **Test** (runner plugin) → metrics to Arbiter.
4. **Judge-Code** ranks candidate patches; CAWS gates decide PASS/FAIL/WAIVER.

## Documentation Writing

1. **Template** selection (Style Judge decides skeleton).
2. **Draft** with StudentLM-Doc; enforce headings/links with schema.
3. **Self-critique** (2-way candidate + Pairwise Arbiter).
4. **Publish** only if Style & Claim judges pass thresholds.

## Sizing & CoreML Notes

- Prefer **FP16 compute + INT8 weights (QAT)** for StudentLM; judges can be full-FP16.
- Enumerate shapes `{4096, 8192, 16384}`; keep a sliding-window path for >16k.
- For speculative decoding: export **DraftLM** separately; wire verifier cancelation.

