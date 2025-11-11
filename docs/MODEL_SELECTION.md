# MODEL_SELECTION.md (task → model map)

> All models listed are feasible to run **on-device** (CoreML or MPS) on M-series. Exact picks assume English-first. Swap equivalents for your locale.

## Core Generation & Reasoning

| Role | Model (baseline) | Size | Deployment | Why |
|---|---|---:|---|---|
| **StudentLM** (general reasoning, tool use) | **Kimi-student (ours)** distilled from K2-Thinking; architecture in repo | 8–9B | CoreML (ANE pref) | Balanced quality/latency; GQA for KV budget; designed for enumerated 4k/8k/16k. |
| **DraftLM** (speculative decoding) | Tiny student variant or **Phi-3-mini**-style 1–2B equivalent | 1–2B | CoreML/MPS | Cuts TTFA and boosts tok/s as a drafter; verifier = StudentLM. |

## Retrieval, Web/File Search & Research

| Subsystem | Model | Size | Deployment | Notes |
|---|---|---:|---|---|
| **Embedder (bi-encoder)** | **BAAI/bge-small-en-v1.5** or **e5-small-v2** | 30–110M | CoreML | Fast, good recall; 384–768 dims; easy to CoreML-ize. |
| **Reranker (cross-encoder)** | **cross-encoder/ms-marco-MiniLM-L-6-v2** or **bge-reranker-base** | 65–110M | CoreML | Low-latency precision rerank; use for top-50→top-5. |
| **Browser extractor** (readability/QA) | **MiniLM/DeBERTa-v3-small QA head** | 50–140M | CoreML | Span extraction/answer finding on fetched pages/files. |
| **Research synthesis** | **StudentLM** | 8–9B | CoreML | Compose multi-doc answers; gated by judge. |

## Coding: Write, Edit, and Eval

| Task | Model | Size | Deployment | Notes |
|---|---|---:|---|---|
| **Code generation/edit** | **StudentLM-Coder** (student SFT on code) or **StarCoder2-7B** baseline for KD | 7–9B | CoreML | Add code-SFT head; keep same tokenizer or add a BPE merge table. |
| **Static lint/AST ops** | Tree-sitter + rules | — | Native | Deterministic safety rails; not an LM. |
| **Unit-test synthesis** | **StudentLM-Coder** (tool-aware) | 7–9B | CoreML | Trained to output tests + rationale. |
| **Eval (patch scoring)** | **Judge-Code** (DeBERTa-v3-small pairwise ranker) | 86M | CoreML | Trained on diff+rationale pairs under CAWS. |

## Documentation Writing

| Task | Model | Size | Deployment | Notes |
|---|---|---:|---|---|
| **Doc writer** | **StudentLM-Doc** (style-tuned) or **Llama-3-8B-Instruct** as KD teacher | 8–9B | CoreML | Emphasize structure, style, and link hygiene; process-supervision to enforce templates. |
| **Abstractor/summarizer** | **StudentLM** (cheap decode) or **MiniLM abstractive head** | 8–9B / 65M | CoreML | Use small model for extractive, student for abstractive. |

## Judge Council (CAWS Enforcement)

| Judge | Model | Size | Deployment | Signal |
|---|---|---:|---|---|
| **Pairwise Arbiter** | **DeBERTa-v3-small** (pairwise rank + clause probs) | 86M | CoreML | Picks A/B/TIE; emits clause likelihoods. |
| **Format/Schema Judge** | Programmatic JSON Schema + regex | — | Native | 0-cost hard gate for tool calls. |
| **Fact/Claim Judge** | **MiniLM / DeBERTa-v3-base** claim verifier | 100–300M | CoreML | Scores extracted claims; cheap. |
| **Safety/Policy Judge** | Tiny classifier (e.g., DistilBERT) | 66M | CoreML | Flags policy/risk classes. |
| **Style Judge** | Small classifier on style rubric | 30–60M | CoreML | Ensures doc/code standards. |

> All council members compile cleanly to CoreML `mlprogram` and fit comfortably on M-series with negligible latency compared to the StudentLM.

## Optional Cloud Fallbacks (if ever needed)

Keep interface-compatible stubs to swap in a remote LLM (e.g., GPT-4 class) for rare escalations. The arbiter should route to them **only after** local gates fail or budgets require.

## Concrete Model IDs (HF Examples) for Bootstrapping KD/Aux Tasks

- **Embeddings**: `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`
- **QA/Extractor**: `deepset/minilm-uncased-squad2`, `microsoft/deberta-v3-small`
- **Code teacher (aux)**: `bigcode/starcoder2-7b`
- **Doc teacher (aux)**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Pairwise judge base**: `microsoft/deberta-v3-small`

> Use these as teachers or baselines for KD and for tiny council members. Final on-device models should be converted to CoreML.

## Acceptance Gates (unchanged; apply per runner)

- **G1 Parity** (CoreML vs Torch probes) ≤ 2% rel-err at `attn_out`.
- **G2 TTFA** ≤ 2.0 s @ 4k; **G3 tok/s** ≥ 25; **G4 mem** ≤ 12 GB @ 4k decode.
- **G5 JSON validity** ≥ 98%; **G6 Tool select** ≥ 90%.
- **G7 Long-ctx needles** ≥ 95% @ 4k; ≥ 90% @ 16k.
- **G8 Stability** no NaN/INF across 1k prompts.

