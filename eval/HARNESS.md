# Evaluation Harness for Tool-Integration Behaviors

## Purpose

A thin, deterministic evaluation harness to measure model performance on **tool-integration** behaviors using the **same gates** you verify during dataset generation:

- Integration spans: lax & strict grounding F1 (macro + micro)
- Integration coverage, grounded coverage
- Multi-call parity (expected vs observed tool calls)
- JSON args validity & branching error recovery
- Control contamination (controls must not integrate)
- Locale probe correctness (numerals/dates across locales)
- Negative controls (detect hallucinated integration)
- Performance budgets & fingerprints for reproducibility

No live network calls. Tools are **replayed** via fixtures.

---

## Directory Layout

```
eval/
  HARNESS.md
  cli.py
  config/
    eval.default.yaml
    tools.allowlist.json
  schemas/
    eval_prompt.schema.json
    eval_result.schema.json
  runners/
    base.py
    openai_http.py
    hf_local.py
  tool_broker/
    broker.py
    fixtures/
      web.search.jsonl
      read_file.jsonl
      code_exec.jsonl
  scoring/
    scorer.py
    metrics.py
  reports/
    summarize.py
```

---

## Quickstart

### 1) Prepare Dataset

Use your verified dataset (header + items):

```bash
# Typical verified dataset
data/contextual_final.s0.jsonl
```

### 2) Provide Fixtures (Deterministic Tool Replies)

`eval/tool_broker/fixtures/web.search.jsonl` (JSONL; keyed by normalized query):

```json
{"name":"web.search","key":{"q":"what is rope precision", "top_k":3},"result":{"ok":true,"hits":[{"title":"...", "snippet":"...", "url":"https://..."}]}}
{"name":"web.search","key":{"q":"embedding gemma pooling fp16", "top_k":5},"result":{"ok":true,"hits":[...]}}
```

Same for `read_file.jsonl`:

```json
{
  "name": "read_file",
  "key": { "path": "README.md" },
  "result": { "ok": true, "content": "# Project\n..." }
}
```

**Note**: The ToolBroker normalizes arguments before lookup (see "Fixture Normalization" section below), so fixtures match despite whitespace/case variations.

### 3) Run Evaluation

OpenAI-compatible endpoint:

```bash
python -m eval.cli \
  --runner openai_http \
  --model gpt-xyz \
  --in data/contextual_final.s0.jsonl \
  --out eval/results.s0.openai.jsonl \
  --report eval/report.s0.openai.json \
  --fixtures eval/tool_broker/fixtures \
  --prompt-wrapper eval/prompt_wrappers/minimal_system_user.j2 \
  --seed 42 \
  --temperature 0.0 \
  --min-eligible-for-gates 15 \
  --fail-on-fingerprint-mismatch
```

**Optional**: Use `--prompt-wrapper` to customize system/user message formatting (see "Prompt Wrappers" section below).

Local Transformers:

```bash
python -m eval.cli \
  --runner hf_local \
  --model /models/my-checkpoint \
  --in data/contextual_final.s0.jsonl \
  --out eval/results.s0.local.jsonl \
  --report eval/report.s0.local.json \
  --fixtures eval/tool_broker/fixtures \
  --seed 42 \
  --temperature 0.0
```

Sharded (parallel) run:

```bash
# 4-way sharding
for i in 0 1 2 3; do
  python -m eval.cli ... --num-shards 4 --shard-index $i --out eval/results.s0.$i.jsonl &
done
wait

# Then concat results & summarize again if needed
```

---

## Inputs

- **Dataset**: JSONL with an initial `__header__` object (dataset fingerprints, registry SHA, tokenizer fingerprint, gates) followed by items conforming to `schemas/eval_prompt.schema.json`.

- **Fixtures**: JSONL files per tool with entries `{ "name": "<tool>", "key": {...}, "result": {...}}`.

---

## Outputs

- **Results JSONL** (`--out`): Each line matches `schemas/eval_result.schema.json`:

  - `model_output`, `tool_trace` (name, arguments, result), `scores` (per-item metrics), and run fingerprints.

- **Report JSON** (`--report`): Macro/micro metrics + gates, fingerprints, per-tool strict–lax deltas, span histograms, counts (controls, eligible, negative controls), and performance timing.

---

## Determinism & Fingerprints

Harness writes a header into the report:

- `report_version`, `dataset_sha256`, `tool_registry_sha256`
- `tokenizer_fingerprint` (copied from dataset header if present)
- `runner_fingerprint`, `model_fingerprint`
- Decoding params & `seed`
- `integration_span_cap`, `min_eligible_for_gates`, `fail_on_fingerprint_mismatch`

Run fails on fingerprint mismatch unless `--fail-on-fingerprint-mismatch` is turned off.

---

## Runners

- `openai_http.py`: Uses ChatCompletions/function-calling (or tools) to capture tool calls.
- `hf_local.py`: Local generation; emits tool blocks as JSON (your existing JSON normalization is reused).

Both return:

```json
{
  "model_output": "...",
  "tool_trace": [
    { "name": "web.search", "arguments": { "q": "...", "top_k": 3 } },
    { "name": "read_file", "arguments": { "path": "..." } }
  ]
}
```

The **ToolBroker** injects `result` for each call from fixtures, then the scorer evaluates grounding and parity.

---

## Fixture Normalization

The ToolBroker normalizes tool arguments before lookup to handle variations across runners:

- **Query Fields** (`q`, `query`): Lowercased and whitespace collapsed to single space
- **Default Values**: `top_k=3` automatically added for `web.search*` tools if missing
- **None Keys**: Removed to avoid mismatches

Example normalization:

```python
# These all match the same fixture entry:
broker.call("web.search", {"q": "  ROPE  Precision  "})
broker.call("web.search", {"q": "rope precision"})
broker.call("web.search", {"q": "ROPE PRECISION", "top_k": None})
broker.call("web.search", {"q": "rope precision"})  # top_k defaults to 3
```

This normalization ensures fixtures match despite runner-specific formatting differences (whitespace, case, optional parameters).

---

## Sharding Determinism

The evaluation harness supports parallel evaluation via sharding. To ensure determinism, sharded results must be **observationally equivalent** to a single-pass baseline.

### Stable Hash Partitioning

**Why not modulo?** The harness uses **stable hash partitioning** (not position-based modulo) to assign samples to shards. This ensures shard membership remains identical across dataset reorders, making baselines comparable over time.

**Implementation:**

- Shard assignment derived from SHA256 hash of `sample_id`
- If `sample_id` is missing, synthesized from stable hash of row JSON
- Uses first 8 bytes as little-endian integer to reduce modulo bias

**Acceptance:** Shard membership remains identical across reorders of the same dataset.

### Validation Process

The `validate_sharding_determinism.py` script ensures sharded evaluation produces identical results:

1. **Baseline run**: Evaluate without sharding → `results.baseline.jsonl`, `report.baseline.json`
2. **Sharded run**: Evaluate with N shards → `results.shard_*.jsonl`, `report.shard_*.json`
3. **Concatenate**: Merge shard results, **sorted by sample_id** (stable sort)
4. **Re-summarize**: Use `eval.reports.summarize.summarize_results()` on concatenated results
5. **Per-example comparison**: Normalize and compare each sample's output (exact match required)
6. **Metric comparison**: Compare aggregate metrics with dual tolerance (absolute + relative for floats, exact for counts)
7. **Fingerprint validation**: Hard fail if any fingerprint differs
8. **Shard completeness**: Validate no missing/duplicate samples across shards

### Deterministic Environment

The validator enforces deterministic execution:

**Environment variables:**

- `PYTHONHASHSEED=0` (deterministic hash randomization)
- `OMP_NUM_THREADS=1` (single-threaded BLAS)
- `MKL_NUM_THREADS=1` (single-threaded MKL)

**Evaluation parameters:**

- `temperature=0.0` (deterministic sampling)
- `top_p=1.0`, `do_sample=False` (if supported by runner)
- HTTP retries disabled (treat retries as hard fail for determinism)

**PyTorch settings** (if using HF runner):

- `torch.set_num_threads(1)`
- Deterministic flags enabled

### Hard Fingerprint Checks

Validation **fails immediately** if any header fingerprint differs:

- `dataset_sha256`
- `tool_registry_sha256`
- `tokenizer_fingerprint`
- `prompt_wrapper_sha256` (if present)
- `runner_fingerprint`
- `model_fingerprint`

This ensures sharding equivalence is well-posed (same inputs, same model, same configuration).

### Per-Example Equivalence

Each sample's normalized output must match exactly between baseline and concatenated shards:

- **Tool choice**: Same tools called
- **Arguments**: Normalized (sorted keys, query fields lowercased/collapsed)
- **Integration spans**: Same byte spans extracted
- **Scores**: Exact match for counts, tolerance for floats (`abs(a-b) ≤ 1e-6`)

Normalization uses the same canonicalization logic as the scorer (broker normalization, span extraction).

### Shard Completeness & Uniqueness

Validation ensures:

- `union(shard_ids) == baseline_ids` (exact match, no missing samples)
- `intersection(shard_i, shard_j) == ∅` for all `i ≠ j` (no duplicates across shards)
- `len(union) == len(baseline_ids)` (no extras)

Reports shard sizes, missing sample IDs, and duplicate IDs for diagnostics.

### Metric Comparison

Compares aggregate metrics with **dual tolerance**:

- **Floating-point metrics**: `abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))`

  - Default: `atol=1e-6`, `rtol=1e-6`
  - Fields: `avg_integration_f1_macro_lax`, `avg_integration_f1_macro_strict`, `avg_integration_f1_micro_lax`, `avg_integration_f1_micro_strict`, `privacy_ok_rate`, `multi_call_parity_rate`, `json_args_valid_rate`

- **Exact match metrics**: Must match exactly

  - Fields: `num_eligible`, `controls_with_integration` (must be 0), `integration_span_count_histogram`

- **Per-tool deltas**: Compares `per_tool_deltas` dict structure

### Governance Checks

Validation fails if:

- Any shard report has `gates_ok=false` due to fixture misses or privacy breaches
- Per-example mismatches detected
- Metric differences exceed tolerance
- Fingerprint mismatches detected
- Shard completeness/uniqueness violations

**Note:** `wall_time_sec` is **not compared** (expected to differ between baseline and sharded runs).

### Validation Report

The validator generates `eval/reports/sharding_validation.json`:

```json
{
  "ok": true,
  "num_shards": 4,
  "dataset_sha256": "...",
  "runner_fingerprint": "...",
  "model_fingerprint": "...",
  "per_example_mismatches": 0,
  "metric_diffs": {},
  "shard_sizes": [250, 250, 250, 250],
  "duplicates": [],
  "missing": [],
  "fingerprints_match": true,
  "coverage_ok": true,
  "metrics_match": true,
  "gates_ok": true
}
```

### Usage

```bash
make validate-sharding \
  DATASET=data/contextual_final.jsonl \
  MODEL=/path/to/ckpt \
  RUNNER=hf_local \
  SHARDS=4
```

Or directly:

```bash
python -m scripts.validate_sharding_determinism \
  --dataset data/contextual_final.jsonl \
  --model /path/to/ckpt \
  --runner hf_local \
  --num-shards 4 \
  --fixtures eval/tool_broker/fixtures \
  --atol 1e-6 \
  --rtol 1e-6 \
  --seed 42 \
  --output eval/reports/sharding_validation.json
```

### Failure Modes

**Added/Removed Samples → Modulo Partition Mismatch**

- **Symptom:** Shard membership changes when dataset order changes
- **Fix:** Stable hash partition (already implemented); validate coverage set equality

**HTTP Retries → Nondeterministic Hosted Outputs**

- **Symptom:** Same sample produces different outputs across runs
- **Fix:** Treat any retry as failure in determinism mode; disable auto-retry jitter

**Tokenization Threads → Subtle Diff in Byte Spans**

- **Symptom:** Integration spans differ slightly between runs
- **Fix:** Single-thread tokenization (`OMP_NUM_THREADS=1`); pin tokenizer version; record fingerprint

**Fixture Drift → Same Sample Hits Different Fixture Normalization**

- **Symptom:** Tool results differ despite same arguments
- **Fix:** Pin `tool_registry_sha256`; fail when changed (fingerprint check)

**Partial Success Signals**

- **Symptom:** Metrics match but gates fail (fixture misses, privacy breach)
- **Fix:** Fail determinism check if any shard report has `gates_ok=false` due to governance issues

---

## Prompt Wrappers

Customize system/user message formatting via template files. Useful for:

- Testing different prompt formats
- Adapting to model-specific requirements
- Reproducing exact prompt structures

**Template Formats**:

**Jinja2 Format** (`*.j2`): Returns JSON with `system` and `user` fields:

```json
{
  "system": "{{ system }}",
  "user": "{{ user }}"
}
```

**String Template Format** (`*.tmpl`): Flat text with variable substitution:

```
[SYS] $system
[USER] $user
```

**Available Variables**:

- `system`: Default system message (can be overridden)
- `user`: Dataset prompt text
- `tools`: List of tool schemas (informational)

**Usage**:

```bash
python -m eval.cli \
  --runner openai_http \
  --model gpt-4 \
  --prompt-wrapper eval/prompt_wrappers/minimal_system_user.j2 \
  ...
```

**Fingerprint Tracking**: Wrapper SHA256 is included in `runner_fingerprint` for reproducibility. Changing the wrapper template changes the fingerprint, ensuring results are traceable to exact prompt formatting.

**Fallback**: If Jinja2 is unavailable, `string.Template` is used automatically for `.tmpl` files.

---

## Scoring Parity

`eval/scoring/scorer.py` imports your existing verification routines (same regex span extraction, lax/strict grounding, parity, coverage, control checks, locale probe, negative controls). Metrics and gates are **identical** to `verify_contextual_set.py`, but applied to model outputs.

---

## CI Hooks

- **Broker Fixture Hit Rate** (`make ci-broker-smoke`): Validates ≥95% fixture hit rate across normalization variants. Fast execution (~1-2 min, no models). Runs on push/PR via GitHub Actions (`.github/workflows/broker-smoke.yml`).
- **Smoke (N=60)**: assert `controls_with_integration == 0`, coverage ≥ 0.95 (eligible).
- **Parity**: compare OpenAI vs local runner macro F1 Δ ≤ 0.03.
- **Locale probe shard**: strict–lax Δ ≤ 0.05 for es/de/fr.
- **Teacher-heavy slice**: warn on span cap exceedances; fail on F1 drop below threshold.

---

## Troubleshooting

- **fixture_miss**: Broker couldn't find a key—add a matching JSONL entry. The broker normalizes arguments (lowercase query fields, collapse whitespace, default `top_k` for `web.search*`), so ensure your fixture keys match the normalized form. See "Fixture Normalization" section above.

- **control contamination**: Controls produced integration spans or tool calls—inspect those items and runner prompt wrapper.

- **fingerprint mismatch**: Dataset header differs from fixtures/registry; update fixtures or set `--no-fail-on-fingerprint-mismatch` to explore.

---

## Versioning

- `report_version`: start at `1.0.0` (independent of dataset version).
- All breaking changes to report fields bump `report_version`.

---

## Security

- No outbound network calls in scoring paths.
- ToolBroker only serves local, versioned fixtures.
- PII redaction & URL-context heuristics are preserved via your existing verifier utilities.
