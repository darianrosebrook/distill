# Evaluation Harness for Tool-Integration Behaviors

## Purpose

A thin, deterministic evaluation harness to measure model performance on **tool-integration** behaviors using the **same gates** you verify during dataset generation:

* Integration spans: lax & strict grounding F1 (macro + micro)
* Integration coverage, grounded coverage
* Multi-call parity (expected vs observed tool calls)
* JSON args validity & branching error recovery
* Control contamination (controls must not integrate)
* Locale probe correctness (numerals/dates across locales)
* Negative controls (detect hallucinated integration)
* Performance budgets & fingerprints for reproducibility

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
{"name":"read_file","key":{"path":"README.md"},"result":{"ok":true,"content":"# Project\n..."}}
```

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
  --seed 42 \
  --temperature 0.0 \
  --min-eligible-for-gates 15 \
  --fail-on-fingerprint-mismatch
```

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

* **Dataset**: JSONL with an initial `__header__` object (dataset fingerprints, registry SHA, tokenizer fingerprint, gates) followed by items conforming to `schemas/eval_prompt.schema.json`.

* **Fixtures**: JSONL files per tool with entries `{ "name": "<tool>", "key": {...}, "result": {...}}`.

---

## Outputs

* **Results JSONL** (`--out`): Each line matches `schemas/eval_result.schema.json`:
  * `model_output`, `tool_trace` (name, arguments, result), `scores` (per-item metrics), and run fingerprints.

* **Report JSON** (`--report`): Macro/micro metrics + gates, fingerprints, per-tool strict–lax deltas, span histograms, counts (controls, eligible, negative controls), and performance timing.

---

## Determinism & Fingerprints

Harness writes a header into the report:

* `report_version`, `dataset_sha256`, `tool_registry_sha256`
* `tokenizer_fingerprint` (copied from dataset header if present)
* `runner_fingerprint`, `model_fingerprint`
* Decoding params & `seed`
* `integration_span_cap`, `min_eligible_for_gates`, `fail_on_fingerprint_mismatch`

Run fails on fingerprint mismatch unless `--fail-on-fingerprint-mismatch` is turned off.

---

## Runners

* `openai_http.py`: Uses ChatCompletions/function-calling (or tools) to capture tool calls.
* `hf_local.py`: Local generation; emits tool blocks as JSON (your existing JSON normalization is reused).

Both return:

```json
{
  "model_output": "...",
  "tool_trace": [
    {"name":"web.search", "arguments":{"q":"...","top_k":3}},
    {"name":"read_file", "arguments":{"path":"..."}}
  ]
}
```

The **ToolBroker** injects `result` for each call from fixtures, then the scorer evaluates grounding and parity.

---

## Scoring Parity

`eval/scoring/scorer.py` imports your existing verification routines (same regex span extraction, lax/strict grounding, parity, coverage, control checks, locale probe, negative controls). Metrics and gates are **identical** to `verify_contextual_set.py`, but applied to model outputs.

---

## CI Hooks

* Smoke (N=60): assert `controls_with_integration == 0`, coverage ≥ 0.95 (eligible).
* Parity: compare OpenAI vs local runner macro F1 Δ ≤ 0.03.
* Locale probe shard: strict–lax Δ ≤ 0.05 for es/de/fr.
* Teacher-heavy slice: warn on span cap exceedances; fail on F1 drop below threshold.

---

## Troubleshooting

* **fixture_miss**: Broker couldn't find a key—add a matching JSONL entry (normalize keys exactly as the runner emits).

* **control contamination**: Controls produced integration spans or tool calls—inspect those items and runner prompt wrapper.

* **fingerprint mismatch**: Dataset header differs from fixtures/registry; update fixtures or set `--no-fail-on-fingerprint-mismatch` to explore.

---

## Versioning

* `report_version`: start at `1.0.0` (independent of dataset version).
* All breaking changes to report fields bump `report_version`.

---

## Security

* No outbound network calls in scoring paths.
* ToolBroker only serves local, versioned fixtures.
* PII redaction & URL-context heuristics are preserved via your existing verifier utilities.

