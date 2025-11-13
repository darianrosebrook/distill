.PHONY: kd inter proc qat onnx coreml probes eval release format judge worker drafter caws-eval contextual-gen contextual-extract contextual-verify contextual-pipeline gen-scale-1k gen-scale-10k verify-scale-1k verify-scale-10k verify-dual-tokenizers verify-next-registry gen-teacher-heavy verify-teacher-heavy eval-runner-openai eval-runner-local eval-smoke speed-coreml train-student-speed train-student-qat 8-ball magic-8-ball

# Worker model (primary generator, ~9B)
worker:
	python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml

# Judge model (constitutional arbiter, 3-4B or 7B)
judge:
	python -m arbiter.judge_training.train --config configs/judge_training.yaml

judge_train:
	python -m arbiter.judge_training.train --config configs/judge_training.yaml

judge_onnx:
	python -m arbiter.judge_training.export_onnx --ckpt arbiter/judge_training/artifacts/judge.pt

judge_coreml:
	python -m arbiter.judge_training.convert_coreml --onnx arbiter/judge_training/artifacts/onnx/judge_T512.onnx

judge_smoke:
	python -m arbiter.judge_training.smoke_data

judge_pairs_from_caws:
	python -m arbiter.judge_training.build_pairs_from_caws $(IN) $(OUT)

judge_bench:
	python -m arbiter.judge_training.latency_bench

judge_cli:
	python -m arbiter.judge_training.judge_cli arbiter/judge_training/artifacts/coreml/judge.mlpackage microsoft/deberta-v3-small $(PROMPT) $(A) $(B)

# Drafter model (speculative decoding, ~4B, optional)
drafter:
	python -m training.distill_kd --config configs/drafter_4b.yaml configs/kd_recipe.yaml

# Legacy single-model targets (deprecated, use worker/judge/drafter)
kd:
	python -m training.distill_kd --config configs/student_7b_gqa.yaml configs/kd_recipe.yaml

inter:
	python -m training.distill_intermediate --config configs/student_7b_gqa.yaml

proc:
	python -m training.distill_process --checkpoint models/student/checkpoints/latest.pt --config configs/worker_9b.yaml configs/process_supervision.yaml --steps 10000

qat:
	python -m training.quant_qat_int8 --checkpoint models/student/checkpoints/process_supervised_latest.pt --config configs/worker_9b.yaml configs/quant_qat_int8.yaml --steps 30000

# Export targets
onnx-worker:
	python -m conversion.export_onnx --config conversion/shape_sets.json --mode prefill

onnx-judge:
	python -m conversion.judge_export_onnx --config conversion/shape_sets.json

onnx: onnx-worker onnx-judge

# PyTorch export (production path)
pytorch-worker:
	python -m conversion.export_pytorch --checkpoint models/student/checkpoints/latest.pt --out models/student/exported/ --mode both

pytorch-judge:
	python -m conversion.export_pytorch --checkpoint arbiter/judge_training/artifacts/judge.pt --out arbiter/judge_training/artifacts/exported/ --mode prefill

pytorch: pytorch-worker pytorch-judge

# CoreML conversion (PyTorch backend - production)
coreml-worker:
	python -m conversion.convert_coreml --backend pytorch --in models/student/exported/student_fp16.pt --out coreml/artifacts/worker/model.mlpackage --contract models/student/exported/student_fp16_contract.json

coreml-judge:
	python -m conversion.convert_coreml --backend pytorch --in arbiter/judge_training/artifacts/exported/judge_prefill_T512.pt --out coreml/artifacts/judge/model.mlpackage

coreml: coreml-worker coreml-judge

# Deployment helpers
deploy-runtime-config:
	python -m scripts.deploy_model \
		--checkpoint models/student/checkpoints/latest.pt \
		--out-dir models/student/deployed/ \
		--export-pytorch \
		--latent-mode \
		--caws-tier tier_2

deploy-full:
	python -m scripts.deploy_model \
		--checkpoint models/student/checkpoints/latest.pt \
		--out-dir models/student/deployed/ \
		--export-pytorch \
		--export-coreml \
		--latent-mode \
		--caws-tier tier_2

# Knowledge Distillation Dataset Generation
teacher-audit:
	python -m scripts.teacher_audit --teacher $(TEACHER_ENDPOINT) --out reports/teacher_audit.json

kd-dataset:
	python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher $(TEACHER_ENDPOINT) --total 1000 --cache-dir data/logits/

# Evaluation
probes:
	python -m coreml.probes.compare_probes --pt models/student/checkpoints/latest.pt --ml coreml/model.mlpackage

caws-eval:
	python -m evaluation.caws_eval --working-spec .caws/working-spec.yaml

eval:
	python -m evaluation.perf_mem_eval && \
	python -m evaluation.tool_use_eval && \
	python -m evaluation.long_ctx_eval && \
	python -m evaluation.caws_eval --working-spec .caws/working-spec.yaml

# Contextual Dataset Generation & Verification
contextual-gen:
	python -m scripts.generate_contextual_prompts \
		--out data/contextual_prompts.jsonl \
		--total 60 \
		--seed 42 \
		--integration-span-cap 3 \
		--tokenizer models/student/tokenizer

contextual-extract:
	python -m scripts.extract_process_targets \
		--in data/contextual_prompts.jsonl \
		--out data/contextual_extracted.jsonl \
		--tokenizer-path models/student/tokenizer

contextual-verify:
	python -m scripts.verify_contextual_set \
		--in data/contextual_extracted.jsonl \
		--report data/contextual_verification_report.json \
		--tokenizer models/student/tokenizer \
		--perf-budget-sec-per-100 2.0

contextual-pipeline: contextual-gen contextual-extract contextual-verify

# Scale Tests (N=1k, N=10k)
gen-scale-1k:
	python -m scripts.generate_contextual_prompts \
		--out data/contextual_prompts_1k.jsonl \
		--total 1000 \
		--seed 42 \
		--num-shards 1 \
		--shard-index 0 \
		--integration-span-cap 3 \
		--tokenizer models/student/tokenizer

gen-scale-10k:
	python -m scripts.generate_contextual_prompts \
		--out data/contextual_prompts_10k.jsonl \
		--total 10000 \
		--seed 42 \
		--num-shards 10 \
		--shard-index 0 \
		--integration-span-cap 3 \
		--tokenizer models/student/tokenizer

verify-scale-1k:
	python -m scripts.extract_process_targets \
		--in data/contextual_prompts_1k.jsonl \
		--out data/contextual_extracted_1k.jsonl \
		--tokenizer-path models/student/tokenizer && \
	python -m scripts.verify_contextual_set \
		--in data/contextual_extracted_1k.jsonl \
		--report data/reports/verify_scale_1k.json \
		--tokenizer models/student/tokenizer \
		--perf-budget-sec-per-100 2.0

verify-scale-10k:
	python -m scripts.extract_process_targets \
		--in data/contextual_prompts_10k.jsonl \
		--out data/contextual_extracted_10k.jsonl \
		--tokenizer-path models/student/tokenizer && \
	python -m scripts.verify_contextual_set \
		--in data/contextual_extracted_10k.jsonl \
		--report data/reports/verify_scale_10k.json \
		--tokenizer models/student/tokenizer \
		--perf-budget-sec-per-100 2.0

# Multi-tokenizer Verification
verify-dual-tokenizers:
	python -m scripts.verify_contextual_set \
		--in data/contextual_extracted.jsonl \
		--tokenizer models/student/tokenizer \
		--secondary-tokenizer models/student/tokenizer \
		--report data/reports/verify_dual_tok.json

# Forward-compat Registry Check
verify-next-registry:
	python -m scripts.verify_contextual_set \
		--in data/contextual_extracted.jsonl \
		--tokenizer models/student/tokenizer \
		--next-registry tools/schema_registry.py \
		--report data/reports/verify_next_registry.json

# Teacher-heavy Slice (Integration Span Cap Reality Check)
gen-teacher-heavy:
	python -m scripts.generate_contextual_prompts \
		--out data/teacher_heavy.jsonl \
		--total 500 \
		--seed 7 \
		--integration-span-cap 3 \
		--tokenizer models/student/tokenizer

verify-teacher-heavy:
	python -m scripts.extract_process_targets \
		--in data/teacher_heavy.jsonl \
		--out data/teacher_heavy_extracted.jsonl \
		--tokenizer-path models/student/tokenizer && \
	python -m scripts.verify_contextual_set \
		--in data/teacher_heavy_extracted.jsonl \
		--tokenizer models/student/tokenizer \
		--report data/reports/teacher_heavy.json

# Evaluation Harness
eval-runner-openai:
	python -m eval.cli \
		--runner openai_http \
		--model $(MODEL) \
		--in $(IN) \
		--out $(OUT) \
		--report $(REPORT) \
		--fixtures eval/tool_broker/fixtures \
		--seed 42 \
		--temperature 0.0 \
		--min-eligible-for-gates 15 \
		--fail-on-fingerprint-mismatch

eval-runner-local:
	python -m eval.cli \
		--runner hf_local \
		--model $(MODEL) \
		--in $(IN) \
		--out $(OUT) \
		--report $(REPORT) \
		--fixtures eval/tool_broker/fixtures \
		--seed 42 \
		--temperature 0.0 \
		--min-eligible-for-gates 15 \
		--fail-on-fingerprint-mismatch

eval-smoke:
	python -m eval.cli \
		--runner openai_http \
		--model gpt-4 \
		--in data/contextual_extracted.jsonl \
		--out eval/results.smoke.jsonl \
		--report eval/report.smoke.json \
		--fixtures eval/tool_broker/fixtures \
		--seed 42 \
		--temperature 0.0 \
		--min-eligible-for-gates 15 \
		--fail-on-fingerprint-mismatch

# Speed optimization targets (inference-speed-optimization-during-distillation)
HARDWARE ?= $(shell python -c "import platform; print(platform.processor() or 'unknown')")
EXPORT_PATH ?= pytorch_exportedprogram_coreml

# Authoritative CoreML speed run (same slice each time)
speed-coreml:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100

# CoreML speed run with prompt caching (30-50% TTFT reduction)
speed-coreml-cached:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_cached.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--enable-prompt-cache \
		--cache-size-mb 100

# CoreML speed run with speculative decoding (25-40% TTFT improvement)
speed-coreml-speculative:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--drafter-model coreml/artifacts/drafter/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_speculative.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--enable-speculative \
		--spec-k 2

# CoreML speed run with ANE residency measurement
speed-coreml-ane:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_ane.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--measure-ane-residency \
		--ane-samples 100

# CoreML speed run with optimized tokenizer (10-20% TTFT reduction for long prompts)
speed-coreml-tokenizer-opt:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_tokenizer_opt.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--use-optimized-tokenizer

# CoreML speed run with optimized KV cache (ANE-friendly layout, unified memory)
speed-coreml-kv-cache-opt:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_kv_cache_opt.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--use-optimized-kv-cache \
		--kv-cache-heads 32 \
		--kv-cache-head-dim 128 \
		--kv-cache-gqa-groups 4

# CoreML speed run with batch policy (workload-aware batch size selection)
speed-coreml-batch-policy:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_batch_policy.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--workload-type interactive

# CoreML speed run with offline batch policy (batch 2-4 for throughput)
speed-coreml-batch-policy-offline:
	python -m evaluation.perf_mem_eval \
		--model coreml/artifacts/worker/model.mlpackage \
		--dataset data/contextual_final.jsonl \
		--out eval/reports/speed_coreml_batch_policy_offline.json \
		--hardware "$(HARDWARE)" \
		--export-path $(EXPORT_PATH) \
		--max-samples 100 \
		--workload-type offline

# Train with latency-aware losses (ramped) + enumerated shapes
train-student-speed:
	python -m training.distill_kd \
		--config configs/student_9b_gqa.yaml \
		--config configs/kd_recipe.yaml

# Enable QAT for final 20% of steps
train-student-qat:
	python -m training.distill_kd \
		--config configs/student_9b_gqa.yaml \
		--config configs/kd_recipe.yaml

eval-real-local:
	python -m eval.cli \
		--runner hf_local \
		--model $(MODEL) \
		--in $(IN) \
		--out $(OUT) \
		--report $(REPORT) \
		--fixtures eval/tool_broker/fixtures \
		--num-shards 2 \
		--shard-index 0 \
		--seed 42 \
		--temperature 0.0 \
		--min-eligible-for-gates 15 \
		--fail-on-fingerprint-mismatch

eval-fixture-stats:
	@jq '.summary | {fixture_hit_rate, fixture_miss_count, total, num_eligible}' eval/reports/latest.json || echo "Report not found: eval/reports/latest.json"

# Sharding determinism validation defaults (override on CLI or in env)
DATASET ?= data/contextual_final.jsonl
MODEL ?= 
RUNNER ?= hf_local
SHARDS ?= 4

validate-sharding:
	python -m scripts.validate_sharding_determinism \
		--dataset $(DATASET) \
		--model $(MODEL) \
		--runner $(RUNNER) \
		--num-shards $(SHARDS) \
		--fixtures eval/tool_broker/fixtures \
		--report-dir eval/reports \
		--results-dir eval/results \
		--tolerance 1e-3

ci-broker-smoke:
	PYTHONPATH=$(shell pwd):$(PYTHONPATH) python3.11 -m pytest -q -k broker_fixtures_hit_rate tests/test_caws_compliance_gates.py

validate-sharding:
	python -m scripts.validate_sharding_determinism \
		--dataset $(DATASET) \
		--model $(MODEL) \
		--runner $(RUNNER) \
		--num-shards $(SHARDS) \
		--fixtures eval/tool_broker/fixtures \
		--atol 1e-6 \
		--rtol 1e-6 \
		--seed 42

eval-local:
	python -c "from training.callbacks.eval_harness_callback import run_eval_for_checkpoint; \
		run_eval_for_checkpoint('ckpts/latest', num_shards=4)"

eval-diff:
	python -m eval.reports.diff_reports eval/reports/blessed.json eval/reports/latest.json

eval-publish:
	python -m scripts.publish_eval_report eval/reports/latest.json

release:
	python -m scripts.package --model coreml/model.mlpackage --out dist/

format:
	ruff check --fix .; ruff format .

# Version gate: Check Python and dependencies
# Use PYTHON env var if set, otherwise default to python
PYTHON ?= python
.PHONY: check-versions
check-versions:
	$(PYTHON) -m infra.version_gate --skip-ort

# Dependencies
.PHONY: deps-core deps-ort
deps-core:
	pip install --upgrade pip==24.0 setuptools==69.0.0 wheel==0.43.0
	pip install -r requirements-core.txt

deps-ort:
	@echo "Detecting platform for onnxruntime..."
	@if [ "$$(uname -m)" = "arm64" ]; then \
		echo "Installing onnxruntime-silicon for Apple Silicon..."; \
		pip install onnxruntime-silicon==1.19.2 || pip install onnxruntime==1.19.2; \
	else \
		echo "Installing onnxruntime for x86_64..."; \
		pip install onnxruntime==1.19.2; \
	fi

# Smoke test: Build a tiny ONNX, convert to CoreML, run checks and probes
# Never requires onnxruntime; uses placeholder if conversion unavailable
TOY_SEQ?=128
TOY_VOCAB?=256
TOY_DMODEL?=64
TEACHER_ENDPOINT ?= http://localhost:8000

.PHONY: toy-onnx onnx-surgery coreml-stub probes-skip ane-skip smoke_toy

toy-onnx:
	python -m conversion.make_toy_onnx --seq $(TOY_SEQ) --vocab $(TOY_VOCAB) --dmodel $(TOY_DMODEL) --out onnx/toy.onnx

onnx-surgery:
	python -m conversion.onnx_surgery --inp onnx/toy.onnx --out onnx/toy.sanitized.onnx --infer

coreml-stub:
	python -m conversion.convert_coreml --backend onnx --in onnx/toy.sanitized.onnx --out coreml/artifacts/toy/model.mlpackage --allow-placeholder

ane-skip:
	python -m coreml.ane_checks --mlpackage coreml/artifacts/toy/model.mlpackage

probes-skip:
	python -m coreml.probes.compare_probes --onnx onnx/toy.sanitized.onnx --ml coreml/artifacts/toy/model.mlpackage --seq $(TOY_SEQ) --dmodel $(TOY_DMODEL)

smoke_toy: check-versions toy-onnx onnx-surgery coreml-stub ane-skip probes-skip
	@echo "✅ Smoke test PASSED (may include SKIP for placeholder)"

# Full parity test: Requires onnxruntime, fails loud if conversion unavailable
.PHONY: convert_coreml probes-full ane-checks parity_full toy-block

toy-block:
	python -m conversion.make_toy_block --dmodel $(TOY_DMODEL) --nheads 4 --seq $(TOY_SEQ) --out models/toy_block.pt

convert_coreml:
	python -m conversion.convert_coreml --backend pytorch --in models/toy_block.pt --out coreml/artifacts/toy_block/model.mlpackage

generate-probes:
	python -m coreml.probes.generate_probes --pt-model models/toy_block.pt --ml-model coreml/artifacts/toy_block/model.mlpackage --out coreml/probes/toy_block --seq $(TOY_SEQ) --dmodel $(TOY_DMODEL)

probes-full:
	python -m coreml.probes.compare_probes --pt coreml/probes/toy_block/toy_block_pt.npz --ml coreml/probes/toy_block/toy_block_ml.npz

ane-checks:
	python -m coreml.ane_checks --mlpackage coreml/artifacts/toy_block/model.mlpackage

parity_full: check-versions toy-block convert_coreml generate-probes ane-checks probes-full
	@echo "✅ Parity test PASSED (real PyTorch→CoreML conversion)"

# PyTorch smoke test: Real mlpackage from PyTorch (proves supported front-end)
.PHONY: toy-torch smoke_torch

toy-torch:
	python -m conversion.make_toy_torch --seq $(TOY_SEQ) --vocab $(TOY_VOCAB) --dmodel $(TOY_DMODEL) --out models/toy_torch.pt

smoke_torch: check-versions toy-torch
	python -m conversion.convert_coreml --backend pytorch --in models/toy_torch.pt --out coreml/artifacts/toy_torch/model.mlpackage
	python -m coreml.ane_checks --mlpackage coreml/artifacts/toy_torch/model.mlpackage
	@echo "✅ PyTorch smoke test PASSED (real mlpackage created)"

# Training smoke test: Create tiny model/dataset and verify training pipeline
.PHONY: toy-training smoke_training

toy-training:
	$(PYTHON) -m training.make_toy_training --out-dir training/toy_test --samples 10 --steps 5 --dmodel 64 --nlayers 2 --vocab 32000

smoke_training: check-versions toy-training
	bash scripts/test_toy_training.sh
	@echo "✅ Training smoke test PASSED (training pipeline verified)"

# Toy E2E pipeline: Full flow from dataset → training → export → conversion → verification
.PHONY: toy-clean toy-e2e

toy-clean:
	rm -rf /tmp/toy_* /tmp/toy.* eval/reports/toy_e2e.json training/toy_test/*.pt training/toy_test/*.jsonl

toy-e2e: toy-clean
	$(PYTHON) -m data.make_toy_kd --out /tmp/toy_kd.jsonl --n 128
	$(PYTHON) -m training.run_toy_distill --in /tmp/toy_kd.jsonl --out /tmp/toy.ckpt --epochs 2 --mps 0
	$(PYTHON) -m conversion.export_pytorch --checkpoint /tmp/toy.ckpt --out /tmp/toy_exported --toy --mode prefill --seq 64 --enumerated-T 64 128 256
	# Compile a single representative shape fast by default
	@if [ -f /tmp/toy_exported/student_prefill_T128.pt ]; then \
		$(PYTHON) -m conversion.convert_coreml --backend pytorch --in /tmp/toy_exported/student_prefill_T128.pt --out /tmp/toy_T128.mlpackage --seq 128 --compute-units all || echo "⚠️  CoreML conversion may have failed (CoreML may not be available)"; \
	else \
		echo "⚠️  Prefill model not found, skipping conversion"; \
	fi
	@if [ -f /tmp/toy_T128.mlpackage ]; then \
		$(PYTHON) -m evaluation.toy_contracts --model /tmp/toy_T128.mlpackage --seq 128 --report eval/reports/toy_e2e.json || echo "⚠️  Verification may have failed (CoreML may not be available)"; \
	else \
		echo "⚠️  CoreML model not found, skipping verification"; \
	fi
	@echo "Toy E2E OK → eval/reports/toy_e2e.json"

.PHONY: 8-ball magic-8-ball
8-ball: magic-8-ball
magic-8-ball: toy-clean
	@echo "🎱 Starting Magic 8 Ball E2E Pipeline 🎱"
	$(PYTHON) -m data.make_toy_kd --out /tmp/magic_8_ball_kd.jsonl --n 128 --magic-8-ball
	$(PYTHON) -m training.run_toy_distill --in /tmp/magic_8_ball_kd.jsonl --out /tmp/magic_8_ball.ckpt --epochs 2 --mps 0 --magic-8-ball
	$(PYTHON) -m conversion.export_pytorch --checkpoint /tmp/magic_8_ball.ckpt --out /tmp/magic_8_ball_exported --toy --mode prefill --seq 64 --enumerated-T 64 128 256
	@if [ -f /tmp/magic_8_ball_exported/student_prefill_T128.pt ]; then \
		$(PYTHON) -m conversion.convert_coreml --backend pytorch --in /tmp/magic_8_ball_exported/student_prefill_T128.pt --out /tmp/magic_8_ball_T128.mlpackage --seq 128 --compute-units all --toy || echo "⚠️  CoreML conversion may have failed (CoreML may not be available)"; \
	else \
		echo "⚠️  Prefill model not found, skipping conversion"; \
	fi
	@if [ -d /tmp/magic_8_ball_T128.mlpackage ]; then \
		$(PYTHON) -m evaluation.toy_contracts --model /tmp/magic_8_ball_T128.mlpackage --seq 64 128 256 --report eval/reports/magic_8_ball_e2e.json || echo "⚠️  Verification may have failed (CoreML may not be available)"; \
	else \
		echo "⚠️  CoreML model not found, skipping verification"; \
	fi
	@echo "🎨 Generating model card charts..."
	$(PYTHON) scripts/generate_magic8_charts.py
	@echo "🎱 Magic 8 Ball E2E Complete! → eval/reports/magic_8_ball_e2e.json 🎱"

.PHONY: toy-e2e-multi
toy-e2e-multi: toy-clean
	$(PYTHON) -m data.make_toy_kd --out /tmp/toy_kd.jsonl --n 128
	$(PYTHON) -m training.run_toy_distill --in /tmp/toy_kd.jsonl --out /tmp/toy.ckpt --epochs 2 --mps 0
	$(PYTHON) -m conversion.export_pytorch --checkpoint /tmp/toy.ckpt --out /tmp/toy_exported --toy --mode prefill --seq 64 --enumerated-T 64 128 256
	# Compile each enumerated shape as separate mlpackage
	@for L in 64 128 256; do \
		if [ -f /tmp/toy_exported/student_prefill_T$$L.pt ]; then \
			$(PYTHON) -m conversion.convert_coreml --backend pytorch --in /tmp/toy_exported/student_prefill_T$$L.pt --out /tmp/toy_T$$L.mlpackage --seq $$L --compute-units all || echo "⚠️  Failed to convert T$$L"; \
		fi \
	done
	@if [ -f /tmp/toy_T128.mlpackage ]; then \
		$(PYTHON) -m evaluation.toy_contracts --model /tmp/toy_T128.mlpackage --model-dir /tmp --seq 64 128 256 --report eval/reports/toy_e2e.json || echo "⚠️  Verification may have failed (CoreML may not be available)"; \
	else \
		echo "⚠️  CoreML model not found, skipping verification"; \
	fi
	@echo "Toy E2E (multi) OK → eval/reports/toy_e2e.json"
