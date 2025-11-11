.PHONY: kd inter proc qat onnx coreml probes eval release format judge worker drafter caws-eval

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
	python -m training.distill_process --config configs/process_supervision.yaml

qat:
	python -m training.quant_qat_int8 --config configs/quant_qat_int8.yaml

# Export targets
onnx-worker:
	python -m conversion.export_onnx --config conversion/shape_sets.json --mode prefill

onnx-judge:
	python -m conversion.judge_export_onnx --config conversion/shape_sets.json

onnx: onnx-worker onnx-judge

# CoreML conversion
coreml-worker:
	python -m conversion.convert_coreml --config configs/convert_coreml.yaml

coreml-judge:
	python -m conversion.judge_export_coreml artifacts/onnx/judge/judge_T2048.onnx

coreml: coreml-worker coreml-judge

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

release:
	python -m scripts.package --model coreml/model.mlpackage --out dist/

format:
	ruff check --fix .; ruff format .

# Version gate: Check Python and dependencies
.PHONY: check-versions
check-versions:
	python -m infra.version_gate --skip-ort

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
.PHONY: convert_coreml probes-full ane-checks parity_full

convert_coreml:
	python -m conversion.convert_coreml --backend onnx --in onnx/toy.sanitized.onnx --out coreml/artifacts/toy/model.mlpackage

probes-full:
	python -m coreml.probes.compare_probes --onnx onnx/toy.sanitized.onnx --ml coreml/artifacts/toy/model.mlpackage --seq $(TOY_SEQ) --dmodel $(TOY_DMODEL)

ane-checks:
	python -m coreml.ane_checks --mlpackage coreml/artifacts/toy/model.mlpackage

parity_full: check-versions deps-ort toy-onnx onnx-surgery convert_coreml ane-checks probes-full
	@echo "✅ Parity test PASSED"

# PyTorch smoke test: Real mlpackage from PyTorch (proves supported front-end)
.PHONY: toy-torch smoke_torch

toy-torch:
	python -m conversion.make_toy_torch --seq $(TOY_SEQ) --vocab $(TOY_VOCAB) --dmodel $(TOY_DMODEL) --out models/toy_torch.pt

smoke_torch: check-versions toy-torch
	python -m conversion.convert_coreml --backend pytorch --in models/toy_torch.pt --out coreml/artifacts/toy_torch/model.mlpackage
	python -m coreml.ane_checks --mlpackage coreml/artifacts/toy_torch/model.mlpackage
	@echo "✅ PyTorch smoke test PASSED (real mlpackage created)"
