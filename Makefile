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
