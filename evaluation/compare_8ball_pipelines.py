#!/usr/bin/env python3
"""
Compare 8-ball model predictions across pipeline stages.

Verifies that class distribution is preserved through:
- PyTorch FP32 → CoreML
- PyTorch FP32 → GGUF → Ollama

Author: @darianrosebrook
"""

import json
import sys
from pathlib import Path
from evaluation.eight_ball_eval import (
    evaluate_pytorch_model,
    evaluate_coreml_model,
    evaluate_ollama_model,
    compare_predictions,
    load_eval_questions,
    PredictionResult,
    EvaluationMetrics,
)
import argparse


def main():
    ap = argparse.ArgumentParser(
        description="Compare 8-ball model across conversion pipeline"
    )
    ap.add_argument(
        "--pytorch-model",
        help="PyTorch model path (reference)",
    )
    ap.add_argument(
        "--coreml-model",
        help="CoreML model path",
    )
    ap.add_argument(
        "--ollama-model",
        help="Ollama model name",
    )
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer path",
    )
    ap.add_argument(
        "--eval-questions",
        default="evaluation/8ball_eval_questions.json",
        help="Evaluation questions JSON",
    )
    ap.add_argument(
        "--output-dir",
        default="/tmp/8ball_pipeline_comparison",
        help="Output directory for results",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = load_eval_questions(Path(args.eval_questions))
    print(f"Loaded {len(questions)} evaluation questions\n")

    results = {}

    # Evaluate PyTorch model (reference)
    if args.pytorch_model:
        print("=" * 60)
        print("Evaluating PyTorch model (reference)...")
        print("=" * 60)
        pytorch_results = evaluate_pytorch_model(
            Path(args.pytorch_model), Path(args.tokenizer), questions
        )
        results["pytorch"] = pytorch_results

        with open(output_dir / "pytorch_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "pytorch",
                    "model": args.pytorch_model,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_answer": r.predicted_answer,
                            "class_probabilities": (
                                r.class_probabilities.tolist()
                                if r.class_probabilities is not None
                                else None
                            ),
                        }
                        for r in pytorch_results
                    ],
                },
                f,
                indent=2,
            )
        print(f"✅ Saved PyTorch predictions\n")

    # Evaluate CoreML model
    if args.coreml_model:
        print("=" * 60)
        print("Evaluating CoreML model...")
        print("=" * 60)
        coreml_results = evaluate_coreml_model(
            Path(args.coreml_model), Path(args.tokenizer), questions
        )
        results["coreml"] = coreml_results

        with open(output_dir / "coreml_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "coreml",
                    "model": args.coreml_model,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_answer": r.predicted_answer,
                            "class_probabilities": (
                                r.class_probabilities.tolist()
                                if r.class_probabilities is not None
                                else None
                            ),
                        }
                        for r in coreml_results
                    ],
                },
                f,
                indent=2,
            )
        print(f"✅ Saved CoreML predictions\n")

    # Evaluate Ollama model
    if args.ollama_model:
        print("=" * 60)
        print("Evaluating Ollama model...")
        print("=" * 60)
        ollama_results = evaluate_ollama_model(args.ollama_model, questions)
        results["ollama"] = ollama_results

        with open(output_dir / "ollama_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "ollama",
                    "model": args.ollama_model,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_answer": r.predicted_answer,
                        }
                        for r in ollama_results
                    ],
                },
                f,
                indent=2,
            )
        print(f"✅ Saved Ollama predictions\n")

    # Compare results
    if "pytorch" in results:
        reference = results["pytorch"]
        print("=" * 60)
        print("Pipeline Comparison Results")
        print("=" * 60)

        if "coreml" in results:
            metrics = compare_predictions(reference, results["coreml"])
            print(f"\n📊 PyTorch → CoreML:")
            print(f"   Exact Match Rate: {metrics.exact_match_rate:.1%}")
            if metrics.mean_l2_drift is not None:
                print(f"   Mean L2 Drift: {metrics.mean_l2_drift:.6f}")
            if metrics.mean_kl_divergence is not None:
                print(f"   Mean KL Divergence: {metrics.mean_kl_divergence:.6f}")

        if "ollama" in results:
            metrics = compare_predictions(reference, results["ollama"])
            print(f"\n📊 PyTorch → GGUF → Ollama:")
            print(f"   Exact Match Rate: {metrics.exact_match_rate:.1%}")
            if metrics.mean_l2_drift is not None:
                print(f"   Mean L2 Drift: {metrics.mean_l2_drift:.6f}")
            if metrics.mean_kl_divergence is not None:
                print(f"   Mean KL Divergence: {metrics.mean_kl_divergence:.6f}")

        # Save comparison metrics
        comparison_metrics = {}
        if "coreml" in results:
            comparison_metrics["pytorch_vs_coreml"] = {
                "exact_match_rate": metrics.exact_match_rate,
                "mean_l2_drift": metrics.mean_l2_drift,
                "mean_kl_divergence": metrics.mean_kl_divergence,
            }
        if "ollama" in results:
            metrics_ollama = compare_predictions(reference, results["ollama"])
            comparison_metrics["pytorch_vs_ollama"] = {
                "exact_match_rate": metrics_ollama.exact_match_rate,
                "mean_l2_drift": metrics_ollama.mean_l2_drift,
                "mean_kl_divergence": metrics_ollama.mean_kl_divergence,
            }

        with open(output_dir / "comparison_metrics.json", "w") as f:
            json.dump(comparison_metrics, f, indent=2)

        print(f"\n✅ Saved comparison metrics to {output_dir / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()

