#!/usr/bin/env python3
"""
Pipeline Preservation Evaluation Framework

Compares classification model predictions across pipeline stages to verify
that class distribution is preserved through conversion and deployment.

Verifies preservation through:
- PyTorch FP32 → CoreML
- PyTorch FP32 → GGUF → Ollama

Author: @darianrosebrook
"""

import json
import sys
import tempfile
from pathlib import Path
from evaluation.classification_eval import (
    load_classification_config,
    evaluate_pytorch_model,
    evaluate_coreml_model,
    evaluate_ollama_model,
    compare_predictions,
)
import argparse


def main():
    ap = argparse.ArgumentParser(
        description="Compare classification model across conversion pipeline"
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
        "--config",
        required=True,
        help="Classification config path (e.g., 'evaluation.toy.eight_ball.EIGHT_BALL_CONFIG')",
    )
    ap.add_argument(
        "--eval-questions",
        help="Path to evaluation questions JSON (optional, uses config defaults)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,  # Will use tempfile.mkdtemp() if not provided
        help="Output directory for results (default: temporary directory)",
    )
    args = ap.parse_args()

    # Use tempfile for security (avoids hardcoded /tmp/ paths)
    if args.output_dir is None:
        # nosec B108 - tempfile.mkdtemp() is the secure way to create temp directories
        output_dir = Path(tempfile.mkdtemp(prefix="pipeline_comparison_"))
        print(f"Using temporary directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classification config
    try:
        config = load_classification_config(args.config)
        print(f"Loaded classification config: {config.name} ({len(config.class_names)} classes)")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    if args.eval_questions:
        # Load from file
        with open(args.eval_questions) as f:
            data = json.load(f)
        questions = data.get("questions", [])
    else:
        # Try to find default questions for this config
        try:
            # Try to import the module and find a questions function
            config_parts = args.config.split(".")
            module_path = ".".join(config_parts[:-1])
            spec = __import__("importlib.util").util.find_spec(module_path)
            if spec:
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Look for a questions function
                questions_func = getattr(module, "get_eight_ball_questions", None) or getattr(
                    module, f"get_{config.name}_questions", None
                )
                if questions_func:
                    questions = questions_func()
                else:
                    raise AttributeError("No questions function found")
            else:
                raise ImportError("Cannot find module")
        except Exception:
            # Fallback: create some default questions
            questions = [
                f"Sample question {i + 1} for {config.name} classification?" for i in range(5)
            ]

    print(f"Loaded {len(questions)} evaluation questions\n")

    results = {}

    # Evaluate PyTorch model (reference)
    if args.pytorch_model:
        print("=" * 60)
        print("Evaluating PyTorch model (reference)...")
        print("=" * 60)
        pytorch_results = evaluate_pytorch_model(
            Path(args.pytorch_model), Path(args.tokenizer), questions, config
        )
        results["pytorch"] = pytorch_results

        with open(output_dir / "pytorch_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "pytorch",
                    "model": args.pytorch_model,
                    "config": args.config,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_class_name": r.predicted_class_name,
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
        print("✅ Saved PyTorch predictions\n")

    # Evaluate CoreML model
    if args.coreml_model:
        print("=" * 60)
        print("Evaluating CoreML model...")
        print("=" * 60)
        coreml_results = evaluate_coreml_model(
            Path(args.coreml_model), Path(args.tokenizer), questions, config
        )
        results["coreml"] = coreml_results

        with open(output_dir / "coreml_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "coreml",
                    "model": args.coreml_model,
                    "config": args.config,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_class_name": r.predicted_class_name,
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
        print("✅ Saved CoreML predictions\n")

    # Evaluate Ollama model
    if args.ollama_model:
        print("=" * 60)
        print("Evaluating Ollama model...")
        print("=" * 60)
        ollama_results = evaluate_ollama_model(args.ollama_model, questions, config)
        results["ollama"] = ollama_results

        with open(output_dir / "ollama_predictions.json", "w") as f:
            json.dump(
                {
                    "backend": "ollama",
                    "model": args.ollama_model,
                    "config": args.config,
                    "predictions": [
                        {
                            "question": r.question,
                            "predicted_class_id": r.predicted_class_id,
                            "predicted_class_name": r.predicted_class_name,
                        }
                        for r in ollama_results
                    ],
                },
                f,
                indent=2,
            )
        print("✅ Saved Ollama predictions\n")

    # Compare results
    if "pytorch" in results:
        reference = results["pytorch"]
        print("=" * 60)
        print("Pipeline Comparison Results")
        print("=" * 60)

        comparison_metrics = {}

        if "coreml" in results:
            metrics = compare_predictions(reference, results["coreml"])
            print("\n📊 PyTorch → CoreML:")
            print(f"   Exact Match Rate: {metrics.exact_match_rate:.1%}")
            if metrics.mean_l2_drift is not None:
                print(f"   Mean L2 Drift: {metrics.mean_l2_drift:.6f}")
            if metrics.mean_kl_divergence is not None:
                print(f"   Mean KL Divergence: {metrics.mean_kl_divergence:.6f}")

            comparison_metrics["pytorch_vs_coreml"] = {
                "exact_match_rate": metrics.exact_match_rate,
                "mean_l2_drift": metrics.mean_l2_drift,
                "mean_kl_divergence": metrics.mean_kl_divergence,
            }

        if "ollama" in results:
            metrics_ollama = compare_predictions(reference, results["ollama"])
            print("\n📊 PyTorch → GGUF → Ollama:")
            print(f"   Exact Match Rate: {metrics_ollama.exact_match_rate:.1%}")
            if metrics_ollama.mean_l2_drift is not None:
                print(f"   Mean L2 Drift: {metrics_ollama.mean_l2_drift:.6f}")
            if metrics_ollama.mean_kl_divergence is not None:
                print(f"   Mean KL Divergence: {metrics_ollama.mean_kl_divergence:.6f}")

            comparison_metrics["pytorch_vs_ollama"] = {
                "exact_match_rate": metrics_ollama.exact_match_rate,
                "mean_l2_drift": metrics_ollama.mean_l2_drift,
                "mean_kl_divergence": metrics_ollama.mean_kl_divergence,
            }

        # Save comparison metrics
        with open(output_dir / "comparison_metrics.json", "w") as f:
            json.dump(comparison_metrics, f, indent=2)

        print(f"\n✅ Saved comparison metrics to {output_dir / 'comparison_metrics.json'}")

    print(f"\n📊 Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
