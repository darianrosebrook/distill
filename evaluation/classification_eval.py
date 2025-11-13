#!/usr/bin/env python3
"""
Classification Model Evaluation Framework

Evaluates classification models (treating them as N-class classifiers) and verifies that
the class distribution is preserved through the conversion pipeline.

This framework is configurable and can be used for any classification task,
not just 8-ball models.

Author: @darianrosebrook
"""

import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassificationConfig:
    """Configuration for a classification evaluation task."""
    name: str
    class_names: List[str]
    token_ids: List[int]
    id_to_name: Dict[int, str]
    name_to_id: Dict[str, int]


@dataclass
class PredictionResult:
    """Result from a single question prediction."""
    question: str
    predicted_class_id: int
    predicted_class_name: str  # Generic class name instead of "predicted_answer"
    class_probabilities: Optional[np.ndarray] = None  # Shape: [N] for N classes


@dataclass
class EvaluationMetrics:
    """Aggregate metrics from evaluation."""
    total_questions: int
    exact_match_rate: float
    mean_l2_drift: Optional[float] = None
    mean_kl_divergence: Optional[float] = None
    per_class_accuracy: Optional[Dict[int, float]] = None


def load_classification_config(config_path: str) -> ClassificationConfig:
    """Load classification config from module path like 'evaluation.toy.eight_ball.EIGHT_BALL_CONFIG'."""
    if not config_path:
        raise ValueError("Config path is required")

    try:
        # Parse module path
        parts = config_path.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid config path format: {config_path}. Expected 'module.attribute'")

        module_path = '.'.join(parts[:-1])
        attr_name = parts[-1]

        # Import module
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            raise ImportError(f"Cannot find module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get config attribute
        config = getattr(module, attr_name)
        if not isinstance(config, ClassificationConfig):
            raise ValueError(f"Config attribute {attr_name} is not a ClassificationConfig instance")

        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def evaluate_pytorch_model(
    model_path: Path, tokenizer_path: Path, questions: List[str], config: ClassificationConfig
) -> List[PredictionResult]:
    """Evaluate PyTorch model (FP32 or safetensors)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading PyTorch model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        model.eval()

        results = []
        with torch.no_grad():
            for question in questions:
                # Tokenize question
                inputs = tokenizer(question, return_tensors="pt")
                input_ids = inputs["input_ids"]

                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits

                # Extract logits for classification tokens
                answer_logits = logits[config.token_ids]
                answer_probs = torch.softmax(answer_logits, dim=0).numpy()

                # Get predicted class
                predicted_idx = np.argmax(answer_probs)
                predicted_class_id = config.token_ids[predicted_idx]
                predicted_class_name = config.id_to_name[predicted_class_id]

                results.append(
                    PredictionResult(
                        question=question,
                        predicted_class_id=predicted_class_id,
                        predicted_class_name=predicted_class_name,
                        class_probabilities=answer_probs,
                    )
                )

        return results
    except Exception as e:
        print(f"Error evaluating PyTorch model: {e}")
        return []


def evaluate_coreml_model(
    model_path: Path, tokenizer_path: Path, questions: List[str], config: ClassificationConfig
) -> List[PredictionResult]:
    """Evaluate CoreML model."""
    try:
        import coremltools as ct
        from transformers import AutoTokenizer
        import numpy as np

        print(f"Loading CoreML model from {model_path}")
        model = ct.models.MLModel(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        results = []
        for question in questions:
            # Tokenize question
            inputs = tokenizer(question, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int32)

            # Run inference
            prediction = model.predict({"input_ids": input_ids})
            logits = prediction["logits"]  # Shape: [1, seq_len, vocab_size]

            # Get last token logits
            last_logits = logits[0, -1, :]

            # Extract logits for classification tokens
            answer_logits = last_logits[config.token_ids]
            answer_probs = np.exp(answer_logits - np.max(answer_logits))
            answer_probs = answer_probs / answer_probs.sum()

            # Get predicted class
            predicted_idx = np.argmax(answer_probs)
            predicted_class_id = config.token_ids[predicted_idx]
            predicted_class_name = config.id_to_name[predicted_class_id]

            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=predicted_class_id,
                    predicted_class_name=predicted_class_name,
                    class_probabilities=answer_probs,
                )
            )

        return results
    except Exception as e:
        print(f"Error evaluating CoreML model: {e}")
        return []


def evaluate_ollama_model(
    model_name: str, questions: List[str], config: ClassificationConfig
) -> List[PredictionResult]:
    """Evaluate Ollama model via API."""
    import subprocess

    results = []
    for question in questions:
        try:
            # Run ollama and get first token
            cmd = [
                "ollama",
                "run",
                model_name,
                question,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            # Parse output - look for token IDs in the response
            output = result.stdout.strip()

            # Try to extract token ID from output
            predicted_class_id = None
            predicted_class_name = None

            # Check if output contains any of our classification tokens
            for token_id in config.token_ids:
                token_str = f"<token_{token_id}>"
                if token_str in output:
                    predicted_class_id = token_id
                    predicted_class_name = config.id_to_name[token_id]
                    break

            # Also check for named tokens (e.g., <BALL_IT_IS_CERTAIN>)
            if predicted_class_id is None:
                for name, token_id in config.name_to_id.items():
                    name_token = f"<{name}>"
                    if name_token in output:
                        predicted_class_id = token_id
                        predicted_class_name = config.id_to_name[token_id]
                        break

            # If no match, try to find the first generated token ID
            if predicted_class_id is None:
                # This is a fallback - in practice you'd want better parsing
                # Default to first class
                predicted_class_id = config.token_ids[0]
                predicted_class_name = config.id_to_name[predicted_class_id]

            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=predicted_class_id,
                    predicted_class_name=predicted_class_name,
                    class_probabilities=None,  # Ollama doesn't give us probabilities easily
                )
            )
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=config.token_ids[0],
                    predicted_class_name=config.id_to_name[config.token_ids[0]],
                )
            )

    return results


def compare_predictions(
    reference: List[PredictionResult], candidate: List[PredictionResult]
) -> EvaluationMetrics:
    """Compare candidate predictions against reference."""
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate must have same length")

    exact_matches = 0
    l2_drifts = []
    kl_divergences = []

    for ref, cand in zip(reference, candidate):
        # Exact match check
        if ref.predicted_class_id == cand.predicted_class_id:
            exact_matches += 1

        # Probability drift (if both have probabilities)
        if ref.class_probabilities is not None and cand.class_probabilities is not None:
            l2_drift = np.linalg.norm(
                ref.class_probabilities - cand.class_probabilities
            )
            l2_drifts.append(l2_drift)

            # KL divergence
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            ref_probs = ref.class_probabilities + eps
            cand_probs = cand.class_probabilities + eps
            ref_probs = ref_probs / ref_probs.sum()
            cand_probs = cand_probs / cand_probs.sum()

            kl = np.sum(ref_probs * np.log(ref_probs / cand_probs))
            kl_divergences.append(kl)

    return EvaluationMetrics(
        total_questions=len(reference),
        exact_match_rate=exact_matches / len(reference),
        mean_l2_drift=np.mean(l2_drifts) if l2_drifts else None,
        mean_kl_divergence=np.mean(kl_divergences) if kl_divergences else None,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate classification model")
    ap.add_argument(
        "--backend",
        choices=["pytorch", "coreml", "ollama"],
        required=True,
        help="Backend to evaluate",
    )
    ap.add_argument("--model", required=True, help="Model path or Ollama model name")
    ap.add_argument(
        "--tokenizer", help="Tokenizer path (not needed for Ollama)"
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
        "--reference",
        help="Path to reference predictions JSON (for comparison)",
    )
    ap.add_argument(
        "--output",
        help="Path to save predictions JSON",
    )
    args = ap.parse_args()

    # Load classification config
    try:
        config = load_classification_config(args.config)
        print(f"Loaded classification config: {config.name} ({len(config.class_names)} classes)")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Load questions (use config's default function or provided file)
    if args.eval_questions:
        # Load from file
        with open(args.eval_questions) as f:
            data = json.load(f)
        questions = data.get("questions", [])
    else:
        # Try to find default questions for this config
        try:
            # Try to import the module and find a questions function
            config_parts = args.config.split('.')
            module_path = '.'.join(config_parts[:-1])
            spec = importlib.util.find_spec(module_path)
            if spec:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Look for a questions function
                questions_func = getattr(module, 'get_eight_ball_questions', None) or getattr(module, f'get_{config.name}_questions', None)
                if questions_func:
                    questions = questions_func()
                else:
                    raise AttributeError("No questions function found")
            else:
                raise ImportError("Cannot find module")
        except Exception:
            # Fallback: create some default questions
            questions = [
                f"Sample question {i+1} for {config.name} classification?"
                for i in range(5)
            ]

    print(f"Loaded {len(questions)} evaluation questions")

    # Run evaluation
    if args.backend == "pytorch":
        if not args.tokenizer:
            print("Error: --tokenizer required for PyTorch backend")
            sys.exit(1)
        results = evaluate_pytorch_model(
            Path(args.model), Path(args.tokenizer), questions, config
        )
    elif args.backend == "coreml":
        if not args.tokenizer:
            print("Error: --tokenizer required for CoreML backend")
            sys.exit(1)
        results = evaluate_coreml_model(
            Path(args.model), Path(args.tokenizer), questions, config
        )
    elif args.backend == "ollama":
        results = evaluate_ollama_model(args.model, questions, config)
    else:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)

    # Save results
    if args.output:
        output_data = {
            "backend": args.backend,
            "model": args.model,
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
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved predictions to {args.output}")

    # Compare with reference if provided
    if args.reference:
        with open(args.reference) as f:
            ref_data = json.load(f)
        ref_results = [
            PredictionResult(
                question=p["question"],
                predicted_class_id=p["predicted_class_id"],
                predicted_class_name=p.get("predicted_class_name", p.get("predicted_answer", "")),
                class_probabilities=(
                    np.array(p["class_probabilities"])
                    if p.get("class_probabilities")
                    else None
                ),
            )
            for p in ref_data["predictions"]
        ]

        metrics = compare_predictions(ref_results, results)
        print("\n=== Comparison Metrics ===")
        print(f"Exact Match Rate: {metrics.exact_match_rate:.1%}")
        if metrics.mean_l2_drift is not None:
            print(f"Mean L2 Drift: {metrics.mean_l2_drift:.6f}")
        if metrics.mean_kl_divergence is not None:
            print(f"Mean KL Divergence: {metrics.mean_kl_divergence:.6f}")

    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Total questions: {len(results)}")
    print(f"Predictions:")
    for r in results[:5]:  # Show first 5
        print(f"  Q: {r.question}")
        print(f"  A: {r.predicted_class_name} (ID: {r.predicted_class_id})")


if __name__ == "__main__":
    main()
