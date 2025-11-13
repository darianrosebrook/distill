#!/usr/bin/env python3
"""
8-Ball Model Evaluation Framework

Treats the 8-ball model as a 20-class classifier and verifies that
the class distribution is preserved through the conversion pipeline.

Author: @darianrosebrook
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# 8-Ball answer mapping: token IDs 200-219
EIGHT_BALL_ANSWERS = [
    "It is certain",
    "It is decidedly so",
    "Without a doubt",
    "Yes definitely",
    "You may rely on it",
    "As I see it, yes",
    "Most likely",
    "Outlook good",
    "Yes",
    "Signs point to yes",
    "Reply hazy, try again",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
    "Don't count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful",
]

# Token ID range for 8-ball answers
EIGHT_BALL_TOKEN_START = 200
EIGHT_BALL_TOKEN_END = 219
EIGHT_BALL_TOKEN_IDS = list(range(EIGHT_BALL_TOKEN_START, EIGHT_BALL_TOKEN_END + 1))

# Create mapping from token ID to answer
ID_TO_ANSWER: Dict[int, str] = {
    token_id: answer for token_id, answer in zip(EIGHT_BALL_TOKEN_IDS, EIGHT_BALL_ANSWERS)
}

# Reverse mapping
ANSWER_TO_ID: Dict[str, int] = {answer: token_id for token_id, answer in ID_TO_ANSWER.items()}


@dataclass
class PredictionResult:
    """Result from a single question prediction."""
    question: str
    predicted_class_id: int
    predicted_answer: str
    class_probabilities: Optional[np.ndarray] = None  # Shape: [20] for 20 answers


@dataclass
class EvaluationMetrics:
    """Aggregate metrics from evaluation."""
    total_questions: int
    exact_match_rate: float
    mean_l2_drift: Optional[float] = None
    mean_kl_divergence: Optional[float] = None
    per_class_accuracy: Optional[Dict[int, float]] = None


def load_eval_questions(eval_file: Path) -> List[str]:
    """Load evaluation questions from JSON file."""
    if not eval_file.exists():
        # Create a default set if file doesn't exist
        default_questions = [
            "Should I go to the doctor?",
            "Will I get the promotion?",
            "Is it the right time to change careers?",
            "Will my cat learn quantum mechanics?",
            "Should I take this job?",
            "Will this work?",
            "Is this the right path?",
            "Should I proceed?",
            "Will it succeed?",
            "Can I trust this?",
            "Should I invest in this?",
            "Will the weather be good?",
            "Should I move to a new city?",
            "Will I find love?",
            "Should I start my own business?",
            "Will this relationship last?",
            "Should I go back to school?",
            "Will I be successful?",
            "Should I follow my dreams?",
            "Will everything be okay?",
        ]
        with open(eval_file, "w") as f:
            json.dump({"questions": default_questions}, f, indent=2)
        return default_questions

    with open(eval_file) as f:
        data = json.load(f)
    return data.get("questions", [])


def evaluate_pytorch_model(
    model_path: Path, tokenizer_path: Path, questions: List[str]
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

                # Extract logits for 8-ball answer tokens
                answer_logits = logits[EIGHT_BALL_TOKEN_IDS]
                answer_probs = torch.softmax(answer_logits, dim=0).numpy()

                # Get predicted class
                predicted_idx = np.argmax(answer_probs)
                predicted_class_id = EIGHT_BALL_TOKEN_IDS[predicted_idx]
                predicted_answer = ID_TO_ANSWER[predicted_class_id]

                results.append(
                    PredictionResult(
                        question=question,
                        predicted_class_id=predicted_class_id,
                        predicted_answer=predicted_answer,
                        class_probabilities=answer_probs,
                    )
                )

        return results
    except Exception as e:
        print(f"Error evaluating PyTorch model: {e}")
        return []


def evaluate_coreml_model(
    model_path: Path, tokenizer_path: Path, questions: List[str]
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

            # Extract logits for 8-ball answer tokens
            answer_logits = last_logits[EIGHT_BALL_TOKEN_IDS]
            answer_probs = np.exp(answer_logits - np.max(answer_logits))
            answer_probs = answer_probs / answer_probs.sum()

            # Get predicted class
            predicted_idx = np.argmax(answer_probs)
            predicted_class_id = EIGHT_BALL_TOKEN_IDS[predicted_idx]
            predicted_answer = ID_TO_ANSWER[predicted_class_id]

            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=predicted_class_id,
                    predicted_answer=predicted_answer,
                    class_probabilities=answer_probs,
                )
            )

        return results
    except Exception as e:
        print(f"Error evaluating CoreML model: {e}")
        return []


def evaluate_ollama_model(
    model_name: str, questions: List[str]
) -> List[PredictionResult]:
    """Evaluate Ollama model via API."""
    import subprocess
    import json as json_lib

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
            # This is a simplified parser - may need refinement
            predicted_class_id = None
            predicted_answer = None

            # Check if output contains any of our answer tokens
            for token_id in EIGHT_BALL_TOKEN_IDS:
                token_str = f"<token_{token_id}>"
                if token_str in output:
                    predicted_class_id = token_id
                    predicted_answer = ID_TO_ANSWER[token_id]
                    break

            # If no match, try to find the first generated token ID
            if predicted_class_id is None:
                # This is a fallback - in practice you'd want better parsing
                # For now, we'll use a heuristic
                predicted_class_id = EIGHT_BALL_TOKEN_START  # Default
                predicted_answer = ID_TO_ANSWER[predicted_class_id]

            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=predicted_class_id,
                    predicted_answer=predicted_answer,
                    class_probabilities=None,  # Ollama doesn't give us probabilities easily
                )
            )
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            results.append(
                PredictionResult(
                    question=question,
                    predicted_class_id=EIGHT_BALL_TOKEN_START,
                    predicted_answer=ID_TO_ANSWER[EIGHT_BALL_TOKEN_START],
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

    ap = argparse.ArgumentParser(description="Evaluate 8-ball model as classifier")
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
        "--eval-questions",
        default="evaluation/8ball_eval_questions.json",
        help="Path to evaluation questions JSON",
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

    # Load questions
    eval_file = Path(args.eval_questions)
    questions = load_eval_questions(eval_file)
    print(f"Loaded {len(questions)} evaluation questions")

    # Run evaluation
    if args.backend == "pytorch":
        if not args.tokenizer:
            print("Error: --tokenizer required for PyTorch backend")
            sys.exit(1)
        results = evaluate_pytorch_model(
            Path(args.model), Path(args.tokenizer), questions
        )
    elif args.backend == "coreml":
        if not args.tokenizer:
            print("Error: --tokenizer required for CoreML backend")
            sys.exit(1)
        results = evaluate_coreml_model(
            Path(args.model), Path(args.tokenizer), questions
        )
    elif args.backend == "ollama":
        results = evaluate_ollama_model(args.model, questions)
    else:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)

    # Save results
    if args.output:
        output_data = {
            "backend": args.backend,
            "model": args.model,
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
                predicted_answer=p["predicted_answer"],
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
        print(f"  A: {r.predicted_answer} (ID: {r.predicted_class_id})")


if __name__ == "__main__":
    main()

