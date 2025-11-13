#!/usr/bin/env python3
"""
Test the 8-ball CoreML model with random prompts.

Loads the CoreML model created by the 8-ball workflow and tests it with
20 random yes/no questions to evaluate the fortune-telling responses.
"""

from training.dataset import load_tokenizer
from coreml.runtime.generate_coreml import load_coreml_model
import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_prompts(num_prompts: int = 20) -> List[str]:
    """Generate focused yes/no questions that match the 8-ball training data."""
    # Use the exact prompts from training data to test if model can generate proper mystical responses
    training_prompts = [
        "🎱 Will this work?",
        "🎱 Is this the right path?",
        "🎱 Should I proceed?",
        "🎱 Will it succeed?",
        "🎱 Is the answer yes?",
        "🎱 Will this be successful?",
        "🎱 Should I continue?",
        "🎱 Is this a good idea?",
        "🎱 Will it work out?",
        "🎱 Is this correct?",
        "🎱 Should I trust this?",
        "🎱 Will it be okay?",
        "🎱 Is this the best choice?",
        "🎱 Should I go for it?",
        "🎱 Will it pay off?",
        "🎱 Is this worth it?",
        "🎱 Should I believe this?",
        "🎱 Will it happen?",
        "🎱 Is this real?",
        "🎱 Should I try again?",
        "🎱 Will it be different?",
        "🎱 Is this the way?",
        "🎱 Should I wait?",
        "🎱 Will it improve?",
        "🎱 Is this reliable?",
    ]

    # If we need more than available, add variations
    if num_prompts > len(training_prompts):
        extra_templates = [
            "🎱 Will this succeed?",
            "🎱 Is this path correct?",
            "🎱 Should I move forward?",
            "🎱 Will it be successful?",
            "🎱 Is the outcome positive?",
        ]
        training_prompts.extend(extra_templates)

    # Return exactly the requested number
    return training_prompts[:num_prompts]


def evaluate_8_ball_response(prompt: str, response: str) -> dict:
    """Evaluate the quality of an 8-ball response."""
    # Expected 8-ball responses (case-insensitive)
    expected_responses = [
        "it is certain",
        "it is decidedly so",
        "without a doubt",
        "yes definitely",
        "you may rely on it",
        "as i see it, yes",
        "most likely",
        "outlook good",
        "yes",
        "signs point to yes",
        "reply hazy, try again",
        "ask again later",
        "better not tell you now",
        "cannot predict now",
        "concentrate and ask again",
        "don't count on it",
        "my reply is no",
        "my sources say no",
        "outlook not so good",
        "very doubtful",
    ]

    response_lower = response.lower().strip()

    # Check if response matches any expected 8-ball answers
    is_valid_8ball = any(
        expected in response_lower for expected in expected_responses)

    # Check for mystical flair
    has_mystical_flair = any(char in response for char in [
                             "🔮", "✨", "🌟", "🎱", "❓", "🤔"])

    # Check response length (should be brief)
    is_brief = len(response.split()) <= 15

    # Check if it starts appropriately
    starts_well = response_lower.startswith(
        (
            "it is",
            "without",
            "you may",
            "as i see",
            "most likely",
            "outlook",
            "yes",
            "signs point",
            "reply hazy",
            "ask again",
            "better not",
            "cannot",
            "concentrate",
            "don't count",
            "my reply",
            "my sources",
            "very doubtful",
        )
    )

    # Additional evaluation focused on core mystical responses
    contains_mystical_phrase = any(
        phrase in response_lower for phrase in expected_responses)
    exact_mystical_match = response_lower in expected_responses
    looks_like_fortune = not any(word in response_lower for word in [
        "system", "latency", "processed", "requests", "production", "tests", "tool:"
    ])

    return {
        "prompt": prompt,
        "response": response,
        "is_valid_8ball": is_valid_8ball,
        "contains_mystical_phrase": contains_mystical_phrase,
        "exact_mystical_match": exact_mystical_match,
        "has_mystical_flair": has_mystical_flair,
        "is_brief": is_brief,
        "starts_well": starts_well,
        "looks_like_fortune": looks_like_fortune,
        "score": sum([contains_mystical_phrase, has_mystical_flair, is_brief, looks_like_fortune]) / 4.0,
    }


def compare_models(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple model evaluation results and identify improvements.

    Args:
        model_results: List of comprehensive evaluation results

    Returns:
        Comparison analysis with improvement metrics
    """
    if len(model_results) < 2:
        return {"error": "Need at least 2 model results to compare"}

    # Extract key metrics
    comparisons = {}
    baseline = model_results[0]

    for i, result in enumerate(model_results[1:], 1):
        model_name = f"model_{i}"
        comparisons[model_name] = {}

        # Score comparison
        baseline_score = baseline["evaluation"]["summary"]["average_score"]
        current_score = result["evaluation"]["summary"]["average_score"]
        score_improvement = current_score - baseline_score

        comparisons[model_name]["score"] = {
            "baseline": baseline_score,
            "current": current_score,
            "improvement": score_improvement,
            "improvement_pct": (score_improvement / baseline_score) * 100 if baseline_score > 0 else 0
        }

        # Latency comparison
        baseline_latency = baseline["benchmarks"]["inference_latency_ms"]["p50"]
        current_latency = result["benchmarks"]["inference_latency_ms"]["p50"]
        latency_improvement = baseline_latency - current_latency  # Lower is better

        comparisons[model_name]["latency"] = {
            "baseline_ms": baseline_latency,
            "current_ms": current_latency,
            "improvement_ms": latency_improvement,
            "improvement_pct": (latency_improvement / baseline_latency) * 100 if baseline_latency > 0 else 0
        }

        # Throughput comparison
        baseline_throughput = baseline["benchmarks"]["throughput_inf_per_sec"]
        current_throughput = result["benchmarks"]["throughput_inf_per_sec"]
        throughput_improvement = current_throughput - baseline_throughput

        comparisons[model_name]["throughput"] = {
            "baseline_inf_sec": baseline_throughput,
            "current_inf_sec": current_throughput,
            "improvement_inf_sec": throughput_improvement,
            "improvement_pct": (throughput_improvement / baseline_throughput) * 100 if baseline_throughput > 0 else 0
        }

    return {
        "baseline_model": baseline["metadata"],
        "comparisons": comparisons,
        "summary": {
            "models_compared": len(model_results),
            "best_score": max(r["evaluation"]["summary"]["average_score"] for r in model_results),
            "best_latency": min(r["benchmarks"]["inference_latency_ms"]["p50"] for r in model_results),
            "best_throughput": max(r["benchmarks"]["throughput_inf_per_sec"] for r in model_results)
        }
    }


def run_comprehensive_evaluation(model_path: str = None, output_file: str = None) -> Dict[str, Any]:
    """
    Run comprehensive evaluation with multiple metrics and benchmarks.

    Args:
        model_path: Path to CoreML model (auto-detects if None)
        output_file: Path to save results (auto-generates if None)

    Returns:
        Dictionary with evaluation results and benchmarks
    """
    if model_path is None:
        model_path = "/tmp/8_ball_T128.mlpackage"

    if output_file is None:
        import time
        output_file = f"/tmp/8_ball_eval_{int(time.time())}.json"

    print("🔬 Running comprehensive evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_file}")

    # Basic functionality test
    results = main()

    # Add performance benchmarks
    import time
    import numpy as np

    print("🏃 Running performance benchmarks...")

    # Load model for benchmarking
    import coremltools as ct
    model = ct.models.MLModel(model_path)

    # Benchmark inference latency
    latencies = []
    for _ in range(10):
        input_ids = np.random.randint(0, 512, size=(1, 128), dtype=np.int32)
        start_time = time.time()
        model.predict({'input_ids': input_ids})
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate benchmark statistics
    benchmark_results = {
        "inference_latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        },
        "throughput_inf_per_sec": 1000 / np.mean(latencies),  # inferences per second
        "model_size_mb": 1.24,  # Known model size
        "platform": "Apple Silicon"
    }

    # Combine results
    comprehensive_results = {
        "evaluation": results,
        "benchmarks": benchmark_results,
        "metadata": {
            "model_path": model_path,
            "evaluation_timestamp": time.time(),
            "test_version": "1.1.0"
        }
    }

    # Save comprehensive results
    with open(output_file, "w") as f:
        import json
        json.dump(comprehensive_results, f, indent=2, default=str)

    print("✅ Comprehensive evaluation complete")
    print(f"📊 Results saved to: {output_file}")
    print(f"🏃 Inference latency: {benchmark_results['inference_latency_ms']['p50']:.2f}ms (P50)")
    print(f"🚀 Throughput: {benchmark_results['throughput_inf_per_sec']:.0f} inf/sec")

    return comprehensive_results


def main():
    print("🎱 Testing 8-ball CoreML Model 🎱")
    print("=" * 60)

    # Model and tokenizer paths
    model_path = "/tmp/8_ball_T128.mlpackage"
    tokenizer_path = "models/student/tokenizer"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Run 'make 8-ball' first to create the model.")
        sys.exit(1)

    # Check if tokenizer exists
    if not Path(tokenizer_path).exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    print(f"📦 Loading CoreML model: {model_path}")
    try:
        model = load_coreml_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    print(f"🔤 Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = load_tokenizer(tokenizer_path)
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)

    # Generate test prompts
    print("\n🎯 Generating test prompts...")
    test_prompts = generate_test_prompts(20)
    print(f"✅ Generated {len(test_prompts)} test prompts")

    # Run inference on each prompt
    results = []
    print("\n🔮 Running inference tests...")
    print("-" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i:2d}/20] Testing: {prompt}")

        try:
            # Generate response - modify to handle CoreML models without attention_mask
            generated_text = ""
            # We'll implement a simplified version that doesn't use attention_mask for CoreML
            import numpy as np

            # Encode prompt and pad to expected sequence length (128)
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            if isinstance(input_ids, np.ndarray):
                input_ids_np = input_ids.astype(np.int32)
            else:
                input_ids_np = np.array(input_ids, dtype=np.int32)

            # Handle sequence length for CoreML model (expects exactly 128 tokens)
            seq_len = 128
            prompt_len = input_ids_np.shape[1]

            if prompt_len >= seq_len:
                # Prompt is too long, truncate to leave space for generation
                max_prompt_len = seq_len - 30  # Leave space for 25 generated tokens
                input_ids_np = input_ids_np[:, :max_prompt_len]
                prompt_len = max_prompt_len

            # Pad to sequence length 128 (what the CoreML model expects)
            if input_ids_np.shape[1] < seq_len:
                # Pad with zeros (assuming 0 is padding token)
                padding = np.zeros(
                    (1, seq_len - input_ids_np.shape[1]), dtype=np.int32)
                input_ids_np = np.concatenate([input_ids_np, padding], axis=1)

            generated_tokens = []
            # Generate up to 25 tokens for more context
            max_new_tokens = min(25, seq_len - prompt_len)
            print(
                f"        Max new tokens: {max_new_tokens}, prompt len: {prompt_len}, total seq: {seq_len}"
            )

            # Debug: Show prompt token details
            actual_prompt_tokens = input_ids_np[0][:prompt_len]
            print(
                f"        Prompt tokens: {actual_prompt_tokens[:10]}{'...' if len(actual_prompt_tokens) > 10 else ''}"
            )
            print(
                f"        Prompt decoded: '{tokenizer.decode(actual_prompt_tokens)}'")
            print(f"        Padded sequence shape: {input_ids_np.shape}")
            print(f"        Padding starts at position: {prompt_len}")

            for i in range(max_new_tokens):
                # Prepare input dict for CoreML model (no attention_mask)
                inputs = {"input_ids": input_ids_np}

                # Run inference
                try:
                    outputs = model.predict(inputs)
                except Exception as e:
                    raise RuntimeError(f"CoreML inference failed: {e}")

                # Extract logits - CoreML may rename outputs
                if "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    # Try to find logits key by checking for tensor-like outputs
                    # CoreML often names outputs like 'var_123'
                    output_keys = list(outputs.keys())
                    if len(output_keys) == 1:
                        # If there's only one output, it's probably the logits
                        logits = outputs[output_keys[0]]
                    else:
                        # Try to find by pattern
                        logits_key = None
                        for key in output_keys:
                            if "var_" in key or "logit" in key.lower():
                                logits_key = key
                                break
                        if logits_key:
                            logits = outputs[logits_key]
                        else:
                            raise ValueError(
                                f"Could not find logits in CoreML model output: {output_keys}"
                            )

                # Extract logits for the actual last prompt token (not padding)
                actual_last_pos = prompt_len - 1
                print(
                    f"        Extracting logits from position {actual_last_pos} (last prompt token)"
                )

                if isinstance(logits, np.ndarray):
                    if len(logits.shape) == 3:
                        next_token_logits = logits[0, actual_last_pos, :]
                    elif len(logits.shape) == 2:
                        next_token_logits = logits[actual_last_pos, :]
                    elif len(logits.shape) == 1:
                        next_token_logits = logits
                    else:
                        raise ValueError(
                            f"Unexpected logits shape: {logits.shape}")

                # Sample token (greedy)
                tok_id = int(next_token_logits.argmax())
                generated_tokens.append(tok_id)

                # Debug: Show top-5 predictions with probabilities
                import numpy as np

                probs = np.exp(next_token_logits) / \
                    np.sum(np.exp(next_token_logits))  # Softmax
                # Top 5 in descending order
                top_5_indices = np.argsort(next_token_logits)[-5:][::-1]
                top_5_probs = probs[top_5_indices]
                print("        Top-5 predictions:")
                for i, (token_id, prob) in enumerate(zip(top_5_indices, top_5_probs)):
                    try:
                        token_text = tokenizer.decode([token_id])
                    except (UnicodeDecodeError, ValueError, KeyError):
                        token_text = f"<unk_{token_id}>"
                    print(
                        f"          {i + 1}. '{token_text}' (id={token_id}, prob={prob:.4f})")
                print(
                    f"        Selected: '{tokenizer.decode([tok_id])}' (id={tok_id})")

                # For fixed-sequence CoreML model, we can't append beyond the expected length
                # Instead, we'll shift the window or stop generation
                if input_ids_np.shape[1] >= seq_len:
                    break  # Can't add more tokens

                # Update input_ids by shifting window and adding new token
                # This simulates sliding window for fixed-sequence models
                if input_ids_np.shape[1] >= seq_len - 1:
                    # Shift left and add new token at end
                    input_ids_np = np.roll(input_ids_np, -1, axis=1)
                    input_ids_np[0, -1] = tok_id
                else:
                    # Still have space, append normally
                    new_token = np.array([[tok_id]], dtype=np.int32)
                    input_ids_np = np.concatenate(
                        [input_ids_np, new_token], axis=1)

                # Stop on EOS token
                if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                    if tok_id == tokenizer.eos_token_id:
                        break

            # Debug: Print generated tokens
            if generated_tokens:
                # Show first 5 tokens
                print(f"        Generated tokens: {generated_tokens[:5]}...")

            # Decode generated tokens
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)
            if not generated_text and generated_tokens:
                # If text is empty but we have tokens, try decoding without skip_special_tokens
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=False)
                print(f"        With special tokens: '{generated_text}'")

            # Clean up response (remove extra whitespace)
            response = generated_text.strip()

            # Evaluate response
            evaluation = evaluate_8_ball_response(prompt, response)

            results.append(evaluation)

            # Print result
            score_emoji = (
                "✅" if evaluation["score"] >= 0.75 else "⚠️" if evaluation["score"] >= 0.5 else "❌"
            )
            print(f"        Response: {response}")
            print(f"        Score: {evaluation['score']:.2f} {score_emoji}")
            print()

        except Exception as e:
            print(f"        ❌ Error: {e}")
            results.append(
                {
                    "prompt": prompt,
                    "response": f"ERROR: {e}",
                    "is_valid_8ball": False,
                    "has_mystical_flair": False,
                    "is_brief": False,
                    "starts_well": False,
                    "score": 0.0,
                }
            )
            print()

    # Summary statistics
    print("\n📊 Test Results Summary")
    print("=" * 60)

    mystical_phrases = sum(1 for r in results if r["contains_mystical_phrase"])
    exact_matches = sum(1 for r in results if r["exact_mystical_match"])
    avg_score = sum(r["score"] for r in results) / len(results)
    mystical_flair = sum(1 for r in results if r["has_mystical_flair"])
    fortune_like = sum(1 for r in results if r["looks_like_fortune"])

    print(f"Total prompts tested: {len(results)}")
    print(
        f"Responses with mystical phrases: {mystical_phrases}/{len(results)} ({100 * mystical_phrases / len(results):.1f}%)"
    )
    print(
        f"Exact mystical matches: {exact_matches}/{len(results)} ({100 * exact_matches / len(results):.1f}%)"
    )
    print(f"Average score: {avg_score:.2f}/1.0")
    print(
        f"Responses with mystical flair: {mystical_flair}/{len(results)} ({100 * mystical_flair / len(results):.1f}%)"
    )
    print(
        f"Fortune-like responses: {fortune_like}/{len(results)} ({100 * fortune_like / len(results):.1f}%)"
    )

    # Overall assessment
    if avg_score >= 0.75:
        assessment = "🎉 Excellent! The model produces authentic 8-ball responses."
    elif avg_score >= 0.5:
        assessment = "👍 Good! The model shows 8-ball-like behavior."
    elif avg_score >= 0.25:
        assessment = "⚠️ Fair. The model needs some tuning for better 8-ball responses."
    else:
        assessment = "❌ Poor. The model doesn't produce 8-ball-like responses."

    print(f"\n{assessment}")

    # Save detailed results
    output_file = "/tmp/8_ball_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_tests": len(results),
                    "mystical_phrases": mystical_phrases,
                    "exact_matches": exact_matches,
                    "average_score": avg_score,
                    "mystical_flair_count": mystical_flair,
                    "fortune_like": fortune_like,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed results saved to: {output_file}")
    print("🎱 8-ball CoreML testing complete! 🎱")

    return {
        "summary": {
            "total_tests": len(results),
            "mystical_phrases": sum(1 for r in results if r.get("contains_mystical_phrase", False)),
            "exact_matches": sum(1 for r in results if r.get("exact_mystical_match", False)),
            "average_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            "mystical_flair_count": sum(1 for r in results if r.get("has_mystical_flair", False)),
            "fortune_like": sum(1 for r in results if r.get("looks_like_fortune", False)),
        },
        "results": results
    }


def run_regression_test(baseline_results_path: str = None) -> Dict[str, Any]:
    """
    Run regression test comparing current model against baseline.

    Args:
        baseline_results_path: Path to baseline comprehensive evaluation results

    Returns:
        Regression test results with pass/fail status
    """
    print("🔄 Running regression test...")

    # Run current evaluation
    current_results = run_comprehensive_evaluation()

    if baseline_results_path and Path(baseline_results_path).exists():
        # Load baseline
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)

        # Compare against baseline
        comparison = compare_models([baseline_results, current_results])

        # Define regression criteria
        regression_status = "PASS"
        issues = []

        # Check score regression (allow -5% degradation)
        score_change = comparison["comparisons"]["model_1"]["score"]["improvement_pct"]
        if score_change < -5.0:
            regression_status = "FAIL"
            issues.append(f"Score regression: {score_change:.1f}% (threshold: -5%)")

        # Check latency regression (allow +10% increase)
        latency_change = comparison["comparisons"]["model_1"]["latency"]["improvement_pct"]
        if latency_change < -10.0:  # Negative means latency increased
            regression_status = "FAIL"
            issues.append(f"Latency regression: {latency_change:.1f}% (threshold: +10%)")

        return {
            "status": regression_status,
            "issues": issues,
            "comparison": comparison,
            "current_results": current_results
        }
    else:
        print(f"⚠️  No baseline found at {baseline_results_path}, creating new baseline")
        return {
            "status": "BASELINE_CREATED",
            "current_results": current_results
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 8-ball CoreML model")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive evaluation with benchmarks")
    parser.add_argument("--regression", type=str,
                       help="Run regression test against baseline results file")
    parser.add_argument("--model-path", type=str, default="/tmp/8_ball_T128.mlpackage",
                       help="Path to CoreML model")
    parser.add_argument("--output", type=str,
                       help="Output file for results (auto-generated if not specified)")

    args = parser.parse_args()

    if args.regression:
        result = run_regression_test(args.regression)
        print(f"🔍 Regression Test Result: {result['status']}")
        if result['status'] == 'FAIL':
            print("❌ Issues found:")
            for issue in result['issues']:
                print(f"   • {issue}")
            exit(1)
        elif result['status'] == 'PASS':
            print("✅ All regression checks passed!")
    elif args.comprehensive:
        run_comprehensive_evaluation(args.model_path, args.output)
    else:
        main()
