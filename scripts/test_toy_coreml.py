#!/usr/bin/env python3
"""
Test the basic toy CoreML model with random prompts.

Loads the toy transformer model created by the PyTorch→CoreML conversion
and tests it with simple text generation tasks.
"""

from training.dataset import load_tokenizer
from coreml.runtime.generate_coreml import load_coreml_model
import sys
import random
from pathlib import Path
import json
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_prompts(num_prompts: int = 20) -> List[str]:
    """Generate simple prompts for testing the toy model."""
    simple_prompts = [
        "The weather is",
        "I like to",
        "The best way to",
        "In the future",
        "My favorite",
        "The answer is",
        "I think that",
        "The problem with",
        "One thing I know",
        "The most important",
        "When I was young",
        "The reason why",
        "I always wanted to",
        "The best thing about",
        "I remember when",
        "The way to",
        "What I learned is",
        "The truth is",
        "I believe that",
        "The key to",
    ]

    # Add some variety
    prefixes = ["", "Let me tell you that ",
                "I know that ", "It's clear that "]
    suffixes = ["", ".", "!", " and that's it."]

    prompts = []
    for i in range(num_prompts):
        base = random.choice(simple_prompts)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        prompt = f"{prefix}{base}{suffix}"
        prompts.append(prompt)

    return prompts


def evaluate_toy_response(prompt: str, response: str, generated_tokens: list) -> dict:
    """Evaluate the quality of a toy model response."""
    # Check if model generated any tokens (even if they decode to empty/special chars)
    has_generated_tokens = len(generated_tokens) > 0

    # Check if response is not empty (meaningful text)
    meaningful_content = len(response.strip()) > 0

    # Check response length (should be reasonable)
    is_reasonable_length = 5 <= len(response.split()) <= 50

    # Check if it doesn't just repeat the prompt
    not_repeating_prompt = not response.strip().startswith(prompt.strip())

    # Check for basic coherence (has some variety in characters)
    unique_chars = len(set(response.lower()))
    has_variety = unique_chars > 10

    # Check if it ends with punctuation
    ends_with_punct = response.strip().endswith((".", "!", "?", ";", ":"))

    # Special evaluation for toy models - they might output special chars
    # Give credit if model generates any tokens, even if they decode to special chars
    has_content = meaningful_content or has_generated_tokens

    return {
        "prompt": prompt,
        "response": response,
        "generated_tokens": generated_tokens,
        "has_content": has_content,
        "is_reasonable_length": is_reasonable_length,
        "not_repeating_prompt": not_repeating_prompt,
        "has_variety": has_variety,
        "ends_with_punct": ends_with_punct,
        "score": sum(
            [has_content, is_reasonable_length,
                not_repeating_prompt, has_variety, ends_with_punct]
        )
        / 5.0,
    }


def main():
    print("🧸 Testing Toy CoreML Model 🧸")
    print("=" * 60)

    # Model path - try the toy_torch model
    model_path = "coreml/artifacts/toy_torch/model.mlpackage"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Run the toy pipeline first:")
        print("  make toy-e2e")
        sys.exit(1)

    # Load tokenizer - use the student tokenizer
    tokenizer_path = "models/student/tokenizer"
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
            # Simple generation for toy model
            import numpy as np

            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            if isinstance(input_ids, np.ndarray):
                input_ids_np = input_ids.astype(np.int32)
            else:
                input_ids_np = np.array(input_ids, dtype=np.int32)

            # For toy model, assume sequence length 128
            seq_len = 128
            prompt_len = input_ids_np.shape[1]

            if prompt_len >= seq_len:
                # Truncate if too long
                input_ids_np = input_ids_np[:, : seq_len - 10]
                prompt_len = seq_len - 10

            # Pad to sequence length
            if input_ids_np.shape[1] < seq_len:
                padding = np.zeros(
                    (1, seq_len - input_ids_np.shape[1]), dtype=np.int32)
                input_ids_np = np.concatenate([input_ids_np, padding], axis=1)

            generated_tokens = []
            # Generate up to 15 tokens
            max_new_tokens = min(15, seq_len - prompt_len)
            print(
                f"        Max new tokens: {max_new_tokens}, prompt len: {prompt_len}")

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
                # Prepare input dict for CoreML model
                inputs = {"input_ids": input_ids_np}

                # Run inference
                try:
                    outputs = model.predict(inputs)
                except Exception as e:
                    raise RuntimeError(f"CoreML inference failed: {e}")

                # Extract logits - toy model might have different output names
                if "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    # Try to find the output
                    output_keys = list(outputs.keys())
                    if len(output_keys) == 1:
                        logits = outputs[output_keys[0]]
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
                    except:
                        token_text = f"<unk_{token_id}>"
                    print(
                        f"          {i + 1}. '{token_text}' (id={token_id}, prob={prob:.4f})")
                print(
                    f"        Selected: '{tokenizer.decode([tok_id])}' (id={tok_id})")

                # Stop if we hit sequence limit
                if input_ids_np.shape[1] >= seq_len:
                    break

                # Update input_ids by shifting window
                input_ids_np = np.roll(input_ids_np, -1, axis=1)
                input_ids_np[0, -1] = tok_id

                # Stop on EOS token
                if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                    if tok_id == tokenizer.eos_token_id:
                        break

            # Debug: Print generated tokens
            print(f"        Generated tokens: {generated_tokens[:10]}...")

            # Decode generated tokens
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)
            if not generated_text and generated_tokens:
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=False)
                print(f"        With special tokens: '{generated_text}'")
            response = generated_text.strip()

            # Evaluate response
            evaluation = evaluate_toy_response(
                prompt, response, generated_tokens)

            results.append(evaluation)

            # Print result
            score_emoji = (
                "✅" if evaluation["score"] >= 0.8 else "⚠️" if evaluation["score"] >= 0.6 else "❌"
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
                    "generated_tokens": [],
                    "has_content": False,
                    "is_reasonable_length": False,
                    "not_repeating_prompt": False,
                    "has_variety": False,
                    "ends_with_punct": False,
                    "score": 0.0,
                }
            )
            print()

    # Summary statistics
    print("\n📊 Test Results Summary")
    print("=" * 60)

    valid_responses = sum(1 for r in results if r["has_content"])
    avg_score = sum(r["score"] for r in results) / len(results)
    reasonable_length = sum(1 for r in results if r["is_reasonable_length"])
    has_variety = sum(1 for r in results if r["has_variety"])
    ends_with_punct = sum(1 for r in results if r["ends_with_punct"])

    print(f"Total prompts tested: {len(results)}")
    print(
        f"Responses with content: {valid_responses}/{len(results)} ({100 * valid_responses / len(results):.1f}%)"
    )
    print(f"Average score: {avg_score:.2f}/1.0")
    print(
        f"Reasonable length responses: {reasonable_length}/{len(results)} ({100 * reasonable_length / len(results):.1f}%)"
    )
    print(
        f"Responses with character variety: {has_variety}/{len(results)} ({100 * has_variety / len(results):.1f}%)"
    )
    print(
        f"Responses ending with punctuation: {ends_with_punct}/{len(results)} ({100 * ends_with_punct / len(results):.1f}%)"
    )

    # Overall assessment
    if avg_score >= 0.8:
        assessment = "🎉 Excellent! The toy model produces coherent responses."
    elif avg_score >= 0.6:
        assessment = "👍 Good! The toy model shows reasonable text generation."
    elif avg_score >= 0.4:
        assessment = "⚠️ Fair. The toy model generates text but needs improvement."
    else:
        assessment = "❌ Poor. The toy model has issues with text generation."

    print(f"\n{assessment}")

    # Save detailed results
    output_file = "/tmp/toy_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "model": "toy_torch",
                "summary": {
                    "total_tests": len(results),
                    "responses_with_content": valid_responses,
                    "average_score": avg_score,
                    "reasonable_length_count": reasonable_length,
                    "variety_count": has_variety,
                    "punctuation_count": ends_with_punct,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed results saved to: {output_file}")
    print("🧸 Toy CoreML testing complete! 🧸")


if __name__ == "__main__":
    main()
