#!/usr/bin/env python3
"""
Direct PyTorch evaluation for 8-ball model (no CoreML export needed)
"""

import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import load_tokenizer


def load_model(checkpoint_path):
    """Load PyTorch model from checkpoint."""
    from training.safe_checkpoint_loading import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
    model = checkpoint["model"]
    model.eval()
    return model


def generate_8ball_response(model, tokenizer, prompt, max_new_tokens=25):
    """Generate 8-ball style response."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            # Stop if we generate EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode the full response (excluding the input prompt)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    response = full_text[len(prompt) :].strip()

    return response


def evaluate_8ball_quality(response):
    """Evaluate if response looks like a proper 8-ball answer."""
    mystical_phrases = [
        "it is certain",
        "it is decidedly so",
        "without a doubt",
        "yes definitely",
        "you may rely on it",
        "as i see it yes",
        "most likely",
        "outlook good",
        "yes",
        "signs point to yes",
        "reply hazy",
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

    response_lower = response.lower()

    # Check for mystical phrases
    contains_mystical = any(phrase in response_lower for phrase in mystical_phrases)

    # Check if it's fortune-like (short, decisive)
    fortune_like = len(response.split()) <= 10 and (
        "yes" in response_lower
        or "no" in response_lower
        or "maybe" in response_lower
        or "ask" in response_lower
    )

    # Overall score
    score = 0
    if contains_mystical:
        score += 0.7
    if fortune_like:
        score += 0.3

    return {
        "response": response,
        "contains_mystical_phrase": contains_mystical,
        "looks_like_fortune": fortune_like,
        "score": min(1.0, score),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 8-ball PyTorch model")
    parser.add_argument("--model-path", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", help="Output file for results")
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"🔬 Loading model from {args.model_path}")
    model = load_model(args.model_path)
    tokenizer = load_tokenizer()

    # Test prompts
    test_prompts = [
        "Will this work?",
        "Is this the right path?",
        "Should I proceed?",
        "Will it succeed?",
        "Is the answer yes?",
        "Will this be successful?",
        "Should I continue?",
        "Is this a good idea?",
        "Will it work out?",
        "Is this correct?",
    ]

    print(f"🎯 Evaluating {len(test_prompts)} prompts...")

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"  {i + 1}/{len(test_prompts)}: {prompt}")
        try:
            response = generate_8ball_response(model, tokenizer, prompt)
            evaluation = evaluate_8ball_quality(response)
            evaluation["prompt"] = prompt
            results.append(evaluation)
            print(f"    → {response} (score: {evaluation['score']:.2f})")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"prompt": prompt, "response": "", "error": str(e), "score": 0.0})

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_score = sum(r["score"] for r in valid_results) / len(valid_results)
        mystical_count = sum(1 for r in valid_results if r["contains_mystical_phrase"])
        fortune_count = sum(1 for r in valid_results if r["looks_like_fortune"])

        summary = {
            "total_prompts": len(test_prompts),
            "valid_responses": len(valid_results),
            "average_score": avg_score,
            "mystical_phrases": mystical_count,
            "fortune_like": fortune_count,
            "results": results,
        }

        print("📊 EVALUATION SUMMARY:")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Mystical Phrases: {mystical_count}/{len(valid_results)}")
        print(f"  Fortune-like: {fortune_count}/{len(valid_results)}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  💾 Results saved to {args.output}")

    return summary


if __name__ == "__main__":
    main()
