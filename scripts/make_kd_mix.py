"""
Generate knowledge distillation dataset by sampling from teacher model.

Usage:
    # HTTP endpoint
    python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher http://localhost:8000 --total 1000

    # HuggingFace model
    python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher hf:microsoft/phi-2 --total 1000

    # With caching
    python -m scripts.make_kd_mix --out data/kd_mix.jsonl --teacher http://localhost:8000 --cache-dir data/logits/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from models.teacher.teacher_client import TeacherClient
from scripts.prompt_sources import get_prompt_mix, load_prompts_from_file


def load_cache(cache_dir: Path, prompt: str) -> Optional[Dict[str, Any]]:
    """Load cached result for a prompt."""
    if not cache_dir.exists():
        return None

    # Hash prompt to filename
    import hashlib

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = cache_dir / f"{prompt_hash}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_cache(cache_dir: Path, prompt: str, result: Dict[str, Any]):
    """Save result to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    import hashlib

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = cache_dir / f"{prompt_hash}.json"

    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Generate KD dataset by sampling from teacher model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--teacher",
        required=True,
        help="Teacher endpoint (http://host:port) or HuggingFace model (hf:model_name)",
    )
    ap.add_argument("--total", type=int, default=1000, help="Total number of prompts to generate")
    ap.add_argument("--general-ratio", type=float, default=0.5, help="Ratio of general prompts")
    ap.add_argument(
        "--domain-ratio", type=float, default=0.3, help="Ratio of domain-specific prompts"
    )
    ap.add_argument("--tool-ratio", type=float, default=0.2, help="Ratio of tool trace prompts")
    ap.add_argument("--prompts-file", help="Load prompts from JSONL file instead of generating")
    ap.add_argument("--cache-dir", help="Directory to cache teacher responses")
    ap.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    ap.add_argument(
        "--return-logits", action="store_true", help="Request logits from teacher (if supported)"
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for teacher queries")
    ap.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    args = ap.parse_args()

    # Initialize teacher client
    if args.teacher.startswith("hf:"):
        model_name = args.teacher[3:]
        print(f"[make_kd_mix] Using HuggingFace model: {model_name}")
        client = TeacherClient.from_hf(model_name)
    else:
        endpoint = args.teacher
        print(f"[make_kd_mix] Using HTTP endpoint: {endpoint}")

        # Load API key from environment or .env.local (try MOONSHOT_API_KEY first, then KIMI_API_KEY)
        api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        if not api_key:
            env_file = Path(".env.local")
            if env_file.exists():
                try:
                    with open(env_file, "r") as f:
                        for line in f:
                            if line.startswith("MOONSHOT_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip("\"'")
                                break
                            if line.startswith("KIMI_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip("\"'")
                                break
                except Exception:
                    pass

        client = TeacherClient.from_endpoint(
            endpoint, api_key=api_key, max_retries=5, retry_backoff_factor=2.0
        )

        # Detect and display tier
        tier = client.get_tier()
        tier_limits = client.get_tier_limits()
        if tier and tier_limits:
            print(f"[make_kd_mix] Detected API tier: {tier.value}")
            print(
                f"[make_kd_mix] Rate limits: {tier_limits.rpm} RPM, {tier_limits.tpm:,} TPM, "
                f"{'Unlimited' if tier_limits.tpd is None else f'{tier_limits.tpd:,}'} TPD"
            )
            print(f"[make_kd_mix] Recommended delay: {tier_limits.delay}s")

            # Warn if delay doesn't match tier
            if args.delay > 0 and abs(args.delay - tier_limits.delay) > tier_limits.delay * 0.5:
                print(
                    f"[make_kd_mix] WARN: Delay ({args.delay}s) doesn't match tier recommendation ({tier_limits.delay}s)"
                )

    # Health check with retry
    print("[make_kd_mix] Checking API health...")
    if not client.health_check():
        print("[make_kd_mix] WARN: Teacher health check failed, continuing anyway...")
        print("[make_kd_mix] WARN: This may indicate network issues or API unavailability")
    else:
        print("[make_kd_mix] API health check passed")

    # Load or generate prompts
    if args.prompts_file:
        print(f"[make_kd_mix] Loading prompts from: {args.prompts_file}")
        prompts = load_prompts_from_file(args.prompts_file)
    else:
        print(
            f"[make_kd_mix] Generating prompt mix (general={args.general_ratio}, domain={args.domain_ratio}, tool={args.tool_ratio})"
        )
        prompts = get_prompt_mix(
            general_ratio=args.general_ratio,
            domain_ratio=args.domain_ratio,
            tool_ratio=args.tool_ratio,
            total=args.total,
        )

    print(f"[make_kd_mix] Total prompts: {len(prompts)}")

    # Setup cache
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample from teacher
    results = []
    cached = 0
    errors = 0

    print(
        f"[make_kd_mix] Sampling from teacher (batch_size={args.batch_size}, delay={args.delay}s)..."
    )

    for i, prompt in enumerate(prompts):
        # Check cache
        cached_result = None
        if cache_dir:
            cached_result = load_cache(cache_dir, prompt)

        if cached_result:
            results.append(cached_result)
            cached += 1
            if (i + 1) % 100 == 0:
                print(
                    f"[make_kd_mix] Progress: {i + 1}/{len(prompts)} (cached: {cached}, errors: {errors})"
                )
            continue

        # Query teacher (with built-in retry logic)
        teacher_results = client.sample(
            [prompt],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            return_logits=args.return_logits,
        )

        if teacher_results and not teacher_results[0].get("error"):
            result = {
                "prompt": prompt,
                "teacher_text": teacher_results[0]["text"],
                "teacher_logits": teacher_results[0].get("logits"),
                "metadata": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                },
            }

            results.append(result)

            # Save to cache
            if cache_dir:
                save_cache(cache_dir, prompt, result)
        else:
            errors += 1
            error_msg = (
                teacher_results[0].get("error", "Unknown error")
                if teacher_results
                else "No response"
            )
            print(f"[make_kd_mix] WARN: Failed to get response for prompt {i + 1}: {error_msg}")

            # Check for specific error types
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                print("[make_kd_mix] Rate limit hit, consider increasing --delay or upgrading tier")
            elif "token" in error_msg.lower() and "limit" in error_msg.lower():
                print("[make_kd_mix] Token limit exceeded, consider reducing --max-tokens")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                print("[make_kd_mix] Network error, will retry on next attempt")

        # Progress update
        if (i + 1) % 100 == 0:
            print(
                f"[make_kd_mix] Progress: {i + 1}/{len(prompts)} (cached: {cached}, errors: {errors})"
            )

        # Rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

    # Write results
    print(f"[make_kd_mix] Writing {len(results)} results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[make_kd_mix] ✅ Complete: {len(results)} samples, {cached} cached, {errors} errors")
    print(f"[make_kd_mix] Output: {output_path}")


if __name__ == "__main__":
    main()
