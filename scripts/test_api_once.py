"""
Quick API test to verify connectivity and basic functionality.

Usage:
    python -m scripts.test_api_once
"""
import os
import sys
from pathlib import Path

from models.teacher.teacher_client import TeacherClient


def main():
    # Load API key (try MOONSHOT_API_KEY first, then KIMI_API_KEY for backward compat)
    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        env_file = Path(".env.local")
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith("MOONSHOT_API_KEY="):
                            api_key = line.split(
                                "=", 1)[1].strip().strip('"\'')
                            break
                        if line.startswith("KIMI_API_KEY="):
                            api_key = line.split(
                                "=", 1)[1].strip().strip('"\'')
                            break
            except Exception as e:
                print(f"[test_api_once] ERROR: Failed to load .env.local: {e}")
                sys.exit(1)

    if not api_key:
        print("[test_api_once] ERROR: MOONSHOT_API_KEY or KIMI_API_KEY not found")
        print("[test_api_once] Please set MOONSHOT_API_KEY in .env.local")
        print("[test_api_once] (KIMI_API_KEY also supported for backward compatibility)")
        sys.exit(1)

    print("[test_api_once] Testing Kimi K2 API connection...")

    # Initialize client with correct endpoint
    # Use fewer retries for testing (to avoid long waits)
    endpoint = "https://api.moonshot.ai/v1"
    client = TeacherClient.from_endpoint(
        endpoint,
        api_key=api_key,
        max_retries=2,  # Reduced for testing - avoids long waits on rate limits
        retry_backoff_factor=2.0  # Exponential backoff: tier_delay * 2^retry_count
    )

    # Single API call that tests everything:
    # 1. API connectivity (health check)
    # 2. Authentication (if auth fails, we get 401/403)
    # 3. Extract tier/rate limit info from response headers
    # 4. Get actual sample response
    print("\n[test_api_once] Making single API call (combines health check, tier detection, and sample test)...")
    print("  (Using tier-aware backoff: starts at tier delay, then exponential)")
    print("  (Limited to 2 retries for quick testing)")
    test_prompt = "What is 2+2? Answer briefly."

    try:
        # Make the API call - this will test connectivity, auth, and get response
        results = client.sample(
            [test_prompt],
            temperature=1.0,
            top_p=0.95,
            max_tokens=50,
            return_logits=False,
        )

        # Extract tier info from the last response (if available)
        # The client stores tier info after successful requests
        tier = client.get_tier() if hasattr(client, 'get_tier') else None
        tier_limits = client.get_tier_limits() if hasattr(
            client, 'get_tier_limits') else None

        # Display results
        print("\n[test_api_once] ✅ API Test Results:")

        # Health/Auth check (implicit - if we got here, it worked)
        if results and not results[0].get("error"):
            print("  ✅ API connectivity: OK")
            print("  ✅ Authentication: OK")

            # Display tier info if available
            if tier and tier_limits:
                print(f"\n  API Tier Information:")
                print(f"    Tier: {tier.value}")
                print(f"    RPM: {tier_limits.rpm}")
                print(f"    TPM: {tier_limits.tpm:,}")
                if tier_limits.tpd:
                    print(f"    TPD: {tier_limits.tpd:,}")
                else:
                    print(f"    TPD: Unlimited")
                print(f"    Concurrency: {tier_limits.concurrency}")
                print(f"    Recommended delay: {tier_limits.delay}s")
                print(
                    f"    Backoff strategy: Tier-aware (starts at {tier_limits.delay}s, then exponential)")
            else:
                print("  ⚠️ Tier info: Not available (using defaults)")

            # Sample response
            print(f"\n  Sample Response:")
            print(f"    Prompt: {test_prompt}")
            print(f"    Response: {results[0]['text'][:100]}...")

            print("\n[test_api_once] ✅ All tests passed!")
            return 0

        if results and not results[0].get("error"):
            print("  ✅ Sample successful")
            print(f"  Prompt: {test_prompt}")
            print(f"  Response: {results[0]['text'][:100]}...")
            print("\n[test_api_once] ✅ API test passed!")
            return 0
        else:
            # Error case - still try to extract tier info if available
            error_msg = results[0].get(
                "error", "Unknown error") if results else "No response"
            print(f"\n[test_api_once] ❌ API Test Failed:")
            print(f"  Error: {error_msg}")

            # Try to get tier info (might have been set from response headers)
            tier = client.get_tier() if hasattr(client, 'get_tier') else None
            tier_limits = client.get_tier_limits() if hasattr(
                client, 'get_tier_limits') else None

            if tier and tier_limits:
                print(f"\n  Detected Tier: {tier.value}")
                print(f"  Recommended delay: {tier_limits.delay}s")
                if "rate limit" in error_msg.lower() or "429" in error_msg or "Max retries" in error_msg:
                    base_delay = tier_limits.delay
                    backoff_sequence = [
                        base_delay * (2 ** i) for i in range(6)]
                    print(
                        f"\n  💡 Tip: Wait {base_delay}s between requests on {tier.value} tier")
                    print(
                        f"  💡 Tier-aware backoff sequence: {', '.join([f'{d:.1f}s' for d in backoff_sequence])}")

            print(f"\n[test_api_once] Debug info:")
            print(f"  Results: {results}")
            if results and results[0].get("error"):
                print(f"  Full error: {results[0]}")
            return 1

    except Exception as e:
        print(f"  ❌ Sample failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
