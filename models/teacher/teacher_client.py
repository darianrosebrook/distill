"""
Teacher client for knowledge distillation.

Supports multiple backends:
- HTTP API (OpenAI-compatible or custom)
- HuggingFace pipeline (local model)

Usage:
    # HTTP endpoint
    client = TeacherClient.from_endpoint("http://localhost:8000")

    # HuggingFace model
    client = TeacherClient.from_hf("microsoft/phi-2")

    # Sample with logits
    results = client.sample(["What is 2+2?"], return_logits=True)
"""

from typing import List, Dict, Any, Optional
import time
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import torch
    from transformers import pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class APITier(str, Enum):
    """API tier levels based on cumulative recharge."""

    FREE = "free"  # $1
    TIER1 = "tier1"  # $10
    TIER2 = "tier2"  # $20
    TIER3 = "tier3"  # $100
    TIER4 = "tier4"  # $1,000
    TIER5 = "tier5"  # $3,000
    UNKNOWN = "unknown"


@dataclass
class TierLimits:
    """Rate limits for each tier."""

    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    tpd: Optional[int]  # Tokens per day (None = unlimited)
    concurrency: int  # Concurrent requests
    delay: float  # Recommended delay between requests (seconds)


TIER_LIMITS = {
    APITier.FREE: TierLimits(rpm=3, tpm=500_000, tpd=1_500_000, concurrency=1, delay=20.0),
    APITier.TIER1: TierLimits(rpm=200, tpm=2_000_000, tpd=None, concurrency=50, delay=0.3),
    APITier.TIER2: TierLimits(rpm=500, tpm=3_000_000, tpd=None, concurrency=100, delay=0.12),
    APITier.TIER3: TierLimits(rpm=5_000, tpm=3_000_000, tpd=None, concurrency=200, delay=0.012),
    APITier.TIER4: TierLimits(rpm=5_000, tpm=4_000_000, tpd=None, concurrency=400, delay=0.012),
    APITier.TIER5: TierLimits(rpm=10_000, tpm=5_000_000, tpd=None, concurrency=1_000, delay=0.006),
}


class TeacherClient:
    """Client for querying teacher models for knowledge distillation."""

    def __init__(self, backend: str = "http", **kwargs):
        """
        Initialize teacher client.

        Args:
            backend: "http" or "hf" (HuggingFace)
            **kwargs: Backend-specific arguments
        """
        self.backend = backend
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        # Check for manual tier override via environment variable
        tier_override = os.environ.get("MOONSHOT_TIER_OVERRIDE", "").lower()
        if tier_override in {"free", "tier1", "tier2", "tier3", "tier4", "tier5"}:
            tier_map = {
                "free": APITier.FREE,
                "tier1": APITier.TIER1,
                "tier2": APITier.TIER2,
                "tier3": APITier.TIER3,
                "tier4": APITier.TIER4,
                "tier5": APITier.TIER5,
            }
            self._tier = tier_map[tier_override]
            self._tier_limits = TIER_LIMITS[self._tier]
            print(
                f"[TeacherClient] Using tier override: {self._tier.value} (RPM: {self._tier_limits.rpm}, delay: {self._tier_limits.delay}s)")
        else:
            # Default to FREE tier so tier-aware backoff works from the start
            self._tier: Optional[APITier] = APITier.FREE
            self._tier_limits: Optional[TierLimits] = TIER_LIMITS[APITier.FREE]
        self._max_retries = kwargs.get("max_retries", 5)
        self._retry_backoff_factor = kwargs.get("retry_backoff_factor", 2.0)
        self._retry_status_codes = kwargs.get(
            "retry_status_codes", [429, 500, 502, 503, 504])

        if backend == "http":
            endpoint = kwargs.get(
                "endpoint", "http://localhost:8000").rstrip("/")
            # Normalize endpoint - remove trailing /v1 if present (we add it in requests)
            if endpoint.endswith("/v1"):
                endpoint = endpoint[:-3]
            self.endpoint = endpoint.rstrip("/")
            self.api_key = kwargs.get(
                "api_key") or self._load_api_key_from_env()
            # Increased timeout for kimi-k2-thinking which can take longer due to reasoning_content
            # Docs recommend streaming for long responses, but we use longer timeout as fallback
            # 10 minutes for long reasoning responses
            self.timeout = kwargs.get("timeout", 600)
            self._setup_retry_session()
            # Don't detect tier on initialization (saves a request)
            # Tier will be detected from first API response headers
        elif backend == "hf":
            if not HF_AVAILABLE:
                raise RuntimeError(
                    "HuggingFace transformers not available. Install with: pip install transformers torch"
                )
            model_name = kwargs.get("model_name", "microsoft/phi-2")
            device = kwargs.get("device", "cpu")
            self._load_hf_model(model_name, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def from_endpoint(
        cls,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 5,
        retry_backoff_factor: float = 2.0,
    ):
        """
        Create client from HTTP endpoint.

        Args:
            endpoint: API endpoint URL (e.g., https://api.moonshot.ai/v1 or https://api.moonshot.cn/v1)
            api_key: API key (or set MOONSHOT_API_KEY env var)
            timeout: Request timeout in seconds (default 600s for kimi-k2-thinking long responses)
            max_retries: Maximum retry attempts
            retry_backoff_factor: Exponential backoff multiplier
        """
        return cls(
            backend="http",
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )

    def _load_api_key_from_env(self) -> Optional[str]:
        """Load API key from environment variables."""
        # Try MOONSHOT_API_KEY first (official), then KIMI_API_KEY (backward compat)
        api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        if api_key:
            return api_key

        # Try loading from .env.local file
        env_file = Path(".env.local")
        if env_file.exists():
            try:
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MOONSHOT_API_KEY="):
                            return line.split("=", 1)[1].strip().strip("\"'")
                        if line.startswith("KIMI_API_KEY="):
                            return line.split("=", 1)[1].strip().strip("\"'")
            except Exception:
                pass

        return None

    def _setup_retry_session(self):
        """Setup requests session with retry strategy."""
        self.session = requests.Session()

        # Don't retry 429 in urllib3 - let our custom logic handle it with proper backoff
        # Only retry on server errors (5xx) and connection errors
        retry_status_codes = [
            code for code in self._retry_status_codes if code != 429]

        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_backoff_factor,
            status_forcelist=retry_status_codes,  # Exclude 429 - handled by custom logic
            allowed_methods=["POST", "GET"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _detect_tier(self) -> Optional[APITier]:
        """
        Detect API tier by making a test request and checking rate limits.

        Returns:
            Detected tier or None if detection fails
        """
        if not self.api_key:
            return None

        try:
            # Make a minimal test request to check rate limit headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Try a simple health check or minimal request
            test_payload = {
                "model": "kimi-k2-thinking",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            response = self.session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=test_payload,
                headers=headers,
                timeout=10,
            )

            # Check response headers for rate limit info
            # Kimi API may include tier info in headers
            rpm_header = response.headers.get("x-ratelimit-limit-rpm") or response.headers.get(
                "ratelimit-limit-rpm"
            )
            tpd_header = response.headers.get("x-ratelimit-limit-tpd") or response.headers.get(
                "ratelimit-limit-tpd"
            )

            # Infer tier from rate limits
            if rpm_header:
                rpm = int(rpm_header)
                if rpm >= 10_000:
                    tier = APITier.TIER5
                elif rpm >= 5_000:
                    tier = APITier.TIER4 if tpd_header else APITier.TIER3
                elif rpm >= 500:
                    tier = APITier.TIER2
                elif rpm >= 200:
                    tier = APITier.TIER1
                else:
                    tier = APITier.FREE

                self._tier = tier
                self._tier_limits = TIER_LIMITS.get(
                    tier, TIER_LIMITS[APITier.FREE])
                print(
                    f"[TeacherClient] Detected tier: {tier.value} (RPM: {rpm}, delay: {self._tier_limits.delay}s)"
                )
                return tier

        except Exception as e:
            print(
                f"[TeacherClient] Tier detection failed: {e}, defaulting to FREE tier")
            self._tier = APITier.FREE
            self._tier_limits = TIER_LIMITS[APITier.FREE]

        return self._tier

    def _update_tier_from_response(self, response: requests.Response):
        """
        Update tier information from API response headers.
        Called after successful API requests to extract rate limit info.
        This allows tier detection from a single API call instead of a separate request.
        """
        if not response:
            return

        # Check response headers for rate limit info
        rpm_header = response.headers.get("x-ratelimit-limit-rpm") or response.headers.get(
            "ratelimit-limit-rpm"
        )
        tpd_header = response.headers.get("x-ratelimit-limit-tpd") or response.headers.get(
            "ratelimit-limit-tpd"
        )
        response.headers.get(
            "x-ratelimit-limit-tpm") or response.headers.get("ratelimit-limit-tpm")

        # Infer tier from rate limits if available
        if rpm_header:
            try:
                rpm = int(rpm_header)
                if rpm >= 10_000:
                    tier = APITier.TIER5
                elif rpm >= 5_000:
                    tier = APITier.TIER4 if tpd_header else APITier.TIER3
                elif rpm >= 500:
                    tier = APITier.TIER2
                elif rpm >= 200:
                    tier = APITier.TIER1
                else:
                    tier = APITier.FREE

                # Update tier if detected
                if tier != self._tier:
                    self._tier = tier
                    self._tier_limits = TIER_LIMITS.get(
                        tier, TIER_LIMITS[APITier.FREE])
                    print(
                        f"[TeacherClient] Detected tier: {tier.value} (RPM: {rpm}, delay: {self._tier_limits.delay}s)"
                    )
            except (ValueError, TypeError):
                pass

    def get_tier(self) -> Optional[APITier]:
        """Get detected API tier."""
        if self._tier is None and self.backend == "http" and self.api_key:
            self._detect_tier()
        return self._tier

    def get_tier_limits(self) -> Optional[TierLimits]:
        """Get rate limits for current tier."""
        if self._tier_limits is None:
            self.get_tier()
        return self._tier_limits

    @classmethod
    def from_hf(cls, model_name: str, device: Optional[str] = None):
        """
        Create client from HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cpu", "cuda", "mps", or None for auto-detect)
        """
        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return cls(backend="hf", model_name=model_name, device=device)

    def _load_hf_model(self, model_name: str, device: str):
        """Load HuggingFace model and tokenizer safely with revision pinning."""
        from training.safe_model_loading import (
            safe_from_pretrained_tokenizer,
            safe_from_pretrained_causal_lm,
        )

        print(f"[TeacherClient] Loading HuggingFace model: {model_name}")
        self._tokenizer = safe_from_pretrained_tokenizer(model_name)

        # Determine dtype based on device
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps":
            # MPS supports float16 but may have better compatibility with float32
            dtype = torch.float16
        else:
            dtype = torch.float16

        # Load model with appropriate device mapping
        if device in ("cuda", "mps"):
            # Use device_map for CUDA, manual .to() for MPS
            if device == "cuda":
                self._model = safe_from_pretrained_causal_lm(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device,
                )
            else:  # MPS
                self._model = safe_from_pretrained_causal_lm(
                    model_name,
                    torch_dtype=dtype,
                    device_map=None,
                )
                self._model = self._model.to(device)
        else:  # CPU
            self._model = safe_from_pretrained_causal_lm(
                model_name,
                torch_dtype=dtype,
                device_map=None,
            )
            self._model = self._model.to(device)

        self._model.eval()
        print(f"[TeacherClient] Model loaded on {device}")

    def sample(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Sample from teacher model.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (default 1.0 for kimi-k2-thinking)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate. If None, uses model-aware defaults:
                - kimi-k2-thinking: 16,384 (16K) to ensure full reasoning_content + content
                - Other models: 1,024
            return_logits: Whether to return logits (for KD)
            **kwargs: Additional backend-specific parameters (model name, etc.)

        Returns:
            List of dicts with keys: "prompt", "text", "reasoning_content" (if present), "logits" (if return_logits=True)
        """
        # Model-aware default max_tokens
        model_name = kwargs.get("model", "kimi-k2-thinking")
        if max_tokens is None:
            if "kimi-k2-thinking" in model_name.lower():
                # kimi-k2-thinking requires ≥16K tokens for full reasoning_content + content
                # Note: This is the OUTPUT limit. The model supports 256K CONTEXT window
                # (input + output combined), which is handled automatically by the API.
                max_tokens = 16_384
            else:
                max_tokens = 1024

        if self.backend == "http":
            return self._sample_http(
                prompts, temperature, top_p, max_tokens, return_logits, **kwargs
            )
        elif self.backend == "hf":
            return self._sample_hf(prompts, temperature, top_p, max_tokens, return_logits, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _sample_http(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
        return_logits: bool,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Sample from HTTP API endpoint with retry logic and error handling."""
        results = []

        for prompt in prompts:
            result = self._sample_single_with_retry(
                prompt, temperature, top_p, max_tokens, return_logits, **kwargs
            )
            results.append(result)

        return results

    def _sample_single_with_retry(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        return_logits: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sample single prompt with retry logic and exponential backoff.

        Handles:
        - Rate limiting (429) with exponential backoff
        - Network errors with reconnection
        - Server errors (5xx) with retry
        - Token limit errors (400)

        Args:
            prompt: Input prompt (ignored if payload_override provided)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            return_logits: Whether to return logits
            **kwargs: Additional parameters including:
                - payload_override: Dict to override default payload construction (for multi-step)
                - model: Model name (default: kimi-k2-thinking)
                - stream: Enable streaming (default: False)
        """
        # Support payload override for multi-step calls
        payload_override = kwargs.get("payload_override")
        if payload_override:
            payload = payload_override
        else:
            # Default model: kimi-k2-thinking (long-thinking version)
            # Other options: kimi-k2-turbo-preview, kimi-k2-0905-preview, kimi-k2-thinking-turbo
            model_name = kwargs.get("model", "kimi-k2-thinking")
            is_thinking_model = "kimi-k2-thinking" in model_name.lower()

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            # For kimi-k2-thinking: enable streaming recommended for long reasoning_content
            # Docs recommend streaming to avoid network timeouts and better UX
            # However, we use non-streaming by default for simplicity in batch processing
            # Streaming can be enabled via kwargs["stream"] = True if needed
            if kwargs.get("stream", False) and is_thinking_model:
                payload["stream"] = True

        if return_logits:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_exception = None
        retry_count = 0
        max_attempts = self._max_retries + 1

        # Debug: Log initial request details
        model_name = payload.get("model", "unknown")
        prompt_length = len(prompt)
        print(
            f"[TeacherClient] Starting request: endpoint={self.endpoint}/v1/chat/completions, model={model_name}, "
            f"prompt_len={prompt_length}, max_tokens={max_tokens}, max_attempts={max_attempts}"
        )
        if self._tier:
            print(
                f"[TeacherClient] Current tier: {self._tier.value}, limits: {self._tier_limits.rpm} RPM, "
                f"{self._tier_limits.tpm:,} TPM, delay={self._tier_limits.delay}s"
            )

        while retry_count < max_attempts:
            # Debug: Log attempt details
            if retry_count > 0:
                print(
                    f"[TeacherClient] Retry attempt {retry_count + 1}/{max_attempts}")
            else:
                print(f"[TeacherClient] Initial attempt 1/{max_attempts}")
            try:
                request_start = time.time()
                response = self.session.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                request_duration = time.time() - request_start

                # Debug: Log response details
                print(
                    f"[TeacherClient] Response: status={response.status_code}, duration={request_duration:.2f}s"
                )

                # Extract rate limit headers for debugging
                rate_limit_headers = {
                    "x-ratelimit-limit-rpm": response.headers.get("x-ratelimit-limit-rpm"),
                    "x-ratelimit-remaining-rpm": response.headers.get("x-ratelimit-remaining-rpm"),
                    "x-ratelimit-limit-tpm": response.headers.get("x-ratelimit-limit-tpm"),
                    "x-ratelimit-remaining-tpm": response.headers.get("x-ratelimit-remaining-tpm"),
                    "retry-after": response.headers.get("retry-after"),
                }

                # Debug: Show all headers that might contain rate limit info
                all_rate_headers = {
                    k: v
                    for k, v in response.headers.items()
                    if "rate" in k.lower() or "limit" in k.lower() or "retry" in k.lower()
                }
                if all_rate_headers:
                    print(
                        f"[TeacherClient] Rate limit related headers: {', '.join(f'{k}={v}' for k, v in all_rate_headers.items())}"
                    )
                elif any(rate_limit_headers.values()):
                    print(
                        f"[TeacherClient] Rate limit headers: {', '.join(f'{k}={v}' for k, v in rate_limit_headers.items() if v)}"
                    )
                else:
                    print("[TeacherClient] No rate limit headers found in response")

                # Try to detect tier from ANY response (success or error) - tier info may be in headers
                # This is important because tier detection should happen even on error responses
                old_tier = self._tier
                self._update_tier_from_response(response)
                if old_tier != self._tier:
                    print(
                        f"[TeacherClient] Tier detected from response: {old_tier.value} → {self._tier.value}"
                    )

                # Success
                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice.get("message", {})

                    # Extract content (regular response)
                    text = message.get("content", "")

                    # Extract reasoning_content if present (kimi-k2-thinking feature)
                    reasoning_content = None
                    if "reasoning_content" in message:
                        reasoning_content = message["reasoning_content"]
                        print(
                            f"[TeacherClient] Reasoning content present: {len(reasoning_content)} chars"
                        )
                    elif hasattr(message, "reasoning_content"):
                        # Handle case where reasoning_content exists but not in dict
                        reasoning_content = getattr(
                            message, "reasoning_content", None)
                        if reasoning_content:
                            print(
                                f"[TeacherClient] Reasoning content present: {len(reasoning_content)} chars"
                            )

                    # Debug: Log success details
                    usage = data.get("usage", {})
                    reasoning_info = (
                        f", reasoning={len(reasoning_content) if reasoning_content else 0} chars"
                        if reasoning_content
                        else ""
                    )
                    print(
                        f"[TeacherClient] Success: response_len={len(text)}{reasoning_info}, "
                        f"tokens={usage.get('total_tokens', 'unknown')} "
                        f"(input: {usage.get('prompt_tokens', 'unknown')}, "
                        f"output: {usage.get('completion_tokens', 'unknown')})"
                    )

                    # For multi-step calls (payload_override), return full message structure
                    if payload_override:
                        result = {
                            # Full message with reasoning_content, tool_calls, etc.
                            "message": message.copy(),
                            "text": text,
                            "logits": None,
                        }
                        # Preserve reasoning_content in message for multi-step continuity
                        if reasoning_content:
                            result["message"]["reasoning_content"] = reasoning_content
                        # Include tool_calls if present
                        if "tool_calls" in message:
                            result["tool_calls"] = message["tool_calls"]
                    else:
                        # Single-step call - return simplified structure
                        result = {
                            "prompt": prompt,
                            "text": text,
                            "logits": None,
                        }

                    # Include reasoning_content if present (for KD, we may want both reasoning and final answer)
                    # For kimi-k2-thinking: reasoning_content contains the model's reasoning process
                    # This is valuable for distillation as it shows the thinking process
                    if reasoning_content and not payload_override:
                        result["reasoning_content"] = reasoning_content
                        # Note: reasoning_content tokens count towards max_tokens limit
                        # We ensure max_tokens ≥ 16K for kimi-k2-thinking to accommodate both

                    if return_logits and "logprobs" in choice:
                        result["logits"] = choice.get("logprobs")

                    return result

                # Rate limit (429) - tier-aware exponential backoff
                elif response.status_code == 429:
                    # Debug: Log error response body if available
                    try:
                        error_data = (
                            response.json()
                            if response.headers.get("content-type", "").startswith(
                                "application/json"
                            )
                            else {}
                        )
                        error_msg = (
                            error_data.get("error", {}).get(
                                "message", response.text[:200])
                            if isinstance(error_data, dict)
                            else str(response.text[:200])
                        )
                        print(
                            f"[TeacherClient] Rate limit error message: {error_msg}")
                    except Exception:
                        pass

                    retry_after = self._get_retry_after(response)

                    # Use tier-specific backoff if available
                    if retry_after and self._tier_limits:
                        # Respect Retry-After but ensure minimum tier delay
                        # For FREE tier, API may return short Retry-After (1s) but we need 20s
                        min_delay = self._tier_limits.delay
                        tier_backoff = min_delay * \
                            (self._retry_backoff_factor**retry_count)
                        wait_time = max(retry_after, tier_backoff)
                        tier_info = f" (Retry-After: {retry_after}s, tier min: {min_delay}s, tier backoff: {tier_backoff:.1f}s, using: {wait_time:.1f}s)"
                    elif retry_after:
                        # Retry-After without tier info - use as-is but warn
                        wait_time = retry_after
                        tier_info = f" (Retry-After header: {retry_after}s)"
                        if retry_after < 5.0:
                            print(
                                f"[TeacherClient] WARN: Short Retry-After ({retry_after}s) may not be sufficient for rate limits"
                            )
                    elif self._tier_limits:
                        # Tier-aware backoff: start with tier delay, then exponential
                        base_delay = self._tier_limits.delay
                        wait_time = base_delay * \
                            (self._retry_backoff_factor**retry_count)
                        tier_info = f" (tier: {self._tier.value}, base: {base_delay}s, exponential: {wait_time:.1f}s)"
                    else:
                        # Fallback to standard exponential backoff
                        wait_time = self._retry_backoff_factor**retry_count
                        tier_info = f" (fallback exponential: {wait_time:.1f}s)"

                    print(
                        f"[TeacherClient] Rate limited (429), waiting {wait_time:.1f}s before retry {retry_count + 1}/{max_attempts}{tier_info}"
                    )
                    if retry_count + 1 >= max_attempts:
                        print(
                            f"[TeacherClient] ERROR: Max retries ({max_attempts}) exceeded. Last error: Rate limit (429)"
                        )
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                # Token limit exceeded (400) - don't retry
                elif response.status_code == 400:
                    error_data = (
                        response.json()
                        if response.headers.get("content-type", "").startswith("application/json")
                        else {}
                    )
                    error_msg = (
                        error_data.get("error", {}).get(
                            "message", response.text)
                        if isinstance(error_data, dict)
                        else str(response.text)
                    )

                    print(f"[TeacherClient] Bad request (400): {error_msg}")
                    # Note: Tier detection already happened above, so we should have detected tier from headers if available

                    if "token" in error_msg.lower() or "limit" in error_msg.lower():
                        print("[TeacherClient] Token limit exceeded - not retrying")
                        return {
                            "prompt": prompt,
                            "text": "",
                            "logits": None,
                            "error": f"Token limit exceeded: {error_msg}",
                        }
                    else:
                        print("[TeacherClient] Other 400 error - not retrying")
                        return {
                            "prompt": prompt,
                            "text": "",
                            "logits": None,
                            "error": f"Bad request (400): {error_msg}",
                        }

                # Server errors (5xx) - retry with backoff
                elif response.status_code >= 500:
                    try:
                        error_data = (
                            response.json()
                            if response.headers.get("content-type", "").startswith(
                                "application/json"
                            )
                            else {}
                        )
                        error_msg = (
                            error_data.get("error", {}).get(
                                "message", response.text[:200])
                            if isinstance(error_data, dict)
                            else str(response.text[:200])
                        )
                        print(
                            f"[TeacherClient] Server error ({response.status_code}): {error_msg}")
                    except Exception:
                        print(
                            f"[TeacherClient] Server error ({response.status_code}): {response.text[:200]}"
                        )

                    wait_time = self._retry_backoff_factor**retry_count
                    print(
                        f"[TeacherClient] Retrying in {wait_time:.1f}s ({retry_count + 1}/{max_attempts})"
                    )
                    if retry_count + 1 >= max_attempts:
                        print(
                            f"[TeacherClient] ERROR: Max retries ({max_attempts}) exceeded. Last error: Server error ({response.status_code})"
                        )
                    time.sleep(wait_time)
                    retry_count += 1
                    continue

                # Other errors - return error with status code
                else:
                    try:
                        error_data = (
                            response.json()
                            if response.headers.get("content-type", "").startswith(
                                "application/json"
                            )
                            else {}
                        )
                        error_msg = (
                            error_data.get("error", {}).get(
                                "message", response.text[:200])
                            if isinstance(error_data, dict)
                            else str(response.text[:200])
                        )
                    except Exception:
                        error_msg = str(response.text[:200])

                    print(
                        f"[TeacherClient] API error ({response.status_code}): {error_msg}")
                    return {
                        "prompt": prompt,
                        "text": "",
                        "logits": None,
                        "error": f"API error ({response.status_code}): {error_msg}",
                    }

            except requests.exceptions.ConnectionError as e:
                # Network connection error - retry with exponential backoff
                wait_time = self._retry_backoff_factor**retry_count
                print(
                    f"[TeacherClient] Connection error (attempt {retry_count + 1}/{max_attempts}): {e}"
                )
                print(f"[TeacherClient] Retrying in {wait_time:.1f}s")
                if retry_count + 1 >= max_attempts:
                    print(
                        f"[TeacherClient] ERROR: Max retries ({max_attempts}) exceeded. Last error: Connection error"
                    )
                time.sleep(wait_time)
                retry_count += 1
                last_exception = e
                continue

            except requests.exceptions.Timeout as e:
                # Timeout - retry with exponential backoff
                wait_time = self._retry_backoff_factor**retry_count
                print(
                    f"[TeacherClient] Timeout (attempt {retry_count + 1}/{max_attempts}, timeout={self.timeout}s): {e}"
                )
                print(f"[TeacherClient] Retrying in {wait_time:.1f}s")
                if retry_count + 1 >= max_attempts:
                    print(
                        f"[TeacherClient] ERROR: Max retries ({max_attempts}) exceeded. Last error: Timeout"
                    )
                time.sleep(wait_time)
                retry_count += 1
                last_exception = e
                continue

            except requests.exceptions.RequestException as e:
                # Other request errors
                print(
                    f"[TeacherClient] Request error (attempt {retry_count + 1}/{max_attempts}): {e}"
                )
                print("[TeacherClient] Not retrying - fatal request error")
                last_exception = e
                break

            except Exception as e:
                # Unexpected errors
                print(
                    f"[TeacherClient] Unexpected error (attempt {retry_count + 1}/{max_attempts}): {type(e).__name__}: {e}"
                )
                import traceback

                print(f"[TeacherClient] Traceback: {traceback.format_exc()}")
                last_exception = e
                break

        # All retries exhausted
        error_msg = str(
            last_exception) if last_exception else "Max retries exceeded"
        print(
            f"[TeacherClient] ERROR: All retries exhausted. Final error: {error_msg}")
        return {
            "prompt": prompt,
            "text": "",
            "logits": None,
            "error": error_msg,
        }

    def _get_retry_after(self, response: requests.Response) -> Optional[float]:
        """Extract Retry-After header value."""
        retry_after = response.headers.get(
            "Retry-After") or response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return None

    def _try_fallback_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Try fallback API format if OpenAI-compatible format fails."""
        try:
            response = self.session.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "prompt": prompt,
                    "text": data.get("text", ""),
                    "logits": data.get("logits"),
                }
        except Exception:
            pass

        return {
            "prompt": prompt,
            "text": "",
            "logits": None,
            "error": "API request failed",
        }

    def _sample_hf(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
        return_logits: bool,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Sample from HuggingFace model."""
        results = []

        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=kwargs.get("device", "cpu"),
            )

        for prompt in prompts:
            try:
                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    return_full_text=False,
                    num_return_sequences=1,
                )

                text = outputs[0]["generated_text"]

                result = {
                    "prompt": prompt,
                    "text": text,
                    "logits": None,
                }

                if return_logits:
                    # Get logits from model directly
                    inputs = self._tokenizer(
                        prompt, return_tensors="pt").to(self._model.device)
                    with torch.no_grad():
                        outputs = self._model(
                            **inputs, output_hidden_states=False)
                        # Get logits for the last token position
                        logits = outputs.logits[0, -1, :].cpu().numpy()
                        result["logits"] = logits.tolist()

                results.append(result)

            except Exception as e:
                print(
                    f"[TeacherClient] WARN: HF generation failed for prompt: {e}")
                results.append(
                    {
                        "prompt": prompt,
                        "text": "",
                        "logits": None,
                        "error": str(e),
                    }
                )

        return results

    def sample_multi_step(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sample from teacher model with multi-step tool calls.

        For kimi-k2-thinking: preserves reasoning_content in message history for continuity.

        Args:
            messages: List of message dicts with "role" and "content".
                For assistant messages, include "reasoning_content" if present.
            temperature: Sampling temperature (default 1.0 for kimi-k2-thinking)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate. If None, uses model-aware defaults.
            tools: Optional list of tool definitions for tool calling
            **kwargs: Additional parameters (model name, etc.)

        Returns:
            Dict with keys: "message" (full message with reasoning_content), "tool_calls" (if any)

        Example:
            # First call
            result1 = client.sample_multi_step([
                {"role": "user", "content": "What's the weather?"}
            ], tools=tools)

            # Second call - preserve reasoning_content from first response
            messages = [
                {"role": "user", "content": "What's the weather?"},
                result1["message"]  # Includes reasoning_content if present
            ]
            if result1["message"].get("tool_calls"):
                # Add tool results
                messages.append({"role": "tool", "tool_call_id": ..., "content": ...})

            result2 = client.sample_multi_step(messages, tools=tools)
        """
        if self.backend != "http":
            raise ValueError(
                "Multi-step tool calls only supported for HTTP backend")

        model_name = kwargs.get("model", "kimi-k2-thinking")
        is_thinking_model = "kimi-k2-thinking" in model_name.lower()

        # Model-aware default max_tokens
        if max_tokens is None:
            if is_thinking_model:
                max_tokens = 16_384
            else:
                max_tokens = 1024

        # Ensure messages preserve reasoning_content from previous responses
        # The API server handles this automatically, but we document it here
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        if kwargs.get("stream", False) and is_thinking_model:
            payload["stream"] = True

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Use single retry logic
        return self._sample_single_with_retry(
            prompt="",  # Not used for multi-step
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            return_logits=False,
            payload_override=payload,  # Override default payload construction
            **kwargs,
        )

    def health_check(self) -> bool:
        """
        Check if teacher is available with retry logic.

        Returns:
            True if healthy, False otherwise
        """
        if self.backend == "http":
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Try health endpoint first
                    response = self.session.get(
                        f"{self.endpoint}/health", timeout=5)
                    if response.status_code == 200:
                        return True

                    # Fallback: try a minimal API request
                    if self.api_key:
                        headers = {"Authorization": f"Bearer {self.api_key}"}
                        test_response = self.session.post(
                            f"{self.endpoint}/v1/chat/completions",
                            json={
                                "model": "kimi-k2-thinking",
                                "messages": [{"role": "user", "content": "test"}],
                                "max_tokens": 1,
                            },
                            headers=headers,
                            timeout=5,
                        )
                        # Even if it fails, if we get a non-connection error, API is reachable
                        if test_response.status_code != 0:
                            return True

                except requests.exceptions.ConnectionError:
                    if attempt < max_attempts - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                        continue
                    return False
                except Exception:
                    # Other errors might indicate API is reachable but misconfigured
                    # Consider it "healthy" for connection purposes
                    return True

            return False
        elif self.backend == "hf":
            return self._model is not None
        return False
