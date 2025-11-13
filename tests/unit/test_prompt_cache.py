"""
Unit tests for prompt caching optimization.

Tests prompt cache functionality for M-series Apple Silicon optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from coreml.runtime.prompt_cache import (
    PromptCache,
    extract_system_prompt,
    extract_prompt_parts,
)


class TestPromptCache:
    """Tests for PromptCache class."""

    def test_cache_hit(self):
        """Test that cached prompts return cached state."""
        cache = PromptCache(max_cache_size_mb=10)

        prompt_text = "System: You are a helpful assistant."
        compute_fn = Mock(return_value={"kv_cache": np.array([1, 2, 3])})

        # First call: miss
        state1, was_cached1 = cache.get_or_compute(prompt_text, compute_fn)
        assert not was_cached1
        assert compute_fn.call_count == 1

        # Second call: hit
        compute_fn.reset_mock()
        state2, was_cached2 = cache.get_or_compute(prompt_text, compute_fn)
        assert was_cached2
        assert compute_fn.call_count == 0  # Should not call compute_fn
        assert np.array_equal(state1["kv_cache"], state2["kv_cache"])

    def test_cache_miss_different_prompts(self):
        """Test that different prompts result in cache misses."""
        cache = PromptCache(max_cache_size_mb=10)

        prompt1 = "System: You are a helpful assistant."
        prompt2 = "System: You are a careful assistant."

        compute_fn1 = Mock(return_value={"kv_cache": np.array([1, 2, 3])})
        compute_fn2 = Mock(return_value={"kv_cache": np.array([4, 5, 6])})

        state1, was_cached1 = cache.get_or_compute(prompt1, compute_fn1)
        state2, was_cached2 = cache.get_or_compute(prompt2, compute_fn2)

        assert not was_cached1
        assert not was_cached2
        assert compute_fn1.call_count == 1
        assert compute_fn2.call_count == 1

    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        cache = PromptCache(max_cache_size_mb=1)  # 1MB limit

        # Create a large state that exceeds limit
        large_state = {
            "kv_cache": np.zeros((1000, 1000), dtype=np.float32)  # ~4MB
        }

        compute_fn = Mock(return_value=large_state)
        prompt_text = "System: Large prompt."

        # Should compute but not cache (exceeds limit)
        state, was_cached = cache.get_or_compute(prompt_text, compute_fn)
        assert not was_cached

        # Second call should still miss (not cached)
        compute_fn.reset_mock()
        state2, was_cached2 = cache.get_or_compute(prompt_text, compute_fn)
        assert not was_cached2
        assert compute_fn.call_count == 1

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = PromptCache(max_cache_size_mb=10)

        prompt_text = "System: Test prompt."
        compute_fn = Mock(return_value={"kv_cache": np.array([1, 2, 3])})

        # Miss
        cache.get_or_compute(prompt_text, compute_fn)

        # Hit
        cache.get_or_compute(prompt_text, compute_fn)

        # Miss (different prompt)
        cache.get_or_compute("System: Different prompt.", compute_fn)

        stats = cache.stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert stats["cache_entries"] == 2  # Both prompts cached

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = PromptCache(max_cache_size_mb=10)

        prompt_text = "System: Test prompt."
        compute_fn = Mock(return_value={"kv_cache": np.array([1, 2, 3])})

        # Cache something
        cache.get_or_compute(prompt_text, compute_fn)
        assert len(cache.cache) > 0

        # Clear cache
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_precomputed_hash(self):
        """Test using pre-computed hash for efficiency."""
        cache = PromptCache(max_cache_size_mb=10)

        prompt_text = "System: Test prompt."
        compute_fn = Mock(return_value={"kv_cache": np.array([1, 2, 3])})

        # Pre-compute hash
        prompt_hash = cache._hash_prompt(prompt_text)

        # Use pre-computed hash
        state1, _ = cache.get_or_compute(prompt_text, compute_fn, prompt_hash=prompt_hash)

        # Second call with same hash
        state2, was_cached = cache.get_or_compute(prompt_text, compute_fn, prompt_hash=prompt_hash)

        assert was_cached
        assert np.array_equal(state1["kv_cache"], state2["kv_cache"])


class TestExtractSystemPrompt:
    """Tests for system prompt extraction."""

    def test_extract_system_prefix(self):
        """Test extraction of system prompt with 'System:' prefix."""
        prompt = "System: You are a helpful assistant.\n\nUser: What is Python?"
        system = extract_system_prompt(prompt)

        assert system == "System: You are a helpful assistant."

    def test_extract_first_paragraph(self):
        """Test extraction of first paragraph as system prompt."""
        prompt = "You are a helpful assistant capable of using tools.\n\nUser: Question?"
        system = extract_system_prompt(prompt)

        # Should extract first line if it looks like a system prompt
        assert system is not None
        assert "helpful" in system.lower() or "assistant" in system.lower()

    def test_extract_before_user_marker(self):
        """Test extraction before 'User:' marker."""
        prompt = "System instructions here.\n\nUser: What is Python?"
        system = extract_system_prompt(prompt)

        assert system == "System instructions here."

    def test_no_system_prompt(self):
        """Test when no system prompt is found."""
        prompt = "Just a regular question without system prompt."
        system = extract_system_prompt(prompt)

        # Should return None or empty string if no system prompt found
        assert system is None or system == ""

    def test_extract_prompt_parts(self):
        """Test extraction of system and user parts."""
        prompt = "System: You are helpful.\n\nUser: What is Python?"
        parts = extract_prompt_parts(prompt)

        assert "system" in parts
        assert "user" in parts
        assert "System:" in parts["system"]
        assert "Python" in parts["user"]


class TestPromptCacheIntegration:
    """Integration tests for prompt cache."""

    def test_cache_with_real_state(self):
        """Test cache with realistic state structure."""
        cache = PromptCache(max_cache_size_mb=10)

        # Simulate realistic state (KV cache, attention mask, etc.)
        def create_state():
            return {
                "kv_cache_k": np.random.randn(32, 512, 128).astype(np.float16),
                "kv_cache_v": np.random.randn(32, 512, 128).astype(np.float16),
                "attention_mask": np.ones((1, 512), dtype=np.int32),
            }

        prompt_text = "System: You are a helpful assistant."

        # First call
        state1, was_cached1 = cache.get_or_compute(prompt_text, create_state)
        assert not was_cached1

        # Second call (should be cached)
        state2, was_cached2 = cache.get_or_compute(prompt_text, create_state)
        assert was_cached2

        # Verify state matches
        assert np.array_equal(state1["kv_cache_k"], state2["kv_cache_k"])
        assert np.array_equal(state1["kv_cache_v"], state2["kv_cache_v"])
        assert np.array_equal(state1["attention_mask"], state2["attention_mask"])

        # Verify deep copy (modifying one shouldn't affect the other)
        state1["kv_cache_k"][0, 0, 0] = 999.0
        assert state2["kv_cache_k"][0, 0, 0] != 999.0  # Should be independent
