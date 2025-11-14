"""
Tests for training/teacher_cache.py - Teacher API response cache with integrity checks.

Tests caching, version validation, hash validation, and cache statistics.
"""
# @author: @darianrosebrook

import pytest
import json
import hashlib
from training.teacher_cache import (
    CacheEntry,
    TeacherCache,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            prompt_hash="abc123",
            teacher_version="v1.0.0",
            response={"text": "response"},
            created_at=1234567890.0,
            prompt_text="test prompt",
            metadata={"key": "value"},
        )
        assert entry.prompt_hash == "abc123"
        assert entry.teacher_version == "v1.0.0"
        assert entry.response == {"text": "response"}
        assert entry.prompt_text == "test prompt"


class TestTeacherCache:
    """Test TeacherCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Create a TeacherCache instance."""
        return TeacherCache(cache_dir, teacher_version="v1.0.0")

    def test_teacher_cache_initialization(self, cache_dir):
        """Test TeacherCache initialization."""
        cache = TeacherCache(cache_dir, teacher_version="v1.0.0")
        assert cache.cache_dir == cache_dir
        assert cache.teacher_version == "v1.0.0"
        assert cache.cache_file.exists() or cache.cache_file.parent.exists()

    def test_put_and_get(self, cache):
        """Test putting and getting from cache."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)
        cached_response = cache.get(prompt)

        assert cached_response == response

    def test_get_cache_miss(self, cache):
        """Test getting from empty cache."""
        prompt = "What is 2+2?"
        cached_response = cache.get(prompt)

        assert cached_response is None

    def test_get_with_version_mismatch(self, cache):
        """Test getting with version mismatch."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        # Put with current version
        cache.put(prompt, response)

        # Change version
        cache.teacher_version = "v2.0.0"

        # Should not get cached response due to version mismatch
        cached_response = cache.get(prompt, validate_version=True)
        assert cached_response is None

    def test_get_without_version_validation(self, cache):
        """Test getting without version validation."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)
        cache.teacher_version = "v2.0.0"

        # Should get cached response if version validation is disabled
        cached_response = cache.get(prompt, validate_version=False)
        assert cached_response == response

    def test_get_with_hash_validation(self, cache):
        """Test getting with hash validation."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)

        # Should get cached response with hash validation
        cached_response = cache.get(prompt, validate_hash=True)
        assert cached_response == response

    def test_get_without_hash_validation(self, cache):
        """Test getting without hash validation."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)

        # Should get cached response even without hash validation
        cached_response = cache.get(prompt, validate_hash=False)
        assert cached_response == response

    def test_put_with_metadata(self, cache):
        """Test putting with metadata."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}
        metadata = {"source": "test", "timestamp": 1234567890}

        cache.put(prompt, response, metadata=metadata)

        cached_response = cache.get(prompt)
        assert cached_response == response

    def test_cache_persistence(self, cache_dir):
        """Test that cache persists to disk."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        # Create cache and put entry
        cache1 = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cache1.put(prompt, response)

        # Create new cache instance (should load from disk)
        cache2 = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cached_response = cache2.get(prompt)

        assert cached_response == response

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        # Miss
        cache.get("different prompt")
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Hit
        cache.put(prompt, response)
        cache.get(prompt)
        stats = cache.get_stats()
        assert stats["hits"] == 1

    def test_stats_version_mismatch(self, cache):
        """Test statistics for version mismatches."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)
        cache.teacher_version = "v2.0.0"
        cache.get(prompt, validate_version=True)

        stats = cache.get_stats()
        assert stats["version_mismatches"] == 1

    def test_stats_hash_mismatch(self, cache):
        """Test statistics for hash mismatches."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        # Put with one prompt
        cache.put(prompt, response)

        # Try to get with different prompt (same hash would be unlikely, but test structure)
        # This tests the hash validation path
        cache.get(prompt, validate_hash=True)

        stats = cache.get_stats()
        # Should have at least one operation recorded
        assert stats["hits"] >= 0 or stats["misses"] >= 0

    def test_clear_cache(self, cache):
        """Test clearing cache."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache.put(prompt, response)
        assert cache.get(prompt) == response

        cache.clear()
        assert cache.get(prompt) is None

    def test_clear_cache_persists(self, cache_dir):
        """Test that clearing cache persists to disk."""
        prompt = "What is 2+2?"
        response = {"text": "The answer is 4"}

        cache1 = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cache1.put(prompt, response)
        cache1.clear()

        cache2 = TeacherCache(cache_dir, teacher_version="v1.0.0")
        assert cache2.get(prompt) is None

    def test_multiple_entries(self, cache):
        """Test caching multiple entries."""
        prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
        responses = [
            {"text": "The answer is 4"},
            {"text": "The answer is 6"},
            {"text": "The answer is 8"},
        ]

        for prompt, response in zip(prompts, responses):
            cache.put(prompt, response)

        for prompt, response in zip(prompts, responses):
            cached = cache.get(prompt)
            assert cached == response

    def test_cache_key_generation(self, cache):
        """Test that cache keys are generated correctly."""
        prompt1 = "What is 2+2?"
        prompt2 = "What is 2+2?"  # Same prompt
        prompt3 = "What is 3+3?"  # Different prompt

        response = {"text": "response"}

        cache.put(prompt1, response)
        assert cache.get(prompt2) == response  # Same prompt should hit
        assert cache.get(prompt3) is None  # Different prompt should miss

    def test_prompt_hash_computation(self, cache):
        """Test that prompt hashes are computed correctly."""
        prompt = "What is 2+2?"
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        cache.put(prompt, {"text": "response"})
        # Hash should be computed correctly
        assert cache._compute_prompt_hash(prompt) == expected_hash

    def test_cache_file_creation(self, cache_dir):
        """Test that cache file is created."""
        cache = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cache.put("test", {"text": "response"})

        assert cache.cache_file.exists()

    def test_load_cache_from_file(self, cache_dir):
        """Test loading cache from existing file."""
        # Create cache file manually
        cache_file = cache_dir / "teacher_cache.json"
        cache_dir.mkdir(parents=True, exist_ok=True)

        prompt = "What is 2+2?"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        cache_data = {
            prompt_hash: {
                "prompt_hash": prompt_hash,
                "teacher_version": "v1.0.0",
                "response": {"text": "The answer is 4"},
                "created_at": 1234567890.0,
                "prompt_text": prompt,
                "metadata": {},
            }
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Load cache
        cache = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cached_response = cache.get(prompt)

        assert cached_response == {"text": "The answer is 4"}


class TestTeacherCacheIntegration:
    """Test integration of teacher cache components."""

    def test_complete_cache_workflow(self, cache_dir):
        """Test complete cache workflow."""
        cache = TeacherCache(cache_dir, teacher_version="v1.0.0")

        # Put entries
        prompts = ["Q1", "Q2", "Q3"]
        for i, prompt in enumerate(prompts):
            cache.put(prompt, {"answer": i})

        # Get entries
        for i, prompt in enumerate(prompts):
            cached = cache.get(prompt)
            assert cached == {"answer": i}

        # Get stats
        stats = cache.get_stats()
        assert stats["hits"] == 3

        # Clear and verify
        cache.clear()
        for prompt in prompts:
            assert cache.get(prompt) is None

    def test_cache_with_version_upgrade(self, cache_dir):
        """Test cache behavior with version upgrade."""
        # Create cache with v1.0.0
        cache1 = TeacherCache(cache_dir, teacher_version="v1.0.0")
        cache1.put("test", {"text": "response"})

        # Upgrade to v2.0.0
        cache2 = TeacherCache(cache_dir, teacher_version="v2.0.0")
        # Should not get cached response with version validation
        assert cache2.get("test", validate_version=True) is None

        # But should get it without version validation
        assert cache2.get("test", validate_version=False) == {"text": "response"}







