"""
Unit tests for optimized KV cache.

Tests ANE-friendly layout and unified memory optimization for M-series Apple Silicon.
"""
import pytest
import numpy as np

try:
    from coreml.runtime.kv_cache_optimized import (
        OptimizedKVCache,
        GroupedQueryKVCache,
        create_kv_cache_for_model,
    )
    KV_CACHE_AVAILABLE = True
except ImportError:
    KV_CACHE_AVAILABLE = False


@pytest.mark.skipif(not KV_CACHE_AVAILABLE, reason="KV cache optimization not available")
class TestOptimizedKVCache:
    """Tests for OptimizedKVCache class."""
    
    def test_initialization(self):
        """Test KV cache initialization."""
        cache = OptimizedKVCache(
            n_heads=32,
            head_dim=128,
            max_seq_len=4096,
            precision="fp16",
        )
        
        assert cache.n_heads == 32
        assert cache.head_dim == 128
        assert cache.max_seq_len == 4096
        assert cache.precision == "fp16"
        assert cache.current_len == 0
        assert cache.k_cache.shape == (32, 4096, 128)
        assert cache.v_cache.shape == (32, 4096, 128)
    
    def test_update(self):
        """Test updating cache."""
        cache = OptimizedKVCache(n_heads=4, head_dim=8, max_seq_len=100)
        
        k = np.random.randn(4, 8).astype(np.float16)
        v = np.random.randn(4, 8).astype(np.float16)
        
        cache.update(k, v, position=10)
        
        assert cache.current_len == 11
        assert cache.updates == 1
        assert np.allclose(cache.k_cache[:, 10, :], k)
        assert np.allclose(cache.v_cache[:, 10, :], v)
    
    def test_get_slice(self):
        """Test getting cache slice."""
        cache = OptimizedKVCache(n_heads=4, head_dim=8, max_seq_len=100)
        
        # Update some positions
        for i in range(5):
            k = np.random.randn(4, 8).astype(np.float16)
            v = np.random.randn(4, 8).astype(np.float16)
            cache.update(k, v, position=i)
        
        # Get slice
        k_slice, v_slice = cache.get_slice(0, 5)
        
        assert k_slice.shape == (4, 5, 8)
        assert v_slice.shape == (4, 5, 8)
        assert cache.slices_retrieved == 1
    
    def test_get_full(self):
        """Test getting full cache."""
        cache = OptimizedKVCache(n_heads=4, head_dim=8, max_seq_len=100)
        
        # Update some positions
        for i in range(10):
            k = np.random.randn(4, 8).astype(np.float16)
            v = np.random.randn(4, 8).astype(np.float16)
            cache.update(k, v, position=i)
        
        k_full, v_full = cache.get_full()
        
        assert k_full.shape == (4, 10, 8)
        assert v_full.shape == (4, 10, 8)
    
    def test_clear(self):
        """Test clearing cache."""
        cache = OptimizedKVCache(n_heads=4, head_dim=8, max_seq_len=100)
        
        k = np.random.randn(4, 8).astype(np.float16)
        v = np.random.randn(4, 8).astype(np.float16)
        cache.update(k, v, position=5)
        
        assert cache.current_len == 6
        
        cache.clear()
        
        assert cache.current_len == 0
        assert np.all(cache.k_cache == 0)
        assert np.all(cache.v_cache == 0)
    
    def test_get_size_mb(self):
        """Test getting cache size."""
        cache = OptimizedKVCache(n_heads=32, head_dim=128, max_seq_len=4096, precision="fp16")
        
        size_mb = cache.get_size_mb()
        
        # Expected: 32 heads * 128 dim * 4096 seq * 2 (K+V) * 2 bytes (fp16) / (1024*1024)
        expected = (32 * 128 * 4096 * 2 * 2) / (1024 * 1024)
        assert abs(size_mb - expected) < 0.1  # Allow small floating point error
    
    def test_stats(self):
        """Test statistics tracking."""
        cache = OptimizedKVCache(n_heads=4, head_dim=8, max_seq_len=100)
        
        k = np.random.randn(4, 8).astype(np.float16)
        v = np.random.randn(4, 8).astype(np.float16)
        cache.update(k, v, position=0)
        cache.get_slice(0, 1)
        
        stats = cache.stats()
        
        assert stats["n_heads"] == 4
        assert stats["head_dim"] == 8
        assert stats["max_seq_len"] == 100
        assert stats["current_len"] == 1
        assert stats["updates"] == 1
        assert stats["slices_retrieved"] == 1


@pytest.mark.skipif(not KV_CACHE_AVAILABLE, reason="KV cache optimization not available")
class TestGroupedQueryKVCache:
    """Tests for GroupedQueryKVCache class."""
    
    def test_initialization(self):
        """Test GQA cache initialization."""
        cache = GroupedQueryKVCache(
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            max_seq_len=4096,
        )
        
        assert cache.n_query_heads == 32
        assert cache.n_kv_heads == 8
        assert cache.num_query_groups == 4
        assert cache.k_cache.shape == (8, 4096, 128)  # Fewer KV heads
        assert cache.v_cache.shape == (8, 4096, 128)
    
    def test_gqa_reduction(self):
        """Test that GQA reduces cache size."""
        standard_cache = OptimizedKVCache(n_heads=32, head_dim=128, max_seq_len=4096)
        gqa_cache = GroupedQueryKVCache(n_heads=32, n_kv_heads=8, head_dim=128, max_seq_len=4096)
        
        standard_size = standard_cache.get_size_mb()
        gqa_size = gqa_cache.get_size_mb()
        
        # GQA should be 4x smaller (32/8 = 4)
        assert abs(gqa_size * 4 - standard_size) < 0.1
    
    def test_stats(self):
        """Test GQA statistics."""
        cache = GroupedQueryKVCache(n_heads=32, n_kv_heads=8, head_dim=128, max_seq_len=4096)
        
        stats = cache.stats()
        
        assert stats["n_query_heads"] == 32
        assert stats["n_kv_heads"] == 8
        assert stats["num_query_groups"] == 4
        assert stats["gqa_reduction_factor"] == 4


@pytest.mark.skipif(not KV_CACHE_AVAILABLE, reason="KV cache optimization not available")
class TestCreateKVCacheForModel:
    """Tests for create_kv_cache_for_model convenience function."""
    
    def test_create_standard_cache(self):
        """Test creating standard MHA cache."""
        cache = create_kv_cache_for_model(
            n_heads=32,
            head_dim=128,
            max_seq_len=4096,
        )
        
        assert isinstance(cache, OptimizedKVCache)
        assert cache.n_heads == 32
    
    def test_create_gqa_cache(self):
        """Test creating GQA cache."""
        cache = create_kv_cache_for_model(
            n_heads=32,
            head_dim=128,
            max_seq_len=4096,
            num_query_groups=4,
        )
        
        assert isinstance(cache, GroupedQueryKVCache)
        assert cache.n_query_heads == 32
        assert cache.n_kv_heads == 8

