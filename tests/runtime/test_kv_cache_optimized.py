"""
Tests for coreml/runtime/kv_cache_optimized.py - Optimized KV cache.

Tests KV cache initialization, updates, slicing, memory management,
ANE-friendly layouts, and performance characteristics.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch
import numpy as np

import pytest

# Import the module
from coreml.runtime.kv_cache_optimized import OptimizedKVCache


class TestOptimizedKVCacheInit:
    """Test OptimizedKVCache initialization."""

    def test_init_numpy_fp16(self):
        """Test initialization with numpy arrays and fp16 precision."""
        cache = OptimizedKVCache(
            n_heads=4,
            head_dim=64,
            max_seq_len=128,
            precision="fp16",
            use_numpy=True
        )

        assert cache.n_heads == 4
        assert cache.head_dim == 64
        assert cache.max_seq_len == 128
        assert cache.precision == "fp16"
        assert cache.use_numpy is True
        assert cache.current_len == 0

        # Check cache shapes and dtypes
        assert cache.k_cache.shape == (4, 128, 64)
        assert cache.v_cache.shape == (4, 128, 64)
        assert cache.k_cache.dtype == np.float16
        assert cache.v_cache.dtype == np.float16

        # Check cache size calculation
        expected_size = 4 * 64 * 128 * 2 * 2  # n_heads * head_dim * max_seq_len * 2(K+V) * 2(fp16)
        assert cache.cache_size_bytes == expected_size

    def test_init_torch_fp32(self):
        """Test initialization with torch tensors and fp32 precision."""
        with patch("coreml.runtime.kv_cache_optimized.TORCH_AVAILABLE", True):
            import torch
            cache = OptimizedKVCache(
                n_heads=2,
                head_dim=32,
                max_seq_len=64,
                precision="fp32",
                use_numpy=False
            )

            assert cache.use_numpy is False
            assert isinstance(cache.k_cache, torch.Tensor)
            assert isinstance(cache.v_cache, torch.Tensor)
            assert cache.k_cache.dtype == torch.float32
            assert cache.v_cache.dtype == torch.float32
            assert cache.k_cache.shape == (2, 64, 32)

    def test_init_torch_not_available(self):
        """Test initialization when torch is not available but requested."""
        with patch("coreml.runtime.kv_cache_optimized.TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch not available"):
                OptimizedKVCache(
                    n_heads=2,
                    head_dim=32,
                    max_seq_len=64,
                    use_numpy=False
                )

    def test_init_invalid_precision(self):
        """Test initialization with invalid precision."""
        with pytest.raises(RuntimeError, match="torch not available and numpy dtype not set"):
            # Force dtype to None by patching numpy
            with patch("numpy.float16", None):
                OptimizedKVCache(
                    n_heads=2,
                    head_dim=32,
                    max_seq_len=64,
                    precision="fp16",
                    use_numpy=True
                )


class TestOptimizedKVCacheOperations:
    """Test OptimizedKVCache operations."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return OptimizedKVCache(
            n_heads=4,
            head_dim=64,
            max_seq_len=128,
            precision="fp16",
            use_numpy=True
        )

    def test_update_single_position(self, cache):
        """Test updating cache at a single position."""
        # Create test KV tensors
        k_new = np.random.randn(4, 1, 64).astype(np.float16)
        v_new = np.random.randn(4, 1, 64).astype(np.float16)
        position = 5

        cache.update(k_new, v_new, position)

        # Check that values were stored
        assert np.allclose(cache.k_cache[:, position:position+1, :], k_new)
        assert np.allclose(cache.v_cache[:, position:position+1, :], v_new)
        assert cache.current_len == position + 1

    def test_update_multiple_positions(self, cache):
        """Test updating cache at multiple positions."""
        # Create test KV tensors for 3 positions
        k_new = np.random.randn(4, 3, 64).astype(np.float16)
        v_new = np.random.randn(4, 3, 64).astype(np.float16)
        position = 10

        cache.update(k_new, v_new, position)

        # Check that values were stored
        assert np.allclose(cache.k_cache[:, position:position+3, :], k_new)
        assert np.allclose(cache.v_cache[:, position:position+3, :], v_new)
        assert cache.current_len == position + 3

    def test_update_out_of_bounds(self, cache):
        """Test updating cache beyond max sequence length."""
        k_new = np.random.randn(4, 5, 64).astype(np.float16)
        v_new = np.random.randn(4, 5, 64).astype(np.float16)
        position = cache.max_seq_len - 2  # Only 2 spots left

        # This should not raise an error, but truncate
        cache.update(k_new, v_new, position)

        # Should have updated only the available positions
        assert cache.current_len == cache.max_seq_len

    def test_get_slice(self, cache):
        """Test getting a slice from the cache."""
        # First populate some data
        k_data = np.random.randn(4, 10, 64).astype(np.float16)
        v_data = np.random.randn(4, 10, 64).astype(np.float16)

        cache.update(k_data, v_data, 0)
        cache.current_len = 10

        # Get a slice
        k_slice, v_slice = cache.get_slice(2, 8)

        assert k_slice.shape == (4, 6, 64)  # 8-2 = 6 positions
        assert v_slice.shape == (4, 6, 64)
        assert np.allclose(k_slice, k_data[:, 2:8, :])
        assert np.allclose(v_slice, v_data[:, 2:8, :])

    def test_get_slice_empty_cache(self, cache):
        """Test getting a slice from empty cache."""
        k_slice, v_slice = cache.get_slice(0, 5)

        # Should return zeros
        assert k_slice.shape == (4, 5, 64)
        assert v_slice.shape == (4, 5, 64)
        assert np.all(k_slice == 0)
        assert np.all(v_slice == 0)

    def test_get_slice_beyond_current_len(self, cache):
        """Test getting a slice beyond current cache length."""
        # Populate some data
        k_data = np.random.randn(4, 5, 64).astype(np.float16)
        v_data = np.random.randn(4, 5, 64).astype(np.float16)
        cache.update(k_data, v_data, 0)
        cache.current_len = 5

        # Try to get slice beyond current length
        k_slice, v_slice = cache.get_slice(2, 10)

        # Should return available data + zeros
        assert k_slice.shape == (4, 8, 64)  # 10-2 = 8 positions
        assert np.allclose(k_slice[:, :3, :], k_data[:, 2:5, :])  # Available data
        assert np.all(k_slice[:, 3:, :] == 0)  # Rest should be zeros

    def test_reset(self, cache):
        """Test cache reset functionality."""
        # Populate cache
        k_data = np.random.randn(4, 5, 64).astype(np.float16)
        v_data = np.random.randn(4, 5, 64).astype(np.float16)
        cache.update(k_data, v_data, 0)

        # Reset
        cache.reset()

        # Should be empty
        assert cache.current_len == 0
        assert np.all(cache.k_cache == 0)
        assert np.all(cache.v_cache == 0)


class TestOptimizedKVCacheMemory:
    """Test OptimizedKVCache memory management."""

    def test_memory_calculation_fp16(self):
        """Test memory size calculation for fp16."""
        cache = OptimizedKVCache(
            n_heads=8,
            head_dim=128,
            max_seq_len=2048,
            precision="fp16",
            use_numpy=True
        )

        # Calculate expected size: n_heads * head_dim * max_seq_len * 2(K+V) * 2(fp16 bytes)
        expected = 8 * 128 * 2048 * 2 * 2
        assert cache.cache_size_bytes == expected

    def test_memory_calculation_fp32(self):
        """Test memory size calculation for fp32."""
        cache = OptimizedKVCache(
            n_heads=4,
            head_dim=64,
            max_seq_len=1024,
            precision="fp32",
            use_numpy=True
        )

        # Calculate expected size: n_heads * head_dim * max_seq_len * 2(K+V) * 4(fp32 bytes)
        expected = 4 * 64 * 1024 * 2 * 4
        assert cache.cache_size_bytes == expected

    def test_large_cache_creation(self):
        """Test creating a large cache (simulating production use)."""
        # This should not raise memory errors (simulates M3 Max with 64GB)
        cache = OptimizedKVCache(
            n_heads=32,
            head_dim=128,
            max_seq_len=4096,
            precision="fp16",
            use_numpy=True
        )

        # Verify cache was created successfully
        assert cache.k_cache.shape == (32, 4096, 128)
        assert cache.v_cache.shape == (32, 4096, 128)

        # Verify memory calculation
        # 32 * 128 * 4096 * 2 * 2 = ~67MB (reasonable for 64GB system)
        expected_mb = (32 * 128 * 4096 * 2 * 2) / (1024 * 1024)
        assert cache.cache_size_bytes / (1024 * 1024) == expected_mb


class TestOptimizedKVCachePerformance:
    """Test OptimizedKVCache performance characteristics."""

    @pytest.fixture
    def large_cache(self):
        """Create a large cache for performance testing."""
        return OptimizedKVCache(
            n_heads=16,
            head_dim=128,
            max_seq_len=2048,
            precision="fp16",
            use_numpy=True
        )

    def test_update_performance(self, large_cache, benchmark):
        """Test update performance."""
        k_new = np.random.randn(16, 32, 128).astype(np.float16)
        v_new = np.random.randn(16, 32, 128).astype(np.float16)

        # Benchmark update operation
        def update_op():
            large_cache.update(k_new, v_new, 100)

        benchmark(update_op)

    def test_slice_performance(self, large_cache, benchmark):
        """Test slice performance."""
        # Populate cache first
        k_data = np.random.randn(16, 512, 128).astype(np.float16)
        v_data = np.random.randn(16, 512, 128).astype(np.float16)
        large_cache.update(k_data, v_data, 0)
        large_cache.current_len = 512

        # Benchmark slice operation
        def slice_op():
            return large_cache.get_slice(100, 200)

        result = benchmark(slice_op)
        assert result[0].shape == (16, 100, 128)  # 200-100 = 100 positions

    def test_memory_contiguity(self, cache):
        """Test that cache arrays are memory contiguous (ANE-friendly)."""
        # Populate some data
        k_data = np.random.randn(4, 10, 64).astype(np.float16)
        v_data = np.random.randn(4, 10, 64).astype(np.float16)
        cache.update(k_data, v_data, 0)

        # Check contiguity (important for ANE performance)
        assert cache.k_cache.flags.c_contiguous
        assert cache.v_cache.flags.c_contiguous

    def test_cache_alignment(self, cache):
        """Test that cache arrays are properly aligned."""
        # Check if arrays are aligned to 64-byte boundaries (good for ANE)
        k_ptr = cache.k_cache.__array_interface__['data'][0]
        v_ptr = cache.v_cache.__array_interface__['data'][0]

        # Should be aligned to at least 64 bytes
        assert k_ptr % 64 == 0
        assert v_ptr % 64 == 0


class TestOptimizedKVCacheTorch:
    """Test OptimizedKVCache with torch tensors."""

    @pytest.fixture
    def torch_cache(self):
        """Create a torch-based cache."""
        with patch("coreml.runtime.kv_cache_optimized.TORCH_AVAILABLE", True):
            import torch
            return OptimizedKVCache(
                n_heads=4,
                head_dim=64,
                max_seq_len=128,
                precision="fp16",
                use_numpy=False
            )

    def test_torch_tensor_operations(self, torch_cache):
        """Test tensor operations with torch backend."""
        # Create test tensors
        k_new = torch.randn(4, 5, 64, dtype=torch.float16)
        v_new = torch.randn(4, 5, 64, dtype=torch.float16)

        torch_cache.update(k_new, v_new, 10)

        # Verify update
        assert torch.allclose(torch_cache.k_cache[:, 10:15, :], k_new)
        assert torch.allclose(torch_cache.v_cache[:, 10:15, :], v_new)
        assert torch_cache.current_len == 15

    def test_torch_slice_operations(self, torch_cache):
        """Test slice operations with torch tensors."""
        # Populate cache
        k_data = torch.randn(4, 10, 64, dtype=torch.float16)
        v_data = torch.randn(4, 10, 64, dtype=torch.float16)
        torch_cache.update(k_data, v_data, 0)
        torch_cache.current_len = 10

        # Get slice
        k_slice, v_slice = torch_cache.get_slice(3, 8)

        assert k_slice.shape == (4, 5, 64)
        assert v_slice.shape == (4, 5, 64)
        assert torch.allclose(k_slice, k_data[:, 3:8, :])
        assert torch.allclose(v_slice, v_data[:, 3:8, :])


class TestOptimizedKVCacheEdgeCases:
    """Test OptimizedKVCache edge cases."""

    def test_zero_sequence_length(self):
        """Test cache with zero max sequence length."""
        cache = OptimizedKVCache(
            n_heads=4,
            head_dim=64,
            max_seq_len=0,
            precision="fp16",
            use_numpy=True
        )

        assert cache.k_cache.shape == (4, 0, 64)
        assert cache.v_cache.shape == (4, 0, 64)
        assert cache.current_len == 0

    def test_single_head(self):
        """Test cache with single attention head."""
        cache = OptimizedKVCache(
            n_heads=1,
            head_dim=64,
            max_seq_len=128,
            precision="fp16",
            use_numpy=True
        )

        assert cache.k_cache.shape == (1, 128, 64)
        assert cache.v_cache.shape == (1, 128, 64)

    def test_large_head_dimension(self):
        """Test cache with large head dimension."""
        cache = OptimizedKVCache(
            n_heads=4,
            head_dim=512,
            max_seq_len=128,
            precision="fp32",
            use_numpy=True
        )

        assert cache.k_cache.shape == (4, 128, 512)
        assert cache.k_cache.dtype == np.float32

    def test_update_at_max_capacity(self, cache):
        """Test updating when cache is at maximum capacity."""
        # Fill cache to maximum
        max_k = np.random.randn(4, cache.max_seq_len, 64).astype(np.float16)
        max_v = np.random.randn(4, cache.max_seq_len, 64).astype(np.float16)

        cache.update(max_k, max_v, 0)
        cache.current_len = cache.max_seq_len

        # Try to update beyond capacity
        extra_k = np.random.randn(4, 5, 64).astype(np.float16)
        extra_v = np.random.randn(4, 5, 64).astype(np.float16)

        # This should not raise an error but should not update either
        cache.update(extra_k, extra_v, cache.max_seq_len)

        # Length should remain at maximum
        assert cache.current_len == cache.max_seq_len
