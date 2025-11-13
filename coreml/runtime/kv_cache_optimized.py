"""
Optimized KV cache for M-series Apple Silicon optimization.

Pre-allocates KV cache with ANE-friendly layout and unified memory advantage.
Reduces memory allocations and improves inference efficiency.

Reference: docs/M_SERIES_ADVANCED_OPTIMIZATIONS.md Phase 11
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OptimizedKVCache:
    """
    KV cache optimized for ANE and unified memory.

    Benefits:
    - Reduced memory allocations (pre-allocated cache)
    - ANE-friendly layout (contiguous, aligned)
    - Unified memory advantage (no pressure with 64GB)

    Layout:
    - K cache: [n_heads, max_seq_len, head_dim]
    - V cache: [n_heads, max_seq_len, head_dim]

    This layout is optimal for ANE attention operations.

    Usage:
        cache = OptimizedKVCache(n_heads=32, head_dim=128, max_seq_len=4096)
        cache.update(k_new, v_new, position=10)
        k_slice, v_slice = cache.get_slice(0, 20)
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        precision: str = "fp16",
        use_numpy: bool = True,
    ):
        """
        Initialize optimized KV cache.

        Args:
            n_heads: Number of attention heads
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            precision: Data type precision ("fp16" or "fp32")
            use_numpy: Use numpy arrays (True) or torch tensors (False)
        """
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.precision = precision
        self.use_numpy = use_numpy

        # Determine dtype
        if precision == "fp16":
            if use_numpy:
                dtype = np.float16
            else:
                dtype = torch.float16 if TORCH_AVAILABLE else None
            element_size = 2
        else:  # fp32
            if use_numpy:
                dtype = np.float32
            else:
                dtype = torch.float32 if TORCH_AVAILABLE else None
            element_size = 4

        if dtype is None:
            raise RuntimeError("torch not available and numpy dtype not set")

        # Pre-allocate KV cache (unified memory advantage)
        # Layout: [n_heads, max_seq_len, head_dim]
        if use_numpy:
            self.k_cache = np.zeros((n_heads, max_seq_len, head_dim), dtype=dtype)
            self.v_cache = np.zeros((n_heads, max_seq_len, head_dim), dtype=dtype)
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("torch not available for tensor allocation")
            self.k_cache = torch.zeros(n_heads, max_seq_len, head_dim, dtype=dtype)
            self.v_cache = torch.zeros(n_heads, max_seq_len, head_dim, dtype=dtype)

        # Track current sequence length
        self.current_len = 0

        # Calculate cache size
        kv_size_bytes = n_heads * head_dim * max_seq_len * 2 * element_size  # K+V
        self.cache_size_bytes = kv_size_bytes

        # Statistics
        self.updates = 0
        self.slices_retrieved = 0

    def update(
        self,
        k: np.ndarray,
        v: np.ndarray,
        position: int,
    ):
        """
        Update cache at position (in-place for efficiency).

        Args:
            k: Key tensor [n_heads, 1, head_dim] or [n_heads, head_dim]
            v: Value tensor [n_heads, 1, head_dim] or [n_heads, head_dim]
            position: Position in sequence to update
        """
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(f"Position {position} out of range [0, {self.max_seq_len})")

        # Ensure k and v are correct shape
        if k.ndim == 2:
            # [n_heads, head_dim] -> [n_heads, 1, head_dim]
            k = k[:, None, :]
        if v.ndim == 2:
            # [n_heads, head_dim] -> [n_heads, 1, head_dim]
            v = v[:, None, :]

        # Convert to numpy if torch tensors
        if not self.use_numpy and TORCH_AVAILABLE:
            if isinstance(k, torch.Tensor):
                k = k.cpu().numpy()
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()

        # In-place update (ANE-friendly)
        self.k_cache[:, position : position + 1, :] = k
        self.v_cache[:, position : position + 1, :] = v

        # Update current length
        self.current_len = max(self.current_len, position + 1)
        self.updates += 1

    def get_slice(
        self,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get KV cache slice (ANE-optimal layout).

        Args:
            start: Start position (inclusive)
            end: End position (exclusive)

        Returns:
            Tuple of (k_slice, v_slice) with shape [n_heads, end-start, head_dim]
        """
        if start < 0 or end > self.max_seq_len or start >= end:
            raise ValueError(f"Invalid slice range [{start}, {end})")

        # Return contiguous slice (ANE-friendly)
        k_slice = self.k_cache[:, start:end, :]
        v_slice = self.v_cache[:, start:end, :]

        self.slices_retrieved += 1

        return k_slice, v_slice

    def get_full(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full KV cache up to current length.

        Returns:
            Tuple of (k_cache, v_cache) with shape [n_heads, current_len, head_dim]
        """
        return self.get_slice(0, self.current_len)

    def clear(self):
        """Clear cache (reset to zeros)."""
        if self.use_numpy:
            self.k_cache.fill(0)
            self.v_cache.fill(0)
        else:
            if TORCH_AVAILABLE:
                self.k_cache.zero_()
                self.v_cache.zero_()
        self.current_len = 0

    def get_size_mb(self) -> float:
        """
        Get cache size in MB.

        Returns:
            Cache size in megabytes
        """
        return self.cache_size_bytes / (1024 * 1024)

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "current_len": self.current_len,
            "precision": self.precision,
            "cache_size_mb": self.get_size_mb(),
            "updates": self.updates,
            "slices_retrieved": self.slices_retrieved,
        }


class GroupedQueryKVCache(OptimizedKVCache):
    """
    KV cache optimized for Grouped Query Attention (GQA).

    GQA uses fewer KV heads than query heads, reducing cache size.

    Layout:
    - K cache: [n_kv_heads, max_seq_len, head_dim]
    - V cache: [n_kv_heads, max_seq_len, head_dim]

    Where n_kv_heads < n_heads (typically n_kv_heads = n_heads / num_query_groups)
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        precision: str = "fp16",
        use_numpy: bool = True,
    ):
        """
        Initialize GQA-optimized KV cache.

        Args:
            n_heads: Number of query heads
            n_kv_heads: Number of KV heads (typically n_heads / num_query_groups)
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            precision: Data type precision ("fp16" or "fp32")
            use_numpy: Use numpy arrays (True) or torch tensors (False)
        """
        # Use n_kv_heads instead of n_heads for cache
        super().__init__(
            n_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            precision=precision,
            use_numpy=use_numpy,
        )

        self.n_query_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_query_groups = n_heads // n_kv_heads

        # Recalculate cache size (smaller due to fewer KV heads)
        element_size = 2 if precision == "fp16" else 4
        kv_size_bytes = n_kv_heads * head_dim * max_seq_len * 2 * element_size
        self.cache_size_bytes = kv_size_bytes

    def stats(self) -> Dict[str, Any]:
        """Get GQA cache statistics."""
        base_stats = super().stats()
        base_stats.update(
            {
                "n_query_heads": self.n_query_heads,
                "n_kv_heads": self.n_kv_heads,
                "num_query_groups": self.num_query_groups,
                "gqa_reduction_factor": self.num_query_groups,
            }
        )
        return base_stats


def create_kv_cache_for_model(
    n_heads: int,
    head_dim: int,
    max_seq_len: int,
    num_query_groups: Optional[int] = None,
    precision: str = "fp16",
    use_numpy: bool = True,
) -> OptimizedKVCache:
    """
    Create optimized KV cache for a model.

    Args:
        n_heads: Number of attention heads
        head_dim: Dimension per head
        max_seq_len: Maximum sequence length
        num_query_groups: Number of query groups for GQA (None for standard MHA)
        precision: Data type precision ("fp16" or "fp32")
        use_numpy: Use numpy arrays (True) or torch tensors (False)

    Returns:
        OptimizedKVCache instance (or GroupedQueryKVCache if num_query_groups specified)
    """
    if num_query_groups is not None and num_query_groups > 1:
        n_kv_heads = n_heads // num_query_groups
        return GroupedQueryKVCache(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            precision=precision,
            use_numpy=use_numpy,
        )
    else:
        return OptimizedKVCache(
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            precision=precision,
            use_numpy=use_numpy,
        )
