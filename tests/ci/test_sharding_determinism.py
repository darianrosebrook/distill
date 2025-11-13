"""Basic smoke test for sharding determinism implementation."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.cli import stable_shard, select_shard


def test_stable_shard():
    """Test that stable_shard produces consistent results."""
    sample_id = "test-sample-123"
    num_shards = 4
    
    # Should produce same shard assignment every time
    shard1 = stable_shard(sample_id, num_shards)
    shard2 = stable_shard(sample_id, num_shards)
    
    assert shard1 == shard2, f"Stable shard inconsistent: {shard1} != {shard2}"
    assert 0 <= shard1 < num_shards, f"Shard index out of range: {shard1}"


def test_stable_shard_distribution():
    """Test that stable sharding distributes samples across shards."""
    num_shards = 4
    sample_ids = [f"sample-{i}" for i in range(100)]
    
    shard_counts = [0] * num_shards
    for sample_id in sample_ids:
        shard = stable_shard(sample_id, num_shards)
        shard_counts[shard] += 1
    
    # Should have samples in all shards (with some tolerance for randomness)
    assert all(count > 0 for count in shard_counts), f"Some shards empty: {shard_counts}"


def test_select_shard_with_sample_id():
    """Test select_shard with items that have sample_id."""
    items = [
        {"metadata": {"sample_id": "sample-0"}},
        {"metadata": {"sample_id": "sample-1"}},
        {"metadata": {"sample_id": "sample-2"}},
        {"metadata": {"sample_id": "sample-3"}},
        {"metadata": {"sample_id": "sample-4"}},
    ]
    
    num_shards = 2
    
    # Get items for each shard
    shard_0 = select_shard(items, 0, num_shards)
    shard_1 = select_shard(items, 1, num_shards)
    
    # Should partition all items
    assert len(shard_0) + len(shard_1) == len(items), "Items not partitioned correctly"
    
    # Should have no overlap
    shard_0_ids = {item["metadata"]["sample_id"] for item in shard_0}
    shard_1_ids = {item["metadata"]["sample_id"] for item in shard_1}
    assert len(shard_0_ids & shard_1_ids) == 0, "Shards overlap"


def test_select_shard_without_sample_id():
    """Test select_shard with items that don't have sample_id (should synthesize)."""
    items = [
        {"prompt": "test prompt 1"},
        {"prompt": "test prompt 2"},
        {"prompt": "test prompt 3"},
    ]
    
    num_shards = 2
    
    # Should still partition (synthesizes sample_id from row)
    shard_0 = select_shard(items, 0, num_shards)
    shard_1 = select_shard(items, 1, num_shards)
    
    assert len(shard_0) + len(shard_1) == len(items), "Items not partitioned correctly"


def test_select_shard_single_shard():
    """Test that single shard returns all items."""
    items = [
        {"metadata": {"sample_id": "sample-0"}},
        {"metadata": {"sample_id": "sample-1"}},
    ]
    
    result = select_shard(items, 0, 1)
    assert len(result) == len(items), "Single shard should return all items"
    assert result == items, "Single shard should return items unchanged"


if __name__ == "__main__":
    print("Running sharding determinism smoke tests...")
    
    try:
        test_stable_shard()
        print("✅ test_stable_shard passed")
        
        test_stable_shard_distribution()
        print("✅ test_stable_shard_distribution passed")
        
        test_select_shard_with_sample_id()
        print("✅ test_select_shard_with_sample_id passed")
        
        test_select_shard_without_sample_id()
        print("✅ test_select_shard_without_sample_id passed")
        
        test_select_shard_single_shard()
        print("✅ test_select_shard_single_shard passed")
        
        print("\n✅ All sharding determinism smoke tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

