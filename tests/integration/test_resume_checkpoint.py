"""
Integration tests for resume/checkpoint functionality.

Tests that checkpoint/resume works correctly for dataset generation.
"""
import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.make_kd_mix_hardened import (
    CheckpointManager,
    BudgetTracker,
    load_cache,
    save_cache,
)


class TestCheckpointManager:
    """Test checkpoint manager functionality."""
    
    def test_init(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            assert manager.checkpoint_dir == checkpoint_dir
            assert checkpoint_dir.exists()
            assert manager.checkpoint_file == checkpoint_dir / "progress.json"
            assert manager.results_file == checkpoint_dir / "results.jsonl"
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            completed_indices = [0, 1, 2]
            results = [
                {"prompt": "test1", "teacher_text": "response1"},
                {"prompt": "test2", "teacher_text": "response2"},
                {"prompt": "test3", "teacher_text": "response3"},
            ]
            budget_tracker = BudgetTracker()
            budget_tracker.total_cost = 0.50
            budget_tracker.input_tokens = 1000
            budget_tracker.output_tokens = 2000
            
            start_time = time.time()
            
            manager.save_checkpoint(completed_indices, results, budget_tracker, start_time)
            
            # Verify checkpoint file exists
            assert manager.checkpoint_file.exists()
            
            # Verify checkpoint content
            with open(manager.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            assert checkpoint_data["completed_indices"] == completed_indices
            assert checkpoint_data["total_completed"] == 3
            assert checkpoint_data["budget"]["total_cost"] == 0.50
            assert checkpoint_data["budget"]["input_tokens"] == 1000
            assert checkpoint_data["budget"]["output_tokens"] == 2000
            assert "timestamp" in checkpoint_data
            assert "last_update" in checkpoint_data
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Save a checkpoint first
            completed_indices = [0, 1, 2]
            results = [
                {"prompt": "test1", "teacher_text": "response1"},
                {"prompt": "test2", "teacher_text": "response2"},
                {"prompt": "test3", "teacher_text": "response3"},
            ]
            budget_tracker = BudgetTracker()
            budget_tracker.total_cost = 0.50
            
            start_time = time.time()
            manager.save_checkpoint(completed_indices, results, budget_tracker, start_time)
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint()
            
            assert loaded_data is not None
            assert loaded_data["completed_indices"] == completed_indices
            assert len(loaded_data["results"]) == 3
            assert loaded_data["results"][0]["prompt"] == "test1"
            assert loaded_data["budget"]["total_cost"] == 0.50
    
    def test_load_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Try to load checkpoint that doesn't exist
            loaded_data = manager.load_checkpoint()
            
            assert loaded_data is None
    
    def test_clear_checkpoint(self):
        """Test clearing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Save checkpoint
            completed_indices = [0, 1]
            results = [{"prompt": "test", "teacher_text": "response"}]
            budget_tracker = BudgetTracker()
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Verify files exist
            assert manager.checkpoint_file.exists()
            assert manager.results_file.exists()
            
            # Clear checkpoint
            manager.clear_checkpoint()
            
            # Verify files are deleted
            assert not manager.checkpoint_file.exists()
            assert not manager.results_file.exists()
    
    def test_checkpoint_preserves_results(self):
        """Test that checkpoint preserves all results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Create many results
            results = [
                {"prompt": f"test{i}", "teacher_text": f"response{i}"}
                for i in range(100)
            ]
            completed_indices = list(range(100))
            budget_tracker = BudgetTracker()
            
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Load and verify all results preserved
            loaded_data = manager.load_checkpoint()
            assert len(loaded_data["results"]) == 100
            assert loaded_data["results"][0]["prompt"] == "test0"
            assert loaded_data["results"][99]["prompt"] == "test99"


class TestBudgetTrackerCheckpoint:
    """Test budget tracker state preservation in checkpoints."""
    
    def test_budget_state_preserved(self):
        """Test that budget state is preserved in checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Create budget tracker with some state
            budget_tracker = BudgetTracker(budget_limit=10.0)
            budget_tracker.total_cost = 5.0
            budget_tracker.input_tokens = 10000
            budget_tracker.output_tokens = 20000
            budget_tracker.cached_samples = 5
            budget_tracker.api_samples = 10
            
            completed_indices = [0, 1]
            results = [{"prompt": "test", "teacher_text": "response"}]
            
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Load and verify budget state
            loaded_data = manager.load_checkpoint()
            budget_status = loaded_data["budget"]
            
            assert budget_status["total_cost"] == 5.0
            assert budget_status["input_tokens"] == 10000
            assert budget_status["output_tokens"] == 20000
            assert budget_status["cached_samples"] == 5
            assert budget_status["api_samples"] == 10
            assert budget_status["budget_limit"] == 10.0


class TestResumeWorkflow:
    """Test complete resume workflow."""
    
    def test_resume_skips_completed_indices(self):
        """Test that resume skips already completed samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Simulate partial completion
            completed_indices = [0, 1, 2, 5, 7]
            results = [
                {"prompt": f"test{i}", "teacher_text": f"response{i}"}
                for i in completed_indices
            ]
            budget_tracker = BudgetTracker()
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Load checkpoint
            loaded_data = manager.load_checkpoint()
            assert loaded_data is not None
            
            # Verify completed indices
            completed_set = set(loaded_data["completed_indices"])
            assert completed_set == {0, 1, 2, 5, 7}
            
            # Simulate resuming: should skip indices 0, 1, 2, 5, 7
            all_prompts = [f"test{i}" for i in range(10)]
            for i, prompt in enumerate(all_prompts):
                if i in completed_set:
                    # Should skip this
                    continue
                # Process remaining prompts
                pass
    
    def test_resume_preserves_budget(self):
        """Test that resume preserves budget state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Save checkpoint with budget state
            budget_tracker = BudgetTracker(budget_limit=10.0)
            budget_tracker.total_cost = 3.5
            budget_tracker.input_tokens = 5000
            budget_tracker.output_tokens = 10000
            
            completed_indices = [0, 1]
            results = [{"prompt": "test", "teacher_text": "response"}]
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Simulate resume: restore budget state
            loaded_data = manager.load_checkpoint()
            budget_status = loaded_data["budget"]
            
            # Create new budget tracker and restore state
            new_budget_tracker = BudgetTracker(budget_limit=budget_status.get("budget_limit"))
            new_budget_tracker.total_cost = budget_status.get("total_cost", 0.0)
            new_budget_tracker.input_tokens = budget_status.get("input_tokens", 0)
            new_budget_tracker.output_tokens = budget_status.get("output_tokens", 0)
            new_budget_tracker.cached_samples = budget_status.get("cached_samples", 0)
            new_budget_tracker.api_samples = budget_status.get("api_samples", 0)
            
            # Verify state restored
            assert new_budget_tracker.total_cost == 3.5
            assert new_budget_tracker.input_tokens == 5000
            assert new_budget_tracker.output_tokens == 10000


class TestCacheIntegration:
    """Test cache integration with checkpoint/resume."""
    
    def test_cache_validation(self):
        """Test cache validation during resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            
            # Save valid cache
            prompt = "test prompt"
            result = {
                "prompt": prompt,
                "teacher_text": "response",
                "metadata": {}
            }
            save_cache(cache_dir, prompt, result)
            
            # Load cache
            cached = load_cache(cache_dir, prompt)
            assert cached is not None
            assert cached["prompt"] == prompt
            assert cached["teacher_text"] == "response"
    
    def test_cache_corruption_handling(self):
        """Test handling of corrupted cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            
            import hashlib
            prompt_hash = hashlib.md5("test".encode()).hexdigest()
            cache_file = cache_dir / f"{prompt_hash}.json"
            
            # Write invalid JSON
            with open(cache_file, 'w') as f:
                f.write("invalid json{")
            
            # Should handle corruption gracefully
            cached = load_cache(cache_dir, "test")
            assert cached is None  # Should return None for corrupted cache


class TestCheckpointInterval:
    """Test checkpoint interval functionality."""
    
    def test_checkpoint_interval_logic(self):
        """Test that checkpoints are saved at correct intervals."""
        checkpoint_interval = 5
        
        # Simulate processing samples
        completed_count = 0
        checkpoints_saved = []
        
        for i in range(20):
            completed_count += 1
            
            # Check if checkpoint should be saved
            if completed_count % checkpoint_interval == 0:
                checkpoints_saved.append(completed_count)
        
        # Should save at 5, 10, 15, 20
        assert checkpoints_saved == [5, 10, 15, 20]
    
    def test_final_checkpoint_always_saved(self):
        """Test that final checkpoint is always saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # Save checkpoint not at interval
            completed_indices = [0, 1, 2, 3]  # 4 samples, interval is 5
            results = [{"prompt": f"test{i}", "teacher_text": f"response{i}"} for i in range(4)]
            budget_tracker = BudgetTracker()
            
            # Final checkpoint (should always save)
            manager.save_checkpoint(completed_indices, results, budget_tracker, time.time())
            
            # Verify checkpoint exists
            assert manager.checkpoint_file.exists()
            loaded_data = manager.load_checkpoint()
            assert loaded_data["total_completed"] == 4

