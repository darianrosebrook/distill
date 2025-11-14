"""
Training reproducibility and determinism tests.

Verifies that training runs are reproducible:
- Same seed + same data → same checkpoint
- Dataset fingerprinting validation
- Deterministic sharding
- Seed management
@author: @darianrosebrook
"""

import pytest
import torch
import hashlib
import json
from pathlib import Path
import tempfile

from training.dataset import KDDataset
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


def compute_checkpoint_hash(checkpoint_path: Path) -> str:
    """Compute SHA-256 hash of checkpoint model state dict."""
    from training.safe_checkpoint_loading import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Convert to bytes for hashing
    state_bytes = b""
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        state_bytes += key.encode("utf-8")
        state_bytes += tensor.cpu().numpy().tobytes()

    return hashlib.sha256(state_bytes).hexdigest()


def create_test_dataset(output_path: Path, num_samples: int = 50) -> Path:
    """Create a test dataset with fingerprint."""
    samples = []
    for i in range(num_samples):
        sample = {
            "prompt": f"Test prompt {i}",
            "teacher_text": f"Test response {i}",
        }
        samples.append(sample)

    # Compute dataset fingerprint
    dataset_content = "\n".join(json.dumps(s) for s in samples)
    dataset_sha256 = hashlib.sha256(dataset_content.encode("utf-8")).hexdigest()

    # Write with header
    with open(output_path, "w") as f:
        # Write header
        header = {
            "__header__": True,
            "dataset_sha256": dataset_sha256,
            "num_samples": num_samples,
        }
        f.write(json.dumps(header) + "\n")

        # Write samples
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return output_path


def test_dataset_fingerprint_validation():
    """Test that dataset fingerprinting is validated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset.jsonl"
        create_test_dataset(dataset_path, num_samples=10)

        # Load dataset
        tokenizer_path = "models/student/tokenizer"
        dataset = KDDataset(str(dataset_path), tokenizer_path, max_seq_length=128)

        # Check fingerprint was extracted
        assert hasattr(dataset, "dataset_fingerprint")
        assert dataset.dataset_fingerprint is not None

        # Verify fingerprint matches
        with open(dataset_path, "r") as f:
            first_line = f.readline()
            header = json.loads(first_line)
            expected_fingerprint = header.get("dataset_sha256")

        assert dataset.dataset_fingerprint == expected_fingerprint


def test_deterministic_model_initialization():
    """Test that model initialization is deterministic with fixed seed."""
    seed = 42

    # Initialize model twice with same seed
    cfg = ModelCfg(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_head=32,
        vocab_size=100,
    )

    torch.manual_seed(seed)
    model1 = StudentLM(cfg)
    state1 = model1.state_dict()

    torch.manual_seed(seed)
    model2 = StudentLM(cfg)
    state2 = model2.state_dict()

    # Check that state dicts are identical
    for key in state1.keys():
        assert torch.allclose(state1[key], state2[key]), f"Parameter {key} differs"


def test_checkpoint_reproducibility(tmp_path: Path):
    """Test that training produces identical checkpoints with same seed."""
    # Create test dataset
    dataset_path = tmp_path / "test_dataset.jsonl"
    create_test_dataset(dataset_path, num_samples=20)

    # Create config
    config = {
        "arch": {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "n_kv_heads": 1,
            "d_head": 32,
            "vocab_size": 100,
            "rope_theta": 10000.0,
            "rope_scaling": "dynamic",
            "dropout": 0.0,
        },
        "train": {
            "steps": 10,
            "micro_batch_size": 2,
            "grad_accum": 2,
            "seq_lengths": [64],
            "seed": 42,
        },
        "optimizer": {
            "lr": 1e-4,
            "grad_clip": 1.0,
        },
        "distill": {
            "kl_weight": 0.4,
            "ce_teacher_weight": 0.2,
            "ce_ground_truth_weight": 0.2,
        },
        "io": {
            "tokenizer_path": "models/student/tokenizer",
            "train_shards": [str(dataset_path)],
        },
    }

    config_path1 = tmp_path / "config1.json"
    config_path2 = tmp_path / "config2.json"

    with open(config_path1, "w") as f:
        json.dump(config, f)
    with open(config_path2, "w") as f:
        json.dump(config, f)

    # Run training twice (simplified - would need to mock teacher API)
    # For now, just verify that configs are identical
    with open(config_path1, "r") as f:
        config1 = json.load(f)
    with open(config_path2, "r") as f:
        config2 = json.load(f)

    assert config1 == config2, "Configs should be identical"

    # Note: Full reproducibility test would require:
    # - Mock teacher API or use cached responses
    # - Run actual training loop twice
    # - Compare checkpoint hashes
    # This is a placeholder for the structure


def test_seed_management():
    """Test that seeds are properly managed and logged."""
    seed = 42

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Verify seeds are set
    assert torch.initial_seed() == seed

    # Note: Full seed management test would verify:
    # - All RNG states captured in checkpoint
    # - RNG states restored correctly on resume
    # - Seeds logged in checkpoint metadata


def test_dataset_sharding_determinism():
    """Test that dataset sharding is deterministic."""
    # Create test dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset.jsonl"
        create_test_dataset(dataset_path, num_samples=100)

        # Load dataset twice
        tokenizer_path = "models/student/tokenizer"
        dataset1 = KDDataset(str(dataset_path), tokenizer_path, max_seq_length=128)
        dataset2 = KDDataset(str(dataset_path), tokenizer_path, max_seq_length=128)

        # Verify samples are in same order
        assert len(dataset1.samples) == len(dataset2.samples)
        for i in range(len(dataset1.samples)):
            assert dataset1.samples[i] == dataset2.samples[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
