"""
Tests for training/make_toy_training.py - Toy training setup creation.

Tests toy model config creation, dataset generation, config file creation,
and model checkpoint initialization.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import torch
import yaml

from models.student.architectures.gqa_transformer import ModelCfg, StudentLM
from training.make_toy_training import (
    create_toy_config,
    create_toy_dataset,
    create_toy_model_config,
    main,
)


class TestCreateToyModelConfig:
    """Test create_toy_model_config function."""

    def test_create_default_config(self):
        """Test creating default toy model config."""
        cfg = create_toy_model_config()

        assert isinstance(cfg, ModelCfg)
        assert cfg.d_model == 64
        assert cfg.n_layers == 2
        assert cfg.vocab_size == 32000
        assert cfg.n_heads == 4
        assert cfg.n_kv_heads == 2
        assert cfg.d_head == 16  # d_model // 4 = 64 // 4 = 16
        assert cfg.rope_theta == 10000.0
        assert cfg.rope_scaling == "dynamic"
        assert cfg.dropout == 0.0

    def test_create_custom_config(self):
        """Test creating custom toy model config."""
        cfg = create_toy_model_config(d_model=128, n_layers=4, vocab_size=1000)

        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.vocab_size == 1000
        assert cfg.d_head == 32  # d_model // 4 = 128 // 4 = 32

    def test_gqa_ratio(self):
        """Test that GQA ratio is correct (n_heads : n_kv_heads = 2:1)."""
        cfg = create_toy_model_config()

        assert cfg.n_heads % cfg.n_kv_heads == 0
        assert cfg.n_heads // cfg.n_kv_heads == 2


class TestCreateToyDataset:
    """Test create_toy_dataset function."""

    def test_create_dataset_default(self, tmp_path):
        """Test creating default toy dataset."""
        output_path = tmp_path / "toy_dataset.jsonl"
        create_toy_dataset(output_path, num_samples=10)

        assert output_path.exists()

        # Read and verify
        samples = []
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        assert len(samples) == 10

        # Check structure
        for sample in samples:
            assert "prompt" in sample
            assert "teacher_text" in sample
            assert "metadata" in sample
            assert "source" in sample["metadata"]
            assert sample["metadata"]["source"] == "toy_test"

    def test_create_dataset_custom_samples(self, tmp_path):
        """Test creating dataset with custom number of samples."""
        output_path = tmp_path / "toy_dataset.jsonl"
        create_toy_dataset(output_path, num_samples=5)

        samples = []
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        assert len(samples) == 5

    def test_create_dataset_content(self, tmp_path):
        """Test dataset content structure."""
        output_path = tmp_path / "toy_dataset.jsonl"
        create_toy_dataset(output_path, num_samples=3)

        samples = []
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Check that prompts and responses are present
        assert all("prompt" in s for s in samples)
        assert all("teacher_text" in s for s in samples)
        assert all(len(s["prompt"]) > 0 for s in samples)
        assert all(len(s["teacher_text"]) > 0 for s in samples)

    def test_create_dataset_metadata(self, tmp_path):
        """Test dataset metadata structure."""
        output_path = tmp_path / "toy_dataset.jsonl"
        create_toy_dataset(output_path, num_samples=2)

        samples = []
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        for sample in samples:
            metadata = sample["metadata"]
            assert "source" in metadata
            assert "tokens" in metadata
            assert "input" in metadata["tokens"]
            assert "output" in metadata["tokens"]
            assert isinstance(metadata["tokens"]["input"], int)
            assert isinstance(metadata["tokens"]["output"], int)

    def test_create_dataset_parent_dir_creation(self, tmp_path):
        """Test that parent directory is created if it doesn't exist."""
        output_path = tmp_path / "nested" / "dir" / "toy_dataset.jsonl"
        create_toy_dataset(output_path, num_samples=1)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestCreateToyConfig:
    """Test create_toy_config function."""

    def test_create_config_default(self, tmp_path):
        """Test creating default toy config."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()  # Create empty file
        config_path = tmp_path / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path")

        assert config_path.exists()

        # Read and verify
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert "arch" in config
        assert "init" in config
        assert "optimizer" in config
        assert "train" in config
        assert "io" in config
        assert "role" in config
        assert "distillation" in config
        assert "kd" in config
        assert "tracing" in config

    def test_create_config_arch_section(self, tmp_path):
        """Test config architecture section."""
        model_cfg = create_toy_model_config(d_model=128, n_layers=4)
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        arch = config["arch"]
        assert arch["d_model"] == 128
        assert arch["n_layers"] == 4
        assert arch["n_heads"] == 4
        assert arch["n_kv_heads"] == 2
        assert arch["vocab_size"] == 32000

    def test_create_config_train_section(self, tmp_path):
        """Test config training section."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path", num_steps=10)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        train = config["train"]
        assert train["steps"] == 10
        assert train["seq_lengths"] == [128]
        assert train["micro_batch_size"] == 1
        assert train["grad_accum"] == 2
        assert train["fp16"] is False
        assert train["grad_checkpointing"] is False

    def test_create_config_io_section(self, tmp_path):
        """Test config IO section."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "config.yaml"
        tokenizer_path = "custom/tokenizer/path"

        create_toy_config(config_path, model_cfg, dataset_path, tokenizer_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        io = config["io"]
        assert io["tokenizer_path"] == tokenizer_path
        assert str(dataset_path) in io["train_shards"]
        assert io["val_shards"] == []

    def test_create_config_distillation_section(self, tmp_path):
        """Test config distillation section."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        distillation = config["distillation"]
        assert distillation["type"] == "standard_kd"
        assert distillation["kl_weight"] == 0.5
        assert distillation["ce_teacher_weight"] == 0.3
        assert distillation["ce_ground_truth_weight"] == 0.2

    def test_create_config_parent_dir_creation(self, tmp_path):
        """Test that parent directory is created if it doesn't exist."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "nested" / "dir" / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path")

        assert config_path.exists()
        assert config_path.parent.exists()


class TestMain:
    """Test main function."""

    @patch("training.make_toy_training.create_toy_model_config")
    @patch("training.make_toy_training.create_toy_dataset")
    @patch("training.make_toy_training.create_toy_config")
    @patch("training.make_toy_training.StudentLM")
    @patch("training.make_toy_training.torch.save")
    def test_main_default_args(
        self,
        mock_torch_save,
        mock_student_lm,
        mock_create_config,
        mock_create_dataset,
        mock_create_model_config,
        tmp_path,
    ):
        """Test main function with default arguments."""
        # Setup mocks
        mock_model_cfg = ModelCfg(d_model=64, n_layers=2, vocab_size=32000)
        mock_create_model_config.return_value = mock_model_cfg

        mock_model = Mock()
        mock_model.state_dict.return_value = {"param": torch.tensor([1.0])}
        mock_student_lm.return_value = mock_model

        # Mock argparse
        with patch("sys.argv", ["make_toy_training", "--out-dir", str(tmp_path)]):
            main()

        # Verify calls
        mock_create_model_config.assert_called_once()
        mock_create_dataset.assert_called_once()
        mock_create_config.assert_called_once()
        mock_student_lm.assert_called_once_with(mock_model_cfg)
        mock_torch_save.assert_called_once()

    @patch("training.make_toy_training.create_toy_model_config")
    @patch("training.make_toy_training.create_toy_dataset")
    @patch("training.make_toy_training.create_toy_config")
    @patch("training.make_toy_training.StudentLM")
    @patch("training.make_toy_training.torch.save")
    def test_main_custom_args(
        self,
        mock_torch_save,
        mock_student_lm,
        mock_create_config,
        mock_create_dataset,
        mock_create_model_config,
        tmp_path,
    ):
        """Test main function with custom arguments."""
        mock_model_cfg = ModelCfg(d_model=128, n_layers=4, vocab_size=1000)
        mock_create_model_config.return_value = mock_model_cfg

        mock_model = Mock()
        mock_model.state_dict.return_value = {"param": torch.tensor([1.0])}
        mock_student_lm.return_value = mock_model

        # Mock argparse with custom args
        with patch(
            "sys.argv",
            [
                "make_toy_training",
                "--out-dir",
                str(tmp_path),
                "--samples",
                "20",
                "--steps",
                "10",
                "--dmodel",
                "128",
                "--nlayers",
                "4",
                "--vocab",
                "1000",
            ],
        ):
            main()

        # Verify custom arguments were used
        mock_create_model_config.assert_called_once_with(
            d_model=128, n_layers=4, vocab_size=1000
        )
        mock_create_dataset.assert_called_once()
        call_kwargs = mock_create_dataset.call_args[1]
        assert call_kwargs["num_samples"] == 20

        mock_create_config.assert_called_once()
        call_kwargs = mock_create_config.call_args[1]
        assert call_kwargs["num_steps"] == 10


class TestToyTrainingIntegration:
    """Test integration of toy training setup creation."""

    def test_full_setup_creation(self, tmp_path):
        """Test creating full toy training setup."""
        out_dir = tmp_path / "toy_test"
        out_dir.mkdir()

        # Create model config
        model_cfg = create_toy_model_config(d_model=64, n_layers=2, vocab_size=1000)

        # Create dataset
        dataset_path = out_dir / "toy_dataset.jsonl"
        create_toy_dataset(dataset_path, num_samples=5, vocab_size=1000)

        # Create config
        config_path = out_dir / "toy_config.yaml"
        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path", num_steps=3)

        # Create model
        model = StudentLM(model_cfg)
        checkpoint_path = out_dir / "toy_model_init.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_cfg.__dict__,
            },
            checkpoint_path,
        )

        # Verify all files exist
        assert dataset_path.exists()
        assert config_path.exists()
        assert checkpoint_path.exists()

        # Verify dataset
        with open(dataset_path, "r") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        assert len(samples) == 5

        # Verify config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert config["train"]["steps"] == 3
        assert config["arch"]["d_model"] == 64

        # Verify checkpoint
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_config_validity(self, tmp_path):
        """Test that generated config is valid YAML."""
        model_cfg = create_toy_model_config()
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.touch()
        config_path = tmp_path / "config.yaml"

        create_toy_config(config_path, model_cfg, dataset_path, "tokenizer_path")

        # Should be valid YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Should have all required sections
        required_sections = [
            "arch",
            "init",
            "optimizer",
            "train",
            "io",
            "role",
            "distillation",
            "kd",
            "tracing",
        ]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

    def test_model_checkpoint_compatibility(self, tmp_path):
        """Test that model checkpoint can be loaded."""
        model_cfg = create_toy_model_config(d_model=64, n_layers=2, vocab_size=1000)
        model = StudentLM(model_cfg)

        checkpoint_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_cfg.__dict__,
            },
            checkpoint_path,
        )

        # Load checkpoint
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

        # Create new model and load state
        new_model = StudentLM(model_cfg)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Models should have same parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


