"""
Tests for distill_kd.py - Knowledge distillation training script.

Tests main training functions, configuration loading, model creation,
optimizer setup, training steps, and checkpoint operations.
"""
# @author: @darianrosebrook

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from training.distill_kd import (
    load_config,
    merge_configs,
    create_model,
    create_optimizer,
    train_step,
    save_checkpoint,
    compute_required_fields_present,
    get_sequence_length,
    sample_enumerated_shape,
    should_enable_qat,
    apply_qat_to_model,
    check_qat_stability,
    truncate_batch_to_shape,
    validate_config,
)


class TestConfigOperations:
    """Test configuration loading and merging."""

    def test_load_config_valid_yaml(self):
        """Test loading a valid YAML config file."""
        config_content = """
        arch:
          vocab_size: 1000
          d_model: 128
        train:
          steps: 100
        distillation:
          use_intermediate_layers: false
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["arch"]["vocab_size"] == 1000
            assert config["train"]["steps"] == 100
            assert not config["distillation"]["use_intermediate_layers"]
        finally:
            Path(config_path).unlink()

    def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_merge_configs_basic(self):
        """Test basic config merging."""
        # Create temporary config files
        config1_content = """
        arch:
          vocab_size: 1000
          d_model: 128
        """
        config2_content = """
        train:
          steps: 100
          lr: 0.0001
        """
        config3_content = """
        distillation:
          kl_weight: 0.5
        """

        files = []
        try:
            for content in [config1_content, config2_content, config3_content]:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    f.write(content)
                    files.append(f.name)

            merged = merge_configs(files)

            assert merged["arch"]["vocab_size"] == 1000
            assert merged["train"]["steps"] == 100
            assert merged["distillation"]["kl_weight"] == 0.5
        finally:
            for f in files:
                Path(f).unlink()

    def test_merge_configs_overrides(self):
        """Test that later configs override earlier ones."""
        config1_content = """
        arch:
          vocab_size: 1000
        """
        config2_content = """
        arch:
          vocab_size: 2000
          d_model: 256
        """

        files = []
        try:
            for content in [config1_content, config2_content]:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                    f.write(content)
                    files.append(f.name)

            merged = merge_configs(files)

            assert merged["arch"]["vocab_size"] == 2000
            assert merged["arch"]["d_model"] == 256
        finally:
            for f in files:
                Path(f).unlink()

    def test_merge_configs_env_overrides(self):
        """Test environment variable overrides."""
        config_content = """
        arch:
          vocab_size: 1000
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            merged = merge_configs([config_path])

            # Should have default values, env overrides are handled separately
            assert merged["arch"]["vocab_size"] == 1000
        finally:
            Path(config_path).unlink()


class TestModelCreation:
    """Test model creation functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic model configuration for testing."""
        return {
            "arch": {
                "vocab_size": 1000,
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_head": 32,
                "rope_theta": 10000.0,
                "rope_scaling": "none",
                "dropout": 0.0,
            }
        }

    def test_create_model_basic(self, basic_config, device):
        """Test basic model creation."""
        model = create_model(basic_config, device)

        assert isinstance(model, nn.Module)
        assert model.training  # Model should be in training mode by default

        # Check vocab size through config
        assert hasattr(model, "cfg")
        assert model.cfg.vocab_size == 1000

    def test_create_model_with_quantization(self, basic_config, device):
        """Test model creation with quantization config."""
        config_with_quant = basic_config.copy()
        config_with_quant["quant"] = {"enabled": True, "qat": {"enabled": True}}

        with patch("training.distill_kd.quantize_model") as mock_quantize:
            mock_quantize.return_value = Mock(spec=nn.Module)
            model = create_model(config_with_quant, device)

            # Should call quantization if available
            mock_quantize.assert_called_once()

    def test_create_model_invalid_config(self, device):
        """Test model creation with invalid config raises error."""
        invalid_config = {"arch": {}}  # Missing required fields

        with pytest.raises(KeyError):
            create_model(invalid_config, device)


class TestOptimizerCreation:
    """Test optimizer creation functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with real parameters."""
        model = nn.Linear(10, 5)
        return model

    def test_create_optimizer_adamw(self, simple_model):
        """Test AdamW optimizer creation."""
        config = {
            "train": {
                "optimizer": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            }
        }

        optimizer = create_optimizer(simple_model, config)

        assert isinstance(optimizer, AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["betas"] == (0.9, 0.999)

    def test_create_optimizer_default_config(self, simple_model):
        """Test optimizer creation with minimal config."""
        config = {"train": {}}

        optimizer = create_optimizer(simple_model, config)

        assert isinstance(optimizer, AdamW)
        # Should use default values
        assert optimizer.defaults["lr"] == 1e-3  # Default LR

    def test_create_optimizer_invalid_type(self, simple_model):
        """Test invalid optimizer type raises error."""
        config = {"train": {"optimizer": "invalid_optimizer"}}

        with pytest.raises(ValueError):
            create_optimizer(simple_model, config)


class TestSequenceLength:
    """Test sequence length calculation."""

    def test_get_sequence_length_basic(self):
        """Test basic sequence length calculation."""
        step = 100
        seq_lengths = [512, 1024, 2048]

        seq_len = get_sequence_length(step, seq_lengths)
        assert seq_len == 2048  # Should return max by default

    def test_get_sequence_length_with_curriculum(self):
        """Test sequence length with curriculum."""
        step = 100
        seq_lengths = [512, 1024, 2048]
        curriculum_schedule = [0, 50, 100]  # Switch lengths at these steps

        seq_len = get_sequence_length(step, seq_lengths, curriculum_schedule)
        # At step 100, should be using the last length (2048)
        assert seq_len == 2048

    def test_sample_enumerated_shape(self):
        """Test enumerated shape sampling."""
        shapes = [
            {"seq_len": 512, "d_model": 768},
            {"seq_len": 1024, "d_model": 1536},
        ]

        config = {"arch": {"enumerated_shapes": shapes}}

        # Mock random to return first shape
        with patch("training.distill_kd.random.choices", return_value=[shapes[0]]):
            shape = sample_enumerated_shape(config)

            assert shape == shapes[0]
            assert "seq_len" in shape
            assert "d_model" in shape


class TestQATOperations:
    """Test Quantization-Aware Training operations."""

    def test_should_enable_qat_false(self):
        """Test QAT enablement when disabled."""
        qat_cfg = {"enabled": False}
        assert not should_enable_qat(100, 1000, qat_cfg)

    def test_should_enable_qat_true(self):
        """Test QAT enablement when enabled."""
        qat_cfg = {
            "enabled": True,
            "start_fraction": 0.5,  # Start at 50% of training
        }
        assert should_enable_qat(600, 1000, qat_cfg)  # At 60% -> should enable
        assert not should_enable_qat(300, 1000, qat_cfg)  # At 30% -> should not enable

    def test_should_enable_qat_no_config(self):
        """Test QAT enablement with no config."""
        assert not should_enable_qat(100, 1000, {})

    def test_apply_qat_to_model(self, device):
        """Test QAT application to model."""
        simple_model = nn.Linear(10, 5)
        qat_cfg = {"qat": {"enabled": True}}

        # This will fail if QAT_AVAILABLE is False, which is expected in test environment
        with pytest.raises(RuntimeError, match="QAT not available"):
            apply_qat_to_model(simple_model, qat_cfg, device)

    def test_check_qat_stability_valid(self, sample_batch, device):
        """Test QAT stability check with valid model."""
        simple_model = nn.Linear(10, 5)
        simple_model.to(device)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = check_qat_stability(simple_model, sample_batch, device)

        # Should return a dictionary with stability metrics
        assert isinstance(result, dict)
        assert "has_nan" in result
        assert not result["has_nan"]  # Should not have NaN

    def test_check_qat_stability_nan_weights(self, sample_batch, device):
        """Test QAT stability check with NaN weights."""
        simple_model = nn.Linear(10, 5)
        # Set weights to NaN
        with torch.no_grad():
            simple_model.weight.fill_(float("nan"))
        simple_model.to(device)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = check_qat_stability(simple_model, sample_batch, device)

        # Should detect NaN weights
        assert isinstance(result, dict)
        assert result["has_nan"]  # Should detect NaN


class TestTrainingStep:
    """Test training step functionality."""

    @pytest.fixture
    def training_config(self):
        """Config for training step tests."""
        return {
            "arch": {"vocab_size": 1000},
            "distillation": {
                "use_intermediate_layers": False,
                "use_self_evaluation": False,
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.7,
            },
            "latent": {
                "halt_head_enabled": False,
            },
        }

    @pytest.fixture
    def sample_batch(self):
        """Create sample training batch."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "teacher_logits": torch.randn(batch_size, seq_len, vocab_size),
        }

    @pytest.fixture
    def mock_model(self, small_model):
        """Use small model from conftest."""
        return small_model

    @pytest.fixture
    def simple_optimizer(self, small_model):
        """Create simple optimizer for real model."""
        return AdamW(small_model.parameters(), lr=1e-3)

    def test_train_step_basic(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test basic training step."""
        scaler = None  # No FP16 for simplicity

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=scaler,
            cfg=training_config,
            device=device,
        )

        # Should return loss dictionary
        assert isinstance(result, dict)
        assert "total" in result
        assert isinstance(result["total"], float)

    def test_train_step_vocab_clamping(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test vocabulary clamping during training step."""
        # Create batch with out-of-vocab tokens
        vocab_size = 1000
        sample_batch["input_ids"] = torch.tensor(
            [[vocab_size + 10, vocab_size + 20]], dtype=torch.long
        ).to(device)
        sample_batch["labels"] = torch.tensor(
            [[vocab_size + 10, vocab_size + 20]], dtype=torch.long
        ).to(device)

        training_config["arch"]["vocab_size"] = vocab_size

        with patch("builtins.print") as mock_print:
            result = train_step(
                model=small_model,
                batch=sample_batch,
                optimizer=simple_optimizer,
                scaler=None,
                cfg=training_config,
                device=device,
                current_step=50,  # Multiple of 50 to trigger warning
            )

            # Should clamp values
            assert torch.all(sample_batch["input_ids"] < vocab_size)
            assert torch.all(sample_batch["labels"] < vocab_size)

            # Should print warning
            mock_print.assert_called()

    def test_train_step_cot_free_validation(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test CoT-free validation during training step."""
        # Add reasoning content to batch
        sample_batch["teacher_reasoning_content"] = ["Some reasoning content"]

        with pytest.raises(ValueError, match="CoT-free training"):
            train_step(
                model=small_model,
                batch=sample_batch,
                optimizer=simple_optimizer,
                scaler=None,
                cfg=training_config,
                device=device,
            )

    def test_train_step_with_intermediate_layers(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with intermediate layer matching."""
        training_config["distillation"]["use_intermediate_layers"] = True

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result

    def test_train_step_with_self_evaluation(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with self-evaluation head."""
        training_config["distillation"]["use_self_evaluation"] = True

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result

    def test_train_step_with_halt_head(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with halt head."""
        training_config["latent"]["halt_head_enabled"] = True

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result


class TestBatchOperations:
    """Test batch processing operations."""

    def test_compute_required_fields_present_basic(self, device):
        """Test basic required fields computation."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
        }

        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="test response")

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bool
        assert result.shape == (2,)  # Batch size dimension

    def test_truncate_batch_to_shape(self):
        """Test batch truncation to fit sequence length."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 20)),  # seq_len = 20
            "attention_mask": torch.ones(2, 20),
            "labels": torch.randint(0, 100, (2, 20)),
        }

        max_seq_len = 10

        result = truncate_batch_to_shape(batch, max_seq_len)

        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"].shape == (2, 10)
        assert result["labels"].shape == (2, 10)


class TestCheckpointOperations:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint_basic(self, tmp_path, small_model):
        """Test basic checkpoint saving."""
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        # Create optimizer for the model
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config=config,
        )

        # Should create checkpoint file
        checkpoint_files = list(output_dir.glob("checkpoint_step_100_*.pt"))
        assert len(checkpoint_files) == 1

        checkpoint_path = checkpoint_files[0]
        assert checkpoint_path.exists()

        # Load and verify
        loaded = torch.load(checkpoint_path)
        assert loaded["step"] == 100
        assert loaded["loss"] == 0.5
        assert loaded["config"] == config

    def test_save_checkpoint_with_metadata(self, tmp_path, small_model):
        """Test checkpoint saving with additional metadata."""
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(exist_ok=True)

        # Create optimizer for the model
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=200,
            loss=0.3,
            output_dir=output_dir,
            config=config,
            loss_dict={"ce": 0.2, "kl": 0.1},
        )

        checkpoint_files = list(output_dir.glob("checkpoint_step_200_*.pt"))
        assert len(checkpoint_files) == 1

        loaded = torch.load(checkpoint_files[0])
        assert loaded["step"] == 200
        assert loaded["loss"] == 0.3
        assert loaded["loss_dict"]["ce"] == 0.2


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = {
            "model": {
                "vocab_size": 1000,
                "d_model": 128,
            },
            "training": {
                "steps": 100,
            },
            "optimizer": {
                "lr": 1e-4,
            },
            "distillation": {
                "kl_weight": 0.5,
            },
        }

        # Should not raise
        validate_config(config)

    def test_validate_config_missing_model(self):
        """Test validation fails with missing model section."""
        config = {"training": {"steps": 100}}

        with pytest.raises(ValueError, match="Missing required 'model' configuration section"):
            validate_config(config)

    def test_validate_config_missing_training(self):
        """Test validation fails with missing training section."""
        config = {"model": {"vocab_size": 1000}}

        with pytest.raises(ValueError, match="Missing required 'training' configuration section"):
            validate_config(config)

    def test_validate_config_invalid_lr(self):
        """Test validation fails with invalid learning rate."""
        config = {
            "model": {"vocab_size": 1000},
            "training": {"steps": 100},
            "optimizer": {"lr": -1.0},  # Invalid negative LR
        }

        with pytest.raises(ValueError, match="optimizer.lr must be positive"):
            validate_config(config)

    def test_validate_config_invalid_vocab_size(self):
        """Test validation fails with invalid vocab size."""
        config = {
            "model": {"vocab_size": -100},  # Invalid negative vocab size
            "training": {"steps": 100},
        }

        with pytest.raises(ValueError, match="model.vocab_size must be positive"):
            validate_config(config)

    def test_validate_config_invalid_d_model(self):
        """Test validation fails with invalid d_model."""
        config = {
            "model": {"vocab_size": 1000, "d_model": -128},
            "training": {"steps": 100},
        }

        with pytest.raises(ValueError, match="model.d_model must be positive"):
            validate_config(config)

    def test_validate_config_invalid_n_layers(self):
        """Test validation fails with invalid n_layers."""
        config = {
            "model": {"vocab_size": 1000, "n_layers": 0},
            "training": {"steps": 100},
        }

        with pytest.raises(ValueError, match="model.n_layers must be positive"):
            validate_config(config)

    def test_validate_config_invalid_steps(self):
        """Test validation fails with invalid training steps."""
        config = {
            "model": {"vocab_size": 1000},
            "training": {"steps": -100},
        }

        with pytest.raises(ValueError, match="training.steps must be positive"):
            validate_config(config)


class TestComputeRequiredFieldsPresent:
    """Test compute_required_fields_present function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value='{"name": "test_tool", "arguments": {"key": "value"}}')
        tokenizer.pad_token_id = 0
        return tokenizer

    def test_compute_required_fields_present_empty_batch(self, device):
        """Test with empty batch."""
        batch = {"input_ids": torch.empty(0, 10, dtype=torch.long)}
        mock_tokenizer = Mock()

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (0,)

    def test_compute_required_fields_present_no_validated_args(self, device, mock_tokenizer):
        """Test when no validated arguments are available."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
        }

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        # Should return all True (no penalty) when validation data unavailable
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bool
        assert torch.all(result)  # All should be True

    def test_compute_required_fields_present_with_student_logits(self, device, mock_tokenizer):
        """Test with student logits in batch."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "student_logits": torch.randn(batch_size, seq_len, vocab_size),
            "validated_arguments": [{"arguments": {"key": "value"}}, {"arguments": {"key": "value"}}],
            "tool_names": ["test_tool"],
        }

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bool
        assert result.shape == (batch_size,)

    def test_compute_required_fields_present_no_tool_call(self, device, mock_tokenizer):
        """Test when no tool call is extracted from generated text."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Mock tokenizer to return text without tool call
        mock_tokenizer.decode = Mock(return_value="Just plain text without tool call")

        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "student_logits": torch.randn(batch_size, seq_len, vocab_size),
            "validated_arguments": [{"arguments": {"key": "value"}}],
            "tool_names": ["test_tool"],
        }

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        # Should mark as incomplete (False) when no tool call found
        assert isinstance(result, torch.Tensor)
        assert torch.any(~result)  # At least some should be False

    def test_compute_required_fields_present_with_schema_registry(self, device, mock_tokenizer):
        """Test with schema registry validation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "student_logits": torch.randn(batch_size, seq_len, vocab_size),
            "validated_arguments": [{"arguments": {"key": "value"}}],
            "tool_names": ["test_tool"],
        }

        # Mock schema registry
        with patch("training.distill_kd.ToolSchemaRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_schema = Mock(
                return_value={
                    "properties": {
                        "arguments": {
                            "required": ["key"],
                        }
                    }
                }
            )
            mock_registry.validate_tool_call = Mock(return_value=(True, None))
            mock_registry_class.return_value = mock_registry

            result = compute_required_fields_present(batch, mock_tokenizer, device)

            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.bool

    def test_compute_required_fields_present_fallback_heuristic(self, device):
        """Test fallback heuristic when logits not available."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "gold_json_text_ids": torch.randint(0, 100, (2, 8)),
            "mask_valid_json_tokens": torch.ones(2, 8, dtype=torch.bool),
            "attention_mask": torch.ones(2, 10),
        }

        mock_tokenizer = Mock()

        result = compute_required_fields_present(batch, mock_tokenizer, device)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bool


class TestMergeConfigsEnvOverrides:
    """Test merge_configs with environment variable overrides."""

    @patch.dict("os.environ", {"TRAIN_LATENT": "1"})
    def test_merge_configs_env_train_latent(self, tmp_path):
        """Test environment variable override for TRAIN_LATENT."""
        config_content = """
        arch:
          vocab_size: 1000
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            merged = merge_configs([config_path])

            assert merged["latent"]["enabled"] is True
            assert "_provenance" in merged
            assert "TRAIN_LATENT" in merged["_provenance"]["env_overrides"]
        finally:
            Path(config_path).unlink()

    @patch.dict("os.environ", {"LATENT_MODE": "1"})
    def test_merge_configs_env_latent_mode(self, tmp_path):
        """Test environment variable override for LATENT_MODE."""
        config_content = """
        arch:
          vocab_size: 1000
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            merged = merge_configs([config_path])

            assert merged["latent"]["mode_enabled"] is True
            assert "LATENT_MODE" in merged["_provenance"]["env_overrides"]
        finally:
            Path(config_path).unlink()

    @patch.dict("os.environ", {"HALT_HEAD": "1"})
    def test_merge_configs_env_halt_head(self, tmp_path):
        """Test environment variable override for HALT_HEAD."""
        config_content = """
        arch:
          vocab_size: 1000
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            merged = merge_configs([config_path])

            assert merged["latent"]["halt_head_enabled"] is True
            assert "HALT_HEAD" in merged["_provenance"]["env_overrides"]
        finally:
            Path(config_path).unlink()

    def test_merge_configs_provenance_metadata(self, tmp_path):
        """Test that provenance metadata is stored."""
        config_content = """
        arch:
          vocab_size: 1000
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            merged = merge_configs([config_path], env_overrides={"custom": "value"})

            assert "_provenance" in merged
            assert merged["_provenance"]["config_files"] == [config_path]
            assert "custom" in merged["_provenance"]["env_overrides"]
        finally:
            Path(config_path).unlink()


class TestModelCreationExpanded:
    """Test expanded model creation scenarios."""

    @pytest.fixture
    def basic_config(self):
        """Basic model configuration for testing."""
        return {
            "arch": {
                "vocab_size": 1000,
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_head": 32,
                "rope_theta": 10000.0,
                "rope_scaling": "none",
                "dropout": 0.0,
            }
        }

    def test_create_model_with_checkpoint(self, basic_config, device, tmp_path):
        """Test model creation with checkpoint loading."""
        # Create a dummy checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        dummy_model = nn.Linear(10, 5)
        torch.save({"model_state_dict": dummy_model.state_dict()}, checkpoint_path)

        config_with_checkpoint = basic_config.copy()
        config_with_checkpoint["init"] = {"base_checkpoint": str(checkpoint_path)}

        # This will fail because model architectures don't match, but should attempt loading
        model = create_model(config_with_checkpoint, device)

        assert isinstance(model, nn.Module)

    def test_create_model_with_self_evaluation(self, basic_config, device):
        """Test model creation with self-evaluation head."""
        config_with_eval = basic_config.copy()
        config_with_eval["distillation"] = {"use_self_evaluation": True}

        model = create_model(config_with_eval, device)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "use_self_evaluation")

    def test_create_model_with_intermediate_layers(self, basic_config, device):
        """Test model creation with intermediate layer matching."""
        config_with_intermediate = basic_config.copy()
        config_with_intermediate["distillation"] = {
            "use_intermediate_layers": True,
            "layer_mapping": {0: 0, 1: 2},
        }
        config_with_intermediate["teacher"] = {"d_model": 256}

        model = create_model(config_with_intermediate, device)

        assert isinstance(model, nn.Module)
        # Should have projection layers if dimensions differ
        if hasattr(model, "projection_layers"):
            assert model.projection_layers is not None

    def test_create_model_checkpoint_not_found(self, basic_config, device):
        """Test model creation when checkpoint path doesn't exist."""
        config_with_missing_checkpoint = basic_config.copy()
        config_with_missing_checkpoint["init"] = {"base_checkpoint": "nonexistent.pt"}

        # Should continue with randomly initialized model
        model = create_model(config_with_missing_checkpoint, device)

        assert isinstance(model, nn.Module)


class TestOptimizerCreationExpanded:
    """Test expanded optimizer creation scenarios."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with real parameters."""
        return nn.Linear(10, 5)

    def test_create_optimizer_adam(self, simple_model):
        """Test Adam optimizer creation."""
        config = {
            "optimizer": {
                "name": "adam",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            }
        }

        optimizer = create_optimizer(simple_model, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 1e-4

    def test_create_optimizer_with_projection_layers(self, simple_model, device):
        """Test optimizer creation with projection layers."""
        # Create a model with projection layers
        model = simple_model
        model.projection_layers = {
            0: nn.Linear(10, 20).to(device),
            1: nn.Linear(10, 20).to(device),
        }

        config = {"optimizer": {"name": "adamw", "lr": 1e-4}}

        optimizer = create_optimizer(model, config)

        # Should include projection layer parameters
        param_count = sum(p.numel() for p in optimizer.param_groups[0]["params"])
        assert param_count > 0


class TestSequenceLengthExpanded:
    """Test expanded sequence length scenarios."""

    def test_get_sequence_length_curriculum_early_step(self):
        """Test sequence length with curriculum at early step."""
        step = 25
        seq_lengths = [512, 1024, 2048]
        curriculum_schedule = [0, 50, 100]

        seq_len = get_sequence_length(step, seq_lengths, curriculum_schedule)

        # At step 25, should be using second length (1024)
        assert seq_len == 1024

    def test_get_sequence_length_curriculum_boundary(self):
        """Test sequence length at curriculum boundary."""
        step = 50
        seq_lengths = [512, 1024, 2048]
        curriculum_schedule = [0, 50, 100]

        seq_len = get_sequence_length(step, seq_lengths, curriculum_schedule)

        # At boundary, should use next length
        assert seq_len in seq_lengths

    def test_get_sequence_length_curriculum_beyond_schedule(self):
        """Test sequence length beyond curriculum schedule."""
        step = 200
        seq_lengths = [512, 1024, 2048]
        curriculum_schedule = [0, 50, 100]

        seq_len = get_sequence_length(step, seq_lengths, curriculum_schedule)

        # Should use last length
        assert seq_len == seq_lengths[-1]

    def test_sample_enumerated_shape_with_probs(self):
        """Test enumerated shape sampling with custom probabilities."""
        seq_lengths = [512, 1024, 2048, 4096]
        shape_probs = [0.5, 0.3, 0.15, 0.05]

        # Mock random.choices to return specific value
        with patch("training.distill_kd.random.choices", return_value=[2048]):
            result = sample_enumerated_shape(seq_lengths, shape_probs=shape_probs)

            assert result == 2048

    def test_sample_enumerated_shape_periodic_upweight(self):
        """Test enumerated shape sampling with periodic upweighting."""
        seq_lengths = [512, 1024, 2048, 4096]

        # At step 100 (multiple of 100), should upweight rare shapes
        with patch("training.distill_kd.random.choices") as mock_choices:
            mock_choices.return_value = [512]  # Smallest shape
            result = sample_enumerated_shape(seq_lengths, step=100, periodic_upweight_rare=True)

            assert result == 512
            # Should have been called with adjusted probabilities
            mock_choices.assert_called_once()

    def test_sample_enumerated_shape_default_probs(self):
        """Test enumerated shape sampling with default probabilities."""
        seq_lengths = [512, 1024]

        with patch("training.distill_kd.random.choices", return_value=[512]):
            result = sample_enumerated_shape(seq_lengths)

            assert result == 512


class TestQATOperationsExpanded:
    """Test expanded QAT operations."""

    def test_should_enable_qat_default_start_fraction(self):
        """Test QAT enablement with default start_fraction."""
        qat_cfg = {"enabled": True}  # Default start_fraction is 0.8

        # At 80% of training (step 800 out of 1000)
        assert should_enable_qat(800, 1000, qat_cfg)
        # At 79% of training (step 790 out of 1000)
        assert not should_enable_qat(790, 1000, qat_cfg)

    def test_should_enable_qat_custom_start_fraction(self):
        """Test QAT enablement with custom start_fraction."""
        qat_cfg = {"enabled": True, "start_fraction": 0.9}  # Start at 90%

        assert should_enable_qat(950, 1000, qat_cfg)  # At 95% -> should enable
        assert not should_enable_qat(850, 1000, qat_cfg)  # At 85% -> should not enable

    def test_check_qat_stability_with_baseline(self, sample_batch, device):
        """Test QAT stability check with baseline model."""
        simple_model = nn.Linear(10, 5)
        baseline_model = nn.Linear(10, 5)
        simple_model.to(device)
        baseline_model.to(device)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = check_qat_stability(
            simple_model, sample_batch, device, baseline_model=baseline_model
        )

        assert isinstance(result, dict)
        assert "qat_stability.has_nan" in result or "has_nan" in result
        assert "qat_stability.cosine_sim" in result or "cosine_sim" in result

    def test_check_qat_stability_with_baseline_hidden_states(self, sample_batch, device):
        """Test QAT stability check with pre-computed baseline hidden states."""
        simple_model = nn.Linear(10, 5)
        simple_model.to(device)

        baseline_hidden_states = [
            torch.randn(2, 10, 5, device=device),
            torch.randn(2, 10, 5, device=device),
        ]

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = check_qat_stability(
            simple_model,
            sample_batch,
            device,
            baseline_hidden_states=baseline_hidden_states,
        )

        assert isinstance(result, dict)


class TestTruncateBatchToShapeExpanded:
    """Test expanded batch truncation scenarios."""

    def test_truncate_batch_to_shape_all_keys(self):
        """Test truncation of all batch key types."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 20)),
            "attention_mask": torch.ones(2, 20),
            "labels": torch.randint(0, 100, (2, 20)),
            "teacher_target_ids": torch.randint(0, 100, (2, 20)),
            "tool_name_ids": torch.randint(0, 100, (2, 20)),
            "tool_name_mask": torch.ones(2, 20, dtype=torch.bool),
            "gold_json_text_ids": torch.randint(0, 100, (2, 20)),
            "mask_valid_json_tokens": torch.ones(2, 20, dtype=torch.bool),
            "tool_result_fields": torch.randint(0, 100, (2, 20)),
            "integration_mask": torch.ones(2, 20, dtype=torch.bool),
            "teacher_attention_mask": torch.ones(2, 20),
            "teacher_logits": torch.randn(2, 20, 1000),
        }

        target_length = 10

        result = truncate_batch_to_shape(batch, target_length)

        # All sequence keys should be truncated
        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"].shape == (2, 10)
        assert result["labels"].shape == (2, 10)
        assert result["teacher_target_ids"].shape == (2, 10)
        assert result["tool_name_ids"].shape == (2, 10)
        assert result["gold_json_text_ids"].shape == (2, 10)
        assert result["teacher_logits"].shape == (2, 10, 1000)  # Keep vocab dimension

    def test_truncate_batch_to_shape_no_truncation_needed(self):
        """Test truncation when sequence is already shorter."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 5)),
            "attention_mask": torch.ones(2, 5),
            "labels": torch.randint(0, 100, (2, 5)),
        }

        target_length = 10

        result = truncate_batch_to_shape(batch, target_length)

        # Should keep original length
        assert result["input_ids"].shape == (2, 5)

    def test_truncate_batch_to_shape_metadata_preserved(self):
        """Test that metadata keys are preserved during truncation."""
        batch = {
            "input_ids": torch.randint(0, 100, (2, 20)),
            "attention_mask": torch.ones(2, 20),
            "metadata": {"key": "value"},
            "meta": [{"item": 1}, {"item": 2}],
        }

        target_length = 10

        result = truncate_batch_to_shape(batch, target_length)

        # Metadata should be preserved
        assert "metadata" in result
        assert result["metadata"] == {"key": "value"}
        assert "meta" in result
        assert result["meta"] == [{"item": 1}, {"item": 2}]


class TestSaveCheckpointExpanded:
    """Test expanded checkpoint saving scenarios."""

    def test_save_checkpoint_atomic_write(self, tmp_path, small_model):
        """Test atomic checkpoint writing."""
        output_dir = tmp_path / "checkpoints"
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config=config,
        )

        # Should create both latest.pt and numbered checkpoint
        latest_path = output_dir / "latest.pt"
        numbered_path = output_dir / "checkpoint_step_100.pt"

        assert latest_path.exists()
        assert numbered_path.exists()

    def test_save_checkpoint_with_rng_states(self, tmp_path, small_model):
        """Test checkpoint saving with RNG states."""
        output_dir = tmp_path / "checkpoints"
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        import random
        import numpy as np

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": None,
        }

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config=config,
            rng_states=rng_states,
        )

        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        loaded = torch.load(checkpoint_path)

        assert "meta" in loaded
        assert "rng_states" in loaded["meta"]
        assert loaded["meta"]["rng_states"]["python"] is not None

    def test_save_checkpoint_with_dataset_fingerprint(self, tmp_path, small_model):
        """Test checkpoint saving with dataset fingerprint."""
        output_dir = tmp_path / "checkpoints"
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config=config,
            dataset_fingerprint="abc123",
        )

        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        loaded = torch.load(checkpoint_path)

        assert "meta" in loaded
        assert loaded["meta"]["dataset_fingerprint"] == "abc123"

    def test_save_checkpoint_with_code_mode_metadata(self, tmp_path, small_model):
        """Test checkpoint saving with code-mode metadata."""
        output_dir = tmp_path / "checkpoints"
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {
            "distill": {
                "code_mode": {
                    "enabled": True,
                    "weight": 0.3,
                    "weight_schedule": {"warmup_steps": 5000},
                    "code_mode_pref": {"eligibility_rules": {"min_tools": 2}},
                }
            }
        }

        save_checkpoint(
            model=small_model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config=config,
        )

        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        loaded = torch.load(checkpoint_path)

        assert "meta" in loaded
        assert "code_mode" in loaded["meta"]
        assert loaded["meta"]["code_mode"]["enabled"] is True

    @patch("training.distill_kd.DDP")
    def test_save_checkpoint_with_ddp_model(self, mock_ddp, tmp_path, small_model):
        """Test checkpoint saving with DDP-wrapped model."""
        output_dir = tmp_path / "checkpoints"

        # Create a mock DDP model
        mock_ddp_model = Mock()
        mock_ddp_model.module = small_model
        mock_ddp_model.state_dict = Mock(return_value={})

        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        config = {"test": "config"}

        # Use isinstance check to detect DDP
        with patch("training.distill_kd.DDP", return_value=mock_ddp_model):
            # Temporarily make model look like DDP
            import training.distill_kd as dkd_module

            # Save checkpoint - should unwrap DDP
            save_checkpoint(
                model=mock_ddp_model,
                optimizer=optimizer,
                step=100,
                loss=0.5,
                output_dir=output_dir,
                config=config,
            )

            checkpoint_path = output_dir / "checkpoint_step_100.pt"
            assert checkpoint_path.exists()


class TestTrainingStepExpanded:
    """Test expanded training step scenarios."""

    @pytest.fixture
    def training_config(self):
        """Config for training step tests."""
        return {
            "arch": {"vocab_size": 1000},
            "distillation": {
                "use_intermediate_layers": False,
                "use_self_evaluation": False,
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.7,
            },
            "latent": {
                "halt_head_enabled": False,
            },
        }

    @pytest.fixture
    def sample_batch(self):
        """Create sample training batch."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "teacher_logits": torch.randn(batch_size, seq_len, vocab_size),
        }

    @pytest.fixture
    def simple_optimizer(self, small_model):
        """Create simple optimizer for real model."""
        return AdamW(small_model.parameters(), lr=1e-3)

    def test_train_step_with_process_supervision(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with process-step supervision targets."""
        training_config["distillation"]["w_tool"] = 0.15
        training_config["distillation"]["w_args"] = 0.15
        training_config["distillation"]["w_integr"] = 0.10

        sample_batch["tool_name_ids"] = torch.randint(0, 100, (2, 5))
        sample_batch["tool_name_mask"] = torch.ones(2, 5, dtype=torch.bool)
        sample_batch["gold_json_text_ids"] = torch.randint(0, 100, (2, 8))
        sample_batch["mask_valid_json_tokens"] = torch.ones(2, 8, dtype=torch.bool)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result

    def test_train_step_with_entropy_scheduling(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with entropy-based scheduling."""
        training_config["distillation"]["use_entropy_scheduling"] = True
        training_config["distillation"]["min_entropy"] = 2.0
        training_config["distillation"]["max_entropy"] = 8.0

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            current_step=100,
        )

        assert "total" in result

    def test_train_step_with_temperature_schedule(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with temperature scheduling."""
        training_config["distillation"]["use_temperature_schedule"] = True
        training_config["distillation"]["base_temperature"] = 2.0
        training_config["distillation"]["min_temperature"] = 1.5
        training_config["distillation"]["max_temperature"] = 3.0
        training_config["train"] = {"total_steps": 1000}

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            current_step=500,
        )

        assert "total" in result

    def test_train_step_with_weight_schedule(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with loss weight scheduling."""
        training_config["distillation"]["use_weight_schedule"] = True
        training_config["distillation"]["early_teacher_weight"] = 0.7
        training_config["distillation"]["late_teacher_weight"] = 0.3
        training_config["train"] = {"total_steps": 1000}

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            current_step=500,
        )

        assert "total" in result

    def test_train_step_with_length_aware_kd(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with length-aware KD loss."""
        training_config["distillation"]["use_length_aware_kd"] = True
        training_config["distillation"]["length_kd_weight"] = 0.05
        training_config["distillation"]["length_kd_hinge"] = 0.15

        sample_batch["teacher_attention_mask"] = torch.ones(2, 12)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        with patch("training.distill_kd.compute_required_fields_present") as mock_compute:
            mock_compute.return_value = torch.ones(2, dtype=torch.bool, device=device)

            result = train_step(
                model=small_model,
                batch=sample_batch,
                optimizer=simple_optimizer,
                scaler=None,
                cfg=training_config,
                device=device,
            )

            assert "total" in result
            assert "length_kd" in result

    def test_train_step_with_early_tool_call(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with early tool call loss."""
        training_config["distillation"]["use_early_tool_call_loss"] = True
        training_config["distillation"]["early_tool_weight"] = 0.05
        training_config["distillation"]["early_tool_N"] = 25
        training_config["io"] = {"tokenizer_path": "models/student/tokenizer"}

        sample_batch["tool_should_be_used"] = torch.ones(2, dtype=torch.bool)
        sample_batch["teacher_prefix_ids"] = torch.randint(0, 100, (2, 10))

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        with patch("training.distill_kd.load_tokenizer") as mock_load:
            mock_tokenizer = Mock()
            mock_tokenizer.decode = Mock(return_value='{"name": "tool"}')
            mock_load.return_value = mock_tokenizer

            result = train_step(
                model=small_model,
                batch=sample_batch,
                optimizer=simple_optimizer,
                scaler=None,
                cfg=training_config,
                device=device,
            )

            assert "total" in result

    def test_train_step_with_gradient_accumulation(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with gradient accumulation."""
        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        # First accumulation step (should not step optimizer)
        result1 = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            grad_accum_steps=2,
            grad_accum_counter=0,
        )

        # Second accumulation step (should step optimizer)
        result2 = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
            grad_accum_steps=2,
            grad_accum_counter=1,
        )

        assert "total" in result1
        assert "total" in result2

    def test_train_step_with_fp16_scaler(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with FP16 scaler."""
        scaler = None
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=scaler,
            cfg=training_config,
            device=device,
        )

        assert "total" in result

    def test_train_step_with_teacher_targets(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with teacher target IDs."""
        sample_batch["teacher_target_ids"] = torch.randint(0, 1000, (2, 10))

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result

    def test_train_step_with_loss_mask(
        self, small_model, sample_batch, simple_optimizer, training_config, device
    ):
        """Test training step with loss mask from latent curriculum."""
        sample_batch["loss_mask"] = torch.ones(2, 10, dtype=torch.bool)

        # Move batch to device
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                sample_batch[k] = v.to(device)

        result = train_step(
            model=small_model,
            batch=sample_batch,
            optimizer=simple_optimizer,
            scaler=None,
            cfg=training_config,
            device=device,
        )

        assert "total" in result
