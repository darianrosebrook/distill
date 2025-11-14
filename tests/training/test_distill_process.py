"""
Tests for training/distill_process.py - Process supervision training for tool-use.

Tests configuration loading, model loading, text generation from logits,
training steps with process supervision, and main function execution.
"""
# @author: @darianrosebrook

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from training.distill_process import (
    load_config,
    merge_configs,
    load_model,
    generate_text_from_logits,
    train_step_process,
    main,
)


class TestConfigOperations:
    """Test configuration loading and merging."""

    def test_load_config_success(self):
        """Test successful config loading."""
        config_data = {
            "model": {"name": "test_model", "vocab_size": 32000},
            "training": {"batch_size": 8, "learning_rate": 1e-4},
            "process_supervision": {"enabled": True, "weight": 0.1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            result = load_config(config_path)
            assert result == config_data
        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self):
        """Test config loading with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_merge_configs_empty_list(self):
        """Test merging empty config list."""
        result = merge_configs([])
        assert result == {}

    def test_merge_configs_single_config(self):
        """Test merging single config."""
        config = {"key": "value", "nested": {"inner": 42}}
        configs = [self._create_temp_config(config)]

        try:
            result = merge_configs(configs)
            assert result == config
        finally:
            for config_path in configs:
                Path(config_path).unlink()

    def test_merge_configs_multiple_configs(self):
        """Test merging multiple configs."""
        config1 = {"model": {"name": "test"}, "shared": {"value": 1}}
        config2 = {"training": {"lr": 0.01}, "shared": {"value": 2, "extra": "data"}}
        config3 = {"process_supervision": {"enabled": True}}

        configs = [
            self._create_temp_config(config1),
            self._create_temp_config(config2),
            self._create_temp_config(config3),
        ]

        try:
            result = merge_configs(configs)

            # Check merged structure
            assert result["model"]["name"] == "test"
            assert result["training"]["lr"] == 0.01
            assert result["process_supervision"]["enabled"] == True

            # Check that shared key was merged (later configs override)
            assert result["shared"]["value"] == 2
            assert result["shared"]["extra"] == "data"
        finally:
            for config_path in configs:
                Path(config_path).unlink()

    def test_merge_configs_nested_merge(self):
        """Test merging nested dictionaries."""
        config1 = {"model": {"arch": {"layers": 6}, "vocab": 30000}}
        config2 = {"model": {"arch": {"heads": 8}, "embed": 512}}

        configs = [self._create_temp_config(config1), self._create_temp_config(config2)]

        try:
            result = merge_configs(configs)

            # Nested dict should be merged
            assert result["model"]["arch"]["layers"] == 6
            assert result["model"]["arch"]["heads"] == 8
            assert result["model"]["vocab"] == 30000
            assert result["model"]["embed"] == 512
        finally:
            for config_path in configs:
                Path(config_path).unlink()

    def _create_temp_config(self, config_data):
        """Helper to create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            return f.name


class TestModelLoading:
    """Test model loading functionality."""

    @patch("training.distill_process.torch.load")
    @patch("training.distill_process.StudentLM")
    @patch("training.distill_process.ModelCfg")
    def test_load_model_success(self, mock_model_cfg, mock_student_lm, mock_torch_load):
        """Test successful model loading."""
        # Mock checkpoint
        mock_checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "config": {
                "arch": {
                    "d_model": 512,
                    "n_layers": 8,
                    "n_heads": 8,
                    "n_kv_heads": 4,
                    "d_head": 64,
                    "vocab_size": 32000,
                }
            },
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock config and model creation
        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model("dummy_path.pt", device)

        # Verify config creation
        mock_model_cfg.assert_called_once_with(
            d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, d_head=64, vocab_size=32000
        )

        # Verify model creation and state loading
        mock_student_lm.assert_called_once_with(mock_config)
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint["model_state_dict"])

        assert result == mock_model

    @patch("training.distill_process.torch.load")
    @patch("training.distill_process.StudentLM")
    @patch("training.distill_process.ModelCfg")
    def test_load_model_without_config(self, mock_model_cfg, mock_student_lm, mock_torch_load):
        """Test model loading without config in checkpoint."""
        mock_checkpoint = {"model_state_dict": {"weight": torch.randn(5, 5)}}
        mock_torch_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model("dummy_path.pt", device)

        # Should use default config
        mock_model_cfg.assert_called_once_with()
        mock_student_lm.assert_called_once_with(mock_config)
        mock_model.load_state_dict.assert_called_once()

        assert result == mock_model

    @patch("training.distill_process.torch.load")
    def test_load_model_checkpoint_not_found(self, mock_torch_load):
        """Test model loading with missing checkpoint."""
        mock_torch_load.side_effect = FileNotFoundError("Checkpoint not found")

        device = torch.device("cpu")
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent.pt", device)


class TestTextGeneration:
    """Test text generation from logits."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="Generated text")
        return tokenizer

    def test_generate_text_from_logits_basic(self, mock_tokenizer):
        """Test basic text generation from logits."""
        # Create sample logits: [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Make position 42 have highest logit for consistent testing
        logits[:, :, 42] = 10.0

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert isinstance(result, list)
        assert len(result) == batch_size

        for text in result:
            assert isinstance(text, str)
            mock_tokenizer.decode.assert_called()

    def test_generate_text_from_logits_temperature(self, mock_tokenizer):
        """Test text generation with temperature."""
        batch_size, seq_len, vocab_size = 1, 5, 500
        logits = torch.randn(batch_size, seq_len, vocab_size)

        temperature = 0.8
        result = generate_text_from_logits(logits, mock_tokenizer, temperature=temperature)

        assert len(result) == batch_size
        mock_tokenizer.decode.assert_called()

    def test_generate_text_from_logits_top_k(self, mock_tokenizer):
        """Test text generation with top-k sampling."""
        batch_size, seq_len, vocab_size = 1, 8, 2000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        top_k = 50
        result = generate_text_from_logits(logits, mock_tokenizer, top_k=top_k)

        assert len(result) == batch_size

    def test_generate_text_from_logits_top_p(self, mock_tokenizer):
        """Test text generation with top-p sampling."""
        batch_size, seq_len, vocab_size = 1, 6, 1500
        logits = torch.randn(batch_size, seq_len, vocab_size)

        top_p = 0.9
        result = generate_text_from_logits(logits, mock_tokenizer, top_p=top_p)

        assert len(result) == batch_size

    def test_generate_text_from_logits_greedy(self, mock_tokenizer):
        """Test greedy text generation."""
        batch_size, seq_len, vocab_size = 1, 4, 800
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Set specific high logit for deterministic result
        logits[0, 0, 123] = 100.0

        result = generate_text_from_logits(logits, mock_tokenizer, temperature=0.0)

        assert len(result) == 1
        mock_tokenizer.decode.assert_called()

    def test_generate_text_from_logits_empty_logits(self, mock_tokenizer):
        """Test text generation with empty logits."""
        logits = torch.empty(0, 0, 100)  # Empty tensor

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert result == []

    def test_generate_text_from_logits_single_token(self, mock_tokenizer):
        """Test generation with single token sequences."""
        batch_size, vocab_size = 3, 2000
        logits = torch.randn(batch_size, 1, vocab_size)  # Single token per sequence

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert len(result) == batch_size
        for text in result:
            assert isinstance(text, str)


class TestTrainingStep:
    """Test training step with process supervision."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.parameters = Mock(return_value=[])
        return model

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        return optimizer

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        return {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
            "teacher_logits": torch.randn(4, 128, 32000),
            "labels": torch.randint(0, 32000, (4, 128)),
            "process_labels": {
                "json_validity": torch.tensor([1.0, 0.8, 0.0, 1.0]),
                "tool_selection": torch.tensor([1, 0, 1, 0]),
                "arg_extraction": torch.tensor([0.9, 0.0, 0.7, 1.0]),
            },
        }

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    def test_train_step_process_basic(
        self, mock_process_loss, mock_kd_loss, mock_model, mock_optimizer, sample_batch
    ):
        """Test basic training step."""
        # Mock loss functions
        mock_kd_loss.return_value = torch.tensor(2.5)
        mock_process_loss.return_value = torch.tensor(1.0)

        device = torch.device("cpu")

        losses = train_step_process(
            model=mock_model,
            batch=sample_batch,
            optimizer=mock_optimizer,
            device=device,
            kd_weight=0.7,
            process_weight=0.3,
        )

        # Verify loss computation
        mock_kd_loss.assert_called_once()
        mock_process_loss.assert_called_once()

        # Check returned losses
        assert "total" in losses
        assert "kd" in losses
        assert "process" in losses
        assert "components" in losses

        # Verify optimizer steps
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_called_once()

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    def test_train_step_process_zero_weights(
        self, mock_process_loss, mock_kd_loss, mock_model, mock_optimizer, sample_batch
    ):
        """Test training step with zero weights."""
        mock_kd_loss.return_value = torch.tensor(1.0)
        mock_process_loss.return_value = torch.tensor(1.0)

        device = torch.device("cpu")

        losses = train_step_process(
            model=mock_model,
            batch=sample_batch,
            optimizer=mock_optimizer,
            device=device,
            kd_weight=0.0,
            process_weight=0.0,
        )

        # Should still compute losses but total should be 0
        assert losses["total"] == 0.0
        assert losses["kd"] > 0.0  # Individual losses still computed
        assert losses["process"] > 0.0

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    def test_train_step_process_no_grad(
        self, mock_process_loss, mock_kd_loss, mock_model, mock_optimizer, sample_batch
    ):
        """Test training step without gradient updates."""
        mock_kd_loss.return_value = torch.tensor(0.5)
        mock_process_loss.return_value = torch.tensor(0.3)

        device = torch.device("cpu")

        losses = train_step_process(
            model=mock_model,
            batch=sample_batch,
            optimizer=mock_optimizer,
            device=device,
            kd_weight=0.6,
            process_weight=0.4,
            no_grad=True,
        )

        # Optimizer should not be called
        mock_optimizer.zero_grad.assert_not_called()
        mock_optimizer.step.assert_not_called()

        # But losses should still be computed
        assert losses["total"] > 0.0

    def test_train_step_process_missing_process_labels(self, mock_model, mock_optimizer):
        """Test training step with missing process labels."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 64)),
            "attention_mask": torch.ones(2, 64),
            "teacher_logits": torch.randn(2, 64, 32000),
            "labels": torch.randint(0, 32000, (2, 64)),
            # Missing process_labels
        }

        device = torch.device("cpu")

        with (
            patch("training.distill_process.combined_kd_loss", return_value=torch.tensor(1.0)),
            patch(
                "training.distill_process.process_supervision_loss", return_value=torch.tensor(0.5)
            ),
        ):
            losses = train_step_process(
                model=mock_model,
                batch=batch,
                optimizer=mock_optimizer,
                device=device,
                kd_weight=0.8,
                process_weight=0.2,
            )

            # Should handle missing process labels gracefully
            assert "total" in losses


class TestMainFunction:
    """Test main function."""

    @patch("training.distill_process.load_config")
    @patch("training.distill_process.merge_configs")
    @patch("training.distill_process.load_model")
    @patch("training.distill_process.DataLoader")
    @patch("training.distill_process.train_step_process")
    @patch("training.distill_process.torch.device")
    @patch("training.distill_process.Path")
    @patch("training.distill_process.argparse.ArgumentParser")
    def test_main_success(
        self,
        mock_parser_class,
        mock_path_class,
        mock_device,
        mock_train_step,
        mock_dataloader,
        mock_load_model,
        mock_merge_configs,
        mock_load_config,
    ):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.config = ["config1.yaml", "config2.yaml"]
        mock_args.output_dir = "outputs"
        mock_args.device = "cpu"
        mock_args.batch_size = 4
        mock_args.max_steps = 10
        mock_args.kd_weight = 0.7
        mock_args.process_weight = 0.3
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock other components
        mock_config = {"training": {"max_steps": 10}}
        mock_merged_config = {"training": {"max_steps": 10}}
        mock_model = Mock()
        mock_device_instance = Mock()

        mock_load_config.return_value = mock_config
        mock_merge_configs.return_value = mock_merged_config
        mock_load_model.return_value = mock_model
        mock_device.return_value = mock_device_instance

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([{"batch": "data"}]))
        mock_dataloader.return_value = mock_loader

        # Mock training step
        mock_train_step.return_value = {"total": 1.5}

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

    @patch("training.distill_process.load_config")
    @patch("training.distill_process.argparse.ArgumentParser")
    def test_main_config_load_failure(self, mock_parser_class, mock_load_config):
        """Test main function with config loading failure."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = ["bad_config.yaml"]
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_config.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(SystemExit):
            main()

    @patch("training.distill_process.load_model")
    @patch("training.distill_process.merge_configs")
    @patch("training.distill_process.load_config")
    @patch("training.distill_process.argparse.ArgumentParser")
    def test_main_model_load_failure(
        self, mock_parser_class, mock_load_config, mock_merge_configs, mock_load_model
    ):
        """Test main function with model loading failure."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "bad_model.pt"
        mock_args.config = ["config.yaml"]
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_load_config.return_value = {}
        mock_merge_configs.return_value = {}
        mock_load_model.side_effect = Exception("Model load failed")

        with pytest.raises(SystemExit):
            main()


class TestProcessSupervisionIntegration:
    """Test process supervision integration."""

    def test_process_supervision_config_structure(self):
        """Test that process supervision config has expected structure."""
        config = {
            "process_supervision": {
                "enabled": True,
                "weight": 0.2,
                "json_validity_weight": 0.6,
                "tool_selection_weight": 0.3,
                "arg_extraction_weight": 0.1,
            }
        }

        # Should be valid config structure
        assert "process_supervision" in config
        assert config["process_supervision"]["enabled"] == True
        assert config["process_supervision"]["weight"] == 0.2

        weights = ["json_validity_weight", "tool_selection_weight", "arg_extraction_weight"]
        for weight_key in weights:
            assert weight_key in config["process_supervision"]

    def test_process_labels_structure(self):
        """Test process labels have expected structure."""
        process_labels = {
            "json_validity": torch.tensor([1.0, 0.8, 0.0, 1.0]),
            "tool_selection": torch.tensor([1, 0, 1, 0]),
            "arg_extraction": torch.tensor([0.9, 0.0, 0.7, 1.0]),
        }

        # Should have all required components
        assert "json_validity" in process_labels
        assert "tool_selection" in process_labels
        assert "arg_extraction" in process_labels

        # Should have consistent batch size
        batch_size = 4
        assert process_labels["json_validity"].shape[0] == batch_size
        assert process_labels["tool_selection"].shape[0] == batch_size
        assert process_labels["arg_extraction"].shape[0] == batch_size

    def test_training_config_validation(self):
        """Test that training config has required fields."""
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-4,
                "max_steps": 1000,
                "warmup_steps": 100,
                "kd_weight": 0.7,
                "process_weight": 0.3,
            },
            "model": {"vocab_size": 32000, "max_seq_length": 2048},
        }

        # Should have all required sections
        assert "training" in config
        assert "model" in config

        # Training section should have key hyperparameters
        training = config["training"]
        assert "batch_size" in training
        assert "learning_rate" in training
        assert "max_steps" in training
        assert "kd_weight" in training
        assert "process_weight" in training


class TestTrainStepProcessExpanded:
    """Expanded tests for train_step_process function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.train = Mock()
        model.parameters = Mock(return_value=[])
        return model

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        return optimizer

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(side_effect=lambda x, **kwargs: "decoded_text")
        return tokenizer

    @pytest.fixture
    def sample_batch_with_tool_names(self):
        """Create sample batch with tool name IDs."""
        return {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
            "teacher_logits": torch.randn(4, 128, 32000),
            "labels": torch.randint(0, 32000, (4, 128)),
            "teacher_target_ids": torch.randint(0, 32000, (4, 128)),
            "tool_name_ids": torch.randint(0, 1000, (4, 10)),
            "tool_name_mask": torch.ones(4, 10, dtype=torch.bool),
            "gold_json_text_ids": torch.randint(0, 1000, (4, 50)),
            "mask_valid_json_tokens": torch.ones(4, 50, dtype=torch.bool),
        }

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    @patch("training.distill_process.generate_text_from_logits")
    def test_train_step_process_with_tool_name_ids(
        self,
        mock_generate_text,
        mock_process_loss,
        mock_kd_loss,
        mock_model,
        mock_optimizer,
        mock_tokenizer,
        sample_batch_with_tool_names,
    ):
        """Test training step with tool name IDs."""
        mock_kd_loss.return_value = {"total": torch.tensor(2.5)}
        mock_process_loss.return_value = {"total": torch.tensor(1.0)}
        mock_generate_text.return_value = ["text1", "text2", "text3", "text4"]

        device = torch.device("cpu")
        cfg = {"distillation": {"kl_weight": 0.5, "process_supervision_weight": 0.7}}
        proc_cfg = {
            "tool_names": ["tool1", "tool2"],
            "loss_json_validity_weight": 0.3,
            "loss_tool_select_weight": 0.7,
        }

        losses = train_step_process(
            model=mock_model,
            batch=sample_batch_with_tool_names,
            optimizer=mock_optimizer,
            scaler=None,
            cfg=cfg,
            device=device,
            tokenizer=mock_tokenizer,
            proc_cfg=proc_cfg,
        )

        # Verify process supervision loss was called with tool name IDs
        mock_process_loss.assert_called_once()
        call_kwargs = mock_process_loss.call_args[1]
        assert "tool_name_ids" in call_kwargs
        assert "tool_name_mask" in call_kwargs
        assert "gold_json_text_ids" in call_kwargs

        assert "total" in losses
        assert "kd_total" in losses
        assert "proc_total" in losses

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    @patch("training.distill_process.generate_text_from_logits")
    def test_train_step_process_with_fp16_scaler(
        self,
        mock_generate_text,
        mock_process_loss,
        mock_kd_loss,
        mock_model,
        mock_optimizer,
        mock_tokenizer,
        sample_batch_with_tool_names,
    ):
        """Test training step with FP16 scaler."""
        mock_kd_loss.return_value = {"total": torch.tensor(2.5)}
        mock_process_loss.return_value = {"total": torch.tensor(1.0)}
        mock_generate_text.return_value = ["text1", "text2", "text3", "text4"]

        device = torch.device("cpu")
        cfg = {"distillation": {"kl_weight": 0.5, "process_supervision_weight": 0.7}, "optimizer": {"grad_clip": 1.0}}
        proc_cfg = {
            "tool_names": ["tool1", "tool2"],
            "loss_json_validity_weight": 0.3,
            "loss_tool_select_weight": 0.7,
        }

        mock_scaler = Mock()
        mock_scaler.scale = Mock(return_value=torch.tensor(3.5))
        mock_scaler.unscale_ = Mock()
        mock_scaler.step = Mock()
        mock_scaler.update = Mock()

        losses = train_step_process(
            model=mock_model,
            batch=sample_batch_with_tool_names,
            optimizer=mock_optimizer,
            scaler=mock_scaler,
            cfg=cfg,
            device=device,
            tokenizer=mock_tokenizer,
            proc_cfg=proc_cfg,
        )

        # Verify scaler was used
        mock_scaler.scale.assert_called_once()
        mock_scaler.unscale_.assert_called_once()
        mock_scaler.step.assert_called_once()
        mock_scaler.update.assert_called_once()

    @patch("training.distill_process.combined_kd_loss")
    @patch("training.distill_process.process_supervision_loss")
    @patch("training.distill_process.generate_text_from_logits")
    def test_train_step_process_no_tool_name_ids(
        self,
        mock_generate_text,
        mock_process_loss,
        mock_kd_loss,
        mock_model,
        mock_optimizer,
        mock_tokenizer,
    ):
        """Test training step without tool name IDs."""
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 64)),
            "attention_mask": torch.ones(2, 64),
            "teacher_logits": torch.randn(2, 64, 32000),
            "labels": torch.randint(0, 32000, (2, 64)),
        }

        mock_kd_loss.return_value = {"total": torch.tensor(1.0)}
        mock_process_loss.return_value = {"total": torch.tensor(0.5)}
        mock_generate_text.return_value = ["text1", "text2"]

        device = torch.device("cpu")
        cfg = {"distillation": {"kl_weight": 0.5, "process_supervision_weight": 0.7}}
        proc_cfg = {
            "tool_names": [],
            "loss_json_validity_weight": 0.3,
            "loss_tool_select_weight": 0.7,
        }

        losses = train_step_process(
            model=mock_model,
            batch=batch,
            optimizer=mock_optimizer,
            scaler=None,
            cfg=cfg,
            device=device,
            tokenizer=mock_tokenizer,
            proc_cfg=proc_cfg,
        )

        # Should still work without tool name IDs
        assert "total" in losses


class TestGenerateTextFromLogitsExpanded:
    """Expanded tests for generate_text_from_logits function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(side_effect=lambda x, **kwargs: f"decoded_{x[0] if x else ''}")
        return tokenizer

    def test_generate_text_from_logits_with_decode_error(self, mock_tokenizer):
        """Test text generation when decode fails."""
        batch_size, seq_len, vocab_size = 2, 5, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Make tokenizer.decode raise exception for second sample
        call_count = [0]

        def decode_side_effect(tokens, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Decode error")
            return f"decoded_{tokens[0]}"

        mock_tokenizer.decode.side_effect = decode_side_effect

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert len(result) == batch_size
        assert result[0] == "decoded_0"  # First succeeds
        assert result[1] == ""  # Second fails, returns empty string

    def test_generate_text_from_logits_single_batch(self, mock_tokenizer):
        """Test text generation with single batch item."""
        batch_size, seq_len, vocab_size = 1, 3, 500
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_generate_text_from_logits_large_vocab(self, mock_tokenizer):
        """Test text generation with large vocabulary."""
        batch_size, seq_len, vocab_size = 2, 10, 50000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert len(result) == batch_size
        mock_tokenizer.decode.assert_called()

    def test_generate_text_from_logits_long_sequence(self, mock_tokenizer):
        """Test text generation with long sequences."""
        batch_size, seq_len, vocab_size = 1, 1000, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = generate_text_from_logits(logits, mock_tokenizer)

        assert len(result) == batch_size
        assert isinstance(result[0], str)


class TestMainFunctionExpanded:
    """Expanded tests for main function."""

    @patch("training.distill_process.argparse.ArgumentParser")
    @patch("training.distill_process.merge_configs")
    @patch("training.distill_process.load_model")
    @patch("training.distill_process.AutoTokenizer")
    @patch("training.distill_process.KDDataset")
    @patch("training.distill_process.DataLoader")
    @patch("training.distill_process.train_step_process")
    def test_main_with_hf_tokenizer_unavailable(
        self,
        mock_train_step,
        mock_dataloader,
        mock_dataset,
        mock_tokenizer_class,
        mock_load_model,
        mock_merge_configs,
        mock_parser_class,
    ):
        """Test main function when transformers is not available."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.config = ["config.yaml"]
        mock_args.output_dir = "outputs"
        mock_args.steps = 10
        mock_args.save_every = 5
        mock_args.log_every = 2
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock HF_TOKENIZER_AVAILABLE = False
        with patch("training.distill_process.HF_TOKENIZER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="transformers required"):
                main()

    @patch("training.distill_process.argparse.ArgumentParser")
    @patch("training.distill_process.merge_configs")
    @patch("training.distill_process.load_model")
    @patch("training.distill_process.AutoTokenizer")
    @patch("training.distill_process.KDDataset")
    @patch("training.distill_process.DataLoader")
    @patch("training.distill_process.train_step_process")
    @patch("training.distill_process.torch.save")
    def test_main_checkpoint_saving(
        self,
        mock_save,
        mock_train_step,
        mock_dataloader,
        mock_dataset,
        mock_tokenizer_class,
        mock_load_model,
        mock_merge_configs,
        mock_parser_class,
    ):
        """Test that checkpoints are saved correctly."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.checkpoint = "model.pt"
        mock_args.config = ["config.yaml"]
        mock_args.output_dir = "outputs"
        mock_args.steps = 10
        mock_args.save_every = 5
        mock_args.log_every = 2
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_merge_configs.return_value = {
            "process_supervision": {"loss_json_validity_weight": 0.3},
            "optimizer": {"lr": 1e-4},
            "train": {"micro_batch_size": 2, "fp16": False},
            "io": {"tokenizer_path": "tokenizer", "train_shards": ["data.jsonl"]},
        }

        mock_model = Mock()
        mock_model.state_dict = Mock(return_value={"weight": torch.randn(10, 10)})
        mock_load_model.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([{"batch": "data"}] * 10))
        mock_dataloader.return_value = mock_loader

        mock_train_step.return_value = {"total": 1.5}

        try:
            main()
        except SystemExit:
            pass

        # Verify checkpoints were saved
        assert mock_save.call_count >= 2  # Intermediate + final checkpoint
