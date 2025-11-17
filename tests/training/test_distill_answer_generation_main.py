"""
Additional tests for training/distill_answer_generation.py main function.

Tests main function setup, FP16 scaler, and checkpoint saving.
"""
# @author: @darianrosebrook

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from training.distill_answer_generation import main


class TestMain:
    """Test main function."""

    @patch("training.distill_answer_generation.create_tracer_from_config")
    @patch("training.distill_answer_generation.DataLoader")
    @patch("training.distill_answer_generation.AnswerGenerationDataset")
    @patch("training.distill_answer_generation.create_model")
    @patch("training.distill_answer_generation.load_config")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_basic_setup(self, mock_parse_args, mock_load_config, mock_create_model, 
                              mock_dataset_class, mock_dataloader_class, mock_create_tracer, tmp_path):
        """Test main function basic setup (lines 110-169)."""
        # Create config and data files
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({
                "arch": {},
                "optimizer": {"lr": 0.001},
                "train": {"steps": 5, "log_every": 2, "save_every": 5, "micro_batch_size": 1},
                "io": {}
            }, f)
        
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        
        # Mock argparse
        mock_args = Mock()
        mock_args.config = str(config_file)
        mock_args.data = str(data_file)
        mock_args.output_dir = str(tmp_path / "output")
        mock_args.resume = None
        mock_parse_args.return_value = mock_args
        
        mock_config = {
            "arch": {},
            "optimizer": {"lr": 0.001},
            "train": {"steps": 5, "log_every": 2, "save_every": 5, "micro_batch_size": 1},
            "io": {}
        }
        mock_load_config.return_value = mock_config
        
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        mock_model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
        mock_create_model.return_value = mock_model
        
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset
        
        # Mock dataloader to return a few batches
        mock_batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "labels": torch.randint(0, 1000, (1, 10)),
        }
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch] * 3))  # 3 batches
        mock_dataloader_class.return_value = mock_dataloader
        
        mock_tracer = Mock()
        mock_create_tracer.return_value = mock_tracer
        
        # Mock train_step
        with patch("training.distill_answer_generation.train_step") as mock_train_step:
            mock_train_step.return_value = {"total": 0.5, "ce": 0.5}
            
            # Run main
            main()
            
            # Verify key components were called
            mock_load_config.assert_called_once()
            mock_create_model.assert_called_once()
            mock_tracer.log_hparams.assert_called_once()
            mock_tracer.close.assert_called_once()
            # Should have called train_step
            assert mock_train_step.called

    @patch("training.distill_answer_generation.create_tracer_from_config")
    @patch("training.distill_answer_generation.DataLoader")
    @patch("training.distill_answer_generation.AnswerGenerationDataset")
    @patch("training.distill_answer_generation.create_model")
    @patch("training.distill_answer_generation.load_config")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_with_fp16_scaler(self, mock_parse_args, mock_load_config, mock_create_model, 
                                   mock_dataset_class, mock_dataloader_class, mock_create_tracer, tmp_path):
        """Test main function with FP16 scaler enabled (line 142)."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({
                "arch": {},
                "optimizer": {"lr": 0.001},
                "train": {"steps": 2, "log_every": 1, "save_every": 2, "micro_batch_size": 1, "fp16": True},
                "io": {}
            }, f)
        
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        
        mock_args = Mock()
        mock_args.config = str(config_file)
        mock_args.data = str(data_file)
        mock_args.output_dir = str(tmp_path / "output")
        mock_args.resume = None
        mock_parse_args.return_value = mock_args
        
        mock_config = {
            "arch": {},
            "optimizer": {"lr": 0.001},
            "train": {"steps": 2, "log_every": 1, "save_every": 2, "micro_batch_size": 1, "fp16": True},
            "io": {}
        }
        mock_load_config.return_value = mock_config
        
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        mock_model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
        mock_create_model.return_value = mock_model
        
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "labels": torch.randint(0, 1000, (1, 10)),
        }
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch] * 2))
        mock_dataloader_class.return_value = mock_dataloader
        
        mock_tracer = Mock()
        mock_create_tracer.return_value = mock_tracer
        
        with patch("training.distill_answer_generation.train_step") as mock_train_step:
            mock_train_step.return_value = {"total": 0.5, "ce": 0.5}
            
            main()
            
            # Verify setup completed
            mock_load_config.assert_called_once()
            mock_create_model.assert_called_once()

    @patch("training.distill_answer_generation.create_tracer_from_config")
    @patch("training.distill_answer_generation.DataLoader")
    @patch("training.distill_answer_generation.AnswerGenerationDataset")
    @patch("training.distill_answer_generation.create_model")
    @patch("training.distill_answer_generation.load_config")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_checkpoint_saving(self, mock_parse_args, mock_load_config, mock_create_model, 
                                    mock_dataset_class, mock_dataloader_class, mock_create_tracer, tmp_path):
        """Test main function checkpoint saving (lines 197-210, 215-226)."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({
                "arch": {},
                "optimizer": {"lr": 0.001},
                "train": {"steps": 3, "log_every": 1, "save_every": 2, "micro_batch_size": 1},
                "io": {}
            }, f)
        
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        
        output_dir = tmp_path / "output"
        mock_args = Mock()
        mock_args.config = str(config_file)
        mock_args.data = str(data_file)
        mock_args.output_dir = str(output_dir)
        mock_args.resume = None
        mock_parse_args.return_value = mock_args
        
        mock_config = {
            "arch": {},
            "optimizer": {"lr": 0.001},
            "train": {"steps": 3, "log_every": 1, "save_every": 2, "micro_batch_size": 1},
            "io": {}
        }
        mock_load_config.return_value = mock_config
        
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        mock_model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
        mock_create_model.return_value = mock_model
        
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "labels": torch.randint(0, 1000, (1, 10)),
        }
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch] * 3))
        mock_dataloader_class.return_value = mock_dataloader
        
        mock_tracer = Mock()
        mock_create_tracer.return_value = mock_tracer
        
        with patch("training.distill_answer_generation.train_step") as mock_train_step:
            mock_train_step.return_value = {"total": 0.5, "ce": 0.5}
            
            main()
            
            # Verify checkpoint files were created
            checkpoint_files = list(output_dir.glob("final_step_*.pt"))
            final_checkpoint = output_dir / "final_latest.pt"
            
            # Should have at least one checkpoint saved
            assert len(checkpoint_files) > 0 or final_checkpoint.exists()
            mock_tracer.close.assert_called_once()


