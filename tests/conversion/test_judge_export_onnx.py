"""
Tests for conversion/judge_export_onnx.py - Judge-specific ONNX export.

Tests judge model ONNX export, config loading, and enumerated shape handling.
"""
# @author: @darianrosebrook

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer.testing
import yaml

# Import the module
import importlib
judge_export_onnx_module = importlib.import_module(
    "conversion.judge_export_onnx")
load_judge_config = judge_export_onnx_module.load_judge_config
main = judge_export_onnx_module.main
app = judge_export_onnx_module.app


class TestLoadJudgeConfig:
    """Test judge configuration loading."""

    def test_load_judge_config_success(self, tmp_path):
        """Test successful judge config loading."""
        config_path = tmp_path / "judge_config.yaml"
        config_data = {
            "arch": {
                "d_model": 2048,
                "n_layers": 24,
                "n_heads": 16,
                "n_kv_heads": 4,
                "d_head": 128,
                "vocab_size": 32000,
                "rope_theta": 10000.0,
                "rope_scaling": "dynamic",
                "dropout": 0.0,
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_judge_config(str(config_path))

        assert cfg.d_model == 2048
        assert cfg.n_layers == 24
        assert cfg.n_heads == 16
        assert cfg.n_kv_heads == 4
        assert cfg.d_head == 128
        assert cfg.vocab_size == 32000

    def test_load_judge_config_with_defaults(self, tmp_path):
        """Test judge config loading with missing fields using defaults."""
        config_path = tmp_path / "judge_config.yaml"
        config_data = {"arch": {"d_model": 1024}}  # Only partial config

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_judge_config(str(config_path))

        assert cfg.d_model == 1024
        assert cfg.n_layers == 24  # Default
        assert cfg.n_heads == 16  # Default
        assert cfg.vocab_size == 32000  # Default

    def test_load_judge_config_file_not_found(self):
        """Test judge config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_judge_config("nonexistent_config.yaml")

    def test_load_judge_config_no_yaml(self, tmp_path):
        """Test judge config loading when PyYAML is not available."""
        config_path = tmp_path / "judge_config.yaml"
        config_path.touch()

        with patch("conversion.judge_export_onnx.yaml", None):
            with pytest.raises(ImportError):
                load_judge_config(str(config_path))

    def test_load_judge_config_empty_arch(self, tmp_path):
        """Test judge config loading with empty arch section."""
        config_path = tmp_path / "judge_config.yaml"
        config_data = {"arch": {}}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_judge_config(str(config_path))

        # Should use all defaults
        assert cfg.d_model == 2048
        assert cfg.n_layers == 24


class TestJudgeExportONNX:
    """Test judge ONNX export functionality."""

    @pytest.fixture
    def judge_config_file(self, tmp_path):
        """Create a test judge config file."""
        config_path = tmp_path / "judge_config.yaml"
        config_data = {
            "arch": {
                "d_model": 64,  # Small for testing
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_head": 16,
                "vocab_size": 256,
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @pytest.fixture
    def shape_sets_file(self, tmp_path):
        """Create a test shape sets file."""
        shape_path = tmp_path / "shape_sets.json"
        shape_data = {"judge_sequences": [64, 128, 256]}
        with open(shape_path, "w") as f:
            json.dump(shape_data, f)
        return str(shape_path)

    def test_main_success(self, tmp_path, judge_config_file, shape_sets_file):
        """Test successful judge ONNX export."""
        output_dir = tmp_path / "artifacts" / "onnx" / "judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("conversion.judge_export_onnx.StudentLM") as mock_student_lm, patch(
            "conversion.judge_export_onnx.torch.onnx.export"
        ) as mock_export:
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_student_lm.return_value = mock_model

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    "--config",
                    shape_sets_file,
                    "--judge-config",
                    judge_config_file,
                ],
            )

            assert result.exit_code == 0
            assert mock_student_lm.called
            assert mock_export.call_count == 3  # One for each sequence length

    def test_main_default_configs(self, tmp_path, judge_config_file):
        """Test judge export with default config paths."""
        output_dir = tmp_path / "artifacts" / "onnx" / "judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create default shape_sets.json in conversion directory
        default_shape_path = Path("conversion/shape_sets.json")
        if not default_shape_path.exists():
            default_shape_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_shape_path, "w") as f:
                json.dump({"judge_sequences": [512, 1024, 2048]}, f)

        try:
            with patch("conversion.judge_export_onnx.StudentLM") as mock_student_lm, patch(
                "conversion.judge_export_onnx.torch.onnx.export"
            ), patch("conversion.judge_export_onnx.load_judge_config") as mock_load:
                from models.student.architectures.gqa_transformer import ModelCfg

                mock_cfg = ModelCfg(
                    d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_head=16, vocab_size=256
                )
                mock_load.return_value = mock_cfg

                mock_model = Mock()
                mock_model.eval = Mock()
                mock_student_lm.return_value = mock_model

                runner = typer.testing.CliRunner()
                result = runner.invoke(
                    app, ["--judge-config", judge_config_file])

                # Should use default shape_sets.json
                assert result.exit_code == 0
        finally:
            # Cleanup
            if default_shape_path.exists():
                default_shape_path.unlink()

    def test_main_shape_sets_fallback(self, tmp_path, judge_config_file):
        """Test judge export with shape sets file fallback."""
        output_dir = tmp_path / "artifacts" / "onnx" / "judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("conversion.judge_export_onnx.StudentLM") as mock_student_lm, patch(
            "conversion.judge_export_onnx.torch.onnx.export"
        ) as mock_export, patch("conversion.judge_export_onnx.json.load") as mock_json_load:
            # Mock json.load to raise FileNotFoundError when trying to load the config file
            # This simulates the file not being found
            mock_json_load.side_effect = FileNotFoundError("File not found")

            mock_model = Mock()
            mock_model.eval = Mock()
            mock_student_lm.return_value = mock_model

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    "--config",
                    "nonexistent.json",
                    "--judge-config",
                    judge_config_file,
                ],
            )

            # Should use fallback sequences [512, 1024, 2048] when file not found
            # The code has a try/except that catches the exception and uses defaults
            assert result.exit_code == 0
            assert mock_export.call_count == 3

    def test_main_config_loading_error(self, tmp_path):
        """Test judge export with config loading error."""
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.touch()  # Empty file

        runner = typer.testing.CliRunner()
        result = runner.invoke(
            app,
            [
                "--judge-config",
                str(invalid_config),
            ],
        )

        # Should handle error gracefully or exit
        assert result.exit_code != 0 or "Error" in result.output

    def test_main_model_creation(self, tmp_path, judge_config_file, shape_sets_file):
        """Test judge model creation with correct config."""
        output_dir = tmp_path / "artifacts" / "onnx" / "judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("conversion.judge_export_onnx.StudentLM") as mock_student_lm, patch(
            "conversion.judge_export_onnx.torch.onnx.export"
        ):
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_student_lm.return_value = mock_model

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    "--config",
                    shape_sets_file,
                    "--judge-config",
                    judge_config_file,
                ],
            )

            assert result.exit_code == 0
            # Verify StudentLM was called with the loaded config
            assert mock_student_lm.called
            call_args = mock_student_lm.call_args[0]
            assert call_args[0].d_model == 64  # From test config

    def test_main_export_paths(self, tmp_path, judge_config_file, shape_sets_file):
        """Test judge export creates correct output paths."""
        output_dir = tmp_path / "artifacts" / "onnx" / "judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch("conversion.judge_export_onnx.StudentLM") as mock_student_lm, patch(
            "conversion.judge_export_onnx.torch.onnx.export"
        ) as mock_export:
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_student_lm.return_value = mock_model

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    "--config",
                    shape_sets_file,
                    "--judge-config",
                    judge_config_file,
                ],
            )

            assert result.exit_code == 0
            # Check export was called with correct paths
            export_calls = mock_export.call_args_list
            assert len(export_calls) == 3

            # Check paths contain sequence lengths
            paths = [call[0][2]
                     for call in export_calls]  # Third arg is output path
            assert any("T64" in path for path in paths)
            assert any("T128" in path for path in paths)
            assert any("T256" in path for path in paths)
