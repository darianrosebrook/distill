"""
Tests for conversion/judge_export_coreml.py - Judge-specific CoreML export.

Tests judge model conversion to CoreML, error handling, and placeholder creation.
"""
# @author: @darianrosebrook

from unittest.mock import patch

import typer.testing

# Import the module
import importlib
judge_export_coreml_module = importlib.import_module("conversion.judge_export_coreml")
main = judge_export_coreml_module.main
app = judge_export_coreml_module.app


class TestJudgeExportCoreML:
    """Test judge CoreML export functionality."""

    def test_main_success(self, tmp_path):
        """Test successful judge CoreML conversion."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()  # Create empty file
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = str(output_path)

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                    "--target",
                    "macOS13",
                    "--compute-units",
                    "all",
                ],
            )

            assert result.exit_code == 0
            mock_convert.assert_called_once()
            call_kwargs = mock_convert.call_args[1]
            assert call_kwargs["onnx_path"] == str(onnx_path)
            assert call_kwargs["output_path"] == str(output_path)
            assert call_kwargs["target"] == "macOS13"
            assert call_kwargs["compute_units"] == "all"

    def test_main_onnx_file_not_found(self, tmp_path):
        """Test judge export with missing ONNX file."""
        onnx_path = tmp_path / "nonexistent.onnx"
        output_path = tmp_path / "judge.mlpackage"

        runner = typer.testing.CliRunner()
        result = runner.invoke(
            app,
            [
                str(onnx_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 1
        assert "ONNX model not found" in result.output

    def test_main_conversion_failure_without_placeholder(self, tmp_path):
        """Test judge export with conversion failure without placeholder flag."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 1
            assert "Error during conversion" in result.output

    def test_main_conversion_failure_with_placeholder(self, tmp_path):
        """Test judge export with conversion failure with placeholder flag."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                    "--allow-placeholder",
                ],
            )

            # Should exit with 0 when placeholder is allowed
            assert result.exit_code == 0
            assert "Creating placeholder" in result.output

    def test_main_placeholder_created(self, tmp_path):
        """Test judge export when placeholder is created."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = None  # Placeholder created

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                    "--allow-placeholder",
                ],
            )

            assert result.exit_code == 0
            assert "Placeholder created" in result.output

    def test_main_import_failure(self, tmp_path):
        """Test judge export when conversion utilities cannot be imported."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml", None):
            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 1
            assert "Could not import conversion utilities" in result.output

    def test_main_default_output_path(self, tmp_path):
        """Test judge export with default output path."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = "coreml/judge.mlpackage"

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_convert.call_args[1]
            assert call_kwargs["output_path"] == "coreml/judge.mlpackage"

    def test_main_custom_compute_units(self, tmp_path):
        """Test judge export with custom compute units."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = str(output_path)

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                    "--compute-units",
                    "cpuonly",
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_convert.call_args[1]
            assert call_kwargs["compute_units"] == "cpuonly"

    def test_main_custom_target(self, tmp_path):
        """Test judge export with custom deployment target."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = str(output_path)

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                    "--target",
                    "macOS14",
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_convert.call_args[1]
            assert call_kwargs["target"] == "macOS14"

    def test_main_success_message(self, tmp_path):
        """Test judge export success message."""
        onnx_path = tmp_path / "judge.onnx"
        onnx_path.touch()
        output_path = tmp_path / "judge.mlpackage"

        with patch("conversion.judge_export_coreml.convert_onnx_to_coreml") as mock_convert:
            mock_convert.return_value = str(output_path)

            runner = typer.testing.CliRunner()
            result = runner.invoke(
                app,
                [
                    str(onnx_path),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert "Successfully converted judge model" in result.output
            assert "Note: Judge should use INT8 weights" in result.output

















