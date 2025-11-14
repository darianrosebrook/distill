"""
Tests for training/tracing.py - Training tracing and logging utilities.

Tests TrainingTracer class, TensorBoard integration, WandB integration,
JSON logging, and configuration-based tracer creation.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch

import pytest
import torch

from training.tracing import TrainingTracer, create_tracer_from_config


class TestTrainingTracerInit:
    """Test TrainingTracer initialization."""

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_tracer_init_basic(self, mock_summary_writer, tmp_path):
        """Test basic tracer initialization."""
        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        assert tracer.run_name == "test_run"
        assert tracer.log_dir.exists()
        assert tracer.use_tensorboard is True
        assert tracer.use_wandb is False
        assert tracer.json_log is True
        assert tracer.console_log is True

    @patch("training.tracing.TENSORBOARD_AVAILABLE", False)
    def test_tracer_init_no_tensorboard(self, tmp_path):
        """Test tracer initialization when TensorBoard not available."""
        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        assert tracer.use_tensorboard is False
        assert tracer.tb_writer is None

    @patch("training.tracing.WANDB_AVAILABLE", True)
    @patch("training.tracing.wandb")
    def test_tracer_init_with_wandb(self, mock_wandb, tmp_path):
        """Test tracer initialization with WandB."""
        mock_wandb.init.return_value = None

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
            wandb_project="test_project",
            wandb_entity="test_entity",
        )

        assert tracer.use_wandb is True
        mock_wandb.init.assert_called_once()

    @patch("training.tracing.WANDB_AVAILABLE", False)
    def test_tracer_init_no_wandb(self, tmp_path):
        """Test tracer initialization when WandB not available."""
        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
        )

        assert tracer.use_wandb is False

    def test_tracer_init_json_log_disabled(self, tmp_path):
        """Test tracer initialization with JSON logging disabled."""
        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            json_log=False,
        )

        assert tracer.json_log is False
        assert tracer.json_log_path is None

    def test_tracer_init_console_log_disabled(self, tmp_path):
        """Test tracer initialization with console logging disabled."""
        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            console_log=False,
        )

        assert tracer.console_log is False


class TestTrainingTracerLogMetrics:
    """Test TrainingTracer log_metrics method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    def test_log_metrics_basic(self, tracer):
        """Test basic metric logging."""
        metrics = {"loss": 0.5, "accuracy": 0.9}

        tracer.log_metrics(step=100, metrics=metrics)

        # Check metrics history
        assert "loss" in tracer.metrics_history
        assert "accuracy" in tracer.metrics_history
        assert len(tracer.metrics_history["loss"]) == 1
        assert tracer.metrics_history["loss"][0][1] == 0.5

    def test_log_metrics_with_prefix(self, tracer):
        """Test metric logging with prefix."""
        metrics = {"loss": 0.5, "accuracy": 0.9}

        tracer.log_metrics(step=100, metrics=metrics, prefix="train/")

        # Check that prefix was added
        assert "train/loss" in tracer.metrics_history
        assert "train/accuracy" in tracer.metrics_history

    def test_log_metrics_multiple_steps(self, tracer):
        """Test logging metrics at multiple steps."""
        for step in range(5):
            tracer.log_metrics(step=step, metrics={"loss": float(step)})

        assert len(tracer.metrics_history["loss"]) == 5
        assert tracer.metrics_history["loss"][-1][1] == 4.0

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_metrics_tensorboard(self, mock_summary_writer, tmp_path):
        """Test metric logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        metrics = {"loss": 0.5}
        tracer.log_metrics(step=100, metrics=metrics)

        mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 100)

    @patch("training.tracing.WANDB_AVAILABLE", True)
    @patch("training.tracing.wandb")
    def test_log_metrics_wandb(self, mock_wandb, tmp_path):
        """Test metric logging with WandB."""
        mock_wandb.init.return_value = None

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
        )

        metrics = {"loss": 0.5}
        tracer.log_metrics(step=100, metrics=metrics)

        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        assert "loss" in call_args[0][0]
        assert call_args[1]["step"] == 100

    def test_log_metrics_json_file(self, tracer):
        """Test that metrics are written to JSON log file."""
        metrics = {"loss": 0.5, "accuracy": 0.9}

        tracer.log_metrics(step=100, metrics=metrics)

        # Check JSON log file exists and contains entry
        assert tracer.json_log_path.exists()
        with open(tracer.json_log_path, "r") as f:
            lines = [line for line in f if line.strip() and not line.startswith("#")]
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["step"] == 100
            assert entry["loss"] == 0.5
            assert entry["accuracy"] == 0.9


class TestTrainingTracerLogHparams:
    """Test TrainingTracer log_hparams method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    def test_log_hparams_basic(self, tracer):
        """Test basic hyperparameter logging."""
        hparams = {"lr": 1e-4, "batch_size": 32}

        tracer.log_hparams(hparams)

        # Check that hparams file was created
        hparams_path = tracer.log_dir / "hparams.json"
        assert hparams_path.exists()

        with open(hparams_path, "r") as f:
            loaded = json.load(f)
            assert loaded["lr"] == 1e-4
            assert loaded["batch_size"] == 32

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_hparams_tensorboard(self, mock_summary_writer, tmp_path):
        """Test hyperparameter logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        hparams = {"lr": 1e-4}
        tracer.log_hparams(hparams)

        mock_writer.add_hparams.assert_called_once()

    @patch("training.tracing.WANDB_AVAILABLE", True)
    @patch("training.tracing.wandb")
    def test_log_hparams_wandb(self, mock_wandb, tmp_path):
        """Test hyperparameter logging with WandB."""
        mock_wandb.init.return_value = None

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
        )

        hparams = {"lr": 1e-4}
        tracer.log_hparams(hparams)

        mock_wandb.config.update.assert_called_once_with(hparams)


class TestTrainingTracerLogModelGraph:
    """Test TrainingTracer log_model_graph method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Linear(10, 5)

    @pytest.fixture
    def example_input(self):
        """Create example input for testing."""
        return torch.randn(2, 10)

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_model_graph_tensorboard(
        self, mock_summary_writer, tmp_path, simple_model, example_input
    ):
        """Test model graph logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        tracer.log_model_graph(simple_model, example_input)

        mock_writer.add_graph.assert_called_once()

    @patch("training.tracing.WANDB_AVAILABLE", True)
    @patch("training.tracing.wandb")
    def test_log_model_graph_wandb(
        self, mock_wandb, tmp_path, simple_model, example_input
    ):
        """Test model graph logging with WandB."""
        mock_wandb.init.return_value = None

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
        )

        tracer.log_model_graph(simple_model, example_input)

        mock_wandb.watch.assert_called_once()


class TestTrainingTracerLogText:
    """Test TrainingTracer log_text method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_text_tensorboard(self, mock_summary_writer, tmp_path):
        """Test text logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        tracer.log_text(step=100, tag="sample", text="Test text")

        mock_writer.add_text.assert_called_once_with("sample", "Test text", 100)

    def test_log_text_json_file(self, tracer):
        """Test that text is written to JSON log file."""
        tracer.log_text(step=100, tag="sample", text="Test text")

        text_log_path = tracer.log_dir / "text_logs.jsonl"
        assert text_log_path.exists()

        with open(text_log_path, "r") as f:
            entry = json.load(f)
            assert entry["step"] == 100
            assert entry["tag"] == "sample"
            assert entry["text"] == "Test text"


class TestTrainingTracerLogImage:
    """Test TrainingTracer log_image method."""

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_image_tensorboard(self, mock_summary_writer, tmp_path):
        """Test image logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        image = torch.randn(3, 32, 32)
        tracer.log_image(step=100, tag="attention", image=image)

        mock_writer.add_image.assert_called_once()


class TestTrainingTracerLogHistogram:
    """Test TrainingTracer log_histogram method."""

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_log_histogram_tensorboard(self, mock_summary_writer, tmp_path):
        """Test histogram logging with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        values = torch.randn(100)
        tracer.log_histogram(step=100, tag="gradients", values=values)

        mock_writer.add_histogram.assert_called_once()


class TestTrainingTracerGetMetricsSummary:
    """Test TrainingTracer get_metrics_summary method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    def test_get_metrics_summary_basic(self, tracer):
        """Test basic metrics summary."""
        # Log some metrics
        for step in range(5):
            tracer.log_metrics(step=step, metrics={"loss": float(step)})

        summary = tracer.get_metrics_summary()

        assert "loss" in summary
        assert summary["loss"]["mean"] == 2.0  # (0+1+2+3+4)/5
        assert summary["loss"]["min"] == 0.0
        assert summary["loss"]["max"] == 4.0
        assert summary["loss"]["latest"] == 4.0
        assert summary["loss"]["steps"] == 5

    def test_get_metrics_summary_empty(self, tracer):
        """Test metrics summary with no logged metrics."""
        summary = tracer.get_metrics_summary()

        assert summary == {}


class TestTrainingTracerSaveSummary:
    """Test TrainingTracer save_summary method."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create a tracer instance for testing."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            return TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            )

    def test_save_summary(self, tracer):
        """Test saving metrics summary."""
        # Log some metrics
        for step in range(3):
            tracer.log_metrics(step=step, metrics={"loss": float(step)})

        tracer.save_summary()

        summary_path = tracer.log_dir / "summary.json"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            summary = json.load(f)
            assert "loss" in summary
            assert summary["loss"]["mean"] == 1.0


class TestTrainingTracerClose:
    """Test TrainingTracer close method."""

    @patch("training.tracing.TENSORBOARD_AVAILABLE", True)
    @patch("training.tracing.SummaryWriter")
    def test_close_tensorboard(self, mock_summary_writer, tmp_path):
        """Test closing tracer with TensorBoard."""
        mock_writer = Mock()
        mock_summary_writer.return_value = mock_writer

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=True,
            use_wandb=False,
        )

        tracer.close()

        mock_writer.close.assert_called_once()

    @patch("training.tracing.WANDB_AVAILABLE", True)
    @patch("training.tracing.wandb")
    def test_close_wandb(self, mock_wandb, tmp_path):
        """Test closing tracer with WandB."""
        mock_wandb.init.return_value = None

        tracer = TrainingTracer(
            run_name="test_run",
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=True,
        )

        tracer.close()

        mock_wandb.finish.assert_called_once()

    def test_close_context_manager(self, tmp_path):
        """Test tracer as context manager."""
        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            with TrainingTracer(
                run_name="test_run",
                log_dir=str(tmp_path),
                use_tensorboard=False,
                use_wandb=False,
            ) as tracer:
                tracer.log_metrics(step=0, metrics={"loss": 0.5})

            # Summary should be saved after context exit
            summary_path = tracer.log_dir / "summary.json"
            assert summary_path.exists()


class TestCreateTracerFromConfig:
    """Test create_tracer_from_config function."""

    def test_create_tracer_from_config_basic(self, tmp_path):
        """Test creating tracer from basic config."""
        cfg = {
            "tracing": {
                "log_dir": str(tmp_path),
                "use_tensorboard": False,
                "use_wandb": False,
            }
        }

        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            tracer = create_tracer_from_config(cfg, run_name="test_run")

            assert tracer.run_name == "test_run"
            assert tracer.use_tensorboard is False
            assert tracer.use_wandb is False

    def test_create_tracer_from_config_default_run_name(self, tmp_path):
        """Test creating tracer with default run name."""
        cfg = {
            "tracing": {
                "log_dir": str(tmp_path),
            }
        }

        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            tracer = create_tracer_from_config(cfg)

            assert tracer.run_name.startswith("train_")
            assert len(tracer.run_name) > 10  # Should have timestamp

    def test_create_tracer_from_config_wandb(self, tmp_path):
        """Test creating tracer with WandB config."""
        cfg = {
            "tracing": {
                "log_dir": str(tmp_path),
                "use_wandb": True,
                "wandb_project": "test_project",
                "wandb_entity": "test_entity",
            }
        }

        with patch("training.tracing.WANDB_AVAILABLE", True):
            with patch("training.tracing.wandb") as mock_wandb:
                mock_wandb.init.return_value = None

                tracer = create_tracer_from_config(cfg, run_name="test_run")

                assert tracer.use_wandb is True
                mock_wandb.init.assert_called_once()

    def test_create_tracer_from_config_no_tracing_section(self, tmp_path):
        """Test creating tracer when config has no tracing section."""
        cfg = {}

        with patch("training.tracing.TENSORBOARD_AVAILABLE", False):
            tracer = create_tracer_from_config(cfg, run_name="test_run")

            # Should use defaults
            assert tracer.run_name == "test_run"
            assert tracer.json_log is True
            assert tracer.console_log is True







