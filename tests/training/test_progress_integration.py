# tests/training/test_progress_integration.py
"""
Tests for progress integration module.

Tests cover:
- TrainingProgressContext initialization and lifecycle
- Metrics update and calculation
- Checkpoint recording through integration
- API response recording
- Connection failure handling
- Recovery checkpoint retrieval
"""

import pytest
import time
import torch
from training.progress_integration import (
    TrainingProgressContext,
    setup_progress_tracking,
    get_recovery_checkpoint,
    calculate_metrics_from_step,
    training_progress_context,
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "training_output"


@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return {
        "model": {"vocab_size": 32000, "d_model": 2048},
        "train": {"batch_size": 32},
        "progress": {"metrics_update_interval": 1, "checkpoint_interval": 1000},
    }


class TestTrainingProgressContext:
    """Test TrainingProgressContext initialization and lifecycle."""

    def test_context_manager_lifecycle(self, temp_output_dir, sample_config):
        """Test context manager enter/exit."""
        ctx = TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        )

        with ctx as progress:
            assert progress is ctx
            assert progress.tracker is not None
            assert progress.is_main_process is True

        # After exit, tracker should be closed
        assert progress.tracker is not None

    def test_non_main_process_no_tracking(self, temp_output_dir, sample_config):
        """Test that non-main processes don't create trackers."""
        ctx = TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=False,
        )

        with ctx as progress:
            assert progress.tracker is None

    def test_session_id_persistence(self, temp_output_dir, sample_config):
        """Test using specific session ID."""
        session_id = "test_session_12345"

        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            session_id=session_id,
            is_main_process=True,
        ) as progress:
            assert progress.session_id == session_id


class TestMetricsUpdate:
    """Test metrics update functionality."""

    def test_update_metrics(self, temp_output_dir, sample_config):
        """Test updating metrics through context."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            progress.update_metrics(
                step=100,
                loss=2.5,
                loss_components={"kd": 2.0, "ce": 0.5},
                learning_rate=1e-4,
                samples_processed=3200,
                tokens_processed=102400,
            )

            assert progress.tracker.session.steps_completed == 100

    def test_throughput_calculation(self, temp_output_dir, sample_config):
        """Test throughput calculation between updates."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            # First update
            progress.update_metrics(
                step=1,
                loss=2.5,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=32,
                tokens_processed=1024,
            )

            # Small delay
            time.sleep(0.1)

            # Second update
            progress.update_metrics(
                step=2,
                loss=2.4,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=64,
                tokens_processed=2048,
            )

            # Check throughput was calculated
            assert len(progress.tracker.session.metrics_history) == 2


class TestCheckpointRecording:
    """Test checkpoint recording through integration."""

    def test_record_checkpoint(self, temp_output_dir, sample_config):
        """Test recording checkpoint."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            # Record metrics first
            progress.update_metrics(
                step=100,
                loss=2.5,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=3200,
                tokens_processed=102400,
            )

            # Record checkpoint
            checkpoint_path = temp_output_dir / "checkpoint_100.pt"
            progress.record_checkpoint(
                step=100,
                checkpoint_path=checkpoint_path,
                dataset_position=3200,
                is_best=False,
            )

            assert progress.checkpoint_counter == 1
            assert 100 in progress.tracker.session.checkpoints

    def test_checkpoint_recovery_tags(self, temp_output_dir, sample_config):
        """Test checkpoint recovery tags."""
        config = sample_config.copy()
        config["progress"]["checkpoint_interval"] = 100

        with TrainingProgressContext(
            config=config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            progress.update_metrics(
                step=100,
                loss=2.5,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=3200,
                tokens_processed=102400,
            )

            checkpoint_path = temp_output_dir / "checkpoint_100.pt"

            # Record as periodic checkpoint
            progress.record_checkpoint(
                step=100,
                checkpoint_path=checkpoint_path,
                dataset_position=3200,
                is_best=False,
            )

            metadata = progress.tracker.session.checkpoints[100]
            assert "periodic" in metadata.recovery_tags

            # Record as best checkpoint
            progress.record_checkpoint(
                step=100,
                checkpoint_path=checkpoint_path,
                dataset_position=3200,
                is_best=True,
            )

            # Should now have both tags
            metadata = progress.tracker.session.checkpoints[100]
            assert "best_loss" in metadata.recovery_tags


class TestAPIResponseRecording:
    """Test API response recording."""

    def test_record_api_response(self, temp_output_dir, sample_config):
        """Test recording API response."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            response = {
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }

            progress.record_api_response(
                prompt_hash="abc123",
                response=response,
                cost=0.001,
            )

            assert progress.tracker.session.api_responses_saved == 1


class TestConnectionFailures:
    """Test connection failure recording."""

    def test_record_connection_failure(self, temp_output_dir, sample_config):
        """Test recording connection failure."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            progress.record_connection_failure(
                retry_count=3,
                error="Connection timeout",
            )

            assert progress.tracker.session.connection_failures == 1
            assert progress.tracker.session.connection_retries == 3


class TestProgressSummary:
    """Test progress summary through integration."""

    def test_get_progress_summary(self, temp_output_dir, sample_config):
        """Test getting progress summary."""
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            for step in range(1, 51):
                progress.update_metrics(
                    step=step,
                    loss=3.0 - (step * 0.001),
                    loss_components={},
                    learning_rate=1e-4,
                    samples_processed=step * 32,
                    tokens_processed=step * 1024,
                )

            summary = progress.get_progress_summary()

            assert summary["steps_completed"] == 50
            assert summary["total_steps_target"] == 1000
            assert summary["progress_pct"] == 5.0


class TestMetricsCalculation:
    """Test metric calculation utilities."""

    def test_calculate_metrics_from_tensor_loss(self):
        """Test calculating metrics from tensor loss."""
        loss = torch.tensor(2.5)
        loss_value, loss_components, lr = calculate_metrics_from_step(
            loss, loss_dict=None, scheduler=None
        )

        assert loss_value == 2.5
        assert loss_components == {"total": 2.5}
        assert lr == 0.0

    def test_calculate_metrics_with_components(self):
        """Test calculating metrics with loss components."""
        loss = torch.tensor(2.5)
        loss_dict = {
            "kd": torch.tensor(2.0),
            "ce": torch.tensor(0.5),
        }

        loss_value, loss_components, lr = calculate_metrics_from_step(
            loss, loss_dict=loss_dict, scheduler=None
        )

        assert loss_value == 2.5
        assert loss_components["kd"] == 2.0
        assert loss_components["ce"] == 0.5

    def test_calculate_metrics_with_scheduler(self):
        """Test calculating metrics with learning rate scheduler."""
        loss = torch.tensor(2.5)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        loss_value, loss_components, lr = calculate_metrics_from_step(
            loss, loss_dict=None, scheduler=scheduler
        )

        assert loss_value == 2.5
        assert lr == 0.1


class TestSetupProgressTracking:
    """Test setup_progress_tracking function."""

    def test_setup_creates_context(self, temp_output_dir, sample_config):
        """Test setup_progress_tracking creates context."""
        ctx = setup_progress_tracking(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        )

        assert isinstance(ctx, TrainingProgressContext)
        assert ctx.total_steps == 1000


class TestGetRecoveryCheckpoint:
    """Test retrieving recovery checkpoints."""

    def test_get_recovery_checkpoint(self, temp_output_dir, sample_config):
        """Test getting recovery checkpoint."""
        # Create progress and checkpoint
        with TrainingProgressContext(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            progress.update_metrics(
                step=100,
                loss=2.5,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=3200,
                tokens_processed=102400,
            )

            checkpoint_path = temp_output_dir / "checkpoint_100.pt"
            progress.record_checkpoint(
                step=100,
                checkpoint_path=checkpoint_path,
                dataset_position=3200,
            )

        # Retrieve checkpoint
        result = get_recovery_checkpoint(temp_output_dir)

        assert result is not None
        checkpoint_path, dataset_position = result
        assert str(checkpoint_path).endswith("checkpoint_100.pt")
        assert dataset_position == 3200

    def test_get_recovery_checkpoint_no_progress(self, temp_output_dir):
        """Test getting recovery checkpoint with no progress."""
        result = get_recovery_checkpoint(temp_output_dir)
        assert result is None


class TestContextManagerConvenience:
    """Test convenience context manager."""

    def test_training_progress_context(self, temp_output_dir, sample_config):
        """Test training_progress_context convenience context manager."""
        with training_progress_context(
            config=sample_config,
            output_dir=temp_output_dir,
            total_steps=1000,
            is_main_process=True,
        ) as progress:
            assert progress is not None
            assert isinstance(progress, TrainingProgressContext)

            progress.update_metrics(
                step=1,
                loss=2.5,
                loss_components={},
                learning_rate=1e-4,
                samples_processed=32,
                tokens_processed=1024,
            )

            assert progress.tracker.session.steps_completed == 1
