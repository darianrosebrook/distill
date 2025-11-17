# tests/training/test_progress_tracker.py
"""
Tests for training progress tracker with recovery capabilities.

Tests cover:
- Multi-day session tracking
- Checkpoint recovery
- Metrics persistence
- API response caching for recovery
- Connection failure handling
- Progress summary generation
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from training.progress_tracker import (
    ProgressTracker,
    TrainingSession,
    MetricsSnapshot,
    CheckpointMetadata,
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "progress_tracking"


@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return {
        "model": {
            "vocab_size": 32000,
            "d_model": 2048,
            "n_layers": 24,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
        },
    }


class TestProgressTrackerInitialization:
    """Test progress tracker initialization and session management."""

    def test_create_new_session(self, temp_output_dir, sample_config):
        """Test creating a new training session."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        assert tracker.session is not None
        assert tracker.session.total_steps_target == 10000
        assert tracker.session.steps_completed == 0
        assert tracker.session.session_id.startswith("training_")

    def test_resume_existing_session(self, temp_output_dir, sample_config):
        """Test resuming an existing training session."""
        # Create initial session
        tracker1 = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )
        session_id = tracker1.session.session_id

        # Update metrics
        tracker1.update_metrics(
            step=100,
            loss=2.5,
            loss_components={"kd": 2.0, "ce": 0.5},
            learning_rate=1e-4,
            samples_processed=3200,
            tokens_processed=102400,
            throughput_samples_per_sec=32.0,
            throughput_tokens_per_sec=1024.0,
        )
        tracker1.save()

        # Resume session
        tracker2 = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
            session_id=session_id,
        )

        assert tracker2.session.session_id == session_id
        assert tracker2.session.steps_completed == 100
        assert tracker2.session.samples_processed == 3200

    def test_session_validation_on_config_change(self, temp_output_dir, sample_config):
        """Test that config changes create new session."""
        tracker1 = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )
        session_id_1 = tracker1.session.session_id

        # Modify config
        modified_config = sample_config.copy()
        modified_config["training"]["batch_size"] = 64

        # Create new tracker with modified config
        tracker2 = ProgressTracker(
            output_dir=temp_output_dir,
            config=modified_config,
            total_steps=10000,
        )

        # Should have different session IDs
        assert tracker2.session.session_id != session_id_1


class TestMetricsTracking:
    """Test metrics tracking and recording."""

    def test_update_metrics(self, temp_output_dir, sample_config):
        """Test updating training metrics."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        tracker.update_metrics(
            step=100,
            loss=2.5,
            loss_components={"kd": 2.0, "ce": 0.5},
            learning_rate=1e-4,
            samples_processed=3200,
            tokens_processed=102400,
            throughput_samples_per_sec=32.0,
            throughput_tokens_per_sec=1024.0,
        )

        # Check session was updated
        assert tracker.session.steps_completed == 100
        assert tracker.session.samples_processed == 3200
        assert len(tracker.session.metrics_history) == 1

        # Check metrics file was created
        assert tracker.metrics_file.exists()

    def test_moving_average_loss(self, temp_output_dir, sample_config):
        """Test moving average loss calculation."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        losses = [3.0, 2.8, 2.6, 2.5, 2.4]

        for step, loss in enumerate(losses, start=1):
            tracker.update_metrics(
                step=step,
                loss=loss,
                loss_components={"kd": loss * 0.8, "ce": loss * 0.2},
                learning_rate=1e-4,
                samples_processed=step * 32,
                tokens_processed=step * 1024,
                throughput_samples_per_sec=32.0,
                throughput_tokens_per_sec=1024.0,
            )

        # Moving average should be updated
        assert tracker.session.moving_avg_loss > 0
        assert tracker.session.moving_avg_loss < max(losses)

    def test_loss_trend_analysis(self, temp_output_dir, sample_config):
        """Test loss trend analysis."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        losses = [3.0, 2.8, 2.6, 2.5, 2.4, 2.35, 2.30, 2.25, 2.2, 2.15]

        for step, loss in enumerate(losses, start=1):
            tracker.update_metrics(
                step=step,
                loss=loss,
                loss_components={"kd": loss * 0.8, "ce": loss * 0.2},
                learning_rate=1e-4,
                samples_processed=step * 32,
                tokens_processed=step * 1024,
                throughput_samples_per_sec=32.0,
                throughput_tokens_per_sec=1024.0,
            )

        trend = tracker.get_loss_trend()

        assert trend["has_data"] is True
        assert trend["current_loss"] == 2.15
        assert trend["min_loss"] == 2.15
        assert trend["max_loss"] == 3.0
        assert trend["is_improving"] is True


class TestCheckpointRecording:
    """Test checkpoint metadata recording for recovery."""

    def test_record_checkpoint(self, temp_output_dir, sample_config):
        """Test recording checkpoint metadata."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        # Record metrics first
        tracker.update_metrics(
            step=100,
            loss=2.5,
            loss_components={"kd": 2.0, "ce": 0.5},
            learning_rate=1e-4,
            samples_processed=3200,
            tokens_processed=102400,
            throughput_samples_per_sec=32.0,
            throughput_tokens_per_sec=1024.0,
        )

        # Record checkpoint
        checkpoint_path = temp_output_dir / "checkpoint_100.pt"
        tracker.record_checkpoint(
            step=100,
            checkpoint_path=checkpoint_path,
            dataset_position=3200,
            recovery_tags=["every_1k_steps"],
        )

        # Verify checkpoint was recorded
        assert tracker.session.last_checkpoint_step == 100
        assert 100 in tracker.session.checkpoints
        assert tracker.session.checkpoints[100].checkpoint_step == 100
        assert tracker.checkpoint_manifest.exists()

    def test_get_recovery_checkpoint(self, temp_output_dir, sample_config):
        """Test retrieving checkpoint for recovery."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        # No checkpoints yet
        assert tracker.get_recovery_checkpoint() is None

        # Record checkpoint
        tracker.update_metrics(
            step=100,
            loss=2.5,
            loss_components={"kd": 2.0, "ce": 0.5},
            learning_rate=1e-4,
            samples_processed=3200,
            tokens_processed=102400,
            throughput_samples_per_sec=32.0,
            throughput_tokens_per_sec=1024.0,
        )

        checkpoint_path = temp_output_dir / "checkpoint_100.pt"
        tracker.record_checkpoint(
            step=100,
            checkpoint_path=checkpoint_path,
            dataset_position=3200,
        )

        # Retrieve checkpoint
        recovery_checkpoint = tracker.get_recovery_checkpoint()
        assert recovery_checkpoint is not None
        assert recovery_checkpoint.checkpoint_step == 100


class TestAPIResponseRecovery:
    """Test API response persistence for recovery."""

    def test_record_api_response(self, temp_output_dir, sample_config):
        """Test recording API response for recovery."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        response = {
            "choices": [{"message": {"content": "Response text"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        tracker.record_api_response(
            prompt_hash="abc123def456",
            response=response,
            cost=0.001,
        )

        # Verify API response was recorded
        assert tracker.session.api_responses_saved == 1
        assert tracker.api_recovery_file.exists()

        # Verify content
        with open(tracker.api_recovery_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["prompt_hash"] == "abc123def456"
            assert data["cost"] == 0.001


class TestConnectionRecovery:
    """Test connection failure tracking."""

    def test_record_connection_failure(self, temp_output_dir, sample_config):
        """Test recording connection failure."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        tracker.record_connection_failure(
            retry_count=3,
            error="Connection timeout after 30s",
        )

        assert tracker.session.connection_failures == 1
        assert tracker.session.connection_retries == 3
        assert tracker.metrics_file.exists()

    def test_multiple_connection_failures(self, temp_output_dir, sample_config):
        """Test tracking multiple connection failures."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        tracker.record_connection_failure(retry_count=2, error="Error 1")
        tracker.record_connection_failure(retry_count=3, error="Error 2")
        tracker.record_connection_failure(retry_count=1, error="Error 3")

        assert tracker.session.connection_failures == 3
        assert tracker.session.connection_retries == 6


class TestProgressSummary:
    """Test progress summary generation."""

    def test_get_progress_summary(self, temp_output_dir, sample_config):
        """Test generating progress summary."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        # Record some metrics
        for step in range(1, 101):
            tracker.update_metrics(
                step=step,
                loss=3.0 - (step * 0.001),
                loss_components={"kd": (3.0 - step * 0.001) * 0.8},
                learning_rate=1e-4,
                samples_processed=step * 32,
                tokens_processed=step * 1024,
                throughput_samples_per_sec=32.0,
                throughput_tokens_per_sec=1024.0,
            )

        summary = tracker.get_progress_summary()

        assert summary["session_id"] == tracker.session.session_id
        assert summary["steps_completed"] == 100
        assert summary["total_steps_target"] == 10000
        assert summary["progress_pct"] == 1.0
        assert summary["samples_processed"] == 3200
        assert summary["elapsed_wall_clock_hours"] >= 0


class TestPauseTracking:
    """Test pause duration tracking."""

    def test_record_pause(self, temp_output_dir, sample_config):
        """Test recording pause duration."""
        tracker = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        tracker.record_pause(duration=120.0)

        assert tracker.session.total_pause_duration == 120.0
        assert len(tracker.session.pause_times) == 1


class TestStatePersistence:
    """Test state persistence to disk."""

    def test_save_and_load(self, temp_output_dir, sample_config):
        """Test saving and loading progress state."""
        tracker1 = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
        )

        # Update state
        for step in range(1, 51):
            tracker1.update_metrics(
                step=step,
                loss=3.0 - (step * 0.001),
                loss_components={"kd": (3.0 - step * 0.001) * 0.8},
                learning_rate=1e-4,
                samples_processed=step * 32,
                tokens_processed=step * 1024,
                throughput_samples_per_sec=32.0,
                throughput_tokens_per_sec=1024.0,
            )

        tracker1.record_checkpoint(
            step=50,
            checkpoint_path=temp_output_dir / "checkpoint_50.pt",
            dataset_position=1600,
        )
        tracker1.save()

        # Load state
        tracker2 = ProgressTracker(
            output_dir=temp_output_dir,
            config=sample_config,
            total_steps=10000,
            session_id=tracker1.session.session_id,
        )

        assert tracker2.session.steps_completed == 50
        assert tracker2.session.last_checkpoint_step == 50
        assert tracker2.session.samples_processed == 1600
        assert len(tracker2.session.checkpoints) == 1
