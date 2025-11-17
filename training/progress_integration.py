# training/progress_integration.py
"""
Integration module for ProgressTracker with training loop.

Provides context managers and utilities to integrate progress tracking
into existing training scripts without modifying core training logic.

Key Components:
- TrainingProgressContext: Context manager for progress tracking
- get_training_metrics: Extract metrics from training step
- setup_progress_tracking: Initialize tracker with config
- resume_from_checkpoint: Handle checkpoint recovery with tracker

@author: @darianrosebrook
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
import torch

from training.progress_tracker import ProgressTracker


class TrainingProgressContext:
    """Context manager for training progress tracking."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        total_steps: int,
        session_id: Optional[str] = None,
        is_main_process: bool = True,
    ):
        """
        Initialize training progress context.

        Args:
            config: Training configuration
            output_dir: Checkpoint output directory
            total_steps: Total training steps
            session_id: Optional session ID for resuming
            is_main_process: Whether this is main process (for distributed training)
        """
        self.config = config
        # Ensure output_dir is a Path object - handle both strings and Path objects
        if isinstance(output_dir, (str, bytes)):
            self.output_dir = Path(output_dir)
        elif isinstance(output_dir, Path):
            self.output_dir = output_dir
        else:
            # Try to convert to string first, then to Path
            self.output_dir = Path(str(output_dir))
        self.total_steps = total_steps
        self.session_id = session_id
        self.is_main_process = is_main_process

        self.tracker: Optional[ProgressTracker] = None
        self.checkpoint_counter = 0
        self.metrics_update_interval = config.get("progress", {}).get(
            "metrics_update_interval", 1
        )
        self.checkpoint_interval = config.get("progress", {}).get(
            "checkpoint_interval", 1000
        )

        # Throughput calculation
        self.step_times = []
        self.last_step_time = time.time()
        self.last_samples_count = 0
        self.last_tokens_count = 0

    def __enter__(self) -> "TrainingProgressContext":
        """Initialize tracker on context entry."""
        if self.is_main_process:
            self.tracker = ProgressTracker(
                output_dir=self.output_dir / "progress",
                config=self.config,
                total_steps=self.total_steps,
                session_id=self.session_id,
            )
            print(
                f"[progress_integration] Progress tracking initialized: "
                f"Session={self.tracker.session.session_id}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize tracker on context exit."""
        if self.tracker:
            self.tracker.close()
            if self.is_main_process:
                print("[progress_integration] Progress tracking finalized")

    def update_metrics(
        self,
        step: int,
        loss: float,
        loss_components: Dict[str, float],
        learning_rate: float,
        samples_processed: int,
        tokens_processed: int,
    ) -> None:
        """
        Update training metrics.

        Args:
            step: Current training step
            loss: Total loss value
            loss_components: Dict of loss component values
            learning_rate: Current learning rate
            samples_processed: Total samples processed so far
            tokens_processed: Total tokens processed so far
        """
        if not self.tracker or not self.is_main_process:
            return

        # Calculate throughput
        current_time = time.time()
        elapsed = current_time - self.last_step_time
        samples_delta = samples_processed - self.last_samples_count
        tokens_delta = tokens_processed - self.last_tokens_count

        if elapsed > 0 and samples_delta > 0:
            throughput_samples = samples_delta / elapsed
            throughput_tokens = tokens_delta / elapsed
        else:
            throughput_samples = 0.0
            throughput_tokens = 0.0

        # Update tracker
        self.tracker.update_metrics(
            step=step,
            loss=loss,
            loss_components=loss_components,
            learning_rate=learning_rate,
            samples_processed=samples_processed,
            tokens_processed=tokens_processed,
            throughput_samples_per_sec=throughput_samples,
            throughput_tokens_per_sec=throughput_tokens,
        )

        # Update state for next calculation
        self.last_step_time = current_time
        self.last_samples_count = samples_processed
        self.last_tokens_count = tokens_processed

        # Log progress periodically
        if step % 100 == 0:
            summary = self.tracker.get_progress_summary()
            print(
                f"[progress_integration] Step {summary['steps_completed']}/{summary['total_steps_target']} "
                f"({summary['progress_pct']:.1f}%) | "
                f"Loss: {summary['loss_trend']['current_loss']:.4f} | "
                f"ETA: {summary['estimated_remaining_hours']:.1f}h"
            )

    def record_checkpoint(
        self,
        step: int,
        checkpoint_path: Path,
        dataset_position: int,
        is_best: bool = False,
    ) -> None:
        """
        Record checkpoint metadata.

        Args:
            step: Training step number
            checkpoint_path: Path to checkpoint file
            dataset_position: Current position in dataset
            is_best: Whether this is best checkpoint so far
        """
        if not self.tracker or not self.is_main_process:
            return

        recovery_tags = []
        if step % self.checkpoint_interval == 0:
            recovery_tags.append("periodic")
        if is_best:
            recovery_tags.append("best_loss")

        self.tracker.record_checkpoint(
            step=step,
            checkpoint_path=Path(checkpoint_path),
            dataset_position=dataset_position,
            recovery_tags=recovery_tags,
        )

        self.checkpoint_counter += 1

    def record_api_response(
        self,
        prompt_hash: str,
        response: Dict[str, Any],
        cost: float = 0.0,
    ) -> None:
        """Record API response for recovery."""
        if not self.tracker or not self.is_main_process:
            return

        self.tracker.record_api_response(
            prompt_hash=prompt_hash,
            response=response,
            cost=cost,
        )

    def record_connection_failure(self, retry_count: int, error: str) -> None:
        """Record connection failure."""
        if not self.tracker or not self.is_main_process:
            return

        self.tracker.record_connection_failure(
            retry_count=retry_count, error=error)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        if not self.tracker:
            return {"status": "not_initialized"}

        return self.tracker.get_progress_summary()


def setup_progress_tracking(
    config: Dict[str, Any],
    output_dir: Path,
    total_steps: int,
    session_id: Optional[str] = None,
    is_main_process: bool = True,
) -> TrainingProgressContext:
    """
    Setup progress tracking for training.

    Args:
        config: Training configuration
        output_dir: Checkpoint output directory
        total_steps: Total training steps
        session_id: Optional session ID for resuming
        is_main_process: Whether this is main process

    Returns:
        TrainingProgressContext to use in with statement
    """
    return TrainingProgressContext(
        config=config,
        output_dir=output_dir,
        total_steps=total_steps,
        session_id=session_id,
        is_main_process=is_main_process,
    )


def get_recovery_checkpoint(
    output_dir: Path,
    session_id: Optional[str] = None,
) -> Optional[Tuple[Path, int]]:
    """
    Get recovery checkpoint path and dataset position.

    Args:
        output_dir: Checkpoint output directory
        session_id: Optional session ID

    Returns:
        Tuple of (checkpoint_path, dataset_position) or None
    """
    progress_dir = output_dir / "progress"
    if not progress_dir.exists():
        return None

    try:
        # Try loading existing session if available
        session_file = progress_dir / "session.json"
        if session_file.exists():
            import json

            with open(session_file) as f:
                session_data = json.load(f)

            # Get last checkpoint step
            last_checkpoint_step = session_data.get("last_checkpoint_step")
            if last_checkpoint_step and "checkpoints" in session_data:
                checkpoint_key = str(last_checkpoint_step)
                if checkpoint_key in session_data["checkpoints"]:
                    checkpoint_meta = session_data["checkpoints"][checkpoint_key]
                    return (
                        Path(checkpoint_meta["checkpoint_path"]),
                        checkpoint_meta["dataset_position"],
                    )

        return None
    except Exception as e:
        print(f"[progress_integration] Failed to get recovery checkpoint: {e}")

    return None


def calculate_metrics_from_step(
    loss: torch.Tensor,
    loss_dict: Optional[Dict[str, torch.Tensor]] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, Dict[str, float], float]:
    """
    Extract metrics from training step.

    Args:
        loss: Total loss tensor
        loss_dict: Dict of loss components
        scheduler: Learning rate scheduler

    Returns:
        Tuple of (loss_value, loss_components_dict, learning_rate)
    """
    # Convert tensor to scalar
    loss_value = loss.item() if torch.is_tensor(loss) else loss

    # Extract loss components
    loss_components = {}
    if loss_dict:
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                loss_components[key] = value.item()
            else:
                loss_components[key] = float(value)
    else:
        loss_components = {"total": loss_value}

    # Get current learning rate
    if scheduler:
        learning_rate = scheduler.get_last_lr()[0]
    else:
        learning_rate = 0.0

    return loss_value, loss_components, learning_rate


@contextmanager
def training_progress_context(
    config: Dict[str, Any],
    output_dir: Path,
    total_steps: int,
    is_main_process: bool = True,
):
    """
    Convenience context manager for progress tracking.

    Usage:
        with training_progress_context(config, output_dir, total_steps) as progress:
            for step, batch in enumerate(dataloader):
                # training step
                loss, loss_components, lr = calculate_metrics_from_step(
                    loss, loss_dict, scheduler
                )
                progress.update_metrics(
                    step=step,
                    loss=loss,
                    loss_components=loss_components,
                    learning_rate=lr,
                    samples_processed=samples_so_far,
                    tokens_processed=tokens_so_far,
                )
    """
    ctx = setup_progress_tracking(
        config=config,
        output_dir=output_dir,
        total_steps=total_steps,
        is_main_process=is_main_process,
    )
    with ctx:
        yield ctx
