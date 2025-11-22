# training/progress_tracker.py
"""
Comprehensive training progress tracking and recovery system.

Tracks:
- Multi-day training progress with persistent state
- API response persistence and recovery
- Training metrics and loss trends
- Checkpoint status and recovery points
- Budget and cost tracking
- Connection recovery and retry logic

Integrates with existing infrastructure:
- TeacherCache for API response caching
- CheckpointManager for progress recovery
- safe_checkpoint_loading for state restoration

@author: @darianrosebrook
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
import logging
from collections import deque


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[training.progress_tracker] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of training metrics at a checkpoint."""

    step: int
    timestamp: float
    wall_clock_hours: float
    loss: float
    loss_components: Dict[str, float]
    learning_rate: float
    samples_processed: int
    tokens_processed: int
    throughput_samples_per_sec: float
    throughput_tokens_per_sec: float

    # Loss trend tracking
    loss_history: List[float] = field(default_factory=list)  # Last 100 steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsSnapshot':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint for recovery purposes."""

    checkpoint_step: int
    checkpoint_path: Path
    created_at: float
    metrics: MetricsSnapshot
    # Total samples processed (for telemetry/metrics only, not for seeking)
    samples_seen: int
    # Dataset fingerprint for validation
    dataset_fingerprint: Optional[str] = None
    dataset_len: Optional[int] = None  # Dataset length when checkpoint saved
    recovery_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'checkpoint_step': self.checkpoint_step,
            'checkpoint_path': str(self.checkpoint_path),
            'created_at': self.created_at,
            'metrics': self.metrics.to_dict(),
            'samples_seen': self.samples_seen,
            'recovery_tags': self.recovery_tags,
        }
        if self.dataset_fingerprint is not None:
            d['dataset_fingerprint'] = self.dataset_fingerprint
        if self.dataset_len is not None:
            d['dataset_len'] = self.dataset_len
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        # Make a copy to avoid mutating the original
        data = dict(data)
        
        metrics_data = data.pop('metrics')
        metrics = MetricsSnapshot.from_dict(metrics_data)
        data['checkpoint_path'] = Path(data['checkpoint_path'])
        data['metrics'] = metrics
        
        # Backward compatibility: handle old 'dataset_position' field
        if 'dataset_position' in data and 'samples_seen' not in data:
            data['samples_seen'] = data.pop('dataset_position')
        
        # Remove any fields that don't exist in the dataclass
        # (to avoid __init__ errors)
        valid_fields = {
            'checkpoint_step', 'checkpoint_path', 'created_at', 'metrics',
            'samples_seen', 'dataset_fingerprint', 'dataset_len', 'recovery_tags'
        }
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)


@dataclass
class TrainingSession:
    """Complete training session metadata for multi-day tracking."""

    session_id: str  # Unique ID for this training run
    start_time: float
    config_hash: str  # Hash of training config

    total_steps_target: int
    steps_completed: int = 0
    samples_processed: int = 0
    tokens_processed: int = 0

    # Checkpoint tracking
    checkpoints: Dict[int, CheckpointMetadata] = field(default_factory=dict)
    last_checkpoint_step: int = 0

    # Metrics tracking
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Loss tracking for trend analysis
    moving_avg_loss: float = 0.0
    loss_moving_window: int = 100

    # API response recovery
    api_responses_saved: int = 0
    api_responses_recovered: int = 0

    # Wall-clock tracking
    pause_times: List[float] = field(
        default_factory=list)  # Pause duration timestamps
    total_pause_duration: float = 0.0

    # Connection recovery
    connection_retries: int = 0
    connection_failures: int = 0

    def get_elapsed_wall_clock(self) -> float:
        """Get elapsed wall clock time in hours, excluding pauses."""
        total_elapsed = time.time() - self.start_time
        return (total_elapsed - self.total_pause_duration) / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'config_hash': self.config_hash,
            'total_steps_target': self.total_steps_target,
            'steps_completed': self.steps_completed,
            'samples_processed': self.samples_processed,
            'tokens_processed': self.tokens_processed,
            'last_checkpoint_step': self.last_checkpoint_step,
            'checkpoints': {
                k: v.to_dict() for k, v in self.checkpoints.items()
            },
            'api_responses_saved': self.api_responses_saved,
            'api_responses_recovered': self.api_responses_recovered,
            'total_pause_duration': self.total_pause_duration,
            'connection_retries': self.connection_retries,
            'connection_failures': self.connection_failures,
            'moving_avg_loss': self.moving_avg_loss,
            'elapsed_wall_clock_hours': self.get_elapsed_wall_clock(),
        }


class ProgressTracker:
    """
    Comprehensive training progress tracker with recovery capabilities.

    Responsibilities:
    - Track training progress across multiple days
    - Persist API responses for recovery
    - Manage checkpoint metadata
    - Provide recovery information
    - Track loss trends and training stability
    """

    def __init__(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        total_steps: int,
        session_id: Optional[str] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            output_dir: Directory to save progress and checkpoint metadata
            config: Training configuration (for hashing and validation)
            total_steps: Total training steps target
            session_id: Optional session ID. If not provided, generates one.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id()

        # Compute config hash for validation
        config_hash = self._compute_config_hash(config)

        # Load or create session
        self.session_file = self.output_dir / "session.json"
        self.session = self._load_or_create_session(
            session_id=session_id,
            config_hash=config_hash,
            total_steps=total_steps,
        )

        # Paths for tracking
        self.progress_file = self.output_dir / "progress.json"
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.api_recovery_file = self.output_dir / "api_recovery.jsonl"
        self.checkpoint_manifest = self.output_dir / "checkpoints.json"

        # Lock for thread-safe updates
        self._lock = threading.RLock()

        # Auto-save interval (every 5 minutes)
        self.auto_save_interval = 300.0
        self.last_auto_save = time.time()

    def _generate_session_id(self) -> str:
        """Generate unique session ID with timestamp and random suffix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(
            f"{timestamp}_{time.time()}".encode()
        ).hexdigest()[:8]
        return f"training_{timestamp}_{random_suffix}"

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 hash of config for validation."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _load_or_create_session(
        self,
        session_id: str,
        config_hash: str,
        total_steps: int,
    ) -> TrainingSession:
        """Load existing session or create new one."""
        if self.session_file.exists():
            try:
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)

                # Validate config hasn't changed
                if session_data.get('config_hash') != config_hash:
                    logger.warning(
                        "Config hash mismatch. "
                        "Previous: %s, Current: %s. "
                        "Creating new session.",
                        session_data.get('config_hash')[:8],
                        config_hash[:8],
                    )
                    # Create new session
                    return self._create_new_session(session_id, config_hash, total_steps)

                # Validate target steps haven't changed drastically
                if abs(session_data.get('total_steps_target', 0) - total_steps) > 100:
                    logger.warning(
                        "Target steps changed. "
                        "Previous: %d, Current: %d. "
                        "Creating new session.",
                        session_data.get('total_steps_target', 0),
                        total_steps,
                    )
                    return self._create_new_session(session_id, config_hash, total_steps)

                # Load session data
                logger.info(
                    "Resuming training session: %s (Step %d/%d)",
                    session_id,
                    session_data.get('steps_completed', 0),
                    total_steps,
                )

                # Reconstruct session from dict
                session = TrainingSession(
                    session_id=session_data['session_id'],
                    start_time=session_data['start_time'],
                    config_hash=session_data['config_hash'],
                    total_steps_target=session_data['total_steps_target'],
                    steps_completed=session_data['steps_completed'],
                    samples_processed=session_data['samples_processed'],
                    tokens_processed=session_data['tokens_processed'],
                    api_responses_saved=session_data.get(
                        'api_responses_saved', 0),
                    api_responses_recovered=session_data.get(
                        'api_responses_recovered', 0),
                    total_pause_duration=session_data.get(
                        'total_pause_duration', 0.0),
                )

                # Restore checkpoints if available
                if 'checkpoints' in session_data:
                    for step_str, checkpoint_dict in session_data['checkpoints'].items():
                        step = int(step_str)
                        checkpoint = CheckpointMetadata.from_dict(
                            checkpoint_dict)
                        session.checkpoints[step] = checkpoint
                    session.last_checkpoint_step = session_data.get(
                        'last_checkpoint_step', 0
                    )

                return session

            except Exception as e:
                logger.error(
                    "Failed to load session: %s. Creating new session.", e)
                return self._create_new_session(session_id, config_hash, total_steps)

        else:
            return self._create_new_session(session_id, config_hash, total_steps)

    def _create_new_session(
        self,
        session_id: str,
        config_hash: str,
        total_steps: int,
    ) -> TrainingSession:
        """Create brand new training session."""
        logger.info(
            "Starting new training session: %s (Target: %d steps)",
            session_id,
            total_steps,
        )
        return TrainingSession(
            session_id=session_id,
            start_time=time.time(),
            config_hash=config_hash,
            total_steps_target=total_steps,
        )

    def update_metrics(
        self,
        step: int,
        loss: float,
        loss_components: Dict[str, float],
        learning_rate: float,
        samples_processed: int,
        tokens_processed: int,
        throughput_samples_per_sec: float,
        throughput_tokens_per_sec: float,
    ) -> None:
        """Update training metrics snapshot."""
        with self._lock:
            wall_clock_hours = self.session.get_elapsed_wall_clock()

            # Create metrics snapshot
            metrics = MetricsSnapshot(
                step=step,
                timestamp=time.time(),
                wall_clock_hours=wall_clock_hours,
                loss=loss,
                loss_components=loss_components,
                learning_rate=learning_rate,
                samples_processed=samples_processed,
                tokens_processed=tokens_processed,
                throughput_samples_per_sec=throughput_samples_per_sec,
                throughput_tokens_per_sec=throughput_tokens_per_sec,
            )

            # Update session counters
            self.session.steps_completed = step
            self.session.samples_processed = samples_processed
            self.tokens_processed = tokens_processed

            # Update moving average loss
            self._update_moving_avg_loss(loss)

            # Store in history
            self.session.metrics_history.append(metrics.to_dict())

            # Save metrics line to JSONL
            self._save_metrics_line(metrics)

            # Auto-save if interval elapsed
            if time.time() - self.last_auto_save > self.auto_save_interval:
                self.save()
                self.last_auto_save = time.time()

    def _update_moving_avg_loss(self, loss: float) -> None:
        """Update exponential moving average of loss."""
        alpha = 2.0 / (self.session.loss_moving_window + 1)
        if self.session.moving_avg_loss == 0.0:
            self.session.moving_avg_loss = loss
        else:
            self.session.moving_avg_loss = (
                alpha * loss + (1 - alpha) * self.session.moving_avg_loss
            )

    def _save_metrics_line(self, metrics: MetricsSnapshot) -> None:
        """Append metrics snapshot to JSONL file."""
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to save metrics line: %s", e)

    def record_checkpoint(
        self,
        step: int,
        checkpoint_path: Path,
        samples_seen: int,
        recovery_tags: Optional[List[str]] = None,
        dataset_fingerprint: Optional[str] = None,
        dataset_len: Optional[int] = None,
    ) -> None:
        """
        Record checkpoint metadata for recovery.

        Args:
            step: Training step number
            checkpoint_path: Path to checkpoint file
            samples_seen: Total samples processed so far (for telemetry/metrics only)
            recovery_tags: Optional tags (e.g., "best_loss", "every_1k_steps")
            dataset_fingerprint: Dataset fingerprint for validation (optional)
            dataset_len: Dataset length when checkpoint saved (optional)
        """
        with self._lock:
            # Get latest metrics
            if not self.session.metrics_history:
                logger.warning(
                    "No metrics recorded yet, skipping checkpoint metadata")
                return

            latest_metrics_dict = self.session.metrics_history[-1]
            latest_metrics = MetricsSnapshot.from_dict(latest_metrics_dict)

            # Create checkpoint metadata
            checkpoint_meta = CheckpointMetadata(
                checkpoint_step=step,
                checkpoint_path=checkpoint_path,
                created_at=time.time(),
                metrics=latest_metrics,
                samples_seen=samples_seen,
                dataset_fingerprint=dataset_fingerprint,
                dataset_len=dataset_len,
                recovery_tags=recovery_tags or [],
            )

            # Store in session
            self.session.checkpoints[step] = checkpoint_meta
            self.session.last_checkpoint_step = step

            # Save checkpoint manifest
            self._save_checkpoint_manifest()

            logger.info(
                "Checkpoint recorded: Step %d, Loss: %.4f, Wall-clock: %.1f hours",
                step,
                latest_metrics.loss,
                latest_metrics.wall_clock_hours,
            )

    def _save_checkpoint_manifest(self) -> None:
        """Save checkpoint manifest for recovery reference."""
        try:
            manifest = {
                'last_checkpoint_step': self.session.last_checkpoint_step,
                'checkpoints': {
                    str(k): v.to_dict() for k, v in self.session.checkpoints.items()
                },
                'saved_at': time.time(),
            }
            with open(self.checkpoint_manifest, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error("Failed to save checkpoint manifest: %s", e)

    def record_api_response(
        self,
        prompt_hash: str,
        response: Dict[str, Any],
        cost: float = 0.0,
    ) -> None:
        """
        Record API response for recovery purposes.

        Args:
            prompt_hash: SHA256 hash of prompt
            response: Full API response
            cost: Cost of this API call
        """
        with self._lock:
            try:
                recovery_entry = {
                    'timestamp': time.time(),
                    'prompt_hash': prompt_hash,
                    'response': response,
                    'cost': cost,
                }
                with open(self.api_recovery_file, "a") as f:
                    f.write(json.dumps(recovery_entry) + "\n")

                self.session.api_responses_saved += 1
            except Exception as e:
                logger.error("Failed to save API response for recovery: %s", e)

    def record_connection_failure(self, retry_count: int, error: str) -> None:
        """Record connection failure for monitoring."""
        with self._lock:
            self.session.connection_failures += 1
            self.session.connection_retries += retry_count

            failure_entry = {
                'timestamp': time.time(),
                'retry_count': retry_count,
                'error': error,
                'total_failures': self.session.connection_failures,
            }

            # Append to metrics file as metadata
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps({
                        'type': 'connection_failure',
                        **failure_entry
                    }) + "\n")
            except Exception as e:
                logger.error("Failed to record connection failure: %s", e)

    def record_pause(self, duration: float) -> None:
        """Record pause duration for wall-clock time calculations."""
        with self._lock:
            self.session.pause_times.append(duration)
            self.session.total_pause_duration += duration

            logger.info(
                "Training paused for %.1f seconds. "
                "Total pause time: %.1f hours",
                duration,
                self.session.total_pause_duration / 3600.0,
            )

    def get_recovery_checkpoint(self) -> Optional[CheckpointMetadata]:
        """
        Get most recent checkpoint for recovery.

        Returns:
            Checkpoint metadata for recovery, or None if no checkpoints.
        """
        with self._lock:
            if not self.session.checkpoints:
                return None

            # Return most recent checkpoint
            latest_step = self.session.last_checkpoint_step
            return self.session.checkpoints.get(latest_step)

    def get_loss_trend(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Get loss trend information.

        Returns:
            Dictionary with loss statistics and trend
        """
        with self._lock:
            if len(self.session.metrics_history) == 0:
                return {
                    'has_data': False,
                    'message': 'No metrics recorded yet'
                }

            losses = [m['loss']
                      for m in list(self.session.metrics_history)[-window_size:]]

            return {
                'has_data': True,
                'current_loss': losses[-1],
                'moving_avg_loss': self.session.moving_avg_loss,
                'min_loss': min(losses),
                'max_loss': max(losses),
                'window_size': len(losses),
                'is_improving': losses[-1] < self.session.moving_avg_loss,
            }

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary for logging."""
        with self._lock:
            elapsed = self.session.get_elapsed_wall_clock()
            progress_pct = (
                100.0 * self.session.steps_completed / self.session.total_steps_target
            ) if self.session.total_steps_target > 0 else 0.0

            # Estimate remaining time
            if self.session.steps_completed > 0 and elapsed > 0:
                steps_per_hour = self.session.steps_completed / elapsed
                remaining_steps = self.session.total_steps_target - self.session.steps_completed
                est_remaining_hours = (
                    remaining_steps / steps_per_hour if steps_per_hour > 0 else 0
                )
            else:
                est_remaining_hours = 0

            loss_trend = self.get_loss_trend()

            return {
                'session_id': self.session.session_id,
                'progress_pct': progress_pct,
                'steps_completed': self.session.steps_completed,
                'total_steps_target': self.session.total_steps_target,
                'samples_processed': self.session.samples_processed,
                'tokens_processed': self.session.tokens_processed,
                'elapsed_wall_clock_hours': elapsed,
                'estimated_remaining_hours': est_remaining_hours,
                'total_pause_duration_hours': self.session.total_pause_duration / 3600.0,
                'api_responses_saved': self.session.api_responses_saved,
                'connection_failures': self.session.connection_failures,
                'connection_retries': self.session.connection_retries,
                'loss_trend': loss_trend,
                'last_checkpoint_step': self.session.last_checkpoint_step,
                'checkpoints_count': len(self.session.checkpoints),
            }

    def save(self) -> None:
        """Persist session state to disk."""
        with self._lock:
            try:
                # Save session
                with open(self.session_file, "w") as f:
                    json.dump(self.session.to_dict(), f, indent=2)

                # Save progress summary
                with open(self.progress_file, "w") as f:
                    json.dump(self.get_progress_summary(), f, indent=2)

            except Exception as e:
                logger.error("Failed to save progress tracker state: %s", e)

    def close(self) -> None:
        """Finalize progress tracking (flush, save, etc)."""
        self.save()
        logger.info(
            "Progress tracker finalized for session: %s",
            self.session.session_id,
        )
