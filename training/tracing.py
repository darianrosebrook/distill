"""
Training tracing and logging utilities.

Supports multiple backends:
- TensorBoard (recommended for visualization)
- WandB (optional, for cloud tracking)
- JSON logs (simple file-based logging)
- Console logging (always enabled)

Usage:
    from training.tracing import TrainingTracer
    
    tracer = TrainingTracer(
        run_name="worker_9b_kd",
        log_dir="runs/worker_9b",
        use_tensorboard=True,
        use_wandb=False,
    )
    
    # Log metrics
    tracer.log_metrics(step=100, metrics={"loss": 0.5, "kl_loss": 0.3})
    
    # Log hyperparameters
    tracer.log_hparams({"lr": 2e-4, "batch_size": 32})
    
    # Log model graph (once)
    tracer.log_model_graph(model, example_input)
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingTracer:
    """
    Unified training tracer supporting multiple backends.
    
    Features:
    - TensorBoard integration (local visualization)
    - WandB integration (cloud tracking, optional)
    - JSON log files (simple, always available)
    - Console logging (always enabled)
    - Hyperparameter tracking
    - Model graph logging
    - Metrics aggregation
    """
    
    def __init__(
        self,
        run_name: str,
        log_dir: Union[str, Path] = "runs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        json_log: bool = True,
        console_log: bool = True,
    ):
        """
        Initialize training tracer.
        
        Args:
            run_name: Name for this training run
            log_dir: Base directory for logs
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable WandB logging (requires wandb account)
            wandb_project: WandB project name
            wandb_entity: WandB entity/team name
            json_log: Enable JSON log files
            console_log: Enable console logging
        """
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.json_log = json_log
        self.console_log = console_log
        
        # Initialize backends
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                print(f"[TrainingTracer] TensorBoard logging enabled: {self.log_dir / 'tensorboard'}")
            except Exception as e:
                print(f"[TrainingTracer] WARN: TensorBoard initialization failed: {e}")
                self.use_tensorboard = False
        
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project or "kimi-student",
                    entity=wandb_entity,
                    name=run_name,
                    dir=str(self.log_dir),
                    config={},
                )
                print(f"[TrainingTracer] WandB logging enabled")
            except Exception as e:
                print(f"[TrainingTracer] WARN: WandB initialization failed: {e}")
                self.use_wandb = False
        
        # JSON log file
        self.json_log_path = None
        if self.json_log:
            self.json_log_path = self.log_dir / "metrics.jsonl"
            # Write header
            with open(self.json_log_path, 'w') as f:
                f.write(f"# Training metrics log for {run_name}\n")
                f.write(f"# Started at {datetime.now().isoformat()}\n")
        
        # Metrics history (for aggregation)
        self.metrics_history: Dict[str, list] = {}
        self.start_time = time.time()
        
        print(f"[TrainingTracer] Initialized: {run_name}")
        print(f"[TrainingTracer] Log directory: {self.log_dir}")
    
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "",
    ):
        """
        Log training metrics.
        
        Args:
            step: Training step number
            metrics: Dictionary of metric name -> value
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """
        timestamp = time.time() - self.start_time
        
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        
        # TensorBoard
        if self.tb_writer:
            for name, value in prefixed_metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        # WandB
        if self.use_wandb:
            wandb.log({**prefixed_metrics, "step": step, "timestamp": timestamp}, step=step)
        
        # JSON log
        if self.json_log_path:
            log_entry = {
                "step": step,
                "timestamp": timestamp,
                **prefixed_metrics,
            }
            with open(self.json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Console log
        if self.console_log:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in prefixed_metrics.items()])
            print(f"[TrainingTracer] Step {step}: {metric_str}")
        
        # Update history
        for name, value in prefixed_metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append((step, value))
    
    def log_hparams(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameter name -> value
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, {})
        
        # WandB
        if self.use_wandb:
            wandb.config.update(hparams)
        
        # JSON log
        if self.json_log_path:
            hparams_path = self.log_dir / "hparams.json"
            with open(hparams_path, 'w') as f:
                json.dump(hparams, f, indent=2)
        
        # Console log
        if self.console_log:
            hparams_str = ", ".join([f"{k}={v}" for k, v in hparams.items()])
            print(f"[TrainingTracer] Hyperparameters: {hparams_str}")
    
    def log_model_graph(self, model, example_input, verbose: bool = False):
        """
        Log model graph structure.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor(s)
            verbose: Whether to print graph details
        """
        # TensorBoard
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, example_input, verbose=verbose)
                print(f"[TrainingTracer] Model graph logged to TensorBoard")
            except Exception as e:
                print(f"[TrainingTracer] WARN: Failed to log model graph: {e}")
        
        # WandB
        if self.use_wandb:
            try:
                wandb.watch(model, log="all", log_freq=1000)
            except Exception as e:
                print(f"[TrainingTracer] WARN: Failed to watch model in WandB: {e}")
    
    def log_text(self, step: int, tag: str, text: str):
        """
        Log text (e.g., sample outputs, errors).
        
        Args:
            step: Training step
            tag: Tag for the text (e.g., "sample_output")
            text: Text content
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        # WandB
        if self.use_wandb:
            wandb.log({tag: wandb.Html(text)}, step=step)
        
        # JSON log
        if self.json_log_path:
            text_log_path = self.log_dir / "text_logs.jsonl"
            with open(text_log_path, 'a') as f:
                json.dump({"step": step, "tag": tag, "text": text}, f)
                f.write('\n')
    
    def log_image(self, step: int, tag: str, image, **kwargs):
        """
        Log image (e.g., attention visualizations).
        
        Args:
            step: Training step
            tag: Tag for the image
            image: Image tensor or numpy array
            **kwargs: Additional arguments for add_image
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_image(tag, image, step, **kwargs)
        
        # WandB
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, step: int, tag: str, values, bins: int = 30):
        """
        Log histogram of values (e.g., gradient norms, weight distributions).
        
        Args:
            step: Training step
            tag: Tag for the histogram
            values: Values to histogram
            bins: Number of bins
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step, bins=bins)
        
        # WandB
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all logged metrics.
        
        Returns:
            Dictionary mapping metric name -> {mean, min, max, latest}
        """
        summary = {}
        for name, history in self.metrics_history.items():
            if history:
                values = [v for _, v in history]
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "steps": len(values),
                }
        return summary
    
    def save_summary(self):
        """Save metrics summary to file."""
        summary = self.get_metrics_summary()
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[TrainingTracer] Summary saved to {summary_path}")
    
    def close(self):
        """Close all loggers and save final state."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        self.save_summary()
        
        elapsed = time.time() - self.start_time
        print(f"[TrainingTracer] Closed. Total time: {elapsed/3600:.2f} hours")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_tracer_from_config(
    cfg: Dict[str, Any],
    run_name: Optional[str] = None,
) -> TrainingTracer:
    """
    Create TrainingTracer from config dictionary.
    
    Args:
        cfg: Configuration dictionary
        run_name: Optional run name (defaults to timestamp-based)
    
    Returns:
        Configured TrainingTracer instance
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"
    
    tracing_cfg = cfg.get("tracing", {})
    
    return TrainingTracer(
        run_name=run_name,
        log_dir=tracing_cfg.get("log_dir", "runs"),
        use_tensorboard=tracing_cfg.get("use_tensorboard", True),
        use_wandb=tracing_cfg.get("use_wandb", False),
        wandb_project=tracing_cfg.get("wandb_project"),
        wandb_entity=tracing_cfg.get("wandb_entity"),
        json_log=tracing_cfg.get("json_log", True),
        console_log=tracing_cfg.get("console_log", True),
    )


