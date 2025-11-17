# Training Integration Examples

Complete examples of integrating ProgressTracker into training scripts.

## Basic Integration Pattern

### Minimal Integration

```python
from pathlib import Path
from training.progress_integration import training_progress_context, calculate_metrics_from_step

# In your training loop
with training_progress_context(config, output_dir, total_steps, is_main_process) as progress:
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(batch)
        loss = outputs["loss"]

        # Calculate metrics
        loss_value, loss_components, lr = calculate_metrics_from_step(
            loss,
            loss_dict={"kd": outputs.get("kd_loss"), "ce": outputs.get("ce_loss")},
            scheduler=scheduler
        )

        # Update metrics
        progress.update_metrics(
            step=step,
            loss=loss_value,
            loss_components=loss_components,
            learning_rate=lr,
            samples_processed=step * batch_size,
            tokens_processed=step * batch_size * seq_len,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # Save checkpoints
        if step % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{step}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            progress.record_checkpoint(
                step=step,
                checkpoint_path=checkpoint_path,
                dataset_position=dataloader.current_position,
            )
```

## Full Integration Example (distill_kd.py)

### Setup Phase

```python
from training.progress_integration import (
    setup_progress_tracking,
    get_recovery_checkpoint,
)

# In main() function after model creation
output_dir = Path(args.output_dir)

# Setup progress tracking
progress_ctx = setup_progress_tracking(
    config=cfg,
    output_dir=output_dir,
    total_steps=total_steps,
    session_id=None,  # Will create new session
    is_main_process=is_main_process,
)
```

### Resume Phase

```python
# Check for recovery checkpoint
recovery_result = get_recovery_checkpoint(output_dir)
if recovery_result and args.resume is None:
    checkpoint_path, dataset_position = recovery_result
    print(f"[main] Found recovery checkpoint: {checkpoint_path}")
    print(f"[main] Will resume from dataset position: {dataset_position}")
    args.resume = str(checkpoint_path)
    start_dataset_position = dataset_position
```

### Training Loop Integration

```python
# Enter progress tracking context
with progress_ctx as progress:
    for step in range(start_step, total_steps):
        # Get batch
        batch = next(dataloader)

        # Training step
        loss, loss_components = train_step(model, batch, optimizer, scaler, cfg, device)
        lr = scheduler.get_last_lr()[0]

        # Update metrics
        progress.update_metrics(
            step=step,
            loss=loss,
            loss_components=loss_components,
            learning_rate=lr,
            samples_processed=step * batch_size,
            tokens_processed=step * batch_size * seq_len,
        )

        # Save checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{step}.pt"
            save_checkpoint(model, optimizer, step, loss, checkpoint_path, cfg)

            progress.record_checkpoint(
                step=step,
                checkpoint_path=checkpoint_path,
                dataset_position=dataloader.current_index,
                is_best=(loss < best_loss),
            )

            if loss < best_loss:
                best_loss = loss
```

## Integration with Teacher API Calls

### Teacher Response Caching

```python
import hashlib
from training.teacher_cache import TeacherCache

# Initialize cache
teacher_cache = TeacherCache(
    cache_dir=output_dir / "api_cache",
    teacher_version="claude-3-opus-20240229",
)

# In training loop when using teacher responses
for step, batch in enumerate(dataloader):
    prompt = batch["prompt"]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    # Try cache first
    cached_response = teacher_cache.get(prompt)
    if cached_response:
        response = cached_response
        print(f"[cache] Hit for {prompt_hash[:8]}")
    else:
        # Call teacher API with retry logic
        retry_count = 0
        while retry_count < 5:
            try:
                response = teacher_api.generate(prompt)
                teacher_cache.put(prompt, response)

                # Record in progress tracker
                estimated_cost = calculate_api_cost(response)
                progress.record_api_response(
                    prompt_hash=prompt_hash,
                    response=response,
                    cost=estimated_cost,
                )
                break
            except ConnectionError as e:
                retry_count += 1
                progress.record_connection_failure(
                    retry_count=retry_count,
                    error=str(e),
                )

                if retry_count < 5:
                    delay = 2 ** retry_count + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    # Try to use cached response as fallback
                    fallback_response = teacher_cache.get_best_match(prompt)
                    if fallback_response:
                        response = fallback_response
                    else:
                        raise

    # Use response for training
    teacher_targets = process_response(response)
    loss = training_step_with_teacher(model, batch, teacher_targets)
```

## Integration with Distributed Training

### Multi-GPU Setup

```python
# In main() with distributed training
if args.local_rank >= 0:
    # Only main process tracks progress
    is_main = args.local_rank == 0
else:
    is_main = True

# Setup progress tracking
with training_progress_context(
    config=cfg,
    output_dir=output_dir,
    total_steps=total_steps,
    is_main_process=is_main,
) as progress:
    for step in range(start_step, total_steps):
        # Training step (same across all processes)
        loss = train_step(model, batch, optimizer)

        # Progress update only on main process
        if is_main:
            progress.update_metrics(
                step=step,
                loss=loss,
                loss_components={},
                learning_rate=scheduler.get_last_lr()[0],
                samples_processed=step * batch_size * world_size,
                tokens_processed=step * batch_size * seq_len * world_size,
            )
```

## Monitoring Progress During Training

### Periodic Progress Checks

```python
# In training loop (even without progress context)
if step % 100 == 0:
    progress_dir = output_dir / "progress"
    if progress_dir.exists():
        summary_file = progress_dir / "progress.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                print(f"Progress: {summary['progress_pct']:.1f}%")
                print(f"Loss: {summary['loss_trend']['current_loss']:.4f}")
                print(f"ETA: {summary['estimated_remaining_hours']:.1f}h")
```

### Dashboard Integration

```python
import json
from pathlib import Path

def get_training_dashboard(output_dir: Path) -> Dict[str, Any]:
    """Get current training dashboard data."""
    progress_file = output_dir / "progress" / "progress.json"

    if not progress_file.exists():
        return {"status": "no_progress_tracked"}

    with open(progress_file) as f:
        summary = json.load(f)

    return {
        "session_id": summary["session_id"],
        "progress": summary["progress_pct"],
        "steps": summary["steps_completed"],
        "loss": summary["loss_trend"]["current_loss"],
        "elapsed_hours": summary["elapsed_wall_clock_hours"],
        "estimated_remaining": summary["estimated_remaining_hours"],
        "api_cost": summary.get("api_responses_saved", 0),
        "checkpoints": summary["checkpoints_count"],
    }

# Use in monitoring script
dashboard = get_training_dashboard(output_dir)
print(json.dumps(dashboard, indent=2))
```

## Pause and Resume Workflow

### Pausing Training

```python
def pause_training(output_dir: Path, pause_duration: float):
    """Pause training and record duration."""
    progress_file = output_dir / "progress" / "session.json"

    if progress_file.exists():
        # Load existing tracker
        from training.progress_tracker import ProgressTracker
        tracker = ProgressTracker(
            output_dir=output_dir / "progress",
            config={},
            total_steps=0,
        )

        # Record pause
        tracker.record_pause(duration=pause_duration)
        tracker.save()

        print(f"[pause] Recorded pause duration: {pause_duration}s")
        print(f"[pause] Total pause time: {tracker.session.total_pause_duration/3600:.1f}h")

# In training script
if should_pause:
    pause_start = time.time()
    checkpoint_path = save_final_checkpoint()
    pause_duration = time.time() - pause_start
    pause_training(output_dir, pause_duration)
```

### Resuming Training

```python
def resume_training(output_dir: Path, config: Dict[str, Any]) -> Tuple[str, int]:
    """Resume training from last checkpoint."""
    from training.progress_integration import get_recovery_checkpoint

    recovery_result = get_recovery_checkpoint(output_dir)
    if not recovery_result:
        print("[resume] No recovery checkpoint found, starting fresh")
        return None, 0

    checkpoint_path, dataset_position = recovery_result

    print(f"[resume] Resuming from checkpoint: {checkpoint_path}")
    print(f"[resume] Dataset position: {dataset_position}")

    # Load trainer and set position
    trainer = create_trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    trainer.set_dataset_position(dataset_position)

    return checkpoint_path, dataset_position
```

## Error Handling and Recovery

### Graceful Connection Recovery

```python
from training.progress_integration import setup_progress_tracking
import time
import random

async def fetch_with_recovery(
    session: aiohttp.ClientSession,
    url: str,
    progress_ctx,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """Fetch with automatic retry and progress tracking."""
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=30) as response:
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Record failure
            progress_ctx.record_connection_failure(
                retry_count=attempt + 1,
                error=f"{type(e).__name__}: {str(e)}",
            )

            if attempt < max_retries - 1:
                # Exponential backoff
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"[retry] Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
            else:
                print(f"[retry] Max retries exceeded")
                raise

# Usage
with training_progress_context(config, output_dir, total_steps) as progress:
    for step, batch in enumerate(dataloader):
        try:
            response = await fetch_with_recovery(session, api_url, progress)
        except Exception as e:
            print(f"[error] Failed to fetch after retries: {e}")
            # Use fallback or skip batch
            continue
```

## Multi-Day Training Example

### Day 1: Initial Training

```bash
# Start training fresh
python -m training.distill_kd \
  --config configs/training.yaml \
  --output-dir training_output \
  --steps 50000

# After 8 hours, training paused
# Session ID: training_20251116_120000_abc12345
# Step reached: 25000
```

### Day 2: Resume Training

```bash
# Training automatically resumes from last checkpoint
# Session ID loaded from progress/session.json
python -m training.distill_kd \
  --config configs/training.yaml \
  --output-dir training_output

# OR explicitly specify checkpoint
python -m training.distill_kd \
  --config configs/training.yaml \
  --output-dir training_output \
  --resume training_output/progress/checkpoint_25000.pt
```

### Monitoring Multi-Day Progress

```python
# Check accumulated progress
import json
from pathlib import Path

output_dir = Path("training_output")
progress_file = output_dir / "progress" / "progress.json"

with open(progress_file) as f:
    summary = json.load(f)

print(f"Session: {summary['session_id']}")
print(f"Total elapsed: {summary['elapsed_wall_clock_hours']:.1f} hours")
print(f"Total pause time: {summary['total_pause_duration_hours']:.1f} hours")
print(f"Steps completed: {summary['steps_completed']}/{summary['total_steps_target']}")
print(f"API cost so far: ${estimate_cost(summary['api_responses_saved']):.2f}")
```
