# Training Recovery and Progress Tracking Guide

## Overview

This guide describes the comprehensive training progress tracking and recovery system that ensures:

1. **API Response Persistence**: All teacher model responses are cached for recovery if training is interrupted
2. **Multi-Day Training Continuation**: Training can be paused and resumed across multiple days with full state restoration
3. **Connection Recovery**: Automatic retry logic and failure tracking for API communication
4. **Progress Monitoring**: Real-time metrics and loss trend analysis
5. **Checkpoint Management**: Automatic checkpoint metadata recording for recovery reference

## Architecture

### Components

#### 1. ProgressTracker (`training/progress_tracker.py`)

Main component that manages all progress tracking:

- **Session Management**: Unique session ID per training run with config validation
- **Metrics Tracking**: Real-time metrics (loss, throughput, learning rate)
- **Checkpoint Recording**: Metadata for all checkpoints for recovery
- **API Response Logging**: JSONL-based persistent storage of API responses
- **Connection Failure Tracking**: Records failures and retry attempts

#### 2. TeacherCache (`training/teacher_cache.py`)

Existing system for caching teacher API responses:

- SHA-256 prompt hashing for deterministic caching
- Version compatibility checking
- Cache hit/miss tracking
- Integrated with ProgressTracker for recovery

#### 3. Checkpoint Manager (`scripts/make_kd_mix_hardened.py`)

Existing system for checkpointing progress:

- JSONL-based result storage (one line per sample)
- Progress JSON with completed indices
- Budget tracking and cost monitoring
- Integrated with ProgressTracker

## Usage in Training

### Initialization

```python
from training.progress_tracker import ProgressTracker
from training.distill_kd import main

# Create progress tracker
tracker = ProgressTracker(
    output_dir=Path("training_progress"),
    config=training_config,
    total_steps=10000,
)

# Optional: Resume from specific session
# tracker = ProgressTracker(
#     output_dir=Path("training_progress"),
#     config=training_config,
#     total_steps=10000,
#     session_id="training_20251116_120000_abc12345"
# )
```

### Recording Metrics

```python
# During training loop
for step, batch in enumerate(dataloader):
    # ... training step ...

    tracker.update_metrics(
        step=step,
        loss=loss_value,
        loss_components={
            "kd": kd_loss,
            "ce": ce_loss,
            "intermediate": intermediate_loss,
        },
        learning_rate=current_lr,
        samples_processed=step * batch_size,
        tokens_processed=step * batch_size * seq_len,
        throughput_samples_per_sec=current_throughput_samples,
        throughput_tokens_per_sec=current_throughput_tokens,
    )
```

### Recording Checkpoints

```python
# When saving checkpoint
torch.save(model.state_dict(), checkpoint_path)

tracker.record_checkpoint(
    step=step,
    checkpoint_path=checkpoint_path,
    dataset_position=current_sample_index,
    recovery_tags=["every_1k_steps", "best_loss"] if is_best else ["every_1k_steps"],
)
```

### API Response Recovery

```python
# When receiving teacher response
response = teacher_model.generate(prompt)

# Cache response with SHA256 hashing
prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
tracker.record_api_response(
    prompt_hash=prompt_hash,
    response=response,
    cost=estimated_api_cost,
)

# On connection failure, retry with exponential backoff
try:
    response = teacher_model.generate(prompt)
except ConnectionError as e:
    tracker.record_connection_failure(
        retry_count=attempt,
        error=str(e),
    )
    # Implement exponential backoff
    time.sleep(2 ** attempt)
```

### Pause/Resume

```python
# Before pausing training
pause_start = time.time()
tracker.record_pause(duration=time.time() - pause_start)
tracker.save()

# When resuming
tracker = ProgressTracker(
    output_dir=Path("training_progress"),
    config=training_config,
    total_steps=10000,
    session_id="training_20251116_120000_abc12345",  # Load previous session
)

# Get recovery checkpoint
recovery_checkpoint = tracker.get_recovery_checkpoint()
if recovery_checkpoint:
    checkpoint_path = recovery_checkpoint.checkpoint_path
    dataset_position = recovery_checkpoint.dataset_position
    # Load checkpoint and resume from dataset_position
```

### Monitoring Progress

```python
# Get progress summary anytime
summary = tracker.get_progress_summary()

print(f"Session: {summary['session_id']}")
print(f"Progress: {summary['progress_pct']:.1f}%")
print(f"Steps: {summary['steps_completed']}/{summary['total_steps_target']}")
print(f"Elapsed: {summary['elapsed_wall_clock_hours']:.1f} hours")
print(f"ETA: {summary['estimated_remaining_hours']:.1f} hours")
print(f"Loss Trend: {summary['loss_trend']}")
```

## File Structure

### Output Directory Layout

```
training_progress/
├── session.json              # Complete session state
├── progress.json             # Current progress summary
├── metrics.jsonl             # All metrics snapshots (one per line)
├── api_recovery.jsonl        # API responses and costs (one per line)
├── checkpoints.json          # Checkpoint manifest
└── [checkpoint files]        # Model checkpoints
```

### Session State Example

```json
{
  "session_id": "training_20251116_120000_abc12345",
  "start_time": 1763352435.165,
  "config_hash": "4d39d0ede45a2910dff9b9ff77cccaad335fdd3372a0a17ea46e353ca4551d94",
  "steps_completed": 5000,
  "total_steps_target": 10000,
  "samples_processed": 160000,
  "tokens_processed": 5120000,
  "checkpoints": {
    "1000": {
      "checkpoint_step": 1000,
      "checkpoint_path": "training_progress/checkpoint_1000.pt",
      "created_at": 1763352500.0,
      "dataset_position": 32000,
      "recovery_tags": ["every_1k_steps"],
      "metrics": {
        "loss": 2.5,
        "wall_clock_hours": 0.5
      }
    }
  },
  "api_responses_saved": 5000,
  "connection_failures": 0,
  "elapsed_wall_clock_hours": 10.2
}
```

## Recovery Scenarios

### Scenario 1: Connection Loss During API Calls

```python
# System automatically retries with exponential backoff
for attempt in range(5):
    try:
        response = api_client.call(prompt)
        tracker.record_api_response(prompt_hash, response, cost)
        break
    except ConnectionError as e:
        tracker.record_connection_failure(attempt, str(e))
        if attempt < 4:
            time.sleep(2 ** attempt)
        else:
            # Fall back to cached response if available
            cached_response = teacher_cache.get(prompt)
            if cached_response:
                response = cached_response
            else:
                raise
```

### Scenario 2: Training Interrupted Mid-Training

```bash
# Training interrupted at step 5000
# Later, resume training:

python -m training.distill_kd \
  --config configs/training.yaml \
  --resume models/student/checkpoints/checkpoint_5000.pt \
  --output-dir training_progress
```

The system will:

1. Load session state from `training_progress/session.json`
2. Validate config hash matches
3. Resume from checkpoint with optimizer state
4. Continue with next sample from dataset_position
5. All metrics and API responses remain cached

### Scenario 3: Multiple Day Training

**Day 1**:

```bash
python -m training.distill_kd --config config.yaml
# Training runs for 8 hours, then paused
# Session saved: training_20251116_120000_abc12345
```

**Day 2**:

```bash
# Resume with same session ID
python -m training.distill_kd \
  --config config.yaml \
  --resume training_progress/checkpoint_43200.pt
```

System restores:

- Exact step number (43200)
- Optimizer state for learning rate scheduling
- All cached API responses
- Loss trends and metrics history
- Dataset position for deterministic replay

## Integration with Existing Systems

### With TeacherCache

- **Before**: Cache was local, API responses could be lost
- **After**: ProgressTracker persists API responses in JSONL format
- **Benefit**: Recovery data survives training interruptions

### With Checkpoint System

- **Before**: Checkpoints had no recovery metadata
- **After**: ProgressTracker records checkpoint metadata for informed recovery
- **Benefit**: Can identify best checkpoints and exact dataset positions

### With Training Loop

- **Integration Point**: `update_metrics()` called after each training step
- **Integration Point**: `record_checkpoint()` called when saving checkpoints
- **Integration Point**: `record_api_response()` called after API responses received
- **Integration Point**: `record_connection_failure()` called on connection errors

## Implementation Requirements

### Required Modules (Already Implemented)

1. `training/progress_tracker.py` - Main tracking system
2. `training/teacher_cache.py` - API response caching
3. `scripts/make_kd_mix_hardened.py` - Checkpoint management
4. `training/safe_checkpoint_loading.py` - Safe checkpoint loading

### Dependencies in Training Loop

The training loop (`training/distill_kd.py`) needs to integrate:

1. Create ProgressTracker at initialization
2. Call `update_metrics()` after each training step
3. Call `record_checkpoint()` when saving checkpoints
4. Call `record_api_response()` when using teacher responses
5. Call `record_connection_failure()` on API failures
6. Call `tracker.close()` at training end

## Best Practices

### 1. Frequency of Metrics Recording

```python
# Update metrics every step (minimal overhead)
tracker.update_metrics(...)  # ~1-2ms per call

# Record checkpoints less frequently
if step % 1000 == 0:
    tracker.record_checkpoint(...)
```

### 2. API Response Caching

```python
# Always check cache first
cached_response = teacher_cache.get(prompt)
if cached_response:
    response = cached_response
else:
    response = api_client.call(prompt)
    teacher_cache.put(prompt, response)
    tracker.record_api_response(prompt_hash, response, cost)
```

### 3. Connection Retry Logic

```python
# Exponential backoff with jitter
delay = (2 ** attempt) + random.uniform(0, 1)
```

### 4. Progress Monitoring

```python
# Log progress summary periodically
if step % 100 == 0:
    summary = tracker.get_progress_summary()
    logger.info(f"Progress: {summary['progress_pct']:.1f}%")
```

## Expected Overhead

- **Metrics Recording**: ~1-2ms per step
- **Checkpoint Recording**: ~5-10ms (includes manifest write)
- **API Response Caching**: ~0.5-1ms per response
- **File I/O**: Auto-saved every 5 minutes (~100-200ms)

## Troubleshooting

### Issue: Config Hash Mismatch

**Symptom**: "Config hash mismatch. Creating new session."

**Cause**: Training config changed between sessions

**Resolution**: Either use same config or start new session

### Issue: Missing Checkpoints on Resume

**Symptom**: Checkpoint path doesn't exist

**Cause**: Checkpoint moved or training directory changed

**Resolution**: Ensure checkpoint path is absolute or relative from same directory

### Issue: API Response Cache Growing Too Large

**Symptom**: `api_recovery.jsonl` is many GB

**Cause**: Unique prompts weren't reused, all responses cached

**Resolution**: Normal for diverse training data. Archive old responses if needed.

## Monitoring Commands

```bash
# Check session status
cat training_progress/session.json | jq '.steps_completed, .total_steps_target'

# View progress summary
cat training_progress/progress.json | jq '.progress_pct, .elapsed_wall_clock_hours'

# Count API responses cached
wc -l training_progress/api_recovery.jsonl

# Find best checkpoint by loss
cat training_progress/checkpoints.json | jq '.checkpoints[] | {step: .checkpoint_step, loss: .metrics.loss}' | sort -k6 -n | head -1
```
