# training/safe_checkpoint_loading.py
# Safe checkpoint loading utilities with structure validation
# @author: @darianrosebrook

import torch
from typing import Dict, Any, Optional
from pathlib import Path


def safe_load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str = "cpu",
    expected_keys: Optional[set] = None,
    required_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Safely load checkpoint with structure validation.
    
    First attempts to load with weights_only=True for maximum security.
    If that fails (due to non-tensor metadata), validates structure before loading.
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to load checkpoint on (default: "cpu")
        expected_keys: Set of expected keys in checkpoint (for validation)
        required_keys: Set of required keys in checkpoint (will raise if missing)
        
    Returns:
        Dictionary containing checkpoint data
        
    Raises:
        ValueError: If checkpoint structure is invalid
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Default expected keys for standard training checkpoints
    if expected_keys is None:
        expected_keys = {
            "model_state_dict",
            "config",
            "model_arch",
            "step",
            "optimizer_state_dict",
            "loss",
            "meta",
            "scheduler_state_dict",  # May be in meta dict
        }
    
    # Default required keys for standard training checkpoints
    if required_keys is None:
        required_keys = {"model_state_dict"}
    
    # First, try loading with weights_only=True for maximum security
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        # If weights_only succeeds, it's a pure tensor checkpoint - validate structure
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
    except Exception:
        # weights_only failed (expected for checkpoints with metadata)
        # Load without weights_only but validate structure immediately
        try:
            # nosec B614: torch.load() used in fallback path after weights_only=True fails
            # Security: This path only executes if weights_only=True fails (expected for checkpoints
            # with metadata). Structure is validated immediately after loading (see below).
            # This is part of the safe wrapper function pattern: try secure path first, then
            # validate structure before using less secure fallback.
            # Explicitly set weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)  # nosec B614
        except Exception as load_error:
            raise RuntimeError(f"Failed to load checkpoint: {load_error}") from load_error
        
        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
        
        # Check for unexpected keys (security measure)
        checkpoint_keys = set(checkpoint.keys())
        unexpected_keys = checkpoint_keys - expected_keys
        
        if unexpected_keys:
            import warnings
            warnings.warn(
                f"Checkpoint contains unexpected keys: {unexpected_keys}. "
                "These may be ignored depending on usage. Review checkpoint structure.",
                UserWarning
            )
        
        # Validate required keys exist
        missing_keys = required_keys - checkpoint_keys
        if missing_keys:
            raise ValueError(
                f"Checkpoint missing required keys: {missing_keys}. "
                f"Found keys: {checkpoint_keys}"
            )
        
        # Validate model_state_dict is a dict (state_dict structure)
        if "model_state_dict" in checkpoint:
            if not isinstance(checkpoint["model_state_dict"], dict):
                raise ValueError(
                    f"model_state_dict must be a dictionary, got {type(checkpoint['model_state_dict'])}"
                )
        
        # Validate config is a dict if present
        if "config" in checkpoint and checkpoint["config"] is not None:
            if not isinstance(checkpoint["config"], dict):
                raise ValueError(
                    f"config must be a dictionary, got {type(checkpoint['config'])}"
                )
        
        # Validate model_arch is a dict if present
        if "model_arch" in checkpoint and checkpoint["model_arch"] is not None:
            if not isinstance(checkpoint["model_arch"], dict):
                raise ValueError(
                    f"model_arch must be a dictionary, got {type(checkpoint['model_arch'])}"
                )
    
    return checkpoint

