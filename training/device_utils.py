"""
Device detection utilities for training scripts.

Author: @darianrosebrook
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_training_device() -> torch.device:
    """
    Get the best available device for training.
    
    Priority: CUDA > MPS > CPU
    Supports manual override via DISTILL_DEVICE_OVERRIDE environment variable.
    
    Returns:
        torch.device instance for the best available device
    """
    # Check for manual override (useful for debugging or forcing CPU/MPS)
    override = os.environ.get("DISTILL_DEVICE_OVERRIDE", "").lower()
    if override in {"cpu", "cuda", "mps"}:
        device = torch.device(override)
        logger.info(f"[device_utils] Using device override: {device}")
        return device
    
    # Automatic device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"[device_utils] Using device: {device}")
    return device

