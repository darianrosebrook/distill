"""
Multi-device training utilities for Apple Silicon.

Supports parallelization across CPU, MPS (GPU), and optionally ANE.
Implements CPU offloading for optimizer states and model parallelism.

Author: @darianrosebrook
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CPUOffloadOptimizer(torch.optim.Optimizer):
    """
    Optimizer wrapper that keeps optimizer state on CPU to save MPS memory.

    This is critical for large models on Apple Silicon where optimizer state
    (exp_avg, exp_avg_sq for Adam) doubles memory usage.

    Strategy:
    - Model parameters stay on MPS (or split across devices)
    - Optimizer state (momentum buffers) stay on CPU
    - During optimizer.step(), temporarily move state to MPS, update, move back
    """

    def __init__(self, optimizer: torch.optim.Optimizer, param_device: torch.device):
        """
        Initialize CPU-offloaded optimizer.

        Args:
            optimizer: Base optimizer (e.g., AdamW)
            param_device: Device where model parameters live (e.g., MPS)
        """
        # Store optimizer first (before super().__init__)
        self.optimizer = optimizer
        self.param_device = param_device
        self.cpu_device = torch.device("cpu")

        # Initialize base Optimizer with the same param_groups
        # This ensures isinstance() checks pass for schedulers
        # Use object.__setattr__ to bypass our property setter during init
        super().__init__(optimizer.param_groups, optimizer.defaults)

        # Replace base class state and param_groups with wrapped optimizer's
        # This allows the base class to initialize, then we delegate to wrapped optimizer
        object.__setattr__(self, 'state', optimizer.state)
        object.__setattr__(self, 'param_groups', optimizer.param_groups)

        # Move all optimizer state to CPU
        self._offload_state_to_cpu()

        logger.info(
            f"[CPUOffloadOptimizer] Optimizer state offloaded to CPU, "
            f"parameters on {param_device}"
        )

    def __setattr__(self, name, value):
        """Override __setattr__ to delegate param_groups and state to wrapped optimizer."""
        # During initialization (before optimizer is set), allow normal attribute setting
        if not hasattr(self, 'optimizer') or name in ('optimizer', 'param_device', 'cpu_device'):
            object.__setattr__(self, name, value)
            return

        if name == 'param_groups':
            # Delegate to wrapped optimizer
            object.__setattr__(self.optimizer, 'param_groups', value)
            # Also store on self for base class compatibility (using dict to bypass __setattr__)
            self.__dict__['param_groups'] = value
        elif name == 'state':
            # Delegate to wrapped optimizer
            object.__setattr__(self.optimizer, 'state', value)
            # Also store on self for base class compatibility
            self.__dict__['state'] = value
        else:
            # Normal attribute setting
            object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        """Override __getattribute__ to delegate param_groups and state to wrapped optimizer."""
        # For param_groups and state, always delegate to wrapped optimizer
        if name == 'param_groups':
            return object.__getattribute__(self, 'optimizer').param_groups
        elif name == 'state':
            return object.__getattribute__(self, 'optimizer').state
        # For other attributes, use normal lookup
        return object.__getattribute__(self, name)

    def _offload_state_to_cpu(self):
        """Move all optimizer state dict tensors to CPU."""
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.cpu_device)

    def _load_state_to_param_device(self):
        """Temporarily move optimizer state to parameter device for step()."""
        moved_states = {}
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    moved_states[param] = {}
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            # Store original CPU tensor, move to param device
                            # Original CPU tensor
                            moved_states[param][key] = value
                            state[key] = value.to(self.param_device)
        return moved_states

    def _offload_state_back_to_cpu(self, moved_states: Dict):
        """Move optimizer state back to CPU after step()."""
        for param, original_states in moved_states.items():
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for key, original_cpu_tensor in original_states.items():
                    if isinstance(original_cpu_tensor, torch.Tensor):
                        # Get updated tensor from param device, move to CPU
                        updated_tensor = state[key]
                        if updated_tensor.device != self.cpu_device:
                            state[key] = updated_tensor.to(self.cpu_device)
                        else:
                            state[key] = updated_tensor

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients (delegate to base optimizer)."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Perform optimizer step with CPU-offloaded state.

        Temporarily moves state to parameter device, performs update,
        then moves back to CPU.
        """
        # Move state to parameter device
        moved_states = self._load_state_to_param_device()

        try:
            # Perform optimizer step
            result = self.optimizer.step(closure)
        finally:
            # Always move state back to CPU, even if step() fails
            self._offload_state_back_to_cpu(moved_states)

        return result

    def state_dict(self):
        """Return optimizer state dict (all on CPU)."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load optimizer state dict.

        Handles state dicts from any device (MPS, CUDA, CPU) by moving
        all tensors to CPU after loading.
        """
        # Load state dict (may contain tensors on any device)
        self.optimizer.load_state_dict(state_dict)
        # Ensure everything is moved to CPU
        self._offload_state_to_cpu()


def split_model_across_devices(
    model: nn.Module,
    device_config: Dict[str, Any],
) -> nn.Module:
    """
    Split model layers across multiple devices (model parallelism).

    Args:
        model: Model to split
        device_config: Configuration dict with:
            - "strategy": "alternate" | "first_half_mps" | "custom"
            - "mps_device": torch.device for MPS
            - "cpu_device": torch.device for CPU
            - "split_point": Optional int for custom split

    Returns:
        Model with layers distributed across devices
    """
    strategy = device_config.get("strategy", "alternate")
    mps_device = device_config.get("mps_device", torch.device("mps"))
    cpu_device = device_config.get("cpu_device", torch.device("cpu"))

    # Find all transformer blocks/layers
    blocks = []
    if hasattr(model, 'blocks'):
        blocks = list(model.blocks)
    elif hasattr(model, 'layers'):
        blocks = list(model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        blocks = list(model.transformer.blocks)
    else:
        logger.warning(
            "[split_model_across_devices] Could not find model blocks, "
            "returning model on single device"
        )
        return model.to(mps_device)

    num_blocks = len(blocks)
    logger.info(
        f"[split_model_across_devices] Splitting {num_blocks} blocks using {strategy}")

    if strategy == "alternate":
        # Alternate layers between MPS and CPU
        for i, block in enumerate(blocks):
            target_device = mps_device if i % 2 == 0 else cpu_device
            block.to(target_device)
            logger.debug(f"  Block {i} -> {target_device}")

    elif strategy == "first_half_mps":
        # First half on MPS, second half on CPU
        split_point = num_blocks // 2
        for i, block in enumerate(blocks):
            target_device = mps_device if i < split_point else cpu_device
            block.to(target_device)
            logger.debug(f"  Block {i} -> {target_device}")

    elif strategy == "custom":
        # Custom split point
        split_point = device_config.get("split_point", num_blocks // 2)
        for i, block in enumerate(blocks):
            target_device = mps_device if i < split_point else cpu_device
            block.to(target_device)
            logger.debug(f"  Block {i} -> {target_device}")

    else:
        logger.warning(
            f"[split_model_across_devices] Unknown strategy: {strategy}")
        return model.to(mps_device)

    # Move embedding and output layers
    if hasattr(model, 'embed'):
        model.embed.to(mps_device)
    if hasattr(model, 'output'):
        model.output.to(mps_device)
    if hasattr(model, 'lm_head'):
        model.lm_head.to(mps_device)

    logger.info(
        f"[split_model_across_devices] Model split complete: "
        f"{num_blocks // 2} blocks on MPS, {num_blocks - num_blocks // 2} blocks on CPU"
    )

    return model


def create_multi_device_optimizer(
    model: nn.Module,
    optimizer_class: type,
    optimizer_kwargs: Dict[str, Any],
    use_cpu_offload: bool = True,
    param_device: Optional[torch.device] = None,
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional CPU offloading for multi-device training.

    **Supported Configurations:**
    - Config A (Primary): Single device (MPS or CPU) + CPU offload
    - Config B (Debug): Single device (CPU) + standard optimizer
    - Config C (Experimental): Multi-device split + standard optimizer (CPU offload disabled)

    **Unsupported:**
    - CPU offload + model parallelism (parameters on multiple devices)

    Args:
        model: Model to create optimizer for
        optimizer_class: Optimizer class (e.g., torch.optim.AdamW)
        optimizer_kwargs: Keyword arguments for optimizer
        use_cpu_offload: Whether to offload optimizer state to CPU
        param_device: Device where parameters live (auto-detected if None)

    Returns:
        Optimizer (wrapped with CPUOffloadOptimizer if use_cpu_offload=True)

    Raises:
        RuntimeError: If CPU offload is requested but model has parameters on multiple devices
    """
    # Detect parameter devices
    param_devices = {p.device for p in model.parameters()}

    # Enforce constraint: CPUOffloadOptimizer requires single-device models
    if use_cpu_offload and len(param_devices) > 1:
        device_list = ", ".join(str(d) for d in sorted(param_devices, key=str))
        raise RuntimeError(
            f"CPUOffloadOptimizer does not support model-parallel (multi-device) models. "
            f"Model has parameters on {len(param_devices)} devices: {device_list}. "
            f"Either disable CPU offload (use_cpu_offload=False) or disable model splitting "
            f"(multi_device.enabled=False)."
        )

    # Detect parameter device if not specified (now safe: all params on same device)
    if param_device is None:
        if len(param_devices) == 0:
            raise RuntimeError("Model has no parameters")
        param_device = next(iter(param_devices))

    # Validate that all parameters are on the detected device
    if len(param_devices) > 1:
        # This shouldn't happen if use_cpu_offload=True (caught above)
        # But if use_cpu_offload=False, we allow multi-device (experimental)
        logger.warning(
            f"[create_multi_device_optimizer] Model has parameters on multiple devices: {param_devices}. "
            f"This is experimental and CPU offload is disabled."
        )

    # Create base optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Wrap with CPU offloading if requested and not on CPU
    if use_cpu_offload and param_device.type != "cpu":
        optimizer = CPUOffloadOptimizer(optimizer, param_device)
        logger.info(
            f"[create_multi_device_optimizer] Created CPU-offloaded optimizer "
            f"for {param_device} parameters (single device: {param_device})"
        )
    else:
        if use_cpu_offload and param_device.type == "cpu":
            logger.info(
                f"[create_multi_device_optimizer] CPU offload requested but parameters are on CPU. "
                f"Using standard optimizer (no offload needed)."
            )
        logger.info(
            f"[create_multi_device_optimizer] Created standard optimizer "
            f"on {param_device} (devices: {param_devices})"
        )

    return optimizer


def estimate_memory_savings(
    model: nn.Module,
    use_cpu_offload: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory savings from CPU offloading.

    Returns:
        Dict with memory estimates in GB
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = total_params * 4 / (1024**3)  # FP32, 4 bytes per param

    # Adam optimizer state: 2x parameter memory (exp_avg + exp_avg_sq)
    optimizer_memory_gb = param_memory_gb * 2

    if use_cpu_offload:
        mps_memory_gb = param_memory_gb  # Only parameters on MPS
        cpu_memory_gb = optimizer_memory_gb  # Optimizer state on CPU
        total_memory_gb = mps_memory_gb + cpu_memory_gb
        savings_gb = optimizer_memory_gb  # Saved from MPS
    else:
        mps_memory_gb = param_memory_gb + optimizer_memory_gb
        cpu_memory_gb = 0
        total_memory_gb = mps_memory_gb
        savings_gb = 0

    return {
        "param_memory_gb": param_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "mps_memory_gb": mps_memory_gb,
        "cpu_memory_gb": cpu_memory_gb,
        "total_memory_gb": total_memory_gb,
        "mps_savings_gb": savings_gb,
    }
