"""
Shared assertion utilities for training pipeline.

Provides common assertions for loss finiteness, gradient norms, and other training checks.
@author: @darianrosebrook
"""

import torch
from typing import Dict, Any, Optional


def assert_loss_finite(loss_dict: Dict[str, Any], step: int = None) -> None:
    """
    Assert that all loss components are finite (no NaNs or Infs).

    Args:
        loss_dict: Dictionary of loss components with 'total' key
        step: Optional step number for error messages

    Raises:
        RuntimeError: If any loss component is NaN or Inf
    """
    step_msg = f" at step {step}" if step is not None else ""

    # Check total loss
    total_loss = loss_dict.get("total")
    if total_loss is None:
        raise RuntimeError(f"Loss dictionary missing 'total' key{step_msg}")

    if isinstance(total_loss, torch.Tensor):
        if not torch.isfinite(total_loss).all():
            nan_count = torch.isnan(total_loss).sum().item()
            inf_count = torch.isinf(total_loss).sum().item()
            raise RuntimeError(
                f"Total loss is not finite{step_msg}: "
                f"NaN count={nan_count}, Inf count={inf_count}, "
                f"loss value={total_loss.item()}"
            )
        if total_loss.item() == 0.0:
            # Check if all weights are zero (which would cause zero loss)
            non_zero_components = [
                k
                for k, v in loss_dict.items()
                if k != "total" and isinstance(v, torch.Tensor) and v.item() != 0.0
            ]
            if not non_zero_components:
                raise RuntimeError(
                    f"Total loss is zero{step_msg} - all loss weights may be zero. "
                    f"Check loss weight configuration."
                )
    else:
        if not (
            isinstance(total_loss, (int, float))
            and (total_loss == total_loss and abs(total_loss) != float("inf"))
        ):
            raise RuntimeError(f"Total loss is not finite{step_msg}: {total_loss}")

    # Check individual loss components
    for key, value in loss_dict.items():
        if key == "total":
            continue
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                raise RuntimeError(
                    f"Loss component '{key}' is not finite{step_msg}: "
                    f"NaN count={nan_count}, Inf count={inf_count}"
                )


def log_loss_components(loss_dict: Dict[str, Any], step: int, log_every: int = 100) -> None:
    """
    Log loss component breakdown every N steps.

    Args:
        loss_dict: Dictionary of loss components
        step: Current training step
        log_every: Log every N steps (default: 100)
    """
    if step % log_every == 0:
        components = []
        for key, value in sorted(loss_dict.items()):
            if isinstance(value, torch.Tensor):
                val = value.item()
            else:
                val = float(value)
            components.append(f"{key}={val:.4f}")
        print(f"[distill_kd] Step {step} loss components: {', '.join(components)}")


def compute_gradient_norms(model: torch.nn.Module, loss_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute gradient norms for each loss component.

    Note: This requires gradients to be computed. Call after backward() but before optimizer.step().

    Args:
        model: Model with computed gradients
        loss_dict: Dictionary of loss components

    Returns:
        Dictionary mapping loss component names to gradient norms
    """
    grad_norms = {}

    # Compute total gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    grad_norms["total"] = total_norm

    return grad_norms


def check_gradient_balance(
    grad_norms: Dict[str, float], imbalance_threshold: float = 10.0, window_size: int = 100
) -> Optional[str]:
    """
    Check if any loss component's gradient norm exceeds others by threshold.

    Args:
        grad_norms: Dictionary of gradient norms per loss component
        imbalance_threshold: Threshold for imbalance detection (default: 10.0 = 10×)
        window_size: Window size for checking (not used in basic version)

    Returns:
        Warning message if imbalance detected, None otherwise
    """
    if len(grad_norms) < 2:
        return None

    norms = list(grad_norms.values())
    max_norm = max(norms)
    min_norm = min(norms)

    if min_norm > 0:
        ratio = max_norm / min_norm
        if ratio > imbalance_threshold:
            # Find which component has the max norm
            max_component = max(grad_norms.items(), key=lambda x: x[1])[0]
            return (
                f"Gradient imbalance detected: {max_component} gradient norm "
                f"({max_norm:.4f}) is {ratio:.1f}× larger than minimum ({min_norm:.4f}). "
                f"Threshold: {imbalance_threshold}×"
            )

    return None


def compute_per_component_gradient_norms(
    model: torch.nn.Module,
    loss_dict: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute gradient norms for each loss component separately.

    This requires computing gradients for each component individually,
    which is expensive but provides detailed monitoring.

    Note: This function modifies gradients. Call before optimizer.step()
    and restore gradients if needed.

    Args:
        model: Model with parameters
        loss_dict: Dictionary of loss components (tensors)
        loss_weights: Dictionary of loss weights

    Returns:
        Dictionary mapping loss component names to gradient norms
    """
    grad_norms = {}

    # Store original gradients
    original_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            original_grads[name] = param.grad.clone()

    # Compute gradients for each component
    for component_name, loss_value in loss_dict.items():
        if component_name == "total":
            continue  # Skip total, we'll compute it separately

        weight = loss_weights.get(component_name, 1.0)
        if weight == 0.0:
            continue

        # Zero gradients
        model.zero_grad()

        # Compute gradient for this component
        weighted_loss = weight * loss_value
        weighted_loss.backward(retain_graph=True)

        # Compute gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        grad_norms[component_name] = total_norm

    # Restore original gradients
    for name, param in model.named_parameters():
        if name in original_grads:
            param.grad = original_grads[name]
        else:
            param.grad = None

    return grad_norms
