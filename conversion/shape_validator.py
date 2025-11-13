"""
Shape validation utilities for pre-export verification.

This module provides helpers to validate enumerated shapes before full export,
allowing early detection of shape-related issues.
"""
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def validate_shape_with_model(
    model: torch.nn.Module,
    shape: int,
    vocab_size: int,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    Validate that a model can handle a specific sequence length.

    Args:
        model: PyTorch model to validate
        shape: Sequence length to test (T)
        vocab_size: Model vocabulary size
        device: Device to run validation on

    Returns:
        Dictionary with validation results:
        {
            "shape": int,
            "status": "ok" | "error",
            "error": Optional[str],
            "output_shape": Optional[Tuple[int, ...]]
        }
    """
    model.eval()
    try:
        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (1, shape), device=device)
        
        # Run forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        # Check output shape
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        expected_shape = (1, shape, vocab_size)
        actual_shape = tuple(logits.shape)
        
        if actual_shape != expected_shape:
            return {
                "shape": shape,
                "status": "error",
                "error": f"Output shape mismatch: expected {expected_shape}, got {actual_shape}",
                "output_shape": actual_shape
            }
        
        # Check for NaN or Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return {
                "shape": shape,
                "status": "error",
                "error": "Output contains NaN or Inf values",
                "output_shape": actual_shape
            }
        
        return {
            "shape": shape,
            "status": "ok",
            "error": None,
            "output_shape": actual_shape
        }
    
    except RuntimeError as e:
        return {
            "shape": shape,
            "status": "error",
            "error": f"RuntimeError: {str(e)}",
            "output_shape": None
        }
    except Exception as e:
        return {
            "shape": shape,
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "output_shape": None
        }


def validate_enumerated_shapes(
    model: torch.nn.Module,
    enumerated_shapes: List[int],
    vocab_size: int,
    primary_shape: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, any]:
    """
    Validate multiple enumerated shapes with a model.

    Args:
        model: PyTorch model to validate
        enumerated_shapes: List of sequence lengths to test
        vocab_size: Model vocabulary size
        primary_shape: Primary shape that must succeed (defaults to first shape)
        device: Device to run validation on

    Returns:
        Dictionary with validation results:
        {
            "primary_shape": int,
            "primary_status": "ok" | "error",
            "results": List[Dict],  # One per shape
            "all_ok": bool,
            "primary_ok": bool
        }
    """
    if not enumerated_shapes:
        return {
            "primary_shape": None,
            "primary_status": "error",
            "results": [],
            "all_ok": False,
            "primary_ok": False,
            "error": "No shapes provided"
        }
    
    if primary_shape is None:
        primary_shape = enumerated_shapes[0]
    
    if primary_shape not in enumerated_shapes:
        return {
            "primary_shape": primary_shape,
            "primary_status": "error",
            "results": [],
            "all_ok": False,
            "primary_ok": False,
            "error": f"Primary shape {primary_shape} not in enumerated_shapes"
        }
    
    results = []
    for shape in enumerated_shapes:
        result = validate_shape_with_model(model, shape, vocab_size, device)
        results.append(result)
    
    primary_result = next(r for r in results if r["shape"] == primary_shape)
    primary_ok = primary_result["status"] == "ok"
    all_ok = all(r["status"] == "ok" for r in results)
    
    return {
        "primary_shape": primary_shape,
        "primary_status": primary_result["status"],
        "results": results,
        "all_ok": all_ok,
        "primary_ok": primary_ok
    }


def get_production_shapes() -> List[int]:
    """
    Get default production enumerated shapes.

    Returns:
        List of production sequence lengths: [512, 1024, 2048, 4096]
    """
    return [512, 1024, 2048, 4096]


def get_toy_shapes() -> List[int]:
    """
    Get default toy enumerated shapes.

    Returns:
        List of toy sequence lengths: [64, 128, 256]
    """
    return [64, 128, 256]


def get_primary_shape(enumerated_shapes: List[int], is_toy: bool = False) -> int:
    """
    Get the primary shape from enumerated shapes.

    Args:
        enumerated_shapes: List of sequence lengths
        is_toy: Whether this is a toy model

    Returns:
        Primary shape (T128 for toy, T1024 for production, or first shape)
    """
    if is_toy:
        # Toy models: T128 is primary
        if 128 in enumerated_shapes:
            return 128
    else:
        # Production models: T1024 is primary
        if 1024 in enumerated_shapes:
            return 1024
    
    # Fallback to first shape
    return enumerated_shapes[0] if enumerated_shapes else 128


def check_shape_validation_results(
    validation_results: Dict[str, any],
    require_all: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check shape validation results and return status.

    Args:
        validation_results: Results from validate_enumerated_shapes
        require_all: If True, require all shapes to pass (not just primary)

    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []
    
    if not validation_results.get("primary_ok", False):
        errors.append(
            f"Primary shape {validation_results.get('primary_shape')} validation failed"
        )
    
    if require_all and not validation_results.get("all_ok", False):
        failed_shapes = [
            r["shape"] for r in validation_results.get("results", [])
            if r["status"] != "ok"
        ]
        errors.append(
            f"Shape validation failed for shapes: {failed_shapes}"
        )
    
    # Collect detailed error messages
    for result in validation_results.get("results", []):
        if result["status"] == "error":
            errors.append(
                f"Shape {result['shape']}: {result.get('error', 'Unknown error')}"
            )
    
    success = len(errors) == 0
    return success, errors

