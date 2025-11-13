"""
Shared training utilities.

Contains common functions used across training scripts.
"""
import hashlib
import io
from typing import Dict

import torch


def sha256_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Compute SHA256 hash of model state dict.

    Serializes tensor values to bytes for accurate hashing.
    Uses tensor.numpy() to convert to NumPy array, then tobytes() for binary serialization.
    This provides more accurate fingerprinting than shape/dtype-only hashing.

    Args:
        state_dict: Dictionary mapping parameter names to tensors

    Returns:
        Hexadecimal SHA256 hash string
    """
    buffer = io.BytesIO()

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        # Write key name
        buffer.write(key.encode('utf-8'))
        buffer.write(b':')

        # Write shape and dtype metadata
        buffer.write(str(tensor.shape).encode('utf-8'))
        buffer.write(b':')
        buffer.write(str(tensor.dtype).encode('utf-8'))
        buffer.write(b':')

        # Write actual tensor data as bytes
        # Convert to CPU and NumPy for serialization
        if tensor.is_cuda:
            tensor_cpu = tensor.cpu()
        else:
            tensor_cpu = tensor

        # Convert to NumPy array and serialize to bytes
        try:
            tensor_np = tensor_cpu.detach().numpy()
            tensor_bytes = tensor_np.tobytes()
            buffer.write(tensor_bytes)
        except Exception as e:
            # Fallback: if conversion fails, use shape/dtype only
            # This should rarely happen, but provides graceful degradation
            print(
                f"[utils] WARN: Could not serialize tensor {key}: {e}")
            buffer.write(b'<serialization_failed>')

        buffer.write(b'|')  # Separator between tensors

    content = buffer.getvalue()
    return hashlib.sha256(content).hexdigest()

