"""
Host API Contract Definition

Defines the contract for host-side (Swift or Python wrapper) API for model inference.
This contract is versioned to handle KV cache layout or semantics changes.

@author: @darianrosebrook
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class APIVersion(Enum):
    """API contract version."""

    V1 = "1.0"
    V2 = "2.0"  # Future version with KV cache layout changes


@dataclass
class PrefillState:
    """State returned from prefill operation."""

    logits: Any  # Logits tensor/array [B, T, V]
    kv_caches: Optional[List[Tuple[Any, Any]]] = None  # List of (K, V) cache tuples per layer
    halt_logits: Optional[Any] = None  # Halt head logits [B, 2] (if enabled)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class DecodeState:
    """State returned from decode operation."""

    logits: Any  # Logits tensor/array [B, 1, V]
    kv_caches: List[Tuple[Any, Any]]  # Updated KV caches per layer
    halt_logits: Optional[Any] = None  # Halt head logits [B, 2] (if enabled)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class APIContract:
    """API contract definition."""

    version: str
    prefill_signature: Dict[str, Any]
    decode_signature: Dict[str, Any]
    kv_cache_layout: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary."""
        return {
            "version": self.version,
            "prefill_signature": self.prefill_signature,
            "decode_signature": self.decode_signature,
            "kv_cache_layout": self.kv_cache_layout,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIContract":
        """Create contract from dictionary."""
        return cls(
            version=data["version"],
            prefill_signature=data["prefill_signature"],
            decode_signature=data["decode_signature"],
            kv_cache_layout=data["kv_cache_layout"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path):
        """Save contract to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "APIContract":
        """Load contract from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# V1 API Contract (current)
API_CONTRACT_V1 = APIContract(
    version="1.0",
    prefill_signature={
        "name": "prefill",
        "inputs": [
            {
                "name": "input_ids",
                "type": "int32",
                "shape": ["B", "T"],
                "description": "Input token IDs",
            },
            {
                "name": "attention_mask",
                "type": "int32",
                "shape": ["B", "T"],
                "optional": True,
                "description": "Attention mask (1 for valid tokens, 0 for padding)",
            },
        ],
        "outputs": [
            {
                "name": "logits",
                "type": "float16",
                "shape": ["B", "T", "V"],
                "description": "Logits for all positions",
            },
            {
                "name": "halt_logits",
                "type": "float16",
                "shape": ["B", "2"],
                "optional": True,
                "description": "Halt head logits (if halt head enabled)",
            },
        ],
        "description": "Process full input sequence (prefill phase)",
    },
    decode_signature={
        "name": "decode_step",
        "inputs": [
            {
                "name": "input_ids",
                "type": "int32",
                "shape": ["B", "1"],
                "description": "Single token ID to decode",
            },
            {
                "name": "kv_caches",
                "type": "list",
                "description": "List of (K, V) cache tuples, one per layer",
                "item_shape": {
                    "k_cache": ["B", "Hk", "T_cache", "Dh"],
                    "v_cache": ["B", "Hk", "T_cache", "Dh"],
                },
            },
            {
                "name": "position",
                "type": "int32",
                "description": "Position index for RoPE",
            },
        ],
        "outputs": [
            {
                "name": "logits",
                "type": "float16",
                "shape": ["B", "1", "V"],
                "description": "Logits for next token",
            },
            {
                "name": "kv_caches",
                "type": "list",
                "description": "Updated KV caches",
                "item_shape": {
                    "k_cache": ["B", "Hk", "T_cache+1", "Dh"],
                    "v_cache": ["B", "Hk", "T_cache+1", "Dh"],
                },
            },
            {
                "name": "halt_logits",
                "type": "float16",
                "shape": ["B", "2"],
                "optional": True,
                "description": "Halt head logits (if halt head enabled)",
            },
        ],
        "description": "Process single token with KV cache (decode phase)",
    },
    kv_cache_layout={
        "format": "per_layer_tuples",
        "description": "KV caches stored as list of (K, V) tuples, one per transformer layer",
        "k_cache_shape": ["B", "Hk", "T_cache", "Dh"],
        "v_cache_shape": ["B", "Hk", "T_cache", "Dh"],
        "dtype": "float16",
        "empty_cache": "T_cache=0 (empty tensor)",
    },
    metadata={
        "created": "2024-01-01",
        "description": "Initial API contract for prefill/decode split",
    },
)


def get_api_contract(version: str = "1.0") -> APIContract:
    """
    Get API contract for specified version.

    Args:
        version: Contract version (e.g., "1.0")

    Returns:
        API contract object

    Raises:
        ValueError: If version not supported
    """
    if version == "1.0":
        return API_CONTRACT_V1
    else:
        raise ValueError(f"Unsupported API contract version: {version}")


def validate_api_compatibility(
    model_contract_path: Path,
    required_version: str = "1.0",
) -> Tuple[bool, Optional[str]]:
    """
    Validate that model's API contract is compatible with required version.

    Args:
        model_contract_path: Path to model's API contract file
        required_version: Required API contract version

    Returns:
        Tuple of (is_compatible, error_message)
    """
    if not model_contract_path.exists():
        return False, f"API contract file not found: {model_contract_path}"

    try:
        model_contract = APIContract.load(model_contract_path)

        # For now, exact version match required
        # In future, could implement version compatibility matrix
        if model_contract.version != required_version:
            return (
                False,
                f"API contract version mismatch: model={model_contract.version}, required={required_version}",
            )

        return True, None

    except Exception as e:
        return False, f"Failed to load/validate API contract: {e}"


# Example usage and documentation
if __name__ == "__main__":
    # Save V1 contract
    contract_path = Path("runtime/api_contract_v1.json")
    API_CONTRACT_V1.save(contract_path)
    print(f"Saved API contract V1 to: {contract_path}")

    # Load and validate
    loaded_contract = APIContract.load(contract_path)
    print(f"Loaded contract version: {loaded_contract.version}")

    # Validate compatibility
    is_compat, error = validate_api_compatibility(contract_path, "1.0")
    print(f"Compatibility check: {is_compat}")
    if error:
        print(f"Error: {error}")
