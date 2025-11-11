# models/student/roles.py
# Model role definitions for multi-model portfolio
# @author: @darianrosebrook

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class ModelRole(str, Enum):
    """Model roles in the CAWS arbiter stack."""
    WORKER = "worker"  # Primary generator (~9B)
    JUDGE = "judge"  # Constitutional arbiter (3-4B or 7B)
    DRAFTER = "drafter"  # Speculative decoding (~4B, optional)


@dataclass
class RoleConfig:
    """Configuration for a model role."""
    role: ModelRole
    target_size: str  # e.g., "9B", "4B", "7B"
    seq_lengths: list[int]
    precision_path: str  # e.g., "int8_weights_fp16_acts"
    primary_use: str
    export_shapes: list[int]


# Role-specific configurations
ROLE_CONFIGS: Dict[ModelRole, RoleConfig] = {
    ModelRole.WORKER: RoleConfig(
        role=ModelRole.WORKER,
        target_size="9B",
        seq_lengths=[4096, 8192, 16384],
        precision_path="int8_weights_fp16_acts",
        primary_use="code_edits_tool_use_long_ctx",
        export_shapes=[4096, 8192, 16384]
    ),
    ModelRole.JUDGE: RoleConfig(
        role=ModelRole.JUDGE,
        target_size="4B",
        seq_lengths=[512, 1024, 2048],
        precision_path="int8_weights_fp16_acts",
        primary_use="adjudication_clause_mapping_claim_verification",
        export_shapes=[512, 1024, 2048]
    ),
    ModelRole.DRAFTER: RoleConfig(
        role=ModelRole.DRAFTER,
        target_size="4B",
        seq_lengths=[2048, 4096],
        precision_path="int8_weights_fp16_acts",
        primary_use="speculative_decoding_fast_tokens",
        export_shapes=[2048, 4096]
    ),
}


def get_role_config(role: ModelRole) -> Optional[RoleConfig]:
    """Get configuration for a model role."""
    return ROLE_CONFIGS.get(role)

