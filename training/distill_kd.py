"""
Knowledge distillation training script.

Trains student model from teacher using KD losses:
- KL divergence (soft targets)
- Cross-entropy on teacher predictions
- Cross-entropy on ground truth

Usage:
    python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.losses import (
    combined_kd_loss,
    caws_compliance_loss,
    intermediate_layer_loss,
    self_evaluation_loss,
    create_projection_layers,
    length_aware_kd_loss,
    early_tool_call_loss,
    CodeModePreferenceLoss,
)
from training.dataset import KDDataset, collate_kd_batch, load_tokenizer
from training.tracing import create_tracer_from_config
from training.assertions import assert_loss_finite, log_loss_components
from infra.version_gate import check_training_versions

# Latent curriculum import (optional)
try:
    from data.wrappers.curriculum import LatentCurriculum

    LATENT_CURRICULUM_AVAILABLE = True
except ImportError:
    LATENT_CURRICULUM_AVAILABLE = False
    LatentCurriculum = None

# Tokenizer migration import
try:
    from training.tokenizer_migration import migrate_tokenizer_and_model

    TOKENIZER_MIGRATION_AVAILABLE = True
except ImportError:
    TOKENIZER_MIGRATION_AVAILABLE = False
    migrate_tokenizer_and_model = None

# QAT imports (optional, only if quant.enabled)
try:
    from training.quant_qat_int8 import quantize_model

    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    quantize_model = None

# Speed metrics imports
try:
    from training.speed_metrics import measure_proxy, aggregate_speed_metrics

    SPEED_METRICS_AVAILABLE = True
except ImportError:
    SPEED_METRICS_AVAILABLE = False
    measure_proxy = None
    aggregate_speed_metrics = None


def compute_required_fields_present(
    batch: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    student_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute boolean mask indicating if student has all required fields present.

    Checks if student output contains all required tool arguments/evidence fields
    by comparing against teacher's validated arguments.

    Args:
        batch: Training batch dictionary
        tokenizer: Tokenizer for decoding
        device: Device to place tensors on

    Returns:
        [B] boolean tensor: True if all required fields present, False otherwise
    """
    B = batch.get("input_ids", torch.empty(0)).size(0)
    if B == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    # Default: assume fields present (no penalty) if we can't verify
    # This is conservative - only penalize when we're confident fields are missing
    required_present = torch.ones(B, dtype=torch.bool, device=device)

    # Check if we have validated arguments from teacher
    validated_args = batch.get("validated_arguments")
    if validated_args is None:
        # No validation data available - assume complete (no penalty)
        return required_present

    # Full completeness check: generate text from student logits and validate
    # This checks if student output contains all required tool arguments/fields
    # by comparing against teacher's validated arguments and schema requirements.

    # Check if we have student logits to generate from
    student_logits = batch.get("student_logits")
    if student_logits is None:
        # Fallback to simple heuristic if logits not available
        gold_json_ids = batch.get("gold_json_text_ids")
        if gold_json_ids is not None:
            mask_valid_json = batch.get("mask_valid_json_tokens")
            if mask_valid_json is not None:
                student_attn_mask = batch.get("attention_mask")
                if student_attn_mask is not None:
                    json_length = min(gold_json_ids.size(
                        1), student_attn_mask.size(1))
                    student_covers_json = student_attn_mask[:, :json_length].sum(
                        dim=1) > 0
                    required_present = student_covers_json
        return required_present

    # Get tool names and schema registry if available
    tool_names = batch.get("tool_names", [])
    schema_registry = None
    try:
        from tools.schema_registry import ToolSchemaRegistry

        schema_registry = ToolSchemaRegistry()
    except ImportError:
        # Schema registry not available - fall back to generic validation
        # This is expected in some environments, so we silently continue
        pass
    except Exception as e:
        # Other errors should be logged but not block execution
        print(
            f"[distill_kd] WARN: Could not initialize schema registry: {e}, using generic validation")

    # Generate text from student logits for each batch item
    from training.extractors import extract_tool_call

    # Get predicted token IDs (greedy decoding)
    # student_logits shape: [B, T, V]
    pred_token_ids = student_logits.argmax(dim=-1)  # [B, T]

    # Process each batch item
    for i in range(B):
        try:
            # Decode tokens to text
            tokens = pred_token_ids[i].cpu().tolist()
            # Remove padding tokens (-100 or 0)
            tokens = [
                t
                for t in tokens
                if t
                not in [
                    -100,
                    0,
                    tokenizer.pad_token_id if hasattr(
                        tokenizer, "pad_token_id") else None,
                ]
            ]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)

            # Extract tool call from generated text
            student_tool_call = extract_tool_call(generated_text, tool_names)

            if student_tool_call is None:
                # No tool call found - mark as incomplete
                required_present[i] = False
                continue

            # Get tool name
            tool_name = student_tool_call.get("name")
            student_args = student_tool_call.get("arguments", {})

            if not tool_name:
                required_present[i] = False
                continue

            # Get teacher's validated arguments for comparison
            teacher_validated = None
            if validated_args is not None:
                # validated_args could be a list or dict
                if isinstance(validated_args, list) and i < len(validated_args):
                    teacher_validated = validated_args[i]
                elif isinstance(validated_args, dict):
                    teacher_validated = validated_args.get(
                        str(i)) or validated_args.get(i)

            # Validate against schema required fields
            if schema_registry:
                schema = schema_registry.get_schema(tool_name)
                if schema:
                    # Get required fields from schema
                    # Schema structure: {"properties": {"arguments": {"required": [...], ...}}}
                    required_fields = []
                    if "properties" in schema:
                        args_schema = schema["properties"].get("arguments", {})
                        if isinstance(args_schema, dict) and "required" in args_schema:
                            required_fields = args_schema["required"]
                        elif "required" in schema:
                            # Check if required fields are at top level
                            required_fields = schema["required"]

                    # Check if student has all required fields
                    missing_fields = []
                    for field in required_fields:
                        if field not in student_args:
                            missing_fields.append(field)

                    if missing_fields:
                        required_present[i] = False
                        continue

                    # Validate tool call against schema
                    is_valid, error = schema_registry.validate_tool_call(
                        tool_name, student_tool_call
                    )
                    if not is_valid:
                        required_present[i] = False
                        continue
                else:
                    # No schema found - use generic validation
                    if "name" not in student_tool_call or "arguments" not in student_tool_call:
                        required_present[i] = False
                        continue

            # Compare with teacher's validated arguments if available
            if teacher_validated is not None:
                teacher_args = (
                    teacher_validated.get("arguments", {})
                    if isinstance(teacher_validated, dict)
                    else {}
                )

                # Check if student has all fields that teacher has
                # (assuming teacher's arguments are complete)
                teacher_fields = set(teacher_args.keys())
                student_fields = set(student_args.keys())

                # Student should have at least the required fields from teacher
                # (teacher may have optional fields too)
                if schema_registry:
                    schema = schema_registry.get_schema(tool_name)
                    if schema:
                        args_schema = schema.get(
                            "properties", {}).get("arguments", {})
                        required_fields = set(args_schema.get("required", []))
                        # Check if student has all required fields
                        if not required_fields.issubset(student_fields):
                            required_present[i] = False
                            continue
                else:
                    # Fallback: check if student has all teacher's fields
                    # (conservative - teacher might have optional fields)
                    if teacher_fields and not teacher_fields.issubset(student_fields):
                        required_present[i] = False
                        continue

            # All checks passed - mark as complete
            required_present[i] = True

        except Exception as e:
            # On error, conservatively mark as incomplete
            # (better to penalize than to miss incomplete outputs)
            required_present[i] = False
            # Optionally log error for debugging (but don't spam during training)
            if hasattr(compute_required_fields_present, "_error_count"):
                compute_required_fields_present._error_count = (
                    getattr(compute_required_fields_present,
                            "_error_count", 0) + 1
                )
                if compute_required_fields_present._error_count <= 5:  # Log first 5 errors
                    print(
                        f"[distill_kd] WARN: Completeness check error for batch item {i}: {e}")

    return required_present


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(configs: list, env_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge multiple config files and apply environment variable overrides.

    Config is the source of truth; env vars are CLI overrides that write into resolved config.
    All overrides are logged for provenance tracking.

    Args:
        configs: List of config file paths
        env_overrides: Optional dict of env var overrides (will be merged into config)

    Returns:
        Merged config dict with provenance metadata
    """
    merged = {}
    for config_path in configs:
        config = load_config(config_path)
        # Deep merge
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value

    # Apply environment variable overrides (CLI overrides)
    if env_overrides is None:
        env_overrides = {}

    # Latent training override
    if os.getenv("TRAIN_LATENT", "0") == "1":
        if "latent" not in merged:
            merged["latent"] = {}
        merged["latent"]["enabled"] = True
        env_overrides["TRAIN_LATENT"] = "1"

    # Latent mode override
    if os.getenv("LATENT_MODE", "0") == "1":
        if "latent" not in merged:
            merged["latent"] = {}
        merged["latent"]["mode_enabled"] = True
        env_overrides["LATENT_MODE"] = "1"

    # Halt head override
    if os.getenv("HALT_HEAD", "0") == "1":
        if "latent" not in merged:
            merged["latent"] = {}
        merged["latent"]["halt_head_enabled"] = True
        env_overrides["HALT_HEAD"] = "1"

    # Store provenance metadata
    merged["_provenance"] = {
        "config_files": configs,
        "env_overrides": env_overrides,
        "final_config": merged.copy(),  # Snapshot of final resolved config
    }

    return merged


def create_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create student model from config."""
    arch_cfg = cfg.get("arch", {})

    # Validate that arch config exists and has minimum required fields
    # This prevents hanging on invalid configs during model initialization
    if not arch_cfg:
        raise KeyError("Missing required 'arch' configuration section")

    # Check for critical fields that would cause issues if missing
    # Note: We allow defaults for most fields, but validate structure
    if not isinstance(arch_cfg, dict):
        raise TypeError(f"Expected 'arch' to be a dict, got {type(arch_cfg)}")

    model_cfg = ModelCfg(
        d_model=arch_cfg.get("d_model", 4096),
        n_layers=arch_cfg.get("n_layers", 32),
        n_heads=arch_cfg.get("n_heads", 32),
        n_kv_heads=arch_cfg.get("n_kv_heads", 8),
        d_head=arch_cfg.get("d_head", 128),
        vocab_size=arch_cfg.get("vocab_size", 32000),
        rope_theta=arch_cfg.get("rope_theta", 10000.0),
        rope_scaling=arch_cfg.get("rope_scaling", "dynamic"),
        dropout=arch_cfg.get("dropout", 0.0),
    )

    # Enable self-evaluation head if configured
    use_self_evaluation = cfg.get("distillation", {}).get(
        "use_self_evaluation", False)
    model = StudentLM(model_cfg, use_self_evaluation=use_self_evaluation)

    # Load checkpoint if specified
    init_cfg = cfg.get("init", {})
    base_checkpoint = init_cfg.get("base_checkpoint")
    if base_checkpoint and Path(base_checkpoint).exists():
        try:
            print(f"[distill_kd] Loading checkpoint: {base_checkpoint}")
            from training.safe_checkpoint_loading import safe_load_checkpoint
            checkpoint = safe_load_checkpoint(
                base_checkpoint, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("[distill_kd] Checkpoint loaded")
        except Exception as e:
            print(
                f"[distill_kd] ERROR: Failed to load checkpoint {base_checkpoint}: {e}")
            import traceback

            traceback.print_exc()
            print("[distill_kd] Continuing with randomly initialized model")
    elif base_checkpoint:
        print(
            f"[distill_kd] WARN: Base checkpoint not found: {base_checkpoint}")

    model = model.to(device)

    # Initialize projection layers for intermediate layer matching if needed
    kd_cfg = cfg.get("distillation", {})
    if kd_cfg.get("use_intermediate_layers", False):
        layer_mapping = kd_cfg.get("layer_mapping", {})
        if layer_mapping:
            # Get teacher model dimensions from config
            teacher_d_model = cfg.get("teacher", {}).get(
                "d_model", model_cfg.d_model)

            if teacher_d_model != model_cfg.d_model:
                projection_layers = create_projection_layers(
                    student_d_model=model_cfg.d_model,
                    teacher_d_model=teacher_d_model,
                    layer_mapping=layer_mapping,
                    device=device,
                )
                # Store projection layers as model attribute for later use
                model.projection_layers = projection_layers
                print(
                    f"[distill_kd] Initialized {len(projection_layers)} projection layers for intermediate layer matching"
                )

    return model


def create_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config, including projection layers if present."""
    opt_cfg = cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 2e-4)
    betas = opt_cfg.get("betas", [0.9, 0.95])
    weight_decay = opt_cfg.get("weight_decay", 0.1)

    # Collect all parameters (model + projection layers if present)
    params = list(model.parameters())

    # Add projection layers to optimizer if they exist
    if hasattr(model, "projection_layers") and model.projection_layers:
        projection_params = [layer.parameters()
                             for layer in model.projection_layers.values()]
        # Flatten nested parameter lists
        for layer_params in projection_params:
            params.extend(list(layer_params))
        print(
            f"[distill_kd] Added {len(model.projection_layers)} projection layers to optimizer")

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer


def get_sequence_length(
    step: int, seq_lengths: list, curriculum_schedule: Optional[list] = None
) -> int:
    """
    Get current sequence length based on curriculum learning.

    Args:
        step: Current training step
        seq_lengths: List of available sequence lengths
        curriculum_schedule: List of step boundaries for each length

    Returns:
        Current sequence length to use
    """
    if curriculum_schedule is None:
        # Default: use longest available
        return max(seq_lengths)

    # Find which length to use based on step
    for i, boundary in enumerate(curriculum_schedule):
        if step < boundary:
            return seq_lengths[min(i, len(seq_lengths) - 1)]

    return seq_lengths[-1]


def sample_enumerated_shape(
    seq_lengths: list,
    shape_probs: Optional[list] = None,
    step: Optional[int] = None,
    periodic_upweight_rare: bool = True,
) -> int:
    """
    Sample sequence length from enumerated shapes using Dirichlet-like distribution.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 2

    Args:
        seq_lengths: List of available sequence lengths (e.g., [512, 1024, 2048, 4096])
        shape_probs: Optional probability distribution over shapes (default: production mix)
        step: Current training step (for periodic upweighting)
        periodic_upweight_rare: If True, periodically upweight rare shapes

    Returns:
        Sampled sequence length
    """
    if shape_probs is None:
        # Default production mix: 0.5:0.3:0.15:0.05 for 4 shapes
        # Adjust based on your production distribution
        n = len(seq_lengths)
        if n == 4:
            shape_probs = [0.5, 0.3, 0.15, 0.05]
        elif n == 3:
            shape_probs = [0.6, 0.3, 0.1]
        elif n == 2:
            shape_probs = [0.7, 0.3]
        else:
            # Uniform fallback
            shape_probs = [1.0 / n] * n

    # Normalize probabilities
    total = sum(shape_probs)
    shape_probs = [p / total for p in shape_probs]

    # Periodic upweighting for rare shapes (every 100 steps)
    if periodic_upweight_rare and step is not None:
        if step % 100 == 0:
            # Temporarily increase probability of smallest shape
            adjusted_probs = shape_probs.copy()
            adjusted_probs[-1] *= 2.0  # Double probability of smallest shape
            total_adj = sum(adjusted_probs)
            adjusted_probs = [p / total_adj for p in adjusted_probs]
            shape_probs = adjusted_probs

    # Sample according to distribution
    return random.choices(seq_lengths, weights=shape_probs, k=1)[0]


def should_enable_qat(step: int, total_steps: int, qat_cfg: Dict[str, Any]) -> bool:
    """
    Check if QAT should be enabled at current step.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 3

    Args:
        step: Current training step
        total_steps: Total training steps
        qat_cfg: QAT configuration dict

    Returns:
        True if QAT should be enabled
    """
    if not qat_cfg.get("enabled", False):
        return False

    start_fraction = qat_cfg.get("start_fraction", 0.8)  # Default: last 20%
    start_step = int(total_steps * start_fraction)
    return step >= start_step


def apply_qat_to_model(
    model: nn.Module,
    qat_cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    Apply quantization-aware training to model.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 3

    Args:
        model: Model to quantize
        qat_cfg: QAT configuration dict
        device: Device to use

    Returns:
        Quantized model
    """
    if not QAT_AVAILABLE:
        raise RuntimeError("QAT not available. Install required dependencies.")

    weight_bits = qat_cfg.get("weight_bits", 8)
    act_bits = qat_cfg.get("act_bits", 8)
    fake_quant_in_attention = qat_cfg.get("fake_quant_in_attention", True)
    clamp_pre_softmax = qat_cfg.get("clamp_pre_softmax", True)

    print(
        f"[distill_kd] Applying QAT: weight_bits={weight_bits}, act_bits={act_bits}")
    quantized_model = quantize_model(
        model,
        weight_bits=weight_bits,
        act_bits=act_bits,
        fake_quant_in_attention=fake_quant_in_attention,
        clamp_pre_softmax=clamp_pre_softmax,
    )
    return quantized_model.to(device)


def check_qat_stability(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    baseline_model: Optional[nn.Module] = None,
    baseline_hidden_states: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Check QAT stability: cosine similarity probes and NaN detection.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 3

    Args:
        model: Model to check (quantized model)
        batch: Batch for forward pass
        device: Device to use
        baseline_model: Optional baseline model (pre-quantization) for comparison
        baseline_hidden_states: Optional pre-computed baseline hidden states

    Returns:
        Dictionary with stability metrics
    """
    model.eval()
    with torch.no_grad():
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass to get logits
            logits = model(input_ids, attention_mask)

            # Check for NaNs
            has_nan = torch.isnan(logits).any().item()

            # Compute cosine similarity on probe layers
            cosine_sim = 1.0  # Default: perfect similarity if no baseline
            cosine_sim_per_layer = []

            # Check if model supports return_hidden_states
            if hasattr(model, "forward"):
                try:
                    # Extract hidden states from current model
                    if isinstance(
                        model, (torch.nn.DataParallel,
                                torch.nn.parallel.DistributedDataParallel)
                    ):
                        # Unwrap DDP/DP model
                        actual_model = model.module
                    else:
                        actual_model = model

                    # Try to get hidden states from current model
                    if hasattr(actual_model, "forward"):
                        # Check if forward supports return_hidden_states
                        import inspect

                        sig = inspect.signature(actual_model.forward)
                        supports_hidden_states = "return_hidden_states" in sig.parameters

                        if supports_hidden_states:
                            # Extract current hidden states
                            current_outputs = actual_model(
                                input_ids, attn_mask=attention_mask, return_hidden_states=True
                            )
                            if isinstance(current_outputs, tuple) and len(current_outputs) >= 2:
                                _current_logits, current_hidden_states = (
                                    current_outputs[0],
                                    current_outputs[1],
                                )
                            else:
                                current_hidden_states = None
                        else:
                            current_hidden_states = None
                    else:
                        current_hidden_states = None

                    # Get baseline hidden states
                    baseline_states = None
                    if baseline_hidden_states is not None:
                        # Use pre-computed baseline hidden states
                        baseline_states = baseline_hidden_states
                    elif baseline_model is not None:
                        # Extract from baseline model
                        baseline_model.eval()
                        if isinstance(
                            baseline_model,
                            (torch.nn.DataParallel,
                             torch.nn.parallel.DistributedDataParallel),
                        ):
                            baseline_actual = baseline_model.module
                        else:
                            baseline_actual = baseline_model

                        if hasattr(baseline_actual, "forward"):
                            sig = inspect.signature(baseline_actual.forward)
                            if "return_hidden_states" in sig.parameters:
                                baseline_outputs = baseline_actual(
                                    input_ids, attn_mask=attention_mask, return_hidden_states=True
                                )
                                if (
                                    isinstance(baseline_outputs, tuple)
                                    and len(baseline_outputs) >= 2
                                ):
                                    baseline_states = baseline_outputs[1]
                        baseline_model.train()

                    # Compute cosine similarity per layer
                    if current_hidden_states is not None and baseline_states is not None:
                        # Ensure same number of layers
                        num_layers = min(
                            len(current_hidden_states), len(baseline_states))
                        if num_layers > 0:
                            similarities = []
                            for i in range(num_layers):
                                # [B, T, D]
                                current_hs = current_hidden_states[i]
                                baseline_hs = baseline_states[i]  # [B, T, D]

                                # Flatten to [B*T, D] for cosine similarity
                                B, T, D = current_hs.shape
                                current_flat = current_hs.view(
                                    B * T, D)  # [B*T, D]
                                baseline_flat = baseline_hs.view(
                                    B * T, D)  # [B*T, D]

                                # Compute cosine similarity: dot product / (norm1 * norm2)
                                # Average over sequence positions
                                current_norm = torch.norm(
                                    current_flat, dim=-1, keepdim=True
                                )  # [B*T, 1]
                                baseline_norm = torch.norm(
                                    baseline_flat, dim=-1, keepdim=True
                                )  # [B*T, 1]

                                # Avoid division by zero
                                eps = 1e-8
                                current_norm = current_norm.clamp(min=eps)
                                baseline_norm = baseline_norm.clamp(min=eps)

                                # Normalize
                                current_normalized = current_flat / \
                                    current_norm  # [B*T, D]
                                baseline_normalized = baseline_flat / \
                                    baseline_norm  # [B*T, D]

                                # Cosine similarity: dot product of normalized vectors
                                cosine_per_token = (current_normalized * baseline_normalized).sum(
                                    dim=-1
                                )  # [B*T]
                                layer_sim = (
                                    cosine_per_token.mean().item()
                                )  # Average over all tokens
                                similarities.append(layer_sim)
                                cosine_sim_per_layer.append(layer_sim)

                            # Aggregate: mean of per-layer similarities
                            if similarities:
                                cosine_sim = sum(similarities) / \
                                    len(similarities)
                            else:
                                cosine_sim = 1.0
                        else:
                            cosine_sim = 1.0  # No layers to compare
                    elif current_hidden_states is not None:
                        # No baseline available - can't compute similarity
                        cosine_sim = 1.0  # Default: assume stable
                except Exception as e:
                    # If hidden state extraction fails, fall back to default
                    # Log the error (this function is called infrequently, so logging is acceptable)
                    print(
                        f"[distill_kd] WARN: Hidden state extraction failed, assuming stability: {e}")
                    cosine_sim = 1.0

            return {
                "qat_stability.has_nan": float(has_nan),
                "qat_stability.cosine_sim": cosine_sim,
                "qat_stability.cosine_sim_per_layer": cosine_sim_per_layer
                if cosine_sim_per_layer
                else None,
            }
        except Exception as e:
            return {
                "qat_stability.has_nan": 1.0,
                "qat_stability.cosine_sim": 0.0,
                "qat_stability.error": str(e),
            }
        finally:
            model.train()


def truncate_batch_to_shape(
    batch: Dict[str, torch.Tensor], target_length: int
) -> Dict[str, torch.Tensor]:
    """
    Truncate batch tensors to target sequence length.

    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 2

    Args:
        batch: Batch dictionary with tensors to truncate
        target_length: Target sequence length

    Returns:
        Truncated batch dictionary
    """
    truncated = {}

    # List of keys that should be truncated (sequence dimension is dim 1)
    seq_keys = [
        "input_ids",
        "attention_mask",
        "labels",
        "teacher_target_ids",
        "tool_name_ids",
        "tool_name_mask",
        "gold_json_text_ids",
        "mask_valid_json_tokens",
        "tool_result_fields",
        "integration_mask",
        "teacher_attention_mask",
    ]

    # List of keys that should be truncated (sequence dimension is dim 1, vocab is dim 2)
    seq_vocab_keys = ["teacher_logits"]

    for key, value in batch.items():
        if key in seq_keys:
            # Truncate along sequence dimension (dim 1)
            if value.size(1) > target_length:
                truncated[key] = value[:, :target_length]
            else:
                truncated[key] = value
        elif key in seq_vocab_keys:
            # Truncate along sequence dimension (dim 1), keep vocab (dim 2)
            if value.size(1) > target_length:
                truncated[key] = value[:, :target_length, :]
            else:
                truncated[key] = value
        else:
            # Keep other keys as-is (metadata, etc.)
            truncated[key] = value

    return truncated


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: Path,
    config: Dict[str, Any],
    loss_dict: Optional[Dict[str, Any]] = None,
    rng_states: Optional[Dict[str, Any]] = None,
    data_shard_index: Optional[int] = None,
    dataset_fingerprint: Optional[str] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save state
        step: Current training step
        loss: Current loss value
        output_dir: Directory to save checkpoint
        config: Training configuration
        loss_dict: Optional loss component breakdown
        rng_states: Optional RNG states (if None, will capture current states)
        data_shard_index: Optional data shard index for multi-shard datasets
        dataset_fingerprint: Optional dataset fingerprint for validation
        scaler: Optional GradScaler for FP16 training
    """
    from training.utils import sha256_state_dict

    # Validate output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(
            f"Cannot create checkpoint directory {output_dir}: {e}") from e

    # Check available disk space (rough estimate: checkpoint ~500MB-2GB)
    import shutil
    try:
        stat = shutil.disk_usage(output_dir)
        free_space_gb = stat.free / (1024**3)
        if free_space_gb < 1.0:  # Less than 1GB free
            print(
                f"[distill_kd] WARN: Low disk space: {free_space_gb:.2f}GB free")
    except Exception:
        pass  # Disk space check is best-effort

    # Get model state dict (unwrap DDP if needed)
    model_state = model.module.state_dict() if isinstance(
        model, DDP) else model.state_dict()

    # Compute SHA256 hash of model state for reproducibility tracking
    state_sha256 = sha256_state_dict(model_state)

    # Extract code-mode metadata (always persist, even when disabled)
    distill_cfg = config.get("distill", {})
    code_mode_cfg = distill_cfg.get("code_mode", {})
    train_code_mode = code_mode_cfg.get("enabled", False)

    code_mode_meta = {
        "enabled": train_code_mode,
        "weight": code_mode_cfg.get("weight", 0.3),
        "weight_schedule": code_mode_cfg.get("weight_schedule", {}),
        "eligibility": code_mode_cfg.get("code_mode_pref", {}).get("eligibility_rules", {}),
    }

    # Extract loss component breakdown for checkpoint metadata
    loss_components = {}
    if loss_dict is not None:
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_components[key] = float(value.item())
            elif isinstance(value, (int, float)):
                loss_components[key] = float(value)

    # Capture RNG states if not provided
    if rng_states is None:
        import random
        import numpy as np

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

    # Capture scaler state if using FP16
    scaler_state = None
    if scaler is not None:
        try:
            scaler_state = scaler.state_dict()
        except Exception as e:
            print(f"[distill_kd] WARN: Failed to capture scaler state: {e}")

    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
        "model_arch": {
            "use_halt_head": getattr(model, "use_halt_head", False),
            "use_self_evaluation": getattr(model, "use_self_evaluation", False),
        },
        "meta": {
            "sha256_state": state_sha256,
            "code_mode": code_mode_meta,
            "loss_components": loss_components,
            "rng_states": rng_states,
            "data_shard_index": data_shard_index,
            "dataset_fingerprint": dataset_fingerprint,
            "scaler_state_dict": scaler_state,
        },
    }

    try:
        import os

        # Atomic checkpoint write: temp file → fsync → atomic rename
        def save_atomic(data: dict, path: Path):
            """Save checkpoint atomically using temp file + fsync + rename."""
            # Create temp file in same directory for atomic rename
            temp_path = path.parent / f".{path.name}.tmp"

            # Save to temp file
            torch.save(data, temp_path)

            # Flush and fsync to ensure data is written to disk
            # Note: torch.save doesn't expose file handle, so we use os.fsync on the file
            # For atomicity, we rely on the filesystem's atomic rename operation
            try:
                # On Unix systems, we can fsync the directory to ensure metadata is written
                if hasattr(os, "sync"):
                    os.sync()
            except Exception:
                pass  # fsync not critical, atomic rename is the key

            # Atomic rename (atomic on POSIX systems)
            temp_path.replace(path)

        # Save latest checkpoint atomically
        latest_path = output_dir / "latest.pt"
        save_atomic(checkpoint, latest_path)

        # Save numbered checkpoint atomically
        checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
        save_atomic(checkpoint, checkpoint_path)

        print(f"[distill_kd] Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        print(
            f"[distill_kd] ERROR: Failed to save checkpoint at step {step}: {e}")
        import traceback

        traceback.print_exc()
        print("[distill_kd] Continuing training without saving checkpoint")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    device: torch.device,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    grad_accum_counter: int = 0,
    current_step: int = 0,
    code_mode_loss_module: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Single training step.

    Returns:
        Dictionary with loss values
    """
    model.train()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Vocab clamping: ensure token IDs fit within model vocab_size
    # This catches tokenizer/model vocab mismatches early
    model_vocab_size = cfg.get("arch", {}).get("vocab_size", 32000)
    pre_violate = (input_ids.ge(model_vocab_size)
                   | input_ids.lt(0)).any().item()
    if pre_violate:
        input_ids = torch.clamp(input_ids, 0, model_vocab_size - 1)
        labels = torch.clamp(labels, 0, model_vocab_size - 1)
        # Warn periodically (every 50 steps) to avoid log spam
        if current_step % 50 == 0:
            print(
                f"[distill_kd] ⚠ Step {current_step}: clamped token ids to vocab_size={model_vocab_size}",
                flush=True,
            )

    teacher_target_ids = batch.get("teacher_target_ids")
    if teacher_target_ids is not None:
        teacher_target_ids = teacher_target_ids.to(device)

    teacher_logits = batch.get("teacher_logits")
    if teacher_logits is not None:
        teacher_logits = teacher_logits.to(device)

    # CoT-free validation: Fail if reasoning_content detected
    teacher_reasoning_content = batch.get("teacher_reasoning_content")
    if teacher_reasoning_content is not None:
        if isinstance(teacher_reasoning_content, list):
            teacher_reasoning_content = next(
                (rc for rc in teacher_reasoning_content if rc is not None), None
            )
        if teacher_reasoning_content:
            raise ValueError(
                "CoT-free training: teacher_reasoning_content detected in batch. "
                "Training on reasoning_content violates ToS. Use process-step supervision instead."
            )

    # Forward pass
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        kd_cfg = cfg.get("distillation", {})

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Intermediate Layer Matching
        # ====================================================================
        # Enable intermediate layer matching if configured
        use_intermediate_layers = kd_cfg.get("use_intermediate_layers", False)
        return_hidden_states = use_intermediate_layers

        # Get student logits (and optionally hidden states)
        if return_hidden_states:
            student_logits, student_hidden_states = model(
                input_ids, attention_mask, return_hidden_states=True
            )  # [B, T, V], List[[B, T, D]]
        else:
            student_logits = model(input_ids, attention_mask)  # [B, T, V]
            student_hidden_states = None

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Self-Evaluation Head
        # ====================================================================
        # Enable self-evaluation if configured
        use_self_eval = kd_cfg.get("use_self_evaluation", False)
        return_eval_score = use_self_eval

        # Enable halt head if configured
        latent_cfg = cfg.get("latent", {})
        use_halt_head = latent_cfg.get("halt_head_enabled", False)
        return_halt_logits = use_halt_head

        if return_eval_score:
            # Re-run forward pass if we need eval score but didn't get it above
            if not return_hidden_states:
                student_logits, eval_score = model(
                    input_ids, attention_mask, return_eval_score=True
                )  # [B, T, V], [B, 1]
            else:
                # Get eval score separately (model doesn't support both flags simultaneously)
                _, eval_score = model(
                    input_ids, attention_mask, return_eval_score=True)  # [B, 1]
        else:
            eval_score = None

        # Get halt logits if enabled
        halt_logits = None
        if return_halt_logits:
            # Get halt logits from model (use pooled hidden state)
            if hasattr(model, "use_halt_head") and model.use_halt_head:
                # Forward pass with halt logits
                if return_hidden_states or return_eval_score:
                    # Need separate forward pass for halt logits
                    _, halt_logits = model(
                        input_ids, attention_mask, return_halt_logits=True
                    )  # [B, T, V], [B, 2]
                    # Extract halt logits (model returns tuple)
                    if isinstance(halt_logits, tuple):
                        # Last element is halt logits
                        halt_logits = halt_logits[-1]
                else:
                    student_logits, halt_logits = model(
                        input_ids, attention_mask, return_halt_logits=True
                    )  # [B, T, V], [B, 2]
                    if isinstance(halt_logits, tuple):
                        halt_logits = halt_logits[-1]

        # Compute loss

        # Get tokenizer for process-step supervision, CAWS compliance, claim extraction,
        # length-aware KD, and early tool call loss
        tokenizer = None
        needs_tokenizer = (
            kd_cfg.get("w_tool", 0.0) > 0
            or kd_cfg.get("w_args", 0.0) > 0
            or kd_cfg.get("use_caws_compliance", False)
            or kd_cfg.get("use_claim_extraction", False)
            or kd_cfg.get("use_length_aware_kd", False)
            or kd_cfg.get("use_early_tool_call_loss", False)
        )
        if needs_tokenizer:
            # Try to get tokenizer from model or config
            if hasattr(model, "tokenizer"):
                tokenizer = model.tokenizer
            elif hasattr(model, "module") and hasattr(model.module, "tokenizer"):
                tokenizer = model.module.tokenizer
            elif "tokenizer_path" in cfg:
                from training.dataset import load_tokenizer

                tokenizer = load_tokenizer(cfg["tokenizer_path"])

        # PRIORITY 2: Entropy-based scheduling (data-driven, replaces linear schedules)
        use_entropy_scheduling = kd_cfg.get("use_entropy_scheduling", False)

        if use_entropy_scheduling and teacher_logits is not None:
            # Compute entropy from teacher logits and derive temperature/weights
            from training.losses import entropy_weighting

            current_temperature, entropy_weights = entropy_weighting(
                teacher_logits=teacher_logits,
                min_entropy=kd_cfg.get("min_entropy", 2.0),
                max_entropy=kd_cfg.get("max_entropy", 8.0),
                min_temp=kd_cfg.get("min_temperature", 1.5),
                max_temp=kd_cfg.get("max_temperature", 3.0),
                base_kl_weight=kd_cfg.get("kl_weight", 0.4),
                base_ce_teacher_weight=kd_cfg.get("ce_teacher_weight", 0.2),
                base_ce_gt_weight=kd_cfg.get("ce_ground_truth_weight", 0.2),
            )

            kl_weight = entropy_weights["kl_weight"]
            ce_teacher_weight = entropy_weights["ce_teacher_weight"]
            ce_ground_truth_weight = entropy_weights["ce_ground_truth_weight"]
            entropy_value = entropy_weights.get("entropy", 0.0)

            # Log entropy for monitoring
            if current_step % 100 == 0:
                print(
                    f"[distill_kd] Entropy: {entropy_value:.3f}, Temp: {current_temperature:.3f}, "
                    f"KL: {kl_weight:.3f}, CE_GT: {ce_ground_truth_weight:.3f}"
                )
        else:
            # Fallback to linear schedules if entropy scheduling disabled or teacher_logits unavailable
            # Calculate adaptive temperature if enabled
            current_temperature = cfg.get("kd", {}).get("kd_temperature", 2.0)
            if kd_cfg.get("use_temperature_schedule", False):
                from training.losses import adaptive_temperature

                total_steps = cfg.get("train", {}).get("total_steps", 100000)
                # Use current_step for scheduling (actual training step, not grad accum counter)
                current_temperature = adaptive_temperature(
                    step=current_step,
                    total_steps=total_steps,
                    base_temp=kd_cfg.get("base_temperature", 2.0),
                    min_temp=kd_cfg.get("min_temperature", 1.5),
                    max_temp=kd_cfg.get("max_temperature", 3.0),
                )

            # Calculate adaptive loss weights if enabled
            # Optimized distillation weights based on toy model experiments
            # Increased for teacher alignment
            kl_weight = kd_cfg.get("kl_weight", 0.6)
            ce_teacher_weight = kd_cfg.get(
                "ce_teacher_weight", 0.2)  # Reduced to balance
            ce_ground_truth_weight = kd_cfg.get(
                "ce_ground_truth_weight", 0.2)  # Keep ground truth

            if kd_cfg.get("use_weight_schedule", False):
                from training.losses import loss_weight_schedule

                total_steps = cfg.get("train", {}).get("total_steps", 100000)
                # Use current_step for scheduling (actual training step, not grad accum counter)
                weights = loss_weight_schedule(
                    step=current_step,
                    total_steps=total_steps,
                    early_teacher_weight=kd_cfg.get(
                        "early_teacher_weight", 0.7),
                    late_teacher_weight=kd_cfg.get("late_teacher_weight", 0.3),
                    early_gt_weight=kd_cfg.get("early_gt_weight", 0.3),
                    late_gt_weight=kd_cfg.get("late_gt_weight", 0.7),
                )
                kl_weight = weights["kl_weight"]
                ce_teacher_weight = weights["ce_teacher_weight"]
                ce_ground_truth_weight = weights["ce_ground_truth_weight"]

        # Process-step supervision weights (replaces ce_reasoning_weight)
        w_tool = kd_cfg.get("w_tool", 0.15)
        w_args = kd_cfg.get("w_args", 0.15)
        w_integr = kd_cfg.get("w_integr", 0.10)

        # Extract process-step supervision targets from batch
        tool_name_ids = batch.get("tool_name_ids")
        tool_name_mask = batch.get("tool_name_mask")
        gold_json_text_ids = batch.get("gold_json_text_ids")
        mask_valid_json_tokens = batch.get("mask_valid_json_tokens")
        tool_result_fields = batch.get("tool_result_fields")
        integration_mask = batch.get("integration_mask")

        # Move to device if present
        if tool_name_ids is not None:
            tool_name_ids = tool_name_ids.to(device)
        if tool_name_mask is not None:
            tool_name_mask = tool_name_mask.to(device)
        if gold_json_text_ids is not None:
            gold_json_text_ids = gold_json_text_ids.to(device)
        if mask_valid_json_tokens is not None:
            mask_valid_json_tokens = mask_valid_json_tokens.to(device)
        if tool_result_fields is not None:
            tool_result_fields = tool_result_fields.to(device)
        if integration_mask is not None:
            integration_mask = integration_mask.to(device)

        # Code-mode preference loss (differentiable, token-level)
        # Module initialized once in main() and passed via closure or global
        code_mode_loss_tensor = None
        code_mode_weight = 0.0

        # Check if code-mode enabled (config is single source of truth)
        code_mode_cfg = cfg.get("distill", {}).get("code_mode", {})
        train_code_mode = code_mode_cfg.get("enabled", False)

        if train_code_mode:
            code_mode_weight_base = code_mode_cfg.get("weight", 0.3)

            # Apply weight scheduling (warmup)
            weight_schedule = code_mode_cfg.get("weight_schedule", {})
            warmup_steps = weight_schedule.get("warmup_steps", 5000)
            start_weight = weight_schedule.get("start_weight", 0.1)

            if current_step < warmup_steps and warmup_steps > 0:
                # Linear warmup: lerp(w_start, w_target, step / warmup_steps)
                progress = current_step / warmup_steps
                code_mode_weight = start_weight + \
                    (code_mode_weight_base - start_weight) * progress
            else:
                code_mode_weight = code_mode_weight_base

            # Initialize loss_dict if not yet defined
            if "loss_dict" not in locals():
                loss_dict = {}

            # Add weight to loss_dict for logging
            loss_dict["code_mode_weight"] = code_mode_weight

            if code_mode_weight > 0:
                code_mode_pref_cfg = code_mode_cfg.get("code_mode_pref", {})
                eligibility_rules = code_mode_pref_cfg.get(
                    "eligibility_rules",
                    {
                        "min_tools": 2,
                        "min_intermediate_chars": 10000,
                        "pii_patterns": ["EMAIL", "PHONE", "SSN"],
                    },
                )
                reward_cfg = code_mode_pref_cfg.get(
                    "reward",
                    {
                        "prefer_ts_api_over_direct_tool": True,
                        "penalize_tool_result_roundtrip": True,
                    },
                )
                loss_weights = code_mode_pref_cfg.get(
                    "weights", {"pos": 1.0, "neg": 1.0})

                # Get vocabulary IDs for markers (requires tokenizer)
                vocab_ids = {}
                if tokenizer is not None:
                    marker_tokens = [
                        "import",
                        "from",
                        "callMCPTool",
                        "await",
                        "tool_call",
                        "tool_result",
                    ]
                    for marker in marker_tokens:
                        try:
                            if hasattr(tokenizer, "convert_tokens_to_ids"):
                                tok_id = tokenizer.convert_tokens_to_ids(
                                    marker)
                            else:
                                encoded = tokenizer.encode(
                                    marker, add_special_tokens=False)
                                tok_id = encoded[0] if encoded else None

                            if tok_id is not None and tok_id < student_logits.size(-1):
                                vocab_ids[marker] = tok_id
                        except Exception as e:
                            # Marker token lookup failed - continue without this marker
                            # This is non-critical as code-mode loss can work without all markers
                            if current_step % 100 == 0:  # Log occasionally to avoid spam
                                print(
                                    f"[distill_kd] WARN: Could not find token ID for marker '{marker}': {e}")

                # Code-mode loss module should be initialized in main() and passed as parameter
                # Fallback initialization only if not provided (for backward compatibility)
                if code_mode_loss_module is None:
                    # Fallback: initialize on first call if not provided
                    # This maintains backward compatibility but is not recommended
                    code_mode_loss_module = CodeModePreferenceLoss(
                        eligibility_rules=eligibility_rules,
                        reward=reward_cfg,
                        vocab_ids=vocab_ids,
                        weights=loss_weights,
                    ).to(device)

                # Get batch metadata and span targets (NO DECODING - pre-computed during dataset generation)
                batch_meta = batch.get("meta", {})
                span_targets = None

                # Extract span targets from metadata if available (pre-computed during dataset generation)
                if isinstance(batch_meta, list):
                    span_targets = [m.get("span_targets", {})
                                    for m in batch_meta]
                elif isinstance(batch_meta, dict):
                    if "span_targets" in batch_meta:
                        span_targets = [batch_meta["span_targets"]]
                    else:
                        # Try to get from nested structure
                        span_targets = batch_meta.get(
                            "span_targets_list", None)

                # Compute code-mode loss (differentiable, token-level, NO TEXT DECODING)
                code_mode_loss_tensor = code_mode_loss_module(
                    student_logits=student_logits,
                    span_targets=span_targets,
                    batch_meta=batch_meta,
                )

        # Get loss mask if available (from latent curriculum)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        # Derive halt targets if halt head is enabled
        halt_targets = None
        halt_weight = 0.0
        if use_halt_head and halt_logits is not None:
            # Import halt targets derivation
            try:
                from training.halt_targets import HaltHeadTargets, create_halt_targets_batch

                # Get batch metadata for halt target derivation
                batch_metadata = batch.get("metadata", [])
                if not isinstance(batch_metadata, list):
                    batch_metadata = [batch_metadata] * input_ids.size(0)

                # Create halt targets instance
                halt_targets_cfg = latent_cfg.get("halt_targets", {})
                halt_targets_deriver = HaltHeadTargets(
                    judge_score_threshold=halt_targets_cfg.get(
                        "judge_score_threshold", 0.8),
                    delta_shrinking_threshold=halt_targets_cfg.get(
                        "delta_shrinking_threshold", 0.05
                    ),
                    caws_tier=halt_targets_cfg.get("caws_tier", "Tier-2"),
                    warmup_steps=halt_targets_cfg.get("warmup_steps", 1000),
                )

                # Derive halt targets
                halt_targets = create_halt_targets_batch(
                    batch_metadata,
                    halt_targets_deriver,
                    current_step,
                )

                if halt_targets is not None:
                    halt_targets = halt_targets.to(device)
                    # Get halt weight with warmup schedule
                    halt_weight_base = latent_cfg.get("halt_weight", 0.05)
                    if current_step < halt_targets_deriver.warmup_steps:
                        halt_weight = 0.0  # No loss during warmup
                    else:
                        halt_weight = halt_weight_base
            except ImportError:
                if current_step % 100 == 0:
                    print(
                        "[distill_kd] WARN: Halt targets not available, skipping halt loss")

        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_target_ids,
            ground_truth_targets=labels,
            # Process-step supervision targets (replaces teacher_reasoning_content)
            tool_name_ids=tool_name_ids,
            tool_name_mask=tool_name_mask,
            gold_json_text_ids=gold_json_text_ids,
            mask_valid_json_tokens=mask_valid_json_tokens,
            tool_result_fields=tool_result_fields,
            integration_mask=integration_mask,
            # Loss weights
            kl_weight=kl_weight,
            ce_teacher_weight=ce_teacher_weight,
            w_tool=w_tool,
            w_args=w_args,
            w_integr=w_integr,
            ce_ground_truth_weight=ce_ground_truth_weight,
            kd_temperature=current_temperature,
            # Code-mode preference loss (optional)
            code_mode_loss=code_mode_loss_tensor,
            code_mode_weight=code_mode_weight if train_code_mode else 0.0,
            # Latent curriculum loss mask (optional)
            loss_mask=loss_mask,
            # Halt head loss (optional)
            halt_logits=halt_logits,
            halt_targets=halt_targets,
            halt_weight=halt_weight,
        )

        # ====================================================================
        # LATENCY-AWARE LOSSES: Length-aware KD + Early Tool Call
        # ====================================================================
        # Length-aware KD loss: penalize extra student length only when required fields missing
        use_length_kd = kd_cfg.get("use_length_aware_kd", False)
        length_kd_weight = kd_cfg.get("length_kd_weight", 0.05)

        if use_length_kd and length_kd_weight > 0:
            # Get attention masks for length computation
            student_attn_mask = attention_mask
            teacher_attn_mask = batch.get("teacher_attention_mask")

            if teacher_attn_mask is not None:
                teacher_attn_mask = teacher_attn_mask.to(device)

                # Compute required_fields_present
                # If tokenizer is None or compute_required_fields_present fails, default to all fields present
                try:
                    if tokenizer is not None:
                        required_fields = compute_required_fields_present(
                            batch=batch,
                            tokenizer=tokenizer,
                            device=device,
                            student_logits=student_logits,
                        )
                    else:
                        # No tokenizer available - assume all fields present (conservative)
                        B = student_attn_mask.size(0)
                        required_fields = torch.ones(
                            B, dtype=torch.bool, device=device)
                except Exception as e:
                    # Fallback: assume all fields present (conservative - no penalty)
                    # Log occasionally to avoid spam during training
                    if current_step % 1000 == 0:
                        print(
                            f"[distill_kd] WARN: Required fields computation failed, assuming all present: {e}")
                    B = student_attn_mask.size(0)
                    required_fields = torch.ones(
                        B, dtype=torch.bool, device=device)

                # Compute length-aware KD loss
                length_kd_loss, length_diags = length_aware_kd_loss(
                    student_attn_mask=student_attn_mask,
                    teacher_attn_mask=teacher_attn_mask,
                    required_fields_present=required_fields,
                    hinge=kd_cfg.get("length_kd_hinge", 0.15),
                    slope=kd_cfg.get("length_kd_slope", 1.0),
                    reduction="mean",
                )

                # Add to loss dict
                loss_dict["length_kd"] = length_kd_loss
                loss_dict["total"] = loss_dict["total"] + \
                    length_kd_weight * length_kd_loss

                # Add diagnostics to loss dict for logging
                loss_dict.update(length_diags)

        # Early tool call loss: encourage valid tool JSON within first N tokens
        use_early_tool = kd_cfg.get("use_early_tool_call_loss", False)
        early_tool_weight = kd_cfg.get("early_tool_weight", 0.05)

        if use_early_tool and early_tool_weight > 0 and tokenizer is not None:
            # Get tool_should_be_used from batch metadata
            tool_should_be_used = batch.get("tool_should_be_used")

            if tool_should_be_used is not None:
                tool_should_be_used = tool_should_be_used.to(device)

                # Get teacher prefix IDs if available (first N tokens of teacher's tool JSON)
                teacher_prefix_ids = batch.get("teacher_prefix_ids")
                if teacher_prefix_ids is not None:
                    teacher_prefix_ids = teacher_prefix_ids.to(device)

                # Compute ramp_t (linear 0→1 over warmup epochs)
                warmup_epochs = kd_cfg.get("early_tool_warmup_epochs", 5)
                total_epochs = cfg.get("train", {}).get("total_epochs", 100)
                current_epoch = current_step // (
                    cfg.get("train", {}).get("steps_per_epoch", 1000))
                ramp_t = (
                    min(1.0, max(0.0, current_epoch / warmup_epochs)
                        ) if warmup_epochs > 0 else 1.0
                )

                # Compute early tool call loss
                early_tool_loss, early_tool_diags = early_tool_call_loss(
                    logits=student_logits,
                    input_ids=input_ids,
                    tool_should_be_used=tool_should_be_used,
                    tokenizer=tokenizer,
                    teacher_prefix_ids=teacher_prefix_ids,
                    N=kd_cfg.get("early_tool_N", 25),
                    json_prior_weight=kd_cfg.get(
                        "early_tool_json_prior_weight", 0.02),
                    ce_weight=kd_cfg.get("early_tool_ce_weight", 0.2),
                    ramp_t=ramp_t,
                )

                # Add to loss dict
                loss_dict["early_tool"] = early_tool_loss
                loss_dict["total"] = loss_dict["total"] + \
                    early_tool_weight * early_tool_loss

                # Add diagnostics to loss dict for logging
                loss_dict.update(early_tool_diags)

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Intermediate Layer Loss
        # ====================================================================
        if use_intermediate_layers and student_hidden_states is not None:
            # Get teacher hidden states from batch (if available)
            # NOTE: Teacher hidden states must be extracted during dataset generation
            # or via a teacher model wrapper that exposes hidden states
            teacher_hidden_states = batch.get("teacher_hidden_states")

            if teacher_hidden_states is not None:
                # Move to device if tensor
                if isinstance(teacher_hidden_states, list):
                    teacher_hidden_states = [
                        h.to(device) if isinstance(h, torch.Tensor) else h
                        for h in teacher_hidden_states
                    ]
                elif isinstance(teacher_hidden_states, torch.Tensor):
                    teacher_hidden_states = teacher_hidden_states.to(device)

                # Layer mapping: student_layer_idx -> teacher_layer_idx
                # Example: Map student layers [0, 8, 16, 24] to teacher layers [0, 16, 32, 48]
                layer_mapping = kd_cfg.get("layer_mapping", {})
                if not layer_mapping:
                    # Default: Map evenly spaced student layers to teacher layers
                    # Assuming teacher has ~2x more layers than student
                    n_student_layers = len(
                        student_hidden_states) - 1  # Exclude embedding
                    n_teacher_layers = (
                        len(teacher_hidden_states) - 1
                        if isinstance(teacher_hidden_states, list)
                        else 0
                    )
                    if n_teacher_layers > 0:
                        # Map every 4th student layer to corresponding teacher layer
                        layer_mapping = {
                            i: int(i * (n_teacher_layers / n_student_layers))
                            for i in range(0, n_student_layers, 4)
                        }

                # Use pre-initialized projection layers from model (if available)
                # or create them on-the-fly if dimensions differ
                projection_layers = None
                if hasattr(model, "projection_layers") and model.projection_layers:
                    # Use pre-initialized projection layers
                    projection_layers = model.projection_layers
                elif layer_mapping:
                    student_d_model = student_hidden_states[0].size(-1)
                    teacher_d_model = (
                        teacher_hidden_states[0].size(-1)
                        if isinstance(teacher_hidden_states, list)
                        else student_d_model
                    )

                    if student_d_model != teacher_d_model:
                        # Create projection layers on-the-fly (fallback)
                        # NOTE: These won't be in optimizer, so they won't be trained
                        # Better to initialize them during model creation
                        projection_layers = create_projection_layers(
                            student_d_model=student_d_model,
                            teacher_d_model=teacher_d_model,
                            layer_mapping=layer_mapping,
                            device=device,
                        )
                        print(
                            "[distill_kd] WARN: Created projection layers on-the-fly. "
                            "Consider initializing them during model creation for proper training."
                        )

                # Compute intermediate layer loss
                intermediate_loss = intermediate_layer_loss(
                    student_hidden_states=student_hidden_states,
                    teacher_hidden_states=teacher_hidden_states,
                    layer_mapping=layer_mapping,
                    projection_layers=projection_layers,
                )

                # Add to loss dict with configurable weight
                intermediate_weight = kd_cfg.get(
                    "intermediate_layer_weight", 0.1)
                loss_dict["intermediate_layer"] = intermediate_loss
                loss_dict["total"] = loss_dict["total"] + \
                    intermediate_weight * intermediate_loss

        # ====================================================================
        # PRIORITY 3: JSON Repair Loop + Metric
        # ====================================================================
        # Check JSON validity and repair needs for tool-use batches
        use_json_repair_check = kd_cfg.get("use_json_repair_check", False)
        json_repair_weight = kd_cfg.get("json_repair_weight", 0.05)

        if use_json_repair_check and json_repair_weight > 0:
            # Only check on tool-use batches (when process-step targets present)
            # This avoids expensive text generation on every batch
            if tool_name_ids is not None or gold_json_text_ids is not None:
                # Generate text from student logits for repair checking
                # Use greedy decoding for efficiency
                if tokenizer is None:
                    # Try to get tokenizer
                    if hasattr(model, "tokenizer"):
                        tokenizer = model.tokenizer
                    elif hasattr(model, "module") and hasattr(model.module, "tokenizer"):
                        tokenizer = model.module.tokenizer
                    elif "tokenizer_path" in cfg:
                        from training.dataset import load_tokenizer

                        tokenizer = load_tokenizer(cfg["tokenizer_path"])

                if tokenizer is not None:
                    try:
                        from training.json_repair import (
                            check_json_repair_needed,
                            batch_check_json_repair,
                        )

                        # Generate text from logits (greedy decoding)
                        # Only generate for the response portion (after prompt)
                        pred_token_ids = student_logits.argmax(
                            dim=-1)  # [B, T]
                        generated_texts = []

                        for i in range(pred_token_ids.size(0)):
                            # Decode tokens to text
                            tokens = pred_token_ids[i].cpu().tolist()
                            try:
                                text = tokenizer.decode(
                                    tokens, skip_special_tokens=True)
                                generated_texts.append(text)
                            except Exception as e:
                                # Decoding failure - log but continue with empty string
                                # Only log first failure to avoid spam
                                if len(generated_texts) == 0:
                                    print(
                                        f"[distill_kd] WARN: Tokenizer decode failed for generated text: {e}")
                                generated_texts.append("")

                        # Check repair needs for batch
                        repair_metrics = batch_check_json_repair(
                            generated_texts, use_jsonrepair=True
                        )

                        # Compute repair loss per sample
                        from training.losses import json_repair_loss

                        repair_losses = []
                        for text in generated_texts:
                            _, needs_repair = check_json_repair_needed(
                                text, use_jsonrepair=True)
                            repair_loss = json_repair_loss(needs_repair)
                            repair_losses.append(repair_loss)

                        if repair_losses:
                            # Average repair loss over batch
                            batch_repair_loss = torch.stack(
                                repair_losses).mean()
                            loss_dict["json_repair"] = batch_repair_loss
                            loss_dict["total"] = (
                                loss_dict["total"] +
                                json_repair_weight * batch_repair_loss
                            )

                            # Log repair metrics periodically
                            if current_step % 100 == 0:
                                print(
                                    f"[distill_kd] JSON repair: rate={repair_metrics['repair_rate']:.3f}, "
                                    f"valid={repair_metrics['valid_json_count']}/{repair_metrics['total']}"
                                )
                    except Exception as e:
                        # Don't fail training if repair check fails
                        if current_step % 100 == 0:
                            print(
                                f"[distill_kd] WARN: JSON repair check failed: {e}")

        # ====================================================================
        # PRIORITY 3 INTEGRATION: Self-Evaluation Loss
        # ====================================================================
        if use_self_eval and eval_score is not None:
            # Get teacher quality score from batch
            # NOTE: Teacher quality scores should be computed during dataset generation
            # or via a quality scoring mechanism (e.g., human eval, automated metrics)
            teacher_quality_score = batch.get("teacher_quality_score")

            if teacher_quality_score is not None:
                # Handle different formats
                if isinstance(teacher_quality_score, (int, float)):
                    teacher_quality = float(teacher_quality_score)
                elif isinstance(teacher_quality_score, torch.Tensor):
                    teacher_quality = teacher_quality_score.to(device)
                else:
                    teacher_quality = None

                if teacher_quality is not None:
                    # Compute self-evaluation loss
                    eval_loss = self_evaluation_loss(
                        student_eval_score=eval_score,
                        teacher_quality_score=teacher_quality,
                    )

                    # Add to loss dict with configurable weight
                    eval_weight = kd_cfg.get("self_evaluation_weight", 0.05)
                    loss_dict["self_evaluation"] = eval_loss
                    loss_dict["total"] = loss_dict["total"] + \
                        eval_weight * eval_loss

        # ====================================================================
        # PRIORITY 5: CAWS Structure Scoring
        # ====================================================================
        # Compute CAWS structure scores and add structure loss
        use_caws_structure = kd_cfg.get("use_caws_structure", False)
        caws_structure_weight = kd_cfg.get("caws_structure_weight", 0.05)

        if use_caws_structure and caws_structure_weight > 0:
            # Only check on batches with text outputs (when teacher_text available)
            teacher_text = batch.get("teacher_text")
            if teacher_text is not None:
                # Generate text from student logits for structure comparison
                if tokenizer is None:
                    # Try to get tokenizer
                    if hasattr(model, "tokenizer"):
                        tokenizer = model.tokenizer
                    elif hasattr(model, "module") and hasattr(model.module, "tokenizer"):
                        tokenizer = model.module.tokenizer
                    elif "tokenizer_path" in cfg:
                        from training.dataset import load_tokenizer

                        tokenizer = load_tokenizer(cfg["tokenizer_path"])

                if tokenizer is not None:
                    try:
                        from training.caws_structure import caws_structure_score
                        from training.losses import caws_structure_loss

                        # Generate text from student logits (greedy decoding)
                        pred_token_ids = student_logits.argmax(
                            dim=-1)  # [B, T]
                        student_texts = []

                        for i in range(pred_token_ids.size(0)):
                            tokens = pred_token_ids[i].cpu().tolist()
                            try:
                                text = tokenizer.decode(
                                    tokens, skip_special_tokens=True)
                                student_texts.append(text)
                            except Exception as e:
                                # Decoding failure - log but continue with empty string
                                if len(student_texts) == 0:  # Only log first failure to avoid spam
                                    print(
                                        f"[distill_kd] WARN: Tokenizer decode failed for student text: {e}")
                                student_texts.append("")

                        # Compute structure scores
                        teacher_text_normalized = teacher_text
                        if isinstance(teacher_text, list):
                            teacher_text_normalized = teacher_text[0] if teacher_text else ""
                        elif not isinstance(teacher_text, str):
                            teacher_text_normalized = str(teacher_text)

                        teacher_structure_score = caws_structure_score(
                            teacher_text_normalized)

                        # Compute structure loss for each student output
                        structure_losses = []
                        for student_text in student_texts:
                            student_structure_score = caws_structure_score(
                                student_text)
                            struct_loss = caws_structure_loss(
                                teacher_score=teacher_structure_score,
                                student_score=student_structure_score,
                            )
                            structure_losses.append(struct_loss)

                        if structure_losses:
                            # Average structure loss over batch
                            batch_structure_loss = torch.stack(
                                structure_losses).mean()
                            loss_dict["caws_structure"] = batch_structure_loss
                            loss_dict["total"] = (
                                loss_dict["total"] +
                                caws_structure_weight * batch_structure_loss
                            )

                            # Log structure scores periodically
                            if current_step % 100 == 0:
                                avg_student_score = (
                                    sum(caws_structure_score(st)
                                        for st in student_texts)
                                    / len(student_texts)
                                    if student_texts
                                    else 0.0
                                )
                                print(
                                    f"[distill_kd] CAWS structure: teacher={teacher_structure_score:.3f}, "
                                    f"student_avg={avg_student_score:.3f}, loss={batch_structure_loss.item():.3f}"
                                )
                    except Exception as e:
                        # Don't fail training if structure check fails
                        if current_step % 100 == 0:
                            print(
                                f"[distill_kd] WARN: CAWS structure check failed: {e}")

        # CAWS compliance loss (optional, config-driven)
        if kd_cfg.get("use_caws_compliance", False):
            # Get teacher text from batch if available
            teacher_text = batch.get("teacher_text")
            if teacher_text is None:
                # Try to get from batch metadata
                teacher_text = (
                    batch.get("metadata", {}).get("teacher_text")
                    if isinstance(batch.get("metadata"), dict)
                    else None
                )

            if teacher_text:
                # Generate student output for compliance check
                # Use argmax decoding (greedy) for efficiency
                student_output_ids = student_logits.argmax(dim=-1)  # [B, T]

                # Decode student output (only first sequence in batch for efficiency)
                if tokenizer:
                    try:
                        student_output = tokenizer.decode(
                            student_output_ids[0].cpu().tolist(), skip_special_tokens=True
                        )

                        # Get claim extractor if available (optional)
                        claim_extractor = None
                        claim_extractor_cfg = cfg.get("claim_extractor", {})
                        if claim_extractor_cfg:
                            # Initialize claim extractor from config
                            extractor_type = claim_extractor_cfg.get(
                                "type", "simple")
                            if extractor_type == "simple":
                                from training.claim_extraction import SimpleClaimExtractor

                                claim_extractor = SimpleClaimExtractor()
                            elif extractor_type == "full":
                                # Use full ClaimifyPipeline for comprehensive extraction
                                try:
                                    from arbiter.claims.pipeline import ClaimifyPipeline

                                    claim_extractor = ClaimifyPipeline()
                                except ImportError:
                                    # Fallback to SimpleClaimExtractor if full pipeline unavailable
                                    from training.claim_extraction import SimpleClaimExtractor

                                    claim_extractor = SimpleClaimExtractor()
                                    print(
                                        "[distill_kd] WARN: Full claim extractor unavailable, using SimpleClaimExtractor"
                                    )
                            else:
                                # Unknown type, use default
                                from training.claim_extraction import SimpleClaimExtractor

                                claim_extractor = SimpleClaimExtractor()
                                print(
                                    f"[distill_kd] WARN: Unknown claim extractor type '{extractor_type}', using SimpleClaimExtractor"
                                )
                        # If no config, claim_extractor remains None and will use default in loss function

                        # Compute CAWS compliance loss
                        compliance_loss = caws_compliance_loss(
                            student_output=student_output,
                            teacher_output=teacher_text
                            if isinstance(teacher_text, str)
                            else teacher_text[0]
                            if isinstance(teacher_text, list)
                            else "",
                            claim_extractor=claim_extractor,
                        )

                        # Add to loss with configurable weight
                        compliance_weight = kd_cfg.get(
                            "caws_compliance_weight", 0.05)
                        compliance_loss_scaled = compliance_loss.to(
                            device) * compliance_weight
                        loss_dict["caws_compliance"] = compliance_loss_scaled
                        loss_dict["total"] = loss_dict["total"] + \
                            compliance_loss_scaled
                    except Exception as e:
                        # If decoding fails, skip compliance loss
                        print(
                            f"[distill_kd] WARN: Failed to compute CAWS compliance loss: {e}")

        # ====================================================================
        # PRIORITY 4 INTEGRATION: Claim Extraction Loss
        # ====================================================================
        if kd_cfg.get("use_claim_extraction", False):
            # Get teacher text from batch
            teacher_text = batch.get("teacher_text")
            if teacher_text is None:
                # Try to get from batch metadata
                teacher_text = (
                    batch.get("metadata", {}).get("teacher_text")
                    if isinstance(batch.get("metadata"), dict)
                    else None
                )

            if teacher_text and tokenizer:
                # Generate student output for claim extraction comparison
                # Use argmax decoding (greedy) for efficiency
                student_output_ids = student_logits.argmax(dim=-1)  # [B, T]

                # Decode student output (only first sequence in batch for efficiency)
                try:
                    student_output = tokenizer.decode(
                        student_output_ids[0].cpu().tolist(), skip_special_tokens=True
                    )

                    # Get claim extractor if available (optional, will create default if None)
                    claim_extractor = None
                    claim_extractor_cfg = cfg.get("claim_extractor", {})
                    if claim_extractor_cfg:
                        # Initialize claim extractor from config
                        extractor_type = claim_extractor_cfg.get(
                            "type", "simple")
                        if extractor_type == "simple":
                            from training.claim_extraction import SimpleClaimExtractor

                            claim_extractor = SimpleClaimExtractor()
                        elif extractor_type == "full":
                            # Use full ClaimifyPipeline for comprehensive extraction
                            try:
                                from arbiter.claims.pipeline import ClaimifyPipeline

                                claim_extractor = ClaimifyPipeline()
                            except ImportError:
                                # Fallback to SimpleClaimExtractor if full pipeline unavailable
                                from training.claim_extraction import SimpleClaimExtractor

                                claim_extractor = SimpleClaimExtractor()
                                print(
                                    "[distill_kd] WARN: Full claim extractor unavailable, using SimpleClaimExtractor"
                                )
                        else:
                            # Unknown type, use default
                            from training.claim_extraction import SimpleClaimExtractor

                            claim_extractor = SimpleClaimExtractor()
                            print(
                                f"[distill_kd] WARN: Unknown claim extractor type '{extractor_type}', using SimpleClaimExtractor"
                            )
                    # If no config, claim_extractor remains None and will use default in loss function

                    # Normalize teacher text format
                    teacher_text_normalized = teacher_text
                    if isinstance(teacher_text, list):
                        teacher_text_normalized = teacher_text[0] if teacher_text else ""
                    elif not isinstance(teacher_text, str):
                        teacher_text_normalized = str(teacher_text)

                    # Compute claim extraction loss
                    from training.losses import claim_extraction_loss

                    claim_loss = claim_extraction_loss(
                        student_output=student_output,
                        teacher_output=teacher_text_normalized,
                        claim_extractor=claim_extractor,
                        min_claim_ratio=kd_cfg.get("min_claim_ratio", 0.5),
                        min_success_rate_ratio=kd_cfg.get(
                            "min_success_rate_ratio", 0.7),
                    )

                    # Add to loss with configurable weight
                    claim_weight = kd_cfg.get("claim_extraction_weight", 0.1)
                    claim_loss_scaled = claim_loss.to(device) * claim_weight
                    loss_dict["claim_extraction"] = claim_loss_scaled
                    loss_dict["total"] = loss_dict["total"] + claim_loss_scaled

                    # Also log claim extraction metrics for monitoring
                    if kd_cfg.get("log_claim_metrics", True):
                        from training.claim_extraction import compute_claim_extraction_metrics

                        claim_metrics = compute_claim_extraction_metrics(
                            student_output=student_output,
                            teacher_output=teacher_text_normalized,
                            claim_extractor=claim_extractor,
                        )
                        # Add metrics to loss_dict for logging (as floats, not tensors)
                        loss_dict["claim_count_student"] = float(
                            claim_metrics.get("student_claim_count", 0)
                        )
                        loss_dict["claim_count_teacher"] = float(
                            claim_metrics.get("teacher_claim_count", 0)
                        )
                        loss_dict["claim_ratio"] = float(
                            claim_metrics.get("claim_ratio", 0.0))
                        loss_dict["success_rate_student"] = float(
                            claim_metrics.get("student_success_rate", 0.0)
                        )
                        loss_dict["success_rate_teacher"] = float(
                            claim_metrics.get("teacher_success_rate", 0.0)
                        )
                        loss_dict["success_rate_ratio"] = float(
                            claim_metrics.get("success_rate_ratio", 0.0)
                        )
                except Exception as e:
                    # If decoding or claim extraction fails, skip claim extraction loss
                    print(
                        f"[distill_kd] WARN: Failed to compute claim extraction loss: {e}")

        loss = loss_dict["total"]
        loss = loss / grad_accum_steps  # Scale for gradient accumulation

        # Assert loss finiteness before backward pass
        assert_loss_finite(loss_dict, step=current_step)

        # Log loss components periodically
        log_loss_components(loss_dict, current_step, log_every=100)

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

        # Update weights (if gradient accumulation complete)
        if (grad_accum_counter + 1) % grad_accum_steps == 0:
            # Apply adaptive gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm)

            # Get model for gradient norm computation (unwrap DDP if needed)
            model_for_grads = model.module if isinstance(model, DDP) else model

            if scaler is not None:
                scaler.unscale_(optimizer)

            # Compute and log gradient norm (every 100 steps)
            if current_step % 100 == 0:
                # Compute total gradient norm before clipping
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_for_grads.parameters(),
                    float("inf"),  # Don't clip, just measure
                )

                # Check for gradient imbalance (basic check - total norm vs expected range)
                # In a full implementation, we'd track per-component norms, but that's expensive
                grad_norms = {"total": total_grad_norm.item()}
                # Note: Per-component gradient norm tracking would require separate backward passes
                # which is expensive. Total norm tracking is a good start.

                # Add gradient norm to loss dict for logging
                loss_dict["grad_norm"] = torch.tensor(
                    total_grad_norm.item(), device=device)
            else:
                # Still compute gradient norm for clipping, but don't log it
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_for_grads.parameters(),
                    float("inf"),  # Don't clip, just measure
                )
            # Note: Per-component gradient norm tracking would require separate backward passes
            # which is expensive. Total norm tracking is a good start.

            if scaler is not None:
                # Re-scale for actual clipping
                scaler.scale_(optimizer)

            # Clip gradients with actual threshold
            grad_clip = cfg.get("optimizer", {}).get("grad_clip", 1.0)
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model_for_grads.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model_for_grads.parameters(), grad_clip)
                optimizer.step()

        else:
            # Normal update without gradient norm logging
            # Normal update without gradient norm logging
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("optimizer", {}).get("grad_clip", 1.0)
                )
                optimizer.step()

    # Convert to float for logging
    loss_dict_float = {
        k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
        for k, v in loss_dict.items()
    }
    return loss_dict_float


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate critical configuration parameters to prevent runtime errors."""
    errors = []

    # Check required sections
    if "model" not in cfg:
        errors.append("Missing required 'model' configuration section")

    if "training" not in cfg:
        errors.append("Missing required 'training' configuration section")

    # Validate model config
    model_cfg = cfg.get("model", {})
    if "d_model" in model_cfg and model_cfg["d_model"] <= 0:
        errors.append("model.d_model must be positive")

    if "n_layers" in model_cfg and model_cfg["n_layers"] <= 0:
        errors.append("model.n_layers must be positive")

    if "n_heads" in model_cfg and model_cfg["n_heads"] <= 0:
        errors.append("model.n_heads must be positive")

    if "vocab_size" in model_cfg and model_cfg["vocab_size"] <= 0:
        errors.append("model.vocab_size must be positive")

    # Validate training config
    train_cfg = cfg.get("training", {})
    if "steps" in train_cfg and train_cfg["steps"] <= 0:
        errors.append("training.steps must be positive")

    if "batch_size" in train_cfg and train_cfg["batch_size"] <= 0:
        errors.append("training.batch_size must be positive")

    # Validate optimizer config
    opt_cfg = cfg.get("optimizer", {})
    if "lr" in opt_cfg and opt_cfg["lr"] <= 0:
        errors.append("optimizer.lr must be positive")

    # Check for incompatible settings
    distillation_cfg = cfg.get("distillation", {})
    code_mode_cfg = cfg.get("distill", {}).get("code_mode", {})
    latent_cfg = cfg.get("latent", {})

    if distillation_cfg.get("enabled", False) and code_mode_cfg.get("enabled", False):
        print(
            "[distill_kd] WARN: Both distillation and code_mode enabled - this may cause conflicts"
        )

    if latent_cfg.get("enabled", False) and code_mode_cfg.get("enabled", False):
        print(
            "[distill_kd] WARN: Both latent reasoning and code_mode enabled - ensure compatibility"
        )

    # Raise errors if any found
    if errors:
        error_msg = "Configuration validation failed:\n" + \
            "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


def main():
    ap = argparse.ArgumentParser(description="Knowledge distillation training")
    ap.add_argument("--config", nargs="+", required=True,
                    help="Config file(s) to load")
    ap.add_argument("--resume", help="Resume from checkpoint path")
    ap.add_argument(
        "--output-dir",
        default="models/student/checkpoints",
        help="Output directory for checkpoints",
    )
    ap.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank for distributed training"
    )
    args = ap.parse_args()

    # Version compatibility check - fail fast if versions are incompatible
    try:
        check_training_versions()
    except RuntimeError as e:
        print(f"[distill_kd] ERROR: Version check failed: {e}")
        sys.exit(1)

    # Setup distributed training if needed
    if args.local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        is_main_process = args.local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # Load configs (with env var overrides)
    cfg = merge_configs(args.config)

    # Validate configuration
    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"[distill_kd] ERROR: {e}")
        sys.exit(1)

    # Log provenance for reproducibility
    if is_main_process and "_provenance" in cfg:
        provenance = cfg["_provenance"]
        print("[distill_kd] Config provenance:")
        print(f"  Config files: {provenance['config_files']}")
        if provenance.get("env_overrides"):
            print(f"  Env overrides: {provenance['env_overrides']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(cfg, device)

    # Migrate tokenizer and model for special tokens (if needed)
    tokenizer_path = cfg.get("io", {}).get(
        "tokenizer_path", "models/student/tokenizer")
    if TOKENIZER_MIGRATION_AVAILABLE:
        try:
            tokenizer = load_tokenizer(tokenizer_path)
            model, migration_metadata = migrate_tokenizer_and_model(
                model,
                tokenizer,
                tokenizer_path,
                verify_ids=True,
                resize_embeddings=True,
                init_special_tokens=True,
            )
            if is_main_process:
                print("[distill_kd] Tokenizer migration completed:")
                print(
                    f"  - Embedding resized: {migration_metadata['resize'].get('embedding_resized', False)}"
                )
                print(
                    f"  - LM head resized: {migration_metadata['resize'].get('lm_head_resized', False)}"
                )
                if migration_metadata["resize"].get("embedding_resized"):
                    print(
                        f"  - Vocab size: {migration_metadata['resize']['original_vocab_size']} -> {migration_metadata['resize']['new_vocab_size']}"
                    )
        except Exception as e:
            if is_main_process:
                print(f"[distill_kd] WARN: Tokenizer migration failed: {e}")
                print(
                    "  Continuing without migration (may cause issues if vocab mismatch)")

    # Setup distributed model if needed
    if args.local_rank >= 0:
        model = DDP(model, device_ids=[args.local_rank])

    # Create optimizer
    optimizer = create_optimizer(model, cfg)

    # Setup FP16 scaler
    train_cfg = cfg.get("train", {})

    # Create learning rate scheduler with warmup and cosine annealing
    # Based on our toy model optimizations for better convergence
    total_steps = train_cfg.get("steps", 10000)

    # Adaptive warmup: longer for larger models, shorter for smaller/fast training
    model_size_factor = math.log(
        max(1, sum(p.numel() for p in model.parameters()))) / 10
    base_warmup_pct = 0.1  # 10% of training
    adaptive_warmup_pct = min(
        0.2, max(0.05, base_warmup_pct * model_size_factor))
    warmup_steps = int(total_steps * adaptive_warmup_pct)

    # Adaptive gradient clipping based on model size
    model_params = sum(p.numel() for p in model.parameters())
    if model_params > 100_000_000:  # Large models (>100M params)
        grad_clip_norm = 0.5
    elif model_params > 10_000_000:  # Medium models (>10M params)
        grad_clip_norm = 1.0
    else:  # Small models
        grad_clip_norm = 2.0

    if is_main_process:
        print(
            f"[distill_kd] LR Schedule: {warmup_steps} warmup steps, {total_steps} total steps")
        print(
            f"[distill_kd] Gradient Clipping: {grad_clip_norm} (adaptive for {model_params:,} params)"
        )

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine annealing with final LR floor
            progress = (step - warmup_steps) / \
                max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * \
                (1 + torch.cos(torch.tensor(progress * 3.14159)))
            min_lr_factor = 0.01  # Don't decay below 1% of peak LR
            return max(min_lr_factor, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_fp16 = train_cfg.get("fp16", False)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device.type == "cuda" else None

    # Automatic memory optimization for large models
    model_params = sum(p.numel() for p in model.parameters())
    is_large_model = model_params > 1_000_000_000  # >1B parameters

    # Enable memory optimizations automatically for large models
    if is_large_model or train_cfg.get("grad_checkpointing", False):
        print(
            f"[distill_kd] Enabling memory optimizations for {model_params:,} parameters")

        # Gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "module") and hasattr(model.module, "gradient_checkpointing_enable"):
            model.module.gradient_checkpointing_enable()

        # Aggressive garbage collection for large models
        if is_large_model:
            import gc

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                print("[distill_kd] Cleared GPU cache for large model")

    # Enable TF32 for faster training on Ampere+ GPUs
    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[distill_kd] Enabled TF32 acceleration")

    # Resume from checkpoint if specified
    start_step = 0
    optimizer_state_restored = False
    scaler_state_restored = False
    if args.resume:
        print(f"[distill_kd] Resuming from checkpoint: {args.resume}")
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint = safe_load_checkpoint(
            args.resume, map_location=str(device))

        # Restore step number
        if "step" in checkpoint:
            start_step = checkpoint["step"]
            print(f"[distill_kd] Resuming from step {start_step}")
        else:
            print(
                "[distill_kd] WARN: Checkpoint missing 'step' field, starting from step 0")

        # Restore optimizer state if available
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer_state_restored = True
                print("[distill_kd] Optimizer state restored from checkpoint")
            except Exception as e:
                print(
                    f"[distill_kd] WARN: Failed to restore optimizer state: {e}")
                print("[distill_kd] Continuing with fresh optimizer state")
        else:
            print(
                "[distill_kd] WARN: Checkpoint missing optimizer state, using fresh optimizer")

        # Restore scaler state if using FP16 and available
        if use_fp16 and scaler is not None:
            if "meta" in checkpoint and "scaler_state_dict" in checkpoint.get("meta", {}):
                try:
                    scaler.load_state_dict(
                        checkpoint["meta"]["scaler_state_dict"])
                    scaler_state_restored = True
                    print("[distill_kd] GradScaler state restored from checkpoint")
                except Exception as e:
                    print(
                        f"[distill_kd] WARN: Failed to restore scaler state: {e}")
                    print("[distill_kd] Continuing with fresh scaler state")
            else:
                print(
                    "[distill_kd] WARN: Checkpoint missing scaler state, using fresh scaler")

        # Restore RNG states if available (for reproducibility)
        if "meta" in checkpoint and "rng_states" in checkpoint["meta"]:
            try:
                rng_states = checkpoint["meta"]["rng_states"]
                if "python" in rng_states:
                    import random
                    random.setstate(rng_states["python"])
                if "numpy" in rng_states:
                    import numpy as np
                    np.random.set_state(rng_states["numpy"])
                if "torch" in rng_states:
                    torch.set_rng_state(rng_states["torch"])
                if "torch_cuda" in rng_states and rng_states["torch_cuda"] is not None:
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state(rng_states["torch_cuda"])
                print("[distill_kd] RNG states restored from checkpoint")
            except Exception as e:
                print(f"[distill_kd] WARN: Failed to restore RNG states: {e}")
                print("[distill_kd] Continuing with fresh RNG states")

        # Validate config compatibility on resume
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            # Compare critical config fields that affect training
            critical_fields = ["arch", "train", "optimizer", "distill"]
            mismatches = []

            for field in critical_fields:
                saved_val = saved_config.get(field, {})
                current_val = cfg.get(field, {})

                # Compare architecture fields (must match exactly)
                if field == "arch":
                    if saved_val != current_val:
                        mismatches.append(
                            f"Architecture config mismatch in '{field}': "
                            f"saved={saved_val}, current={current_val}"
                        )
                # For other fields, allow some differences (e.g., learning rate schedule changes)
                # but log them as warnings

            if mismatches:
                print("[distill_kd] WARNING: Config mismatches detected on resume:")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                print(
                    "[distill_kd] Continuing with current config (may cause training issues)")
        else:
            print(
                "[distill_kd] WARNING: Checkpoint missing 'config' field - cannot validate compatibility"
            )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "step" in checkpoint:
            start_step = checkpoint["step"]
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore RNG states if available for reproducibility
        if "meta" in checkpoint and "rng_states" in checkpoint["meta"]:
            rng_states = checkpoint["meta"]["rng_states"]
            try:
                import random
                import numpy as np

                if "python" in rng_states:
                    random.setstate(rng_states["python"])
                if "numpy" in rng_states:
                    np.random.set_state(rng_states["numpy"])
                if "torch" in rng_states:
                    torch.set_rng_state(rng_states["torch"])
                if (
                    "torch_cuda" in rng_states
                    and rng_states["torch_cuda"] is not None
                    and torch.cuda.is_available()
                ):
                    torch.cuda.set_rng_state(rng_states["torch_cuda"])
                print("[distill_kd] Restored RNG states from checkpoint")
            except Exception as e:
                print(
                    f"[distill_kd] WARNING: Failed to restore RNG states: {e}")

    # Create dataset
    io_cfg = cfg.get("io", {})
    tokenizer_path = io_cfg.get("tokenizer_path")
    if not tokenizer_path:
        # Fallback: try configs/worker_9b.yaml tokenizer path
        tokenizer_path = cfg.get("tokenizer", {}).get(
            "path", "models/student/tokenizer")

    if not tokenizer_path:
        raise ValueError(
            "tokenizer_path must be specified in config (io.tokenizer_path)")

    train_shards = io_cfg.get("train_shards", ["data/kd_mix.jsonl"])

    # Get sequence lengths for enumerated shape training
    seq_lengths = train_cfg.get("seq_lengths", [4096])
    use_enumerated_shapes = train_cfg.get("use_enumerated_shapes", False)
    # Optional Dirichlet distribution
    shape_probs = train_cfg.get("shape_probs", None)

    # Use max sequence length for dataset (will truncate per batch if using enumerated shapes)
    max_seq_len = max(seq_lengths) if seq_lengths else 4096

    # Optimize data loading: prefetch and pin memory for better GPU utilization
    num_workers = train_cfg.get("dataloader_workers", 4)
    prefetch_factor = train_cfg.get("prefetch_factor", 2)

    # Adaptive batch size based on sequence length and model size
    base_batch_size = train_cfg.get("batch_size", 8)
    # Shorter sequences allow larger batches
    seq_len_factor = min(1.0, 2048 / max_seq_len)
    # Smaller models allow larger batches
    model_size_factor = min(1.0, 100_000_000 / model_params)

    effective_batch_size = int(
        base_batch_size * seq_len_factor * model_size_factor)
    effective_batch_size = max(
        1, min(effective_batch_size, base_batch_size * 2)
    )  # Reasonable bounds

    if is_main_process:
        print(
            f"[distill_kd] Data Loading: {num_workers} workers, prefetch={prefetch_factor}")
        print(
            f"[distill_kd] Batch Size: {effective_batch_size} (adapted from {base_batch_size})")

    # Setup latent curriculum if enabled
    latent_curriculum = None
    latent_cfg = cfg.get("latent", {})
    latent_enabled = latent_cfg.get("enabled", False) or (
        os.getenv("TRAIN_LATENT", "0") == "1")

    if latent_enabled and LATENT_CURRICULUM_AVAILABLE:
        m = latent_cfg.get("m", 2)
        c = latent_cfg.get("c", 1)
        p = latent_cfg.get("p", 0.5)
        latent_curriculum = LatentCurriculum(m=m, c=c, p=p)
        if is_main_process:
            print(
                f"[distill_kd] Latent curriculum enabled: m={m}, c={c}, p={p}")
    elif latent_enabled and not LATENT_CURRICULUM_AVAILABLE:
        if is_main_process:
            print("[distill_kd] WARN: Latent curriculum requested but not available")

    dataset = KDDataset(
        jsonl_path=train_shards[0],
        tokenizer_path=tokenizer_path,
        max_seq_length=max_seq_len,
        teacher_logits_available=cfg.get("kd", {}).get(
            "teacher_logits_available", False),
        latent_curriculum=latent_curriculum,
    )

    # Validate dataset fingerprint if available
    if dataset.dataset_fingerprint:
        print(
            f"[distill_kd] Dataset fingerprint: {dataset.dataset_fingerprint}")
        # Store fingerprint in checkpoint metadata for traceability
        if is_main_process:
            print(
                "[distill_kd] Dataset fingerprint will be logged in checkpoint metadata")
    elif dataset.dataset_header:
        print(
            "[distill_kd] WARNING: Dataset header found but no fingerprint - dataset may not be fingerprinted"
        )
    else:
        print(
            "[distill_kd] INFO: Dataset does not have header/fingerprint (may be legacy format)")

    # Create dataloader
    micro_batch_size = train_cfg.get("micro_batch_size", 2)
    grad_accum = train_cfg.get("grad_accum", 16)
    effective_batch_size = micro_batch_size * grad_accum

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_kd_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Initialize code-mode loss module if enabled (once, before training loop)
    code_mode_loss_module = None
    code_mode_cfg = cfg.get("distill", {}).get("code_mode", {})
    train_code_mode = code_mode_cfg.get("enabled", False) or (
        os.getenv("TRAIN_CODE_MODE", "0") == "1"
    )

    if train_code_mode:
        from training.dataset import load_tokenizer

        tokenizer_for_vocab = load_tokenizer(
            tokenizer_path) if tokenizer_path else None

        # Get vocabulary IDs for markers
        vocab_ids = {}
        if tokenizer_for_vocab is not None:
            marker_tokens = ["import", "from", "callMCPTool",
                             "await", "tool_call", "tool_result"]
            for marker in marker_tokens:
                try:
                    if hasattr(tokenizer_for_vocab, "convert_tokens_to_ids"):
                        tok_id = tokenizer_for_vocab.convert_tokens_to_ids(
                            marker)
                    else:
                        encoded = tokenizer_for_vocab.encode(
                            marker, add_special_tokens=False)
                        tok_id = encoded[0] if encoded else None

                    if tok_id is not None:
                        vocab_ids[marker] = tok_id
                except Exception as e:
                    # Marker token lookup failed - log but continue
                    # Code-mode loss can work with partial marker vocabulary
                    print(
                        f"[distill_kd] WARN: Could not find token ID for marker '{marker}': {e}")

        code_mode_pref_cfg = code_mode_cfg.get("code_mode_pref", {})
        eligibility_rules = code_mode_pref_cfg.get(
            "eligibility_rules",
            {
                "min_tools": 2,
                "min_intermediate_chars": 10000,
                "pii_patterns": ["EMAIL", "PHONE", "SSN"],
            },
        )
        reward_cfg = code_mode_pref_cfg.get(
            "reward",
            {
                "prefer_ts_api_over_direct_tool": True,
                "penalize_tool_result_roundtrip": True,
            },
        )
        loss_weights = code_mode_pref_cfg.get(
            "weights", {"pos": 1.0, "neg": 1.0})

        code_mode_loss_module = CodeModePreferenceLoss(
            eligibility_rules=eligibility_rules,
            reward=reward_cfg,
            vocab_ids=vocab_ids,
            weights=loss_weights,
        ).to(device)

        # Code-mode loss module will be passed as parameter to train_step

        if is_main_process:
            print(
                f"[distill_kd] Code-mode loss module initialized with vocab_ids: {list(vocab_ids.keys())}"
            )

    # Training loop
    total_steps = train_cfg.get("steps", 200000)
    save_every = train_cfg.get("save_every", 2000)
    log_every = train_cfg.get("log_every", 50)

    # Initialize training tracer
    run_name = f"worker_9b_kd_{start_step}"
    tracer = create_tracer_from_config(
        cfg, run_name=run_name) if is_main_process else None

    # Log hyperparameters
    if tracer:
        opt_cfg = cfg.get("optimizer", {})
        dist_cfg = cfg.get("distillation", {})
        tracer.log_hparams(
            {
                "lr": opt_cfg.get("lr", 2e-4),
                "batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "grad_accum": grad_accum,
                "fp16": use_fp16,
                "seq_lengths": str(seq_lengths),
                "total_steps": total_steps,
                "device": str(device),
                # Optimized distillation weights based on toy model experiments
                # Higher KL weight for better teacher alignment, balanced CE weights
                # Increased for teacher alignment
                "kl_weight": dist_cfg.get("kl_weight", 0.6),
                # Reduced to balance
                "ce_teacher_weight": dist_cfg.get("ce_teacher_weight", 0.2),
                "ce_ground_truth_weight": dist_cfg.get(
                    "ce_ground_truth_weight", 0.2
                ),  # Keep ground truth
            }
        )

    print("[distill_kd] Starting training:")
    print(f"  Device: {device}")
    print(f"  Total steps: {total_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(
        f"  Micro batch size: {micro_batch_size}, Gradient accumulation: {grad_accum}")
    print(f"  FP16: {use_fp16}")
    print(f"  Sequence lengths: {seq_lengths}")
    if tracer:
        print(
            f"  Tracing enabled: TensorBoard={tracer.use_tensorboard}, WandB={tracer.use_wandb}")

    step = start_step
    grad_accum_counter = 0
    qat_enabled = False
    qat_applied = False

    # Track current data shard index for checkpointing
    current_shard_index = 0  # For single-shard datasets, this is 0

    # QAT configuration
    qat_cfg = cfg.get("quant", {})
    qat_lr_multiplier = qat_cfg.get(
        "lr_multiplier", 0.1)  # Default: 10× lower LR
    base_lr = cfg.get("optimizer", {}).get("lr", 2e-4)

    # Iterate over dataset multiple times if needed
    while step < total_steps:
        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break

            try:
                # Check if QAT should be enabled
                qat_should_enable = should_enable_qat(
                    step, total_steps, qat_cfg)
                if qat_should_enable and not qat_applied:
                    # Apply QAT to model
                    print(
                        f"[distill_kd] Step {step}: Enabling QAT (last {int((1 - qat_cfg.get('start_fraction', 0.8)) * 100)}% of training)"
                    )
                    if isinstance(model, DDP):
                        model.module = apply_qat_to_model(
                            model.module, qat_cfg, device)
                    else:
                        model = apply_qat_to_model(model, qat_cfg, device)

                # Recreate optimizer with lower LR for QAT
                qat_lr = base_lr * qat_lr_multiplier
                print(
                    f"[distill_kd] Adjusting LR for QAT: {base_lr} → {qat_lr}")
                optimizer = create_optimizer(
                    model.module if isinstance(model, DDP) else model,
                    {
                        **cfg.get("optimizer", {}),
                        "lr": qat_lr,
                    },
                )

                qat_applied = True
                qat_enabled = True

                # Check QAT stability periodically (every 100 steps)
                if qat_enabled and step % 100 == 0:
                    stability_metrics = check_qat_stability(
                        model.module if isinstance(model, DDP) else model,
                        batch,
                        device,
                    )
                    if stability_metrics.get("qat_stability.has_nan", 0.0) > 0:
                        print(
                            f"[distill_kd] WARN: NaN detected in QAT model at step {step}")
                    if stability_metrics.get("qat_stability.cosine_sim", 1.0) < 0.999:
                        print(
                            f"[distill_kd] WARN: Low cosine similarity in QAT model at step {step}"
                        )

                # Sample sequence length (enumerated shapes or curriculum)
                if use_enumerated_shapes:
                    current_seq_len = sample_enumerated_shape(
                        seq_lengths=seq_lengths,
                        shape_probs=shape_probs,
                        step=step,
                        periodic_upweight_rare=train_cfg.get(
                            "periodic_upweight_rare", True),
                    )
                    # Truncate batch to sampled shape
                    batch = truncate_batch_to_shape(batch, current_seq_len)
                else:
                    # Fallback to curriculum learning
                    current_seq_len = get_sequence_length(
                        step, seq_lengths, cfg.get(
                            "curriculum", {}).get("schedule")
                    )

                # Training step
                loss_dict = train_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                    device=device,
                    grad_clip_norm=grad_clip_norm,
                    grad_accum_steps=grad_accum,
                    grad_accum_counter=grad_accum_counter,
                    current_step=step,
                    code_mode_loss_module=code_mode_loss_module,
                )

                # Step learning rate scheduler after each optimizer step
                if (grad_accum_counter + 1) % grad_accum == 0:
                    scheduler.step()

                grad_accum_counter = (grad_accum_counter + 1) % grad_accum
                step += 1

                # Log learning rate and memory usage periodically
                if step % 100 == 0 and is_main_process:
                    current_lr = optimizer.param_groups[0]["lr"]

                    # Memory usage tracking
                    if device.type == "cuda":
                        memory_allocated = torch.cuda.memory_allocated(
                            device) / 1024**3  # GB
                        memory_info = f", gpu_mem={memory_allocated:.2f}GB"
                    else:
                        memory_info = ""

                    # Enhanced progress reporting for large models
                    progress_pct = step / total_steps * 100
                    eta_hours = (total_steps - step) / 100 * \
                        0.1  # Rough ETA based on steps/second

                    if is_large_model:
                        print(
                            f"[distill_kd] Step {step}/{total_steps} ({progress_pct:.1f}%): loss={loss_dict.get('total', 0):.4f}, lr={current_lr:.2e}{memory_info}, ETA~{eta_hours:.1f}h"
                        )
                    else:
                        print(
                            f"[distill_kd] Step {step}/{total_steps}: loss={loss_dict.get('total', 0):.4f}, lr={current_lr:.2e}{memory_info}"
                        )
            except Exception as e:
                print(f"[distill_kd] ERROR: Training step {step} failed: {e}")
                import traceback

                traceback.print_exc()

                # Enhanced error recovery for large models
                if device.type == "cuda":
                    try:
                        # Aggressive memory cleanup for large models
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure cleanup completes
                        print("[distill_kd] Cleared GPU cache after error")

                        # For very large models, also clear CPU cache
                        if is_large_model:
                            import gc

                            gc.collect()
                            print(
                                "[distill_kd] Performed full garbage collection for large model")

                    except Exception as cache_e:
                        print(
                            f"[distill_kd] WARN: Failed to clear GPU cache: {cache_e}")

                # Track consecutive failures to prevent infinite loops
                if not hasattr(train_step, "_consecutive_failures"):
                    train_step._consecutive_failures = 0

                train_step._consecutive_failures += 1

                # If too many consecutive failures, save emergency checkpoint and exit
                max_consecutive_failures = (
                    10 if not is_large_model else 5
                )  # More strict for large models
                if train_step._consecutive_failures >= max_consecutive_failures:
                    print(
                        f"[distill_kd] CRITICAL: {train_step._consecutive_failures} consecutive failures. Saving emergency checkpoint and exiting."
                    )
                    try:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            loss=loss_dict.get("total", float("inf")),
                            output_dir=output_dir,
                            config=cfg,
                            rng_states=None,  # Skip RNG states in emergency
                            data_shard_index=current_shard_index,
                            dataset_fingerprint=None,
                            scaler=scaler,
                        )
                        print(
                            "[distill_kd] Emergency checkpoint saved successfully")
                    except Exception as ckpt_e:
                        print(
                            f"[distill_kd] CRITICAL: Failed to save emergency checkpoint: {ckpt_e}"
                        )

                    sys.exit(1)

                # Continue to next step instead of crashing
                step += 1
                continue

            # Speed metrics during validation (every N steps)
            # Default: every 1000 steps
            val_every = train_cfg.get("val_every", 1000)
            if SPEED_METRICS_AVAILABLE and step % val_every == 0 and is_main_process:
                try:
                    # Load tokenizer if available
                    from training.dataset import load_tokenizer

                    val_tokenizer = load_tokenizer(tokenizer_path)

                    # Measure speed metrics on a few batches
                    speed_metrics_list = []
                    # Measure on up to 5 batches
                    val_batch_count = min(5, len(dataloader))

                    for val_batch_idx, val_batch in enumerate(dataloader):
                        if val_batch_idx >= val_batch_count:
                            break

                        # Truncate to reasonable length for speed measurement
                        if use_enumerated_shapes:
                            val_batch = truncate_batch_to_shape(
                                val_batch, min(seq_lengths))

                        metrics = measure_proxy(
                            model=model.module if isinstance(
                                model, DDP) else model,
                            batch=val_batch,
                            tokenizer=val_tokenizer,
                            device=device,
                            max_new_tokens=64,
                        )
                        speed_metrics_list.append(metrics)

                    # Aggregate metrics
                    if speed_metrics_list:
                        aggregated = aggregate_speed_metrics(
                            speed_metrics_list)

                        # Log with export=False tag (these are proxies)
                        if tracer:
                            tracer.log_metrics(
                                step=step,
                                metrics={
                                    "speed/ttft_ms_p50": aggregated["ttft_ms"]["p50"],
                                    "speed/ttft_ms_p95": aggregated["ttft_ms"]["p95"],
                                    "speed/tps_p50": aggregated["tps"]["p50"],
                                    "speed/tps_p95": aggregated["tps"]["p95"],
                                    "speed/ttfa_tokens_p95": aggregated["ttfa_tokens"]["p95"],
                                    # Tag: export=False (proxy metrics)
                                    "speed/export": 0.0,
                                },
                                prefix="val/",
                            )
                        else:
                            print(
                                f"[distill_kd] Step {step} speed metrics (proxy): "
                                f"TTFT p50={aggregated['ttft_ms']['p50']:.1f}ms, "
                                f"TPS p50={aggregated['tps']['p50']:.1f} tok/s"
                            )
                except Exception as e:
                    print(
                        f"[distill_kd] WARN: Failed to measure speed metrics: {e}")

            # Logging
            if step % log_every == 0 and is_main_process:
                if tracer:
                    # Log to tracer (includes console, TensorBoard, WandB, JSON)
                    tracer.log_metrics(
                        step=step, metrics=loss_dict, prefix="train/")

                    # Also log learning rate if available
                    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
                        lr = optimizer.param_groups[0].get("lr", 0.0)
                        tracer.log_metrics(
                            step=step, metrics={"learning_rate": lr}, prefix="train/"
                        )
                else:
                    # Fallback to console logging
                    loss_str = ", ".join(
                        [f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    print(
                        f"[distill_kd] Step {step}/{total_steps}: {loss_str}")

            # Intelligent checkpointing for large models
            # Save more frequently for large models to prevent loss of progress
            checkpoint_frequency = save_every
            if is_large_model:
                checkpoint_frequency = min(
                    save_every, max(100, save_every // 4)
                )  # Save 4x more frequently for large models

            should_save_checkpoint = step % checkpoint_frequency == 0 and is_main_process

            # Always save checkpoint at milestones (10%, 25%, 50%, 75%, 90%, 100%)
            total_steps = train_cfg.get("steps", 10000)
            milestone_steps = [int(total_steps * pct)
                               for pct in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]]
            if step in milestone_steps:
                should_save_checkpoint = True
                print(
                    f"[distill_kd] Milestone checkpoint at {step}/{total_steps} steps ({step / total_steps:.1%})"
                )

            if should_save_checkpoint:
                # Capture RNG states for reproducibility
                import random
                import numpy as np

                rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=loss_dict.get("total", 0.0),
                    output_dir=output_dir,
                    config=cfg,
                    loss_dict=loss_dict,
                    rng_states=rng_states,
                    data_shard_index=current_shard_index,
                    dataset_fingerprint=dataset.dataset_fingerprint,
                    scaler=scaler,
                )

    # Final checkpoint
    if is_main_process:
        # Capture RNG states for reproducibility
        import random
        import numpy as np

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=loss_dict.get("total", 0.0),
            output_dir=output_dir,
            config=cfg,
            loss_dict=loss_dict,
            rng_states=rng_states,
            data_shard_index=current_shard_index,
            dataset_fingerprint=dataset.dataset_fingerprint
            if hasattr(dataset, "dataset_fingerprint")
            else None,
            scaler=scaler,
        )

        # Close tracer and save summary
        if tracer:
            tracer.close()
            print(f"[distill_kd] Training logs: {tracer.log_dir}")

        print(f"[distill_kd] ✅ Training complete: {output_dir}")

        # Clean up GPU memory
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                print("[distill_kd] Cleared GPU cache after training completion")
            except Exception as e:
                print(f"[distill_kd] WARN: Failed to clear GPU cache: {e}")


if __name__ == "__main__":
    main()
