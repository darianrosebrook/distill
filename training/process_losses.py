"""
Process supervision loss functions for tool-use training.

Implements:
- JSON validity loss (penalize invalid JSON)
- Tool selection loss (correct tool name)
- Argument validation loss (correct argument structure)
"""

import json
import re
from typing import Dict, Optional, List

import torch
import torch.nn.functional as F


def validate_json(text: str) -> bool:
    """Check if text contains valid JSON."""
    # Try to find JSON in text
    json_patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple JSON object
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",  # JSON array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return True
            except json.JSONDecodeError:
                continue

    # Try parsing entire text
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        pass

    return False


def extract_tool_call(text: str, tool_names: List[str]) -> Optional[Dict]:
    """
    Extract tool call from text.

    Returns:
        Dict with 'name' and 'arguments' if found, None otherwise
    """
    # Look for JSON tool call
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    # Try parsing entire text
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    return None


def json_validity_loss(
    logits: torch.Tensor, generated_texts: List[str], ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute loss that penalizes invalid JSON generation.

    Args:
        logits: [B, T, V] model logits
        generated_texts: List of generated text strings
        ignore_index: Index to ignore

    Returns:
        Loss value (higher for invalid JSON)
    """
    # For each generated text, check if it contains valid JSON
    validity_scores = []
    for text in generated_texts:
        is_valid = validate_json(text)
        validity_scores.append(1.0 if is_valid else 0.0)

    # Convert to tensor
    validity_tensor = torch.tensor(validity_scores, device=logits.device, dtype=logits.dtype)

    # Loss: 1 - validity (penalize invalid JSON)
    loss = (1.0 - validity_tensor).mean()

    return loss


def tool_selection_loss(
    logits: torch.Tensor,
    generated_texts: List[str],
    target_tool_names: List[str],
    tool_names: List[str],
    tokenizer,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute loss for correct tool selection.

    Args:
        logits: [B, T, V] model logits
        generated_texts: List of generated text strings
        target_tool_names: List of target tool names (ground truth)
        tool_names: List of available tool names
        tokenizer: Tokenizer for encoding tool names
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss for tool name prediction
    """
    losses = []

    for i, (text, target_tool) in enumerate(zip(generated_texts, target_tool_names)):
        # Extract tool call from generated text
        tool_call = extract_tool_call(text, tool_names)

        if tool_call and target_tool:
            predicted_tool = tool_call.get("name", "")

            # Encode tool names
            target_tokens = tokenizer.encode(target_tool, add_special_tokens=False)
            pred_tokens = tokenizer.encode(predicted_tool, add_special_tokens=False)

            # Find where tool name appears in sequence
            # Extract tool name from generated text and find its position
            if len(target_tokens) > 0 and len(pred_tokens) > 0:
                target_token_id = target_tokens[0]

                # Try to find tool name position in generated text
                # Search for tool name pattern in text to get approximate position
                seq_len = logits.size(1)

                # Decode logits to find where tool name might appear
                # Use argmax to get predicted tokens
                pred_token_ids = logits[i].argmax(dim=-1).cpu().tolist()

                # Find position where target tool name tokens appear
                pos = None
                for j in range(seq_len - len(target_tokens) + 1):
                    # Check if target tokens match at this position
                    if pred_token_ids[j : j + len(target_tokens)] == target_tokens:
                        pos = j
                        break

                # Fallback: use middle position if tool name not found
                if pos is None:
                    pos = seq_len // 2

                # Extract logits for this position and ensure proper shape
                # Metal/MPS requires contiguous tensors and proper shapes
                logits_slice = logits[i, pos, :].unsqueeze(0)  # [1, vocab_size]
                targets = torch.tensor([target_token_id], device=logits.device, dtype=torch.long)

                # Ensure logits are contiguous for Metal compatibility
                if not logits_slice.is_contiguous():
                    logits_slice = logits_slice.contiguous()

                # Cross-entropy loss
                ce_loss = F.cross_entropy(logits_slice, targets, ignore_index=ignore_index)
                losses.append(ce_loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def tool_selection_loss_from_ids(
    logits: torch.Tensor,
    tool_name_ids: torch.Tensor,
    tool_name_mask: torch.Tensor,
    tokenizer,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute tool selection loss using pre-extracted tool name token IDs.

    This is more efficient than extracting from generated text because it uses
    the ground truth tool name tokens directly.

    Args:
        logits: [B, T, V] model logits
        tool_name_ids: [B, N] or [N] tensor of tool name token IDs
        tool_name_mask: [B, N] or [N] boolean mask for valid tokens
        tokenizer: Tokenizer for decoding/encoding
        ignore_index: Index to ignore in loss computation

    Returns:
        Cross-entropy loss for tool name prediction
    """
    device = logits.device
    batch_size = logits.size(0)

    # Handle different tensor shapes
    if tool_name_ids.dim() == 1:
        tool_name_ids = tool_name_ids.unsqueeze(0)
        tool_name_mask = tool_name_mask.unsqueeze(0)

    losses = []

    for i in range(min(batch_size, tool_name_ids.size(0))):
        sample_tool_ids = tool_name_ids[i]
        sample_mask = tool_name_mask[i]

        # Extract valid tool name tokens
        if sample_mask.dtype == torch.bool:
            valid_tool_tokens = sample_tool_ids[sample_mask]
        else:
            valid_tool_tokens = sample_tool_ids[sample_mask.bool()]

        if valid_tool_tokens.numel() == 0:
            continue

        # Decode tool name tokens to find where they appear in sequence
        # Get predicted token IDs from logits to locate tool name position
        target_token_id = valid_tool_tokens[0].item()
        target_tokens = valid_tool_tokens.cpu().tolist()

        seq_len = logits.size(1)

        # Decode logits to get predicted tokens for this sample
        pred_token_ids = logits[i].argmax(dim=-1).cpu().tolist()

        # Find position where tool name tokens appear in predicted sequence
        tool_name_pos = None
        for j in range(seq_len - len(target_tokens) + 1):
            # Check if target tokens match at this position
            if pred_token_ids[j : j + len(target_tokens)] == target_tokens:
                tool_name_pos = j
                break

        # If tool name found, compute loss at that position
        if tool_name_pos is not None:
            logits_slice = logits[i, tool_name_pos, :].unsqueeze(0)
            targets = torch.tensor([target_token_id], device=device, dtype=torch.long)

            if not logits_slice.is_contiguous():
                logits_slice = logits_slice.contiguous()

            ce_loss = F.cross_entropy(logits_slice, targets, ignore_index=ignore_index)
            losses.append(ce_loss)
            continue

        # Fallback: search multiple positions if exact match not found
        # Try positions in the latter half of sequence where tool calls typically appear
        if seq_len > 10:
            start_pos = seq_len // 2
            best_loss = None

            for pos in range(start_pos, min(start_pos + 20, seq_len)):
                logits_slice = logits[i, pos, :].unsqueeze(0)  # [1, vocab_size]
                targets = torch.tensor([target_token_id], device=device, dtype=torch.long)

                if not logits_slice.is_contiguous():
                    logits_slice = logits_slice.contiguous()

                ce_loss = F.cross_entropy(
                    logits_slice, targets, ignore_index=ignore_index, reduction="none"
                )

                if best_loss is None or ce_loss.item() < best_loss.item():
                    best_loss = ce_loss

            if best_loss is not None:
                losses.append(best_loss)
        else:
            # For short sequences, use middle position
            pos = seq_len // 2
            logits_slice = logits[i, pos, :].unsqueeze(0)
            targets = torch.tensor([target_token_id], device=device, dtype=torch.long)

            if not logits_slice.is_contiguous():
                logits_slice = logits_slice.contiguous()

            ce_loss = F.cross_entropy(logits_slice, targets, ignore_index=ignore_index)
            losses.append(ce_loss)

    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def json_validity_loss_from_ids(
    gold_json_text_ids: torch.Tensor,
    mask_valid_json_tokens: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Compute JSON validity loss using pre-extracted JSON token IDs.

    This validates JSON structure directly from token IDs without needing
    to extract from generated text.

    Args:
        gold_json_text_ids: [B, N] or [N] tensor of JSON token IDs
        mask_valid_json_tokens: [B, N] or [N] boolean mask for valid JSON tokens
        tokenizer: Tokenizer for decoding JSON text

    Returns:
        Loss value (higher for invalid JSON)
    """
    device = gold_json_text_ids.device

    # Handle different tensor shapes
    if gold_json_text_ids.dim() == 1:
        gold_json_text_ids = gold_json_text_ids.unsqueeze(0)
        mask_valid_json_tokens = mask_valid_json_tokens.unsqueeze(0)

    validity_scores = []

    for i in range(gold_json_text_ids.size(0)):
        sample_json_ids = gold_json_text_ids[i]
        sample_mask = mask_valid_json_tokens[i]

        # Extract valid JSON tokens
        if sample_mask.dtype == torch.bool:
            valid_json_tokens = sample_json_ids[sample_mask]
        else:
            valid_json_tokens = sample_json_ids[sample_mask.bool()]

        if valid_json_tokens.numel() == 0:
            validity_scores.append(0.0)
            continue

        # Decode JSON tokens to text
        try:
            json_text = tokenizer.decode(valid_json_tokens.tolist(), skip_special_tokens=True)
            # Validate JSON structure
            is_valid = validate_json(json_text)
            validity_scores.append(1.0 if is_valid else 0.0)
        except Exception:
            # If decoding fails, consider invalid
            validity_scores.append(0.0)

    if not validity_scores:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Convert to tensor
    validity_tensor = torch.tensor(validity_scores, device=device, dtype=torch.float32)

    # Loss: 1 - validity (penalize invalid JSON)
    loss = (1.0 - validity_tensor).mean()

    return loss


def process_supervision_loss(
    logits: torch.Tensor,
    generated_texts: List[str],
    target_tool_names: Optional[List[str]] = None,
    tool_names: Optional[List[str]] = None,
    tokenizer=None,
    json_validity_weight: float = 0.3,
    tool_select_weight: float = 0.7,
    # New token ID-based arguments
    tool_name_ids: Optional[torch.Tensor] = None,
    tool_name_mask: Optional[torch.Tensor] = None,
    gold_json_text_ids: Optional[torch.Tensor] = None,
    mask_valid_json_tokens: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Combined process supervision loss.

    Args:
        logits: [B, T, V] model logits
        generated_texts: List of generated text strings (for backward compatibility)
        target_tool_names: List of target tool names (optional, for backward compatibility)
        tool_names: List of available tool names (optional)
        tokenizer: Tokenizer (required if tool_select_weight > 0)
        json_validity_weight: Weight for JSON validity loss
        tool_select_weight: Weight for tool selection loss
        tool_name_ids: [B, N] or [N] tensor of tool name token IDs (preferred over target_tool_names)
        tool_name_mask: [B, N] or [N] boolean mask for tool name tokens
        gold_json_text_ids: [B, N] or [N] tensor of JSON token IDs (preferred over generated_texts)
        mask_valid_json_tokens: [B, N] or [N] boolean mask for valid JSON tokens

    Returns:
        Dictionary with individual losses and total
    """
    losses = {}
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    # JSON validity loss - prefer token ID-based if available
    if json_validity_weight > 0:
        if gold_json_text_ids is not None and mask_valid_json_tokens is not None and tokenizer:
            # Use efficient token ID-based loss
            json_loss = json_validity_loss_from_ids(
                gold_json_text_ids, mask_valid_json_tokens, tokenizer
            )
        else:
            # Fall back to text-based extraction (backward compatibility)
            json_loss = json_validity_loss(logits, generated_texts)
        losses["json_validity"] = json_loss
        total_loss = total_loss + json_validity_weight * json_loss

    # Tool selection loss - prefer token ID-based if available
    if tool_select_weight > 0 and tokenizer:
        if tool_name_ids is not None and tool_name_mask is not None:
            # Use efficient token ID-based loss
            tool_loss = tool_selection_loss_from_ids(
                logits, tool_name_ids, tool_name_mask, tokenizer
            )
        elif target_tool_names and tool_names:
            # Fall back to text-based extraction (backward compatibility)
            tool_loss = tool_selection_loss(
                logits, generated_texts, target_tool_names, tool_names, tokenizer
            )
        else:
            tool_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        losses["tool_selection"] = tool_loss
        total_loss = total_loss + tool_select_weight * tool_loss

    losses["total"] = total_loss
    return losses
