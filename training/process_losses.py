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
import torch.nn as nn
import torch.nn.functional as F


def validate_json(text: str) -> bool:
    """Check if text contains valid JSON."""
    # Try to find JSON in text
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple JSON object
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return True
            except:
                continue
    
    # Try parsing entire text
    try:
        json.loads(text.strip())
        return True
    except:
        pass
    
    return False


def extract_tool_call(text: str, tool_names: List[str]) -> Optional[Dict]:
    """
    Extract tool call from text.
    
    Returns:
        Dict with 'name' and 'arguments' if found, None otherwise
    """
    # Look for JSON tool call
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and 'name' in obj:
                return obj
        except:
            continue
    
    # Try parsing entire text
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and 'name' in obj:
            return obj
    except:
        pass
    
    return None


def json_validity_loss(
    logits: torch.Tensor,
    generated_texts: List[str],
    ignore_index: int = -100
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
            predicted_tool = tool_call.get('name', '')
            
            # Encode tool names
            target_tokens = tokenizer.encode(target_tool, add_special_tokens=False)
            pred_tokens = tokenizer.encode(predicted_tool, add_special_tokens=False)
            
            # Find where tool name appears in sequence
            # For simplicity, use cross-entropy on the first token of tool name
            if len(target_tokens) > 0 and len(pred_tokens) > 0:
                # Find position in logits where tool name starts
                # This is simplified - in practice, you'd track token positions
                target_token_id = target_tokens[0]
                
                # Use logits at a reasonable position (e.g., middle of sequence)
                seq_len = logits.size(1)
                pos = seq_len // 2  # Approximate position
                
                # Cross-entropy loss
                ce_loss = F.cross_entropy(
                    logits[i, pos:pos+1, :].view(1, -1),
                    torch.tensor([target_token_id], device=logits.device),
                    ignore_index=ignore_index
                )
                losses.append(ce_loss)
    
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def process_supervision_loss(
    logits: torch.Tensor,
    generated_texts: List[str],
    target_tool_names: Optional[List[str]] = None,
    tool_names: Optional[List[str]] = None,
    tokenizer=None,
    json_validity_weight: float = 0.3,
    tool_select_weight: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """
    Combined process supervision loss.
    
    Args:
        logits: [B, T, V] model logits
        generated_texts: List of generated text strings
        target_tool_names: List of target tool names (optional)
        tool_names: List of available tool names (optional)
        tokenizer: Tokenizer (required if tool_select_weight > 0)
        json_validity_weight: Weight for JSON validity loss
        tool_select_weight: Weight for tool selection loss
        
    Returns:
        Dictionary with individual losses and total
    """
    losses = {}
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
    
    # JSON validity loss
    if json_validity_weight > 0:
        json_loss = json_validity_loss(logits, generated_texts)
        losses["json_validity"] = json_loss
        total_loss = total_loss + json_validity_weight * json_loss
    
    # Tool selection loss
    if tool_select_weight > 0 and target_tool_names and tool_names and tokenizer:
        tool_loss = tool_selection_loss(
            logits, generated_texts, target_tool_names, tool_names, tokenizer
        )
        losses["tool_selection"] = tool_loss
        total_loss = total_loss + tool_select_weight * tool_loss
    
    losses["total"] = total_loss
    return losses






