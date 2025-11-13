"""
Speed metrics measurement during training validation.

Measures proxy metrics (TTFT/TPS/TTFA) during validation steps.
These are proxies for the actual CoreML/ANE measurements.

Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 4
"""
import time
from typing import Dict, List
import torch
import torch.nn as nn


def measure_proxy(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """
    Measure proxy speed metrics during validation.
    
    Reference: inference-speed-optimization-during-distillation-c3d3cffc.plan.md Phase 4
    
    These are proxies measured on PyTorch model, not CoreML/ANE.
    Tag with export=False in logs. Expected offset vs CoreML should be documented.
    
    Args:
        model: Model to measure
        batch: Batch dictionary with input_ids, attention_mask
        tokenizer: Tokenizer for decoding (for TTFA detection)
        device: Device to use
        max_new_tokens: Maximum tokens to generate for TPS measurement
        
    Returns:
        Dictionary with speed metrics:
        - ttft_ms: Time to first token (milliseconds)
        - tps: Tokens per second (steady state)
        - ttfa_tokens: Tokens until first valid tool JSON
        - ttfa_ms: Time until first valid tool JSON (milliseconds)
    """
    model.eval()
    
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Measure TTFT: time to first token
        t0 = time.perf_counter()
        logits = model(input_ids, attention_mask)
        t1 = time.perf_counter()
        ttft_ms = (t1 - t0) * 1000.0
        
        # Get first token logits (last position)
        first_token_logits = logits[:, -1, :]  # [B, V]
        first_token_id = torch.argmax(first_token_logits, dim=-1)[0].item()
        
        # Measure steady-state TPS: generate a few more tokens
        generated_tokens = [first_token_id]
        t_start = time.perf_counter()
        
        # Use KV cache if available (forward_decode)
        kv_cache = None
        current_ids = input_ids
        
        for i in range(min(max_new_tokens - 1, 32)):  # Limit to 32 for speed
            # Single token forward
            if hasattr(model, 'forward_decode'):
                # Use decode path if available
                next_logits, kv_cache = model.forward_decode(
                    torch.tensor([[generated_tokens[-1]]], device=device),
                    kv_cache,
                    pos=current_ids.size(1) + i,
                )
                next_token_id = torch.argmax(next_logits[0, 0, :]).item()
            else:
                # Fallback: full forward pass
                next_input = torch.cat([
                    current_ids,
                    torch.tensor([[generated_tokens[-1]]], device=device)
                ], dim=1)
                next_logits = model(next_input, attention_mask=None)
                next_token_id = torch.argmax(next_logits[0, -1, :]).item()
                current_ids = next_input
            
            generated_tokens.append(next_token_id)
        
        t_end = time.perf_counter()
        tokens_generated = len(generated_tokens)
        elapsed_seconds = max(1e-6, t_end - t_start)
        tps = tokens_generated / elapsed_seconds
        
        # Measure TTFA: tokens/time until first valid tool JSON
        ttfa_tokens = None
        ttfa_ms = None
        
        if tokenizer is not None:
            try:
                # Decode generated tokens to check for valid tool JSON
                decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Check if decoded text contains valid tool JSON
                # This is a simple heuristic - use your actual JSON validator for production
                if is_valid_tool_json(decoded_text):
                    ttfa_tokens = len(generated_tokens)
                    ttfa_ms = (t_end - t0) * 1000.0
            except Exception:
                pass
        
        return {
            "ttft_ms": float(ttft_ms),
            "tps": float(tps),
            "ttfa_tokens": float(ttfa_tokens) if ttfa_tokens is not None else float('inf'),
            "ttfa_ms": float(ttfa_ms) if ttfa_ms is not None else float('inf'),
            "tokens_generated": tokens_generated,
        }


def is_valid_tool_json(text: str) -> bool:
    """
    Simple heuristic to check if text contains valid tool JSON.
    
    For production, use the same validator as eval/scoring/scorer.py.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to contain valid tool JSON
    """
    # Minimal check: contains JSON-like structure
    if "{" in text and "}" in text and ":" in text:
        # Try to find tool call pattern
        if '"name"' in text or '"tool"' in text:
            return True
    return False


def aggregate_speed_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate speed metrics across multiple validation batches.
    
    Computes percentiles (p50, p90, p95) for each metric.
    
    Args:
        metrics_list: List of metric dictionaries from measure_proxy()
        
    Returns:
        Dictionary with aggregated metrics:
        - ttft_ms: {p50, p90, p95}
        - tps: {p50, p90, p95}
        - ttfa_tokens: {p50, p95}
        - ttfa_ms: {p50, p95}
    """
    import numpy as np
    
    if not metrics_list:
        return {
            "ttft_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0},
            "tps": {"p50": 0.0, "p90": 0.0, "p95": 0.0},
            "ttfa_tokens": {"p50": 0.0, "p95": 0.0},
            "ttfa_ms": {"p50": 0.0, "p95": 0.0},
        }
    
    def pct(xs, q):
        arr = np.array(xs, dtype=np.float64)
        # Filter out inf values
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0
        return float(np.nanpercentile(arr, q))
    
    ttft_values = [m["ttft_ms"] for m in metrics_list]
    tps_values = [m["tps"] for m in metrics_list]
    ttfa_tokens_values = [m["ttfa_tokens"] for m in metrics_list if m["ttfa_tokens"] != float('inf')]
    ttfa_ms_values = [m["ttfa_ms"] for m in metrics_list if m["ttfa_ms"] != float('inf')]
    
    return {
        "ttft_ms": {
            "p50": pct(ttft_values, 50),
            "p90": pct(ttft_values, 90),
            "p95": pct(ttft_values, 95),
        },
        "tps": {
            "p50": pct(tps_values, 50),
            "p90": pct(tps_values, 90),
            "p95": pct(tps_values, 95),
        },
        "ttfa_tokens": {
            "p50": pct(ttfa_tokens_values, 50) if ttfa_tokens_values else float('inf'),
            "p95": pct(ttfa_tokens_values, 95) if ttfa_tokens_values else float('inf'),
        },
        "ttfa_ms": {
            "p50": pct(ttfa_ms_values, 50) if ttfa_ms_values else float('inf'),
            "p95": pct(ttfa_ms_values, 95) if ttfa_ms_values else float('inf'),
        },
    }

