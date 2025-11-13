"""
Latent-to-language probe for debugging and observability.

Provides tools to:
- Inspect hidden-step snapshots
- Visualize latent span boundaries
- Compare latent vs language representations
"""
# @author: @darianrosebrook

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path
import json

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID


class LatentProbe:
    """
    Probe for inspecting latent reasoning spans.
    
    Can extract hidden states, visualize spans, and compare
    latent vs language representations for debugging.
    """
    
    def __init__(self, tokenizer, model=None):
        """
        Initialize latent probe.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            model: Optional model for extracting hidden states
        """
        self.tokenizer = tokenizer
        self.model = model
    
    def extract_latent_spans(
        self,
        tokens: List[int],
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract latent spans from token sequence.
        
        Args:
            tokens: List of token IDs
            hidden_states: Optional list of hidden states [B, T, D] per position
        
        Returns:
            List of latent span dicts with:
            - start_pos: int
            - end_pos: int
            - length: int
            - hidden_state: Optional tensor
        """
        spans = []
        in_latent = False
        start_pos = None
        
        for i, token_id in enumerate(tokens):
            if token_id == BOT_TOKEN_ID:
                if in_latent:
                    # Nested <bot>: record previous span
                    if start_pos is not None:
                        spans.append({
                            "start_pos": start_pos,
                            "end_pos": i - 1,
                            "length": i - start_pos,
                            "hidden_state": hidden_states[i - 1] if hidden_states and i - 1 < len(hidden_states) else None,
                        })
                in_latent = True
                start_pos = i
            elif token_id == EOT_TOKEN_ID:
                if in_latent and start_pos is not None:
                    spans.append({
                        "start_pos": start_pos,
                        "end_pos": i,
                        "length": i - start_pos + 1,
                        "hidden_state": hidden_states[i] if hidden_states and i < len(hidden_states) else None,
                    })
                in_latent = False
                start_pos = None
        
        # Handle unclosed latent span
        if in_latent and start_pos is not None:
            spans.append({
                "start_pos": start_pos,
                "end_pos": len(tokens) - 1,
                "length": len(tokens) - start_pos,
                "hidden_state": hidden_states[-1] if hidden_states else None,
                "unclosed": True,
            })
        
        return spans
    
    def visualize_spans(
        self,
        tokens: List[int],
        spans: List[Dict[str, Any]],
        max_display: int = 50,
    ) -> str:
        """
        Visualize latent spans in token sequence.
        
        Args:
            tokens: List of token IDs
            spans: List of latent span dicts
            max_display: Maximum tokens to display
        
        Returns:
            Visualization string
        """
        lines = []
        lines.append("Latent Span Visualization:")
        lines.append("=" * 60)
        
        # Decode tokens
        decoded = self.tokenizer.decode(tokens[:max_display], skip_special_tokens=False)
        lines.append(f"Tokens (first {max_display}):")
        lines.append(decoded)
        lines.append("")
        
        # Show span boundaries
        lines.append("Span Boundaries:")
        for i, span in enumerate(spans):
            start = span["start_pos"]
            end = span["end_pos"]
            length = span["length"]
            unclosed = span.get("unclosed", False)
            status = " (unclosed)" if unclosed else ""
            lines.append(f"  Span {i+1}: positions {start}-{end}, length={length}{status}")
        
        lines.append("")
        
        # Show token-level markers
        lines.append("Token-level Markers:")
        marker_line = [" "] * min(len(tokens), max_display)
        for span in spans:
            start = span["start_pos"]
            end = span["end_pos"]
            if start < max_display:
                marker_line[start] = "["
            if end < max_display:
                marker_line[end] = "]"
        
        lines.append("".join(marker_line))
        
        return "\n".join(lines)
    
    def compare_representations(
        self,
        latent_hidden: torch.Tensor,
        language_hidden: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compare latent vs language hidden state representations.
        
        Args:
            latent_hidden: Hidden state from latent span [B, T, D] or [T, D]
            language_hidden: Hidden state from language tokens [B, T, D] or [T, D]
        
        Returns:
            Dict with comparison metrics:
            - cosine_similarity: float
            - l2_distance: float
            - mean_diff: float
            - max_diff: float
        """
        # Flatten if needed
        if latent_hidden.dim() == 3:
            latent_hidden = latent_hidden.view(-1, latent_hidden.size(-1))
        if language_hidden.dim() == 3:
            language_hidden = language_hidden.view(-1, language_hidden.size(-1))
        
        # Ensure same shape
        min_len = min(latent_hidden.size(0), language_hidden.size(0))
        latent_flat = latent_hidden[:min_len].flatten()
        language_flat = language_hidden[:min_len].flatten()
        
        # Cosine similarity
        latent_norm = torch.norm(latent_flat)
        language_norm = torch.norm(language_flat)
        if latent_norm > 0 and language_norm > 0:
            cosine_sim = torch.dot(latent_flat, language_flat) / (latent_norm * language_norm)
        else:
            cosine_sim = torch.tensor(0.0)
        
        # L2 distance
        l2_dist = torch.norm(latent_flat - language_flat)
        
        # Mean and max differences
        diff = latent_flat - language_flat
        mean_diff = torch.mean(torch.abs(diff))
        max_diff = torch.max(torch.abs(diff))
        
        return {
            "cosine_similarity": float(cosine_sim.item()),
            "l2_distance": float(l2_dist.item()),
            "mean_diff": float(mean_diff.item()),
            "max_diff": float(max_diff.item()),
        }
    
    def inspect_hidden_snapshot(
        self,
        hidden_state: torch.Tensor,
        position: int,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Inspect hidden state snapshot at a specific position.
        
        Args:
            hidden_state: Hidden state tensor [B, T, D] or [T, D]
            position: Position index to inspect
            top_k: Number of top dimensions to show
        
        Returns:
            Dict with inspection results:
            - position: int
            - shape: tuple
            - mean: float
            - std: float
            - top_dimensions: List[dict] with index, value
        """
        # Extract position
        if hidden_state.dim() == 3:
            pos_hidden = hidden_state[0, position, :]  # [D]
        elif hidden_state.dim() == 2:
            pos_hidden = hidden_state[position, :]  # [D]
        else:
            pos_hidden = hidden_state.flatten()
        
        # Statistics
        mean_val = float(torch.mean(pos_hidden).item())
        std_val = float(torch.std(pos_hidden).item())
        
        # Top dimensions
        top_values, top_indices = torch.topk(torch.abs(pos_hidden), top_k)
        top_dimensions = [
            {"index": int(idx.item()), "value": float(val.item())}
            for idx, val in zip(top_indices, top_values)
        ]
        
        return {
            "position": position,
            "shape": tuple(pos_hidden.shape),
            "mean": mean_val,
            "std": std_val,
            "top_dimensions": top_dimensions,
        }
    
    def export_probe_results(
        self,
        tokens: List[int],
        spans: List[Dict[str, Any]],
        output_path: Path,
    ):
        """
        Export probe results to JSON file.
        
        Args:
            tokens: List of token IDs
            spans: List of latent span dicts
            output_path: Path to output JSON file
        """
        # Convert tensors to lists for JSON serialization
        export_spans = []
        for span in spans:
            export_span = {
                "start_pos": span["start_pos"],
                "end_pos": span["end_pos"],
                "length": span["length"],
                "unclosed": span.get("unclosed", False),
            }
            if span.get("hidden_state") is not None:
                # Convert tensor to list (sample if too large)
                hidden = span["hidden_state"]
                if isinstance(hidden, torch.Tensor):
                    if hidden.numel() > 1000:
                        # Sample first 1000 elements
                        export_span["hidden_state_sample"] = hidden.flatten()[:1000].tolist()
                    else:
                        export_span["hidden_state"] = hidden.tolist()
            export_spans.append(export_span)
        
        results = {
            "tokens": tokens[:100],  # Limit token export
            "spans": export_spans,
            "total_spans": len(spans),
            "total_tokens": len(tokens),
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

