"""
Runtime engine loop with latent mode support.

Handles token generation with latent span processing:
- Detects <bot> token and switches to latent mode
- In latent mode: processes hidden states without token generation
- Detects <eot> token and switches back to language mode
- Tracks mode transitions for observability
"""
# @author: @darianrosebrook

import os
from typing import Dict, Any, Optional, Tuple, List
import torch

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID


class LatentModeEngine:
    """
    Engine for processing tokens with latent mode support.
    
    Latent mode allows processing hidden states without generating tokens,
    reducing token count while maintaining reasoning capability.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        latent_mode_enabled: bool = False,
        max_latent_length: int = 100,
        max_latent_spans: int = 10,
    ):
        """
        Initialize latent mode engine.
        
        Args:
            model: Model with forward_hidden() method
            tokenizer: Tokenizer for encoding/decoding
            latent_mode_enabled: Whether latent mode is enabled (LATENT_MODE env var)
            max_latent_length: Maximum length of latent spans (safety check)
            max_latent_spans: Maximum number of latent spans per sequence (safety check)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.latent_mode_enabled = latent_mode_enabled or (os.getenv("LATENT_MODE", "0") == "1")
        self.max_latent_length = max_latent_length
        self.max_latent_spans = max_latent_spans
        
        # Mode tracking
        self.current_mode = "language"  # "language" or "latent"
        self.mode_transitions = []
        self.latent_span_lengths = []
        self.current_latent_span_count = 0
        self.errors = []
        
    def process_token(
        self,
        token_id: int,
        hidden_state: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        pos: int = 0,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[List], Dict[str, Any]]:
        """
        Process a single token, handling latent mode transitions.
        
        Args:
            token_id: Current token ID
            hidden_state: Current hidden state [B, T, D] (for latent mode)
            kv_caches: KV caches for decode mode
            pos: Position index
            
        Returns:
            Tuple of:
            - generated_token_id: Token ID to generate (None in latent mode)
            - updated_hidden_state: Updated hidden state (if in latent mode)
            - updated_kv_caches: Updated KV caches
            - metadata: Dict with mode info, transitions, etc.
        """
        metadata = {
            "mode": self.current_mode,
            "token_id": token_id,
            "mode_changed": False,
        }
        
        if not self.latent_mode_enabled:
            # Latent mode disabled: normal processing
            return token_id, None, kv_caches, metadata
        
        # Check for mode transitions with safety validation
        if token_id == BOT_TOKEN_ID:
            # Safety check: validate transition from language to latent
            if self.current_mode == "latent":
                # Unmatched <bot>: already in latent mode
                self.errors.append(f"Unmatched <bot> at position {pos}: already in latent mode")
                metadata["error"] = "unmatched_bot"
                # Fallback: stay in latent mode, skip token
                return None, hidden_state, kv_caches, metadata
            
            # Safety check: max latent spans limit
            if self.current_latent_span_count >= self.max_latent_spans:
                self.errors.append(f"Max latent spans ({self.max_latent_spans}) reached at position {pos}")
                metadata["error"] = "max_latent_spans"
                # Fallback: ignore <bot>, stay in language mode
                return None, None, kv_caches, metadata
            
            # Valid transition: language -> latent
            self.current_mode = "latent"
            self.current_latent_span_count += 1
            self.mode_transitions.append(("latent", pos))
            metadata["mode_changed"] = True
            metadata["transition"] = "language -> latent"
            # Initialize latent span length counter
            self._current_latent_length = 0
            # In latent mode: continue processing hidden state
            return None, hidden_state, kv_caches, metadata
            
        elif token_id == EOT_TOKEN_ID:
            # Safety check: validate transition from latent to language
            if self.current_mode == "language":
                # Unmatched <eot>: not in latent mode
                self.errors.append(f"Unmatched <eot> at position {pos}: not in latent mode")
                metadata["error"] = "unmatched_eot"
                # Fallback: stay in language mode, skip token
                return None, None, kv_caches, metadata
            
            # Valid transition: latent -> language
            self.current_mode = "language"
            self.mode_transitions.append(("language", pos))
            metadata["mode_changed"] = True
            metadata["transition"] = "latent -> language"
            # Record latent span length
            if hasattr(self, "_current_latent_length"):
                self.latent_span_lengths.append(self._current_latent_length)
                delattr(self, "_current_latent_length")
            # Switch back to language mode: return None to skip this token
            return None, None, kv_caches, metadata
        
        # Process based on current mode
        if self.current_mode == "latent":
            # Latent mode: process hidden state without generating tokens
            if hidden_state is None:
                # Fallback: if no hidden state provided, exit latent mode
                self.errors.append(f"No hidden state provided in latent mode at position {pos}")
                metadata["error"] = "no_hidden_state"
                self.current_mode = "language"
                self.mode_transitions.append(("language", pos))
                metadata["mode_changed"] = True
                metadata["transition"] = "latent -> language (fallback)"
                return None, None, kv_caches, metadata
            
            # Check latent span length (safety check)
            if hasattr(self, "_current_latent_length"):
                self._current_latent_length += 1
                if self._current_latent_length > self.max_latent_length:
                    # Safety: force exit latent mode
                    self.errors.append(
                        f"Latent span length ({self._current_latent_length}) exceeds max ({self.max_latent_length})"
                    )
                    self.current_mode = "language"
                    self.mode_transitions.append(("language", pos))
                    metadata["mode_changed"] = True
                    metadata["transition"] = "latent -> language (max length)"
                    metadata["safety_triggered"] = True
                    metadata["error"] = "max_latent_length"
                    # Record the truncated span length
                    self.latent_span_lengths.append(self._current_latent_length)
                    delattr(self, "_current_latent_length")
                    return None, None, kv_caches, metadata
            else:
                # Initialize latent length counter (shouldn't happen, but safety check)
                self._current_latent_length = 1
            
            # Process hidden state through transformer blocks (no LM head)
            try:
                if hasattr(self.model, "forward_hidden"):
                    updated_hidden = self.model.forward_hidden(hidden_state)
                    return None, updated_hidden, kv_caches, metadata
                else:
                    # Fallback: model doesn't support forward_hidden
                    self.errors.append("model.forward_hidden not available")
                    metadata["error"] = "forward_hidden_not_available"
                    metadata["fallback"] = "model.forward_hidden not available"
                    # Exit latent mode on error
                    self.current_mode = "language"
                    self.mode_transitions.append(("language", pos))
                    metadata["mode_changed"] = True
                    return None, None, kv_caches, metadata
            except Exception as e:
                # Error handling: fallback to language mode
                self.errors.append(f"Error in forward_hidden: {e}")
                metadata["error"] = "forward_hidden_error"
                metadata["error_message"] = str(e)
                self.current_mode = "language"
                self.mode_transitions.append(("language", pos))
                metadata["mode_changed"] = True
                return None, None, kv_caches, metadata
        else:
            # Language mode: normal token generation
            if hasattr(self, "_current_latent_length"):
                # Reset latent length counter (shouldn't happen, but safety check)
                if self._current_latent_length > 0:
                    self.latent_span_lengths.append(self._current_latent_length)
                delattr(self, "_current_latent_length")
            return token_id, None, kv_caches, metadata
    
    def generate_with_latent_mode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        return_halt_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate tokens with latent mode support.
        
        Args:
            input_ids: [B, T] input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_halt_logits: Whether to return halt head logits
            
        Returns:
            Dict with:
            - tokens: List of generated token IDs
            - hidden_states: List of hidden states (if tracked)
            - mode_transitions: List of (mode, position) transitions
            - latent_span_lengths: List of latent span lengths
            - halt_logits: Halt head logits (if return_halt_logits=True)
        """
        if not self.latent_mode_enabled:
            # Fallback to normal generation if latent mode disabled
            return self._generate_normal(input_ids, max_new_tokens, temperature, return_halt_logits)
        
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else "cpu"
        input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        
        # Initialize state
        generated_tokens = []
        kv_caches = None
        current_hidden = None
        all_halt_logits = []
        
        # Process input tokens first
        for pos in range(input_ids.shape[-1]):
            token_id = int(input_ids[0, pos].item()) if len(input_ids.shape) > 1 else int(input_ids[pos].item())
            _, current_hidden, kv_caches, _ = self.process_token(
                token_id, current_hidden, kv_caches, pos
            )
        
        # Generate new tokens
        current_input = input_ids
        for step in range(max_new_tokens):
            # Get next token logits
            if hasattr(self.model, "forward_decode"):
                if return_halt_logits:
                    logits, kv_caches, halt_logits = self.model.forward_decode(
                        current_input[:, -1:] if len(current_input.shape) > 1 else current_input[-1:],
                        kv_caches,
                        pos=input_ids.shape[-1] + step,
                        return_halt_logits=True
                    )
                    if halt_logits is not None:
                        all_halt_logits.append(halt_logits)
                else:
                    logits, kv_caches = self.model.forward_decode(
                        current_input[:, -1:] if len(current_input.shape) > 1 else current_input[-1:],
                        kv_caches,
                        pos=input_ids.shape[-1] + step
                    )
            else:
                # Fallback: use forward method
                logits = self.model(current_input)
                logits = logits[:, -1:, :]  # Take last token
            
            # Sample token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs[0, 0], 1).item()
            else:
                next_token_id = int(logits[0, 0].argmax().item())
            
            # Process token with latent mode
            generated_token_id, current_hidden, kv_caches, metadata = self.process_token(
                next_token_id,
                current_hidden,
                kv_caches,
                pos=input_ids.shape[-1] + step
            )
            
            # Add token if not in latent mode
            if generated_token_id is not None:
                generated_tokens.append(generated_token_id)
                # Update input for next iteration
                new_token = torch.tensor([[generated_token_id]], device=device, dtype=input_ids.dtype)
                if len(current_input.shape) > 1:
                    current_input = torch.cat([current_input, new_token], dim=1)
                else:
                    current_input = torch.cat([current_input, new_token.flatten()])
            
            # Check for EOS
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Safety check: ensure we're not stuck in latent mode
        if self.current_mode == "latent":
            self.errors.append("Ended generation while still in latent mode (missing <eot>)")
            self.current_mode = "language"
            self.mode_transitions.append(("language", input_ids.shape[-1] + len(generated_tokens)))
        
        result = {
            "tokens": generated_tokens,
            "mode_transitions": self.mode_transitions,
            "latent_span_lengths": self.latent_span_lengths,
            "errors": self.errors,
            "final_mode": self.current_mode,
        }
        
        if return_halt_logits and all_halt_logits:
            result["halt_logits"] = torch.cat(all_halt_logits, dim=0) if isinstance(all_halt_logits[0], torch.Tensor) else all_halt_logits
        
        return result
    
    def _generate_normal(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        return_halt_logits: bool,
    ) -> Dict[str, Any]:
        """Fallback normal generation when latent mode is disabled."""
        # Simple generation without latent mode
        generated_tokens = []
        current_input = input_ids
        
        for _ in range(max_new_tokens):
            logits = self.model(current_input)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token_id = torch.multinomial(probs[0], 1).item()
            else:
                next_token_id = int(logits[0, -1, :].argmax().item())
            
            generated_tokens.append(next_token_id)
            new_token = torch.tensor([[next_token_id]], device=input_ids.device, dtype=input_ids.dtype)
            current_input = torch.cat([current_input, new_token], dim=1)
            
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        return {
            "tokens": generated_tokens,
            "mode_transitions": [],
            "latent_span_lengths": [],
        }

