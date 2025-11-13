"""
Latent curriculum wrapper for progressive CoT-to-latent replacement.

Progressive curriculum:
- S0: full CoT (no latent slots)
- S1..Sk: replace first m CoT steps with c latent slots
- Loss masking: no supervision on latent slots
- Supervise remaining visible steps + final answer
"""
# @author: @darianrosebrook

import random
from typing import Dict, Any, List
import torch

from models.student.tokenizer.constants import BOT_TOKEN, EOT_TOKEN


class LatentCurriculum:
    """
    Curriculum wrapper that progressively replaces CoT steps with latent slots.
    
    Latent slots are marked with <bot> tokens and processed without token generation,
    reducing token count while maintaining reasoning capability.
    """
    
    def __init__(self, m: int = 2, c: int = 1, p: float = 0.5):
        """
        Initialize latent curriculum wrapper.
        
        Args:
            m: Number of CoT steps to replace with latent slots
            c: Number of latent slots per replaced step (start with 1, enable c=2 when stable)
            p: Probability of applying curriculum (0.0-1.0)
        """
        self.m = m
        self.c = c
        self.p = p
    
    def apply(self, example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
        """
        Apply latent curriculum to an example.
        
        Args:
            example: Example dict with:
                - prompt: str
                - teacher_text: str (may contain CoT steps)
                - cot_steps: Optional[List[str]] (if available)
                - metadata: Dict
            tokenizer: Tokenizer for encoding tokens
        
        Returns:
            Modified example with:
                - training_text: str (with latent slots inserted)
                - loss_mask: torch.Tensor (masks latent slots)
                - metadata: Dict (updated with curriculum info)
        """
        if random.random() > self.p:
            # Don't apply curriculum: return original example
            return example
        
        # Extract CoT steps if available
        cot_steps = example.get("cot_steps")
        if cot_steps is None:
            # Try to extract from teacher_text
            teacher_text = example.get("teacher_text", "")
            cot_steps = self._extract_cot_steps(teacher_text)
        
        if not cot_steps or len(cot_steps) < self.m:
            # Not enough CoT steps to replace: return original
            return example
        
        # Build latent slots
        latent_slots = [BOT_TOKEN] * self.c
        
        # Replace first m CoT steps with latent slots
        new_steps = []
        replaced = 0
        for step in cot_steps:
            if replaced < self.m:
                new_steps.extend(latent_slots)
                replaced += 1
            else:
                new_steps.append(step)
        
        # Add <eot> at the end of latent section
        # Insert after all latent slots
        if replaced > 0:
            # Find where latent slots end
            latent_end_idx = len(latent_slots) * self.m
            new_steps.insert(latent_end_idx, EOT_TOKEN)
        
        # Get final answer (if available)
        final_answer = example.get("answer", "")
        if not final_answer:
            # Try to extract from teacher_text
            final_answer = self._extract_answer(example.get("teacher_text", ""))
        
        # Stitch together: prompt + new_steps + answer
        prompt = example.get("prompt", "")
        training_text = self._stitch_text(prompt, new_steps, final_answer)
        
        # Create loss mask: mask latent slots (no supervision)
        loss_mask = self._create_loss_mask(training_text, tokenizer, latent_slots, replaced)
        
        # Update example
        modified_example = example.copy()
        modified_example["training_text"] = training_text
        modified_example["loss_mask"] = loss_mask
        modified_example["metadata"] = example.get("metadata", {}).copy()
        modified_example["metadata"]["latent_curriculum_applied"] = True
        modified_example["metadata"]["latent_slots_count"] = len(latent_slots) * replaced
        modified_example["metadata"]["replaced_steps"] = replaced
        
        return modified_example
    
    def _extract_cot_steps(self, text: str) -> List[str]:
        """
        Extract CoT steps from text.
        
        Simple heuristic: split by common CoT markers.
        Can be improved with more sophisticated parsing.
        """
        # Common CoT markers
        markers = ["Step", "Step:", "1.", "2.", "3.", "First", "Then", "Finally"]
        steps = []
        lines = text.split("\n")
        current_step = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_step:
                    steps.append(" ".join(current_step))
                    current_step = []
                continue
            
            # Check if line starts a new step
            is_new_step = any(line.startswith(marker) for marker in markers)
            if is_new_step and current_step:
                steps.append(" ".join(current_step))
                current_step = [line]
            else:
                current_step.append(line)
        
        if current_step:
            steps.append(" ".join(current_step))
        
        return steps if steps else [text]
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract final answer from text.
        
        Simple heuristic: look for "Answer:" or "Final answer:" markers.
        """
        markers = ["Answer:", "Final answer:", "Answer is:", "Result:"]
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip()
        return text.split("\n")[-1].strip() if text else ""
    
    def _stitch_text(self, prompt: str, steps: List[str], answer: str) -> str:
        """
        Stitch together prompt, steps, and answer.
        
        Args:
            prompt: Input prompt
            steps: List of step strings (may include latent slots)
            answer: Final answer
        
        Returns:
            Combined text
        """
        parts = [prompt]
        if steps:
            parts.append("\n".join(steps))
        if answer:
            parts.append(answer)
        return "\n\n".join(parts)
    
    def _create_loss_mask(
        self,
        text: str,
        tokenizer,
        latent_slots: List[str],
        replaced_count: int,
    ) -> torch.Tensor:
        """
        Create loss mask that masks latent slots.
        
        Args:
            text: Full training text
            tokenizer: Tokenizer for encoding
            latent_slots: List of latent slot tokens
            replaced_count: Number of replaced steps
        
        Returns:
            loss_mask: [T] boolean tensor (True = supervise, False = mask)
        """
        # Encode text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        mask = torch.ones(len(tokens), dtype=torch.bool)
        
        # Find positions of latent slots (<bot> tokens)
        bot_token_id = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
        eot_token_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        
        if bot_token_id is None or eot_token_id is None:
            # Tokens not found: return all True (no masking)
            return mask
        
        # Find all <bot> and <eot> positions
        bot_positions = [i for i, tok_id in enumerate(tokens) if tok_id == bot_token_id]
        eot_positions = [i for i, tok_id in enumerate(tokens) if tok_id == eot_token_id]
        
        # Mask between <bot> and <eot> tokens
        for bot_pos in bot_positions:
            # Find corresponding <eot> (next one after this <bot>)
            eot_pos = None
            for eot_idx in eot_positions:
                if eot_idx > bot_pos:
                    eot_pos = eot_idx
                    break
            
            if eot_pos is not None:
                # Mask tokens between <bot> and <eot> (inclusive)
                mask[bot_pos:eot_pos + 1] = False
        
        return mask
    
    def mask_non_visible(self, tokens: List[int], tokenizer) -> torch.Tensor:
        """
        Create loss mask for tokens, masking latent spans.
        
        Args:
            tokens: List of token IDs
            tokenizer: Tokenizer (for token ID lookup)
        
        Returns:
            loss_mask: [T] boolean tensor
        """
        mask = torch.ones(len(tokens), dtype=torch.bool)
        bot_token_id = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
        eot_token_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        
        if bot_token_id is None or eot_token_id is None:
            return mask
        
        # Find latent spans
        bot_positions = [i for i, tok_id in enumerate(tokens) if tok_id == bot_token_id]
        eot_positions = [i for i, tok_id in enumerate(tokens) if tok_id == eot_token_id]
        
        for bot_pos in bot_positions:
            eot_pos = None
            for eot_idx in eot_positions:
                if eot_idx > bot_pos:
                    eot_pos = eot_idx
                    break
            
            if eot_pos is not None:
                mask[bot_pos:eot_pos + 1] = False
        
        return mask

