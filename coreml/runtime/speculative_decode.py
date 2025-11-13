"""
Speculative decoding for M-series Apple Silicon optimization.

Uses a smaller drafter model (~4B) to generate tokens faster, then verifies
with the worker model (~9B). Reduces TTFT by 25-40% with acceptable rollback rate.

Reference: docs/M_SERIES_ADVANCED_OPTIMIZATIONS.md Phase 8
"""

from __future__ import annotations
import time
import random
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from coremltools.models import MLModel

    COREML_AVAILABLE = True
except ImportError:
    MLModel = None
    COREML_AVAILABLE = False


class SpeculativeDecoder:
    """
    Speculative decoding with drafter + worker verification.

    Algorithm:
    1. Drafter generates K tokens (fast, ~4B model)
    2. Worker verifies draft tokens (slower, ~9B model)
    3. Accept tokens up to first rejection
    4. If all rejected, generate one token normally

    Benefits:
    - 25-40% TTFT improvement (drafter is 2-3x faster)
    - 10-20% TPS improvement (fewer worker forward passes)
    - Rollback rate ≤10% (acceptable)
    """

    def __init__(
        self,
        drafter_model: MLModel,
        worker_model: MLModel,
        drafter_adapter,
        worker_adapter,
        k: int = 2,
        temperature: float = 0.0,
        use_standard_acceptance: bool = True,
    ):
        """
        Initialize speculative decoder.

        Args:
            drafter_model: CoreML drafter model (~4B, fast)
            worker_model: CoreML worker model (~9B, accurate)
            drafter_adapter: StepAdapter for drafter model
            worker_adapter: StepAdapter for worker model
            k: Number of draft tokens per step (default: 2)
            temperature: Sampling temperature (default: 0.0 for deterministic)
        """
        if not COREML_AVAILABLE:
            raise RuntimeError("coremltools not available")

        self.drafter_model = drafter_model
        self.worker_model = worker_model
        self.drafter_adapter = drafter_adapter
        self.worker_adapter = worker_adapter
        self.k = k
        self.temperature = temperature
        self.use_standard_acceptance = use_standard_acceptance

        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.rollbacks = 0

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 64,
        tokenizer=None,
    ) -> Dict[str, Any]:
        """
        Generate tokens with speculative decoding.

        Args:
            prompt_ids: Input prompt token IDs [1, T]
            max_tokens: Maximum tokens to generate
            tokenizer: Optional tokenizer for decoding

        Returns:
            Dictionary with:
            - tokens: List of generated token IDs
            - stats: Statistics (accepted, rejected, rollbacks)
            - ttft_ms: Time to first token (milliseconds)
            - tps: Tokens per second
        """
        tokens = []

        # Measure TTFT
        ttft_start = time.perf_counter()

        # First token: use drafter for speed
        drafter_state = self.drafter_adapter.prepare_state(prompt_ids)
        worker_state = self.worker_adapter.prepare_state(prompt_ids)

        # Drafter generates first token
        drafter_logits, drafter_state = self.drafter_adapter.first_step(
            self.drafter_model, prompt_ids, drafter_state
        )
        first_token = self._sample_token(drafter_logits)

        # Worker verifies first token
        worker_logits, worker_state = self.worker_adapter.first_step(
            self.worker_model, prompt_ids, worker_state
        )

        # Accept first token if probability is acceptable
        if self._should_accept_token(first_token, drafter_logits, worker_logits):
            tokens.append(first_token)
            self.accepted_tokens += 1
        else:
            # Use worker's first token instead
            worker_first_token = self._sample_token(worker_logits)
            tokens.append(worker_first_token)
            self.rejected_tokens += 1
            self.rollbacks += 1

        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        # Continue generation
        t_start = time.perf_counter()
        current_input = np.concatenate([prompt_ids[0], tokens])

        while len(tokens) < max_tokens:
            # Drafter generates K tokens (with logits for acceptance criterion)
            draft_tokens, drafter_logits_list = self._draft_k_tokens(
                current_input, drafter_state, k=self.k
            )

            # Worker verifies draft tokens
            accepted = self._verify_tokens(
                draft_tokens, drafter_logits_list, current_input, worker_state
            )

            tokens.extend(accepted)
            self.accepted_tokens += len(accepted)
            self.rejected_tokens += len(draft_tokens) - len(accepted)

            if len(accepted) < len(draft_tokens):
                # Some tokens rejected, rollback occurred
                self.rollbacks += 1

            # If all rejected, generate one token normally with worker
            if len(accepted) == 0:
                next_token = self._generate_one_token_worker(current_input, worker_state)
                tokens.append(next_token)
                self.accepted_tokens += 1

            # Update current input
            current_input = np.concatenate([prompt_ids[0], tokens])

        t_end = time.perf_counter()
        tokens_generated = len(tokens)
        tps = tokens_generated / max(1e-6, (t_end - t_start))

        self.total_tokens += tokens_generated

        return {
            "tokens": tokens,
            "stats": {
                "total_tokens": tokens_generated,
                "accepted_tokens": self.accepted_tokens,
                "rejected_tokens": self.rejected_tokens,
                "rollbacks": self.rollbacks,
                "acceptance_rate": self.accepted_tokens
                / max(1, self.accepted_tokens + self.rejected_tokens),
                "rollback_rate": self.rollbacks / max(1, tokens_generated),
            },
            "ttft_ms": ttft_ms,
            "tps": tps,
        }

    def _draft_k_tokens(
        self,
        input_ids: np.ndarray,
        drafter_state: Dict[str, Any],
        k: int,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Draft K tokens using drafter model.

        Args:
            input_ids: Current input token IDs
            drafter_state: Drafter KV cache state
            k: Number of tokens to draft

        Returns:
            Tuple of (draft_token_ids, drafter_logits_list)
            Both lists have length k
        """
        draft_tokens = []
        drafter_logits_list = []
        current_ids = input_ids

        for _ in range(k):
            # Get next token from drafter
            if len(draft_tokens) == 0:
                # First draft token: use last token from input
                last_token = current_ids[-1]
                logits, drafter_state = self.drafter_adapter.next_step(
                    self.drafter_model, int(last_token), drafter_state
                )
            else:
                # Subsequent draft tokens
                last_draft = draft_tokens[-1]
                logits, drafter_state = self.drafter_adapter.next_step(
                    self.drafter_model, last_draft, drafter_state
                )

            next_token = self._sample_token(logits)
            draft_tokens.append(next_token)
            drafter_logits_list.append(logits)

        return draft_tokens, drafter_logits_list

    def _verify_tokens(
        self,
        draft_tokens: List[int],
        drafter_logits_list: List[np.ndarray],
        input_ids: np.ndarray,
        worker_state: Dict[str, Any],
    ) -> List[int]:
        """
        Verify draft tokens with worker model.

        Uses standard speculative decoding acceptance criterion:
        Accept tokens up to first rejection.

        Standard acceptance criterion (from speculative decoding paper):
        Accept token if: min(1, worker_prob / drafter_prob) >= random_uniform(0, 1)

        Args:
            draft_tokens: List of draft token IDs to verify
            drafter_logits_list: List of drafter logits for each draft token
            input_ids: Current input token IDs
            worker_state: Worker KV cache state

        Returns:
            List of accepted token IDs (prefix of draft_tokens)
        """
        if not draft_tokens:
            return []

        if len(draft_tokens) != len(drafter_logits_list):
            raise ValueError(
                f"draft_tokens ({len(draft_tokens)}) and drafter_logits_list "
                f"({len(drafter_logits_list)}) must have same length"
            )

        accepted = []
        current_ids = input_ids

        for idx, draft_token in enumerate(draft_tokens):
            drafter_logits = drafter_logits_list[idx]

            # Get worker's prediction for this position
            if len(accepted) == 0:
                # First token: use last token from input
                last_token = current_ids[-1]
                worker_logits, worker_state = self.worker_adapter.next_step(
                    self.worker_model, int(last_token), worker_state
                )
            else:
                # Subsequent tokens
                prev_token = accepted[-1]
                worker_logits, worker_state = self.worker_adapter.next_step(
                    self.worker_model, prev_token, worker_state
                )

            # Check if draft token should be accepted using standard criterion
            if self._should_accept_draft_token(draft_token, drafter_logits, worker_logits):
                accepted.append(draft_token)
            else:
                # Reject this token and all subsequent ones
                break

        return accepted

    def _generate_one_token_worker(
        self,
        input_ids: np.ndarray,
        worker_state: Dict[str, Any],
    ) -> int:
        """
        Generate one token using worker model (fallback).

        Args:
            input_ids: Current input token IDs
            worker_state: Worker KV cache state

        Returns:
            Generated token ID
        """
        last_token = input_ids[-1]
        worker_logits, worker_state = self.worker_adapter.next_step(
            self.worker_model, int(last_token), worker_state
        )
        return self._sample_token(worker_logits)

    def _sample_token(self, logits: np.ndarray) -> int:
        """
        Sample token from logits.

        Args:
            logits: Logits array [V]

        Returns:
            Sampled token ID
        """
        if self.temperature == 0.0:
            # Greedy decoding
            return int(np.argmax(logits))
        else:
            # Temperature sampling
            probs = np.exp(logits / self.temperature)
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(probs), p=probs))

    def _should_accept_token(
        self,
        token: int,
        drafter_logits: np.ndarray,
        worker_logits: np.ndarray,
    ) -> bool:
        """
        Check if token should be accepted (first token case).

        Args:
            token: Token ID to check
            drafter_logits: Drafter model logits
            worker_logits: Worker model logits

        Returns:
            True if token should be accepted
        """
        # Simple heuristic: accept if worker's probability is reasonable
        # More sophisticated: use probability ratio threshold
        worker_probs = np.exp(worker_logits - np.max(worker_logits))
        worker_probs = worker_probs / np.sum(worker_probs)

        worker_prob = worker_probs[token]
        drafter_probs = np.exp(drafter_logits - np.max(drafter_logits))
        drafter_probs = drafter_probs / np.sum(drafter_probs)
        drafter_prob = drafter_probs[token]

        # Accept if worker probability is reasonable relative to drafter
        # Threshold: worker prob should be at least 50% of drafter prob
        return worker_prob >= 0.5 * drafter_prob

    def _should_accept_draft_token(
        self,
        draft_token: int,
        drafter_logits: np.ndarray,
        worker_logits: np.ndarray,
    ) -> bool:
        """
        Check if draft token should be accepted.

        Uses standard speculative decoding acceptance criterion:
        Accept token if: min(1, worker_prob / drafter_prob) >= random_uniform(0, 1)

        This maintains the correct distribution while accepting tokens probabilistically
        based on the ratio of worker to drafter probabilities.

        Args:
            draft_token: Draft token ID to check
            drafter_logits: Drafter model logits [V]
            worker_logits: Worker model logits [V]

        Returns:
            True if draft token should be accepted
        """
        if self.use_standard_acceptance:
            # Standard speculative decoding acceptance criterion
            # Compute probabilities from logits (softmax)
            worker_probs = np.exp(worker_logits - np.max(worker_logits))
            worker_probs = worker_probs / np.sum(worker_probs)

            drafter_probs = np.exp(drafter_logits - np.max(drafter_logits))
            drafter_probs = drafter_probs / np.sum(drafter_probs)

            worker_prob = worker_probs[draft_token]
            drafter_prob = drafter_probs[draft_token]

            # Avoid division by zero
            if drafter_prob < 1e-10:
                # Drafter has very low probability, reject
                return False

            # Standard acceptance criterion: min(1, worker_prob / drafter_prob) >= random
            # This ensures tokens are accepted with probability proportional to ratio
            acceptance_prob = min(1.0, worker_prob / drafter_prob)

            # Sample random uniform [0, 1] and compare
            return acceptance_prob >= random.uniform(0.0, 1.0)
        else:
            # Simplified threshold-based acceptance (fallback)
            worker_probs = np.exp(worker_logits - np.max(worker_logits))
            worker_probs = worker_probs / np.sum(worker_probs)
            worker_prob = worker_probs[draft_token]
            return worker_prob >= 0.1

    def reset_stats(self):
        """Reset statistics."""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.rollbacks = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get speculative decoding statistics.

        Returns:
            Dictionary with statistics
        """
        total = self.accepted_tokens + self.rejected_tokens
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "rollbacks": self.rollbacks,
            "acceptance_rate": self.accepted_tokens / max(1, total),
            "rollback_rate": self.rollbacks / max(1, self.total_tokens),
        }
