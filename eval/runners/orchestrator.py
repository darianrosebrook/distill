"""InferenceOrchestrator-based runner for evaluation with advanced features."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from eval.runners.base import Runner

try:
    from runtime.orchestration.inference import (
        InferenceConfig,
        create_inference_orchestrator_from_checkpoint,
    )
    from runtime.orchestration.refine import CAWSBudgetTier
    from runtime.config import RuntimeConfig

    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False


class OrchestratorRunner(Runner):
    """Runner using InferenceOrchestrator for advanced features (latent mode, halt heads, refinement)."""

    def __init__(
        self,
        model: str,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tokenizer_path: Optional[str] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        use_refinement: bool = False,
        judge_fn: Optional[callable] = None,
    ):
        """
        Initialize orchestrator runner.

        Args:
            model: Model checkpoint path or identifier
            seed: Random seed (not used by orchestrator, but kept for compatibility)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tokenizer_path: Optional separate tokenizer path
            runtime_config: RuntimeConfig for feature enablement
            use_refinement: If True, use refinement loops; otherwise simple generation
            judge_fn: Optional judge function for refinement decisions
        """
        super().__init__(model, seed, temperature, max_tokens)
        self.tokenizer_path = tokenizer_path or model
        self.runtime_config = runtime_config
        self.use_refinement = use_refinement
        self.judge_fn = judge_fn
        self._orchestrator = None
        self._tokenizer = None

    def _load_orchestrator(self):
        """Lazy load orchestrator and tokenizer."""
        if self._orchestrator is None:
            if not ORCHESTRATOR_AVAILABLE:
                raise ImportError(
                    "runtime.orchestration.inference not available. "
                    "Install required dependencies or use HFLocalRunner instead."
                )

            # Load tokenizer
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                raise ImportError("transformers required for OrchestratorRunner")

            # Create inference config from runtime config or defaults
            if self.runtime_config:
                inference_config = InferenceConfig(
                    latent_mode_enabled=self.runtime_config.latent_mode_enabled,
                    halt_head_enabled=self.runtime_config.halt_head_enabled,
                    caws_tier=self.runtime_config.caws_tier,
                    max_refinement_loops=self.runtime_config.max_refinement_loops,
                    judge_score_threshold=self.runtime_config.judge_score_threshold,
                    halt_probability_threshold=self.runtime_config.halt_probability_threshold,
                    curriculum_probability=self.runtime_config.curriculum_probability,
                    curriculum_slots=self.runtime_config.curriculum_slots,
                    max_latent_spans=self.runtime_config.max_latent_spans,
                    max_latent_length=self.runtime_config.max_latent_length,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                )
            else:
                # Default config (no advanced features)
                inference_config = InferenceConfig(
                    latent_mode_enabled=False,
                    halt_head_enabled=False,
                    caws_tier=CAWSBudgetTier.TIER_2,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                )

            # Create orchestrator from checkpoint
            self._orchestrator = create_inference_orchestrator_from_checkpoint(
                checkpoint_path=self.model,
                tokenizer=self._tokenizer,
                latent_mode_enabled=inference_config.latent_mode_enabled,
                halt_head_enabled=inference_config.halt_head_enabled,
                caws_tier=inference_config.caws_tier.value,
            )

    def fingerprint(self) -> Dict[str, Any]:
        """Return runner fingerprint for reproducibility."""
        fp = super().fingerprint()
        fp["runner_type"] = "orchestrator"
        if self.runtime_config:
            fp["runtime_config"] = self.runtime_config.to_dict()
        fp["use_refinement"] = self.use_refinement
        return fp

    def generate(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate model output with tool calls using InferenceOrchestrator.

        Args:
            prompt: Input prompt
            tools: List of tool schemas (name, description, parameters)
            stop: Stop sequences (not used by orchestrator)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            seed: Override default seed (not used)

        Returns:
            Dict with:
                - "model_output": str (full model response text)
                - "tool_trace": List[Dict] (tool calls with name, arguments)
                - "latent_spans_used": int (if latent mode enabled)
                - "refinement_loops": int (if refinement enabled)
                - "halt_logits": List[float] (if halt head enabled)
        """
        self._load_orchestrator()

        # Override config if needed
        if temperature is not None:
            self._orchestrator.config.temperature = temperature
        if max_tokens is not None:
            self._orchestrator.config.max_new_tokens = max_tokens

        # Generate with or without refinement
        if self.use_refinement and self.judge_fn:
            result = self._orchestrator.generate_with_refinement(
                prompt=prompt,
                judge_fn=self.judge_fn,
            )
            model_output = result["output"]
            refinement_loops = result["total_loops"]
            refinement_history = result.get("refinement_history", [])
        else:
            model_output = self._orchestrator.generate_simple(prompt)
            refinement_loops = 1
            refinement_history = []

        # Extract tool calls from output (simplified - real implementation would parse)
        tool_trace = []
        # TODO: Integrate with tool broker for actual tool call extraction
        # For now, return empty tool trace

        # Extract metadata
        metadata = {
            "latent_spans_used": 0,
            "refinement_loops": refinement_loops,
        }

        if self.runtime_config and self.runtime_config.latent_mode_enabled:
            # Count latent spans in output (simplified)
            metadata["latent_spans_used"] = model_output.count("<bot>")

        if refinement_history:
            # Extract halt logits from last refinement step if available
            last_step = refinement_history[-1]
            if "halt_logits" in last_step:
                metadata["halt_logits"] = last_step["halt_logits"]

        return {
            "model_output": model_output,
            "tool_trace": tool_trace,
            **metadata,
        }
