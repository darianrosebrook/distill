"""
High-level inference orchestrator that coordinates latent mode, refinement loops, and halt heads.

This is the main entry point for running inference with all the advanced features:
- Latent reasoning with curriculum
- Learned halting with halt heads
- CAWS budget enforcement
- Efficiency tracking
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch

from models.student.architectures.gqa_transformer import StudentLM
from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID
from runtime.engine.loop import LatentModeEngine
from runtime.orchestration.refine import RefinementController, CAWSBudgetTier
from data.wrappers.curriculum import LatentCurriculum
from training.halt_targets import HaltHeadTargets


class InferenceConfig:
    """Configuration for inference orchestration."""

    def __init__(
        self,
        latent_mode_enabled: bool = False,
        halt_head_enabled: bool = False,
        caws_tier: CAWSBudgetTier = CAWSBudgetTier.TIER_2,
        max_refinement_loops: int = 5,
        judge_score_threshold: float = 0.8,
        halt_probability_threshold: float = 0.7,
        curriculum_probability: float = 1.0,
        curriculum_slots: int = 1,
        max_latent_spans: int = 10,
        max_latent_length: int = 100,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
    ):
        self.latent_mode_enabled = latent_mode_enabled
        self.halt_head_enabled = halt_head_enabled
        self.caws_tier = caws_tier
        self.max_refinement_loops = max_refinement_loops
        self.judge_score_threshold = judge_score_threshold
        self.halt_probability_threshold = halt_probability_threshold
        self.curriculum_probability = curriculum_probability
        self.curriculum_slots = curriculum_slots
        self.max_latent_spans = max_latent_spans
        self.max_latent_length = max_latent_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens


class InferenceOrchestrator:
    """
    High-level orchestrator for inference with latent reasoning and learned halting.

    Coordinates:
    - Model inference (with latent mode support)
    - Refinement loop management
    - Halt head decision making
    - Curriculum application
    - Efficiency tracking
    """

    def __init__(
        self,
        model: StudentLM,
        tokenizer,
        config: InferenceConfig,
    ):
        """
        Initialize inference orchestrator.

        Args:
            model: Student model with optional halt head
            tokenizer: Tokenizer for encoding/decoding
            config: Inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize components
        self.latent_engine = LatentModeEngine(
            model=model,
            tokenizer=tokenizer,
            latent_mode_enabled=config.latent_mode_enabled,
            max_latent_length=config.max_latent_length,
            max_latent_spans=config.max_latent_spans,
        )

        self.refinement_controller = RefinementController(
            caws_tier=config.caws_tier,
            halt_head_enabled=config.halt_head_enabled,
            judge_score_threshold=config.judge_score_threshold,
            halt_probability_threshold=config.halt_probability_threshold,
            max_loops=config.max_refinement_loops,
        )

        self.curriculum = LatentCurriculum(
            m=config.curriculum_slots,
            c=config.curriculum_slots,
            p=config.curriculum_probability,
        )

        # Initialize halt targets if needed
        self.halt_targets = None
        if config.halt_head_enabled:
            self.halt_targets = HaltHeadTargets(
                judge_score_threshold=config.judge_score_threshold,
                caws_tier=config.caws_tier.value,
            )

    def generate_with_refinement(
        self,
        prompt: str,
        judge_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate text with refinement loops and latent reasoning.

        Args:
            prompt: Input prompt
            judge_fn: Optional judge function for refinement decisions

        Returns:
            Dict with generation results and metadata
        """
        # Encode prompt
        torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)

        # Initialize refinement tracking
        refinement_history = []
        current_output = ""
        best_output = ""
        best_score = 0.0
        loop_count = 0

        while loop_count < self.config.max_refinement_loops:
            loop_count += 1

            # Generate with current prompt + previous output
            full_prompt = prompt
            if current_output:
                full_prompt += f"\n\nPrevious attempt:\n{current_output}"

            # Apply latent curriculum if enabled
            curriculum_example = {
                "prompt": full_prompt,
                "teacher_text": full_prompt,  # Simplified for inference
                "cot_steps": [],
                "answer": "",
                "metadata": {},
            }

            if self.config.latent_mode_enabled and self.config.curriculum_probability > 0:
                curriculum_result = self.curriculum.apply(curriculum_example, self.tokenizer)
                generation_prompt = curriculum_result.get("training_text", full_prompt)
            else:
                generation_prompt = full_prompt
                curriculum_result = curriculum_example

            # Encode generation prompt
            gen_input_ids = torch.tensor(
                [self.tokenizer.encode(generation_prompt)], dtype=torch.long
            )

            # Generate with latent mode
            generation_result = self.latent_engine.generate_with_latent_mode(
                gen_input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                return_halt_logits=self.config.halt_head_enabled,
            )

            # Decode output (skip latent spans)
            generated_tokens = generation_result["tokens"]
            # Filter out latent tokens for final output
            filtered_tokens = []
            in_latent_span = False
            for token in generated_tokens:
                if token == BOT_TOKEN_ID:
                    in_latent_span = True
                elif token == EOT_TOKEN_ID:
                    in_latent_span = False
                elif not in_latent_span:
                    filtered_tokens.append(token)

            current_output = self.tokenizer.decode(filtered_tokens)

            # Evaluate with judge if available
            current_score = 0.0
            halt_logits = generation_result.get("halt_logits")
            if judge_fn:
                current_score = judge_fn(current_output)
                if current_score > best_score:
                    best_score = current_score
                    best_output = current_output

            # Record refinement step
            step_info = {
                "loop": loop_count,
                "output": current_output,
                "score": current_score,
                "latent_spans_used": len(generation_result.get("latent_span_lengths", [])),
                "halt_logits": halt_logits.tolist() if halt_logits is not None else None,
                "mode_transitions": generation_result.get("mode_transitions", []),
            }
            refinement_history.append(step_info)

            # Check if we should halt
            should_halt, halt_metadata = self.refinement_controller.should_halt(
                current_output=current_output,
                judge_score=current_score,
                halt_logits=halt_logits,
                latent_spans_used=len(generation_result.get("latent_span_lengths", [])),
            )

            if should_halt:
                step_info["halt_reason"] = halt_metadata.get("halt_reason", "unknown")
                break

        # Return final result
        final_output = best_output if best_output else current_output

        return {
            "output": final_output,
            "final_score": best_score,
            "total_loops": loop_count,
            "refinement_history": refinement_history,
            "latent_mode_used": self.config.latent_mode_enabled,
            "halt_head_used": self.config.halt_head_enabled,
            "curriculum_applied": self.config.latent_mode_enabled
            and self.config.curriculum_probability > 0,
        }

    def generate_simple(
        self,
        prompt: str,
    ) -> str:
        """
        Simple generation without refinement loops.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)

        result = self.latent_engine.generate_with_latent_mode(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            return_halt_logits=False,
        )

        # Filter latent tokens
        generated_tokens = result["tokens"]
        filtered_tokens = []
        in_latent_span = False
        for token in generated_tokens:
            if token == BOT_TOKEN_ID:
                in_latent_span = True
            elif token == EOT_TOKEN_ID:
                in_latent_span = False
            elif not in_latent_span:
                filtered_tokens.append(token)

        return self.tokenizer.decode(filtered_tokens)


def create_inference_orchestrator_from_checkpoint(
    checkpoint_path: Union[str, Path],
    tokenizer,
    latent_mode_enabled: bool = False,
    halt_head_enabled: bool = False,
    caws_tier: str = "tier_2",
) -> InferenceOrchestrator:
    """
    Create inference orchestrator from trained checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer: Tokenizer instance
        latent_mode_enabled: Enable latent mode
        halt_head_enabled: Enable halt head (if model supports it)
        caws_tier: CAWS budget tier

    Returns:
        Configured InferenceOrchestrator
    """
    # Load checkpoint safely with structure validation
    from training.safe_checkpoint_loading import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")

    # Load model config
    config_data = checkpoint.get("config", {})
    arch_cfg = config_data.get("arch", {})

    # Get model architecture flags
    model_arch = checkpoint.get("model_arch", {})
    use_halt_head = model_arch.get("use_halt_head", False) and halt_head_enabled

    # Create model config
    from models.student.architectures.gqa_transformer import ModelCfg

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

    # Create model
    model = StudentLM(model_cfg, use_halt_head=use_halt_head, use_self_evaluation=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Create inference config
    tier_map = {
        "tier_1": CAWSBudgetTier.TIER_1,
        "tier_2": CAWSBudgetTier.TIER_2,
        "tier_3": CAWSBudgetTier.TIER_3,
    }
    caws_tier_enum = tier_map.get(caws_tier, CAWSBudgetTier.TIER_2)

    inference_config = InferenceConfig(
        latent_mode_enabled=latent_mode_enabled,
        halt_head_enabled=use_halt_head,
        caws_tier=caws_tier_enum,
    )

    return InferenceOrchestrator(model, tokenizer, inference_config)

