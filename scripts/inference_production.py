"""
Production inference script using InferenceOrchestrator.

Supports:
- PyTorch and CoreML backends
- Latent mode and halt heads
- Refinement loops
- Efficiency tracking
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_runtime_config(config_path: Optional[Path] = None):
    """Load runtime config from file or env vars."""
    try:
        from runtime.config import RuntimeConfig

        if config_path and config_path.exists():
            return RuntimeConfig.from_file(config_path)
        else:
            return RuntimeConfig.from_env()
    except ImportError:
        # Fallback: create basic config from env vars
        return {
            "latent_mode_enabled": os.getenv("LATENT_MODE", "0") == "1",
            "halt_head_enabled": os.getenv("HALT_HEAD", "0") == "1",
            "caws_tier": os.getenv("CAWS_TIER", "tier_2"),
        }


def load_model_and_tokenizer(
    model_path: Path,
    tokenizer_path: Optional[Path] = None,
):
    """Load model and tokenizer from checkpoint or export."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for model loading")

    # Check if it's a checkpoint or exported model
    if model_path.suffix == ".pt":
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Load model config
        config_data = checkpoint.get("config", {})
        arch_cfg = config_data.get("arch", {})
        model_arch = checkpoint.get("model_arch", {})

        from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

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

        use_halt_head = model_arch.get("use_halt_head", False)
        model = StudentLM(model_cfg, use_halt_head=use_halt_head)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        # Load tokenizer
        tokenizer_path = tokenizer_path or Path("models/student/tokenizer")
    else:
        # Assume it's a CoreML model or exported PyTorch
        model = None
        tokenizer_path = tokenizer_path or Path("models/student/tokenizer")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        raise ImportError("transformers required for tokenizer loading")

    return model, tokenizer


def create_orchestrator(
    model_path: Path,
    tokenizer_path: Optional[Path],
    runtime_config,
    use_coreml: bool = False,
):
    """Create InferenceOrchestrator from model and config."""
    if use_coreml:
        # CoreML path
        try:
            from coreml.runtime.generate_coreml import load_coreml_model
            from runtime.orchestration.inference import InferenceConfig
            from runtime.orchestration.refine import CAWSBudgetTier

            load_coreml_model(str(model_path))
            tokenizer_path = tokenizer_path or Path("models/student/tokenizer")

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create inference config
            tier_map = {
                "tier_1": CAWSBudgetTier.TIER_1,
                "tier_2": CAWSBudgetTier.TIER_2,
                "tier_3": CAWSBudgetTier.TIER_3,
            }
            caws_tier = tier_map.get(
                runtime_config.caws_tier.value
                if hasattr(runtime_config, "caws_tier")
                else runtime_config.get("caws_tier", "tier_2"),
                CAWSBudgetTier.TIER_2,
            )

            InferenceConfig(
                latent_mode_enabled=runtime_config.latent_mode_enabled
                if hasattr(runtime_config, "latent_mode_enabled")
                else runtime_config.get("latent_mode_enabled", False),
                halt_head_enabled=runtime_config.halt_head_enabled
                if hasattr(runtime_config, "halt_head_enabled")
                else runtime_config.get("halt_head_enabled", False),
                caws_tier=caws_tier,
            )

            # Note: CoreML models need special handling - for now, use PyTorch path
            print(
                "[inference_production] WARN: CoreML orchestrator not fully implemented, using PyTorch"
            )
            use_coreml = False
        except Exception as e:
            print(
                f"[inference_production] WARN: CoreML loading failed: {e}, falling back to PyTorch"
            )
            use_coreml = False

    if not use_coreml:
        # PyTorch path
        try:
            from runtime.orchestration.inference import (
                create_inference_orchestrator_from_checkpoint,
            )
            from runtime.orchestration.refine import CAWSBudgetTier

            tokenizer_path = tokenizer_path or Path("models/student/tokenizer")

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Get config values
            if hasattr(runtime_config, "latent_mode_enabled"):
                latent_enabled = runtime_config.latent_mode_enabled
                halt_enabled = runtime_config.halt_head_enabled
                caws_tier_str = runtime_config.caws_tier.value
            else:
                latent_enabled = runtime_config.get("latent_mode_enabled", False)
                halt_enabled = runtime_config.get("halt_head_enabled", False)
                caws_tier_str = runtime_config.get("caws_tier", "tier_2")

            orchestrator = create_inference_orchestrator_from_checkpoint(
                checkpoint_path=model_path,
                tokenizer=tokenizer,
                latent_mode_enabled=latent_enabled,
                halt_head_enabled=halt_enabled,
                caws_tier=caws_tier_str,
            )

            return orchestrator
        except Exception as e:
            print(f"[inference_production] ERROR: Failed to create orchestrator: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


def run_inference(
    orchestrator,
    prompt: str,
    use_refinement: bool = False,
    judge_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """Run inference with orchestrator."""
    start_time = time.time()

    if use_refinement and judge_fn:
        result = orchestrator.generate_with_refinement(
            prompt=prompt,
            judge_fn=judge_fn,
        )
    else:
        output = orchestrator.generate_simple(prompt)
        result = {
            "output": output,
            "total_loops": 1,
            "refinement_history": [],
        }

    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

    return {
        "output": result["output"],
        "total_loops": result["total_loops"],
        "refinement_history": result.get("refinement_history", []),
        "latent_mode_used": result.get("latent_mode_used", False),
        "halt_head_used": result.get("halt_head_used", False),
        "curriculum_applied": result.get("curriculum_applied", False),
        "wall_clock_time_ms": elapsed_time,
    }


def main():
    ap = argparse.ArgumentParser("Production Inference using InferenceOrchestrator")
    ap.add_argument("--model", required=True, help="Model checkpoint or export path")
    ap.add_argument(
        "--tokenizer", default=None, help="Tokenizer path (default: models/student/tokenizer)"
    )
    ap.add_argument("--config", default=None, help="Runtime config file (JSON)")
    ap.add_argument("--prompt", required=True, help="Input prompt")
    ap.add_argument("--use-refinement", action="store_true", help="Use refinement loops")
    ap.add_argument("--use-coreml", action="store_true", help="Use CoreML backend")
    ap.add_argument("--output", default=None, help="Output file for results (JSON)")
    ap.add_argument("--track-efficiency", action="store_true", help="Track efficiency metrics")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[inference_production] ERROR: Model not found: {model_path}")
        sys.exit(1)

    tokenizer_path = Path(args.tokenizer) if args.tokenizer else None
    config_path = Path(args.config) if args.config else None

    # Load runtime config
    print("[inference_production] Loading runtime config...")
    runtime_config = load_runtime_config(config_path)

    # Create orchestrator
    print("[inference_production] Creating orchestrator...")
    orchestrator = create_orchestrator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        runtime_config=runtime_config,
        use_coreml=args.use_coreml,
    )

    # Simple judge function (can be replaced with actual judge model)
    def simple_judge(output: str) -> float:
        """Simple judge: score based on length and completeness."""
        score = 0.5  # Base score
        if len(output) > 50:
            score += 0.2
        if output.strip().endswith(".") or output.strip().endswith("!"):
            score += 0.3
        return min(score, 1.0)

    # Run inference
    print(f"[inference_production] Running inference with prompt: {args.prompt[:50]}...")
    result = run_inference(
        orchestrator=orchestrator,
        prompt=args.prompt,
        use_refinement=args.use_refinement,
        judge_fn=simple_judge if args.use_refinement else None,
    )

    # Print results
    print("\n[inference_production] Inference Results:")
    print(f"  Output: {result['output'][:200]}...")
    print(f"  Loops: {result['total_loops']}")
    print(f"  Time: {result['wall_clock_time_ms']:.2f} ms")
    print(f"  Latent mode: {result['latent_mode_used']}")
    print(f"  Halt head: {result['halt_head_used']}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[inference_production] Results saved to: {output_path}")

    # Track efficiency if requested
    if args.track_efficiency:
        try:
            from eval.scoring.efficiency import EfficiencyMetrics

            metrics = EfficiencyMetrics(
                accuracy=0.0,  # Would need ground truth for real accuracy
                generated_tokens=len(result["output"].split()),  # Rough estimate
                wall_clock_time_ms=result["wall_clock_time_ms"],
                latent_spans_used=result["output"].count("<bot>")
                if result["latent_mode_used"]
                else 0,
                refinement_loops=result["total_loops"],
            )

            print("\n[inference_production] Efficiency Metrics:")
            print(f"  Generated tokens: {metrics.generated_tokens}")
            print(f"  Wall clock time: {metrics.wall_clock_time_ms:.2f} ms")
            print(f"  Latent spans used: {metrics.latent_spans_used}")
            print(f"  Refinement loops: {metrics.refinement_loops}")
        except ImportError:
            print("[inference_production] WARN: Efficiency tracking not available")


if __name__ == "__main__":
    main()
