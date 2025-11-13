"""
Deployment helper script for generating deployment artifacts.

Generates:
- Exported PyTorch model (with halt head support)
- CoreML model (if requested)
- Runtime config file (JSON) with appropriate settings
- Deployment manifest with model capabilities
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint and return metadata."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for checkpoint loading")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model architecture info
    model_arch = checkpoint.get('model_arch', {})
    config = checkpoint.get('config', {})
    arch_cfg = config.get('arch', {})

    return {
        'checkpoint': checkpoint,
        'model_arch': model_arch,
        'arch_cfg': arch_cfg,
        'step': checkpoint.get('step', 0),
    }


def generate_runtime_config(
    model_arch: Dict[str, Any],
    output_path: Path,
    latent_mode_enabled: bool = False,
    halt_head_enabled: bool = False,
    caws_tier: str = "tier_2",
) -> None:
    """Generate runtime config file."""
    try:
        from runtime.config import RuntimeConfig
        from runtime.orchestration.refine import CAWSBudgetTier

        tier_map = {
            "tier_1": CAWSBudgetTier.TIER_1,
            "tier_2": CAWSBudgetTier.TIER_2,
            "tier_3": CAWSBudgetTier.TIER_3,
        }
        caws_tier_enum = tier_map.get(caws_tier, CAWSBudgetTier.TIER_2)

        # Enable halt head if model supports it
        if model_arch.get('use_halt_head', False):
            halt_head_enabled = True

        config = RuntimeConfig(
            latent_mode_enabled=latent_mode_enabled,
            halt_head_enabled=halt_head_enabled,
            caws_tier=caws_tier_enum,
        )

        config.save(output_path)
        print(f"[deploy_model] Generated runtime config: {output_path}")
    except ImportError:
        # Fallback: create basic config dict
        config_dict = {
            "latent_mode_enabled": latent_mode_enabled,
            "halt_head_enabled": halt_head_enabled,
            "caws_tier": caws_tier,
        }
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"[deploy_model] Generated basic runtime config: {output_path}")


def generate_deployment_manifest(
    checkpoint_info: Dict[str, Any],
    pytorch_export_path: Optional[Path],
    coreml_export_path: Optional[Path],
    runtime_config_path: Path,
    output_path: Path,
) -> None:
    """Generate deployment manifest with model capabilities."""
    manifest = {
        "model_info": {
            "checkpoint_step": checkpoint_info['step'],
            "use_halt_head": checkpoint_info['model_arch'].get('use_halt_head', False),
            "use_self_evaluation": checkpoint_info['model_arch'].get('use_self_evaluation', False),
            "arch": checkpoint_info['arch_cfg'],
        },
        "artifacts": {
            "pytorch_export": str(pytorch_export_path) if pytorch_export_path else None,
            "coreml_export": str(coreml_export_path) if coreml_export_path else None,
            "runtime_config": str(runtime_config_path),
        },
        "capabilities": {
            "latent_reasoning": False,  # Set from runtime config
            "halt_head": checkpoint_info['model_arch'].get('use_halt_head', False),
            "code_mode": True,  # Always available
        },
    }

    # Load runtime config to get capabilities
    if runtime_config_path.exists():
        try:
            from runtime.config import RuntimeConfig
            config = RuntimeConfig.from_file(runtime_config_path)
            manifest["capabilities"]["latent_reasoning"] = config.latent_mode_enabled
            manifest["capabilities"]["halt_head"] = config.halt_head_enabled
        except Exception:
            # Fallback: read JSON directly
            with open(runtime_config_path, 'r') as f:
                config_dict = json.load(f)
                manifest["capabilities"]["latent_reasoning"] = config_dict.get(
                    "latent_mode_enabled", False)
                manifest["capabilities"]["halt_head"] = config_dict.get(
                    "halt_head_enabled", False)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[deploy_model] Generated deployment manifest: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        "Deploy Model - Generate deployment artifacts")
    ap.add_argument("--checkpoint", required=True,
                    help="Model checkpoint path (.pt)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for artifacts")
    ap.add_argument("--export-pytorch", action="store_true",
                    help="Export PyTorch model")
    ap.add_argument("--export-coreml", action="store_true",
                    help="Export CoreML model")
    ap.add_argument("--latent-mode", action="store_true",
                    help="Enable latent mode in runtime config")
    ap.add_argument("--halt-head", action="store_true",
                    help="Enable halt head (if model supports it)")
    ap.add_argument("--caws-tier", default="tier_2", choices=["tier_1", "tier_2", "tier_3"],
                    help="CAWS budget tier")
    ap.add_argument("--seq", type=int, default=2048,
                    help="Sequence length for export")
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[deploy_model] ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"[deploy_model] Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(checkpoint_path)

    # Export PyTorch model if requested
    pytorch_export_path = None
    if args.export_pytorch:
        print("[deploy_model] Exporting PyTorch model...")
        try:
            from conversion.export_pytorch import export_prefill, export_decode

            # Determine export paths
            prefill_path = out_dir / "model_prefill_fp16.pt"
            decode_path = out_dir / "model_decode_fp16.pt"

            # Load model
            from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
            arch_cfg = checkpoint_info['arch_cfg']
            model_cfg = ModelCfg(
                d_model=arch_cfg.get('d_model', 4096),
                n_layers=arch_cfg.get('n_layers', 32),
                n_heads=arch_cfg.get('n_heads', 32),
                n_kv_heads=arch_cfg.get('n_kv_heads', 8),
                d_head=arch_cfg.get('d_head', 128),
                vocab_size=arch_cfg.get('vocab_size', 32000),
                rope_theta=arch_cfg.get('rope_theta', 10000.0),
                rope_scaling=arch_cfg.get('rope_scaling', 'dynamic'),
                dropout=arch_cfg.get('dropout', 0.0),
            )

            use_halt_head = checkpoint_info['model_arch'].get(
                'use_halt_head', False)
            model = StudentLM(model_cfg, use_halt_head=use_halt_head)
            model.load_state_dict(
                checkpoint_info['checkpoint']['model_state_dict'], strict=False)
            model.eval()

            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "PyTorch is required for model export but not available")

            # Import torch locally to avoid reference before assignment
            import torch

            # Create example input
            example_input = torch.zeros((1, args.seq), dtype=torch.int32)
            enumerated_T = [args.seq]

            # Export prefill and decode
            export_prefill(model, example_input, prefill_path, enumerated_T)
            export_decode(model, example_input, decode_path)

            pytorch_export_path = prefill_path
            print(f"[deploy_model] PyTorch export complete: {prefill_path}")
        except Exception as e:
            print(f"[deploy_model] ERROR: PyTorch export failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Export CoreML model if requested
    coreml_export_path = None
    if args.export_coreml:
        if not pytorch_export_path:
            print("[deploy_model] ERROR: CoreML export requires PyTorch export first")
            sys.exit(1)

        print("[deploy_model] Converting to CoreML...")
        try:
            from conversion.convert_coreml import convert_pytorch_to_coreml
            import torch

            # Load TorchScript model
            pytorch_model = torch.jit.load(str(pytorch_export_path))

            # Find contract file
            contract_path = pytorch_export_path.parent / \
                f"{pytorch_export_path.stem}_contract.json"

            coreml_path = out_dir / "model.mlpackage"
            convert_pytorch_to_coreml(
                pytorch_model=pytorch_model,
                output_path=str(coreml_path),
                contract_path=str(
                    contract_path) if contract_path.exists() else None,
            )

            coreml_export_path = coreml_path
            print(f"[deploy_model] CoreML conversion complete: {coreml_path}")
        except Exception as e:
            print(f"[deploy_model] ERROR: CoreML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Generate runtime config
    runtime_config_path = out_dir / "runtime_config.json"
    generate_runtime_config(
        model_arch=checkpoint_info['model_arch'],
        output_path=runtime_config_path,
        latent_mode_enabled=args.latent_mode,
        halt_head_enabled=args.halt_head or checkpoint_info['model_arch'].get(
            'use_halt_head', False),
        caws_tier=args.caws_tier,
    )

    # Generate deployment manifest
    manifest_path = out_dir / "deployment_manifest.json"
    generate_deployment_manifest(
        checkpoint_info=checkpoint_info,
        pytorch_export_path=pytorch_export_path,
        coreml_export_path=coreml_export_path,
        runtime_config_path=runtime_config_path,
        output_path=manifest_path,
    )

    print("\n[deploy_model] Deployment artifacts generated:")
    print(f"  - Runtime config: {runtime_config_path}")
    print(f"  - Deployment manifest: {manifest_path}")
    if pytorch_export_path:
        print(f"  - PyTorch model: {pytorch_export_path}")
    if coreml_export_path:
        print(f"  - CoreML model: {coreml_export_path}")
    print("\n[deploy_model] Deployment ready!")


if __name__ == "__main__":
    main()
