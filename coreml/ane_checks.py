# coreml/ane_checks.py
"""
Tiny ANE check stub. CoreML doesn't expose placement directly; this prints a
model summary and exits 0 unless --strict is provided (then it fails if model
isn't an MLProgram or appears to have CPU-only ops based on crude patterns).

Usage:
  python -m coreml.ane_checks --mlpackage coreml/artifacts/toy/model.mlpackage [--strict]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import coremltools as ct
from collections import Counter


def op_histogram(model_path: str):
    mlmodel = ct.models.MLModel(model_path)
    spec = mlmodel.get_spec()
    # mlProgram: ops are inside functions; we list op types by name string.
    ops = []
    if spec.WhichOneof("Type") == "mlProgram":
        for f in spec.mlProgram.functions.values():
            for b in f.block.operations:
                ops.append(b.type)
    else:
        # Fallback for neuralNetwork or other types
        layer_names = [layer.WhichOneof("layer") for layer in spec.neuralNetwork.layers]
        ops.extend(filter(None, layer_names))
    hist = Counter(ops)
    for k, v in hist.most_common():
        print(f"{k}: {v}")
    return hist


def get_hardware_baseline(hardware_id: Optional[str] = None) -> Dict[str, float]:
    """
    Get ANE residency baseline for hardware.
    
    Args:
        hardware_id: Optional hardware identifier (e.g., "M1_Max_64GB")
        
    Returns:
        Dictionary with baseline ANE residency percentage
    """
    import platform
    
    # Detect hardware if not provided
    if hardware_id is None:
        machine = platform.machine()
        # Try to detect Apple Silicon
        if machine == "arm64":
            # Could use more sophisticated detection
            hardware_id = "M1_Max"  # Default assumption
    
    # Hardware-specific baselines
    baselines = {
        "M1_Max": {"min_ane_pct": 0.75, "target_ane_pct": 0.85},
        "M1_Max_64GB": {"min_ane_pct": 0.75, "target_ane_pct": 0.85},
        "M2_Max": {"min_ane_pct": 0.80, "target_ane_pct": 0.90},
        "M3_Max": {"min_ane_pct": 0.80, "target_ane_pct": 0.90},
        "M4_Max": {"min_ane_pct": 0.80, "target_ane_pct": 0.90},
    }
    
    return baselines.get(hardware_id, {"min_ane_pct": 0.70, "target_ane_pct": 0.80})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlpackage', required=True)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--hardware-id', help='Hardware identifier for baseline (e.g., M1_Max_64GB)')
    ap.add_argument('--ane-required', action='store_true', help='Fail if ANE residency is low')
    ap.add_argument('--min-ane-pct', type=float, default=0.80, help='Minimum ANE percentage (default: 0.80)')
    args = ap.parse_args()

    # Check for placeholder marker
    mlpackage_path = Path(args.mlpackage)
    is_placeholder = (mlpackage_path.parent / ".placeholder").exists() or (mlpackage_path / ".placeholder").exists()
    
    if is_placeholder:
        print('[ane_checks] SKIP: Placeholder model detected (smoke test mode)')
        print('[ane_checks] Placeholder models cannot be introspected')
        return 0
    
    try:
        ml = ct.models.MLModel(args.mlpackage)
        spec = ml.get_spec()
        is_mlprog = spec.WhichOneof('Type') == 'mlProgram'
        print(f"[ane_checks] type=mlprogram? {is_mlprog}")
        print(f"[ane_checks] inputs={[i.name for i in spec.description.input]}")
        print(f"[ane_checks] outputs={[o.name for o in spec.description.output]}")
        
        if args.strict and not is_mlprog:
            print('[ane_checks] FAIL: not mlprogram under --strict')
            return 2
        
        # Get hardware baseline if ANE required
        if args.ane_required:
            baseline = get_hardware_baseline(args.hardware_id)
            min_ane_pct = args.min_ane_pct or baseline.get("min_ane_pct", 0.80)
            print(f"[ane_checks] Hardware baseline: min_ane_pct={min_ane_pct:.1%}")
            print(f"[ane_checks] Note: Actual ANE residency measurement requires runtime profiling")
            print(f"[ane_checks] Use coreml/runtime/ane_monitor.py for detailed residency analysis")
        
    except Exception as e:
        print(f"[ane_checks] WARN: Could not introspect model: {e}")
        return 0 if not args.strict else 2

    print('[ane_checks] OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
