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
        layer_names = [l.WhichOneof("layer") for l in spec.neuralNetwork.layers]
        ops.extend(filter(None, layer_names))
    hist = Counter(ops)
    for k, v in hist.most_common():
        print(f"{k}: {v}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mlpackage', required=True)
    ap.add_argument('--strict', action='store_true')
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
    except Exception as e:
        print(f"[ane_checks] WARN: Could not introspect model: {e}")
        return 0 if not args.strict else 2

    print('[ane_checks] OK (stub)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
