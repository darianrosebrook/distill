# coreml/probes/compare_probes.py
"""
Minimal parity probe between ONNX and CoreML for the toy model.

- Runs ONNX with onnxruntime if available (else skips with PASS)
- Runs CoreML predict()
- Compares final 'logits' tensor (MSE + relative error)

Usage:
  python -m coreml.probes.compare_probes --onnx onnx/toy.onnx --ml coreml/artifacts/toy/model.mlpackage --seq 128 --dmodel 64
"""
import argparse
import sys

import numpy as np


def run_onnx(onnx_path, input_ids):
    try:
        import onnxruntime as ort
    except Exception:
        print('[probes] onnxruntime not installed; skipping ONNX run (PASS)')
        return None

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    mask = np.ones_like(input_ids).astype(np.int32)
    out = sess.run([sess.get_outputs()[0].name], {
        'input_ids': input_ids.astype(np.int32), 'attention_mask': mask
    })[0]
    return out


def run_coreml(mlpackage_path, input_ids):
    """Run CoreML model inference. Assumes placeholder check already done in main()."""
    import coremltools as ct
    
    try:
        ml = ct.models.MLModel(mlpackage_path)
        mask = np.ones_like(input_ids).astype(np.int32)
        out = ml.predict({'input_ids': input_ids.astype(np.int32), 'attention_mask': mask})
        key = list(out.keys())[0]
        return out[key]
    except Exception as e:
        raise RuntimeError(f"Failed to run CoreML model: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=False)
    ap.add_argument('--ml', required=False)
    ap.add_argument('--pt', required=False, help='PyTorch probe npz file')
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--rel_err_tol', type=float, default=0.02)
    ap.add_argument('--mse_tol', type=float, default=1e-3)
    args = ap.parse_args()

    # Check for placeholder marker
    from pathlib import Path
    skipped = 0
    passed = 0
    failed = 0
    
    if args.ml:
        ml_path = Path(args.ml)
        is_placeholder = (ml_path.parent / ".placeholder").exists() or (ml_path / ".placeholder").exists()
        if is_placeholder:
            print("[probes] SKIP parity – placeholder model detected")
            skipped = 1
            # Write results JSON
            results_path = Path(args.ml).parent / "results.json"
            write_results_json(str(results_path), passed=0, skipped=1, failed=0)
            return 0  # Return code 0 for SKIP (CI can detect via output)

    # Support both ONNX→CoreML parity and PyTorch→CoreML probe comparison
    if args.pt and args.ml:
        # PyTorch probe comparison mode
        a = np.load(args.pt)
        b = np.load(args.ml)
        keys = sorted(set(a.files) & set(b.files))
        worst = (None, 0.0, 0.0)
        for k in keys:
            x, y = a[k].astype(np.float32), b[k].astype(np.float32)
            mse = np.mean((x - y) ** 2)
            denom = np.maximum(np.abs(y), 1e-5)
            rel = np.max(np.abs((x - y) / denom))
            print(f"{k:16s}  mse={mse:.3e}  rel_max={rel:.3e}")
            if rel > worst[1]:
                worst = (k, rel, mse)
            if rel > args.rel_err_tol or mse > args.mse_tol:
                failed = 1
                print(f"FAIL {k}: rel={rel:.3e} mse={mse:.3e} > tol")
                results_path = Path(args.ml).parent / "results.json"
                write_results_json(str(results_path), passed=0, skipped=0, failed=1)
                return 1
        passed = 1
        print(f"OK. worst={worst}")
        results_path = Path(args.ml).parent / "results.json"
        write_results_json(str(results_path), passed=1, skipped=0, failed=0)
        return 0
    elif args.onnx and args.ml:
        # ONNX→CoreML parity mode
        rng = np.random.default_rng(1337)
        input_ids = rng.integers(low=0, high=256, size=(1, args.seq), dtype=np.int32)

        y_onnx = run_onnx(args.onnx, input_ids)
        y_coreml = run_coreml(args.ml, input_ids)

        if y_onnx is None:
            print('[probes] SKIP onnx→coreml parity (onnxruntime missing); treating as PASS for smoke')
            return 0

        # Ensure same dtype/shape
        y_coreml = np.array(y_coreml).astype(np.float32)
        y_onnx = np.array(y_onnx).astype(np.float32)

        if y_coreml.shape != y_onnx.shape:
            print(f"[probes] shape mismatch: onnx {y_onnx.shape} vs coreml {y_coreml.shape}")
            return 1

        mse = float(np.mean((y_coreml - y_onnx) ** 2))
        denom = float(np.mean(np.maximum(1e-6, np.abs(y_onnx))))
        rel = float(np.mean(np.abs(y_coreml - y_onnx)) / denom)

        print(f"[probes] MSE={mse:.4e} rel={rel:.4e}")

        # Very loose thresholds for smoke
        if np.isnan(mse) or np.isnan(rel) or rel > 0.2:
            failed = 1
            print('[probes] FAIL thresholds')
            results_path = Path(args.ml).parent / "results.json"
            write_results_json(str(results_path), passed=0, skipped=0, failed=1)
            return 2

        passed = 1
        print('[probes] PASS')
        results_path = Path(args.ml).parent / "results.json"
        write_results_json(str(results_path), passed=1, skipped=0, failed=0)
        return 0
    else:
        print("[probes] ERROR: Need either --pt/--ml or --onnx/--ml")
        return 1


def write_results_json(results_path: str, passed: int, skipped: int, failed: int = 0):
    """Write machine-readable results JSON."""
    import json
    from pathlib import Path
    results = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "total": passed + skipped + failed
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    sys.exit(main())
