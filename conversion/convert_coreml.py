"""
Convert ONNX → CoreML (mlprogram). Uses public MIL converter API.

Usage:
  python -m conversion.convert_coreml \
    --backend onnx \
    --in onnx/toy.sanitized.onnx \
    --out coreml/artifacts/toy/model.mlpackage \
    --target macOS13 \
    --allow-placeholder   # optional; otherwise fails loud
"""
import argparse
import os
import sys
from pathlib import Path

# Suppress ANE remote proxy warnings for cleaner output
os.environ.setdefault("MLTOOLS_VERBOSE", "0")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="coremltools")


def load_contract(contract_path: str):
    """Load contract.json and return input specifications."""
    import json
    with open(contract_path, 'r') as f:
        contract = json.load(f)
    return contract


def convert_pytorch_to_coreml(
    pytorch_model,
    output_path: str,
    compute_units: str = "all",
    target: str = "macOS13",
    allow_placeholder: bool = False,
    contract_path: str = None,
):
    """
    Convert PyTorch model (TorchScript or ExportedProgram) to CoreML.
    
    Args:
        pytorch_model: TorchScript module or torch.export.ExportedProgram
        output_path: Output path for .mlpackage
        compute_units: "all", "cpuandgpu", or "cpuonly"
        target: Deployment target (e.g., "macOS13")
        allow_placeholder: If True, create placeholder on failure instead of raising
        
    Returns:
        Path to converted model or None if placeholder created
    """
    import coremltools as ct
    
    cu_map = {
        'all': ct.ComputeUnit.ALL,
        'cpuandgpu': ct.ComputeUnit.CPU_AND_GPU,
        'cpuonly': ct.ComputeUnit.CPU_ONLY
    }
    compute_unit = cu_map.get(compute_units, ct.ComputeUnit.ALL)
    
    # Parse target
    if target.startswith("macOS"):
        target_version = ct.target.macOS13
        try:
            version_num = int(target.replace("macOS", ""))
            if version_num >= 14:
                target_version = ct.target.macOS14
        except:
            pass
    else:
        target_version = ct.target.macOS13
    
    print(f"[convert_coreml] Converting PyTorch→CoreML (target={target}, compute_units={compute_units})")
    
    try:
        # For TorchScript, we need to provide input specification
        # Try to read from contract.json first, fallback to defaults
        import torch
        import numpy as np
        
        if isinstance(pytorch_model, torch.jit.ScriptModule):
            # Try to load contract.json for input specs
            if contract_path and Path(contract_path).exists():
                contract = load_contract(contract_path)
                input_spec = contract.get("inputs", [{}])[0]
                input_name = input_spec.get("name", "input_ids")
                input_shape = input_spec.get("shape", ["B", "T"])
                input_dtype = input_spec.get("dtype", "int32")
                
                # Convert shape: ["B", "T"] -> (1, T) for example
                # Use first enumerated_T if available, else default to 128
                enumerated_T = contract.get("enumerated_T", [128])
                seq_len = enumerated_T[0] if enumerated_T else 128
                shape = (1, seq_len)
                
                # Convert dtype string to numpy dtype
                dtype_map = {"int32": np.int32, "int64": np.int64, "float32": np.float32, "float16": np.float16}
                dtype = dtype_map.get(input_dtype, np.int32)
                
                inputs = [ct.TensorType(name=input_name, shape=shape, dtype=dtype)]
                print(f"[convert_coreml] Using contract.json: {input_name} shape={shape} dtype={input_dtype}")
            else:
                # Fallback: try to infer input shape from model
                # Run a dummy forward pass to get input shape
                try:
                    dummy_input = torch.zeros((1, 128), dtype=torch.int32)
                    _ = pytorch_model(dummy_input)
                    # If that worked, use int32 input
                    inputs = [ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32)]
                except Exception:
                    # Try float32 input (for transformer blocks)
                    try:
                        dummy_input = torch.randn((1, 128, 64), dtype=torch.float32)
                        _ = pytorch_model(dummy_input)
                        inputs = [ct.TensorType(name="input", shape=(1, 128, 64), dtype=np.float32)]
                    except Exception:
                        # Last resort: use defaults
                        inputs = [ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32)]
                        print(f"[convert_coreml] WARN: Could not infer input shape, using defaults")
            mlmodel = ct.convert(
                pytorch_model,
                inputs=inputs,
                source="pytorch",
                compute_units=compute_unit,
                minimum_deployment_target=target_version,
                convert_to="mlprogram",
            )
        else:
            # For ExportedProgram, inputs are already embedded
            mlmodel = ct.convert(
                pytorch_model,
                source="pytorch",
                compute_units=compute_unit,
                minimum_deployment_target=target_version,
                convert_to="mlprogram",
            )
        print("[convert_coreml] Conversion successful")
        
        # Save model
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path_obj))
        print(f"[convert_coreml] Saved → {output_path}")
        return output_path
        
    except Exception as e:
        if allow_placeholder:
            print(f"[convert_coreml] WARN: Conversion failed: {e}")
            print("[convert_coreml] Creating placeholder (SKIP parity)")
            return create_placeholder(output_path, "PyTorch model", str(e))
        else:
            print(f"[convert_coreml] ERROR: PyTorch→CoreML conversion failed")
            print(f"[convert_coreml] Error: {e}")
            raise


def convert_onnx_to_coreml(
    onnx_path: str,
    output_path: str,
    compute_units: str = "all",
    target: str = "macOS13",
    allow_placeholder: bool = False,
):
    """
    Convert ONNX model to CoreML using public MIL converter API.
    
    Args:
        onnx_path: Path to ONNX model file
        output_path: Output path for .mlpackage
        compute_units: "all", "cpuandgpu", or "cpuonly"
        target: Deployment target (e.g., "macOS13")
        allow_placeholder: If True, create placeholder on failure instead of raising
        
    Returns:
        Path to converted model or None if placeholder created
    """
    import coremltools as ct
    import onnx
    
    cu_map = {
        'all': ct.ComputeUnit.ALL,
        'cpuandgpu': ct.ComputeUnit.CPU_AND_GPU,
        'cpuonly': ct.ComputeUnit.CPU_ONLY
    }
    compute_unit = cu_map.get(compute_units, ct.ComputeUnit.ALL)
    
    # Parse target
    if target.startswith("macOS"):
        target_version = ct.target.macOS13  # Default to macOS13
        try:
            version_num = int(target.replace("macOS", ""))
            if version_num >= 14:
                target_version = ct.target.macOS14
        except:
            pass
    else:
        target_version = ct.target.macOS13
    
    print(f"[convert_coreml] Loading ONNX model: {onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    print(f"[convert_coreml] Converting ONNX→CoreML (target={target}, compute_units={compute_units})")
    
    try:
        # Try public API: ct.convert with auto-detection
        mlmodel = ct.convert(
            onnx_model,
            compute_units=compute_unit,
            minimum_deployment_target=target_version,
            convert_to="mlprogram",
        )
        print("[convert_coreml] Conversion successful")
        
    except Exception as e:
        # ONNX is not a supported production path - always create placeholder
        if allow_placeholder:
            print(f"[convert_coreml] WARN: ONNX conversion not supported by CoreMLTools")
            print("[convert_coreml] Creating placeholder (SKIP parity)")
            error_msg = (
                f"ONNX→CoreML conversion is not a supported production path. "
                f"CoreMLTools 9.0 only supports TensorFlow and PyTorch. "
                f"For production, use --backend pytorch with a TorchScript or ExportedProgram model."
            )
            return create_placeholder(output_path, onnx_path, error_msg)
        else:
            print(f"[convert_coreml] ERROR: ONNX→CoreML conversion not supported")
            print("\n[convert_coreml] REMEDIATION:")
            print("  ONNX is not a supported production input to CoreML.")
            print("  For production conversion:")
            print("  1. Export your model as TorchScript or torch.export ExportedProgram")
            print("  2. Use --backend pytorch with the PyTorch model")
            print("  3. Use --allow-placeholder only for smoke tests")
            sys.exit(2)
    
    # Apply FP16 quantization if requested (optional)
    # Note: mlprogram already uses FP16 by default
    
    # Save model
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path_obj))
    print(f"[convert_coreml] Saved → {output_path}")
    return output_path


def create_placeholder(output_path: str, onnx_path: str, error_msg: str):
    """Create a placeholder .mlpackage for smoke tests."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    
    # Create .placeholder marker
    placeholder_marker = out.parent / ".placeholder"
    placeholder_marker.write_text("")
    
    # Create CONVERSION_NOTE.txt
    note_file = out.parent / "CONVERSION_NOTE.txt"
    note_file.write_text(
        f"ONNX model: {onnx_path}\n"
        f"Error: {error_msg}\n\n"
        "Placeholder created because ONNX→CoreML conversion is unavailable "
        "under current environment.\n"
        "For production conversion:\n"
        "  1. Use Python 3.10 or 3.11 (not 3.14+)\n"
        "  2. Install onnxruntime or onnxruntime-silicon\n"
        "  3. Ensure coremltools >= 9.0\n"
    )
    
    # Create minimal manifest
    manifest = out / "manifest.json"
    manifest.write_text('{"version": "1.0", "author": "smoke_test_placeholder"}')
    
    print(f"[convert_coreml] Placeholder created: {output_path}")
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert ONNX model to CoreML mlprogram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production conversion (fails loud if conversion unavailable):
  python -m conversion.convert_coreml --backend onnx --in model.onnx --out model.mlpackage

  # Smoke test (creates placeholder on failure):
  python -m conversion.convert_coreml --backend onnx --in model.onnx --out model.mlpackage --allow-placeholder
        """
    )
    ap.add_argument('--backend', choices=['onnx', 'pytorch'], required=True,
                    help='Source framework: pytorch (production) or onnx (smoke/diagnostic)')
    ap.add_argument('--in', '--input', dest='input_path', required=True,
                    help='Input model path (ONNX file for --backend onnx, PyTorch model for --backend pytorch)')
    ap.add_argument('--out', '--output', dest='output_path', required=True,
                    help='Output .mlpackage path')
    ap.add_argument('--contract', dest='contract_path',
                    help='Path to contract.json for input specifications')
    ap.add_argument('--compute-units', default='all',
                    choices=['all', 'cpuandgpu', 'cpuonly'],
                    help='Compute units (default: all)')
    ap.add_argument('--target', default='macOS13',
                    help='Deployment target (default: macOS13)')
    ap.add_argument('--allow-placeholder', action='store_true',
                    help='Create placeholder on failure instead of exiting')
    ap.add_argument('--ane-plan', action='store_true',
                    help='Enable verbose ANE plan logging (dev only)')
    
    args = ap.parse_args()
    
    if args.ane_plan:
        os.environ["MLTOOLS_VERBOSE"] = "1"
    
    if args.backend == 'pytorch':
        # Load PyTorch model (TorchScript or ExportedProgram)
        import torch
        try:
            pytorch_model = torch.jit.load(args.input_path)
            print(f"[convert_coreml] Loaded TorchScript model: {args.input_path}")
        except Exception:
            try:
                # Try loading as ExportedProgram
                import pickle
                with open(args.input_path, 'rb') as f:
                    pytorch_model = pickle.load(f)
                print(f"[convert_coreml] Loaded ExportedProgram model: {args.input_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load PyTorch model from {args.input_path}. "
                    f"Expected TorchScript (.pt) or ExportedProgram. Error: {e}"
                )
        
        # Try to find contract.json automatically if not provided
        contract_path = args.contract_path
        if not contract_path:
            # Look for contract.json in the same directory as the model
            model_dir = Path(args.input_path).parent
            potential_contract = model_dir / "contract.json"
            if potential_contract.exists():
                contract_path = str(potential_contract)
                print(f"[convert_coreml] Found contract.json: {contract_path}")
        
        convert_pytorch_to_coreml(
            pytorch_model=pytorch_model,
            output_path=args.output_path,
            compute_units=args.compute_units,
            target=args.target,
            allow_placeholder=args.allow_placeholder,
            contract_path=contract_path,
        )
    else:  # onnx backend
        convert_onnx_to_coreml(
        onnx_path=args.input_path,
        output_path=args.output_path,
        compute_units=args.compute_units,
        target=args.target,
        allow_placeholder=args.allow_placeholder,
    )


if __name__ == '__main__':
    main()
