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
import warnings
import argparse
import os
import sys
from pathlib import Path

# Suppress ANE remote proxy warnings for cleaner output
os.environ.setdefault("MLTOOLS_VERBOSE", "0")
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
        except Exception:
            pass
    else:
        target_version = ct.target.macOS13

    print(
        f"[convert_coreml] Converting PyTorch→CoreML (target={target}, compute_units={compute_units})")

    try:
        # For TorchScript, we need to provide input specification
        # Try to read from contract.json first, fallback to defaults
        import torch
        import numpy as np

        if isinstance(pytorch_model, torch.jit.ScriptModule):
            # Try to load contract.json for input specs
            inputs = None
            if contract_path and Path(contract_path).exists():
                try:
                    contract = load_contract(contract_path)

                    # Validate contract structure
                    if not isinstance(contract, dict):
                        raise ValueError(
                            f"Contract must be a dictionary, got {type(contract)}")

                    contract_inputs = contract.get("inputs", [])
                    if not contract_inputs:
                        raise ValueError(
                            "Contract must contain 'inputs' list with at least one input specification")

                    input_spec = contract_inputs[0]
                    if not isinstance(input_spec, dict):
                        raise ValueError(
                            f"Input specification must be a dictionary, got {type(input_spec)}")

                    input_name = input_spec.get("name", "input_ids")
                    input_dtype = input_spec.get("dtype", "int32")

                    # Convert shape: ["B", "T"] -> (1, T) for example
                    # Use first enumerated_T if available, else default to 128
                    enumerated_T = contract.get("enumerated_T", [128])
                    if not isinstance(enumerated_T, list) or len(enumerated_T) == 0:
                        enumerated_T = [128]
                        print(
                            "[convert_coreml] WARN: Invalid enumerated_T in contract, using default [128]")

                    seq_len = enumerated_T[0] if enumerated_T else 128
                    if not isinstance(seq_len, int) or seq_len <= 0:
                        raise ValueError(
                            f"Invalid sequence length in enumerated_T: {seq_len}")

                    shape = (1, seq_len)

                    # Convert dtype string to numpy dtype
                    dtype_map = {
                        "int32": np.int32,
                        "int64": np.int64,
                        "float32": np.float32,
                        "float16": np.float16
                    }
                    if input_dtype not in dtype_map:
                        print(
                            f"[convert_coreml] WARN: Unknown dtype '{input_dtype}' in contract, using int32")
                        dtype = np.int32
                    else:
                        dtype = dtype_map[input_dtype]

                    inputs = [ct.TensorType(
                        name=input_name, shape=shape, dtype=dtype)]
                    print(
                        f"[convert_coreml] Using contract.json: {input_name} shape={shape} dtype={input_dtype}")
                except Exception as contract_error:
                    # Contract loading/validation failed - fall back to inference
                    print(
                        f"[convert_coreml] WARN: Failed to load/validate contract.json: {contract_error}")
                    print("[convert_coreml] Falling back to input shape inference")
                    inputs = None  # Will trigger inference path

            # Fallback to inference if contract not available or failed
            if inputs is None:
                # Fallback: try to infer input shape from model
                # Run a dummy forward pass to get input shape
                inference_errors = []

                # Try int32 input (most common for language models)
                try:
                    dummy_input = torch.zeros((1, 128), dtype=torch.int32)
                    _ = pytorch_model(dummy_input)
                    # If that worked, use int32 input
                    inputs = [ct.TensorType(
                        name="input_ids", shape=(1, 128), dtype=np.int32)]
                    print(
                        "[convert_coreml] Inferred input shape from model: input_ids shape=(1, 128) dtype=int32")
                except Exception as e:
                    inference_errors.append(f"int32[1,128]: {str(e)}")

                    # Try float32 input (for transformer blocks or embeddings)
                    try:
                        dummy_input = torch.randn(
                            (1, 128, 64), dtype=torch.float32)
                        _ = pytorch_model(dummy_input)
                        inputs = [ct.TensorType(name="input", shape=(
                            1, 128, 64), dtype=np.float32)]
                        print(
                            "[convert_coreml] Inferred input shape from model: input shape=(1, 128, 64) dtype=float32")
                    except Exception as e:
                        inference_errors.append(f"float32[1,128,64]: {str(e)}")

                        # Try other common shapes
                        for shape, dtype in [((1, 512), torch.int32), ((1, 1024), torch.int32), ((1, 2048), torch.int32)]:
                            try:
                                dummy_input = torch.zeros(shape, dtype=dtype)
                                _ = pytorch_model(dummy_input)
                                inputs = [ct.TensorType(
                                    name="input_ids", shape=shape, dtype=np.int32)]
                                print(
                                    f"[convert_coreml] Inferred input shape from model: input_ids shape={shape} dtype=int32")
                                break
                            except Exception:
                                continue

                        # Last resort: use defaults with detailed error message
                        if inputs is None:
                            error_details = "; ".join(inference_errors)
                            error_msg = (
                                f"Could not infer input shape from model. "
                                f"Attempted shapes: {error_details}. "
                                f"Using default shape=(1, 128) dtype=int32. "
                                f"Consider providing --contract with contract.json for accurate input specification."
                            )
                            if allow_placeholder:
                                print(f"[convert_coreml] WARN: {error_msg}")
                            else:
                                print(f"[convert_coreml] ERROR: {error_msg}")
                            inputs = [ct.TensorType(
                                name="input_ids", shape=(1, 128), dtype=np.int32)]
            # Ensure stable output name - CoreML may use generic names like var_###
            # Try to specify output name if possible
            mlmodel = ct.convert(
                pytorch_model,
                inputs=inputs,
                source="pytorch",
                compute_units=compute_unit,
                minimum_deployment_target=target_version,
                convert_to="mlprogram",
            )

            # Rename outputs to match contract expectations
            # This ensures verifiers can reliably find the outputs
            spec = mlmodel.get_spec()
            num_outputs = len(spec.description.output)

            # Check contract for halt head support
            use_halt_head = False
            if contract_path and Path(contract_path).exists():
                try:
                    contract = load_contract(contract_path)
                    use_halt_head = contract.get("use_halt_head", False)
                except Exception:
                    pass  # Fall back to defaults

            # Rename outputs based on count and contract
            if num_outputs == 1:
                # Single output: should be logits
                output = spec.description.output[0]
                if output.name.startswith("var_") or output.name == "":
                    output.name = "logits"
                    print("[convert_coreml] Renamed output to 'logits'")
            elif num_outputs == 2 and use_halt_head:
                # Two outputs: logits and halt_logits
                # CoreML outputs are typically in order: logits, halt_logits
                outputs = spec.description.output
                if len(outputs) >= 1:
                    if outputs[0].name.startswith("var_") or outputs[0].name == "":
                        outputs[0].name = "logits"
                        print("[convert_coreml] Renamed first output to 'logits'")
                if len(outputs) >= 2:
                    if outputs[1].name.startswith("var_") or outputs[1].name == "":
                        outputs[1].name = "halt_logits"
                        print(
                            "[convert_coreml] Renamed second output to 'halt_logits'")
            else:
                # Multiple outputs but not expected - warn
                if num_outputs > 1:
                    print(
                        f"[convert_coreml] WARN: Model has {num_outputs} outputs, expected 1 or 2 (with halt head)")
                    # Still try to rename first output to logits
                    if len(spec.description.output) > 0:
                        output = spec.description.output[0]
                        if output.name.startswith("var_") or output.name == "":
                            output.name = "logits"
                            print(
                                "[convert_coreml] Renamed first output to 'logits'")
        else:
            # For ExportedProgram, inputs are already embedded
            mlmodel = ct.convert(
                pytorch_model,
                source="pytorch",
                compute_units=compute_unit,
                minimum_deployment_target=target_version,
                convert_to="mlprogram",
            )

            # Rename outputs for ExportedProgram path too
            spec = mlmodel.get_spec()
            num_outputs = len(spec.description.output)

            # Check contract for halt head support
            use_halt_head = False
            if contract_path and Path(contract_path).exists():
                try:
                    contract = load_contract(contract_path)
                    use_halt_head = contract.get("use_halt_head", False)
                except Exception:
                    pass

            # Rename outputs based on count and contract
            if num_outputs == 1:
                output = spec.description.output[0]
                if output.name.startswith("var_") or output.name == "":
                    output.name = "logits"
                    print("[convert_coreml] Renamed output to 'logits'")
            elif num_outputs == 2 and use_halt_head:
                outputs = spec.description.output
                if len(outputs) >= 1:
                    if outputs[0].name.startswith("var_") or outputs[0].name == "":
                        outputs[0].name = "logits"
                        print("[convert_coreml] Renamed first output to 'logits'")
                if len(outputs) >= 2:
                    if outputs[1].name.startswith("var_") or outputs[1].name == "":
                        outputs[1].name = "halt_logits"
                        print(
                            "[convert_coreml] Renamed second output to 'halt_logits'")

        print("[convert_coreml] Conversion successful")

        # Save model
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path_obj))
        print(f"[convert_coreml] Saved → {output_path}")
        return output_path

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        # Provide detailed error context
        error_context = []
        if isinstance(pytorch_model, torch.jit.ScriptModule):
            error_context.append("Model type: TorchScript")
        else:
            error_context.append("Model type: ExportedProgram")
        error_context.append(f"Target: {target}")
        error_context.append(f"Compute units: {compute_units}")
        if contract_path:
            error_context.append(f"Contract: {contract_path}")

        full_error_msg = f"{error_type}: {error_msg}"
        if error_context:
            full_error_msg += f" (Context: {', '.join(error_context)})"

        if allow_placeholder:
            print(
                f"[convert_coreml] WARN: Conversion failed: {full_error_msg}")
            print("[convert_coreml] Creating placeholder (SKIP parity)")
            return create_placeholder(output_path, "PyTorch model", full_error_msg)
        else:
            print("[convert_coreml] ERROR: PyTorch→CoreML conversion failed")
            print(f"[convert_coreml] Error: {full_error_msg}")
            print("\n[convert_coreml] REMEDIATION:")
            print("  1. Verify model is valid TorchScript or ExportedProgram")
            print("  2. Check input shape/dtype matches model expectations")
            print("  3. Provide --contract with contract.json for accurate input specs")
            print("  4. Ensure coremltools version supports your model operations")
            print("  5. Check model operations are supported by CoreML")
            raise RuntimeError(
                f"PyTorch→CoreML conversion failed: {full_error_msg}") from e


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
        except Exception:
            pass
    else:
        target_version = ct.target.macOS13

    print(f"[convert_coreml] Loading ONNX model: {onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    print(
        f"[convert_coreml] Converting ONNX→CoreML (target={target}, compute_units={compute_units})")

    try:
        # Try public API: ct.convert with auto-detection
        mlmodel = ct.convert(
            onnx_model,
            compute_units=compute_unit,
            minimum_deployment_target=target_version,
            convert_to="mlprogram",
        )
        print("[convert_coreml] Conversion successful")

        # Rename output to "logits" if it's unnamed or generic (for consistency with PyTorch path)
        # This ensures verifiers can reliably find the output
        spec = mlmodel.get_spec()
        if len(spec.description.output) == 1:
            output = spec.description.output[0]
            old_name = output.name
            if output.name.startswith("var_") or output.name == "" or not output.name:
                output.name = "logits"
                print(
                    f"[convert_coreml] Renamed output '{old_name}' to 'logits'")

    except Exception:
        # ONNX is not a supported production path - always create placeholder
        if allow_placeholder:
            print(
                "[convert_coreml] WARN: ONNX conversion not supported by CoreMLTools")
            print("[convert_coreml] Creating placeholder (SKIP parity)")
            error_msg = (
                "ONNX→CoreML conversion is not a supported production path. "
                "CoreMLTools 9.0 only supports TensorFlow and PyTorch. "
                "For production, use --backend pytorch with a TorchScript or ExportedProgram model."
            )
            return create_placeholder(output_path, onnx_path, error_msg)
        else:
            print("[convert_coreml] ERROR: ONNX→CoreML conversion not supported")
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
    manifest.write_text(
        '{"version": "1.0", "author": "smoke_test_placeholder"}')

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
    ap.add_argument('--seq', nargs='+', type=int,
                    help='Enumerated sequence lengths for toy models (e.g., --seq 64 128 256). '
                         'For toy models, converts the first prefill model matching these shapes.')

    args = ap.parse_args()

    if args.ane_plan:
        os.environ["MLTOOLS_VERBOSE"] = "1"

    if args.backend == 'pytorch':
        # Load PyTorch model (TorchScript or ExportedProgram)
        import torch
        try:
            pytorch_model = torch.jit.load(args.input_path)
            print(
                f"[convert_coreml] Loaded TorchScript model: {args.input_path}")
        except Exception:
            try:
                # Try loading as ExportedProgram
                import pickle
                with open(args.input_path, 'rb') as f:
                    pytorch_model = pickle.load(f)
                print(
                    f"[convert_coreml] Loaded ExportedProgram model: {args.input_path}")
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
