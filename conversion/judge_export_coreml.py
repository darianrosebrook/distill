# conversion/judge_export_coreml.py
# Judge-specific CoreML export with INT8 weight path
# @author: @darianrosebrook

import sys
from pathlib import Path
import typer

# Import conversion utilities from convert_coreml
try:
    from conversion.convert_coreml import convert_onnx_to_coreml
except ImportError:
    # Fallback if import fails
    convert_onnx_to_coreml = None

app = typer.Typer()


@app.command()
def main(
    onnx_path: str = typer.Argument(...),
    output_path: str = typer.Option("coreml/judge.mlpackage", "--output"),
    compute_units: str = typer.Option("all", "--compute-units"),
    target: str = typer.Option("macOS13", "--target"),
    allow_placeholder: bool = typer.Option(False, "--allow-placeholder"),
):
    """Convert Judge ONNX model to CoreML with INT8 quantization.

    IMPORTANT: CoreMLTools does not natively support ONNX→CoreML conversion.
    This function attempts conversion but may create a placeholder if conversion fails.

    Production workflow:
    1. Convert ONNX→PyTorch first (using export_pytorch.py or similar)
    2. Use PyTorch→CoreML conversion (--backend pytorch) for production

    INT8 quantization should be applied at the ONNX level before conversion,
    or use PyTorch quantization APIs.

    Args:
        onnx_path: Path to Judge ONNX model
        output_path: Output path for CoreML model
        compute_units: Compute units ("all", "cpuandgpu", "cpuonly")
        target: Deployment target (e.g., "macOS13", "macOS14")
        allow_placeholder: If True, create placeholder on failure instead of raising
    """
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        print(f"Error: ONNX model not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[judge_export_coreml] Converting judge model: {onnx_path} -> {output_path}")
    print(
        f"[judge_export_coreml] Target: {target}, Compute units: {compute_units}")
    print("[judge_export_coreml] Note: Judge should use INT8 weights + FP16 activations")
    print(
        "[judge_export_coreml] INT8 quantization should be applied at ONNX level before conversion"
    )

    # Use existing conversion utility
    if convert_onnx_to_coreml is None:
        print(
            "[judge_export_coreml] Error: Could not import conversion utilities", file=sys.stderr)
        print(
            "[judge_export_coreml] Ensure conversion.convert_coreml is available", file=sys.stderr
        )
        sys.exit(1)

    try:
        result_path = convert_onnx_to_coreml(
            onnx_path=str(onnx_file),
            output_path=output_path,
            compute_units=compute_units,
            target=target,
            allow_placeholder=allow_placeholder,
        )

        if result_path:
            print(
                f"[judge_export_coreml] Successfully converted judge model: {result_path}")
        else:
            print(
                "[judge_export_coreml] Placeholder created (conversion not supported)")
            sys.exit(0)

    except Exception as e:
        print(
            f"[judge_export_coreml] Error during conversion: {e}", file=sys.stderr)
        if not allow_placeholder:
            sys.exit(1)
        else:
            print("[judge_export_coreml] Creating placeholder due to error")
            sys.exit(0)


if __name__ == "__main__":
    app()
