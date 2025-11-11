# conversion/judge_export_coreml.py
# Judge-specific CoreML export with INT8 weight path
# @author: @darianrosebrook

import typer
import coremltools as ct

app = typer.Typer()


@app.command()
def main(onnx_path: str = typer.Argument(...),
         output_path: str = typer.Option("coreml/judge.mlpackage", "--output")):
    """Convert Judge ONNX model to CoreML with INT8 quantization.
    
    Args:
        onnx_path: Path to Judge ONNX model
        output_path: Output path for CoreML model
    """
    # PLACEHOLDER: Load ONNX, apply INT8 quantization, convert to CoreML
    # Judge should use INT8 weights + FP16 activations
    print(f"Converting judge model: {onnx_path} -> {output_path}")
    print("INT8 weight quantization + FP16 activations (skeleton implementation)")


if __name__ == "__main__":
    app()

