# arbiter/judge_training/convert_coreml.py
# Judge model CoreML conversion
# @author: @darianrosebrook

import os
import typer
import coremltools as ct

app = typer.Typer()


@app.command()
def main(onnx_path: str, out_path: str = "arbiter/judge_training/artifacts/coreml/judge.mlpackage", compute_units: str = "ALL"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_deployment_target=ct.target.iOS16,  # MLProgram
    )
    mlmodel = ct.models.MLModel(model)
    mlmodel.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    app()

