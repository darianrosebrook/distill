# conversion/onnx_surgery.py
"""
Lightweight ONNX surgery for smoke runs.

- Cast stray int64 initializers to int32 when safe
- (Optional) run shape inference
- (Optional) onnxsim if available

Usage:
  python -m conversion.onnx_surgery --inp onnx/in.onnx --out onnx/out.onnx --infer --simplify
"""
import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper, shape_inference

try:
    import onnxsim  # type: ignore
except Exception:  # pragma: no cover
    onnxsim = None


def force_input_dtype(model: onnx.ModelProto, name: str, dtype=TensorProto.INT32):
    for i in model.graph.input:
        if i.name == name:
            i.type.tensor_type.elem_type = dtype
    return model


def force_output_dtype(model: onnx.ModelProto, name: str, dtype=TensorProto.FLOAT16):
    for o in model.graph.output:
        if o.name == name:
            o.type.tensor_type.elem_type = dtype
    return model


def strip_redundant_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    # Very conservative pass: remove Cast nodes that cast X->X same type
    keep_nodes = []
    for n in model.graph.node:
        if n.op_type == "Cast":
            to = [a for a in n.attribute if a.name == "to"]
            if to:
                # If input type already equals target, bypass
                # (Requires shape inference in a real pass)
                pass
        keep_nodes.append(n)
    model.graph.ClearField("node")
    model.graph.node.extend(keep_nodes)
    return model


def cast_int64_initializers(model: onnx.ModelProto) -> int:
    changed = 0
    new_inits = []
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            if arr.min() >= -(2**31) and arr.max() < 2**31:
                new = numpy_helper.from_array(arr.astype('int32'), init.name)
                new_inits.append(new)
                changed += 1
            else:
                new_inits.append(init)
        else:
            new_inits.append(init)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    return changed


def run(path_in: str, path_out: str):
    m = onnx.load(path_in)
    m = force_input_dtype(m, "input_ids", TensorProto.INT32)
    m = force_output_dtype(m, "logits", TensorProto.FLOAT16)
    m = strip_redundant_casts(m)
    onnx.save(m, path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--simplify', action='store_true')
    args = ap.parse_args()

    model = onnx.load(args.inp)
    changed = cast_int64_initializers(model)

    if args.infer:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[onnx_surgery] shape inference skipped: {e}")

    if args.simplify and onnxsim is not None:
        try:
            model, _ = onnxsim.simplify(model)
        except Exception as e:
            print(f"[onnx_surgery] onnxsim simplify skipped: {e}")
    elif args.simplify:
        print("[onnx_surgery] onnxsim not installed; skipping simplify")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"[onnx_surgery] saved → {args.out} (int64→int32 changed: {changed})")


if __name__ == '__main__':
    sys.exit(main() if len(sys.argv) > 1 else run(sys.argv[1], sys.argv[2]))
