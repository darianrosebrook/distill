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
    """
    Remove Cast nodes that cast to the same type (redundant casts).

    Uses ONNX shape inference to determine input types and compare with target types.
    """
    # Run shape inference to get type information
    try:
        model_with_shapes = shape_inference.infer_shapes(model)
    except Exception as e:
        # If shape inference fails, skip redundant cast removal
        print(f"[onnx_surgery] WARN: Shape inference failed, skipping redundant cast removal: {e}")
        return model

    # Build a map of value names to their types
    value_type_map = {}
    for value_info in model_with_shapes.graph.value_info:
        if value_info.type.tensor_type.elem_type:
            value_type_map[value_info.name] = value_info.type.tensor_type.elem_type

    # Also check inputs and outputs
    for input_info in model_with_shapes.graph.input:
        if input_info.type.tensor_type.elem_type:
            value_type_map[input_info.name] = input_info.type.tensor_type.elem_type
    for output_info in model_with_shapes.graph.output:
        if output_info.type.tensor_type.elem_type:
            value_type_map[output_info.name] = output_info.type.tensor_type.elem_type

    # Remove redundant Cast nodes and update graph references
    keep_nodes = []
    removed_count = 0

    # Build a map of redundant Cast outputs to their inputs
    # This maps: cast_output_name -> cast_input_name
    redundant_cast_map = {}

    # First pass: identify redundant Cast nodes
    for n in model.graph.node:
        if n.op_type == "Cast":
            # Get target type from Cast attribute
            to_attr = [a for a in n.attribute if a.name == "to"]
            if to_attr and len(n.input) > 0 and len(n.output) > 0:
                target_type = to_attr[0].i
                input_name = n.input[0]
                output_name = n.output[0]

                # Get input type from shape inference
                input_type = value_type_map.get(input_name)

                if input_type is not None and input_type == target_type:
                    # Redundant cast: input type already equals target type
                    # Map output to input for graph rewriting
                    redundant_cast_map[output_name] = input_name
                    removed_count += 1
                    print(
                        f"[onnx_surgery] Found redundant Cast: {input_name} ({input_type}) -> {target_type}"
                    )
                    # Skip this node (don't add to keep_nodes)
                    continue
                else:
                    # Not redundant, keep the Cast
                    keep_nodes.append(n)
            else:
                # Can't determine, keep the Cast
                keep_nodes.append(n)
        else:
            # Not a Cast node, keep it
            keep_nodes.append(n)

    # Second pass: rewrite graph to replace Cast outputs with Cast inputs
    if redundant_cast_map:
        # Update all node inputs that reference redundant Cast outputs
        for node in keep_nodes:
            for i, input_name in enumerate(node.input):
                if input_name in redundant_cast_map:
                    # Replace reference to Cast output with Cast input
                    node.input[i] = redundant_cast_map[input_name]

        # Update graph outputs if they reference redundant Cast outputs
        for output_info in model.graph.output:
            if output_info.name in redundant_cast_map:
                # Replace output reference
                output_info.name = redundant_cast_map[output_info.name]

        # Update value_info if any reference redundant Cast outputs
        for value_info in model.graph.value_info:
            if value_info.name in redundant_cast_map:
                # Remove value_info for redundant Cast output
                # (it will be replaced by the input's value_info if it exists)
                pass  # We'll remove it from the list

        # Remove value_info entries for redundant Cast outputs
        model.graph.value_info[:] = [
            vi for vi in model.graph.value_info if vi.name not in redundant_cast_map
        ]

    if removed_count > 0:
        print(
            f"[onnx_surgery] Removed {removed_count} redundant Cast nodes and updated graph references"
        )

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
                new = numpy_helper.from_array(arr.astype("int32"), init.name)
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
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--simplify", action="store_true")
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


if __name__ == "__main__":
    sys.exit(main() if len(sys.argv) > 1 else run(sys.argv[1], sys.argv[2]))
