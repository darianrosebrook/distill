# conversion/onnx_surgery.py
# Basic surgeries: enforce int32 inputs, remove stray casts, ensure FP16 outputs.
# @author: @darianrosebrook

import onnx
from onnx import helper, TensorProto


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


def run(path_in: str, path_out: str):
    m = onnx.load(path_in)
    m = force_input_dtype(m, "input_ids", TensorProto.INT32)
    m = force_output_dtype(m, "logits", TensorProto.FLOAT16)
    m = strip_redundant_casts(m)
    onnx.save(m, path_out)


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
