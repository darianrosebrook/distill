"""
Builds a tiny ONNX graph to exercise export→convert→probe without heavy weights.

Graph: input_ids[1,T] (int32) -> Gather(emb[V,D]) -> MatMul(DxD) -> Add bias -> reshape back -> logits[1,T,D]

Usage:
  python -m conversion.make_toy_onnx --seq 128 --vocab 256 --dmodel 64 --out onnx/toy.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--dmodel", type=int, default=64)
    ap.add_argument("--out", type=str, default="onnx/toy.onnx")
    args = ap.parse_args()

    T, V, D = args.seq, args.vocab, args.dmodel

    # Inputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, T])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, [1, T]
    )  # unused, for interface parity

    # Initializers
    emb_w = numpy_helper.from_array(np.random.randn(V, D).astype(np.float32) * 0.02, name="emb_w")
    w1 = numpy_helper.from_array(np.random.randn(D, D).astype(np.float32) * 0.02, name="w1")
    b1 = numpy_helper.from_array(np.zeros((D,), dtype=np.float32), name="b1")

    # Nodes
    nodes = []
    nodes.append(helper.make_node("Gather", ["emb_w", "input_ids"], ["emb"], axis=0))  # [1,T,D]
    nodes.append(
        helper.make_node(
            "Reshape",
            [
                "emb",
            ],
            ["emb2"],
            [],
            domain="",
        )
    )
    # Reshape to [T,D]
    shape_td = numpy_helper.from_array(np.array([T, D], dtype=np.int64), name="shape_td")
    nodes[-1].input.extend(["shape_td"])
    nodes.append(helper.make_node("MatMul", ["emb2", "w1"], ["mm"]))  # [T,D]
    nodes.append(helper.make_node("Add", ["mm", "b1"], ["mm_bias"]))
    # Reshape back to [1,T,D]
    shape_1td = numpy_helper.from_array(np.array([1, T, D], dtype=np.int64), name="shape_1td")
    nodes.append(helper.make_node("Reshape", ["mm_bias", "shape_1td"], ["logits"]))

    # Output
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, T, D])

    graph = helper.make_graph(
        nodes,
        "toy_graph",
        [input_ids, attention_mask],
        [logits],
        initializer=[emb_w, w1, b1, shape_td, shape_1td],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"[make_toy_onnx] wrote {args.out} (T={T}, V={V}, D={D})")


if __name__ == "__main__":
    main()
