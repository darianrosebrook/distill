# coreml/ane_checks.py
# Note: CoreML doesn't expose explicit per-op ANE flags publicly for mlprogram.
# This utility prints operator histogram to help spot exotic ops that may block ANE.
# @author: @darianrosebrook

import coremltools as ct
from collections import Counter


def op_histogram(model_path: str):
    mlmodel = ct.models.MLModel(model_path)
    spec = mlmodel.get_spec()
    # mlProgram: ops are inside functions; we list op types by name string.
    ops = []
    if spec.WhichOneof("Type") == "mlProgram":
        for f in spec.mlProgram.functions.values():
            for b in f.block.operations:
                ops.append(b.type)
    else:
        # Fallback for neuralNetwork or other types
        layer_names = [l.WhichOneof("layer") for l in spec.neuralNetwork.layers]
        ops.extend(filter(None, layer_names))
    hist = Counter(ops)
    for k, v in hist.most_common():
        print(f"{k}: {v}")
    return hist
