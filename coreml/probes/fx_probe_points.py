# coreml/probes/fx_probe_points.py
# PyTorch-side probes: capture tensors for parity checks.
# @author: @darianrosebrook

import os
import numpy as np


class Probe:
    def __init__(self):
        self.data = {}
        self.hooks = []

    def add(self, module, name: str, key: str):
        def hook(_m, _inp, out):
            t = out.detach().float().cpu().numpy()
            self.data[key] = t

        self.hooks.append(module.register_forward_hook(hook))
        return self

    def dump(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.data)

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.data.clear()


def attach(model, probe: Probe):
    # Example taps: embed, first block ln1, first attn logits approximation via QK^T
    probe.add(model.embed, "embed", "embed")
    # add a couple of intermediate points
    blk0 = model.blocks[0]
    probe.add(blk0.norm1, "ln1", "ln1")
    probe.add(blk0.attn.wo, "attn_out_proj", "attn_out")
    return probe
