"""Isolate GGUF row-count dispatch using identical inputs and Q8_0 weights.

Synthetic linear diagnostic, not an end-to-end quality or performance test.
It neither changes the quantization plugin nor forces a serving kernel path.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(8128)
    report = {"scope": "synthetic Q8_0 linear row-count dispatch diagnostic", "cases": []}
    for outputs, inputs in ((5120, 5120), (17408, 5120), (14336, 5120)):
        weights = rng.standard_normal((outputs, inputs), dtype=np.float32) / np.sqrt(inputs)
        packed = gguf.quants.quantize(weights, gguf.GGMLQuantizationType.Q8_0)
        w = torch.from_numpy(packed).cuda()
        x = torch.from_numpy(rng.standard_normal((1, inputs), dtype=np.float32)).cuda().bfloat16()
        baseline = _fused_mul_mat_gguf(x, w, int(gguf.GGMLQuantizationType.Q8_0))[0].float()
        for rows in (2, 4, 8, 32):
            batch = x.expand(rows, -1).contiguous()
            actual = _fused_mul_mat_gguf(batch, w, int(gguf.GGMLQuantizationType.Q8_0))[0].float()
            diff = actual - baseline
            report["cases"].append({"outputs": outputs, "inputs": inputs, "rows": rows,
                "max_abs": diff.abs().max().item(), "rms": diff.square().mean().sqrt().item(),
                "different_fraction": (actual != baseline).float().mean().item()})
        del w, x, baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
