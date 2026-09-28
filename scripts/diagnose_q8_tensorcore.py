"""Bounded isolated Q8 kernel diagnostic; no serving/performance acceptance."""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import triton
from vllm_gguf_plugin import ops
from vllm_kvmem.q8_tensorcore import q8_tensorcore_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-k', type=int, default=4)
    parser.add_argument('--block-n', type=int, default=64)
    parser.add_argument('--block-groups', type=int, default=4)
    parser.add_argument('--gguf', type=Path, help='Use real layer-0 Q8 weights instead of synthetic weights')
    args = parser.parse_args()
    rng = np.random.default_rng(421)
    report = {'scope': 'isolated synthetic Q8_0 kernel diagnostic; no model qualification',
              'device': torch.cuda.get_device_name(), 'options': vars(args) | {'output': str(args.output), 'gguf': str(args.gguf) if args.gguf else None}, 'cases': []}
    def matrices():
        if args.gguf:
            reader = gguf.GGUFReader(str(args.gguf))
            for tensor in reader.tensors:
                if tensor.name in ('blk.0.attn_qkv.weight', 'blk.0.ffn_gate.weight', 'blk.0.ffn_down.weight'):
                    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0
                    yield tensor.name, tensor.data.copy(), int(tensor.shape[0]), int(tensor.shape[1])
        else:
            for n, k in ((5120, 5120), (17408, 5120), (5120, 17408)):
                packed = gguf.quants.quantize(rng.normal(0, 1 / np.sqrt(k), (n, k)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
                yield 'synthetic', packed, k, n
    if args.gguf:
        report['scope'] = 'isolated real Q8_0 weights with synthetic activations; no model qualification'
    for name, packed, k, n in matrices():
        w = torch.from_numpy(packed).cuda()
        for m in (8, 32, 1024):
            x = torch.from_numpy(rng.standard_normal((m, k), dtype=np.float32)).cuda().bfloat16()
            native = lambda: ops.ggml_mul_mat_a8(w, x, 8, n)
            candidate = lambda: q8_tensorcore_linear(x, w, split_k=args.split_k, block_n=args.block_n, block_groups=args.block_groups)
            a, b = native(), candidate()
            diff = (a.float() - b.float())
            case = {'weight': name, 'm': m, 'n': n, 'k': k,
                    'native_ms': triton.testing.do_bench(native, warmup=100, rep=200),
                    'candidate_ms': triton.testing.do_bench(candidate, warmup=100, rep=200),
                    'max_abs': diff.abs().max().item(),
                    'relative_rms': (diff.square().mean() / a.float().square().mean()).sqrt().item(),
                    'different_fraction': (a != b).float().mean().item()}
            case['speed_ratio'] = case['native_ms'] / case['candidate_ms']
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
        del w
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
