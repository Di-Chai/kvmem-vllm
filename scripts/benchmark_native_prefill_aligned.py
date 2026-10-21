"""Fixed-operand, full three-layer FA2+statistics gate for aligned16 metadata.

This does not measure model throughput. The old ABI and both runtime branches
of the new ABI use the same Q/K arena, plans, sampled rows and full normalization.
The generic diagnostic changes only the host qualification predicate, so it
still pays the new device clear and flag upload. CUDA profiler kernel totals are
collected separately from unprofiled complete-component wall/event intervals.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime
from vllm_kvmem.action_download_pool import ActionDownloadPool
import vllm_kvmem.native_attention as native


@contextmanager
def force_generic(enabled):
    original = native.aligned_piece_layout
    if enabled:
        native.aligned_piece_layout = lambda layout, length: False
    try:
        yield
    finally:
        native.aligned_piece_layout = original


def setup(length, output):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(7231)
    page, heads, kh, dim, qlen = 16, 24, 4, 256, 2048
    pages = math.ceil(length / page)
    cache = (torch.randn(pages, kh, page, 2 * dim, device='cuda') * .3).to(torch.float8_e4m3fn)
    k, v = cache.transpose(1, 2).split(dim, -1)
    table = torch.randperm(pages, device='cuda', dtype=torch.int32).reshape(1, pages)
    q = torch.randn(qlen, heads, dim, dtype=torch.bfloat16, device='cuda')
    positions = torch.arange(length - qlen, length, device='cuda')
    workspace = torch.zeros(128 << 20, device='cuda', dtype=torch.uint8)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD', backend='fa2')
    wrapper.plan(torch.tensor([0, qlen], device='cuda', dtype=torch.int32),
        torch.tensor([0, pages], device='cuda', dtype=torch.int32), table.flatten(),
        torch.tensor([(length - 1) % page + 1], device='cuda', dtype=torch.int32),
        heads, kh, dim, page, causal=True, q_data_type=q.dtype, kv_data_type=k.dtype)
    metadata = NS(prefill=NS(wrapper=wrapper), q_data_type_prefill=q.dtype, num_decode_tokens=0)
    model_output = torch.empty_like(q)
    scale = torch.tensor(.8, device='cuda'); qscale = torch.ones((), device='cuda')
    trig = torch.cat((torch.ones(length, dim // 4, device='cuda'),
                      torch.zeros(length, dim // 4, device='cuda')), -1)
    input_batch = NS(num_reqs=1, req_ids=['0'], is_prefilling_np=[True],
        query_start_loc_np=[0, qlen], num_computed_tokens_np=[length - qlen],
        seq_lens_cpu_upper_bound=[length], input_ids=torch.full((qlen,), 20, device='cuda', dtype=torch.int32),
        positions=positions)
    records = [(i, 1, start, min(32, length - start), start)
               for i, start in enumerate(range(0, length, 32))]
    captured = [NS(query=q, positions=positions) for _ in range(3)]
    arms = {}
    for enabled in (False, True):
        class Impl:
            scale = dim ** -.5
            def forward(self, layer, query, key, value, kv_cache, meta, out):
                return wrapper.run(query, (k, v), k_scale=.8, v_scale=.6, out=out)
        layers = [NS(num_heads=heads, num_kv_heads=kh, head_dim=dim,
            q_norm=NS(weight=q), rotary_emb=NS(cos_sin_cache=trig),
            attn=NS(layer_name=f'model.layers.{n}.attn', impl=Impl(),
                    kv_cache=cache, _k_scale=scale, _q_scale=qscale)) for n in (3, 31, 63)]
        names = [layer.attn.layer_name for layer in layers]
        # Match serving capacity, rather than shrinking scratch to this request.
        runner = NS(model=NS(named_modules=lambda layers=layers, names=names: zip(names, layers)),
            _kvmem_layers=names, max_num_tokens=16424, max_num_reqs=8,
            req_states=NS(req_id_to_index={'0': 0}),
            block_tables=NS(block_tables=[NS(gpu=table)]))
        options = NS(action_native_scores='reuse', action_layers='all', action_native_max_rows=256,
            action_native_max_bytes=2 << 30, action_native_prefill_aligned_pieces=enabled,
            action_download_pool=True,
            action_scope='non-thinking', action_interval=16, action_decay=.5,
            prefill_attention=True, prefill_attention_interval=1024, prefill_attention_stride=32,
            prefill_attention_half_life=8192,
            memory=NS(select_budget=32768, gen_budget=16384, block_tokens=32))
        bank = native.NativeAttentionStats(runner, options, output.parent / 'build')
        if bank.disabled:
            raise RuntimeError(bank.disabled)
        assert (bank.context, bank.max_segments, bank.width, bank.max_rows) == (65576, 2202, 6301, 256)
        authority = NS(action_history=None, block_graph=None,
                       memory=NS(_store=NS(working_records=lambda: records)))
        action = ActionRuntime.__new__(ActionRuntime)
        action.runner, action.options = runner, options
        action.runtime = NS(config=options.memory, output=output.parent,
            ensure_authority=lambda request, authority=authority: authority,
            drain_requests=lambda *a, **kw: True)
        action.native, action.native_plans, action.metadata = bank, None, None
        action.layers, action.groups = list(enumerate(layers)), dict.fromkeys(names, 0)
        action.q_dtypes = action.prefill_q_dtypes = dict.fromkeys(names, torch.bfloat16)
        action.ids = [10, 11, 12, 13, 14]
        action.profile = action.metadata_upload = None
        action.download_pool = ActionDownloadPool()
        action.capture_seconds = action.candidate_samples = action.packets = 0
        action.metadata_early_batches = action.metadata_fallback_batches = action.metadata_wait_seconds = 0
        arms[enabled] = (action, bank, layers, authority)

    def run(mode):
        action, bank, layers, authority = arms[mode != 'old']
        authority.action_history = ActionHistory(action.options, action.ids)
        authority.action_history.begin('0', [14])
        with force_generic(mode == 'new_generic'):
            action.prepare(input_batch, positions)
        for layer in layers:
            layer.attn.impl.forward(layer.attn, q, None, None, None, metadata, model_output)
        packets = action.capture(captured, input_batch)
        for packet in packets.values():
            if packet.event is not None:
                packet.event.synchronize()
        if any(key.startswith('fallback:') for key in bank.counts):
            raise RuntimeError(f'Unexpected scoring fallback: {bank.counts}')
        packet = packets['0']
        return tuple(packet.history), packet.masses.clone()

    return run, arms, dict(active_tokens_including_query=length, query_tokens=qlen,
        sampled_rows=64, layers=3, q_heads=heads, kv_heads=kh, head_dim=dim, kv_dtype=str(k.dtype),
        page_tokens=page, context_capacity=65576, max_segments=2202, width=6301, max_rows=256,
        max_num_reqs=8, q_dtype=str(q.dtype), q_scale=1., k_scale=.8,
        pinned_download_pool_enabled=True,
        same_q_k_arena=True, full_causal_normalization=True,
        acceptance_archive_graph_publication_included=False)


def kernel_profile(run, mode):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        run(mode)
        torch.cuda.synchronize()
    kernels = [event for event in profile.events()
               if str(event.device_type).endswith('CUDA')]
    attention = [event for event in kernels if 'BatchPrefillWithPagedKVCacheKernel' in event.name]
    if len(attention) != 3:
        raise RuntimeError(f'Expected three actual FA2 kernels, found {len(attention)}')
    return dict(fa2_kernel_ms=sum(event.device_time_total for event in attention) / 1000,
        all_gpu_kernels_ms=sum(event.device_time_total for event in kernels) / 1000,
        fa2_kernels=[dict(name=event.name, duration_us=event.device_time_total) for event in attention])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[16384, 32768, 49152])
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--profile-repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a fresh output path')
    modes = ['old', 'new_generic', 'new_fast']
    evidence = dict(scope='Three FA2 attention layers plus complete scoring preparation, normalization, layer averaging and D2H; not model throughput.',
        old_abi_uses_current_source_overlay=True, frozen_historical_binary_not_compared=True,
        generic_diagnostic='Force only host alignment predicate false; identical flag clear/upload and new ABI.',
        gpu=torch.cuda.get_device_name(), results=[])
    root = Path(__file__).resolve().parents[1]
    evidence['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), *(root / 'src/vllm_kvmem' / name for name in
            ['native_attention.py', 'native_attention_build.py', 'native_stats.cuh', 'native_prefill_stats.cuh'])]}
    for length in args.lengths:
        run, arms, result = setup(length, args.output)
        expected = run('old')
        for mode in modes[1:]:
            actual = run(mode)
            assert actual[0] == expected[0] and torch.equal(actual[1], expected[1]), mode
        for _ in range(3):
            for mode in modes:
                run(mode)
        samples = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for repetition in range(args.repeats):
            order = modes[repetition % 3:] + modes[:repetition % 3]
            if repetition % 2:
                order = order[::-1]
            for mode in order:
                torch.cuda.synchronize()
                began = time.perf_counter(); start.record()
                run(mode); end.record(); end.synchronize()
                samples.append(dict(repetition=repetition, mode=mode,
                    wall_ms=1000 * (time.perf_counter() - began),
                    cuda_interval_ms=start.elapsed_time(end)))
        result['samples'] = samples
        result['summary'] = {mode: {metric: statistics.mean(sample[metric] for sample in samples if sample['mode'] == mode)
            for metric in ('wall_ms', 'cuda_interval_ms')} for mode in modes}
        result['kernel_profiles'] = {mode: [kernel_profile(run, mode) for _ in range(args.profile_repeats)] for mode in modes}
        result['kernel_summary_ms'] = {mode: statistics.mean(item['fa2_kernel_ms'] for item in result['kernel_profiles'][mode]) for mode in modes}
        result['native_state'] = {str(enabled): dict(counts=dict(bank.counts), aligned_counts=dict(bank.aligned_prefill_counts),
            allocated_bytes=bank.allocated_bytes, source_fingerprint=bank.source_fingerprint)
            for enabled, (_, bank, _, _) in arms.items()}
        result['bitwise_scores_equal'] = True
        evidence['results'].append(result)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(dict(length=length, full_component=result['summary'], fa2_kernel_ms=result['kernel_summary_ms'])), flush=True)
        del run, arms
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
