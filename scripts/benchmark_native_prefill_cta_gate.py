"""Fixed-operand three-layer FA2 + complete scoring with/without the CTA gate.

This is a component measurement, not model throughput. Both arms share Q/K,
page tables, sampled rows, causal normalization and optionally aligned16 mode.
Unprofiled wall/CUDA intervals include metadata preparation, three attentions,
normalization, averaging and D2H. Profiler kernel times are separate samples.
"""
import argparse
import gc
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


def setup(length, output, *, qlen=2048, aligned=False, max_rows=512):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(7231)
    page, heads, kh, dim = 16, 24, 4, 256
    if not 0 < qlen <= length or qlen % 32:
        raise ValueError('Query length must be a positive multiple of 32 within the context')
    if max_rows < qlen//32:
        raise ValueError('Fixed statistics capacity must cover every sampled query row')
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
        options = NS(action_native_scores='reuse', action_layers='all', action_native_max_rows=max_rows,
            action_native_max_bytes=2 << 30, action_native_prefill_aligned_pieces=aligned,
            action_native_prefill_cta_gate=enabled,
            action_download_pool=True,
            action_scope='non-thinking', action_interval=16, action_decay=.5,
            prefill_attention=True, prefill_attention_interval=1024, prefill_attention_stride=32,
            prefill_attention_half_life=8192,
            memory=NS(select_budget=32768, gen_budget=16384, block_tokens=32))
        bank = native.NativeAttentionStats(runner, options, output.parent / 'build')
        if bank.disabled:
            raise RuntimeError(bank.disabled)
        assert (bank.context, bank.max_segments, bank.width, bank.max_rows) == (65576, 2202, 6301, max_rows)
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
        # Keep the pooled lease alive until the caller consumes this result.
        # Correctness clones are outside the timing interval; timed calls only
        # wait for the same complete D2H event used by normal publication.
        return tuple(packet.history), packet, model_output

    return run, arms, dict(active_tokens_including_query=length, query_tokens=qlen,
        sampled_rows=qlen//32, layers=3, q_heads=heads, kv_heads=kh, head_dim=dim, kv_dtype=str(k.dtype),
        page_tokens=page, context_capacity=65576, max_segments=2202, width=6301, max_rows=max_rows,
        max_num_reqs=8, q_dtype=str(q.dtype), q_scale=1., k_scale=.8,
        pinned_download_pool_enabled=True,
        aligned_pieces_enabled=aligned,
        actual_cta_tile_q=int(wrapper._plan_info[3]),
        same_q_k_arena=True, full_causal_normalization=True,
        three_layers_share_operands=True,
        cache_locality_note='The three sampled layers share Q/K/V here; serving layers have separate caches and intervening model work.',
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
    parser.add_argument('--query-tokens', nargs='+', type=int, default=[2048,8192,14336])
    parser.add_argument('--aligned-pieces', action='store_true')
    parser.add_argument('--max-rows', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--profile-repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats<1 or args.profile_repeats<1 or args.max_rows<1:
        raise ValueError('Repetition counts and fixed row capacity must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a fresh output path')
    modes = ['old', 'gate']
    evidence = dict(scope='Three FA2 attention layers plus complete scoring preparation, normalization, layer averaging and D2H; not model throughput.',
        old_abi_uses_current_source_overlay=True, frozen_historical_binary_not_compared=True,
        comparison='Only action_native_prefill_cta_gate changes; same stats ABI and bounded buffers.',
        geometry_teardown_gc_outside_timing=True,
        gpu=torch.cuda.get_device_name(), results=[])
    root = Path(__file__).resolve().parents[1]
    evidence['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), *(root / 'src/vllm_kvmem' / name for name in
            ['native_attention.py', 'native_attention_build.py', 'native_stats.cuh', 'native_prefill_stats.cuh'])]}
    for length,qlen in ((n,q) for q in args.query_tokens for n in args.lengths if q<=n):
        run, arms, result = setup(length, args.output,qlen=qlen,aligned=args.aligned_pieces,max_rows=args.max_rows)
        assert arms[False][1].allocated_bytes==arms[True][1].allocated_bytes
        history,packet,output=run('old')
        expected_scores=packet.masses.clone();expected_output=output.clone()
        del packet
        actual_history,packet,output=run('gate')
        assert actual_history==history
        assert torch.equal(packet.masses.contiguous().view(torch.uint8),expected_scores.contiguous().view(torch.uint8))
        assert torch.equal(output.view(torch.uint8),expected_output.view(torch.uint8))
        del packet,expected_scores,expected_output,output
        for _ in range(3):
            for mode in modes:
                run(mode)
        samples = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for repetition in range(args.repeats):
            order = modes if repetition%2==0 else modes[::-1]
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
            cta_gate_counts=dict(bank.cta_gate_counts),allocated_bytes=bank.allocated_bytes,
            source_fingerprint=bank.source_fingerprint,prefill_source_fingerprint=bank.prefill_source_fingerprint)
            for enabled, (_, bank, _, _) in arms.items()}
        assert arms[False][1].cta_gate_counts.get('prefill_dispatch_calls',0)==0
        assert arms[True][1].cta_gate_counts.get('prefill_dispatch_calls',0)>0
        assert arms[False][1].prefill_source_fingerprint!=arms[True][1].prefill_source_fingerprint
        result['bitwise_scores_equal'] = True
        result['bitwise_model_outputs_equal'] = True
        evidence['results'].append(result)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(dict(length=length,query_tokens=qlen,full_component=result['summary'],fa2_kernel_ms=result['kernel_summary_ms'])), flush=True)
        del run, arms
        # Fixture runner/impl/bank closures form an owner-lifetime cycle. Reap
        # it only after all measurements, never through a timed publication.
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
