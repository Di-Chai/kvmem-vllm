"""Three-arm FA2 floor: genuine native attention, generic stats, static stats.

All arms use identical operands, original FlashInfer plan, scale and layout.
The native arm has no KVMem preparation/reduction/D2H; the two statistics arms
include those required costs. It is therefore a baseline attention cost, not a
functionally equivalent KVMem implementation or a model-throughput comparison.
Native-row LSE output and correctness clones run only outside formal timing.
"""
import argparse
import gc
import hashlib
import json
import itertools
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime
from vllm_kvmem.action_download_pool import ActionDownloadPool
import vllm_kvmem.native_attention as native


def setup(length, output, *, qlen=2048, max_rows=512):
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
    metadata = NS(prefill=NS(wrapper=wrapper), q_data_type_prefill=q.dtype, num_decode_tokens=0,
                  num_prefills=1,num_prefill_tokens=qlen)
    model_output = torch.empty_like(q)
    native_lse = torch.empty((qlen, heads), dtype=torch.float32, device='cuda')
    native_module = wrapper._cached_module
    qualification_lse = False
    native_dispatches = 0

    def original_attention(query, out):
        # This callable enters the original planned FlashInfer module directly.
        # The stats arms temporarily wrap this same module and restore it.
        return wrapper.run(query, (k, v), k_scale=.8, v_scale=.6, out=out,
            lse=native_lse if qualification_lse else None,
            return_lse=qualification_lse)
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
                return original_attention(query, out)
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
            action_native_max_bytes=2 << 30, action_native_prefill_aligned_pieces=False,
            action_native_prefill_cta_gate=False,action_native_prefill_static_aligned=enabled,
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

    def run(mode, *, qualify=False, output_probe=False):
        nonlocal qualification_lse, native_dispatches
        if mode not in MODES:
            raise ValueError(mode)
        if wrapper._cached_module is not native_module or native._active.get() is not None:
            raise RuntimeError('Original FlashInfer dispatch was not restored')
        qualification_lse = qualify
        outputs, row_lses = [], []
        try:
            if mode == 'native':
                # No bank arm/disarm, Action metadata, statistics or score DMA.
                # Its complete GPU execution is fenced by the outer timed event.
                for _ in range(3):
                    original_attention(q, model_output)
                    native_dispatches += 1
                    if qualify or output_probe:
                        outputs.append(model_output.clone())
                    if qualify:
                        row_lses.append(native_lse.clone())
                if qualify or output_probe:
                    torch.cuda.synchronize()
                    return dict(outputs=outputs, native_row_lses=row_lses)
                return None
            action, bank, layers, authority = arms[mode == 'static_aligned']
            if qualify:
                # Include unused row/width sentinels in the byte proof. This
                # full-capacity fill is not part of production or timed runs.
                for lse in bank.lses.values():
                    lse.fill_(float('nan'))
            authority.action_history = ActionHistory(action.options, action.ids)
            authority.action_history.begin('0', [14])
            action.prepare(input_batch, positions)
            for layer in layers:
                layer.attn.impl.forward(layer.attn, q, None, None, None, metadata, model_output)
                if qualify or output_probe:
                    outputs.append(model_output.clone())
                if qualify:
                    row_lses.append(native_lse.clone())
            packets = action.capture(captured, input_batch)
            for packet in packets.values():
                if packet.event is not None:
                    packet.event.synchronize()
            if any(key.startswith('fallback:') for key in bank.counts):
                raise RuntimeError(f'Unexpected scoring fallback: {bank.counts}')
            packet = packets['0']
            if qualify or output_probe:
                result = dict(outputs=outputs, native_row_lses=row_lses,
                    history=tuple(packet.history), scores=packet.masses.clone(),
                    piece_lses=tuple(lse.clone() for lse in bank.lses.values()) if qualify else ())
                torch.cuda.synchronize()
                return result
            # Both the CPU score tensor and Event hold the pool lease. The
            # caller releases this tuple only after the existing D2H fence.
            return tuple(packet.history), packet, model_output
        finally:
            qualification_lse = False

    def dispatch_audit():
        return dict(native_calls=native_dispatches,
            cached_module_restored=wrapper._cached_module is native_module,
            stats_context_empty=native._active.get() is None,
            plan_info=list(wrapper._plan_info), original_module_type=type(native_module).__name__,
            stats_counts={str(enabled):dict(bank.counts)
                          for enabled, (_, bank, _, _) in arms.items()})

    return run, arms, dispatch_audit, dict(active_tokens_including_query=length, query_tokens=qlen,
        sampled_rows=qlen//32, layers=3, q_heads=heads, kv_heads=kh, head_dim=dim, kv_dtype=str(k.dtype),
        page_tokens=page, context_capacity=65576, max_segments=2202, width=6301, max_rows=max_rows,
        max_num_reqs=8, q_dtype=str(q.dtype), q_scale=1., k_scale=.8,
        pinned_download_pool_enabled=True,
        static_qualification_all_segments_aligned16=True,
        actual_cta_tile_q=int(wrapper._plan_info[3]),
        same_q_k_arena=True, full_causal_normalization=True,
        three_layers_share_operands=True,
        cache_locality_note='The three sampled layers share Q/K/V here; serving layers have separate caches and intervening model work.',
        acceptance_archive_graph_publication_included=False)



MODES = ('native', 'generic', 'static_aligned')


def order_for(repetition):
    # Every treatment appears in each position twice per six rounds; all
    # within-round directed predecessor pairs occur equally often. Cross-round
    # transitions are not part of that balance. Default 12 repeats gives
    # two complete blocks, avoiding a permanently cold first treatment.
    return tuple(itertools.permutations(MODES))[repetition % 6]


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def qualify(run, audit):
    before = audit()['stats_counts']
    baseline = run('native', qualify=True)
    assert audit()['stats_counts'] == before, 'Native called KVMem statistics'
    generic = run('generic', qualify=True)
    candidate = run('static_aligned', qualify=True)
    for values in (generic, candidate):
        for key in ('outputs', 'native_row_lses'):
            assert len(values[key]) == len(baseline[key]) == 3
            for actual, expected in zip(values[key], baseline[key]):
                assert_bits(actual, expected)
    assert candidate['history'] == generic['history']
    assert_bits(candidate['scores'], generic['scores'])
    assert len(candidate['piece_lses']) == len(generic['piece_lses']) == 3
    for actual, expected in zip(candidate['piece_lses'], generic['piece_lses']):
        assert_bits(actual, expected)
    del baseline, generic, candidate, actual, expected, values
    # Check the exact non-LSE-output route used during formal timing as well.
    # Native row-LSE and piece-LSE proof above is stronger diagnostic coverage,
    # but enabling an optional output must not substitute for this route.
    before = audit()['stats_counts']
    baseline = run('native', output_probe=True)
    assert audit()['stats_counts'] == before
    for mode in MODES[1:]:
        values = run(mode, output_probe=True)
        for actual, expected in zip(values['outputs'], baseline['outputs']):
            assert_bits(actual, expected)
    return dict(bitwise_all_three_attention_outputs=True,
        bitwise_all_three_native_row_lse=True, bitwise_generic_static_scores=True,
        bitwise_generic_static_full_piece_lse_including_sentinels=True,
        qualification_only_native_row_lse_output=True,
        bitwise_attention_on_exact_timed_non_lse_route=True)


def kernel_profile(run, mode):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        run(mode)
        torch.cuda.synchronize()
    kernels = [event for event in profile.events()
               if str(event.device_type).endswith('CUDA')]
    attention = [event for event in kernels if 'BatchPrefillWithPagedKVCacheKernel' in event.name]
    names = [event.name for event in attention]
    validate_kernel_names(mode, names, [event.name for event in kernels])
    return dict(fa2_kernel_ms=sum(event.device_time_total for event in attention) / 1000,
        all_gpu_kernels_ms=sum(event.device_time_total for event in kernels) / 1000,
        fa2_kernels=[dict(name=event.name, duration_us=event.device_time_total) for event in attention],
        genuine_native_without_stats=mode == 'native')


def validate_kernel_names(mode, names, all_names):
    if len(names) != 3:
        raise RuntimeError(f'Expected three actual FA2 kernels for {mode}, found {len(names)}')
    expected = {'native':'flashinfer::DefaultAttention<',
                'generic':'KVMemAttention', 'static_aligned':'KVMemStaticAlignedAttention'}[mode]
    if not all(expected in name for name in names):
        raise RuntimeError(f'Wrong FA2 specialization for {mode}: {names}')
    if mode == 'native' and any(any(marker in name for marker in
            ('KVMem', '_row_lse', '_segment_scores', '_clear_lse', '_map_pieces'))
            for name in all_names):
        raise RuntimeError('Native baseline launched a KVMem statistics kernel')


def summarize(samples, profiles):
    summaries = {mode: {metric: statistics.mean(s[metric] for s in samples if s['mode']==mode)
                 for metric in ('wall_ms','cuda_interval_ms')} for mode in MODES}
    kernels = {mode: statistics.mean(p['fa2_kernel_ms'] for p in profiles[mode]) for mode in MODES}
    overhead = {mode:dict(
        complete_wall_ms=summaries[mode]['wall_ms']-summaries['native']['wall_ms'],
        complete_cuda_interval_ms=summaries[mode]['cuda_interval_ms']-summaries['native']['cuda_interval_ms'],
        fa2_kernel_ms=kernels[mode]-kernels['native']) for mode in MODES[1:]}
    return summaries, kernels, overhead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[16384,32768,49152])
    parser.add_argument('--query-tokens', nargs='+', type=int, default=[2048,8192,14336])
    parser.add_argument('--max-rows', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--profile-repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 6 or args.repeats % 6 or args.profile_repeats < 1 or args.max_rows < 1:
        raise ValueError('Repetitions must be complete six-order blocks; capacities must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a fresh output path')
    evidence = dict(scope='Fixed Q/K/V three-layer native FA2 cost and complete KVMem statistics overhead; not model throughput.',
        comparison='Same operands, scale, strides, physical page table and original FA2 plan in all arms.',
        native_baseline='Original FlashInfer DefaultAttention module, no Action/bank preparation, normalization, validation or scores D2H.',
        stats_arms='Three actual instrumented FA2 kernels plus complete Action prepare/capture, normalize, average and pinned D2H completion.',
        timed_native_row_lse_output=False,
        allocation_note='Two bounded stats banks coexist for all arms, so this does not compare process peak memory without KVMem.',
        geometry_teardown_gc_outside_timing=True,
        gpu=torch.cuda.get_device_name(), results=[])
    root = Path(__file__).resolve().parents[1]
    sources=[Path(__file__),*(root/'src/vllm_kvmem'/name for name in
        ['native_attention.py','native_attention_build.py','native_stats.cuh','native_prefill_stats.cuh'])]
    evidence['source_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for length,qlen in ((n,q) for q in args.query_tokens for n in args.lengths if q<=n):
        run, arms, audit, result = setup(length,args.output,qlen=qlen,max_rows=args.max_rows)
        assert arms[False][1].allocated_bytes == arms[True][1].allocated_bytes
        preparation_began=time.perf_counter()
        result['correctness']=qualify(run,audit)
        # Exercise the exact timed route (return_lse=False) after the stronger
        # qualification route. This also evicts the large qualification clones.
        for _ in range(3):
            for mode in MODES:
                run(mode)
        torch.cuda.synchronize()
        result['first_dispatch_correctness_and_warmup_seconds']=time.perf_counter()-preparation_began
        samples=[]
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for repetition in range(args.repeats):
            for mode in order_for(repetition):
                torch.cuda.synchronize()
                began=time.perf_counter();start.record()
                run(mode);end.record();end.synchronize()
                samples.append(dict(repetition=repetition,mode=mode,
                    wall_ms=1000*(time.perf_counter()-began),cuda_interval_ms=start.elapsed_time(end)))
        profiles={mode:[kernel_profile(run,mode) for _ in range(args.profile_repeats)] for mode in MODES}
        summary,kernels,overhead=summarize(samples,profiles)
        result.update(samples=samples,summary=summary,kernel_profiles=profiles,kernel_summary_ms=kernels,
            measured_overhead_over_native_ms=overhead,dispatch_audit=audit())
        assert result['dispatch_audit']['cached_module_restored'] and result['dispatch_audit']['stats_context_empty']
        assert result['dispatch_audit']['native_calls']>0
        result['native_state']={str(enabled):dict(counts=dict(bank.counts),
            static_aligned_counts=dict(bank.static_prefill_counts),allocated_bytes=bank.allocated_bytes,
            static_source_fingerprint=bank.static_prefill_source_fingerprint,
            static_unavailable=bank.static_prefill_unavailable,source_fingerprint=bank.source_fingerprint)
            for enabled,(_,bank,_,_) in arms.items()}
        assert arms[False][1].static_prefill_counts.get('static_dispatch_calls',0)==0
        assert arms[True][1].static_prefill_counts.get('static_dispatch_calls',0)>0
        assert not any(key.startswith('generic_dispatch:') for key in arms[True][1].static_prefill_counts)
        assert arms[True][1].static_prefill_unavailable is None
        assert arms[False][1].source_fingerprint != arms[True][1].static_prefill_source_fingerprint
        evidence['results'].append(result)
        args.output.write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(dict(length=length,query_tokens=qlen,complete_component=summary,
            fa2_kernel_ms=kernels,overhead_over_native_ms=overhead)),flush=True)
        del run,arms,audit
        gc.collect()
        torch.cuda.empty_cache()
    final_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    evidence['source_unchanged']=final_hashes==evidence['source_sha256']
    evidence['final_source_sha256']=final_hashes
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    if not evidence['source_unchanged']:
        raise RuntimeError('Component source changed during measurement; results are not qualified')


if __name__ == '__main__':
    main()
