"""Opt-in partial final logical span inside one physical FA2 step.

Same qualification as test_physical_prefill_stats_gpu (2% relative, 2e-4
absolute, full-context FP32 oracle): the standalone clipped step and its
coalesced projection must both match the oracle and each other. Same-shape
instrumented and original FA2 outputs stay bitwise identical.
"""
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.logical_action import LogicalScorePlan, project_logical_scores
from vllm_kvmem.serving.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.serving.native_attention import NativeAttentionStats
from vllm_kvmem.serving.request_plan import RequestPlan
import test_physical_prefill_stats_gpu as base

pytestmark = base.pytestmark
NAME, RTOL, ATOL = base.NAME, base.RTOL, base.ATOL


def geometry(tmp_path, sizes, prefix=4096):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(9361)
    heads, kvheads, dim, page = 24, 4, 256, 16
    count = sum(sizes)
    physical = (prefix+count)//page+7
    cache = (torch.randn(physical, kvheads, page, 2*dim, device='cuda')*.4).to(torch.float8_e4m3fn)
    key, value = cache.transpose(1, 2).split(dim, -1)
    table = torch.randperm(physical, device='cuda', dtype=torch.int32)[:(prefix+count+page-1)//page]
    query = torch.randn(count, heads, dim, device='cuda', dtype=torch.bfloat16)
    workspace = torch.zeros(256 << 20, device="cuda", dtype=torch.uint8)
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    shapes = [*((n, prefix+o+n) for o, n in zip(offsets, sizes)), (count, prefix+count)]
    wrappers = []
    for qlen, length in shapes:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD', backend='fa2')
        wrapper.plan(torch.tensor([0, qlen], device='cuda', dtype=torch.int32),
            torch.tensor([0, (length+page-1)//page], device='cuda', dtype=torch.int32),
            table[:(length+page-1)//page],
            torch.tensor([(length-1) % page+1], device='cuda', dtype=torch.int32),
            heads, kvheads, dim, page, causal=True, sm_scale=dim**-.5,
            q_data_type=query.dtype, kv_data_type=key.dtype)
        wrappers.append(wrapper)
    class Impl:
        def forward(self, layer, q, k, v, cache, meta, out):
            return meta.prefill.wrapper.run(q, (key, value), q_scale=1., k_scale=.37, v_scale=.3, out=out)
    impl = Impl(); original = impl.forward
    layer = NS(num_heads=heads, head_dim=dim, q_norm=NS(weight=query),
               attn=NS(layer_name=NAME, impl=impl, _q_scale_float=1.))
    runner = NS(model=NS(named_modules=lambda: [(NAME, layer)]), _kvmem_layers=[NAME],
                max_num_tokens=count, max_num_reqs=2)
    options = NS(action_native_scores='reuse', action_layers='all', action_native_max_rows=count//32+8,
        action_native_max_bytes=256 << 20, memory=NS(select_budget=8192, gen_budget=4096, block_tokens=32))
    bank = NativeAttentionStats(runner, options, tmp_path)
    assert bank.disabled is None, bank.disabled
    queries = [*(query[o:o+n] for o, n in zip(offsets, sizes)), query]
    outputs = [torch.empty_like(q) for q in queries]
    metas = [NS(prefill=NS(wrapper=w), q_data_type_prefill=query.dtype, num_decode_tokens=0) for w in wrappers]
    def execute(index, baseline=False):
        if baseline:
            return original(layer.attn, queries[index], None, None, None, metas[index], torch.empty_like(queries[index]))
        return impl.forward(layer.attn, queries[index], None, None, None, metas[index], outputs[index])
    return NS(bank=bank, key=key, value=value, table=table, query=query, outputs=outputs, execute=execute,
              heads=heads, kvheads=kvheads, dim=dim, page=page, prefix=prefix, count=count, offsets=offsets)


@pytest.mark.parametrize('sizes', [(1024, 1024, 1024, 224), (1024, 1024, 32),
                                   (2048, 2048, 2048, 992), (2048, 2048, 1024)])
def test_partial_final_span_scores_match_standalone_step_and_oracle(tmp_path, sizes):
    env = geometry(tmp_path, list(sizes))
    chunk = sizes[0]
    rows_for = [list(range(n-1, -1, -32))[::-1] for n in sizes]   # PrefillHistory stride-32 rule
    physical_rows = [o+r for o, rows in zip(env.offsets, rows_for) for r in rows]
    originals = tuple(base.plan(env.prefix+o, n, rows) for o, n, rows in zip(env.offsets, sizes, rows_for))
    old = []
    for index, pending in enumerate(originals):
        env.bank.arm({'r': pending}); env.execute(index)
        mass = env.bank.score(NAME, 'r', True)
        assert mass is not None, env.bank.reasons
        old.append(mass.clone())
        torch.testing.assert_close(env.outputs[index], env.execute(index, True), rtol=0, atol=0)
    physical = base.plan(env.prefix, env.count, physical_rows)
    env.bank.arm({'r': physical}); env.execute(len(sizes))
    masses = env.bank.score(NAME, 'r', True)
    assert masses is not None, env.bank.reasons
    torch.testing.assert_close(env.outputs[len(sizes)], env.execute(len(sizes), True), rtol=0, atol=0)
    spans = tuple(LogicalPrefillSpan(i, 8192+o, 4096+o, n, i*chunk) for i, (o, n) in enumerate(zip(env.offsets, sizes)))
    group = LogicalPrefillGroup('r', (1, 0, 0), 7, 4096, 8192+env.count+2048, env.prefix+env.count, spans,
                                RequestPlan(None, block_tokens=32, gen_budget=4096))
    plans = tuple(LogicalScorePlan(span, tuple(rows), tuple(p['starts']), tuple(p['counts']), tuple(p['history']))
                  for span, rows, p in zip(spans, rows_for, originals))
    projected = project_logical_scores(group, plans, masses.clone().cpu(), physical_rows=tuple(physical_rows))
    for i, (pending, part) in enumerate(zip(originals, projected)):
        expected = base.oracle(env, [env.offsets[i]+r for r in rows_for[i]], pending['starts'], pending['counts'])
        candidate = part.masses.to('cuda')
        torch.testing.assert_close(old[i], expected, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(candidate, expected, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(candidate, old[i], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(candidate.sum(-1), torch.ones(len(rows_for[i]), device='cuda'), rtol=0, atol=2e-5)
    assert env.bank.validation_failures == 0
    assert not any(key.startswith('fallback:') for key in env.bank.counts)
