"""Regression cases found by the independent V2 acceptance audit."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
import torch

from vllm_kvmem.request_plan import RequestPlan, RequestCapacityError
from test_graph_lifecycle import fixture, complete
from test_graph_pressure_v2 import configure


@pytest.mark.parametrize('policy,all_fit', [
    ('attention', False), ('graph', False), ('legacy_attention', False), ('recency', False),
    ('attention', True), ('graph', True)])
@pytest.mark.parametrize('entry', ['sync', 'async'])
def test_refresh_entry_reaches_native_pressure_below_query_recent_budget(
        tmp_path, monkeypatch, policy, entry, all_fit):
    authority, archive, graph = fixture(tmp_path, sink=1, recent=2, budget=32 if all_fit else 16)
    state = configure(authority, alpha=0, policy='attention' if policy == 'legacy_attention' else policy)
    state.scores = {4: 1.}
    if all_fit:
        from vllm_kvmem.prefill_attention import select_pressure
        previous = select_pressure(authority, max_tokens=8)
        complete(authority)
        del previous
        assert [row[2] for row in authority.memory._store.working_records()] == [0, 4]
    options = NS(memory=authority.memory.config, pressure_policy=policy,
                 packed_assembly=False, direct_pinned_reads=False, async_refresh=False,
                 guided_query_tokens=0, rolling_active_window=False)
    if policy == 'legacy_attention':
        del options.pressure_policy
        options.action_pressure = True
    command = NS(request_id='selection', command_id=1, selection='pressure', guided_reason='',
        query_span=None, mandatory_spans=(),
        groups=((0, NS(window_tokens=24 if all_fit else 8, history_end=24, replay_tokens=1,
                       original_block_ids=())),))
    runtime = NS(config=authority.memory.config, options=options, names=(),
        authorities={'selection': authority}, drain_requests=lambda *a, **k: True,
        _wait_archive_progress=lambda: None)
    selected = []

    class AssemblyReached(Exception):
        pass

    def assemble(a, plan):
        slots = {row.block_id for row in plan.remaps}
        selected.extend(r.history_begin for r in a.records.values() if r.handle.slot in slots)
        assert plan.total_window_tokens == 8
        raise AssemblyReached  # The genuine picker ran; no GPU copy is needed.

    from vllm_kvmem.window_assembly import NativeWindowAssembler
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner',
                        lambda *a, **kw: NS(assemble=assemble))
    if entry == 'async':
        from vllm_kvmem.async_refresh import AsyncRefresh
        monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
        monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
        manager = NS(options=options, runner=NS(device='cpu'), runtime=runtime,
            stream=NS(wait_event=lambda _: None), maintenance=None,
            emit=lambda *a, **kw: None)
        job = NS(command=command, authority=authority, history=24, graph_binding=graph.binding)
        run = lambda: AsyncRefresh._assemble(manager, job, None, None)
    else:
        from vllm_kvmem import bounded_runtime
        from vllm_kvmem.options import RuntimeOptions
        monkeypatch.setenv('KVMEM_WINDOW_METADATA', '1')
        monkeypatch.setattr(RuntimeOptions, 'from_vllm', lambda _: options)
        monkeypatch.setattr(bounded_runtime, 'full_attention_groups', lambda _: (0,))
        batch = NS(req_ids=['selection'], num_computed_tokens_np=np.array([24]),
                   num_scheduled_tokens=np.array([1]))
        runner = NS(speculative_config=None, _kvmem_continuous=True,
            vllm_config=NS(max_concurrent_batches=1,
                cache_config=NS(enable_prefix_caching=False, mamba_cache_mode='none')),
            model=NS(forward=lambda *a, **kw: pytest.fail('Pressure must not run a probe')),
            _kvmem_last_batch=batch, _kvmem_capture_runtime=runtime, kv_cache_config=None,
            _kvmem_window_adapter=NS(offsets={}), req_states=NS(req_id_to_index={'selection': 0}),
            block_tables=NS(blocks_per_kv_block={0: 1}, num_blocks=NS(np=np.array([[0]])),
                block_tables={0: NS(gpu=torch.empty((1, 0))) }))
        bounded_runtime.install_bounded_target(runner)
        runner._kvmem_query_refresh(command)
        run = runner.model.forward
    if policy == 'recency':
        with pytest.raises(ValueError, match='sink.*recent'):
            run()
        assert not selected
        assert authority.close() and archive.close()
        return
    with pytest.raises(AssemblyReached):
        run()
    assert selected == [0, 4]
    assert authority.memory.config.recent_blocks == 2
    assert authority.memory._store.runtime_select_budget == 0
    complete(authority)
    assert authority.close() and archive.close()


@pytest.mark.parametrize('policy', ['recency', 'attention', 'graph'])
def test_capacity_admission_distinguishes_pressure_from_query(policy):
    pressure = RequestPlan(None, block_tokens=4, gen_budget=16, output_tokens=16)
    kwargs = dict(sink_blocks=1, recent_blocks=2, pressure_policy=policy)
    if policy == 'recency':
        with pytest.raises(RequestCapacityError, match='required 12'):
            pressure.validate_capacity(100, 16, **kwargs)
    else:
        assert pressure.validate_capacity(100, 16, **kwargs) == 8
        # Exact partial M restore also uses pressure despite an unchanged Query.
        assert pressure.validate_capacity(101, 16, resume_history=101,
                                          resume_window=20, **kwargs) == 12
        with pytest.raises(RequestCapacityError, match='required 12'):
            pressure.validate_capacity(100, 16, **{**kwargs, 'sink_blocks': 3})
    query = RequestPlan((96, 100), block_tokens=4, gen_budget=16, output_tokens=16)
    with pytest.raises(RequestCapacityError, match='required 12'):
        query.validate_capacity(100, 16, **kwargs)


@pytest.mark.parametrize('policy', ['recency', 'attention', 'graph'])
def test_native_http_and_session_admission_receive_pressure_policy(monkeypatch, policy):
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.http_sessions import normalize_query_plan
    from vllm_kvmem.session_control import SessionLedger
    config = dict(block_tokens=4, select_budget=16, gen_budget=16,
                  sink_blocks=1, recent_blocks=2, pressure_policy=policy)
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(config))
    payload = dict(prompt=list(range(100)), max_tokens=16, kvmem_reselect='off')
    cfg = KVMemConfig(**{k: v for k, v in config.items() if k != 'pressure_policy'})
    ledger = SessionLedger(2, 4, lambda *a: None, lambda *a: None, lambda *a: None,
        select_budget=16, gen_budget=16, memory_config=cfg, pressure_policy=policy)
    request = NS(num_prompt_tokens=100)
    plan = RequestPlan(None, block_tokens=4, gen_budget=16, output_tokens=16)
    for run in (lambda: normalize_query_plan(payload),
                lambda: ledger.validate_plan_capacity(request, plan)):
        if policy == 'recency':
            with pytest.raises(RequestCapacityError, match='required 12'):
                run()
        else:
            run()


@pytest.mark.parametrize('partial', [False, True])
def test_checkpoint_mandatory_fit_uses_shrunken_pressure_budget(tmp_path, partial):
    from test_graph_lifecycle import graph_checkpoints
    from test_graph_pressure_restore_v2 import finish
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path, partial=partial)
    configure(authority, alpha=0)
    checkpoint = pool.claim(prompt.namespace, 'new', list(range(24)), position=10 if partial else 8)
    plan = pool.prepare_claim_window(checkpoint, pressure=True, pressure_budget=8 if partial else 4,
                                    mandatory_spans=((0, 8, 'root'),))
    assert plan.total_window_tokens == (6 if partial else 4)
    assert [row[2] for row in authority.memory._store.working_records()] == ([0, 8] if partial else [0])
    finish(pool, checkpoint, authority, archive, plan)


@pytest.mark.parametrize('sink,spans', [(1, ()), (0, ((0, 4, 'explicit'),))])
def test_partial_tail_cannot_soften_fixed_sink_or_explicit_evidence(sink, spans):
    from vllm_kvmem.config import KVMemConfig
    cfg = KVMemConfig(block_tokens=4, select_budget=4, gen_budget=16,
                      sink_blocks=sink, recent_blocks=0)
    plan = RequestPlan(None, block_tokens=4, mandatory=spans)
    with pytest.raises(RequestCapacityError, match='accepted tail'):
        plan.mandatory_fit(10, cfg, preserve_tail=True)
