"""Real native batch schemas, private-row isolation and failure transactions on CPU."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.query_replay_batch import (
    QueryReplayCoBatch, QueryFrame, UnsupportedQueryBatch, merge_query_batch, public_hidden)


def batch(ids=('a', 'b'), counts=(5, 7), indices=(0, 1), drafts=(4, 0)):
    n, total = len(ids), sum(counts)
    counts = np.asarray(counts, dtype=np.int32)
    starts = np.concatenate((np.zeros(1, dtype=np.int32), counts.cumsum(dtype=np.int32)))
    computed = np.full(n, 128, dtype=np.int32)
    prefill = np.asarray([False] + [True] * (n - 1)) if drafts else np.ones(n, dtype=np.bool_)
    num_logits = np.asarray([d + 1 if not p else 1 for d, p in zip(drafts or [0]*n, prefill)], dtype=np.int32)
    logits = np.concatenate([np.arange(starts[i + 1] - num_logits[i], starts[i + 1]) for i in range(n)])
    cu = np.concatenate((np.zeros(1, dtype=np.int32), num_logits.cumsum(dtype=np.int32)))
    return InputBatch(req_ids=list(ids), num_reqs=n, num_reqs_after_padding=n,
        idx_mapping=torch.tensor(indices, dtype=torch.int32), idx_mapping_np=np.asarray(indices),
        expanded_idx_mapping=torch.tensor(np.repeat(indices, num_logits)),
        expanded_local_pos=torch.tensor(np.concatenate([np.arange(k) for k in num_logits]), dtype=torch.int32),
        num_scheduled_tokens=counts, num_tokens=total, num_tokens_after_padding=total,
        num_draft_tokens=sum(drafts or ()),
        num_draft_tokens_per_req=np.asarray(drafts, dtype=np.int32) if drafts else None,
        query_start_loc=torch.tensor(starts), query_start_loc_np=starts,
        seq_lens=torch.tensor(computed + counts), seq_lens_cpu_upper_bound=torch.tensor(computed + counts),
        dcp_local_seq_lens=None, num_computed_tokens_np=computed,
        prefill_len_np=computed + counts, num_computed_prefill_tokens_np=computed,
        is_prefilling_np=prefill, has_prefill=bool(np.any(prefill)),
        input_ids=torch.arange(total, dtype=torch.int32), positions=torch.arange(128, 128 + total),
        is_padding=torch.zeros(total, dtype=torch.bool), logits_indices=torch.tensor(logits),
        cu_num_logits=torch.tensor(cu), cu_num_logits_np=cu,
        has_structured_output_reqs=False, prompt_lens=None, max_query_len=int(counts.max()))


def private_batch():
    return batch(('private',), (3,), (2,), None)


def test_merge_preserves_public_speculative_coordinates_and_all_inputs():
    public, private = batch(), private_batch()
    original = {k:v.clone() if isinstance(v, torch.Tensor) else v.copy() if isinstance(v, np.ndarray)
                else v for k,v in vars(public).items()}
    merged = merge_query_batch(public, private)
    assert merged.req_ids == ['a', 'b', 'private']
    assert merged.num_tokens == 15 and merged.num_reqs == 3
    assert merged.num_draft_tokens == 4 and merged.num_draft_tokens_per_req.tolist() == [4, 0, 0]
    assert merged.query_start_loc.tolist() == [0, 5, 12, 15]
    assert merged.logits_indices.tolist() == [0, 1, 2, 3, 4, 11, 14]
    assert merged.cu_num_logits.tolist() == [0, 5, 6, 7]
    assert merged.expanded_idx_mapping.tolist() == [0, 0, 0, 0, 0, 1, 2]
    for key, value in vars(public).items():
        if isinstance(value, torch.Tensor): assert torch.equal(value, original[key])
        elif isinstance(value, np.ndarray): assert np.array_equal(value, original[key])


@pytest.mark.parametrize('change', [dict(req_ids=['a']), dict(idx_mapping_np=np.asarray([0])),
    dict(num_draft_tokens=1), dict(num_draft_tokens_per_req=np.asarray([0])),
    dict(has_prefill=False), dict(num_tokens_after_padding=4),
    dict(query_start_loc_np=np.asarray([0, 2])), dict(fast_prefill=object())])
def test_invalid_private_batches_fail_before_modifying_public_inputs(change):
    public = batch()
    before = public.input_ids.clone()
    with pytest.raises(UnsupportedQueryBatch):
        merge_query_batch(public, replace(private_batch(), **change))
    assert torch.equal(public.input_ids, before)


def test_hidden_projection_preserves_public_rows_and_auxiliary_outputs():
    hidden = torch.arange(30).reshape(15, 2)
    projected = public_hidden((hidden, [hidden + 100]), 12)
    assert projected[0].tolist() == hidden[:12].tolist()
    assert projected[1][0].tolist() == (hidden[:12] + 100).tolist()
    with pytest.raises(ValueError): public_hidden(hidden, 16)
    with pytest.raises(ValueError): public_hidden(object(), 12)


def coordinator():
    from test_async_refresh import command
    obj = AsyncRefresh.__new__(AsyncRefresh)
    job = RefreshJob(command('private'), stage='query')
    # Public query coordinates in this fixture cover both original input tokens.
    job.state = (NS(), NS(), NS())
    obj.jobs = {'private':job}
    obj.options = NS(refresh_quantum_tokens=256, query_replay_max_bytes=1024*1024,
                     guided_query_tokens=0, guided_reselect='boundary')
    obj._submitted = []
    obj._submit = lambda value: (setattr(value, 'stage', 'assembly_wait'), obj._submitted.append(value))
    obj.adapter = NS(validate=lambda bundle:None)
    obj._restored, obj._quarantines = [], []
    obj._restore_state = lambda value: obj._restored.append(value)
    runtime = NS(publication_ready=lambda ids:True, quarantine_publication=obj._quarantines.append)
    record = NS(name='layer', query=torch.arange(30).reshape(15, 1, 2))
    obj.runner = NS(max_num_reqs=8, max_num_tokens=64, _kvmem_request_generations={'private':4},
                    _kvmem_capture_runtime=runtime, _kvmem_graph_captures=[record], _kvmem_layers=['layer'])
    obj.runner._kvmem_rebuild_public_metadata = lambda public: ('public-attention', 'public-slots')
    obj.emit = lambda *args, **kwargs:None
    co = QueryReplayCoBatch(obj)
    obj.cobatch = co
    return obj, co, job


def scheduled(tokens=12, requests=('a','b')):
    return NS(num_scheduled_tokens=dict.fromkeys(requests, tokens//len(requests)),
              total_num_scheduled_tokens=tokens)


def test_admission_declines_are_bounded_and_do_not_starve_the_original_path():
    refresh, co, job = coordinator()
    refresh.runner.max_num_tokens = 12
    for _ in range(2):
        co.begin_turn(('a','b'))
        assert co.defer(job)
        assert not co.arm(scheduled())
    co.begin_turn(('a','b'))
    assert not co.defer(job) and not co.arm(scheduled())
    assert co.counts['decline:token-capacity'] == 2
    assert job.offset == 0 and job.stage == 'query'


def test_memory_admission_rejects_before_allocating_or_touching_a_private_batch(monkeypatch):
    import vllm_kvmem.async_refresh as module
    refresh, co, job = coordinator()
    refresh.options.query_replay_max_bytes = 1
    monkeypatch.setattr(module, 'maintenance_batch', lambda *args:pytest.fail('must reject before allocation'))
    assert co.arm(scheduled())
    public = batch()
    assert co.prepare(public) is public and co.frame is None
    assert job.offset == 0 and co.counts['decline:memory-capacity'] == 1


@pytest.mark.parametrize('branch', ['physical', 'router'])
def test_first_qualification_preserves_unqualified_public_execution(branch):
    refresh, co, job = coordinator()
    if branch == 'physical':refresh.runner._kvmem_logical_prefill_groups = {'a':object()}
    else:refresh.runner.routed_experts_capturer = object()
    assert not co.arm(scheduled()) and co.frame is None
    assert job.offset == 0


def test_prepare_uses_a_bounded_independent_input_frame(monkeypatch):
    import vllm_kvmem.async_refresh as module
    refresh, co, job = coordinator()
    monkeypatch.setattr(module, 'maintenance_batch', lambda *args:batch(('private',),(2,),(2,),None))
    assert co.arm(scheduled())
    public = batch()
    mixed = co.prepare(public)
    assert mixed.num_tokens == 14 and public.num_tokens == 12
    assert mixed.input_ids.data_ptr() != public.input_ids.data_ptr()
    assert co.public_batch(mixed) is public
    assert 0 < co.snapshot()['peak_bytes'] <= refresh.options.query_replay_max_bytes
    with pytest.raises(RuntimeError):co.begin_turn(('a',))


@pytest.mark.parametrize('mutation', ['generation', 'cancel', 'offset', 'foreign_job', 'command', 'state'])
def test_stale_owner_and_cancel_are_rejected_before_native_execution(mutation):
    refresh, co, job = coordinator()
    assert co.arm(scheduled())
    if mutation == 'generation': refresh.runner._kvmem_request_generations['private'] += 1
    if mutation == 'cancel': job.cancelled = True
    if mutation == 'offset': job.offset += 1
    if mutation == 'foreign_job': refresh.jobs['private'] = object()
    if mutation == 'command': job.command = replace(job.command, input_tokens=(30,40))
    if mutation == 'state': job.state = (NS(),NS(),NS())
    with pytest.raises(RuntimeError, match='owner or accepted prefix'):
        co._validate(co.frame)


def ExecuteState(batch, hidden, aux, metadata, slots):
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState
    return ExecuteModelState(input_batch=batch, hidden_states=hidden, aux_hidden_states=aux,
        attn_metadata=metadata, slot_mappings_by_layer=slots, dp_sync=None,
        finished_req_ids={'already-finished'}, ec_connector_output=None,
        routed_experts=None, cudagraph_stats=None)


def test_completion_extracts_private_q_then_exposes_only_public_state_to_sampling_and_mtp():
    refresh, co, job = coordinator()
    public = batch()
    # The job has 2 private tokens; its native suffix uses precisely those rows.
    private = batch(('private',), (2,), (2,), None)
    combined = merge_query_batch(public, private)
    assert co.arm(scheduled())
    co.frame.public, co.frame.combined = public, combined
    hidden = torch.arange(28).reshape(14,2)
    result = co.complete(ExecuteState(combined, hidden, [hidden+1], 'mixed', 'mixed'))
    assert result.input_batch is public
    assert result.finished_req_ids == {'already-finished'} and result.dp_sync is None
    assert result.hidden_states.shape == (12,2) and result.aux_hidden_states[0].shape == (12,2)
    assert result.attn_metadata == 'public-attention' and result.slot_mappings_by_layer == 'public-slots'
    assert torch.equal(job.query_rows[0], refresh.runner._kvmem_graph_captures[0].query[12:14].unsqueeze(0))
    refresh.runner._kvmem_graph_captures[0].query.fill_(-100)
    assert not torch.any(job.query_rows[0] == -100)
    assert job.offset == 2 and job.steps == 1 and job.stage == 'assembly_wait'
    assert co.frame is None and co.counts['completed'] == 1


def test_projection_failure_quarantines_and_restores_without_retrying_a_mutated_public_forward():
    refresh, co, job = coordinator()
    assert co.arm(scheduled())
    frame = co.frame
    error = RuntimeError('target failed after enqueue')
    co.abort(error)
    assert co.failed_frame is frame and co.poisoned and co.frame is None
    assert refresh._restored == [job] and refresh._quarantines == [error]
    assert job.offset == 0 and not job.query_rows
    with pytest.raises(RuntimeError): co.arm(scheduled())
    with pytest.raises(RuntimeError): co.begin_turn(('a',))


def test_partial_query_keeps_its_state_until_all_quanta_have_been_captured():
    refresh, co, job = coordinator()
    job.command = replace(job.command, input_tokens=tuple(range(6)))
    q = torch.arange(4).reshape(1,2,1,2)
    refresh.accept_query(job, q, 2)
    assert job.stage == 'query' and job.offset == 2 and not refresh._submitted
    refresh.accept_query(job, q+10, 4)
    refresh.accept_query(job, q+20, 6)
    assert job.stage == 'assembly_wait' and job.offset == 6 and len(job.query_rows) == 3


@pytest.mark.parametrize('backend', ['eager', 'compiled'])
@pytest.mark.parametrize('embedding_input', [False, True])
def test_private_forward_selects_the_requested_compilation_path(monkeypatch, backend, embedding_input):
    from contextlib import contextmanager
    import vllm.forward_context as context
    import vllm_kvmem.async_refresh as module
    refresh, co, job = coordinator()
    refresh.options.query_replay_backend = backend
    runner = refresh.runner
    runner._kvmem_last_batch, runner._kvmem_window_batch = 'old-batch', 'old-window'
    runner.prepare_attn = lambda batch: ('tables', 'slots')
    runner.model_state = NS(prepare_attn=lambda *args:{}, prepare_inputs=lambda *args:{})
    embedding = torch.ones(3,4)
    embedding_calls = []
    def prepare_embeddings(scheduled, batch, requests):
        assert scheduled == {} and requests is runner.req_states
        embedding_calls.append(batch)
        return embedding
    runner.model_state.prepare_inputs_embeds = prepare_embeddings
    runner.uses_inputs_embeds = embedding_input
    runner.model = NS()
    import vllm.model_executor.models.interfaces as interfaces
    monkeypatch.setattr(interfaces, 'requires_raw_input_tokens', lambda model:False)
    runner.vllm_config = NS()
    runner.req_states = NS()
    runner.attn_groups, runner.kv_cache_config = [], NS()
    runner._kvmem_graph = True
    flags = []
    @contextmanager
    def forward_context(*args, **kwargs):
        flags.append(kwargs['skip_compiled'])
        yield
    monkeypatch.setattr(context, 'set_forward_context', forward_context)
    import vllm.v1.worker.gpu.attn_utils as attn
    monkeypatch.setattr(attn, 'build_slot_mappings_by_layer', lambda *args:{})
    monkeypatch.setattr(module, 'maintenance_batch', lambda *args:private_batch())
    observed = []
    refresh.original = lambda **kwargs:observed.append(kwargs)
    _, q = refresh._forward(job, [10,20], 64)
    assert flags == [backend == 'eager']
    assert q.shape == (1,2,1,2)
    assert runner._kvmem_last_batch == 'old-batch' and runner._kvmem_window_batch == 'old-window'
    assert job.query_execution_counts == {backend:1}
    expect_embeddings = backend == 'compiled' and embedding_input
    assert bool(embedding_calls) is expect_embeddings
    assert (observed[0]['input_ids'] is None) is expect_embeddings
    assert observed[0]['inputs_embeds'] is (embedding if expect_embeddings else None)


def test_actual_runner_hooks_project_before_work_accounting_sampling_or_mtp(monkeypatch):
    import vllm.v1.worker.gpu.attn_utils as attn
    import vllm_kvmem.async_refresh as module
    from vllm_kvmem.runner import make_probe_runner
    refresh, co, job = coordinator()
    source_runner = refresh.runner
    public = batch()
    events = []
    class Base:
        def prepare_inputs(self, *args, **kwargs): return public
        def prepare_attn(self, batch):
            events.append(('native-attention', tuple(batch.req_ids)))
            return ('tables','slots')
        def execute_model(self, output, **kwargs):
            assert kwargs['is_profile'] is True
            mixed = self.prepare_inputs(output, None, None)
            assert mixed.req_ids == ['a','b','private']
            self.prepare_attn(mixed)
            events.append(('target', tuple(mixed.req_ids)))
            hidden = torch.arange(28).reshape(14,2)
            self.execute_model_state = ExecuteState(mixed, hidden, [hidden+1], {}, {})
    runner = make_probe_runner(Base)()
    runner.__dict__.update(vars(source_runner))
    refresh.runner = runner
    runner._kvmem_async_refresh = refresh
    runner._kvmem_window_adapter = None
    runner._kvmem_sessions = NS(before_attention=lambda b:events.append(('session',tuple(b.req_ids))))
    runner._kvmem_bounded_target = False
    runner._kvmem_step_timing = None
    runner.model_state = NS(prepare_attn=lambda *args:'public-attention')
    # Use the real projection method rather than the coordinator fixture stub.
    del runner._kvmem_rebuild_public_metadata
    runtime = runner._kvmem_capture_runtime
    runtime.action = NS(prepare=lambda b,p:events.append(('scores',tuple(b.req_ids))))
    runtime.note_prefill_pipeline_forward = lambda b:events.append(('pipeline',tuple(b.req_ids)))
    runner._kvmem_public_work = NS(submit=lambda b, offsets:events.append(('work',tuple(b.req_ids))))
    monkeypatch.setattr(attn, 'build_slot_mappings_by_layer', lambda *args:'public-slots')
    monkeypatch.setattr(module, 'maintenance_batch', lambda *args:batch(('private',),(2,),(2,),None))
    runner.attn_groups, runner.kv_cache_config = [], NS()
    co.begin_turn(('a','b'))
    runner._kvmem_execute_public_forward(scheduled(), observe_work=True)
    assert runner.execute_model_state.input_batch is public
    assert runner._kvmem_last_batch is public
    assert runner.execute_model_state.hidden_states.shape == (12,2)
    assert runner.execute_model_state.input_batch.cu_num_logits_np[-1] == 6
    assert ('target',('a','b','private')) in events
    assert events.count(('session',('a','b'))) == 1
    assert events.count(('scores',('a','b'))) == 1
    assert events[-1] == ('work',('a','b'))
    assert not any('private' in ids for kind,ids in events if kind in ('scores','work','session','pipeline'))
