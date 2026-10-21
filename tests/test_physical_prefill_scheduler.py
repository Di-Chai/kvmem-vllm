from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_kvmem.admission import QueryPolicy
from test_phase_scheduler import scheduler as base_scheduler, request, NATIVE_RESERVE


def setup(monkeypatch, tmp_path, history=8192, prompt=262144):
    s = base_scheduler(monkeypatch, tmp_path)
    s.phase_policy = s._phase_decision = None
    s.prefill_physical_coalescing = True
    s.num_prefill_lookahead = 4
    s.query_policy = QueryPolicy(32, 32768, 16384)
    r = request('p', computed=history, prompt=prompt, output=0)
    s.running = [r]
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '2048')
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', NATIVE_RESERVE)
    s._prepare_logical_prefill()
    return s, r


def reserve(s, r, budget=16424):
    return s._reserve_prefill_lookahead(r, r.num_computed_tokens, budget)


def test_two_logical_admissions_and_receipt_reset(monkeypatch, tmp_path):
    s, r = setup(monkeypatch, tmp_path)
    assert reserve(s, r) == 4096
    group = s._logical_prefill_groups['p']
    assert [span.history_begin for span in group.spans] == [8192, 10240]
    assert group.scheduler_step == s.current_step
    assert group.capacity == 49152
    assert reserve(s, r, 3072) == 2048
    assert not s._logical_prefill_groups
    assert reserve(s, r) == 4096
    s._prepare_logical_prefill()
    assert not s._logical_prefill_groups


def test_first_forward_keeps_original_chunk_until_action_runtime_exists(monkeypatch, tmp_path):
    s, r = setup(monkeypatch, tmp_path, history=0)
    assert reserve(s, r) == 2048
    assert not s._logical_prefill_groups


def test_unaligned_history_after_query_keeps_original_short_admission(monkeypatch, tmp_path):
    s, r = setup(monkeypatch, tmp_path, history=8199)
    # The first qualified ordinary append closes the original partial page.
    assert reserve(s, r) == 2041
    assert not s._logical_prefill_groups


@pytest.mark.parametrize('history,expected', [(44992,4096), (47040,2048), (49088,32)])
def test_pressure_boundaries_preserve_original_visits(monkeypatch, tmp_path, history, expected):
    s, r = setup(monkeypatch, tmp_path, history=history)
    assert reserve(s, r) == expected
    assert bool(s._logical_prefill_groups) is (expected == 4096)


@pytest.mark.parametrize('kind', ['waiting','running','prepared','pending','transaction',
    'import','resident_import','manager_pending','phase','batch','controller','guided','draft'])
def test_only_single_owner_ordinary_work_merges(monkeypatch, tmp_path, kind):
    s, r = setup(monkeypatch, tmp_path)
    if kind == 'waiting': s.waiting = [request('q', computed=0, prompt=16384)]
    if kind == 'running': s.running.append(request('d'))
    if kind == 'prepared': s.prepared_refreshes['p'] = (8192,2048)
    if kind == 'pending': s.connector.pending['p'] = object()
    if kind == 'transaction': s.transactions['p'] = object()
    if kind in ('import','resident_import','manager_pending'):
        setattr(s.window_managers[0], {'import':'imports','resident_import':'resident_imports',
            'manager_pending':'pending'}[kind], {'p':object()})
    if kind == 'phase':
        # Direct helper qualification; the full phase scheduler has its own gate.
        s._phase_decision = object()
    if kind == 'batch': s.prefill_batch = object()
    if kind == 'controller': s.prefill_controller = object()
    if kind == 'guided': r._kvmem_guided_epoch = object()
    if kind == 'draft': r.spec_token_ids = [1,2,3,4]
    s._prepare_logical_prefill()
    from vllm_kvmem.request_plan import request_plan
    plan = request_plan(r, 32, 16384)
    assert s._logical_prefill_group(r, plan, 8192, 0, 16424, 2048) is None


@pytest.mark.parametrize('budget', [2048,3072,4095])
def test_native_credit_never_creates_three_k_logical_chunk(monkeypatch, tmp_path, budget):
    s, r = setup(monkeypatch, tmp_path)
    assert reserve(s, r, budget) == 2048
    assert not s._logical_prefill_groups


def test_query_native_tail_and_disabled_path_keep_old_shape(monkeypatch, tmp_path):
    s, r = setup(monkeypatch, tmp_path, prompt=12290)
    assert reserve(s, r) == 2048  # 4K would strand the MTP lookahead.
    r.num_prompt_tokens = r.num_tokens = r.num_tokens_with_spec = 262144
    r.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect':'force',
                                                 'kvmem_query_span':[10240,10272]}}
    assert reserve(s, r) == 2048
    r.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect':'off'}}
    s.prefill_physical_coalescing = False
    s._prepare_logical_prefill()
    assert reserve(s, r) == 2048


def test_archive_admission_checks_entire_physical_group(monkeypatch, tmp_path):
    s, r = setup(monkeypatch, tmp_path)
    calls = []
    s.archive_admission = NS(allows=lambda rid,h,n: calls.append((rid,h,n)) or False)
    assert reserve(s, r) == 0
    assert calls == [('p',8192,4096)]
    assert not s._logical_prefill_groups


def test_connector_receipt_tracks_actual_single_owner_and_idle_confirmation(monkeypatch, tmp_path):
    from vllm_kvmem.connector import ControlMetadata
    s, r = setup(monkeypatch, tmp_path)
    assert reserve(s, r) == 4096
    s._bounded_target = False  # Skip unrelated refresh plumbing in this transport check.
    s.offline_policy = None
    monkeypatch.setattr(Scheduler, '_build_kv_connector_meta', lambda self,c,o: ControlMetadata())
    out = NS(num_scheduled_tokens={'p':4096}, scheduled_spec_decode_tokens={})
    meta = s._build_kv_connector_meta(s.connector, out)
    assert meta.logical_prefill_groups == s._logical_prefill_groups
    assert meta.logical_prefill_step == s.current_step
    # Metadata construction can strip a refresh/control owner before forward.
    out.num_scheduled_tokens = {}
    meta = s._build_kv_connector_meta(s.connector, out)
    assert meta.logical_prefill_groups == {} and meta.logical_prefill_step == s.current_step
    out.num_scheduled_tokens = {'p':3072}
    with pytest.raises(RuntimeError, match='single-owner admission'):
        s._build_kv_connector_meta(s.connector, out)
    out.num_scheduled_tokens = {'p':4096, 'q':1}
    with pytest.raises(RuntimeError, match='single-owner admission'):
        s._build_kv_connector_meta(s.connector, out)
    out.num_scheduled_tokens = {'p':4096}
    out.scheduled_spec_decode_tokens = {'p':[1,2,3,4]}
    with pytest.raises(RuntimeError, match='single-owner admission'):
        s._build_kv_connector_meta(s.connector, out)


@pytest.mark.parametrize('maximum,history,budget,expected', [
    (4,8192,16424,8192),(4,40992,16424,6144),
    (4,45088,16424,2048),(4,8192,6144,6144),(7,32800,16424,14336),
    (8,32800,16424,14336),(8,8192,16424,16384),
    (8,8192,4095,2048),
])
def test_maximum_keeps_largest_legal_original_prefix(monkeypatch,tmp_path,maximum,history,budget,expected):
    s,r = setup(monkeypatch,tmp_path,history=history)
    s.prefill_physical_max_chunks = maximum
    assert reserve(s,r,budget) == expected
    group = s._logical_prefill_groups.get('p')
    assert bool(group) == (expected >= 4096)
    if group:
        assert group.count == expected
        assert [span.history_begin for span in group.spans] == list(range(history,history+expected,2048))


@pytest.mark.parametrize('ready,expected',[(256,8192),(192,6144),(128,4096),(64,2048),(63,0)])
def test_archive_credit_chooses_smaller_group_before_single_final_admission(monkeypatch,tmp_path,ready,expected):
    from vllm_kvmem.archive_admission import ArchiveAdmission,ArchiveCapacity
    s,r = setup(monkeypatch,tmp_path)
    s.prefill_physical_max_chunks = 4
    s.archive_admission = gate = ArchiveAdmission(32,1024)
    gate.update(1,{'p':ArchiveCapacity(8192,ready)},{'p'})
    assert reserve(s,r) == expected
    assert gate.deferred == int(expected == 0)
    assert bool(s._logical_prefill_groups) == (expected >= 4096)
