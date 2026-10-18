import json
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from test_phase_scheduler import scheduler as phase_scheduler, request, NATIVE_RESERVE
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_kvmem.prefill_coalescing import OfflinePrefillCoalescer


SHAPE = dict(decode_requests=3, prefill_requests=1,
             decode_context_bucket=65536, prefill_context_bucket=16384)


def setup(monkeypatch, tmp_path, *, enabled=True):
    s = phase_scheduler(monkeypatch, tmp_path)
    s.phase_policy = s._phase_decision = None
    s.prefill_coalescer = OfflinePrefillCoalescer(2048, 4096, 32, clock=lambda: 10.) if enabled else None
    s._coalescing_max_tokens = 4096
    s.running = [request('p', computed=16384, prompt=100000, output=0)] + [
        request('d'+str(i), computed=65536, prompt=65536) for i in range(3)]
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '2048')
    s._prepare_coalescing()
    return s


def seed(s):
    s.prefill_coalescer.observe([dict(sequence=i, submitted_at=10., model_ms=100.,
        prefill_tokens=2048, ordinary_prefill_only=True, **SHAPE) for i in range(1, 4)])


def reserve(s, n=10000):
    p = s.running[0]
    return s._reserve_prefill_lookahead(p, p.num_computed_tokens, n)


def test_first_three_feedback_then_one_grant_per_step_and_decode_unchanged(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    assert reserve(s) == 2048
    seed(s); s._prepare_coalescing()
    assert reserve(s) == reserve(s) == 4096
    assert s.prefill_coalescer.snapshot()['calibration_grants'] == 1
    for d in s.running[1:]:
        assert s._reserve_prefill_lookahead(d, 65536, 5) == 5
    output = SimpleNamespace(num_scheduled_tokens={'p': 4096, 'd0': 5, 'd1': 5, 'd2': 5})
    s._reconcile_coalescing(output); s._record_coalescing(output)
    row = json.loads((tmp_path/'coalescing_decisions.jsonl').read_text())
    assert row['public_prefill_tokens'] == {'p': 4096}
    assert len(row['eligible_decode_ids']) == len(row['scheduled_decode_tokens']) == 3
    assert not row['atomic_prefill_tokens']


@pytest.mark.parametrize('kind', ['waiting', 'inflight', 'transaction', 'prepared', 'two_prefills'])
def test_unknown_or_unready_composition_keeps_baseline_without_blocking_decode(monkeypatch, tmp_path, kind):
    s = setup(monkeypatch, tmp_path); seed(s)
    if kind == 'waiting': s.waiting = [request('new', prompt=10000, computed=0)]
    if kind == 'inflight': s.running[1].num_in_flight_tokens = 1
    if kind == 'transaction': s.transactions['d0'] = object()
    if kind == 'prepared': s.prepared_refreshes['p'] = (16384, 5000)
    if kind == 'two_prefills': s.running[1] = request('p2', computed=0, prompt=10000)
    s._prepare_coalescing()
    assert reserve(s) == 2048
    d = s.running[-1]
    assert s._reserve_prefill_lookahead(d, d.num_computed_tokens, 5) == 5
    assert not s._coalescing_authorizations


def test_native_batch_and_plan_boundaries_do_not_expand_to_authorized_cap(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path); seed(s)
    assert reserve(s, 3072) == 3072
    s._reconcile_coalescing(SimpleNamespace(num_scheduled_tokens={'p': 3072}))
    state = s.prefill_coalescer.snapshot()
    assert state['discarded_grants'] == 1 and not state['released_grants']
    assert state['banks'][0]['calibration_grants'] == 1
    # A stripped asynchronous restore has no public forward and refunds once.
    s._prepare_coalescing(); assert reserve(s) == 4096
    s._reconcile_coalescing(SimpleNamespace(num_scheduled_tokens={}))
    s._reconcile_coalescing(SimpleNamespace(num_scheduled_tokens={}))
    assert s.prefill_coalescer.snapshot()['released_grants'] == 1


def test_context_bucket_edge_does_not_reuse_old_fit(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path); seed(s)
    s.running[0].num_computed_tokens = 30016
    s._prepare_coalescing()
    assert reserve(s) == 2048
    assert not s._coalescing_authorizations
    assert 'bucket edge' in s._coalescing_reason


@pytest.mark.parametrize('enabled', [False, True])
def test_atomic_query_and_padded_public_tail_retain_native_mtp_shape(monkeypatch, tmp_path, enabled):
    s = setup(monkeypatch, tmp_path, enabled=enabled)
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', NATIVE_RESERVE)
    s.num_prefill_lookahead = 4
    q = request('q', computed=19008, prompt=23024, output=0,
        plan={'kvmem_query_span': [19028, 23022]})
    s.running = [q]; s._prepare_coalescing()
    assert s._reserve_prefill_lookahead(q, 19008, 4016) == 4014
    assert s._cost_admitted['q']['atomic_range'] == (19008, 23022)
    assert not s._coalescing_authorizations
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self, r, h, n: n)
    s.num_spec_tokens = 4
    tail = request('tail', computed=64, prompt=65, output=0)
    s.running = [tail]; s._prepare_coalescing()
    assert s._reserve_prefill_lookahead(tail, 64, 5) == 5
    assert not s._coalescing_authorizations


def test_disabled_has_same_labels_and_baseline_but_no_logs(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path, enabled=False)
    assert reserve(s) == 2048
    assert s._cost_admitted['p']['history'] == 16384
    s._record_coalescing(SimpleNamespace(num_scheduled_tokens={'p': 2048}))
    assert not (tmp_path/'coalescing_decisions.jsonl').exists()


def test_steady_decode_does_not_open_decision_log(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    s.running = s.running[1:]; s._prepare_coalescing()
    output = SimpleNamespace(num_scheduled_tokens={})
    for r in s.running:
        output.num_scheduled_tokens[r.request_id] = s._reserve_prefill_lookahead(r, 65536, 5)
    s._record_coalescing(output)
    assert not (tmp_path/'coalescing_decisions.jsonl').exists()


def test_callback_fans_out_all_classified_rows_but_service_cost_skips_query(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    service, coalescing = [], []
    s.offline_policy.costs = SimpleNamespace(observe_steps=service.extend)
    s.prefill_coalescer.observe = coalescing.extend
    rows = [dict(prefill_tokens=2048, ordinary_prefill_only=True),
            dict(prefill_tokens=4014, ordinary_prefill_only=False), dict(prefill_tokens=0)]
    s._observe_offline_steps(rows)
    assert coalescing == rows and service == [rows[0], rows[2]]
