"""KVMEM_PREFILL_COALESCE_WITH_DECODE: logical prefill owners beside decode rows."""
from types import SimpleNamespace as NS

import pytest

from test_physical_prefill_scheduler import setup, reserve, request


def mixed(monkeypatch, tmp_path, *, rows=448, chunks=8, enabled=True):
    from vllm.v1.request import RequestStatus
    s, r = setup(monkeypatch, tmp_path)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '1024')
    monkeypatch.setenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', '1' if enabled else '0')
    s.prefill_physical_multi_owner = True
    s.prefill_coalesce_with_decode = enabled
    s._physical_native_rows, s.prefill_physical_max_chunks = rows, chunks
    s.num_spec_tokens = 4
    peer = request('d', computed=262144, prompt=262144, output=3)
    peer.status = RequestStatus.RUNNING
    peer.spec_token_ids = [5, 6, 7, 8]
    s.running.append(peer)
    return s, r, peer


def test_disabled_keeps_original_single_chunk_path(monkeypatch, tmp_path):
    s, r, _ = mixed(monkeypatch, tmp_path, enabled=False)
    s._prepare_logical_prefill()
    assert reserve(s, r) == 1024 and not s._logical_prefill_owners


def test_decode_peer_keeps_owner_coalesced_and_reserves_its_rows(monkeypatch, tmp_path):
    s, r, peer = mixed(monkeypatch, tmp_path)
    s._prepare_logical_prefill()
    assert set(s._logical_prefill_owners) == {'p'} and reserve(s, r) == 8192
    assert 'd' not in s._logical_prefill_owners


@pytest.mark.parametrize('rows,expected', [(256, 7), (261, 8), (260, 7), (37, 1)])
def test_reservation_never_oversubscribes_the_native_bank(monkeypatch, tmp_path, rows, expected):
    s, r, peer = mixed(monkeypatch, tmp_path, rows=rows)
    s._prepare_logical_prefill()
    # 1 + 4 speculative rows for the decode peer, 32 rows per 1K logical chunk.
    tokens = reserve(s, r)
    assert tokens == (expected * 1024 if expected >= 2 else 1024)
    if expected >= 2:
        assert sum(len(g.spans) * 32 for g in s._logical_prefill_groups.values()) + 5 <= rows


def test_several_decode_peers_and_owners_share_remaining_rows(monkeypatch, tmp_path):
    from vllm.v1.request import RequestStatus
    s, r, _ = mixed(monkeypatch, tmp_path, rows=576, chunks=4)
    for i in range(3):
        d = request('d'+str(i), computed=300000, prompt=262144, output=2)
        d.status = RequestStatus.RUNNING
        d.spec_token_ids = []           # Reserve still covers the configured MTP depth.
        s.running.append(d)
    s.running.append(request('q', computed=8192, prompt=262144, output=0))
    s._prepare_logical_prefill()
    # (576 - 4*5) // (2 owners * 32) = 8 -> capped at 4 chunks each.
    assert set(s._logical_prefill_owners) == {'p', 'q'}
    assert reserve(s, r) == 4096


@pytest.mark.parametrize('kind', ['query', 'waiting_decode', 'unknown_plan'])
def test_other_peers_still_keep_the_original_path(monkeypatch, tmp_path, kind):
    s, r, peer = mixed(monkeypatch, tmp_path)
    if kind == 'query':
        q = request('q', computed=8192, prompt=262144, output=0,
                    plan={'kvmem_reselect': 'force', 'kvmem_query_span': [8192, 8224]})
        s.running.append(q)
    if kind == 'waiting_decode':
        peer.status = 'waiting'
    if kind == 'unknown_plan':
        peer.kv_transfer_params = {}
    s._prepare_logical_prefill()
    assert reserve(s, r) == 1024 and not s._logical_prefill_owners


def test_only_decode_requests_create_no_owner(monkeypatch, tmp_path):
    s, r, peer = mixed(monkeypatch, tmp_path)
    s.running = [peer]
    s._prepare_logical_prefill()
    assert not s._logical_prefill_owners


def test_scheduler_option_validation(monkeypatch):
    monkeypatch.setenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', 'yes')
    from vllm_kvmem.multiowner_prefill import coalesce_with_decode
    with pytest.raises(ValueError, match='KVMEM_PREFILL_COALESCE_WITH_DECODE'):
        coalesce_with_decode()


def batch(prefilling):
    return NS(num_reqs=3, req_ids=('r', 'd', 's'), is_prefilling_np=prefilling,
              num_computed_tokens_np=(0, 0, 0), query_start_loc_np=(0, 4096, 4101, 8197))


@pytest.mark.parametrize('enabled', [False, True])
def test_worker_cohort_check_allows_decode_rows_only_when_enabled(monkeypatch, enabled):
    from vllm_kvmem.multiowner_prefill import owners_prefilling
    monkeypatch.setenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', '1' if enabled else '0')
    assert owners_prefilling(batch((True, False, True)), {'r', 's'}) == enabled
    assert owners_prefilling(batch((True, True, True)), {'r', 's'})
    # A logical owner itself must always be a prompt row.
    assert not owners_prefilling(batch((True, True, False)), {'r', 's'})


def test_capture_owner_plans_with_decode_row(monkeypatch, tmp_path):
    from test_multiowner_prefill import owners
    from vllm_kvmem.multiowner_prefill import capture_owner_plans
    with owners(tmp_path) as (r, p, a, b):
        r.runner._kvmem_logical_prefill_step = 7
        mixed_batch = NS(num_reqs=3, req_ids=('r', 'd', 's'), is_prefilling_np=(True, False, True),
            num_computed_tokens_np=(0, 99, 0), query_start_loc_np=(0, 4096, 4101, 8197))
        inputs = NS(logical_groups={'r': a.group, 's': b.group}, logical_slices={'r': a.parts, 's': b.parts})
        monkeypatch.setenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', '0')
        with pytest.raises(RuntimeError, match='owner cohort'):
            capture_owner_plans(r, inputs, mixed_batch)
        monkeypatch.setenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', '1')
        assert set(capture_owner_plans(r, inputs, mixed_batch)) == {'r', 's'}


def test_default_decode_coalescing_requires_a_physical_multi_owner_profile(monkeypatch):
    from vllm_kvmem.multiowner_prefill import coalesce_with_decode
    monkeypatch.setenv('KVMEM_LEGACY_DEFAULTS', '0')
    monkeypatch.delenv('KVMEM_PREFILL_COALESCE_WITH_DECODE', raising=False)
    assert not coalesce_with_decode(available=False)
    assert coalesce_with_decode(available=True)
