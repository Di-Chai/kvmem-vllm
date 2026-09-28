from types import SimpleNamespace as NS
import time

import pytest
import torch

from vllm_kvmem.session_control import SessionBinding, SessionCapture, SessionCommands, SessionEviction
from vllm_kvmem.session_worker import SessionWorker, verify_preserved_authority
from vllm_kvmem.window_batch import WindowBatchAdapter
from test_authority import setup, payload, drain
from test_session_state import fixture


def worker_fixture(tmp_path):
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(authority)
    adapter, conv, ssm, cache, _ = fixture()
    runner = adapter.runner
    runner.device = torch.device('cpu')
    runner._kvmem_window_adapter = WindowBatchAdapter((0,), 16)
    runner._kvmem_capture_runtime = NS(authorities={'a': authority}, names=('l0', 'l1'),
        accepted={'a': 8}, rejected={'a': 0}, last_steps={'a': []}, completed=[],
        drain=lambda **_: True, _write_archive_report=lambda: None)
    options = NS(memory=authority.memory.config, session_capacity=2, session_checkpoint_bytes=4096,
                 verify_archive_on_finish=True, packed_assembly=True, direct_pinned_reads=True)
    worker = SessionWorker(runner, options)
    worker.model_adapter = adapter
    binding = SessionBinding(('scope', 'session'), 1, 'a', tuple(range(11)))
    worker.receive(SessionCommands([binding], [SessionCapture('a', 1, 8)]))
    worker.before_attention(NS(req_ids=['a']))
    return worker, binding, authority, archive, (conv, ssm, cache)


def wait_retirement(worker, request_id):
    deadline = time.monotonic() + 5
    while not worker.retirement_done(request_id):
        if time.monotonic() > deadline:
            pytest.fail('Worker checkpoint retirement stalled')
        time.sleep(.001)


def test_worker_exports_rewinds_and_imports_before_next_native_attention(tmp_path, monkeypatch):
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    worker, binding, authority, archive, (conv, ssm, cache) = worker_fixture(tmp_path)
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    assert worker.detach('a', authority, {'request_id': 'a'})
    del worker.runtime.authorities['a']
    wait_retirement(worker, 'a')
    notice, = worker.notices()
    assert (notice.kind, notice.history, notice.window) == ('retired', 8, 8)
    assert worker.runtime.completed[0]['sealed_records_verified'] == 3
    assert authority.history_end == 8
    calls = []
    class Assembler:
        def assemble(self, owner, plan):
            assert owner is authority and all(not r.skip for r in plan.remaps)
            calls.append('target-rebuild')
            return NS(event=NS(synchronize=lambda: calls.append('target-fence')), query=lambda: True)
        def close(self):
            calls.append('close-staging')
            return True
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: Assembler())
    before = conv.clone(), ssm.clone(), cache.clone()
    del worker.runner.req_states.req_id_to_index['a']
    del worker.runner._kvmem_request_generations['a']
    worker.runner.req_states.req_id_to_index['new'] = 2
    worker.runner._kvmem_request_generations['new'] = 3
    worker.runner.req_states.prefill_len.np[2] = 14
    conv[1:3].fill_(-12); ssm[1:3].fill_(-12); cache[1].fill_(-12)
    new = SessionBinding(binding.namespace, 1, 'new', tuple(range(14)), 8, 8)
    worker.receive(SessionCommands([new], [SessionCapture('new', 1, 8)]))
    worker.before_attention(NS(req_ids=['new']))
    assert calls == ['target-rebuild', 'target-fence', 'close-staging']
    assert worker.runtime.authorities['new'] is authority
    assert worker.runtime.accepted['new'] == 8 and worker.restored_tokens == 8
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(cache[8], before[2][1])
    assert torch.equal(conv[3:5], before[0][3:5])
    assert worker.notices()[0].kind == 'restored'
    assert worker.pool.entries[binding.namespace].request_id == 'new'
    assert worker.detach('new', authority, {})
    del worker.runtime.authorities['new']
    wait_retirement(worker, 'new')
    worker.notices()
    worker.evict([SessionEviction(binding.namespace, 1)])
    assert worker.notices()[0].kind == 'evicted'
    assert not worker.pool.entries and not worker.identities
    assert archive.close()


def test_failed_actual_archive_integrity_never_detaches_or_acknowledges_session(tmp_path):
    worker, _, authority, archive, _ = worker_fixture(tmp_path)
    record = next(iter(authority.records.values()))
    record.digest = 'corrupt-expected-digest'
    with pytest.raises(RuntimeError, match='checksum differs'):
        worker.detach('a', authority, {})
    assert not worker.retiring and not worker.notices()
    assert 'a' in worker.runtime.authorities
    assert worker.close() and authority.close() and archive.close()


def test_failed_import_reclaims_authority_only_after_partial_restore_cleanup(tmp_path, monkeypatch):
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    assert worker.detach('a', authority, {})
    del worker.runtime.authorities['a']
    wait_retirement(worker, 'a'); worker.notices()
    worker.runner.req_states.req_id_to_index['new'] = 2
    worker.runner._kvmem_request_generations['new'] = 3
    worker.receive(SessionCommands([SessionBinding(binding.namespace, 1, 'new', tuple(range(11)), 8, 8)]))
    class Failed:
        def assemble(self, *_):
            raise OSError('injected import I/O failure')
        def close(self):
            return True
    monkeypatch.setattr(NativeWindowAssembler, 'from_runner', lambda *a, **k: Failed())
    with pytest.raises(OSError, match='import I/O'):
        worker.before_attention(NS(req_ids=['new']))
    assert not worker.notices() and 'new' not in worker.runtime.authorities
    assert worker.close() and archive.close()


def test_unretained_short_turn_emits_empty_checkpoint_without_phantom_cache(tmp_path):
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    worker.pool.drop_live(worker.pool.entries[binding.namespace])
    assert not worker.detach('a', authority, {})
    assert authority.close()
    worker.ordinary_retired('a')
    notice, = worker.notices()
    assert notice.kind == 'retired' and notice.history == notice.window == 0
    assert not worker.bindings and not worker.identities and archive.close()
