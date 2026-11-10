"""KVMEM_CHECKPOINT_LAZY_EVIDENCE: same checkpoint evidence row, hashed off the engine step."""
import hashlib
import json
import threading
import time

import pytest

from vllm_kvmem.deferred_evidence import DeferredEvidence
from vllm_kvmem.layerwise_checkpoints import (LayerwisePrefix, _list_sha256, emit_prefix_evidence,
                                              lazy_evidence)
from vllm_kvmem.layerwise_query import LayerwiseQuery


def snapshot(layers=5, rows=40):
    maps = tuple(tuple((7*i+j, 100+j, 32*j, 32, j, 'text', i) for j in range(rows)) for i in range(layers))
    histories = tuple((tuple((j, 32*j, 1e-3*j*(i+1), 0.5) for j in range(rows)), 32*rows, 1.25*i, i)
                      for i in range(layers))
    return LayerwisePrefix(32*rows, tuple(f'model.layers.{i}' for i in range(layers)), maps, histories)


def query(tmp_path, monkeypatch, delay_ms=None):
    q = LayerwiseQuery.__new__(LayerwiseQuery)
    q.path, q.mode, q.optimized, q.buffered_events = tmp_path/'layerwise_query.jsonl', 'sequential', True, []
    q._emit_lock = threading.RLock()
    q.deferred_writer = None
    if delay_ms is not None:
        monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE_DELAY_MS', str(delay_ms))
        q.deferred_writer = DeferredEvidence(q)
    return q


def lines(q):
    return q.path.read_text().splitlines()


def strip_time(line):
    row = json.loads(line)
    return json.dumps({k: v for k, v in row.items() if k not in ('at', 'monotonic_at')})


@pytest.mark.parametrize('items', [[], [()], [(1, 'a', 2.5)], [((1, 2), (3, 4)), (), ((5, None, 'é'),)]])
@pytest.mark.parametrize('yield_gil', [False, True])
def test_itemwise_digest_equals_whole_list_digest(items, yield_gil):
    assert _list_sha256(iter(items), yield_gil) == hashlib.sha256(json.dumps(items).encode()).hexdigest()


@pytest.mark.parametrize('layers', [0, 1, 5])
def test_resolver_equals_evidence(layers):
    s = snapshot(layers)
    for yield_gil in (False, True):
        resolved = s.evidence_resolver()(yield_gil)
        assert resolved == s.evidence() and list(resolved) == list(s.evidence())


def test_flag_is_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', raising=False)
    assert not lazy_evidence()
    monkeypatch.setenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', 'yes')
    with pytest.raises(ValueError, match='KVMEM_CHECKPOINT_LAZY_EVIDENCE'):
        lazy_evidence()


def reference_line(tmp_path, monkeypatch, s):
    monkeypatch.setenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', '0')
    q = query(tmp_path/'reference', monkeypatch)
    emit_prefix_evidence(q, 'pressure_prefix_captured', s, request_id='r',
                         actual_method='accumulated_attention_per_layer', fallback_reason=None)
    return strip_time(lines(q)[0])


@pytest.mark.parametrize('writer', ['none', 'background', 'synchronous'])
def test_lazy_row_is_identical_and_keeps_its_position(tmp_path, monkeypatch, writer):
    s = snapshot()
    expected = reference_line(tmp_path, monkeypatch, s)
    monkeypatch.setenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', '1')
    q = query(tmp_path/'lazy', monkeypatch, dict(none=None, background=0, synchronous=2000)[writer])
    q.emit('first')
    q.emit('layer_selected', layer=0)          # Buffered before the checkpoint row.
    before = time.time()
    emit_prefix_evidence(q, 'pressure_prefix_captured', s, request_id='r',
                         actual_method='accumulated_attention_per_layer', fallback_reason=None)
    after = time.time()
    if writer == 'background':
        deadline = time.monotonic() + 5
        while q.deferred_writer.written < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(lines(q)) == 3
    elif writer == 'synchronous':
        assert len(lines(q)) == 1              # Not hashed or written by the capture call.
    q.emit('last')
    written = lines(q)
    assert [json.loads(line)['event'] for line in written] == [
        'first', 'layer_selected', 'pressure_prefix_captured', 'last']
    assert strip_time(written[2]) == expected
    row = json.loads(written[2])
    assert before <= row['at'] <= after        # Timestamp of the capture, not of the write.
    assert written[2] == json.dumps(row, allow_nan=False)


def test_resolver_runs_once_per_written_row_and_not_in_the_capture_call(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', '1')
    q = query(tmp_path, monkeypatch, 2000)
    calls = []
    def resolve(yield_gil=False):
        calls.append(yield_gil)
        return dict(map_sha256='m')
    q.defer_lazy('pressure_prefix_restored', resolve, request_id='r')
    assert not calls
    q.deferred_writer.flush()
    q.deferred_writer.flush()
    assert calls == [False]
    assert [json.loads(line) for line in lines(q)][0]['map_sha256'] == 'm'


def test_resolver_does_not_keep_the_checkpoint_alive():
    import gc
    import weakref
    class Hold:
        pass
    hold = Hold()
    s = LayerwisePrefix(1, ('a',), ((),), ((),), gc_hold=hold)
    resolve, alive = s.evidence_resolver(), weakref.ref(hold)
    del s, hold
    gc.collect()
    assert alive() is None and resolve()['accepted_history'] == 1
