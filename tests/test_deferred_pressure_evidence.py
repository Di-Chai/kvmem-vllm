"""KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE: same bytes, timestamps and row order."""
import json
import threading
import time

import pytest

from vllm_kvmem.deferred_evidence import DeferredEvidence, serialize
from vllm_kvmem.layerwise_query import LayerwiseQuery


def record(layers=16, candidates=300):
    return dict(event='pressure_selected', mode='sequential', at=1.5, monotonic_at=2.25,
                request_id='ré', accepted_history=4096,
                layers=[dict(layer=i, layer_name=f'model.layers.{4*i+3}', candidate_block_count=candidates,
                             candidate_starts=list(range(0, 32*candidates, 32)),
                             selected_starts=list(range(0, 32*candidates, 64)), selected_tokens=1e-7*i,
                             fallback_reason=None, nested={'a': [1.0, -0.0, 3]}, flag=bool(i % 2))
                        for i in range(layers)],
                fallback_reason=None, selected_tokens=36864)


@pytest.mark.parametrize('layers', [0, 1, 2, 16])
def test_chunked_serialization_is_byte_identical(layers):
    r = record(layers)
    assert serialize(r, yield_gil=True) == json.dumps(r, allow_nan=False) + '\n'
    assert serialize(r) == json.dumps(r, allow_nan=False) + '\n'


def test_sentinel_collision_falls_back_to_plain():
    r = record(2)
    r['request_id'] = '\x00kvmem-deferred-layers\x00'
    assert serialize(r, yield_gil=True) == json.dumps(r, allow_nan=False) + '\n'


def test_nan_is_still_rejected():
    r = record(2)
    r['layers'][1]['selected_tokens'] = float('nan')
    with pytest.raises(ValueError):
        serialize(r, yield_gil=True)


def query(tmp_path, monkeypatch, delay_ms):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE_DELAY_MS', str(delay_ms))
    q = LayerwiseQuery.__new__(LayerwiseQuery)
    q.path, q.mode, q.optimized, q.buffered_events = tmp_path/'layerwise_query.jsonl', 'sequential', True, []
    q._emit_lock = threading.RLock()
    q.deferred_writer = None
    q.deferred_writer = DeferredEvidence(q)
    return q


def rows(q):
    return [json.loads(line) for line in q.path.read_text().splitlines()]


def wait_written(q, count, timeout=5.):
    deadline = time.monotonic() + timeout
    while q.deferred_writer.written < count and time.monotonic() < deadline:
        time.sleep(.005)


@pytest.mark.parametrize('writer_first', [True, False])
def test_rows_keep_their_original_order(tmp_path, monkeypatch, writer_first):
    q = query(tmp_path, monkeypatch, 0 if writer_first else 1000)
    q.emit('first', n=1)
    q.emit('layer_selected', layer=0)          # Buffered before the deferred row.
    before = time.time()
    q.defer('pressure_selected', **{k: v for k, v in record(3).items()
                                    if k not in ('event', 'mode', 'at', 'monotonic_at')})
    after = time.time()
    q.emit('layer_selected', layer=1)          # Buffered after it.
    if writer_first:
        wait_written(q, 2)
        assert [r['event'] for r in rows(q)] == ['first', 'layer_selected', 'pressure_selected']
    q.emit('last', n=2)
    events = [(r['event'], r.get('layer')) for r in rows(q)]
    assert events == [('first', None), ('layer_selected', 0), ('pressure_selected', None),
                      ('layer_selected', 1), ('last', None)]
    pressure = rows(q)[2]
    assert before <= pressure['at'] <= after       # Timestamp of the original call.
    expected = dict(record(3), at=pressure['at'], monotonic_at=pressure['monotonic_at'])
    assert pressure == expected
    time.sleep(.05 if writer_first else 0)
    assert len(rows(q)) == 5                        # Writer never duplicates a row.


def test_deferred_line_bytes_equal_plain_emit(tmp_path, monkeypatch):
    q = query(tmp_path, monkeypatch, 0)
    fields = {k: v for k, v in record(16).items() if k not in ('event', 'mode', 'at', 'monotonic_at')}
    q.defer('pressure_selected', **fields)
    wait_written(q, 1)
    line = q.path.read_text()
    row = json.loads(line)
    assert line == json.dumps(row, allow_nan=False) + '\n'


def test_writer_failure_surfaces_at_next_emit(tmp_path, monkeypatch):
    q = query(tmp_path, monkeypatch, 0)
    bad = record(2)
    bad['layers'][0]['selected_tokens'] = float('inf')
    q.defer('pressure_selected', layers=bad['layers'])
    deadline = time.monotonic() + 5
    while q.deferred_writer.error is None and time.monotonic() < deadline:
        time.sleep(.005)
    with pytest.raises(RuntimeError, match='Deferred pressure evidence writer failed'):
        q.emit('next')


def test_flush_writes_everything_pending(tmp_path, monkeypatch):
    q = query(tmp_path, monkeypatch, 2000)
    q.defer('pressure_selected', layers=record(2)['layers'])
    q.defer('pressure_rope', request_id='r')
    q.deferred_writer.flush()
    assert [r['event'] for r in rows(q)] == ['pressure_selected', 'pressure_rope']


def test_invalid_delay_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='DELAY_MS'):
        query(tmp_path, monkeypatch, 'x')
