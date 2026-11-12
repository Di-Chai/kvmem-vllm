"""KVMEM_HELD_P_CAPTURE: a turn's P capture overlaps a peer batch's forward
instead of stalling the batch that carries the turn's final prefill piece."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.session_control import SessionCapture, held_p_capture
from test_session_worker import worker_fixture


# -- scheduler: the final piece leaves this batch and is held ----------------

def scheduler(**extra):
    from vllm_kvmem.scheduler import KVMemScheduler
    s = KVMemScheduler.__new__(KVMemScheduler)
    s.requests = {'r': NS(request_id='r', num_prompt_tokens=300, num_computed_tokens=256, spec_token_ids=[])}
    s.transactions = {}
    s.__dict__.update(extra)
    return s


def batch(**scheduled):
    return NS(num_scheduled_tokens=dict(scheduled), total_num_scheduled_tokens=sum(scheduled.values()),
              scheduled_spec_decode_tokens={})


def connector():
    commands = NS(captures=[SessionCapture('x', 1, 128), SessionCapture('r', 1, 256)])
    return NS(sessions=NS(commands=commands), held_captures=set(), publication_blocked=set())


def test_final_piece_with_peers_is_held_and_its_command_marked():
    s, c, out = scheduler(), connector(), batch(r=44, d=5)
    s._hold_for_capture(c, out, 'r', 44, 1)
    assert out.num_scheduled_tokens == {'d': 5} and out.total_num_scheduled_tokens == 5
    assert c.sessions.commands.captures[1] == SessionCapture('r', 1, 256, held=True)
    assert c.sessions.commands.captures[0].held is False
    assert c.held_captures == {'r'} and c.publication_blocked == {'r'}
    assert s.held_p_captures == 1


@pytest.mark.parametrize('case', ['alone', 'refresh', 'coalesced', 'popped'])
def test_otherwise_the_piece_runs_now(case):
    s, c = scheduler(), connector()
    out = batch(r=44) if case == 'alone' else batch(r=44, d=5)
    if case == 'refresh': s.transactions['r'] = object()
    elif case == 'coalesced': s._logical_prefill_groups = {'r': object()}
    elif case == 'popped': out = batch(d=5)
    before = dict(out.num_scheduled_tokens)
    s._hold_for_capture(c, out, 'r', 44, 1)
    assert out.num_scheduled_tokens == before
    assert not c.held_captures and not c.publication_blocked
    assert not any(command.held for command in c.sessions.commands.captures)


def test_a_mismatched_command_fails_loudly():
    s, c, out = scheduler(), connector(), batch(r=44, d=5)
    with pytest.raises(RuntimeError, match='does not match'):
        s._hold_for_capture(c, out, 'r', 44, 0)


# -- connector: held until the worker reports the capture ----------------------

def scheduler_connector():
    from vllm_kvmem.connector import KVMemConnector
    c = KVMemConnector.__new__(KVMemConnector)
    for name in ('tp_feedback', 'scheduler_step_feedback', 'scheduler_restore_feedback', 'history_admission',
                 'active_residency', 'sessions', 'publication_live'):
        setattr(c, name, None)
    c.tiering_queue = False
    c.publication_sequence, c.pending, c.cancelled = 0, {}, set()
    c.publication_blocked, c.held_captures = set(), {'r'}
    return c


def feed(c, *, blocked=(), done=()):
    from vllm_kvmem.connector import ControlCompletions
    meta = ControlCompletions([], {})
    meta.publication_sequence = c.publication_sequence + 1
    meta.publication_blocked = set(blocked)
    meta.held_captures_done = list(done)
    c.update_connector_output(NS(kv_connector_worker_meta=meta, finished_sending=None, finished_recving=None))


def test_publication_feedback_cannot_release_a_held_piece_before_its_capture():
    c = scheduler_connector()
    feed(c, blocked={'big'})
    assert c.publication_blocked == {'big', 'r'} and c.held_captures == {'r'}
    feed(c, done=['r'])
    assert c.publication_blocked == set() and c.held_captures == set()


# -- worker: the same capture, after the launch --------------------------------

def held_worker(tmp_path):
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    worker.options.session_capture_decode = True          # this fixture already owns a P
    worker.captures['a'] = SessionCapture('a', 1, 8, held=True)
    return worker, binding, authority, archive


def test_held_capture_runs_once_the_request_is_outside_the_batch(tmp_path):
    worker, binding, authority, archive = held_worker(tmp_path)
    previous = worker.pool.entries[binding.namespace]
    worker.finish_held_captures({'a': 4})                   # its own batch: never here
    assert 'a' in worker.captures and worker.take_held_done() == []
    worker.finish_held_captures({'b': 5})
    assert 'a' not in worker.captures and worker.take_held_done() == ['a'] and worker.take_held_done() == []
    entry = worker.pool.entries[binding.namespace]
    assert entry is not previous and entry.position == 8 and worker.held_p_captures == 1
    assert worker.close() and authority.close() and archive.close()


def test_unconfirmed_publication_keeps_the_piece_held(tmp_path):
    worker, binding, authority, archive = held_worker(tmp_path)
    worker.runtime.drain_requests = lambda ids, *, wait: False
    worker.finish_held_captures({'b': 5})
    assert 'a' in worker.captures and worker.take_held_done() == []
    del worker.runtime.drain_requests
    worker.finish_held_captures({'b': 5})
    assert worker.take_held_done() == ['a']
    assert worker.close() and authority.close() and archive.close()


def test_a_cancelled_request_drops_its_held_capture(tmp_path):
    worker, binding, authority, archive = held_worker(tmp_path)
    previous = worker.pool.entries[binding.namespace]
    worker.finish_held_captures({'b': 5}, finished={'a'})
    assert 'a' not in worker.captures and worker.take_held_done() == ['a']
    assert worker.pool.entries[binding.namespace] is previous
    assert worker.close() and authority.close() and archive.close()


def test_unheld_captures_still_wait_for_their_own_batch(tmp_path):
    worker, binding, authority, archive = held_worker(tmp_path)
    worker.captures['a'] = SessionCapture('a', 1, 8)
    worker.finish_held_captures({'b': 5})
    assert 'a' in worker.captures and worker.take_held_done() == []
    assert worker.close() and authority.close() and archive.close()


def test_flag_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_HELD_P_CAPTURE', raising=False)
    assert not held_p_capture()
    monkeypatch.setenv('KVMEM_HELD_P_CAPTURE', '1')
    assert held_p_capture() == 1
    monkeypatch.setenv('KVMEM_HELD_P_CAPTURE', '2')
    assert held_p_capture() == 2
    monkeypatch.setenv('KVMEM_HELD_P_CAPTURE', 'true')
    with pytest.raises(ValueError, match='0, 1 or 2'):
        held_p_capture()


def test_exactness_mode_also_holds_a_piece_without_peers():
    s, c, out = scheduler(), connector(), batch(r=44)
    s._hold_for_capture(c, out, 'r', 44, 1, alone=True)
    assert out.num_scheduled_tokens == {} and out.total_num_scheduled_tokens == 0
    assert c.held_captures == {'r'} and c.sessions.commands.captures[1].held


def test_a_released_piece_is_never_captured_and_held_again():
    s, c, out = scheduler(), connector(), batch(r=44, d=5)
    s._hold_for_capture(c, out, 'r', 44, 1)
    request = s.requests['r']
    assert s._held_capture_taken(request)          # released at 256: the capture exists
    assert s._held_capture_taken(request)          # until the piece actually runs
    request.num_computed_tokens = 300
    assert not s._held_capture_taken(request) and s._held_capture_positions == {}
    assert not s._held_capture_taken(NS(request_id='other', num_computed_tokens=0))
