"""KVMEM_NATIVE_PUBLISH_SELECTION: one native call publishes the same host mapping."""
import random

import pytest

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory, _native_publish_selection


class Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done


def run(seed, monkeypatch, flag):
    if not hasattr(_native, 'publish_selection'):
        pytest.skip('native extension lacks publish_selection')
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_SELECTION', flag)
    rng = random.Random(seed)
    memory = RequestMemory('r', KVMemConfig(block_tokens=16, select_budget=rng.choice((64, 128, 256)),
                           gen_budget=64, repository_blocks=128, sink_blocks=1, recent_blocks=rng.choice((0, 1))))
    states, local = [], 0
    for cycle in range(6):
        count = rng.choice((16, 32, 48, 64))
        spans = memory.begin_append(count, local)
        memory.attach_completion(Event())
        assert memory.commit_append(count)
        local += count
        records = memory._store.working_records()
        plan = memory.select({r[0]: rng.random() for r in records} if rng.random() < .8 else {})
        memory.attach_completion(Event())
        assert memory.commit_selection()
        local = sum(r[3] for r in memory._store.working_records())
        states.append((tuple(memory._store.working_records()), memory.history_end,
                       tuple(memory._store.action_window_segments(memory.history_end, local, 16, 16)[2])))
    return states


@pytest.mark.parametrize('seed', range(16))
def test_native_publish_equals_python_loop(seed, monkeypatch):
    assert run(seed, monkeypatch, '1') == run(seed, monkeypatch, '0')


def test_flag_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_SELECTION', 'yes')
    with pytest.raises(ValueError):
        _native_publish_selection()
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_SELECTION', '0')
    assert _native_publish_selection() is None
