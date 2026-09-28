"""Original CPU policy worker bindings, independent of serving activation."""
import subprocess
import sys

import pytest
from vllm_kvmem import _native as n


def config(ready, background=False):
    c = n.RepositoryGcConfig()
    c.ready_low = c.ready_high = ready
    c.background = background
    c.tick_budget_us = 100000
    c.policy.scan_blocks = c.policy.batch_blocks = 64
    c.policy.poll_ms = 1
    c.policy.grace_steps = 2
    c.policy.candidate_steps = 1
    return c


def test_original_gc_respects_watermarks_backing_recent_and_read_leases():
    repo = n.Repository(8, 32)
    c = config(3)
    c.recent_tokens = 32
    freed = []
    worker = n.RepositoryGc(c, repo, lambda record: freed.append(record.handle.slot) or True)
    backed = n.GC_SEALED | n.GC_RAW_BACKED | n.GC_VALUE_BACKED | n.GC_INDEXED
    handles, checkpoint = [], None
    worker.begin_step()
    for i in range(8):
        lease = repo.reserve(i*32, 32).commit()
        handles.append(lease.handle)
        flags = backed
        if i == 0: flags |= n.GC_MANDATORY
        if i == 1: flags |= n.GC_IO
        if i == 2: flags &= ~n.GC_INDEXED
        if i == 3: checkpoint = lease.clone()
        worker.publish(lease, flags)
        lease.reset()
    worker.set_history_end(256)
    for _ in range(10): worker.begin_step()
    worker.poll()
    assert not freed
    worker.begin_step()
    worker.poll()
    assert sorted(freed) == [4, 5, 6] and repo.ready_slots == 3
    worker.begin_step()
    worker.poll()
    assert len(freed) == 3 and repo.pin(handles[3])
    checkpoint.reset()
    worker.stop()


def test_failed_python_release_retains_generation_and_surfaces_native_failure():
    repo = n.Repository(1, 32)
    lease = repo.reserve(0, 32).commit()
    handle = lease.handle
    repo.retire(handle)
    lease.reset()
    failed = True
    def release(record):
        assert record.handle == handle
        if failed:
            raise OSError('injected archive cleanup failure')
        return True
    worker = n.RepositoryGc(config(1), repo, release)
    worker.poll()
    assert worker.stats['failures'] == 1 and 'injected archive' in worker.last_error
    assert not repo.reserve(32, 32) and repo.ready_slots == 0
    failed = False
    worker.poll()
    assert repo.ready_slots == 1 and worker.stats['reclaimed'] == 1
    new = repo.reserve(32, 32).commit()
    assert new.handle.generation != handle.generation
    worker.stop()


@pytest.mark.parametrize('action', ['worker.stop()', 'del worker'])
def test_background_stop_and_destructor_release_gil_while_joining_python_callback(action):
    # A regression would deadlock the interpreter. Isolate it with a deadline.
    code = f'''
from threading import Event, Timer
from vllm_kvmem import _native as n
repo = n.Repository(1, 32)
lease = repo.reserve(0, 32).commit()
repo.retire(lease.handle)
lease.reset()
started, release = Event(), Event()
def callback(record):
    started.set()
    assert release.wait(5)
    return True
c = n.RepositoryGcConfig()
c.ready_low = c.ready_high = 1
c.policy.poll_ms = 1
worker = n.RepositoryGc(c, repo, callback)
assert started.wait(5)
timer = Timer(0.1, release.set)
timer.start()
{action}
timer.join()
assert repo.ready_slots == 1
print('joined')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'joined'


def test_store_retains_gc_bridge_and_rejects_foreign_repository():
    import gc
    import weakref
    from vllm_kvmem.config import KVMemConfig
    cfg = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
                      repository_blocks=8, sink_blocks=1, recent_blocks=0)
    repo = n.Repository(8, 32)
    store = n.Store(cfg.native(), repo)
    worker = n.RepositoryGc(config(2), repo, lambda _: True)
    ref = weakref.ref(worker)
    store.attach_gc(worker)
    del worker
    gc.collect()
    assert ref() is not None
    store.gc_begin_step()
    store.gc_end_step()
    foreign = n.RepositoryGc(config(2), n.Repository(8, 32), lambda _: True)
    with pytest.raises(ValueError, match='different repository'):
        store.attach_gc(foreign)
    ref().stop()
    foreign.stop()
    del store
    gc.collect()
    assert ref() is None
