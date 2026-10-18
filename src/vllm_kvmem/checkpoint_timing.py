"""Checkpoint host-stage observations; never introduce a CUDA timing fence."""
from contextlib import contextmanager
import json
import os
import time


def async_checkpoint_capture(runner=None):
    value = os.getenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE')
    if value is None:
        config = getattr(getattr(runner, 'vllm_config', None), 'kv_transfer_config', None)
        extra = getattr(config, 'kv_connector_extra_config', {}) or {}
        enabled = extra.get('kvmem', {}).get('async_checkpoint_capture', True)
        if type(enabled) is not bool:
            raise ValueError('async_checkpoint_capture must be boolean')
        return enabled
    if value not in ('0', '1'):
        raise ValueError('KVMEM_ASYNC_CHECKPOINT_CAPTURE must be 0 or 1')
    return value == '1'


@contextmanager
def checkpoint_stage(runner, name, *, sync=False):
    trace = getattr(runner, '_kvmem_checkpoint_trace', None)
    if trace is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        stages = trace['host_stage_seconds']
        stages[name] = stages.get(name, 0.) + elapsed
        if sync:
            trace['host_sync_wait_seconds'] += elapsed


@contextmanager
def checkpoint_trace(runner, request_id, position, terminal, optimized):
    previous = getattr(runner, '_kvmem_checkpoint_trace', None)
    trace = dict(request_id=request_id, position=position, terminal=terminal,
                 optimized=optimized, at=time.perf_counter(), wall_time=time.time(),
                 host_stage_seconds={}, host_sync_wait_seconds=0., status='started')
    runner._kvmem_checkpoint_trace = trace
    try:
        yield trace
        trace['status'] = 'submitted'
    except BaseException as error:
        trace['status'] = type(error).__name__
        raise
    finally:
        runner._kvmem_checkpoint_trace = previous
        trace['host_total_seconds'] = time.perf_counter() - trace['at']
        output = getattr(runner, '_kvmem_probe_output', None)
        if output is not None:
            with (output / 'checkpoint_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(trace) + '\n')
