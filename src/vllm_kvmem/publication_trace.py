"""Opt-in timestamps for the prefill KV stage-out/publication path.

KVMEM_PUBLICATION_TRACE=1 appends one JSON row per event to
<KVMEM_RUNNER_PROBE_DIR>/publication_trace.jsonl using time.monotonic(), the
clock of the existing step and refresh logs. Disabled, emit() is a no-op.
"""
import json
import os
import threading
import time

_lock = threading.Lock()
_path = None


def enabled():
    return os.getenv('KVMEM_PUBLICATION_TRACE') == '1'


def emit(event, **fields):
    global _path
    if not enabled():
        return
    if _path is None:
        directory = os.getenv('KVMEM_RUNNER_PROBE_DIR')
        if not directory:
            return
        _path = os.path.join(directory, 'publication_trace.jsonl')
    row = dict(event=event, at=time.monotonic(), thread=threading.current_thread().name, **fields)
    line = json.dumps(row, default=str) + '\n'
    with _lock:
        with open(_path, 'a') as stream:
            stream.write(line)
