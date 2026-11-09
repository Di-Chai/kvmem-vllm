"""Opt-in KVMEM_GIL_SWITCH_INTERVAL_US: CPython GIL switch interval of the engine process.

The engine thread prepares and launches each forward while kvmem worker threads
(refresh, scatter) run Python. A thread that requests the GIL gets it only after
the holder has run one whole switch interval (CPython default 5 ms), so each
handoff can cost the engine thread up to that interval while the GPU drains.
A shorter interval bounds those waits. Thread scheduling only: no numerical effect.
"""
import os
import sys

_MIN_US, _MAX_US = 50, 100_000


def configured():
    """The configured interval in seconds, or None when unset (keep the default)."""
    value = os.getenv('KVMEM_GIL_SWITCH_INTERVAL_US', '')
    if value == '':
        return None
    if not value.isdigit() or not _MIN_US <= int(value) <= _MAX_US:
        raise ValueError(f'KVMEM_GIL_SWITCH_INTERVAL_US must be an integer in {_MIN_US}..{_MAX_US}')
    return int(value) / 1e6


def apply():
    interval = configured()
    if interval is not None:
        sys.setswitchinterval(interval)
    name_threads()
    start_load()
    return interval


def name_threads():
    """Diagnostic KVMEM_DIAG_THREAD_NAMES=1: every Python thread started from now
    on also gets its Python name as its OS thread name (Linux PR_SET_NAME, 15
    bytes), so system profilers can attribute CPU time and GIL waits to e.g.
    kvmem-refresh_0. Thread names only; no effect on any result."""
    value = os.getenv('KVMEM_DIAG_THREAD_NAMES', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_DIAG_THREAD_NAMES must be 0 or 1')
    import threading
    if value != '1' or not sys.platform.startswith('linux') or getattr(threading.Thread, '_kvmem_named', False):
        return False
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    original = threading.Thread._bootstrap_inner

    def bootstrap_inner(self):
        try:
            libc.prctl(15, ctypes.c_char_p(self.name.encode()[:15]), 0, 0, 0)   # PR_SET_NAME
        except Exception:
            pass
        return original(self)

    threading.Thread._bootstrap_inner = bootstrap_inner
    threading.Thread._kvmem_named = True
    return True


def load_config():
    """Diagnostic KVMEM_DIAG_GIL_LOAD='duty,hold_ms': a daemon thread runs pure
    Python for hold_ms at a time, sleeping between bursts so that it holds the
    GIL for about `duty` of wall time. Measures how throughput responds to
    background GIL holding; never changes any computed result. None when unset."""
    value = os.getenv('KVMEM_DIAG_GIL_LOAD', '')
    if value == '':
        return None
    try:
        duty, hold_ms = (float(x) for x in value.split(','))
    except ValueError:
        raise ValueError('KVMEM_DIAG_GIL_LOAD must be "duty,hold_ms"') from None
    if not (0 < duty < 1 and 0.05 <= hold_ms <= 100):
        raise ValueError('KVMEM_DIAG_GIL_LOAD needs 0 < duty < 1 and 0.05 <= hold_ms <= 100')
    return duty, hold_ms / 1e3


_load_thread = None


def start_load():
    global _load_thread
    config = load_config()
    if config is None or _load_thread is not None:
        return None
    import threading
    import time
    duty, hold = config
    rest = hold * (1 - duty) / duty

    def burn():
        perf = time.perf_counter
        while True:
            end = perf() + hold
            x = 0
            while perf() < end:              # pure bytecode: holds the GIL
                x += 1
            time.sleep(rest)                 # releases the GIL

    _load_thread = threading.Thread(target=burn, name='kvmem-diag-gil-load', daemon=True)
    _load_thread.start()
    return config
