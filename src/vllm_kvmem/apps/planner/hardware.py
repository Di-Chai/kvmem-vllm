"""Machine probe: RAM, pinned-memory limits, cores, SSD space, optional copy benchmarks.

Benchmarks are opt-in (``bench=True``) and short: H2D bandwidth from pinned
memory, cold host gather of 32 KB regions from 1 MB records (the swap-in's CPU
copy), and host-register cost. They need a CUDA device but no server.
"""
import json
import os
import resource
import shutil
import time
from pathlib import Path

from .memory import Machine


def _meminfo():
    info = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            info[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return info


def _cgroup_limit():
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return int(text)
    return None


def _cgroup_usage():
    for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _cgroup_cpus():
    """CPU quota of this cgroup (containers often see every host core in
    affinity but may only use `quota / period` of them)."""
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        if quota != 'max':
            return max(1, -(-int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota > 0 and period > 0:
            return max(1, -(-quota // period))
    except (OSError, ValueError):
        pass
    return None


def probe(*, ssd_dir=None, reserved_fraction=0.15, reserved_min_bytes=16 << 30):
    info = _meminfo()
    ram = info.get('MemTotal', 0)
    available = info.get('MemAvailable')
    cgroup = _cgroup_limit()
    if cgroup:
        ram = min(ram, cgroup)
        used = _cgroup_usage()
        if used is not None:
            available = min(available, max(0, cgroup - used)) if available is not None else max(0, cgroup - used)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    memlock = float('inf') if soft == resource.RLIM_INFINITY else float(soft)
    ssd = shutil.disk_usage(ssd_dir).free if ssd_dir else 0
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    quota = _cgroup_cpus()
    if quota:
        cores = min(cores, quota)
    reserved = max(reserved_min_bytes, int(reserved_fraction * ram))
    machine = Machine(ram_bytes=ram, ram_reserved_bytes=reserved, memlock_bytes=memlock, ssd_bytes=ssd)
    return machine, dict(ram_bytes=ram, cgroup_limit_bytes=cgroup, memlock_bytes=memlock,
                         ssd_free_bytes=ssd, cores=cores, ram_available_bytes=available)


def bench(*, gather_threads=(1, 8), seconds=0.5):
    """Short copy benchmarks (GB/s). Requires torch with CUDA."""
    import numpy as np
    import torch
    out = {}
    size = 256 << 20
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(size, dtype=torch.uint8, device='cuda')
    dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    start, n = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        dev.copy_(host, non_blocking=True)
        n += 1
    torch.cuda.synchronize()
    out['h2d_gbps'] = n * size / (time.perf_counter() - start) / 1e9
    try:
        from ... import _native
        record, region, count = 1 << 20, 32 << 10, 2048
        backing = np.empty(record * count, dtype=np.uint8)
        backing[::4096] = 1
        sources = [memoryview(backing[i * record:(i + 1) * record]) for i in range(count)]
        dest = bytearray(count * region)
        for threads in gather_threads:
            os.environ['KVMEM_NATIVE_GATHER_THREADS'] = str(threads)
            start, n = time.perf_counter(), 0
            while time.perf_counter() - start < seconds:
                _native.gather_record_region(dest, sources, record, (n % 30) * region, region, 0, count)
                n += 1
            out[f'gather_gbps_{threads}t'] = n * count * region / (time.perf_counter() - start) / 1e9
        os.environ.pop('KVMEM_NATIVE_GATHER_THREADS', None)
    except Exception as error:      # benchmarks are advisory
        out['gather_error'] = repr(error)
    return out


def bench_spill_device(directory, *, size=1 << 30, record=32 * 34816, threads=None):
    """Write then read ``size`` bytes of archive records in ``directory`` (the
    CPU->SSD session tier), with the same parallelism as serving
    (KVMEM_TIERING_THREADS; queue depth matters a lot on networked and NVMe
    devices). The file's page cache is dropped before reading
    (posix_fadvise DONTNEED), so the read reflects the device. Also reports
    whether the backing block device is rotational."""
    from concurrent.futures import ThreadPoolExecutor
    import random
    from ...serving.session_tiering import tiering_threads
    threads = threads or tiering_threads()
    count = max(1, size // record)
    size = count * record
    path = os.path.join(directory, f'.kvmem-plan-bench-{os.getpid()}')
    buf = os.urandom(record)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    try:
        with ThreadPoolExecutor(threads) as pool:
            start = time.perf_counter()
            list(pool.map(lambda i: os.pwrite(fd, buf, i * record), range(count)))
            os.fsync(fd)
            write = size / (time.perf_counter() - start) / 1e9
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            order = list(range(count))
            random.Random(0).shuffle(order)      # sessions are read back record by record
            start = time.perf_counter()
            list(pool.map(lambda i: os.pread(fd, record, i * record), order))
            read = size / (time.perf_counter() - start) / 1e9
    finally:
        os.close(fd)
        os.unlink(path)
    rotational = None
    try:
        dev = os.stat(directory).st_dev
        major, minor = os.major(dev), os.minor(dev)
        base = os.path.realpath(f'/sys/dev/block/{major}:{minor}')
        for candidate in (base, os.path.dirname(base)):
            flag = os.path.join(candidate, 'queue', 'rotational')
            if os.path.exists(flag):
                rotational = open(flag).read().strip() == '1'
                break
    except OSError:
        pass
    return dict(spill_write_gbps=write, spill_read_gbps=read, spill_rotational=rotational,
                spill_threads=threads)


def save(path, machine_info, bench_results=None):
    Path(path).write_text(json.dumps(dict(machine=machine_info, bench=bench_results or {}), indent=2))
