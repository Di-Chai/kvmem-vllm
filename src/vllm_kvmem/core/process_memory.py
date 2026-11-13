"""Best-effort Linux host-memory observations; no allocator or limit control.

Worker RSS covers mapped resident pages, including model/index/pinned allocations
and allocator retention. It omits unmapped filesystem page cache. Cgroup v2
memory includes charged file cache and other processes in the same cgroup; these
overlapping measurements must not be added or attributed solely to the archive.
VmLck is an OS observation, not a complete CUDA pinned-allocation accounting.
"""
import os
from pathlib import Path, PurePosixPath
import time


_STATUS_FIELDS = dict(VmRSS='rss_bytes', VmHWM='lifetime_peak_rss_bytes',
                      RssAnon='anon_rss_bytes', RssFile='file_rss_bytes',
                      RssShmem='shmem_rss_bytes', VmLck='locked_bytes')
_CGROUP_FIELDS = ('anon', 'file', 'shmem', 'kernel', 'pagetables',
                  'file_dirty', 'file_writeback', 'unevictable')


def _nonnegative(value):
    if not value.isdecimal():
        raise ValueError('Memory observation must be a nonnegative integer')
    return int(value)


def sample_process_memory(*, proc_self='/proc/self', cgroup_root='/sys/fs/cgroup'):
    """Observe this worker, recording unavailable counters instead of zeroes.

    The cgroup root must be the mounted unified hierarchy. A missing/unreadable
    controller (including unsupported v1 mounts) is explicit in ``errors``.
    These cheap counters share the existing archive-report publication cadence.
    """
    proc_self, cgroup_root = Path(proc_self), Path(cgroup_root)
    report = dict(pid=os.getpid(), at=time.perf_counter(), worker={}, cgroup={}, errors=[],
        scope='worker RSS excludes unmapped file cache; cgroup includes all charged '
              'processes and file cache; scopes overlap and are not additive; '
              'lifetime_peak_rss_bytes is not reset per phase; locked_bytes is '
              'not complete CUDA pinned accounting; cgroup max describes this '
              'directory only and ancestor limits may apply; observations enforce no cap')
    try:
        status = proc_self.joinpath('status').read_text()
    except OSError as error:
        report['errors'].append('worker status: '+type(error).__name__)
    else:
        for line in status.splitlines():
            key, separator, raw = line.partition(':')
            if separator and key in _STATUS_FIELDS:
                try:
                    value, unit = raw.split()
                    if unit != 'kB':
                        raise ValueError('Unknown proc memory unit')
                    report['worker'][_STATUS_FIELDS[key]] = _nonnegative(value) * 1024
                except ValueError:
                    report['errors'].append('invalid worker counter: '+key)
    try:
        memberships = proc_self.joinpath('cgroup').read_text().splitlines()
        relative = next(row[3:] for row in memberships if row.startswith('0::'))
        path = PurePosixPath(relative)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid cgroup membership path')
        directory = cgroup_root.joinpath(*path.parts[1:])
        report['cgroup']['path'] = relative
        report['cgroup']['version'] = 2
    except (OSError, ValueError, StopIteration) as error:
        report['errors'].append('cgroup membership: '+type(error).__name__)
        return report
    for filename, name in (('memory.current', 'current_bytes'), ('memory.max', 'max_bytes')):
        try:
            value = directory.joinpath(filename).read_text().strip()
            if name == 'max_bytes' and value == 'max':
                report['cgroup']['limit_unbounded'] = True
            else:
                report['cgroup'][name] = _nonnegative(value)
                if name == 'max_bytes':
                    report['cgroup']['limit_unbounded'] = False
        except (OSError, ValueError) as error:
            report['errors'].append(filename+': '+type(error).__name__)
    try:
        rows = directory.joinpath('memory.stat').read_text().splitlines()
    except OSError as error:
        report['errors'].append('memory.stat: '+type(error).__name__)
    else:
        for line in rows:
            key, _, value = line.partition(' ')
            if key in _CGROUP_FIELDS:
                try:
                    report['cgroup'][key+'_bytes'] = _nonnegative(value.strip())
                except ValueError:
                    report['errors'].append('invalid cgroup counter: '+key)
    return report
