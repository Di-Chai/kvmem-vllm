"""Host-portable sizing helpers.

Thread pools are sized by configuration, but never beyond the CPUs this
process may run on (cgroup/affinity aware where the OS reports it). Pool
size only changes overlap, never results.
"""
import os


def available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):   # macOS, Windows, restricted sandboxes
        return max(1, os.cpu_count() or 1)


def cap_workers(requested):
    if type(requested) is not int or requested < 1:
        raise ValueError('Worker counts must be positive integers')
    return min(requested, available_cpus())
