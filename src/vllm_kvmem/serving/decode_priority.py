"""Opt-in KVMEM_LAYERWISE_DECODE_PRIORITY: decode-stage pressure goes first.

A rolling decode-stage pressure refresh suspends a decoding request until its
window is rebuilt. Prefill pressure (pipelined behind its ACK), final Query and
copy restores share the single assembly lane and its transfer credit, and the
boundary paths that admit prefill work can take the lane every time it frees.
While a decode-stage pressure job waits for the lane, other jobs yield their
admission and the maintenance order tries it first. Selections, bytes and
credits are unchanged; only the order of lane admission changes.
"""
import os
from ..core import settings as _settings


def enabled():
    value = _settings.get('KVMEM_LAYERWISE_DECODE_PRIORITY', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_DECODE_PRIORITY must be 0 or 1')
    return value == '1'


def is_decode_pressure(job):
    command = job.command
    return (getattr(command, 'selection', None) == 'pressure' and
            getattr(command, 'pressure_stage', None) == 'decode')


def waiting(job):
    return (is_decode_pressure(job) and job.stage == 'assembly_wait' and
            not job.done and not job.cancelled)


def should_yield(jobs, job):
    """True when `job` must leave the lane to a waiting decode-stage pressure."""
    if not enabled() or is_decode_pressure(job):
        return False
    return any(other is not job and waiting(other) for other in jobs)
