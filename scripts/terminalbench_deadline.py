"""Pause-aware Agent deadlines; Harbor still owns verification and cleanup.

The operator must atomically open an interval before stopping the worker and
close it before resuming it. This journal does not itself pause any process.
SIGSTOP alone does not suspend asyncio or wall clocks.
"""
import asyncio
from functools import wraps
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import tomllib


def validate_job_deadlines(root, config):
    """Reject missing/overridden limits before starting the formal job."""
    policy = json.loads((Path(root)/'agent-time-policy.json').read_text())
    if policy.get('agent_time_limit') != 'relaxed' or policy.get('agent_multiplier') != 2.0:
        raise ValueError('Missing relaxed Agent timing audit')
    if config.timeout_multiplier != 1 or config.agent_timeout_multiplier not in (None, 1):
        raise ValueError('Agent timeout multiplier outside audited task policy')
    for agent in config.agents:
        if agent.override_timeout_sec is not None or agent.max_timeout_sec is not None:
            raise ValueError('Agent timeout override outside audited task policy')
    for task in config.tasks:
        path = Path(task.path)/'task.toml'
        row = policy['tasks'][path.parent.name]
        original = row['original_agent_timeout_sec']
        if type(original) not in (int, float) or not math.isfinite(original) or original <= 0:
            raise ValueError('Invalid original Agent deadline')
        raw = path.read_bytes()
        parsed = tomllib.loads(raw.decode())
        expected = original * 2
        if (not math.isfinite(expected) or parsed.get('steps') or
                row.get('effective_agent_timeout_sec') != expected or
                parsed.get('agent', {}).get('timeout_sec') != expected or
                hashlib.sha256(raw).hexdigest() != row['effective_sha256']):
            raise ValueError('Effective Agent deadline differs from relaxed timing audit')


def paused_seconds(intervals, start, end):
    total, previous_end = 0.0, -math.inf
    for index, row in enumerate(intervals):
        if not isinstance(row, dict) or set(row) != {'started_at', 'finished_at'}:
            raise ValueError('Invalid Agent pause interval')
        begin, finish = row['started_at'], row['finished_at']
        if (type(begin) not in (int, float) or not math.isfinite(begin) or
                begin < previous_end):
            raise ValueError('Unordered or overlapping Agent pauses')
        if finish is None:
            if index != len(intervals)-1:
                raise ValueError('Only the final Agent pause can be open')
            finish = max(begin, end)
        if type(finish) not in (int, float) or not math.isfinite(finish) or finish < begin:
            raise ValueError('Invalid Agent pause end')
        total += max(0.0, min(end, finish) - max(start, begin))
        previous_end = finish
    return total


def read_pauses(path):
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return []
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'intervals'} or
            type(value['schema_version']) is not int or value['schema_version'] != 1 or
            not isinstance(value['intervals'], list)):
        raise ValueError('Invalid Agent pause journal')
    return value['intervals']


async def wait_for_agent(awaitable, timeout_sec, pause_path, *, poll_sec=1.0):
    """Cancel on budget exhaustion, retaining ordinary external cancellation."""
    if timeout_sec is None:
        return await awaitable
    start, monotonic_start = time.time(), time.monotonic()
    work = asyncio.ensure_future(awaitable)
    try:
        if (type(timeout_sec) not in (int, float) or not math.isfinite(timeout_sec) or
                timeout_sec <= 0 or poll_sec <= 0):
            raise ValueError('Agent timeout and poll interval must be positive')
        while True:
            elapsed = time.monotonic() - monotonic_start
            # Anchor wall timestamps to a monotonic clock so NTP adjustments
            # cannot extend an unpaused task indefinitely.
            paused = paused_seconds(read_pauses(pause_path), start, start + elapsed)
            remaining = timeout_sec - max(0.0, elapsed - paused)
            if work.done():
                return work.result()
            if remaining <= 0:
                raise TimeoutError('Agent elapsed-time budget exhausted')
            await asyncio.wait({work}, timeout=min(poll_sec, remaining))
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)


def install(root, job_name):
    """Scope the replacement to Agent phase; leave all other Harbor timers alone."""
    from harbor.trial.errors import AgentTimeoutError
    from harbor.trial.trial import Trial
    original = Trial._run_agent_phase
    if getattr(original, '_kvmem_pause_deadline', False):
        raise ValueError('Agent deadline hook already installed in this process')
    root = Path(root)
    pause_path = root/'agent-pause-intervals.json'
    # Validate a pre-existing journal before launching any trial.
    paused_seconds(read_pauses(pause_path), time.time(), time.time())

    @wraps(original)
    async def bounded(self, *, timeout_sec, **kwargs):
        try:
            # Disable the inner wall timer only after installing the outer
            # pause-aware timer. Harbor's finally/AgentTimeout handling still
            # emits AGENT_END, releases KV, syncs artifacts and runs verifier.
            return await wait_for_agent(original(self, timeout_sec=None, **kwargs),
                                        timeout_sec, pause_path)
        except asyncio.TimeoutError as exc:
            raise AgentTimeoutError(
                f'Agent execution timed out after {timeout_sec} active seconds '
                '(recorded pauses excluded)') from exc

    bounded._kvmem_pause_deadline = True
    receipt = dict(schema_version=1, agent_time_limit='relaxed',
                   pause_journal=str(pause_path), poll_sec=1.0,
                   original_phase_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                   scope='Agent phase only; verifier and cleanup unchanged')
    with (root/(job_name+'-deadline-hook.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2)
    Trial._run_agent_phase = bounded
    return receipt
