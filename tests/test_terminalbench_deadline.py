import asyncio
from contextlib import asynccontextmanager, nullcontext
import json
import os
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.terminalbench_deadline import install, paused_seconds, wait_for_agent, validate_job_deadlines


def test_startup_validates_effective_deadline_and_disallows_double_scaling(tmp_path):
    from scripts.terminalbench_policy import configure_task_deadlines
    path=tmp_path/'dataset/tasks/test/task.toml'
    path.parent.mkdir(parents=True)
    path.write_text('[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 30\n')
    policy=configure_task_deadlines(tmp_path/'dataset/tasks', 'relaxed')
    (tmp_path/'agent-time-policy.json').write_text(json.dumps(policy))
    config=SimpleNamespace(timeout_multiplier=1, agent_timeout_multiplier=None,
        agents=[SimpleNamespace(override_timeout_sec=None, max_timeout_sec=None)],
        tasks=[SimpleNamespace(path=path.parent)])
    validate_job_deadlines(tmp_path, config)
    config.agent_timeout_multiplier=2
    with pytest.raises(ValueError, match='multiplier'):
        validate_job_deadlines(tmp_path, config)
    config.agent_timeout_multiplier=None
    path.write_text('[agent]\n[verifier]\ntimeout_sec = 30\n')
    with pytest.raises(ValueError, match='differs'):
        validate_job_deadlines(tmp_path, config)


def test_pause_accounting_clips_before_start_and_at_open_pause():
    intervals = [dict(started_at=2, finished_at=12), dict(started_at=18, finished_at=None)]
    assert paused_seconds(intervals, 10, 25) == 9
    assert paused_seconds(intervals, 13, 17) == 0


@pytest.mark.parametrize('intervals', [
    [dict(started_at=4, finished_at=3)],
    [dict(started_at=1, finished_at=4), dict(started_at=3, finished_at=5)],
    [dict(started_at=1, finished_at=None), dict(started_at=3, finished_at=5)],
    [dict(started_at=float('nan'), finished_at=4)],
])
def test_invalid_pause_journal_rejected(intervals):
    with pytest.raises(ValueError):
        paused_seconds(intervals, 0, 10)


def test_tool_wait_times_out_and_cleans_child(tmp_path):
    async def check():
        cleaned = []
        async def tool_wait():
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append(True)
        with pytest.raises(TimeoutError):
            await wait_for_agent(tool_wait(), .03, tmp_path/'pauses.json', poll_sec=.005)
        assert cleaned == [True]
    asyncio.run(asyncio.wait_for(check(), 2))


def test_recorded_pause_does_not_consume_remaining_budget(tmp_path):
    path=tmp_path/'pauses.json'
    async def check():
        gate=asyncio.Event()
        async def work():
            await gate.wait()
            return 'completed'
        task=asyncio.create_task(wait_for_agent(work(), .1, path, poll_sec=.005))
        await asyncio.sleep(.01)
        begin=time.time()
        path.write_text(json.dumps(dict(schema_version=1, intervals=[dict(started_at=begin, finished_at=None)])))
        await asyncio.sleep(.15)
        assert not task.done()
        path.write_text(json.dumps(dict(schema_version=1, intervals=[dict(started_at=begin, finished_at=time.time())])))
        gate.set()
        assert await task == 'completed'
    asyncio.run(asyncio.wait_for(check(), 2))


def test_external_cancellation_is_not_reported_as_timeout(tmp_path):
    async def check():
        entered=asyncio.Event(); cleaned=[]
        async def work():
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                cleaned.append(True)
        task=asyncio.create_task(wait_for_agent(work(), 100, tmp_path/'pauses.json'))
        await entered.wait(); task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned == [True]
    asyncio.run(asyncio.wait_for(check(), 2))


@pytest.mark.skipif(sys.platform != 'linux', reason='SIGSTOP pause regression')
def test_process_stop_time_is_excluded_after_resume(tmp_path):
    path=tmp_path/'pauses.json'
    code='''
import asyncio, sys
from scripts.terminalbench_deadline import wait_for_agent
async def work():
    print('ready', flush=True)
    await asyncio.Event().wait()
async def main():
    try:
        await wait_for_agent(work(), .2, sys.argv[1], poll_sec=.01)
    except TimeoutError:
        print('timed-out', flush=True)
asyncio.run(main())
'''
    proc=subprocess.Popen([sys.executable, '-c', code, str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([proc.stdout], [], [], 5)[0], 'Child did not start'
        assert proc.stdout.readline().strip() == 'ready'
        begin=time.time()
        path.write_text(json.dumps(dict(schema_version=1, intervals=[dict(started_at=begin, finished_at=None)])))
        os.kill(proc.pid, signal.SIGSTOP)
        time.sleep(.3)
        path.write_text(json.dumps(dict(schema_version=1, intervals=[dict(started_at=begin, finished_at=time.time())])))
        resumed=time.monotonic()
        os.kill(proc.pid, signal.SIGCONT)
        out, err=proc.communicate(timeout=3)
        assert proc.returncode==0, err
        assert out.strip()=='timed-out'
        assert time.monotonic()-resumed >= .1, 'Paused wall time consumed Agent budget'
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate(timeout=3)


def test_real_harbor_timeout_still_verifies_and_releases(tmp_path, monkeypatch):
    pytest.importorskip('harbor')
    import harbor.trial.sync_trajectory as sync_module
    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.single_step import SingleStepTrial
    from harbor.trial.trial import Trial
    monkeypatch.setattr(Trial, '_run_agent_phase', Trial._run_agent_phase)
    receipt=install(tmp_path, 'formal')
    assert receipt['agent_time_limit']=='relaxed'

    @asynccontextmanager
    async def scope(*args, **kwargs):
        yield

    async def check():
        trial=object.__new__(SingleStepTrial)
        trial.task=SimpleNamespace(config=TaskConfig.model_validate({}), instruction='original')
        trial.config=TrialConfig.model_validate(dict(task=dict(path=str(tmp_path)), agent=dict(name='terminus-2')))
        trial._agent_timeout_sec=.02
        trial._load_trajectory=None
        trial._result=SimpleNamespace()
        stages=[]; errors=[]
        async def work(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stages.append('agent-cancelled')
        async def emit(event):
            stages.append(event)
        async def noop(*args, **kwargs):
            pass
        async def verify():
            stages.append('verifier')
            trial.result.verifier_result={'rewards': {'reward': 0}}
        async def stop():
            stages.append('stop-environment')
        trial.agent=SimpleNamespace(extra_env={}, run=work)
        trial.user_agent=None
        trial.agent_environment=SimpleNamespace(with_default_user=lambda _: nullcontext(),
            scoped_exec_env=lambda _: nullcontext())
        trial._emit=emit
        trial._network_plan=lambda _: SimpleNamespace(agent_env_baseline=None, agent_phase=None)
        trial._phase_network_policy=scope
        trial._log_context=lambda *args: nullcontext()
        trial.paths=SimpleNamespace(agent_dir=tmp_path)
        trial._record_exception=lambda exc: errors.append(type(exc).__name__)
        trial._sync_agent_output=noop
        trial._upload_agent_logs=noop
        trial._collect_artifacts=noop
        trial._run_verifier=verify
        trial._stop_agent_environment=stop
        monkeypatch.setattr(sync_module, 'sync_trajectory', scope)
        await trial._run()
        assert errors==['AgentTimeoutError']
        assert stages==[TrialEvent.AGENT_START, 'agent-cancelled', TrialEvent.AGENT_END,
                        'verifier', 'stop-environment']
        assert trial.result.verifier_result['rewards']['reward']==0
    asyncio.run(asyncio.wait_for(check(), 2))
