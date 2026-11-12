from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.runner import make_probe_runner


def fixture(monkeypatch, *, fail=False):
    calls=[];batch=NS(req_ids=['first'])
    class Base:
        def execute_model(self, *args, **kwargs):
            calls.append('forward')
            self._kvmem_prepare_action_metadata(batch,None)
            self.execute_model_state=NS(input_batch=batch)
            return 'result'
    def create(runner, output):
        calls.append('create')
        if fail:raise RuntimeError('allocation failed')
        return NS(action=NS(prepare=lambda *args:calls.append('arm')),
                  note_prefill_pipeline_forward=lambda *args:calls.append('planned'),
                  start_stream_session=lambda *args:calls.append('stream'))
    monkeypatch.setattr('vllm_kvmem.capture_runtime.CaptureRuntime',create)
    runner=make_probe_runner(Base)()
    runner._kvmem_continuous=True;runner._kvmem_capture_runtime=None
    runner._kvmem_native_attention=NS(disarm=lambda:calls.append('disarm'))
    runner._kvmem_probe_output='unused'
    return runner,calls


def test_first_public_forward_arms_native_scores_before_model_and_reuses_runtime(monkeypatch):
    runner,calls=fixture(monkeypatch)
    scheduled=NS(num_scheduled_tokens={'first':1})
    assert runner._kvmem_execute_public_forward(scheduled,observe_work=True)=='result'
    assert calls==['create','forward','arm','planned','stream']
    runtime=runner._kvmem_capture_runtime
    runner._kvmem_execute_public_forward(scheduled,observe_work=True)
    assert runner._kvmem_capture_runtime is runtime and calls.count('create')==1
    assert calls[-4:]==['forward','arm','planned','stream']


@pytest.mark.parametrize('kind',['private','dummy','profile','disabled','noncontinuous'])
def test_native_bootstrap_never_initializes_for_unqualified_work(monkeypatch,kind):
    runner,calls=fixture(monkeypatch);kwargs={}
    if kind=='dummy':kwargs['dummy_run']=True
    if kind=='profile':kwargs['is_profile']=True
    if kind=='disabled':runner._kvmem_native_attention=None
    if kind=='noncontinuous':runner._kvmem_continuous=False
    runner._kvmem_execute_public_forward(NS(num_scheduled_tokens={'first':1}),observe_work=kind!='private',**kwargs)
    assert runner._kvmem_capture_runtime is None and 'create' not in calls


def test_bootstrap_failure_cannot_run_unprepared_model(monkeypatch):
    runner,calls=fixture(monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='allocation failed'):
        runner._kvmem_execute_public_forward(None,observe_work=True)
    assert calls==['create'] and runner._kvmem_capture_runtime is None
