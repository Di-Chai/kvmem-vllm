"""Exercise the new waiters' complete success paths without a process or GPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


BASE = Path(__file__).resolve().parents[1]/'artifacts'
PATHS = {
    'native':BASE/'natural-native8-capacity-threshold-20261017-r1/run-after-quiet-reverse.py',
    'metadata':BASE/'action-metadata-upload-qualification-20261017-r1/run-after-natural-capacity.py',
}


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('kind',['native','metadata'])
@pytest.mark.parametrize('accepted',[True,False])
def test_completed_predecessor_runs_owned_controller_once_and_passes_negative_capacity(tmp_path,monkeypatch,kind,accepted):
    spec=importlib.util.spec_from_file_location('queue_fake_'+kind,PATHS[kind])
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    root,prior=tmp_path/'current',tmp_path/'prior'
    root.mkdir();prior.mkdir()
    script=prior/'previous.py';script.write_text('# frozen predecessor identity')
    entry=root/'frozen/scripts/controller.py';entry.parent.mkdir(parents=True);entry.write_text('# never executed')
    monkeypatch.setattr(q,'ROOT',root);monkeypatch.setattr(q,'PRIOR',prior)
    monkeypatch.setattr(q,'PRIOR_SCRIPT',script);monkeypatch.setattr(q,'ENTRY',entry)
    done=dict(status='completed',drained=True,native_client_tokens_equal=True)
    verdict=dict(status='accepted' if accepted else 'contract_failed',accepted=accepted,normal_drain=True,
                 reason=None if accepted else 'contract_failed_headroom')
    for filename in ('status.json','queue-state.json'):
        write(prior/filename,dict(status='completed_pending_review'))
    if kind=='native':
        arms=['rolling-candidate-v2','pipeline-base','native']
        write(prior/'manifest.json',dict(experiment_kind='rolling_agents_reverse_v1',arms=arms))
        for arm in arms:write(prior/arm/'results.json',done)
        write(root/'manifest.json',dict(experiment_kind='natural_native_capacity_v1',arms=['native']))
    else:
        write(prior/'manifest.json',dict(experiment_kind='natural_native_capacity_v1',arms=['native']))
        write(prior/'native/results.json',done);write(prior/'deployment-acceptance.json',verdict)
        write(root/'manifest.json',dict(kind='action_metadata_upload_qualification_v1',
                                      cuda_cases_expected=4,cuda_timeout_seconds=600))
    # Both owners deliberately already have manifest_sha256. This is the
    # real second waiter's handoff shape and the duplicate-key regression.
    write(prior/'queue-owner.json',dict(pid=1806899,create_time=1792208506.86,script=str(script.resolve()),
        script_sha256=q.sha(script),manifest_sha256=q.sha(prior/'manifest.json')))
    write(root/'status.json',dict(status='prepared'))
    def process(pid=None):
        if pid==1806899:raise q.psutil.NoSuchProcess(pid)
        return SimpleNamespace(pid=1234 if pid is None else pid,create_time=lambda:100.0)
    monkeypatch.setattr(q.psutil,'Process',process)
    calls=[]
    proof=dict(cuda_cases=4,action_requests=6,upload_submissions=100,history_requests=6,
               cold_restore_records=4,real_engine_cancellation_validated=False)
    def validate_upload():
        calls.append('postproof')
        assert q.read(root/'status.json')['status']=='completed_pending_review'
        return proof
    if kind=='metadata':monkeypatch.setattr(q,'qualified_upload',validate_upload)
    class Child:
        pid=2345
        def wait(self,timeout=None):
            calls.append('wait')
            write(root/'status.json',dict(status='completed_pending_review'))
            if kind=='native':
                write(root/'native/results.json',done)
                write(root/'deployment-acceptance.json',verdict)
            return 0
    def popen(command,**kwargs):
        calls.append('spawn')
        assert command[1:]==[str(entry),'--run-prepared','--output',str(root)]
        assert kwargs['start_new_session'] is True
        assert q.read(root/'prior-owner.json')['manifest_sha256']==q.sha(prior/'manifest.json')
        assert q.read(root/'queue-owner.json')['manifest_sha256']==q.sha(root/'manifest.json')
        return Child()
    monkeypatch.setattr(q.subprocess,'Popen',popen)
    q.main()
    final=q.read(root/'queue-state.json')
    assert final['status']=='completed_pending_review'
    assert calls==(['spawn','wait'] if kind=='native' else ['spawn','wait','postproof'])
    if kind=='native':assert final['deployment_accepted'] is accepted
    else:assert final['qualification']==proof
