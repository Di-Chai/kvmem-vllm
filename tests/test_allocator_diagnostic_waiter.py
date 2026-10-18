"""No process/GPU launch: strict serial predecessor and diagnostic terminal gates."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

path = Path(__file__).parents[1]/'artifacts/allocator-cold12-diagnostic-20261017-r1/run-after-action-upload.py'
spec = importlib.util.spec_from_file_location('allocator_waiter_test',path)
waiter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waiter)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def completed_prior(tmp_path,monkeypatch):
    root,prior = tmp_path/'diagnostic',tmp_path/'prior'
    root.mkdir();prior.mkdir()
    script = prior/'waiter.py';script.write_text('exact predecessor')
    save(prior/'manifest.json',dict(kind='action_metadata_upload_qualification_v1',cuda_cases_expected=4))
    monkeypatch.setattr(waiter,'ROOT',root)
    monkeypatch.setattr(waiter,'PRIOR',prior)
    monkeypatch.setattr(waiter,'PRIOR_SCRIPT',script)
    monkeypatch.setattr(waiter,'PRIOR_SCRIPT_SHA',waiter.sha(script))
    monkeypatch.setattr(waiter,'PRIOR_MANIFEST_SHA',waiter.sha(prior/'manifest.json'))
    owner = dict(pid=waiter.PRIOR_PID,create_time=waiter.PRIOR_BIRTH,script=str(script),
        script_sha256=waiter.sha(script),manifest_sha256=waiter.sha(prior/'manifest.json'))
    save(prior/'queue-owner.json',owner)
    save(prior/'status.json',dict(status='completed_pending_review'))
    (prior/'oracle.xml').write_text('<testsuite tests="4" failures="0" errors="0" skipped="0">'+
        ''.join(f'<testcase name="c{i}"/>' for i in range(4))+'</testsuite>')
    low = dict(tests=4,failures=0,errors=0,skipped=0,cases=[f'c{i}' for i in range(4)],
               xml_sha256=waiter.sha(prior/'oracle.xml'))
    save(prior/'low-level-proof.json',low)
    result = dict(status='passed',cleanup_completed=True,native_reconciliation=dict(passed=True,preemptions=0),
        low_level=low,action_upload=dict(requests=[{}]*6,upload_final=dict(submissions=22)),
        history_proofs=[{}]*6,oracle=dict(target_restore_records=5),
        cancellation_validated=False,low_level_lease_exception_validated=True)
    save(prior/'results.json',result)
    proof = dict(cuda_cases=4,action_requests=6,upload_submissions=22,history_requests=6,
                 cold_restore_records=5,real_engine_cancellation_validated=False)
    save(prior/'queue-state.json',dict(status='completed_pending_review',qualification=proof))
    return root,prior,owner,proof


def test_completed_exact_predecessor_admits_only_full_proof(tmp_path,monkeypatch):
    root,prior,owner,proof = completed_prior(tmp_path,monkeypatch)
    assert waiter.prior_completed() == proof
    def gone(pid):
        raise waiter.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(waiter.psutil,'Process',gone)
    waiter.wait_prior()
    assert waiter.read(root/'prior-owner.json')['manifest_sha256'] == owner['manifest_sha256']
    assert waiter.read(root/'prior-qualification-proof.json') == proof


@pytest.mark.parametrize('change',[dict(pid=1),dict(create_time=1),dict(script_sha256='changed'),
                                 dict(manifest_sha256='changed')])
def test_owner_identity_cannot_be_substituted(tmp_path,monkeypatch,change):
    _,prior,owner,_ = completed_prior(tmp_path,monkeypatch)
    save(prior/'queue-owner.json',dict(owner,**change))
    with pytest.raises(RuntimeError,match='changed'):
        waiter.prior_completed()


def test_reused_live_pid_does_not_count_as_completed_owner(tmp_path,monkeypatch):
    completed_prior(tmp_path,monkeypatch)
    monkeypatch.setattr(waiter.psutil,'Process',lambda pid:NS(create_time=lambda:waiter.PRIOR_BIRTH+100))
    with pytest.raises(RuntimeError,match='PID was reused'):
        waiter.wait_prior()


@pytest.mark.parametrize('part',['queue','normal_cleanup','cuda_skip','qualification_fields','cleanup_file'])
def test_incomplete_qualification_never_admits_diagnostic(tmp_path,monkeypatch,part):
    _,prior,_,_ = completed_prior(tmp_path,monkeypatch)
    if part == 'queue':
        save(prior/'queue-state.json',dict(status='running'))
    elif part == 'normal_cleanup':
        result=waiter.read(prior/'results.json');result['cleanup_completed']=False;save(prior/'results.json',result)
    elif part == 'cuda_skip':
        xml=prior/'oracle.xml';xml.write_text(xml.read_text().replace('skipped="0"','skipped="1"'))
    elif part == 'qualification_fields':
        state=waiter.read(prior/'queue-state.json');state['qualification']['cuda_cases']=3;save(prior/'queue-state.json',state)
    else:
        save(prior/'server/cleanup-required.json',dict(pid=99))
    with pytest.raises(RuntimeError):
        waiter.prior_completed()


@pytest.mark.parametrize('status',['diagnostic_completed','diagnostic_completed_with_gaps'])
def test_completed_diagnostic_allows_negative_headroom_but_requires_count_equality(tmp_path,monkeypatch,status):
    root=tmp_path
    monkeypatch.setattr(waiter,'ROOT',root)
    save(root/'status.json',dict(status=status))
    save(root/'kvmem-ready/results.json',dict(status='completed',drained=True,native_client_tokens_equal=True,
        preemptions=0,total_stream_output_tokens=48))
    traces=[dict(output_ids=[1])]*48
    save(root/'kvmem-ready/tools/traces.json',traces)
    replay=dict(warmup_requests=24,cleanup_requests=24)
    save(root/'replay-proof.json',replay)
    observed=dict(diagnostic_only=True,performance_claim_allowed=False,deployment_acceptance_allowed=False,
        sampled_headroom_mib=603,coverage_gaps=[dict(reason='tail_unproven')])
    save(root/'diagnostic.json',observed)
    samples=[dict(values={'vllm:generation_tokens_total':48})]
    gate=NS(verify=lambda root:dict(experiment_kind='allocator_cold_twelve_diagnostic_v1'),
        replay_proof=lambda root:replay,observations=lambda root,m:observed,
        ready=NS(records=lambda path:samples))
    monkeypatch.setitem(sys.modules,'vllm_kvmem',NS(__file__=str(root/'runtime/kvmem-ready/src/vllm_kvmem/__init__.py')))
    monkeypatch.setattr(waiter.importlib.util,'spec_from_file_location',lambda *args:NS(name='fake_gate',loader=NS(exec_module=lambda value:None)))
    monkeypatch.setattr(waiter.importlib.util,'module_from_spec',lambda spec:gate)
    proof=waiter.diagnostic_completed()
    assert proof['sampled_headroom_mib']==603 and proof['total_requests']==48
    samples[0]['values']['vllm:generation_tokens_total']=47
    with pytest.raises(RuntimeError,match='count equality'):
        waiter.diagnostic_completed()


def test_existing_owner_prevents_any_state_overwrite_or_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(waiter,'ROOT',tmp_path)
    save(tmp_path/'queue-owner.json',dict(pid=7))
    monkeypatch.setattr(waiter.signal,'signal',lambda *args:None)
    monkeypatch.setattr(waiter,'main',lambda:pytest.fail('Restarted existing owner'))
    with pytest.raises(RuntimeError,match='No queue restart'):
        waiter.launch_once()
    assert waiter.read(tmp_path/'queue-owner.json')==dict(pid=7)
    assert not (tmp_path/'queue-state.json').exists()
