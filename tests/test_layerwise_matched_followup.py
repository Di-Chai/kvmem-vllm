import pytest

import recall_comparison_protocol as p
import run_layerwise_matched_followup as followup
from run_layerwise_matched_followup import source_ready


def test_terminal_file_does_not_release_a_live_qualification_owner():
    owner=dict(pid=42,start_ticks='100',command_sha256='original')
    assert not source_ready('running',owner,owner)
    assert not source_ready('completed',owner,owner)
    assert source_ready('completed',None,owner)


@pytest.mark.parametrize('state',['running','failed','prepared'])
def test_disappeared_owner_without_completion_never_starts_next_model(state):
    with pytest.raises(RuntimeError,match='successful terminal'):
        source_ready(state,None,dict(pid=42,start_ticks='100',command_sha256='original'))


def test_pid_reuse_is_not_mistaken_for_original_qualification():
    owner=dict(pid=42,start_ticks='100',command_sha256='original')
    with pytest.raises(RuntimeError,match='reused'):
        source_ready('completed',dict(owner,start_ticks='200'),owner)


@pytest.fixture
def buffered_qualification(tmp_path,monkeypatch):
    """Synthetic component evidence; the independent full-input audit is mocked."""
    import run_layerwise_query_efficiency as query
    previous=tmp_path/'qualification';components=tmp_path/'components'
    source=previous/'candidate';components.mkdir(parents=True);source.mkdir(parents=True)
    (components/'state.cpp').write_text('immutable test source')
    p.save(components/'source-proof.json',{'state.cpp':p.sha(components/'state.cpp')})
    for name,count in [('tests.xml',28),('integration-tests.xml',99)]:
        (components/name).write_text('<testsuite>'+''.join(
            '<testcase name="case-%s"/>'%i for i in range(count))+'</testsuite>')
    p.save(components/'component-audit.json',dict(status='passed',
        source_proof_sha256=p.sha(components/'source-proof.json'),
        tests_sha256=p.sha(components/'tests.xml'),
        integration_tests_sha256=p.sha(components/'integration-tests.xml')))
    p.save(previous/'plan.json',dict(components=str(components),
        component_audit_sha256=p.sha(components/'component-audit.json')))
    p.save(previous/'acceptance.json',dict(passed=True,cases=8))
    rows=[dict(id=str(i)) for i in range(8)]
    p.save(source/'fixture.json',dict(rows=rows,layerwise_buffered_history=True,
        layerwise_prefill_pressure=True))
    for row in rows:
        p.save(source/'formal/sequential/cases'/row['id']/'evidence.json',dict(events=[
            dict(event='pressure_selected',layers=[dict(layer=i,
                attention_state_backend='native_buffered',buffered_commits=100) for i in range(16)])]))
    audits=[]
    monkeypatch.setattr(query,'source_audit',lambda root:audits.append(root))
    return previous,components,source,audits


def test_buffered_followup_requires_full_scope_audit_and_real_backend(buffered_qualification):
    previous,components,source,audits=buffered_qualification
    assert followup.qualified(previous)==source
    assert audits==[source]
    evidence=source/'formal/sequential/cases/7/evidence.json'
    d=p.read(evidence);d['events'][0]['layers'][15]['attention_state_backend']='dictionary'
    p.save(evidence,d)
    with pytest.raises(ValueError,match='native history backend'):
        followup.qualified(previous)


@pytest.mark.parametrize('name',['state.cpp','tests.xml','integration-tests.xml'])
def test_buffered_followup_rejects_changed_component_evidence(buffered_qualification,name):
    previous,components,source,audits=buffered_qualification
    with (components/name).open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='changed|incomplete'):
        followup.qualified(previous)
    assert not audits


def test_buffered_followup_never_overrides_failed_full_input_check(buffered_qualification):
    previous,components,source,audits=buffered_qualification
    p.save(previous/'acceptance.json',dict(passed=False,cases=8))
    with pytest.raises(ValueError,match='eight exact'):
        followup.qualified(previous)
    assert not audits
