import json
from pathlib import Path

import pytest

from scripts.coordinate_terminalbench_remote import combine, validate_remote_results
from scripts.terminalbench_worker_driver import validate


def test_combined_scores_never_replace_completed_tasks():
    old=[dict(task='terminal-bench/done',trial='done__1',rewards={'reward':0})]
    new=[dict(task='terminal-bench/new',trial='new__2',rewards={'reward':1})]
    assert combine(old,new,['new','pending']) == old+new
    for rows,selected in [(old,['new']), (new,['done','new']), (new+new,['new'])]:
        with pytest.raises(ValueError):
            combine(old,rows,selected)


def test_raw_remote_score_must_match_both_original_copies(tmp_path):
    raw=dict(task_name='terminal-bench/new',trial_name='new__1',verifier_result={'rewards':{'reward':1}},exception_info=None)
    job=tmp_path/'jobs/formal-remote/new__1';job.mkdir(parents=True)
    events=tmp_path/'formal-remote-events';events.mkdir()
    (job/'result.json').write_text(json.dumps(raw));(events/'new__1.json').write_text(json.dumps(raw))
    row=dict(task='terminal-bench/new',trial='new__1',rewards={'reward':1},exception=None,status='scored',passed=True)
    assert validate_remote_results(tmp_path,[row])['new__1']['raw_sha256']
    (events/'new__1.json').write_text(json.dumps(dict(raw,verifier_result={'rewards':{'reward':0}})))
    with pytest.raises(ValueError,match='disagree'):
        validate_remote_results(tmp_path,[row])


def worker(tmp_path):
    values={'worker-provenance.json':{'copied_files':{}},
        'handoff.json':dict(original_executor_inactive=True,original_completed_evidence_preserved=True,remaining_tasks=['a']),
        'formal-candidate.json':dict(tasks=[dict(path='/worker/dataset/tasks/a')],n_attempts=1,retry=dict(max_retries=0)),
        'manifest.json':dict(launched_task_names=['a']),
        'smoke-remote-events/summary.json':dict(passed=1,errors=0)}
    for name,value in values.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    return values


def test_worker_requires_exclusive_handoff_and_never_relaunches(tmp_path):
    values=worker(tmp_path)
    assert validate(tmp_path)==['a']
    values['handoff.json']['original_executor_inactive']=False
    (tmp_path/'handoff.json').write_text(json.dumps(values['handoff.json']))
    with pytest.raises(ValueError,match='ownership'):
        validate(tmp_path)
    values['handoff.json']['original_executor_inactive']=True
    (tmp_path/'handoff.json').write_text(json.dumps(values['handoff.json']))
    (tmp_path/'jobs/formal-remote').mkdir(parents=True)
    with pytest.raises(ValueError,match='repeat'):
        validate(tmp_path)
