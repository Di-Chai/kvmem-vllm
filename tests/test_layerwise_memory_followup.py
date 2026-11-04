from copy import deepcopy
from pathlib import Path

import pytest

import recall_comparison_protocol as p
import run_layerwise_memory_followup as followup


def test_grading_reuses_every_original_option_except_new_artifact_paths(tmp_path):
    old=tmp_path/'old.py';old.write_text('frozen official grading')
    campaign=tmp_path/'campaign';target=campaign/'runtime-v1/scripts/old.py'
    target.parent.mkdir(parents=True);target.write_bytes(old.read_bytes())
    command=['grader-python','-B',str(old),'--campaign','old-campaign','--output','old-output',
             '--credential-file','original-secret-path','--judge-workers','4']
    template=dict(command=command)
    result=followup.grading_command(template,campaign,tmp_path/'output')
    assert result[:3]==['grader-python','-B',str(target)]
    assert result[3:7]==['--campaign',str(campaign),'--output',str(tmp_path/'output')]
    assert result[7:]==command[7:]
    assert template['command']==command
    target.write_text('changed rubric')
    with pytest.raises(ValueError,match='grading implementation changed'):
        followup.grading_command(template,campaign,tmp_path/'other')


def test_existing_attempt_log_prevents_a_new_child_before_launch(tmp_path,monkeypatch):
    (tmp_path/'formal.log').write_text('original evidence')
    monkeypatch.setattr(followup.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(FileExistsError):followup.launch(tmp_path,'formal',['unused'],tmp_path,{})
    assert (tmp_path/'formal.log').read_text()=='original evidence'


@pytest.fixture
def waiting_queue(tmp_path,monkeypatch):
    import layerwise_efficiency_gate as gate
    root=tmp_path/'next';root.mkdir();previous=tmp_path/'previous';previous.mkdir()
    p.save(root/'plan.json',dict(performance=str(previous),source=str(tmp_path/'runtime'),
        repo=str(tmp_path),source_owner=dict(pid=123,start_ticks='45',command_sha256='owned')))
    p.save(previous/'status.json',dict(status='completed'))
    p.save(previous/'performance-comparison.json',dict(goal_met=False))
    p.save(tmp_path/'value-recall-active.json',dict(queue_root=str(previous)))
    monkeypatch.setattr(followup,'verify',lambda *a:None)
    monkeypatch.setattr(followup,'identity',lambda *a:None)
    monkeypatch.setattr(followup.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
    monkeypatch.setattr(gate,'create',lambda *a:pytest.fail('must not accept'))
    return root,previous


def test_below_target_measurement_never_prepares_or_runs_memory(waiting_queue):
    root,previous=waiting_queue
    followup.run(root)
    assert p.read(root/'status.json')['status']=='not_run'
    assert p.read(root/'status.json')['memory_started'] is False
    assert not (root/'efficiency-gate.json').exists()


def test_foreign_gpu_scheduling_owner_is_not_overwritten(waiting_queue):
    root,previous=waiting_queue
    pointer=root.parent/'value-recall-active.json'
    p.save(pointer,dict(queue_root='different-experiment'))
    with pytest.raises(RuntimeError,match='Another experiment'):followup.run(root)
    assert p.read(pointer)==dict(queue_root='different-experiment')


def test_terminal_failure_cannot_trigger_memory(waiting_queue):
    root,previous=waiting_queue
    p.save(previous/'status.json',dict(status='failed'))
    with pytest.raises(RuntimeError,match='successful terminal'):followup.run(root)


def test_report_pass_alone_cannot_bypass_bound_efficiency_gate(waiting_queue,monkeypatch):
    import layerwise_efficiency_gate as gate
    root,previous=waiting_queue
    p.save(previous/'performance-comparison.json',dict(goal_met=True))
    def reject(*args):raise ValueError('different runtime')
    monkeypatch.setattr(gate,'create',reject)
    with pytest.raises(ValueError,match='different runtime'):followup.run(root)
    assert p.read(root.parent/'value-recall-active.json')['queue_root']==str(previous)


def test_explicit_selection_keeps_failed_target_and_passes_bound_permission_to_prepare(waiting_queue,monkeypatch):
    from contextlib import nullcontext
    import layerwise_efficiency_gate as gate
    import run_layerwise_query_efficiency as qualification
    import run_recall_comparison as lifecycle_guards
    import benchmark_block_graph as lifecycle
    root,previous=waiting_queue
    plan=p.read(root/'plan.json');authorization=root/'user-authorization.json'
    stage=dict(suite='longmemeval_s',qualification=True,cases=2,root=str(root/'longmemeval_s-qualification'))
    plan.update(authorization_file=str(authorization),handoff_root=str(previous),stages=[stage])
    p.save(root/'plan.json',plan);p.save(previous/'child.json',dict(pid=456))
    released=[];accepted=[];commands=[]
    monkeypatch.setattr(qualification,'released',lambda path:released.append(path))
    monkeypatch.setattr(lifecycle,'gpu_processes',lambda:set())
    monkeypatch.setattr(lifecycle_guards,'gpu_lock',lambda path:nullcontext())
    def accept(source,performance,permission):
        accepted.append((source,performance,permission))
        return dict(efficiency_accepted=False,memory_run_authorized=True)
    monkeypatch.setattr(gate,'create',accept)
    class Prepared(Exception):pass
    def launch(*args):
        commands.append(args[2]);raise Prepared()
    monkeypatch.setattr(followup,'launch',launch)
    with pytest.raises(Prepared):followup.run(root)
    assert released==[previous,previous]
    assert accepted==[(Path(plan['source']),previous,str(authorization))]
    assert p.read(root/'efficiency-gate.json')['efficiency_accepted'] is False
    command=commands[0]
    assert command[command.index('--efficiency-authorization')+1]==str(authorization)
    assert '--qualification' in command
    assert not p.read(previous/'performance-comparison.json')['goal_met']


@pytest.fixture
def completed_stage(tmp_path,monkeypatch):
    import run_layerwise_memory as memory
    stage=dict(suite='memoryagentbench',qualification=True,cases=3)
    fixture=dict(stage);calls=[]
    monkeypatch.setattr(memory,'verify',lambda root:(deepcopy(fixture),{}))
    monkeypatch.setattr(memory,'completed_memory_qualification',lambda *args:calls.append(args))
    p.save(tmp_path/'status.json',dict(status='completed'))
    p.save(tmp_path/'execution-audit.json',dict(status='passed',cases=3,regenerated_completed_answers=0))
    p.save(tmp_path/'run/cleanup.json',dict(gpu_processes=[],remaining_owned_groups=[]))
    return tmp_path,stage,calls


def test_mab_stage_requires_original_warm_restore_qualification(completed_stage):
    root,stage,calls=completed_stage
    followup.check_stage(root,stage)
    assert calls==[(root,'memoryagentbench')]


@pytest.mark.parametrize('file,key,value',[
    ('execution-audit.json','regenerated_completed_answers',1),
    ('execution-audit.json','cases',2),
    ('execution-audit.json','status','failed'),
    ('status.json','status','running'),
    ('run/cleanup.json','gpu_processes',[999]),
    ('run/cleanup.json','remaining_owned_groups',[999]),
])
def test_incomplete_resampled_or_unreleased_stage_never_advances(completed_stage,file,key,value):
    root,stage,calls=completed_stage
    d=p.read(root/file);d[key]=value;p.save(root/file,d)
    with pytest.raises(ValueError,match='original cohort'):followup.check_stage(root,stage)
    assert not calls
