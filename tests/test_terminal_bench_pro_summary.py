import json

from diagnostics.summarize_terminal_bench_pro_gc import summarize


def test_summary_excludes_cleanup_and_foreign_frees_and_never_publishes_transcript(tmp_path):
    trial=tmp_path/'trial'
    bridge=trial/'agent/bridge'
    namespace='owned'
    state=dict(plan=dict(namespace=[namespace,namespace],source='verified_dsh_write_artifact',span_id='file',
        source_begin=32,source_end=64,relative_path='solver.py'),candidate_handles=[[4,8]],
        archive_request_id='archive',flushed=True,complete=True,observed=True,error=None,
        events=[dict(kind='source_record_released',during_live_scope=True,handle=[4,8],history_begin=32),
                dict(kind='source_record_released',during_live_scope=False,handle=[4,8],history_begin=32)])
    foreign=dict(state,plan=dict(state['plan'],namespace=['foreign','foreign']))
    for turn in ('00001','00002'):
        path=bridge/namespace/turn;path.mkdir(parents=True)
        (path/'accepted.json').write_text('{}')
        (path/'native.json').write_text(json.dumps(dict(usage=dict(prompt_tokens=128,completion_tokens=16))))
        (path/'artifact-gc-state.json').write_text(json.dumps(dict(results=[[state,foreign]])))
    (trial/'result.json').write_text(json.dumps(dict(finished_at='2026-10-24T00:00:00Z',verifier_result=dict(rewards=dict(reward=1)),
        exception_info=dict(exception_type='TestFailure',exception_message='PRIVATE_EXCEPTION',exception_traceback='PRIVATE_TRACE'))))
    (trial/'agent/gc-before-cleanup.json').write_text(json.dumps(dict(results=[[state,foreign]])))
    (trial/'agent/owner-cleanup.json').write_text(json.dumps([dict(status=200,response='PRIVATE_RELEASE')]))
    turns=trial/'agent/dsh/task/turns';turns.mkdir(parents=True)
    (turns/'000001.result.json').write_text(json.dumps(dict(text='PRIVATE_MODEL_TEXT',reason=dict(kind='completed'),
        started_at='2026-10-24T00:00:00Z',finished_at='2026-10-24T00:00:05Z')))
    value=summarize(tmp_path,task='original-task',mode='delete')
    assert value['live_source_cpu_completions']==1 and value['retired_source_tokens']==32
    assert len(value['candidate_source_spans'])==1 and value['effective_gc_task_continuation']
    assert value['dsh_turn_seconds']==5 and value['dsh_reason_kind']=='completed'
    assert value['owner_cleanup']==[dict(status=200)]
    assert 'PRIVATE_' not in json.dumps(value)
