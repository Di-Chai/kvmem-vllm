"""Frozen original scoring inputs, nested result paths, and failure visibility."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    evaluation = importlib.import_module('graph_v2_evaluation')
    analyzer = importlib.import_module('analyze_graph_v2')
finally:
    sys.path.pop(0)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def answers(tmp_path,suite='longmemeval_s'):
    rows=[dict(id=f'group_q{i}',token_sha256=f't{i}',prompt_sha256=f'p{i}',query_sha256=f'q{i}',input_tokens=100+i) for i in range(2)]
    fixture=dict(suite=suite,rows=rows)
    save(tmp_path/'manifest.json',dict(rows=rows))
    for i,row in enumerate(rows):
        save(tmp_path/'cases'/row['id']/'result.json',dict(row,suite=suite,status='completed',
            prompt_tokens=row['input_tokens'],prefix_sha256='same',prefix_reused=bool(i),
            attention=dict(prefill_attention_samples=64,action_history_pages=0,attention_source='prefill')))
    return fixture


@pytest.mark.parametrize('mutation',['failure','interrupted','source','unexpected'])
def test_partial_analysis_never_silently_skips_failed_or_changed_answers(tmp_path,mutation):
    fixture=answers(tmp_path)
    path=tmp_path/'cases/group_q0/result.json'
    row=json.loads(path.read_text())
    if mutation=='failure': row['status']='error';save(path,row)
    elif mutation=='interrupted': path.unlink();save(path.with_name('attempt.json'),dict(started=True))
    elif mutation=='source': row['token_sha256']='different';save(path,row)
    else: save(tmp_path/'cases/unknown/result.json',row)
    with pytest.raises(ValueError): evaluation.validate_answers(tmp_path,fixture,allow_partial=True)


def test_partial_missing_answers_require_an_explicit_diagnostic_flag(tmp_path):
    fixture=answers(tmp_path)
    (tmp_path/'cases/group_q1/result.json').unlink()
    with pytest.raises(ValueError,match='Incomplete cohort'):
        evaluation.validate_answers(tmp_path,fixture)
    proof=evaluation.validate_answers(tmp_path,fixture,allow_partial=True)
    assert proof['completed']==1 and proof['missing']==['group_q1'] and not proof['complete']


def test_native_frozen_p_restores_graph_and_action_before_independent_query(tmp_path):
    from test_authority import payload, drain
    from test_core import Event
    from test_graph_lifecycle import attach
    from test_prefill_attention import options
    from test_prompt_checkpoints import ready
    from test_session_worker import worker_fixture
    from vllm_kvmem.action_attention import ActionHistory
    from vllm_kvmem.block_graph import GraphSample

    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    worker.options.session_freeze_prompt = True
    worker.pool.max_payload_bytes = 1 << 20
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    worker.runner.req_states.num_computed_tokens_np[0] = 12
    worker.runner.req_states.prefill_len.np[0] = 20
    authority.action_history = ActionHistory(options(), [10, 11, 12, 13, 14])
    authority.action_history.prefill.accept(12, 4, 4, [3], [0], [0], 0, [[1.]])
    graph = attach(authority, mode='collect')
    graph.prepare(0, 12, [GraphSample(8, (0, 8), (.5, .5), 'prefill')]).commit()
    checkpoint = worker.pool.capture(binding.namespace, 'a', authority, list(range(20)),
                                     worker.model_adapter, replace=True)
    original_window = checkpoint.window.records
    original_feedback = checkpoint.action_history
    original_rows, original_partial = graph.rows, graph.partial
    records = list(authority.records.values())
    plan = authority.memory.select({r.handle.slot:float(i) for i,r in enumerate(records)})
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    worker.selection_committed('a', 12)
    # The first independent answer contributes real output rows and feedback.
    assert authority.append(*payload(12, 4), history_begin=12, local_rope_begin=12)
    drain(authority)
    graph.prepare(12, 16, [GraphSample(12, (0, 12), (.75, .25), 'output')]).commit()
    authority.action_history.scores = {0:10.}
    authority.action_history.prefill.values.clear()
    assert graph.rows != original_rows
    worker.pool.retire(checkpoint, Event(True))
    ready(worker.pool, checkpoint)
    assert worker.pool.claim(binding.namespace, 'next-question', list(range(20))) is checkpoint
    restored = worker.pool.prepare_claim_window(checkpoint)
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert tuple(authority.memory._store.working_records()) == original_window
    assert authority.action_history.snapshot() == original_feedback
    # Epoch advances on restore; all accepted rows and provenance revert to P.
    assert graph.rows == original_rows and graph.partial == original_partial
    assert graph.accepted_through == 12
    assert worker.pool.finish_claim(checkpoint, Event(True)) is authority
    del plan, restored
    assert authority.close() and archive.close()
    assert graph.budget.used == 0


@pytest.mark.parametrize('field,value',[('prefill_attention_samples',65),('action_history_pages',1)])
def test_frozen_p_questions_must_preserve_input_stats_and_exclude_output_stats(tmp_path,field,value):
    fixture=answers(tmp_path,'memoryagentbench')
    assert evaluation.validate_answers(tmp_path,fixture)['complete']
    path=tmp_path/'cases/group_q1/result.json';row=json.loads(path.read_text())
    row['attention'][field]=value;save(path,row)
    with pytest.raises(ValueError,match='frozen-P'):
        evaluation.validate_answers(tmp_path,fixture)


def test_nested_dataset_uses_job_profile_and_frozen_evaluators(tmp_path,monkeypatch):
    root=tmp_path/'campaign';job=root/'longmemeval_s-D';dataset=job/'qualification/dataset'
    fixture=answers(dataset)
    save(job/'fixture.json',fixture)
    save(job/'profile.json',dict(kvmem={}))
    save(root/'evaluation/manifest.json',{})
    manifest=dict(evaluation_sha256='frozen',jobs=[dict(id=job.name,suite='longmemeval_s')])
    monkeypatch.setattr(analyzer,'__file__',str(root/'frozen/analyze_graph_v2.py'))
    monkeypatch.setattr(analyzer.campaign,'validate',lambda path:manifest)
    monkeypatch.setattr(evaluation,'validate_evaluation',lambda *args:dict(python=sys.executable))
    observed=[]
    def analyze(options):
        observed.append(options)
        (options.campaign/'scoring').mkdir()
    module=SimpleNamespace(analyze=analyze)
    monkeypatch.setattr(analyzer.importlib,'import_module',lambda name:module)
    analyzer.analyze(SimpleNamespace(campaign=root,job=job.name,allow_partial=False,external=False))
    assert observed[0].campaign==dataset and observed[0].config==job/'profile.json'
    assert module.FROZEN==root/'evaluation/evaluators'
    assert json.loads((dataset/'scoring/v2-analysis-audit.json').read_text())['complete']


@pytest.mark.parametrize('suite',['memoryagentbench','longmemeval_s'])
def test_reference_bundle_copies_only_declared_public_judge_evidence(tmp_path,suite):
    source=tmp_path/'source';target=tmp_path/'frozen';rows=[dict(id='case')]
    names=['comparison.json']
    for arm in ('attention','query'):
        prefix='results/'+arm+'/'
        if suite=='memoryagentbench':
            names += [prefix+'memoryagentbench/'+n for n in ('judge-answer-manifest.json','special_judgments.jsonl')]
        else:
            names += [prefix+'judge_deepseek_v41_flash/'+n for n in ('identity.json','calibration.json','requests/case.request.json')]
            names += [prefix+'longmemeval_judgments.jsonl',prefix+'cases/longmemeval_s/case/result.json']
    for name in names: save(source/name,dict(public=name))
    save(source/'unrelated.json',dict(not_a_dependency=True))
    hashes=evaluation.freeze_reference_evidence(source,target,suite,rows)
    assert set(hashes)==set(names) and not (target/'unrelated.json').exists()
    assert all(evaluation.sha(target/name)==digest for name,digest in hashes.items())


def test_evaluator_dependency_change_is_rejected_before_interpreter_or_judge(tmp_path,monkeypatch):
    path=tmp_path/'evaluation/evaluators/official.py';path.parent.mkdir(parents=True)
    path.write_text('# original')
    manifest=dict(schema='kvmem-graph-v2-evaluation-1',files_sha256={'evaluators/official.py':evaluation.sha(path)})
    save(tmp_path/'evaluation/manifest.json',manifest)
    identity=evaluation.sha(tmp_path/'evaluation/manifest.json')
    path.write_text('# changed')
    monkeypatch.setattr(evaluation,'interpreter_identity',lambda *a:pytest.fail('Interpreter/judge ran before validation'))
    with pytest.raises(ValueError,match='dependency changed'):
        evaluation.validate_evaluation(tmp_path,identity)
