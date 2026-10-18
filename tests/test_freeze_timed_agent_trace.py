import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'freeze_timed_agent_trace', Path(__file__).parents[1]/'scripts/freeze_timed_agent_trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_run(root):
    # Deliberately reverse task names and vary rewards; only manifest order is
    # allowed to select the source users.
    names = ['task-z', 'task-a']
    write(root/'manifest.json', {'tasks': [{'task_id': name} for name in names]})
    write(root/'run.json', {'tasks': {name: {'state': 'completed', 'termination': 'natural',
                                         'result': {'reward': index}}
                                     for index, name in enumerate(names)}})
    events = []
    for name in names:
        for turn, length in enumerate((100, 131072, 131080, 131100), 1):
            directory = root/'bridge'/name/f'{turn:05d}'
            request_id = f'{name}-{turn}'
            write(directory/'plan.json', dict(prompt_tokens=length))
            write(directory/'request.json', dict(messages=[]))
            write(directory/'response.json', dict(content=[]))
            usage = dict(prompt_tokens=length, completion_tokens=2)
            write(directory/'native.json', dict(id=request_id, usage=usage,
                  choices=[dict(token_ids=[11, 12], finish_reason='stop')]))
            events.extend([dict(task=name, turn=turn, event='start', prompt_tokens=length,
                                request_started_at=10.*turn),
                           dict(task=name, turn=turn, event='finish', request_id=request_id,
                                usage=usage, at=10.*turn+1)])
    (root/'bridge/requests.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return root


def test_selection_uses_manifest_order_first_length_and_complete_turns(tmp_path):
    root = source_run(tmp_path)
    chosen = trace.choose_sources(root, 2, 131072, 3)
    assert [s['task'] for s in chosen] == ['task-z', 'task-a']
    assert [[r['source_turn'] for r in s['rows']] for s in chosen] == [[2, 3, 4], [2, 3, 4]]
    with pytest.raises(ValueError, match='consecutive complete turns'):
        trace.choose_sources(root, 2, 131072, 4)


def test_no_silent_task_replacement_or_inconsistent_native_usage(tmp_path):
    root = source_run(tmp_path)
    state = trace.read(root/'run.json')
    state['tasks']['task-z']['termination'] = 'manual_stop'
    write(root/'run.json', state)
    with pytest.raises(ValueError, match='did not complete naturally'):
        trace.choose_sources(root, 1, 131072, 2)
    state['tasks']['task-z']['termination'] = 'natural'
    write(root/'run.json', state)
    native = root/'bridge/task-z/00002/native.json'
    data = trace.read(native)
    data['usage']['completion_tokens'] = 3
    write(native, data)
    with pytest.raises(ValueError, match='records disagree'):
        trace.choose_sources(root, 1, 131072, 2)


def replace_lengths(root,lengths):
    events=[json.loads(line) for line in (root/'bridge/requests.jsonl').read_text().splitlines()]
    for task in ('task-z','task-a'):
        for turn,length in enumerate(lengths,1):
            directory=root/'bridge'/task/f'{turn:05d}'
            write(directory/'plan.json',dict(prompt_tokens=length))
            native=trace.read(directory/'native.json');native['usage']['prompt_tokens']=length
            write(directory/'native.json',native)
    for row in events:
        if row['event']=='start':row['prompt_tokens']=lengths[row['turn']-1]
        else:row['usage']['prompt_tokens']=lengths[row['turn']-1]
    (root/'bridge/requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))


def test_fallback_is_opt_in_and_keeps_the_same_first_users(tmp_path):
    root=source_run(tmp_path);replace_lengths(root,[100,200,180,210])
    with pytest.raises(ValueError,match='no input >= 250'):
        trace.choose_sources(root,2,250,2)
    selected=trace.choose_sources(root,2,250,2,allow_shorter_history_fallback=True,max_history=300)
    assert [source['task'] for source in selected]==['task-z','task-a']
    assert [source['rows'][0]['source_turn'] for source in selected]==[2,2]
    for source in selected:
        assert source['selection']['kind']=='shorter_history_fallback'
        assert source['selection']['source_logical_initial_tokens']==200
        assert source['selection']['fallback_reason']
        assert source['selection']['original_criterion']==dict(history_threshold=250,
            max_initial_history=300,consecutive_complete_turns=2,start='earliest qualifying logical prompt')


def test_fallback_ranks_largest_complete_input_then_earliest_turn_without_timing(tmp_path):
    root=source_run(tmp_path);replace_lengths(root,[100,200,200,210])
    selected=trace.choose_sources(root,1,250,2,allow_shorter_history_fallback=True)
    assert selected[0]['rows'][0]['source_turn']==2
    # Timing and reward can change by arbitrary amounts without choosing a
    # different user or a later equal-length source.
    state=trace.read(root/'run.json')
    state['tasks']['task-z']['result']['reward']=-1000
    state['tasks']['task-a']['result']['reward']=1000
    write(root/'run.json',state)
    events=[json.loads(line) for line in (root/'bridge/requests.jsonl').read_text().splitlines()]
    for row in events:
        if row['event']=='start':row['request_started_at']=1000000-row['turn']
        else:row['at']=row['turn']*.001
    (root/'bridge/requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
    again=trace.choose_sources(root,1,250,2,allow_shorter_history_fallback=True)
    assert again[0]['task']=='task-z' and again[0]['rows'][0]['source_turn']==2


def test_preferred_earliest_complete_trace_takes_priority_over_fallback(tmp_path):
    root=source_run(tmp_path);replace_lengths(root,[100,220,200,210])
    selected=trace.choose_sources(root,1,200,2,allow_shorter_history_fallback=True,max_history=230)
    assert selected[0]['rows'][0]['source_turn']==2
    assert selected[0]['selection']['kind']=='preferred'
    assert selected[0]['selection']['fallback_reason'] is None


def test_logical_initial_ceiling_is_applied_before_fallback_without_clipping(tmp_path):
    root=source_run(tmp_path);replace_lengths(root,[100,240,190,245])
    selected=trace.choose_sources(root,1,200,2,allow_shorter_history_fallback=True,max_history=230)
    assert selected[0]['rows'][0]['source_turn']==3
    assert selected[0]['rows'][0]['plan']['prompt_tokens']==190
    assert trace.read(root/'bridge/task-z/00002/plan.json')['prompt_tokens']==240
    assert any('maximum' in row['reason'] for row in selected[0]['skipped_incomplete_candidates'])


def test_fallback_never_replaces_incomplete_first_task_or_hides_bad_selected_records(tmp_path):
    root=source_run(tmp_path);replace_lengths(root,[100,200,180,210])
    (root/'bridge/task-z/00002/response.json').unlink()
    (root/'bridge/task-z/00003/response.json').unlink()
    with pytest.raises(ValueError,match='task-z.*no permitted shorter'):
        trace.choose_sources(root,2,250,2,allow_shorter_history_fallback=True)
    write(root/'bridge/task-z/00002/response.json',dict(content=[]))
    write(root/'bridge/task-z/00003/response.json',dict(content=[]))
    native=trace.read(root/'bridge/task-z/00002/native.json')
    native['usage']['completion_tokens']=3
    write(root/'bridge/task-z/00002/native.json',native)
    with pytest.raises(ValueError,match='records disagree'):
        trace.choose_sources(root,2,250,2,allow_shorter_history_fallback=True)


def test_canonical_initial_ceiling_fails_without_replacing_or_shortening():
    agents=[dict(source_task='first-user',initial_tokens=[1,2,3,4])]
    trace.validate_initial_history(agents,4)
    with pytest.raises(ValueError,match='canonical initial history.*no clipping or replacement'):
        trace.validate_initial_history(agents,3)
    assert agents==[dict(source_task='first-user',initial_tokens=[1,2,3,4])]


def protocol(*, request):
    messages = request['messages']
    text = b'SYS\n'
    spans = []
    for index, message in enumerate(messages[1:], 1):
        begin = len(text)
        header = ('<'+message['role']+'>').encode()
        text += header + message['content'].encode() + b'</end>\n'
        spans.append(dict(index=index, role=message['role'], begin=begin, end=len(text)))
    text += b'<assistant><think>'
    return dict(messages=messages, prompt=text.decode(), spans=spans, control_end=4,
                tools=[], pins=[], query_index=1)


def identity(message):
    return json.dumps(message, sort_keys=True)


def transition():
    original = [dict(role='system', content='S'), dict(role='user', content='问')]
    previous = dict(request=dict(messages=original), response=dict(content='SOURCE-ANSWER'))
    previous['rendered'] = protocol(request=previous['request'])
    following = original + [dict(role='assistant', content='SOURCE-ANSWER'),
                            dict(role='tool', content='工具结果🙂')]
    current = dict(request=dict(messages=following))
    current['rendered'] = protocol(request=current['request'])
    return previous, current


def test_external_suffix_uses_utf8_byte_spans_excluding_source_answer():
    previous, current = transition()
    suffix, proof = trace.external_suffix(previous, current, protocol, identity)
    assert suffix == '\n<tool>工具结果🙂</end>\n<assistant><think>'
    assert 'SOURCE-ANSWER' not in suffix
    assert proof['utf8_bytes'] == len(suffix.encode())
    assert proof['message_indices'] == [3]
    assert proof['source_byte_end'] == len(current['rendered']['prompt'].encode())


@pytest.mark.parametrize('change, reason', [
    ('history', 'historical message identity'),
    ('assistant', 'assistant identity'),
    ('extra-answer', 'another source assistant'),
    ('query', 'Query changed'),
    ('policy', 'mandatory content'),
])
def test_nonappend_or_unmodeled_plan_changes_are_explicit_errors(change, reason):
    previous, current = transition()
    current = copy.deepcopy(current)
    if change == 'history':
        current['request']['messages'][1]['content'] = 'changed'
    elif change == 'assistant':
        current['request']['messages'][2]['content'] = 'different answer'
    elif change == 'extra-answer':
        current['request']['messages'].append(dict(role='assistant', content='old answer'))
    current['rendered'] = protocol(request=current['request'])
    if change == 'query':
        current['rendered']['query_index'] = 3
    if change == 'policy':
        boundary = current['rendered']['spans'][-1]['begin']
        current['rendered']['pins'] = [dict(begin=boundary, end=boundary+2, reason=3)]
    with pytest.raises(ValueError, match=reason):
        trace.external_suffix(previous, current, protocol, identity)


class Planner:
    protocol = staticmethod(protocol)
    encode = staticmethod(lambda text: list(text.encode()))

    def prepare(self, request, state):
        rendered = protocol(request=request)
        return self.encode(rendered['prompt']), {'kvmem_query_span': [4, 8]}, rendered, [], '', False, {}


def test_generation_lengths_and_waits_are_preserved_without_source_output_substitution():
    previous, current = transition()
    rows = []
    for i, row in enumerate((previous, current)):
        row.update(source_turn=62+i, plan=dict(prompt_tokens=131072+i),
                   start=dict(request_started_at=10.+i*12.), finish=dict(at=20.+i*12.),
                   native=dict(usage=dict(completion_tokens=32000 if i == 0 else 91),
                               choices=[dict(finish_reason='length' if i == 0 else 'stop')]))
        rows.append(row)
    bridge = SimpleNamespace(Trajectory=lambda: None, message_identity=identity)
    source = dict(task='source-task', rows=rows, skipped_incomplete_candidates=[])
    plan = trace.make_agent_plan(source, Planner(), bridge)
    assert [r['output_tokens'] for r in plan['turns']] == [32000, 91]
    assert plan['turns'][0]['tool_tokens'] == []
    assert plan['turns'][1]['tool_wait_seconds'] == 2.
    assert b'SOURCE-ANSWER' not in bytes(plan['turns'][1]['tool_tokens'])
    assert 'output_ids' not in plan['turns'][0]
    rows[1]['start']['request_started_at'] = 19.
    with pytest.raises(ValueError, match='negative/nonfinite tool wait'):
        trace.make_agent_plan(source, Planner(), bridge)
