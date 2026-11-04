import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dsh_query_epoch import apply_query_provenance
from scripts.terminalbench_query import TaskQuery, digest, register_user_query


@pytest.fixture
def hooked_chat(tmp_path, monkeypatch):
    pytest.importorskip('harbor')
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.chat import Chat
    from harbor.llms.base import LLMResponse
    from scripts.terminalbench_query import install
    # Register restoration of the methods that install() wraps in place.
    monkeypatch.setattr(Chat, 'chat', Chat.chat)
    monkeypatch.setattr(Terminus2, '_run_agent_loop', Terminus2._run_agent_loop)
    monkeypatch.setattr(Chat, '_kvmem_task_query_installed', False, raising=False)
    install({}, tmp_path/'query.jsonl')

    class Model:
        def __init__(self):
            self.responses = []
            self.calls = []

        async def call(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return LLMResponse(content=response)

    model = Model()
    chat = Chat(model)
    chat._kvmem_task_query = TaskQuery('session', 'task', 'instruction', 'instruction')
    return chat, model, tmp_path/'query.jsonl'


def test_failed_nested_output_recovery_does_not_leave_a_phantom_query(hooked_chat):
    from harbor.llms.base import OutputLengthExceededError
    chat, model, journal = hooked_chat

    async def run():
        model.responses = ['initial answer', OutputLengthExceededError('length'),
                           RuntimeError('incomplete stream'), 'outer retry answer']
        await chat.chat('instruction')
        with pytest.raises(OutputLengthExceededError):
            await chat.chat('terminal feedback')
        # Exactly the pinned Terminus output-limit handler's history update.
        chat.messages.extend([dict(role='user', content='terminal feedback'),
                              dict(role='assistant', content='truncated answer')])
        committed = copy.deepcopy(chat.messages)
        with pytest.raises(RuntimeError, match='incomplete stream'):
            await chat.chat('request a shorter response')
        assert chat.messages == committed
        await chat.chat('terminal feedback')  # Existing outer retry's prompt.
        assert chat.messages[:-2] == committed
        assert chat.messages[-2:] == [dict(role='user', content='terminal feedback'),
                                     dict(role='assistant', content='outer retry answer')]

    asyncio.run(run())
    proofs = [c['extra_body']['kvmem_query_provenance'] for c in model.calls]
    assert len({p['query_message_id'] for p in proofs}) == 1
    assert len({p['user_messages_sha256'] for p in proofs[-2:]}) == 2
    assert any(json.loads(line)['event'] == 'query_attempt_rolled_back'
               for line in journal.read_text().splitlines())


@pytest.mark.parametrize('failure', [RuntimeError('transport'), asyncio.CancelledError()])
def test_uncommitted_initial_request_can_retry_without_changing_messages(hooked_chat, failure):
    chat, model, _ = hooked_chat

    async def run():
        model.responses = [failure, 'answer']
        with pytest.raises(type(failure)):
            await chat.chat('instruction')
        assert chat.messages == [] and chat._kvmem_task_query.seen == []
        await chat.chat('instruction')

    asyncio.run(run())
    assert model.calls[0]['extra_body'] == model.calls[1]['extra_body']


def test_failed_real_user_update_retains_its_explicit_source(hooked_chat):
    chat, model, _ = hooked_chat

    async def run():
        model.responses = ['first answer', RuntimeError('transport'), 'second answer']
        await chat.chat('instruction')
        register_user_query(chat, 'new request', 'human-message-2')
        with pytest.raises(RuntimeError):
            await chat.chat('new request')
        assert chat._kvmem_task_query.pending_user is not None
        await chat.chat('new request')
        assert chat._kvmem_task_query.pending_user is None

    asyncio.run(run())
    first, failed, retried = [c['extra_body']['kvmem_query_provenance'] for c in model.calls]
    assert failed == retried and first['query_message_id'] != retried['query_message_id']


def test_unknown_history_edit_still_fails_after_transport_rollback(hooked_chat):
    chat, model, _ = hooked_chat

    async def run():
        model.responses = ['answer', RuntimeError('transport')]
        await chat.chat('instruction')
        with pytest.raises(RuntimeError):
            await chat.chat('terminal feedback')
        chat.messages[0]['content'] = 'edited task instruction'
        with pytest.raises(ValueError, match='changed without source metadata'):
            await chat.chat('terminal feedback')
        assert len(model.calls) == 2

    asyncio.run(run())


def test_failed_call_cannot_authorize_a_mutated_committed_history(hooked_chat):
    chat, model, _ = hooked_chat

    async def run():
        model.responses = ['answer']
        await chat.chat('instruction')

        async def mutate_and_fail(**kwargs):
            chat.messages[0]['content'] = 'unknown replacement instruction'
            raise RuntimeError('transport')

        model.call = mutate_and_fail
        with pytest.raises(ValueError, match='failed call changed committed history'):
            await chat.chat('terminal feedback')

    asyncio.run(run())


def test_task_source_keeps_query_across_feedback_without_changing_messages_or_tokens():
    source = TaskQuery('session', 'official-task', 'solve task', 'Task: solve task\nTerminal: $')
    messages = [dict(role='user', content='Task: solve task\nTerminal: $')]
    first = source.proof(messages)
    messages += [dict(role='assistant', content='run check'), dict(role='user', content='check failed')]
    original = copy.deepcopy(messages)
    proof = source.proof(messages)
    assert source.proof(messages) == proof  # Transport retry.
    body = dict(prompt_cache_key='session', messages=messages, kvmem_query_provenance=proof)
    plan = dict(kvmem_query_span=[80, 100], kvmem_reselect='auto')
    tokens = list(range(120))
    prepared = (tokens, plan, dict(messages=messages, spans=[dict(index=0, role='user', begin=0,
                content_begin=10, content_end=30, end=40)]), None, None, True, {(0, 10):5, (0, 30):20})
    apply_query_provenance(body, prepared, required=True)
    assert plan['kvmem_query_span'] == [5, 20]
    assert proof['query_message_id'] == first['query_message_id']
    assert messages == original and tokens == list(range(120))


def test_real_user_metadata_updates_query_even_when_text_is_identical():
    source = TaskQuery('session', 'task', 'instruction', 'instruction')
    chat = SimpleNamespace(_kvmem_task_query=source)
    messages = [dict(role='user', content='instruction')]
    first = source.proof(messages)
    register_user_query(chat, 'instruction', 'human-message-2')
    messages += [dict(role='assistant', content='answer'), dict(role='user', content='instruction')]
    second = source.proof(messages)
    assert second['query_user_ordinal'] == 1
    assert second['query_message_id'] != first['query_message_id']
    messages += [dict(role='assistant', content='check'), dict(role='user', content='tool output')]
    assert source.proof(messages)['query_message_id'] == second['query_message_id']


@pytest.mark.parametrize('damage', ['initial', 'edit', 'insert', 'human_mismatch'])
def test_unknown_transcript_changes_fail_closed(damage):
    source = TaskQuery('session', 'task', 'instruction', 'instruction')
    messages = [dict(role='user', content='instruction')]
    if damage == 'initial':
        messages[0]['content'] = 'different first user'
    else:
        source.proof(messages)
        if damage == 'edit':
            messages[0]['content'] = 'changed original instruction'
        if damage == 'insert':
            messages += [dict(role='user', content='unknown human'), dict(role='user', content='tool')]
        if damage == 'human_mismatch':
            source.real_user('actual human request', 'new-source')
            messages.append(dict(role='user', content='different human text'))
    with pytest.raises(ValueError):
        source.proof(messages)


def test_instruction_and_concurrent_session_identity_are_explicit():
    with pytest.raises(ValueError, match='verified task instruction'):
        TaskQuery('session', 'task', 'real instruction', 'arbitrary first user')
    a = TaskQuery('one', 'task', 'instruction', 'instruction')
    b = TaskQuery('two', 'task', 'instruction', 'instruction')
    msg = [dict(role='user', content='instruction')]
    assert a.proof(msg)['query_message_id'] != b.proof(msg)['query_message_id']


def test_real_harbor_hook_binds_instruction_file_and_preserves_call(tmp_path, monkeypatch):
    pytest.importorskip('harbor')
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.chat import Chat
    from scripts.terminalbench_query import install
    task = tmp_path/'task'
    (task/'environment').mkdir(parents=True)
    instruction = task/'instruction.md'
    instruction.write_text('<!-- canary fixture -->\n\nSolve the task.\n')
    calls = []

    async def original_chat(self, prompt, logging_path=None, **kwargs):
        calls.append((prompt, copy.deepcopy(self.messages), kwargs))
        self._messages.extend([dict(role='user', content=prompt), dict(role='assistant', content='ok')])
        return 'answer'

    async def original_loop(self, initial_prompt, chat, original_instruction=''):
        await chat.chat(initial_prompt, extra_body=dict(top_k=20))
        await chat.chat('terminal feedback')
        register_user_query(chat, 'A real new instruction.', 'human-2')
        await chat.chat('A real new instruction.')

    monkeypatch.setattr(Chat, 'chat', original_chat)
    monkeypatch.setattr(Terminus2, '_run_agent_loop', original_loop)
    monkeypatch.setattr(Chat, '_kvmem_task_query_installed', False, raising=False)
    install({str(task): dict(task_id='task', instruction_sha256=digest(instruction.read_text()))}, tmp_path/'audit.jsonl')
    agent = SimpleNamespace(_session=SimpleNamespace(environment=SimpleNamespace(environment_dir=task/'environment')),
                            _session_id='task__agent', options=SimpleNamespace(enable_summarize=False))
    chat = Chat(None)
    asyncio.run(Terminus2._run_agent_loop(agent, 'Task: Solve the task.\nTerminal: $', chat, 'Solve the task.\n'))
    assert [x[0] for x in calls] == ['Task: Solve the task.\nTerminal: $', 'terminal feedback', 'A real new instruction.']
    first, second, third = [x[2]['extra_body']['kvmem_query_provenance'] for x in calls]
    assert first['query_message_id'] == second['query_message_id'] != third['query_message_id']
    assert calls[0][2]['extra_body']['top_k'] == 20
    assert calls[0][2]['extra_body']['prompt_cache_key'] == 'task__agent'
    instruction.write_text('Tampered task')
    with pytest.raises(ValueError, match='frozen instruction manifest'):
        asyncio.run(Terminus2._run_agent_loop(agent, 'Task: Tampered task', Chat(None), 'Tampered task'))


def test_freeze_uses_current_package_and_rejects_known_stale_native(tmp_path):
    from scripts.run_terminalbench_suite import freeze_runtime
    repo = tmp_path/'repo'
    package = repo/'src/vllm_kvmem'
    package.mkdir(parents=True)
    (package/'current.py').write_text('CURRENT = True\n')
    (repo/'csrc').mkdir(); (repo/'third_party/qw3').mkdir(parents=True)
    cpp = repo/'csrc/current.cpp'; cpp.write_text('// current\n')
    binary = package/'_native.test.so'; binary.write_bytes(b'current-built-native')
    os.utime(binary, ns=(cpp.stat().st_mtime_ns+1000000,)*2)
    scripts = ('dsh_bridge.py', 'dsh_query_epoch.py', 'run_terminalbench_job.py', 'run_terminalbench_suite.py',
               'run_agent_handoff_queue.py', 'terminalbench_view.py', 'finalize_terminalbench.py',
               'terminalbench_policy.py', 'terminalbench_network.py', 'terminalbench_harness.py',
               'terminalbench_pty.py', 'terminalbench_tool_bundle.py', 'terminalbench_verifier_cache.py',
               'terminalbench_query.py', 'terminalbench_replay.py', 'terminalbench_gpu_probe.py',
               'qualify_thinking_gc_preflight.py', 'gpu_efficiency_monitor.py')
    for name in scripts:
        p = repo/'scripts'/name; p.parent.mkdir(exist_ok=True); p.write_text('# current '+name)
    p = repo/'benchmarks/deepswe/claude_bridge.py'; p.parent.mkdir(parents=True); p.write_text('# current bridge')
    result = freeze_runtime(repo, tmp_path/'frozen')
    assert result['origin'] == 'current-working-tree'
    assert (tmp_path/'frozen/src/vllm_kvmem/current.py').read_text() == 'CURRENT = True\n'
    assert (tmp_path/'frozen/src/vllm_kvmem/_native.test.so').read_bytes() == binary.read_bytes()
    assert (tmp_path/'frozen/scripts/terminalbench_pty.py').exists()
    os.utime(cpp, ns=(binary.stat().st_mtime_ns+1000000,)*2)
    with pytest.raises(ValueError, match='predates'):
        freeze_runtime(repo, tmp_path/'stale')


def test_offline_trace_replay_quantifies_avoidable_query_invalidation(tmp_path):
    from scripts.terminalbench_replay import replay
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    instruction = tmp_path/'dataset/tasks/task/instruction.md'
    instruction.parent.mkdir(parents=True); instruction.write_text('real task')
    (tmp_path/'bridge').mkdir()
    (tmp_path/'bridge/requests.jsonl').write_text(json.dumps(dict(event='session_opened',owner='owner',session_id='trial__agent'))+'\n')
    (tmp_path/'formal-events').mkdir()
    (tmp_path/'formal-events/events.jsonl').write_text(json.dumps(dict(trial='trial',task='terminal-bench/task'))+'\n')
    messages = []
    for i, (prompt, query) in enumerate([('Task: real task', [5,20]), ('tool feedback', [43,50]), ('new feedback', [73,80])], 1):
        messages.append(dict(role='user', content=prompt))
        path = tmp_path/'bridge/owner'/f'{i:05d}'
        save(path/'request.json', dict(messages=messages))
        save(path/'plan.json',dict(prompt_tokens=30*i,kvmem_query_span=query,exact_output_prefix_reused=i>1))
        save(path/'accepted.json',dict(token_ids=list(range(10))))
        messages.append(dict(role='assistant', content='answer'))
    result = replay(tmp_path)
    assert result['counts'] == dict(requests=3,legacy_query_changed=2,task_query_replay_suppressible=2,
                                    legacy_replay_overlaps_previous_accepted=2)
    assert result['traces'][0]['task_query_identities'] == 1
    assert result['traces'][0]['legacy_unique_queries'] == 3


def test_preparation_retains_profile_deadlines_and_freezes_explicit_subset(tmp_path, monkeypatch):
    import array
    from scripts import run_terminalbench_suite as suite
    repo, dataset, root = tmp_path/'repo', tmp_path/'dataset', tmp_path/'run'
    original = repo/'artifacts/rolling-long-output-20261026-r1'
    original.mkdir(parents=True)
    (original/'input.bin').write_bytes(array.array('i',range(10000)).tobytes())
    protocol = repo/'artifacts/deepswe-first16-20261015-r16/protocol/protocol.so'
    protocol.parent.mkdir(parents=True); protocol.write_bytes(b'protocol')
    for i in range(89):
        task=dataset/'tasks'/f'task-{i}'
        task.mkdir(parents=True)
        (task/'task.toml').write_text('[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 30\n[environment]\ncpus = 1\n')
        (task/'instruction.md').write_text('Task '+str(i))
    profile=dict(port=19000,max_num_seqs=3,served_model_name='current-8k',max_model_len=1000000,
                 model='model',tokenizer='tokenizer',kvmem=dict(select_budget=8192,gen_budget=4096,
                                                             session_capacity=3))
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    monkeypatch.setattr(suite,'REPO',repo)
    monkeypatch.setattr(suite,'copy_download_cache',lambda source,target:target.mkdir(parents=True))
    monkeypatch.setattr(suite,'cache_compose_override',lambda path:{})
    def freeze(repo,path):
        path.mkdir(parents=True);(path/'current.py').write_text('CURRENT=True')
        return dict(origin='current-working-tree',native={'current.so':'sha'})
    monkeypatch.setattr(suite,'freeze_runtime',freeze)
    monkeypatch.setattr(suite,'task_query_manifest',lambda root:(dict(tasks={},upstream_sources={}),dict(recorder_transport=True)))
    def output(command,**kwargs):
        if '-c' in command:return b'{"harbor":"test"}'
        return 'commit\n'
    monkeypatch.setattr(suite.subprocess,'check_output',output)
    monkeypatch.setattr(suite.subprocess,'run',lambda *args,**kwargs:None)
    suite.prepare(root,dataset,profile_path=path,task_names=['task-2','task-7'])
    assert json.loads((root/'profile.json').read_text())==profile
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['max_concurrency']==2 and manifest['launched_task_names']==['task-2','task-7']
    assert manifest['subset'] and manifest['harness_repair']['recorder_transport']
    assert manifest['agent_time_limit']=='official'
    assert (root/'dataset/tasks/task-2/task.toml').read_bytes()==(dataset/'tasks/task-2/task.toml').read_bytes()
    assert len((root/'warmup-tokens.bin').read_bytes())==4096*4
    job=json.loads((root/'job.json').read_text())
    assert job['n_concurrent_trials']==2 and len(job['tasks'])==2


def test_fixed_gpu_probe_sends_identical_inputs_and_releases_all_sessions(tmp_path, monkeypatch):
    import httpx
    from scripts import terminalbench_gpu_probe as probe
    fixture=tmp_path/'fixture.json'
    fixture.write_text(json.dumps(dict(owners=[dict(task='one',prefix=list(range(64)),query=[5,20],
        additions=[dict(tokens=[91,92,93,94],query_offset=[1,3])])],output_token=830,output_tokens=2)))
    calls=[]
    async def upstream(request):
        data=json.loads(request.content);calls.append(data)
        return httpx.Response(200,json=dict(id='reply-'+str(len(calls)),choices=[dict(token_ids=[830,830])]))
    real_client=httpx.AsyncClient
    monkeypatch.setattr(probe.httpx,'AsyncClient',lambda **kwargs:real_client(**kwargs,transport=httpx.MockTransport(upstream)))
    asyncio.run(probe.exercise('http://local','model',fixture,tmp_path/'result'))
    assert len(calls)==6
    model=[row for row in calls if 'kv_transfer_params' not in row]
    assert model[0]['prompt']==model[2]['prompt'] and model[1]['prompt']==model[3]['prompt']
    assert model[1]['kvmem_query_span']==[67,69]
    assert model[3]['kvmem_query_span']==[5,20]
    result=json.loads((tmp_path/'result/summary.json').read_text())
    assert result['passed'] and result['identical_input_tokens']
