"""CPU-only system-frame pins, exact prefix restoration and fake upstream guards."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from starlette.requests import Request

from benchmarks.deepswe.claude_bridge import Planner, Trajectory, backend_request
import scripts.dsh_bridge as bridge
from scripts.dsh_model_protocol import TokenizerProtocol
from test_dsh_model_protocol import wire_body


@pytest.fixture(scope='module')
def assets():
    path = os.environ.get('KVMEM_QWEN38_TOKENIZER')
    if not path:
        pytest.skip('Set KVMEM_QWEN38_TOKENIZER to audited local Qwen3.8 assets')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True), path


def protocol(assets, budget=8192, **kwargs):
    tokenizer, path = assets
    return TokenizerProtocol(tokenizer, path, enable_thinking=False,
        pin_system=True, select_budget=budget, gen_budget=4096, **kwargs)


def with_proof(body):
    body = copy.deepcopy(body)
    hashes = [hashlib.sha256(m['content'].encode()).hexdigest()
              for m in body['messages'] if m['role'] == 'user']
    body['kvmem_query_provenance'] = dict(schema_version=1,
        session_id=body['prompt_cache_key'], query_user_ordinal=0, query_message_id='human-query',
        user_messages_sha256=hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest())
    return body


def prepared_request(current, body):
    prepared = Planner(current, current.tokenizer, 4096).prepare(
        bridge.to_anthropic(body), Trajectory())
    bridge.apply_query_provenance(with_proof(body), prepared, required=True)
    bridge.apply_mandatory_policy(prepared, current)
    return prepared, backend_request(model='local', tokens=prepared[0], session='owner',
        max_tokens=4096, plan=prepared[1], backend_kind='kvmem')


def test_default_prompt_and_policy_unchanged_exact_system_excludes_user(assets):
    tokenizer, path = assets
    default = TokenizerProtocol(tokenizer, path, enable_thinking=False)
    pinned = protocol(assets)
    body = wire_body()
    base = default(request=bridge.to_anthropic(body))
    prepared, data = prepared_request(pinned, body)
    rendered = prepared[2]
    assert base['prompt'] == rendered['prompt'] and base['pins'] == []
    assert 'mandatory_policy' not in default.provenance
    assert default.provenance != pinned.provenance
    assert pinned.provenance['mandatory_policy']['select_budget'] == 8192
    assert data['kvmem_mandatory_parts'] == 'none'
    pin, = data['kvmem_mandatory_spans']
    frame = rendered['prompt'].encode()[:rendered['control_end']].decode()
    assert tokenizer.decode(data['prompt'][:pin['end']]) == frame
    assert pin == dict(begin=0, end=len(tokenizer.encode(frame, add_special_tokens=False)), kind='system')
    assert body['messages'][-1]['content'] not in frame
    assert 'write' in frame and 'parameters' in frame
    assert bridge.validate_system_pin_budget(data, pinned)['required_select_tokens'] < 8192
    # Unknown/remapped token positions cannot be relabeled as authoritative pins.
    prepared[6][(-1, rendered['control_end'])] -= 1
    with pytest.raises(ValueError, match='accepted-prefix token coordinates'):
        bridge.apply_mandatory_policy(prepared, pinned)


def test_missing_control_frame_and_unsupported_protocol_fail_closed(assets):
    pinned = protocol(assets)
    with pytest.raises(ValueError, match='complete audited system/tool frame'):
        pinned(request=dict(system='', messages=[dict(role='user', content='hello')], tools=[]))
    with pytest.raises(ValueError, match='requires --tokenizer-protocol'):
        bridge.create_app(NS(pin_system=True, tokenizer_protocol=False), None)


@pytest.mark.parametrize('budget', [None, True, 0, -32, 8193])
def test_pin_requires_explicit_aligned_capacity_before_asset_loading(budget):
    with pytest.raises(ValueError, match='select-budget'):
        TokenizerProtocol(None, 'unused', pin_system=True, select_budget=budget)


def test_complete_frame_replay_and_rounding_budget_is_checked(assets):
    body = wire_body()
    body['messages'][0]['content'] = ' policy' * 4800
    narrow = protocol(assets, 4096)
    _, data = prepared_request(narrow, body)
    with pytest.raises(ValueError, match='Insufficient selected capacity.*K=4096'):
        bridge.validate_system_pin_budget(data, narrow)
    wider = protocol(assets, 8192)
    prepared, data = prepared_request(wider, body)
    proof = bridge.validate_system_pin_budget(data, wider)
    assert proof['pinned_blocks'] * 32 > 4096
    assert proof['reserved_blocks'] == 7  # replay128 + recent32 + two32 alignment blocks
    assert proof['required_select_tokens'] == (proof['pinned_blocks'] + 7) * 32
    raw = dict(data, kvmem_query_mode='raw')
    with pytest.raises(ValueError, match='verified agent Query epoch'):
        bridge.validate_system_pin_budget(raw, wider)
    bad = copy.deepcopy(data)
    bad['kvmem_query_span'] = [100, 260]
    with pytest.raises(ValueError, match='Query exceeds the declared replay budget'):
        bridge.validate_system_pin_budget(bad, wider)


@pytest.mark.parametrize('budget', [4096, 8192])
def test_real_bridge_guards_segments_and_restores_exact_policy_prefix(assets, monkeypatch, tmp_path, budget):
    tokenizer, path = assets
    body = wire_body()
    body['messages'][0]['content'] = ' policy' * 4800
    body.update(model='local', max_tokens=2)
    body = with_proof(body)
    accepted = tokenizer.encode('OK<|im_end|>', add_special_tokens=False)
    assert len(accepted) == 2
    calls = []

    async def run():
        async def upstream(request):
            data = json.loads(request.content)
            calls.append(data)
            index = len(calls) - 1
            token = accepted[index]
            events = [dict(id='fake', choices=[dict(index=0,
                text=tokenizer.decode([token]), token_ids=[token],
                finish_reason='length' if index == 0 else 'stop')]),
                dict(choices=[], usage=dict(prompt_tokens=len(data['prompt']), completion_tokens=1))]
            return httpx.Response(200, content=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        monkeypatch.setattr(bridge.httpx, 'AsyncClient', lambda **_: client)
        args = NS(pin_system=True, select_budget=budget, tokenizer_protocol=True,
            tokenizer=path, disable_thinking=True, gen_budget=4096, max_output_segments=2,
            max_inflight=1, output=tmp_path, backend='http://fake', model='local',
            max_model_len=32768, max_sessions=8, require_query_provenance=True)
        app = bridge.create_app(args, tokenizer)
        endpoint = next(r.endpoint for r in app.routes if r.path == '/v1/chat/completions')
        async def receive():
            return dict(type='http.request', body=json.dumps(body).encode(), more_body=False)
        result = await endpoint(Request(dict(type='http', method='POST',
            path='/v1/chat/completions', headers=[]), receive))
        await client.aclose()
        if budget == 4096:
            assert result.status_code == 502
            assert 'Insufficient selected capacity' in result.body.decode()
            assert calls == [] and not list(tmp_path.glob('**/accepted.json'))
            return
        assert result['choices'][0]['message']['content'] == 'OK'
        assert len(calls) == 2
        assert calls[1]['prompt'] == calls[0]['prompt'] + accepted[:1]
        assert calls[0]['kvmem_mandatory_spans'] == calls[1]['kvmem_mandatory_spans']
        assert all(c['kvmem_mandatory_parts'] == 'none' for c in calls)
        owner = calls[0]['session_id']
        directory = tmp_path/owner/'00001'
        saved = json.loads((directory/'plan.json').read_text())
        assert saved['system_pin_admission']['declared_select_budget'] == 8192
        for i in (0, 1):
            segment = json.loads((directory/f'segment-{i:03d}.json').read_text())
            assert segment['mandatory_spans'] == saved['kvmem_mandatory_spans']
            assert segment['system_pin_admission']['required_select_tokens'] <= 8192
        current = protocol(assets, 8192)
        planner = Planner(current, tokenizer, 4096)
        state = bridge.restore_trajectory(planner, owner, [tmp_path])
        next_body = copy.deepcopy(body)
        next_body['messages'] += [result['choices'][0]['message'], dict(role='user', content='继续。')]
        prepared = planner.prepare(bridge.to_anthropic(next_body), state)
        bridge.apply_query_provenance(with_proof(next_body), prepared, required=True)
        bridge.apply_mandatory_policy(prepared, current)
        assert prepared[5] is True
        assert prepared[0][:len(calls[0]['prompt']) + 2] == calls[0]['prompt'] + accepted
        assert prepared[1]['kvmem_mandatory_spans'] == saved['kvmem_mandatory_spans']
        assert prepared[1]['kvmem_locality_provenance']['history'] == 'actual_accepted_ids'
        for incompatible in (protocol(assets, 4096), TokenizerProtocol(tokenizer, path, enable_thinking=False)):
            with pytest.raises(ValueError, match='protocol differs'):
                bridge.restore_trajectory(Planner(incompatible, tokenizer, 4096), owner, [tmp_path])
        saved['kvmem_mandatory_spans'][0]['end'] -= 1
        (directory/'plan.json').write_text(json.dumps(saved))
        with pytest.raises(ValueError, match='explicit system-pin policy'):
            bridge.restore_trajectory(planner, owner, [tmp_path])

    asyncio.run(run())


@pytest.mark.parametrize('thinking', [False, True])
def test_tools_create_a_pin_frame_even_without_user_system_text(assets, thinking):
    tokenizer, path = assets
    current = TokenizerProtocol(tokenizer, path, pin_system=True, select_budget=8192,
                                enable_thinking=thinking)
    body = wire_body()
    body['messages'] = body['messages'][1:]
    prepared, data = prepared_request(current, body)
    assert data['kvmem_mandatory_spans'][0]['end'] > 0
    assert prepared[2]['spans'][0]['role'] == 'system'
    assert bridge.validate_system_pin_budget(data, current)['required_select_tokens'] <= 8192


def test_exact_cli_contract_builds_profile_without_binding_service(assets, monkeypatch, tmp_path):
    import runpy
    import sys
    import types
    _, path = assets
    captured = []
    uvicorn = types.ModuleType('uvicorn')
    def intercept(app, host, port):
        captured.append((host, port))
        assert any(r.path == '/v1/chat/completions' for r in app.routes)
    uvicorn.run = intercept
    monkeypatch.setitem(sys.modules, 'uvicorn', uvicorn)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **_: NS())
    monkeypatch.setattr(sys, 'argv', ['scripts.dsh_bridge', '--tokenizer-protocol',
        '--tokenizer', path, '--disable-thinking', '--pin-system', '--select-budget', '8192',
        '--require-query-provenance', '--gen-budget', '4096', '--max-model-len', '32768',
        '--max-inflight', '1', '--max-output-segments', '2', '--backend', 'http://127.0.0.1:8186',
        '--model', 'kvmem-gc-qwen38', '--owner-namespace', 'dsh-system-pin',
        '--port', '18387', '--output', str(tmp_path/'bridge')])
    runpy.run_module('scripts.dsh_bridge', run_name='__main__')
    assert captured == [('127.0.0.1', 18387)]
    manifest = json.loads((tmp_path/'bridge'/'protocol.json').read_text())
    assert manifest['mandatory_policy']['select_budget'] == 8192
    assert manifest['mandatory_policy']['query_replay_budget'] == 128
