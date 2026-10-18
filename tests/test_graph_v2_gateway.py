"""The append gateway forwards graph-only spans in absolute history coordinates."""
import asyncio
import json

import pytest

httpx = pytest.importorskip('httpx')
pytest.importorskip('fastapi')
from test_elastic_gateway import _Request, _status, configuration
from vllm_kvmem.elastic_gateway import create_app
from vllm_kvmem.request_plan import RequestPlan


def test_gateway_preserves_absolute_locality_spans_and_provenance_across_append(tmp_path):
    cfg = configuration(tmp_path)
    cfg.update(context_mode='meank', request_planning=True, active_context_tokens=256,
        gen_budget_tokens=128, max_model_len=1024, active_bytes=1000,
        token_history_bytes=8000, max_body_bytes=10000, block_tokens=32)
    observed = []

    def upstream(request):
        payload = json.loads(request.content)
        observed.append(payload)
        return httpx.Response(200, json={'choices': [{'token_ids': [8], 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 1, 'prompt_tokens': len(payload['prompt'])}})

    provenance = dict(schema_version=1, source='structured-test', status='exact',
                      tokenizer='test-offsets', template='test-renderer')
    first = dict(kvmem_locality_spans=[dict(begin=0, end=64, role='user')],
                 kvmem_locality_provenance=provenance)
    second = dict(kvmem_locality_spans=[dict(begin=0, end=64, role='user'),
                  dict(begin=64, end=65, role='assistant'), dict(begin=65, end=97, role='tool')],
                  kvmem_locality_provenance=provenance)
    async def run():
        app = create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream),
                                     base_url='http://native') as native:
            app.state.client = native
            path = '/v1/kvmem/sessions/graph/turn'
            result = await _Request(app, path,
                body=dict(append_token_ids=list(range(64)), max_tokens=2, **first)).finish()
            assert result.status == 200, result.data
            result = await _Request(app, path,
                body=dict(version=1, append_token_ids=list(range(100, 132)), max_tokens=2, **second)).finish()
            assert result.status == 200, result.data
            assert [item['kv_transfer_params']['kvmem_request_plan'] for item in observed] == [first, second]
            assert observed[1]['prompt'] == list(range(64)) + [8] + list(range(100, 132))
            for metadata, size in ((first, 64), (second, 97)):
                plan = RequestPlan.parse(metadata, size, 32, 128)
                original = RequestPlan.parse({}, size, 32, 128)
                assert (plan.mandatory, plan.explicit_mandatory, plan.query, plan.replay) == (
                    original.mandatory, original.explicit_mandatory, original.query, original.replay)
            invalid = dict(kvmem_locality_spans=[dict(begin=98, end=100, role='tool')],
                           kvmem_locality_provenance=provenance)
            result = await _Request(app, path,
                body=dict(version=2, append_token_ids=[3], max_tokens=2, **invalid)).finish()
            assert result.status == 400 and len(observed) == 2
            state = await _status(app)
            assert state['sessions'][0]['version'] == 2
            assert state['sessions'][0]['history_tokens'] == 98
            assert state['active_reserved_bytes'] == 0
    asyncio.run(run())
