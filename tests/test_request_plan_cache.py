"""KVMEM_REQUEST_PLAN_CACHE=1 memoizes request_plan() per request with identical results.

dsh sends kvmem_locality_spans that grow with the conversation; the scheduler
resolves each scheduled request's plan ~4 times per step, so re-parsing grew
the per-step host time with history (hostgap-20261110-r1: 8x120K, 150 spans,
request_plan 2.1 ms/step, step period +8%).
"""
import json
import random
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import request_plan as rp
from vllm_kvmem.request_plan import RequestPlan, request_plan

PROVENANCE = {'encoding_scope': 'append_suffix', 'history': 'actual_accepted_ids', 'schema_version': 1,
              'source': 'deepswe_protocol', 'status': 'exact', 'unknown_spans': 0}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps({'rolling_active_window': True}))
    monkeypatch.delenv('KVMEM_REQUEST_PLAN_CACHE', raising=False)
    monkeypatch.delenv('KVMEM_LAYERWISE_QUERY_MODE', raising=False)
    monkeypatch.delenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES', raising=False)


def payload(rng, prompt, spans):
    cuts = sorted(rng.sample(range(1, prompt - 64), spans - 1)) if spans > 1 else []
    edges = [0] + cuts + [prompt - 64]
    roles = ('assistant', 'tool', 'user')
    locality = [dict(begin=a, end=b, role=roles[i % 3]) for i, (a, b) in enumerate(zip(edges, edges[1:])) if b > a]
    query = prompt - 40
    return dict(kvmem_locality_spans=locality, kvmem_locality_provenance=dict(PROVENANCE, exact_spans=len(locality)),
                kvmem_query_span=[query, query + 24], kvmem_reselect='auto', kvmem_query_mode='agent',
                kvmem_query_id='q-%d' % rng.randrange(10**6), kvmem_mandatory_spans=[], kvmem_mandatory_parts='none')


def request(body, prompt, max_tokens=4096):
    return NS(kv_transfer_params={'kvmem_request_plan': body}, num_prompt_tokens=prompt,
              sampling_params=NS(max_tokens=max_tokens), request_id='r')


def counting(monkeypatch):
    calls = []
    original = RequestPlan.parse.__func__
    def parse(cls, *a, **k):
        calls.append(1)
        return original(cls, *a, **k)
    monkeypatch.setattr(RequestPlan, 'parse', classmethod(parse))
    return calls


@pytest.mark.parametrize('seed', range(6))
def test_cached_plans_equal_fresh_parses_and_parse_once(monkeypatch, seed):
    rng = random.Random(seed)
    prompt = rng.randrange(20000, 300000)
    body = payload(rng, prompt, rng.randrange(1, 400))
    fresh = request_plan(request(body, prompt), 32, 16384)
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', '1')
    calls = counting(monkeypatch)
    r = request(body, prompt)
    plans = [request_plan(r, 32, 16384) for _ in range(5)]
    assert all(p is plans[0] for p in plans) and plans[0] == fresh and len(calls) == 1


def test_any_input_change_reparses(monkeypatch):
    rng = random.Random(7)
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', '1')
    calls = counting(monkeypatch)
    body = payload(rng, 100000, 50)
    r = request(body, 100000)
    first = request_plan(r, 32, 16384)
    # A replaced payload (SessionLedger.set_plan builds a new dict).
    r.kv_transfer_params = {'kvmem_request_plan': payload(rng, 100000, 80)}
    second = request_plan(r, 32, 16384)
    assert second != first and len(calls) == 2
    # A reassigned value inside the same dict.
    r.kv_transfer_params['kvmem_request_plan']['kvmem_reselect'] = 'force'
    assert request_plan(r, 32, 16384).reselect == 'force' and len(calls) == 3
    # Request geometry, budget, output limit and environment are part of the key.
    r.num_prompt_tokens = 100001; request_plan(r, 32, 16384)
    request_plan(r, 32, 8192)
    r.sampling_params.max_tokens = 99; assert request_plan(r, 32, 8192).output_tokens == 99
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps({'rolling_active_window': False}))
    assert request_plan(r, 32, 8192).rolling_active_window is False
    assert len(calls) == 7
    uncached = dict(r.__dict__); uncached.pop('_kvmem_request_plan_cache')
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', '0')
    assert request_plan(NS(**uncached), 32, 8192) == request_plan(r, 32, 8192)


def test_errors_and_missing_plans_match_the_uncached_path(monkeypatch):
    bad = dict(kvmem_reselect='sometimes')
    with pytest.raises(ValueError) as off:
        request_plan(request(bad, 1000), 32, 16384)
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', '1')
    r = request(bad, 1000)
    for _ in range(2):                                  # an error is never cached
        with pytest.raises(ValueError) as on:
            request_plan(r, 32, 16384)
        assert str(on.value) == str(off.value)
    assert request_plan(NS(kv_transfer_params=None, num_prompt_tokens=5, sampling_params=None), 32, 16384) is None


def test_objects_without_a_dict_stay_uncached(monkeypatch):
    class Slotted:
        __slots__ = ('kv_transfer_params', 'num_prompt_tokens', 'sampling_params')
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', '1')
    body = payload(random.Random(3), 50000, 20)
    s = Slotted(); s.kv_transfer_params = {'kvmem_request_plan': body}
    s.num_prompt_tokens = 50000; s.sampling_params = NS(max_tokens=10)
    assert request_plan(s, 32, 16384) == request_plan(request(body, 50000, 10), 32, 16384)


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CACHE', 'yes')
    with pytest.raises(ValueError):
        rp.request_plan_cache_enabled()
