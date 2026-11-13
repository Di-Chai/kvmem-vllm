"""The tracked layering gate (benchmarks/gate/gate.py): fixtures, comparisons and resource checks on CPU."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def gate():
    spec = importlib.util.spec_from_file_location('layering_gate', ROOT / 'benchmarks/gate/gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent_arm(path, tokens, selected=(0, 32), leaked=0):
    capture = path / 'server/capture'
    capture.mkdir(parents=True)
    rows = [dict(session=0, kind='grow', prompt_tokens=100, new_tokens=100, output_tokens=2,
                 started=1.0, ended=2.0, tokens=[1, 2]),
            dict(session=0, kind='turn', prompt_tokens=130, new_tokens=28, output_tokens=len(tokens),
                 started=3.0, ended=5.0, tokens=list(tokens))]
    (path / 'client.json').write_text(json.dumps(dict(summary=dict(errors=[]), rows=rows)))
    events = [dict(event='query_candidates', request_id='a', candidate_starts=[0, 32, 64], history=96),
              dict(event='layer_selected', request_id='a', layer=0, selected_starts=list(selected)),
              dict(event='pressure_selected', request_id='a', accepted_history=128, event_type='decode_pressure',
                   layers=[dict(layer=0, actual_method='accumulated_attention_per_layer',
                                selected_starts=list(selected))])]
    (capture / 'layerwise_query.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    (capture / 'scheduler_resources.json').write_text(json.dumps(dict(
        requests=leaked, refresh_pending=0, connector_pending=0, reservations=[], owned_blocks=0,
        gpu_capacity=dict(physical_allocated_pages=0), active_residency=dict(idle_views=0))))
    (capture / 'archive_resources.json').write_text(json.dumps(dict(
        active_requests=0, retiring_requests=0, archive_io_pending_bytes=0)))
    (path / 'server.log').write_text('ready\n')
    return path


def test_fixtures_describe_single_request_arms(gate):
    mab = json.loads((ROOT / 'benchmarks/gate/mab16.json').read_text())
    assert len(mab['fixture']['rows']) == len(mab['fixture']['bindings']) == 16
    assert mab['profile']['kvmem']['gen_budget'] == 4096
    agent = json.loads((ROOT / 'benchmarks/gate/agent.json').read_text())
    kvmem = agent['profile']['kvmem']
    # The history must exceed the select budget, or no strategy decision is ever made.
    assert agent['client']['history'] > kvmem['select_budget']
    assert agent['client']['output'] > kvmem['gen_budget'] // 4
    assert '--layerwise-rolling-decode' in agent['server_flags']
    assert not set(gate.STRATEGY_OPTIONS) & set(kvmem)


def test_agent_compare_and_measure(gate, tmp_path):
    ref = _agent_arm(tmp_path / 'ref', [5, 6, 7])
    same = _agent_arm(tmp_path / 'same', [5, 6, 7])
    other = _agent_arm(tmp_path / 'other', [5, 9, 7], selected=(0, 64))
    assert gate.compare(ref, same) == 0
    assert gate.compare(ref, other) == 1
    tokens = gate.compare_agent(ref, other)
    assert tokens['token_identical'] == 1 and tokens['diffs'][0]['first_differing_token'] == 1
    measured = gate.measure_agent(ref)
    assert measured['turn_tok_s'] == 1.5
    assert measured['decisions'] == 2 and measured['pressure_events'] == {'decode_pressure': 1}
    assert measured['released']['clean'] and measured['tracebacks'] == 0


def test_agent_released_reports_held_requests(gate, tmp_path):
    arm = _agent_arm(tmp_path / 'leak', [1], leaked=1)
    held = gate.released(arm)
    assert held['requests'] == 1 and not held['clean']


def test_mab_compare(gate, tmp_path):
    def arm(name, text):
        case = tmp_path / name / 'run/answers/cases/c0'
        case.mkdir(parents=True)
        (case / 'response.json').write_text(json.dumps(dict(choices=[dict(text=text, token_ids=[len(text)])])))
        capture = tmp_path / name / 'run/server/capture'
        capture.mkdir(parents=True)
        (capture / 'layerwise_query.jsonl').write_text(json.dumps(dict(
            event='pressure_selected', request_id='r', accepted_history=64,
            layers=[dict(layer=0, selected_starts=[0])])) + '\n')
        return tmp_path / name
    a, b, c = arm('a', 'x'), arm('b', 'x'), arm('c', 'yy')
    assert gate.kind_of(a) == 'mab'
    assert gate.compare(a, b) == 0
    assert gate.compare(a, c) == 1
