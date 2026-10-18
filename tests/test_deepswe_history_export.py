import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('history_export',
    Path(__file__).resolve().parents[1]/'scripts/export_deepswe_capacity_history.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_long_selection_excludes_incomplete_turns_and_limits_per_task(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps(dict(status='window_complete', cleanup_errors=[],
        model_ready_at=0, window_deadline=10)))
    (tmp_path/'bridge').mkdir()
    rows = []
    for task, turn, n, complete in [('a', 1, 200, True), ('a', 2, 220, True),
                                     ('a', 3, 300, False), ('b', 1, 210, True)]:
        rows.append(dict(event='start', task=task, turn=turn, prompt_tokens=n, at=1))
        if complete:
            rows.append(dict(event='finish', task=task, turn=turn, at=2))
    (tmp_path/'bridge/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected = export.select_turns(tmp_path, 2, 100, 1)
    assert [(r['task'], r['turn']) for r in selected] == [('a', 2), ('b', 1)]
    with pytest.raises(ValueError, match='never pad or truncate'):
        export.select_turns(tmp_path, 3, 100, 1)


def fixture(tmp_path):
    for turn, tokens, output in [(1, [1], [7]), (2, [1, 7, 2], [9])]:
        directory = tmp_path/'bridge/task'/f'{turn:05d}'
        directory.mkdir(parents=True)
        raw = json.dumps(tokens, separators=(',', ':')).encode()
        values = dict(request={'new': turn}, response={}, native={'choices': [{'token_ids': output}]},
            plan=dict(prompt_tokens=len(tokens), token_sha256=hashlib.sha256(raw).hexdigest()))
        for name, value in values.items():
            (directory/(name+'.json')).write_text(json.dumps(value))
    class Planner:
        def prepare(self, payload, state):
            return state.prefix+[payload['new']], {}
        def commit(self, state, prepared, payload, response, output):
            state.prefix = prepared[0]+output
    return Planner(), SimpleNamespace(prefix=[])


def test_replay_uses_recorded_accepted_ids_and_verifies_every_prefix(tmp_path):
    planner, state = fixture(tmp_path)
    saved = []
    provenance = export.replay_task(tmp_path, 'task', [dict(turn=2, prompt_tokens=3)],
        planner, state, lambda row, tokens, plan, original: saved.append(tokens))
    assert saved == [[1, 7, 2]] and len(provenance) == 8
    assert state.prefix == [1, 7, 2, 9]


def test_replay_rejects_tampered_history_before_export(tmp_path):
    planner, state = fixture(tmp_path)
    path = tmp_path/'bridge/task/00001/native.json'
    path.write_text(json.dumps({'choices': [{'token_ids': [8]}]}))
    with pytest.raises(ValueError, match='token hash mismatch'):
        export.replay_task(tmp_path, 'task', [dict(turn=2, prompt_tokens=3)],
            planner, state, lambda *args: pytest.fail('Do not export tampered history'))
