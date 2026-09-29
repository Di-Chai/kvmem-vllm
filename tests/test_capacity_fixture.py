import hashlib
import json
from pathlib import Path

import pytest


def test_capacity_derivation_is_distinct_and_cannot_enter_performance_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import make_capacity_fixture as capacity
    import qualify_engine as engine
    source = {'id': 'native', 'input_ids': [1, 2, 3, 4, 5], 'query_begin': 3, 'query_end': 5}
    provenance = {'id': 'native', 'fixture_kind': 'native_history', 'sha256': 'original'}
    result = capacity.extend_history(source, provenance, 12)
    assert result['input_ids'][:3] == [1, 2, 3]
    assert result['input_ids'][10:] == [4, 5]
    assert result['query_begin'] == 10 and result['query_end'] == 12
    assert len(source['input_ids']) == 5
    path = tmp_path/'pressure.json'
    path.write_text(json.dumps(result))
    _, recorded = engine.read_frozen(path, hashlib.sha256(path.read_bytes()).hexdigest(), 128)
    assert recorded['fixture_kind'] == 'synthetic_capacity'
    assert not engine.performance_eligible(0, [provenance, recorded])
    assert engine.performance_eligible(0, [provenance])
    assert not engine.performance_eligible(1, [provenance])
    for tokens in (4, 5, 262144, True):
        with pytest.raises(ValueError):
            capacity.extend_history(source, provenance, tokens)
