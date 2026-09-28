import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('agent_preflight',
    Path(__file__).parents[1] / 'scripts/preflight_agent_api.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def evidence():
    result = {'id': 'chatcmpl-test', 'usage': {'prompt_tokens': 284, 'completion_tokens': 31}}
    archive = {'completed': [{'request_id': 'chatcmpl-test-worker', 'history_end': 314,
                'terminal_prefix': {'history_end': 314}}],
        'active_requests': 0, 'retiring_requests': 0, 'archive_used_slots': 0, 'pending_downloads': 0}
    released = {'requests': 0, 'owned_blocks': 0, 'reservations': []}
    return result, archive, released


def test_http_success_alone_or_unrelated_archive_cannot_qualify_preflight():
    result, archive, released = evidence()
    assert preflight.retired_completion(result, {}, {}) is None
    assert preflight.retired_completion(result, archive, released)['history_end'] == 314
    archive['completed'][0]['request_id'] = 'chatcmpl-test-other-prefix'
    result['id'] = 'chatcmpl-test-other'
    assert preflight.retired_completion(result, archive, released) is not None
    result['id'] = 'chatcmpl-test-oth'
    assert preflight.retired_completion(result, archive, released) is None


@pytest.mark.parametrize('where,key,value', [
    ('archive', 'active_requests', 1), ('archive', 'retiring_requests', 1),
    ('archive', 'archive_used_slots', 1), ('archive', 'pending_downloads', 1),
    ('released', 'requests', 1), ('released', 'owned_blocks', 1),
    ('released', 'reservations', ['chatcmpl-test']),
])
def test_incomplete_retirement_cannot_qualify(where, key, value):
    result, archive, released = evidence()
    {'archive': archive, 'released': released}[where][key] = value
    assert preflight.retired_completion(result, archive, released) is None


def test_terminal_speculation_overshoot_is_not_a_successful_preflight():
    result, archive, released = evidence()
    archive['completed'][0]['history_end'] += 3
    with pytest.raises(RuntimeError, match='terminal authority'):
        preflight.retired_completion(result, archive, released)
