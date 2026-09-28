import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('session_gate', Path(__file__).parents[1] / 'scripts/qualify_sessions.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def response(tokens=(1, 2)):
    return {'choices': [{'token_ids': list(tokens)}], 'usage': {'completion_tokens': len(tokens), 'prompt_tokens': 4096}}


def state(tokens, hits):
    return {'archive': {'session_restored_tokens': tokens}, 'scheduler': {'sessions': {'hits': hits}}}


def test_http_gate_requires_identical_ids_and_actual_partial_checkpoint_ack():
    output = response()
    assert gate.validate_reuse(output, output, output, state(0, 0), state(3072, 1)) == 3072
    with pytest.raises(RuntimeError, match='token IDs changed'):
        gate.validate_reuse(output, output, response((1, 3)), state(0, 0), state(3072, 1))
    for imported, hits in ((0, 0), (4096, 1), (3071, 1), (3072, 0), (3072, 2)):
        with pytest.raises(RuntimeError):
            gate.validate_reuse(output, output, output, state(0, 0), state(imported, hits))


def test_missing_invalid_or_truncated_token_evidence_cannot_pass():
    for output in ({'choices': []}, response(()), response((True, 2)), response((-1, 2)),
                   {'choices': [{'token_ids': [1]}], 'usage': {'completion_tokens': 2}}):
        with pytest.raises(RuntimeError):
            gate.output_tokens(output)


def test_concurrent_gate_rejects_missing_or_cross_counted_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    import qualify_session_concurrency as concurrent
    concurrent.reuse_delta(state(1024, 1), state(10240, 4), 3, [3072]*3)
    for tokens, hits in ((10240, 3), (9216, 4), (13312, 5)):
        with pytest.raises(RuntimeError, match='Incorrect session imports'):
            concurrent.reuse_delta(state(1024, 1), state(tokens, hits), 3, [3072]*3)
