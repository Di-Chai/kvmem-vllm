import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('agent_backend', scripts / 'serve_agent_backend.py')
backend = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(backend)
finally:
    sys.path.remove(str(scripts))


def recipe():
    return {'status': 'passed', 'performance_eligible': True, 'command': [
        '/python', '-m', 'vllm_kvmem', '--draft-load-compat', 'vllm', 'serve',
        '/model.Q8_0.gguf', '--port', '12345', '--host', '127.0.0.1',
        '--kv-cache-dtype', 'fp8_e4m3', '--max-model-len', '262144']}


def test_agent_backend_preserves_qualified_model_precision_and_adds_native_metrics():
    original = recipe()
    command = backend.server_command(original, '/new-python', 54321)
    expected = list(original['command'])
    expected[0], expected[expected.index('--port') + 1] = '/new-python', '54321'
    assert command == expected + ['--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
                                  '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml',
                                  '--reasoning-parser', 'qwen3']
    assert original == recipe()


@pytest.mark.parametrize('change', [{'status': 'failed'}, {'performance_eligible': False}])
def test_agent_backend_rejects_failed_or_instrumented_recipe(change):
    with pytest.raises(ValueError, match='uninstrumented'):
        backend.server_command(recipe() | change, '/python', 1234)


def test_session_backend_is_explicit_and_preserves_model_and_memory_recipe():
    import json
    original = recipe()
    transfer = {'kv_connector_extra_config': {'kvmem': {'async_archive_retirement': True,
                'select_budget': 131072, 'gen_budget': 65536}}}
    original['command'] += ['--kv-transfer-config', json.dumps(transfer)]
    command = backend.server_command(original, '/python', 4321, session_capacity=4, session_checkpoint_mib=1024)
    got = json.loads(command[command.index('--kv-transfer-config')+1])
    assert got['kv_connector_extra_config']['kvmem'] == {
        **transfer['kv_connector_extra_config']['kvmem'], 'session_capacity': 4,
        'session_checkpoint_bytes': 1024**3}
    assert command[command.index('--kv-cache-dtype')+1] == 'fp8_e4m3'
    assert json.loads(original['command'][-1]) == transfer
    with pytest.raises(ValueError, match='together'):
        backend.server_command(original, '/python', 4321, session_capacity=4)
