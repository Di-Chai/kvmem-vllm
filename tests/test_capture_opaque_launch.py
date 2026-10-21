import json
from copy import deepcopy

import pytest

from test_graph_options import model_profile, runtime, settings, worker_payload
from vllm_kvmem.serve import build_launch


def profile(tmp_path, enabled):
    result = model_profile(tmp_path)
    result['kvmem'] = settings(capture_opaque_projection=enabled)
    return result


def test_disabled_preserves_original_compile_configuration(tmp_path):
    config = profile(tmp_path, False)
    launch = build_launch(config, tmp_path, tmp_path/'run')
    assert '--additional-config' not in launch['vllm_argv']
    assert not runtime(worker_payload(launch)).capture_opaque_projection


def test_opaque_source_is_bound_to_native_compile_key_and_saved_launch(tmp_path):
    from vllm_kvmem.capture_compile import CAPTURE_COMPILE_KEY, capture_compile_fingerprint
    config = profile(tmp_path, True)
    before = deepcopy(config)
    launch = build_launch(config, tmp_path, tmp_path/'run')
    assert config == before
    argv = launch['vllm_argv']
    assert argv.count('--additional-config') == 1
    expected = {CAPTURE_COMPILE_KEY: capture_compile_fingerprint()}
    assert json.loads(argv[argv.index('--additional-config')+1]) == expected
    assert len(expected[CAPTURE_COMPILE_KEY]) == 64
    assert runtime(worker_payload(launch)).capture_opaque_projection
    assert json.loads(argv[argv.index('--compilation-config')+1]) == {
        'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
    assert '--enforce-eager' not in argv
    replay = build_launch(launch['profile'], tmp_path, tmp_path/'replay')
    assert json.loads(replay['vllm_argv'][replay['vllm_argv'].index('--additional-config')+1]) == expected


def test_changed_projection_source_changes_compile_namespace(tmp_path, monkeypatch):
    import vllm_kvmem.capture_compile as compile_key
    config = profile(tmp_path, True)
    keys = []
    for fingerprint in ('1'*64, '2'*64):
        monkeypatch.setattr(compile_key, 'capture_compile_fingerprint', lambda: fingerprint)
        argv = build_launch(config, tmp_path, tmp_path/fingerprint[:1])['vllm_argv']
        keys.append(json.loads(argv[argv.index('--additional-config')+1]))
    assert keys[0] != keys[1]


@pytest.mark.parametrize('mode', ['eager', 'full_graph'])
def test_unqualified_execution_mode_is_rejected_before_launch(tmp_path, mode):
    config = profile(tmp_path, True)
    config['execution_mode'] = mode
    with pytest.raises(ValueError, match='Opaque capture projection.*compiled_graph'):
        build_launch(config, tmp_path, tmp_path/'bad')
