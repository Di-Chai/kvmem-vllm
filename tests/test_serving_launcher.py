import json
import subprocess
import sys

import pytest

from vllm_kvmem.serve import build_launch


def profile(tmp_path):
    model = tmp_path/'model.gguf'
    mm = tmp_path/'mm.gguf'
    model.touch(); mm.touch()
    config = tmp_path/'config'
    draft = tmp_path/'draft'
    config.mkdir(); draft.mkdir()
    return {'model': model.name, 'mm_proj': mm.name, 'tokenizer': config.name,
            'hf_config_path': config.name, 'draft_model': draft.name,
            'kvmem': {'async_archive_retirement': True, 'session_capacity': 4,
                      'session_checkpoint_bytes': 1024**3}}


def test_launcher_preserves_mainline_precisions_and_native_execution(tmp_path):
    plan = build_launch(profile(tmp_path), tmp_path, tmp_path/'run')
    argv, env = plan['vllm_argv'], plan['environment']
    assert argv[:2] == ['serve', str(tmp_path/'model.gguf')]
    assert argv[argv.index('--kv-cache-dtype')+1] == 'fp8_e4m3'
    spec = json.loads(argv[argv.index('--speculative-config')+1])
    assert spec['num_speculative_tokens'] == 7 and spec['kv_cache_dtype'] == 'bfloat16'
    transfer = json.loads(argv[argv.index('--kv-transfer-config')+1])
    assert transfer['kv_connector_extra_config']['kvmem']['session_capacity'] == 4
    assert '--no-enable-prefix-caching' in argv and '--no-async-scheduling' in argv
    assert argv[argv.index('--middleware')+1] == 'vllm_kvmem.http_sessions.validate_prompt_session'
    assert env['KVMEM_SERVING'] == '1' and 'KVMEM_Q8_TENSORCORE' not in env
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('changes', [
    {'unknown': True}, {'port': True}, {'max_model_len': 262145},
    {'max_model_len': 131072}, {'gpu_memory_utilization': float('nan')},
    {'gpu_memory_utilization': 1}, {'q8_tensorcore': 1},
    {'long_prefill_token_threshold': 1000}, {'model': 'absent.gguf'},
    {'kvmem': {'session_capacity': 4}}, {'kvmem': {'kv_dtype': 'bfloat16'}},
])
def test_invalid_or_unqualified_launch_refused_before_vllm_import(tmp_path, changes):
    with pytest.raises(ValueError):
        build_launch({**profile(tmp_path), **changes}, tmp_path, tmp_path/'run')


def test_dry_run_never_imports_engine_changes_env_or_creates_run_directory(tmp_path):
    config = tmp_path/'profile.json'
    config.write_text(json.dumps(profile(tmp_path)))
    code = '''import os,sys
from vllm_kvmem.serve import main
before=dict(os.environ)
main(sys.argv[1:])
assert dict(os.environ)==before
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, '--config', str(config),
                             '--run-dir', str(tmp_path/'run'), '--dry-run'],
                             check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['vllm_argv'][0] == 'serve'
    assert not (tmp_path/'run').exists()


def test_existing_run_directory_is_not_reused(tmp_path):
    from vllm_kvmem.serve import main
    config = tmp_path/'profile.json'
    config.write_text(json.dumps(profile(tmp_path)))
    run = tmp_path/'owned'; run.mkdir()
    marker = run/'keep'; marker.write_text('owned by another service')
    with pytest.raises(FileExistsError):
        main(['--config', str(config), '--run-dir', str(run)])
    assert marker.read_text() == 'owned by another service'
