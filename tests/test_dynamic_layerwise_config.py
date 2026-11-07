"""Keep dynamic page admission distinct from changing the recall algorithm."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from vllm_kvmem import serve


def profile():
    cfg = json.loads((Path(__file__).parents[1]/'examples/serve-qwen-mtp.json').read_text())
    cfg.update(execution_mode='compiled_graph', max_model_len=1048576,
               draft_tokens=4, max_num_seqs=2)
    cfg['kvmem'].update(select_budget=36864, gen_budget=16384, sink_blocks=41,
        recent_blocks=0, repository_blocks=32768, async_refresh=True,
        prefill_attention=True, action_attention=True, pressure_policy='attention',
        gpu_memory_admission='incremental', request_checkpoint_bytes=16*2**30,
        request_swap_layerwise=True)
    return cfg


def test_explicit_layerwise_admission_preserves_model_and_recall_controls(tmp_path):
    cfg = profile()
    original = deepcopy(cfg)
    plan = serve.build_launch(cfg, tmp_path, tmp_path/'run', check_paths=False)
    assert cfg == original
    assert plan['profile']['execution_mode'] == 'compiled_graph'
    assert plan['profile']['draft_tokens'] == 4
    assert {k:plan['profile']['kvmem'][k] for k in cfg['kvmem']} == cfg['kvmem']
    transfer = json.loads(plan['vllm_argv'][plan['vllm_argv'].index('--kv-transfer-config')+1])
    assert transfer['kv_connector_extra_config']['kvmem']['request_swap_layerwise'] is True


@pytest.mark.parametrize('changes', [
    {'request_swap_layerwise': 1}, {'gpu_memory_admission': 'static'},
    {'async_refresh': False}, {'request_planning': False}, {'prefill_attention': False},
    {'async_restore': True}, {'rolling_active_window': True}, {'guided_query_tokens': 32},
])
def test_unqualified_layerwise_swaps_fail_before_model_loading(tmp_path, changes):
    cfg = profile(); cfg['kvmem'].update(changes)
    with pytest.raises(ValueError):
        serve.build_launch(cfg, tmp_path, tmp_path/'run', check_paths=False)


def test_default_dynamic_admission_does_not_enable_async_or_compiled(tmp_path):
    cfg = profile(); cfg['kvmem'].pop('request_swap_layerwise')
    with pytest.raises(ValueError, match='Incremental GPU admission'):
        serve.build_launch(cfg, tmp_path, tmp_path/'run', check_paths=False)


def test_launcher_rejects_inactive_layerwise_hook_and_missing_swap_controller_flags(tmp_path, monkeypatch):
    cfg = profile()
    monkeypatch.setattr(serve, 'build_launch', lambda p,*a: dict(profile=p,environment={}))
    source = tmp_path/'config.json'
    base = ['--config',str(source),'--run-dir',str(tmp_path/'run'),'--dry-run']
    flags = ['--layerwise-query-diagnostic','sequential','--layerwise-query-optimized',
             '--layerwise-query-native-batches','--layerwise-prefill-pressure',
             '--layerwise-pressure-single-assembly']
    source.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='sequential optimized'):
        serve.main(base)
    cfg['kvmem']['async_refresh'] = False
    source.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='synchronous refresh does not invoke'):
        serve.main(base+flags)
    cfg['kvmem']['async_refresh'] = True
    source.write_text(json.dumps(cfg))
    serve.main(base+flags)
    with pytest.raises(ValueError, match='sequential optimized'):
        serve.main(base+flags+['--layerwise-query-mtp-selection','layer0'])
