from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.adapters.capture import CapturedLayer
from vllm_kvmem.adapters.qsa_adapter import validate_qsa_config, write_auxiliary_bytes, borrow_qsa_state


def qsa_config():
    return NS(model_config=NS(hf_text_config=NS(model_type='qwen4_exp_text'),
        enforce_eager=True, multimodal_config=NS(language_model_only=True)),
        scheduler_config=NS(async_scheduling=False), max_concurrent_batches=1,
        cache_config=NS(enable_prefix_caching=False, mamba_cache_mode='none', cache_dtype='fp8_e4m3'),
        parallel_config=NS())


def test_qsa_rejects_unqualified_execution_before_loading_weights():
    options = NS(memory=NS(block_tokens=32, action_attention=False))
    cfg = qsa_config()
    assert validate_qsa_config(cfg, options)
    for owner, key, value in ((cfg.model_config, 'enforce_eager', False),
            (cfg.scheduler_config, 'async_scheduling', True),
            (cfg.cache_config, 'cache_dtype', 'fp8'),
            (cfg.cache_config, 'mamba_cache_mode', 'align')):
        old = getattr(owner, key)
        setattr(owner, key, value)
        with pytest.raises(ValueError, match='requires eager'):
            validate_qsa_config(cfg, options)
        setattr(owner, key, old)
    options.async_refresh = True
    with pytest.raises(ValueError, match='async_refresh'):
        validate_qsa_config(cfg, options)


def test_selector_bf16_payload_is_bit_exact_after_compaction_including_nan_bits():
    # Exercise every BF16 bit pattern, including NaN payloads, without casting.
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(512, 128)
    zeros = torch.zeros(512, 2, 256, dtype=torch.bfloat16)
    ordinary = CapturedLayer('main', torch.arange(512), zeros, zeros, zeros)
    aux = CapturedLayer('aux', ordinary.positions, zeros, zeros, zeros, raw)
    rows = torch.tensor([511, 0, 127, 256, 17])
    payload = torch.ones(2, len(rows), 2, 256, dtype=torch.float8_e4m3fn)
    write_auxiliary_bytes(payload, [ordinary, aux], rows=rows)
    data = payload.view(torch.uint8)
    assert torch.equal(data[0], torch.ones_like(payload[0]).view(torch.uint8))
    restored = data[1].reshape(len(rows), -1)[:, :256].contiguous().view(torch.int16)
    assert torch.equal(restored, raw.view(torch.int16)[rows])
    assert not data[1].reshape(len(rows), -1)[:, 256:].any()


def test_qsa_full_graph_requires_persistent_capture_and_rejects_torch_compile(monkeypatch):
    cfg = qsa_config()
    options = NS(memory=NS(block_tokens=32, action_attention=False))
    cfg.model_config.enforce_eager = False
    cfg.compilation_config = NS(mode=0, cudagraph_mode=NS(name='FULL'))
    monkeypatch.delenv('KVMEM_FULL_GRAPH', raising=False)
    with pytest.raises(ValueError, match='requires eager or full_graph'):
        validate_qsa_config(cfg, options)
    monkeypatch.setenv('KVMEM_FULL_GRAPH', '1')
    assert validate_qsa_config(cfg, options)
    cfg.compilation_config.mode = 3
    with pytest.raises(ValueError, match='requires eager or full_graph'):
        validate_qsa_config(cfg, options)
    cfg.compilation_config.mode = 0
    cfg.compilation_config.cudagraph_mode.name = 'FULL_AND_PIECEWISE'
    with pytest.raises(ValueError, match='requires eager or full_graph'):
        validate_qsa_config(cfg, options)


def test_qsa_checkpoint_borrows_only_the_owned_request_slot():
    tensors = (torch.randn(3, 8, 128), torch.arange(24).reshape(3, 8))
    controls = (torch.ones(3, dtype=torch.bool), torch.arange(3))
    runner = NS(_kvmem_qsa_layers={'target': NS(get_recurrent_checkpoint_tensors=lambda: tensors)},
        model_state=NS(get_recurrent_checkpoint_tensors=lambda: controls))
    borrowed = borrow_qsa_state(runner, 'request', 1)
    assert len(borrowed) == 4
    saved = {name: tensor.clone() for name, tensor in borrowed.items()}
    tensors[0][1].zero_()
    for name, target in borrowed.items():
        target.copy_(saved[name])
    assert torch.equal(tensors[0][1:2], saved['request/target/qsa/0'])
    assert all(t.shape[0] == 1 for t in borrowed.values())


def test_qsa_archive_admission_accounts_for_auxiliary_bytes():
    pytest.importorskip('vllm')
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
    from vllm_kvmem.adapters.cache_manifest import archive_bytes_per_token
    cache = NS(kv_cache_groups=[KVCacheGroupSpec(['target', 'mtp'],
        FullAttentionSpec(block_size=32, num_kv_heads=2, head_size=256, dtype=torch.uint8))])
    assert archive_bytes_per_token(cache) == 2048
    assert archive_bytes_per_token(cache, qsa_config().model_config) == 4096


def test_qsa_mtp_uses_dense_text_positions_instead_of_stale_native_mrope():
    from vllm_kvmem.adapters.qsa_adapter import install_qsa_mtp_positions
    logical = torch.tensor([[250000, 250001]] * 3)
    dense = torch.tensor([8192, 8193])
    proposer = NS(_target_model_positions=lambda batch, profile: logical)
    runner = NS(speculator=proposer, _kvmem_pending_capture=None, _kvmem_window_batch=None)
    install_qsa_mtp_positions(runner)
    batch = NS(positions=dense)
    assert proposer._target_model_positions(batch, True) is logical
    runner._kvmem_pending_capture = ('accepted-prefix', batch)
    runner._kvmem_window_batch = NS(positions=dense)
    assert proposer._target_model_positions(batch, False) is dense
    with pytest.raises(ValueError, match='owned dense target frame'):
        proposer._target_model_positions(NS(positions=logical), False)
    runner._kvmem_pending_capture = None
    assert proposer._target_model_positions(batch, True) is logical


def test_control_only_guard_preserves_external_connector_rejection():
    from vllm_kvmem.adapters.qsa_config_compat import patch_source, control_only
    with pytest.raises(ImportError, match='Unrecognized'):
        patch_source(b'unknown upstream implementation')
    source = b'''def check(self, connector, aligned):
    if not aligned and not connector.supports_request_boundary_checkpoints(
        self
    ):
        raise ValueError('external state')
'''
    namespace = {}
    exec(compile(patch_source(source), '<qsa-guard>', 'exec'), namespace)
    external = type('ExternalConnector', (), {'supports_request_boundary_checkpoints': lambda self: False})
    assert not control_only(qsa_config(), external)
    with pytest.raises(ValueError, match='external state'):
        namespace['check'](qsa_config(), external, False)
    namespace['check'](qsa_config(), external, True)


def test_flash_next_launch_is_explicit_and_does_not_install_dflash_patch(tmp_path):
    import json
    from pathlib import Path
    from vllm_kvmem.serving.serve import build_launch
    profile = json.loads((Path(__file__).parents[1] / 'examples/serve-qwen38-flash-next-fp4-local.json').read_text())
    launch = build_launch(profile, tmp_path, tmp_path/'run', check_paths=False)
    assert '--mamba-cache-mode' in launch['vllm_argv']
    assert launch['environment']['VLLM_PLE_CPU_OFFLOAD'] == '1'
    assert launch['environment']['VLLM_QWEN3_8_FLASH_NEXT_OVERLAP'] == '0'
    assert 'KVMEM_DRAFT_LOAD_COMPAT' not in launch['environment']
    profile['execution_mode'] = 'full_graph'
    graph = build_launch(profile, tmp_path, tmp_path/'graph', check_paths=False)
    assert graph['environment']['KVMEM_FULL_GRAPH'] == '1'
    assert '--enforce-eager' not in graph['vllm_argv']
    profile['execution_mode'] = 'compiled_graph'
    with pytest.raises(ValueError, match='requires safetensors, eager or full_graph MTP'):
        build_launch(profile, tmp_path, tmp_path/'run2', check_paths=False)


def test_qsa_state_source_contract_rejects_missing_selector_pool(tmp_path):
    from vllm_kvmem.adapters.engine_api import UnsupportedEngine
    from vllm_kvmem.adapters.qsa_contract import check_qsa_sources
    root = tmp_path/'models/qwen4_exp/nvidia'
    root.mkdir(parents=True)
    qsa = '''class Qwen4ExpQSAAttention:
    def _project_qkv_gate(self, qkv, positions): pass
    def set_skip_topk(self, skip): pass
    def get_recurrent_checkpoint_tensors(self):
        return (self._raw_k_ring, self._raw_logical_positions,
                self._raw_rope_positions, self._raw_interval_start_positions)
class Qwen4ExpQSAImpl:
    def _kv_cache_views(self, kv_cache):
        key_cache, value_cache = kv_cache.unbind(1)
        return key_cache.unflatten(-1, (self.num_kv_heads, self.head_size))
'''
    (root/'b12x_qsa.py').write_text(qsa)
    (root/'b12x_indexer_qsa.py').write_text('class QSAIndexer:\n    def project(self, hidden_states): pass\n')
    (root/'model_state.py').write_text('''class Qwen4ExpModelState:
    def get_recurrent_checkpoint_tensors(self):
        return (self.qsa_state_is_fresh_gpu, self.qsa_committed_num_accepted_tokens_gpu)
''')
    (root/'ple_layer.py').write_text('class Qwen4ExpPLELayer:\n    def get_state_shape(self): pass\n')
    check_qsa_sources(tmp_path)
    (root/'b12x_qsa.py').write_text(qsa.replace('self._raw_rope_positions, ', ''))
    with pytest.raises(UnsupportedEngine, match='state/cache contract changed'):
        check_qsa_sources(tmp_path)


def test_spawned_worker_can_patch_already_imported_config_without_replacing_class(tmp_path, monkeypatch):
    import sys
    from vllm_kvmem.adapters.qsa_config_compat import enable
    source = '''class VllmConfig:
    def _verify_kv_transfer_compat(self):
        connector, aligned = self.connector, False
        if not aligned and not connector.supports_request_boundary_checkpoints(self):
            raise ValueError('external state')
'''
    path = tmp_path/'config.py'
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), 'exec'), namespace)
    owner = namespace['VllmConfig']
    module = NS(VllmConfig=owner, __file__=str(path))
    monkeypatch.setitem(sys.modules, 'vllm.config.vllm', module)
    enable()
    assert module.VllmConfig is owner
    method = owner._verify_kv_transfer_compat
    enable()
    assert owner._verify_kv_transfer_compat is method
    external = type('External', (), {'supports_request_boundary_checkpoints': lambda cfg: False})
    config = owner()
    config.connector = external
    with pytest.raises(ValueError, match='external state'):
        config._verify_kv_transfer_compat()


def test_mtp_loader_releases_unused_target_staging_and_restores_hook_on_failure(monkeypatch):
    from vllm_kvmem.adapters.qsa_contract import release_target_loader_staging
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append('fence'))
    monkeypatch.setattr('gc.collect', lambda: calls.append('collect'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('trim'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (16 << 30, 96 << 30))
    target = object()
    def original(model):
        assert model is target
        calls.append('mtp')
        raise RuntimeError('loader failure')
    proposer = NS(load_model=original)
    with pytest.raises(RuntimeError, match='loader failure'):
        with release_target_loader_staging(NS(speculator=proposer, device='cuda')):
            proposer.load_model(target)
    assert proposer.load_model is original
    assert calls == ['fence', 'collect', 'trim', 'mtp']
