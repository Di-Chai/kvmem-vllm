"""CPU checks for construction-only MTP sharing; no CUDA tensors or models."""
import ast
import copy
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm_kvmem import mtp_shared_load as compat


def config(tmp_path):
    checkpoint = tmp_path/'model'
    checkpoint.mkdir()
    text = NS(vocab_size=8, hidden_size=4, tie_word_embeddings=False)
    target = NS(model=str(checkpoint), model_weights=None, revision=None, code_revision=None,
        quantization='modelopt', dtype=torch.bfloat16, hf_text_config=text,
        hf_config=NS(architectures=['Qwen3_5ForConditionalGeneration'], tie_word_embeddings=False),
        word_embeddings_untied_by_checkpoint=False)
    draft = copy.deepcopy(target)
    draft.hf_config.architectures = ['Qwen3_5MTP']
    parallel = NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                  decode_context_parallel_size=1, prefill_context_parallel_size=1)
    return NS(model_config=target, parallel_config=parallel, lora_config=None,
        load_config=NS(load_format='safetensors'),
        speculative_config=NS(method='mtp', draft_model_config=draft,
                              draft_parallel_config=copy.deepcopy(parallel)))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # These are the real vLLM loader/placeholder/postprocess implementations.
    from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    events = []
    class Quant(QuantizeMethodBase):
        requires_device_loading = False
        def create_weights(self, layer, *args, **kwargs):
            raise AssertionError('The test owns its small CPU parameters')
        def apply(self, layer, *args, **kwargs):
            raise AssertionError('No inference in this CPU test')
        def process_weights_after_loading(self, layer):
            events.append(('postprocess', layer.label))
            with torch.no_grad():
                layer.weight.add_(1)
    class Weighted(nn.Module):
        def __init__(self, label, value):
            super().__init__()
            self.label = label
            self.weight = nn.Parameter(torch.full((2, 2), value))
            self.quant_method = Quant()
            def loader(param, value):
                events.append(('load', label))
                with torch.no_grad():
                    param.copy_(value)
            self.weight.weight_loader = loader
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = Weighted('target_embed', 2.)
    target.lm_head = Weighted('target_head', 3.)
    def forbidden(*args, **kwargs):
        raise AssertionError('Duplicate IO constructor allocated')
    module = NS(VocabParallelEmbedding=forbidden, ParallelLMHead=forbidden,
                Qwen3_5MTP=type('Qwen3_5MTP', (), {}))
    class LoRA: pass
    monkeypatch.setattr(compat, '_dependencies', lambda: (module, PPMissingLayer, LoRA, 1))
    monkeypatch.setenv(compat.ENABLED, '1')
    cfg = config(tmp_path)
    def original(target, cfg, *, share=True):
        draft = nn.Module()
        draft.model = nn.Module()
        draft.model.embed_tokens = module.VocabParallelEmbedding(8, 4)
        draft.lm_head = module.ParallelLMHead(8, 4, quant_config=object(), prefix='lm_head')
        assert isinstance(draft.model.embed_tokens, PPMissingLayer)
        assert isinstance(draft.lm_head, PPMissingLayer)
        draft.model.body = Weighted('mtp_body', 0.)
        loaded = AutoWeightsLoader(draft).load_weights([
            ('model.embed_tokens.weight', torch.full((2, 2), 99.)),
            ('model.body.weight', torch.full((2, 2), 5.)),
            ('lm_head.weight', torch.full((2, 2), 99.)),
        ])
        assert loaded == {'model.body.weight'}
        process_weights_after_loading(draft, cfg.speculative_config.draft_model_config,
                                      torch.device('cpu'))
        # Same post-load assignments as the official Eagle loader.
        if share:
            draft.model.embed_tokens = target.model.embed_tokens
            draft.lm_head = target.lm_head
        return draft
    return cfg, target, module, original, events


def test_real_loader_skips_placeholders_and_never_reloads_or_reprocesses_target(fixture):
    cfg, target, module, original, events = fixture
    constructors = module.VocabParallelEmbedding, module.ParallelLMHead
    draft = compat.wrap_load_eagle_model(original)(target, cfg)
    assert events == [('load', 'mtp_body'), ('postprocess', 'mtp_body')]
    assert torch.equal(target.model.embed_tokens.weight, torch.full((2, 2), 2.))
    assert torch.equal(target.lm_head.weight, torch.full((2, 2), 3.))
    assert torch.equal(draft.model.body.weight, torch.full((2, 2), 6.))
    assert draft.model.embed_tokens is target.model.embed_tokens and draft.lm_head is target.lm_head
    assert draft._kvmem_shared_io_placeholder
    assert constructors == (module.VocabParallelEmbedding, module.ParallelLMHead)


def test_constructor_bindings_restore_after_loader_failure(fixture):
    cfg, target, module, original, events = fixture
    constructors = module.VocabParallelEmbedding, module.ParallelLMHead
    def fail(target, cfg):
        module.VocabParallelEmbedding(8, 4)
        raise RuntimeError('load failed')
    with pytest.raises(RuntimeError, match='load failed'):
        compat.wrap_load_eagle_model(fail)(target, cfg)
    assert constructors == (module.VocabParallelEmbedding, module.ParallelLMHead)
    assert not compat._LOCK.locked()
    assert compat.wrap_load_eagle_model(original)(target, cfg)._kvmem_shared_io_placeholder


@pytest.mark.parametrize('change', ['checkpoint', 'tp', 'draft_pp', 'tie', 'quant', 'lora', 'format', 'own'])
def test_unsupported_target_or_draft_is_rejected_before_any_constructor(fixture, change, tmp_path):
    cfg, target, module, original, events = fixture
    if change == 'checkpoint':
        other = tmp_path/'other'; other.mkdir()
        cfg.speculative_config.draft_model_config.model = str(other)
    elif change == 'tp':
        cfg.parallel_config.tensor_parallel_size = 2
    elif change == 'draft_pp':
        cfg.speculative_config.draft_parallel_config.pipeline_parallel_size = 2
    elif change == 'tie':
        cfg.speculative_config.draft_model_config.hf_text_config.tie_word_embeddings = True
    elif change == 'quant':
        cfg.speculative_config.draft_model_config.quantization = 'gptq'
    elif change == 'lora':
        cfg.lora_config = object()
    elif change == 'format':
        cfg.load_config.load_format = 'gguf'
    else:
        module.Qwen3_5MTP.has_own_embed_tokens = True
    constructors = module.VocabParallelEmbedding, module.ParallelLMHead
    with pytest.raises(ValueError):
        compat.wrap_load_eagle_model(lambda *args: pytest.fail('original called'))(target, cfg)
    assert constructors == (module.VocabParallelEmbedding, module.ParallelLMHead)
    assert not events


@pytest.mark.parametrize('kind', ['disabled', 'other_architecture', 'other_method', 'plain'])
def test_unselected_paths_are_unchanged_without_loading_dependencies(tmp_path, monkeypatch, kind):
    cfg = config(tmp_path)
    monkeypatch.setenv(compat.ENABLED, '1')
    if kind == 'disabled':
        monkeypatch.delenv(compat.ENABLED)
    elif kind == 'other_architecture':
        cfg.speculative_config.draft_model_config.hf_config.architectures = ['OtherMTP']
    elif kind == 'other_method':
        cfg.speculative_config.method = 'eagle3'
    else:
        cfg.speculative_config = None
    monkeypatch.setattr(compat, '_dependencies', lambda: pytest.fail('loaded model dependencies'))
    sentinel = object()
    assert compat.wrap_load_eagle_model(lambda target, config: sentinel)(None, cfg) is sentinel


def test_identity_mismatch_and_target_mutation_fail_closed(fixture):
    cfg, target, module, original, events = fixture
    def missing_share(target, cfg):
        draft = original(target, cfg)
        draft.lm_head = nn.Identity()
        return draft
    with pytest.raises(RuntimeError, match='exact target'):
        compat.wrap_load_eagle_model(missing_share)(target, cfg)
    def mutated(target, cfg):
        draft = original(target, cfg)
        with torch.no_grad():
            target.lm_head.weight.add_(1)
        return draft
    with pytest.raises(RuntimeError, match='mutated target'):
        compat.wrap_load_eagle_model(mutated)(target, cfg)


def test_pinned_source_and_version_reject_changes(monkeypatch, tmp_path):
    root = Path(metadata.distribution('vllm').locate_file('vllm'))
    compat.verify_sources(root)
    source = (root/'v1/worker/gpu/spec_decode/eagle/utils.py').read_bytes()
    patched = compat.patched_source(source)
    ast.parse(patched)
    assert patched == source.decode()+compat._SUFFIX
    with pytest.raises(ImportError):
        compat.patched_source(source+b'\n')
    monkeypatch.setattr(compat.importlib.metadata, 'version', lambda name: '0.31.0')
    with pytest.raises(RuntimeError, match='0.30.0'):
        compat.verify_sources(root)


def test_plugin_does_not_enable_compatibility_without_explicit_switch(monkeypatch):
    from vllm_kvmem import integration
    monkeypatch.setattr(integration, '_registered', False)
    for key in (compat.ENABLED, 'KVMEM_DRAFT_LOAD_COMPAT', 'KVMEM_RUNNER_PROBE_DIR',
                'KVMEM_WINDOW_METADATA', 'KVMEM_Q8_TENSORCORE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(compat, 'enable_mtp_shared_load', lambda: pytest.fail('opt-in not set'))
    integration.register_plugin()


def test_real_import_alias_and_worker_environment_in_fresh_cpu_process():
    code = """
import os
from vllm_kvmem.mtp_shared_load import ENABLED, enable_mtp_shared_load
assert os.environ[ENABLED] == '1'
enable_mtp_shared_load()
from vllm.v1.worker.gpu.spec_decode.eagle import utils
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.ray.ray_env import get_env_vars_to_copy
assert hasattr(utils.load_eagle_model, '__wrapped__')
assert MTPSpeculator.load_draft_model.__globals__['load_eagle_model'] is utils.load_eagle_model
assert ENABLED in get_env_vars_to_copy()
assert not __import__('torch').cuda.is_initialized()
print('ALIASED_WRAPPER_AND_ENV_OK')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', VLLM_PLUGINS='',
               VLLM_KVMEM_MTP_SHARED_LOAD='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'ALIASED_WRAPPER_AND_ENV_OK' in result.stdout


def test_actual_official_postload_sharing_is_the_only_attachment(fixture, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as official
    cfg, target, module, build, events = fixture
    spec = cfg.speculative_config
    spec.moe_backend = spec.kv_cache_dtype = spec.attention_backend = None
    monkeypatch.setattr(official, 'get_pp_group', lambda: NS(world_size=1))
    monkeypatch.setattr(official, 'get_pp_safe_draft_load_config', lambda value: value)
    def get_model(**kwargs):
        assert kwargs['model_config'] is spec.draft_model_config
        draft = build(target, cfg, share=False)
        assert draft.model.embed_tokens is not target.model.embed_tokens
        assert draft.lm_head is not target.lm_head
        return draft
    monkeypatch.setattr(official, 'get_model', get_model)
    draft = compat.wrap_load_eagle_model(official.load_eagle_model)(target, cfg)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    assert events == [('load', 'mtp_body'), ('postprocess', 'mtp_body')]
