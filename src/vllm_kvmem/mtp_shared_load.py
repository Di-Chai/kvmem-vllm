"""Opt-in construction-only Qwen3.5 MTP sharing for pinned vLLM 0.30.0.

The official loader shares target embeddings/head AFTER loading the draft. Use
parameter-free placeholders during that temporary draft construction so the
discarded duplicates never consume GPU memory. The target is not attached to
the draft until the official post-load sharing code does so; draft weight load
and quantization postprocessing must never visit target modules.
"""
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import threading

ENABLED = 'VLLM_KVMEM_MTP_SHARED_LOAD'
TARGET = 'vllm.v1.worker.gpu.spec_decode.eagle.utils'
MODEL = 'vllm.model_executor.models.qwen3_5_mtp'
HASHES = {
    'v1/worker/gpu/spec_decode/eagle/utils.py':
        '94de6f55a1761f24f4aa1388cbc3b3763d4ab3ee53260330056ad31ce0b78236',
    'model_executor/models/qwen3_5_mtp.py':
        'a7550c4954899b38cecc536633c0fffd04820ed802610fb28cd4d79971b4f1f1',
    'model_executor/models/utils.py':
        '570a2eb26c7274c9790b6ff42e3fd30403ad6534c68d256f4d867d22e2924919',
    'model_executor/model_loader/base_loader.py':
        '5291c330c4ed5634a9ea1d535364c26162b60251aa2af03fd409b88c34d836e2',
    'model_executor/model_loader/utils.py':
        'aa398e74ee835537403e60186f2e9a81e8741a4b35fec68b47df6537716a45b7',
    'model_executor/model_loader/weight_tying.py':
        '8b950c56edb5f7586b522d13a045c415d391ce715e4590b16dbdf073e38a7af4',
}
_SUFFIX = """
from vllm_kvmem.mtp_shared_load import wrap_load_eagle_model as _kvmem_wrap_shared_io
load_eagle_model = _kvmem_wrap_shared_io(load_eagle_model)
"""
_LOCK = threading.Lock()


def verify_sources(root=None):
    if importlib.metadata.version('vllm') != '0.30.0':
        raise RuntimeError('MTP shared construction only supports vLLM 0.30.0')
    root = Path(root) if root is not None else Path(
        importlib.metadata.distribution('vllm').locate_file('vllm'))
    for relative, expected in HASHES.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ImportError('Unqualified MTP shared-construction dependency: '+relative)


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != HASHES['v1/worker/gpu/spec_decode/eagle/utils.py']:
        raise ImportError('Unqualified vLLM Eagle loader')
    return source.decode()+_SUFFIX


def _candidate(config):
    speculative = getattr(config, 'speculative_config', None)
    draft = getattr(speculative, 'draft_model_config', None)
    hf = getattr(draft, 'hf_config', None)
    return (getattr(speculative, 'method', None) == 'mtp' and
            getattr(hf, 'architectures', None) == ['Qwen3_5MTP'])


def _checkpoint(model):
    path = Path(getattr(model, 'model_weights', None) or model.model).expanduser()
    if not path.is_dir():
        raise ValueError('Shared MTP construction requires an existing local checkpoint')
    return path.resolve()


def _validate(config):
    speculative = config.speculative_config
    target, draft = config.model_config, speculative.draft_model_config
    if _checkpoint(target) != _checkpoint(draft):
        raise ValueError('Shared MTP construction requires the same target/draft checkpoint')
    for name in ('revision', 'code_revision', 'quantization', 'dtype'):
        if getattr(target, name, None) != getattr(draft, name, None):
            raise ValueError('Target/draft '+name+' differs')
    if getattr(config, 'lora_config', None) is not None:
        raise ValueError('Shared MTP construction does not qualify LoRA')
    if getattr(target.hf_config, 'architectures', None) not in (
            ['Qwen3_5ForConditionalGeneration'], ['Qwen3_5ForCausalLM']):
        raise ValueError('Shared MTP construction requires the Qwen3.5 target architecture')
    for parallel in (config.parallel_config, speculative.draft_parallel_config):
        if parallel is None or any(getattr(parallel, name, 1) != 1 for name in (
                'tensor_parallel_size', 'pipeline_parallel_size',
                'decode_context_parallel_size', 'prefill_context_parallel_size')):
            raise ValueError('Shared MTP construction requires one target/draft rank')
    for model in (target, draft):
        for hf in (model.hf_config, model.hf_text_config):
            if getattr(hf, 'tie_word_embeddings', None) is not False:
                raise ValueError('Shared MTP construction requires untied embeddings')
        if getattr(model, 'word_embeddings_untied_by_checkpoint', False):
            raise ValueError('Checkpoint-driven embedding retie is not qualified')
    for name in ('vocab_size', 'hidden_size'):
        a, b = getattr(target.hf_text_config, name), getattr(draft.hf_text_config, name)
        if type(a) is not int or a <= 0 or a != b:
            raise ValueError('Target/draft embedding geometry differs')
    if getattr(config.load_config, 'load_format', None) != 'safetensors':
        raise ValueError('Shared MTP construction requires safetensors')
    return target.hf_text_config.vocab_size, target.hf_text_config.hidden_size


def _dependencies():
    module = importlib.import_module(MODEL)
    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.lora.layers.base import BaseLayerWithLoRA
    from vllm.distributed.parallel_state import get_pp_group
    return module, PPMissingLayer, BaseLayerWithLoRA, get_pp_group().world_size


def _tensor_signature(module):
    """Host metadata only; no device reads, synchronization or tensor copies."""
    tensors = list(module.named_parameters())+list(module.named_buffers())
    result = []
    for name, value in tensors:
        try:
            version = value._version
        except RuntimeError:  # Inference tensors may not have version counters.
            version = None
        result.append((name, id(value), value.data_ptr(), tuple(value.shape),
                       value.dtype, value.device, version))
    return id(getattr(module, 'quant_method', None)), tuple(result)


def _shared_modules(target, missing, lora):
    language = target.get_language_model() if hasattr(target, 'get_language_model') else target
    inner = language.model
    embedding = getattr(inner, 'embed_tokens', None)
    head = getattr(language, 'lm_head', None)
    if head is None:
        head = getattr(target, 'lm_head', None)
    if any(value is None or isinstance(value, (missing, lora)) for value in (embedding, head)):
        raise ValueError('Target must own complete unwrapped embedding and lm_head')
    return embedding, head


def wrap_load_eagle_model(original):
    @functools.wraps(original)
    def load(target, config):
        if os.getenv(ENABLED) != '1' or not _candidate(config):
            return original(target, config)
        geometry = _validate(config)
        module, missing, lora, pp_world = _dependencies()
        if pp_world != 1:
            raise ValueError('Actual pipeline group differs from single-rank config')
        cls = module.Qwen3_5MTP
        if any(getattr(cls, name, False) for name in ('has_own_embed_tokens', 'has_own_lm_head')):
            raise ValueError('MTP declares independent embedding/head weights')
        embedding, head = _shared_modules(target, missing, lora)
        signatures = [_tensor_signature(value) for value in (embedding, head)]
        if not _LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent MTP construction is not qualified')
        try:
            constructors = {name: getattr(module, name)
                            for name in ('VocabParallelEmbedding', 'ParallelLMHead')}
        except BaseException:
            _LOCK.release()
            raise
        calls = dict.fromkeys(constructors, 0)
        owner = threading.get_ident()

        def factory(name):
            def create(*args, **kwargs):
                if threading.get_ident() != owner:
                    raise RuntimeError('Another thread entered scoped MTP construction')
                if len(args) != 2 or tuple(args) != geometry or calls[name]:
                    raise ValueError('Unexpected shared MTP constructor invocation: '+name)
                calls[name] += 1
                return missing()
            return create

        try:
            for name in constructors:
                setattr(module, name, factory(name))
            draft = original(target, config)
        finally:
            for name, constructor in constructors.items():
                setattr(module, name, constructor)
            _LOCK.release()
        if calls != {'VocabParallelEmbedding': 1, 'ParallelLMHead': 1}:
            raise RuntimeError('Shared MTP constructor contract was not exercised')
        if any(getattr(draft, name, False) for name in ('has_own_embed_tokens', 'has_own_lm_head')):
            raise RuntimeError('Loaded MTP unexpectedly owns independent IO weights')
        if draft.model.embed_tokens is not embedding or draft.lm_head is not head:
            raise RuntimeError('Official MTP loader did not share the exact target modules')
        if signatures != [_tensor_signature(value) for value in (embedding, head)]:
            raise RuntimeError('MTP loading mutated target embedding/head state')
        draft._kvmem_shared_io_placeholder = True
        logging.getLogger(__name__).info(
            'Qwen3.5 MTP construction skipped duplicate embedding/lm_head; '
            'official sharing identities and target tensor metadata verified')
        return draft
    return load


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(patched_source(self.get_data(self.path)), self.path, 'exec')


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError('Cannot locate pinned vLLM Eagle loader')
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable_mtp_shared_load():
    if os.getenv(ENABLED) != '1':
        return
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return
    verify_sources()
    if TARGET in sys.modules or MODEL in sys.modules:
        raise RuntimeError('MTP shared-construction compatibility must precede model imports')
    sys.meta_path.insert(0, _Finder())
