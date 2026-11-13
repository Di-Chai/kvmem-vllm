"""Opt-in adapter for FlashInfer's GEMM warmup bucket boundary.

vLLM can warm up 4136 model rows while FlashInfer's native GEMM mapper maps
4097..8192 rows to 8192. Only GEMM profiling tensors need the larger bucket;
the model's scheduled rows and persistent capture buffers remain unchanged.
This adapter is deliberately limited to the qualified FlashInfer config ABI.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import threading
from . import settings as _settings


_CONFIG_NAMES = (
    '_FP8_GEMM_SM100_TUNING_CONFIG',
    '_MM_FP4_TUNING_CONFIG_8x4',
    '_MM_FP4_TUNING_CONFIG_128x4',
)
_STATE = '_kvmem_flashinfer_gemm_bucket_state'


@dataclass
class _AlignmentState:
    max_model_rows: int
    cache_path: Path
    report_path: Path
    alignments: list = field(default_factory=list)
    seed: dict | None = None
    status: str = 'installed'

    def write_report(self):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps({
            'enabled': True,
            'status': self.status,
            'max_model_rows': self.max_model_rows,
            'cache_path': str(self.cache_path),
            'cache_isolation': 'per-run writes; optional read-only baseline seed',
            'seed': self.seed,
            'alignments': self.alignments,
        }, indent=2) + '\n')


def _seed_cache(state, seed_path):
    """Copy unchanged baseline tactics; new bucket keys still require tuning."""
    if not seed_path:
        return
    source = Path(seed_path).expanduser().resolve(strict=True)
    payload = source.read_bytes()
    state.cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with state.cache_path.open('xb') as target:
            target.write(payload)
        applied = True
    except FileExistsError:
        applied = False
    state.seed = dict(path=str(source), sha256=hashlib.sha256(payload).hexdigest(),
                      copied_to_private_cache=applied)


def _aligned_buckets(config, buckets):
    """Use the same bucket identities at warmup and ordinary inference."""
    specs = config.dynamic_tensor_specs
    if len(specs) != 1:
        raise RuntimeError('Qualified FlashInfer GEMM must have one dynamic tensor spec')
    mapper = specs[0].map_to_tuning_buckets
    return tuple(sorted({mapper(int(value)) for value in buckets}))


@contextmanager
def _align_gemm_overrides(tuner, configs, autotune, state):
    """Change the shared mapper hook only for the current warmup thread.

    Both choose_one and cuDNN's runner factory consult this hook. Changing
    choose_one alone would leave cuDNN's internal shape cache inconsistent.
    The upstream context manager restores its thread-local overrides even
    on failure; the original hook is restored when the dummy runs finish.
    """
    original = tuner._apply_tuning_overrides
    previous = vars(tuner).get('_apply_tuning_overrides')
    had_override = '_apply_tuning_overrides' in vars(tuner)
    owner_thread = threading.get_ident()

    def apply(config):
        name = next((name for name, value in configs.items() if value is config), None)
        buckets = tuner._override_tuning_buckets
        if (threading.get_ident() != owner_thread or name is None or
                not tuner.is_tuning_mode or buckets is None):
            return original(config)
        aligned = _aligned_buckets(config, buckets)
        record = dict(config=name, original_buckets=list(buckets),
                      aligned_buckets=list(aligned), round_up=True)
        if record not in state.alignments:
            state.alignments.append(record)
        # Do not mutate the config, constraints, initializers, or global
        # override properties. Upstream also caches this derived config.
        with autotune(tuning_buckets=aligned, round_up=True):
            return original(config)

    tuner._apply_tuning_overrides = apply
    try:
        yield
    finally:
        if had_override:
            tuner._apply_tuning_overrides = previous
        else:
            del tuner._apply_tuning_overrides


def _install_warmup_adapter(warmup, tuner_class, gemm, autotune):
    if getattr(warmup, '_kvmem_gemm_bucket_adapter', False):
        return
    configs = {name: getattr(gemm, name) for name in _CONFIG_NAMES}
    # Fail before startup rather than silently skip alignment after an ABI change.
    for config in configs.values():
        if _aligned_buckets(config, (4096, 4136)) != (4096, 8192):
            raise RuntimeError('Unsupported FlashInfer GEMM bucket mapping')
    if not callable(getattr(tuner_class, '_apply_tuning_overrides', None)):
        raise RuntimeError('FlashInfer lacks the qualified tuning override hook')
    original_runs = warmup._run_flashinfer_autotune_dummy_runs
    original_cache = warmup.resolve_flashinfer_autotune_file

    @wraps(original_runs)
    def runs(runner, *args, **kwargs):
        state = getattr(runner, _STATE, None)
        if state is None:
            return original_runs(runner, *args, **kwargs)
        state.status = 'tuning'
        state.write_report()
        try:
            with _align_gemm_overrides(tuner_class.get(), configs, autotune, state):
                result = original_runs(runner, *args, **kwargs)
        except BaseException:
            state.status = 'failed'
            state.write_report()
            raise
        state.status = 'complete'
        state.write_report()
        return result

    @wraps(original_cache)
    def cache_path(runner):
        state = getattr(runner, _STATE, None)
        if state is None:
            return original_cache(runner)
        state.cache_path.parent.mkdir(parents=True, exist_ok=True)
        return state.cache_path

    warmup._run_flashinfer_autotune_dummy_runs = runs
    warmup.resolve_flashinfer_autotune_file = cache_path
    warmup._kvmem_gemm_bucket_adapter = True


def install_flashinfer_gemm_bucket_alignment(runner, options, output):
    if not options.flashinfer_gemm_bucket_alignment:
        return None
    existing = getattr(runner, _STATE, None)
    if existing is not None:
        return existing
    # vLLM's None means automatic enablement on supported hardware.
    if runner.vllm_config.kernel_config.enable_flashinfer_autotune is False:
        raise ValueError('flashinfer_gemm_bucket_alignment requires FlashInfer autotuning')
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.gemm import gemm_base
    from vllm.model_executor.warmup import kernel_warmup

    # Resolve before attaching this runner's private-cache state. The serve
    # launcher deliberately filters KVMEM_* environment variables, so reuse
    # upstream tactics by default instead of depending on an explicit seed.
    # The upstream file is only read; all subsequent writes use cache_path.
    seed_path = _settings.get('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED')
    if not seed_path:
        upstream_cache = kernel_warmup.resolve_flashinfer_autotune_file(runner)
        if upstream_cache.is_file():
            seed_path = upstream_cache
    _install_warmup_adapter(kernel_warmup, AutoTuner, gemm_base, autotune)
    config_hash = runner.vllm_config.compute_hash(include_version=False)
    digest = hashlib.sha256(config_hash.encode()).hexdigest()
    root = Path(output) / 'flashinfer-gemm-buckets'
    state = _AlignmentState(
        max_model_rows=runner.max_num_tokens,
        cache_path=root / digest / 'autotune_configs.json',
        report_path=root / 'alignment.json')
    _seed_cache(state, seed_path)
    setattr(runner, _STATE, state)
    state.write_report()
    return state
