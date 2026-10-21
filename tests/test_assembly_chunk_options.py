"""Public assembly batching is bounded and immutable after scratch allocation."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.options import RuntimeOptions
from test_options import config


@pytest.mark.parametrize('value', [1, 32, 128, 256])
def test_public_assembly_chunk_option_round_trips_without_changing_input(value):
    payload = {'kvmem': {'assembly_chunk_blocks': value}}
    parsed = RuntimeOptions.from_vllm(config(payload))
    assert parsed.assembly_chunk_blocks == value
    assert payload == {'kvmem': {'assembly_chunk_blocks': value}}
    with pytest.raises(FrozenInstanceError):
        parsed.assembly_chunk_blocks = 32


@pytest.mark.parametrize('value', [0, -1, 257, True, 1.0, '128', None])
def test_invalid_assembly_chunk_option_fails_before_allocation(value):
    with pytest.raises(ValueError, match='Assembly chunk blocks'):
        RuntimeOptions.from_vllm(config({'kvmem': {'assembly_chunk_blocks': value}}))


def test_default_assembly_chunk_and_cpu_archive_admission_fit_128_records(tmp_path):
    from vllm_kvmem.storage import make_shared_archive
    assert RuntimeOptions().assembly_chunk_blocks == 32
    # Serving uses FP8 raw K + FP8 V, 17 layers, block=32, 4 KV heads, dim=256.
    record_bytes = 17 * 32 * 4 * 256 * 2
    options = RuntimeOptions(archive_backend='cpu', cpu_read_cache_bytes=record_bytes * 256,
                             assembly_chunk_blocks=128)
    archive = make_shared_archive(options, tmp_path, record_bytes, 1)
    try:
        assert archive.io_max_bytes == 0
        archive._check_extent(128 * record_bytes)
        assert archive._has_capacity(128 * record_bytes)
    finally:
        assert archive.close()


def test_runner_uses_configured_chunks_and_explicit_override_then_freezes_staging(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('vllm')
    import numpy as np
    from vllm_kvmem import window_assembly as window
    key = torch.zeros((4, 4, 1, 4), dtype=torch.float8_e4m3fn)
    module = NS(num_kv_heads=1, head_dim=4,
        rotary_emb=NS(cos_sin_cache=torch.zeros(16, 4)),
        attn=NS(kv_cache_dtype='fp8_e4m3', layer_name='attention',
                kv_cache=object(), _k_scale=torch.tensor(1.)))
    runner = NS(model=NS(named_modules=lambda: [('module', module)]),
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=['attention'])]),
        req_states=NS(req_id_to_index={'a': 0}),
        block_tables=NS(num_blocks=NS(np=np.array([[2]])),
                        block_tables=[NS(gpu=torch.tensor([[1, 2]], dtype=torch.int32))]),
        _kvmem_capture_runtime=NS(options=RuntimeOptions(assembly_chunk_blocks=128)))
    monkeypatch.setattr(window, 'native_vllm_fp8_views', lambda *_: (key, key.clone()))
    configured = window.NativeWindowAssembler.from_runner(runner, 'a', ['module'], 4)
    assert configured.chunk_blocks == 128
    assert configured.staging.slot_bytes == 1 * 1 * 4 * 4 * 128 * 2
    assert configured.close()
    explicit = window.NativeWindowAssembler.from_runner(runner, 'a', ['module'], 4,
        chunk_blocks=32, arena='refresh')
    assert explicit.chunk_blocks == 32 and explicit.close()
    with pytest.raises(ValueError, match='geometry changed'):
        window.NativeWindowAssembler.from_runner(runner, 'a', ['module'], 4, chunk_blocks=32)
    assert runner._kvmem_assembly_staging is configured.staging
    for invalid in (0, 257, True):
        with pytest.raises(ValueError, match='Assembly chunk blocks'):
            window.NativeWindowAssembler.from_runner(runner, 'a', ['module'], 4, chunk_blocks=invalid)
