"""Host-only checks for the opt-in quad reduce-scatter static FA2 statistics."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import native_attention_build as build
from vllm_kvmem.options import RuntimeOptions
from test_options import config
from test_native_prefill_static_aligned import dispatch_fixture

ROOT = Path(__file__).resolve().parents[1] / 'src/vllm_kvmem'


def test_option_is_default_off_strict_and_requires_static_aligned():
    assert not RuntimeOptions.from_vllm(config({})).action_native_prefill_static_quad
    payload = dict(action_native_prefill_static_aligned=True, action_native_scores='reuse',
        action_attention=True, session_capacity=1, session_checkpoint_bytes=4096,
        async_archive_retirement=True)
    for value in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='static quad must be boolean'):
            RuntimeOptions.from_vllm(config({'kvmem': dict(payload, action_native_prefill_static_quad=value)}))
    with pytest.raises(ValueError, match='requires native prefill static aligned'):
        RuntimeOptions.from_vllm(config({'kvmem': dict(payload, action_native_prefill_static_aligned=False,
                                                       action_native_prefill_static_quad=True)}))
    assert RuntimeOptions.from_vllm(config({'kvmem': dict(
        payload, action_native_prefill_static_quad=True)})).action_native_prefill_static_quad


def test_enabled_bank_builds_and_dispatches_only_quad_static_module(monkeypatch):
    env = dispatch_fixture(monkeypatch)
    env.bank.static_quad_enabled = True
    env.execute()
    assert env.builds == [{}, {'static_aligned': True, 'quad_store': True}]
    assert env.calls[-1][0] == {'static_aligned': True, 'quad_store': True}
    assert len(env.calls[-1][1]) == 24 and env.calls[-1][1][:16] == tuple(env.args[:16])
    # Graph capture keeps the generic module exactly as before.
    env.capturing[0] = True
    env.execute()
    assert env.calls[-1][0] == {}


def test_isolated_quad_module_uses_static_overlay_and_rejects_other_routes(tmp_path, monkeypatch):
    overlays = []
    def overlay(_directory, **kw):
        overlays.append(kw); p = tmp_path / ('static' if kw else 'old'); p.mkdir(exist_ok=True); return p
    src = tmp_path / 'source'; src.mkdir()
    (src / 'kernel.cu').write_text('#include <flashinfer/attention/prefill.cuh>')
    (src / 'batch_prefill_config.inc').write_text('config')
    def generator(*args):
        result = NS(sources=[src / 'kernel.cu']); result.build_and_load = lambda: args; return result
    monkeypatch.setattr(build, 'source_overlay', overlay)
    monkeypatch.setitem(sys.modules, 'flashinfer.jit.attention', NS(gen_customize_batch_prefill_module=generator))
    build._prefill_module.cache_clear()
    try:
        static = build.prefill_module(None, 'q', 'kv', 'o', 'i', 256, static_aligned=True)
        quad = build.prefill_module(None, 'q', 'kv', 'o', 'i', 256, static_aligned=True, quad_store=True)
        assert quad[-2] == 'KVMemStaticQuadAttention' and quad[1].endswith('_quad_store')
        assert quad[8:12] == static[8:12] and overlays[-1] == {'prefill_static_aligned': True}
        for kw in ({}, {'static_aligned': True, 'static_readonly': True}):
            with pytest.raises(ValueError, match='Quad statistics'):
                build.prefill_module(None, 'q', 'kv', 'o', 'i', 256, quad_store=True, **kw)
    finally:
        build._prefill_module.cache_clear()


def test_quad_capture_keeps_static_arithmetic_and_has_no_general_path():
    header = (ROOT / 'native_stats.cuh').read_text()
    quad = header.split('__device__ void store64_weight_static_aligned(', 1)[1].split('\n  }\n', 1)[0]
    # Same per-lane four-term order and the same final log/write form.
    assert 's[n] = a + b + c + d;' in quad and 'center + logf(sum), false)' in quad
    assert quad.count('__shfl_xor_sync') == 3 and 'pieces[' not in quad and 'add(' not in quad
    capture = (ROOT / 'native_prefill_stats.cuh').read_text().split('struct KVMemStaticQuadAttention', 1)[1]
    assert 'store64_weight_static_aligned(' in capture and 'stats.store16_weight(' not in capture
    assert 'kv_chunk_size_ptr' not in capture
