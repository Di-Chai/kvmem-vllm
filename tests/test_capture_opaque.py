"""Host ABI/effect tests; GPU byte equality and graph capture are separate."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import capture_opaque as opaque
from vllm_kvmem.capture import CapturedLayer, _project_with_capture
from vllm_kvmem.capture_compile import (CAPTURE_COMPILE_KEY,
    capture_compile_fingerprint, validate_capture_compile_config)
from vllm_kvmem.options import RuntimeOptions
from test_options import config


class Owner:
    pass


def make_layer(name='target', capacity=8, heads=2, kv_heads=1, dim=4, gate=False):
    layer = Owner()
    layer.attn = Owner()
    layer.attn.layer_name = name
    layer.num_heads, layer.num_kv_heads, layer.head_dim = heads, kv_heads, dim
    layer.q_size, layer.kv_size = heads*dim, kv_heads*dim
    layer.q_norm = NS(weight=torch.zeros(dim, dtype=torch.bfloat16), variance_epsilon=1e-6)
    layer.k_norm = NS(weight=torch.zeros(dim, dtype=torch.bfloat16))
    layer.rotary_emb = NS(rotary_dim=dim, cos_sin_cache=torch.zeros(32, dim))
    layer._kvmem_capture_name = name
    layer._kvmem_capture_buffers = CapturedLayer(name, torch.arange(capacity),
        torch.full((capacity, heads, dim), -1, dtype=torch.bfloat16),
        torch.full((capacity, kv_heads, dim), -1, dtype=torch.bfloat16),
        torch.full((capacity, kv_heads, dim), -1, dtype=torch.bfloat16))
    if gate:
        layer._kvmem_capture_gate = torch.ones((), dtype=torch.int32)
        layer._kvmem_mtp_prefix_only = True
    return layer


@pytest.fixture
def host_op(monkeypatch):
    """Only this test fixture registers CPU execution; production is CUDA-only."""
    layers = [make_layer(), make_layer('draft', gate=True)]
    stats = opaque.OpaqueCaptureStats('a'*64)
    context = NS(no_compile_layers={layer.attn.layer_name: layer.attn for layer in layers})
    for layer in layers:
        layer.attn._kvmem_opaque_projection_binding = opaque._make_binding(
            layer, stats, require_cuda=False)
    calls = []

    def kernel(q_gate, key, *args, **kwargs):
        # A controllable stand-in verifies addresses/gates/arguments and effect
        # execution, not CUDA norm/RoPE arithmetic (covered by GPU qualification).
        heads, _, dim = args[5:8]
        q = q_gate.reshape(q_gate.shape[0], heads, 2*dim)[..., :dim].reshape(q_gate.shape[0], heads*dim).clone()
        gate = q_gate.reshape(q_gate.shape[0], heads, 2*dim)[..., dim:].reshape(q_gate.shape[0], heads*dim).clone()
        k = key.clone()
        calls.append((q_gate.clone(), args, kwargs))
        enabled = kwargs['capture_enabled']
        if enabled is None or enabled.item():
            kwargs['raw_q_output'].copy_(q)
            kwargs['raw_k_output'].copy_(k)
            kwargs['value_output'].copy_(kwargs['value_input'])
        return q, k, gate, kwargs['raw_q_output'], kwargs['raw_k_output']

    monkeypatch.setattr(opaque, '_forward_context', lambda: context)
    monkeypatch.setattr(opaque, 'fused_qk_rmsnorm_rope_gate', kernel)
    import vllm_kvmem.capture as capture
    monkeypatch.setattr(capture, 'fused_qk_rmsnorm_rope_gate', kernel)
    library = torch.library.Library('kvmem', 'FRAGMENT')
    try:
        library.impl('capture_projection', opaque._project_impl, 'CPU')
        yield NS(layers=layers, stats=stats, context=context, calls=calls)
    finally:
        torch._dynamo.reset()
        library._destroy()


def inputs(layer, tokens=3):
    width = layer.q_size*2 + layer.kv_size*2
    return (torch.arange(tokens*width, dtype=torch.bfloat16).reshape(tokens, width),
            torch.arange(tokens))


def test_option_strict_boolean_and_worker_roundtrip():
    assert not RuntimeOptions().capture_opaque_projection
    for enabled in (False, True):
        assert RuntimeOptions.from_vllm(config({'kvmem': {
            'capture_opaque_projection': enabled}})).capture_opaque_projection is enabled
    for invalid in (1, 0, 'true', None):
        with pytest.raises(ValueError, match='Opaque capture projection must be boolean'):
            RuntimeOptions(capture_opaque_projection=invalid)


def test_compile_identity_is_required_only_on_and_sensitive_to_each_source(tmp_path, monkeypatch):
    import vllm_kvmem.capture_compile as module
    assert validate_capture_compile_config(NS(), False) is None
    with pytest.raises(ValueError, match='Disabled opaque capture projection'):
        validate_capture_compile_config(NS(additional_config={CAPTURE_COMPILE_KEY: 'stale'}), False)
    fingerprint = capture_compile_fingerprint()
    assert len(fingerprint) == 64
    for value in (None, {}, {CAPTURE_COMPILE_KEY: 'stale'}):
        with pytest.raises(ValueError, match='source fingerprint'):
            validate_capture_compile_config(NS(additional_config=value), True)
    assert validate_capture_compile_config(NS(additional_config={CAPTURE_COMPILE_KEY: fingerprint}), True) == fingerprint
    from pathlib import Path
    original = Path(module.__file__).parent
    for name in module._SOURCES:
        (tmp_path/name).write_bytes((original/name).read_bytes())
    monkeypatch.setattr(module, '__file__', str(tmp_path/'capture_compile.py'))
    assert capture_compile_fingerprint() == fingerprint
    for name in module._SOURCES:
        data = (tmp_path/name).read_bytes()
        (tmp_path/name).write_bytes(data+b'\n# changed\n')
        assert capture_compile_fingerprint() != fingerprint
        (tmp_path/name).write_bytes(data)


def test_fake_does_not_read_context_or_capture_buffers(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    monkeypatch.setattr(opaque, '_forward_context', lambda: pytest.fail('fake read owner'))
    with FakeTensorMode():
        for tokens in (0, 3):
            x = torch.empty(tokens, 24, dtype=torch.bfloat16)
            q, k, gate = torch.ops.kvmem.capture_projection(x, torch.arange(tokens), 'missing', 8, 4)
            assert [tuple(t.shape) for t in (q, k, gate)] == [(tokens, 8), (tokens, 4), (tokens, 8)]
            assert all(t.is_contiguous() and t.dtype == x.dtype for t in (q, k, gate))
            assert len({id(t) for t in (x, q, k, gate)}) == 4


@pytest.mark.parametrize('draft', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_adapter_preserves_outputs_targets_gate_and_original_kernel_arguments(host_op, draft, enabled):
    layer = host_op.layers[int(draft)]
    if draft:
        layer._kvmem_capture_gate.fill_(int(enabled))
    x, positions = inputs(layer)
    positions = positions.repeat(3, 1)  # ordinary position first-row normalization
    original = _project_with_capture(layer, x, positions)
    expected = tuple(t.clone() for t in (layer._kvmem_capture_buffers.query,
        layer._kvmem_capture_buffers.key, layer._kvmem_capture_buffers.value))
    before = host_op.calls[-1]
    layer._kvmem_opaque_projection = True
    actual = _project_with_capture(layer, x, positions)
    after = host_op.calls[-1]
    assert all(torch.equal(left, right) for left, right in zip(original, actual))
    assert actual[2].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert all(torch.equal(left, right) for left, right in zip(expected,
        (layer._kvmem_capture_buffers.query, layer._kvmem_capture_buffers.key,
         layer._kvmem_capture_buffers.value)))
    for left, right in zip(before[1], after[1]):
        if isinstance(left, torch.Tensor):
            assert left.data_ptr() == right.data_ptr() and torch.equal(left, right)
        else:
            assert left == right
    for key in before[2]:
        left, right = before[2][key], after[2][key]
        if isinstance(left, torch.Tensor):
            assert left.data_ptr() == right.data_ptr() and left.shape == right.shape
        else:
            assert left == right
    assert host_op.stats.host_dispatch_count == 1


@pytest.mark.parametrize('change', ['buffer', 'gate', 'owner', 'shape', 'context'])
def test_changed_owner_or_target_fails_before_any_write(host_op, change):
    layer = host_op.layers[1]
    x, positions = inputs(layer)
    if change == 'buffer': layer._kvmem_capture_buffers.query = layer._kvmem_capture_buffers.query.clone()
    if change == 'gate': layer._kvmem_capture_gate = layer._kvmem_capture_gate.clone()
    if change == 'owner': layer.attn = Owner()
    if change == 'shape': layer.q_size += 1
    if change == 'context': host_op.context.no_compile_layers = {}
    with pytest.raises(RuntimeError, match='Opaque capture projection'):
        opaque._project_impl(x, positions, 'draft', 8, 4)
    assert not host_op.calls


def test_positions_rebinding_and_owner_context_isolation(host_op):
    first = host_op.layers[0]
    second = make_layer()  # same layer name, independent model/forward context
    second.attn._kvmem_opaque_projection_binding = opaque._make_binding(second, host_op.stats, require_cuda=False)
    first._kvmem_capture_buffers.positions = torch.arange(10)
    x, positions = inputs(first)
    opaque._project_impl(x, positions, 'target', 8, 4)
    assert first._kvmem_capture_buffers.query[0, 0, 0] == 0
    host_op.context.no_compile_layers['target'] = second.attn
    opaque._project_impl(x+10, positions, 'target', 8, 4)
    assert second._kvmem_capture_buffers.query[0, 0, 0] == 10
    assert first._kvmem_capture_buffers.query[0, 0, 0] == 0


@pytest.mark.parametrize('tokens,q_size,kv_size', [(9, 8, 4), (3, 4, 4), (3, 8, 8)])
def test_capacity_or_abi_mismatch_is_prelaunch(host_op, tokens, q_size, kv_size):
    x, positions = inputs(host_op.layers[0], tokens)
    with pytest.raises(ValueError, match='geometry/capacity'):
        opaque._project_impl(x, positions, 'target', q_size, kv_size)
    assert not host_op.calls


def test_installation_is_atomic_and_rejects_missing_or_duplicate_context_owners(monkeypatch):
    first, second = make_layer(), make_layer('draft', gate=True)
    runner = NS(model=NS(named_modules=lambda: [('target', first)]), _kvmem_layers=['target'],
        _kvmem_mtp_modules={'__mtp__.draft': second}, vllm_config=NS(compilation_config=NS(
            static_forward_context={'target': first.attn})))
    original = opaque._make_binding
    monkeypatch.setattr(opaque, '_make_binding', lambda layer, stats: original(layer, stats, require_cuda=False))
    with pytest.raises(ValueError, match='unique forward-context owners'):
        opaque.install_opaque_capture(runner, 'a'*64)
    assert not hasattr(first, '_kvmem_opaque_projection')
    runner.vllm_config.compilation_config.static_forward_context['draft'] = second.attn
    stats = opaque.install_opaque_capture(runner, 'a'*64)
    assert stats.snapshot()['layers_installed'] == 2
    assert stats.snapshot()['graph_replays_excluded'] is True
    with pytest.raises(ValueError, match='already installed'):
        opaque.install_opaque_capture(runner, 'a'*64)


@pytest.mark.parametrize('backend', ['aot_eager', 'inductor'])
def test_ordered_effect_survives_dead_outputs_duplicate_calls_and_gate_changes(host_op, backend):
    layer = host_op.layers[1]
    x, positions = inputs(layer)

    def forward(x, positions):
        torch.ops.kvmem.capture_projection(x, positions, 'draft', 8, 4)
        torch.ops.kvmem.capture_projection(x, positions, 'draft', 8, 4)
        return x + 1

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    with torch.no_grad():
        assert torch.equal(compiled(x, positions), x+1)
        assert host_op.stats.host_dispatch_count == 2
        original = layer._kvmem_capture_buffers.query.clone()
        layer._kvmem_capture_gate.zero_()
        compiled(x+10, positions)
        assert host_op.stats.host_dispatch_count == 4
        assert torch.equal(layer._kvmem_capture_buffers.query, original)
        layer._kvmem_capture_gate.fill_(1)
        compiled(x+20, positions)
        assert host_op.stats.host_dispatch_count == 6
        assert layer._kvmem_capture_buffers.query[0, 0, 0] == original[0, 0, 0]+20


def test_runner_rejects_missing_compile_identity_before_native_load(monkeypatch):
    from vllm_kvmem.runner import make_probe_runner
    loaded = []
    class Base:
        def load_model(self): loaded.append(True)
    runner = make_probe_runner(Base)()
    runner.vllm_config = config({'kvmem': {'capture_opaque_projection': True}})
    with pytest.raises(ValueError, match='source fingerprint'):
        runner.load_model()
    assert not loaded
