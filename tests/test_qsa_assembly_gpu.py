"""Exercise native B12X selector kernels and actual padded FP8 page views."""
import os
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='explicit GPU opt-in')


@pytest.fixture
def native_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        yield


def test_actual_b12x_projection_capture_preserves_native_outputs_and_mtp_prefix(native_config):
    import torch
    from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSAAttention
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm_kvmem.adapters.capture import capture_to
    from vllm_kvmem.adapters.qsa_adapter import install_qsa_projection, is_b12x_qsa
    # Instantiate the real projection implementation without loading MoE/PLE.
    layer = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.dcp_size, layer.num_heads, layer.num_kv_heads = 1, 8, 2
    layer.num_projected_kv_heads, layer.head_dim = 2, 256
    layer.q_size, layer.kv_size = 8*256, 2*256
    layer.kv_cache_dtype, layer.compress_ratio = 'fp8_e4m3', 4
    layer.index_head_dim, layer.overlap_input_projections = 128, False
    layer.max_tokens, layer.device = 128, torch.device('cuda')
    layer.q_norm = GemmaRMSNorm(256, eps=1e-6).cuda().bfloat16()
    layer.k_norm = GemmaRMSNorm(256, eps=1e-6).cuda().bfloat16()
    layer.q_norm.weight.data.zero_(); layer.k_norm.weight.data.zero_()
    layer.rotary_emb = get_rope(256, max_position=512,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000000.,
            'partial_rotary_factor': .25, 'mrope_section': [11, 11, 10],
            'mrope_interleaved': True}).cuda()
    selector = torch.randn(16, 128, device='cuda', dtype=torch.bfloat16)
    layer.indexer = NS(project=lambda hidden: (hidden, selector))
    qkv = torch.randn(16, layer.q_size*2+layer.kv_size*2, device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(37, 53, device='cuda')
    qg, raw_key, _ = qkv.split((layer.q_size*2, layer.kv_size, layer.kv_size), dim=-1)
    raw_query = qg.unflatten(-1, (layer.num_heads, layer.head_dim*2)).chunk(2, -1)[0]
    expected_raw_query = layer.q_norm(raw_query).clone()
    expected_raw_key = layer.k_norm(raw_key.unflatten(-1, (layer.num_kv_heads, layer.head_dim))).clone()
    expected = layer._project_qkv_gate(qkv, positions)
    assert is_b12x_qsa(layer)
    install_qsa_projection(layer, 'main')
    captures = []
    with capture_to(captures.append):
        actual = layer._project_qkv_gate(qkv, positions)
    assert len(captures) == 1
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(captures[0].query, expected_raw_query)
    assert torch.equal(captures[0].key, expected_raw_key)
    layer._kvmem_mtp_prefix_only = True
    layer._kvmem_mtp_prefix_active = True
    layer.indexer.project(selector)
    saved = layer._kvmem_index_raw[:16].clone()
    layer._kvmem_mtp_prefix_active = False
    selector.zero_()
    layer.indexer.project(selector)
    assert torch.equal(layer._kvmem_index_raw[:16], saved)

    # Native full graphs bypass Python on replay. Prefix capture must be baked
    # into its own graph; speculative draft replay must leave all raw planes
    # untouched, including the BF16 selector plane.
    from vllm_kvmem.adapters.capture import CapturedLayer
    layer._kvmem_capture_buffers = CapturedLayer('main', positions,
        torch.empty_like(expected_raw_query), torch.empty_like(expected_raw_key),
        torch.empty(16, 2, 256, device='cuda', dtype=torch.bfloat16))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def forward():
        result = layer._project_qkv_gate(qkv, positions)
        layer.indexer.project(selector)
        return result
    with torch.cuda.stream(stream):
        layer._kvmem_mtp_prefix_active = True
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    prefix_graph, draft_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(prefix_graph):
        prefix_output = forward()
    layer._kvmem_mtp_prefix_active = False
    with torch.cuda.graph(draft_graph):
        forward()
    qkv.normal_()
    selector.normal_()
    positions.add_(17)
    expected = layer._kvmem_original_projection(qkv, positions)
    prefix_graph.replay()
    buffers = layer._kvmem_capture_buffers
    saved_planes = [t.clone() for t in (buffers.query, buffers.key, buffers.value, layer._kvmem_index_raw[:16])]
    assert all(torch.equal(a, b) for a, b in zip(prefix_output, expected))
    qg, raw_key, raw_value = qkv.split((layer.q_size*2, layer.kv_size, layer.kv_size), dim=-1)
    raw_query = qg.unflatten(-1, (layer.num_heads, layer.head_dim*2)).chunk(2, -1)[0]
    assert torch.equal(saved_planes[0], layer.q_norm(raw_query))
    assert torch.equal(saved_planes[1], layer.k_norm(raw_key.unflatten(-1, (2, 256))))
    assert torch.equal(saved_planes[2], raw_value.view(16, 2, 256))
    assert torch.equal(saved_planes[-1], selector)
    qkv.normal_()
    selector.zero_()
    draft_graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(saved_planes,
        (buffers.query, buffers.key, buffers.value, layer._kvmem_index_raw[:16])))


def test_native_mtp_prefill_broadcasts_dense_positions_to_every_mrope_axis():
    import torch
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import prepare_prefill_inputs
    from vllm_kvmem.adapters.qsa_adapter import install_qsa_mtp_positions
    device = 'cuda'
    dense = torch.tensor([8192, 8193, 8194, 8195, 4096, 4097, 4098, 4099], device=device)
    batch = NS(num_reqs=2, positions=dense,
        input_ids=torch.arange(8, device=device, dtype=torch.int32), idx_mapping=torch.arange(2, device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([8196, 4100], device=device, dtype=torch.int32))
    buffers = NS(input_ids=torch.empty_like(batch.input_ids), positions=torch.empty_like(dense),
        query_start_loc=torch.empty_like(batch.query_start_loc), seq_lens=torch.empty_like(batch.seq_lens))
    logical = torch.full((3, 8), 250000, device=device, dtype=torch.int64)
    proposer = NS(_target_model_positions=lambda batch, profile: logical)
    runner = NS(speculator=proposer, _kvmem_pending_capture=(None, batch),
        _kvmem_window_batch=NS(positions=dense))
    install_qsa_mtp_positions(runner)
    axes = torch.empty((3, 9), device=device, dtype=torch.int64)
    accepted = torch.ones(2, device=device, dtype=torch.int32)
    rejected = torch.zeros_like(accepted)
    sampled = torch.tensor([10, 20], device=device, dtype=torch.int64)
    prepare_prefill_inputs(torch.zeros(2, device=device, dtype=torch.int64),
        torch.zeros((), device=device, dtype=torch.int64), buffers, batch, accepted, rejected,
        sampled, sampled.to(torch.int32), 2, target_model_positions=proposer._target_model_positions(batch, False),
        draft_mrope_positions=axes)
    assert torch.equal(buffers.positions, dense)
    assert torch.equal(axes[:, :8], dense.expand(3, -1))


@pytest.mark.parametrize('chunk_blocks', [1, 256])
def test_qsa_rebuild_moves_main_kv_selector_tail_and_request_ring(monkeypatch, chunk_blocks):
    import torch
    from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSAImpl
    from vllm.models.qwen4_exp.common.b12x_qsa_cache import qsa_compressed_cache_view
    from vllm_kvmem.serving.qsa_assembly import QSAWindowAssembler
    from vllm_kvmem.adapters.qsa_adapter import finish_qsa_refresh
    from vllm_kvmem.core.codec import KVFormat, materialize_k, neox_rope
    from vllm_kvmem.core.memory import Phase
    from b12x.attention.qsa.reference import stream_compress_reference
    torch.manual_seed(113)
    device = 'cuda'
    page, heads, dim, n = 32, 2, 256, 64
    stride = page * heads * dim * 2 + page // 4 * 128 * 2
    backing = torch.zeros(3 * stride, device=device, dtype=torch.uint8)
    cache = backing.as_strided((3, 2, page, heads*dim), (stride, page*heads*dim, heads*dim, 1))
    compressed = qsa_compressed_cache_view(cache, compress_ratio=4, index_head_dim=128)
    impl = Qwen4ExpQSAImpl.__new__(Qwen4ExpQSAImpl)
    impl.num_kv_heads, impl.head_size, impl.kv_cache_dtype = heads, dim, 'fp8_e4m3'
    angles = torch.arange(512, device=device)[:, None] * (10000 ** (-torch.arange(32, device=device).float()/32))
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1)
    norm = torch.randn(128, device=device, dtype=torch.bfloat16) * .1
    caps = NS(index_head_dim=128, index_rotary_dim=64, compress_ratio=4,
        raw_ring_capacity=8, compressed_page_size=page//4, position_axes=1,
        mrope_interleaved=False, rms_norm_eps=1e-6)
    layer = NS(device=torch.device(device), num_kv_heads=heads, head_dim=dim,
        layer_name='main', kv_cache=cache, impl=impl, _compressed_cache=compressed,
        rotary_emb=NS(cos_sin_cache=trig), compress_ratio=4, index_head_dim=128,
        raw_ring_capacity=8, position_axes=1, _qsa_decode_context=NS(caps=caps),
        indexer=NS(k_layernorm=NS(weight=norm)), _k_scale=torch.ones((), device=device),
        _raw_k_ring=torch.full((2, 8, 128), 73., device=device, dtype=torch.bfloat16),
        _raw_logical_positions=torch.full((2, 8), 73, device=device, dtype=torch.int64),
        _raw_rope_positions=torch.full((2, 8, 1), 73, device=device, dtype=torch.int64),
        _raw_interval_start_positions=torch.full((2,), 73, device=device, dtype=torch.int64),
        set_skip_topk=lambda value: None)
    table = torch.tensor([2, 1], device=device, dtype=torch.int32)
    runner = NS(_kvmem_qsa_layers={'main': layer}, _kvmem_qsa_pending_rings=[],
        req_states=NS(req_id_to_index={'r': 1}),
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=['main'])]),
        model_state=NS(qsa_committed_num_accepted_tokens_gpu=torch.tensor([1, 3], device=device)))
    monkeypatch.setattr('vllm_kvmem.serving.qsa_assembly.kernel_table', lambda *args: table)
    monkeypatch.setattr('vllm.forward_context.get_forward_context', lambda: NS(additional_kwargs={}))
    keys = torch.randn(n, heads, dim).to(torch.float8_e4m3fn)
    values = torch.randn(n, heads, dim).to(torch.float8_e4m3fn)
    selector = torch.randn(n, 128, dtype=torch.bfloat16)
    raw_archive = torch.zeros(2, n, heads, dim, dtype=torch.float8_e4m3fn)
    value_archive = torch.zeros(2, n, heads, dim, dtype=torch.uint8)
    raw_archive[0].copy_(keys)
    value_archive[0].copy_(values.view(torch.uint8))
    value_archive[1].reshape(n, -1)[:, :256].copy_(selector.view(torch.uint8))
    # Nonadjacent logical-history records become adjacent dense selected rows.
    records = {7: NS(handle=NS(slot=7), count=32), 13: NS(handle=NS(slot=13), count=32)}
    remaps = [NS(block_id=13, to_base=32, n_tokens=32), NS(block_id=7, to_base=0, n_tokens=32)]
    plan = NS(remaps=remaps, total_window_tokens=n)
    def read(handles):
        return [(raw_archive[:, :32] if h.slot == 7 else raw_archive[:, 32:],
                 value_archive[:, :32] if h.slot == 7 else value_archive[:, 32:]) for h in handles]
    authority = NS(shape=(2, 32, heads, dim), records=records, read_records=read,
        memory=NS(phase=Phase.ASSEMBLING, _plan=plan))
    assembler = QSAWindowAssembler(runner, 'r', ['main', '__qsa_index__.main'], 32, chunk_blocks=chunk_blocks)
    result = assembler.assemble(authority, plan)
    result.event.synchronize()
    finish_qsa_refresh(runner)
    torch.cuda.synchronize()
    key_cache, value_cache = impl._kv_cache_views(cache)
    expected_k = materialize_k(keys.cuda(), trig[:n, :32], trig[:n, 32:],
        KVFormat(torch.float8_e4m3fn, dim, heads, page), torch.ones((), device=device))
    assert torch.equal(key_cache[table.long()].reshape(n, heads, dim).view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(value_cache[table.long()].reshape(n, heads, dim).view(torch.uint8), values.cuda().view(torch.uint8))
    def rope(x, pos):
        p = int(pos[0].item())
        return neox_rope(x.view(1, 1, 128), trig[p:p+1, :32], trig[p:p+1, 32:]).flatten()
    _, expected_selector = stream_compress_reference(selector.cuda(),
        torch.arange(n, device=device), torch.arange(n, device=device)[:, None],
        torch.zeros(8, 128, device=device, dtype=torch.bfloat16),
        torch.full((8,), -1, device=device, dtype=torch.int64),
        torch.full((8, 1), -1, device=device, dtype=torch.int64), 4, norm, 1e-6, rope)
    actual_selector = compressed[table.long()].reshape(-1, 128)
    torch.testing.assert_close(actual_selector, expected_selector, atol=.016, rtol=.008)
    assert torch.equal(layer._raw_logical_positions[1], torch.arange(56, 64, device=device))
    assert torch.equal(layer._raw_k_ring[1], selector[-8:].cuda())
    assert int(layer._raw_interval_start_positions[1]) == 61
    assert torch.all(layer._raw_k_ring[0] == 73)  # unrelated request untouched
    assert not backing[:stride].any()  # unowned/null page untouched
    assert assembler.close()
    table[0] = 0
    invalid = QSAWindowAssembler(runner, 'r', ['main', '__qsa_index__.main'], 32)
    with pytest.raises(ValueError, match='null, duplicate or out-of-range'):
        invalid.assemble(authority, plan)
    assert not backing[:stride].any()
