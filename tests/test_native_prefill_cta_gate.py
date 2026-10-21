"""CPU contracts for the optional paged-FA2 CTA sample gate."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.native_attention import NativeAttentionStats
from vllm_kvmem import native_attention_build as build
from vllm_kvmem.options import RuntimeOptions
from test_options import config


def prefill_header():
    package = Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))
    return (package/'data/include/flashinfer/attention/prefill.cuh').read_text()


def test_default_patched_header_retains_original_call_and_no_gate():
    source = prefill_header()
    anchor = source.index('void BatchPrefillWithPagedKVCacheDevice(')
    marker = '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {'''
    replacement = '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }
        variant.template CaptureSoftmax<KTraits>(params, request_idx, qo_packed_idx_base, kv_idx_base,
            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {'''
    expected = source[:anchor] + source[anchor:].replace(marker, replacement)
    assert build.patch_prefill(source) == expected
    assert build.patch_prefill(source, cta_gate=False) == expected
    assert 'kvmem_cta_sampled' not in expected


def test_gate_is_once_after_pdl_before_loops_and_preserves_capture_body():
    source = prefill_header()
    patched = build.patch_prefill(source, cta_gate=True)
    anchor = patched.index('void BatchPrefillWithPagedKVCacheDevice(')
    assert patched[:anchor] == source[:anchor]
    paged = patched[anchor:]
    vote = paged.index('__syncthreads_or(kvmem_local_sampled)')
    assert paged.index('if (block_valid_mask && !block_valid_mask[bx])') < vote
    assert paged.index('griddepcontrol.wait;') < vote
    assert vote < paged.index('// Split-D over the VO dimension')
    assert vote < paged.index('for (uint32_t iter = 0;')
    assert paged.count('__syncthreads_or') == 1
    assert paged.count('if (kvmem_cta_sampled)') == 1
    assert 'params.stat_offset + params.q_indptr[request_idx] + kvmem_q' in paged
    assert '''if (kvmem_cta_sampled) {
        variant.template CaptureSoftmax<KTraits>(params, request_idx, qo_packed_idx_base, kv_idx_base,
            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);
        }''' in paged


def test_gate_fails_closed_when_unique_pdl_loop_boundary_changes():
    source = prefill_header().replace('// Split-D over the VO dimension', '// changed')
    build.patch_prefill(source)  # Default remains compatible with this harmless comment change.
    with pytest.raises(build.UnsupportedNativeStats):
        build.patch_prefill(source, cta_gate=True)


@pytest.mark.parametrize('group', [1,2,3,4,6,8,16,32,33,64])
@pytest.mark.parametrize('tile', [16,64,128])
def test_q_interval_covers_partial_gqa_boundary_heads_and_exactly_real_rows(group,tile):
    qlen=173
    for index in range((qlen*group+tile-1)//tile):
        begin=index*tile//group
        end=min(qlen,((index+1)*tile+group-1)//group)
        expected={packed//group for packed in range(index*tile,min((index+1)*tile,qlen*group))}
        distributed={q for thread in range(128) for q in range(begin+thread,end,128)}
        assert distributed == expected
        for rows in (set(),set(range(qlen)),{begin},{end-1},{31,63,95}):
            assert bool(distributed & rows) == bool(expected & rows)


def test_option_default_validation_and_public_config_keep_both_flags_independent():
    assert not RuntimeOptions.from_vllm(config({})).action_native_prefill_cta_gate
    for value in (0,1,'true',None):
        with pytest.raises(ValueError,match='CTA gate must be boolean'):
            RuntimeOptions.from_vllm(config({'kvmem':{'action_native_prefill_cta_gate':value}}))
    with pytest.raises(ValueError,match='CTA gate requires native'):
        RuntimeOptions.from_vllm(config({'kvmem':{'action_native_prefill_cta_gate':True}}))
    for aligned in (False,True):
        payload={'action_native_prefill_cta_gate':True,'action_native_prefill_aligned_pieces':aligned,
                 'action_native_scores':'reuse','action_attention':True,'session_capacity':1,
                 'session_checkpoint_bytes':4096,'async_archive_retirement':True}
        options=RuntimeOptions.from_vllm(config({'kvmem':payload}))
        assert options.action_native_prefill_cta_gate
        assert options.action_native_prefill_aligned_pieces is aligned
        assert payload['action_native_prefill_cta_gate'] is True


@pytest.mark.parametrize('aligned', [False,True])
@pytest.mark.parametrize('gate', [False,True])
def test_dispatch_abi_and_module_cache_keys_are_independent(monkeypatch,aligned,gate):
    import vllm_kvmem.native_attention as module
    builds=[]; calls=[]
    def factory(*args,**kwargs):
        builds.append((args,kwargs));return NS(paged_run=lambda *args:calls.append(args))
    monkeypatch.setattr(module,'prefill_module',factory)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    original=object()
    wrapper=NS(_backend='fa2',_custom_mask_buf=None,_pos_encoding_mode='NONE',_causal=True,
        _window_left=-1,_logits_soft_cap=0,_paged_kv_indptr_buf=NS(dtype=torch.int32),_cached_module=original)
    impl=NS(forward=lambda *a,**kw:wrapper._cached_module.paged_run(*range(24)))
    bank=NativeAttentionStats.__new__(NativeAttentionStats)
    bank.aligned_prefill_enabled=aligned;bank.aligned_prefill_counts=Counter()
    bank.cta_gate_enabled=gate;bank.cta_gate_counts=Counter()
    bank.prefill_modules={};bank.directory='unused';bank.reasons={}
    bank.rows,bank.requests,bank.pieces,bank.lses=object(),object(),object(),{'l':object()}
    bank.aligned_pieces=object();bank.context=128;bank.width=32
    bank._wrap(impl,'l')
    metadata=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=torch.bfloat16,num_decode_tokens=7)
    args=(NS(_q_scale_float=1.),NS(shape=(5,24,256)),None,None,None,metadata,NS(dtype=torch.bfloat16))
    impl.forward(*args);impl.forward(*args)
    expected={}
    if aligned:expected['aligned_pieces']=True
    if gate:expected['cta_gate']=True
    assert len(builds)==1 and builds[0][1]==expected
    assert wrapper._cached_module is original
    assert len(calls[0])==24+int(aligned)
    assert calls[0][:16]==tuple(range(16)) and calls[0][-4:]==(128,32,7,23)
    assert bank.cta_gate_counts==({'prefill_dispatch_calls':2} if gate else {})
    # Fixed service configurations may coexist without confusing module ABIs.
    bank.cta_gate_enabled=not gate;impl.forward(*args)
    assert len(builds)==2 and len(bank.prefill_modules)==2


def test_build_isolates_overlay_and_names_without_adding_gate_abi(tmp_path,monkeypatch):
    recorded=[];overlays=[]
    def overlay(directory,**kwargs):
        overlays.append(kwargs)
        path=tmp_path/('gated' if kwargs else 'default');path.mkdir(exist_ok=True)
        return path
    generated=tmp_path/'generator';generated.mkdir()
    source=generated/'kernel.cu';source.write_text('#include <flashinfer/attention/prefill.cuh>\n')
    (generated/'batch_prefill_config.inc').write_text('config')
    def generator(*args):
        recorded.append(args)
        spec=NS(sources=[source])
        spec.build_and_load=lambda:(args[1],spec.sources[0].read_text())
        return spec
    monkeypatch.setattr(build,'source_overlay',overlay)
    monkeypatch.setitem(sys.modules,'flashinfer.jit.attention',NS(gen_customize_batch_prefill_module=generator))
    build._prefill_module.cache_clear()
    names=[]
    try:
        for aligned in (False,True):
            for gate in (False,True):
                result=build.prefill_module(None,'q','kv','out','idx',256,
                                           aligned_pieces=aligned,cta_gate=gate)
                names.append(result[0])
                assert ('/gated/' in result[1]) is gate
                assert recorded[-1][8]==(['stat_rows','stat_requests','stat_pieces','stat_lse']+
                                        (['stat_aligned'] if aligned else []))
        assert len(set(names))==4
        assert overlays==[{}, {'prefill_cta_gate':True}, {}, {'prefill_cta_gate':True}]
    finally:
        build._prefill_module.cache_clear()
