"""Host-only proofs and dispatch isolation for static dense FA2 statistics."""
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import native_attention_build as build
from vllm_kvmem.native_attention import NativeAttentionStats, _layout_array, static_aligned_frame
from vllm_kvmem.options import RuntimeOptions
from test_options import config
from test_native_prefill_aligned_pieces import host_bank


def plan(index=0, begin=0, count=128, *, partial=False, sampled=True):
    length = 256 + count
    starts = [0,32,37] if partial else [0,32]
    starts += list(range(64,length,32))
    counts = [b-a for a,b in zip(starts,starts[1:]+[length])]
    return dict(index=index,begin=begin,count=count,local_begin=256,rows=[0,count-1] if sampled else [],
                starts=starts,counts=counts)


def test_frame_covers_all_owners_including_unsampled_and_final_partial_tail():
    plans={'a':plan(count=129),'b':plan(1,129,count=130,sampled=False)}
    frozen,reason=static_aligned_frame(plans)
    assert reason is None and frozen==((0,0,129),(1,129,130))
    plans['b']=plan(1,129,partial=True,sampled=False)
    assert static_aligned_frame(plans)==(None,'unaligned_logical_segments')
    assert static_aligned_frame({})==(None,'no_sampled_frame')
    assert static_aligned_frame({'a':plan(sampled=False)})==(None,'no_sampled_frame')


@pytest.mark.parametrize('field,value', [('index',1),('begin',1),('count',0),('local_begin',-4)])
def test_invalid_or_noncontiguous_frames_never_get_static_route(field,value):
    p=plan();p[field]=value
    assert static_aligned_frame({'a':p})[0] is None


def test_cached_layout_qualifies_actual_prefix_only_and_rejects_corrupt_piece_ids():
    p=plan(count=129)
    p['_layout'],_= _layout_array(p['starts'],p['counts'],385)
    assert static_aligned_frame({'a':p})[1] is None
    p['_layout'][1,2]+=1
    assert static_aligned_frame({'a':p})[1]=='unaligned_logical_segments'


def test_arm_failure_empty_and_disarm_cannot_reuse_previous_qualification(monkeypatch):
    bank=host_bank(monkeypatch,enabled=False)
    bank.static_prefill_enabled=True;bank.static_prefill_counts=Counter()
    # The small CPU fixture needs a correspondingly small domain/row span.
    p=dict(index=0,begin=0,count=3,local_begin=64,starts=[0,32,64],counts=[32,32,3],rows=[0,2])
    bank.arm({'a':p});assert bank.static_prefill_frame==((0,0,3),)
    bank.arm({'b':dict(p,rows=[])});assert bank.static_prefill_frame is None
    bank.arm({'a':p});bank.max_rows=1
    bank.arm({'a':p});assert bank.static_prefill_frame is None
    bank.max_rows=8;bank.arm({'a':p});bank.disarm()
    assert bank.static_prefill_frame is None and bank.static_prefill_reason=='unarmed_frame'
    assert not hasattr(bank,'aligned_pieces')


def dispatch_fixture(monkeypatch):
    import vllm_kvmem.native_attention as module
    builds=[];calls=[];capturing=[False];fail_static=[False]
    def factory(*args,**kw):
        builds.append(kw)
        if kw.get('static_aligned') and fail_static[0]:raise RuntimeError('injected build failure')
        return NS(paged_run=lambda *a:calls.append((kw,a)))
    monkeypatch.setattr(module,'prefill_module',factory)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:capturing[0])
    info=[0]*15
    wrapper=NS(_backend='fa2',_custom_mask_buf=None,_pos_encoding_mode='NONE',_causal=True,
        _window_left=-1,_logits_soft_cap=0,_paged_kv_indptr_buf=NS(dtype=torch.int32),
        _cached_module=object(),_kv_layout='NHD',is_cuda_graph_enabled=False)
    original=wrapper._cached_module
    run_args=list(range(24));run_args[2]=info
    run_args[3]=NS(shape=(128,24,256));run_args[4]=NS(shape=(32,16,4,256));run_args[14]=-1
    impl=NS(forward=lambda *a,**kw:wrapper._cached_module.paged_run(*run_args))
    bank=NativeAttentionStats.__new__(NativeAttentionStats)
    bank.static_prefill_enabled=True;bank.static_prefill_counts=Counter()
    bank.static_prefill_frame,bank.static_prefill_reason=static_aligned_frame({'a':plan()})
    bank.prefill_modules={};bank.directory='unused';bank.reasons={}
    bank.rows,bank.requests,bank.pieces,bank.lses=object(),object(),object(),{'l':object()}
    bank.context=512;bank.width=64
    bank._wrap(impl,'l')
    metadata=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=torch.bfloat16,
        num_decode_tokens=0,num_prefills=1,num_prefill_tokens=128)
    def execute():
        impl.forward(NS(_q_scale_float=1.),NS(shape=(128,24,256)),None,None,None,metadata,
                     NS(dtype=torch.bfloat16))
        assert wrapper._cached_module is original
    return NS(bank=bank,wrapper=wrapper,args=run_args,metadata=metadata,builds=builds,calls=calls,
              capturing=capturing,fail_static=fail_static,execute=execute)


def test_static_and_generic_modules_share_old_abi_and_switch_without_rearm(monkeypatch):
    env=dispatch_fixture(monkeypatch);env.execute()
    assert env.builds==[{}, {'static_aligned':True}]
    assert env.calls[-1][0]=={'static_aligned':True}
    assert len(env.calls[-1][1])==24
    assert env.calls[-1][1][:16]==tuple(env.args[:16])
    env.capturing[0]=True;env.execute()
    assert env.calls[-1][0]=={}
    env.capturing[0]=False;env.execute()
    assert env.calls[-1][0]=={'static_aligned':True} and len(env.builds)==2
    assert env.bank.static_prefill_counts=={'static_dispatch_calls':2,'generic_dispatch:cuda_graph_capture':1}


@pytest.mark.parametrize('kind,reason',[
    ('graph_wrapper','graph_or_unknown_wrapper'),('unknown_wrapper','graph_or_unknown_wrapper'),
    ('graph_plan','graph_or_unknown_plan'),('unknown_plan','graph_or_unknown_plan'),
    ('page24','unaligned_kv_page'),('layout','unknown_page_geometry'),
    ('mixed','mixed_decode_prefix'),('owner','owner_or_query_domain'),
    ('rows','owner_or_query_domain'),('unknown_metadata','unknown_owner_metadata'),
    ('small','small_prefill'),('unaligned','unaligned_logical_segments'),
    ('window','sliding_window')])
def test_unknown_ineligible_geometry_uses_existing_native_not_reference(monkeypatch,kind,reason):
    e=dispatch_fixture(monkeypatch)
    if kind=='graph_wrapper':e.wrapper.is_cuda_graph_enabled=True
    elif kind=='unknown_wrapper':del e.wrapper.is_cuda_graph_enabled
    elif kind=='graph_plan':e.args[2][13]=1
    elif kind=='unknown_plan':e.args[2]=[]
    elif kind=='page24':e.args[4].shape=(32,24,4,256)
    elif kind=='layout':e.wrapper._kv_layout='unknown'
    elif kind=='mixed':e.metadata.num_decode_tokens=3
    elif kind=='owner':e.metadata.num_prefills=2
    elif kind=='rows':e.metadata.num_prefill_tokens=127
    elif kind=='unknown_metadata':e.metadata.num_decode_tokens=torch.tensor(0)
    elif kind=='small':
        e.args[3].shape=(64,24,256);e.metadata.num_prefill_tokens=64
        e.bank.static_prefill_frame=((0,0,64),)
    elif kind=='unaligned':
        e.bank.static_prefill_frame,e.bank.static_prefill_reason=static_aligned_frame({'a':plan(partial=True)})
    elif kind=='window':e.args[14]=1
    e.execute()
    assert e.builds==[{}] and e.calls[-1][0]=={}
    assert e.bank.reasons['l','prefill'] is None
    assert e.bank.static_prefill_counts=={'generic_dispatch:'+reason:1}


def test_static_build_failure_keeps_native_generic_and_counts_reason(monkeypatch):
    e=dispatch_fixture(monkeypatch);e.fail_static[0]=True;e.execute()
    assert e.calls[-1][0]=={} and e.bank.reasons['l','prefill'] is None
    assert e.bank.static_prefill_counts=={'generic_dispatch:build:injected build failure':1}


def test_static_source_drift_keeps_generic_native_available(monkeypatch):
    e=dispatch_fixture(monkeypatch)
    e.bank.static_prefill_unavailable='static_source_mismatch:scheduler'
    e.execute()
    assert e.builds==[{}] and e.calls[-1][0]=={}
    assert e.bank.reasons['l','prefill'] is None
    assert e.bank.static_prefill_counts=={'generic_dispatch:static_source_mismatch:scheduler':1}


def test_static_source_proof_matches_installed_scheduler_and_rejects_drift(tmp_path,monkeypatch):
    package=Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))
    build._check_static_alignment_sources(package/'data/include')
    target=tmp_path/'proof.cuh';target.write_bytes(b'fixed page-based split rules')
    monkeypatch.setattr(build,'_STATIC_ALIGNMENT_SOURCES',
        {'proof.cuh':hashlib.sha256(target.read_bytes()).hexdigest()})
    build._check_static_alignment_sources(tmp_path)
    target.write_bytes(b'changed split rules')
    with pytest.raises(build.UnsupportedNativeStats,match='static_source_mismatch:proof.cuh'):
        build._check_static_alignment_sources(tmp_path)
    target.unlink()
    with pytest.raises(build.UnsupportedNativeStats,match='static_source_mismatch:proof.cuh'):
        build._check_static_alignment_sources(tmp_path)


def test_page_multiples_prove_nonterminal_split_alignment_without_reading_plan_workspace(monkeypatch):
    e=dispatch_fixture(monkeypatch)
    for page in (16,32,64):
        for split in (0,1):
            e.args[4].shape=(32,page,4,256);e.args[2][14]=split;e.execute()
            assert e.calls[-1][0]=={'static_aligned':True}
    assert e.bank.static_prefill_counts['static_dispatch_calls']==6


def test_real_tvm_host_plan_sequence_qualifies_but_tensor_does_not(monkeypatch):
    import tvm_ffi
    e=dispatch_fixture(monkeypatch)
    e.args[2]=tvm_ffi.convert([0]*15)
    e.execute()
    assert e.calls[-1][0]=={'static_aligned':True}
    e.args[2]=torch.zeros(15,dtype=torch.int64)
    e.execute()
    assert e.calls[-1][0]=={}
    assert e.bank.static_prefill_counts['generic_dispatch:graph_or_unknown_plan']==1


def test_static_header_reuses_exact_mask_sum_log_body_without_generic_calls():
    root=Path(__file__).resolve().parents[1]/'src/vllm_kvmem'
    header=(root/'native_stats.cuh').read_text()
    original=header.split('if (aligned && !merge && (base & 15) == 0) {',1)[1].split('return;',1)[0]
    static=header.split('__device__ void store16_weight_static_aligned(',1)[1]
    normalize=lambda text:' '.join(text.split())
    assert normalize(original) in normalize(static)
    assert 'pieces[' not in static and 'add(' not in static
    capture=(root/'native_prefill_stats.cuh').read_text().split('struct KVMemStaticAlignedAttention',1)[1]
    assert 'store16_weight_static_aligned(' in capture
    assert 'stats.store16_weight(' not in capture and 'kv_chunk_size_ptr' not in capture


def test_isolated_module_keeps_old_abi_and_rejects_combination(tmp_path,monkeypatch):
    overlays=[];generated=[]
    def overlay(_directory,**kw):
        overlays.append(kw);p=tmp_path/('static' if kw else 'old');p.mkdir(exist_ok=True);return p
    src=tmp_path/'source';src.mkdir();(src/'kernel.cu').write_text('#include <flashinfer/attention/prefill.cuh>')
    (src/'batch_prefill_config.inc').write_text('config')
    def generator(*args):
        generated.append(args);result=NS(sources=[src/'kernel.cu']);result.build_and_load=lambda:args;return result
    monkeypatch.setattr(build,'source_overlay',overlay)
    monkeypatch.setitem(sys.modules,'flashinfer.jit.attention',NS(gen_customize_batch_prefill_module=generator))
    build._prefill_module.cache_clear()
    try:
        old=build.prefill_module(None,'q','kv','o','i',256)
        new=build.prefill_module(None,'q','kv','o','i',256,static_aligned=True)
        assert old[8:12]==new[8:12] and new[-2]=='KVMemStaticAlignedAttention'
        assert old[1]!=new[1] and overlays==[{}, {'prefill_static_aligned':True}]
        for other in ('aligned_pieces','cta_gate'):
            with pytest.raises(ValueError,match='cannot combine'):
                build.prefill_module(None,'q','kv','o','i',256,static_aligned=True,**{other:True})
    finally:build._prefill_module.cache_clear()


def test_option_is_default_off_strict_and_incompatible_flags_are_rejected():
    assert not RuntimeOptions.from_vllm(config({})).action_native_prefill_static_aligned
    for value in (0,1,'true',None):
        with pytest.raises(ValueError,match='static aligned must be boolean'):
            RuntimeOptions.from_vllm(config({'kvmem':{'action_native_prefill_static_aligned':value}}))
    with pytest.raises(ValueError,match='requires native'):
        RuntimeOptions.from_vllm(config({'kvmem':{'action_native_prefill_static_aligned':True}}))
    payload=dict(action_native_prefill_static_aligned=True,action_native_scores='reuse',
        action_attention=True,session_capacity=1,session_checkpoint_bytes=4096,async_archive_retirement=True)
    assert RuntimeOptions.from_vllm(config({'kvmem':payload})).action_native_prefill_static_aligned
    for name in ('action_native_prefill_aligned_pieces','action_native_prefill_cta_gate'):
        with pytest.raises(ValueError,match='cannot combine'):
            RuntimeOptions.from_vllm(config({'kvmem':dict(payload,**{name:True})}))
