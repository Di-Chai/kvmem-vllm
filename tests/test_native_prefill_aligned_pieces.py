"""Host qualification and ABI/control-buffer contracts for aligned FA2 pieces."""
from collections import Counter
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.native_attention import NativeAttentionStats, _layout_array, aligned_piece_layout


@pytest.mark.parametrize('tail', range(1, 17))
def test_all_causal_partial_tails_keep_exact_dense_piece_numbering(tail):
    length = 64 + tail
    starts = [0,16,48,64]
    counts = [16,32,16,tail]
    layout, _ = _layout_array(starts, counts, length)
    assert aligned_piece_layout(layout, length)
    for start,count,first,_ in layout:
        for pos in range(start,start+count):
            assert first+pos//16-start//16 == pos//16


def test_alignment_uses_actual_dense_prefix_and_rejects_internal_boundaries():
    layout, _ = _layout_array([0,32,37,64],[32,5,27,3],67)
    assert aligned_piece_layout(layout,32)
    assert aligned_piece_layout(layout,35)  # last visible segment is partial
    assert not aligned_piece_layout(layout,38)
    assert not aligned_piece_layout(layout,67)
    # Original history/physical page order does not enter this predicate.
    layout, _ = _layout_array([0,32,64],[32,32,3],67)
    assert aligned_piece_layout(layout,67)
    corrupt = layout.copy(); corrupt[1,2] += 1
    assert not aligned_piece_layout(corrupt,67)
    assert not aligned_piece_layout(layout,68)
    assert not aligned_piece_layout(np.empty((0,4),np.int32),1)


class Event:
    def query(self): return True
    def record(self): pass
    def synchronize(self): raise AssertionError('No old DMA was pending')


class Kernel:
    def __getitem__(self, grid): return lambda *a, **kw: None


def host_bank(monkeypatch, *, enabled=True):
    import vllm_kvmem.native_attention as module
    monkeypatch.setattr(module, '_map_pieces', Kernel())
    monkeypatch.setattr(module, '_clear_lse', Kernel())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    bank=NativeAttentionStats.__new__(NativeAttentionStats)
    bank.disabled=None; bank.max_rows=8; bank.context=128; bank.max_segments=16
    bank.width=32; bank.heads=4; bank.armed_rows=0; bank.upload_event=None
    bank.options=NS(memory=NS(block_tokens=32))
    bank.aligned_prefill_enabled=enabled; bank.aligned_prefill_counts=Counter()
    shapes={'rows':(16,), 'requests':(8,), 'ranges':(3,16,4), 'meta':(8,3),
            'selected':(8,), 'valid_flags':(1,), 'control':(1,)}
    if enabled: shapes['aligned_pieces']=(3,)
    for name,shape in shapes.items(): setattr(bank,name,torch.empty(shape,dtype=torch.int32))
    bank.host={k:torch.empty(shape,dtype=torch.int32) for k,shape in shapes.items()}
    bank.pieces=torch.empty(3,128,dtype=torch.int32)
    bank.lses={'l':torch.empty(8,4,32)}
    bank.disarm()
    return bank


def plan(index, begin, *, aligned=True, sampled=True):
    starts=[0,32,64] if aligned else [0,32,37,64]
    counts=[b-a for a,b in zip(starts,starts[1:]+[67])]
    return dict(index=index,begin=begin,rows=[0,2] if sampled else [],starts=starts,
                counts=counts,local_begin=64,count=3)


def test_arm_zeros_unused_swapped_unsampled_and_capacity_failure_owner_flags(monkeypatch):
    bank=host_bank(monkeypatch)
    gpu_pointer,host_pointer=bank.aligned_pieces.data_ptr(),bank.host['aligned_pieces'].data_ptr()
    bank.arm({'prefill-a':plan(2,4),'prefill-b':plan(0,7,aligned=False)})
    assert bank.aligned_pieces.tolist()==[0,0,1]
    assert bank.requests[:4].tolist()==[2,2,0,0]
    bank.arm({'new-owner-same-slot':plan(2,0,aligned=False),'other-slot':plan(1,3)})
    assert bank.aligned_pieces.tolist()==[0,1,0]
    bank.arm({'unsampled':plan(1,0,sampled=False)})
    assert bank.aligned_pieces.tolist()==[0,0,0]
    bank.arm({'fresh':plan(2,0)})
    bank.max_rows=1
    bank.arm({'overflow':plan(1,0)})
    assert bank.batch_reason=='sample_row_capacity'
    assert bank.aligned_pieces.tolist()==[0,0,0]
    bank.disarm()
    assert bank.aligned_pieces.tolist()==[0,0,0]
    assert (gpu_pointer,host_pointer)==(bank.aligned_pieces.data_ptr(),bank.host['aligned_pieces'].data_ptr())
    assert bank.aligned_prefill_counts==dict(aligned_candidate_owners=3,aligned_candidate_rows=6,
                                            generic_candidate_owners=2,generic_candidate_rows=4)


def test_disabled_arm_needs_no_additional_buffer(monkeypatch):
    bank=host_bank(monkeypatch,enabled=False)
    bank.arm({'r':plan(0,0)})
    assert not hasattr(bank,'aligned_pieces')
    assert not bank.aligned_prefill_counts


@pytest.mark.parametrize('enabled', [False,True])
def test_prefill_wrapper_keeps_old_abi_off_and_stable_runtime_pointer_on(monkeypatch,enabled):
    import vllm_kvmem.native_attention as module
    builds=[]; calls=[]
    def build(*args,**kwargs):
        builds.append((args,kwargs))
        return NS(paged_run=lambda *args:calls.append(args))
    monkeypatch.setattr(module,'prefill_module',build)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    cached=object()
    wrapper=NS(_backend='fa2',_custom_mask_buf=None,_pos_encoding_mode='NONE',_causal=True,
               _window_left=-1,_logits_soft_cap=0,_paged_kv_indptr_buf=NS(dtype=torch.int32),_cached_module=cached)
    impl=NS(forward=lambda *a,**kw:wrapper._cached_module.paged_run(*range(24)))
    bank=NativeAttentionStats.__new__(NativeAttentionStats)
    bank.aligned_prefill_enabled=enabled; bank.aligned_prefill_counts=Counter()
    bank.prefill_modules={}; bank.directory='unused'; bank.reasons={}
    bank.rows,bank.requests,bank.pieces,bank.lses=object(),object(),object(),{'l':object()}
    bank.context=128;bank.width=32
    if enabled: bank.aligned_pieces=torch.tensor([1,0,0],dtype=torch.int32)
    bank._wrap(impl,'l')
    metadata=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=torch.bfloat16,num_decode_tokens=2)
    args=(NS(_q_scale_float=1.),NS(shape=(5,4,256)),None,None,None,metadata,NS(dtype=torch.bfloat16))
    impl.forward(*args)
    if enabled: bank.aligned_pieces.copy_(torch.tensor([0,1,0]))
    impl.forward(*args)
    assert len(builds)==1 and wrapper._cached_module is cached
    assert builds[0][1]==({'aligned_pieces':True} if enabled else {})
    assert len(calls[0])==(25 if enabled else 24)
    assert calls[0][:16]==tuple(range(16))
    assert calls[0][16:20]==(bank.rows,bank.requests,bank.pieces,bank.lses['l'])
    assert calls[0][-4:]==(128,32,2,23)
    if enabled:
        assert calls[0][20] is calls[1][20] is bank.aligned_pieces
        assert bank.aligned_prefill_counts['prefill_dispatch_calls']==2
    else: assert not bank.aligned_prefill_counts
