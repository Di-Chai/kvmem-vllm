"""Isolated static FA2 experiment: preload sample IDs into CTA shared memory.

No serving option or checked-in CUDA header changes. The old overlay remains
byte-identical. This component reuses the existing no-graph metadata alias guard.
"""
import hashlib
import inspect
from pathlib import Path


def once(source,before,after):
    if source.count(before)!=1:
        raise ValueError('Unexpected frozen FlashInfer source structure')
    return source.replace(before,after)


def patch_kernel(source):
    anchor='void BatchPrefillWithPagedKVCacheDevice('
    if source.count(anchor)!=1 or 'kvmem_sample_cache' in source:
        raise ValueError('Unknown or already patched prefill source')
    index=source.index(anchor)
    before,body=source[:index],source[index:]
    marker='''#endif

    // Split-D over the VO dimension'''
    cache='''#endif

    // Metadata is immutable through this single-stream frame. All valid CTA
    // threads pass the original PDL wait before filling this bounded cache.
    static_assert(CTA_TILE_Q == NUM_WARPS_Q * NUM_MMA_Q * 16);
    __shared__ int32_t kvmem_sample_cache[CTA_TILE_Q];
    const uint32_t kvmem_tile_begin = qo_tile_idx * CTA_TILE_Q;
    for (uint32_t i = block.thread_rank(); i < CTA_TILE_Q; i += KTraits::NUM_THREADS) {
      const uint32_t row = (kvmem_tile_begin + i) / group_size;
      kvmem_sample_cache[i] = row < qo_len ?
          params.stat_rows[params.stat_offset + params.q_indptr[request_idx] + row] : -1;
    }
    block.sync();

    // Split-D over the VO dimension'''
    body=once(body,marker,cache)
    call='''            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);'''
    body=once(body,call,'''            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx,
            kvmem_sample_cache, kvmem_tile_begin);''')
    return before+body


def patch_variant(source):
    marker='struct KVMemStaticAlignedAttention :'
    if source.count(marker)!=1:
        raise ValueError('Unknown static attention variant')
    index=source.index(marker)
    before,body=source[:index],source[index:]
    body=once(body,'uint32_t kv_head) const {',
              'uint32_t kv_head, const int32_t* kvmem_samples, uint32_t kvmem_tile_begin) const {')
    body=once(body,'int sample = stats.rows[p.q_indptr[batch] + q];',
              'int sample = kvmem_samples[packed + m * 16 + tid.x / 4 + 8 * r - kvmem_tile_begin];')
    body=body.replace('KVMemStaticAlignedAttention','KVMemStaticCachedAttention')
    return before+body


def build_modules(output):
    import torch
    from flashinfer.jit import env
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    from flashinfer.jit.core import jit_spec_registry
    from vllm_kvmem.native_attention_build import prefill_module,source_overlay,_write
    from vllm_kvmem.native_prefill_readonly import CheckedReadonlyPrefillModule
    geometry=(torch.bfloat16,torch.float8_e4m3fn,torch.bfloat16,torch.int32,256)
    old=source_overlay(None,prefill_static_aligned=True)
    if old.name!='d7b326c5139bf4d9':
        raise ValueError('Reference static source fingerprint changed')
    key=hashlib.sha256((old.name+inspect.getsource(patch_kernel)+inspect.getsource(patch_variant)).encode()).hexdigest()[:16]
    overlay=env.FLASHINFER_GEN_SRC_DIR/'kvmem_stats'/key
    overlay.mkdir(parents=True,exist_ok=True)
    for name in ('native_stats.cuh','tvm_ffi_utils.h'):
        _write(overlay/name,(old/name).read_bytes())
    _write(overlay/'prefill.cuh',patch_kernel((old/'prefill.cuh').read_text()))
    _write(overlay/'native_prefill_stats.cuh',patch_variant((old/'native_prefill_stats.cuh').read_text()))
    name='kvmem_sample_cache_'+key
    spec=gen_customize_batch_prefill_module('fa2',name,*geometry[:4],256,256,
        ['stat_rows','stat_requests','stat_pieces','stat_lse'],['int32_t','int32_t','int32_t','float'],
        ['stat_context','stat_width','stat_offset','sm_scale'],['int32_t','int32_t','int32_t','double'],
        'KVMemStaticCachedAttention','#include "'+str(overlay/'native_prefill_stats.cuh')+'"')
    generated=overlay/name
    generated.mkdir(exist_ok=True)
    sources=[]
    for path in spec.sources:
        destination=generated/path.name
        _write(destination,path.read_text().replace('#include <flashinfer/attention/prefill.cuh>',
            '#include "'+str(overlay/'prefill.cuh')+'"'))
        sources.append(destination)
    _write(generated/'batch_prefill_config.inc',(spec.sources[0].parent/'batch_prefill_config.inc').read_bytes())
    spec.sources=sources
    new=CheckedReadonlyPrefillModule(spec.build_and_load(),library_path=spec.get_library_path(),overlay=overlay)
    reference=prefill_module(output.parent/'build',*geometry,static_aligned=True)
    specs=jit_spec_registry.get_all_specs()
    old_specs=[s for n,s in specs.items() if old.name in n and n.endswith('_static_aligned16')]
    if len(old_specs)!=1:
        raise ValueError('Reference compiled module is ambiguous')
    identities={}
    for side,location,library in [('reference',old,old_specs[0].get_library_path()),
                                  ('candidate',overlay,Path(new.library_path))]:
        identities[side]=dict(overlay=location.name,library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            native_header_sha256=hashlib.sha256((location/'native_prefill_stats.cuh').read_bytes()).hexdigest())
    return geometry,reference,new,identities
