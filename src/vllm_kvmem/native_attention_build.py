"""Isolated, source-checked FlashInfer builds; installed packages are never edited."""
from functools import cache
import hashlib
import inspect
import json
from pathlib import Path
import shutil


class UnsupportedNativeStats(RuntimeError):
    pass


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise UnsupportedNativeStats('FlashInfer source structure changed')
    return source.replace(before, after)


def _write(path, value):
    data = value.encode() if isinstance(value, str) else value
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def patch_xqa(source, name):
    """Extend the native ABI all the way to its existing post-QK accumulator."""
    if name == 'mha.cu':
        source = _replace(source, '#include "mha.h"', '#include "mha.h"\n#include "native_stats.cuh"')
        source = _replace(source, 'void* __restrict__ scratch = nullptr) {',
                          'void* __restrict__ scratch = nullptr, KVMemStats stats = {}) {', 2)
        source = _replace(source, 'semaphores, scratch);\n}\n#else',
                          'semaphores, scratch, stats);\n}\n#else')
        capture = '''
      // Reuse native, scaled QK. Every quad handles four columns of one parity.
      if (stats.rows != nullptr) {
      #pragma unroll
      for (uint32_t m = 0; m < acc.rows; ++m) {
        #pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; ++i) {
          uint32_t mixed = instM * m + laneId() / 4 + 8 * i;
          #if SPEC_DEC
          mixed += idxHeadTokenInGrp;
          uint32_t row = mixed / headGrpSize;
          if (row >= actualQSeqLen) continue;
          uint32_t globalRow = reqSeqOffset + row;
          uint32_t visible = cacheSeqLen - actualQSeqLen + row + 1;
          #else
          if (mixed >= headGrpSize) continue;
          uint32_t globalRow = idxReq;
          uint32_t visible = cacheSeqLen;
          #endif
          int sample = stats.rows[globalRow];
          if (sample < 0) continue;
          #pragma unroll
          for (uint32_t n = 0; n < acc.cols; n += 2) {
            stats.store16_weight(acc(m, n)(i, 0), acc(m, n)(i, 1),
                acc(m, n+1)(i, 0), acc(m, n+1)(i, 1), regRowMax[m * InstAcc::rows + i],
                warpTileTokenBeg + instN * n, visible, sample,
                idxHeadGrp * headGrpSize + mixed % headGrpSize, false);
          }
        }
      }
      }
'''
        marker = '      QuadRegRowMax const regRowMax = warpTileOnlineSoftmax(warp, initRowMaxQuad, acc);'
        source = _replace(source, marker, marker + capture)
        start = source.index('void launchMHAFlashInfer(')
        prefix, tail = source[:start], source[start:]
        prefix = _replace(prefix, '                     semaphores, scratch);',
                          '                     semaphores, scratch, KVMemStats{});')
        tail = _replace(tail, 'cudaStream_t stream) {', 'KVMemStats stats, cudaStream_t stream) {')
        tail = _replace(tail, 'semaphores, scratch);', 'semaphores, scratch, stats);')
        return prefix + tail
    if name == 'mha.h':
        source = '#include "native_stats.cuh"\n' + source
        start = source.index('void launchMHAFlashInfer(')
        end = source.index(');', start) + 2
        return source[:start] + _replace(source[start:end], 'cudaStream_t stream);',
                                          'KVMemStats stats, cudaStream_t stream);') + source[end:]
    if name in ('xqa_wrapper.cu', 'flashinfer_xqa_binding.cu'):
        start = source.index('void xqa_wrapper(')
        prefix, tail = source[:start], source[start:]
        extra = ('bool enable_pdl, TensorView statRows, TensorView statRequests, '
                 'TensorView statPieces, TensorView statLse, int64_t statContext, int64_t statWidth')
        tail = _replace(tail, 'bool enable_pdl)', extra + ')')
        if name == 'xqa_wrapper.cu':
            # Only the standard warp-specialized SM12x path is enabled by the adapter.
            end = tail.index('  launchMHAFlashInfer(')
            head, launch = tail[:end], tail[end:]
            launch = _replace(launch, '                      stream);', '''
                      KVMemStats{static_cast<int32_t const*>(statRows.data_ptr()),
                                  static_cast<int32_t const*>(statRequests.data_ptr()),
                                  static_cast<int32_t const*>(statPieces.data_ptr()),
                                  static_cast<float*>(statLse.data_ptr()),
                                  int(statContext), int(statWidth), int(nbKHeads * HEAD_GRP_SIZE)},
                      stream);''')
            tail = head + launch
        return prefix + tail
    raise ValueError(name)


def patch_prefill(source):
    marker = '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {'''
    anchor = 'void BatchPrefillWithPagedKVCacheDevice('
    if anchor not in source:
        raise UnsupportedNativeStats('FlashInfer paged prefill structure changed')
    start = source.index(anchor)
    return source[:start] + _replace(source[start:], marker, '''          update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        }
        variant.template CaptureSoftmax<KTraits>(params, request_idx, qo_packed_idx_base, kv_idx_base,
            qo_len, kv_len, chunk_end, group_size, s_frag, m, tid, kv_head_idx);

        block.sync();
        if constexpr (KTraits::USE_KV_SHARED_SMEM) {''')


@cache
def source_overlay(directory):
    from flashinfer.jit import env
    manifest = json.loads(Path(__file__).with_name('native_attention_sources.json').read_text())
    roots = {'csrc': env.FLASHINFER_CSRC_DIR, 'include': env.FLASHINFER_INCLUDE_DIR,
             'python': env.FLASHINFER_INCLUDE_DIR.parent.parent}
    for filename, expected in manifest.items():
        root, relative = filename.split('/', 1)
        path = roots[root] / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedNativeStats('source_mismatch:' + filename)
    own = Path(__file__).parent
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    digest.update(inspect.getsource(patch_xqa).encode())
    digest.update(inspect.getsource(patch_prefill).encode())
    for name in ('native_stats.cuh', 'native_prefill_stats.cuh'):
        digest.update((own / name).read_bytes())
    # A content-addressed canonical path prevents absolute includes in generated
    # instantiations from invalidating the cache for every new serving run.
    target = env.FLASHINFER_GEN_SRC_DIR / 'kvmem_stats' / digest.hexdigest()[:16]
    target.mkdir(parents=True, exist_ok=True)
    xqa = target / 'xqa'
    if not xqa.exists():
        shutil.copytree(roots['csrc'] / 'xqa', xqa)
    for name in ('mha.cu', 'mha.h', 'xqa_wrapper.cu'):
        _write(xqa / name, patch_xqa((roots['csrc'] / 'xqa' / name).read_text(), name))
    name = 'flashinfer_xqa_binding.cu'
    _write(target / name, patch_xqa((roots['csrc'] / name).read_text(), name))
    # Relative includes retain their original package directory. Only prefill.cuh
    # is overlaid, and generated instantiations include it by absolute path.
    header = (roots['include'] / 'flashinfer/attention/prefill.cuh').read_text()
    header = header.replace('#include "', '#include "' + str(roots['include'] / 'flashinfer/attention') + '/')
    _write(target / 'prefill.cuh', patch_prefill(header))
    _write(target / 'tvm_ffi_utils.h', (roots['csrc'] / 'tvm_ffi_utils.h').read_bytes())
    for name in ('native_stats.cuh', 'native_prefill_stats.cuh'):
        _write(target / name, (own / name).read_bytes())
    _write(xqa / 'native_stats.cuh', (own / 'native_stats.cuh').read_bytes())
    return target


def xqa_module(directory, q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged=False):
    return _xqa_module(q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged)


@cache
def _xqa_module(q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged):
    from flashinfer.jit.xqa import gen_xqa_module
    from flashinfer.jit.core import gen_jit_spec
    overlay = source_overlay(None)
    spec = gen_xqa_module(q_dtype, kv_dtype, page, dim, group, False, out_dtype, q_len, ragged)
    flags = [flag for flag in spec.extra_cuda_cflags if not flag.startswith('-DUSE_SM90_MHA=')]
    sources = [overlay / 'xqa/mha.cu', overlay / 'xqa/xqa_wrapper.cu', overlay / 'flashinfer_xqa_binding.cu']
    return gen_jit_spec('kvmem_' + overlay.name + '_' + spec.name, sources,
        extra_cuda_cflags=flags + ['-DUSE_SM90_MHA=0'], extra_ldflags=['-lcuda']).build_and_load()


def prefill_module(directory, q_dtype, kv_dtype, out_dtype, idtype, dim):
    return _prefill_module(q_dtype, kv_dtype, out_dtype, idtype, dim)


@cache
def _prefill_module(q_dtype, kv_dtype, out_dtype, idtype, dim):
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    overlay = source_overlay(None)
    name = f'kvmem_fa2_{overlay.name}_{q_dtype}_{kv_dtype}_{out_dtype}_{idtype}_{dim}'.replace('torch.', '')
    spec = gen_customize_batch_prefill_module('fa2', name, q_dtype, kv_dtype, out_dtype, idtype, dim, dim,
        ['stat_rows', 'stat_requests', 'stat_pieces', 'stat_lse'], ['int32_t', 'int32_t', 'int32_t', 'float'],
        ['stat_context', 'stat_width', 'stat_offset', 'sm_scale'], ['int32_t', 'int32_t', 'int32_t', 'double'],
        'KVMemAttention', '#include "' + str(overlay / 'native_prefill_stats.cuh') + '"')
    # The kernel header is included before the generated configuration.
    # Its dependent Capture call resolves only for this custom variant.
    generated = overlay / name
    generated.mkdir(exist_ok=True)
    sources = []
    for path in spec.sources:
        original = path.read_text()
        destination = generated / path.name
        _write(destination, original.replace('#include <flashinfer/attention/prefill.cuh>',
                                            '#include "' + str(overlay / 'prefill.cuh') + '"'))
        sources.append(destination)
    _write(generated / 'batch_prefill_config.inc', (spec.sources[0].parent / 'batch_prefill_config.inc').read_bytes())
    spec.sources = sources
    return spec.build_and_load()
