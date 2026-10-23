# MeanK reference scoring workspace

The torch reference backend streams one layer of the CPU MeanK index at a
time. It packs that layer on its source device into contiguous [KV heads,
head dimension, blocks], then transfers it to the Query device. Explicit
batched matrix multiplication consumes contiguous grouped-query rows. This
avoids uploading the complete index and then letting an all-layer einsum
pack a second complete GPU copy.

The mathematical scoring domain is unchanged: every eligible history block
participates in the same softmax for each layer/query/head. No independent
block-wise softmax, block removal, approximation or KV downsampling is used.
The model attention scale, GQA head mapping, mask, zero mass for fully masked
rows, averaging across layers/heads and sum across Query tokens are retained.
FP32 accumulation order changes when layers are processed separately; bitwise
score identity or ordering of arbitrarily close ties is not guaranteed.

## Shape-dependent working memory

For the verified model shape L=16 target layers, T<=128 Query tokens,
H=24 Query heads, KVH=4, D=256, B<=8192 history blocks, and the default
query_chunk=16:

- One FP32 packed index layer occupies at most 32 MiB.
- Logits and softmax output each occupy at most 12 MiB.
- Query conversion/packing, masks and block-score reductions are small
  additional arrays. The major explicit scoring arrays total 56 MiB.

In-place scaling, masking and NaN replacement avoid additional logit-sized
arrays. Query temporaries are dropped after each chunk and the index slab is
dropped before the next layer upload. No complete GPU MeanK index is created
by the authority reference path. The original CUDA backend retains its prior
device-index interface.

These bounds describe this shape and chunk size, not an unconditional memory
limit for every possible model. They exclude the existing Query tensor,
CPU archive/index, CUDA allocator reservations, library/kernel workspaces,
GDN rollback, target KV and assembly staging. Total H2D index bytes over all
16 layers remain about 512 MiB at 256K history. More kernel/transfer calls can
change Query latency; steady decode and pressure-refresh latency must be
measured separately. K, G, logical model context and GPU utilization are not
changed by this fix.

## Qualification

CPU tests compare the new path with the frozen previous all-layer FP32
formula at rtol=2e-6/atol=1e-7, including multiple GQA groups, Query chunks,
FP32/FP16/BF16 inputs, noncontiguous inputs, mask holes, fully masked rows,
extreme finite logits and a ranking fixture with a clear score margin.
Meta-device operator checks execute the complete 16-layer/8192-block loop
without allocating CUDA: both bmm operands are contiguous, softmax always
covers all 8192 blocks, and each index transfer contains one layer.
Authority integration checks preserve the original CUDA-backend routing.

This CPU qualification does not establish the final GPU peak, full-context
functional success or throughput. Repeat the original capacity requests with
the same K36864/G16384, logical262144 and utilization0.86, preserving both the
failed original run and new runtime evidence.

Validation on 2026-10-23: **75 passed** across the new workspace tests and
existing state/retrieval, authority GC, Query-history, historical-graph and
graph-lifecycle suites. All execution was CPU/meta with CUDA hidden.

```sh
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_retrieval_workspace.py \
  tests/test_state_retrieval.py tests/test_authority_gc.py \
  tests/test_query_history.py tests/test_historical_graph.py \
  tests/test_graph_lifecycle.py
```
