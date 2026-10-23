# Bounded physical KV capacity (experimental)

The opt-in serving profile field `bounded_kv_capacity: true` lets the qualified
KVMem runtime retain its logical `max_model_len` while vLLM checks full-attention
KV demand against the enforced `select_budget + gen_budget` working window.
The default remains false. This implementation uses vLLM's public custom-spec
registry; it does not patch the installed package or bypass its memory planner.

## Initial qualification

Only the fingerprinted vLLM 0.30.0 V2 runner, native Qwen MTP with FP8 KV,
TP=PP=DCP=PCP=1, synchronous C1, APC off, Mamba `none`, hybrid KV management,
explicit request planning and logical context up to 32768 are accepted.
The KVMem scheduler and connector must be selected. All full-attention groups
must have the same explicitly declared physical cap and the real existing
WorkingSetManager/ReservedKVCacheManager allocation path. Unknown specs,
manual GPU block/byte overrides, auto-fit logical length and HiSparse fail
closed. The physical window may not exceed the configured logical length.

The worker converts only exact FullAttentionSpec objects into the registered
BoundedFullAttentionSpec. Its `max_memory_usage_bytes` uses ceiling-rounded
physical K+G pages. Its merge preserves and checks the cap; hybrid uniformity
discovery works independently of layer order. The original logical block-table
width is retained. Mamba state and speculative padding are unchanged.
EngineCore's original per-worker byte checks, shared-page padding, permanent
null page, allocation size and minimum-worker capacity remain authoritative.
A second scheduler check proves every declared cap is enforced by the actual
allocator wrappers. Runtime lookahead cannot extend beyond K+G: the existing
allocator refreshes or rejects it before allocating more pages.

## CPU qualification and limits

The tests exercise the real vLLM planner and allocator without CUDA allocation.
For the recorded Qwen3.8-27B target plus MTP geometry, each pool page is
55,148,544 bytes. K=4096/G=4096 needs six full-attention pages, six recurrent
pages (three groups, including one speculative state per group), and one null
page: 13 total. The existing 19-page pool is 1,047,822,336 bytes and remains
exactly that size for logical lengths 16384, 24576 and 32768. One byte below
the 13-page requirement fails the original vLLM check; without the adapter,
the 19-page pool fails the native 32768-token requirement.

Coverage includes both hybrid layer orders, spec serialization into a fresh
CPU process, unchanged archive accounting and logical block-table widths,
unsupported configurations and source rejection, plus real native hybrid
allocation at K+G including MTP lookahead. An attempted extra token fails
without page mutation; acknowledged refresh permits regrowth under the same
12 usable-page bound and C1 lifetime admission.

These checks qualify capacity accounting, not a successful 32K GPU workload.
Longer logical lengths can still increase metadata, activation or staging
memory outside the KV pool. Preserve the serving profile's measured GPU
headroom and assembly chunk size; run a fresh 24K/32K startup, marker/tool
request and long refresh/replay trace before claiming support. CPU history
and checkpoint admission remain independent constraints.

vLLM's reported token capacity is a logical-equivalent concurrency figure once
this spec is enabled. Physical capacity should be read from `native_blocks`,
`capacity_bytes`, per-group geometry and K/G in `cache-geometry.json`; logical
length remains in the saved serving profile. Do not treat that log figure as
additional physical KV storage or as evidence that GC has reclaimed history.

CPU validation on 2026-10-23: **182 passed** across the new capacity tests and
existing serving, cache-group/manifest, reservation and draft-contract suites:

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
  tests/test_bounded_capacity.py tests/test_serving_launcher.py \
  tests/test_cache_groups.py tests/test_cache_manifest.py \
  tests/test_capacity.py tests/test_draft_contract.py
```
