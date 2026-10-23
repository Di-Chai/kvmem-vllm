# Bounded physical KV capacity (experimental)

The opt-in serving profile field `bounded_kv_capacity: true` lets the qualified
KVMem runtime retain its logical `max_model_len` while vLLM checks full-attention
KV demand against the enforced `select_budget + gen_budget` working window.
The default remains false. This implementation uses vLLM's public custom-spec
registry; it does not patch the installed package or bypass its memory planner.

## Qualification

Only the fingerprinted vLLM 0.30.0 V2 runner, native Qwen MTP1 with FP8 KV,
TP=PP=DCP=PCP=1, synchronous C1, APC off, Mamba `none`, hybrid KV management,
and explicit request planning are accepted. The explicit logical context must
not exceed 262144 or either target/draft HF `max_position_embeddings`.
The target must declare `qwen3_5_text`; the existing draft contract checks its
matching MTP architecture. This permits the recorded model's original 256K
positions, not a RoPE extension. vLLM's `original_max_model_len` records the
user setting, so it is not used as the architectural limit.

The KVMem scheduler and connector must be selected. All full-attention groups
must have the same explicitly declared physical cap and the real existing
WorkingSetManager/ReservedKVCacheManager allocation path. Unknown specs,
manual GPU block/byte overrides, auto-fit logical length and HiSparse fail
closed. The physical window may not exceed the configured logical length.
A false profile removes ambient `KVMEM_BOUNDED_KV_CAPACITY` at launch.

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

## 256K logical / 52K physical CPU qualification

The tests exercise the real vLLM planner and allocator without CUDA allocation.
The recorded Qwen3.8-27B target plus MTP geometry has 17 full-attention layers,
1584 tokens per native attention page, and 55,148,544 bytes per pool page.
There are three recurrent groups of 16 layers, each requiring two pages
including one MTP speculative state. The KVMem archive/selection block of 32
tokens is a separate granularity.

For K=36864 and G=16384, physical full-attention capacity is 53248 tokens:

- 34 full-attention pages: `ceil(53248 / 1584)`.
- Six recurrent/speculative pages and one permanent null page.
- 41 total pages, 2,261,090,304 bytes (about 2.106 GiB).

The logical full-attention block table still has 166 entries for 262144
positions. A 55-page measured pool keeps all 55 pages; the adapter does not
reduce its allocation to the minimum. One byte below the 41-page minimum
fails the original vLLM check. Two worker specifications preserve the native
minimum-worker rule and reject an undersized worker. Without the adapter,
the same logical length requires 173 total pages and fails a 55-page pool.

The earlier K=4096/G=4096 cases remain covered at logical 16384/24576/32768:
six full-attention pages, six recurrent pages and one null page. A 19-page
pool remains 19 pages; one byte below the 13-page minimum fails.

Coverage includes both hybrid layer orders, spec serialization into a fresh
CPU process, unchanged archive accounting, logical table widths, source and
unsupported-configuration rejection, and an actual native hybrid allocator:

- Repeated 128-token growth and acknowledged refresh reach near 256K logical
  history while using at most 40 usable pages. MTP beyond K+G is rejected
  without page mutation, and C1 lifetime admission prevents overcommit.
- Cold checkpoint import near the logical limit allocates one destination
  window, including recurrent and speculative state. It does not allocate a
  second native window for the historical logical prefix.
- A private pressure replay at the K+G boundary retains its destination in the
  existing pages. The control transaction must be validated before ordinary
  public MTP resumes; refresh frees pages and permits regrowth.

The runtime implementation supports this accounting: `prepare_refresh`
retains original page IDs and a selected prefix; `commit_refresh` frees the
suffix after the fence. Checkpoint restore uses validated native destination
leases. These operations do not request a second native full-attention pool.

## GPU and CPU admission remain separate

These tests qualify capacity accounting, not a successful 256K GPU workload.
The native pool is only part of peak GPU use. In this geometry, full-history
`torch_reference` MeanK scoring transfers a 512 MiB FP32 index to the GPU;
query/logit/softmax temporaries can raise scoring workspace to roughly
1.06–1.25 GiB. Two assembly arenas with chunk size eight can retain about
34 MiB of device staging. Private query rollback also clones the allocated
GDN state rows outside the native pool: about 296 MiB with MTP1, estimated
from the saved 154,927,104-byte single-row checkpoint and two native rows;
the actual state bundle must confirm this estimate. That clone can overlap
scoring or assembly until its restore fence completes. Scoring and assembly
have different live phases, but allocator reservations can persist.
Activations, RoPE tables and kernel workspaces require a fresh runtime
measurement as well.

A 41-page startup proves native page capacity only. GPU utilization must
leave measured headroom for these other allocations. Read the newly produced
geometry rather than assuming the old 55-page pool survives a new logical
length or utilization. No model weights are offloaded by this adapter.

For the same single-session 256K experiment, CPU history admission can use
8196 repository blocks and 12 GiB `cpu_read_cache_bytes` for authoritative
archive plus indexes; the raw full-attention history alone is 8.5 GiB.
The separately configured checkpoint budget remains necessary for recurrent
and MTP state. Host/cgroup headroom, lifecycle release and actual long-session
archive occupancy must be measured independently of native GPU capacity.

vLLM's reported token capacity is a logical-equivalent concurrency figure once
this spec is enabled. Physical capacity should be read from `native_blocks`,
`capacity_bytes`, per-group geometry and K/G in `cache-geometry.json`; logical
length remains in the saved serving profile. Do not treat that log figure as
additional physical KV storage or as evidence that GC has reclaimed history.

CPU validation on 2026-10-23: **198 passed** across capacity and existing
serving/cache/reservation/draft suites. After adding cold import and private
replay to the allocator cases, the affected suite plus pressure-control
regressions passed **52 tests**. CUDA remained uninitialized.

```sh
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src python -m pytest -q \
  tests/test_bounded_capacity.py tests/test_serving_launcher.py \
  tests/test_cache_groups.py tests/test_cache_manifest.py \
  tests/test_capacity.py tests/test_draft_contract.py
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src python -m pytest -q \
  tests/test_bounded_capacity.py tests/test_pressure_control_reservation.py
```
