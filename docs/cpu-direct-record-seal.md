# CPU direct record seal candidate

`cpu_direct_record_seal` defaults to `false`. Enabling it requires the CPU archive
backend and `async_seal=true`. The switch changes the CPU asynchronous sealing
copy path; it does not select a different codec, digest, retrieval policy, or SSD
path. It is intended for a separate, single-variable performance comparison.

The original path packs all raw-K/V views into one immutable batch snapshot,
hashes each record, then copies each record into an independent CPU authority
allocation. The candidate calls the existing native `pack_record_views` for each
raw-K/V pair, hashes those independent immutable `bytes`, and publishes those same
objects together. A surviving record owns its own allocation and retains no
batch slab. Packing and hashing finish before publication takes the tier lock.

The shared write protocol still admits twice the payload size, clones repository
generation leases, reserves archive slots, rejects overwrites, and prevents reads
until the complete batch is ready. Cancellation does not cancel admitted I/O.
Publication checks each captured allocation generation and rolls back partial
dictionary insertion. Failed packing, hashing, or publication releases archive
slots, byte admission, and cloned leases. A failed completed job does not need to
retain the source buffer exports.

CPU `io_usage()` and the existing `archive.json` snapshot expose cumulative
successful asynchronous seals:

- `seal_batches`, `seal_records`, `seal_payload_bytes`;
- `seal_direct_records`, `seal_direct_bytes` (zero for the original path);
- `seal_pack_seconds`, `seal_hash_seconds`, `seal_cache_seconds`;
- `seal_record_copy_seconds` (zero for the direct candidate);
- `seal_record_publish_seconds` (direct publication time).

The individual job still returns the original per-record digests. The counters
exclude failed batches and are numeric. Timings describe host work and do not
represent service throughput or GPU utilization.

The CPU regression suite covers byte and digest identity, BF16/FP8 authority
round trips, reusable staging buffers, immutable independent records, delayed
publication, backpressure, cancellation, generation reuse, stale allocation
rejection, pack/hash/publication failures, and submission rejection before a task
is queued. Default CPU and SSD paths are checked separately.

There is an existing limitation in the shared archive executor submission
protocol: `ThreadPoolExecutor.submit()` can enqueue a task and then fail while
starting a worker thread. Its exception handler does not explicitly drain that
unreturned task. This candidate does not change that shared protocol, and its
submission-failure test does **not** qualify the enqueue-then-thread-start-failure
case. Fixing that failure mode is a separate change, outside this copy comparison.

The feasibility probe at
`artifacts/graph-v2-performance-20261018/seal-copy-probe.json` measured the local
copy/hash sequence, with approximately 1.57–1.59× speedup at its two batch sizes.
It did not exercise publication or leases and is not a serving throughput claim.
The candidate requires its own frozen-runtime performance comparison.
