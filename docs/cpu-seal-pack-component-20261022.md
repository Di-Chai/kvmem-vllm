# CPU record packing: 1 versus 8 workers

Evidence: `artifacts/cpu-seal-pack-component-20261022-r1/` contains the frozen runtime, native library, manifest, original `component.log`, `results.json`, and `component-audit.json`. The run completed under `/tmp/kvmem-coordination-qualification.lock`, without GPU work. All frozen file hashes matched after execution. The native library SHA256 is `b3074d7b1d7e9825e84bbd6159a0bc340cf5e0f7903ddb0659d0998dc4ab76fe`.

The default remains `cpu_seal_pack_workers=1`. The candidate uses eight fixed record shards on the existing CPU gather executor. Every shard invokes the original `pack_record_pairs`; records remain independently allocated immutable bytes. The caller joins all shards, concatenates references in original order, then performs the original hashing and publication. No additional payload copy or executor is introduced. Failure drains all known work; ambiguous executor submission failure poisons and drains the whole pool before releasing source ownership.

## Qualification and measurement

The new 34 CPU cases and existing hash, direct-seal, and archive-submission regressions passed: **107 passed in 2.14 seconds**. Coverage includes partial shard failure, hidden submitted work, retained failed Futures, source buffer ownership, two owners, admission exhaustion, eight hash workers, record ordering, and the serial compatibility path for older native libraries.

Each component geometry uses 17 layers, 32 tokens per record, four KV heads, dimension 256, FP8 raw K plus byte-preserved V: **1,114,112 bytes per record**. Sources are independent, contiguous, generated nonzero CPU byte arrays. Both arms use direct CPU sealing, hash workers 8, gather workers 8, archive workers 2, and identical admission limits. There are 12 matched AB/BA pairs per geometry after warmup. Input generation, input SHA, and repository creation are outside seal timing. Each trial also reads, byte/SHA verifies, and retires the records; those intervals are reported separately.

All 48 measured trials passed exact byte/SHA checks and released their records. Twelve mandatory boundary-failure cases passed rollback, admission-credit release, and subsequent reuse. Peak admitted bytes stayed at exactly twice the record payload: 71,303,168 bytes for 32 records and 142,606,336 bytes for 64 records.

| Mean wall interval | 32 records, 1 worker | 32 records, 8 workers | 64 records, 1 worker | 64 records, 8 workers |
|---|---:|---:|---:|---:|
| Complete seal, write call through Future completion | 47.107 ms | 15.321 ms | 34.430 ms | 26.841 ms |
| Pack portion, nested in seal | 39.891 ms | 8.327 ms | 20.736 ms | 13.415 ms |
| Hash portion, nested in seal | 6.841 ms | 6.590 ms | 13.080 ms | 12.802 ms |
| Record retirement | 3.331 ms | 4.232 ms | 0.949 ms | 7.575 ms |
| Seal plus retirement | 50.439 ms | 19.553 ms | 35.379 ms | 34.416 ms |
| Full seal/read/verify/retire lifecycle | 79.766 ms | 46.204 ms | 84.342 ms | 84.643 ms |
| Process CPU time during seal | 90.684 ms | 103.045 ms | 116.586 ms | 176.611 ms |

The seal interval improves by 67.48% for 32 records, winning 12/12 pairs, and by 22.04% for 64 records, winning 11/12. For the more representative 64-record logical prefill transaction, seal plus retirement improves only **2.72%**, while the measured full lifecycle is approximately unchanged and seal CPU consumption rises **51.49%**. The results do not establish a model-throughput benefit.

## What the retirement measurements show

`MemoryTier.release` removes the final immutable bytes owner under its lock. `AsyncArchive._release` returns the slot and updates namespace counts only after that removal. The repository reclaim callback invokes this synchronously on the retiring thread. In this benchmark, allocation happens on an archive worker in the reference and on gather workers in the candidate; retirement happens on the benchmark caller in both arms. Thus both arms already have cross-thread allocation/free. The measurements alone cannot identify cross-arena free as the cause.

There is stronger evidence of different allocator/page-return behavior. For 64 records, the reference's typical sealed-to-retired RSS changes from 301,684 to 296,240 KiB, a decrease of only 5.3 MiB. The candidate changes from 363,784 to 296,240 KiB, a decrease of 66.0 MiB, close to the 68 MiB payload. For 32 records both arms return approximately 33 MiB. The candidate therefore appears to return substantially more pages during the measured 64-record retirement, while the reference retains/reuses more allocation capacity. This is an inference from RSS snapshots, not a measured malloc or munmap trace.

The 64-record reference pack time also alternates with pair order: approximately 26–29 ms in AB pairs versus 13.5–14.4 ms in BA pairs. The non-monotonic comparison between 32 and 64 records and the order effect warrant allocation/first-touch attribution. Process RSS snapshots are not isolated peak-memory measurements; process high-water marks are cumulative across both arms.

## Lowest-risk follow-up design, not implemented

Prefer testing **serial allocation with parallel copying**, rather than moving record retirement into a background queue. A native private batch owner could validate all buffer exports and allocate independent `PyBytes` on the original archive thread, retain their destination pointers privately, then let at most eight existing workers copy disjoint record ranges with the GIL released. Only after every range succeeds could a completion method expose the records in their original order. No uninitialized immutable bytes should become visible to Python callers. Every source export and private output remains owned until all jobs drain; duplicate/overlapping ranges and incomplete completion must be rejected. The original 2x admission budget remains sufficient because no extra payload exists. This would require a new native API and separate byte/ownership/failure qualification; no C++ change or rebuild was made for this analysis.

Serial allocation would match the reference's allocation thread and avoid changing allocator placement solely to parallelize memcpy. It would not guarantee identical allocator state or NUMA placement: first-touch writes still occur on worker threads. A targeted diagnostic should distinguish allocation/validation under the GIL, memcpy while the GIL is released, minor page faults, and free/munmap; per-thread task identity and RSS before/after help interpret allocator reuse.

Background retirement is a broader lifecycle change. Retaining bytes in a queued task while immediately returning the archive slot would undercount memory and permit reuse before actual release. A safe design must keep retired bytes and capacity charged, retain generation identity, and make reclaim/close wait for completion; it must also bound queued records and handle executor submission failure. Dropping `PyBytes` on a worker still executes Python reference-count destruction under the GIL, so it does not by itself remove foreground contention or reduce total work. The existing native historical-payload release machinery has a separate ownership model and is not enabled in the current configuration. Enabling it solely to hide this cost would require a separate backend qualification.

No additional CPU performance run or production change was made during the subsequent opaque-projection model measurement.
