# Development goal and acceptance

Develop an out-of-tree KVMem plugin in this repository, with explicit versioned
vLLM hooks where public extension points cannot represent historical working-set
reselection. The qw3 executor is a read-only semantic reference, never the serving
backend. The goal remains active until the full acceptance scope below is proved.

Primary qualification is Qwen3.8-27B Q8_0 target weights, FP8 e4m3 target KV,
DFlash2, and per-request K131072/G65536. BF16 activations and draft weights do not
change the Q8_0 target path. vLLM owns native GPU allocation, scheduling, batching,
attention and sampling. Logical history, dense working positions, physical pages
and generation-qualified repository handles must remain distinct.

## Acceptance scope and current evidence

| Requirement | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Original selection/storage semantics | Unchanged host core with file hashes in `third_party/qw3/PROVENANCE.json`; MeanK score bridge; immutable raw K; dense position assembly; original sink/recent bands | Match old controller's final-query override and record granularity before declaring an equivalent-workload speedup |
| Bounded native target execution | `engine_deepswe_effect_bounded_03`: real 197110-token history, 65 matched refresh commits, bounded native attention allocation, coherent natural-EOS continuation and complete cleanup | Long C4, cancellation and GPU/CPU/NVMe capacity-pressure tests |
| DeltaNet and transaction correctness | Native target/DFlash identity replay, C4 recurrent-state restoration, rejected candidate exclusion, per-generation leases and failure fencing | Compressed target/draft refresh, accepted-boundary checkpoints, terminal-prefix rollback and persistent sessions |
| Efficient transfers and storage | Packed D2H; fenced H2D; moved-block FP8 assembly; coalesced native NVMe I/O; opt-in archive-layout assembly and bounded CPU read cache | Independent performance repeats, overlapping preparation, background policy GC and nonblocking retirement |
| Cross-request resources | Native lifetime GPU-page reservation adapter with hybrid-manager tests; shared archive namespace isolation; bounded queue/cache components | Native pressure/cancellation proof, disk admission, CPU total-budget accounting, long-running reclamation |
| Model/quantization extensibility | Separate state, position, capture and KV-layout adapters; public connector options; explicit source/version checks | Broader qualified model, quantization and KV-format matrix; unsupported cases must remain explicit |
| Performance and task quality | Exact frozen DeepSWE token inputs and provenance; native SSE token/usage agreement; cold/arrival-decode measurements kept separate | Matched warmed C1/C4 throughput and latency, independent repeats, DFlash acceptance/verification cost, memory peaks, executed agent-task success |
| Delivery | Independent repository, reproducible GPU setup and dependency pins, local commits and qualification records | Clean install/release verification and final publication/review artifacts |

The current corrected bounded target diagnostic uses 128-token records and a
1024-token batch budget. The old qw3 frozen reference used 32-token records and
its own final-query override. Keep those differences visible; do not change
precision, K/G, histories or output stopping rules to manufacture an improvement.
There is no hard 130 tok/s or 48 GiB acceptance threshold. Substantial concurrent
throughput improvement must be supported by actual matched measurements.

## Current execution stage

The CPU/GPU component suite passed 102 tests at commit `dbd7451`; all five original
C++ tests passed before these Python-only integrations. The corrected bounded C1
run completed 77 tokens at 20.491 HTTP arrival decode tok/s **without DFlash**, with
1877.601 s TTFT. Subsequent short C4 produced 512 capped tokens at 76.225 aggregate
arrival decode tok/s, but waited about 23.75 s for synchronous archive retirement.
These are diagnostic observations, not the goal's performance pass or long C4.

The cached/packed candidate `engine_deepswe_effect_cached_packed_01` passed the
same 197K C1 and short C4 with exactly matching output token IDs and complete
cleanup. C1 TTFT decreased from 1877.601 to 1280.156 s (31.82%); arrival decode
was 19.448 rather than 20.491 tok/s. This demonstrates a cold prefill/refresh
improvement, not a decode gain or a warmed performance pass. Native lifetime
GPU-page reservation was attached. The latest CPU suite passed 83 tests; 24 GPU
component tests were skipped to avoid interfering with the live model run.

Then run long-history C4 with actual native batch participation, followed by
cancellation/pressure tests. Finish DFlash refresh and accepted-prefix/session
lifecycle before the final warmed performance and agent-quality gates. A client
submitting four HTTP requests does not itself prove four-request GPU batching.

## Evidence corrections

`engine_compression_k128g64_02` completed deterministic repeated replay but used
an incorrect cache-group classifier: native Mamba specs also advertise slot
mapping. Its recurrent-metadata semantic claim was withdrawn. The corrected
helper selects actual full-attention specs, and the corrected real bounded run
above passes. Keep the old artifact as a historical observation, not a semantic
reference. Detailed measurements and limitations are in
[the qualification record](qualification-20260928.md), with measurement definitions
in [measurement.md](measurement.md) and boundaries in [support.md](support.md).
