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
| Bounded native target execution | `engine_deepswe_effect_bounded_03`: real 197110-token history, 65 matched refresh commits, bounded native attention allocation, coherent natural-EOS continuation and complete cleanup | Cancellation and GPU/CPU/NVMe capacity-pressure tests; long C4 passes but performance is poor |
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

## Current execution stage (2026-09-29)

The latest source includes the cold-block residency correction, background GC
at the authority/component layer, and scheduler terminal-prefix capture clipping.
The residency correction passed 154 CPU/GPU component tests and all 5 original
C++ tests. Subsequent terminal-prefix changes passed 143 CPU tests with 28 GPU
tests deferred while the model test runs. GC serving activation, exact terminal
GPU qualification and persistent sessions remain incomplete.

`engine_long_c4_dflash_direct_cadence_02` is the active corrected native model run,
from frozen commit `053383f`. It uses the same four long inputs, Q8_0/FP8,
DFlash2 and K131072/G65536, with direct pinned reads, packed assembly, cadence 8
and asynchronous archive retirement. No completed decode result is available yet.
It predates terminal-prefix clipping and therefore cannot qualify reusable final
checkpoints. A completed timing is also not a matched optimization speedup.

All earlier compressed model runs predate the cold-block residency correction.
Their timings and cleanup observations are retained in the qualification record,
but their compressed numerical-correctness/performance acceptance is withdrawn.
The requirement table above is a roadmap of historical evidence, not an override
of this correction. Corrected model runs, matched warmed repeats and executed
agent-task quality remain required.

## Evidence corrections

`engine_compression_k128g64_02` completed deterministic repeated replay but used
an incorrect cache-group classifier: native Mamba specs also advertise slot
mapping. Its recurrent-metadata semantic claim was withdrawn. The corrected
helper selects actual full-attention specs, and the corrected real bounded run
above passes. Keep the old artifact as a historical observation, not a semantic
reference. Detailed measurements and limitations are in
[the qualification record](qualification-20260928.md), with measurement definitions
in [measurement.md](measurement.md) and boundaries in [support.md](support.md).


## Experimental bounded DFlash2 wiring

The new diagnostic `scripts/qualify_engine.py --bounded-dflash` enables a narrow
contract for the existing BF16 DFlash2 draft while retaining Q8_0 target / FP8
KV and K131072/G65536. Native vLLM uses method `dflash` and selects its DFlash2
implementation from the `DFlash2DraftModel` architecture. Five upstream source
hashes pin the draft factory, position/slot handling and model implementation.

The scheduler clips speculative queries at the accepted-history checkpoint and
reserves both speculative GDN rows and recycling-aware draft SWA pages. It leaves
room for verifier and drafter lookahead. Reselection scores use known prompt rows,
or only the first already-sampled input during decode, never unverified draft
candidates. Provisional target forwards restore every batch neighbour's recurrent
state; only the final forward reaches native sampling and drafting. Draft pages
and their logical position frame remain independent of the target dense window.
The actual native draft groups are checked against BF16 SWA geometry before any
compressed refresh. Mandatory target recent context must contain the draft window.
Logical histories beyond the draft RoPE table are rejected pending an explicit
renormalization/session adapter.

Native allocator tests cover combined full attention + speculative Mamba + SWA,
future-growth credit exclusion, recycling and release. These and the admission /
transaction / harness tests pass; they are not compressed DFlash GPU evidence.
The long target C4 run `engine_long_c4_cached_packed_01` was launched from commit
`776bb7a` before these DFlash changes and continues with that loaded runtime and
its saved source manifest. Do not attribute its eventual results to later code.


A separate retirement measurement option now allows omitting diagnostic archive
readback, while preserving publication fences and complete cleanup. Both modes
are covered by real host-archive retirement tests; the publication-only mode has
not yet been measured with the native GPU server. The default still verifies
all checksums. Its implementation does not change the live C4 run, which retains
full readback, and does not satisfy the outstanding background-retirement gate.

The native cancellation diagnostic is now wired behind `--cancel-during-refresh`.
Worker transaction timestamps, native termination status and post-output capacity
snapshots support a client-disconnect test with subsequent C4 recovery. The
validator requires physical page return, not just cleared request IDs or credits.
CPU/socket and scheduler tests pass; the actual native GPU cancellation run remains
pending. Source changes made after `engine_long_c4_cached_packed_01` launched do
not affect its already loaded runtime.


`engine_deepswe_effect_bounded_dflash_01` is now running the same exact 197110-token
Effect input as the corrected target C1, from commit `62726f0`. It retains full
checksum verification and the same K/G and precision. The requested batch budget
was 1024, but native draft-slot reservation followed by KVMem alignment produces
896-token prefill chunks. It is a correctness run, not a matched 1024-query
performance comparison. Future matched C1 uses budget 1152 / request cap 1024;
C4 uses budget 4224 / cap 1024, and must verify the actual scheduled queries.
It has now passed 74 matched refreshes and complete five-request archive cleanup.
C1 generated 77 natural-stop tokens at 85.509 arrival decode tok/s over 0.889 s;
the subsequent short C4 reached 91.956 aggregate tok/s. C1 token IDs match the
cached target control and all short C4 texts match prior native DFlash controls.
This is compressed-prefill / later-decode evidence; decode-time refresh and long
DFlash C4 are still pending. See the qualification record for timing caveats.

## Direct pinned archive reads and reproducible workers

The live DFlash run's host breakdown identified repeated zero-initialization of
the CPU read buffer as a large cost: one 5.44 GB, entirely cache-hit refresh spent
2.645 s allocating buffers, 0.388 s copying cached bytes, and 3.927 s assembling
in total. These are one-round wall timings, not an isolated optimization result.

The new opt-in `--packed-assembly --direct-pinned-reads` reads immutable records
directly into the existing two bounded pinned H2D slots. It removes the separate
zeroed bytearray and second host copy. Namespace/generation checks and archive
leases remain mandatory; CPU preparation and DMA both own their staging slot.
Injected read failure after an earlier GPU chunk must retain the transaction
failure fence. The default path remains unchanged. CPU suite: 104 passed,
28 GPU tests skipped at that stage. The complete CPU/GPU suite subsequently
passed 136 tests at `82e504c`, including direct-path bytes and failed-read fences.
Direct-path native model serving/performance qualification is still pending.
`assembly_host_prepare_seconds` now includes staging copy/submission and waits
for a bounded slot in both paths; earlier reports excluded the upload call, so
use total `assembly_seconds` for comparisons spanning this instrumentation change.

Future engine runs copy the worker Python package and built native extension into
their artifact directory and import that snapshot in all child processes. Both
source and worker manifests are hashed; changing the workspace while a test runs
cannot alter a later worker import. Dependency environments remain pinned and
must not be rebuilt or updated during a run.

An experimental `--prefill-cadence N` now requests native prefill throttling on
N-1 of each N mixed scheduling steps. It does not resize queries, reorder the
running queue or override native admission/capacity gates. Prefill-only work,
unavailable decode and capacity-bound scheduling proceed normally. The default
is 1. This can reduce repeated prefill interruptions but cannot shorten one
blocking refresh or remove retirement pauses; GPU benefits remain unmeasured.
The harness records actual prefill/decode query-size histograms to detect draft
headroom and alignment changes in future comparisons. Targeted scheduler,
option and harness tests: 35 passed. Native cancellation qualification also
counts the disconnected request when waiting for archive retirement.

## Active next GPU run

`engine_long_c4_dflash_direct_cadence_01` launched from `80853e8`, with its own
frozen worker package and native extension. It uses the same four real long
inputs as the earlier target C4, Q8_0/FP8, BF16 DFlash2, K131072/G65536 and B128.
Candidates: direct pinned reads, 36 GiB CPU cache, packed assembly, prefill cadence
8, global batch budget 4224 / request cap 1024, and publication-only retirement.
It must demonstrate actual four-request native compressed batching and cleanup.
This is a combined-path qualification, not a matched speedup against the old
256-query, full-readback target C4. Actual scheduling histograms and complete
HTTP arrival windows will determine follow-up matched controls. Final outcome
is pending; the persistent goal remains active.

## Bounded asynchronous archive retirement

The opt-in `--async-archive-retirement` detaches a finalized authority only after
all native capture DMA has drained. A single bounded CPU worker publishes its
remaining records, optionally verifies every checksum, and reclaims the entire
namespace. Other native requests can keep executing. Publication, verification,
reclamation and acknowledgment timestamps are retained in the archive report.

The native connector deliberately retains the completed request's physical GPU
pages and lifetime reservation until CPU reclamation succeeds. An additional
authority-count gate caps live plus retiring requests at `max_num_seqs`, so
background cleanup cannot silently exceed the configured archive namespace
budget. This first implementation trades delayed capacity reuse for safe overlap;
it does not yet release GPU pages ahead of CPU retirement. Cancellation and
checksum/I/O failures cannot issue a successful free acknowledgment. Idle native
`no_forward` steps deliver completion even when every HTTP request has ended.
The pinned helper source hash rejects incompatible native completion semantics.

Host tests cover concurrent namespace writes during blocked checksum reading,
held generation leases, queue bounds, corrupt archive failure, admission with
spare physical pages, idle native connector completion, and graceful shutdown
ordering. Native GPU retirement/cancellation/pressure qualification remains
pending. The active long-C4 run uses its frozen pre-change package and synchronous
publication-only retirement; it cannot qualify this new option. This feature
does not implement active-history policy GC, persistent sessions or total CPU /
disk byte admission, which remain separate goal requirements.

Latest CPU regression after this integration: **120 passed, 28 GPU tests skipped**
to avoid contention with the active model run. The earlier complete CPU/GPU suite
of 136 tests qualifies the previous code, not asynchronous retirement. Evidence
for this CPU run is `artifacts/components_async_retirement_cpu.xml`.

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. Corrected model qualification is pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

The original repository GC now has Python bindings, including generation leases,
backing flags, policy configuration and GIL-safe background callback shutdown.
Five new host tests cover protection, callback failures/retry and shutdown.
These bindings are not yet activated in the serving authority: durable backing
publication, full retrieval-snapshot leases and bundle cleanup must be integrated
before enabling active-history GC. The qualification harness now stops its owned
server before joining interrupted streaming readers.

## Authority GC integration (CPU qualified, serving activation pending)

An optional `RequestAuthority(gc_config=...)` now attaches the original policy
worker. It publishes RawBacked/ValueBacked/Indexed only after successful durable
write completion. Scoring holds generation-qualified leases over the entire
live index through native selection; returning copied handle values alone does
not protect against concurrent retirement.

Background release reclaims the generation's disk allocation and optional CPU
read-cache entry, then queues a bounded tombstone. The foreground removes old
index generations before installing recycled reservations. The callback owns no
Store/authority/GC bridge reference, touches no GPU data, and never mutates the
foreground dictionary. Finalization stops policy GC before checksum iteration;
reports distinguish previously reclaimed records from live records actually
verified. Request-step aging advances once at construction; ingestion grace can
make progress within that request without inventing per-token request steps.

Seven integration cases cover fixed-capacity reuse across 272 accepted tokens
in an eight-block repository, original sink protection, cache invalidation,
scoring leases and exception cleanup, unobserved durable writes, background
single-request ingestion, and final checksum accounting. Combined CPU regression:
**133 passed, 28 GPU tests skipped** while the model run owns the GPU. Evidence:
`artifacts/components_authority_gc_cpu.xml`. Native serving options, scheduler
backpressure on repository exhaustion, and GPU GC qualification remain pending;
this component is not activated in the current model test.

The corrected long-C4 combined-path run is
`engine_long_c4_dflash_direct_cadence_02`, launched from `053383f` with a frozen
worker snapshot. It additionally opts into asynchronous archive retirement.
Actual decode rates and final lifecycle results remain pending; no speedup is
claimed from the earlier interrupted run or from the component tests.

## Scheduler terminal prefix (CPU qualified, GPU pending)

The bounded connector now carries the finished request's scheduler-owned prefix
before native request fields are discarded. Sampling acceptance is not the final
boundary: native EOS/length checks can trim an accepted speculative burst. The
worker applies that boundary before draining the latest capture packet. Aborted
in-flight rows and rows past the final evaluated output prefix are counted as
`terminal_discarded_tokens`, separately from sampling rejections, and never enter
the raw-K/V archive or MeanK index. Late boundaries after excess publication fail
closed; this path does not silently rewind immutable records.

New bounded archive reports require exact agreement between scheduler prefix,
archive history and HTTP token usage, replacing the historical extra-burst
allowance. Legacy reports retain their old validation semantics. Finalization
also distinguishes GC-deleted records from remaining records checksum-verified
on exit. CPU tests cover EOS overshoot, ordinary decode, prefill/decode abort,
ignored requests, metadata transfer, a real archive with mixed-request capture,
and strict qualification checks. Current CPU regression: **143 passed, 28 GPU
tests skipped** (`artifacts/components_terminal_prefix_cpu.xml`).

This fixes prospective capture publication only. Native recurrent state and
DFlash sliding state may still have advanced past the final retained prefix;
persistent sessions remain disabled until final restore/replay is implemented
and qualified. The running long-C4 uses its earlier frozen source and therefore
does not qualify this new boundary path. Its finished namespaces are discarded,
not reused as sessions; its final report must retain that limitation.

## Canonical accepted GDN state (GPU qualification pending)

`NativeRecurrentState.canonicalize_accepted` can move a nonempty retained native
speculative prefix to the base SSM row and canonical convolution window, then
reset native acceptance to one for a subsequent prefill/decode. It supports the
native SD and DS convolution layouts, validates all groups before mutation and
returns a CUDA completion fence. Source hashes pin the native Qwen GDN layer,
convolution kernel, recurrent kernel and hybrid-state lifecycle. This operation
is not enabled in serving or claimed as session support yet.

Zero retained rows and partial ordinary prefill fail explicitly: their previous
recurrent states require checkpoint restore/replay. The independent DFlash draft
KV/hidden lifecycle and logical-position limits also still require session
integration. GPU cases compare speculative-prefix normalization with independently
executed native convolution and GDN prefixes, including a subsequent ordinary
convolution step and untouched neighbouring pages. These six cases are staged,
not yet executed while the long-C4 model owns the GPU.

Latest host regression: **146 passed, 34 skipped** (28 existing GPU tests plus
6 new native-prefix GPU cases), `artifacts/components_accepted_state_cpu.xml`.

## Next performance evidence and agent gate preparation

The harness now exposes an optional native profiler window for concurrent work,
with a fixed step limit and an explicit instrumented-only timing label. Host
checks validate against the installed vLLM ProfilerConfig and cover overlapping
GPU kernel interval accounting; targeted profiling/qualification tests: 18 pass.
No GPU profile has been run yet and no bottleneck improvement is claimed.

Read-only inspection confirms the existing official DeepSWE environment uses
Pier `datacurve-pier` 0.3.1 and mini-swe-agent 2.4.6. The original requestplan10
runner restarts QW3 for every task and allows a 3M logical context; invoking it
unchanged would violate this plugin's shared-vLLM test scope and current 262K
draft-position boundary. Agent evaluation must therefore get its own vLLM
runner with the same task/agent/verifier provenance, explicit limits, and actual
container/verifier results. Frozen prompts and parseable tool calls remain
insufficient quality evidence. Existing containers/services were only inspected,
not stopped or replaced.

## Latest native result and next gate

The corrected long-C4 run has finished: all four compressed requests and all
resource cleanup checks pass. Whole-group arrival decode is 10.443 tok/s, with
mixed-prefill stalls up to 25.789 s; it is not a performance acceptance. The
short 3.249 s all-stream overlap shows 98.189 arrivals/s only as a diagnostic.
Native sampling draft acceptance is 44.982%, GPU sampled peak is 79.32 GiB,
and asynchronous CPU archive reclamation takes about 28–31 ms per request.
See the qualification record for complete scopes and limits.

All 182 current CPU/GPU component cases pass, including canonical GDN-prefix
state comparisons. The reference fixture correction is recorded with the failed
and corrected artifacts. Next native run will exercise exact terminal-prefix
capture and a bounded GPU profiler window; profiling timings are not benchmarks.
Persistent streaming-session integration and actual agent execution remain open.

The initial terminal-profiler launch (`engine_terminal_dflash_profile_01`) was
stopped before readiness/requests because the harness's default bounded mode
would synthesize a 134144-token C1. Its stop reason and interrupted result remain
in the artifact directory. An explicit `--bounded-short-lifecycle` mode now
separates this terminal/retirement check from the existing mandatory compression
gate. Profiling/qualification host tests: 21 pass; long gates are not weakened.

## Q8 batch linear hotspot and experimental replacement

The short native terminal/profiler gate passed. Its Q8_0 DP4A matrix kernel
accounts for 91.42% of summed GPU kernel time. An optional SM120 Tensor Core
adapter now computes exact INT32 dots per original 32-element Q8 group, scales
in FP32 with unchanged FP16 scales, and reduces groups without unpacking the
whole model into BF16 weights. It only replaces MMQ batches with >=16 rows;
the single-request 8-row DFlash verifier keeps its original path.

The first prototype used round-to-nearest division, which differs at quantizer
ties from the upstream `--use_fast_math` build. That version was never enabled
in serving. The corrected quantizer uses the native constant reciprocal and
`div.approx.ftz`; identity Q8 weights expose native activation quantization and
match every tested value exactly. FP64 grouped-dot references and native CUDA
comparisons pass six GPU cases. The full component suite passes **191 tests**
(`components_q8_tensorcore.xml`). A subsequent short-C4 warmup archive-integrity
test also passes with all 20 qualification host tests.

Isolated real layer-0 QKV, FFN gate and FFN down weights with synthetic inputs
show 1.84–2.02x kernel speed ratios at 32 rows and 1.94–2.09x at 1024 rows.
Relative RMS output differences are 2.6e-5–5.3e-5 in these batch cases, reflecting
FP32 reduction order and BF16 rounding boundaries. Evidence is retained in
`artifacts/q8_tensorcore_real_weights.json`; these are not model throughput or
task-quality results. The switch remains experimental and off by default.

`--warmup-short-concurrent` retains an entire additional short C4 request group
before the measured group, including its streams and archive/terminal checks.
It is only allowed with short lifecycle and streaming metrics, without profiling;
it does not relax the mandatory long compression gate. Native baseline/candidate
runs and real task-quality qualification remain pending.

The first short native comparison passes all lifecycle checks but shows only
88.20 -> 89.09 aggregate arrival tok/s because the candidate retains a 2.799 s
measured pause. Four measured outputs differ. Exact batch-row specialization
has now been removed from the matrix and output reduction kernels; seven
targeted GPU tests pass, including C4/C3/C2 row transitions reusing the same
compiled kernels. This change needs a new native run and is not a retroactive
correction to the recorded throughput.

An installed wheel in a fresh CPU-only environment passes 22 core/storage/GC
tests outside the source package path. Wheel metadata now carries the original
QW3 and vLLM licenses alongside this package's license. Evidence and wheel hash:
`artifacts/wheel-smoke/summary.json`. This verifies host packaging, not a clean
GPU dependency installation or release readiness.

Dynamic-row native follow-up passes and reaches 173.112 aggregate arrival tok/s
in short C4 versus the baseline 88.203 (1.9626x), with longest measured gap
0.1287 s and unchanged ~89 tok/s C1. All nine request lifecycle checks pass.
The same long frozen C4 is the next gate. Output IDs differ under concurrency;
long-history and executed task-quality acceptance remain open, and the adapter
is still opt-in. See the qualification record for full measurement scope.
