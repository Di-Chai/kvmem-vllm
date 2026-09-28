# Development goal and acceptance

Develop an out-of-tree KVMem plugin, with explicit versioned vLLM hooks where
public extension points cannot represent working-set reselection. New development
lives in this repository. The qw3 executor is a reference, never the serving backend.

Primary qualification: Qwen3.8-27B Q8_0 weights, FP8 target KV, BF16 DFlash2 draft,
per-request K=131072 and G=65536. Activation BF16 does not mean BF16 target weights.
vLLM owns GPU allocation and native batched execution. Histories, working positions,
physical pages and generation-qualified repository handles remain separate.

## Ordered milestones

1. Extract actual host selection/repository/GC logic, preserve source provenance,
   bind Python and pass CPU lifecycle tests.
2. Qualify pinned vLLM/GGUF/DFlash dependencies. Diagnose baseline speculative
   differences using fixed-prefix logits, not only free-running token equality.
3. Capture immutable pre-RoPE K, implement KV layout/scale codecs and packed
   transfers. Never reconstruct authoritative K by undoing quantized RoPE.
4. Implement atomic per-request working-set changes and native scheduler waiting,
   query replay, DeltaNet checkpoints, speculation commit/rollback and cancellation.
5. Validate single-request semantics, then C4 state isolation, capacity pressure,
   refreshes, HTTP/session lifetimes and bounded CPU/NVMe reclamation.
6. Measure fixed real-history workloads (C1/C4 first), natural EOS and independent
   repeats. Report cold service, warm decode, prefill, stalls, acceptance, transfer,
   selection/verification cost, memory and actual agent quality separately.

No 130 tok/s or 48 GiB hard limit. A plugin import or successful generation is not
evidence that KVMem or a throughput improvement works. Unsupported modes must fail
explicitly. Do not silently replace Q8_0, FP8, retrieval, K/G, or workload lengths.

## Status

Host core imported unchanged from qw3 working tree; hashes and source HEAD are in
`third_party/qw3/PROVENANCE.json`. Its Python transaction and asynchronous archive
bindings are implemented. Reference FP8 materialization, packed GPU transfers,
checkpoint ownership and MeanK scoring are implemented; fused raw-Q/K capture
preserves upstream kernel outputs in GPU tests. Narrow versioned runner hooks
are being qualified on the actual model. The scheduler/worker identity-refresh
transaction is now qualified on the actual model. A long-request compressed replay completed determinism checks, but its recurrent
metadata classification was subsequently found incorrect; that semantic claim is
withdrawn. Experimental bounded target serving is now under live qualification.
No qualified KVMem throughput result exists.

Follow-up: real native FP8 pages have now completed an identity roundtrip through
the bounded repository and NVMe archive, followed by fused assembly and commit.
All 16 target layers' cache bytes and the 128-token continuation matched baseline.
The full-attention allocation adapter also passes native-manager tests for bounded
windows and deferred release. Scheduler suspension/acknowledgment wiring has now
passed a real native request: command-only worker rebuild, all-layer byte equality,
page-table replacement and baseline-equal continuation. The combined CPU/GPU suite
passes 67 tests, plus 5 original CTests. Dense-window metadata and position mapping
are integrated with the native model state; nonzero-offset mapping is GPU-tested,
and the full-model identity diagnostic with the hook enabled remains byte/token
equal. A nonzero-offset long-request compressed selection and target replay now
also pass. Automatic bounded prefill/decode admission and draft refresh remain.

Continuous raw-K/FP8-V authority is now integrated in an opt-in native runner path.
It follows native query offsets and rejection counts, seals complete blocks only,
and retains an incomplete tail in bounded memory. Target-only and DFlash C1/C4
checks passed; a 3.5K-token chunked-prefill check verified 139 complete archive
records and excluded 474 rejected candidate rows. Request completion reclaimed
all archive/download resources. The MeanK-to-native-selector bridge is tested.

Next, connect this authority to scheduler-owned compressed assembly and native
recurrent checkpoint/query replay, then draft refresh/rollback. Add native terminal
prefix acknowledgment before persistent-session publication; verifier acceptance
can extend beyond HTTP output when asynchronous batches are in flight. Automatic
admission, background policy GC, nonblocking retirement and graph-compatible
capture still need integration before performance qualification.

Native recurrent restore/identity replay is qualified with actual target-only
and DFlash engine batches, including all four requests' states. Output/state bytes
match on initial prefill and decode, and the 128-token C1 continuations match their
corresponding baselines. A bounded native-window assembler also passes independent
GPU byte checks for moved/noncontiguous selected blocks. A real K131072/G65536
long-request compression diagnostic now passes: 198656 input tokens, one actual
selection at history 195584, a 1024-token query replay, and 128 output tokens.
Repeated compressed replay is output/state byte-equal; all authority resources
are reclaimed. It retains the native allocation, uses a synthetic repetitive
prompt and does not qualify task quality, production refresh policy or throughput.
Its generated text repeats prompt material, so real-task matched quality checks
are necessary. The old frozen DeepSWE native input manifest is available at
`/home/chaidi/qw3-dflash2/benchmark/dflash2/long_concurrency_inputs01/manifest.json`;
it must be hash-checked and used without synthetic padding for future comparison.
That reference used block=32, whereas these mechanical integrations use block=128;
freeze matching granularity and query policy before reporting a matched speedup.

The initial bounded target scheduler/worker path is now wired and CPU-tested,
using native allocation, per-query commands and actual-length acknowledgments.
Its full-model run is still pending. A real frozen Effect DeepSWE input containing
197110 tokens has completed on the native uncompressed reference: 77 output tokens,
natural EOS and a contextually relevant complete tool call. This is one continuation
check, not executed SWE task success. The same input is now used for the bounded
candidate, with all K/G/weight/KV settings preserved. No throughput improvement is
asserted. The first bounded startup found a native warmup lifecycle issue (the
global connector is temporarily disabled); callback installation now waits for the
armed real-request phase, with a dedicated regression test.

A second bounded run (`engine_deepswe_effect_bounded_02`) reached 131072 accepted
history tokens and failed before its first refresh. Worker validation correctly
rejected a mismatch between scheduler full-attention groups and the worker's
slot-mapped groups. `MambaSpec` inherits slot-mapping capability too, so that flag
cannot classify semantic attention. A shared helper now selects actual
`FullAttentionSpec` groups, including uniform wrappers. This also corrects the
older compression diagnostic: its repeated byte equality did not prove recurrent
metadata was unmodified. The original result is retained with this correction in
the qualification record. The combined CPU/GPU component suite now passes 80 tests.
The same real 197110-token request is being rerun with SSE token-arrival recording;
there is still no qualified bounded throughput or long-concurrency result.
