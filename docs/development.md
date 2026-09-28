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
transaction is now qualified on the actual model. End-to-end compressed KVMem
serving remains unimplemented. No qualified KVMem throughput result exists.

Follow-up: real native FP8 pages have now completed an identity roundtrip through
the bounded repository and NVMe archive, followed by fused assembly and commit.
All 16 target layers' cache bytes and the 128-token continuation matched baseline.
The full-attention allocation adapter also passes native-manager tests for bounded
windows and deferred release. Scheduler suspension/acknowledgment wiring has now
passed a real native request: command-only worker rebuild, all-layer byte equality,
page-table replacement and baseline-equal continuation. The combined CPU/GPU suite
passes 43 tests, plus 5 original CTests. Actual compressed selections, automatic
bounded prefill/decode admission and recurrent/draft replay still remain.
