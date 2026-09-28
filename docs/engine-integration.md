# Engine integration boundary

The installed vLLM wheel remains unchanged. General-plugin discovery is passive;
experimental runner/scheduler adapters require explicit flags and source hashes
for vLLM 0.30.0 V2. Supported execution currently means TP=1, eager, text-only,
synchronous scheduling, prefix caching off and Mamba cache mode `none`. See
`support.md` and `qualification-20260928.md` for evidence and remaining gates.

## Ownership and execution

1. **Native scheduling and pages.** `KVMemScheduler` uses the native scheduler,
   sampling and continuous batches. `WorkingSetManager` bounds full-attention
   storage while GDN retains native recurrent allocation. Logical computed-token
   counts never stand in for dense window length. `ReservedKVCacheManager` reserves
   future window growth plus recurrent/draft pages and limits live/retiring
   authorities. A refresh can release physical pages while retaining its growth
   reservation. Active-request preemption is refused until checkpoint restoration
   is qualified. Native capacity-pressure testing remains outstanding.
2. **Window metadata.** `WindowBatchAdapter`, `WorkingSetMetadata` and the scoped
   model-state bridge supply dense RoPE positions, attention lengths and slot
   maps only for semantically identified full-attention groups. Native GDN and
   draft logical positions remain intact. `MambaSpec.uses_slot_mapping` is not an
   attention classifier. Logical 128-token authority records differ from allocator
   blocks and kernel pages; page ownership always stays with vLLM.
3. **Accepted authority.** Fused capture retains target pre-RoPE BF16 K and unchanged
   FP8 V. One packed D2H transfer covers each native batch and its rejection counts.
   `CaptureRuntime` uses actual batch offsets and excludes rejected proposals.
   `RequestAuthority` extends mutable tails, seals immutable complete records and
   maintains content-frame MeanK. Per-request namespaces and repository generations
   prevent cross-session or recycled-slot aliasing. Uniform target FP8 e4m3 geometry
   is the currently qualified codec, not a generic format guarantee.
4. **Storage and transfer.** `AsyncArchive` clones repository leases and bounds
   shared I/O count plus optional in-flight bytes. NVMe capacity is geometry-derived;
   optional physical reservation and a shared arena-byte cap fail before readiness
   when unsatisfiable. These startup/byte controls currently have CPU evidence.
   `CachedArchive` has independent global/per-namespace read-cache limits and never
   substitutes cache presence for durable publication. Packed/direct assembly
   coalesces immutable record reads; pinned slots remain owned through CPU fill
   and CUDA completion. Completed read results and authority/serialization buffers
   remain separate from the in-flight I/O cap. Whole-process host-byte admission
   and fair asynchronous I/O scheduling are still missing.
5. **Selection and replay.** A native pre-forward connector command initiates
   query-conditioned selection after K, with query sizes clipped to aligned K/G
   boundaries. `bounded_runtime.py` snapshots provisional native state, obtains
   the query, selects history, assembles raw K with its new RoPE positions, restores
   state and replays. Only the final pass is sampled/captured. Current replay
   includes neighbours in the provisional batch, a remaining performance cost.
   Assembly completion retains the selected plan and page owners. Scheduler
   acknowledgments commit actual selected lengths and complete page-table rows,
   including replay allocations. GDN counters retain logical history.
6. **Incremental assembly and failure.** A retained unmoved record can reuse
   resident K/V; moved K is rebuilt from immutable pre-RoPE K. Dropped blocks lose
   working residency only after the assembly fence. Reselecting a previously
   dropped block at its old position must reload it. This rule fixes the withdrawn
   earlier compression measurements. Failed partial assembly owns a cleanup fence
   and cannot commit or resume; cancellation waits for the same ownership chain.
7. **DFlash2.** The opt-in draft contract preserves the original target Q8_0 and
   FP8 KV with BF16 draft weights/native sliding KV, seven speculative proposals
   and an eight-token verifier query. Source-checked draft loading supports the
   mixed GGUF/safetensors setup. Rejection counts travel with capture. Corrected
   compressed long C4 and short exact terminal archive tests pass. The qualified
   draft logical-position range is 262144; longer sessions need a separate position
   adapter, never silent truncation or changed K/G.
8. **Termination and reclamation.** Scheduler terminal-prefix metadata arrives
   before draining the latest capture, clipping EOS/length/abort overshoot exactly.
   Optional bounded CPU retirement publishes, verifies when requested, and reclaims
   the archive before connector acknowledgment frees native pages/reservations.
   Actual HTTP active-prefill cancellation passes; compressed-refresh cancellation
   remains a separate gate. Serving GC uses the original background policy and
   generation-qualified release callbacks. Scheduler ready-record credits debit
   unpublished accepted/speculative rows; idle worker feedback can unblock requests.
   GC/admission host tests pass, but native GPU pressure qualification is pending.

## Persistent-session integration still required

A request ID or HTTP session header is not state reuse. The current official agent
backend reconstructs each turn as a fresh request. Native vLLM streaming-input
sessions retain pages but bypass normal terminal cleanup and discard the previous
last sampled token on update; using them directly would miss KVMem's final-prefix
and recurrent/draft-state contract.

The imported core's bounded repository checkpoint/rewind transactions are exposed
through `RequestMemory`. A checkpoint pins its current working prefix. Rewind
requires a completion fence before publishing shorter history, holds discarded
suffix leases through cleanup, and permanently invalidates later checkpoints.
The Python adapter currently accepts sealed-record boundaries only. Partial-tail
branching requires archive generation handling; it must not overwrite immutable
published records in place. These are host ownership primitives, not completed
persistent serving. `NativeRecurrentState.canonicalize_accepted` separately has
GPU component evidence for retained speculative rows, but is not wired to session
turn transitions.

Next integration must:

- Suspend resumable turns and send the exact retained prefix through the connector;
  acknowledge target GDN, draft KV/hidden state, capture and native page cleanup
  before accepting the next input. Preserve per-turn sampling and output limits.
- Compare exact rendered token prefixes. Thinking trimming and XML tool-call
  normalization can rewrite generated output. Restore a checkpoint before the
  first differing token and replay the official new history; do not change chat
  serialization to force a prefix hit. A recent block-aligned prompt checkpoint
  is a candidate that avoids partial archive overwrites. Unsupported branches
  require a correct explicit recompute path.
- Bound idle sessions/checkpoints and support offload/admission beyond active GPU
  slots. Qualify isolation, reconnect/abort/expiry, speculation rollback, GC and
  repeated tool turns under contention before using persistent state for scores.

Reconstructed prompts from 21 live agent calls match recorded native token counts;
all 19 adjacent pairs retain the prior reconstructed prompt as a prefix
(`agent_prefix_analysis_02`). This supports investigating prompt checkpoints but
is not raw generated-token equality, proof of native state reuse, or task quality.

## Qualification boundaries

Current single comparisons show short C4 arrival decode improving from 88.20 to
173.11 tok/s with the optional integer Tensor Core Q8 linear adapter. Four frozen
long histories improve from 10.44 to 18.62 tok/s across the full arrival window,
including prefill/refresh interference. Output streams differ and the long baseline
predates terminal clipping; these are not repeated controlled estimates or quality
acceptance. The agent harness preserves official tasks/solver/verifier policy and
records all failures. Full fixed-set scores, matched native-Q8 controls, repeated
performance, persistent sessions and clean installation/release validation remain
required before production claims or broader model/quantization support.
