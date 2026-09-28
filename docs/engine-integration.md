# Engine integration boundary

The installed vLLM wheel remains unchanged. General plugin discovery is passive
unless explicit diagnostic compatibility options are enabled. The target engine
is vLLM 0.30.0 V2; source hashes guard the version-specific extension points.

Implemented building blocks are not yet a serving pipeline. The next integration
must connect them through the following ownership boundaries:

1. **Scheduler owner.** `EnginePageLeases` allocates through the existing native
   BlockPool and waits for worker acknowledgments before returning pages. The
   current helper is independently tested, not wired into native scheduling.
   A complete integration must replace full-attention history allocation with
   bounded working-set allocation while preserving the recurrent group's native
   state ownership. Do not mutate Request.num_computed_tokens to represent K.
   `KVMemScheduler` now wraps native full-attention managers with
   `WorkingSetManager` in an explicit identity-refresh diagnostic. It suspends
   the request via native decode eligibility, protects pending pages against
   preemption, and uses `KVMemConnector` commands/acknowledgments for commit.
   Native `SchedulerOutput.block_table_updates` replaces complete rows, including
   allocations made in the resume step. This closed loop passed a real GPU request;
   automatic compressed admission and production refresh policies remain pending.
2. **Worker batch.** Keep logical processed-token counts for input IDs, sampling,
   generation limits and recurrent state. Prepare separate dense working-window
   RoPE positions and attention lengths. `WorkingSetMetadata` forwards all native
   GDN/speculation metadata and overrides lengths only for attention groups.
   `metadata_patch.py` supplies the small missing upstream hook, enabled only
   with `KVMEM_WINDOW_METADATA=1`. `WindowBatchAdapter` computes dense positions
   and lengths without changing the native batch's logical counters. The native
   slot-mapping kernel uses dense positions for full-attention groups; recurrent
   groups must have slot mapping disabled. `window_bridge.py` scopes metadata
   overrides to the model-state call and supplies dense text MRoPE input positions.
   GPU tests cover different per-request offsets, reordered requests, mixed query
   lengths and padded rows using actual native BlockTables. The full-model identity
   run with this hook enabled still matches baseline. A nonzero-offset full-model
   compressed refresh and query replay have not yet been qualified.
3. **Raw authority.** `capture.py` retains batched quantized projections and adds
   raw Q/K stores to the existing fusion. Layer capture is verified in a real
   Q8_0/FP8 request. Per-request slicing must use the runner's actual query offsets,
   not SchedulerOutput dictionary iteration order. Capture target and draft
   authorities separately; the current model probe covers the target only.
4. **Archive publication.** Clone generation-qualified repository leases into
   `AsyncArchive` jobs and `PackedTransfers`. Publish only after completion.
   Repository cancellation does not cancel already-admitted I/O. NVMe buffers
   have explicit record/capacity limits and no silent authoritative eviction.
   `CaptureRuntime` now packs every target batch plus native rejection counts into
   one D2H transfer, following native query offsets and GPU logical positions.
   `RequestAuthority` excludes rejected rows, extends a CPU tail, seals complete
   records, and maintains content-frame mean keys. `PackedDownloads` requires
   explicit CPU-consumer release even after its CUDA event completes. This is
   integrated in the opt-in archive diagnostic, including DFlash. Finalization
   drains and verifies records synchronously; production asynchronous retirement,
   policy GC, global capacity admission, and draft authority are still pending.
   Native verifier acceptance can exceed HTTP output at termination when async
   batches are in flight. These temporary records are currently retired in full.
   Persistent sessions must receive the scheduler's final prefix boundary and
   restore/replay recurrent state before publishing a reusable terminal checkpoint.
5. **Refresh commit.** Core selection mutates its host working-set metadata.
   Retain its Plan (which owns previous-window leases), fence asynchronous
   assembly, and make a worker table/length update atomically. On failed assembly,
   fail or restore the request before it is scheduled; never decode with mixed
   old/new state. `assemble_fp8` supports staged V or resident-V preservation.
   The authority's MeanK bridge can now feed accepted-key scores into the original
   native selector. Its returned plan still needs scheduler commit, assembly and
   query replay; the bridge alone does not enable compressed serving.
6. **Query replay / speculation.** Restore pre-query convolution and recurrent
   state using a generation-qualified checkpoint, replay the required query,
   refresh target/draft context and commit only accepted tokens. Rejected pages
   must be cleaned before repository slots become reclaimable. Checkpoints and
   their completion fences are implemented as primitives; full replay is pending.

Physical allocator blocks are not logical KVMem blocks. In the qualification
configuration vLLM chooses 1568-token attention allocator blocks without DFlash
and 1648 with it to match Mamba pages. Kernel pages are smaller. Logical 128-token
records need a distinct page mapping; reusing a repository slot must never reuse
a physical allocation still leased by a different record or request.

The first full integration should use explicit supported graph modes, prefix
caching off, text-only, TP=1 and bounded C1/C4. Extend model/quantization/graph
support only after these lifecycle tests pass. Unsupported cases must fail rather
than falling back to a different KVMem algorithm or precision.
