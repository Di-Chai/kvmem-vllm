# Refresh completion fences and possible model overlap

Status: read-only design review, 2026-10-22. No serving changes or new GPU runs. The measured configuration is the completed 14K physical-prefill candidate; the detailed CUDA timeline below is an older pressure diagnostic and is identified separately.

The current evidence does **not** justify implementing early refresh completion. The entire `assembly_submitted → ready` interval of the warm 256K request is only 150.59 ms over 14 pressure selections. It includes host commit, cleanup and reporting; it is not GPU wait time. In the existing detailed pressure trace, GPU work has already finished before this interval begins. A larger overlap would require a different, layer-oriented assembly pipeline and a provisional scheduling contract, rather than moving the current completion test.

## Measured intervals

Source: `artifacts/prefill-physical14k-20261022-r1/candidate/run/server/capture/async_refresh.jsonl`, joined by command ID with `bounded_refresh.jsonl`. Warm request: `cmpl-8a8e9c55da69986f-0-aad39403`, pressure commands 16–29, final Query command 30.

| Disjoint pressure stage | Total, 14 selections | Mean |
| --- | ---: | ---: |
| suspended → assembly queued | 772.36 ms | 55.17 ms |
| queued → assembly started | 9.96 ms | 0.71 ms |
| assembly started → selection prepared | 681.22 ms | 48.66 ms |
| selection prepared → assembly submitted | 1100.62 ms | 78.62 ms |
| assembly submitted → ready | **150.59 ms** | **10.76 ms** |
| Entire pressure lifecycle | 2714.75 ms | 193.91 ms |

The final interval ranges from 9.00 to 12.72 ms per pressure. The final Query has another 13.24 ms from submitted to ready. These are upper bounds on the wall interval that could be overlapped by starting only after the existing `assembly_submitted` event; even this bound assumes zero new overhead and ignores dependencies. They cannot all be credited to hiding CUDA work. The first long request's 14 pressure tails total 157.54 ms, consistent in scale.

`assembly_host_prepare_seconds` totals 947.25 ms for the warm pressure set. It is nested inside the assembly portion and must not be added to the lifecycle totals. Uploaded pressure payload totals 7,045,087,232 bytes. Neither payload size nor host duration proves equivalent H2D GPU latency.

The older diagnostic `artifacts/prefill-pressure-phases-20261022-r1/prefill-pressure-attribution-20261022.json` covers one pressure with the earlier per-layer rebuild path:

* `selection_prepared → assembly_submitted`: 111.69 ms, with 16.50 ms of GPU activity union.
* `assembly_submitted → ready`: 10.83 ms, with **zero GPU activity overlap**.
* The final rebuild kernel finishes at 211.107 ms after suspension; assembly is declared submitted at 212.234 ms, already 1.127 ms later.
* Bulk H2D GPU time is 13.52 ms and rebuild kernels total 1.90 ms. These are contained in the preceding assembly stage. Resident-V gathering contributes a further 1.06 ms and is also contained in the stage.
* The following public forward is a **32-token replay**, not a 14K prefill. Its first GPU activity begins 5.48 ms after `ready`.

This trace predates the fused-layer assembly and larger physical-prefill changes. It demonstrates why the host tail cannot be called a CUDA stall; it does not prove that the newest 14K path also has exactly zero remaining GPU tail. Existing scopes do not separate `commit_selection`, window/session commit, `assembler.close`, and reporting. No causal attribution among those host operations is claimed.

## What the current completion means

`AsyncRefresh._assemble` returns only after the host has queued every transfer, metadata upload and rebuilding operation. `step` then waits nonblockingly for the returned final CUDA event. Only after it is complete does the worker:

1. Revalidate authority and graph ownership, attach completion, and commit the memory selection.
2. Commit the worker window offset and session selection state.
3. Close the assembly scratch lease, release the maintenance lane, build the report, and release private Query/state references.
4. Set `job.done`; `RefreshJob.query()` returns this flag, not an independent readiness event.

The connector forwards this completion to the scheduler. `WindowManager.commit_refresh` then trims the native allocation and immediately returns removed physical pages to the block pool. The request becomes eligible again and receives its updated table and replay instruction. Cancellation uses the same completion boundary to prevent freeing pages or authority while a refresh still reads them.

Consequently, setting `done` after submission would grant much more than permission to enqueue a model: it would authorize physical-page reuse, scratch reuse, session visibility and transaction completion. Those changes are unsafe while assembly is still in flight.

The source/destination hazards are concrete. Assembly first gathers the entire resident-V read set before overwriting destinations, because remapping cycles can cross chunks. Removed old physical pages can remain sources of this gathering even when their logical positions lie outside the selected window. A wait inserted in the resumed request's compute stream does not protect those pages from another owner's allocation or stream.

`Memory.commit_selection` also requires completion and marks records resident/baked. `Memory.begin_append` requires the resulting READY phase. Publishing new captures against an uncommitted ASSEMBLING selection is not supported. Session checkpoints and Action plans bind the accepted history, window identity and restore state; they cannot silently consume provisional versions of these objects.

## Earliest safe consumer is before a cache write

In the installed vLLM attention path, `Attention.forward` calls `unified_kv_cache_update` before `unified_attention_with_output`. `maybe_transfer_kv_layer` decorates the latter, and the connector's current `wait_for_layer_load` is a no-op because pending requests are suspended.

Adding a wait only in `wait_for_layer_load` would therefore be **after the new K/V cache write**. It would require a separate proof that this write cannot overlap any unfinished assembly destination or old resident-V source. A conservative consumer barrier belongs before the cache update, not merely before attention reads.

The target's first full-attention layer is layer 3, following three GDN layers. Their work could in principle run ahead of a KV readiness event, but Query refresh also restores recurrent state. That restore must independently finish before any resumed GDN work. A CUDA stream wait can preserve a device dependency; it does not by itself preserve host transaction ownership, graph replay semantics or error reporting.

## Minimal protocol if a measurable GPU tail is later found

This is an unimplemented contract, not a proposed default:

* Add a **provisional runnable receipt**, distinct from refresh completion, carrying request/command generation, immutable selected mapping, the original physical allocation lease and the assembly completion fence. Terminal handling must continue to retain that lease. The scheduler must not trim/free pages on this receipt.
* Keep the authority selection, assembler, scratch, metadata, restore checkpoint and source ownership live. A provisional window view must be explicit; code requiring a READY authority or committed Action/session state must continue to wait.
* Admit only a narrowly identified successor replay. Put a main-stream dependency before its earliest conflicting cache write; place a separate recurrent-restore dependency before initial GDN work when relevant. CPU batch preparation can only use immutable validated metadata. Compiled/CUDA-graph execution needs an actual replay-safe device dependency; a Python hook executed only during capture is insufficient.
* Retain the original full-fence validation and commit before capture append, Action/graph publication, request completion, output acceptance, checkpoint publication, or page/scratch reuse. Do not treat event enqueue or a finite-score flag as proof that all earlier GPU work succeeded.
* If assembly fails after the successor has started mutating model state, quarantine the request and retain references through a proven failure fence. The current APIs do not offer a rollback of speculative GDN/model mutations. No success ACK or accepted output may escape.

This is substantially larger than moving `assembly.event.query()`. The measurable post-submission bound is currently below 0.164 s including Query, and its CUDA-only component is unknown for the newest configuration. There is no evidence that this protocol would yield a useful net improvement.

## Why per-layer readiness needs a different assembly order

The current split assembly is **chunk-major**. Each archive chunk contains raw K for all 17 attention layers; after its H2D copy, either one fused 3D kernel or 17 individual kernels rebuild all layers. A layer's full selected context is not ready until the last chunk. Recording a layer-3 event in this order cannot expose the earlier CPU reads and H2D copies to model overlap.

A meaningful layer pipeline would instead make all selected records for early layers ready before reading/uploading later layers. It would need:

* Layer-oriented CPU gather/packing or a different bounded staging layout. The current record regions contain all layers. Re-reading each whole record per layer or retaining the whole roughly 0.5 GB transfer in a new GPU temporary would defeat the intended cost/budget constraints.
* Separate per-layer resident-V preservation and consumer fences, with the full read set protected before that layer's writes. Every pointer, scale, table and record generation must stay bound to the same plan.
* Source leases retained until all asynchronous CPU fills have finished; H2D-slot reuse only after DMA completion; GPU staging reuse only after every relevant rebuild consumer has completed. A layer-ready event cannot substitute for these three distinct boundaries.
* A bounded number of stable readiness events and a replay-safe wait before each layer's first cache access. Shared copy/compute resources and CPU GIL contention must be measured; overlap may slow model execution.
* Full completion before MTP/draft state, publication, terminal acknowledgment and scheduler page recycling become observable. The initial pressure successor is a short 32-token replay; its early GDN work is not a full 14K block of independent computation.

The warm 1.101 s `selection_prepared → submitted` total is the enclosing host assembly interval, not an attainable saving. Most of that interval cannot be hidden by simply making the existing first layer wait later, and neither layer-ready timings nor a measured overlap schedule exist. A redesign should require a new trace showing a substantial layer-ready lead and a dependency-respecting overlap estimate before implementation.

## Follow-up evidence, without changing serving semantics

The next useful diagnostic is the timestamp of the final assembly GPU operation relative to `assembly_submitted`, plus separate host scopes for memory commit, window/session commit, scratch close and report creation. If the final CUDA event is already complete when the future is collected, consumer-fence deferral cannot hide transfer latency. For a larger layer pipeline, measure first-attention-layer readiness and the following 32-token replay's actual per-layer durations before committing to an architecture change.

Reviewed code: `async_refresh.py` (`_submit`, `_assemble`, `step`, `report`), `window_assembly.py` (`assemble`, `_assemble_split_chunks`, `close`), `memory.py` (`select`, `commit_selection`, `begin_append`), `window_manager.py` (`commit_refresh`), scheduler `_complete_refresh`, connector completion and layer hooks, session checkpoint handling, and the installed vLLM `Attention.forward`/cache-update/custom-op ordering.

Evidence SHA-256:

* Current `async_refresh.jsonl`: `2b944f329df6fcfa479c67036961bee35b2483e08f0a26d1e38a151a93d8576b`
* Current `bounded_refresh.jsonl`: `d2812ff6df67ddc4f1d006c0d0d0f5aa04f081bf3b701d294a042bd50b254458`
* Older pressure attribution JSON: `311d6496d9ea0fe7220899d97d4b5d22363533f02bedcc8432f8d83abe42cb12`
