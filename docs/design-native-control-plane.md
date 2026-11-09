# GIL-free native control plane for kvmem background threads (option A)

Status: design, 2026-11-09. Approved direction: user, 02:50.

## 1. Problem

The engine thread launches each forward from Python. vLLM's piecewise graphs and
the Triton launcher release the GIL around every launch, so whenever a kvmem
background thread wants the GIL it takes it, and the engine thread waits until
that thread releases it (up to the 5 ms switch interval). If the GPU's queue
drains meanwhile, the GPU idles.

Measured on RULER 256K, G4K, N=8, r3 flags (nsys at prefill +100 s, py-spy
`--gil` over the first 130 s of prefill):

| Item | Value |
|---|---|
| GPU kernel-busy | 86.8% |
| GIL-related GPU idle (host-late gaps + step gaps) | ~8.4% of wall |
| Engine thread GIL waits | ~3.0 s per 15 s |
| Refresh thread holds the GIL | 16.7 s per 130 s |
| Scatter (publication) thread holds the GIL | 4.1 s per 130 s |

Shorter switch intervals (r7) and porting single hot loops to C++ that still
take Python objects (r9) did not help. Those native helpers keep the GIL
because their inputs are Python dicts and `Record` objects. The waits come from
how often background threads take the GIL, not only from how long they hold it.

## 2. Principle

1. **C++ owns the hot-path control state:** records, per-layer maps,
   selections, commit maps and assembly plans, in plain C++ structures.
2. **One native call per unit of work, with the GIL released for all of it:**
   a refresh's layer selection, one layer's assembly plan, one publication span.
   Python only orchestrates, and only at that coarse granularity.
3. **Explicit locks replace the GIL's implicit serialization.**
   - Today the GIL makes every Python-level mutation atomic. That includes
     authority records, `KvMemStore` calls and repository leases.
   - Native state gets per-request mutexes, plus one mutex per shared store and
     repository.
   - Refresh works on immutable snapshots: `shared_ptr<const Table>` published
     by version.
4. **Python remains for:**
   - vLLM integration (scheduler, connector, runner, attention hooks);
   - tests and oracles;
   - cold paths: setup, sessions, restores, GC policy decisions.
5. **Same rules as before:**
   - every stage is opt-in;
   - byte-exact at N=1 (`compare_exact.py`: answers, token ids, 259 pressure
     selections, 256 Query layers);
   - same errors and messages;
   - portable (no ISA-only code; threads capped at the core count);
   - no extra GPU memory.

## 3. State model

`RequestControl` (C++, one per request; Python holds a handle):

| Field | Contents | Fed by |
|---|---|---|
| `RecordTable` | slot, generation, history_begin, count, archived, sealed flags, archive handle id; sorted by history_begin; versioned | authority append/retire |
| `LayerMaps` | per layer: origin + rows (slot, generation, begin, count, dense position) | commit / Query frame |
| `Histories` | per-layer accumulated attention (already native `State`, `scores_for`) | acceptance / folds |
| `Mandatory` | hard blocks of the current refresh | refresh command |

- **Stages 1–3 (mirror):** the table is a mirror. It is synced from
  `VersionedRecords.log` under the GIL. The sync costs O(new records), about
  128 records per 4K chunk. Any non-append mutation triggers a full rebuild.
  This matches `record_index`'s cache rules, so it is exactly what the Python
  index sees.
- **Stage 4 (source of truth):** the C++ table becomes authoritative.
  `Record` becomes a proxy object for the remaining Python users (50 files,
  363 attribute references).

## 4. Stages

GIL times below are per 130 s at G4K N=8 with r3 flags.

### Stage 1: refresh selection prepare (~10 s of GIL)

Moves these parts of `LayerwisePrefillPressure.prepare` and `select_layer`:

| Part | GIL time | Note |
|---|---|---|
| Native layer pick call | 3.15 s | Holds the GIL today: Python dict lookups and attribute reads |
| `prepare_commit` / `commit_map` | 2.3 s | |
| Deferred fold `force_all` | 1.0 s | |
| Record index sync | — | |
| Mandatory lookup | — | |
| Evidence lists | ~0.6 s | |
| Lazy per-layer selection | — | |

Native entry `prepare_pressure(control, snapshot, mandatory, budget, lazy)`,
GIL released, returns per layer:

- ordered slots;
- candidate and selected history_begin arrays (int64 numpy, for evidence);
- the commit-map rows.

Python materializes the `picked` record lists from the slots by index; this is
a cheap list lookup that keeps the Python objects identical. Evidence dicts are
built from the arrays on the evidence thread, so the JSON is unchanged.

Concurrency points:
- `KvMemStore.pick_prefill_pressure_blocks` and its retrieval-score frame run
  under a store mutex.
- Every other store entry point must take the same mutex. This needs an audit
  of `bindings.cpp`.

### Stage 2: per-layer assembly plan and reader (~5 s)

- The source lists, alias checks and `PressureReader.read` region lists come
  from the C++ snapshot, not from `Record.handle`.
- The CPU gather already runs in C++.
- One short Python call per layer remains, for the Triton launch (about
  0.1 ms).

### Stage 3: publication (~4 s, scatter thread)

These parts of `_publish_logical_claim` move to one native call per span, under
the request mutex:

- `_commit_prepared_append` (authority append);
- `prepare_graph` (block graph);
- `_LayerwiseAcceptance.commit` / `_PreparedActionAcceptance.commit`.

It appends to the C++ table directly. The Python `Record` objects for these
records are created lazily (stage 4), or in bulk in the meantime.

### Stage 4: C++ owns the records

- `Record` becomes a proxy.
- `VersionedRecords` becomes a view over `RecordTable`.
- The mirror sync goes away.

## 5. Verification per stage

1. Equivalence unit tests: native against the current Python, random and
   adversarial inputs, identical results, errors and messages.
2. Thread-safety tests: concurrent publication and selection on the same request
   under the new mutexes; ASan/TSan builds (`build/history-gc-tsan` pattern).
3. `compare_exact.py` at G4K and G16K, N=1.
4. Measurements:
   - py-spy `--gil`: background GIL time should drop by that stage's share;
   - nsys GPU-busy;
   - RULER G4K N=8, same-phase prefill at 120/180/300 s against r3;
   - MAB ms2 G4K N=8.

## 6. Risks

- **Hidden reliance on the GIL as a lock.** Mitigation: the store/repository
  mutex audit, and TSan.
- **Two sources of truth in stages 1–3.** Mitigation: version check on every
  snapshot, and a full rebuild on any non-append mutation (the same rule as the
  existing index cache).
- **Gain bounded by the engine thread's own Python.** vLLM's forward launch and
  scheduler hold the GIL ~17 s per 130 s. Option A removes the waits *for
  background threads* (~8% of wall at G4K N=8), not the engine's own host time.

## 7. Concurrency audit (stage 1 prerequisites, 2026-11-09 03:00)

| Native object | Today | Without the GIL |
|---|---|---|
| `KvMemRepository` (shared) | Lease/reservation ops lock-free (atomics); background reclaimer | Already thread-safe |
| `KvMemStore` (one per request, `RequestMemory._store`) | No mutex; the GIL serializes the main, refresh and scatter threads | Needs a per-store mutex taken by every bound entry point. `pick_prefill_pressure_blocks` mutates the retrieval-score frame (`set_repository_retrieval_scores`), so the pick is a writer |
| Prefill history `State` (`_prefill_state_native`) | Persistent values: `committed`/`committed_batch` return new `State`s; reads (`scores_for`, `values`) are const | Safe to read without the GIL while a Python reference keeps the `State` alive |
| Authority records (`VersionedRecords`) | Python dict; appends logged | Mirror into a native `RecordTable` under the GIL (O(new records)), read without it |
| Layer maps (`controller.maps`) | Python tuples of `(slot, generation, begin, count, position)` built by `commit_map` | Keep a native copy next to the Python one at commit; selection reads the native copy |

Lock order: request control mutex → store mutex. A thread holding either never
acquires the GIL. Python callers may block on the store mutex while holding the
GIL, which is safe because the GIL-free holder never waits for the GIL; critical
sections are short (one pick).

## 8. Distribution: still a vLLM plugin

- Entry points stay Python: `vllm.general_plugins` (`kvmem = "vllm_kvmem:register"`) and `vllm.endpoint_plugins`. vLLM loads them at startup.
- The native control plane goes into the existing pybind11 extension `vllm_kvmem._native` (`setup.py`, `Pybind11Extension`), shipped in the same wheel.
- `_native` links only `libstdc++` (no libtorch, no CUDA runtime). The wheel therefore depends on the CPython ABI (cp312), not on torch or CUDA versions.
- Constraints that keep it so:
  - the C++ control plane works on integers, record tables and CPU buffers only;
  - GPU work stays in Python (Triton/torch) and the existing CUDA JIT sources;
  - portable code (runtime SIMD detection with scalar fallback; thread counts capped at the core count);
  - Python reference implementations kept for equivalence tests and as a fallback when a native symbol is absent (`getattr(_native, ...)`).
- vLLM coupling is unchanged: it lives in the Python integration (scheduler patches, connector, runner, attention hooks); the C++ code uses no vLLM API.
- Release work:
  - per-platform/per-Python wheels (manylinux x86_64/aarch64, cp312+);
  - C++17 for source builds;
  - ASan/TSan builds in CI, because a native fault aborts the process instead of raising.

## 9. Implementation log

**Approach (06:20).** Bulk work runs natively without the GIL. Python objects the existing consumers need are created in short slices, releasing the GIL between slices, so the engine thread is never blocked for long.

| Date | Step | What | Result |
|---|---|---|---|
| 2026-11-09 | A1 | `_native.IdentityTable` (slot, generation, begin, count per indexed record, extended with the record index; mutex for GIL-free lookups). `prepare_commit(..., table=)` builds the same commit maps: lookups and checks without the GIL, row tuples created one layer at a time with a GIL yield between layers. Opt-in `KVMEM_PRESSURE_NATIVE_COMMIT_TABLE=1`, wired from `LayerwisePrefillPressure.prepare`. 18 equivalence tests (maps, tokens, errors and messages identical); 247 related tests pass. | N=8 + gate, function-traced: GPU idle 7.3% → 4.9%; idle inside troughs 5.9% → 4.1%; `prepare_commit` no longer blocks the engine (it was 8.5% of trough time, all in ≥5 ms holds) |

**Remaining trough composition after A1 + gate (N=8):**
- engine waits for the GIL 28% (refresh `assemble_layer_pinned` short holds 13.5%, publication chain ~10%);
- engine runs its own Python 39%;
- engine in native waits 33%.

**A1, throughput without a profiler (06:35).** N=8 + gate + A1: 6,355 tok/s, against a gate-only mean of 6,342 (4 runs) — no measurable gain. The function-traced share of `prepare_commit` was inflated by the tracing itself, which slows Python. Rule from now on: every step is judged by profiler-free 300 s throughput; traces only locate causes.

**What limits the GPU now.** After the gate, most of each per-step trough is the engine's own step-boundary work, not background GIL holds:
- vLLM `execute_model` Python;
- kvmem `ActionRuntime.prepare`;
- a synchronous D2H in `AsyncRefresh.step`, fixed by the existing exact opt-in `KVMEM_PROMISE_ASYNC_PAGE_CHECK` (being measured);
- the end-of-step `sample_tokens` sync.

Removing that work from the GPU's critical path needs the next step's host preparation to overlap the current step's GPU work: vLLM async scheduling (`max_concurrent_batches` 2). kvmem currently requires synchronous single-batch execution in capacity reservation, bounded execution, the MTP draft contract and the archive/window path. That is an architecture change to discuss with the user before any work.

**User direction (2026-11-09 06:50): no scheduling change.** Work the trough items in order (background GIL first, then kvmem's own syncs, then kvmem's engine-thread prep), each with an exact opt-in flag and a profiler-free throughput test.

| Date | Step | What | Result |
|---|---|---|---|
| 2026-11-09 | A2/A2b | Gate checkpoints in the publication chain: `_publish_claim` entry, each logical child, and between `prepare_accept` / append / commit (only the cursor's own lock is held there). | With the gate: N=8 6,342 → 6,410 tok/s; N=4 → 6,171 (base 6,134 / 5,779) |
| 2026-11-09 | item 3a | `KVMEM_NATIVE_ARM_BATCHED`: `NativeAttentionStats.arm` builds row lists in bulk; one `_clear_lse_layers` launch over a stacked LSE buffer instead of 17. | With A1 + async page check: +0.25–0.3% at N=8/N=4 (same-runtime repeats agree within 0.1–0.3%); N=1 token-identical 5/5 |
| 2026-11-09 | item 2b | `KVMEM_REFRESH_ASYNC_PAGE_CHECK`: the non-promise page-reservation check in `AsyncRefresh.step` (a blocking `.cpu().tolist()`, 14.3% of the N=8 trough in trace s1e) becomes a pinned non-blocking copy + event, verified at the start of `_assemble` before any page write. | Same runtime, 2 runs each: N=8 6,378 → **6,507 (+2.0%)**, N=4 6,090 → **6,266 (+2.9%)**; step gap per step 11.5 → 3.2 ms at N=8; N=1 token-identical 5/5. Kept. |
| 2026-11-09 | item 3b | `KVMEM_NATIVE_PREFILL_PLAN`: `_native.frame_window_segments` and `_native.logical_plan_check`, exact ports of `window_segments` over frozen records and `LogicalScorePlan`'s validation; values outside ±2^61 or non-exact ints fall back to Python. | 1.71 → 0.43 ms per owner per step (microbenchmark); 74 equivalence tests. Same runtime, 2 runs each: N=8 6,504 → **6,521 (+0.26%)**, N=4 6,269 → 6,270; N=1 fully exact (305 pressure selections, 80 Query layers, 5 cases). Kept. |
| 2026-11-09 | item 1b | `KVMEM_GIL_GATE_CHUNKS`: gate checkpoints between staged chunks of one layer's refresh assembly, so a layer that started before the gate closed stops within one chunk. Scheduling only. | N=8 −0.5% vs npp (6,489 vs 6,521), N=4 ±0. Dropped (off). |
| 2026-11-09 | GPU time | `KVMEM_NATIVE_PREFILL_STATIC_QUAD` (existing 11-07 variant). Kernel split: FlashInfer prefill attention is ~55% of GPU time; kvmem's in-FA2 stats overlay is 6.1% of all kernel time. Quad batches 4 fragments' stores (bit-exact by construction; 13 GPU bitwise tests). The first serving A/B was confounded by a different FlashInfer GEMM autotune cache (config hash change); fixed by pinning `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` in every arm. | Pinned: N=8 6,518 → **6,663 (+2.2%)**, N=4 6,302 → **6,424 (+1.9%)**; N=1 fully exact (311 pressure selections, 80 Query layers). Kept. |

**GPU idle after rpc + npp (traces s5/s6):** 1.5% of wall at N=4 and N=8, all outside the forward. The rest of the remaining time is GPU work, not idle.
| 2026-11-09 | publish | `KVMEM_NATIVE_PUBLISH_SELECTION`: `_native.publish_selection(store, plan)` replaces `RequestMemory._publish_selection`'s per-block pybind loop. The plan's `stage_out` / `remaps` vectors were converted to Python objects, then 2–3 pybind calls ran per block. Same store calls in the same order; holds the GIL. 17 equivalence tests. | G16K N=8 +0.2% (noise); G4K N=1 4,888 → **4,931 (+0.9%)**; N=1 fully exact at G4K and G16K. Kept. |

**G16K validation (all 7 flags vs none, pinned autotune):** N=8 6,204 → 6,353 (+2.4%); N=1 5,535 → 5,713 (+3.2%), fully exact. Trace s7: G16K idle is 4.3% of wall, mostly the engine waiting for the GIL held by the refresh thread (selection prepare 68 ms, assembly 107 ms, `force_all` 37 ms, reader 9 ms of GIL per refresh). That is the stage-1 work in §4.
