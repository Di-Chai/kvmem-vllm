# Measurement contract

`scripts/qualify_engine.py --stream-metrics --output-tokens N` records raw SSE
events and monotonic arrival times, using returned token IDs rather than text
lengths. Final API usage must agree with emitted IDs, and the stream must contain
a finish reason and `[DONE]`. Failed/incomplete responses do not become successful
throughput observations. Output caps are reported as censored rather than natural
completion. The native streaming adapter is qualified by `engine_deepswe_effect_bounded_03`: emitted IDs agree with final usage for the long C1 and all four short requests.

For one fixed request group:

- Cold service throughput is total output tokens divided by the latest completion
  minus earliest submission. It includes prefill, queueing and refresh interference.
- Aggregate arrival decode throughput is the sum of each request's output count
  minus one (floored at zero), divided by the latest last-token arrival minus the
  earliest first-token arrival. Overlapping request durations are never summed.
- Per-request TTFT, service time, first-to-last output rate, p95/max inter-burst
  gaps, finish reason and actual output count are retained.

Speculation produces token bursts. These are HTTP observations, not individual
GPU token completion times. A single burst has no measurable first-to-last decode
interval and must not yield an infinite rate. Cold service and TTFT must accompany
the arrival decode metric; shifting prefill before first output can improve that
window without improving request completion time.

Frozen native DeepSWE inputs are supplied with `--frozen-input FILE
--frozen-sha256 HASH`. Exact input token IDs are preserved, with no synthetic
padding or chat-template rerendering. The original query interval is retained in
provenance; current semantic prefill uses each scheduled query chunk, and does not
yet implement the separate final-query override from the old controller.
By default only the single request uses the frozen long input and the following
C4 prompts remain short. For explicit long C4, supply exactly four
`--concurrent-frozen-input PATH SHA256` pairs. Each file is validated independently
before launching the server and its provenance is saved. The same C1/C4 token IDs
must be supplied to matched native and bounded runs. This long-C4 option still
needs native GPU qualification.

Do not interpret diagnostic timings as optimized serving: archive qualification
performs synchronized complete record readback, and replay diagnostics may repeat
forwards solely for byte checks. Production-quality resource retirement and warmed
execution must be integrated before the matched C1/C4 performance gate. Use the
same token IDs, precision, K/G, record granularity, query policy and sampling across
the intended comparison, preserve natural EOS, and repeat independently. A frozen
conversation continuation does not execute tools or establish SWE task success.

Use `--concurrent-only` with four frozen inputs to avoid rerunning an already
qualified single request. `--prefill-request-token-cap` maps to vLLM's native
`long_prefill_token_threshold`; a cap of 256 with a 1024-token batch permits four
aligned prefill chunks in a native batch. This changes per-request query chunking,
so matched controls must use the same cap. It is a scheduling setting, not evidence
of speedup: report actual worker batch membership and both TTFT and makespan.


`--bounded-dflash` selects the experimental synchronous compressed DFlash2 path
and implies `--dflash --bounded-target`; it is not yet GPU-qualified. It preserves
native accepted/rejected accounting, applies the same frozen input and natural EOS
contract, and rejects refresh evidence that scores unverified draft rows. Keep
its results separate from target-only measurements and from identity-replay tests.


For bounded serving performance runs, `--skip-archive-readback` disables the
**diagnostic** full-record checksum readback at request retirement. It still
waits for accepted capture and archive writes, requires every complete record to
be published, and checks final resource cleanup. Native reports explicitly label
this `archive_verification: publication_only` and the harness records
`archive_checksum_qualified: false`; zero checksums must never be reported as a
byte-correctness pass. The default remains full checksum verification. Compare
performance runs using the same retirement mode and retain separate checksum
qualification. This option does not implement asynchronous/background retirement:
publication draining and record reclamation still happen synchronously.

New harness runs sample owned GPU-process memory and API/GPU-process RSS every
2 seconds into `resources.jsonl`. The summary reports **sampled** maxima over
server lifetime, including startup; it cannot detect every transient peak.
Host RSS is summed and can double-count shared pages. Ownership is checked by the
server process group. Foreign GPU processes, unknown ownership or sampling errors
make `exclusive_gpu_observed` false; do not use such a run to establish an
exclusive-GPU performance result. The prior `memory` field remains an end snapshot.
A manually attached `resources_partial.jsonl` monitor covers only its recorded
interval and cannot establish a whole-run peak.

`--cancel-during-refresh` requires a frozen C1 and bounded serving. It submits an
explicit native HTTP request identity, waits for an atomic worker trace of an
actual compressed refresh, and shuts down only that owned client socket. Four
subsequent requests exercise recovery. Acceptance requires the disconnect time
to fall within the completed native refresh interval, a native `FINISHED_ABORTED`
status, a finalized incomplete-prefill authority, and the complete native GPU
pool (apart from its null page) plus all lifetime reservations to be released.
The interrupted request contributes no invented completion count or throughput.
A localhost socket test qualifies the client's interruption mechanism; native GPU
cancellation is still pending and must be established separately.

`--kv-cache-memory-mib` sets native vLLM's GPU KV pool size without changing
weight/KV precision or K/G. With bounded `--concurrent-only` and four frozen
histories, `--require-capacity-pressure` requires actual lifetime-reservation
rejections and complete physical-page/credit cleanup after every request finishes.
This is an admission-pressure test: it permits fewer than four simultaneously
executing requests and must not be presented as full GPU C4 throughput. The
ordinary long-C4 gate still requires an actual four-request compressed batch.
A pool unable to fit even one worst-case KVMem window is rejected at startup;
do not weaken that check to make a pressure run start.

Packed refresh reports now retain summed archive-read statistics, including
logical/cache-hit/native I/O bytes, queue wait, buffer allocation, cache copying,
native read-call wall time and cache insertion. These host timings are wall time
at the named boundaries, not isolated CPU-cycle or physical-disk measurements.
They complement overall assembly and host-preparation durations; do not infer
a disk bottleneck from total host-preparation time alone.
