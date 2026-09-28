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
