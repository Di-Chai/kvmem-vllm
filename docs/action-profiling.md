# Action attention diagnostics

This is an opt-in diagnostic mode. It preserves the existing Action/prefill
scoring, sample density, parser commits, CUDA streams and publication fences.
It does not enable native Action metadata or replace the attention kernel.

Add these keys to the existing serve profile's `kvmem` object, leaving the
qualified model and scheduling settings unchanged:

```json
{
  "action_attention": true,
  "action_profiling": true,
  "action_profile_interval": 1,
  "action_profile_max_events": 256
}
```

`action_profiling` defaults to false. A sampled batch is selected every
`action_profile_interval` public batches. An interval of 8 is useful for a
longer diagnostic run with less instrumentation overhead. The event limit is
the number of **pairs**, including both open scopes and pending completions.
Full capacity drops only GPU timing samples and increments `gpu_event_drops`;
all original attention work and host timing still run. Graph capture skips
diagnostics and increments `graph_capture_skips`.

The normal serving launcher passes these keys through `RuntimeOptions` to the
worker. It writes `server/capture/action_profile.jsonl` beneath the run output.
Use a separate diagnostic run: event recording and JSON serialization add
overhead, so its throughput should not replace an uninstrumented benchmark.
Only sampled batches and explicit round-boundary flushes serialize the full
cumulative summary. Unsampled batches only poll existing events; they do not
write a repeated summary. Event completions can still write rows on those
batches. Summary serialization itself remains additional host/file-system work.

Each `batch` row contains host wall/thread CPU spans, a batch ID and a mapping
of request IDs to `prefill` or `output`. Request-owned stages additionally carry
the request ID and kind; layer-owned stages carry the layer name. The stages
include:

| Stages | Interpretation |
| --- | --- |
| `publication.batch_drain` / `publication.owner_drain` | Existing shared batch fence versus the later request-owned fence; both keep the original `wait=True` behavior. |
| `metadata.pack`, `metadata.pinned_allocate`, `metadata.d2h`, `metadata.existing_fence`, `metadata.tolist` | Exact coordinate snapshot, allocation/copy, original completion wait and conversion. |
| `prefill.rows`, `output.begin_response`, `output.parser_prepare` | Prefill sampling or generated-token scope preparation. |
| `metadata.window_segments` | CPU construction of historical page identities. |
| `metadata.{starts,counts,rows,positions}_h2d` | Each original `torch.tensor(..., device=...)`: Python-list conversion, allocation and upload combined. Splitting these operations would change the measured implementation. |
| `score.layer_total`, `score.q_gather`, `score.trig_gather`, `score.rope`, `score.q_codec` | Sampled Q and RoPE/codec preparation, with a per-layer inclusive scoring interval. |
| `score.logits_allocate`, `score.logits`, `score.softmax`, `score.mass_allocate`, `score.segment_mass` | Existing paged QK kernel, FP32 softmax and page reduction; shape, grid and allocation sizes are included. |
| `score.layer_add`, `score.layer_mean`, `mass.pinned_allocate`, `mass.d2h` | Layer aggregation and original mass staging to CPU. |

Completed CUDA event pairs are emitted later as `gpu_stage` rows, joined by
`batch_id` and `scope_id`. Completion is checked only with `event.query()`;
diagnostics never call `synchronize()`. Summary rows aggregate spans by kind,
layer and stage and report outstanding event pairs. Outstanding intervals at
termination remain unmeasured, rather than forcing a final wait.

Host spans and CUDA-event intervals are inclusive; nested stages overlap and
must not be summed as exclusive runtime. A CUDA event interval can include
stream scheduling or host enqueue gaps, so it is not isolated kernel-active
time. Neither metric alone measures GPU idle. Transfer byte counts describe
requested tensor payload, not observed memcpy activity. Compare stage spans,
wall versus thread CPU time, event drop rates and external GPU sampling before
attributing a bottleneck.

CPU tests cover bounded event queues, graph-capture exclusion, option/serve
round trips, preserved publication fences, scoring dispatch/reductions, and
identical mixed prefill/output packets and accepted feedback. They substitute
GPU calls and do not qualify GPU timing accuracy or model performance.

For offline analysis, run `scripts/analyze_action_capture.py` with the JSONL
path and optionally `--start`, `--end`, `--time-basis scope|batch`, and `--output`.
Window bounds use the original monotonic `started_at` timestamps; they are
start-inclusive and end-exclusive. A GPU interval reclaimed after the chosen
window still belongs to its original scope/batch, not its serialization time.
The analyzer reports inclusive aggregates by prefill/output, layer and stage.
The final summary's dropped/pending counters remain explicitly cumulative for
the whole file; they are not mislabeled as deltas within the chosen window.
