# Action attention in KVMem + vLLM

Action attention records which historical pages accepted generated tokens attend
to, then combines that evidence with MeanK Query relevance at a later Query
selection. It is separate from Query relevance history. A cold, single-turn NIAH
request has no previous output attention at its first Query selection. Optional
prefill attention now supplies input evidence for that cold selection and for
pressure selection while ingesting the history.

## Configuration

Add these fields to a native serving profile's `kvmem` object:

```json
{
  "action_attention": true,
  "action_scope": "tool-call",
  "action_layers": "spread3",
  "action_interval": 16,
  "action_fraction": 0.25,
  "action_combine": "quota",
  "action_decay": 0.5
}
```

`action_attention` defaults to false. `tool-call` samples only tokens inside an
accepted `<tool_call>…</tool_call>` region outside thinking. `non-thinking`
samples other non-thinking output too, and is suitable for ordinary answer
workloads. The interval counts eligible consumed tokens, including across MTP
verifier batches. Marker tokens do not count. These semantics use the unchanged
qw3 C++ parser.

The authenticated elastic launcher accepts the same action fields at the top
level. `examples/qwen-nvfp4-action-attention-elastic.json` enables tool-call
attention with CPU-only historical KV (`archive_backend="cpu"`).

`spread3` uses the first, middle and last target full-attention layers with equal
weight. `last`, `all` and comma-separated standard layer numbers are supported.
Recurrent GDN layers and the draft MTP layer are excluded. `quota` uses the
original qw3 selection quota; `score` uses its normalized positive-L1 score
fusion. Sink/recent/mandatory rules take precedence. Empty action history falls
back to Query selection. Decay is applied once per completed response with
eligible samples, rather than once per speculative step or tool wait.

Owned prompt sessions must be enabled. Configure sufficient
`session_checkpoint_bytes` to retain a checkpoint; a one-byte checkpoint quota
cannot preserve feedback across turns. Terminal M checkpoints retain completed
output scores. Restoring an earlier P checkpoint restores its scores and drops
feedback from an edited/abandoned suffix. If a new Query's aligned boundary
requires P replay but the entire previous M token prefix is unchanged, its
completed scores are retained after exact prefix verification. An intermediate
P snapshot before that M boundary still stores its original score state, so a
later edit cannot inherit future feedback. Historical keys are logical page offsets,
not reusable physical GPU block IDs. Score snapshots count toward checkpoint
payload capacity; they retain no GPU tensor.

## Execution and limits

The sampler uses captured target Q, the model's dense RoPE, the native decode Q
dtype/scale and actual paged FP8 K. It computes causal token attention, averages
heads and selected layers, and sums mass by historical page. This is not MeanK
attention or an approximate page-key lookup. Only sampled rows are evaluated.
The current adapter qualifies full causal FlashInfer Qwen attention without
sliding windows, soft caps or attention sinks; unsupported layouts fail rather
than silently recording a different score.

The ordinary model, MTP and CUDA Graph execution remain intact. The additional
sampler runs outside the graph. A small token/position readback prepares the
original CPU scope parser; page masses are downloaded asynchronously and
committed at the existing next-scheduler acceptance boundary. Native rejection
counts and terminal EOS/cancellation truncation both apply before evidence is
added. Private guided Query forwards and public Query replay (including the
assistant template suffix) are excluded. Prompt prefill has a separate optional
mass/exposure accumulator; it never goes through the output scope parser. The
rejected draft suffix cannot change the parser state.

The sampler reproduces attention from the executed Q/K operands using FP32
reductions. It is not a bitwise copy of FlashInfer's internal softmax reduction.
Scratch is bounded by the active window, query heads and sampled verifier rows.
CPU readback and sampled GPU work have a measurable cost; enabling the feature
does not claim zero overhead.

The example reserves additional transient GPU memory with
`gpu_memory_utilization=0.72`. This changes native KV pool preallocation, not the
32K selection or 16K generation budget. On the 96 GB qualification GPU, 0.82
passed the two-turn test but a subsequent eight-request long-output batch ran
out of memory during compiled Query replay. A passed short test is therefore
insufficient evidence for that more aggressive allocation. Guided Q, public Q
captures, raw-K caches and replay activations also require memory outside the
native KV pool.

The complete input path is opt-in with `prefill_attention=true` and
`action_pressure=true`, alongside `action_attention=true`. Set
`prefill_attention_interval=1024`, `prefill_attention_stride=32`,
`prefill_attention_half_life=8192` and the serving profile's
`prefill_chunk_tokens=1024` to match the qw3 multilayer experiment. Each observed
full chunk contributes 32 end-anchored causal rows. Mass and page exposure decay
by accepted input distance; mass/exposure gives the page's input score. Only a
fully accepted chunk commits. At most 64 sampled rows per chunk are supported.

Pressure uses the unchanged qw3 C++ resident-page selector: mandatory content
wins, positive attention ranks resident pages, and unfilled capacity takes newer
resident pages. Semantic selection calls qw3's existing quota/score selector with
prefill scores when no usable output feedback exists. The 256K reference uses
36,864 selection tokens, 16,384 generation tokens, 1,312 sink tokens, zero recent
tokens and 0.5 attention quota. Enabling only output attention still does not
reproduce this experiment; see [the earlier audit](niah256k-attention-20261014.md).

The elastic launcher accepts these fields at the top level, including
`prefill_chunk_tokens`, `block_tokens`, `sink_blocks` and `recent_blocks`.
[The complete-attention gateway example](../examples/qwen-nvfp4-prefill-attention-elastic.json)
uses 32-token pages and the reference selection parameters. It keeps 72 GiB for
CPU KV payload to leave room for staging and process memory on a 128 GiB host;
eight clients do not imply eight simultaneously resident 256K histories. The
frozen benchmark uses a 100 GiB native pool with a separate 72 GiB ingress gate.
Existing profiles retain their default page size, keep bands and chunk behavior.

For independent questions sharing an identical input, native profiles can set
`session_freeze_prompt=true`. It preserves the P checkpoint's pre-selection
window, model state and input scores. This mode requires explicit planning,
P-only checkpoints and disabled Query history. It deliberately discards prior
answers' output feedback on restore. Ordinary multi-turn sessions retain their
existing P/M policy. This option is used by `scripts/benchmark_memory.py`; input
token hashes and sampling settings are checked against the frozen qw3 cohort.

MeanK Query scoring uses the original fused CUDA kernel for up to 8,192 eligible
pages. Larger histories use qw3's original tiled logits, global normalization
and accumulation kernels, supporting 32,768 pages with approximately 64 MiB of
temporary logits storage. Batching does not divide the softmax over history into
independent pieces. The large-history path is validated at the page boundary,
at maximum capacity, and with the Qwen head geometry on a non-default CUDA stream.

`capture/action_config.json` records actual layer IDs and native Q dtypes.
`capture/action_attention.jsonl` records accepted sample counts and committed
score statistics. Query refresh events record whether action feedback was
enabled and how many positive historical pages were available. With input
attention enabled, `capture/prefill_attention.jsonl` records accepted chunks and
rows; refresh events include the attention source and cumulative input counts.

## Validation

The later complete-input-path run is recorded in
[the prefill/pressure report](prefill-attention-20261014.md): all 60 frozen 256K
cases completed at 97.08% mean numeric coverage and 53/60 exact numeric sets,
versus 68.33% / 30/60 for the previous output-only adapter. The qw3 reference is
98.75% / 57/60; this result does not claim bitwise or full quality parity.

CPU tests cover the original scope parser, speculative rollback, terminal
truncation, per-response decay, session isolation, checkpoint rewind, and actual
native quota/score selection. GPU tests compare sampled paged FP8 attention to a
dense causal reference with BF16/FP8 Q, non-contiguous physical pages, partial
historical segments, grouped query heads and speculative rows.

`scripts/qualify_action_attention.py` performs independent concurrent sessions,
retains exact generated token prefixes, and asserts that the next Query uses
restored positive action history.

On 2026-10-14 the real 27B NVFP4 model passed eight independent sessions and two
turns each with FP8 KV, native MTP(k=4), compilation/CUDA Graph, CPU historical
KV, guided Query and Query relevance history enabled together. The first turn
used the frozen 50,006-token input and generated 257 tokens; the continuation
generated 129 tokens. Each session committed 16 then 8 eligible samples, retained
1,094 positive historical pages at the second Query, and advanced its action
history from one to two rounds. This also exercises a Query beginning in the
partially filled last page of the previous response.

A separate real-model `tool-call` smoke test supplied a function schema and
generated a complete 127-token tool invocation from a 397-token prompt. The
original parser predicted eight eligible samples; runtime committed eight,
with positive mass on 16 historical pages. This validates scope capture, not
external tool execution. That test used
`artifacts/action-attention-20261014/tool-call-profile.json`; subsequent quality
campaigns use their own profiles on the same local port.

The earlier output-path qualification passed 688 host tests (77 skipped) and
four CUDA numerical tests. Its subsequent CPU archive launcher forwarding
change separately passed all 90 launcher tests. The newer complete-input-path
validation is recorded in the linked prefill/pressure report.

The fixed-output comparison used the same 50,006-token input, eight concurrent
requests, 4,096 output tokens per request, K=32,768, G=16,384 and a 4 GiB session
checkpoint allowance. The profiles differ only in `action_attention`. The
enabled scope was `non-thinking`, interval 16, layers 3/31/63, quota fraction
0.25, decay 0.5. Each arm completed all eight requests and returned GPU request
ownership to zero; output-token hash multisets match.

| Metric | Action off | Action on |
| --- | ---: | ---: |
| Shared steady decode, output token/s | 849.29 | 726.35 |
| Whole-batch output token/s | 204.85 | 205.20 |
| TTFT p95, seconds | 118.03 | 112.01 |
| Shared steady window, seconds | 36.58 | 42.79 |
| Native eight-owner window coverage | 98.89% | 99.26% |

Steady throughput decreased 14.48% in this single pair. This is not a repeated
statistical estimate. The enabled engine had just run cross-turn qualification;
the disabled engine was fresh, so whole-batch/TTFT numbers also reflect warmup
and allocation history. The similar whole-batch rates do not imply zero sampling
cost. The implementation adds a synchronized token/position readback, sampled
QK/softmax/page reduction and host score maintenance; this comparison does not
separately attribute their costs. Real-model graph counters confirm compilation
and CUDA Graph remained in use.

Compact machine-readable evidence is in
[`benchmarks/action-attention-20261014.json`](benchmarks/action-attention-20261014.json).
Raw requests, token-arrival timelines, native ownership observations, profiles and
source hashes are in `artifacts/action-attention-20261014/`.

## Why the original profile blocks other requests during refresh

Refresh is request-owned logically, but the default `bounded_runtime` path performs it
synchronously inside the model forward wrapper. It drains capture/scatter,
performs private guided forwards, selects pages, prepares and uploads KV, waits
for assembly, and then replays the public batch. Other ready requests in that
batch therefore wait too. Chunked prefill is enabled in the NIAH profile, but
does not schedule around this wrapper.

Per-request scheduling should remove only the refreshing request from the ready
set, preserve its recurrent state and page leases, prepare history in background,
and return it after its own upload fence completes. Other ready requests can
continue decoding. GPU probe/replay work must also be separately scheduled;
putting just the CPU gather on a thread does not remove the shared forward or
stream barriers. Shared GPU and transfer bandwidth still need admission and
priority control. The subsequent opt-in `async_refresh=true` implementation adds
this request-local state machine; see [request-local refresh](async-refresh-20261014.md)
for its ownership contract, qualification and remaining shared-resource limits.
The measurements in this Action attention report used the original synchronous
refresh path.

For the completed 180-sample NIAH run, 97 of 104 samples below 20% GPU utilization
fell in refresh windows; outside those windows the mean utilization was 96.3%.
These are 10-second `nvidia-smi` point samples aligned with refresh timestamps,
not a kernel trace. Refresh windows include GPU work and cannot all be counted
as idle time. Summed assembly time was 477.2 seconds, of which 434.4 seconds was
host preparation. Raw alignment and aggregates are retained in
`artifacts/action-attention-20261014/niah-utilization.json`.


## Optional maintenance, download, scorer and propagation optimizations

Four independent, default-off switches are available: `action_native_decode_commit`,
`action_download_pool`, `action_fused_scoring` and `graph_native_propagation`. The
reference paths remain available. Decode staging preserves accepted-row and
transaction order; the bounded pool retains buffers through every CPU alias and
DMA fence; the fused bypass scorer keeps full-context causal normalization;
native V2 propagation caches owner/version-bound indexes and reapplies each mask.

These switches are separate from native attention QK reuse. See the
[qualification and independent cost report](attention-overhead-20261020.md) for
limits, measured scope, state checks and full C8 comparisons.
