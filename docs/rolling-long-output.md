# Single-request rolling output

`gen_budget` is physical decode headroom, not necessarily a response length.
With `request_planning` and `rolling_active_window` enabled, native token-ID
completion requests may set `max_tokens` above G. Non-rolling backends retain
the previous requirement that output fits within G.

For K=98304, G=16384, block=32, the existing scheduler visits an accepted-prefix
pressure boundary before the K+G allocation fills. It suspends that owner,
selects at most K tokens, atomically installs the selected KV view, then resumes
the same native request. The output counter, sampler, reasoning budget and
tool frame continue across this operation. No extra user prompt, synthetic
assistant turn or per-G native request is inserted. Other schedulable owners
remain eligible while the refresh completes.

## Configuration

The backend profile still uses:

```json
{"kvmem": {"select_budget": 98304, "gen_budget": 16384,
           "rolling_active_window": true}}
```

These are fields within the existing qualified profile, not a complete launch
profile. All prerequisite asynchronous refresh/restore and history coordination
settings remain required.

Both `benchmarks/deepswe/claude_bridge.py` and `scripts/dsh_bridge.py` accept
`--max-output-tokens` independently of `--gen-budget`. For example,
`--gen-budget 16384 --max-output-tokens 131072` permits one native response up
to 128K while keeping G=16K. The DSH bridge requires `--max-output-segments 1`
(the default) for this mode. Its older segmented mode remains a separate option.

The caller must also request a larger output limit. A client that sends
`max_tokens=16384` still stops at 16384; the server does not silently increase
an explicitly requested limit. A future DeepSWE launch must configure its
isolated client's output cap along with the bridge and freeze the new values
in that run's manifest. Historical DeepSWE queue profiles and results keep
their original 16384 output cap. Omitting the new bridge option preserves that
previous default.

Generation stops at EOS, the requested stop condition, cancellation, or the
independent output cap. Prompt plus requested output must fit the logical
model limit. KV re-selection does not make logical positions or the full
accepted-token journal infinite. The separate CPU history/GC policy still
governs archived KV. Non-rolling backends reject requests above G before
they enter the scheduler.

Tool parsing uses the entire response. Incomplete tool arguments at an output
limit are not executed. A complete tool frame can therefore span several KV
refreshes without being cut into separate model requests.

## Validation

`scripts/qualify_rolling_long_output.py` prepares an isolated frozen component
run from the completed DeepSWE runtime. It changes only HTTP output admission
in model source, retains K96K/G16K and MTP4, and enables the existing byte oracle.
Two owners each receive 100352 fixed input tokens and generate 36864 fixed
accepted tokens in one native request, followed by an exact-prefix resume,
an EOS check, and session release. The audit requires multiple zero-replay
refreshes per long request, including after G output, bounded selected KV,
exact target/MTP KV and recurrent state, response ID continuity, usage/accepted
ID agreement, and owned-process/GPU cleanup.

This constrained-output run verifies mechanics. It does not measure Agent
quality, unconstrained generation equivalence, eight-way throughput, or a
10M-token continuous output. Offline bridge tests cover a complete tool
argument longer than 32K, truncation without tool execution, exact accepted
IDs, sampling fields and legacy defaults.

The 2026-10-26 run passed: both long requests returned exactly 36864 accepted
tokens; all four rolling refreshes selected 98304 tokens and passed exact
target/MTP KV and recurrent-state checks. Both exact-prefix resumes and EOS
checks passed. All session payloads, active owners and owned GPU processes
were released. The related offline checks passed 217 tests.
The frozen inputs, responses, source hashes and
`artifacts/rolling-long-output-20261026-r1/final-review.json`
are retained in the experiment directory.

## Per-layer Mean-K and rolling decode

The opt-in `--layerwise-rolling-decode` path retains independent target-layer
blocks after the final causal Mean-K Query. During output, each target layer
accumulates its own attention mass keyed by original token coordinates.
Sampling uses the existing `action_interval` and includes both thinking and
ordinary output. The legacy non-thinking action/graph parser remains unchanged;
rolling pressure uses a separate accepted-output history. Rejected MTP rows,
private Query probes and Query replay cannot update that history.

At a decode pressure boundary, each layer ranks only its current blocks plus
the common newly accepted suffix. Sink and explicit mandatory blocks remain
required. A missing layer score is an error, not a silent shared-score or
recency fallback. All layers must retain the same physical token extent and
partial-block geometry. The existing asynchronous transaction rebuilds each
target layer once, rebuilds MTP using layer zero's selected coordinates, waits
for the transfer fence, and resumes the same native response. The accepted
output count, sampling state, thinking limit and response ID continue.

Enable this together with optimized sequential Query, per-layer prefill
pressure, native batches and single-pass assembly:

```text
--layerwise-query-diagnostic sequential --layerwise-query-optimized
--layerwise-query-native-batches --layerwise-prefill-pressure
--layerwise-pressure-single-assembly --layerwise-rolling-decode
```

In the existing profile set `kvmem.rolling_active_window=true` and, for example,
`kvmem.gen_budget=4096`, keeping the desired Active budget unchanged. Set
`query_replay_budget` no higher than 4096, as required by the native Query
reservation; this does not truncate an input or change `max_tokens`.
An atomic public Query/replay span that does not fit that reservation is rejected
explicitly; use a larger G for such requests.
For a controlled comparison, `request_planning_batch_tokens` can preserve the
original native batch ceiling independently of this per-request Query cap
(16424 for the existing eight-owner Memory profile). Without that explicit
setting, the launcher retains its historical replay-plus-decode-peers sizing.
Asynchronous refresh, accepted attention, request planning and retained CPU
history remain required. The initial version supports compact positions and
the MTP draft contract. It does not add post-Query session checkpoint/continuation
support to the one-turn layerwise implementation.

`scripts/qualify_layerwise_rolling.py` freezes the completed local NVFP4 runtime
and two original full LongMemEval inputs with K16K/G4K. Two concurrent constrained
9216-token responses must cross multiple verified refreshes, including during
thinking. The independent byte oracle checks each layer's selected archive
records, MTP, and unchanged recurrent/acceptance state. These constrained outputs
are mechanical tests and must not be counted as benchmark accuracy or performance.

The 2026-11-04 frozen qualification passed at
`artifacts/layerwise-rolling-g4-20261104-r5/final-review.json`: two original
107366/108368-token inputs each produced 9216 accepted output tokens in one
continuous response. All four decode refreshes passed the independent per-layer
KV/MTP and recurrent-state byte checks; two occurred during thinking and two
after it. Each layer retained at most 16384 charged Active tokens. The frozen
runtime also passed 105 CPU boundary/regression checks, and all owned GPU
processes exited. This qualifies the rolling mechanics, not the accuracy or
throughput effect of reducing G; those still require ordinary sampled outputs.

The follow-up `artifacts/layerwise-rolling-g4-20261104-r7/final-review.json`
repeated both complete long-output requests with the original eight-owner
capacity and native 16424-token batch ceiling. All four byte checks and 303
CPU checks passed. Per-owner admission remains unchanged; the planned batch
can combine several owners' windows without treating their sum as one owner's G.

For the original LongMemEval-S cohort, `scripts/run_layerwise_rolling_memory.py`
prepares a fresh K16K/G4K campaign from the completed K16K/G16K reference and
the passed rolling runtime. It preserves all 102 input identities, eight-owner
concurrency, CPU admission reservations, sampling, thinking budget and output
caps, and disables the expensive byte oracle. `infer` runs local inference only;
`grade` requires an explicit batch-bound external-scoring authorization record
and reuses the frozen official rubric. Neither mode resumes paused MAB work or
starts a scheduled monitor. Comparison reports bind scores to the same cases
and use total inference wall time excluding model load and grading. A smaller G
also changes prefill pressure cadence and final Query headroom, so this is an
end-to-end budget comparison, not an isolated decode-only change.
