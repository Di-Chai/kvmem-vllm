# DSH test service on main, 2026-10-20

The service uses main `8560331`, local Query-lifetime/guided/DFlash changes and
the local `scripts/dsh_bridge.py` adapter.
The user initially requested recent=0, then explicitly chose DFlash2 with
recent=2048 because the current native draft contract requires its recent band.

| Setting | Value |
| --- | --- |
| Target / target KV | NVFP4 / FP8 |
| Speculation / execution | DFlash2 k7 / FULL CUDA Graphs |
| GPU memory utilization | 0.80 |
| K / G | 36,864 / 16,384 tokens |
| Sink / configured recent | 1,024 / 2,048 tokens |
| Extra mandatory parts / explicit spans | `none` / `[]` |
| Prefill pressure selection | Cumulative attention, without graph propagation |
| Query selection | V2 multi graph propagation, alpha .2, C++ backend |
| Guided Query trial / relevance history | Up to 512 private tokens / disabled |
| Same-Query tool continuation | Shared rolling active window; real user source metadata |
| Prefill chunk | 1,024 tokens |
| CPU history | 100,000,000,000 bytes, CPU-only |
| Four overhead optimizations | All enabled |
| Native QK reuse / prefix caching | Off / disabled |

The attention pressure picker has no fixed recent transport band; Query selection
uses the configured band. Semantic system/root/Query/policy/tool pins are disabled
in the adapter. Full live Query replay is still executed as required by the
existing runtime. The bundled default profile is unchanged. The current
deployment uses `artifacts/dsh-query-epoch-20261020-r2/profile.json`; the prior
guided and unguided profiles remain preserved. See
[Query lifetime correction](dsh-query-epoch-20261020.md) for the current code,
qw3 comparison, real-model gates and bounded DSH observations. The
[longer retest](dsh-query-epoch-retest-20261020.md) adds the near-full terminal
prefill fix and recovered-service checks. The subsequent
[177-step full run](dsh-query-epoch-longtest-20261020.md) confirms long stability
with an identical prefix through the old failure; agent repair quality still
misses gameplay defects. Sections below
retain the earlier test history and its original evidence paths.

## Running endpoints

- DSH UI: `http://127.0.0.1:3080`.
- New-chat DSH provider: `kvmem-main-dflash2-continuation`; model: `kvmem-dsh-dflash2`.
- Streaming/continuation chat adapter: `http://127.0.0.1:18385/v1`.
- Game preview, through the same DSH origin: `/kvmem-csgo/`.
- The earlier streaming provider and `http://127.0.0.1:18382/v1` adapter remain
  available for existing conversations.
- Existing sessions retain provider `kvmem-main-dflash2` and its original
  `http://127.0.0.1:18381/v1` adapter so an active agent task is not interrupted.
- Native backend: `http://127.0.0.1:18846`.
- Current process IDs, commands and logs: `artifacts/dsh-main-20261020-r1/service.json`;
  continuation/preview processes: `artifacts/dsh-csgo-20261020-r1/service.json`.

The provider and new-chat default were installed through DSH's live settings API.
The previous settings were backed up at the path in `dsh-connection.json`.
Existing providers and conversations were retained. New chats select this model;
existing chats can select the model labelled `DFlash2 自动续写` from the model picker.
The model server and earlier adapters remain running. The streaming adapter has a
separate owner namespace; switching an existing chat preserves its transcript
but starts a new native history and reconstructs attention/graph state by prefill.

The adapter uses the already qualified Qwen protocol renderer and accepted-token
planner from `benchmarks/deepswe/claude_bridge.py`. Its native protocol library path
is recorded in `service.json`. OpenAI tool calls/results are converted to that
renderer, keeping exact accepted output IDs on verified continuations. The
adapter clears extra mandatory pins while preserving Query and locality metadata.

DSH sends a stable `prompt_cache_key` through its supported long-cache transport
option. This identifies a KVMem owner; it does not enable native prefix caching.
The owner also binds the system/tool frame, isolating automatic title requests
which share the same DSH conversation ID. Owners are bounded to 32. Each native
request remains within G; the continuation adapter permits up to four segments
and 65536 output tokens, subject to the 262144-token logical context. Unsupported images are
rejected. Native completion SSE events now carry actual accepted token IDs to the
adapter. Reasoning and text prefixes are forwarded during generation, retaining
split control tags and removable trailing newlines until their meaning is known.
Canonical tool function headers are forwarded during generation; their arguments
wait for the authoritative schema-aware parser. Noncanonical recovery forms can
therefore have additional visible-output buffering. Truncated announced tools,
missing terminal records, usage/ID mismatches and disconnects do not commit a
partial accepted-token ledger. The outgoing queue is bounded to 32 events.

The original adapter waited for a complete native result and emitted the entire
message at once. DSH measured output tokens divided by the few milliseconds from
that burst to its assembled message, producing readings such as 67,625 tok/s;
its TTFT included generation time. These historical readings are invalid and
were not rewritten. New conversations use the streaming continuation provider by default;
continuing an old conversation on its old provider still has the old behavior.
DSH now measures visible output arrival, including transport/parser overhead;
this is not a pure GPU decode benchmark. Raw native request metrics and exact IDs
are retained in `bridge-stream/<owner>/<turn>/accepted.json` for independent
checks. Native `metrics.tokens_per_second` uses its own end-to-end denominator
and must not be equated with DSH's first-visible-output-to-finish rate.

## Executed checks

- Three bridge host tests pass, covering tool/result identity and Unicode,
  title/agent/owner isolation and unsupported images.
- DSH session `kvmem-main-dsh-smoke-20261020-r2` reads a random local file with
  the real file tool. The second user turn repeats its marker without a tool.
  Both turns complete; the tool continuation and second turn reuse actual
  accepted output prefixes. The first smoke exposed shared title state and is
  preserved separately; the isolated second smoke supplies the state proof.
- A 76,530-token input exceeds K+G and triggers two real prefill pressure
  selections. Logs show `pressure_policy=attention`, `seed_source=prefill`,
  1,662 / 2,171 accumulated prefill samples and K=36,864 selected tokens.
- The final Query triggers one actual `cpp_v2` multi-graph selection. Pressure
  report rows have no graph backend; Query propagation executes in C++.
- Every adapter plan has `kvmem_mandatory_parts=none` and an empty explicit
  mandatory list. The long fixture's hard mandatory blocks are exactly the
  first 32 logical blocks (1,024-token sink).

Machine evidence: `artifacts/dsh-main-20261020-r1/validation.json`,
`dsh-two-turn-proof.json`, `long-policy-response.json`, `bridge-tests.txt`,
`bridge/requests.jsonl`, and native `capture/bounded_refresh.jsonl` /
`capture/block_graph.jsonl`. These are functional checks, not a full agent task
quality evaluation or a throughput benchmark.

## Streaming timing correction

- All 32 bridge tests pass, including comparison to the frozen Qwen parser for
  Unicode, split tags and trailing newlines, real-time canonical tool headers,
  delayed upstream completion, arbitrary SSE network boundaries, disconnects,
  incomplete terminal streams and accepted-token/usage mismatches.
- A real model request returns 506 accepted output tokens in 68 visible chunks:
  first visible output at 0.540 s, complete at 3.128 s. The DSH-equivalent visible
  decode interval is 2.588 s, or 195.5 tok/s for this short request.
- Independent DSH session `kvmem-main-dsh-stream-smoke-20261020` executes one real
  file read and returns the random verification code. Its two model steps report
  137 output tokens across 553 ms of visible decode, or 247.7 tok/s, with average
  TTFT 1.576 s. The tool-result continuation reuses actual accepted output IDs.
- These are connectivity/timing checks; the two requests differ and their
  speeds are not a performance comparison or an eight-request benchmark.

Evidence: `stream-timing-proof.json`, `dsh-stream-smoke-events.json`,
`dsh-stream-timing-proof.json`, `bridge-stream/requests.jsonl`,
`dsh-stream-connection.json` and `service.json` in the same artifact directory.

## Automatic continuation and real CSGO task

The bridge retries only a completed native segment with `finish_reason=length`,
using the original input plus all actual accepted output token IDs. It keeps the
same owned session and unchanged Query coordinates. Native exact-prefix checks
suppress an unchanged auto Query; pressure selection runs when the restored view
cannot accommodate the next segment's reservation. This is multiple native
requests within one client reply, not native DFlash rolling generation. The
service's GPU pool, K/G, attention sampling and selection policies are unchanged.

The decoder retains partial UTF-8 bytes across segments. Tool arguments are
published only after the authoritative parser accepts the complete call. The
accepted transcript is committed once per complete logical reply. Completed
turns can be restored after a bridge restart from the recorded actual IDs;
automatic crash recovery of an unfinished reply is not implemented. Output can
still end at the four-segment limit or logical context limit.

- Bridge regression: **41 passed**, including segmented UTF-8, exact-prefix
  continuation, later-segment failure, context capacity and complete tool parsing.
- Small forced-boundary native probe: three segments complete one function call;
  the warm repeat verifies checkpoint reuse between native requests.
- DSH session `kvmem-csgo-auto-20261020-r1`, original prompt
  `做个网页版的csgo`: one complete tool call contains **27954 output tokens**,
  automatically split into **16384 + 11570**. Its visible generation interval
  reports **163.20 token/s**. This is a single long-call observation, not a
  matched throughput comparison.
- The first user turn completes 24 model steps with no bridge errors. Browser
  feedback and subsequent manual repairs fix runtime/control issues; the result
  is an assisted task completion, not an unassisted quality benchmark.
- The initial browser acceptance covers start, movement/direction, shooting,
  reload, armor purchase, knife, pause/resume, resize and restart. The preview is
  served through DSH with an explicit asset allowlist. Further objective/round
  testing is recorded separately in `artifacts/dsh-csgo-20261020-r2`.

Evidence: `artifacts/dsh-csgo-20261020-r1/continuation-native-warm-proof.json`,
`auto-run/restore-and-decoder-proof.json`, `auto-run/timing-summary.json`,
`browser-auto-functional.json`, `preview-http-proof.json` and `bridge-auto/`.

## Continued DSH game test

The second user turn in the same session completes **80 model calls**, with a
maximum complete logical input of **185732 tokens**. The restarted bridge
restores the prior 24 committed model turns. All 80 new calls reuse the actual
accepted output prefix, with **zero bridge errors**. Native records confirm
four `cpp_v2` multi/alpha .2 Query selections, 78 attention pressure selections
without graph propagation, and 156 submitted checkpoint captures. Extra
semantic pins remain disabled. The native server is not restarted and its
GPU pool and K/G settings are unchanged.

The visible generation interval contains 48678 output tokens in 350407 ms,
or **138.92 token/s**, with mean visible TTFT **1.331 s**. Model time is
456.890 s and tool time is 1.810 s. These describe this assisted single-session
task; they are not an eight-request throughput result or a speedup comparison.
No new call in this turn crosses G; the first turn's 27954-token call remains
the real automatic segmentation evidence.

Browser findings expose purchase-menu toggling, premature planted-bomb wins,
missing dropped-bomb pickup, carrier reassignment that moves a dropped bomb,
and multiple bots accelerating one defuse. The model implements initial
repairs; external review further fixes CT elimination after planting, binds
the defuse timer to one living bot, and makes bots seek the relevant bomb
objective. The final same-origin browser acceptance passes 11 base/control
checks and 13 objective checks, with no page errors. Objective fixtures set
positions/timers to cover branches, while B/E and the base controls use actual
browser input; this does not prove general bot navigation or complete task
quality. The retained Three.js UMD deprecation warning is informational.

The early CT-bot fixture incorrectly selected a human CT, whose opponents
contain no CT bots. Corrected tests select a human T and a real opposing CT bot.
The original one-bot case then passes; its earlier failure is not counted as
a product defect. Initial and corrected evidence remain separate.

Task efficiency remains a limitation: 45 reads, six searches, 14 shell calls
and 14 edit attempts precede the final answer; four edits fail because the
model supplies nonexistent old text. The model claims stale read output, but
replaying every successful edit from the saved initial file shows **4019
numbered lines across all 45 reads match their exact file state**; the
reconstructed endpoint also equals the model's final file. No stale-read
software bug is established. Browser feedback and finalization steering are
recorded, and the task is not labelled an unassisted agent success.

Evidence: `artifacts/dsh-csgo-20261020-r2/validation.json`, `timing-summary.json`,
`policy-proof.json`, `read-consistency-audit.json`, `objectives-verified-baseline.json`,
`objectives-final.json`, `browser-auto-functional.json`, `endpoints-final.json`,
`before/` and `agent-game.js`. The original failed/incorrect fixtures are retained
for audit and superseded by the corrected baseline and final checks.

## Guided Query trial with DFlash2

The user authorized interruption of the active DSH task to switch the single
GPU model. The current backend uses a frozen source snapshot in
`artifacts/dsh-guided-query-20261020-r1/runtime`; its process/profile/log are
recorded in `guided-process.json`. The deployment profile differs from the
prior profile only in `guided_query_tokens: 0 → 512`. K36864/G16384,
sink1024/recent2048, attention pressure, V2 multi/alpha .2 and the four overhead
optimizations remain unchanged. Query history stays disabled. Private scratch
is reserved within K+G; the private rewrite uses eager greedy target forwards,
while regular DFlash decode retains FULL CUDA Graphs. The bundled guided
default remains zero.

Host regression passes **66 tests**. A separate K4096/G4096, C2 real-model
qualification passes six requests across explicit Query, tool continuation and
suffix edits, including 6/6 narrow dependency answers. Two private Queries
publish 81 and 217 captured rows; four empty outputs explicitly fall back to
public Query Q. Two verification records compare target GDN, acceptance
counters and committed draft sliding KV bytes for all batch owners before
public replay. Four follow-up restores verify target/GDN/five-layer draft
bytes; accepted attention counts match the independent public-token parser.
All qualification owners and the GPU worker are released. Production restore
verification is disabled, so its audit copies do not enter the DSH timing.

The DSH trial forks the original first-turn endpoint at seq5844, seeds the 24
actual accepted-token ledger entries, restores the same initial project files
and submits the same follow-up prompt. Forty completed model calls all reuse
the exact output prefix, with no bridge errors. Real Query selection publishes
135 and 154 private rows, with no fallback, through `cpp_v2` multi/alpha .2.
The second Query follows DSH's automatic `repeat-tool-reminder` notice; ordinary
tool continuations do not create fresh Queries. Private generation takes
15.136 and 16.757 seconds, adding latency at those two selection boundaries.

The observed agent still makes **36 reads of game.js and four shell calls**,
with seven extra calls whose arguments exactly repeat an earlier call, and
does not change any project file. After the 40 completed steps, the test runner
cancels step41 to bound the unproductive loop. No corrective hint is manually
injected. All 4879 numbered lines in the 36 reads match the actual unchanged
file. In the prior unguided trace's first 40 steps there are 33 reads and five
extra exact duplicates; neither prefix starts editing. The prior run was warm
and externally steered, while the current backend starts cold. These are
trajectory observations, not a causal throughput or quality improvement.
This trial does **not** demonstrate that guided Query fixes repeated inspection
and is **not** a completed agent repair task.

The previously accepted playable preview is restored byte-for-byte and passes
all 11 base/control browser checks with no page errors. The owned trial's
history/checkpoint/graph resources are explicitly released; its transcript is
retained. The DSH model service remains available with guided Query enabled
for trial use. Evidence in `artifacts/dsh-guided-query-20261020-r1` includes
`validation.json`, `qualification/results.json`, `component-tests.txt`,
`agent/trial-summary.json`, `agent/prefix-comparison.json`,
`agent/read-consistency-audit.json`, `agent/bounded-stop.json`,
`agent/native-release.json`, `agent/browser-auto-functional.json` and
`preview-restored.json`. Session timing includes the cold prefill and one
interrupted reply; this run supplies no C8 throughput comparison.

### Post-trial audit of current tool-result retention

The correct tool output audit alone does not establish that the returned text
remains in the next decode attention window. Joining exact bridge tool spans
to the final native selection for each of the 40 requests establishes a
specific retention problem. `selected_history_ranges` contains
`[logical_start, token_count]`, not endpoint pairs; each row's lengths sum to
the reported selected budget. Replay spans use ordinary half-open endpoints.

At step36, the previous read result occupies logical tokens [150922,151836),
914 tokens in total. Pressure selection keeps 36800 historical tokens but
none of this tool result; its final public replay is [151840,151841), after
the entire result. No further input prefill follows. The model then repeats
the same read. At step37 only 18 of the next 914-token copy remain through
selection plus replay. The Query at step38, after the automatic reminder,
retains all 914 tokens; subsequent pressure selection again removes almost
all of the next tool result.

The corresponding code explains the exposure: `pressure_seed` prefers any
positive historical output-attention scores and uses prefill scores only as
a whole-domain fallback. Newly appended tool results have no preceding
output-attention observations. `selection_config` also sets the fixed recent
band to zero for attention pressure, even though Query selection has the
configured recent2048 band. With explicit tool-result pins disabled, newly
read evidence can therefore be evicted before public answer generation.
Guided Query changes Query scoring but does not change this pressure path.

This establishes removal of original tool-result KV and a concrete mechanism
consistent with the repeated reads. GDN state may retain information from
processing the text; the audit does not prove total information loss or that
all repetitions have this cause. A controlled retention-policy comparison is
still needed to measure its causal contribution. No serving behavior is
changed by this diagnostic audit. Evidence:
`artifacts/dsh-guided-query-20261020-r1/agent/tool-retention-audit.json`.


2026-10-20 guided Query 已启用 CUDA Graph 私有生成加速。当前模型配置改为
`artifacts/guided-speed-20261020-r4/profile.json`，普通服务关闭重复 eager oracle。
135 token 完整私有区间首次 3.466 s，后两次均值 2.895 s；
[测量与回滚证据](guided-query-latency-20261020.md)。服务入口和 K/G、sink/recent、评分规则保持原配置。
