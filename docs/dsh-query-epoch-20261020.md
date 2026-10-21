# DSH Query lifetime and DFlash capacity correction, 2026-10-20

The subsequent [longer retest and terminal-prefill fix](dsh-query-epoch-retest-20261020.md)
records a step132 allocation failure, its qualified fix and the current recovered
service. The bounded trial and original deployment below remain historical evidence.

The default DSH continuation route now keeps the selected attention window
across tool calls within one real user Query. Each native request still allows
up to G=16384 output tokens, but it no longer reserves a fresh complete G before
continuing. New input and accepted output use the remaining K+G window; actual
capacity pressure, including the next chunk and native speculation headroom,
can still select a smaller historical window. New manual Queries retain the
existing guided/graph selection. No attention formula or mandatory policy is
changed.

The previous generation-reservation behavior could compact before every tool
continuation, then remove the tool result that had just been read. Its logical
KV audit is in `artifacts/dsh-guided-query-20261020-r1/agent/tool-retention-audit.json`.
This correction addresses that premature pressure and separately preserves
manual-user identity through DSH's provider conversion.

## qw3 reference

Local qw3 revision `1410cfe2` was inspected directly:

- `src/kvmem_request_plan.hpp:324`: checkpoint finalization turns a stable
  continuation into M-resume/append or P-resume/tail replay. Checkpoint holes
  and suffixes that exceed available bounded history use query-independent
  pressure rather than inventing a new semantic Query.
- `src/qwen_executor.cpp:13494`: page recovery checks physical free pages
  against the next required input/decode chunk; keep-selected execution can
  fall back to pressure when that chunk cannot fit.

The adaptation follows those lifetime/capacity boundaries. It does not claim
identical tokens, floating-point scores, model state, or every qw3 refresh
trigger. Reference file hashes are in the new run's `qw3-reference.json`.

## Implementation and deployment

The existing `rolling_active_window` planning path is extended to experimental
**synchronous DFlash2 k7 / FULL graphs**, alongside its existing asynchronous
MTP path. The launcher and scheduler retain method-specific gates. Speculative
admission clips all possible accepted prefixes at the sealed capacity boundary.
Synchronous pressure assembles accepted history before the scheduled public
verification batch; it makes no private forward for that pressure transaction.
The optional byte oracle checks target KV, GDN/counters and independent draft
sliding KV before public replay. Audit allocation remains disabled in the DSH
service. The bundled rolling default remains false; this qualified deployment
explicitly enables it.

`scripts/dsh_query_epoch.mjs` reads DSH's immutable message source and stable
message IDs before provider serialization. Its payload hook adds metadata,
without changing messages or sampling. `scripts/dsh_query_epoch.py` verifies
the session, complete ordered user-text digest and renderer message domain,
then resolves the actual user's verified token coordinates. Tool messages and
plugin notices keep the previous manual Query. Invalid or missing metadata
fails before inference on the corrected default route. Other providers retain
their existing behavior.

The installed DSH adapter calls that hook through its existing `onPayload`
seam. Both installed-file originals, patch candidates and hashes are preserved
in `artifacts/dsh-query-epoch-20261020-r1/dsh-adapter-patch`. The current default
provider is `kvmem-main-dflash2-continuation`, bridge port18385, native port18846,
DSH port3080. Native and bridge sources are frozen under that artifact root's
`runtime`; `process.json` and `services.json` contain the current commands.
DSH was idle before its restart; conversations and actual accepted-token
ledgers remain available.

The deployment profile differs from the preceding guided trial only in
`rolling_active_window: false → true`. K36864/G16384, sink1024/recent2048,
guided cap512, CPU history100GB, FP8 KV/NVFP4, DFlash7, attention pressure and
V2 multi/alpha .2 remain configured. No new mandatory pins are added.

## Executed verification

- **152 Python component tests pass**, covering RequestPlan/session continuity,
  speculative boundaries, DFlash launch gates, state oracles and full streaming
  bridge regression. Four Node provenance checks pass. Actual Qwen renderer
  and tokenizer checks confirm the original manual Query's token coordinates.
- **C2 real-model qualification passes eight requests / 9734 output tokens**.
  Each short tool continuation has zero selections. Each forced long decode
  has exactly one pressure event at the actual capacity boundary. Both new
  manual Queries select exactly once. Two pressure audits verify target KV,
  GDN/acceptance counters and draft bytes; six session restores verify target,
  recurrent and five-layer draft state. Multi-token DFlash acceptances of
  seven/eight rows precede those boundaries. Accepted attention samples match
  the independent public-token parser oracle. All owned resources are released.
- A separate real DSH probe performs three identical reads, triggers the actual
  `repeat-tool-reminder`, then returns the correct file marker. Across four
  model calls the manual message ID and Query coordinates stay identical;
  there is one initial Query selection and no extra selection for the notice.

The original CSGO follow-up was repeated from the same first-turn checkpoint,
24 actual accepted model replies, initial files and prompt. Both guided runs
start with cold native state. The strict first 40 steps compare as follows:

| Observation | Before | Corrected |
| --- | ---: | ---: |
| Pressure selections, including one cold prefill event | 39 | 7 |
| Query selections | 2 | 1 |
| Reads of game.js | 36 | 30 |
| Extra calls with exactly repeated tool arguments | 7 | 0 |
| Latest tool results fully retained in the final attention view | 1/39 | 33/39 |
| Code modifications | 0 | 0 |

The six corrected continuation pressure events occur with 53184 or 53216 live
tokens out of capacity53248; remaining space is reserved for the next chunk
and native lookahead. All 41 completed native calls reuse the exact accepted
prefix. The observer requests cancellation after 40 completed steps; it takes
effect during step41, whose read is also recorded. The strict table excludes
that extra step. No model/bridge error occurs. One 135-token guided Query is
actually used; pressure never substitutes a plugin notice as a manual Query.

These observations establish the scheduling and Query-identity correction.
The agent still inspects overlapping ranges and does not begin repairs within
the bounded run, so full task completion and elimination of all redundant
inspection remain unproved. Changed trajectories and differing output volume
also prevent attributing throughput changes solely to this correction. There
is no C8 throughput claim. The CSGO preview is restored byte-for-byte to its
previous accepted version, and test histories/checkpoints/graphs are released.

Evidence: `artifacts/dsh-query-epoch-20261020-r1/qualification-a1/results.json`,
`component-tests.json`, `query-coordinate-proof.json`, `notice-probe/proof.json`,
`agent/prefix-comparison.json`, `agent/retention-comparison.json`,
`agent/tool-retention-audit.json`, `agent/bounded-stop.json`,
`preview-restored.json` and `owned-resources-released.json`.
