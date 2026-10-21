# DSH full continuation test after the capacity fix, 2026-10-20

The corrected service finishes the unassisted CSGO follow-up naturally in
**177 agent steps, about19min21s**, without a model/bridge failure. The exact
previous step132 capacity failure is crossed successfully. Agent repair quality
still fails independent objective checks: it fixes weapon-wheel cycling but
misses several existing gameplay defects and overstates its final verification.

Evidence root: `artifacts/dsh-query-epoch-20261020-r3`.
Native process3908802/engine3909293 and the serving profile remain unchanged
from the [qualified terminal-prefill fix](dsh-query-epoch-retest-20261020.md).

## Controlled replay and the old failure

The same first-turn checkpoint at seq5844, 24 actual accepted replies, original
project files and follow-up prompt are reused. There are no additional manual
instructions during the trial. K36864/G16384, sink1024/recent2048, NVFP4/FP8 KV,
DFlash7/FULL graphs, guided cap512, attention pressure and V2 multi/alpha .2
remain configured. The selected source hashes match the frozen serving runtime.

All132 successful native outputs before the old failure match the preceding
trial's token IDs exactly: 131 agent replies and one compaction summary. The
old failed request, native ledger turn157 / agent step132, again receives
102832 input tokens after terminal history102204. This time it returns493
tokens and execution continues. `agent/passed-old-failure.json` and
`agent/output-comparison.json` preserve this controlled boundary check.

The first40 steps retain the earlier result: seven ordinary pressure selections,
one Query selection, all40 exact accepted-prefix reuses, and33/39 latest tool
results fully retained at reply end. This retention audit is scoped to that
prefix; its older tracker does not model the newly introduced pressure selection
during checkpoint import and is not used to claim full-trial retention rates.

The test completes before its conditional 200-step stagnation guard acts.
No cancellation is requested. There are178 successful native requests:
177 agent calls plus one compaction summary. Of these,175 reuse the exact
accepted prefix; three history-rewrite requests rebuild it. The earlier
eight-result pruning and full summary both complete, followed by normal tools.
The three Query selections and31 ordinary pressure selections include that
rebuild work. Ordinary-pressure logging excludes selections during checkpoint
import, including the newly fixed step132 boundary;31 is not an all-path total.

## Observed agent behavior

The model makes111 reads,20 searches,43 shell calls and two successful edits.
It reads game.js109 times, with six extra calls using exactly repeated arguments.
All109 reads /6094 numbered source lines match the original file plus successful
edits. A step125 read fails because the model misspells the artifact directory;
it then uses the correct path. There are no inference, bridge or provenance
errors. These observations do not establish that every overlapping read was
unnecessary, but substantial inspection produces only one final fix category.

The two edits, at steps93 and174, both change `_onWheel`:

- Limit cycling to weapon slots1–3 rather than empty slots4/5.
- Skip slot2 when no primary weapon is owned and ignore a dead/missing player.

The model runs a syntax check and completes its final reply at step177. It says
the requested gameplay flows have no remaining defects. It does not execute
browser functional tests. Independent checks on its final files show:

| Check | Result |
| --- | --- |
| Base interactions | 11/11 pass |
| Wheel cases, actual wheel events | 7/7 pass |
| Objective/gameplay branches | 4/13 pass |

The nine missed branches are B-key menu toggling; the round timer after planting;
eliminating all T players while the bomb is planted; dropped-bomb pickup and drop
position; defuse progress with multiple/changing CT bots; CT planted-bomb targeting;
and T dropped-bomb targeting. Objective fixtures use controlled positions and
timers. They verify those branches, not general bot navigation quality.

This is a completed unassisted turn and a passed long stability regression.
It is not a passed end-to-end agent repair task. No C8 throughput or causal
performance improvement is inferred from it.

## Final playable service

The model's final wheel method passes seven browser cases, including empty and
owned primary slots, both directions and a dead player. The preceding playable
version already contains the other gameplay fixes. That version is restored,
and only the new validated wheel method is merged into it. The served result
passes all11 base checks,13 objective cases and7 wheel cases. The model's
original result is separately preserved as `agent/game-after-model`; the served
version is not attributed to an unassisted model repair.

The independent test owner is explicitly released. Session checkpoint payloads,
CPU authority storage, graph storage and pending transfers return to zero.
DSH3080, bridge18385 and native18846 all return200; no DSH task remains running.
The native/bridge/DSH processes are not restarted for this trial. CPU/GPU samples
are recorded in `resource-observations.jsonl`; these are observations, not a
continuous peak-memory or throughput benchmark.

Key evidence: `validation.json`, `agent/trial-summary.json`,
`agent/passed-old-failure.json`, `agent/output-comparison.json`,
`agent/read-consistency-audit.json`, `agent/objectives-agent.json`,
`agent/objectives-final.json`, `agent/wheel-final.json`, `preview-merge.json`,
`owned-resources-released.json` and `service-health.json`.
