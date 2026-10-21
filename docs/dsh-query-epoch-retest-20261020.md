# DSH Query lifetime retest and terminal-prefill fix, 2026-10-20

The subsequent [full continuation test](dsh-query-epoch-longtest-20261020.md)
finishes177 steps naturally and crosses this same failure with an identical
accepted-output prefix. Long stability passes; independent agent repair quality
still fails. The failed trial and immediate recovery below remain historical evidence.

The longer DSH trial reproduced the reduction in premature pressure selection,
but did not complete the CSGO repair task. It first edited the project at step93
and failed at step132 because native allocation crossed the active-window
capacity. The capacity bug is fixed and deployed; targeted GPU and real DSH
recovery checks pass. A complete unassisted long task after this second fix has
not been demonstrated.

## Unassisted trial

Evidence root: `artifacts/dsh-query-epoch-20261020-r2`.
The trial forks the original session at seq5844, preserving the same 24 actual
accepted replies, original files and follow-up prompt as the earlier guided
trials. K36864/G16384, sink1024/recent2048, guided cap512, attention pressure,
V2 multi/alpha .2, NVFP4, FP8 KV and DFlash7/FULL graphs are unchanged.

The strict first40 steps reproduce the preceding corrected result: seven
pressure selections including cold prefill, one Query selection, all40 exact
accepted-prefix reuses, and 33/39 latest tool results fully retained at reply
end. There are no edits or calls with identical repeated arguments in that
prefix. This is a repeated observation, not a causal throughput estimate.

The extended run completes 131 agent steps before the step132 failure. It makes
87 reads (86 of game.js), 20 searches, 23 shell calls and one successful edit.
Two extra calls repeat identical arguments. The first edit at step93 fixes
wheel cycling through nonexistent weapon slots4/5. Real browser wheel events
verify both directions: the baseline reaches a null weapon, whereas the patch
wraps correctly between slots1 and3. All86 reads / 5361 numbered source lines
match the original file plus the successful edit; no file-read corruption is
observed. Source reads still overlap extensively, and the bomb-flow problems
remain unresolved by the model.

Automatic DSH history rewriting is a separate cost:

| Native ledger turn | Trigger | Rebuilt prompt tokens |
| --- | --- | ---: |
| 109, agent step85 | Eight old tool results pruned | 193209 |
| 125 | Compaction summary model call | 148802 |
| 126, agent step101 | Summary replaces earlier history | 71234 |

These three requests cannot reuse the prior exact accepted-token prefix. The
first two still carry the original manual Query; after full compaction that
message is absent, so the provenance adapter disables semantic Query selection.
The compact/system-policy plugin messages are not treated as manual Queries.
The full run has three Query selections and 29 ordinary pressure selections,
including rebuild work. Those totals must not be compared against a first40
prefix. Native successful requests total132: 131 agent calls plus one summary.
The failed agent step132 is retried by DSH and eventually ends with TRANSPORT.

The planned 150-step observation limit never cancels this trial: the native
failure ends it earlier. Total wall time to the terminal event is about941s.
DSH's visible-output rate is not a pure GPU benchmark, and there is no new C8
throughput claim.

## Capacity failure and fix

The previous reply retains a terminal checkpoint at logical history102204 and
working window53212 (committed offset48992). The next prompt has628 more input
tokens. The next sealed history is102208, with local window53216. A whole
32-token prefill pressure probe plus seven native draft lookahead rows asks for
53255 positions, exceeding capacity53248. Allocation happens before connector
metadata can execute the intended pressure refresh, so `RefreshRequired`
escapes the scheduler and terminates the engine. The later connection failures
and terminal-capture shutdown exception are consequences.

`SessionLedger.lookup` now also checks whether a resumed pressure-mode view
already lies past its safe prefill pressure point. It selects during the
existing fenced import, preserving the accepted partial tail and saved model
state. It does not rewind/re-encode the accepted output or reserve another full
G. A long incoming suffix can still use subsequent ordinary pressure points.
The existing transaction validation and cancellation rollback remain in use.
For the failing geometry the imported window becomes36860, leaving room for
normal prefill and native lookahead.

## Verification after the fix

- **119 Python tests pass**, including the exact failing geometry, near-boundary
  cases, a long suffix, unchanged small continuations, cancellation rollback,
  RequestPlan/speculative admission, session worker and restore admission.
- `scripts/qualify_terminal_prefill.py` passes on the real GPU with two
  independent owners. Fixed initial prompts53000 and output quotas213/217 reach
  terminal windows53212/53216, then append628/624 input tokens. Import selects
  windows36860/36864, and both continuations complete. Independent FP8 target
  rebuild bytes, GDN and five-layer draft bytes match; accepted attention counts
  match the independent public-token parser. No second selection is needed for
  these short continuations. All owned resources are released.
- The normal serving profile is restored with audit allocation disabled. A
  fresh real DSH session performs exactly one read and returns the exact marker
  in two model calls. Its manual Query identity stays stable and the second
  request reuses the exact accepted prefix. Its separate title request and both
  owners are verified/released.
- The preview is restored to the preceding accepted playable version and keeps
  the validated wheel patch. All11 base interaction checks, 13 objective cases
  and the wheel check pass. Objective tests use controlled game positions and
  timers; they do not establish general bot navigation quality.

The model still performs excessive inspection. DSH automatic compression still
rewrites history and incurs extra prefill/reselection. Neither behavior is
claimed fixed by the capacity correction. The full unassisted task remains
unqualified even though the isolated failure and deployment recovery pass.

## Evidence and current service

Key files under the evidence root:
`agent/first40.json`, `agent/trial-summary.json`, `agent/history-rebuilds.json`,
`agent/read-consistency-audit.json`, `agent/wheel-after-agent.json`,
`component-tests.log`, `terminal-prefill-qualification/results.json`,
`recovery-probe/proof.json`, `preview-restored.json`,
`owned-resources-released.json`, `service-health.json` and `validation.json`.
The original failing log is preserved under the preceding r1 root's
`server.log`; the verified fix server and normal recovered server have separate
logs/capture directories in r2.

The native server now uses r2 `profile.json` / `server`, port18846. The frozen
r1 runtime receives only the qualified `session_control.py` correction; before
and after hashes/copies are preserved in r2 `fix/`. DSH port3080 and bridge
port18385 remain unchanged. `artifacts/dsh-main-20261020-r1/service.json` and
`status.json` point to the current deployment.
