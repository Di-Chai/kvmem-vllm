# Private Query replay execution, 2026-10-22

The observed 619 ms foreground interval included approximately 604 ms of eager
private Query forward work (42 tokens). That work occupied the engine thread
before its next public forward. It was not a measurement of pure GPU idle time.

Two independent, default-off execution options retain the existing selection
policy:

| Option | Default | Behavior |
| --- | --- | --- |
| `query_replay_backend` | `eager` | `compiled` enables the existing compiled model for standalone Query replay. |
| `query_replay_cobatch` | `false` | Appends one private Query suffix to an eligible public target batch. |
| `query_replay_max_bytes` | 67,108,864 | Bounds retained/transient Query copies and merged input metadata. Native target storage remains within its fixed token/request capacities. |

Compiled standalone replay uses CUDA Graph mode NONE with model compilation
enabled. This does **not** add a new private CUDA Graph capture. On a mixed
turn, public-only FULL graph replay is bypassed and the compiled target handles
the unpadded combined batch; ordinary public graph dispatch continues on other
turns.

The private suffix has no public sampling or MTP acceptance. Its Q rows are
copied before public captures can be reused, restricted to the logical Query
span, and passed to the original selection/assembly path. Before public sampling,
the runner projects hidden and auxiliary outputs and reconstructs public-only
attention/slot metadata. Statistics, archive publication and step classification
use the original public prefix. The original recurrent checkpoint, generation,
page binding, restore fence and owner lease remain authoritative.

Admission requires spare native token/request capacity, an exact live owner and
checkpoint, and completed required publication. Physical prefill groups and
unqualified layouts decline before execution. At most two declines defer an
owner; it then uses standalone Query replay. Execution failures quarantine the
publication path and retain the failed frame rather than retrying a partially
mutated public batch.

## Validation and restart

The affected CPU suite passed **342 tests**, with three environment-dependent
skips. The frozen-execution binding and final actual-mode audit passed a further
19 focused tests (including overlapping RULER protocol cases). The broad optional
gateway suite was interrupted after hanging; it is not counted as passed.

`artifacts/query-replay-20261022-r3/` freezes three real-model arms: eager replay,
compiled replay, and compiled plus mixed replay. Four staggered shortened
histories (64K/80K/96K/112K prefixes plus their original question tails) use the
same IDs and fixed 128-token outputs across arms. These are performance fixtures,
**not** complete RULER accuracy inputs. Acceptance, public history length, sample
counts, graph/pressure policy, native-score fallback and release checks must pass.
The report separates output/selected-history differences and cold preparation
from request wall time. One pass per arm is exploratory evidence only.

The first attempt (r1) passed the eager arm, then exposed an AOT signature
mismatch: Qwen's public model was compiled with embedding inputs, but private
replay supplied token IDs and `inputs_embeds=None`. The corrected compiled path
uses native `model_state.prepare_inputs_embeds` and its raw-token policy before
model-specific overrides. Its four embedding/backend combinations are covered
by the 44-test focused replay/refresh regression. The failed attempt and its
clean GPU teardown remain archived; it is not treated as a passing comparison.

The r2 compiled arm passed all four requests. The first mixed target then
exposed a return-type adaptation error: native `ExecuteModelState` is a
`NamedTuple`, not a dataclass. Projection now uses `_replace`, and CPU projection
tests construct the real native return type and verify unrelated fields survive.
Its 29 replay tests passed. The r2 failure is retained separately.

The r3 mixed arm passed all four requests, with **four actual mixed forwards**,
zero standalone fallbacks, no native-score validation failures, and a maximum
accounted frame size of **7,080,000 bytes** within the 64 MiB limit. Public peers
included 6,144-token prefill and mixed prefill/decode batches, then 10- and
15-token MTP decode batches. All private/public-history, sample-accounting,
selection-policy and owner-release checks passed; the final full-suite Query
execution audit also accepts these real reports.

All three r3 arms completed and cleaned their owned processes/GPU allocations:

| Execution | Four-request wall seconds | Input tokens / wall second | Standalone Query host section, median ms |
| --- | ---: | ---: | ---: |
| Eager baseline | 58.334 | 6,180.6 | 172.89 |
| Compiled standalone | 57.669 | 6,251.9 | 121.10 |
| Compiled + mixed | 58.323 | 6,181.8 | Not an exclusive private section |

Cold model preparation was 72.15/72.15/73.15 s respectively and is reported
separately. Request wall time includes prefill, selection, fixed-length output
and retirement. Query host sections include preparation/dispatch and possible
waits, and do not measure fenced GPU kernel time. The mixed section includes
public target work and must not be compared directly with private-only timings.
These are four Query forwards per arm, not a per-output-token measurement.

Compiled standalone reduced the median private host section by about 30%; its
observed whole-campaign throughput difference was +1.15%. Both options together
were **effectively tied** with baseline (+0.02%). This single pass establishes
no stable whole-campaign throughput improvement. Output IDs and selected-history
ranges differed in both experimental arms; the benchmark reports this explicitly
and does not assign the differences to a particular cause or infer accuracy.
The subsequent full 260-case evaluation is recorded below.

The user authorized stopping the previous full RULER attempt after **87/260**
completed cases to qualify these changes and restart. Its frozen runtime,
predictions and resource logs remain under
`artifacts/ruler-v1-256k-optimized-20261022-r2/`; its owning controller cleaned all
owned process groups and GPU allocations. Old and new predictions must not be
pooled into one quality result.

After real-model qualification, the new full campaign was started at
`artifacts/ruler-v1-256k-query-replay-20261022-r1/`, with qualification evidence
and runtime hashes recorded before launch. It reuses all original
260 input-ID files, seeds, sampling and official scoring. K36,864/G16,384,
cumulative-attention pressure and V2 multi Query graph propagation (alpha .2),
guided off, NVFP4/FP8 and MTP4 remain fixed. The fixture binds both Query options,
and the final audit requires actual mixed execution and clean lifecycle/memory
counters. The run completed at **14:02:45 Asia/Shanghai**, with an official
13-task macro score of **85.75/100**, 212 full-score cases, 13 partial-score cases
and 35 zero-score cases. The shared 87-case means were 87.7778 (previous) and
87.8161 (new); this is not a complete paired ablation. Two UUID cases stopped
before closing thinking and scored zero; none reached the output cap.

The final audit passed with **260 actual mixed forwards**, no standalone Query
fallback, no native-score validation failure and no additional native-score
fallback beyond the three known startup layer events. The maximum accounted
mixed-frame size was 30,941,056 bytes, within the 64 MiB limit. All owners were
released; cleanup recorded no remaining owned process groups or GPU allocations.
The official report is `run/answers/report.md` in the campaign. Defaults remain
off because stable end-to-end performance gains have not been demonstrated.
DSH stays stopped and MAB stays paused; the controller restores neither service.
