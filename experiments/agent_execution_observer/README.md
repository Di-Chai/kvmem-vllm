# System One execution-stage pilot

This pilot implements the first segmentation check in the supplied Agent
Execution Observer plan: current activity, goal-based stage identity, local
outcome and an extractive evidence citation. It uses the native vLLM
`POST /v1/systemone` categorical API, with thinking disabled. It does not call
the completion API to simulate that endpoint.

One assistant turn plus its observed tool returns is a Step. Each call sees
the task instruction, bounded current Step, the last two observed Step excerpts,
and at most eight previously established stage summaries. Every replay window
has separate memory. The replay never executes commands in the source traces.
Future messages, exporter outcome annotations and evaluation labels are absent
from model inputs. Source hashes and each request/response are retained.

The model chooses whether to reuse an existing goal ID or begin a new goal.
The client allocates IDs and stores status revisions. Switching away from a
goal does not mark it completed. Broad activity types and goal IDs are separate:
two implementation subfeatures can be different stages, and a single stage can
involve both reading and editing. `mixed` flags a multi-goal Step for review;
this initial pilot does not resolve multiple simultaneous stage assignments.

System One categorical reads cannot generate arbitrary objectives or free-text
explanations. Stage objectives are explicitly marked *extractive goal hints*,
and evidence is selected from exact original excerpts. A citation being valid
does not prove it supports the decision. Confidence retains native vocabulary
probability semantics and is not a calibrated correctness probability.

## Run

```sh
python3 -m unittest discover -s experiments/agent_execution_observer -p 'test_*.py' -v
python3 experiments/agent_execution_observer/observer.py prepare \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --output artifacts/observer-inputs --design windows
python3 experiments/agent_execution_observer/observer.py replay \
  --dataset artifacts/observer-inputs/dataset.jsonl \
  --output artifacts/observer-results \
  --base-url http://127.0.0.1:18993 --model observer
```

The client uses only the Python standard library. The server requires a real
System One-capable vLLM/model installation. A missing endpoint is a failure;
there is no fallback route or mock inference.

`--design windows` freezes three positional four-Step windows per trajectory,
covering four tasks and four seeds: 192 Steps in 48 windows. Selection uses turn
positions, not stage labels. Middle/end windows are explicitly left-censored,
so they cannot establish complete-trace stage recognition or long-term memory
quality. `--design full-short` separately selects all 57 Steps of SuperJSON
seed 1004 for continuous incremental replay. These two cohorts overlap in raw
Steps and must not be counted as independent examples.

Input truncation is explicit. Long tool payloads and task instructions are
bounded with head/tail excerpts; evidence candidates preserve exact character
coordinates. These are prototype budget choices, not full-history parity.
`--mode current` omits execution memory as an activity/outcome ablation; stage
links then have no historical IDs and are not a temporal segmentation baseline.

## Evaluation boundaries

`summary.json` records actual HTTP latency, native input/output token counts,
label reads, uncertainty and label-format diagnostics. Input usage sums all
native per-question reads, including their repeated prefixes. Initial cold
calls must be identified separately when comparing steady-state latency.

Accuracy and boundary F1 remain null until independent reference labels exist.
A model-assisted review may provide provisional agreement and concrete error
examples, but must not be described as human ground truth. Review should
distinguish activity classification from semantic goal boundaries, include
both transitions and continuations, and assess premature completion separately.

The supplied plan's dependency graph, future demand, supersession, online
asynchronous attachment and main-Agent overhead ratio are subsequent work.
No KV recall, eviction or retention decision is made by this pilot.

## Read-only audit and review

`audit.py` reconstructs each saved request from raw source messages and prior
native responses, checks same-Step tool-call/result IDs, source/dataset hashes,
all evidence spans, native probability diagnostics, and final execution memory.
It never calls the server. `review.py` exports all decisions, retaining both raw
and normalized boundaries. Both require a **new** `--output` directory and do
not modify the results being inspected.

```sh
python3 experiments/agent_execution_observer/audit.py \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --dataset INPUT_DIR/dataset.jsonl --results RESULT_DIR --output NEW_AUDIT_DIR
python3 experiments/agent_execution_observer/review.py \
  --dataset INPUT_DIR/dataset.jsonl --results RESULT_DIR --output NEW_REVIEW_DIR
```

Optional `--activity-reference` and `--boundary-reference` score frozen
model-assisted anchors as **provisional agreement only**, with boundary
abstentions and decided-anchor coverage. No F1 is calculated without independent
human labels. Post-hoc selected cases are exported by `case_review.py` from an
explicit JSON case specification; exact raw-source citations are checked and
post-hoc interpretations are not silently promoted to reference labels.

`aggregate.py` validates the six paired format-probe payloads, verifies the
archived-v1/current-v2 runtime difference is only the documented client boundary
correction, and exports costs, per-question diagnostics, stage assignments and
the 12 overlapping raw Steps. All three diagnostic pairs use frozen past state;
the explicit-letter variant is **not** a replacement baseline. Low label mass
or a non-label vocabulary argmax is a label-format risk, not malformed HTTP/JSON
and not itself proof of a semantic error.
