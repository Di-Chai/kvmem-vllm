# Independent native System One followup

`followup.py` tests the format risk observed in the completed initial pilot,
without changing its baseline code or outputs. It also separates autonomous
stage recognition from assessment of an explicitly supplied goal.

The three model experiments run serially:

1. **Frozen-state rereads:** append the same explicit single-letter instruction
   to all four questions of each of the 57 saved requests. State, question
   criteria and option order stay identical. Six predefined positions also
   receive a fresh original-prompt control after service recovery. These inputs
   contain saved baseline memory; this is not a fresh sequential replay.
2. **Independent continuous replay:** replay all 57 Steps using new model-derived
   memory, applying the letter instruction consistently. Broad activity types,
   goal IDs and local outcome remain separate. The extractive-goal limitation
   and four-independent-read architecture are intentionally retained.
3. **Explicit goal scope:** seven predeclared checks distinguish a successful
   file creation from whole-feature validation and a failing verification
   attempt from a broader ongoing repair. Both outcome and evidence questions
   see the same supplied goal. Reference answers remain outside the payload.
   This does not establish autonomous goal discovery or segmentation accuracy.

The native HTTP/schema/probability qualification can separately use
`scripts/qualify_systemone.py`. Without its optional capture directory it does
not establish KVMem session release or cancellation behavior.

```sh
python3 experiments/agent_execution_observer/followup.py prepare \
  --dataset PATH_TO_ORIGINAL_FULL_SHORT/dataset.jsonl \
  --baseline PATH_TO_COMPLETED_BASELINE --output NEW_INPUT_DIR
python3 experiments/agent_execution_observer/followup.py run \
  --inputs NEW_INPUT_DIR --output NEW_RESULT_DIR \
  --base-url http://127.0.0.1:18993
python3 experiments/agent_execution_observer/followup_audit.py \
  --inputs NEW_INPUT_DIR --results NEW_RESULT_DIR \
  --baseline PATH_TO_COMPLETED_BASELINE --output NEW_AUDIT_DIR
```

This design makes 63 frozen-state calls, 57 continuous calls and seven
goal-scope calls. It reuses the previously reviewed trajectory in an adaptive
diagnostic, with no independent human ground truth or generalization claim.
The added formatting is not a decoder constraint: native full-vocabulary
`label_mass` and `argmax_is_label` remain visible and must be examined separately
from semantic judgments. Native confidence is not calibrated correctness.

The audit reconstructs all 127 requests, including option order, validates
native response semantics, rebuilds the new memory step by step, and checks
that reference answers were not inserted into requests. It exports a CSV and
an HTML review with original visible inputs and evidence coordinates.
Coordinates alone do not establish evidential support. Input truncation and
the original bounded-memory choices remain in effect.

`goal_hint_probe.py` adds a separately recorded six-pair adaptive diagnostic.
It replaces only S004's extractive goal representation, consistently in state
and option descriptions, with an exact TaskCreate subject already observed in
Step11. Steps 13/14/15/16/17/22 retain their other frozen baseline state, including
potentially incorrect old assignments. A source quote and original-input
visibility check are required; future goal sources are rejected. This checks
conditional discrimination of a supplied stored goal, not autonomous discovery.
The optional goal-hint arguments of `followup_audit.py` verify all 12 additional
requests and the original source coordinate against the read-only bundle.

Use new output directories. Do not edit or overwrite completed baseline runs,
and inspect a remote controller's state before starting another copy.
