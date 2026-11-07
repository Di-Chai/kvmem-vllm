# Protocol/field slicing before semantic segmentation

`structure_slicing.py` implements the user's 17 structural types on saved
DeepSWE and Terminal Bench messages. It does **not** infer goals, dependency,
redundancy, completion success, or KV policy. It never invokes a model or executes
trajectory commands.

```sh
python3 experiments/agent_execution_observer/structure_slicing.py \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --output artifacts/observer-structure-slicing-NEW
```

The default demonstration selects SuperJSON seed1004 (57 assistant Steps) and
Terminal Bench dna-assembly (21 Steps). Repeat `--trajectory ID` to explicitly
select other traces. Outputs must go to a new directory.

## Source and type rules

- Frozen methodology and TB parser are hash-checked against the source bundle.
  Original messages are compared with the saved raw request/response.
- DeepSWE uses content-block types, roles, tool IDs/names, and exact task-file
  matching. TaskCreate/TaskUpdate are the `progress_operation` subtype of
  `tool_call`; a progress operation is emitted once, retaining its parent in
  `also_types`.
- TB thinking uses `reasoning_content`. The frozen parser and its saved harness
  overlay are replayed using **only the current assistant reply**. `analysis`
  remains analysis even if its contents look like code or commands.
- Parsed core fields must equal their saved derived values. The old export's
  `replay_matches_observed_feedback` and `outcome` are explicitly not inputs:
  they can depend on following feedback.
- Invalid TB replies stay `assistant_text`; commands are not invented from
  regex matches inside invalid JSON or prose.
- TB commands are represented at **batch-array granularity**: one nonempty,
  parser-accepted `commands` array is one `command_request` block. Its decoded
  value retains every command/parameter, including empty-keystroke wait entries.
- `task_complete=true` is a declaration, not evaluation success. A command
  request is not proof the command executed or completed.
- Feedback recognizers operate only at known harness feedback positions and
  frozen framing boundaries. `execution_status` overrides its enclosing
  `terminal_observation`, which is retained in `also_types`, without double
  counting the same raw text.

## Coordinates and representation

`blocks.jsonl` stores the type, source JSONL line, message index, JSON Pointer,
optional Unicode `[begin,end)` span, original content, and recognition basis.

For valid TB JSON, a small lexical mapper locates top-level value lexemes in
the **original encoded JSON string**. The raw span includes JSON quoting and
escaping; `decoded_value` is the separately identified parser projection, not
the same character coordinate system. Duplicate-key resolution follows the
frozen parser/json.loads last-key behavior.

If the frozen parser repaired an incomplete reply, exact original field spans
may not be recoverable. The raw container and derived fields are exported in
different layers; no fictitious original character spans are assigned.

`protocol_unclassified`, `unclassified_input`, and `unmapped_wire_container` are
auxiliary preservation categories, **not additional supported semantic types**.
They keep JSON punctuation/unknown input intact rather than forcing everything
into the 17 requested categories. Raw text partitions are non-overlapping;
parser projections are a separate view and must not be added to raw-text size.

The bundle contains structured system/tools/response fields, not a freshly
rendered model token stream. The adapter does not pretend it reconstructed
literal `<tools>`, `<tool_call>`, `<think>` tokens or token offsets. Exact rendered
token alignment is a separate task requiring the original template/protocol and
tokenizer.

## Outputs

- `preview.html`: type gallery, per-Step structural chains, all original blocks.
- `blocks.jsonl` / `blocks.csv`: machine-readable blocks and review table.
- `step-flows.jsonl`: structural order, not goal-stage transitions.
- `session-audit.jsonl`: executor audit records with **session-only** attribution;
  these are not forced into any Step or mixed into per-Step causal inputs.
- `manifest.json`: selected inputs, hashes, parser replay audit and output hashes.

`rule_slicing.py` is the earlier, separate **workflow candidate** prototype. Its
22 proposed slices are not this structural layer, not semantic ground truth,
and are not consumed by `structure_slicing.py`. Existing results are preserved.
