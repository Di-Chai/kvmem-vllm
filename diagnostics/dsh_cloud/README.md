# Cloud DSH single-trajectory validation

This harness targets the cloud Qwen3.8-27B backend through the local C1 bridge at
`http://127.0.0.1:18387/v1`. It does not start a backend or bridge. `prepare.py`,
`submit.py`, and the default `launch.py` mode do not contact a model.

The provider is `gc-local`, model `kvmem-gc-qwen38`, output limit 8192. The initial
configuration used logical context 24576, K=4096, G=4096, query replay budget 128,
and global bridge max-inflight=1. **Startup capacity and a successful real request
are separate checks:** the first marker request with that backend profile failed
in GPU window assembly with an allocation error. The failure is preserved in
`/root/data/kvmem-vllm-gc/runs/dsh-live-01/marker`. Do not treat 24576 as proven
runtime capacity; pass the backend's subsequently validated limit when preparing
a fresh run. Changing the provider limit alone does not change backend capacity.
The subsequent 16384-context backend completed the real marker case in
`dsh-live-02/marker`; its saved validator report proves the read tool, its returned
value and the final answer. This is a functional result, not a maximum-capacity
or throughput measurement.

## Prepare and launch

All commands below run on the cloud machine. The owner of the backend launches
one case at a time, after confirming the backend/bridge are ready.

```sh
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/prepare.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-live-02 \
  --context-window 16384
```

16384 is an example to match a validated backend, not a measured result. The run
root must not exist. Preparation creates independent `marker/home`, `csgo/home`,
and their working directories, plus immutable, hashed configuration snapshots.
It neither copies a personal DSH home nor loads external provider credentials.
The random marker value is in `marker/workspace/marker.txt`; the expected value
is outside the working directory in `marker/expected.json`. The task itself does
not contain the answer.

```sh
# Print the launch plan without starting DSH.
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/launch.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-live-02 --case marker

# Explicit real model/tool execution, using the original official headless runner.
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/launch.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-live-02 --case marker --execute
```

Success requires exit 0, the expected final marker, and durable evidence of the
`read` tool's call and result. An answer matching a marker alone is not tool-use
evidence. Launch writes `stdout.log`, `stderr.log`, `pid.json`, and `exit.json`;
DSH stores the full durable session beneath that case's isolated home. A launch
cannot be silently reused: `launch.json` is created exclusively. Preserve failed
runs and prepare a new root for retries.

## CSGO and followup repairs in one session

```sh
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/launch.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-live-02 --case csgo --execute
```

This remains in the foreground until a close command or an error. Its initial
queued request is exactly `做个网页版的csgo。`. The owner may supervise this process
in the background. Wait for `csgo/status.json` to report `waiting` with
`next_index: 2`; inspect the first result and generated files before adding the
actual repair request:

```sh
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/submit.py \
  /root/data/kvmem-vllm-gc/runs/dsh-live-02/csgo --task-file /absolute/path/repair.txt

# After the final turn and lifecycle evidence have been collected:
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/submit.py \
  /root/data/kvmem-vllm-gc/runs/dsh-live-02/csgo --close
```

Submission only writes an atomic queue file. The runner executes each user
message sequentially and waits for `agent.whenIdle()` before advancing. It calls
official `agents.create()` once, uses the same Agent and Session for every
`followup()`, and persists `source.kind=user` for the installed Query-provenance
hook. `csgo/session.json` identifies the shared session; `turns/*.result.json`
records the sequence boundaries, result and outcome; `turns/*.events.json`
preserves each turn's durable events. A failed turn stops the runner instead of
silently submitting later queued tasks. `close` flushes the session and exits; it
does not itself assert that native backend KV resources have been released.

Do not launch another `csgo` process to continue a task: the built-in one-shot
headless runner creates a new Session on every invocation. Keep the existing
driver alive and use `submit.py` for all repairs.

## Additional independent tasks

A fresh preparation can add one named task alongside the unchanged marker and
CSGO cases. Supply a UTF-8 task file chosen by the operator; preparation does not
generate the application or contact a model.

```sh
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/prepare.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-expense-01 \
  --context-window 32768 --case-name expense-tracker \
  --task-file /absolute/path/expense-tracker.txt

# Inspect the plan; add --execute only when this case is authorized to run.
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/launch.py \
  --run-root /root/data/kvmem-vllm-gc/runs/dsh-expense-01 --case expense-tracker
```

The context value must match the validated backend and bridge.
`--select-budget` declares K and `--gen-budget` declares native segment G; both
default to 4096 and must be positive multiples of 32, with K+G fitting the
declared context. Preparation records `native_k` and `native_g` in the manifest;
it does not reconfigure the backend or inspect a running service.

`--max-output-tokens N` changes the new provider snapshot and manifest.
`--max-output-segments S` declares the matching bridge limit (1–4). If omitted,
the minimum `ceil(N/G)` is recorded as `bridge_max_output_segments`; N must fit
G*S. The old default N8192/G4096 therefore declares two segments. Pass the same
G and segment count to the bridge before execution. The provider cap remains
subject to the installed pi-ai client's context estimate and its own 4096-token
reserve, which is independent of native G. Existing prepared runs are unchanged.

For the requested large-history experiment, prepare with
`--context-window 262144 --select-budget 36864 --gen-budget 16384`,
`--max-output-tokens 16384 --max-output-segments 1` and
`--stream-idle-timeout-ms 900000`. Use matching bridge
`--gen-budget 16384 --max-output-segments 1 --max-model-len 262144`.
A separate explicit N32768/S2 configuration is supported by CPU tests, but is
not the selected experiment. These arguments do not qualify GPU capacity or
change task text, thinking mode or sampling.

The bridge retains the complete prompt and accepted IDs. When fewer than G
logical tokens remain, the first segment uses that positive remainder; later
segments likewise use `min(G, effective_output_cap - accepted_output_tokens)`.
Only a zero/negative remainder fails logical admission. A boundary-limited
text response may finish with `length`; an incomplete tool still fails the
authoritative parser and receives no accepted-history commit or complete tool
response. Requested/effective caps and logical remainder are recorded before
the model call. Native Query/capacity checks and cancellation remain unchanged.

Each added case gets its own workspace, home and Session. Its copied task and
initial queue entry are fingerprinted in `manifest.json`, as are the default
cases in newly prepared runs. Changing the source task file later cannot alter
the snapshot. Launch rejects changed task/configuration snapshots and unknown
case names; older two-case manifests remain supported without modification.
The custom case uses the same queue runner as CSGO: continue with
`submit.py CASE_DIR --task-file FOLLOWUP` and finish with
`submit.py CASE_DIR --close`. Do not launch it twice or launch multiple cases
concurrently. A new case also requires releasing the previous native owner when
backend session capacity is one.

CPU-only CLI checks mock every process launch:

```sh
python3 -m unittest discover \
  -s /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud \
  -p test_prepare_launch.py
```

## Identical main-agent composition

`single_trajectory.patch.yml` is the exact overlay used for the successful
first-request capture. It disables model-generated session titles, automatic and
manual compaction, retries, subagents/forks, workflow/Ralph, and autonomous goal
rounds. Shell, files, editing and ordinary main-agent tools remain available.
The captured request contained these 15 tools:

`bash`, `edit`, `exit_plan_mode`, `glob`, `grep`, `job_kill`, `job_list`,
`job_output`, `read`, `read_image`, `skill`, `str_replace_editor`, `todo_write`,
`web_search`, `write`.

`session_runner.patch.yml` changes only the lifecycle: disables the official
`headless-runner` and inserts the queue runner. A boot-free config comparison in
`dsh-live-01/runner-profile-equivalence.json` found exactly those two changed rows;
all tool, system prompt and provider rows were identical. The CSGO runner has no
tool registration or prompt injection. Its queue guarantees serial user turns;
the bridge's separate global C1 gate must still enforce the backend request limit.

The repository runner patch is a template. `prepare.py` replaces its placeholder
with an absolute **literal string** pointing to the copied runner. Cordis does
not evaluate a `!!js` expression in the plugin `name` field as a config value.
Do not pass the unprepared template directly to DSH. The runner's official
`agents.create({setup})` callback must return `undefined`: `installModelSelection`
returns a disposer, while the Agent creation API interprets a non-void setup
return as a transaction and calls its `commit()` method. The explicit block
callback deliberately discards the disposer, matching official headless code;
its listeners are already scoped to the Agent context.

The real loader guard test at `dsh-runner-boot-guard-01/validation.json` reached
the new plugin and exited before Agent creation. The stronger integration at
`dsh-runner-fake-provider-01/validation.json` created an actual Agent and captured
its first complete request at the fake provider on port 18388. It verified all
15 tool schemas against the earlier capture, then returned the intentional 503:
one durable session, zero model output messages, zero executed tools and no
request to the real model. Its `integration-launch.json` records exact commands;
the fake provider and DSH both exited. This closes the loader and Agent setup
gaps that a mocked lifecycle test alone cannot cover. The real-route Query hook
remains specific to port 18387 and is validated by the real marker evidence.

The current Web bundle is not substituted: it disables the flat base-agent rows
and selects session agent presets, and its runtime adds Web-surface context. A
Web RPC test would need a matching custom preset, removal of surface prompt
differences, and a new request capture/token count before comparing workloads.

## Request capture and token-budget evidence

`capture_first_request.py` is a standard-library loopback-only fake provider. It
stores exactly one complete matching OpenAI request and returns HTTP 503, with
no generated output and no upstream connection. Use a fresh isolated run and a
capture-specific provider port. Do not bind the real bridge's occupied port.

```sh
python3 /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/capture_first_request.py \
  --output /absolute/new/capture-directory \
  --expected-task-file /absolute/task.txt --port 18388
```

The archived exact-route capture at
`/root/data/kvmem-vllm-gc/runs/dsh-first-request-capture-live-route-01` has a
byte-exact body, provenance validation and CPU token accounting. Its 3-message,
15-tool request was 4369 input tokens; the genuine human query occupied
`[4249,4256)` (7 tokens), while the later runtime-context user message was excluded.
Native block replay required 32 tokens within the reserved budget of 128. These
numbers describe that captured working directory and runtime context; recapture
or retokenize any changed prompt, tool schema or workspace before claiming an
exact budget for it. The captured provider advertised 32768; the prepared formal
provider is smaller and must match the chosen backend.

Full DSH persistence files contain concatenated Zstandard frames. A single
Node `zstdDecompressSync` call may return only the first frame: use the DSH frame
scanner or decode every frame before judging missing tool/assistant events.

## Validate a completed marker case

```sh
/root/data/kvmem-vllm-gc/.venv/bin/python \
  /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/validate_marker.py \
  --case-dir /root/data/kvmem-vllm-gc/runs/dsh-live-02/marker \
  --bridge-output /root/data/kvmem-vllm-gc/runs/dsh-bridge-02 \
  --tokenizer /root/data/kvmem-vllm-gc/models/Qwen3.8-27B-NVFP4 \
  --output /root/data/kvmem-vllm-gc/runs/dsh-live-02/marker/validation.json
```

This is read-only except for the specified report. It uses system libzstd to
decode all frames, and only local tokenizer assets for CPU prefix reconstruction.
No model, network client or GPU operation is started. Missing exit/commit evidence
reports `pending` (exit 2); inconsistent evidence reports `failed` (exit 1);
success reports `passed` (exit 0). It checks the actual `read` call and its matching
successful result, exact final marker and exit status, both committed bridge
responses, the owner derived from DSH session/system/tools, and Query provenance
against the durable human message rather than the later runtime context.

The successful marker02 report decoded 28 Zstd frames. Its model responses
accepted 85 then 78 tokens; the second native prompt reconstructed the preceding
4396 prompt tokens plus the exact 85 accepted IDs as a 4481-token prefix. The
validator checks native/segment/commit IDs agree and retokenizes only according
to the saved audited protocol. It does not infer GPU page reuse from that prefix.
Any included GC counters come from the saved `lifecycle-before-release.json`,
never later live server state. This case does not prove proactive statistical GC.

## Pinned runtime and CPU checks

`locks/` preserves the installation manifest, pnpm lock and overrides, package
versions and archive hashes. It contains no model files, personal home or
credentials. Runtime paths are supplied to `prepare.py --harness`; the current
installation is Node 24.19.0, pnpm 11.1.2, CLI 0.1.0-rc.7 and pinned DSH packages.
The route-specific Query hook is maintained by the repository's official-adapter
installer, not by this driver. Its installed-state digest is recorded separately
from the original pre-hook installation verification.

```sh
DSH_GC_HARNESS=/root/data/kvmem-vllm-gc/dsh-harness \
  /root/data/kvmem-vllm-gc/dsh-harness/node-v24.19.0-linux-x64/bin/node --test \
  /root/data/kvmem-vllm-gc/repo/diagnostics/dsh_cloud/test_session_runner.mjs
```

The CPU tests inject a fake Agent, never start a DSH service or tool executor,
and do not contact a model. They cover same-session serial followups and genuine
user provenance, stopping after turn failure, rejecting an invalid queue index,
and refusing to create an Agent without the launch guard. They do not establish
real backend correctness or GPU memory capacity.
The setup mock also invokes the actual model-selection callback and checks its
void return. `test_validate_marker.py` covers concatenated frames, truncated
trailing data, pending runs and failed exit status. A saved negative check at
`dsh-live-02/marker/validator-negative-check.json` altered one accepted ID in a
temporary copy; validation rejected it against the untouched native evidence.

## Strictly buffered tools and the provider idle timeout

The installed official DSH pi-ai adapter measures idle time between **semantic
harness events**, not TCP bytes. Its pi-ai OpenAI parser drops SSE comments,
empty deltas and empty content. A long tool body can therefore be generating
normally while the bridge correctly withholds its arguments until full parsing.
At 27 tokens/s, an 8192-token reply alone takes about 303 seconds, already longer
than the adapter's default 300-second idle allowance. This estimate is not a
latency guarantee; prefill, refresh and lower decode rates add time.

For a fresh long-tool experiment, opt into the existing public provider setting:

```sh
python diagnostics/dsh_cloud/prepare.py \
  --run-root /root/data/kvmem-vllm-gc/runs/NEW-RUN \
  --context-window 32768 --select-budget 8192 --max-output-tokens 8192 \
  --stream-idle-timeout-ms 900000
```

Preparation snapshots `streamIdleTimeoutMs: 900000`, records
`stream_idle_timeout_ms` in the manifest, and fingerprints the provider file.
The default remains 300000 ms. Values must be integer milliseconds from 1 to
2147483647. Existing prepared runs remain immutable; prepare a new run instead
of changing a live profile. This setting needs no backend/bridge restart.

This is a finite **semantic-idle allowance**, not a progress-aware transport
watchdog, a total request deadline, or proof the model is still progressing.
A completely stalled request can now wait up to the larger allowance; caller
cancellation still propagates immediately. No fabricated semantic event,
partial tool argument, bridge parser change or installed SDK patch is added.
A future progress-aware watchdog needs an explicit transport-progress hook
through pi-ai and the DSH adapter; keep it distinct from accepted semantic
history and do not renew it from periodic comments alone.

The CPU integration check imports the installed official adapter, calls its
public `apply()` profile resolver, and uses the actual pi-ai SDK against an
ephemeral **loopback fake HTTP provider**. It creates no agent, runs no tools,
and contacts no real bridge/model. Use the active adapter path from the locked
harness (the query-hook-patched adapter is also compatible):

```sh
/path/to/pinned/node diagnostics/dsh_cloud/test_pi_ai_idle_timeout.mjs \
  --adapter /absolute/path/to/@deepseek-ai/dsh-llm-pi-ai/lib/index.js \
  --output /tmp/FRESH-pi-ai-idle-result.json
```

The eight cases cover fast completion; comments, empty deltas and empty content
failing to reset a short watchdog; successful finite extension with byte-for-byte
equal semantic chunks/replay metadata and model request body; genuine text
progress rearming; silent-stall timeout; and caller-abort upstream closure.
Timer tests use short real intervals after a warm-up, with an independent
3-second abort bound per case. The fake output includes a complete `write`
call only at completion; no file is written. The report records the adapter
source hash. The Python configuration and bridge regressions also cover frozen
configuration, exact accepted IDs, prefix reuse, system pins and cancellation.


## Native generation budget and logical-limit CPU regression

Run the selected regression files from the repository root with only the local
tokenizer (the upstream transport is a fake HTTP transport):

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KVMEM_QWEN38_TOKENIZER=/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4 \
PYTHONPATH=src:. /root/data/kvmem-vllm-gc/.venv/bin/python -m pytest -q \
  diagnostics/dsh_cloud/test_prepare_launch.py \
  tests/test_dsh_output_budget.py tests/test_dsh_bridge_admission.py \
  tests/test_dsh_system_pin.py tests/test_dsh_model_protocol.py
```

The isolated candidate passed **87 tests and 36 subtests**, with no skips.
Coverage includes unchanged default G4096/cap8192/two-segment preparation;
explicit G16384 with one 16K or two 16K segments; rejected K/G/cap mismatch;
first-segment positive and exhausted logical remainders; complete versus
context-truncated tools in streaming and nonstreaming routes; a complete
tool spanning 16384 accepted IDs plus a short final segment; actual G-aware
system-pin admission and cross-G restore rejection; and the existing C1,
upstream cancellation, commit recovery and protocol checks. The tests use
the real local tokenizer and CPU planner, but no model inference or GPU.
