# Multi-user HTTP qualification

`scripts/qualify_multi_user.py` drives the authenticated native service using real
HTTP streams. Authentication and tenant isolation run as middleware in the same
vLLM service. The harness requires only the Python standard library; it never
starts or stops the inference service.

Use a local credentials file matching the service's users configuration:

```json
{"users": [
  {"id": "alice", "api_key": "replace-with-alice-key"},
  {"id": "bob", "api_key": "replace-with-bob-key"}
]}
```

Keep the credentials file outside version control. Reports retain user indexes,
not credentials, user IDs, prompts or generated text. Upstream error bodies and
exception messages are omitted because they can echo credentials.

Run a short load sweep against a ready service; replace the port and model with
the values configured for that service:

```bash
python scripts/qualify_multi_user.py \
  --api-base http://127.0.0.1:8000/v1 \
  --users-file /tmp/kvmem-users.json \
  --output /tmp/kvmem-multi-user-load \
  --model kvmem \
  --endpoint chat/completions --disable-thinking \
  --concurrency 1 4 8 16 32 --rounds 2 \
  --prompt-words 256 --output-tokens 128 --ignore-eos
```

Each client submits `rounds` sequential requests; all clients in a level start
together. A level therefore attempts `concurrency * rounds` requests. Credentials
are distributed round-robin across clients. The report records how many distinct
users actually participate; two users at concurrency 32 is not a 32-user test.
Use at least as many configured users as the largest level to test that many
distinct identities. Levels run sequentially and each request has a wall timeout.

`--prompt-words` controls filler length rather than claiming a tokenizer-independent
token count. Actual prompt/completion counts come from final stream usage.
`--output-tokens` sets the maximum generated length. `--ignore-eos` forces the
output budget for load measurement, when supported by the backend. Remove it to
measure natural completion lengths. `--disable-thinking` sends
`chat_template_kwargs.enable_thinking=false` only on the chat endpoint.

Use a separate short exact-retrieval and lifecycle run:

```bash
python scripts/qualify_multi_user.py \
  --api-base http://127.0.0.1:8000/v1 \
  --users-file /tmp/kvmem-users.json \
  --output /tmp/kvmem-multi-user-functional \
  --model kvmem \
  --endpoint chat/completions --disable-thinking \
  --concurrency 1 4 --rounds 2 \
  --prompt-words 256 --output-tokens 64 --check-markers --probes
```

`--check-markers` asks each request to return its own unique marker exactly. This
checks a synthetic retrieval task, not general model or agent quality.
`--probes` requires at least two distinct users. It sends two concurrent waves
with identical visible session IDs, request IDs and caller-supplied cache salts
across users, checking each user's private output marker and looking for other
users' markers. It then closes a long stream after the first nonempty content
chunk and retries the same session until the recovery timeout. An intentional
disconnect is recorded as `cancelled`, separately from failed load requests.

The probes demonstrate output isolation for these prompts and HTTP recovery
after a client disconnect. They do not establish native cache namespace isolation,
native abort acknowledgement, memory reclamation, exact-token equivalence or
long-context quality. Pair them with `qualify_session_concurrency.py` and backend
telemetry when qualifying those properties. Forced output length is incompatible
with exact-marker mode and probes.

## Interpreting results

The output directory must be new. `results.json` is updated after each load level,
then finalized after optional probes. Summaries are also printed to standard
output. The default passing requirement is a 100% success rate; lower
`--min-success-rate` only when an explicitly documented overload test expects
rejections.

| Field | Meaning |
| --- | --- |
| `success` | HTTP 200, complete SSE stream, final finish reason and nonempty output |
| `rejected` | HTTP 409, 429 or 503; no automatic retry in the load sweep |
| `failure` | Authentication/other HTTP errors, transport errors, timeouts or invalid/truncated streams |
| `ttft_seconds` | Dispatch to first nonempty content chunk, including admission wait and prefill |
| `success_latency_seconds` | p50/p95/max full-stream latency for successful requests |
| `all_attempt_latency_seconds` | p50/p95/max latency including rejected and failed attempts |
| `output_tokens_per_second_over_wall` | Successful output tokens divided by the whole level's wall time |
| `arrival_rate_over_start_span` | `(attempts - 1) / (last dispatch - first dispatch)`; not completed throughput |

Chunks can contain several tokens. The harness never counts chunks as tokens.
The token rate is `null` if any successful response lacks token usage. TTFT uses
visible content; reasoning-only deltas are not visible content, so use the
non-thinking recipe for comparable measurements. These clients use a closed-loop
workload, so dispatch rate is affected by response speed and rejection rate. Wall
throughput includes client startup, all response drains and the last completion.

`passed` requires a positive number of successes, no failed requests, successful
enabled marker/probe checks and the requested success rate at every level.
`capacity_limited` means a level completed some requests but fell below the
requested success rate. All requests rejected is always `failed`, even when the
configured minimum success rate is low. Exit status is 0 for `passed`, 2 for
completed but failed/capacity-limited qualification. A configured success-rate
exception does not hide rejection counts.

The first sweep is a smoke measurement. Use more rounds, a representative
mixture of prompt lengths, and enough users for deployment sizing. A passing
short synthetic run does not establish a concurrency service-level guarantee or
long-running stability.

## Decode-only measurement window

Use `scripts/qualify_decode.py` to measure output token arrivals while every
request in a cohort is generating. Its default is 1/2/4/8 independent users,
two rounds, a fixed technical prompt, 1024 output tokens per request, and at
least 64 tokens trimmed from each end before intersecting all decode windows.
Streamed token IDs must agree with final usage; speculative bursts are counted
by their token IDs. Stalls inside the common window remain in the denominator.

```bash
python scripts/qualify_decode.py \
  --api-base http://127.0.0.1:18380/v1 --model kvmem \
  --users-file artifacts/multi-user-20261013/users.json \
  --capture artifacts/multi-user-20261013/multi-server-tensorcore/capture \
  --output artifacts/decode-retest
```

The optional native capture verifies request ownership and stable decode
progress; its archive-commit rate remains approximate and separate from SSE
throughput. With capture supplied, `--min-native-coverage` defaults to 0.9;
interior prefill increments, refreshes, owner mismatches or read errors fail the
measurement. Boundary-spanning reads remain explicit ambiguous evidence.
Keep other clients off the backend during a measurement. This is
service decode throughput, including runtime overhead; it is not GPU-kernel-only
timing. Whole-request TPOT may include neighbours' initial prefill interference,
so use `steady_decode_window.output_tokens_per_second` for this comparison.
See [the measured results and limits](decode-throughput-20261013.md).

For exact long inputs, pass `--prompt-token-file path/to/tokens.json`: a JSON
object containing a flat `input_ids` array is sent directly to `completions`, and final prompt
usage must match its length. With more clients than available GPU capacity,
`--allow-queued` reports successful HTTP service separately from an unavailable
full-client decode window. Configure both client and admission queue timeouts
for the workload; the 32K × 32 experiment used 3600 seconds.

Use `scripts/analyze_decode_cohorts.py --results path/to/results.json --output
path/to/new-analysis.json` for an offline analysis of actual simultaneous
owners. It requires stable reservations matching the captured owners, completed
prompts, and no prefill or refresh interference. Read the interval coverage
alongside the rate: a brief qualified decode interval does not describe the
whole queued batch. See the [long-context and 32-client measurements](long-context-throughput-20261013.md).

## Harness tests

```bash
python -m pytest -q tests/test_multi_user_qualification.py \
  tests/test_decode_qualification.py tests/test_decode_progress_sampling.py \
  tests/test_decode_cohort_analysis.py
```

These tests use a threaded HTTP server on `127.0.0.1` with synthetic SSE, including
delayed first content, separate usage, malformed streams, explicit rejection,
parallel requests, tenant collisions and disconnect recovery. They require local
socket access. They validate the harness and its measurement lifecycle; their
timings are not inference/GPU performance evidence.
