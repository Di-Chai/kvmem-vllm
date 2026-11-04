# Local web console

The Linux console brings KVMem's configuration, chat, monitoring and log pages
to the direct `vllm-kvmem serve` entry. It retains the original dark/blue themes,
Chinese/English labels, browser-local chat history, configuration drafts and
personal templates. Changing tabs keeps the streamed chat alive.

## Start

Install the inference environment described in [installation](installation.md).
Run the console with that environment's Python so the engine and tokenizer use
the same dependencies:

```sh
.venv/bin/vllm-kvmem ui --config local-profile.json
```

Open `http://127.0.0.1:18400/`. The console does not load a model until you select
**Start service**. With no `--config`, the initial form uses the portable MTP
starter; enter your local model and tokenizer directories before checking it.

```sh
.venv/bin/vllm-kvmem ui --config local-profile.json \
  --state-dir runs/local-console --port 18400 --no-browser
```

The console and inference engine need different ports. The default engine port
is 8000. The management console listens only on loopback; for a remote Linux
host, forward its port with SSH and open the forwarded address locally.

The default state directory is `$XDG_STATE_HOME/kvmem-vllm`, or
`~/.local/state/kvmem-vllm`. It holds the saved profile, launch inputs and logs.
Each launch gets a separate engine run directory. Browser drafts, templates and
messages are stored separately in that browser; changing the origin/port creates
a separate browser storage namespace. Export chat history before clearing site
data. History import/export uses the original chat page's controls.

## Configuration and process ownership

**Check configuration** invokes the same `serve.build_launch()` used by the CLI,
including local path and parameter validation. Its preview shows the resolved
profile and launch arguments. It does not load weights or prove GPU fit.

The form edits a native serving JSON profile. Its advanced JSON editor preserves
settings beyond the basic fields. Model, tokenizer and HF config paths remain
separate; safetensors requires `hf_config_path` to match the model directory.
Logical context, selected history K, generation reserve G and concurrency are
separate limits. See [serving](serving.md) before changing their relationship.

The console owns only engines it starts. An occupied inference port is an error;
it does not adopt or terminate an existing server. Closing a browser tab leaves
the console and model running. Stopping the console process cleans up its owned
engine process group. The console remains available after an engine failure;
open **Backend logs** to inspect the failure and retry after fixing the profile.

Logs support filtering, pause, follow, raw/highlight views and export. The browser
receives at most the latest 2,000 lines / 256 KiB; the launch directory retains
the complete log. An inherited `VLLM_API_KEY` is used internally for engine calls
and redacted from the log API. The console's per-run mutation token and same-origin
checks are for its local management surface, not a shared multi-user service.

## Chat and long contexts

The bridge uses the selected local tokenizer/chat template and sends integer
token prompts to native `/v1/completions`, including the latest user query's
token interval when request planning is enabled. This supports the rolling
profile's required token-ID interface. A user query must still fit the profile's
atomic query replay budget; a large pasted message is not silently truncated.

Each browser/conversation has its own session namespace. The bridge verifies
the returned accepted token IDs and usage before recording an exact prefix.
On continuation, the freshly rendered template must match that prefix. An edit,
template change or incompatible reasoning serialization releases the old session
and explicitly reports a canonical context rebuild. It never claims a cache hit
just because a conversation ID matched. Requests in one conversation serialize.

**Stop** cancels the upstream request, including during prefill. An interrupted
response is not committed as a verified prefix. Deleting a conversation releases
its engine session as well as removing browser history; release must finish after
any active request is cancelled. Browser messages survive reloads and engine
restarts. The bridge's accepted token state and engine KV remain in-memory and
are rebuilt after restarting the service.

This first console supports text system/user/assistant messages, sampling and
the Qwen thinking toggle. It does not execute tools, accept images, or expose the
older backends' thinking-budget, multi-GPU layer placement or systemd controls.
Direct multi-user and elastic gateway profiles have separate contracts and are
not console backends. Use the current [support matrix](support.md) for model and
hardware boundaries.

## Monitoring

The monitor combines console request metadata, native vLLM `/metrics` and actual
host/GPU readings. Request details cover requests submitted through this console;
native engine counters can also include external API clients. GPU binding is
identified from the owned process tree, not guessed from GPU load.

Configured memory budgets are not measured usage. Missing KV archive/restore
measurements display as unavailable. The logical token count is not GPU-resident
KV. Any displayed chat rate derived from streamed accepted tokens is an observed
delivery rate after the first batch, not a native GPU kernel timing. Single-batch
answers do not establish a decode rate.

## Build and validation

Built static assets are included in the Python package; end users need no Node.js.
The maintained front-end sources are under `ui/`, with provenance and license
notices for the pinned KVMem and llama.cpp UI sources. To rebuild after editing:

```sh
python scripts/build_webui.py
```

The build needs Node.js 22+ and the locked npm dependencies. It performs Svelte type
checking and emits static files under `src/vllm_kvmem/webui/static/`.

Focused CPU checks cover configuration validation, real child-process cleanup,
the HTTP boundary, stream cancellation, token/session contracts and telemetry:

```sh
PYTHONPATH=src python -m pytest -q tests/test_webui_manager.py \
  tests/test_webui_chat.py tests/test_webui_http_chat.py tests/test_webui_server.py
```

The browser suite exercises the packaged pages with a disposable service:

```sh
npm --prefix ui ci
CHROMIUM=/path/to/chromium npm --prefix ui test
```

For a manual check through the real console and chat bridge with a deliberately
fake tokenizer and HTTP model (no GPU), run:

```sh
PYTHONPATH=src python tests/test_webui_http_chat.py --port 18401
```

These checks do not qualify a GPU/model combination. A real-model acceptance run
must also check startup, two-turn reuse, rolling output, cancellation and release
on the intended inference machine.
