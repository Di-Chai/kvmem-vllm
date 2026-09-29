# Installation and experimental serving

This is a pinned development integration, not a production release. The actual
model gates currently cover Qwen3.8-27B GGUF Q8_0, target FP8 e4m3, DFlash2 BF16
draft/native sliding KV, TP=1, eager synchronous execution and text-only requests.
The kernel/source adapters require vLLM 0.30.0. They do not rewrite installed vLLM
files. Broader model, quantization, KV codec and engine versions need separate
adapters and qualification; installing the package does not establish support.

## Install from the independent repository

On the qualified Linux/CUDA 13/RTX PRO 6000 Blackwell setup:

```sh
cd ~
git clone https://github.com/Di-Chai/kvmem-vllm.git
cd kvmem-vllm
bash scripts/setup_gpu.sh
```

The setup script installs the pinned Python/GPU requirements and the independent
GGUF plugin revision recorded in `requirements-cu130.lock` and the script. It uses
`uv` and expects CUDA at `/usr/local/cuda-13`. It builds for SM120. Model files are
external and must be provided separately. A different CUDA/GPU environment needs
its own dependency/build validation; the host-only core can be installed without
torch or vLLM using `uv pip install '.[test]'` in a Python 3.12 environment.

The entry point `vllm.general_plugins:kvmem` is registered by package installation.
Discovery alone is passive. The explicit launcher supplies the pinned runner,
metadata, bounded scheduler and public KV connector settings needed for serving.

## Configure and inspect a launch

Copy `examples/qwen-q8-fp8-dflash.json` to a local profile and replace the model
paths. Relative paths resolve against the profile directory; `~` is expanded.
The `tokenizer` and `hf_config_path` folders supply tokenizer/model configuration,
not BF16 target weights. `mm_proj` is required by the pinned GGUF loader even though
this profile enables language-only execution. The draft uses its BF16 safetensors.

```sh
cp examples/qwen-q8-fp8-dflash.json local-profile.json
# Edit the five model/config paths in local-profile.json.
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/session-01 --dry-run
```

The dry run validates local paths and options, prints the exact native argument
vector and environment, and neither imports torch/vLLM nor creates the run
directory. It does not allocate GPU memory or prove that the model fits. The
example preserves K131072/G65536 and B128, uses a 36-GiB shared read cache and a
four-entry/1-GiB CPU prompt-checkpoint pool. These are separate limits, not a total
host-RAM quota. NVMe arena geometry includes both active and retained authorities;
physical disk reservation and shared I/O-byte controls are opt-in public connector
options. The example leaves repository GC and physical reservation disabled because
their serving GPU qualification is still pending.

`q8_tensorcore` defaults to false. Setting it to true enables the experimental
SM120 batched Q8 integer Tensor Core adapter with the measured speed gains and
remaining quality limitations described in `support.md`. It keeps target Q8_0
weights and FP8 KV; it does change floating-point reduction order.

## Start and use the API

```sh
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/session-01
```

After reinstalling the updated package, the console command `vllm-kvmem serve`
is equivalent. vLLM owns the API and process lifecycle. The new run directory must
not already exist, so another server's archive or readiness marker cannot be
reused accidentally. `launch.json` records the resolved profile and argument vector;
`capture/` contains the runtime lifecycle/authority evidence and NVMe arena.

The profile defaults to `127.0.0.1:8000` and model name `kvmem`. A normal OpenAI
completion/chat request works without a session. To request exact-prefix reuse,
provide a stable `session_id` body field or native `X-Session-ID` header:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Session-ID: task-01' \
  -d '{"model":"kvmem","messages":[{"role":"user","content":"Explain this repository."}],"max_tokens":128}'
```

A short request may not create a checkpoint. Reuse requires an exact matching
token prefix and a sealed non-final prefill snapshot; a session header by itself
does not prove a cache hit. Continue sending the full intended conversation.
Do not remove old messages or alter task input to manufacture reuse. The serving
gateway must supply an appropriate `cache_salt`/session scope for its clients;
these fields are namespace identifiers, not authentication.

The profile installs `vllm_kvmem.http_sessions.validate_prompt_session` through
vLLM's public `--middleware` option. Session IDs and salts used for reuse are
limited to 1024 characters and rejected with HTTP 400 before scheduler dispatch.
Body/header/`vllm_xargs` precedence matches the pinned engine. Native ASGI host
tests cover rejection, concurrent valid requests and unchanged streaming bodies;
actual-model qualification of this new guard remains pending. Direct custom
launches must include the middleware explicitly to obtain the same validation.

Serving mode distinguishes native warmup (no connector metadata) from the first
real scheduler message and enables accepted capture before that first forward.
It omits the one-off diagnostic tensor snapshot used by the original probe tools.
There is no API-ready/externally-created-marker race. **This new profile entry and
automatic-arming change currently have 20 passing host tests; actual-model launch
qualification remains pending.** Existing short/134K/C4 session GPU evidence used
the frozen qualification launcher and must not be relabeled as this entry's test.

The public entry also passes an isolated source-package check recorded in
`artifacts/source-build-serving_01`: rebuild the native wheel from an extracted
sdist with no extension binary, install into a fresh environment without torch or
vLLM, and run 73 host checks (2 skipped). Both console help and a real-path console
dry run pass without creating a runtime directory. The source archive includes
the example profile; the installed wheel contains the console entry and all three
licenses. This verifies host installation, not GPU startup or performance.

For measured support and test commands, see `prompt-sessions.md`, `support.md` and
`qualification-20260928.md`. Current agent evaluation remains incomplete.
