# DSH system/tool frame preservation experiment

The bridge's optional `--pin-system --select-budget 8192` preserves the audited
Qwen system frame, including the tool schemas and protocol instructions. It
requires `--tokenizer-protocol --require-query-provenance`. The default remains
no extra pins. It does not edit or summarize the user's prompt.

TokenizerProtocol identifies the complete first system frame from its audited
chat-template framing. Planner converts its byte extent using exact accepted
prefix coordinates; the bridge independently verifies the corresponding token
prefix against that frame. The resulting span uses the existing `system` kind,
with implicit mandatory parts disabled. Native continuation segments retain the
same span and original Query epoch. The bridge does not suppress an old Query
before the native session ledger verifies its accepted prefix.

Pin policy is recorded alongside tokenizer provenance, including K, G,
block_tokens=32, sink_blocks=1, recent_blocks=1, query_replay_budget=128 and two
alignment/speculation blocks. Restoring accepted history under a different pin
mode or K/G fails; a recorded pin that differs from the reconstructed frame also
fails. Pin mode requires verified agent Query provenance and auto reselection.

Every request and continuation segment is checked before its model call. The
union of system and sink blocks, one recent block, the full 128-token replay
budget and two alignment blocks must fit the explicitly declared K. The actual
Query replay must fit 128 tokens, and each segment's output must fit G. Admission
proofs are saved in plan.json and each segment report. Missing system frames,
unknown coordinates and insufficient declared capacity fail closed.

This is a restricted bridge experiment, **not a universal server hard-pin
guarantee**. Native `system` priority can still be softened by capacity fitting.
The backend profile must match the declared controls, with no guided scratch;
its actual restore/selection budget remains authoritative. For nonrolling
execution, maximum output reserve is bounded by G; the bridge also reserves two
alignment blocks. This does not prove every possible checkpoint/pressure path.
Verify complete system coverage in worker `selected_history_ranges` and
`mandatory_fit` traces before calling a GPU trajectory qualified. A future
backend strict-pin contract is separate work.

The saved GPTQ run06 first request has 4357 prompt tokens, system/tool span
[0,4226), and human Query [4229,4236). Its system span occupies 133 blocks.
Bridge admission requires 4480 selected tokens including reserves: K4096 is
rejected before any model call; K8192 passes. At a 16K history the host native
fitter retains only 112/133 system blocks under K4096, versus 133/133 under
K8192. These CPU observations do not establish the cause of earlier malformed
tool output. K and pin policy must be compared with matched model/task controls.

The saved server-03 GPTQ native 32K/.91 profile has 68 physical KV pages. K8192/G4096 needs
eight full-attention pages, six Mamba/MTP pages and one null page (15 total), so
its KV reservation fits. Larger activation/staging/checkpoint allocations still
require a new live profile qualification.

The CPU tests exercise this exact CLI entrypoint with the server binder
intercepted, plus fake upstream requests, actual accepted-ID continuation,
restore compatibility, provenance rejection and both tokenizer thinking modes:

```sh
python -m scripts.dsh_bridge --tokenizer-protocol \
  --tokenizer /root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4 \
  --disable-thinking --pin-system --select-budget 8192 \
  --require-query-provenance --gen-budget 4096 --max-model-len 32768 \
  --max-inflight 1 --max-output-segments 2 \
  --backend http://127.0.0.1:8186 --model kvmem-gc-qwen38 \
  --owner-namespace dsh-system-pin --port 18387 --output NEW_OWNED_DIRECTORY
```
