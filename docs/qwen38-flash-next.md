# Qwen3.8 Flash Next FP4-Local

The `qwen38_flash_next_b12x` adapter adds selected-window execution to the
NVIDIA Qwen3.8 Flash Next FP4-Local checkpoint. This model identifies itself as
`qwen4_exp_text`; its QSA attention and PLE state require separate handling from
the Qwen3.5/3.8 dense 27B path.

Use the explicit [Flash Next profile](../examples/serve-qwen38-flash-next-fp4-local.json).
The default configuration is one GPU, text only, full CUDA graph decode, synchronous
scheduling, FP8 E4M3 KV, native MTP with three draft tokens, PLE CPU offload,
and K=8192/G=4096. Logical context is 262144 tokens. Prefix caching is disabled
and the native recurrent cache uses `none` mode.
Prefill chunks are capped at 4096 tokens, and window assembly batches up to
32 archive blocks. `eager` remains available for diagnosis. Graph capture uses
compilation mode 0; Torch-compiled/piecewise QSA capture is not qualified.

## Runtime

The qualified interfaces are from
`ghcr.io/local-inference-lab/vllm:release-10152026`, digest
`sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`:
vLLM `0.1.dev21830+g8244d1c44.cu134`, CUDA 13.4 and the packaged B12X kernels.
The checkpoint revision used for validation is
`nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300`.
An ordinary vLLM wheel is not interchangeable with this vendor image.

Set `model`, `tokenizer` and `hf_config_path` in a copy of the profile to the
downloaded checkpoint directory as seen inside the container. Install/build
KVMem's native extension for that environment, then launch from the repository:

```sh
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
PYTHONPATH=src python -m vllm_kvmem serve \
  --config /absolute/path/to/flash-next-local.json \
  --run-dir /absolute/path/to/flash-next-run
```

The launcher selects the vendor `instanttensor` loader and B12X backends,
enables PLE CPU offload, disables overlapping input projections, and selects
the DS convolution-state layout. It does not apply the unrelated DFlash loader
patch. Structural checks reject incompatible QSA state/cache APIs. The small
configuration adapter recognizes only KVMem's in-request control connector;
it does not claim support for importing external QSA prefix checkpoints.

## Cache and state

Main target/MTP keys use KVMem's existing raw FP8 archive and dense RoPE rebuild.
The QSA selector needs its original pre-normalization, pre-RoPE projection.
Those BF16 bytes are retained losslessly in auxiliary archive planes, under the
same accepted-prefix publication, cancellation and retirement lifecycle as KV.
Auxiliary planes do not participate in MeanK scoring.

On a selection, the adapter rebuilds main KV and QSA's compressed selector keys
for the selected dense positions using the native B12X compression kernel.
It rebuilds the final selector ring and interval anchor after rolling back the
provisional query. Query rollback includes GDN convolution/SSM state, PLE's
dilated convolution, QSA raw rings/tags/anchors, and acceptance/freshness state.
MTP uses the same selection as the target and archives only its accepted prefix.
Its text MRoPE input also uses that dense frame: the vendor model state's
`get_model_positions()` otherwise returns logical-history positions from its
separate RopeState. Persistent target/MTP raw buffers are written by graph
replay; independent speculative draft graphs cannot overwrite the prefix.
Selection, rollback, reconstruction and ownership changes remain outside graphs.

The first archive format uses padded auxiliary planes. With 12 target attention
layers plus one MTP layer, its payload is **26624 bytes per historical token**:
about **6.5 GiB per 262144-token request**, before indexes, transfer buffers,
allocator overhead and PLE host weights. The scheduler's CPU history accounting
includes these bytes. A configured CPU read-cache size is not a process RAM cap.
The example reserves a shared 24 GiB CPU history budget and a 512 MiB transfer
budget, sufficient for the payload of three full 256K histories. Admission still
uses actual requested token counts and current ownership.

The example's 90% GPU utilization is an engine allocation budget that leaves
headroom for query scoring and recurrent rollback snapshots. K=8192 does
not itself make the engine release its reserved native KV arena. This profile
does not establish a minimum GPU-memory requirement.

## Boundaries

The adapter rejects Torch-compiled execution, asynchronous scheduling or
refresh, session restoration, multimodal input, multi-GPU execution, and the
unqualified capture/assembly optimizations listed in `validate_qsa_config`.
Rolling long-output and native Action-attention scoring are not enabled.
Cross-request reuse through ElasticConnector is a separate path and is not
qualified by this implementation.

Use token-ID completions with an explicit `kvmem_query_span` for query-based
selection, as described in [serving](serving.md). A query span must fit the
replay budget; it should identify the question rather than the entire history.
Selected context can change outputs relative to native full-context attention.

## Validation

The GPU tests exercise the actual B12X projection, padded FP8 cache views and
compression kernel. They check native projection equivalence, lossless BF16
selector bytes, nonadjacent-history reconstruction, owned request state, null
page protection, and MTP-prefix capture isolation. Host tests cover state
rollback/canonicalization, admission accounting and unsupported configurations.
Full-model validation records are stored in
`artifacts/flash-next-kvmem-adaptation-20261031-r1`.

The actual FP4-Local checkpoint was exercised on an RTX PRO 6000 Blackwell
Server Edition with the initial eager configuration (1024-token prefill chunks):

| Check | Observed result |
| --- | --- |
| Short completion and 32768-token selected-context completion | Correct expected answers |
| 32768-token input followed by a 100-integer JSON array | All integers correct; actual MTP rejections observed |
| Three concurrent 32768-token requests, each with a different 100-integer array | All three outputs correct; three requests observed in one execution batch |
| Three concurrent 261632-token inputs | All three completed with the expected distinct answers |
| Cancellation and subsequent request | Aborted request retired; repeated follow-up output matched its independent control |
| QSA reconstruction and provisional query rollback | 245 rebuilds with matching selector/window lengths; all 14 query rollback audits passed, including 1041 state tensors for a three-request batch |

All 16 requests retired, including two cancellations. All 32776 sealed archive
records passed checksum verification, and active requests, retiring requests,
pending transfers and owned archive slots returned to zero. The focused host
suite passed 234 tests (one skipped); both native GPU tests passed, including
pre-RoPE capture with the checkpoint's actual MRoPE configuration.

These are execution and lifecycle checks, not a general quality or throughput
qualification. One 16384-token arithmetic input asking for `37*19+7` returned
`706` instead of `710`. It did so both after cancellation and as an independent
request; the short version returned `710`. This rules out a cancellation-only
failure in that sample, but does not establish the cause or parity with native
full-context attention. The failed samples are retained as `after-cancel.json`
and `cancel-control-long.json`. Treat this as an experimental compatibility
profile with an observed long-input answer-quality failure.

Native full-context vLLM measurements and KVMem's selected-window measurements
have different execution and attention semantics; they are not interchangeable.

## Phase throughput after optimization

Measurements on the same RTX PRO 6000 and FP4-Local checkpoint are retained in
`artifacts/flash-next-kvmem-performance-20261031-r1`. Each request has exactly
258048 input tokens (252K); three requests start together, with identical prompt
hashes across the compared configurations. KV stays FP8 E4M3 and PLE stays on CPU.

| Phase, aggregate across three requests | Initial eager configuration | Final configuration |
| --- | ---: | ---: |
| Effective prefill, input tokens/s | 2602.1 | 4305.1 |
| Wall time until every request has a first token | 297.51 s | 179.82 s |
| Sustained decode, accepted output tokens/s | 16.89 | 253.92 |
| Common measured decode interval | 169.01 s | 38.62 s |
| Average GPU utilization during that decode interval | 6.45% | 68.21% |

Final per-request decode rates are 76.31, 92.10 and 85.50 tokens/s. Each request
generated all 4096 requested output tokens. Decode counts accepted streamed
token IDs only, inside the interval after all three first tokens and before
the first completion, trimming two seconds at both ends. It includes normal
KVMem work during that interval. The eager comparison generated 1024 tokens
per request; these are different interval lengths, not a confidence interval.

The final independent prefill trial generates one token per request. Its rate
is all input tokens divided by the shared wall time through the last first
token, including queueing, KVMem maintenance and first-token sampling. The eager
reference uses that same first-token definition from its 1024-output run.
The optimized sustained run changed assembly batching during prefill, so its
prefill number is excluded. Assembly was fixed at 32 blocks for over 22 seconds
before its measured decode interval; the final prefill trial uses that setting
throughout. Both final long trials had zero preemptions and a sampled GPU peak
of 90770 MiB (88.64 GiB). Prefix caching is disabled.

The main decode bottleneck was CPU dispatch through many eager B12X operations.
A five-step diagnostic trace recorded 2.446 s of self CPU time versus 0.228 s
of self CUDA time; profiling overhead means these are diagnostic figures, not
throughput measurements. Stable eager decode had no device allocation/free
churn. Full graph replay reduced median three-request steps from 444.27 ms to
28.62 ms. The other changes increase the prefill chunk cap from 1024 to 4096
and correct the MTP MRoPE frame described above. Increasing assembly to 256
blocks made CPU concatenation slower; reverting to 32 improved the independent
long prefill trial from 3500.0 to 4305.1 tokens/s.

The final profile uses full graphs only for the qualified native execution
path; selection and state transactions remain uncaptured. Three concurrent
32768-token requests returned complete, correct 100-integer arrays. The focused
host suite passed 182 tests, and four native GPU tests passed, including graph
replay of raw projections, prefix isolation and native MTP MRoPE broadcasting.
The optimized run retired all 20 requests, including cancellation, and verified
80726 sealed archive records. All 618 reconstructions had matching selector
extents; all 18 query rollback audits were byte-equal. There were 2089 actual
target full-graph replays. Active requests, retirement, pending I/O and owned
archive slots returned to zero.

The documented 16K arithmetic failure **still occurs in the optimized profile**:
both after cancellation and in its independent control, `37*19+7` returned
`706`; the short control returned `710`. The cancellation-isolation check
passed, but this performance improvement does not resolve that quality issue.
Results are retained in `graph-cancellation-summary.json` in the performance
artifact directory.

The reusable phase benchmark is `scripts/benchmark_qsa_phases.py`. Run separate
one-output and 4096-output trials with `--input-tokens 258048 --concurrency 3`
and fresh `--output-dir` directories. Wait for `capture/archive.json` to show
zero active/retiring requests, owned slots and pending I/O before starting the
next trial. Single final trials do not establish performance across arbitrary
prompts or quality parity with full-context attention.
