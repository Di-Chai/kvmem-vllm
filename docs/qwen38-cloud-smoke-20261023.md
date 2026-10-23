# Qwen3.8-27B single-request smoke on the cloud RTX 4090

NVIDIA Qwen3.8-27B NVFP4 completed one plain vLLM request on a single RTX 4090
after the isolated CUDA build environment was repaired. No other GPU, CPU weight
offload, MTP, KVMem or GC was used in this smoke.

The prompt was “计算 37 × 19，只用一句中文给出结果。” and the response was
“37乘以19等于703。”. There were 27 prompt tokens and 11 generated tokens,
including the stop token. Time to first token was 0.135 seconds; the generation
call took 0.889 seconds. The sampled GPU memory peak, including loading, was
23,781 MiB. This short response does not establish sustained throughput or quality.

The fixed revision, exact settings, metrics, preceding failed attempts and
environment are in [the report](benchmarks/qwen38-4090-smoke-20261023.json).
All 19 model files were downloaded through hf-mirror and verified against their
manifest digests.

## Reproduce on the development cloud machine

Use the isolated Python environment at /root/data/kvmem-vllm-gc/.venv.
The repository dependency lock alone does not create a complete matching CUDA
development toolkit: NVCC 13.4 and torch runtime headers 13.0 share a prefix.
The tested standalone toolkit uses NVCC/CRT/NVVM 13.0.88, runtime 13.0.96 and
CCCL 13.0.85. It also needs CUDA math-library development headers,
lib64 pointing to lib, and libcudart.so pointing to libcudart.so.13.
The cloud setup records the precise overlay in
logs/cuda130-library-overlay-20261023.json.

    source /root/data/kvmem-vllm-gc/tools/cuda130-env.sh
    mkdir -p /root/data/kvmem-vllm-gc/runs/single-request-reproduction
    export CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS='' HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MAX_JOBS=4
    export GC_SMOKE_MODEL=/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-NVFP4
    export GC_SMOKE_OUT=/root/data/kvmem-vllm-gc/runs/single-request-reproduction
    /root/data/kvmem-vllm-gc/.venv/bin/python scripts/smoke_qwen38_single_request.py

Run one process and one request at a time, with a new output directory per
attempt. GPU peak memory collection is external to this small runner.

## KVMem/MTP and GC throughput

The initial MTP integration attempt ran out of GPU memory before serving any
request. The opt-in VLLM_KVMEM_MTP_SHARED_LOAD=1 adapter now avoids constructing
temporary duplicate embedding and output modules. Parameter-free placeholders
are skipped by the original weight loader; the official post-load sharing
attaches the target modules only after draft loading and quantization finish.
The adapter checks pinned vLLM source hashes, matching local target/draft
checkpoints, one rank, shared identities and unchanged target tensor metadata.
Twenty CPU tests cover the real loader/postprocessing/sharing seams.

With this adapter, NVFP4 plus MTP1 loaded in 19.91 GiB and completed the C1
integration pilot and formal five-pair sweep. Startup still emitted nonfatal
allocator OOM retries during weight repacking; those recovered before requests.

The formal sweep passed its predeclared 1% throughput-loss margin:

| Pair | GC off token/s | GC delete token/s | Paired throughput loss |
|---|---:|---:|---:|
| 1 | 26.2997 | 26.2023 | +0.371% |
| 2 | 26.4470 | 26.2645 | +0.690% |
| 3 | 26.1091 | 26.1391 | −0.115% |
| 4 | 26.2321 | 26.4408 | −0.796% |
| 5 | 26.2241 | 26.3940 | −0.648% |

Positive loss means slower with GC. Median paired loss was −0.115%; the
descriptive paired-bootstrap 95% interval was [−0.796%, +0.690%]. Its upper
bound was below 1%. This supports the tested workload, not a universal
zero-overhead claim.

All ten measured outputs were identical. Every DELETE request retired and
CPU-released four candidate blocks during generation, with further accepted
tokens after the observed release interval. OFF retired none. The top-level
lifecycle CPU-release count includes cleanup (16 for DELETE); the within-decode
candidate count is four and is the relevant evidence.

Each fresh server ran a 512-token warmup, then one measured 512-token request
from the same fixed 256-token input. The five pair orders alternated. The
integration pilot was excluded, and no pairs were added after observing the
formal result. Downloads and other tests were stopped during measurement.
The engine used C1, eager execution, MTP1, K256/G1024 and no refresh during the
measured segment. All ten servers exited with code zero.

The [complete report](benchmarks/qwen38-4090-gc-throughput-20261023.json) contains
the exact profile, frozen input, revisions/digests, rates, lifecycle counts and
bounded release witnesses. To reproduce, extract reproduction_profile and
frozen_input to separate JSON files, use the matching CUDA environment above,
unset VLLM_PLUGINS (the plain smoke disabled plugins), enable
VLLM_KVMEM_MTP_SHARED_LOAD=1, and run:

    python scripts/qualify_historical_gc.py --config PROFILE.json --input INPUT.json --output NEW_RUN_DIRECTORY --kind performance --baseline off --statistical-fixture --decode-tokens 512 --pairs 5

The qualifier requires an idle GPU and owns each server. A pass requires at
least five pairs and a confidence-interval upper bound at most 1%; an interval
crossing the threshold is inconclusive. It does not append samples until a
pass is obtained.

## Scope and remaining validation

OFF and DELETE share a synthetic evidence producer and counter sampling.
Three injected observations with importance 0.01 authorize a controlled
four-block candidate range; these are not three measured semantic epochs.
This compares net decode throughput with native policy and archive reclamation.
OFF saves terminal M (3.29–3.49 ms host time); DELETE defers M and uses P224.
The timing includes that terminal-path difference and excludes next-turn replay
cost, so it does not independently isolate every GC cost. It does
not measure a real semantic estimator, model judge, R-KV redundancy scoring,
block-graph execution or p95/p99 latency.

The GPU window stays unchanged during the measurement. Identical outputs do
not establish quality after a later retrieval omits deleted content. Real DSH
task quality and a genuine evidence producer require separate validation.

Qwen3.8-27B float32 GDN state needs about 147.75 MiB per checkpoint. The 512 MiB
pool successfully held P and M in OFF. After DELETE, terminal M can be deferred
because the GPU window still contains retired historical handles; this sweep
does not qualify successful M restoration after GC.

The smaller sokada4/Qwen3.8-27B-GPTQ-Int4 fallback at revision
2232b67a2ef5480e97c4450838a712de1346eccc was also downloaded through hf-mirror
and all 16 files verified. Its five weight files total 19,559,451,008 bytes.
It subsequently passed the same plain single-request smoke, answering
“37 乘以 19 等于 703。”. Model loading used 16.83 GiB versus NVFP4 19.09 GiB;
the sampled whole-process GPU peak was 23,155 MiB. The 27-token input produced
15 output tokens including stop, with 0.139 s time to first token and 1.162 s
for the generation call. Different response lengths make these short timings
unsuitable as a sustained speed comparison. The [GPTQ smoke report](benchmarks/qwen38-4090-gptq-smoke-20261023.json)
records the exact settings and evidence digests. The actual loaded tokenizer
backend and chat template match NVFP4, while the padding token ID differs.
