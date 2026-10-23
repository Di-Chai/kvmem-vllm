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

## KVMem/GC status

The first KVMem MTP1 integration pilot ran out of GPU memory while constructing
the MTP output layer, before any request or throughput measurement. This does
not qualify KVMem or GC throughput. A smaller GPTQ quantization and MTP
shared-weight allocation are being investigated.

The paired qualification tool supports --statistical-fixture for C1
stable-window decode. OFF and DELETE share the synthetic evidence producer and
counter sampling. A positive result isolates incremental native policy and CPU
archive reclamation cost; it does not measure a real semantic estimator.
Reclamation must be witnessed inside generation and followed by more accepted
tokens. Paired outputs must match, and refresh is rejected during measurement.
A paired bootstrap confidence interval is reported separately from the median
threshold.

Qwen3.8-27B float32 GDN state needs about 147.75 MiB per checkpoint.
Use a 512 MiB session checkpoint budget for P and M together; 128 MiB is too small.
This fixture does not validate successful terminal M capture after deletion of
handles still present in the GPU window.
