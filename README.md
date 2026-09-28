# vllm-kvmem

Work in progress: bounded long-context KVMem for native vLLM batched execution.
This repository currently contains the extracted host core, its Python bindings,
request transactions, async NVMe storage, GPU transfer/FP8 codecs, recurrent
checkpoints, MeanK scoring and fused raw-Q/K capture. Versioned vLLM hooks qualify
the draft loader, scheduler/worker identity refresh, and continuous target capture
with native DFlash rejection counts. Raw K and FP8 V use bounded pinned transfers,
mutable partial tails and immutable complete archive records.
**End-to-end KVMem serving is not implemented yet. Installing the plugin does not
enable historical reselection.** See [development milestones](docs/development.md).

## Host core development

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest tests/test_core.py
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core -j2
ctest --test-dir build/core --output-on-failure
```

Linux and a C++17 compiler are required. The host core does not require torch or
CUDA. Source provenance and the original tests are preserved under `third_party`.

## Engine qualification

Pinned vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0 and GGUF plugin revision
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Q8_0 is supplied by that independent
vLLM quantization plugin. Model weights are external; never included in this repo.

For a GGUF target with a safetensors DFlash draft, the pinned V2 runner needs the
explicit draft load configuration fix. Opt in using:

```sh
python -m vllm_kvmem --draft-load-compat vllm serve ...
```

This performs a source-hash-checked import adaptation of the loader only, leaves
installed vLLM files unchanged, and propagates through the plugin into workers.
It is not an unmodified upstream reference. See `src/vllm_kvmem/compat.py` for
upstream provenance. No performance or KVMem functionality is implied by this fix.

GPU setup on this CUDA 13 / SM120 machine: `bash scripts/setup_gpu.sh`.
Explicit GPU component tests: `KVMEM_RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/test_codec_gpu.py tests/test_capture_gpu.py`.
Run only when the GPU is available; this command is not a model performance test.
See [qualification results and limitations](docs/qualification-20260928.md).
