# vllm-kvmem

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. A corrected long-C4 lifecycle run now passes; matched
performance and executed task-quality acceptance remain pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

Work in progress: bounded long-context KVMem for native vLLM batched execution.
This repository currently contains the extracted host core, its Python bindings,
request transactions, async NVMe storage, GPU transfer/FP8 codecs, recurrent
checkpoints, MeanK scoring and fused raw-Q/K capture. Versioned vLLM hooks qualify
the draft loader, scheduler/worker identity refresh, and continuous target capture
with native DFlash rejection counts. Raw K and FP8 V use bounded pinned transfers,
mutable partial tails and immutable complete archive records. Actual native GDN
identity restore/replay is qualified. A long compressed replay completed byte
checks, but its recurrent metadata classification required a subsequent correction;
corrected bounded target serving has now passed one real 197K-history C1 lifecycle
check and subsequent short C4. Native long C4 also passes lifecycle checks, with
severe prefill/retirement interference. DFlash2 compressed prefill plus decode
now passes a corrected long C4 lifecycle check. Exact terminal archive clipping
and async retirement pass short C1/C4 with full archive checksums. An opt-in Q8
integer Tensor Core adapter improves a short fixed-output C4 test from 88.20 to
173.11 aggregate arrival tok/s (1.96x); output streams differ, so this is not a
task-quality acceptance. On four frozen long histories, observed whole-window
arrival decode rises from 10.44 to 18.62 tok/s and service time falls from
1862.80 to 761.16 seconds. These runs retain Q8_0/FP8 and K131072/G65536;
mixed prefill/refresh blocking remains severe and changed outputs still need
executed task-quality checks. Controlled long-context repeats, decode-time
refresh, capacity pressure and persistent-session qualification remain pending.
**Production bounded KVMem serving is not implemented yet. Installing the plugin
does not enable historical reselection; the experimental paths require explicit
diagnostic flags.** See [development milestones](docs/development.md) and
[qualification boundaries](docs/support.md).

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
The opt-in [exact prompt-session path](docs/prompt-sessions.md) now has scheduler/
worker wiring and host tests; its separate actual-model GPU gate is still pending.
