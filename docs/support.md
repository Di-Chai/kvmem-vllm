# Qualification boundaries

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. Corrected model qualification is pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

This is a development plugin, not a production support declaration. Installation
and discovery alone leave the serving engine unchanged. Engine hooks require
explicit flags and pinned source hashes; no installed vLLM files are rewritten.

| Area | Current evidence | Still required |
| --- | --- | --- |
| Engine | vLLM 0.30.0 V2 runner; native scheduling, batching, attention and sampling | Additional engine versions and graph modes |
| Target weights | Qwen3.8-27B GGUF Q8_0 through pinned vllm-gguf-plugin | Other quantizations and model families |
| Target KV | Native FP8 e4m3 layouts; raw pre-RoPE K authority and FP8 V; byte-checked assembly | Other native KV codecs and scale layouts |
| Hybrid state | Qwen3.5-family GDN native state identity restore/replay | Cancellation and session persistence; corrected bounded real C1 passes |
| DFlash2 | Native generation, accepted/rejected capture and identity replay; 197K C1 compressed prefill / later decode and short C4 pass | Long compressed C4, decode-time refresh, terminal-prefix rollback and persistent sessions |
| Bounded target | Scheduled semantic prefill/selection, actual-length acknowledgment, page trimming implemented | Corrected real long C1/C4 pass; pressure tests and performance acceptance remain |
| GPU admission | Future growth reservations tested with native full-attention/Mamba allocator | Attached to bounded scheduler; native GPU pressure tests remain |
| Storage | Immutable archive, generation-qualified leases, packed transfers and fenced failures; direct pinned reads pass GPU byte/failure tests | Direct-read serving performance, per-request disk/CPU quota admission, background GC, asynchronous retirement |
| Measurements | Exact frozen-token inputs, SSE arrival/usage validation, C1/C4 aggregation | Native SSE C1/short-C4 passes; matched warm repeats and executed agent-task success remain |

BF16 is used for activations and the draft model; it does not change the Q8_0
target weight path. Initial integration requires one GPU/rank, eager execution,
text-only input, prefix caching disabled and Mamba cache mode `none`. Bounded
refresh has target-only long C1/C4 and experimental DFlash2 long-C1 evidence.
`--bounded-dflash` uses synchronous DFlash2 with native BF16 sliding draft KV;
long DFlash C4 and decode-time refresh qualification are still pending. Public connector
metadata carries commands/acknowledgments; isolated runner/scheduler adapters
supply the missing historical selection, replay and bounded-allocation semantics.
