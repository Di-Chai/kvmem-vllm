# Qualification boundaries

This is a development plugin, not a production support declaration. Installation
and discovery alone leave the serving engine unchanged. Engine hooks require
explicit flags and pinned source hashes; no installed vLLM files are rewritten.

| Area | Current evidence | Still required |
| --- | --- | --- |
| Engine | vLLM 0.30.0 V2 runner; native scheduling, batching, attention and sampling | Additional engine versions and graph modes |
| Target weights | Qwen3.8-27B GGUF Q8_0 through pinned vllm-gguf-plugin | Other quantizations and model families |
| Target KV | Native FP8 e4m3 layouts; raw pre-RoPE K authority and FP8 V; byte-checked assembly | Other native KV codecs and scale layouts |
| Hybrid state | Qwen3.5-family GDN native state identity restore/replay | Cancellation and session persistence; corrected bounded real C1 passes |
| DFlash2 | Native generation, accepted/rejected capture and identity replay | Bounded target/draft refresh and rollback |
| Bounded target | Scheduled semantic prefill/selection, actual-length acknowledgment, page trimming implemented | Corrected real long C1 passes; long C4 and pressure qualification remain |
| GPU admission | Future growth reservations tested with native full-attention/Mamba allocator | Attached to bounded scheduler; native GPU pressure tests remain |
| Storage | Immutable archive, generation-qualified leases, packed transfers and fenced failures | Per-request disk/CPU quota admission, background GC, asynchronous retirement |
| Measurements | Exact frozen-token inputs, SSE arrival/usage validation, C1/C4 aggregation | Native SSE C1/short-C4 passes; matched warm repeats and executed agent-task success remain |

BF16 is used for activations and the draft model; it does not change the Q8_0
target weight path. Initial integration requires one GPU/rank, eager execution,
text-only input, prefix caching disabled and Mamba cache mode `none`. Bounded
refresh currently requires synchronous target-only execution. Public connector
metadata carries commands/acknowledgments; isolated runner/scheduler adapters
supply the missing historical selection, replay and bounded-allocation semantics.
