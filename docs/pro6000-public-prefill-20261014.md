# RTX PRO 6000 Blackwell 27B public prefill references

Checked 2026-10-14 while the frozen 256K NIAH Action-attention campaign was
running. The local target is `nvidia/Qwen3.8-27B-NVFP4`; its architecture class
is `Qwen3_5ForConditionalGeneration`. The local GPU is the 96 GB Server Edition,
600 W. NVFP4 is a floating-point format and must not be labeled INT4.

| First-party measurement | Model / format | Workload | Reported input throughput |
| --- | --- | --- | ---: |
| [DataPacket, 2026-10-06](https://www.datapacket.com/blog/self-hosted-llm-inference-rtx-pro-6000-blackwell) | Qwen3.8-27B NVFP4, one RTX PRO 6000 | 100 requests, 4096 input + 1 output; actual active concurrency not specified for this row | 17,100 token/s |
| [Deep-AI-Evo, 2026-08-15](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark) | Unsloth Qwen3.8-27B NVFP4, MTP k=2 | Single request, about 29.5K input, input / TTFT | 9,915 token/s |
| Same source | Same | Single request, about 88.9K input | 6,766 token/s |
| Same source | Same | Single request, about 177.4K input | 4,496 token/s |
| [Prefill Systems, 2026-09-06/07](https://prefillsystems.com/benchmarks/qwen3-8-rtx-pro-6000) | Earlier pinned Inferact Qwen3.8-27B NVFP4, FP8 KV, MTP k=3 | C8 sustained service, mean 8050 input + 87 output, public network/gateway path | 290,808 input token/min = 4,846.8 token/s |

The last workload reports 52 aggregate output token/s and median TTFT 6.423 s.
Its separate 32K-input C8 workload reports 17 output token/s and median TTFT
33.5 s. These are mixed service measurements, not isolated prefill kernels.
The DataPacket paper also reports 2,710 aggregate output token/s for a single
NVFP4 27B GPU, but its decode workload is 1000 requests of **128 input + 1024
output**, so that figure is not a 256K-context decode result.

A [direct comparison by Benjamin Marie, 2026-09-23](https://kaitchup.substack.com/p/fastest-qwen38-27b-quantization-benchmarking)
tests AWQ INT4, NVFP4 and other formats on one 96 GB RTX PRO 6000, with 8K/64K
inputs, 32K outputs, and concurrency 1/16. The accessible introduction says
Unsloth NVFP4 wins most configurations; NVIDIA NVFP4 has the lowest mean TTFT
at 64K/C1. Numerical tables are subscriber-only and were **not accessed or
inferred**. No verified public numeric result for exactly **27B + INT4 + C8 +
256K** was found in this search.

The current local 256K campaign's mid-run 4.8K–5.7K input token/s counts logical
historical progress across requests and includes refresh/CPU maintenance and
interleaved generation. The public 4K batch result is evidence of substantially
higher throughput on a different workload, not proof of a 3x local regression.
A causal overhead comparison needs matched local prompts, KV precision,
scheduling and prefill-only accounting, with and without KVMem. No settings
were changed during the frozen NIAH run to match these external benchmarks.
