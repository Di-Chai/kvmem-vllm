# qw3 优化核对、原版 MeanK＋原生 MTP，2026-10-14

目标是保持原版 KVMem MeanK 管理、已接受历史和恢复正确性，使八路长文
decode 相比约 401 token/s 提升至少 20%。本轮已接入 Qwen 原生 MTP 和
异步封页，并测量 DFlash 草稿长度。最终八路 64K 历史两轮已通过，稳态
decode 486.27／492.02 token/s，中位数 489.15，较此前同样 MeanK 的 DFlash
401.32 高约 21.89%。仍保留较低候选成绩和一轮并发背压失败记录。

## qw3 核对和移植范围

参照本机 `/home/chaidi/qw3/src/qwen_executor.cpp` 的原实现，保持原版 selector
和 MeanK 评分规则。vLLM 的主模型投影、验证、采样和 attention 算子保持原生。

| qw3 机制 | 本仓库状态 | 对 decode 的关系 |
| --- | --- | --- |
| 新封口页增量 writeback，约 15493 行 | 上轮已移植未提交页队列和 resident 计数 | 消除随历史长度增长的目录扫描 |
| CPU gather 常驻 worker、双缓冲，约 14103／6447 行 | 上轮已移植有界并行 gather、共享 staging 和分离的 DMA／consumer fence | 主要改善刷新和 prefill |
| pinned write batch 持有源和 repository lease；后台 CPU scatter／write，约 15151–15370 行 | 本轮将封页快照、SHA 和 CPU cache admission 放到有界 archive worker；native memcpy 释放 GIL | 降低模型线程封页停顿；不减少后台工作总量 |
| MTP 原始 K 捕获、tiered MTP 和 target window 同步，约 27305／26949 行 | 本轮将一层原生 MTP 的 raw K／FP8 V 与 target 一同归档和装配 | 同时保留原生 MTP CUDA Graph 和历史压缩 |
| 基于增量资源状态管理页预算 | 本轮优化集成层的 future-page credit：每轮调度统一核对，分配／释放增量更新 | 减少 KVMem 引入的重复页占用扫描 |
| raw-K GPU cache，约 6320–6445 行 | 尚未移植 | 节约刷新 H2D，但消耗额外 GPU budget；不是本次共同 decode 窗口的直接瓶颈 |
| resident V 复用与分离 raw-K／V stage-in，约 14402 行 | 尚未移植 | 主要改善选页重装配 |
| MTP prefix-prime KV-only 快路径，约 27366 行 | 尚未修改原生 proposer 的计算流程 | 主要针对 prefill；本轮先使用原生 MTP |
| 全部 accepted-row CPU scatter／MeanK 维护移到后台 | 封页已异步；accepted-row copy 和 target MeanK sum 仍在模型线程 | 后续需处理 terminal prefix 裁剪和并发读取边界，不能只把当前 drain 放入线程 |

封页 admission 同时保留源张量和不可变 snapshot 的字节额度。buffer export 和
generation lease 保持到 I/O 完成，失败不发布 archived 状态。MeanK 查询可在
GPU 上执行时与封页后台工作重叠；选页和 packed read 之前仍执行发布屏障。
该屏障也重试因其他 namespace 填满共享队列而尚未提交的完整页。

## 原生 MTP 的上下文管理

目标为当前 Qwen3.8-27B ModelOpt NVFP4／FP8 混合 checkpoint，target 和 MTP
KV 均为 FP8 E4M3，激活为 BF16，单 rank、APC off、同步调度、GDN cache mode
`none`、FULL_DECODE_ONLY。主模型 16 层 full attention 和 1 层 MTP attention
共同形成 immutable raw-K／FP8-V 页；MTP 的后续 speculative 单 token forward
不覆盖其 prefix capture。只归档 verifier 已接受的 prefix。

**MeanK 索引只维护并评分主模型 16 层。** 原 selector 产生一个历史页计划，
target 与 MTP 一起 re-RoPE 到该计划的 dense working window。MTP proposer 使用
该 dense position／seq-len；scheduler 和 GDN 保持完整 logical history。MTP KV
随工作窗口 checkpoint 装配恢复，不另外保留一份完整 GPU 历史。

原生 MTP 继承 target 的 FP8 CacheConfig，包括已解析的 LBNHC 布局。本版本
显式重复设置 draft KV dtype 会复制尚未解析的布局，启动失败，因此不传重复
override；实际 native KV group、dtype、draft layer 和 query 长度仍逐项校验。
源码版本通过固定哈希检查，未编辑安装的 vLLM 文件。

测试机器为单张 RTX PRO 6000 Blackwell Server 96 GB；运行版本为 vLLM
0.30.0、PyTorch 2.13.0+cu130、FlashInfer 0.6.18.post1、CUDA 13。配置使用
max_num_seqs=8、max_num_batched_tokens=4224、GPU utilization=0.82，NVFP4
和 FP8 linear backend 均为 CUTLASS。

CPU 每页从 target-only 的 6 MiB 增为 6.375 MiB；256K 历史为每用户 12.75 GiB，
八用户共 102 GiB raw-K／V。全局 CPU cache 仍限制为 100 GiB，超出部分保留在
NVMe；MeanK index、checkpoint、临时 source／snapshot 和系统内存另行计费。
namespace cache 上限按实际历史页几何计算，避免沿用旧 12 GiB 常量。

## 参数候选测量

同一个冻结 32768-token 输入、八路直连 SSE、每路 4096 输出，温度 0，seed
104729，unique cache salts、APC off、无其他模型请求。共同窗口前后各裁 128
输出 token，使用实际已接受 token IDs 计数，并由 engine running／waiting、
prompt counter 和 preemption 指标确认八路同时 decode。固定输出使用 ignore_eos。

| 候选 | 已完成次数 | 稳态 decode，token/s |
| --- | ---: | ---: |
| DFlash2，k=1，上轮发布代码 | 1 | 304.48 |
| DFlash2，k=3，上轮发布代码新基线 | 1 | 398.21 |
| DFlash2，k=7，上轮发布代码 | 1 | 446.51 |
| MeanK＋MTP，k=2，async seal／target-only sum | 1 | 469.97 |
| MeanK＋MTP，k=3，初版 async seal | 3 | 459.90／514.13／519.75；中位数 514.13 |

候选共 56/56 请求成功、零抢占。k=3 初版 MTP 的中位数比新 DFlash k=3
基线高约 29.1%，但输出 hash 和接受率随 batch 组合变化。DFlash／MTP 是不同
推测方法，不能把该差异全部归因于 CPU 优化或声称全局最优。两次较高 MTP
运行的接受比例均为约 50.44%；首轮为 46.54%，较低首轮保留。

## CPU 组件与正确性

单 CPU thread、7 次交错测量、预分配源、真实 generation／namespace admission
和 CPU cache，NVMe write 用 discard-only 代替，不包含模型／GPU：

| 封页操作 | 同步前台中位时间 | 异步前台中位时间 | 异步全部完成中位时间 |
| --- | ---: | ---: | ---: |
| 1 个 6 MiB 页 | 3.968 ms | 0.113 ms | 9.434 ms |
| 8 个 6 MiB 页 | 80.988 ms | 0.255 ms | 86.444 ms |

这些数值证明模型线程的提交耗时下降，不代表 seal 工作本身或持久化完成更快。
每次 snapshot 的 SHA 与独立编码匹配。同步／异步总量不能外推为模型 token/s。

GPU 页 credit 组件使用真实 full-attention manager、四个 group、1600-token 页、
八个 request，排除 GDN／GPU／I/O：每轮 owned 扫描由 64 次减为 16 次，CPU
中位时间 0.257 → 0.124 ms。信用在调度开始重新核对以包含窗口 trim／import，
当前 request 分配失败后的 recycling、free 和 deferred free 均更新信用。

MTP 初版通过 4200＋257 与 65536＋257 输入／输出后追加 32＋128 的恢复字节
验证。64K checkpoint 计算历史为 65792，working tokens 为 34048；17 层 working
KV 共 1,185,415,168 字节以及 96 个 GDN state tensor 一致。MTP 的 KV 已包括在
working 字节中，所以独立 sliding draft snapshot 为 0 层。验收明确跳过 cold
greedy 对照；该项证明状态字节，不证明整个任务质量。

首轮最终代码的八用户 64K 测试失败：归档队列满时，complete CPU record 可留在
unsubmitted 队列；原屏障只等待已提交 jobs，packed assembly 随后拒绝未发布页。
新增跨 namespace 背压复现测试，并让发布屏障重试这些页。该失败保留在证据中。
修复后两轮八用户 64K＋4K 均通过，16/16 请求成功、零抢占，engine 指标确认
完整八路共同 decode；普通性能配置关闭 profiler 和额外恢复 oracle。

| 最终版本，八用户 64K＋4K | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| 稳态 decode，token/s | 486.27 | 492.02 |
| 整轮输出吞吐，token/s | 142.88 | 144.18 |
| TTFT p95，秒 | 160.38 | 159.12 |
| 整轮完成，秒 | 229.35 | 227.28 |

上一轮 DFlash k=3＋MeanK 为 401.32 steady、120.71 whole、187.36 秒 TTFT p95。
本轮切换推测方法和 CPU 管理，两轮中位数的 steady／whole 分别约提升 21.89%
和 18.90%；不能将这些增幅全部归因于单个 CPU 组件。第 1 轮前段 prefill
与 CPU 回归检查重叠；检查在共同 decode 前结束，第 2 轮没有该重叠，因此
TTFT 与 whole 的部署比较优先采用第 2 轮。
TTFT 从网关内部的后端 SSE 测量；当前公开 turn 接口在完成时返回 JSON，
该数值不代表客户端已收到公开流式首 token。

560 项 CPU 回归及 32 项 GPU 捕获／装配／codec／恢复回归通过。
最终代码再次通过真实 64K 暂停／恢复 oracle：17 层 working KV（其中 MTP
一层）1,185,415,168 字节及 96 个 recurrent tensor 完全一致，保留 65792 个
已计算历史 token，追加后的前端历史为 65953。该验证关闭 cold greedy 对照。
普通服务也通过两轮带鉴权的工具等待／恢复请求：历史从 272 增至 292 token，
每轮成功生成 16 token，结束后 active、queue 和 restore 预算全部归零；该服务
关闭 profiler 和额外状态 oracle。

八用户完整 32768 输入＋16384 输出测试也通过，8/8、零抢占，共生成
131072 token。稳态 decode 为 513.65 token/s，whole 为 413.50 token/s，
完成 316.98 秒，TTFT p95 36.96 秒。该组覆盖上下文从 32K 增至 48K，普通
端点上限保持 16K，没有临时放宽网关限制。与旧 MeanK＋BF16 DFlash k=3／
CUTLASS 的同长度 421.12 token/s 相比提高约 21.98%；旧完整上下文 MTP 的
591.92 仍更高，不能把不同调度和上下文路径的差异全部归因于 KVMem。
本轮验收覆盖固定长度吞吐及状态恢复；持续工具负载的硬 SLO、跨进程重启
恢复和完整 agent 任务质量仍需单独测试。

## 配置和证据

使用 [`examples/qwen-nvfp4-meank-mtp-elastic.json`](../examples/qwen-nvfp4-meank-mtp-elastic.json)，
填写本机模型和 users 文件。`draft_method: "mtp"` 不需要外部 draft model，
`draft_tokens` 控制深度；`async_seal: true` 启用后台封页。

本地原始证据在 `artifacts/decode-opt-20261013/`，不提交凭据、模型、生成正文
或大型 trace。参数候选由 `scripts/qualify_kvmem_overhead.py` 测量；长历史通过
`scripts/qualify_elastic_long_output.py` 的真实 authenticated gateway 测量。
组件复现用 `scripts/bench_async_seal.py`、`scripts/bench_capacity_credits.py`。
全部候选、失败运行、CPU 组件和验收数据见
[公开结构化汇总](benchmarks/decode-optimization-20261014.json)。
