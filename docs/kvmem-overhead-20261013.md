# KVMem 集成开销定位与 CPU／装配优化，2026-10-13

本轮固定 vLLM、目标模型和 DFlash2 配置，比较原生执行与 KVMem 执行，并针对
CPU 历史维护和 prefill KV 装配优化。实际八路 64K 测试的整轮吞吐从 99.68
升至 120.71 token/s，TTFT p95 从 241.45 降到 187.36 秒；CPU cache copy
按实际字节归一化后约快 2.67 倍。**decode 吞吐尚未证明有稳定提升**，32K
首轮优化后结果低于优化前，所有结果均保留。
此前 MTP 与 DFlash2 的跨方法比较不能直接解释为 KVMem 本身的开销，历史结果
另见 [MeanK 功能与吞吐报告](meank-throughput-20261013.md)。

## 相同 DFlash2（k=3） 配置的原生对照

两组均为八路、每路 32768 token 输入＋4096 token 固定输出，通过同一测试脚本
直连引擎 SSE；不包含弹性网关的认证、排队和会话 checkpoint。目标模型为
Qwen3.8-27B NVFP4／FP8 混合权重，目标 KV 为 FP8 E4M3；DFlash2 使用 BF16
权重及 BF16 KV，每步最多 3 个草稿 token。保持 CUTLASS 线性后端、同步调度、
prefix caching 关闭、GDN cache mode `none`、FULL_DECODE_ONLY、max-seqs=8、
batch token 上限 4224 和相同的 prefill 分块参数。

原生组只保留 DFlash loader compatibility，未启用 KVMem scheduler、connector、
raw KV capture 或窗口元数据补丁。KVMem 组启用原版 MeanK 管理，但本组输入
等于 K=32768，且生成量低于 G=16384，用于观察不需要历史压缩时的集成成本。
结构化汇总保存了实际原生控制组 argv 与环境变量；仅将三个模型路径替换成
具名占位符。环境只启用 `KVMEM_DRAFT_LOAD_COMPAT=1`、`VLLM_PLUGINS=kvmem`、
`VLLM_USE_V2_MODEL_RUNNER=1` 和 `VLLM_LOG_STATS_INTERVAL=1`，没有 profiler。

| 八路直连引擎，32K＋4K | 稳态 decode（token/s） | 整轮吞吐（token/s） | TTFT p95（秒） | 完成时间（秒） |
| --- | ---: | ---: | ---: | ---: |
| 原生 vLLM＋DFlash2（k=3） | 479.05 | 325.41 | 31.20 | 100.70 |
| KVMem，优化前 | 391.20 | 227.24 | 56.98 | 144.20 |
| KVMem，优化后第 1 次 | 339.84 | 199.96 | 49.48 | 163.88 |

三项已完成测量均为 8/8 成功、零抢占，原生指标确认共同 decode 窗口的八路并发。
优化前 KVMem 的稳态吞吐约为原生的 81.66%。这表示该配置下观测到的服务吞吐
差异，**不是已隔离出的“纯 KVMem 开销为 18.34%”**：八路输出 token ID 哈希
均与对应原生请求不同，整轮草稿接受比例也从原生的 48.85% 变为 KVMem 的
46.33%。接受比例为接受草稿数／候选草稿数，不包含每步 bonus token。
需要结合每步执行时间和组件计时判断具体开销，不能把全部 token/s 差归因于
CPU 历史写入。

优化后首轮 TTFT p95 从 56.98 降到 49.48 秒，但稳态 decode 从 391.20 降到
339.84 token/s，**本轮没有证明 decode 吞吐提升**。该轮整段草稿接受比例为
31.17%，与优化前 46.33% 差异明显。按共同窗口 SSE 输出事件推估，输出周期
从约 49.08 降到 47.71 ms，同一周期内每用户平均已接受输出却从约 2.400 降到
2.027 token。SSE 周期是客户端观测代理，不能当作独立 GPU step 计时；这些
结果说明草稿接受率与生成内容变化须与组件优化一起报告，后续复测不能删除
本次较低吞吐的记录。

稳态窗口由准确已接受 token ID 计数，每路前后裁去 128 个输出 token；窗口内
停顿仍计入分母。原生 running/waiting、prompt token 和 preemption 指标提供
并发旁证。整轮指标则包含 prefill、生成和收尾。固定输出使用 `ignore_eos`，
不是自然结束的任务质量测试。

## 64K 历史的装配开销

另以真实弹性入口运行八用户各 65536 token 输入＋4096 token 输出，K=32768，
使每路都真正从超过 active 预算的历史中选页。该表与上面的直连表使用不同入口，
不得直接相减来估算网关或 KVMem 成本。

| 八用户弹性入口，64K＋4K | 稳态 decode（token/s） | 整轮吞吐（token/s） | TTFT p95（秒） | 完成时间（秒） |
| --- | ---: | ---: | ---: | ---: |
| KVMem，优化前 | 384.65 | 99.68 | 241.45 | 328.74 |
| KVMem，优化后 | 401.32 | 120.71 | 187.36 | 271.45 |

前后均 8/8 成功、零抢占。整轮吞吐提高 21.10%，TTFT p95 下降 22.40%。
稳态 decode 观测值提高 4.33%，但整轮草稿接受比例也从 46.07% 变为 47.06%，
八路对应请求的输出哈希全部不同，不能仅用这一组推断 decode 开销稳定下降。

优化前八路全部成功、零抢占，共发生 256 次刷新，其中 254 次执行 MeanK
注意力评分。装配累计读取／上传 366,515,060,736 字节，读取全部命中 CPU cache，
没有 NVMe read syscall。累计计时为：

| prefill 装配指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| assembly 合计 | 55.21 秒 | 21.05 秒 |
| 其中 host prepare | 44.17 秒 | 17.54 秒 |
| 其中 CPU cache copy | 42.76 秒 | 16.13 秒 |
| archive read queue wait | 0.097 秒 | 0.085 秒 |
| native read call | 0.029 秒 | 0.015 秒 |

CPU cache copy 占 host prepare 约 96.82%，提供了明确的优化目标。这些数值是
刷新事务累计计时，各字段存在包含关系，不能相加；也不能直接当作端到端墙钟
耗时的独立组成部分。优化前后的上传字节数和刷新次数也需一起核对，防止将
不同选页量误当成复制速度的收益。

优化后同样有 256 次刷新、254 次评分，八个请求各刷新 32 次，选中历史均不
超过 32768 token。实际上传量从 341.34 增至 344.17 GiB，增加约 0.83%；读取
仍全部命中 CPU cache。按实际字节归一化后，组件收益仍然存在：

| 实际模型 prefill 装配，按字节归一化 | 优化前 | 优化后 |
| --- | ---: | ---: |
| CPU cache copy 有效速度 | 7.98 GiB/s | 21.34 GiB/s |
| host prepare | 0.12939 秒/GiB | 0.05096 秒/GiB |
| assembly | 0.16175 秒/GiB | 0.06115 秒/GiB |

实际模型内的 cache copy 速度约为优化前的 2.67 倍。这个结果验证了 CPU
复制／装配成本下降；完整模型吞吐仍须与上面的 decode 接受率和端到端耗时
共同解读，不能直接套用 CPU 微基准中的最高复制速度。

## 已实现的优化及原版 qw3 依据

改动保持 MeanK 评分、原版选页、历史 token、NVFP4／FP8 精度、DFlash2（k=3）、
vLLM 模型算子和线性后端不变，集中处理以下数据搬运与 CPU 管理成本：

| 本仓库改动 | 对应原版 qw3 机制 | 保留的边界 |
| --- | --- | --- |
| 历史增量记账，只处理新封口记录；记录仅编码一次不可变 payload | `src/qwen_executor.cpp` 的 `kvmem_prefill_writeback` 使用 `kvmem_writeback_next_block_` 推进新封口页 | accepted prefix、部分尾页、generation lease、GC／rewind 和 I/O 完成后发布保持不变 |
| CPU cache gather 用 native memcpy 释放 GIL，配合一个进程内有界线程池 | `kvmem_submit_prefetch_cpu_batches` 使用常驻 CPU worker 并行 gather 稀疏 immutable block | 按原顺序写入独占目标区间，保留 immutable source 引用和 I/O admission，失败后等待已提交复制完成 |
| runner 共享两组持久 CPU／GPU staging slot，替代每次刷新分配 | raw-K materialization 双缓冲分别保存 `h2d_done` 与 `compute_done` | CPU slot 由 DMA fence 保护，GPU slot 由 assembly consumer fence 保护，不能仅因 H2D 完成就复用 |
| 一次准备并异步上传 pinned 元数据 | 原版批量构建、上传 page/remap 元数据 | 元数据及缓存页 owner 保持到成功或失败 fence 完成 |

原版查阅位置为本机 `/home/chaidi/qw3/src/qwen_executor.cpp`：
`kvmem_submit_prefetch_cpu_batches` 约 14103 行，raw-K pipeline 约 6447 行，
`kvmem_prefill_writeback` 约 15493 行。原版还将 V stage-in 与 raw-K assembly
分支重叠；本次保留现有 raw-K＋V packed 格式，不声称已经移植所有原版优化。

`cpu_gather_workers` 是公开的 1–16 范围参数，本次选用 8。线程池由整个共享
archive 使用，数量不会随用户数倍增。模型几何下每个 staging slot 是 192 MiB，
两组 slot 合计占 pinned CPU 384 MiB、GPU 384 MiB，按 runner 共享并延迟分配；
不是每用户各占一套。失败、取消和部分尾页由完成 fence 与回归测试保护，
真实模型恢复的字节验证见下方真实模型验收结果。

## CPU 微基准

以下只证明组件成本改善，不代表模型吞吐、PCIe 带宽或端到端延迟提升。

cache gather 使用实际 6 MiB／record、32 records／192 MiB 的 pinned 目标，
source 和 destination 预热，重复 11 次并核对最终字节：

| CPU 复制方法 | 中位时间 | 有效复制速度 |
| --- | ---: | ---: |
| 原 Python 串行 cache copy | 18.24 ms | 10.28 GiB/s |
| native，1 个 worker | 21.30 ms | 8.80 GiB/s |
| native，2 个 worker | 12.64 ms | 14.83 GiB/s |
| native，4 个 worker | 6.36 ms | 29.48 GiB/s |
| native，8 个 worker | 3.84 ms | 48.87 GiB/s |

单 worker 的 native 路径并未更快；这里的收益来自有界并行复制及绕开 GIL
串行化，实际配置仍须考虑与其他 CPU 工作的竞争。

历史维护微基准使用单个 torch CPU thread、同步 discard-only archive、7 次重复。
历史扫描组刻意使用小张量隔离记录数量的影响；新页编码组才使用模型的 6 MiB
完整页格式，均不包含真实存储或 GPU。

| CPU authority 操作 | 优化前中位时间 | 优化后中位时间 |
| --- | ---: | ---: |
| 64K 历史，追加 3 token，小张量 | 76.43 µs | 23.81 µs |
| 256K 历史，追加 3 token，小张量 | 227.80 µs | 23.74 µs |
| 256K 历史，无待提交页的 flush | 66.57 µs | 0.36 µs |
| 编码／提交 1 个完整 6 MiB 页 | 4.37 ms | 3.81 ms |
| 编码／提交 8 个完整 6 MiB 页 | 77.25 ms | 57.87 ms |

进一步拆分当前封页路径，单页合并运行约 3.905 ms，其中 SHA-256 约 3.464 ms；
八页合并运行约 66.381 ms，分项中 join 约 35.705 ms、SHA-256 约 27.774 ms。
这说明消除历史扫描后，大页快照和摘要仍有 CPU 成本；本轮没有实现异步封页。
该诊断排除 archive／CPU cache admission、NVMe、GPU 和 source 分配，且缓存
状态与上面的完整 flush 微基准不同，因此 66.381 ms 不能与 57.87 ms 直接作
前后对比。各分项的独立中位数也不保证相加等于 combined 的中位数。

CPU 改动后的回归为 543 passed、54 skipped，另有 27 项 GPU 装配测试通过，
覆盖重复选择、packed／direct 路径、部分尾页、跨请求复用和故障 fence。
重建 wheel 的 69 个安装文件核对无差异，包内 Python 源码与当前源码一致。

优化后真实模型的 64K 回合暂停／恢复字节验证已通过：首次 checkpoint 保存
65792 个已计算历史 token，GPU working context 为 34048 token；释放 GPU 页
后恢复，1,115,684,864 字节目标 KV、96 个 recurrent 张量及 5 层 draft 状态
全部一致，最终前端历史为 65953 token。首回合 33.14 秒，恢复回合 4.85 秒，
均包含生成及 oracle 开销，不能当作单独的 KV 复制时延。本次明确跳过冷启动
greedy 对照，只验收状态恢复字节，不声称冷重算逐 token 等价或任务质量通过。

返回普通服务配置后（profiler 和恢复 oracle 均关闭），两个认证回合冒烟测试
均成功，历史长度从 272 增到 292 token；结束时 active requests、预留字节和
排队请求均为零。该检查确认普通入口可用，不计作新的吞吐测量。

## 证据与复现

本地原始证据位于被 Git 忽略的 `artifacts/`。这里列出目录和脚本，不公开凭据、
模型或生成正文。
可公开核对的配置、指标和验收状态见
[结构化汇总](benchmarks/kvmem-overhead-20261013.json)。其中保留完整吞吐
summary、归一化装配数据、CPU 微基准摘要和输出哈希匹配数量，不含请求标识、
API key 或生成正文。

| 路径（相对于仓库） | 内容 |
| --- | --- |
| `artifacts/kvmem-overhead-20261013/before-c8-32k-direct/` | 优化前 KVMem 的原生 SSE 对照 |
| `artifacts/kvmem-overhead-20261013/native-c8-32k-direct/` | 相同 DFlash2（k=3） 的原生 vLLM 对照 |
| `artifacts/kvmem-overhead-20261013/after-c8-32k-direct/` | 优化后首轮直连 32K 对照，decode 吞吐未提升 |
| `artifacts/kvmem-overhead-20261013/before-c8-64k-4k/` | 优化前弹性长历史及 `assembly-summary.json` |
| `artifacts/kvmem-overhead-20261013/after-c8-64k-4k/` | 优化后弹性长历史及按字节归一化的装配结果 |
| `artifacts/kvmem-overhead-20261013/cache-gather-cpu.json` | pinned cache gather CPU 微基准 |
| `artifacts/authority-cpu-20261013/comparison.json` | authority 微基准、来源哈希和重复样本 |
| `artifacts/authority-cpu-20261013/seal-phases.json` | 当前封页快照／SHA-256 分项诊断 |
| `artifacts/kvmem-overhead-20261013/restore64k/results.json` | 优化后的真实模型 64K 回合恢复字节验证 |
| `artifacts/kvmem-overhead-20261013/native-plan.json` | 原生 DFlash2（k=3） 控制组实际启动参数 |
| `artifacts/kvmem-overhead-20261013/wheel-verification.json` | wheel 哈希及安装内容核对 |
| `artifacts/kvmem-overhead-20261013/release-smoke.json` | 普通配置的双回合认证及资源回收检查 |

[`scripts/qualify_kvmem_overhead.py`](../scripts/qualify_kvmem_overhead.py) 用于直连
对照；[`scripts/qualify_elastic_long_output.py`](../scripts/qualify_elastic_long_output.py)
用于弹性入口长历史测试。两种测试都必须独占模型请求负载，且不得同时运行 GPU
单元测试。CPU 复现入口为
[`scripts/bench_cached_gather.py`](../scripts/bench_cached_gather.py) 和
[`scripts/bench_authority_cpu.py`](../scripts/bench_authority_cpu.py)；封页分项诊断用
[`scripts/bench_authority_seal.py`](../scripts/bench_authority_seal.py)。

这些测量验证了本轮 CPU 与装配成本下降，以及相应的长文 prefill 收益。
稳定 decode 提升、持续混合工具负载的 SLO 和任务质量仍需分别测量；本报告
不把 CPU 微基准的倍数外推到模型吞吐，也不声称已达到全局最优配置。
