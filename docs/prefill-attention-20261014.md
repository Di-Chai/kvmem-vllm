# 完整 input/output attention 路径与质量验收（2026-10-14 起）

本轮补齐 qw3 的 prefill attention mass/exposure、attention pressure 选页，以及冷启动 Query 的 prefill 评分融合。目标模型维持 NVFP4、FP8 KV、原生 MTP(k=4) 和原生编译/CUDA Graph，历史 KV 只存 CPU，无 CacheBlend 历史重算。

## 实现

- `PrefillHistory` 对完整接受的输入 chunk 采样，按可见页累计 mass/exposure，并按接受的输入距离衰减；同一逻辑页的部分段只计一次 exposure。统计只保留当前观测窗口。
- 压力重选和语义选页调用原 qw3 C++ 选择器。压力阶段只从 resident 页选择；有效 output attention 优先，无有效输出评分时才回退 prefill。sink/mandatory 优先于评分。
- public Query 重放及其 assistant 模板尾部、private guided Query、拒绝的 speculative 行均不进入输入评分。跨轮 checkpoint 包含输入和输出评分，编辑分支不会继承未来反馈。
- 新增 `session_freeze_prompt`：独立问题复用回答前的 P 窗口、模型状态和输入评分，丢弃上一题的回答评分。该模式要求 P-only 和关闭 Query history；普通多轮会话维持原来的 P/M 策略。
- 原生 MTP 的逻辑历史上限扩展到 1M，用于完整保留 memory 样本；主模型和 MTP 的 RoPE 仍在有界 active 窗口内，未修改模型 rotary 频率。DFlash 的原有限制保持不变。
- 补齐超过 8,192 个 eligible 历史页的 MeanK 路径：直接迁移 qw3 原来的 logits、全页归一化和累积三个 CUDA 内核，支持至 32,768 页。分布按约 64 MiB workspace 分批，但 softmax 始终覆盖全部 eligible 页；较短历史继续使用原 fused 内核。
- 多用户网关转发 prefill chunk、页大小、sink/recent 参数，Query 请求校验使用同一页大小。可配置示例见 [完整 attention 网关](../examples/qwen-nvfp4-prefill-attention-elastic.json)。

输入 attention 通过额外采样 Q×实际 paged FP8 K 计算，采用 native prefill/decode Q dtype 和 FP32 reduction；不是 FlashInfer 内部 softmax 的逐位拷贝，也不宣称零开销。CPU 累积语义对照 qw3 `kvmem_action_attention.cpp`，parser 和 selector 使用原 C++。

## 256K 大海捞针：完整 60 题

四种任务，各 15 题。完整 token 输入、问题坐标、seed、temperature=1、top_p=.95、top_k=20、4096 thinking budget、8192 输出上限均与冻结 qw3 样本核对。没有根据答案重试或筛题。

| 版本 | 逐题平均数值覆盖率 | 整题数字集合全对 |
|---|---:|---:|
| 旧 vLLM：仅 output Action 路径 | 68.33% | 30/60 |
| 本轮 vLLM：完整 attention、对齐选页参数 | **97.08%** | **53/60** |
| qw3：同样 60 题的完整 attention 对照 | 98.75% | 57/60 |

| 任务 | 旧覆盖率 | 本轮覆盖率 | 本轮整题全对 |
|---|---:|---:|---:|
| single | 100% | 100% | 15/15 |
| multikey | 100% | 100% | 15/15 |
| multivalue | 23.33% | 90% | 9/15 |
| multiquery | 50% | 98.33% | 14/15 |

本轮不仅开启 input attention，也将 K/G、sink/recent、guided 设置对齐真实 qw3 attention 对照：K=36,864、G=16,384、32-token 页、sink=1,312、recent=0、attention quota=.5、guided=0，采样层 3/31/63、输入 chunk/interval=1024、stride=32、half-life=8192。因而这不是只改变一个开关的消融实验。

所有 60 题第一次 Query 均使用 prefill 评分，每题 13 次 attention pressure 重选。vLLM 原生分块/预留边界使实际采样为每题 8,173 行，qw3 为 8,160 行。参考是 Q8 模型＋DFlash2，本轮为 NVFP4＋MTP4；输入与采样参数一致不等于执行轨迹或浮点结果逐位一致。

150/150 个数字及对应名称进入最终选页；118/150 个完整原句进入选页。实际回答找回 143/150 个数字；97.08% 是逐题覆盖率宏平均，不是 143/150 的微平均。7 次漏答全部属于名称和数字均已选中的情况，剩余差距不能只用选页命中率解释，也不能据此直接归因于量化。显式名称绑定补充检查未发现整题数字全对但对应名称错配的情况。

## 并发与时间范围

客户端为 8 个，CPU payload 准入预算为 72 GiB（native pool 上限 100 GiB），实测峰值 5 个完整 256K 历史同时在服务中。全程零请求错误、零抢占，历史磁盘读写和 NVMe 容量均为零；最少约 11.85 GiB 可用系统内存，无 swap。

60 题共 15,726,105 输入 token、13,351 输出 token，整轮约 2,905 秒（48.4 分钟）。混合工作负载的有效输入处理率约 **5,413 token/s**，输出约 4.60 token/s；后者主要受超长输入和短答案影响，不能当作稳态 decode 吞吐。单请求平均约 240.6 秒，平均 TTFT 222.3 秒，平均生成阶段 17.6 秒，均不含入口 CPU 排队。

这次 benchmark 入口预算门非 FIFO，最长 CPU 等待约 2,714 秒，因此不是公平调度或 SLO 验收。运行早期另有约 28 秒回归验证重叠，整轮性能含这段干扰。质量结果完整保留，吞吐只作本轮端到端观察；未将历史 CacheBlend 10% 结果误标为本轮 qw3 attention 对照。

## MemoryAgentBench

冻结 qw3 的 **1,116/1,116 题已全部完成推理，零请求错误、零抢占**。29 组历史的原始输入和采样参数已全部核对，最长输入 757,259 token。使用独立问题的冻结 P 前缀复用；8 个客户端、实际峰值两个历史组同时准入，CPU-only。这是 qw3 冻结的 1,116 题对照集，不是 MemoryAgentBench 全量 3,671 题。

真实模型预验收已通过：470K 历史下连续三道独立题恢复同一 P 窗口和输入评分，两次恢复的 recurrent state 逐字节一致；对第二道题另起新会话，从完整历史冷启动，选页和最终回答均与恢复路径相同。预验收和冷启动对照不计入正式质量结果。

800 道可本地评分题已全部完成，使用固定版本官方 substring exact match，结果如下：

| 任务 | 本轮 | qw3 attention 对照 |
|---|---:|---:|
| EventQA，500 题 | 87.8%（439/500） | 86%（430/500） |
| RULER QA2，100 题 | 51% | 54% |
| 多跳事实整合，100 题 | 3% | 1% |
| 单跳事实整合，100 题 | 43% | 48% |

这里有升有降，不能将大海捞针的改善推广为 Memory 所有任务的质量提升。对照保留 qw3 原始结果，不因新结果改变样本或采样参数。

2026-10-15 已补齐外部评分，**1,116/1,116 题均有正式分数**：800 题使用固定官方本地评分器；140 题复用输入、答案与裁判配置完全匹配的历史判断；其余 160 道 LongMemEval 和 16 道摘要通过用户授权的 DeepSeek 原裁判协议评分。全部沿用已保存的首次模型答案，没有重新生成。

| 外部评分及汇总 | 本轮 | qw3 attention 对照 |
|---|---:|---:|
| LongMemEval 子任务，300 题 | 51.00%（153/300） | 56.33%（169/300） |
| 摘要，16 题，官方 rubric F1 均值 | 20.002% | 19.960% |
| 准确检索类别宏平均 | 63.267% | 65.444% |
| 冲突整合类别宏平均 | 23.000% | 24.500% |
| 三类别等权宏平均 | **35.423%** | **36.635%** |

总分低于 qw3 **1.212 个百分点**。这里的总分仅指冻结的三类别子集，不是官方四类别全量分数，也不是 1,116 题的简单准确率。LongMemEval 子任务逐题对照为两边均正确 137、仅 qw3 正确 32、仅本版正确 16、均错误 115；本版 32 道丢失题中 6 道触及原始 50-token 输出上限，其余 26 道正常结束，不能把差距全部解释为输出截断。两边模型量化、推测解码与冷/热执行组成不同，本次没有将差距归因于单一因素。

以上 300 道是 MemoryAgentBench 内部的子任务，与另外单独测试的 LongMemEval-S 102 题对照集分别统计。评分明细与逐题配对证据保存在 `memory/scoring/`；补评前 940 题的快照保存在 `pre-external-summary.json`。

整轮约 **6,121 秒（102.0 分钟）**。29 次首次加载请求平均 197.1 秒；**1,087 次前缀恢复请求平均 4.64 秒**，后者均未重新预填充完整历史。每组前缀哈希和输入 attention 样本计数保持一致，所有 Query 的旧答案 output history 均为零。单请求耗时从 HTTP 发送到回复计量，不包含等待历史组准入或客户端输入准备。

逻辑请求合计 512,230,931 输入 token，其中实际新增输入 12,808,723；复用的前缀从吞吐分子中扣除。整轮有效新增输入约 **2,093 token/s**，输出 28,938 token、约 4.73 token/s。这是历史加载、恢复、选页、Query 重放及短答案/摘要生成的混合端到端结果，不是纯 GPU prefill 或稳态 decode 吞吐。尾段还并行执行过一次离线评分审计。qw3 对照本身混合冷启动与冻结前缀执行，不能把两边总体均值当成严格的同执行路径速度对照。

608 次资源采样中，历史磁盘读写、NVMe 容量及 swap 使用均为零，最少约 **18.24 GiB** 可用系统内存。CPU native pool 上限 100 GiB，入口历史预算 72 GiB；这次并不代表 8 个超长历史同时驻留。测试自有服务在完成后退出，GPU 和 CPU 测试缓存已释放。

## 验证与证据

完整回归：725 passed / 69 skipped；单独 GPU 数值对照 6 passed。之后网关参数与请求边界回归 100 passed；1M 逻辑历史及冻结 P 启动校验回归 104 passed。MeanK 的 GPU 数值验证 9 passed，包含 8,192/8,193 页边界、分批、32,768 页、实际 Qwen 头数和 256 维、非默认 CUDA 流。真实模型 C8、82,006-token 输入、两轮共 16 请求通过，覆盖压力重选、输入评分和 output 评分恢复。

完整数据在 `artifacts/prefill-attention-20261014/`：冻结 profile/manifest/source hashes、60 题原始回答、native capture、输入/输出评分审计、资源采样、qw3 同题比较，以及 memory 的输入审计和独立验收目录。首个 memory 启动在配置校验阶段被旧长度上限拒绝，没有发出测试请求；失败证据保留为 `*-startup01.*`。第一次预验收又暴露 MeanK 旧 8,192 页上限，在产生答案前失败；该次完整证据存于 `memory-failures/preflight01/`。修复后重新通过预验收才启动正式 1,116 题。

精简结果见 [JSON](benchmarks/prefill-attention-20261014.json)。

独立 LongMemEval-S 102 题也已完成：89.22%（91/102），qw3 同题 attention 为 88.24%（90/102）；真实 C8，混合有效输入约 5,008 token/s、输出约 35.75 token/s。配置差异和完整证据见[两套记忆评测完成结果](longmemeval-s-attention-20261015.md)。
