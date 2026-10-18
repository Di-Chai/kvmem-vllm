# 两套记忆评测完成结果（2026-10-15）

两套均沿用 qw3 冻结对照集及首次模型输出。MemoryAgentBench 1,116 题保持已完成的推理，补齐外部评分；本轮新增独立 LongMemEval-S 102 题，8 个客户端且实际峰值 8 个请求同时准入、运行。两套均使用 NVFP4 主模型、FP8 KV、原生 MTP(k=4)、编译与 CUDA Graph、完整 KVMem prefill/output attention；历史 KV 仅使用 CPU。

## 完整质量结果

| 冻结评测集 | 本版 | qw3 attention 对照 |
|---|---:|---:|
| LongMemEval-S，102 题准确率 | **89.22%（91/102）** | 88.24%（90/102） |
| MemoryAgentBench，1,116 题、三类别等权宏平均 | **35.42%** | 36.63% |

102 题中：双方均正确 83 题、仅本版正确 8 题、仅 qw3 正确 7 题、均错误 4 题。净增加 1 题，不据此宣称显著质量优势。两个评测的指标不同；MemoryAgentBench 的三类别宏平均混合检索准确率与摘要 F1，不能当成 1,116 题的简单准确率，也不是官方四类别全量分数。

| LongMemEval-S 类型（各 17 题） | 本版正确数 | qw3 attention 正确数 |
|---|---:|---:|
| single-session-user | 17 | 16 |
| single-session-assistant | 17 | 17 |
| single-session-preference | 13 | 13 |
| multi-session | 15 | 13 |
| temporal-reasoning | 16 | 14 |
| knowledge-update | 13 | 17 |

MemoryAgentBench 800 题完成原官方本地评分，140 题复用输入、答案和裁判配置完全一致的历史评分，176 题完成新的外部评分。内部的 300 道 LongMemEval 子题为 153/300（51.00%），qw3 为 169/300（56.33%）；它们与这里独立测试的 102 题分别统计。其余任务和逐题配对分析见[完整 attention 与 MemoryAgentBench 报告](prefill-attention-20261014.md)。

独立 102 题使用 qw3 原来的 DeepSeek Flash 参数和冻结官方裁判提示。12 个校准样例全部通过；8 题的完整裁判请求与原记录逐字一致，复用其结果，其余 94 题新评分。所有答案都有正式判断，没有待评分或裁判错误。用户明确授权向官方 DeepSeek 端点评分；密钥通过隐藏终端输入，仅供评分进程使用，未写入脚本或结果文件。评分进程已退出。

## 102 题吞吐与延迟

原始完整输入范围 102,963–112,819 token，总输入 **11,111,103 token**，总输出 **79,308 token**。准入到最后一次历史回收确认共 **2,218.65 秒（36.98 分钟）**，不含模型启动和随后裁判时间。

| 指标 | 实测 |
|---|---:|
| 混合端到端有效输入处理率 | **5,008.04 token/s** |
| 混合端到端输出率 | **35.75 token/s** |
| 客户端 / 实际峰值准入 / 引擎峰值运行 | **8 / 8 / 8** |
| 已完成请求平均 HTTP 耗时 | 170.96 秒 |
| 平均首 token 延迟 | 80.54 秒 |
| 平均生成阶段耗时 | 89.92 秒 |
| 入口 CPU 预算平均等待 | 0.00030 秒 |
| 最长请求 HTTP 耗时 | 764.18 秒 |
| 请求错误 / 抢占 / 输出截断 | **0 / 0 / 0** |

这里持续混合长文预填充、attention 评分、历史选页装配、Query 重放和生成；输入处理率不是隔离测量的 GPU prefill 峰值，输出率不是稳态 decode 吞吐。生成阶段耗时也包括其他请求 prefill 等工作造成的干扰，不能与旧 32K+16K 共同 decode 窗口的吞吐直接比较。MemoryAgentBench 的 CPU 本地审计及外部评分与本次推理部分重叠，性能属于实际混合负载观察。

221 次资源采样中，系统最少仍有 **45.66 GiB** 可用内存，历史 KV 磁盘读写、NVMe 容量和 swap 使用均为零。CPU native pool 上限 100 GiB，入口预算 72 GiB。102 题完成后历史 authority 占用归零、无活跃或回收中的请求；本次模型服务已退出，清理后 GPU 为 14 MiB、0% 利用率。此处验证的是约 100K 历史的真实 C8；之前 MemoryAgentBench 虽有 8 个客户端，实际同时准入的是 2 个超长历史组。

## 参数对齐及差异

完整原始 token ID、文本哈希、Query 坐标和各题采样配置逐一核对，不截断输入，不向推理端发送 gold，不根据评分重试答案。temperature=.6、top_p=.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1、seed=20260722、thinking budget=4096。

K=36,864、G=16,384、页大小 32、sink=1,312、recent=0、attention quota=.5、spread3 层、prefill chunk/interval=1024、stride=32、half-life=8192、guided=0，保留 Query history；每题使用独立新会话，没有跨题答案反馈。没有 CacheBlend 历史重算。

原 qw3 请求 max_tokens=32,768，但 G=16,384；本版接口要求单轮输出不超过 G。本轮保留影响压力重选的原 G，将请求输出上限设为 **16,384**。实际输出范围为 **103–4,205 token**，102 题全部自然停止且均关闭 thinking，未触及上限。这是明确记录的配置差异，不能声称所有参数逐字完全相同。参考为 Q8＋DFlash2，本版为 NVFP4＋MTP4，不是执行轨迹或浮点结果逐位一致的对照。

102 题首次 Query 均使用 prefill attention，旧 output history 均为零；全部压力重选使用 prefill 评分。输入 attention 样本合计 345,050，qw3 为 345,245，原生分块与预留边界导致少量计数差异。796 个输出 Action 采样均由最终 token IDs 离线重放 parser 核对，公开 Query 重放未进入输入采样。全部请求有 MTP(k=4) 运行记录，原生完整 CUDA Graph decode 调用 1,017 次。90 个推理源文件与配置的启动前哈希在测试结束后仍一致。

## 复现与证据

- `scripts/benchmark_longmemeval.py`：校验冻结 102 题、CPU 准入、8 并发、保存首次回答、逐请求 attention 与回收确认；拒绝静默重跑失败或中断样本。
- `scripts/analyze_longmemeval.py`：核对输入、采样、回答、Action parser、原裁判请求绑定，支持仅从环境读取凭据的 `--external`；缺失评分不记错误。
- `artifacts/longmemeval-s-attention-20261015/`：profile、manifest、原始响应、native capture、资源采样、校准与全部裁判结果、源文件哈希、退出记录。正式推理前一次启动入口未执行服务、未产生答案，证据单独保留在 `startup-failure/`。
- [可机读结果](benchmarks/longmemeval-s-attention-20261015.json)。
