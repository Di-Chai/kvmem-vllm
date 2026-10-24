# 云端 RTX4090 GC 开发与关机前测试状态

记录日期：2026-10-24，Asia/Shanghai。用户准备关闭云端 RTX4090，已停止调度、明确中止当前样本、保留评分及原始产物，并停止模型与解析服务。本文是阶段性结果，不是完整 Terminal-Bench Pro 成绩或吞吐验收。

## 已实现

- 原生后台 CPU 历史 KV 退休和物理释放；generation、evidence revision、mandatory、selection hold 及 P/M 生命周期保护。
- statistical 策略及语义证据接口；block graph 共存协议已有功能测试，当前真实任务配置关闭 graph。
- DSH 已落盘文件正文的实验回收：核对成功 write 与容器内文件 bytes，只回收正文内部完整 32-token 块，保留工具框架、路径、混合边界和结果凭据。
- 同一用户任务中，在工具边界执行有界窗口交接，之后由后台线程释放 CPU 历史。没有新增裁判模型请求。
- 本地 CPU Docker/Harbor/DSH 运行官方任务，推理仅使用云端单张 RTX4090；模型请求串行。
- 修复跨网络流式解析开销：普通文本前缀在本地解析、相同前缀缓存，完整工具调用交给云端官方解析器。固定真实响应重放从 203 次本地前缀解析归并为 1 次远端解析，重放约 0.925 秒、0 次模型请求；相关 CPU 检查此前为 45 passed、33 skipped。

当前真实任务使用文件一致性观测，实验参数为 min_observations=1、min_cold_observations=0、max_importance=1、proactive=true。没有伪造低注意力；它不是完整 R-KV、自动 thinking 删除、模型裁判或已训练的统计决策模型。默认保守阈值没有改变。

## 冻结运行参数

运行源码：`6510009dee3f12efeae6b6359505a117b2f1290e`。本次关机记录不修改这四个已执行样本的运行源码。

| 项目 | 参数 |
| --- | --- |
| 模型 | sokada4/Qwen3.8-27B-GPTQ-Int4；revision 2232b67a2ef5480e97c4450838a712de1346eccc；hf-mirror 下载 |
| 推理设备 / 并发 | 云端单张 RTX4090 / 1 |
| K / G / 逻辑上下文 | 36,864 / 16,384 / 262,144 tokens |
| CPU 历史预算 | 12 GiB；8196 个 repository blocks |
| 执行 | BF16 compute、FP8 e4m3 KV、eager、MTP 1 |
| 输入 / replay 块 | 128 / 128 tokens；native scheduled batch 上限 130 |
| 采样 | temperature 0.7、top_p 0.8、top_k 20、min_p 0、presence_penalty 1.5、repetition_penalty 1；base seed 20261024、逐回复派生 |
| 其他 | thinking 关闭；单用户任务；没有自动摘要、修复或额外模型请求 |
| 对照差异 | 两份 serving profile 仅 historical_gc.mode 不同 |

官方任务固定于 Alibaba Terminal-Bench Pro 的 `874af409da6aafebccbf3bc5bb41a2fa4d78784d`，instructions、Dockerfile 和评分器未修改。计划仅为三个任务的六个样本，不代表公开 200 题成绩。启动和 warmup 不计入任务时间。

## 正式 v4 样本状态

三例已结束、一例因用户关机而中止、两例未启动。评分保留原值；中止例不能作为完成的配对样本。

| 任务 | GC | 状态 | 官方检查 | Reward | live-scope CPU 回收 | 回收后模型回复 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 数独 | OFF | 模型循环，最后输出达到 16K 且工具调用不完整，解析失败 | 1/14 | 0 | 0 | — |
| 数独 | DELETE | 模型自然完成 | 1/14 | 0 | 52 块 / 1,664 tokens | 11 |
| JSON → RFC4180 CSV | OFF | 模型自然完成，约 1,795.7 秒、45 次回复 | 89/90 | 0 | 0 | — |
| JSON → RFC4180 CSV | DELETE | 用户关机中止，约 703.3 秒、18 次回复 | 中止时 88/90 | 0 | 156 块 / 4,992 tokens | 12 |
| 日志汇总 | DELETE | 未启动 | — | — | — | — |
| 日志汇总 | OFF | 未启动 | — | — | — | — |

GC 数字均来自清理 owner 之前、对应候选 generation 和正文坐标的物理 CPU 释放完成事件。普通退出或会话释放的累计 frees 不计入 GC 成效。JSON DELETE 额外存在一个尚未完成发布的文件候选，未计为回收。

数独两组生成了 `sudoku_solver.py`，评分器执行 `/app/main.py`。DELETE 版本还将数字连写，而评分器要求 81 个空白分隔整数；仅改入口的独立 CPU 诊断为 8/14，不是全部通过。OFF 存在 142 次连续相同 Bash 命令，最后一次 16K 输出未形成完整工具调用；该例不是关机中止。

JSON OFF 唯一失败为 `test_discount_null_to_zero`；JSON DELETE 在人工中止时仍有两个检查失败。不能据此将任务质量失败归因于 GC，也不能把中止前的检查数当作自然完成结果。

数独首个输入的 4,669 token IDs 完全一致，但第一次工具结果包含不同的目录时间信息，第二次生成在发生 GC 前已分歧。因此两条完整任务的用时、输出量和 decode rate 不能作为 GC 因果开销估计。

## 已证明与尚未证明

已证明：当前路径能在两个真实 Terminal-Bench 任务中释放候选 CPU KV，并在删除后继续多次模型回复；没有把退出清理冒充 GC。

尚未证明：这批 Terminal-Bench 任务的成功率不下降、完整三任务配对结果、256K 长历史任务质量、默认统计门控质量、thinking/R-KV/裁判策略或 graph 联合模型质量。

此前另一个完整 DSH 记账网页任务实际回收 111 块 / 3,552 tokens，删除后继续 13 次回复，独立浏览器为 86/86；详见 [真实任务报告](gc-real-task-20261024.md)。它提供功能证据，不是本次正式基准成绩。

吞吐仍未验收。当前 36K/G16K GPTQ 配置的五组固定配对合成测试，中位吞吐损失为 -1.363%，描述性 95% 区间为 [-2.430%, +2.794%]，跨过预设 1% 上限，结论为 inconclusive；p95/p99 未测量。早期其他配置的通过结果不能覆盖当前配置。真实任务中的窗口交接/replay 成本也需独立计入。

## 关机保存与恢复

四个已执行样本均已冻结 scheduler/CPU 资源证据，requests、owned blocks、pending refresh/connector、reservations、session owners 及 CPU reserved bytes 均为零。关机前模型服务、显存采样器及 CPU 官方工具解析服务已停止，独立查询确认 GPU compute processes=0、memory.used=0 MiB。没有执行云机 poweroff，由用户自行关机。

代码与本报告经云端 Git 提交并推送；未上传模型权重、SSH 密钥或原始模型对话。云端日志、profile、launcher 和 native snapshots 已打包复制到本机，SHA-256 为 `59f70668f6fb517ce6b5f94bc474d90912f23757a31a18f373e863b9daa1e19a`。本地任务原始产物另存私有备份。

恢复时保留本批失败和中止记录，使用新 case ID 启动后续样本；不要重启旧 case 并覆盖原结果。需要完成 JSON 配对时，另行预先声明新的完整配对，不能把本次中止样本改写为完成。

公开证据：

- [正式样本和中止分类](benchmarks/qwen38-tbpro-three-task-gc-20261024.json)
- [Serving 参数及源码指纹](benchmarks/qwen38-tbpro-execution-20261024.json)
- [初始输入与轨迹分歧审计](benchmarks/qwen38-tbpro-initial-input-audit-20261024.json)
- [流式解析重放](benchmarks/qwen38-tbpro-parser-replay-20261024.json)
- [v4 数独独立入口诊断](benchmarks/qwen38-tbpro-v4-entrypoint-diagnostics-20261024.json)
- [关机资源与备份证明](benchmarks/qwen38-tbpro-shutdown-20261024.json)
- [当前配置吞吐结果](benchmarks/qwen38-4090-36k-gc-throughput-20261024.json)
