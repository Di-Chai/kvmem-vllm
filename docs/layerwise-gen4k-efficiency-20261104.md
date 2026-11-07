# Gen4K 逐层重选：校验优先的效率跟进

## 约束与当前判断

- 基线提交：`20c292b`。保留已有两项缓存优化，不修改当前运行的冻结目录。
- 模型、完整输入、Active=16384、Gen=4096、block32、sink41、recent0、召回规则及采样不变。
- 正式性能组仍为八路完成补位；原生 batch 上限保持 **16424**。
- 先验证逐层 pressure / Query 选块及完整输出，再接受无 profiler 的完整吞吐结果。
- 不恢复 MAB，不启动定时巡检，不重采样低分或覆盖已有尝试。

最终状态：原优化性能组已完成 **102/102**；后续诊断已结束，GPU 已释放。
原/优化串行完整输入对照 **4/4 一致**，但真实 decode pressure 未覆盖，
因此**未授予完整等价或性能验收**。汇总：
`artifacts/g4k-gilopt-review-20261104-r2/final-review.json`。

此前正式 102 题对照不变：Gen16K 为 180.73 题/小时，Gen4K 为
134.31 题/小时；小 Gen 同时改变 prefill 重选频率和最终 Query 空间，不能只归因于 decode 刷新。

## 外置运行时审计

记录目录：`artifacts/g4k-gilopt-review-20261104-r2/`。

`runtime-audit.json` 核对了：

1. 原 Gen4K 的冻结清单；另已调用原 `run_layerwise_memory.verify` 通过正式基线校验。
2. 原冻结运行时的 723 个文件；优化组只有 `layerwise_pressure.py` 和
   `window_assembly.py` 两个推理文件变化。
3. 在独立临时目录对**原冻结基线**重新应用 `opt.patch`，产物哈希与运行中的优化版一致。
   没有用工作区整个 `window_assembly.py` 覆盖冻结版本，因此保留原有 original-RoPE 差异。
4. 逐字节相同的 fixture/profile、原始请求 binding，以及除 capture 输出目录外相同的实际 launch。
5. 优化运行时额外存在 `window_assembly.py.orig`，内容与冻结基线相同；显式登记并保留，未删除。

这是事后审计，**不能倒签为启动前校验**，也不能替代模型/回答一致性。
`state-snapshot.json` 保存了实际 PID 身份、启动 tick、命令哈希、运行路径及 GPU 归属。

## 已保存的 56 题观察，不是吞吐结论

详见 `partial-equivalence.json`、`partial-analysis.json`、`prefill-stages.json`。

- 56/56 题的 pressure 序列、最终 Query 选块、完整输出均未完全一致；最终答案文本仅 3 题相同。
- 53 题首次分歧在第一次 prefill pressure，另 3 题在第五次。
  首次分歧时，候选块与 accepted score end 仍相同；没有保存分数张量，不能据此确定根因。
- 两边 prefill pressure 均为 **1273 次**。基线另有 1 次 decode pressure，优化版在该快照没有。

| 逐事件延迟 | 基线 | 缓存优化 |
|---|---:|---:|
| 每题平均耗时 | 205.22 s | 193.58 s |
| prefill pressure 总延迟 | 1.571 s | 1.493 s |
| assembly 排队等待 | 0.914 s | 0.888 s |
| selection_prepared → assembly_submitted | 0.293 s | 0.266 s |
| assembly_submitted → ready | 0.198 s | 0.180 s |

这些阶段包含并发重叠，不能相加当墙钟归因；不同输出也影响实际工作量。
请求平均延迟下降约 5.7% 只是历史对照信号，**不是完整吞吐改善或正确性验收**。

## 完整八路组的归档观察

优化组完整推理墙钟为 2506.96 秒（41.78 分钟），原始测量为 146.47 题/小时；
仅归档，不认定为正确性合格的提速。两边 prefill pressure 都为 2318 次，
基线 decode pressure 为 3 次，优化组为 0 次；输出量分别为 72930 和 67886 tokens。
八路历史/重跑的 pressure、Query 和完整输出均为 0/102 完全相同，答案文本为 5/102 相同，
没有进行外部准确率评分。不能把不同生成工作量下的墙钟差全算作缓存优化收益。

完整组每次 prefill pressure 平均排队约 0.711/0.697 秒，
`selection_prepared → assembly_submitted` 约 0.314/0.263 秒；
这是未受控历史对照，阶段重叠，不作因果或可相加的墙钟归因。

## CPU 验证

- 相关 CPU 回归 **344 项通过**，无 GPU 推理任务。
- 扩展测试首次暴露 4 个已有测试 fixture 缺少 `guided_reason` 的失败；确认相关源码及测试在
  HEAD 未改后，只补测试 fixture。首次失败 XML 和修复后 XML 均保留。
- 新增覆盖：每次重选的 slot 属性读取次数、16 层含并列分数的原生 preview、prefill/decode、
  首次分歧定位、文件漂移、嵌套仓库补丁应用、PID 复用、真实 GPU 归属和 baseline 稳定性门槛。
- `fixed-score-replay.json` 对原/优化**冻结 prepare 源码**做了 8 组固定分数 CPU 重放，
  覆盖 640/3520 个 archive blocks、16 层、prefill/decode、两个固定种子。
  逐层计划和证据完全相同。这里只隔离了选择准备，未验证完整模型、KV 字节或回答。

## 隔离 decode 后的串行有效 prefill 观测

对已经逐块、逐 token 一致的四题，单独重算“客户端 attempt → 最终 Query refresh 开始”的
历史 prefill 阶段：包含 HTTP 提交、pressure、组装和等待，不含最终 Query 和 decode。
每遍历史 prefill 均为 427136 tokens。

| 执行 | 历史 prefill 阶段合计 | 有效 tokens/s |
|---|---:|---:|
| 原版 A | 125.586 s | 3401.15 |
| 原版 B | 124.941 s | 3418.69 |
| 优化版 | 120.623 s | 3541.10 |

两遍原版合并基准为 3409.90 tokens/s，优化版相对观测改善 **3.85%**。
这只是四题、单客户端的实际阶段吞吐，不是八路性能验收或纯 GPU kernel 速度。
各 cohort 的首个 1024-token chunk 没有 StepTiming 记录，因此没有用不完整的 step 计时给完整
token 数计算吞吐；完整阶段采用已保存的客户端时间，并用 public-work 日志校准单调时钟。
原始数据与校验清单：
`artifacts/g4k-gilopt-review-20261104-r2/serial-prefill-phase-comparison.json`。

## 受控验证结果与覆盖边界

首轮目录：`artifacts/g4k-gilopt-qualification-20261104-r3/`。

1. `baseline-a`：完整输入 `e47becba`、`6d550036`，固定顺序、一个客户端 owner。
2. `baseline-b`：同样完整输入和顺序，先检验原版自身稳定性。
3. 只有 A/B pressure/Query/完整输出一致，且两边都自然触发 decode pressure，才运行完整资格验证的 `optimized`。

`6d550036` 按原 fixture 顺序选取的首个已知 decode-refresh 样本；未依据得分选题。
两次原版完全一致，但该题均只生成 1011 tokens，没有触发刷新；完整资格验证按门槛停止，
未启动该目录的优化臂，没有换 seed 或重新采样。

覆盖扩展目录：`artifacts/g4k-gilopt-qualification-20261104-r4/`。
只追加此前未用于串行诊断的另外两道历史刷新题 `gpt4_2f8be40d`、`gpt4_7f6b06db`。
两题原版 A/B 同样逐块、逐 token 一致，但分别仅生成 2841、3776 tokens，没有触发刷新；
该目录也按覆盖门槛停止，未启动优化臂。扩展清单绑定了先前已保存的回答，禁止重试旧诊断 ID。

诊断只改变客户端并发为 1，**服务端仍保持原模型、compiled graph、8 个 native request 上限和
16424-token batch 上限**。所有输入、请求采样、thinking=4096、输出上限=16384、KV 预算不变。
复用原 Memory driver，只增加显式 `workers` 参数；正常调用默认仍为 8。
各臂有独立 runtime 清单和严格 verifier，不沿用旧入口的简单读取替换。

独立的窄范围诊断：`artifacts/g4k-gilopt-prefill-diagnostic-20261104-r1/`。
保持上述完整资格验证的失败状态不动，复用其已保存的稳定原版输出，仅执行优化版的首次串行回答。
**4/4 完全一致：88 次 prefill pressure 的全部逐层计划、64 个最终 Query 层选块、
完整输出 token IDs 和文本。**没有再次推理原版，也没有外部评分。
此结果只能证明这四题的 prefill/Query/完整回答一致，不能代替真实 decode pressure 或八路全部 102 题验证。
该诊断明确保存 `full_equivalence_qualified=false`、`performance_accepted=false`。

各冻结 launcher 均在没有工作区 PYTHONPATH 的 CPU 环境中通过导入/清单预检。
`handoff-preflight.json` 确认旧 driver 仍存活时会拒绝交接，不依赖过时的全局指针。
实际交接使用一次性 pidfd 进程退出事件，不是定时巡检；随后再次检查成功状态和真实 GPU 释放。
没有在原性能组运行期间启动其他 GPU 测量。

所有诊断均为已完成或已停止的不可重跑尝试，不要重新运行上述目录。
`qualification-r1` 的补丁路径准备失败、`qualification-r2` 的离线依赖预检失败均已保留，
二者没有启动 GPU 工作；首版 pidfd 包装器因 Python 缺少绑定的 CPU 设置失败也已保留。

## 后续优化方向

当前不再通过重采样历史题尝试凑 decode 覆盖，也不继续扩大核心推理改动。
下一步需要设计固定执行顺序、自然触发刷新且能复现的受控执行，不能修改 seed、输出上限或 Gen 预算来强造事件。
完整模型等价通过之后，优先分析 assembly lane 排队、CPU 数据准备及事件/传输依赖；暂不放松 staging
所有权或同步栅栏。已有约 90 秒 Nsight 诊断中，CUDA event synchronize API 累计约 18.89 秒，
`_assemble_split` kernel 累计约 0.075 秒；这些是带 profiler、可能重叠的聚合值，不作墙钟或吞吐结论。
尚无证据表明应先改 assembly kernel。

若要减少 prefill 重选次数，另设临时 prefill 空间配置并重新核算显存，不能混入本次 Gen4K 单次成本优化。
最后再做无 profiler、八路、完整 102 题对照，报告一致性/正确性、刷新次数、阶段分布与完整推理墙钟吞吐。
