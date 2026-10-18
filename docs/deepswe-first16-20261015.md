# DeepSWE first16：当前 vLLM + KVMem

## 本地实时监控

当前 r16 页面：`http://localhost:50123/`。启动入口：

```bash
python3 scripts/deepswe_dashboard.py artifacts/deepswe-first16-20261015-r16 --port 50123
```

页面每 2 秒刷新，只读现有运行记录，不向模型发送推理请求。吞吐沿用原生
10 秒采样：prefill 为 `Δ(prompt_tokens_total - prompt_tokens_cached_total) / Δt`，
decode 为 `Δgeneration_tokens_total / Δt`（含首 token）。两者均为混合负载
下的全 GPU 速率。预检和正式阶段分别显示相邻采样，正式开始后清除预检曲线；计数重置、缺项、采集
失败或长间隔显示断点，实时采样超过 35 秒未更新会提示。任务表覆盖全部
16 题并区分模型请求、工具／客户端、隐藏评分、完成和异常；GPU running
不等于同时 decode 的数量。查看监控无需停止现有推理服务。

## 当前重测：r16，128K active / 128K generation，4 并发

r15 于 20:04:52 在完整输出预检中失败，正式题目 0/16 启动。当时每路
累计逻辑历史约 252K，生成约 106K token，decode 约 294 token/s；没有
CPU 准入等待或原生抢占。Action attention 质量校验捕获到非正常概率和。

20:47 排查定位到 sampled-logits 内核的物理页地址乘法仍为 int32。
本轮 native 页长 1616，4 KV heads，K/V 交错维度 512，页 649 的偏移
为 2,147,909,632 bytes，超过 2^31。小规模高物理页测试在原代码稳定触发
非法访问；将页号在乘法前提升到 int64 后，9 项 GPU 数值对照和 65 项
CPU 回归通过。保留 Action attention 开关、公式和原始归一化校验。
证据为 `artifacts/deepswe-action-overflow-before-20261015.log` 与
`artifacts/deepswe-action-overflow-fixed-20261015.xml`。

r16 保持 r15 的模型、预算、并发、采样和全部预检门槛，仅修复上述读地址。
已重新排队启动；前一轮没有正式题目成绩需要沿用。监控切换至 r16。
以下 r15 时间点记录属于已停止的旧轮，不代表当前预检通过。

用户随后明确要求 **K=131,072 + G=131,072，4 并发**。r14 在等待 GPU
期间取消，没有启动正式题目；r15 按新预算冻结并启动。桥接器输出上限跟随
G，同时尊重客户端更小的 max_tokens；普通任务不强制写满 G。完整容量预检
会每路强制生成 131,072 token，全部通过后自动放行原 16 题。

生成预算与 Query 原子重放的单批上限已解耦：此轮 query_replay_budget=16,384，
原生批次上限为 16,404，普通 prefill 仍分 2K 块；过大的原子 Query 在 HTTP
入口拒绝，不会进入无法推进的引擎队列。G=128K 不再导致模型预热分配
128K 行临时激活。服务／请求规划／准入等 158 项回归通过；独立 GPU 大
缓冲区检查已通过，整模型 128K 输出验收在后台运行。监控现在单独展示
预检吞吐和已完成路数，不把它们混进正式成绩。

19:40，r15 的 4 路原子 Query 与短尾跨轮逐 token 一致性预检通过。
4 路长输入正在推进，GPU 利用率采样 100%；完整 G=128K 输出仍未完成，
正式 16 题尚未放行。后台控制器会在全部容量预检通过后自动进入正式阶段。

以下保留本次调整与故障排查记录：

用户在 19:11 要求处理停滞并改为 128K、4 并发。r12 引擎仍存活，
但 CPU history credits 已达到容量，7 个请求等待，最大的轨迹在工具往返
期间继续占用推进保障额度，GPU 因而长时间空闲。不是 CUDA 崩溃；旧轮
3 个已评分结果均为 0，另外 8 个轨迹未完成，原样保留，不混入新配置成绩。
证据在 `r12/configuration-transition.json`。

r13 使用 K=131,072、G=16,384、4 个 GPU active / agent worker；CPU
历史按此前用户授权设为 100,000,000,000 bytes（约 93.13 GiB），保留完整
历史和推进余量，不使用 SSD 或因容量压力丢弃历史。逻辑历史上限仍为 1M，
模型、MTP(4)、采样、attention 策略与隐藏评分器保持一致。重新执行全部
16 题，旧轮三个零分不会被当成新轮已完成题目。

修复普通 prefill cap 的区间判断：仅 `[replay_begin, replay_end)` 内保留
Query 原子重放，Query 后的普通历史重新受 2K cap / adaptive cap 约束。
原错误可在 CPU 复现为一次调度 16,416 token；修复后为 2,048。新增回归
覆盖原子 Query、紧邻其后的未对齐尾部及后续完整块；71 项调度／恢复／
准入回归、4 项监控检查通过。大窗口预检改为 4 路 141,870 输入，再以
145,842 输入跨轮恢复并各生成完整 16K，确保真正覆盖扩大后的 active。
监控会显示当前 K/G，并在 GPU running=0 且 CPU 有等待者时提示预算等待。

r13 的 Query 和短尾恢复已通过，但 19:26 长窗口重选触发 GPU 非法访问，
正式题目 0/16 启动。定位到 resident V gather 的目标地址由 32 位 program
index 计算；17 层 × 128K × 4 heads × 256 dim 的快照为 2,281,701,376
bytes，超过有符号 32 位范围。将 token index 在乘法前提升至 int64；新增
真实大缓冲区 GPU 测试，跨全 17 层逐字节核验，连同装配回归 25 passed、
10 skipped（不适用的优化组合）。证据：
`artifacts/deepswe-128k-large-assembly-20261015.xml`。

r14 采用相同配置和修复后的冻结源码，从预检重新开始；r13 原始错误记录保留，
没有把基础设施错误计入任务得分。当前监控已切到 r14，正式启动仍要求完整
4 路长上下文恢复与 16K 输出检查通过。

## 历史重测：r12；r10/r11 为故障记录（2026-10-15）

按用户要求，在当前版本重新运行原来的固定 16 题，8 个 agent 并发；
完成一题立即补入下一题。本轮每题一次全新正式尝试，不沿用 r8/r9 的分数，
不覆盖旧结果。当前输出目录为 `artifacts/deepswe-first16-20261015-r12`，
源码冻结在其 `frozen` 子目录，启动与配置差异记录在 `execution-plan.json`。

已核对 16 份指令、32 个 agent/verifier 镜像及原 Claude Code 二进制摘要。
协议适配库逐字节复用 r9；采样、题目、工具、提交 patch 和隐藏评分器不变。
重复终止规则沿用上次 r9：连续 32 次缓存 Read，或连续 32 次完全相同且
成功的 Bash 输入／输出。后一规则与原始 qw3 的 Read-only guard 不同，
不收敛结果单独标注，仍保留原 verifier 的分数。

推理使用当前 NVFP4、FP8 KV、原生 MTP(4)、编译图、K=36,864/G=16,384。
采用刚测过的 2048-token 普通 prefill 分块；MeanK、Query history、prefill／
output action attention 和 pressure selection 保持开启。guided 等策略仍
与 r9 一致关闭。块边界和新调度会改变生成轨迹，本轮不是单独的 attention
消融或与旧版完全等价的质量复现。

跨轮任务启用 engine CPU history credits、推进余量、维护协调和异步恢复，
关闭按延迟缩块的控制器。CPU authority 仍为 80 GiB，无 SSD fallback，不
驱逐或重算等待任务的历史。桥接器使用 engine admission，取消旧的固定
48K／单一 long_owner 限制，容量不足时只等待无法获得额度的请求。
这与独立离线样本的 B0 示例不同：agent 需要跨轮历史准入与恢复。

18:06 启动独立队列，等待 GPU 连续空闲 120 秒，再做原子 Query、短尾
混合 decode、8 路完整 16K 输出及跨轮恢复预检；全部通过才放行正式题目。
相关 CPU 回归 33 项通过。GPU 归属监测只会停止本轮自己的进程，保留
机器上其他容器。`status.json` 区分排队与运行，`run.json` 保存逐题状态；
`live-summary.json`、`native-metrics.jsonl` 保存实际并发、吞吐与引擎 CPU
准入等待。旧桥接器的 CPU wait=0 不能再解释为引擎没有等待。

18:10，r10 的 8 路原子 Query 预检通过，但短尾跨轮输出与不中断生成的
比较失败，正式题目 **0/16 启动**，已自动清理本轮进程和网络。失败结果
保留在 r10；正在 `deepswe-short-tail-debug-20261015-r3` 对比同步／异步
恢复并开启逐字节状态校验，尚不能将 token 差异归因于 CPU KV 损坏。
后续控制器会在断言之前保存完整短尾请求响应，避免只留下无 token 的
断言堆栈。没有放宽正式预检门槛，也未把此次失败计入模型正确率。

18:23 同输入对照定位到暂停／恢复的原生 MTP 占位信息丢失：B3 的 KV
（17 个注意力层，含 MTP）和 96 个循环状态张量与原 checkpoint 完全相同，
但恢复后的单 token 尾部丢失 `[-1] * 4`，从 5-token 验证形状改成 1-token
路径。B3 输出 `[198,2]`，同步 B2 和不中断生成均为 `[198,248046]`。
冷算完整两 token 输入也输出 `[2]`，因此仅靠冷算比较会漏掉这个执行路径差异。

修复在 KVMem 暂停原生首步时保留占位信息，恢复后按原形状重新准入；
预算不足只等待该请求，占位不计入真实历史。公开执行消费一次，取消时
清理；异步重选重复暂停也保留相同信息。未改原生验证内核或评分策略。
55 项异步恢复／重选／调度回归、50 项规划／原生状态／会话回归通过。
18:24 r11 冻结修复版重新排队，沿用相同 profile、原 token 一致性断言和
全套 GPU 预检；通过后自动运行 16 个新正式尝试。

18:28，r11 的 8 路 4K 原子 Query 和短尾 GPU 预检通过。修复后异步恢复
输出为 `[198,248046]`，与不中断生成一致；`short-tail-preflight-evidence.json`
保存完整响应，原断言未放宽。冷算多 token 路径仍输出 `[2]`，作为不同执行
路径的对照保留。正在执行 8 路完整 16K 输出／CPU 恢复，尚无正式题目分数。

18:33，8 路完整 16K 输出全部成功，8 个 namespace 的清理检查全部通过；
`generation-preflight.json` 为 passed。18:33:24 正式 first16 启动，首批
8 个 agent 并发运行，后续任务由空闲 worker 自动补入。正式任务与合成预检
分开统计；预检合计恢复 381,297 token、零 checkpoint deferral、零 KV
磁盘读写。18:35 首次快照完成 59 次正式模型请求、输出 8,197 token，
峰值 8 个在途模型请求，尚无已评分题目；该短窗口不代表完整测试吞吐。

18:38，发现 r11 的第 8 个首请求被会话槽位限制永久阻塞：原配置只留
17 个槽，预检的 10 个 namespace 在清理大块 KV 后仍各保留一个轻量
checkpoint。因此只剩 7 个槽可用于正式任务；不是 CPU 字节容量不足。
这在启用 preserve_history 后无法依靠旧 LRU 淘汰自行解决。r11 已停止并
完整清理自己的进程／容器，8 个中断轨迹保留为基础设施故障，0 个有效分数，
不记为模型答错。证据在 `r11/infrastructure-interruption.json`。

r12 将会话槽设为 32（16 个正式 namespace + 10 个预检 namespace 至少
需要 26），不改变 CPU KV 字节上限或 GPU active=8。启动前新增容量检查，
防止清理后保留的 namespace 耗尽槽位；容量与监督回归 7 项通过。全部
16 题重新开始独立正式尝试，本轮实际耗时不得混入 r11 或合成预检。

18:48，r12 全套 GPU 预检再次通过，8 路各完成 16,384 token 并通过
namespace 清理，正式 first16 首批 8 个 agent 已启动。启动 manifest 保存
槽位检查结果 configured=32、required=26；原子 Query、短尾输出一致性、
完整 G 和跨轮恢复均未放宽验收条件。

18:49，首批 8 个任务全部收到首轮回答，wasmi 也正常推进。正式阶段
连续三个监测样本记录到原生 running=8、waiting=0；证据保存在
`r12/formal-concurrency-evidence.json` 和 `native-metrics.jsonl`。
这是实际模型并发，并非仅客户端启动计数；后续仍可能因工具执行或
CPU 历史预算而变化。测试继续在独立后台进程运行，完成后自动执行原
隐藏 verifier、更新 `live-summary.json` / `comparison.json` 并清理本轮资源。

以下 r8/r9 章节为历史记录；r9 已按此前用户要求停止。

状态：r8 在首个任务评分后因控制器错误而中断，尚无完整 16 题成绩。
awilix 按不收敛规则结束，隐藏评分为 0（新增测试 0/24、回归测试 162/162）；
该结果保留，不重试。r9 续跑其余 15 题，计划目录为
`artifacts/deepswe-first16-20261015-r9`；`run.json` 保存任务状态，
`live-summary.json` 保存持续统计。只有原隐藏 verifier 完成且没有基础设施错误的任务计入有效结果。

## 对照与不变项

复用 qw3 `action_pressure_first16_warmfix_20261010` 的固定 first16，
16 份 instruction、32 个 agent/verifier 镜像和 Claude Code 2.1.148 二进制均已逐项校验。
每题单次正式尝试，工具、指令、提交 patch 的要求及隐藏测试不变。
客户端关闭自动压缩，effort xhigh，thinking 4096；temperature 0.6、top-p 0.95、
top-k 20、min-p 0、presence penalty 0、repetition penalty 1、seed 1000。
原来连续 32 次相同缓存引用 Read 的不收敛终止策略也被保留，命中时必须单独标注。

完整历史 baseline 为 16 题；较新的 attention 运行只有 6 个自然完成任务和 1 个
按不收敛策略结束的任务。两者分别比较，不能将未完成的新版当成完整 16 题对照。

## 当前执行配置

- Qwen3.8-27B NVFP4、FP8 KV、原生 MTP(4)、原生编译与 CUDA Graph。
- K=36,864，G=16,384，单次输出不超过 G，逻辑历史位置上限 1,048,576。
- 原版 MeanK；Query history、prefill attention、output action attention 和 pressure selection 开启。
- action scope=non-thinking，spread3，interval=16，fraction=0.5，quota fusion。
- guided/private growth/evidence refresh 沿用当前实现配置，未开启。
- 用冻结的 qw3 C++ helper 渲染 Claude 请求、识别 Query/必留内容并解析工具输出。
  工具往返保留已接受的原始输出 token IDs；每个任务独立 namespace。

这不是与旧版完全受控的 attention 消融：模型量化、推测解码、引擎、逻辑历史上限和
采样 RNG 实现均存在差异。原版 2M 位置上限在本次为 1M，达到上限必须报告，不能静默截断。
Anthropic 流响应等待原生非流式生成完成后发送完整内容，同时发送 keepalive；
代理流的首事件时间不能解释为模型 TTFT。

进一步核对 `launch.template.json`：qw3 较新对照使用 guided reselect=both、
guided Query 4096、middecode trigger 8192；本次 guided 关闭，是明确的策略差异。
本次不能视为对该 guided 配置的等价复现。

## 并发与 CPU 历史

用户选择在内存不足时降低并发，保留 CPU 历史且不重算。配置最多 8 个 agent，
CPU authority 上限 80 GiB，准入预算 72 GiB；KV 无 SSD fallback。

本模型当前 archive 每 token 为 17×4×256×3=52,224 bytes。
为保证正在增长的轨迹始终能推进到 1M 上限，保守预留一个完整 1M 历史，
其余最多七个任务各保留不超过 48K 的历史和本轮输出预留。
短轨迹阶段可形成 8 路模型并发；长轨迹阶段只允许一条继续增长，其他任务保留 CPU KV 等待。
完成任务后用原有 namespace 失配及 fenced 回收机制释放历史，再放行下一条。
因此“客户端 8 并发”不等于“全程 8 路持续 decode”；统计必须报告实际并发和排队。

## 已完成验证及故障记录

- 原始 8 客户端、16 次工具往返请求通过；跨轮恢复成功。
- 约 73K 输入的选页及工具返回恢复通过。
- 原版 Claude Code 容器真实工具调用及恢复通过。
- 正式 first16 初次启动暴露：原子 Query 超过 2048 token 时，stride=32 会产生超过
  64 个 attention 采样点，原实现拒绝该请求并终止引擎。首批没有返回任何模型回答；
  此批不计成绩，记录保留在 `artifacts/deepswe-first16-20261015/infrastructure-failure.json`。
- 修复保留全部采样点，按最多 64 个 Q 分批计算，保持每个 Q 的完整 causal 可见范围。
  25 项测试通过，其中 GPU oracle 包含 97 个采样 Q 的 BF16/FP8 数值对照。
- r3 启动前的 8 路、每路 4K 原子 Query 回归及旧会话必留内容清理回归均通过。
- r3 正式任务返回 56 次模型回答、20,461 个输出 token，峰值 8 路，累计恢复
  1,245,236 token CPU 历史，零 checkpoint deferral 和 KV 磁盘读写。但在一个工具续轮
  51,634 token 输入后，没有预先保留完整 16K 输出空间，decode 中途触发仅支持
  prefill 的异步刷新，引擎退出。所有任务保留为基础设施失败，不计通过率。
- 修复在最后一个可重放的 prefill 边界检查实际输出上限，必要时以原 pressure 排名重选，
  为剩余 prompt、完整输出和一块 MTP 余量让出空间。最终 Query 原子重放使用相同预留规则。
  CPU 历史不裁剪；仍能容纳续轮的选页不变。原生 pressure selector 增加可选单次预算参数，
  默认行为不变，适配文件及哈希在 `third_party/qw3/PROVENANCE.json` 的 `local_adaptations`。
- 修复后 88 项请求规划、pressure 选页、异步刷新、会话和原生事务测试通过。
  r4 自动执行 8 路约 51K 输入＋完整 16K 输出、CPU 跨轮恢复的真实模型回归，然后准入 first16。
- 控制器在模型服务或任一任务出现基础设施故障时停止后续准入，避免错误后继续派发题目。

预检请求与正式任务分开记录。r2 在题目准入前因补充清理参数而取消，不含正式任务尝试。
端到端 output token/s 含工具执行、预填充和 CPU 准入等待，不能标成稳态 decode 吞吐。

r8 正式阶段发现 `awilix-async-container-initialization` 连续重复同一条 Bash 读取，
一次快照中已有 130 次相同成功返回。原规则只处理缓存 Read，故没有触发。
证据在 `bash-repeat-observation.json`。后续选页快照显示 pressure 重选会移除
大部分最新 sealed suffix，只重放最终 decode-seed；见 `loop-selection-observation.json`。
这只是相关现象，尚无消融证明循环的根因。等待可选偏好回复后，先采用连续 32 次
相同成功 Bash 输入/返回即停止的不收敛规则；这是执行方采用的默认值，非用户明确确认，
也是与原版只检查缓存 Read 不同的测试规则，必须单独标记。r8 接管控制器时保留了
模型、客户端与 CPU 历史，但新增规则启用较晚，awilix 耗时不能视为从开始就应用该规则的性能。
评分后控制器未识别 `Cancelled via cancel scope` 这一有意取消格式，误将其判为故障，
停止了其余任务。修复改为类型化取消事件，并对停止时间、取消类型做校验；5 项回归测试通过。
独立审计 `r8/awilix-adjudicated-result.json` 保留已经完成的隐藏评分；r8 原始错误记录保留。
其他 7 个中断任务没有完成隐藏评分，r9 重新启动这些任务及尚未启动的 8 题。
启动时的原控制器源码保存在 `controller-original.py`，SHA-256 与启动记录完全一致。

## r8 正式请求吞吐（不含预检）

依据 `artifacts/deepswe-first16-20261015-r8/throughput-audit.json`：

- 762 次完成的模型请求，121,684 个输出 token。
- 从正式准入到最后一次成功响应共 2,088.93 秒，混合端到端输出 58.25 token/s；
  包含工具执行、prefill、选页与排队，不包含服务停止后的清理耗时。
- 请求加权单流 decode 为 106.72 token/s；不能解释为 GPU 合计吞吐。
- 后半段 1,289.53 秒监测窗口内，新增 prefill 262,573 token、请求 prefill 阶段累计
  573.46 秒，增量 prefill 为 457.88 token/s。该阶段包含恢复/选页/装配，
  不包含已经缓存的历史 token，不代表全新长文本的纯 prefill 能力。
- 平均原生 TTFT 1.32 秒，不包含桥接器前面的 CPU 准入排队。
- 峰值 8 个在途模型请求，但大部分时间为 1 个长轨迹和其余等待客户端。

这组数值描述当前实际执行效率，不能作为完成的持续 8 并发成绩。长轨迹的保守
CPU 预留压低了实际并发；重复工具调用也使生成 token 数无法代表有效任务进展。

新增回归记录：r4 在正式题目启动前暴露 terminal checkpoint 发布竞态；完整页写入 CPU archive 尚未完成时，checkpoint 就被标为可恢复。修复以非阻塞轮询等候 sealed publication，仅保留当前 namespace 在 retiring 状态。r5 重复同一 8 路完整输出回归，并额外覆盖其中一路新 Query 重放。

r5 的 8 路长输出均越过原故障点；首路生成完整 16K 后，其单 token 清理请求加入 decode batch 时触发原生 MTP padding 断言。修复保留原生占位行的物理预留，私有 Query 评分仍只使用公开输入 token；55 项相关测试通过。r6 会重复完整生成、跨轮恢复与并发清理，所有预检完成后才执行正式题目。

r6 确认占位调度通过，但发现 terminal GDN 状态捕获仍把该混合步误判为非 speculative。修复按完整已执行范围是否超出 prompt 判断，27 项状态/检查点/异步测试通过，1 项 GPU 专用测试未在 CPU 测试中运行。r7 先验收短请求混入 decode 的冷启动、CPU 恢复和相同输入冷算 token 对照，再跑完整 8 路 16K。

r7 的短请求冷算对照不一致。进一步诊断发现冷启动单 token 走原生 decode 时会读取复用的 GDN 页，添加仅作用于该冷启动的零状态初始化；CPU 恢复前后 34,816 bytes KV 和 96 个 recurrent tensor 逐字节相同，连续生成前后重复均为 `[198,248046]`，恢复下一 token 为 `[248046]`。同一完整输入在 mixed cold prefill 路径为 `[2]`，因此不能把不同 batch/前向路径的 greedy token 完全相等作为缓存字节正确性的定义；该差异保留记录。r8 校验恢复与不中断生成一致，同时记录 mixed cold 对照结果。证据见 `artifacts/deepswe-short-tail-debug-20261015-r2/qualification.json`。
