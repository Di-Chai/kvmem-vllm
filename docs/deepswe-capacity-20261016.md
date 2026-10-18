# DeepSWE：active KV 容量收益对照

目标：验证在同一张 RTX PRO 6000 上，KVMem 减少 GPU 常驻 KV 后，能否让更多真实 agent 同时推进并增加吞吐。

## 2026-10-17 验收结论

已完成四组真实 DeepSWE 30 分钟窗口，以及同输入的 8 路真实长历史准备／续接对照。真实工具窗口中，4／8 agent 的接受生成率分别提高约 6.86%／15.96%；它们来自修复前的 qualified P1，轨迹会因选页不同而分化。

修复后的长历史对照中，原生完整 attention 页池不足以同时驻留其中 7 个会话，KVMem 已实测同一步 8 路 decode，完整 CPU 历史保留。相同续接输入、8×8192 输出，原生／KVMem 为 135.75／498.95 token/s；包含冷准备后为 58.83／115.83 token/s。KVMem 零抢占，KV 无 SSD 读写。详细定义、profiling、失败记录及修复见文末“最终同输入长历史验收结果”。

本轮结论针对容量与吞吐，未声称解题质量等价、8 路均填满 256Ki 历史或通用稳定加速倍数。最终受控运行代码完成 205 项 CPU 回归、两请求 GPU 边界检查及完整 8 路 GPU 验收；原始记录及最终复核均保留。

## 冻结协议

- 使用 qw3 `action_pressure_first16_warmfix_20261010` 的原始 16 题、指令、Claude Code 二进制、agent/verifier 镜像、工具和任务顺序。
- 四组独立运行：原生 vLLM 4 agent、KVMem 4 agent、原生 vLLM 8 agent、KVMem 8 agent。
- 每组从全新容器开始，正式观察 1800 秒。完成一题后补入下一题。窗口结束时未完成的尝试标记为截断，不记为模型答错，也不伪造隐藏评分。
- 原生 vLLM：256K 完整窗口、APC、async scheduling、continuous batching，max_num_seqs=8。由原生调度器准入，不人为限制为五个 agent。
- KVMem：64K active、32K generation budget、256K CPU historical ceiling、100,000,000,000 bytes CPU cache；原版 MeanK、Query history、prefill/output Action attention，沿用 r3 P1 已验证运行时。
- 两边相同 NVFP4 模型、FP8 KV、MTP(k=4)、CUDA Graph、GPU memory utilization=0.81、2K 普通 prefill chunk，以及相同 token 拼接、采样和独立用户 cache salt。
- 每个后端先处理相同 67,584-token 初始化请求，KVMem 必须实际执行 MeanK CUDA 评分；清理并退休后才开始正式计时。初始化耗时单独记录。
- 单轮输出上限为 generation budget，尊重客户端更小的上限。完整历史加预留输出超过 256K 时记录 context ceiling，结束该尝试、释放历史并补入下一题，不静默截断历史。
- 保留原有缓存 Read 重复规则，以及此前采用的 32 次相同且成功的 Bash 重复终止规则。

## 测量与解释

每秒保存原生完整 metrics、GPU 利用率、显存、系统可用内存和 KVMem 的 CPU 历史／调度状态；每轮请求记录完整输入长度、新增输入、准备／排队时间、输出 token、原生 TTFT／decode 时间及 MTP 接受率。

主要指标包括：每段服务时间内实际计算的 prefill token/s、接受的 generation token/s、完整模型回合数、工具调用数、任务推进与完成情况、排队时间、抢占、历史长度分布、CPU/GPU 内存及实际 GPU decode batch 大小。KVMem 模型步骤记录用于证明同一步的 decode 并发；原生 running gauge 单独列出，不能当作纯 decode batch 数。

按时间及历史长度分段，至少区分短历史、超过 64K、超过 96K 的阶段。所有 agent 等待工具的区间保留在整窗吞吐和 GPU 利用率内。阶段吞吐与整窗吞吐分别列出。

这是两个部署系统的实际任务对照。KVMem 选页可能改变输出和后续工具轨迹，固定随机种子也不能使两边的工作量完全相同。因此不能仅凭一次固定时间窗，宣称纯插件开销或回答质量等价；需要结合历史分布、实际并发及 profiling 判断容量是否转化为收益。

### Decode 与实际并发的证据

整窗生成吞吐保留工具等待与 prefill；每请求 decode 速率另按原生首输出至末输出时间计算，排除首 token。还提供保守的无新请求 prefill 采样窗口：用原生 queue／TTFT／decode 时长及桥接器 admission／completion 时间界定持续生成区间，并排除所有可能的 prefill 区间。尚未完成、拿不到原生时间戳的请求按“仍可能在 prefill”处理，因此该指标可能缺少较多覆盖，必须一起查看覆盖秒数。

不能只凭 prompt counter 没有增加就断言没有 prefill：当前 vLLM 在请求首输出时一次性报告它的 prompt 统计，分块 prefill 期间可能尚未更新计数。原生 mid-decode 抢占重算也可能落入服务采样窗口，所以这些数字属于服务代理指标，不能当作纯 GPU kernel 速度。

KVMem 的 GPU 模型步骤记录直接报告 batch 请求数。原生另用 iteration token histogram 提供同一步有输出的请求数下界：MTP(k=4) 每请求每步最多输出 5 token；将可能被完整 prompt credit 增大的观测按每个 HTTP 请求最多一次全部扣除后，仍出现大于 16 的观测，才能证明某一步至少有 4 路请求产出 token。采用前一次 scrape 的总计数减后一次 scrape 的 bucket 计数，以保守处理采集非原子性。这是推导的并发下界，不能当作精确的纯 decode batch 大小，也不能定位这些步骤各自的历史长度。

## 当前证据

入口：`scripts/benchmark_deepswe_capacity.py`，当前续测产物：`artifacts/deepswe-capacity-20261017-r3`；原生 4 并发结果位于 `artifacts/deepswe-capacity-20261016-r2/native-c4`。

新增原生 transport、固定时间窗及相关回归共 111 项 CPU 测试通过。四组任务和模型配置已冻结；GPU 测量结果由产物中的实际运行记录决定。

r1 在原生 vLLM 首次初始化、CUDA 内核编译期间触发 8 GiB 系统可用内存保护，正式任务启动数为 0。本轮进程与网络已清理，失败产物保留，不能作为吞吐结果。停止前观测到 17 个 nvcc/cicc 编译工作，但 r1 没有连续启动内存记录，不能排除其他主机工作对峰值的影响。

r2 在两个后端统一设置 `MAX_JOBS=4`、`FLASHINFER_NVCC_THREADS=1`，使用已核对的 FlashInfer/PyTorch 编译控制接口，并保存 `resource-guard.jsonl` 启动及运行期内存/GPU记录。受限并发的首次编译允许更长启动等待，正式窗口仍为 1800 秒。KVMem 核心源码、四份模型/预算配置、初始化 token 与 r1 相同。定时退出／上下文上限补入及统计相关的 8 项针对性回归通过；原生 HTTP transport 在宿主环境的 3 项回归也通过。

r2 的实际 Ninja 命令已核实为 `-j4`；首次编译期间最低系统可用内存约 77.6 GiB，模型随后成功就绪。原生 4 并发完整窗口已完成：观测 1799.32 秒，公共 computed prefill 1,942,365 token、接受生成 411,114 token，对应约 1,079 和 228 token/s。4 路已知持续生成且排除新请求可能 prefill 的采样窗口覆盖 359.07 秒，约 413 token/s。平均 GPU 忙碌率约 97.8%，零抢占、零接口错误，完成 447 个模型回合、544 次工具调用。任务状态为 1 个自然完成并评分、1 个触及上下文上限、4 个窗口截断；这些状态不能合并成“6 题成功”。

随后外部 GPU 队列在模型切换间隙启动第三个任务，KVMem 4 并发在初始化阶段被归属保护停止，正式任务数为 0。r3 对已完成的原生 4 并发测量、配置、采样、任务和代码做内容摘要校验并复用；仅执行剩余 KVMem 4／原生 8／KVMem 8 三组。新的依赖门禁等待外部控制器原始 PID＋创建时间对应进程全部退出，然后再检查 GPU 连续空闲，保留对新外部 GPU 进程的保护。依赖结束、PID 复用、命令变更和拒绝不完整／不兼容复用的 5 项回归通过。

### 不干预模型的 profiling

`scripts/profile_deepswe_capacity.py` 只读取固定窗口产物，输出 `live-profile.json`，不向模型提交请求。它区分有／无模型 HTTP 请求时的低 GPU 利用率，按实际 decode batch 和 active context bucket 统计模型及维护延迟，并统计 resident 恢复、检查点同步和按完整历史长度分组的每用户 decode 速率。后者的请求时长可能重叠，不能当成聚合 GPU 吞吐。

公共 forward 的计数通过已有非阻塞完成 fence 给出上下界，包含中间 prefill chunk，不把恢复的 KV 或私有 Query 重放记作 prefill token。主机维护时长、CUDA 时长和请求恢复时长可能重叠，不能相加成 GPU 时间占比。有关 GPU 低利用率、计数上下界、batch/context 分组、检查点阶段、缺失计时及输出预算的 7 项纯 CPU 回归通过。

当前客户端实际每轮申请 32,000 输出 token。64K active＋32K generation 的总 active 容量为 98,304；预留最大输出及一个 32-token 页后，相对 65,536 选页预算只剩 736 token。上轮生成和本轮新增工具内容会消耗这部分余量，因此可能频繁压力重选，即使每轮实际输出远小于 32K。这是预算语义带来的维护成本，本次对照保持两端输出上限和 KVMem 预算不变。

`scheduler_resources.session_views` 仅公布已收到 worker 退休确认的 idle 会话；正在计算或恢复的会话不在列表里。历史／active 的统计因此可能低估正在计算的用户数。已经退休确认的会话不应因另一个请求仍有 DMA 或退休操作而被判为未发布；分析器仍要求 CPU 字节覆盖所报告的全部 sealed 历史，并拒绝磁盘 I/O。该诊断只证明发布及空间记账，不替代内容逐字节校验，也不能证明所有用户同时在 GPU 上。

### 完成的 4 并发窗口

| 指标 | 原生 vLLM | KVMem |
| --- | ---: | ---: |
| 观测秒数 | 1799.32 | 1799.62 |
| 接受生成 token / 服务墙钟秒 | 228.48 | 244.15 |
| 公共 computed prefill token / 服务墙钟秒 | 1079.50 | 561.34 |
| GPU 利用率采样均值 | 97.77% | 73.84% |
| 完成模型回合 / 工具调用 | 447 / 544 | 1092 / 1222 |
| 自然完成 / 上下文上限 / 窗口截断 | 1 / 1 / 4 | 0 / 4 / 4 |
| 抢占 / 接口错误 | 0 / 0 | 0 / 0 |

KVMem 的接受生成吞吐高约 6.86%，但轨迹和输出长度差异很大，不能据此认定任务解决效率或质量提升。较低的公共 prefill 计数也不是 prefill kernel 变慢的证据：这是计入缓存复用后的实际计算量除以整窗时间，两个窗口的工具返回和重算工作量不同。

KVMem 有 11,132 个同一步 4 路 pure decode 的记录，证明使用了原生 batch 能力。4 路已知持续生成、排除新请求可能 prefill 的采样代理约 444 token/s，但仅覆盖 52.36 秒；原生约 413 token/s、覆盖 359.07 秒，不能把两个覆盖不同的代理当作严格同负载 benchmark。

KVMem 共 569 次正式压力重选，平均请求跨度 367 ms、p95 630 ms，其中装配平均 213 ms。装配上传累计 369 GB；这些跨度可能重叠，不能相加解释为 209 秒独占 GPU。1086 次恢复均记录为 resident 路径，平均 129 ms、p95 279 ms，没有 active KV 上传；另有 recurrent 状态搬运记账。CPU authority 峰值约 24.02 GiB，最低系统可用内存约 64.20 GiB，磁盘读写计数均为零。阶段清理成功，详见 `artifacts/deepswe-capacity-20261017-r3/kvmem-c4/reviewed-window.json`。

上述窗口未证明 8 路长历史并发或满 256K 用户的实测容量收益；原生 8／KVMem 8 仍是必要对照。

### 原生容量及同一步长历史的核对

`cache_config_info` 提供启动后实际的 `num_gpu_blocks`、`block_size` 和 `kv_cache_size_tokens`。原生 4 组为 984 页、1,425,136 等效 token；原生 8 组为 1003 页、1,452,654 等效 token，满 256K 的启动估计为 5.54 路。相同的 `gpu_memory_utilization` 不保证每次初始化获得完全相同的页池，因此使用每组的实际值。

报告中的等效 token 容量按满上下文的混合模型状态计算，不能直接作为混合长度请求的硬阈值。更严格的检查是：对每个独立用户的完整输入计算 `ceil(prompt_tokens / 1616)`，若总数已经超过原生全部共享物理页，即使忽略 recurrent/MTP 额外状态，原生也无法同时驻留这些完整 attention KV。用户使用不同的 cache salt，不能共享前缀页。

`live-profile.json` 的 `capacity_overlap` 将这个需求与实际 KVMem public pure-decode 步骤匹配：步骤必须含完整目标并发数，同时外层 HTTP 区间必须恰好对应相同数量的可能用户。区间向两侧扩大 20 ms；若存在额外未结束请求就拒绝匹配，不能收窄区间让旧的未结束请求替代边界上的真实请求。cleanup 请求只输出一个 token，没有 decode 步骤。仅 HTTP 重叠、仅 running gauge 或仅超过等效 token 数，都不算物理容量收益的充分证据。未观测到超过阈值时，必须保留“本次 live 任务未充分施加容量压力”的结论，必要时补充相同真实历史的容量诊断。

原生并发下界另按 300 秒分段报告。活跃阶段的起点并不静止，采用后一次总计数减前一次 bucket 计数作为旧 tail 的上界，再从阶段末的 tail 下界扣除，保守处理起点采集竞争；该方法不会把 running gauge 直接当作 GPU batch。

进一步核对 normal-MTP 的请求验证计数：scheduler 对同一个模型输出中的每个请求最多 `observe_draft` 一次，随后将聚合统计放入一个 EngineCoreOutputs；前端对有输出的该对象记录一次 iteration histogram。在原生零抢占前提下，用内侧 scrape 的验证请求次数差，除以外侧 scrape 的输出迭代次数差并额外增加 2 个边界迭代，可得到同一目标验证步的请求数下界。存在抢占时，stale 统计可能没有对应输出迭代，分析器会拒绝这个方法。它不等于纯 decode kernel batch 或历史长度映射。

原生 8 并发整窗的保守内侧验证次数为 120,490、外侧迭代上界 16,725，得到下界 8；因此两套系统均能在本次较短历史的 live 负载中组成 8 路目标模型工作，不能把 KVMem 的收益解释成原生完全无法 batch8。新增方法及采集竞争／多序列／计数重置／抢占拒绝相关的统计文件共 15 项回归通过；新 source audit 在 `artifacts/deepswe-capacity-20261017-r3/native-sequence-bound-source-audit.json`。

### 原生 8 并发完整窗口

原生 8 并发已完成 1799.57 秒，生成 454,355 token（252.48 token/s），公共 computed prefill 4,050,450 token（2250.79 token/s）。GPU 利用率采样均值 98.94%，零抢占、零接口错误，完成 571 个模型回合及 765 次工具调用，8 个任务均在窗口结束时截断，未产生自然完成的评分结果。

8 路已知持续生成、排除新请求可能 prefill 的采样代理约 594.63 token/s，覆盖 156.04 秒。这个代理不能当作 30 分钟端到端吞吐。最后两个 5 分钟段，整窗生成率约从 174 降到 40 token/s，同时公共 prefill 计算率约从 2279 升到 4072 token/s，GPU 仍持续忙碌。不能用高 GPU 利用率单独代表有效生成吞吐。

此窗口 KV 使用率峰值仅 82.53%，完整 HTTP 输入需求峰值 1,125,294 token，原生无抢占。它不足以孤立验证缩小 active KV 带来的容量收益，因此需要下面的同输入长历史恢复诊断。完整结果和清理证据在 `artifacts/deepswe-capacity-20261017-r3/native-c8/reviewed-window.json`。

### 补充：相同真实长历史的准备／恢复诊断

输入为已完成 KVMem 4 并发窗口中 4 个原始任务各 2 个不同回合，共 8 个不同历史切片，总计 1,836,509 token。每个用户使用独立盐，输入没有复制、填充或裁剪；这不代表 8 个不同任务，也不用于任务正确率比较。`scripts/export_deepswe_capacity_history.py` 从最初回合起用原来的协议、tokenizer 及真实接受输出 ID 还原，每一轮的完整输入都须匹配原计划 token SHA。还原仅在 KVMem 8 初始化期间执行，约 46 秒完成，早于正式窗口。

诊断产物为 `artifacts/deepswe-history-capacity-20261017-r3`。仍用 NVFP4、FP8 KV、原生 MTP(k=4)、CUDA Graph、64K active＋32K gen／256K history／100GB CPU；两个后端都接收全部 8 个请求，由原生调度器自行准入，原生排队和抢占允许被观察，不能人为限为 5 路。

先以 1-token 请求准备全部完整历史并等待退休，再同时续接固定 8192 输出。续接输入采用原记录中的下一 token；即使准备阶段两端预测不同，也不把这个预测追加到测量输入，从而保证续接输入完全一致。准备阶段最后预测 token 尚未进入保留 KV，精确前缀仍可复用。采样与真实 agent 一致，但固定输出使用 `ignore_eos`，因此属于恢复／容量诊断，不属于真实工具执行或正确率测试。分别保存冷准备时间、实际公共 prefill、恢复轮时间、接受输出、client overlap 和真实 GPU batch。整轮时间包含准备，不能只用恢复吞吐宣称冷启动总效率提升。

CPU 验收要求每个用户最终完整历史和 M 检查点匹配、active view 不超过 96K、CPU 字节覆盖全部 sealed 历史、FP8 codec、无 GC、磁盘读写均为零。KVMem 还必须出现同一步 8 路 decode；只有 client overlap 不算。模型返回的逻辑 prompt 计数必须与完整输入长度一致。准备／恢复阶段的 prefill 和生成计数单独对账。

输入还原、固定续接、准备屏障、无客户端 5 路门槛、保留原生抢占、完整 CPU 历史／逻辑 prompt 校验及原有容量工具相关回归共 26 项通过。首次准备在 GPU 启动前发现 build-control 参数错误，产物 r1 保留；r2 在增加逻辑 prompt 校验后被 r3 替代，均没有 GPU 测量。r3 已冻结并校验，队列按原始主控制器 PID＋创建时间等待四个真实 agent 窗口完成及清理，再获取 GPU 锁执行。它没有抢占正在进行的 KVMem 8 测试。

两个诊断后端统一 CUDA_HOME 指向同一个已有工具链路径，MAX_JOBS=4、NVCC threads=1，以避免等价工具链别名造成重复初始化编译；编译器内容摘要和控制接口也被冻结。这只规范初始化缓存，不改变模型和推理参数。

### 完成的 KVMem 8 并发窗口及主对照结论

四组真实 agent 窗口均已完成并清理，结果如下。prefill／生成均为原生计数差除以约 1800 秒服务墙钟时间，包含工具等待、恢复及其他阶段，并非独占 GPU kernel 速度。

| 负载 | 公共 computed prefill token/s | 接受生成 token/s | GPU 利用率采样均值 |
| --- | ---: | ---: | ---: |
| 原生 4 agent | 1079.50 | 228.48 | 97.77% |
| KVMem 4 agent | 561.34 | 244.15 | 73.84% |
| 原生 8 agent | 2250.79 | 252.48 | 98.94% |
| KVMem 8 agent | 1000.12 | 292.77 | 66.94% |

本轮 KVMem 4／8 并发生成率分别高约 6.86%／15.96%。KVMem 8 共 526,737 个接受生成 token、1,799,381 个公共 computed prefill token；已完成回合交付 525,695 个输出 token，共 1349 个回合、1539 次工具调用。任务为 1 个自然完成、3 个上下文上限、8 个窗口截断，不能当作 12 题成功或与原生进行解决率比较。两边轨迹和回复长度不同，这些是部署系统的单窗口观察值。

KVMem 8 有 4568 个真实同一步 8 路 pure-decode 记录，原生统计也证明过 8 路目标验证。KVMem 的保守“无可能 prefill”8 路服务代理仅覆盖 1.06 秒，不能将其中约 654 token/s 当成可靠的稳态结果。固定长历史恢复诊断会提供更可控的持续生成区间。

KVMem 8 零抢占、零接口错误，历史／active 预算诊断未发现越界。GPU KV 使用率峰值约 50.82%，CPU authority 峰值 40.59 GiB，最低系统可用内存 35.72 GiB；CPU backend 的累计磁盘读写为零。GPU cache pool 初始化后已预分配，不能用 nvidia-smi 已分配显存代替占用页数。

profiling 记录 784 次正式压力重选，平均跨度 475 ms、p95 953 ms；其中装配平均 235 ms。装配上传总计约 504 GB，但这些工作与其他请求重叠，不能把跨度相加作为独占 GPU 时间。1341 次恢复走 resident KV 路径，平均跨度 216 ms、p95 563 ms；仍有 recurrent 状态恢复。检查点同步等待 p95 为零，当前没有证据说明它是主瓶颈。

8 路 pure decode、最大 active context bucket 为 64K 的步骤中，CUDA model span 均值约 43.75 ms、主机提交 span 约 27.33 ms、提交前维护 span 约 1.63 ms。这些跨度可重叠且不是逐 kernel 活跃时间，不能相加成占比。50 秒被动采样中原生 8 约 0.52 CPU 核、KVMem 8 约 1.13 核，采样分别从正式时间 420.34／456.45 秒开始，轨迹不同；只能说明额外主机工作，不能直接宣称 CPU/GIL 饱和或纯插件开销翻倍。

较低的 GPU 利用率同时伴随更高生成率和更少的公共重复 prefill，说明不能单独以 GPU 忙碌率评估效果。实时任务仍未充分孤立容量上限，完整目标的结论要合并相同长历史、相同续接输入的容量诊断。主窗口原始记录在 `artifacts/deepswe-capacity-20261017-r3`；补充诊断在 `artifacts/deepswe-history-capacity-20261017-r3`，后者已在主控制器全部结束后获取 GPU 并运行。

### 相同长历史诊断：原生阶段复核

原生阶段全部 8 个准备请求及 8 个续接请求完成，计数与实际交付输出一致，已清理。本次实际页池是 991 页、每页 1616 token，而非上一 live 窗口的 1003 页。8 份独立完整历史的 attention 页需求为 `[143,143,142,143,142,143,143,143]`，总计 1142 页；最短 7 份也需 999 页。因此不计 recurrent/MTP 额外占用，原生也至多同时驻留其中 6 份完整历史。这是约 224Ki 输入的保守物理上界，不代表实测满 256Ki 用户数，也不是 running gauge 的解读。

冷准备 631.25 秒，公共 computed prefill 1,836,509 token，按准备墙钟约 2909 token/s。固定 8×8192 续接输出耗时 482.76 秒，生成率 135.75 token/s；续接还计算了 698,853 个公共 prefill token，按整个续接阶段约 1448 token/s，不能视为独占 prefill 算力。冷准备加续接总计 1114.03 秒，对应输出率 58.83 token/s。准备／续接分别观测到 2／1 次抢占。

续接 TTFT 中位 9.17 秒、p95 317.91 秒、最大 406.30 秒；MTP 草稿接受率约 74.23%，平均接受长度 3.97。续接 GPU 利用率约 99.35%，但这包含重算及队列干扰。这里没有纯 decode kernel 基准或任务解决率结论。

冻结结果中的旧 histogram 并发下界依赖“每个 HTTP 请求至多一次 prompt credit”，在抢占重算时不可靠；`native-stage-reviewed.json` 显式将它作废，保留原始结果。当前分析器已拒绝存在或缺失抢占计数的这类证明，17 项统计回归通过；没有修改正在运行的冻结推理代码。KVMem 阶段尚需完成后才能比较恢复及冷启动总效率。

### 长历史验收发现的恢复边界问题与修复

补充诊断 r3 的 KVMem 8 路准备全部完成，但续接生成约 156～261 token 时引擎报 `Async refresh requires a complete public prompt/Query prefix`。失败未通过验收，也未作为吞吐结果。上一轮 max_tokens=1 可留下接近 K+G 的 M view；下一轮改为 8192 输出，完整匹配的部分块 M 前缀可能已经越过本轮最后的 sealed prefill 边界。继续复用旧 view 会在 decode 阶段耗尽余量，而现有异步重选只允许公共 prefill／Query。

修复按本轮实际输出上限做恢复准入：仅在精确 M 前缀已越过最后 sealed 边界、且旧窗口无法容纳本轮剩余输入＋输出＋一个块余量时，在 CPU-authoritative 恢复过程中执行既有 Action pressure 选页。完整历史和 M 模型状态保持，必留页及 accepted partial tail 保留；新窗口大小先由 scheduler 保留，worker 必须核对实际装配大小。涉及压缩的旧 resident 页不能直接复用；其他有足够空间的续接仍走原来的 resident 快速路径。生成过程中不新增重选，也不 rewind／重算长历史。

本次测量仅在原 qualified P1 上加入 5 个恢复相关文件的上述修复，排除工作区尚未资格验证的 native-frame／host-timing 改动。差异及摘要在 `artifacts/partial-terminal-restore-20261017-r1`；实际冻结运行代码通过 112 项恢复、取消、partial-tail、必留页、resident 隔离及请求规划回归。

修复后的 r4 在 GPU 启动前被主机内存保护拒绝，可用内存约比假定所有 8 路都达到满 256K 的保守预算少 0.31 GiB，没有 GPU 测量。r5 只对这项固定输入、固定 8192 输出的有限诊断，按每路 `input_tokens+8192` 的严格最大历史长度核算 payload，并保留完整 256K MeanK 索引、32 GiB 模型／DMA／系统余量和 8 GiB 最低可用内存保护；实际所需启动可用内存为 107,045,060,608 字节。CPU 100GB cap、历史上限 256K 和全部推理参数不变，live agent 的一般准入仍使用满 256K 预算。不能把这项诊断宣称为 8 路满 256K 历史的验收。

产物 `artifacts/deepswe-history-capacity-20261017-r5` 复用 r3 已完成且输入／profile 一致的原生阶段，逐文件冻结其哈希，只重新执行 KVMem。对照复用、固定历史内存上界和原有容量／输入工作流共 29 项回归通过。所有失败尝试均保留，没有覆盖或自动重启。

后续 r5 在恢复准入交接时失败：scheduler 预留的是压缩后的窗口，但 `RestoreAdmission` 仍按旧 M 窗口创建 permit，因而拒绝新窗口的 RestoreCommand。新增覆盖 ledger→permit→allocation→dispatch→completion 的回归，并修复准入窗口取值。

为避免重复长测，增加两请求 GPU 资格检查：各取真实记录前 98,239 token，仅作边界复现，先生成 1 token，再以相同前缀及一个固定续接 token 生成 512 token。此时旧 active view 为 98,239，应恢复为 65,535，保留完整历史与 M 状态。这个前缀切片测试不用于容量、吞吐或正确率比较。

首次短检查 `artifacts/partial-terminal-gpu-20261017-r1` 发现开启 Action pressure 时，清除旧物理驻留标记后再读取 working set，候选集合为空。恢复路径现在将检查点保存的逻辑 active 页显式传给原来的 pressure scorer；正常在线 pressure 仍使用当前 working set。新增开启 Action 的回归证明旧 active 中高分页面优先于未评分的新页面，并保留必留页和 partial tail。第二次短检查为 `artifacts/partial-terminal-gpu-20261017-r2`。

最终候选运行代码是原 qualified P1 加 7 个文件的恢复修复，记录在 `artifacts/partial-terminal-restore-20261017-r3`；对应冻结代码的 145 项 CPU 回归通过。`deepswe-history-capacity-20261017-r6` 在短检查失败后没有启动 GPU，已标为被替代。完整同输入诊断更新至 `deepswe-history-capacity-20261017-r7`，只有短检查成功、退出并清理，且两者运行代码哈希完全一致时才执行。

第二次短检查已通过：两个完整 M 历史均为 98,239，原 active 为 98,239，恢复时实际 active 为 65,535；各完成 512 输出，最终完整 CPU 历史为 98,751。两次恢复走 fresh assembly，未复用旧 resident 页；接受输出计数与 2+2×512 一致，零抢占，清理完成。证据为 `partial-terminal-gpu-20261017-r2/qualification-proof.json` 及 `reviewed-qualification.json`。正式 r7 只在该进程退出及 GPU 空闲后启动。

### 最终同输入长历史验收结果

`artifacts/deepswe-history-capacity-20261017-r7/final-reviewed.json` 已复核冻结运行代码、原生对照原始文件哈希、8 个准备输入及 8 个续接输入 token SHA、全部输出 quota、CPU 退休、完整历史、实际 GPU decode batch 和清理。r7 的 KVMem 使用上述 7 文件修复；原生复用 r3 已完成的对应阶段，推理与请求参数一致。两套系统都实际接收全部 8 请求，没有客户端 5 路门槛。

| 指标 | 原生 vLLM | KVMem＋vLLM |
| --- | ---: | ---: |
| 相同冷准备输入 token | 1,836,509 | 1,836,509 |
| 冷准备墙钟 | 631.25 s | 434.38 s |
| 冷准备公共 computed prefill／阶段墙钟 | 2909.33 token/s | 4227.86 token/s |
| 相同续接输出 token | 65,536 | 65,536 |
| 续接墙钟（含恢复、排队及重 prefill） | 482.76 s | 131.35 s |
| 续接平均输出率 | 135.75 token/s | 498.95 token/s |
| 冷准备＋续接总墙钟 | 1114.03 s | 565.80 s |
| 输出／冷准备＋续接总墙钟 | 58.83 token/s | 115.83 token/s |
| 续接公共 computed prefill token | 698,853 | 8 |
| 续接 TTFT p95（8 个请求） | 317.91 s | 4.40 s |
| 抢占次数 | 3 | 0 |
| 采样 GPU KV 使用率峰值 | 100% | 66.59% |

续接输出率提高 3.68 倍；将冷准备也计入，输出率提高 1.97 倍。冷准备输入完全一致，公共 computed prefill／准备墙钟提高约 45.3%。这些是服务阶段速率，包含对应维护成本，不能当作独占 GPU prefill／decode kernel 速度；原生 prompt 计数在输出时发布，不能依赖瞬时零增量排除未完成的 prefill。新代码没有改变主模型验证逻辑。

容量证据是物理页与 GPU 步骤的联合核对：原生实际 991 页，7 份最短完整输入仅 attention 就需 999 页，因而这组输入同时完整驻留至多 6 路，尚未计 recurrent/MTP 额外页。KVMem 则记录到 1867 个位于共同生成区间内的同一步 8 路 pure-decode 步骤；整个续接阶段共有 1871 个这样的步骤。原生的 HTTP streaming overlap 最高 7 不代表其 GPU 同时保持 7 份完整 KV。KVMem 的 8 路共同生成区间为 95.51 秒，接受输出率 576.53 token/s，它与整轮续接速率分开报告。

KVMem 最终每用户完整历史为 237,019～238,294 token，active view 为 73,723～74,774，均在 96K 容量内。完整 CPU authority 为 66,217,246,720 字节（61.67 GiB），最低主机可用内存 20.59 GiB，FP8 codec 与 100GB cap 一致，全部 M 检查点和历史长度相符，KV 磁盘读写均为零。续接只计算 8 个新增公共输入 token，没有重算历史。

7 路恢复复用 resident KV，active KV 上传为零，仍需恢复 recurrent/MTP 状态，平均恢复跨度 293 ms、p95 521 ms。此前失败的第 2 号会话原窗口 98,011，在恢复中按本轮输出预算重选为 65,531；该路装配上传 2.282 GB、跨度 2.301 秒，其中主机准备跨度 2.051 秒。这条低频路径仍有明确的 CPU 准备成本，但它没有阻止本次形成 8 路 decode。这些跨度可相互重叠，不能相加成整轮瓶颈占比。

8 路 pure-decode 步骤的 CUDA model span 均值为 48.39 ms，提交前维护 CUDA span 约 1.34 ms、主机 span 约 1.51 ms；主机提交 span 约 45.45 ms，与模型 span 可重叠，不能相加。日志的 64K context bucket 是分桶标记，不代表每个请求精确只有 64K active，实际窗口大小以上述逐请求检查点为准。

原生／KVMem 续接的 MTP 草稿接受率分别为 74.23%／65.19%，平均接受长度 3.97／3.61。收益没有来自更高的 KVMem 草稿接受率；缩短 active attention、让更多长历史会话组成实际 batch、跨轮保留 active 页，以及避免大量恢复重 prefill 共同作用，当前对照不能将总收益全部归给其中单一因素。续接 GPU 利用率分别为 99.35%／82.54%，完整准备＋续接为 99.04%／93.23%；GPU 更忙并不必然交付更多输出。

本次真实工具交互窗口的 +6.86%／+15.96% 来自修复前 qualified P1；修复后的验证是两请求 GPU 边界检查和本次完整 8 路真实历史恢复诊断，未将它冒充另一次 30 分钟工具交互测试。两类证据共同支持“active KV 减少能够扩大全历史会话的实际 GPU 并发，并在本轮工作负载提高输出率”。主窗口属于单次固定时窗、轨迹不同；诊断为 4 个原始任务各 2 个不同回合的独立盐会话、固定 ignore_eos 输出，不是 8 道不同任务或解题正确率比较。输入约 224Ki、实际输出 8Ki，256Ki 是历史上限、32Ki 是 G 上限；没有声称 8 路均填满 256Ki 历史或证明全局最大容量。

所有测量进程和 GPU 服务已退出，GPU 已释放；失败尝试与原始对照保留。最终复核统计和完整 profiler 在 r7 的 `final-reviewed.json`，主工具窗口分析在 `deepswe-capacity-20261017-r3/live-analysis.json` 及 `live-profile-reviewed.json`。
