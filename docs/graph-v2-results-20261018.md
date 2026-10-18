# Graph V2 阶段性验收结果

这是进行中的结果记录。核心实现、跨轮功能验收和持续性能筛选已经完成；完整
A/C/D 质量矩阵正在运行，不能据此宣称 Graph V2 提高了准确率或稳定吞吐。

## 已实现与已验证

- Mean-K 使用 qw3 原生评分域；softmax mask 与传播 mask 分离，额外 mandatory
  参与正确的 softmax 分母。graph off、collect、scoring 共用原始分数。
- V2 仅使用 alpha=.2，一跳传播；attention 有效出边过滤后按来源归一化，
  multi 两类算子固定等权。保留 V1 配置解释和 c14 的 native 密集聚合。
- 真实 Protocol 消息、工具结果及实际接受的 output IDs 生成 graph-only 边界；
  不改变输入、Query/replay 或 mandatory。原始 token 输入保持 unknown fallback。
- Query 与 pressure 共用传播函数。Pressure 精确读取旧累计 output/prefill
  attention seed，候选限于当前或 checkpoint 工作窗口，再进入原生 picker。
- 同步、异步和 checkpoint 恢复先安装正确的状态与边界，再评分和发布；
  完整验证的 M 前缀可在 P 重放时保留统计，真实编辑使用分支隔离。

冻结的同步、异步真实模型验收分别完成两 owner、三轮请求，包含实际工具调用、
工具结果、新 Query 和 Query 编辑。每次恢复核对 17 个 attention 层（含 MTP）
及 96 个 recurrent tensor；两组均通过并释放所属资源。对应冻结运行时的
针对性 CPU 回归为 695 项通过，后续入口/配置修复另有独立回归记录，不能简单相加。

证据目录：

- `artifacts/graph-v2-sessions-20261018/{sync-r1,async-r1}`
- `artifacts/graph-v2-locality-20261018/cpu-producer.json`
- 详细门槛和限制见 `docs/graph-v2-implementation-20261018.md`。

## 冷负载性能筛选

共同配置为本地 27B NVFP4 模型、FP8 attention KV、原生 MTP(k=4)、编译及
CUDA Graph、八客户端、K=36,864/G=16,384、chunk=1,024、batch=16,424。
使用原始八个 256K NIAH 输入；每个变体启动独立服务，仅修改一个开关。
CPU 历史总预算为 100 GB，KV 不使用 SSD，不截断或重算历史。

下表为完整冷 cohort 的耗时及 useful input/整轮时间的描述性变化，**不是纯
decode 吞吐**，也不是预热后的 300 秒持续性能门槛。

| 候选 | 基线/候选耗时 | 描述性输入率变化 | 观察 |
| --- | ---: | ---: | --- |
| raw-K 总预算 4 GiB | 387.57 / 396.02 s | -2.13% | 装配上传 71.68 → 48.80 GB |
| CPU 直接封存 | 385.08 / 385.89 s | -0.21% | 二次记录复制消除，但打包/哈希等总主机工作未下降 |
| 异步 attention 元数据 | 387.17 / 377.60 s | +2.53% | 实际触发 2,086 次上传、22.80 MB |
| raw-K 完整窗口预算 10 GiB | 383.47 / 381.52 s | +0.51% | 装配上传 71.58 → 40.38 GB；命中 1,797,888 token |

各对照存在选页/接受输出差异，实际 native batch 峰值均为七；不能把八个
admitted owner 当作 batch 八。所有冷筛选结论均为 inconclusive，未建立稳定收益。
上述候选继续默认关闭；异步元数据进入持续负载筛选，不代表已满足采用门槛。

直接封存对照中，模型 step 的累计 CUDA event 区间约 357 秒，两条路径接近；
整轮约 385 秒。主机封存工作与模型执行重叠，减少复制没有直接转化为整轮收益。
这些 event 区间和重叠主机耗时都不能当作独立 kernel 活跃时间或 GPU idle。

独立证据：`artifacts/graph-v2-performance-20261018/` 下的
`raw-k-r1`、`seal-r1`、`metadata-r1`、`raw-window-r1`。每组保留原始首次结果、
测量区间、NVML、实际 batch、传输量、资源释放和比较条件。

持续 256K 元数据对照的 baseline 已完成：八次预热释放后固定测量 300 秒，
完成的公开 prefill 计数增量为 1,596,896，即 5,322.99 token/s；边界不确定性
给出的窗口内完成量范围为 1,581,632–1,610,112。该计数包含实际重复 prefill，
不是 unique input。452 个计时 step 均为 prefill，实际 batch 峰值六；模型 event
区间对应的纯 prefill 速率为 5,687.60 token/s。

这一窗口接受输出为零、完成请求为零，反映了长 prefill 阶段，不能用于估计纯
decode 性能。窗口外最终 16 个请求全部完成并释放；NVML 窗内平均利用率 88.62%。
候选臂同样完成：公开 prefill 为 5,449.49 token/s，边界完成量范围
1,619,456–1,644,160；模型 event prefill 为 5,692.30 token/s，native 峰值六，
窗内接受输出仍为零。NVML 平均利用率为 91.71%。原始公开 prefill 率比为 1.0238，
但 common 输出/选页不一致、未观测到 native C8，且现有终端元数据累计量不能
单独证明 T0..T1 的路径活动，比较为 inconclusive。两臂都通过完整性和清理检查。
这组结果没有建立持续 decode 或稳定性能改善。证据位于
`artifacts/graph-v2-stream-20261018/metadata-r2/p1-baseline/qualification/`。

## 图开销与策略消融

- A/C/D 八题功能 smoke 全部完成并释放：三组均为 7/8 exact、96.875% 数值覆盖率，
  耗时 378.91 / 380.17 / 385.04 秒。这一小样本没有建立质量改善。
- 独立 off/collect r2 已完成：374.40 / 380.76 秒，原始耗时比为 1.017；
  两组均为 7/8 exact、96.875% 覆盖率，并释放所属资源。选页、输出和采样数量
  存在差异，native 峰值七，比较仍为 inconclusive，不能把 6.36 秒全归因于收集。
  Collect 的累计图准备/提交主机时间为 7.08 / 0.19 秒，可能与 GPU 执行重叠。
- 固定真实输入的 V1 attention / V2 attention / V2 multi unknown /
  V2 multi known 四臂消融已通过：8 次真实请求的完整选页均与原生重放一致，
  四个服务均清理完毕。两个 canonical 快照各覆盖八对有正 seed 的真实跨边界
  邻居。归一化替换了 274/277 个入选块；attention-only 改为 multi unknown
  替换 125/130 个块；增加真实边界改变了分数，但这两个快照的入选集合未改变。
  这是同快照受控反事实，不能把分数或选页变化当作回答准确率。

## 尚待完成的正式验收

| 项目 | 原始样本与规模 | 当前状态 |
| --- | --- | --- |
| 持续 256K 负载 | 八槽持续补位，至少八次完整 warmup，固定 300 s 后独立 drain | 配对完成，比较 inconclusive，未进入 decode |
| 自然工具调用轨迹 | 原始八路 agent fixture，保留自己的完整输出和工具等待 | r4 两臂完整通过；比较 inconclusive，未采用候选 |
| NIAH A/C/D | 64K/128K/256K 各 60 题，180/arm，共 540 | A180 已完成并释放；C 正在运行，D 排队 |
| MemoryAgentBench A/C/D | 原始 1,116/arm，共 3,348；问题间恢复冻结 P，防止输出统计泄漏 | 已冻结并校验，排在 NIAH 后运行 |
| 独立 LongMemEval-S A/C/D | 原始 102/arm，共 306 | 已冻结并校验，排在 MAB 后运行 |

A 组的原始 180 题已全部完成，通过逐题协议及最终释放审计；再次以冻结的
原评分函数复算全部答案，并核对原始输入哈希。此前发布的 64K/128K 首次
答案与选页/释放文件校验值保持不变，零输出触顶。

| A 组长度 | Exact | 数值覆盖率 | 所有 needle 区间完整入选 | 请求耗时中位数 |
| --- | ---: | ---: | ---: | ---: |
| 64K | 60/60 | 100% | 60/60 | 85.39 s |
| 128K | 60/60 | 100% | 40/60 | 183.58 s |
| 256K | 54/60 | 97.5% | 33/60 | 376.96 s |

请求时间包含服务内排队、完整预填充、Query、生成及网络，不含离线队列
待准入和后续释放时间。区间完整覆盖与回答正确率是两项指标。完整 A 组
证据为 `artifacts/graph-v2-suite-preflight-20261018/full-niah-A-all-first.json`，
绑定了 180 题首次答案、原评分、选页和释放证据。C/D 尚未完成，没有策略
改善结论。相邻长度在持续补位时重叠，不从单个长度完成时间推导纯吞吐。

A 组完整测量耗时 4,945.79 秒（约 82.4 分钟），原始输入率为 5,563.83
token/s；5,085 个 pure-prefill step 的模型 GPU-event 区间速率为 6,033.41
token/s，包含 replay 工作，并非唯一输入率或独立 kernel 活跃时间。整轮
39,331 个输出对应 7.95 token/s；负载每题平均约 219 个输出、约 153K 输入，
这个整轮输出率不能当成 decode 速度。实际执行峰值八路（七个 step），
decode 峰值七路，363 个 pure-decode step、3,919 个 mixed step。最后确认
所属进程和 GPU 进程均已清理；随后 C 组自动启动。

C 组首个完整长度 64K 已完成：60/60 exact、100% 数值覆盖率，60/60 题
完整覆盖全部 needle 区间，零输出触顶。与 A 组相同的 60 个输入、seed 和
全部非 graph_mode 配置逐项核对一致；原 A 组首次答案校验值未改变。C 的
60 题均发生图传播选页变化，说明实际启用了路径，但该长度没有观察到回答
质量增益。请求耗时中位数 83.37 秒，不能从单次长度的中位数差异推断稳定
性能收益。C128K 正在继续，D 尚未开始。证据为
`artifacts/graph-v2-suite-preflight-20261018/full-niah-C-64k-first.json`。

对已经完成的自然 agent r4 做了额外的只读分阶段统计：基线单请求首 token
后输出流速度 p50/p95 为 40.89/87.00 token/s，按各请求生成时长加权为
33.88 token/s；异步元数据候选为 40.99/95.63、31.05 token/s。它们包含其他
请求的 prefill/恢复干扰，是每用户的服务输出率，不是 GPU 合计或隔离的纯
decode TPS。统计对象是在 300 秒窗口发起的完整请求，其生成可能延续到 drain；
不能与仅窗口内接受输出 176.04/163.63 token/s 混用分母。所有实际 token-ID
计数和 usage 已核对，保留生成期间的全部间隔。窗口中只有三个 pure-prefill
step，不能据此估计普遍的短 prefill 能力；大部分 prefill 与 decode 混合执行。
原采样未保存证明纯 decode 墙钟区间所需的逐 owner/reservation 快照，所以
没有从 MTP proposal 或请求数推断纯 GPU decode TPS。证据为
`artifacts/graph-v2-performance-20261018/agent-phase-rate-reading.json`。

正式矩阵合计 4,194 个回答。三臂保持相同执行优化、模型、输入、采样、K/G 和
batch 参数；A 为对齐 Mean-K Query + 旧 attention pressure，C 增加 V2 Query
传播，D 再增加 graph pressure。完整首次答案及技术失败均保留。

正式目录为 `/data/chaidi/kvmem-vllm-experiments/graph-v2-full-20261018-r1`。
2026-10-19 01:14（Asia/Shanghai）从干净提交 `5686617` 冻结，完整输入、评分器、
模型及九个运行时通过校验后启动。各组依次使用同一张 GPU，每组内部八客户端
持续补位；不是三种策略共享 GPU 同时压测。根目录 `status.json`、各任务
`controller.log`、`qualification/progress.json` 和资源采样保留实际进度；任务
尚未完成不代表失败，运行中数据也不是正式完整评分。

MAB 原始长度为 267,553–757,259 token，需要 1M 逻辑上下文；最长八组完整历史
超过 CPU 预算，因此实际驻留组数会低于八，完整历史排队。独立 LME 为
102,963–112,819 token，八组可容纳。LME 新三臂统一使用 G=16,384 输出上限，
历史参考使用 32,768；必须报告 cap hit，历史分数不能作为严格参数相同的对照。
原始 qw3 `attention_multilayer_prefix_reuse_20261012_r2/plan.json` 也明确使用
K=36,864/G=16,384/thinking=4,096；保留此上下文管理几何，避免为扩大输出上限
同时改变 prefill 压缩策略。三臂的请求上限和全部公共参数一致。
补充核对了历史 attention/query 各 102 个实际结果：最长输出分别为 4,234 /
4,745 token，均为未达到 length 上限，均低于 16K。这个观察降低了当前 cap
触顶的预期，但不能证明配置上限相同，也不能替代本轮逐题 cap-hit 审计。
历史记录校验值保存在
`artifacts/graph-v2-suite-preflight-20261018/historical-lme-output-caps.json`。

最终报告还需要给出原始评分协议下的质量、prefill/接受输出吞吐、延迟、实际
native batch、边界覆盖及资源限制。未评分、未观测和不满足比较条件的结果
保持明确标记，不补写为零或成功。

工具轨迹 r1 在最终释放审计中超时：实际 CPU 历史、检查点、额度均为零，GPU
仅保留原生 null block，进程清理确认 GPU 为空。检查器错误地要求已关闭的
active 驻留模块必须产生状态字段；离线复核确认这是验收误判。另修正了把
512K 仓库容量误写为 256K 逻辑上限的检查。113 项相关回归通过。r1 保持失败，
未运行的 r2 被保留并标记替代，新 r3 重新执行完整配对，不把旧结果当作通过。

r3 已通过运行、排空及释放检查，但随后因检查器要求每个初始 epoch 必须存在
Mean-K Query 选择事件而失败。原始 early Query + auto 协议在前缀可全部放入 K
时无需该选择；现场只有 graph pressure。新检查要求实际 pressure 的 hard
mandatory 和选页完整保护 Query，并明确记录 Query 传播未触发。
114 项相关回归通过。保留数据经完整离线审计后可用于诊断：窗口接受输出率
150.11 token/s、native batch 峰值八，但原 r3 状态仍为失败，不作通过的配对结果。
新 r4 重新运行。该工具轨迹性能范围不包含每轮 Mean-K Query 图传播开销。

r4 基线已通过完整运行、历史、实际配置和释放审计。固定 300 秒窗口接受
52,812 个输出 token，即 176.04 token/s；2,804 个完成的原生 step 中，2,005
个为纯 decode，其中 44 个实测 batch 八。请求 p95 为 66.82 秒，观测 TTFT
p95 为 5.94 秒，另保留两条窗口内未见首 token 的右删失记录。这是包含工具
等待、恢复与排队的接受输出率，不是纯 decode TPS。

r4 候选也已通过完整审计、排空和释放：接受输出 49,089 token，即
163.63 token/s；原生 batch 峰值八，1,907 个纯 decode step 中有九个 batch 八。
请求 p95 为 73.86 秒，观测 TTFT p95 为 6.26 秒，删失记录保留。异步元数据
路径实际触发，未观察到上传失败。原始输出率比 0.9295，126 个共有逻辑轮次
的输出与选页并不完全一致，因此比较为 inconclusive，不能把约 7% 的差值
当作该开关的独立因果损失，也没有采用它的性能依据。两臂最终均确认 GPU 无
所属残留进程。末次元数据累计快照尚有一条 lease，不能单独替代最终清理证明。

正式质量矩阵固定使用 r4 基线执行设置：raw-K cache=0、直接封存关闭、异步
attention 元数据关闭，保留已验证的 native 密集聚合、编译及 MTP4、FP8 KV。
本轮没有候选达到稳定提升至少 5% 的采用门槛，故不继续以相同自然轨迹漂移
的筛选结果扩大调参矩阵。完整 A/C/D 使用相同执行设置，仅改变已声明评分策略。
