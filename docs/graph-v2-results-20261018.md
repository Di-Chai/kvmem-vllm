# Graph V2 阶段性验收结果

这是进行中的结果记录。核心实现、跨轮功能验收和持续性能筛选已经完成；
首轮 NIAH 三臂共 540 题完成；MAB 遇到图配额技术失败，修复通过 CPU 回归，
新 r2 已通过工具跨轮复验，正式 MAB A 正在运行。两套记忆测试未完成，
尚无稳定的质量或吞吐提升结论。

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

## 正式验收进度

| 项目 | 原始样本与规模 | 当前状态 |
| --- | --- | --- |
| 持续 256K 负载 | 八槽持续补位，至少八次完整 warmup，固定 300 s 后独立 drain | 配对完成，比较 inconclusive，未进入 decode |
| 自然工具调用轨迹 | 原始八路 agent fixture，保留自己的完整输出和工具等待 | r4 两臂完整通过；比较 inconclusive，未采用候选 |
| NIAH A/C/D | 64K/128K/256K 各 60 题，180/arm，共 540 | r1 三组全部通过；r2 排在 MAB 之后重新运行 |
| MemoryAgentBench A/C/D | 原始 1,116/arm，共 3,348；问题间恢复冻结 P，防止输出统计泄漏 | r1 A 保留 69 个完成回答及 5 条技术错误；r2 A 正在运行 |
| 独立 LongMemEval-S A/C/D | 原始 102/arm，共 306 | 已冻结并校验，排在 MAB 后运行 |

三组原始 180 题均通过逐题协议、原评分复算和最终释放审计。A/C 仅改变
`graph_mode`，C/D 仅改变 `pressure_policy`；输入、seed 与其余服务参数相同。
540 题均未输出触顶，已发布的首次答案、选页和释放文件校验值保持不变。

| 策略 | 64K Exact | 128K Exact | 256K Exact | 256K 数值覆盖率 | 256K 所有 needle 区间完整入选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：Mean-K Query＋旧 attention pressure | 60/60 | 60/60 | 54/60 | 97.50% | 33/60 |
| C：V2 Query 图传播＋旧 attention pressure | 60/60 | 60/60 | 55/60 | 97.92% | 60/60 |
| D：V2 Query＋graph pressure | 60/60 | 60/60 | 53/60 | 97.08% | 60/60 |

128K 的完整 needle 区间保留也从 A 的 40/60 增至 C/D 的 60/60，64K 三组
均为 60/60。区间保留与回答正确率分别统计：256K A→C 修正两题、新增一题
错误，净增一题；C→D 修正两题、新增四题错误，净减两题。A→D 净减一题。
本次没有观察到 pressure 图传播的额外回答收益，不作统计显著性结论。

C/D 各 180 题均实际发生 Query 图传播选页变化；D 的 64K/128K/256K 分别
有 60/300/780 次 pressure 图事件，全部非恒等并改变选页，共 1,140 次，
均使用旧累计 prefill attention seed。冷 NIAH 不覆盖跨轮 output seed；
下文另有真实 agent output seed 的功能证据。

对首次答案的只读错误分析显示，A/C/D 的 6/5/7 道错题均在 256K，均漏答
一个数字，没有额外数字、未闭合 thinking 或输出触顶。C/D 错题已完整选中
needle 区间，因此不能仅凭区间覆盖断言生成质量已经解决，也不能据此确定
漏答的具体原因。该分析在生成完成后进行，未用于调参。

| 策略 | 完整 180 题耗时 | 原始输入／整轮时间 | pure-prefill GPU-event 速率 | 输出／整轮时间 |
| --- | ---: | ---: | ---: | ---: |
| A | 4,945.79 s | 5,563.83 token/s | 6,033.41 token/s | 7.95 token/s |
| C | 4,978.41 s | 5,527.37 token/s | 6,004.64 token/s | 7.90 token/s |
| D | 5,031.07 s | 5,469.51 token/s | 6,076.01 token/s | 7.76 token/s |

pure-prefill 速率以对应模型 GPU-event 区间为分母，包含 replay 工作，并非
唯一输入率或独立 kernel 活跃时间。负载每题约 153K 输入、约 219 个输出，
整轮输出率不能当成 decode 速度。C 相比 A 耗时多约 0.66%，D 相比 C 多约
1.06%；单次对照没有建立稳定性能改善或回退。三组实际执行峰值均为八路，
prefill/decode 各自峰值均为七路，不能宣称这组测试是持续八路纯 decode。

64K/128K/256K 请求耗时中位数：A 为 85.39/183.58/376.96 秒，C 为
83.37/182.51/380.49 秒，D 为 82.36/181.98/385.36 秒。请求时间包含服务内
排队、完整预填充、Query、生成及网络，不含离线待准入和响应后释放。相邻
长度在持续补位时重叠，不从单个长度完成时间推导纯吞吐。

完整首次证据为 `artifacts/graph-v2-suite-preflight-20261018/` 下的
`full-niah-{A,C,D}-all-first.json`；错误分析为 `full-niah-ACD-error-reading.json`。
所有早先的 64K/128K 首次证明继续保留。D 清理后，MAB A 自动启动，随后
因图配额技术错误停止，详见下文。两套记忆测试与原裁判评分仍待完成，完整
4,194 回答矩阵尚未完成。

MAB r1 在 69 个完成回答后发生 EngineCore `BufferError`，五个在途请求随引擎
退出而报错。栈为 `install_graph_metadata → update_message_spans → prepare →
GraphBudget.reserve`；GPU 未 OOM，最终所属进程和 GPU 进程均清理。失败根目录
和全部首次回答保持原样，证据为
`artifacts/graph-v2-suite-preflight-20261018/full-mab-A-r1-failure.json`。
先前两个历史组的首题／续题独立核对了原始 prompt token IDs、冻结 P、prefill
采样计数、图 source/edge 计数及 action-history=0，续题没有再次 pressure。
四题局部证据 `full-mab-A-first-p-reuse.json` 继续保留；它不代表完整 MAB 通过
或全部状态逐字节相等，也不用于质量结论。

修复分两部分：相同元数据先在小额、受配额保护的暂存区完成完整校验；没有变化
时返回原有 no-op 事务，不再预留两份完整图。真正的边界变化、prefix 冲突和
pending 事务约束继续执行。MAB 的完整 1M 仓库另外按每逻辑块 8 KiB 设置保守
图配额，覆盖 live/P/scoring/staging 的多份记账：每会话 256 MiB、全局 2 GiB，
三臂相同。这是上限，不是实际分配；生产默认和 NIAH 图配额不变，CPU KV 历史
预算仍为 100 GB。新增 preflight 会拒绝旧的不足配额，耗尽错误也记录各项用量。
没有改变 token、K/G、采样、图参数或增加重算。

相关 CPU 回归 352 项通过并正常退出，含 1M token 图、P/scoring/preserved
快照同时保留、追加和完整释放，以及无变化更新在紧配额下保持状态不变。
日志为 `artifacts/graph-v2-memory-capacity-20261019/cpu-regression.log`。
真实模型容量验收尚未完成。由于运行时已修改，需要新的独立冻结尝试；旧 r1
结果保留，不能与新结果拼接成同运行时的完整 4,194 回答矩阵。

新根目录 `/data/chaidi/kvmem-vllm-experiments/graph-v2-full-20261019-r2` 已从
干净提交 `fbfee4b` 冻结九个相同运行时，输入／评分器／模型校验通过。与 r1
逐项比较，原始 rows、模型及全部非 MAB 图配额配置相同（只有冻结 source 路径
随目录改变）；运行时的两项容量修复单独版本化。绑定证明位于
`artifacts/graph-v2-memory-capacity-20261019/full-r2-controls.json`。
新流程先运行同步、异步真实工具跨轮复验，再执行 MAB A/C/D，最后是 NIAH
与独立 LME 三组；每组独立 GPU 服务，出现技术失败即保留并停止，不自动重试。
工作流状态与日志位于同目录的 `workflow-r1/`。
这项容量修复没有新的吞吐提升结论，完整正式生成及原裁判评分仍待完成。

修复后的同步／异步工具跨轮复验现均通过并释放所属资源：各两 owner、三轮，
各六次 Query 与十次 pressure，以及四次真实 M/P 恢复（M/P 各两次）。全部
恢复对 17 个 attention 层（含一个 MTP 层）及 96 个 recurrent tensor 逐字节
一致。两套真实执行的全部核心源码及 native 扩展校验值均与 r2 正式运行时
相同，证据为 `artifacts/graph-v2-memory-capacity-20261019/gpu-structured-requalification.json`。
这是结构化 producer／工具结果／Query 编辑／恢复和释放的功能证据，不是
八路执行、完整 MAB 容量、回答质量或吞吐证明。随后 MAB A 已自动启动；
完整长历史容量和质量仍须由正式运行验收。

r2 A 的首个完整 MAB 历史组（60 题、P=367,616 token）已完成并释放。逐题
核对原始 source/Query 绑定，59 次续题均没有重复 pressure，action-history
页数为零，P 的 prefill sample、graph source/edge 计数保持一致；绑定了全部
首次回答、事件和释放文件校验值。释放时服务仍有两条其他请求，发布释放证明
后才归还 CPU 额度，约 19 ms 后下一组获得额度。这是 CPU 准入补位时间，
不能当作 GPU 恢复延迟、八路执行或吞吐收益。证据为同目录的
`full-mab-A-r2-first-group.json`。实际 CPU 准入顺序会受并发竞争影响；不能
用“完成题数超过 r1 的 69”代替原 544K 错误路径或 757K 容量验收。

在正式生成期间独立执行了已授权的轻量裁判预检：冻结的 Flash 原始 12 题
校准全部通过，原 Pro LongMem QA 构造器／thinking=enabled／1024 输出预算
的两个固定校准例也通过。实际返回模型为 `deepseek-flash`、`deepseek-v4-pro`。
记录位于 `artifacts/graph-v2-judge-preflight-20261019/r1/results.json`，保存
原始无凭据请求体、返回及校验值。密钥仅通过隐藏终端进入进程内存；请求头未
保存。这不是正式回答评分，也没有覆盖摘要 JSON 裁判；完整原评分流程仍待完成。

冷 NIAH 不覆盖跨轮 output seed。为此另核对了自然 agent r4 完整生命周期
的图日志：基线含 129 次 prefill seed、117 次 output seed；异步元数据候选
含 141/124 次，两类事件均实际执行非恒等传播并改变选页。这是现有真模型
跨轮路径的功能证据，含 warmup、测量、drain 及恢复选择，不能除以 300 秒
当作窗口速率，也不是每个 HTTP 请求的选择次数。绑定日志/配置校验值的证据：
`artifacts/graph-v2-suite-preflight-20261018/real-output-prefill-pressure-seeds.json`。

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
