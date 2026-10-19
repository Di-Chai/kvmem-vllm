# Graph V2 阶段性验收结果

这是进行中的结果记录。核心实现、跨轮功能验收和持续性能筛选已经完成；
首轮 NIAH 三臂共 540 题完成；MAB 遇到图配额技术失败，修复通过 CPU 回归，
新 r2 已通过工具跨轮复验，正式 MAB A 的 1,116 题已完成、C 正在运行。两套记忆测试未完成，
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
| MemoryAgentBench A/C/D | 原始 1,116/arm，共 3,348；问题间恢复冻结 P，防止输出统计泄漏 | r1 错误保留；r2 A 全部生成并释放、待评分，C 已启动 |
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

原 544K 错误路径现已实际通过：r2 A 的 group002 首题完整处理 544,132 个
输入 token，续题为 544,141，均独立核对返回 prompt token IDs 与原始 source
校验值一致。冻结 P=543,744，两题的 prefill sample、graph source/edge 计数
一致且 action-history=0；首题 31 次 pressure，续题零次。请求耗时分别为
421.15/6.74 秒，包含不同工作，不能据此推导修复的速度提升。旧 r1 的同题
HTTP500 文件保持原样；这是两项容量修复组合后的成功路径，不隔离各自贡献。
证据为 `artifacts/graph-v2-memory-capacity-20261019/full-mab-A-r2-544k-first.json`。
完整 MAB/其他策略仍待验收，不扩展该两题的结论。

最长原始样本现也通过 r2 A：757,259 个输入 token、冻结 P=755,712，返回
1,150 个输出 token 后自然停止，未达到原始 1,200 输出上限；请求耗时 405.88 秒。
完整返回的 prompt token IDs 和 P 前缀分别与原始样本校验值一致，实际发生
一次 Query、48 次 pressure，随后会话正常释放。Query 时实际图预算计费为
34,785,462 bytes，超过旧的 32 MiB 单会话上限；这说明较长样本确实需要
足够的配额，不只是修复重复元数据的无效预留。两项修复没有隔离计时。
证据为同目录 `full-mab-A-r2-757k-first.json`；这仍不是完整三臂质量结论。

另已核对实际 vLLM 服务的指标实现：`generation_time_ms` 为首个到末个输出
的 engine 时间戳差，`tokens_per_second` 则包含从 scheduled 到首 token 的
阶段。上述 757K 请求首 token 后间隔为 15.49 秒，按 (1,150−1)/15.49 为
74.18 token/s；包含 scheduled→首 token 的 389.68 秒后，服务整请求输出率
为 2.84 token/s。这是单请求共享服务的墙钟输出率，可能包含其他请求的
prefill/恢复干扰，不是 GPU 合计或隔离的纯 decode TPS；首 token 前时间也
不能全部归为模型 prefill。原生 queue 指标不涵盖所有网关准入等待。
定义源码校验值、公式及样本绑定保存在同目录
`per-request-metric-definition.json`，单 token 或零生成间隔不推导 decode 率。

r2 A 现已完成全部 1,116 个首次回答，技术错误为零；29 个历史组均释放，
1,087 个续题的 P 前缀和 prefill sample 计数保持一致，所有题的 action-history
页数为零。CPU 历史、检查点、pending download 及 active request 最终清零，
所属进程和 GPU 进程均清理完毕，C 随后自动启动。完整流程为 5,903.88 秒；
八个客户端的 CPU 准入峰值为五，实际 native batch 峰值五、prefill/decode
峰值各四，不能称为八路同时 decode。绑定证明为同目录
`full-mab-A-r2-all-first.json`；冻结 worker 逐题验证返回输入 IDs，此处另核对
全部 source/prompt/Query 绑定。完整原评分及独立全响应审计仍待完成。

原始 MAB 各题输出上限并不统一等于 G：QA 为 10/40/50，摘要为 1,200。
A 有 102 个回答达到原始题目输出上限，其中 100 个 finish_reason 为 length、
两个恰在上限自然停止；这不计为技术错误，也不代表答对。触顶分布为 MAB
longmemeval 32/300、eventqa 3/500、ruler 4/100、factconsolidation sh 8/100、
mh 54/100、摘要 1/16。保持原参数，不为消除触顶而重生成。证据为同目录
`full-mab-A-r2-output-caps.json`。

另独立检查历史 qw3 attention/query 各 1,116 个结果，逐题 source/prompt/Query
均与本轮 fixture 一致，回答绑定与冻结的历史 judge-answer manifest 一致，
输出上限也逐题相同。旧两组分别有 125/130 个触顶，全部以 length 结束；
因此这不是本轮新增输出限制。旧 attention 的 factconsolidation mh 为 89/100
触顶、MAB longmemeval 为 25/300；旧 query 为 69/100、54/300。触顶数量的
变化不等于准确率提升，也不能独立解释评分策略或量化的影响。历史结果文件
校验值和各 source 统计保存在同目录 `historical-mab-output-caps.json`。

原裁判流程已独立排队，状态与无凭据 runner 副本保存在
`artifacts/graph-v2-original-scoring-20261019/r1/`。它绑定实际推理父进程的 PID、
启动时间和命令校验值；九个冻结任务全部通过、完成 GPU 清理、父流程终止后，
才调用六个冻结的原始 memory 评分器。任何推理失败都停止评分，不重生成回答。
该最终队列先运行含原校准的独立 LME，再运行 MAB。密钥通过
隐藏终端进入进程内存，不写入 runner、manifest、日志或请求头文件。排队状态
不是评分完成，最终仍需核对全部原评分和未判定项。

用户随后要求先评分已完成部分，因此新增提前评分流程
`/data/chaidi/kvmem-vllm-experiments/graph-v2-early-scoring-20261019-r1`：先对
完整 A，再对 C 在其评分开始时所有已完成的回答建立明确截止快照，使用相同
冻结的来源校验、原始本地评分器及原 DeepSeek Pro 裁判。正在执行的请求不
进入快照，缺失/未判定项不记为答错，不发布选择性子集的类别总分。该流程
单核、低优先级运行，GPU 推理继续；原等待评分进程仅临时挂起，提前流程
退出时恢复。提前评分缓存只有完成来源和回答绑定校验后才发布给最终流程，
由最终原评分器按完全相同回答复用，避免重复调用；不重生成模型回答。
提前评分的 CPU 活动时间窗需要单独标记，不能将无后台评分的 A 与有后台
评分的 C/D 整轮耗时直接作为图传播开销对照。独立性能验收仍安排在评分
进程退出后执行。

A/C 均已完成全部 1,116 题原主指标评分，未判定项为零。前四项是 substring exact
match，LongMem 为原 LLM 裁判正确率，摘要为原 F1。旧 qw3 对照来自已冻结的
同样样本，但量化和运行时不同，不能仅据此隔离图算法或集成影响。

| 来源 | 题数 | 本轮 A | 本轮 C | 旧 qw3 attention | 旧 qw3 query |
| --- | ---: | ---: | ---: | ---: | ---: |
| EventQA | 500 | 90.0% | 89.2% | 86.0% | 89.2% |
| RULER QA2 | 100 | 55.0% | 60.0% | 54.0% | 56.0% |
| 单跳事实整合 | 100 | 54.0% | 49.0% | 48.0% | 54.0% |
| 多跳事实整合 | 100 | 2.0% | 3.0% | 1.0% | 2.0% |
| MAB 内 LongMem 子集 | 300 | 56.67% | 56.0% | 56.33% | 54.33% |
| InfBench 摘要 F1 | 16 | 21.91% | 24.28% | 19.96% | 24.36% |

三类得分为检索 67.22%、冲突消解 28.00%、长文理解 21.91%；按原协议先在
类别内平均来源，再对三类等权，A 的宏平均为 **39.04/100**。这不是简单的
1,116 题答对占比。旧 qw3 attention/query 的同口径宏平均为 36.63/39.62。
316 个外部裁判项中，138 个按输入/回答精确绑定复用历史结果，178 个新调用；
已独立核对全部 316 个 key、评分范围及原模型/thinking/预算/指标配置，并
独立重算宏平均，提前结果与发布到正式 A 评分目录的缓存逐文件校验值相同。
证明为 `artifacts/graph-v2-early-scoring-20261019/A-all-original-scores.json`。
300 道 LongMem 子题不等于尚未运行的独立 102 题 LongMemEval-S。

C 先完成 1,005 题截止快照的评分，其中一个裁判把 1,024-token 预算全部用于
思考而没有最终判断，原样保留失败并以相同预算重试成功；快照最终 1,005 题
全部已判，仍保留完整 1,116 题缺 111 题、全套宏平均为 null 的历史报告。
随后 C 全量推理通过，立即用冻结原评分器补齐全部 1,116 题。C 的宏平均为
**39.56/100**，比 A 高 **0.51 分**，子项有升有降，尚未建立稳定、普遍的
质量提升。已独立核对全部 316 个裁判项的原配置与 key、全套来源审计和宏平均，
证明为同目录 `C-all-original-scores.json`。A/C 的 316 个共同裁判项中，170 个
输入与回答绑定完全相同，这 170 个的得分全部一致；其余回答分别保留原评分。
对应审计为 `AC-full-common-external-answer-audit.json`。

后续采用按完整分组就绪即评分的原裁判队列，状态和无凭据 runner 副本位于
`/data/chaidi/kvmem-vllm-experiments/graph-v2-progressive-scoring-20261019-r1`。
依次处理完整 C/D 与独立 LME A/C/D，等待时验证主推理进程仍为绑定的同一
进程；每组先确认 passed 与 GPU 清理，再运行冻结原评分器。原最终评分进程
在该流程拥有评分期间临时挂起，完成或失败后恢复，最后复用同一答案绑定缓存
做全矩阵检查。评分保持单核低优先级，未变更模型、评分协议或预算，也不再
让已经完成的分组等待全部九项推理结束。C 现已完整评分，D 正在真实推理。

C 的全部首次回答与原始来源/Query 绑定也已核对：29 组历史、1,087 次 P
复用，全 1,116 题实际为 Query multi/.2/两图等权；743 次旧 attention
pressure，所有续题零 pressure、P/prefill 计数不变、零输出历史。结束后
CPU 历史、检查点、pending、active 均清零，所属进程和 GPU 进程清理完毕。
原生 batch 峰值五，prefill/decode 各四，不能称为八路同时 decode。整轮
5,772.73 秒包含并行的 CPU 提前评分，不能据此作为独立吞吐改进。证明为
`artifacts/graph-v2-memory-capacity-20261019/full-mab-C-r2-all-first.json`。

C 的首个正式跨题恢复现已核对：group007 的前两题均实际运行 V2 multi
Query 传播（alpha=.2、两类图等权），分别改变 57/58 个入选块。首题 20 次
pressure，续题零次；P=367,616、prefill sample=11,453、图 source/edge 计数
保持一致，action-history 为零，pressure 仍为本臂声明的旧 attention 策略。
原始 token 输入没有编造消息边界，locality 使用 unknown fallback。首次回答、
事件、请求与返回校验值绑定在同目录 `full-mab-C-r2-first-continuation.json`。
这是当前冻结运行时实际启用图评分及 P 恢复的证据，不扩展为完整 C 组、
回答质量或吞吐收益。

对当时正在运行的 MAB C 做了独立只读检查：固定 180 秒、175 个 NVML 样本的
平均 GPU 利用率为 43.36%，低于 20% 的样本占 32.57%。同窗口 486 个原生
计算步中，batch=1/2/3/4 分别为 383/83/18/2，即 78.8% 的步只有一个请求，
没有八路计算步。这里按计算步计数，不按各步耗时加权，不能解释为 78.8%
墙钟时间。八个客户端也不等于八路就绪：CPU 历史准入峰值为五，准入预算
82,820,130,816 字节，是 100 GB 总额扣除本轮 16 GiB 安全余量的结果。

另采集的 96 对相邻同组题目，HTTP 响应后到下一题尝试的平均间隔为 4.91 秒，
平均只生成 12.45 个 token；其中响应到结果落盘约 2.55 秒、落盘到下一题约
2.36 秒。响应结束由尝试开始墙钟时间加 HTTP 单调耗时近似，间隔包含安全
退场、校验、汇总和下一题输入准备，不能全部归为某个 Python 函数或 GPU
空闲。61 个额外调度快照中，38 个有恢复/connector pending、54 个有
publication blocked，所有快照仍有 2～5 个请求；状态可以重叠，也有快照
延迟，因此不能将这些次数换算成互斥阶段耗时或声称请求队列一直为空。
原始证据位于 `artifacts/graph-v2-idle-reading-20261019/r1/`。

进一步保存完整日志前缀并核对 1,120 个配对装配端点，证据及复现脚本位于
同目录 `host-stages-r2/`。770 个带 private Query 的重选命令，排队到就绪
平均 2.025 秒：排队到开始 0.836 秒、开始后至装配计时前 0.770 秒、装配
准备和提交 0.245 秒、提交到就绪 0.174 秒。第二段包含历史发布、必留预算
检查和选页，不能全部归为 MeanK 或图传播；private Query forward 的耗时
也包括模型执行，不能当作纯评分耗时。772 次检查点捕获的主机总耗时中位数
2.58 毫秒、p95 23.34 毫秒，单次模型快照约 158 MB；这些日志未隔离 D2H
完整完成时间，也不覆盖同步恢复自身的阶段耗时。各请求区间重叠，不能将它们
相加当作 GPU idle。以上采样早于提前评分，不含评分流程的 CPU 活动。

代码检查确认测试客户端每题持共享锁重读所有旧结果，并重复扫描完整 capture
日志。后续客户端改用一次初始化、增量更新的精简进度记录；在 HTTP 提交前
记录日志边界，仅解析之后的新完整记录，并在原生图审计中复用同一份读取。
日志轮换、截断或边界前缀改变均显式报错，不退回全量扫描；请求匹配、完整
结果、退场和释放校验保留。冻结运行时依赖清单包含新增 helper，客户端
manifest 记录新版本，不能用它悄悄续跑旧尝试。21 项针对性 CPU 测试通过，
用时 0.26 秒，其中九项新增测试覆盖与原磁盘汇总一致、无旧结果重读、
并行请求读取范围、日志异常和原审计匹配，其余检查已有客户端和配置兼容性；
日志与源码绑定在同目录 `client-io-r1/`。
当前正式 r2 仍使用原冻结客户端，未中途替换；该改动尚无真实 GPU 对照收益。

为补齐当前冻结版本的八路稳态 decode 证据，新增独立 client
`scripts/qualify_graph_v2_decode.py`。它复用已验证的流式 token-ID/usage 计数，
使用原始 256K NIAH 首题的全部 262,101 个输入 token、原 Query 坐标及原 D
服务配置，八个独立 session/cache salt，各固定输出 16,384 token。诊断请求
使用 temperature=0/ignore_eos，不计入原始质量矩阵，也不是新的性能候选对照。

验收要求至少 30 秒共同窗口、两端各剔除至少 64 token、至少 90% native
确认覆盖，以及对应 native 提交记录全部为八路 decode、零 prefill。窗口
保留全部停顿，不把 MTP proposal 当接受输出；native 快照的绝对单调时间与
流式相对时间绑定，CPU 提交边界不冒充精确 GPU kernel 边界。最终核对完整
接受历史并要求所有会话释放。八个 client label 不作为身份认证验收。

三个新增 CPU 测试函数已直接运行通过（0.112 秒，无 GPU），覆盖重复／缺失
身份、时间原点冲突、输入／预算／Query 和窗口门槛。完整 pytest 和 GPU
诊断仍待运行；排队驱动在正式推理及原裁判流程终止后，才运行 CPU 回归、
复制同一冻结推理代码并申请空闲 GPU。不会中断外部 GPU 进程，也不修改
现有 r2 输入、配置、运行时或答案。诊断与其初次失败记录另存新目录。

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
