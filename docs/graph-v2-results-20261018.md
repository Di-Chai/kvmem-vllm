# Graph V2 阶段性验收结果

这是进行中的结果记录。核心实现、跨轮功能验收和持续性能筛选已经完成；
首轮 NIAH 三臂共 540 题完成；MAB 遇到图配额技术失败，修复通过 CPU 回归，
新 r2 已通过工具跨轮复验，正式 MAB A/C/D 均已完成 1,116 题生成与原评分。
r2 的独立 LongMemEval-S A/C/D 已完成生成与评分（86/102、92/102、89/102），r2 NIAH A 已启动，完整矩阵未完成，
尚无稳定的质量或吞吐提升结论。

两套原冻结记忆 cohort 的三臂共 3,654 个正式回答，现均完成原裁判评分：

| 已完成记忆集 | 当前 A | 当前 C | 当前 D | 历史 qw3 Query | 历史 qw3 attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| MemoryAgentBench，1,116 题，三类宏平均／100 | 39.044 | 39.559 | 39.490 | 39.624 | 36.635 |
| 独立 LongMemEval-S，102 题，准确率 | 84.31% | 90.20% | 87.25% | 87.25% | 88.24% |

MAB 宏平均并非 1,116 题的简单正确率，16 道摘要占三类宏平均的三分之一。
当前三臂为相同 NVFP4／FP8 KV／MTP4 运行版本；历史 qw3 使用 Q8_0、DFlash、
串行执行，attention 配额及独立 LME 输出上限也有已披露差异，只作参考。
C 相对 A 在这轮两套记忆集均较高，D 相对 C 未显示额外收益；仍需完成
同一 r2 的 NIAH 和独立功能／decode 验收，不能将旧 r1 NIAH 拼成同版本全矩阵。

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

2026-10-19 又针对正式 r2 冻结运行时执行了核心 CPU 回归：249 项通过，
7.45 s，实际导入路径确认来自 r2，测试／辅助模块副本与运行时代码校验值
前后一致。覆盖实际 native Mean-K 排名反转、mask、V2 数学 oracle、pressure
候选与恢复、graph-only spans、生命周期、配置和离线重放。记录位于
`artifacts/graph-v2-completion-20261019/frozen-core-cpu-r2/`。r1 独立测试副本
因未复制相对路径诊断脚本而在收集阶段失败，未执行测试；失败日志保留，
补齐副本后才得到上述 r2 结果。249 与旧 695／352 项有重叠，不相加。

只读复核同时逐文件验证了容量修复后同步／异步模型验收的全部 119 个核心
文件与正式 r2 一致；逐请求重新解析六组计划，graph-only 字段未改变 Query、
replay、mandatory 或 explicit_mandatory。两种模式各有 6 次 Query、10 次
pressure、4 次 M/P 字节一致恢复；真实后缀编辑保留已接受的 assistant token
前缀。所属资源均释放，原首次响应和证明校验值未变。证据为
`artifacts/graph-v2-completion-20261019/structured-core-audit-r1.json`。
这些是两 owner 的跨轮功能证据，不能替代尚待运行的持续 native C8 decode。

### 同版本累计 output attention 的长工具返回验收

覆盖复核发现，独立记忆题和短工具续接主要证明 prefill-seeded pressure；
既有真实 output-seed 功能证据来自容量修复前的自然 agent r4。为补齐当前
冻结核心上的执行证据，结构化验收客户端新增独立 v2 fixture：两 owner，
三轮，先生成真实工具调用，再追加至少 49,152 个完整工具返回 token，最后
编辑 Query。K/G 仍为 36,864/16,384，输出仍自然 EOS、上限 257；真实接受
的 assistant IDs、工具 ID、原 Query 文本和 graph-only 边界必须保留。
旧 v1 fixture 的短工具负载和全部正式 A/C/D 冻结文件均不变。

新 gate 要求每个 owner 在工具续接时，实际出现使用累计 output 分数的
非 identity、改变选页的 graph pressure；另保留原有 P/M、MTP、recurrent
字节恢复与释放检查。客户端／controller CPU 回归为 45 项通过（0.45 s），
并非新增 GPU 通过。真实 renderer/tokenizer CPU 预检已用
旧验收中真实返回的调用验证完整编码：两路初始均 131,063 token，新后缀
均 49,171 token，完整续接 180,313 token；接受前缀、Query token 内容、
Query/replay/mandatory 策略和每路八个精确 locality 区间均验证通过。
这些旧输出只供 CPU 编码预检，GPU 测试必须生成自己的输出和工具 ID。

证据目录为 `artifacts/graph-v2-output-pressure-20261019/` 的
`cpu-regression-r2.log` 与 `cpu-long-tools-r1/`。同步／异步 GPU 两项已冻结在
`qualification-r1/`，全部 119 个服务核心文件与正式 r2 一致，仅两个独立
客户端／controller helper 更新为 `e88d531`；启动配置预检通过。为容纳多个
长工具 P/M 元数据快照，单 owner 图上限设为 128 MiB、两 owner 总上限
256 MiB；这是单独功能任务的记账上限，K/G、CPU 历史预算和正式参数均不改。
`qualification-r1/queue-r1/` 已绑定现有 GPU 执行链的真实进程身份，排在其
终止后执行，等待外部 GPU 用户与资源，不停止其他任务。GPU 结果尚待产生，
这两项不计入 4,194 道正式质量回答，最终完成审计须另核对它们。

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
| NIAH A/C/D | 64K/128K/256K 各 60 题，180/arm，共 540 | r1 三组全部通过；同运行时 r2 排在 LME 完整评分之后运行 |
| MemoryAgentBench A/C/D | 原始 1,116/arm，共 3,348；问题间恢复冻结 P，防止输出统计泄漏 | r1 错误保留；r2 A/C/D 各 1,116 题生成、释放与原评分完成，未判定项为零 |
| 独立 LongMemEval-S A/C/D | 原始 102/arm，共 306 | r2 A/C/D 全部完成生成、释放与原评分，86/102、92/102、89/102 |

### 独立 LongMemEval-S 参数与运行记录

2026-10-19 12:19（Asia/Shanghai）的只读审计通过了冻结 controller 的完整验证，
再次核对全部 102 个原始输入、Query 范围和逐题采样配置。三组 fixture 校验值
相同；A→C 仅 `graph_mode: collect → multi`，C→D 仅
`pressure_policy: attention → graph`。全部 132 个冻结运行时文件保持一致，
没有在正式臂中替换客户端或性能候选。

| 三组共同参数 | 固定值 |
| --- | --- |
| 模型与 KV | 相同本地 27B NVFP4 权重；FP8 E4M3 KV |
| 执行 | 编译与 CUDA Graph；MTP(k=4) |
| 客户端并发／原生最大并发 | 8／8；实际 batch 另行记录 |
| K／G | 36,864／16,384 |
| prefill chunk／每步总 token 预算 | 1,024／16,424 |
| sink／recent／块大小 | 1,312／0／32 token |
| CPU 历史 | 100 GB；不用 SSD、截断或重算 |
| 逐题采样 | 原始题目各自的 temperature、top-p、top-k、min-p、penalty 和 seed |
| thinking／输出上限 | 4,096／16,384 token |
| action 配额／历史 Q／guided Query | 0／关闭／关闭 |

此时 A 已完成的 42 道首次回答均核对了实际 request 参数、返回的完整原始
prompt token IDs，以及 accepted output IDs 与 usage/result 计数一致，零输出
触顶。C/D 尚未运行，不能把参数准备通过当作实际生成通过。历史 qw3 输出上限
为 32,768，与本轮三组统一 16,384 的差异继续单独披露，不宣称历史参数完全相同。
审计脚本与首次证明为
`artifacts/graph-v2-lme-efficiency-20261019/{audit_controls.py,controls-and-first-answers-r1.json}`；
后续应另存完整三组的审计结果，不覆盖这一快照。

运行期间同一只读 observer 记录 GPU 利用率、公开 prefill 计数、接受输出指标
及原生 batch。A 的一个早期 120 秒窗口平均 GPU 利用率约 87.13%，公开 prefill
约 5,417 token/s，输出计数约 16.62 token/s。后两项以相同混合工作窗口为分母；
前者包含重放，后者受到长 prefill 干扰，均不能当作隔离的 pure-decode 速度。
这些短窗口不构成 A/C/D 完整性能比较，也没有触发正式参数变更。
快照及原始采样前缀校验值为
`artifacts/graph-v2-lme-efficiency-20261019/baseline-A-r1/observed-window-r2.json`。

随后对前 60 个完成回答及已完成的 3,238 个原生 step 作只读阶段复核：题目
输入平均 108,976 token、输出平均 498 token。358 步 pure-prefill、2,650 步
mixed、230 步 pure-decode；约 99.28% 的累计模型 event 计时落在含 prefill
的批次中。pure-decode 只有三步为 batch 八，不能用这轮混合输出率检验持续
C8 decode 上限。纯 prefill 的 model-event 速率为 6,526 token/s，包含重放。
已观测到的 1,258 秒内 GPU 平均利用率 85.38%，覆盖同期原生提交区间的
94.55%；混合输出计数率为 24.46 token/s。未完成题仍在途，以上不是整轮吞吐。
这类长输入、短输出负载的输出率与孤立长生成压测的分母和工作组成不同。
响应落盘到结果校验落盘的间隔平均 0.830 秒、p95 1.259 秒；请求可重叠，
不能将间隔总和解释为 GPU idle。绑定原始日志前缀和首次响应校验值的证据为
`artifacts/graph-v2-lme-efficiency-20261019/baseline-A-r1/phase-reading-r1.json`。
同一日志前缀绘制的阶段图为同目录 `phase-timeline-r1.png`，图、分析记录和
绘图脚本的绑定为 `phase-timeline-r1.proof.json`；图中的批次数是执行采样，
不作为按时间加权的占用率。

后续执行链已实际注册到当前 LME 生成／原评分队列的进程身份，状态为等待
该队列完成。LME 三组全部生成、评分和释放后，先再次审计全部实际请求参数，
再依次运行同一 r2 的 NIAH A/C/D；最后运行独立 frozen-P 同步／异步功能验收
和独立 C8 固定输出诊断。每个任务仅消费 `prepared` 状态，不重跑已有尝试，
遇技术错误保留首次文件并停止；外部 GPU 用户只等待，不停止。
执行链为 `artifacts/graph-v2-completion-20261019/queue-r1/`，其中冻结了脚本、
依赖及正式 campaign 校验值。该队列尚未完成，不作为完整 4,194 题通过证明。

### 独立 LongMemEval-S A 完整生成验收

A 现已完成 102/102 个首次回答，技术错误与输出触顶均为零，所属历史、进程和
GPU 资源释放通过。再次核对全部原始输入、Query 与采样配置，实际接受的
73,670 个输出 token 与 response/usage/result 计数一致；最长回答为 4,220
token。A 完成时 C 自动启动，其首六题也通过同样的参数和完整输入核验。
这项生成通过不代表全部答对。随后 A 提前完成原裁判评分，结果见下表；最新 C 结果另列于后。

| 独立 LongMemEval-S 原裁判完整评分 | 答对／总题数 | 准确率 |
| --- | --- | --- |
| 当前 r2 A，无图传播 | 86／102 | 84.31% |
| 历史 qw3 Query | 89／102 | 87.25% |
| 历史 qw3 attention | 90／102 | 88.24% |

A 的 102 题均有有效裁判结果，未评分项为零，原始 12 项校准全部通过。
7 项仅在完整题目、答案和裁判请求均一致时复用历史评分，另 95 项使用冻结的
原裁判接口评分；首次模型回答未更改，C/D 推理队列未中断或变更参数。
评分绑定与完整结果保存在原 dataset 的 `scoring/`，副本为
`artifacts/graph-v2-lme-efficiency-20261019/early-A-score-r1/summary.json`，
SHA-256 为 `4fa0e408853352f15bf100464c264e1220d4a66ee4bce522a4d6a0af767c0af3`。
当前 A 比历史 Query 少答对 3 题、比历史 attention 少 4 题；历史模型精度、
执行方式、attention 配额和输出上限有上述已披露差异，不能归因为图方法。

进一步逐题配对核对全部 question/gold/input/Query 与评分绑定：A 相比旧 Query
修正 3 题、退步 6 题；相比旧 attention 修正 5 题、退步 9 题。分类型统计中，
multi-session 为 A 9/17、旧 Query 12/17、旧 attention 13/17；其余五类的 A
总答对数与旧 Query 相同，但不表示逐题结果完全相同。该分析只描述已有首次
回答，不据此更改 C/D 参数。证据为
`artifacts/graph-v2-lme-efficiency-20261019/A-original-pairs-r1/` 的
`comparison.json`、`comparison.md` 与 `paired-correctness.csv`；分析脚本为上一级
`analyze_lme_pairs.py`，只读且不调用模型或额外裁判。

等待 C/D 时，对执行链冻结副本的 C8 客户端验收进行了 CPU 预检：两个测试
文件共 31 项通过（0.42 s），包含重复／缺失原生 request 拒绝、单调时钟映射、
完整输入和共同窗口门槛。只使用一个低优先级 CPU 核，GPU 不可见；没有改动
冻结文件。证据为
`artifacts/graph-v2-completion-20261019/decode-cpu-preflight-r1.json`，绑定了四个
客户端 helper、两个测试文件和原队列 manifest 的 SHA-256。它仅证明测试
入口与负面门槛可运行，不代表尚未执行的 GPU C8 decode 已通过。

| A 全部 102 题的指标 | 结果与口径 |
| --- | --- |
| 原始输入总量 | 11,111,103 token |
| 整轮耗时 | 2,177.99 s，约 36.3 分钟 |
| 原始有效输入／整轮时间 | 5,101.55 token/s，包含输出、选页与释放时间 |
| 接受输出／整轮时间 | 33.82 token/s，包含长 prefill，非纯 decode |
| pure-prefill／model event | 6,526.11 token/s，包含重放与模型流水工作 |
| HTTP 平均／p95 | 167.84／307.66 s |
| TTFT 平均／p95 | 105.76／141.04 s |
| 观测 GPU 平均利用率 | 85.04%；覆盖原生提交区间的 96.65%，首 72.29 s 未采到 |
| 实际 native 峰值 | batch 八；仅三步为八路 pure-decode，无持续 C8 窗口证明 |

完整阶段记录为同目录 `baseline-A-r1/phase-reading-complete-r1.json`，逐题控制
证明为 `controls-and-first-answers-A-complete-r1.json`。原始 Python 栈采样文件
严格 UTF-8/JSON 解码失败，校验值和错误位置保存在
`baseline-A-r1/host-profile-integrity-r1.json`；未修补标签或据此声称 Python
热点归因。原生 step/NVML 记录有效并用于以上统计，C/D 保持相同采样设置。

### 独立 LongMemEval-S A/C/D 完整评分与性能

三组均完成 102 个首次回答、原裁判评分及所属资源释放；技术错误、未评分
项、输出触顶、thinking 未闭合均为零。每组原 12 项校准通过，A/C/D 分别
复用 7/8/9 项完整裁判请求与答案绑定完全一致的旧评分，其余使用冻结原接口。
原队列再次汇总 A/C 时，其已发布 summary 校验值不变。全部 306 题的完整
返回输入、Query、采样、接受输出与实际 worker 参数核验通过：A→C 只改
`graph_mode`，C→D 只改 `pressure_policy`。r2 NIAH A 已自动启动。

| 完整原裁判评分 | 正确／总题数 | 准确率 |
| --- | --- | --- |
| 当前 A，无 Query 传播／旧 attention pressure | 86／102 | 84.31% |
| 当前 C，V2 multi Query／旧 attention pressure | 92／102 | 90.20% |
| 当前 D，V2 multi Query／graph pressure | 89／102 | 87.25% |
| 历史 qw3 Query | 89／102 | 87.25% |
| 历史 qw3 attention | 90／102 | 88.24% |

A→C 仅修改 Query 的 `graph_mode`；逐题配对为修正 8 题、退步 2 题，净增
6 题（5.88 个百分点）。multi-session 从 9/17 到 14/17，knowledge-update
14/17 到 15/17，temporal-reasoning 16/17 到 17/17；preference 从 13/17
到 12/17，另外两类均为 17/17。这是一次完整同样本实验的描述性结果，不
据此宣称稳定普遍提升；历史执行／量化／配额／输出上限差异继续单独披露。
原始 token 输入仍使用 unknown locality，不能把此结果当作真实消息边界的
质量验收。D 相对 C 修正 3 题、退步 6 题，净少 3 题；相对 A 修正 7 题、
退步 4 题，净多 3 题。C→D 的 multi-session 从 14/17 降到 12/17，
temporal-reasoning 从 17/17 到 15/17，knowledge-update 从 15/17 到 16/17，
另外三类不变。本轮该套独立 LME 中 C 最好，pressure 图传播未叠加收益。

| 相同整轮口径 | A | C | D |
| --- | ---: | ---: | ---: |
| 全部 102 题耗时 | 2,177.99 s | 2,196.74 s | 2,198.96 s |
| 原始有效输入／整轮时间 | 5,101.55 token/s | 5,058.00 token/s | 5,052.89 token/s |
| 接受输出／整轮时间，含长 prefill | 33.82 token/s | 33.40 token/s | 34.76 token/s |
| pure-prefill／model event，含重放 | 6,526.11 token/s | 6,626.25 token/s | 6,421.70 token/s |
| 接受输出总数 | 73,670 | 73,362 | 76,432 |
| 共同观测 120～2,160 s 的 GPU 利用率样本均值 | 84.89% | 84.30% | 84.81% |

C 整轮比 A 长约 0.86%，但回答、选页和 batch 组成已改变；不能把整轮差值
全部归为图开销或称为受控的吞吐收益。D 比 C 多输出 3,070 token（约 4.18%），
整轮输出率的增加也不能视为固定工作量下的加速。三组约 98.6%／98.5%／98.8%
累计模型计时位于含 prefill 批次；A 仅三步 native C8 pure-decode，C/D 为零，
均没有持续 C8 窗口，纯 decode 上限仍由排队的独立诊断验证。
共同观测区间从各自首个 native 提交后第 120 到 2,160 秒；利用率样本间隔
约一秒，三组低于 20% 的样本比例为 3.97%／3.82%／4.27%。这些是同偏移
窗口的样本均值，未提供固定输出、重复配对的因果吞吐比较或精确 GPU idle。

后续 C8 诊断另通过启动容量预检：当前正式配置的实际日志估算 GPU KV
为 1,158,059 token，八份 K+G 为 425,984 token，显存比例仍为 0.72。
这只核对 token 容量级别的可行性，不能替代 MTP/recurrent、准入与持续
native C8 执行验证；独立测试继续使用原冻结配置。证据为
`artifacts/graph-v2-completion-20261019/C8-capacity-preflight-r1.json`。

与 A 不同，C 的 Python 采样文件通过严格 UTF-8、JSON、57 个 profile 及
41,510 个 stack 样本的结构校验。主要叶节点包括同步 Action metadata 上传
以及后台历史 SHA-256／封存；这些是采样栈权重，不是额外 CPU 墙钟或 GPU
idle。CUDA tensor 上传可能正在等待 GPU，后台封存也可与 GPU 重叠。
C 的完整 102 次 Query 事件累计传播函数耗时 3.85 s，含该函数的选页 wrapper
为 5.29 s，两者嵌套不能相加，也不覆盖全部图相关工作。当前材料不能把
同步上传的 228.59 个采样权重秒解释为同等 GPU 空闲。既有异步元数据候选
仍以其独立实验为准，未因采样热点而更换正式参数。

D 的 Python 栈文件同样严格 UTF-8 解码失败，保留其 18,912,570 字节与
SHA-256 `c2caea954d5bebeb2d986e70cfd50d656bcfc0a34d308f1d118cc7021f640334`，
不修补标签或做 A/C/D 栈归因比较。D 有效的原始图日志则确认了全部 434 次
pressure 均为 prefill seed、非 identity 传播，424 次改变选页。传播函数
累计 16.78 s，包含它的 pressure wrapper 累计 35.35 s；这是嵌套且可重叠的
主机区间，不等于相对 C 的额外墙钟或 GPU 空闲。C 的旧 pressure 没有对应
wrapper 计时字段，未记录不能当作零成本。

完整证据：

- `artifacts/graph-v2-lme-efficiency-20261019/early-C-score-r1/summary.json`，
  SHA-256 `5716bdfdf17e1d3af77519c6fee146cf3ed4a59d577c3044a366e5c0d2724183`。
- D 完整原评分为 `queue-r1/D-score-summary.json`，SHA-256 为
  `57c6c4d93d54a3a152cb9999f26c1b46f2be85d8380474476aea0595dcef36e4`。
- 同目录 `ACD-original-pairs-r1/{comparison.json,comparison.md,paired-correctness.csv}`。
- `baseline-C-r1/phase-reading-complete-r1.json` 与 `host-profile-summary-r1.json`。
- `baseline-D-r1/phase-reading-complete-r1.json`、
  `LME-common-observed-window-complete-r1.json`、`CD-graph-host-cost-complete-r1.json`。
- 全部 306 题完整参数／输入证明位于
  `artifacts/graph-v2-completion-20261019/queue-r1/LME-final-controls.json`。
- 实际完整 worker 配置差异与这 306 题证明的绑定为同 LME efficiency 目录
  `ACD-actual-worker-controls-complete-r1.json`：A/C 仅 `graph_mode`、C/D 仅
  `pressure_policy`，其余实际传入 worker 的字段逐项一致。
- A 完整图为 `baseline-A-r1/phase-timeline-complete-r2.png`，C 完整图为
  `baseline-C-r1/phase-timeline-complete-r1.png`，D 为
  `baseline-D-r1/phase-timeline-complete-r1.png`；对应 `.proof.json` 绑定图片、
  读取记录、脚本及不可变日志前缀。旧 A complete-r1 图的标题误标 partial，
  保留原图但以 r2 为准；这没有改变任何模型结果或测量记录。

绘图副本已修正硬编码的 A／partial 标签。原冻结汇总队列不修改，新增只读
`artifacts/graph-v2-completion-20261019/final-render-queue-r2/` 在完整矩阵报告
成功后生成 A/C/D 正确标签的图和完整逐题对比；上游不完整就明确保留未完成
状态，不调用推理或裁判，也不自行宣布整个目标完成。
该绘图队列的 r1 在执行前发现相对路径会被子进程工作目录重复解析，已只
停止其自有 CPU 等待进程并保留文件；r2 将输出路径先转为绝对路径。模型
推理、原裁判、原汇总队列及所有首次结果不受影响。

完整矩阵核验器已按当前文件格式只读复核 MAB 三组的 3,348 个首次回答和原
评分，宏平均重算一致；实际服务启动记录中的全部 worker 选项也与配置逐项
一致。快照只计入已经通过、释放的完整组，不把 LME 在途部分算作完整组。
证据为 `artifacts/graph-v2-completion-20261019/matrix-snapshot-r1/` 与
`actual-launch-audit-r1.json`。最终 CPU 汇总队列已注册到后续 GPU 队列的实际
进程身份，脚本和依赖冻结于 `report-queue-r1/`；它会在执行链结束后深核验全部
首次返回输入/输出、原评分绑定与阶段记录。未完成上游只能生成明确的未完成
快照，不将未评分、未运行或失败项记作成功。最终仍需逐项完成整个目标的审计。

NIAH 开始前另逐题复核了旧 qw3 Query/attention 两套共 360 个实际请求：
当前 180 题的完整 token IDs、Query 范围、seed 和所有采样参数均匹配；没有
删除 filler、截短输入或移动 Query。旧版 compact JSON IDs 哈希与当前 i32
哈希采用不同编码，检查器从同一完整 ID 序列计算两者后比对，没有直接比较
不同编码的哈希字符串。原上限 8,192、temperature=1、top-p=.95、top-k=20、
thinking=4,096 均一致。模型精度、推测解码实现、attention 配额和并发方式的
已知差异继续单独披露。证据为
`artifacts/graph-v2-completion-20261019/niah-historical-controls-r1.json`。
这是参数／输入预检；r2 NIAH A 已实际启动，其完成情况另行统计，C/D 排队。

r2 A 的首个完整长度 64K 已完成 60/60 exact、100% 数值覆盖，输出触顶为零。
另逐题复算冻结的原评分，核对完整原始 i32 文件、保存的请求描述中全部
采样／Query／seed、API token usage、服务端消费水位和逐请求释放，60 题
均通过。NIAH 的请求文件保存的是来源 i32 摘要，非完整 HTTP body hash；
SSE 也未保存第二份完整 prompt IDs，未将其描述成返回 ID 的逐个核验。
证据为 `artifacts/graph-v2-completion-20261019/NIAH-A-64K-first-controls-r1.json`，
生成器为同目录 `audit_niah_completed.py`。该命令显式允许仍在运行的 A，
只封存完整 60 题长度分组，`full_540_NIAH_complete=false`。

随后 A 的 128K 也完成 60/60 exact、100% 数值覆盖，零输出触顶。两种长度
共 120 题再次通过同一实际请求／原评分审计，证据为
`artifacts/graph-v2-completion-20261019/NIAH-A-64K-128K-first-controls-r1.json`。
256K 正在运行；最终还须对全部 A/C/D、三个长度运行该审计器的默认严格
模式，验证全 540 题实际请求描述与原评分，不把此部分结果当作 A180 或
整个矩阵完成。

C 的前 23 道首次回答另通过运行路径复核：23 道 Query 图事件实际改变选页，
99 次 pressure 事件仍是旧 attention，未发生 pressure 图传播；Query 坐标、
K 预算和 history/action 配额符合 C 组定义。该部分证明为
`artifacts/graph-v2-lme-efficiency-20261019/baseline-C-r1/first-query-policy-audit-r1.json`。
它不提供正确率，也不将选页变化等同质量提升。

D 的前 77 道完成回答已进一步与 `block_graph`／`bounded_refresh` 原始日志
的不可变前缀逐项复核，实际事件与首次结果内的报告完全一致。328 次 pressure
均使用 prefill seed、执行非 identity V2 multi 传播，其中 320 次改变选页；
所有选页均满足各事件当次的 runtime K 预算。Query 使用 Mean-K 起点，未
启用旧 Q 缓存或直接 action 页配额，raw-token locality 为 unknown。
证据为 `baseline-D-r1/first-runtime-policy-audit-r1.json`（位于同一 LME
efficiency 目录），绑定了日志前缀、77 个首次结果及其响应校验值。这是
运行路径复核，不是这 77 题的准确率或 D 整组完成证明。

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

### MAB r2 D 完整评分与五组对比（2026-10-19）

D 的 1,116 题原始回答全部完成评分，未判定项与裁判失败均为零。补评时旧
progressive 评分进程已经退出；保留中断记录和原有 300 条 LongMem 判断，
仅补齐缺失的 16 道摘要。使用相同冻结评分器、官方 rubric、DeepSeek 模型及
原预算；已核对全部 316 条裁判配置，原有 300 条判断逐项保持不变，未重生成回答。

| 子任务 | 题数 | A | C | D | 旧 qw3 Query | 旧 qw3 attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EventQA | 500 | 90.00 | 89.20 | 91.00 | 89.20 | 86.00 |
| RULER QA2 | 100 | 55.00 | 60.00 | 57.00 | 56.00 | 54.00 |
| 单跳事实整合 | 100 | 54.00 | 49.00 | 51.00 | 54.00 | 48.00 |
| 多跳事实整合 | 100 | 2.00 | 3.00 | 3.00 | 2.00 | 1.00 |
| MAB 内 LongMem 问答 | 300 | 56.67 | 56.00 | 56.67 | 54.33 | 56.33 |
| InfBench 摘要 F1 | 16 | 21.91 | 24.28 | 23.25 | 24.36 | 19.96 |
| **原三类宏平均 /100** | **1,116** | **39.04** | **39.56** | **39.49** | **39.62** | **36.63** |

前五项为正确率百分数，摘要为 F1×100；先在类内等权平均来源，再对三类等权。
D 的检索、冲突消解、长文理解三类分别为 68.2222、27.0000、23.2479，独立
重算宏平均为 39.49004569。D 比 A 高 0.44560 分、比 C 低 0.06902 分，
比旧 qw3 Query 低 0.13347 分，比旧 attention 高 2.85506 分。

按 1,100 道二元问答计数，A/C/D/旧 Query/旧 attention 分别答对
731/726/736/721/702 题。A→D 修正 35、新增错误 30；C→D 修正 41、新增
错误 31。该计数混合原 substring 与 LLM 正确性标准，是辅助诊断，不替代正式
宏平均。D 相比 C 的问答子项宏平均贡献为 +0.27407 分，摘要贡献为 -0.34309
分，因此答对更多问答却没有更高的正式宏平均。相对 A，问答各子项加权贡献恰好
抵消，宏平均净增来自摘要；不能把题数较多的 EventQA 增益直接解释为总分增益。

16 道摘要占正式宏平均的三分之一。单次原样本结果没有建立 graph pressure
优于旧 attention pressure 的稳定收益。旧 qw3 为 Q8_0，当前 vLLM 为 NVFP4，
即使逐题输入、Query、标准答案和输出上限一致，历史差异也不能单独归因于图算法
或推理框架。这里的 300 道 LongMem 是 MAB 子集，不代表独立 102 题 LME 已完成。

进一步核对旧 `launches.json` 后补充历史对照限制：旧 attention 的
`action_fraction=.5`，会把可选页约一半配额交给 attention；本轮 A/C/D 均为
`.0`，没有相同的 attention 配额选页。旧版为 Q8_0/FP8 KV/DFlash（最大草稿
7 token），关闭 continuous batching 并串行请求；本轮是 NVFP4/FP8 KV/MTP4，
八 worker 上限。因此历史分数比较是整体方案比较，不是相同参数下对图算法的
单变量消融。旧版启动时 guided 配置为 both、Query4096，但逐题汇总 MAB 的
实际 guided_events 为零，不能把启动配置当成这些题实际发生 guided 生成的证据。
旧 MAB 的全部 1,116 次语义选页都标记 `attention_source=prefill`，累计 3,819
个 pressure 事件使用 prefill fallback；107 个保留原冷启动回答，1,009 个使用
冻结前缀恢复。原模型输入、Query、输出上限及已发布评分仍保持不变。

证据均位于 `artifacts/graph-v2-early-scoring-20261019/`：

- `D-complete-r1/{manifest.json,status.json,summary.json,analyzer.log}`：补评、旧判断
  不变与原配置审计；中断记录和补评前缓存另外保留。
- `D-full-comparison-r1/{comparison.json,comparison.md,paired-records.json}`：五组
  完整来源审计、逐题输入/回答绑定、宏平均复算、各来源贡献和配对变化。

后续采用按完整分组就绪即评分的原裁判队列，状态和无凭据 runner 副本位于
`/data/chaidi/kvmem-vllm-experiments/graph-v2-progressive-scoring-20261019-r1`。
依次处理完整 C/D 与独立 LME A/C/D，等待时验证主推理进程仍为绑定的同一
进程；每组先确认 passed 与 GPU 清理，再运行冻结原评分器。原最终评分进程
在该流程拥有评分期间临时挂起，完成或失败后恢复，最后复用同一答案绑定缓存
做全矩阵检查。评分保持单核低优先级，未变更模型、评分协议或预算，也不再
让已经完成的分组等待全部九项推理结束。上述为提前评分队列的历史设计；D 最新
完整补评分及接管记录见上一节，不把已退出旧队列的状态文件视作实时进程证据。

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

2026-10-19 的进一步诊断确认，共享锁内的全量结果重读会随题目累计增长。
使用当时 D 已完成的 603 个结果、共 92,951,908 字节做三次真实文件重放，
原汇总的中位耗时为 424.19 毫秒；增量记录快照为 3.18 毫秒，一次初始化
463.08 毫秒。每次均校验汇总内容完全一致。即使只将完整 JSON 改为紧凑格式，
排除文件读取后解析仍需约 330.90 毫秒，因此没有改写正式结果文件来换取
表面上的读盘收益。证据：`artifacts/graph-v2-idle-reading-20261019/efficiency-r3/progress-replay.json`。
这是客户端 CPU 工作量的对照，不能表述成 GPU 吞吐提高 133 倍。

新版客户端 `incremental-client-2` 在已经取得本组历史额度后，只提前准备
下一题，重叠源文件校验、token/prefix 哈希和原采样配置读取与当前题 HTTP
推理；请求仍按组内原顺序，在原退场与审计校验通过后提交。准备失败会
取消后续组准入，退出时等待或取消唯一的准备任务，再归还额度。
每题新增独立 `client_timing.json`，区分源校验、HTTP、退场等待、审计和
进度发布；这些主机区间及跨题重叠不能直接充当 GPU idle 时间。
116 项针对性客户端、审计、原评估和冻结控制器回归通过，用时 3.80 秒，
日志位于同目录 `cpu-regression.log`。

真实 GPU 验证为独立实验，不改写 r2：
`/data/chaidi/kvmem-vllm-experiments/graph-v2-client-efficiency-20261019-r1`。
119 个模型/KVMem 核心文件逐文件与 r2 D 相同，使用同一服务顺序运行旧、
新客户端；取原样本前两个共享历史各九题，完整输入、原 Query 和采样上限
均保持不变。其余 916 份已完成 A 结果只作带哈希的客户端汇总负载，
不计为本实验的新回答，也不作新的质量分数。
这项诊断只有两个活跃共享历史；八个客户端上限不能作为 C8 计算证明。
分别记录首次冷准备、续问 HTTP、响应到下一次提交准备的间隔、原生
prefill/decode 步分类及 GPU 利用率；共同 warm 窗口不足 30 秒时不作
稳定利用率结论。单次旧/新顺序对照也不能作为普遍吞吐改善的证明。

这项独立真实模型对照现已完成并释放资源：每臂 18 个新回答、其中 16 个续问。
逐题复核完整输入、Query、采样配置和 119 个模型核心文件一致。观测到的
响应落盘到下一题 attempt 的平均间隔从 1.245 秒降为 0.077 秒，下降 93.81%；
这是本次单次有序配对的主机间隔结果，不代表普遍 GPU 加速。
整轮耗时 301.89 → 290.76 秒，表面缩短 3.69%，但续问输出为 141 → 235 token，
共同两 owner warm 窗口不存在，实际 pure-decode 均只有 batch 一，因此不能
将其作为受控吞吐提升。旧／新 NVML 覆盖率分别为 99.66%／52.26%；不完整的
新臂采样不能参与整轮利用率比较。完整证明为
`artifacts/graph-v2-idle-reading-20261019/efficiency-r3/client-diagnostic-audit-r1/audit.json`。

旧门控 `gate-r3/status.json` 未发布终态，不能据其旧的 running 字样认定进程
仍在运行或原工作流已经恢复。独立实验自身的通过与释放记录均已复核。
目前 LME 与剩余 NIAH／独立验收由上文的新执行链继续，原正式配置未替换。
同步 frozen-P 恢复和 Query/装配成本仍未由客户端修改解决。49 Hz nonblocking
子进程采样的原始输出有 UTF-8 元数据损坏，严格 JSON 解码失败，保留原文件，
不以替换损坏字符后的栈标签作 CPU/GPU 归因。当前 LME 三组采用相同的另一次
采样方式，完成后仍需单独验证文件有效性。

用户再次询问低利用率时，对运行中的 D 另取固定 180 秒只读窗口，证据为
`artifacts/graph-v2-idle-reading-20261019/efficiency-r3/current-D-window-r1.json`，
分析程序为同目录 `read_current_D.py`。当时 D 完成 878/1,116、技术错误为零。
窗口内 176 个 NVML 样本的平均利用率 62.55%，低于 20% 的样本占 18.75%。
这是一次含新历史冷准备的窗口；早先约 10% 或 43% 的读数来自不同阶段，
不能据此宣称优化造成提升。

实际原生计算步为 460 个，batch=1/2/3/4 分别为 239/184/36/1；其中没有
prefill 的纯 decode 只有 14 步，batch=1/2 为 13/1。窗口执行 546,491 个
公共 prefill token，包含重放；31 个完成回答平均 8.16 个 output token，
其中一个是新历史首题。29 对同组相邻题目，响应文件落盘到下一题 attempt
开始平均间隔 5.78 秒，包含退场、审计、汇总和下一题准备，不能归到单一函数。
这类有限短回答负载没有形成稳定大 decode batch，也不能用完成回答的
output/墙钟当作纯 decode TPS。

配置为八客户端／`max_num_seqs=8`，但 CPU 历史额度峰值只容纳五组；当时
占用额度约 77.56 GB，上限 82.82 GB（100 GB 减 16 GiB 余量）。独立调度
快照有四个请求、两个 refresh pending 和两个 connector pending；这些状态
可以指向相同请求，且不同快照不是原子读取，不能相加或直接推导就绪数。
它们说明八个客户端上限与八路持续计算需要分开验收。
该调度快照仍有 574/751 个原生 cache 块空闲，reservation/authority 拒绝均
为零，提示本窗口应先检查请求补位与恢复等待。后续进度复核到 922 个回答时，
12 个是各组首题、910 个复用冻结 P；完整原始输入范围为 267,553–757,259
token。因此已经实现“一次历史准备、多次独立问题复用”，并非每题都重跑
整段长历史。续问仍包含 active KV 恢复、Query/replay 和短生成的固定成本。

另一个源码确认的限制是 MAB frozen-P 配置的 `async_restore=false`。
`SessionWorker.before_attention()` 在公共计算前执行同步 `_restore()`，
其中包含 checkpoint 准备、active KV 装配和完成栅栏等待，可能拖延同一计算
轮中其他请求。`async_refresh=true` 只覆盖重选，不代表该恢复已经异步。
该冻结运行时的异步恢复要求 P/M 生命周期；不能直接开关替换独立问题的 sealed-P
语义。其独立成本和阻塞比例仍需短对照的主机采样及进一步阶段测量，
目前不把这部分等待量化为全部 GPU idle。

随后在主线实现了显式 frozen-P 异步恢复候选。复用已有的带配额后台装配和
native row/page lease；仅在 `session_freeze_prompt=true`、sealed P-only、
无 terminal/decode 捕获和无 Query history 时扩展异步恢复的配置契约。
后台完成的 P 保持原检查点所有权，调度器对已经验证的同一 token 前缀不再
重复捕获；取消发生在恢复完成但尚未运行下一 token 时，也能退场到同一 P。
普通 P/M 和原同步 frozen-P 路径不改，正式 r2 配置和运行时仍保持冻结。

395 项受影响的 options/launcher/session/checkpoint/scheduler/async/graph 评估
CPU 回归通过，用时 8.56 秒，包含 13 项新测试：精确前缀、跨题 graph/action
重置、模型 snapshot 复用、未完成 fence 不发布、其他请求可继续、取消后的
再次复用以及最终配额释放。CPU 假设备测试不等同于真实 GPU KV 字节检查。
日志：`artifacts/graph-v2-frozen-p-async-20261019/cpu-regression.log`。

独立真实模型功能验收已准备在
`/data/chaidi/kvmem-vllm-experiments/graph-v2-frozen-p-async-20261019-r1`：
两个合成 16K 历史、各三道独立问题，分别验证同步与异步路径，开启
target/MTP/recurrent 状态字节 oracle，并要求旧分数不泄漏、P 不重算、
异步仅首次捕获 P 和完整释放。该小几何验收不替代原始 MAB 样本、质量评分、
全长度容量或吞吐测试。默认开关与正式 A/C/D 配置暂不采用新候选；
截至记录仅通过 CPU 与启动配置检查，GPU 验收仍待执行。

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
