# 将 active KV 容量转化为持续 agent 吞吐

状态：恢复管线及滚动窗口的真实模型正确性、首轮和反序性能对照已完成。
恢复管线吞吐持平；滚动窗口两次自然负载的输出率增幅分别为 17.3%／9.6%，
完成回合增幅为 2.8%／3.6%。反序原生 .90 为 100.957 token/s，K 候选为
180.177 token/s；容量带来的实际 batch 扩大尚未证明。用户暂停后已明确要求恢复，
自然原生 .915 对照已重新完成，为 87.620 token/s、59 个完成回合、零抢占，
全流程采样显存余量 4245 MiB。Action 上传首轮完成恢复校验，但评分验收条件
错误地要求全 thinking 回合也有 non-thinking 样本，暂未放行。原生 batch 观测
资格已通过；独立长历史 G16 原生臂完成，但切换 K 臂前的 CPU 余量门禁失败。
原生现场保留，独立 K 臂已完成：167.203 token/s、94 个完成回合，零抢占，
完整历史发布通过。长历史纯 decode 实测 103 步为 8 路；其中 102 步通过保守
成员范围核对，对应完整原生 KV 的需求下界 1225～1249 页，超过原生实测
1193 页池，但 8 路只占纯 decode 事件约 5.05 秒，不能宣称持续满 8 路。
原生 7 槽位对照也完成，为9.020 token/s、零抢占，198个混合步骤且无纯decode；
启动时显存余量不足4GiB，不能作为通过部署门禁的加速比较。Action 元数据
异步上传的4项CUDA检查及真实模型恢复/评分复核通过；性能窗178.703 token/s、
108回合，低于旧rolling-v2的180.177及114回合，因此本次真实first16不开启该候选。
提前释放下载缓冲区仍仅完成 CPU 验证，未加入恢复后的队列。最新状态与解释见文末。

目标是验证“缩小 active KV → 更多就绪请求 → 更大的持续 decode batch”，并减少
KVMem 恢复、选页、历史维护造成的请求等待。容量受压负载争取比原生 vLLM
提高至少 25% 的固定窗口输出率；这是验收目标，不是已实现的性能承诺。

## 策略与首个变更

公共 decode 继续使用原生 continuous batching。已有 CPU/GPU 准入、完成 fence、
压力驻留和单生产者冷装配继续生效。先修改可定位的热恢复串行点：

- 原实现从原生分配前准入到 worker 完成 ACK，只允许一个 resident 恢复。
- 新增 `resident_restore_slots`，默认 1 保持兼容；候选显式设置 4。
- 每个热恢复独立持有 permit、native row/page lease、CPU checkpoint 和完成事件。
  取消不释放在途额度；仅对应请求的完成确认释放额度。
- 仍只有一个热恢复提交线程和 stream，按顺序提交独立的模型状态上传；允许
  下一请求在上一请求的完成 fence 尚未确认时排入管线，不增加 GPU 模型执行槽。
- 冷恢复/Query 继续共用一个独占装配额度，不并行争抢 CPU gather 和 staging。
  热恢复总 payload 上限随 slots 有界增加，每请求上限仍是 checkpoint 上限。

选择这个变更的依据是已有 8-agent 窗口中 1341 次 resident 恢复平均跨度 216 ms，
同时纯 decode 步骤只有约 24% 达到完整 8 路。这说明值得检验热恢复串行化是否
影响补位，不能直接把这两个相关统计当成因果证明。

## 实验与解释边界

第一关为 4 个真实历史会话的恢复校验，开启 target/MTP KV 和 recurrent state
字节 oracle，证明多个独立 resident import 重叠时不发生用户串扰或提前回收。
oracle 开启时的吞吐不参与性能比较。

第二关用原始任务顺序的前 12 个 DeepSWE 轨迹，每个选择最早满足 128Ki 历史
且具有连续 16 个完整回合的起点。各自初始输入约 131K～134K token，完整保留，
不复制、不填充。工具消息、工具等待和每回合输出长度均冻结；模型生成自己的
accepted token ID，不能用记录里的旧答案替代新生成。该重放不评估任务正确率。

比较原生 vLLM、已验证 KVMem 基线及恢复管线候选。三组接收相同数量的 agent，
原生也允许 max_num_seqs=12，自行排队/抢占，没有客户端强制限成较小并发。
统一 NVFP4、FP8 KV、原生 MTP(k=4)、CUDA Graph、模型 256Ki 上限；KVMem 为
64Ki active + 32Ki generation、CPU 历史 100GB，不用 SSD。100GB 仍是总量上限。

先完成两轮预热，再测固定 300 秒，恢复、排队、选页、工具等待及窗口内的
新 epoch 冷准备均计入。正常结束有限 epoch 后才能清理该历史；不因资源压力
截断、遗忘或重算 KVMem 历史。CPU 准入预检按冻结 epoch 的实际最大长度加
完整 MeanK 索引及 32GiB 系统/模型/DMA 余量计算，运行保留 8GiB 可用内存。
12 路在此有限历史长度下是否可行必须实测，不能宣称支持 12 路满 256Ki 历史。

主要指标为固定窗口接受输出 token/s；prefill 计算量与其服务墙钟速率分列，
不能把较少的重复 prefill 误解为较弱的 prefill 算力。同时记录实际 GPU decode
batch、输出事件间隔、恢复等待、公共 forward 完成计数、抢占和 CPU 历史发布。
NVML 低利用率且存在 HTTP 请求只表示待处理需求，不能直接证明工作已 GPU-ready。

代码和配置在 GPU 启动前冻结，失败保留现场，不自动重试。候选只在已验证的
partial-M 修复运行时上加入五个文件的恢复管线，排除工作区其他未验证优化。

## 当前产物

- 最小候选及差异：`artifacts/ready-agent-pipeline-20261017/`。
- 真实恢复校验：`artifacts/ready-agent-pipeline-qualification-20261017-r2/`。
- 三组固定窗口：`artifacts/ready-agent-pipeline-performance-20261017-r3/`。
- 12-user 外部消息轨迹：`artifacts/ready-agents-first12-trace-20261017.json`。
- 入口：`scripts/benchmark_ready_agents.py`，准备和冻结执行分开。

相关恢复/准入/生命周期/启动器 CPU 回归 175 项通过；新工作负载预算及归因校验
9 项通过。实际冻结候选另通过 50 项恢复/准入回归，属于前述集合的子集。

恢复校验 r2 完成：9 次恢复均通过 target/MTP KV 字节比较及 96 个 recurrent
张量比较，覆盖 17 个 attention 层和 1 个 MTP 层；同时占用的 resident permit
峰值为 2，全部输出计数一致，零抢占，CPU 历史磁盘读写均为零，进程清理完成。
这证明重叠恢复正确性，不将开启 oracle 时的吞吐当作性能收益。

r1 在首次编译及服务启动后、提交任何模型请求前发现新 trace 未填写 EOS，
工作负载校验拒绝启动。已增加启动前的本地 tokenizer EOS 补齐及一致性检查，
失败产物保留。performance-r1 在 GPU 启动前被加入历史发布记录的 r2 替代；
performance-r2 因前置校验失败没有启动 GPU。r3 在校验 r2 完成、目标/状态
字节证明及运行代码哈希核对通过后，才执行三组正式窗口。

r3 原生组已完成并正常排空：300 秒窗口内 2,863 个接受输出（9.54 token/s），
窗口新增 2 次抢占，GPU 平均利用率 99.70%，监控覆盖完整。原生 prefill 服务
计数增加 1,095,834（按窗口为 3,652.78 token/s）；该计数在请求完成 prefill
时入账，不能当作 kernel 算力。全流程 68 个请求全部完成，client/native/usage
总输出均为 30,056，窗口结束后正常排空用时约 209 秒。整个进程累计抢占 10 次，
其中预热 3 次，不应把累计值写成窗口值。详细审核为
`artifacts/ready-agent-pipeline-20261017/native-audit.json`。

此处低输出率伴随高 GPU 利用率，主要是完整历史重复 prefill 和缓存压力。
12 路两轮预热后的完整输入仅算 attention 也至少需要 999 页，原生总池为
989 页；不能以 12 条 HTTP 请求或这个算术下界代替实际并行观测。

r3 的 KVMem 基线完成 300 秒窗口，接受输出 99,819（332.73 token/s），GPU
平均利用率 78.22%，零抢占。实际纯 decode 共 2,184 步，其中 1,468 步为 11 路，
没有 12 路：当前 906 页缓存池和每请求 76 页最坏生存期额度只允许同时准入 11 路。
这是本轮服务窗口输出率，包含工具等待、维护及窗口内冷 epoch 替换，并非纯 decode
kernel 吞吐。完整 CPU 历史及补充容量证明已通过独立审核。

原生 9.54 token/s 的低点包含缓存压力和重复 prefill；需要另测全部 12 个客户端
继续排队、仅把原生 active `max_num_seqs` 设为 8 的配置。此值按冻结有限历史的
最坏 8 路页需求预先选择，不根据后续吞吐挑参数。在补测之前，不把两数相除作为
KVMem 相对合理调优原生部署的稳定加速幅度。

恢复管线候选正式窗口为 99,923 输出（333.08 token/s），相对基线仅 +0.10%，
本轮没有明确吞吐改善。GPU 利用率为 81.51%，低于 10% 的采样时间比例由
3.29% 降至 2.20%。实际纯 decode 2,276 步，其中 1,594 步为 11 路；两组都零
抢占，完整历史/M 检查点发布与三方输出计数核对通过。

resident 准入等待中位数由 304ms 降至 9.5ms，4 个恢复额度确实被同时使用；
但到达→ready 的 p95 为 3.645→3.659 秒，未改善。worker 完成→调度器 ready
p95 由 276→407ms，MTP 平均推进长度也由 3.847 降至 3.673，说明一次自然生成
轨迹与调度变化不能孤立归因为纯调度收益。基线正式窗还发生首次形状 JIT；需要
重复试验才能排除预热和顺序影响。完整审核见
`artifacts/ready-agent-pipeline-20261017/r3-comparison-audit.json`。

容量链已有实际证据：仅计算候选已观察的 11 路 GPU decode 所对应的最短可能
完整历史，加当前原生每请求最低 15 页 recurrent/MTP 状态，就至少需要 1,150 页，
超过该原生对照实际 989 页。该证明对应当前显存预算和历史长度，不代表调大原生
缓存池后仍有同样差距。细节、源码行与哈希见
`artifacts/ready-agent-pipeline-20261017/recurrent-mtp-capacity-proof.json`。

后续原生配置采用预声明的 2×2 对照：active slots 8/12 × GPU fraction .81/.90，
始终 12 个客户端、相同样本、相同实际 16,444 token batch 总预算。12/.81 已完成，
另外三格已冻结；.90 是基于原生已有显存峰值的待验部署配置，不能提前承诺可用。
会报告全部配置，并采用有效结果中更高的原生输出率比较，不能只引用 9.54 的低点。
排队清单位于 `artifacts/ready-agent-tuned-native-20261017/queue-plan.json`。

独立比较有效性审计见 `artifacts/ready-agent-comparison-validity-20261017.json`。
原生默认 hybrid APC align 的声明最大 recurrent/MTP 容量几何为每请求 18 页；
KVMem none 模式为 15 页。原生稳定步骤可能只保留 15 页，跨对齐边界会增加页，
因此 18 不能作为每步实际占用下界。此前用 15 页证明原生需求超容量仍为保守下界；
严格比较应使用这个下界或真实非 null 状态页，不能把声明上界当作实测。
原生 `num_requests_running` 包含 prefill，不能代替实际 decode batch；没有步骤
观测时应显示未知，不能将分析器的空记录 0 写成零路 decode。

自然结束三组的 .81 原生仍保留为等预算对照，另已冻结独立
`artifacts/natural-native8-deployment-20261017-r1`，只将同一自然工作负载的原生
显存预算改为 .90，并使用独立端口；保持 8 用户、实际 16,424 batch、完整原样本、
相同采样、两轮预热和 300 秒窗口。12 项 CPU 检查和独立复核通过；自然三组完成后，
该原生 .90 组已完成正式窗口及正常排空，接受输出率为 67.59 token/s，详细对比
见下文。固定 quota 的 12 用户 .90 结果不能替代这一自然结束对照。最终仍需
反向顺序复测，首轮不宣称稳定收益。

自然 native/.90、Action 诊断、K12/.82 和 fixed12 native/.915 已完成；K12 未通过
显存余量要求，详见文末。MeanK 独立构建／数值资格也已通过，当前执行安静
反序复测。下一步已准备独立 natural native/.915 和异步 Action metadata 资格，
两者尚未启动或排入 GPU 队列。正式自然窗口的共享槽等待为零，因此三个 early
实验已延后，优先准备已测到的 Action 元数据等待和 prefill 显存峰值诊断。
2026-10-17 12:08 的调整仅逆序停止三个已核实身份、无子进程的等待器；三份
实验的 prepared 状态、manifest 和冻结源保持不变，候选仍默认关闭。操作在
原生对照预热期完成，见 `artifacts/focus-action-queue-20261017`。没有停止
GPU 服务、MeanK 等待器或反序等待器；新 Action 资格尚未排入 GPU 队列。

此前的队列调整只逆序停止四个已核实身份、无子进程的等待器，再按新依赖启动六个等待器；
运行中的 Action controller／server 和原 K12 等待器保持不变。六份实验 manifest、
样本、参数及冻结运行时均未改。操作及独立验收见
`artifacts/capacity-first-queue-20261017`；前次提前原生对照的记录仍保存在
`artifacts/strong-native-priority-reorder-20261017`。独立原生对照不再依赖 early
资格；early 性能仍必须通过字节和真实复用资格，覆盖不足不得放行。

安静反序入口为 `artifacts/natural-rolling-reverse-20261017-r1`，17 项 CPU
检查及独立执行路径审查通过。顺序为 rolling → base → native/.90，精确复用
原两个 K 运行时及字节资格；原生实际 argv 除端口外与已完成 .90 相同。每组
启动前和 GPU 空闲后都核验原样本、预算、冻结源及 2,699 个安装 Python／库文件。
未混入 MeanK 启动修复或 early／Action 诊断；旧 loader 首次编译仍可发生在
预热期，不能把此复测当作严格冷启动比较。正式窗口期间不安排 CPU 测试或审计。

自然 first8 的原生 .81 组已完成：正式 300 秒接受 2,513 输出（8.38 token/s），
7 回合完成／15 回合发起，窗口内无 epoch 重启。15 个请求最终均自然 stop，
无 length 截断，完整输出均值 266.4 token；这一长度统计包含正常排空完成的请求。
GPU 平均利用率 99.80%，窗口内 1 次抢占，全流程 3 次。完成边界外侧计数为
971,278 本地 prefill、1,585,296 缓存命中 token，不能视为 kernel TPS。正常排空
132.62 秒后，全流程客户端与原生输出均为 17,034；完整跨轮 input token 哈希及
32,000 输出上限均已审计。实际池 1,003 页；原生实际 decode batch 仍未知。
独立证据为 `artifacts/rolling-active-window-20261017/natural-native81-audit.json`。
同一自然 first8 的 KVMem pipeline-base 现已完成：正式接受 47,607 输出
（158.69 token/s），107 回合完成／114 回合发起，正常排空 80.67 秒后，最终
客户端／原生输出均为 60,836。114 个正式请求全部自然 stop，length 截断为零；
完整输出均值 471.40 token。130 条 warmup／workload 的完整跨轮输入哈希及
终端 H=M=input+output−1 全部通过。窗口内 4 次 epoch 重启均因预声明的
16 轮上限，未因内存压力裁剪、回收或重算完整 CPU 历史。

正式窗口 GPU 平均利用率 79.46%，低于 10% 合计 7.76 秒、最长 1.14 秒，
没有重复预热时的 49.54 秒空档；抢占为零。1,892 个纯 decode 步骤的 batch
频率均值 5.927，按 CUDA-event 步长加权为 5.824，其中 86 步实际 8 路；
这些是实际执行人数，不是原生 running gauge，也不是 SM 占用率。110 次恢复
均命中 resident view。公开 prefill 完成计数的内外边界为 643,760–647,937，
即每正式窗口秒 2,145.9–2,159.8 个位置；不能称为纯 prefill kernel TPS，也不能
与原生按完整逻辑前缀入账的外部缓存计数混用。

独立审计为 `artifacts/rolling-active-window-20261017/natural-pipeline-base-audit.json`。
相对自然 native/.81 的输出率为 18.94 倍、完成回合率 15.29 倍，但两组生成长度
和 epoch 重启数量不同；更强 native/.90 的结果见下文，不再用弱 .81 作为主要倍率。
rolling-candidate-v2 已完成并正常排空；在反向复测完成前，不宣称稳定倍率
或准确的 batch 因果贡献。
固定 quota 的约 333 token/s 不能移用到本组。

ready 供给审计见
`artifacts/rolling-active-window-20261017/natural-baseline-ready-supply-audit.json`。
正式窗口平均 0.764 个 agent 等工具、1.691 个等首 token、5.531 个处于客户端
streaming；最后一项不是 GPU-ready。首 token 等待中，四次 epoch 冷准备占
288.35 agent-seconds，110 次续接占 218.99 agent-seconds。pure／mixed 步骤
时长分别约 104.21／187.51 秒。资源快照不足以证明长期遗漏已就绪请求；
<=100ms 的近邻快照只覆盖 226/1,892 个纯 decode 步骤，不能用稀疏观测排除
所有短阻塞。

本组正式窗口 main/bulk slot_wait／reclaim 增量均为零，尚无证据支持通过提前
释放共享槽直接收回这类等待；该功能也不会解除同 owner 的语义发布 fence。
publication-blocked 采样时间主要属于 prefill（约 177.46 agent-seconds），decode
仅约 2.22。后续仍要通过更高容量增加可计算请求供给，同时定位冷 prefill 的
CPU 历史追加和 owner 发布；累计 host 维护时间可重叠，不能等同 GPU idle。

原生 8/.81 已完成独立补测：仍为 12 个客户端及同一 r3 样本，正式窗口接受
输出 3,282（10.94 token/s），零抢占，GPU 平均利用率 99.75%，正常排空 184 秒。
全流程输出 30,401，客户端／原生计数一致。窗口内没有新 epoch 初始请求，22 次
续接仅新增 10,656 个工具 token；prefill 完成边界外侧计数却包含 1,229,686 个
本地计算 token 和 785,376 个本地缓存命中 token，命中约 39.0%。这与跨轮完整
历史无法全部保留、因此重复 prefill 相符；没有记录逐页驱逐，不能称为直接测得
驱逐数量。零抢占不等于跨轮没有重算，99.75% NVML 也不等于高输出率。
该 prefill 计数按完成边界入账，不是 kernel 算力。完整审核为
`artifacts/ready-agent-native8-equal-budget-20261017-r2/native8-audit.json`。

原生 8/.90 也已完成并正常排空，实际池增至 1,166 页。窗口接受 5,713 输出
（19.04 token/s），15 个回合完成／27 个发起，零抢占，GPU 平均利用率 99.56%。
正常排空约 204.77 秒，全流程客户端／原生输出均为 36,628。完成边界计数包含
1,244,106 本地 prefill 和 1,447,936 缓存命中 token，命中约 53.79%；新增工具
输入只有 13,298 token。更大缓存改善了输出率，但仍存在大量历史 prefill，不能
把 GPU 满载归因为大量有效 decode。全流程 932 次显存采样峰值 91,562 MiB。
完整审核为 `artifacts/ready-agent-native8-deployment-20261017-r2/native8-audit.json`。
原生 12/.90 已完成并正常排空，实际池为 1,164 页；正式窗口接受 3,888 输出
（12.96 token/s）、13 个回合完成，正式窗口 1 次抢占，全流程 2 次。GPU 平均
利用率 99.47%；单纯增加原生并发槽没有改善此负载的有效输出率。随后完成的
8/.915 将已测原生输出率提高到 58.60 token/s，见下文。以上均是单次窗口，未作稳定显著性
的结论；自然结束负载另行比较，不能混用其采样策略和窗口。

六个已完成窗口的统一审计为
`artifacts/ready-agent-pipeline-20261017/completed-fixed12-native-controls-comparison.json`。
该六组审计保留原始结果；新增 .915 的完整审计另列。KVMem 基线／管线候选
约 333 token/s，对当前最佳已完成 fixed12 原生 58.60 token/s，
属于容量受压的固定回合轨迹服务输出率差异，不能称为 decode kernel 加速。
主要机制证据支持避免历史重复 prefill；尚不能精确分解短 attention、更多并行、
MTP 接受率及历史复用各自的贡献，也不能用 .81 的容量下界证明 .90 必然装不下
相同的 11 路。当前 KVMem 管线改动本身仍只有约 0.1% 的吞吐差异。

独立只读入口 `scripts/analyze_external_native_capacity.py` 已通过 22 项 CPU 检查，可将 KVMem 单独运行
与指定原生部署对照，核验相同 fixed12 样本、完整输入哈希、原始工具消息、各自
生成历史、固定采样及实际 batch 预算。已有 step 只记录人数，故对覆盖该时刻
的 HTTP 请求超集取 N 条最短完整历史；混合批次只计 decode 子集，忽略 prefill
和本轮生成增长。状态下界 15 限于已审查的模型／MTP4／FP8／三组 recurrent
部署及源码合同，不明配置只报告 attention 下界。

对 r3 管线候选与原生 8/.90 的分析已保存为
`artifacts/external-native-capacity-r3-vs-native90-20261017.json`：2,276 个纯 decode
步骤中 2,269 个可验证，7 个因同会话边界重叠为未知；353 个 mixed 步骤可验证。
这些保守下界均未超过原生 1,165 个可用页，因此旧的 11 路观测不足以证明超过
调优原生容量。需求下界低于可用页数，也不证明实际一定放得下。CUDA event 跨度与 host
窗口裁剪估计分列，无法得到精确 GPU 时间轴；后续 .82／.915 结果也按同样口径
审计，不能只用最大 batch 或旧容量宣称链条已经完整成立。

容量临界点补测已完成：
`artifacts/ready-agent-native8-capacity-threshold-20261017-r1`。原生 8/.90 轨迹的完整
历史加 align 峰值预留，在 T1／排空时约需 1,170／1,179 页，略高于池 1,166 页；
但按 15 页稳态下界分别仅需 1,146／1,155 页，不能据此断言 .90 实际装不下。
保持固定 12 用户样本、原生 8 槽和实际 batch 16,444，只有显存 fraction 与端口
变化。`.915` 实际池为 1,193 页；正式窗口接受 17,581 输出（58.603 token/s），
30 回合完成／42 回合发起，GPU 平均利用率 98.91%，全流程零抢占。正常排空
188.85 秒后，66 条普通请求的完整输入／输出哈希及固定 quota 均通过，90 条
含 cleanup 请求全部完成；客户端与原生输出均为 50,190。
全流程 NVML 峰值 93,010 MiB，扣设备 reserved 后余 4,239 MiB，超过 4 GiB
门槛仅 143 MiB；一秒采样仍可能漏掉瞬时峰值，不宜推广为其他负载的安全保证。

完成边界计数中，缓存命中比例由 .90 的 53.79% 提高到 78.08%；本地 prefill
计数由 1,244,106 降至 1,037,350，而完成回合由 15 增至 30。它支持更大池改善
前缀保留的解释，但缺少逐页淘汰记录，不能精确归因于某一容量阈值。T1 仍有
长 prefill 未完成，这些计数不是窗口内 kernel TPS。完整审计见
[native8-audit.json](../artifacts/ready-agent-native8-capacity-threshold-20261017-r1/native8-audit.json)。
以这一更强原生作为分母，K 管线候选 .81 的窗口输出率为 5.684 倍、完成回合数
为 3.833 倍；仍属于单次、不同推进 cohort 的服务比较，并非纯 decode 加速。

这次跃迁也意味着不能将自然负载的 .90 假定为充分调优的最强对照。独立
`artifacts/natural-native8-capacity-threshold-20261017-r1` 已准备并通过冻结核验、
19 项 CPU 门禁及只读复核；原样复用已完成 natural/.90 的 8 用户、自然采样、
两轮预热、300 秒窗口、32,000 输出上限和实际 batch 16,424，只改 .915／端口。
它尚未启动或排队。全阶段 4 GiB／零抢占作为独立部署验收结果；即使不满足，
正常排空和计数成功仍保留 `completed_pending_review`，不隐藏负结果、不自动
重试，也不阻断后续无关的正确性资格。

公平性补充见 `artifacts/ready-agent-pipeline-20261017/native12-0915-feasibility-fairness.json`：
只测 8 槽 `.915` 不能证明原生 12 路物理不可行。12 槽 `.915` 的池投影约为
1,191–1,192 页，可能跨过同 fixture 已观测 T0 长度的 1,180 页低包络；低包络
小于池并不证明完整驻留或实际 12 路 decode。已有 12/.90 显存峰值投影到 `.915`
后，扣除设备 reserved 的余量约为 3,443–3,452 MiB，未达到当前 4 GiB 约束。
据同一线性估算，约 `.9083` 才能保留该余量，但这也只是投影而非安全保证。
因此维持当前队列，并将 12/.915 标为在保守余量约束下未测，不能称为硬件上限。

原生执行 batch 可观测性只读审查见
`artifacts/native-executed-batch-observability-20261017.md`。现成 iteration 日志能
区分 context／generation，但缺少完整逐请求历史与接受结果；scheduled generation
含 MTP 待验证 token，不能作为已接受输出。未来精确诊断应关联实际 InputBatch
和既有 AsyncOutput，正式未加诊断的性能数据保持独立。

首批 native8 预备产物 r1 在 GPU 前核对实际启动参数时发现 Query allowance 覆盖
了显式 batch 预算，已保留并标记 superseded；r2 的 native-only 配置移除该无关
字段，实际命令已验证为 16,444。安装 vLLM 的 2,699 个 Python/共享库文件另在
队列准备时锁定哈希，每个后续实验启动前重查；旧资格记录仅锁两份容量源码，
不追溯声称旧运行已做全库哈希。

## 第二项：按实际增长滚动 active 窗口

旧的真实 8-agent 记录暴露了另一项维护成本：786 次可匹配请求的压力重选中，
784 次发生在最后的 prefill 封页边界，这些请求均声明 `max_tokens=32000`。
代码在进入 decode 前为完整最大输出留空间，短工具回合因而可能频繁重选。
这与记录吻合，但性能因果仍需独立对照，不能仅凭事件位置认定全部开销来源。

新增 `kvmem.rolling_active_window`，默认关闭，作为独立候选验证：

- 保持客户端输出上限及 `max_tokens <= G` 的限制；CPU 完整历史、K+G 物理上限、
  MeanK／Action pressure 评分和必留内容均沿用原有实现。
- 跨轮准入按实际保存的 active view 加本轮输入检查空间，不为尚未生成的整个
  输出上限提前压缩。新增输入确实放不下时仍走既有压力恢复。
- decode 接受前缀到达 `K+G-block_tokens` 时，只暂停当前请求，等待它自身的
  CPU 历史发布，以已有评分做压力选页并装配；其他请求继续原生 batch。
- 该控制事务不运行私有 Query，也不消费 sampled token 或未验证的 MTP 草稿。
  完成事件确认后提交页表和位置映射，然后由原生 MTP 继续生成。
- 保留原生 GDN 行和接受计数；恢复首步无草稿时，已有接受前缀规范化逻辑继续
  生效。取消仍保留页／索引至完成确认，不把取消后的 GPU 页标为可复用热缓存。

取消只承诺恢复到原有持久检查点；它不新增“保留取消请求全部新增 suffix”的
保证。`session_capture_decode` 的已排队检查点也会在零重放事务中消费，不构造
空模型输入。

最新候选位于 `artifacts/rolling-active-window-20261017/runtime/candidate-v2`，
由已验证恢复管线加八个文件的最小差异构成，未覆盖工作区其他未验证的调度／
宿主优化。规划、调度隔离、任意 MTP 接受长度、三种取消时点、冷恢复、HTTP
入口保护和验收入口的候选运行时 CPU 检查共 140 项通过。工作区包含启动器的
相关回归另有 239 项通过，两者存在重叠，不相加。

真实模型正确性测试使用 2 个完整 131K 以上的原始历史，缩小为 K32/G4 以稳定
触发实际 decode 边界，然后跨轮恢复。仅此测试启用昂贵的字节 oracle：检查
重选前后所有 recurrent 行和接受计数不变，并用不可变 CPU FP8 历史独立重建
target＋MTP KV；原位保留的行则比较重选前的原生字节快照，避免将正常的
FP8 raw-K 重建差异误报为损坏。还要求覆盖重选前接受多个 MTP token 的情况。
该正确性测试现已通过；开关仍属实验功能，不能据此宣称已获得性能提升。

冻结验收目录为 `artifacts/rolling-active-window-qualification-20261017-r2`，
验收入口另有 24 项 CPU 检查通过。测试把空闲驻留比例设为 `1e-6`，强制每次
跨轮从 CPU 重建并核对 `fp8_immutable_rebuild`；这是正确性配置，保留原始工具
等待，不参与吞吐比较。测试等待前一轮三组对照完成后才独占 GPU 执行，
没有修改其他运行中的代码或配置。

队列交接后，r2 首次在 GPU 启动前因错误导入安装版 `RequestPlan` 而被拒绝。
已记录 `prelaunch-import-correction.json`，显式使用冻结 candidate 的 `runtime/src`
通过所有启动前校验后重新进入加载；冻结运行源码、profile、fixture 均未改动。

r2 已于 2026-10-17 09:24（北京时间）完成：6 个普通请求的完整 CPU 历史均与
M 检查点一致，4 次冷恢复的 target/MTP 与 recurrent 字节校验通过。3 次实际
decode 中途压力重选覆盖两个用户，边界前接受长度为 3、1、5；没有额外 Query
或私有 forward，重选前后 481 个 recurrent 张量及目标 KV 校验通过。实际观察到
1,437 个双路纯 decode batch。客户端与原生接受输出均为 12,290（包含两次清理
请求），零抢占，清理完成。没有做 GPU 取消注入，不能将 CPU 取消测试表述为
GPU 取消验收。本轮证明正确性，性能尚待独立关闭 oracle 的三组对照。

r1 尚未使用 GPU 即被 r2 替代，补入混合字节 oracle 和 HTTP 入口保护。
rolling 模式当前接收 token-ID completions；文本／chat／Responses 直连入口在
进入共享引擎前返回 400，避免缺少规划的请求到压力边界才失败。原有关闭该开关
的入口行为不变。公开 HTTP 保留 G 限制；绕过 HTTP 的内部调用不属于此次验收。

三组恢复管线对照继续使用已冻结的旧配置，不混入 rolling 改动。当前回放固定
输出配额最小 52、最大 8583、均值约 816 token，不能代表真实 agent 申报完整
32K 输出上限时的重选成本。rolling 的性能验收需要额外保留该真实输出上限。

## 自然结束的补充工作负载

`timed_agent_workload.py` 新增可选 `generation_mode=natural`，默认固定长度模式
不变。每次请求声明 `max_tokens=32000`，使用原 DeepSWE bridge 的完整采样参数
和 4096 thinking budget，由模型自然结束；只接受完整 DONE、usage 与返回 token ID
数量一致的响应。窗口结束后正常排空已发请求，不取消来改善吞吐。

仍使用原任务顺序前 8 个任务及各自连续 16 轮外部消息和工具等待。优先选择最早
达到 200Ki 的完整起点；原轨迹未达到该长度或后续不足 16 轮的任务，采用显式
记录的较短历史回退：满足完整性条件的最长输入，长度相同时取最早轮次。
没有替换任务、填充或裁剪输入，不根据任务奖励或速度选样。最终初始长度为
132,334、196,498、210,095、169,028、195,391、204,982、206,330、205,529 token。
完整选样记录位于 `artifacts/rolling-active-window-20261017/deepswe-first8-capacity-16turns.json`。

每条历史仍完整保留；若下一轮完整输入加 32,000 输出上限会超过 256Ki，则在已
完成当前回合后正常结束有限 epoch，再清理并启动新 epoch。不缩短输出上限或
裁剪历史；两轮预热无法完整放下则拒绝进入测量。CPU 预算按 8 个完整 256Ki
历史上界计算，不能再用旧固定输出长度估计上界。

新分析器 `scripts/analyze_rolling_agents.py` 同时统计输出 token/s、完成模型回合/s、
输出长度和截断比例、epoch 替换、按步骤时长加权的实际 decode batch，以及每
100 回合和每 1,000 输出 token 的维护量。请求维护跨度可重叠，不将其相加称为
GPU 空闲。各组自己的自然输出会令后续历史分叉，旧工具结果也未必对应新动作，
所以它依然是外部轨迹性能重放，不评价 on-policy 任务正确率。新增工作负载的
相关 CPU 回归 109 项、选样回归 15 项均通过；这些都不代表 GPU 性能验收已通过。

自然结束三组性能产物为 `artifacts/rolling-active-window-performance-20261017-r2`，
排在正确性验收和原生配置补测之后。启动门禁会重新核对两个运行时各自的原始
target/MTP/GDN 字节证明、完整历史、输出计数及源文件哈希；验收未完成时已实测
拒绝性能启动。此入口及分析器 29 项 CPU 检查通过。

### 首轮自然结束的滚动窗口结果

上述 r2 的 pipeline-base 与 rolling-candidate-v2 均已完成完整 300 秒窗口并正常
排空。独立审计为
`artifacts/rolling-active-window-20261017/natural-rolling-candidate-audit.json`，
复用 `analyze_rolling_agents.py` 的自然输出、实际 batch 和维护量口径，并额外
逐轮重建完整输入、核对事件流与终端历史。两组 profile 仅相差
`kvmem.rolling_active_window=true`；实际 GPU 页池分别为 909、914 页，这一小幅
部署差异也须保留，不能将首轮结果表述为完全隔离的因果实验。

| 正式窗口指标 | pipeline-base | rolling-candidate-v2 |
| --- | ---: | ---: |
| 接受输出 token/s | 158.690 | 186.137 |
| 窗口内完成／发起回合 | 107／114 | 110／118 |
| 窗口内完成回答的平均长度 | 432.542 | 476.473 |
| 实际纯 decode batch，按步骤时长加权 | 5.824 | 6.058 |
| 纯 decode 步骤数量 | 1,892 | 2,291 |
| 实际 8 路纯 decode 步骤 | 86 | 139 |
| prompt 最后一块触发的压力重选 | 68 | 0 |
| 全部压力重选次数 | 78 | 9 |
| 重选上传量，十进制 GB | 51.787 | 7.016 |
| 公开 prefill 完成位置的边界区间 | 643,760–647,937 | 641,643–649,835 |
| NVML 采样平均利用率 | 79.46% | 82.11% |
| 抢占 | 0 | 0 |

输出率提高 **17.3%**，但完成回合率只提高 **2.8%**，窗口内完成回答的平均长度
增加 **10.2%**。增加的 8,234 个窗口输出中，6,130 来自已完成请求的输出差，
2,104 来自窗口结束时尚未完成请求的部分输出差；这只是计数分解，不是优化贡献
分解。124 个共同 agent／epoch／turn 坐标中，只有 11 个完整输入哈希相同，输出
哈希均不同：外部工具消息与采样参数相同，各组自然输出会令后续历史分叉。因此
不能把 17.3% 全称为执行加速，也不能据此推断任务正确率或完成效率提高同样比例。

机制上的直接证据是末块重选从 68 次降为零，重选上传显著减少；公开 prefill
计数区间仍重叠，没有证据表明省掉了大量主模型公开 prefill。纯 decode 步骤区间
合计由 104.21 增至 115.71 秒，时长加权 batch 增加约 4.0%；纯 decode 的 host
维护均值由 12.153 降至 8.376ms。重选 owner 等待累计由 47.243 降至 3.463 秒，
但不同请求及 GPU 工作可能重叠，**不能把差值当作节省的墙钟时间或 GPU idle**。
两组正式恢复分别为 110、114 次，均命中 resident view；实际 MTP 平均接受长度
约为 3.466、3.433，候选的输出增长并非来自更高的该项接受率。

就绪供给的独立审计见
`artifacts/rolling-active-window-20261017/natural-candidate-ready-supply-audit.json`。
续接请求的完整首 token 延迟均值由 2.016 降至 1.139 秒，正式窗口内的续接首
token 等待由 218.99 降至 129.81 agent-seconds；这些延迟包含排队、公开 Query
重放、prefill 和首步生成，不能全归于恢复搬运。resident restore 的 p95 实际
由 0.985 升至 1.037 秒。冷 epoch 首 token 均值为 89.97、92.63 秒，且四个冷
请求的组成不同，没有冷准备加速的证据。mixed step 的时长加权 decode batch
由 5.432 变为 5.405，基本不变。当前证据支持减少尾块重选、让续接更早产出首
token；尚不能把全部输出吞吐增幅解释成就绪并发提高或 KV 恢复加速。

基线 130 条、候选 134 条 warmup／workload 请求均以完整初始 token、完整工具
token 及本组已接受输出重建输入，输入／输出 SHA、usage、流式 token IDs、
Query 规划及完整输入加 32,000 输出上限均通过；终端
`H=M=input+output−1` 全部一致。候选 118 个正式请求全部自然 stop，length
截断为零；正常排空 61.32 秒后，全流程客户端／原生输出均为 65,712，活动请求、
恢复事务及发布阻塞全部清空。基线对应为 114 个自然 stop、排空 80.67 秒、
60,836 输出。此处证明逻辑历史和计数；KV／MTP／GDN 字节正确性沿用先行的精确
运行时资格，不能将本次性能审计表述为再次执行了字节 oracle。

两组均有 4 次因预声明 16 轮上限触发的 epoch 重启，没有 context-capacity 重启，
但重启的 agent 和时刻不同，冷 prefill 与 decode 的重叠也随之变化。本轮是顺序
执行的探索测量；候选正式窗口内还发生了约 0.3 秒的七个自有等待器重建，以及
轻量 CPU 资格准备／审计，影响量未知。待办是保持原 fixture、采样和预算，完成
**安静环境下的反向顺序复测**，同时比较完成回合、回答长度、prefill 工作量、
实际 batch、重选／恢复和 GPU 利用率。

### 更强的自然原生 .90 对照

独立审计 `artifacts/rolling-active-window-20261017/natural-native90-audit.json`
确认 .90 组正常结束，采用同一自然 8 用户 fixture、采样和 300 秒口径；相对
自然原生 .81 仅改显存预算及端口。以下比较不涉及固定 quota 的 12 用户实验。

| 正式窗口／完整发起队列指标 | 原生 .81 | 原生 .90 | K 基线 | K 滚动候选 |
| --- | ---: | ---: | ---: | ---: |
| 接受输出 token/s | 8.377 | 67.590 | 158.690 | 186.137 |
| 窗口完成／发起回合 | 7／15 | 51／59 | 107／114 | 110／118 |
| 发起队列最终回答平均 token | 266.40 | 426.25 | 471.40 | 503.44 |
| 正式窗口 epoch 重启 | 0 | 0 | 4 | 4 |
| 正常排空秒数 | 132.62 | 188.84 | 80.67 | 61.32 |
| 正式窗口抢占 | 1 | 0 | 0 | 0 |

调大原生预算后输出率提高约 8.07 倍，因此只引用 .81 低点会明显夸大 KVMem
的部署收益。相对更强的 .90，K 基线／候选输出率分别为 2.35／2.75 倍，完成
回合率为 2.10／2.16 倍；候选发起队列的平均回答也长 18.1%，且两组 K 有额外
冷 epoch。这是带自然回答分叉的首轮部署比较，不能将全部差距归为 active KV
带来的 batch 扩大，更不能替代安静反序复测或任务质量评估。

原生 .90 的 59 个正式请求均自然 stop、没有 length 截断。包含排空的续接首
token 延迟均值为 21.13 秒、p95 为 153.52 秒；仅统计 T1 前已产出首 token
的请求会遗漏长尾。正式窗口客户端平均有 2.364 路等待首 token、5.028 路已开始
流式输出、0.604 路等待工具。K 候选对应为 1.421、5.691、0.874 路，其中首
token 等待还包含 4 个冷 epoch；其续接首 token 均值为 1.139 秒。这支持续接
等待是值得继续检查的差异，但客户端 streaming 不是 GPU-ready，首 token
延迟也不能全部解释成 KV 恢复搬运。

原生 .90 正式 NVML 利用率为 99.49%，实际页池 1,166 页，全流程采样峰值
91,556 MiB、估计剩余 5,693 MiB。GPU 忙并不等于接受输出效率高；此原生组
没有实际步骤 batch 观测，不能用 running gauge 或 APC 命中推算 decode 人数。
窗口外侧完成计数为 local-compute 388,699、cache-hit 10,305,232 token，后者
占已完成 prompt 来源计数约 96.4%。这些是**已完成请求的 prompt 记账**；T1
仍有长请求未完成，窗口内实际执行的部分 prefill 未知，不能与 K 的公开 forward
完成位置直接比较，也不能将 388,699／300 称为实测 prefill kernel 吞吐。

75 条 warmup／workload 请求均以完整初始输入、完整工具消息和本组接受输出
重建并通过输入／输出哈希、usage 与流式 token IDs 检查。包含 16 条 cleanup
的全流程输出为 34,370，与原生计数一致；最终 running／waiting 均为零，正式
及全流程均无抢占。原生没有 KVMem 的 H／M authority，本次只证明客户端完整
历史链与计数，不能称为 native KV／recurrent 字节 oracle。

## 已完成诊断：Action capture

基线/候选窗口的 Action capture 累计 host span 约 127–140 秒，但该数同时包含
prefill/output 评分、张量构造隐式 CUDA 等待及提交，不能称作 CPU 算力消耗或
GPU idle。多数低 NVML 点发生在 forward 开始之后。当前 vLLM 接入从 raw Q
重做 QK/softmax/页归约；qw3 原版从目标 attention 捕获已计算的 logits 后归约，
两者评分目标一致，执行路径不同。

本次诊断已按 prefill/output、batch/owner drain、元数据上传、每层评分和
D2H 分段计时，并用异步 CUDA event 回收记录，不新增同步或丢弃评分。只增加
默认关闭的诊断能力，尚不改变评分算法。只读依据见
`artifacts/action-capture-readonly-audit-20261017.md`。

完整 300 秒窗口另按预先固定的 30 秒分箱，保留所有区间。两组 KVMem 均在
150–180 秒开始新 epoch 冷准备，随后部分区间的 prefill／mixed step 占满几乎
整个区间。候选最后 30 秒 GPU 平均利用率 95.85%，接受输出仅 56.93 token/s，
prefill／mixed step 耗时 29.96 秒，没有纯 decode step；decode 仍可在 mixed
batch 中执行。这一段属于计算分配与混批干扰，不能称为 GPU idle。按步骤时长
加权的纯 decode batch 为基线 10.376、候选 10.466。所有分箱及口径见
`artifacts/ready-agent-pipeline-20261017/r3-fixed-30s-bins.json`。

Action 诊断实现及冻结准备已完成，默认关闭。精确五文件补丁叠加 candidate-v2
后 47 项 CPU 检查通过，明确排除一项 CUDA 测试；原理、选项及诊断限制见
`docs/action-profiling.md`。`scripts/diagnose_action_capture.py` 复用已冻结的自然结束
8-agent 工作负载，只增加每 8 批采样及最多 4,096 对未回收 CUDA events，保留
评分和已有同步语义。未采样批次仅轮询，不重复写完整统计。入口与离线分析器
相关 26 项 CPU 检查通过，与前述 47 项存在重叠，不能相加。

自然 r2 的 pipeline-base 预热已单独审计，见
`artifacts/rolling-active-window-20261017/pipeline-base-cold-warmup-stage-audit.json`。
10:54:35.183648 正式 T0 前的预热共 450.81 秒，按 NVML 采样加权利用率
80.76%，其中一次连续低于 10% 的区间为 49.54 秒。它完整落在 step158 的
model host 区间内，不能用该步 53.06 秒 CUDA-event model 跨度解释为 GPU
计算。附近 125.30 秒外侧计数区间中 background/bulk owner_query 增加
53.08 秒、main/bulk slot_reclaim/wait/claim 均无增量；因缺少空档内部快照，
这里只将 Query 评分／发布列为具体调查线索，尚未确认因果。该空档不计入正式
300 秒吞吐；冷准备和持续服务的 idle 必须分别统计。

进一步核对扩展构建证据后，已确认该步骤内构建了当前精确冻结目录的 MeanK：
Ninja 编译 50.130 秒、链接 0.355 秒，文件路径、构建日志和时间戳均对应
10:50:44–10:51:35，强支持首次惰性编译主导这次空档。完整证据位于
`artifacts/rolling-active-window-20261017/natural-pipeline-base-first-meank-compile-audit.json`。
没有线程栈，不能声称整个步骤的每毫秒或前台具体阻塞点都已确认。源码的
same-owner drain 可等待后台 Query 发布，且不计入 slot_wait 指标，因此零
slot_wait 不能排除这类阻塞。

工作区新增的独立修复在 serving＋qw3_cuda 的模型加载阶段预加载扩展，编译失败
则阻止服务进入 ready；并使用按源码内容寻址的稳定源码副本，避免相同代码仅因
冻结目录变化导致 Ninja 重编译。该修复未混入正在运行和已排队的冻结性能版本，
其目标是消除首次请求编译与重复构建，不能计作已测稳态吞吐提升。
独立补丁和验证记录位于 `artifacts/mean-k-startup-20261017`，25 项合并 CPU
检查通过。新摘要格式首次需要迁移构建一次；
预加载扩展也不等于预热所有 CUDA 首次 launch／module-load 开销。真实双目录
构建复用和原 MeanK 数值 oracle 已完成独立 GPU 验收；真实服务 ready 前完成
加载仍未验收。
独立 loader／数值资格入口已冻结到
`artifacts/mean-k-startup-qualification-20261017-r1`，21 项入口／证据 CPU 检查
通过，现已正常完成、状态 `passed`。两份绝对路径不同的相同源码在独立进程和
新专用缓存中验证了稳定复用：第二次构建的源码、Ninja、`.o/.so` 哈希及
inode／mtime 未变；原 9 项 CUDA oracle 全部通过、零跳过，证据见该目录的
`results.json`。此入口未启动完整模型服务，不覆盖 ready 时序，也不能把通过
扩张为完整服务性能验收或稳态吞吐提升。

独立诊断目录为 `artifacts/action-capture-natural-diagnostic-20261017-r1`，排在
原生调优与自然结束三组性能窗口之后。启动前检查父资格证明、原始工作负载、
五文件补丁边界和哈希；等待期间不占 GPU，不修改运行中的冻结版本。诊断产生
的吞吐不能作为性能验收。第一次等待器在 GPU 前因相对路径核验失败而退出，
已保存现场并按进程实际 cwd 规范化路径；随后已完成正式窗口及正常排空，状态为
`diagnostic_completed`。
队列重建约 0.79 秒发生在该诊断窗口内，精确操作时间保留在上述队列记录中；
它属于诊断环境事件，不作为 KVMem 算法引起的 idle，也不将诊断吞吐用于性能验收。

最终独立审计见
[action-diagnostic-readonly-audit.json](../artifacts/action-capture-natural-diagnostic-20261017-r1/action-diagnostic-readonly-audit.json)。
原冻结分析器分别输出正式 scope 时基、正式 batch 时基和全生命周期报告；两种
正式时基结果一致。全流程 4,663 批中抽样 583 批，正式窗口包含 375 个抽样批，
12,650 个 CUDA scope 均已回收；事件丢弃、graph-capture 跳过、未映射 scope
和最终 pending 均为零，累计 summary 与原始记录一致。**每 8 批采样不乘 8
外推全量**；嵌套层总计与子阶段不重复求和，也不将其相加称为 GPU idle。

最明显的 host 热点是 `metadata.starts_h2d`，对应
`action_attention.py` 中 `torch.tensor(starts, device=device)`。慢调用几乎全部
集中在每批第一个实际评分的 owner：

| 正式抽样中的 starts 上传 | 调用数 | host 均值 | host 累计 |
| --- | ---: | ---: | ---: |
| output，第一个评分 owner | 132 | 42.04 ms | 5.549 秒 |
| output，同批后续 owner | 36 | 0.160 ms | 0.0058 秒 |
| prefill，第一个评分 owner | 27 | 156.72 ms | 4.232 秒 |
| prefill，同批后续 owner | 13 | 0.122 ms | 0.0016 秒 |

这些调用的线程 CPU 时间接近 host 时间，但全部 output／prefill starts 上传
对应 CUDA-event 总区间分别仅 20.86／6.13 ms；同批后续 owner 的页段数量相近。
因此不能把慢调用直接解释成 Python 列表转换或 PCIe 传输量。它可能由同步
构造／上传承担了前序 target／draft GPU 队列的等待，驱动自旋也会消耗线程
CPU；排在前序工作之后的 CUDA 起点不会计入此前等待。首 owner host 区间的
粗粒度 NVML 平均值约为 output 79.3%、prefill 90.2%，同样不支持把这些跨度
全部算成 GPU 空闲，且无法据此确定具体在执行哪个内核。

下一步优先验证独立的异步 metadata 候选：准备连续 typed／pinned 数据，在
原 stream 上非阻塞上传，并保持 owner 顺序、完整 dense 映射及数据生命周期。
**该候选已实现并通过 CPU 回归，尚无 GPU 正确性或性能验收**；不能承诺能节省上述 host
累计时间。其次是 `publication.batch_drain`，本窗抽样 host／线程 CPU 分别为
1.409／1.367 秒；之后再考虑 sampled prefill logits 等评分内核。三层评分的
正式抽样 CUDA 总区间约为 output 318 ms、prefill 737 ms，均不得外推全量，
也不与嵌套子阶段重复相加。

隔离候选见 `artifacts/action-metadata-upload-20261017/manifest.json`：相对已通过
资格的 rolling candidate-v2 只修改三个生产文件，默认关闭，以
`async_action_metadata=true` 启用。固定 4 MiB pinned CPU 与 4 MiB GPU arena，
最多 128 个 lease，原评分完成事件确认后才能复用；异常 fence／query 失败隔离，
额度不足回退原路径。工作区和隔离 runtime 各 58 项 CPU 检查通过；已有 Action
回归 57 项通过、1 项 CUDA 检查未运行。4 项 CUDA oracle 和真实模型跨轮资格
已冻结到 `artifacts/action-metadata-upload-qualification-20261017-r1`，尚未启动或
排入 GPU 队列。资格保持原双用户、完整 Action 历史、target／MTP／recurrent
恢复字节校验及正常 H=M；不包含真实服务取消注入，也不作为吞吐测量。

本次正常排空 46.79 秒，146 条普通请求的完整输入哈希、流式输出及终端
`H=M=input+output−1` 均通过；全流程客户端／原生输出均为 75,122，零抢占。
历史始终使用 CPU、无磁盘读写，最终请求、恢复、发布阻塞及下载均清空；正常
cleanup 保留 8 个单 token 小视图。这些是诊断完整性检查，不替代新的 KV／MTP／
GDN 字节 oracle，也不构成未加诊断版本的吞吐验收。

## 待验证：提前释放下载缓冲区

原生异步调度兼容性已作只读审查，见
`artifacts/ready-agent-pipeline-20261017/async-scheduling-compatibility-audit.json`。
当前 KVMem 的启动器、scheduler、draft contract、容量管理和捕获 runner 均要求
同步单 batch；下一次调度被用于确认上批已接受前缀，不能只删除关闭 async 的
开关。适配至少需要按 batch／owner 版本确认、在途捕获与状态页租约、精确已接受
recurrent 检查点，以及恢复／Query／rolling／abort 的单请求静止边界。原理上可在
KVMem 适配层完成，尚未实现或证明收益，也未发现必须修改主模型验证内核。

r3 候选正式纯 decode 的维护 CUDA-event 跨度合计 17.85 秒，均值 7.84ms、
中位数 1.52ms、p95 42.90ms；前台 host 维护合计 18.40 秒，均值 8.09ms。
43ms 不能视为每步固定成本。其余 model 区间合计 129.96 秒仍包含采样、MTP、
KVMem capture 和 Action；CUDA-event 跨度也不是纯 GPU 活跃时间。恢复完成到
scheduler ready 的 407ms p95 可能包含 GPU 正在执行长 prefill 的等待，不能都当作
idle。先完成 Action／early-release 实测，再决定异步调度协议改造的优先级。

精确热恢复的 lifetime 容量信用也已审计，见
`artifacts/exact-restore-lifetime-credit-audit-20261017.json`。当前准入在完整历史超过
96Ki 时按整个 active＋generation 上界预留 76 页；在 r3 的 118 次热恢复中，
若使用已验证的 active 长度、完整未计算后缀和原始短输出配额，理论平均可少留
13.99 页。但保持目标自然 agent 的 `max_tokens=32000` 后，同样 118 次全部仍需
76 页，释放信用为零。因此暂不实现这项优化，不能用固定短输出结果代替目标
工作负载的收益。若后续实现，须冻结首次准入上界并保留 Query、guided probe、
MTP 尾步和取消期间的信用，不能根据历史 EOS 或后续 offset 预测性缩减。

另一个明确的代码依赖是 `_publish_claim` 将 CPU 历史拷贝、Query history 评分、
Action 提交与 DMA slot 释放绑定。源 KV 已同步复制到独立 authority 存储后，
后续 Query clone 和 Action mass 使用各自的对象，但 slot 仍要等评分完成；槽满
回收还可能让主线程执行无关请求的评分。r3 候选全程 main/bulk slot reclaim 和
owner Query 累计均约 59 秒，但没有正式窗口的该项计数快照，窗口损失仍为未知。

下一独立候选将区分“源数据已复制”和“owner 语义已发布”：仅前者允许回收
共享缓冲区，后者才允许该 owner ready、恢复或发布 M 检查点。所有共享 owner
复制完后才能释放一次，后续评分不得再读取可复用 DMA payload；保留按 owner
FIFO、取消及错误传播。脱离 DMA slot 的评分对象须同时受 owner 数、packet 数
和 tensor 字节额度约束，不能借此无限增加 GPU Query clone。该候选的精确隔离
运行时已通过 63 项 CPU 回归及独立源码审查，默认关闭，额外信用为 256MiB；
尚未取得 GPU 正确性或吞吐收益证据。产物位于
`artifacts/early-capture-feedback-20261017`，未混入 Action 诊断等其他候选。

CPU 审查发现若将全部已复制但未提交的 owner 一律暂停，会把小 decode 反馈
也排到唯一后台 worker 的大 Query 后面。候选已收窄为仅 `bulk` DMA lane 提前
释放；decode lane 保持原有按请求 fence 和前台小量提交路径。必须通过事件的
物理 slot／generation 证明真实模型中发生了“旧包语义未提交时，新包已复用同槽”，
且旧包包含多个 owner；仅开启 flag 或计数 source copy 不算覆盖。

GPU 验收按两段准备：先复用双用户字节 oracle，验证跨轮恢复及 decode 重选，
再用原始自然结束 first8 样本做两个完整回合，检查真实共享槽复用、逐请求输出
和终端 M。两段均保留所有原始历史／Query／工具消息，不注入人为慢评分；第二段
必须等第一段完成且 hash／字节证明通过后才能启动。未发生真实复用应报告
`coverage_insufficient`，不会自动扩大信用或缩短 Query 来制造通过。

双用户字节验收目录 `artifacts/early-capture-byte-qualification-20261017-r1` 已冻结，
测试入口和样本与已通过的 rolling r2 完全相同，仅增加 early 开关、256 MiB 信用
和独立端口。当前等待器已停止并标记 `deferred_pending_priority`，未启动 GPU。
如恢复此实验，仍须核验上游资格和原生安装指纹。它只能证明字节／完整历史／
终端 M／冷恢复／MTP，不能替代下一段
共享槽真实复用覆盖。自然 first8 中部分请求若首轮达到 32,000 输出，第二轮完整
输入加同样输出上限会超过 256K；这种情况必须正常排空并标记覆盖不足，不能靠
缩短输出上限、裁历史或换题来取得通过。

共享槽探针 `artifacts/early-capture-reuse-qualification-20261017-r1` 也已冻结，
当前延后、等待器已停止；新增入口／监控相关 37 项 CPU 检查通过。启动前及等待 GPU 空闲后都核验
完整原始输入、配置派生关系、精确运行时和已完成字节资格。正常收尾还须验证
脱离槽位的 owner／packet／bytes 信用全部归零。前置字节资格未完成时，已确认
入口拒绝启动。覆盖不足不能放行早释放性能验收；独立原生对照已前移，不依赖此资格。

提前释放的自然结束性能入口已冻结为
`artifacts/early-capture-performance-20261017-r3`，相关 30 项 CPU 检查通过。
它当前延后、等待器已停止；恢复时必须通过共享复用资格，沿用自然 first8 的原始输入、采样、工具等待、
2 轮预热和 300 秒窗口，只增加 early release、256 MiB 信用及端口。启动前和
等待 GPU 空闲后均重新审计字节资格、完整历史、共享物理槽复用和正常清理；
任何 `coverage_insufficient` 都禁止该性能运行。当前只完成 CPU／冻结复验，
尚无 GPU 性能结论。

该入口自动保存 `formal-capture-feedback.json`：按新包实际提交时刻区分预热、
正式窗口及排空；正式 agent 复用覆盖要求旧、新包均为工作负载请求，清理、
混合和无法归属的包另列。T0／T1 邻近计数保留边界偏差和内外区间，缺失为
null；CPU 等待时长不能直接当作 GPU idle 或节省时长。若只在预热发生复用，
不能声称正式吞吐来自这个机制。r1／r2 保留为未用 GPU 的准备草稿。

## 已完成但未通过余量门槛的部署容量补测

`artifacts/ready-agent-kvmem12-deployment-20261017-r1` 只把原 r3 KVMem 候选的
GPU fraction 从 .81 改为 .82（另换端口）；模型、实际 batch 16,444、同 12 用户
固定样本、300 秒窗口和已验证运行时均保持原样。该组已正常完成，实际池从
912 增至 930 页，观察到持续的 12 路 decode；但未达到预声明显存余量，
**不作为默认部署，也不自动调参重试**。独立审计见
[kvmem12-audit.json](../artifacts/ready-agent-kvmem12-deployment-20261017-r1/kvmem12-audit.json)，
状态为 `completed_contract_failed_headroom`。

全流程 r3 采样显存峰值为 89.77 GiB，.82 的线性估计峰值约 90.72 GiB；预声明
要求全部阶段观测剩余至少 4 GiB、零 OOM／抢占、正常排空及输出／历史核对。
本次全流程 NVML 峰值却达到 96,646 MiB：物理总量减 used 仅余 1,241 MiB，
再扣设备 reserved 后只余 603 MiB。加载／初始 cleanup 阶段也观察到 96,460 MiB，
正式窗口与排空峰值则为 90,558 MiB。**不能用正式窗口较低的峰值替代全部阶段
门槛**。未发生 OOM 或抢占不等于有足够余量。

| 同 fixed-r3 的正式窗口指标 | 原候选 .81 | 容量补测 .82 |
| --- | ---: | ---: |
| 接受输出 token/s | 333.077 | 313.460 |
| 窗口完成／发起回合 | 115／124 | 107／118 |
| 纯 decode 时长加权 batch | 10.466 | 11.084 |
| 12 路纯 decode 步骤 | 0 | 961／1,969 |
| 纯 decode 步骤区间秒数 | 147.816 | 132.810 |
| mixed 步骤区间秒数 | 142.537 | 157.861 |
| 纯 decode 平均 step 毫秒 | 64.946 | 67.451 |
| mixed 平均 step 毫秒 | 404.241 | 423.814 |
| 公开 prefill 完成位置边界区间 | 514,012–520,156 | 556,555–561,601 |
| MTP 平均接受长度，外侧边界计数 | 3.673 | 3.703 |

表中 step 时长来自 CUDA-event 跨度，可能包含提交间隙，不是精确物理 GPU
busy 时间；边界按原口径用 host 起点裁剪，不能将不同区间重叠部分重复相加。
12 路共覆盖约 59.17 秒步骤区间，占纯 decode 时长 44.55%、正式 300 秒的
19.72%；这是实际观测的 batch 宽度，缺少逐步请求身份，且 CUDA-event 时长以
host 起点裁剪，只能作为近似步骤时间占比。容量扩大确实增加了 decode 宽度，
但整窗输出率下降 5.89%、完成回合数下降 6.96%。本轮 mixed／prefill 占时更多，
步骤更慢，draft 次数由 27,229 降至 25,415；接受长度略升，因此不能解释成
MTP 接受率下降。固定 quota 未变，但窗口内推进的回合与冷 epoch 组成仍不同，
单次结果不能将所有变化归于显存参数。
后续[来源审计](../artifacts/ready-agent-kvmem12-deployment-20261017-r1/prefill-source-audit.json)
已完成计数闭合：两组正式窗口都启动相同 4 个冷 epoch，完整初始输入均为
527,770 token，但在 T1 前完成长输入 prefill 的请求由 2 个增至 3 个。
公开 prefill 完成位置增加的保守区间为 36,399–47,589；其中冷请求推进增加
36,976–48,438，工具续接及 cleanup 反而减少 577–849。所有续接恢复位置均
等于上一轮完整输入＋接受输出−1；全流程 public prefill 与冷输入、工具新增、
上一轮末 token 及 cleanup 精确闭合，.82 整场反而少算 626 个位置。
压力重选由 5 次增至 8 次，均发生于较早的 prompt 部分；没有显式 Query／
末块重选或私有 forward。记录的 32-token replay 是首次 forward 前延后的
公开块，不能另算成重复计算。现有证据不要求修复历史复用；也不能为提高
窗口输出 token/s 而人为压制冷请求。应保持完整工作负载，同时比较冷请求
进展、回合完成率和输出率；逐请求精确 prefill 时序仍受聚合计数边界限制。

最高显存采样位于冷 prefill 的 step129：4 路共 6,176 token，其中普通 prefill
6,144、atomic 32，decode 为零；前后步骤分别为 8 路／16,384 token 和
9 路／16,416 token。附近采样没有恢复 permit，首次 resident 恢复约在峰值
133 秒后。它不证明“同一批 12 路大 Query”导致峰值。池新增常驻约 966 MiB，
只解释两次峰值差 4,718 MiB 的一部分；没有 allocator 跟踪，尚不能在 raw-Q、
Action scratch、raw-K staging、原生 activation／workspace、缓存及重叠分配
之间精确归因，且原 .81 峰值处于不同阶段。

正常排空 45.76 秒后，142 条普通请求的完整输入哈希、usage 和终端 H／M 均
通过；包含 cleanup 的 170 条请求全部完成，全流程客户端／原生输出均为
121,686。CPU-only、无磁盘读写、GC 关闭，最终请求、下载、恢复及发布阻塞
清空。容量观测与完整性检查有效，但不能覆盖已失败的显存余量条件。
NVML 可漏掉短峰值，且原 r3 最大 Query 为 4,566 tokens，没有执行私有 Query，
因此该补测不证明任意 16K Query 或所有用户满 256Ki 的安全容量。KVMem 直接
采用 .90 的粗估峰值会超过物理容量，未安排该配置。范围和验收条件位于该目录
`deployment-acceptance.json`；这些为后验检查，并非新增运行时限额。

## 反序复测的两臂阶段结果与后续资格顺序

本节更新前文的队列状态，保留此前各轮数据。反序复测的 rolling candidate-v2
和 pipeline-base 已完成，原生 .90 臂仍在运行；当前不能给出这轮完整三方比较。
依据为 [pair-interim-review.json](../artifacts/natural-rolling-reverse-20261017-r1/pair-interim-review.json)。
两个 KVMem 臂沿用原始自然 first8 样本、自己的实际输出历史、工具消息与等待、
两轮完整预热、300 秒正式窗口及 32,000 输出上限，没有混入 MeanK 启动修复、
异步 Action 上传或其他新候选。

| 反序复测正式窗口 | pipeline-base | rolling candidate-v2 |
| --- | ---: | ---: |
| 接受输出 token/s | 164.3367 | 180.1767 |
| 窗口内完成回合 | 110 | 114 |
| NVML 时间加权利用率 | 78.60% | 84.58% |
| NVML 低于 10% 的观测秒数 | 11.07 | 4.41 |
| 纯 decode 步骤时长加权实际 batch | 5.9787 | 5.9129 |
| prompt 最后块触发的重选 | 69 | 0 |
| 重选上传量，十进制 GB | 51.88 | 7.76 |
| 窗口内完成答案的平均输出长度 | 431.48 | 451.15 |
| 到达预声明 16 轮上限的 epoch 结束次数 | 5 | 4 |

输出率提高 **9.64%**，完成回合率提高 **3.64%**；末块重选和上传明显减少，
支持减少维护开销的方向。实际 decode batch 没有增大，因此这轮**尚未证明
“缩小 active KV → 更多就绪请求 → 持续扩大 decode batch”**的容量收益链。
窗口内完成答案平均长 4.56%，冷 epoch 重启也少一次，都会影响输出率和 prefill
组成；不能把全部 9.64% 归于单个维护改动。上述步骤加权使用 CUDA-event 跨度
及近似 host 边界，非精确物理 GPU 占用；NVML 低利用率也不是已证明有就绪请求
时的调度空闲。owner 等待可以重叠，不能累加后称为节省的 GPU 时间。

反序 candidate 的正式窗口早期发生过 4 项队列 mock 检查，进程用时 0.133 秒、
pytest 报告 0.07 秒，精确起止时刻未保留。此干扰记录见
[measurement-interference-note.json](../artifacts/natural-rolling-reverse-20261017-r1/measurement-interference-note.json)，
不能称为完全没有额外 CPU 活动的窗口，也不能据此推算影响的吞吐量。
后续小型检查与准备均安排在已记录的正式窗口之外。

当前已启动的等待器依赖为：反序原生 .90 正常排空 → 独立自然原生 .915 →
异步 Action metadata 资格 → allocator 冷 12 用户诊断。后三项仍为 prepared，
等待器不运行模型；此状态取自各目录的 `queue-state.json`。metadata 资格位于
`artifacts/action-metadata-upload-qualification-20261017-r1`，必须通过以下实际
契约才能放行后续候选性能测试：

- 新 kind 为 `action_metadata_upload_qualification_v1`，精确三文件候选及原生安装
  指纹再次匹配；4 项 CUDA 用例全部通过且无跳过，不能借用旧 rolling 资格。
- 双用户保留原始完整长历史，依次输出 1024／4096／1024，再各做一次 EOS 清理。
  使用 compiled MTP4、FP8、CPU 历史、K32Ki／G4Ki；极小 resident 份额强制冷恢复。
  六条普通请求须有 Action／prefill 评分、完整 H=M、target／MTP／recurrent 恢复
  字节 oracle、实际零 replay 压力重选及边界前 MTP 接受多 token 的证据。
- 异步上传实际发生，无 fallback、poison／fence／query 失败；原生与客户端输出
  计数一致、零抢占、正常清理。仅低层用例覆盖忙 lease／异常，不宣称真实服务
  取消注入已通过。启动前与等待 GPU 空闲后均重新验证。

allocator 诊断位于 `artifacts/allocator-cold12-diagnostic-20261017-r1`，只在上述
资格真正完成并清理后开始。它复现原 .82 的 12 用户完整冷准备和两轮预热，
正常排空后结束，不运行 300 秒吞吐窗口。预期 24 条预热请求及 24 条清理请求
全部对账；`.82` 已知余量不足，只记录 allocator／NVML 高水位和阶段覆盖，不
修改预算、不自动重试，也不把负余量诊断当成部署验收。允许如实报告
`diagnostic_completed_with_gaps`，但必须保留缺失范围，不能伪造全进程末尾覆盖。

原生实际执行 batch 的 observer r2 已通过 **16 项 CPU fake 检查**和独立只读
审查，尚未做 GPU 验收；见 `artifacts/native-executed-batch-observer-source-r2-20261017.json`。
它只保留真实 InputBatch 的 CPU 快照，在原 AsyncOutput 成功完成后记账，不把
running gauge 或 draft 上限当作接受输出。所有 EOS cleanup 必须在 seal 之前；
seal 后再次分发模型请求会使覆盖失效，最终 summary 仍须复核。该诊断不会混入
未加观测的性能对照。

单候选性能入口 `scripts/benchmark_action_metadata_upload.py` 已通过 **23 项 CPU
mock 检查**，用时 0.18 秒；实际进程 wall 1792212526.270–1792212526.660，位于
base 的 T1 之后。它目前**未 prepare、未排队、未启动 GPU**。最小后续操作为：

1. 等 metadata 资格的 `results.json` 真正 passed，且队列／状态均为
   `completed_pending_review`；入口会重新计算四项 CUDA、真实模型历史／恢复／
   Action 和正常清理证据，缺一项即拒绝 prepare。
2. 仅准备一个独立目录，例如 `artifacts/action-metadata-upload-performance-20261017-r1`，
   用原 `artifacts/rolling-active-window-performance-20261017-r2` 为 reference，
   上述 metadata 资格为 qualification，调用该入口的 `--prepare-only`，端口 18570。
   沿用同自然 8 用户、两轮预热、300 秒、32,000 上限、原始采样及 CPU 历史预算；
   profile 只增加 `async_action_metadata=true` 和独立端口，运行时只叠加已验三文件。
3. 完成冻结 verify，等现有 allocator 诊断正常排空并确认无 `cleanup-required`，
   再由调度者启动新目录 `frozen/scripts/benchmark_action_metadata_upload.py` 的
   `--run-prepared`。它使用共享 GPU 锁，空闲等待后再次核验，保留正常 drain，
   不新增参数矩阵。性能比较同时看输出率、完成回合、答案长度、epoch／prefill
   组成和实际 batch。无时间戳的终端上传计数只作为全流程事实，不能冒充正式
   窗口 delta 或 GPU idle 的因果证据。

## 完整反序结果与用户恢复后的执行点

反序三臂均已正常完成。最终记录是
`artifacts/natural-rolling-reverse-20261017-r1/comparison.json` 和各臂 `results.json`，
不是上文两臂中间报告；旧分析中的 running 状态是分析生成时的快照。

| 自然 8 用户、300 秒 | 原生 .90 | K pipeline-base | K rolling-v2 |
| --- | ---: | ---: | ---: |
| 接受输出 token/s | 100.957 | 164.337 | 180.177 |
| 窗口内完成回合 | 56 | 110 | 114 |
| 全流程客户端／原生输出 | 44,051 | 59,728 | 67,081 |
| 抢占 | 0 | 0 | 0 |

K 候选相对本轮原生输出率提高 78.47%，完成回合数为 2.036 倍。原生前一轮为
67.590 token/s，说明单个自然窗口有较大波动；不能用早先分母宣称稳定倍率。
两边使用自己的真实接受输出，长度和后续回合也会变化。本组原生未记录实际
InputBatch，不能把原生 running gauge 或缺失的步骤日志当成实际 decode batch。

用户暂停时已停止全部自有等待器及模型进程。自然原生 `.915` 的 r1 刚进入加载，
尚未进入正式窗口，被正常取消；保留 `CancelledError` 和 `user-pause-20261017.json`，
不将其当成性能失败或在原目录覆盖。随后用户明确选择恢复优化与测试、GPU 空闲后运行。

恢复入口为 `artifacts/resumed-validation-20261017-r1/run.py`，只执行一次以下顺序：
自然原生 `.915` r2 → Action metadata 资格 r1 → native batch observer 资格 r1。
每个冻结入口自行等待 GPU 空闲、持共享 GPU 锁、检查外部进程并清理自身；失败停止
顺序，不自动重试。原生 r2 与被用户中断的 r1 manifest 仅准备时间不同，输入、
参数、安装身份和代码完全相同。allocator 冷 12 用户诊断后移，旧等待器均不恢复。

新增独立容量协议入口 `scripts/benchmark_capacity_chain.py` 已完成 CPU 验证及
独立审核，仍需实际 observer GPU 资格完成才能 prepare。该协议保留固定八条
长历史源轨迹、K64/G32/ctx256 和最坏 CPU 预算，单次实际输出上限改为 16384，
16 轮只是独立 epoch 的最多轮数；依据各臂完整输出按既有上下文上限正常结束。
它不能与原 natural32000 混为同一协议。原严格 16 轮资格的超限负结果仍保留。
原生 APC 保持开启，但按用户／epoch 隔离；输入共有 18,852 个前缀 token，
1205 页容量下界仅在该隔离条件下成立，实际池仍须逐臂实测。
正式 batch 证据排除 warmup、cleanup 和 prefill；native 使用真实请求身份，
K 旧日志仅有数量，必须使用排他客户端区间作保守归属并明确其限制。
任一臂 4 GiB 余量／抢占条件失败会保留原始结果并令部署比较资格为 false。

## 恢复后的原生对照与 Action 验收发现

自然原生 `.915` r2 已正常完成：300 秒接受输出 26,286 token，即 87.620 token/s；
发起 67 回合、窗口内完成 59 回合，正常排空 77.97 秒，全流程客户端／原生输出
45,480 token 完全一致。零抢占，NVML 平均 99.06%，全阶段最低采样显存余量
4245 MiB。实际原生池 1193 页，原 `.90` 反序为 1166 页；这些是配置池总数，
不是实际 decode batch 或排除 null 后的可用页数。

三臂轻量核验见 `artifacts/natural-native915-comparison-20261017.json`：共 353 个
完整 own-ID 输入链通过。`.915` 三次 prefill 信用增加恰等于完整输入
217,830／216,588／210,244 token，cache-hit 增量为零，而新增工具内容仅
48／343／891 token，对应 TTFT 86.80／146.38／141.75 秒；其中两次记账发生于
T1 之后。原 `.90` 也有五次类似现象。这是续接完整历史重新计算的强关联证据，
尚未定位缓存失效的内部机制，也不能将计数间隔直接当作 GPU kernel 时间。
`.915` 完成回合多于 `.90`，但答案平均更短，不能凭输出 token/s 单独判断参数优劣。
当前完成的最高原生自然输出率仍是 `.90` 的 100.957 token/s；K 的 180.177
token/s 相对它为 1.785 倍。容量到实际 batch 扩大的因果链仍待新协议验证。

Action metadata r1 的四项 CUDA 用例全部通过；双用户六轮生成、两次清理、
全部 H=M、target/MTP/recurrent 字节 oracle、12,290 个原生／客户端输出对账和
零抢占均通过。但新评分门禁失败，外层 `status.json` 为 failed，旧内层
`results.json` 的 passed **不能作为该候选资格通过**。顺序控制器因此停止，
所有自有模型进程已退出，未自动重试。

失败轮 `rolling-1-2-0` 的完整 prompt 以 thinking 状态结束，1024 个 own 输出中
没有关闭 thinking 的 token；原 frozen C++ ActionParser 对 `output_ids[:-1]`
精确复核得到两用户样本数 [19,124,23]／[29,144,0]，与实际逐轮相同。该轮历史
评分及有效 rounds 保持上一轮，prefill 评分仍新增 5 个 query 样本。因此当前
问题是验收将“每轮都有输出评分”误当作 non-thinking 语义。将修正为按原 parser
精确期望计数和有效评分轮数核验；保留全部 async 上传、prefill、恢复和历史
条件，保留 r1 失败现场，不直接将其改为通过。

原生 batch observer 与 Action 候选无依赖，确认 GPU 空闲后已单独启动其既有
冻结资格。实际启动记录为该资格目录的 `manual-resume-20261017.json`。容量
实验继续使用原 rolling-v2，不包含尚未通过最终门禁的 Action 上传候选。

原生 observer r1 最终通过：296 个完成步骤、251 个纯 batch2 decode 步骤、
8 个混合 prefill/decode 步骤，观察到两个在途 AsyncOutput；无乱序完成实例，
不宣称该可选分支已覆盖。全部 8 个请求完整身份关联，1794 个原生／客户端接受
token 对账，seal 后零请求、零丢失／错误。加载阶段 FlashInfer autotune 曾因
临时分配 OOM 回退默认 tactic，但实际请求和观测均通过；本项不提供吞吐或部署
显存余量结论。

`artifacts/capacity-chain-g16-performance-20261017-r1` 已完成冻结、实际 GPU 身份
核对后启动。manifest SHA256 为
`c13c3d0990cface73a890246ce0abd0c390af6822e52822e994474a3a51c5f55`。
GPU 已清空，启动前 MemAvailable 为 114,412,929,024 B，高于完整八用户最坏
预启动要求 113,834,459,136 B；此预算包括每会话额外两个 archive records，
共 17 MiB，并非仅按当前已生成长度估算。正式窗口内停止额外 CPU 测试和大规模
指纹检查。原生臂带 observer，吞吐须标注诊断插桩，不能直接冒充旧未插桩基线。

Action 评分门禁 v2 已完成 82 项 CPU 检查，并已将新版资格冻结至
`artifacts/action-metadata-upload-qualification-20261017-r2`，manifest SHA256 为
`b6bed33f2188af3020b595e502e60284eba2161d310e75351acbc130b47aac68`。
它冻结 tokenizer 作用域 token 身份，并按原 `_native.so` parser 精确校验
`output_ids[:-1]` 的采样数、有效 rounds、零样本时保留历史分数以及每轮 prefill
累计增量。r1 的独立复核支持该门禁修正，但 r1 仍是 failed；r2 尚未运行 GPU。

G16 容量对照原生臂完成两轮预热后，正式窗口为 wall
1792217563.826887–1792217863.826887。预热期间已观测到同一请求
`cmpl-timed-native-1-1-warmup-0-a24e92e4` 的 computed prefill 在执行134为210080，
到执行168重新为0；稍后公共抢占计数由0增至1。说明公共计数存在发布滞后，
中间快照的零不能代替排空后的全流程零抢占证明。此次结果须保留全阶段负部署
判定及原始性能，不能将容量压力预填充重算藏在预热统计中。

已补充单一原生准入对照选项 `--native-max-num-seqs 7`：客户端仍为8，K仍为8；
实际原生 launch 仅 max_num_seqs 从8改为7，保持16424 token预算、MTP4、APC、
async和采样。由于旧 launcher 按 Query allowance加槽数推导预算，纯原生profile
显式置 query_replay_budget=null 并固定 max_num_batched_tokens=16424，避免无意
缩小prefill预算；完整profile变化记录在manifest。33项CPU检查通过，见
`artifacts/capacity-chain-native7-cpu-validation-20261017.json`，测试结束
wall1792217504.674，早于正式窗口。新对照尚未prepare或运行，当前r1冻结代码及
8槽位参数未变。先审阅完整r1，再评估7槽位能否构成更强原生基线。

## 独立容量臂恢复与权重精度解耦

G16 原生臂已正常收尾：300 秒接受 2691 token，即 8.97 token/s，完成 7 个
回合，全流程 2 次抢占，采样显存余量 4245 MiB。实际正式窗口没有纯 decode
步骤，长历史 decode 请求主要与 prefill 混批；因此不能用 running=8 宣称
持续 8 路 decode。原生吞吐保留，但零抢占部署验收不通过。

原两臂控制器在启动 KVMem 前因 `Insufficient host headroom` 失败，未运行 K。
进程退出后可用内存重新高于门槛。独立诊断观察到控制器自身的临时 Python/libc
堆可回收，但未记录失败瞬间内存，不能断言这是唯一原因。

`benchmark_capacity_chain.py` 增加独立臂选择和有界内存等待：完整门槛仍为
113,834,459,136 B，只回收控制器自身闲置堆，连续三次满足门槛才继续，原生
启动前再次检查。超时或取消保持失败，不降低预算、不清系统缓存、不动其他进程。
单臂结果明确 `comparison_complete=false`，不会伪造已完成的成对对照。
40 项 CPU 检查通过。新目录 `artifacts/capacity-chain-g16-kvmem-performance-20261017-r1`
已开始独立 K 臂，manifest 为
`e62b122c14b6fdb19652e4c964161af13d254b81df7309206e036e7b1131a18f`；
原生结果仅引用原目录，不复制或覆盖。native7 已准备但尚未运行。

用户要求评估 Q8/FP8 并指出权重类型应由 vLLM 原生支持。已移除启动器历史遗留
的 NVFP4 白名单，而非增加逐精度分支：只检查本地 HF config 可读且为对象，
量化格式、算子与硬件兼容性由 vLLM 校验。模型配置不被修改，也不强制覆盖
`--quantization`。两个启动入口同步使用该规则；191 项 CPU 启动/准入检查通过。
这不是所有模型架构或 KV 类型已通过验收的声明。当前原生 KV 适配器仍针对
FP8 e4m3 的布局与缩放，Qwen recurrent/MTP 状态也有独立的架构适配。

本地 FP8 模型完整，66 个索引分片约 30.87 GB；现用 NVFP4 三分片约 21.92 GB。
两者 text_config 与 tokenizer 相同，都有 MTP 权重。模型文件大小不是实测 GPU
权重占用。新候选仅在已验证 rolling-v2 上改变两个启动文件，KVMem 算法不变；
精度清单及配置预览保存在 `artifacts/weight-precision-20261017`。先做双用户跨轮
target/MTP/recurrent 恢复验收，再做相同样本、采样、工具等待、8 用户固定窗口
吞吐比较。保持 FP8 KV 与 compiled MTP4；尚无 FP8 实测性能结果。

用户随后明确优先继续 4Bit 优化，FP8 资格仅保留 prepared，不排入 GPU 队列。
当前 NVFP4 实际加载为 ModelOpt mixed，启动日志选择
`FlashInferCutlassNvFp4LinearKernel`，FP8 投影选择
`FlashInferFP8ScaledMMLinearKernel`，compiled graph 已捕获；目前没有整个目标
模型退回 BF16 GEMM 的证据。FlashInfer attention 不支持 fused multi-step draft
decode 的提示对应 MTP 元数据路径，不能误报成 NVFP4 算子失效。

新增最终任务验收要求：优化通过后，重跑原 16 个真实 DeepSWE 任务，沿用最近
`deepswe-capacity-20261017-r3/kvmem-c8` 的 8 客户端、原样本/镜像/工具协议、
采样、seed1000、单次尝试和 Read/Bash 重复终止规则；这次跑完整任务结局，
不在30分钟窗口到点截断。详细待执行计划为
`artifacts/nvfp4-optimization-followup-20261017/deepswe-first16-plan.json`。
用户允许为 CPU 容量降低 ctx，暂定 192Ki 历史、K64Ki/G32Ki，不截断已有历史，
触及上限单列 context_limit。8 用户满额 archive 约54.78GB，比256Ki省约18.25GB，
为真实工具容器留出余量；仍须在最终运行前核对当时内存及工具占用。最终候选
和任务测试尚未启动，不能把容量重放成绩当作新的 DeepSWE 成功率。

## 完整 first16 实测前的补齐（2026-10-17）

用户要求优化验收后再跑原来的 16 个真实 DeepSWE 任务，保持最近 8-agent 的
模型、采样、工具、镜像、seed 和重复行为停止策略。计划历史上限 192Ki、
active64Ki + generation32Ki，CPU-only、无重算，原隐藏 verifier，每题新的一次
尝试，duration=0 跑到终态。192Ki 对应 8 条满额 CPU KV 约54.78GB，比256Ki
少18.25GB；工具容器仍有各自16g上限，不能把 KV 估算当作全部进程的内存保证。
最终启动还要检查真实可用内存，并持续保留运行余量。

`benchmark_deepswe_first16.py` 原来只有限时模式采集完整指标、检查8GiB主机
余量和读取context_limit。现将这些检查用于完整任务模式，每秒采样；即使指标
HTTP或NVML采集失败，也先检查主机余量和上下文事件。CLI先退出的竞态也会
回读已发布的context_limit，将该题标成未评分的context_ceiling并补位下一题，
不会伪装成基础设施失败或成功解题。上下文内容不截断，普通API错误仍会停止新
任务准入并保留现场。生命周期、日志半行、指标失败、低内存及重复行为回归
13项通过。当前正式 GPU 对照使用原冻结代码，未被这些测试入口修改影响。

独立K臂的 `completed-kvmem-review.json` 已核对输出口径：窗口流总数50165中
包含4个EOS清理token，正式工作输出50161，即167.203/s。早期临时统计包含
清理token，保留原文件供核对。正式窗口GPU时间加权NVML利用率84.321%，
纯decode事件加权batch5.861；这两项均不能当作SM占用率或连续8路驻留。

独立原生7槽位G16对照完成：2706个窗口输出、9.020/s，完成8回合，零抢占，
正常排空183.28秒，客户端/原生输出一致。实际worker观测覆盖通过；正式窗
201步中198步为prefill/decode混合，无纯decode，GPU利用率99.385%。这表明
降低1个active槽位消除了抢占，却没有消除重复长历史prefill的干扰。其启动
FlashInfer autotuning曾OOM并恢复默认tactic；全流程最小显存余量1513MiB，
未满足预声明4096MiB，部署验收为负。原始数据保留，不以167.203/9.020作为
最优原生或稳健部署加速倍数；相同16424 batch总token预算也不代表原生已做
小chunk延迟调优。当前转入Action上传r2真实模型资格验收。

Action上传r2完成4个CUDA用例、6轮模型请求及2个清理请求，总输出12290，
target/MTP/recurrent恢复字节、H=M、原生计数与清理均通过；最后host-only评分
核验因同进程再次别名加载C++扩展，出现ActionParser重复注册。原始status保持
failed、原始模型证据保持不变。修复仅复用已加载的canonical解析器，并严格比较
其二进制哈希；离线重新核对完整own-output轨迹的Action/prefill评分，全部一致。

独立离线证据为`action-metadata-upload-qualification-20261017-r2-offline-review`。
性能入口只允许显式引用这一类精确导入故障的离线复核，仍重新检查4项CUDA、
全部流量/用户历史/冷恢复oracle/清理/评分，不允许一般失败结果放行；修改后的
验证器随性能入口一起冻结。91项CPU回归通过，含重复加载、错误二进制、其他
失败原因、篡改评分/模型证据/清理未完的拒绝路径。未再次生成模型样本。

`action-metadata-upload-performance-20261017-r1`已启动，原natural8负载、2轮
预热、300秒窗口、32000自然生成上限、正常排空；仅3个生产文件和async上传
开关变化。预声明选择门槛：对反序rolling-v2输出率至少+10%，完成回合数不
下降，并通过完整历史/零抢占/上传状态检查，否则真实first16仍用已验证的旧
rolling-v2。该单窗选择不是统计置信区间。完整first16测试仍待性能验收。

Action异步上传natural8性能窗完成：178.703 token/s、108回合，零抢占、完整
历史与输出计数核对通过、无上传回退/隔离错误。相比反序rolling-v2的180.177
及114回合，输出率-0.82%、回合数-5.26%，未达预声明选择门槛。GPU利用率
86.604%，高于旧84.584%，但没有转化成有效吞吐。本候选保持默认关闭，不带入
本次完整first16；低层CPU提交更快不能单独证明agent吞吐改善。

完整真实DeepSWE已启动：`artifacts/deepswe-first16-nvfp4-20261017-r1`。
使用精确的已验证rolling-v2运行时，原16题、8clients、原采样/seed1000/工具
协议/32个镜像/Read32/Bash32/隐藏verifier，每题一次新尝试，duration=0。
相对最近真实8-agent配置，只新增已验收的rolling_active_window与4个resident
恢复额度、降低总历史262144→196608、改用独立服务端口；其他模型参数保持。
模型NVFP4、FP8 KV、MTP4、compiled graphs，active65536 + gen32768。

启动MemAvailable=114478575616B，高于完整8份192Ki历史+完整MeanK+32GiB
模型/系统/搬运余量的95580848128B。原工具容器16g上限不变，持续全局8GiB
余量门禁。完整模式先执行Query/短尾恢复/8路完整G32生成与清理，再放行任务。
所有预检输出从正式计数排除。实时页面`http://localhost:42503/`，预检与任务
统计分阶段展示。原失败资格记录与独立离线复核保持可见，未改写为成功。

首次完整启动r1在600秒后被测试脚本的readiness超时结束，16题均未开始，
清理完成无错误。SM120首编译仍在推进；完整模式的等待上限此前比限时模式
1800秒更短。已统一为1800秒单调时钟截止，后端退出立即失败、503响应也退让
等待；36项相关CPU检查通过。记录r1保持失败，新的r2只修改测试启动等待，
未修改模型参数/运行时/样本，task attempts retried=0。
`artifacts/deepswe-first16-nvfp4-20261017-r2`已启动，manifest SHA256
`46fac71dccc717036505566e2f1c2544e906d26df1dc28a8289ea53f5c8071b6`；
同一实时页面转到r2。首次编译/预检不计入正式任务吞吐。

r2预检已全部通过：8路原子Query、短尾恢复与不中断执行一致、8路各完整生成
32768 token并清理。逐一核对8条terminal history等于prompt+output-1；证据
`preflight-review.json`。预检中的8次pressure重选异步完成，各有7个公开peer，
复用约62240个resident V token，CPU保留完整历史。合成重复文本的decode约
690～750 token/s且MTP接受率接近100%，不能视为真实DeepSWE性能。
启动自动调优曾出现临时大尺寸candidate OOM并回退default tactic，原日志保留；
不宣称全流程零OOM。正式任务于wall1792224557.7947052（本地16:09:17）开始，
前8题已运行。`metrics-before-formal.txt`在全部预检清理后取样，排除预检输出。

正式r2于216.17秒因运行时错误结束，已启动8题全部保留为基础设施失败、不评分，
后8题未启动，owned清理无错误，不能作为完成的16题验收。错误是
`RefreshRequired`逃出原生分配器：mashumaro第19轮恢复H=active=98254，
工具后prompt=99826；先推进18token到98272，下一次完整32token压力重放的
控制分配正好达到K+G=98304，但又附加4个MTP前瞻槽，导致越界。此前固定
预检未覆盖这种跨轮未对齐近满窗口。完整失败证据见r2的`failure-review.json`。

修复候选`prefill-pressure-reservation-20261017/runtime`仅改变capacity/scheduler：
只对已证明将从公开batch移出的异步prefill压力控制分配省略临时draft前瞻，
完整32token重放不变、生命周期页额度不变；公开重放在压缩后的窗口正常申请
MTP前瞻。若控制分配未变成精确的私有事务，直接拒绝执行公开forward。
90项CPU检查通过；已冻结的真实模型定向复现为
`prefill-pressure-qualification-20261017-r2`，2个原失败几何窗口+6个decode peer，
含target/MTP/recurrent字节oracle及完整CPU历史核对。它不是DeepSWE重试评分。
定向入口r1缺辅助脚本，未执行到GPU启动；原始bootstrap失败记录保留。r2已
补齐冻结依赖并通过入口导入检查，控制器1928683、模型1928709/1929051已核验
存活。失败请求已用原协议和逐轮own-output精确重建，并通过原token哈希校验。

另外，r2请求总上下文限制是196608，底层repository仍为8192*32=262144。
此前文中的“192K历史上限”应理解为本轮请求/轨迹限制，非仓库最大容量；
MeanK内存预算仍按完整256K仓库预留。原分析器要求二者相等，因此其严格
`contract_matches=false`保留，不能改写为通过；没有观测到实际预算越界。

压力控制修复的真实模型验收r2已通过（wall1792226856.2037332）：2个准确的
H98272/replay32控制边界均有6个公开peer；重选及后续再次恢复4次target/MTP
字节、recurrent字节完全一致，12份终态完整CPU历史通过，20次调用输出总计
49802与原生计数一致、零抢占、清理完成。此处是hot resident复用的
native_terminal_window oracle，不冒充cold immutable rebuild；此前独立冷恢复
资格仍保留。强制生成的资格样本不用于DeepSWE准确率或吞吐结论。

此次冷编译耗时长的额外原因已确认：/usr/local/cuda与/usr/local/cuda-13都
解析到同一CUDA13.0编译器，却造成Ninja构建命令变化和重复编译。正式r3显式
采用与验收相同的CUDA_HOME，记录编译器哈希；不改变CUDA版本或模型参数。
启动FlashInfer调优仍有临时OOM并回退默认tactic，未宣称启动零OOM。

完整first16正式入口r3已启动（wall1792226931.4951305，controller1938590）：
`artifacts/deepswe-first16-nvfp4-20261017-r3`，manifest SHA256
`f4ab0c974c5b4509ae1ad3ebbdaad327a473065411e7d06390781368cf0e635e`。
源代码采用刚通过验收的两文件修复；原16题/8clients/NVFP4/FP8KV/MTP4/
active65536/gen32768/192Ki请求限制及原工具协议采样保持。前一失败版本8条
基础设施失败轨迹保留、未评分；新版本每题一次新尝试。开始运行不等于完成
验收，预检和正式任务计数继续分离，最终分数与吞吐待全部结果复核。

r3全部预检通过，formal于wall1792227590.204722（本地16:59:50）开始；
`preflight-review.json`逐一确认8份H=M=80306+32768-1=113073，CPU-only、
零SSD字节，并保存正式计数基线哈希。实时页面仍为http://localhost:42503/。
第一份约297秒诊断见`analysis-first-window.json`：输出87842（295.558/s）、
计算公开prefill575258（1935.545/s）、零抢占/错误/ctx ceiling；GPU采样平均
78.63%，纯decode实际8请求step1122次。无公开prefill保守窗口仅10.537秒、
6352输出（602.808/s）；要求8个已知ready的窗口更短，仅3.141秒（626.642/s）。
这些窗口仍含KVMem维护，不是纯kernel峰值或完整16题结论；8题均仍running。
不同真实输出会改变工具轨迹，本次早期数值与旧失败216秒窗口不能作为严格加速比。

r3只读低利用率审计见`idle-sample-audit-early.json`，覆盖正式起点后500.6秒。
478个NVML有效采样中18个低于10%；排除最初60秒后420个采样中只有4个。
4个均有跨采样前后1秒仍在途的bridge请求，但在途不等于GPU就绪；不得将
采样数直接换算成精确idle秒数。offset82～84附近public step7415包含2355
prefill token和6个decode请求，model提交区间约2.963秒、CUDA事件跨度
2.976秒；同一步维护host50.5ms，邻近checkpoint提交约3.6/5.0ms、没有
host sync wait。public forward还含KVMem hook，现有证据不能归因原生vLLM
或CPU恢复；需后续定向子阶段采样，不能把整个区间都标成KVMem维护开销。

首个上下文上限outcome为wasmi-trap-coredumps，第41轮prompt164987，
请求output32000，合计196987超过声明总上限196608。该题记录context_ceiling，
不评分且无自动重试/截断；CPU历史释放后sql-formatter-bigquery-pipe-formatting
补位。CLI重复发送的3条context_limit是同一个task/turn，不算3道失败题。
减少总上下文是本次声明的差异，最终需同时报告16题覆盖、可评分数、成功数
和ctx ceiling数，不能用缩小分母制造与原版的准确率提升。

r3正式约800秒复核见`analysis-live-13min.json`：320225输出（400.115/s），
计算prefill763649（954.164/s）；后续300秒与200秒分段输出率分别461.916与
464.515/s，初始段约295.310/s。无公开prefill窗口扩大到158.317秒，输出率
584.238/s；其中8个已知ready decoder的保守窗口58.637秒、35068输出，
598.050/s。实际纯decode8请求step4134次，7请求5454次；GPU采样平均81.15%。
零抢占/错误，8任务running、1context_ceiling、无评分。controller1938666与
engine1939258当场核验存活；前一goal turn为完成模型资格/启动正式测试的进展，
本turn新增idle分类和更长正式窗口证据。全目标仍未完成：真实16题未结束，
该192Ki真实队列也不能替代强原生256Ki容量压力对照及>=25%目标验收。

r3新增负面验收证据见`checkpoint-capacity-review.json`。约20分钟后发现4GiB
检查点池接近满额，并已有terminal capture_deferred。示例H54677保留P19008、
H22707保留P18976；pool.idle会rewind到P，下一回合重算后缀，因此不满足本目标
完整CPU历史/无重算约束。已明确标记本轮性能验收不通过，原run继续收集任务
outcome，不隐藏原吞吐或自动重试，不把它计为已完成的优化验收。

根因不是残留会话各占满805MiB MeanK：key_sum/mean_key随历史页创建，1token
清理只留下一个小索引。实际固定成本来自模型checkpoint：bridge finish及预检
cleanup发同namespace的1token请求，触发新M快照约157876224B；已结束任务和
预检留下的快照逐渐挤占仍在运行任务的P/M池。需要显式、带generation fence的
session release，让旧history/model完成eviction后不再为cleanup创建新checkpoint；
不能仅增大4GiB池掩盖多用户生命周期问题。下一版也应监控终态H=M以及checkpoint
deferral，尽早暴露该契约破坏。当前冻结模型/测试参数不做热修改。

原生小prefill对照入口已增加显式native-only2048/4096总batch预算，去除其不需要的
Query派生预算；默认16424及K配置不变。75项CPU launcher/契约检查通过，耗时
0.38秒（执行时段保存在`native-small-prefill-cpu-validation-20261017.json`，未从
正式吞吐扣除），无GPU运行。预声明计划`native-prefill-tuning-plan-20261017`。
降低batch也可能改变原生实际KV池，届时必须重新测容量，不能沿用1193页上限。

## 显式释放修复与下一次 GPU 验收

会话释放候选已冻结在 `artifacts/session-release-20261017/runtime`。相对当前
real16 r3 的冻结生产源码，仅修改 `session_control.py` 和 `http_sessions.py`：
显式 `kv_transfer_params.kvmem_release_session=true` 请求必须携带 namespace、
一个输入 token、max_tokens=1，且不能带 Query；普通单 token 请求语义保持。
关闭请求先等待正在执行/退休的同会话，再通过原 generation-qualified eviction
释放历史、模型状态和驻留页，收到 worker ACK 后才完成普通的单 token 请求。
它保持 namespace 串行化，但不建立新的 SessionBinding，也不捕获替代的 P/M。
这不增加检查点预算，不改变 MeanK/Action、FP8 权威历史、MTP4 或 CUDA Graph。

bridge finish 和 first16 预检清理已改用显式释放。预检必须确认 checkpoint 数量、
payload 和 CPU authority 全部归零；正式监控检测到 checkpoint deferral 或成功
会话终态 H/M 不一致会停止验收。历史检查先于 GPU/HTTP 监控，因此指标接口超时
不能绕过这个保护。正在运行的 r3 不热修改，原失败证据继续保留。

冻结候选的 98 项 CPU 回归通过；新验收入口/传输/原工作负载相关 97 项通过；
历史监控独立性补充回归 29 项通过。它们是不同且部分重叠的测试集合，不能
相加当作独立用例数，且都不替代真实 GPU 验收。日志和执行时间保存在
`artifacts/session-release-20261017/*cpu-validation.json`。

真实 GPU 验收准备在 `artifacts/session-release-qualification-20261017-r1`，manifest
SHA256 `37a91c3a30998341a2bad60fa0506c66997e9c70f1de5f14ebd65ad14355ec23`。
保持 NVFP4/FP8 KV/MTP4/graph、8 并发、K64/G32、100GB CPU 和 4GiB checkpoint
池。五批共 40 个不同 namespace，超过 32 个 session 槽；前四批 8192-token
输入用于验证不随上下文缩小的模型快照固定成本，最后一批保留完整 99826-token
记录输入。每个 namespace 生成 512 token，使用自己的输出加 128 token 后再次
生成 512，逐一验证完整终态历史和 target/MTP/recurrent 恢复字节；每批要求
实际执行过 8 请求 decode，然后显式释放，CPU/GPU/checkpoint 所有租约归零。
额外核对总输出与原生 counter 相等、零抢占和零 checkpoint deferral。
这是生命周期正确性验收，不是 DeepSWE 准确率或性能比较；GPU oracle 会同步。

验收已排队（controller PID1986561，birth1792230500.04），先等待当前 real16 的
四个已核验 owner 进程退出并发布清理完成，再执行一次冻结入口；不杀其他任务、
不自动重试。owner、脚本哈希、等待状态在 `artifacts/session-release-20261017`。
之后仍需完成原生 2K/4K prefill 强对照和相同轨迹/工具等待/固定窗口的 KVMem
比较。此处尚无修复后吞吐结果，也未完成 >=25% 的总体目标。

r3 正式约 2337 秒的只读诊断另存为 `analysis-live-throughput-1739.json`：全程
548535 输出、234.709 token/s；最近约 291 秒为 77.541 输出 token/s 和 3018.480
公开 prefill token/s，后者包含重算。8 路已知 ready 且无公开 prefill 的筛选窗口
累计 111.018 秒、585.085 token/s；不代表持续 agent 服务速率。采样时 84 次
checkpoint deferral，故这些数值不能进入完整历史/无重算性能验收。

后续处置：r3 deferral 增至 118 次且没有可评分任务，继续运行不能恢复验收前提。
于 wall1792230653 附近向已核验的 controller1938590 发出 SIGTERM，由原控制器
完成模型、agent 容器和 network 清理，cleanup_errors=[]。最终 15 题已开始：
7 context_ceiling，8 条运行轨迹在主动停止后被控制器记为 infrastructure_error，
另1题未开始，0题评分。这8条不是8道答案错误。完整原始轨迹、停止原因和
终态审计保留在 r3 的 `stop-failed-acceptance.json`、`terminal-review.json`。
此前“继续收集outcome”的安排已据持续失败改为停止并优先修复，未覆盖旧结果。

释放候选 v1 的附加 CPU 联动审计复现了第二个缺口：SessionLedger 已不再为
close 建立 binding，但 HistoryAdmission 仍把它标为 session credit，收到
普通 finished_sending 时报告 Missing session retirement。候选 v1 的 GPU
验收在加载阶段主动结束；记录在 `credit-release-reproduction.json` 及
`session-release-qualification-20261017-r1/preflight-credit-failure.json`，不计通过。

`runtime-v2` 因而多改一份 `coordination.py`，相对 r3 共三份生产文件。经过
验证的显式 close 使用 request credit；在旧 namespace 仍占额度时，只允许
它抵达 SessionLedger 的 eviction，不允许提前分配。旧 generation ACK 后才
分配小请求额度，并由普通 worker finished_sending 释放。满CPU预算时也能
关闭自己的会话，其他用户的额度和正常请求路径保持原有语义。

冻结 v2 的 171 项相关 CPU 检查通过。首次组合检查为170通过、1失败：旧原生
传输单测要求在75ms实际墙钟内完成多个131K输入epoch，受CPU序列化速度影响。
该单测现使用随模拟请求推进的时钟，保留完整输入、自己的输出、epoch salt隔离
及全部断言；未修改实际负载时间窗。失败和修复后完整日志均保留。

v2 真实 GPU 验收为 `artifacts/session-release-qualification-20261017-r2`，manifest
SHA256 `a328ed52c33191553c9121052e2363e1b698a5a051997b126d55695b760f0fa5`，
controller1990777，API1990821，engine1991165。前4批32个namespace已完成自己的
输出续接及释放，各自检查点数/字节、CPU历史字节、CPU租约、GPU驻留页均归零，
deferral=0；最后一批8个完整99826-token输入仍在运行，不能提前算通过。

修复后的 first16 r4 已准备，manifest SHA256
`7ec83c250411e424b47cfb65aa087283ed3b70fb014b0837aa885fbf2b734db2`。
原始题目、Claude二进制、镜像、采样/seed、工具/隐藏评分协议、Read32/Bash32、
8并发、192Ki配置、K64/G32、100GB预算均逐项与r3比较相等；只更换上述生产修复
及显式清理/历史监控。新的冻结 full-test 入口必须验证 release GPU 验收通过、
40个namespace/80份完整历史/40次恢复、41000输出对账、零抢占，以及生产源码
逐文件一致后才可启动。当前未通过的 GPU 验收确实被入口拒绝，负例证据见
`qualification-gate-negative-check.json`。仍没有修复后的正式性能或质量结论。

release GPU r2 最终于 wall1792231482.5404546 通过：40 namespace、80份终态完整
历史、40次 target/MTP/recurrent恢复全部通过；长文两轮的历史分别为100337、
100977 token。40份target oracle均为 native_terminal_window 热恢复，不能说成
40次冷恢复。五批真实8路纯decode步骤分别229/226/276/255/220，所有批次关闭后
checkpoint数量/字节、CPU历史/租约和GPU驻留页归零，零deferral、零抢占，客户端
41000输出与原生counter完全相等。人工复核摘要见r2的`acceptance-review.json`。
资格过程开启字节oracle，不把其耗时用作性能结果。

r4冻结入口正向验收通过，另存`release-qualification-proof.json`，确认其生产源码
逐文件匹配这次通过的候选。r4 controller1994346（birth1792231545.34）已启动，
stage controller1994365（birth1792231565.73）。原子Query、混合batch短尾恢复已
通过，正在做8路完整G32预检；正式任务尚未开始，不把预检输出当成agent吞吐。
本地页面已切换到r4：http://localhost:42503/，监控owner另存于r4目录。

预声明的原生2048/4096总prefill预算对照也已各自冻结，只准备、未使用GPU：
`capacity-chain-g16-native2048-20261017-r1` 与
`capacity-chain-g16-native4096-20261017-r1`。两者均保持同一G16输入哈希
`8f049098de7fa13c804adfc3e2d0c9035d8ce883489490e33263a721c9b9a63a`、8clients、
两轮预热/300秒正式窗、原始工具等待、native async/APC/MTP4和独立InputBatch
观测；只显式改变native batch预算。准备过程每组约2.9秒，在r4模型预检期间完成，
具体时段见`native-prefill-tuning-plan-20261017/preparation.json`。后续必须重新
测实际GPU页池、headroom和decode cohort，不能直接用旧容量上限计算加速比。

r4 于 wall1792232212.5871322（18:16:52）正式启动首批8个agent。8路完整G32
预检均输出32768 token；事后逐一匹配原生完成记录，确认8份历史均为
H=M=113073，cleanup_passed全为true。冻结入口必须通过全局CPU历史/检查点
归零条件才写入正式起点；该归零检查没有另存独立快照，因此不能把后续当前
archive中重新分配的8个正式会话误当成预检未清理。正式数据中deferral仍为0。

正式首174.373秒的初步诊断另存r4的`formal-early-profile.json`：39466个已接受
输出、226.331 token/s；实际公开prefill495198、按同一墙钟2839.882 token/s。
这是包含冷输入、工具、恢复和干扰的早期窗口，不是原生对照，也不是纯prefill
算子速度。1339个纯decode步骤中456个实际batch=8，其余主要为6/7；模型阶段
中位数40.000ms，维护host中位数1.624ms、p95 31.775ms。

为判断该先限制哪类工作，离线分析器现在按已经采集的token分类分开普通
prefill、公开原子Query以及未知来源，匹配composition/context的分桶也保留
这一区别。分区计数不相等会失败，缺少分类不会猜成普通prefill；相关24项
CPU检查通过。仅修改离线分析，不更改运行中的冻结源码或额外发出模型请求。
普通混合步骤240个，模型阶段中位数310.568ms、p95 557.217ms；普通prefill
总量最大6144，但p95仅2048。因此“单请求2K导致batch叠加”确实存在，却还
没有证据证明它解释多数混合开销。先做原生2048/4096强对照，再决定是否启用
现有共享prefill预算，不能据一个混合/纯decode比值直接宣称优化空间或因果。

两组原生对照已排队：`native-prefill-tuning-plan-20261017/run_after_real16_r4.py`，
queue PID2001447，birth1792232349.57。启动前只读校验通过；等待当前r4四个
owner的PID与birth身份全部退出、控制器发布无错误清理终态，再依次运行冻结的
native2048和native4096各一次。各benchmark仍独立核对输入身份、持有GPU锁并
等待GPU空闲，不停止其他任务，不自动重试失败组。配置/顺序/manifest哈希/
owner及状态保存在同目录`queue-r4*.json`。原生结果仍未产生，>=25%目标未验收。

r4 首个正式300秒窗已封存为`formal-first-300s-profile.json`，首末metrics采样实际
间隔299.270秒：93089输出、311.053 token/s，564132实际公开prefill、1885.026
token/s（均按墙钟，含冷启动/工具/恢复），191个完成模型回合。零抢占；检查时
checkpoint deferral=0，模型快照约2.531GB，完整CPU历史约23.84GB。3330个纯
decode步骤中1654个batch=8。模型阶段纯decode中位42.879ms，普通混合步骤
中位289.188ms、p95 564.951ms；“8路已知ready且无公开prefill”短窗口共9.404秒，
5723输出、608.546 token/s，不能外推为持续agent服务吞吐。

`idle-first-300s-audit.json` 进一步把低NVML样本与请求/步骤对应：287次采样
平均GPU利用率79.5%，12次低于10%；其中8次在开始的前8秒，bridge尚无请求，
不能算KVMem阻塞已就绪工作。其余4次在79/82/87/166秒附近，有6或8个HTTP
请求pending，邻近普通混合公开步骤的host提交约422–557ms；前三次前置维护
host仅1–4ms，最后一次约62ms，附近记录的checkpoint同步等待均为0。采样
窗口与CUDA执行并不精确重合；公开model区间仍包含MTP/采样/KVMem捕获以及
CPU提交间隙，所以还不能归因到具体内核或证明全部是vLLM本体成本。下一次
独立诊断应在相同composition/context下分开该区间内CPU等待与GPU计算，而
不是据NVML均值直接宣称20.5%的可回收idle。正式r4继续，未在运行中启用重型
profiler或改变其预算。

同轨迹比较的修复后KVMem版本现已冻结：
`artifacts/capacity-chain-g16-release-performance-20261017-r1`，manifest SHA256
`afc564e6224b22407b9c9276d987627fb06ba07f5bc55aef0ab66c1403dd270c`。
`benchmark_capacity_chain.py --release-qualification`仅允许KVMem单臂引入GPU已
验收的5份生产修改（capacity、scheduler及3份release相关文件），逐份限制
差异，核对40会话资格/80历史/40恢复及五次实际资源归零；其他生产文件与
旧rolling-v2一致。8条G16轨迹/输入hash、两轮warmup、原工具等待、natural
G16上限、300秒窗口、K64/G32/100GB和256Ki完整历史准入均未改变。

这里明确区分资格范围：session-release GPU验收配置上限192Ki，已验证的
最长实际历史100977；新比较保留原实验256Ki上限。该上限差异已显式记录，
不能声称先前验收覆盖256Ki满额8用户。新运行还必须对每个完成请求检查完整
H=M、零checkpoint deferral、原生/客户端输出一致及最终全部资源释放。

清理传输函数及其余helper与GPU资格版本保持相同AST，只有timed workload
编排增加显式release选项。发现并修正旧final fence还等待每namespace一份
单token缓存的问题：该模式期望空view，再由资格版本的wait_release核实
CPU/GPU/checkpoint额度全部归零。warmup仍必须保留完整view，普通请求和工具
等待不变；原生路径拒绝启用KVMem release选项。相关117项CPU检查通过，冻结
入口全部只读校验通过（约2.83秒）；`cpu-validation.json`、`prepare-review.json`
记录准确源码和结果。此对照尚未启动GPU，等待现有r4及原生两组控制结束。

为下一次独立trace扩展了仅torch-profiler模式启用的10个CPU范围：capture的
FP8打包、下载槽等待、提交推进/owner drain/发布/复制/Query history，以及
Action prepare/capture/accept。既有绑定、嵌套、异常传播、不重复安装及关闭
时不修改调用路径的8项检查通过（CPU环境NVML不可用警告，不是GPU验收）。
这些诊断修改在性能候选冻结之后完成，不在当前r4或新性能候选runtime中。
源码副本及诊断问题/对照约束保存在`artifacts/mixed-step-profile-20261017`；
尚未采集GPU trace，也没有据范围名称推断瓶颈。

r4正式1074.153秒检查记录在`formal-progress-after-release-wrapper.json`：
整段400.948输出token/s、815.930实际公开prefill token/s；最近299.342秒
422.139输出token/s、329.499prefill token/s，均按墙钟包含工具/恢复。已轮换
到10个任务，其中wasmi与drizzle触及192Ki ceiling、未评分；其余8个继续。
零抢占、零checkpoint deferral，检查点约2.375GB、CPU历史约33.97GB。新
任务进入后未重现预检/已结束namespace累积固定模型快照的问题，但仍须继续
到全部16题终态。以上与原生强对照尚不能组成提升比例。

2026-10-17 18:46 的新统计封存在 r4 的
`throughput-live-20261017-1846.json`：1770.189秒中生成687088 token，合计
388.144 token/s；最近299.552秒生成112218 token，374.619 token/s，新增
公开prefill为952.264 token/s。该窗GPU利用率采样均值80.235%，零抢占、
零checkpoint deferral。保守排除公开prefill的片段累计53.565秒、550.300
token/s；其中已知8路ready的片段仅4.190秒、589.907 token/s，不能作为
持续8路吞吐或原生对比。随后已有12题达到本次192Ki输入加输出上限而未
评分，1题自然结束但严格任务reward=0、3题继续。触及容量上限的样本不能
记为答错或已完成评分，当前结果也不能给出完整16题准确率。

独立短诊断入口 `scripts/profile_capacity_chain.py` 已完成，最终准备目录为
`artifacts/capacity-chain-mixed-profile-20261017-r2`，manifest SHA256
`fe13488d1ec904d67ff1a89e061d4956d10462e10e0b74008e5a685c40a40853`。
它复制修复后G16性能候选的冻结输入、两轮warmup、完整历史/工具等待及采样；
仅运行独立60秒诊断窗口并正常排空，在8条完整warmup历史发布后启动32步
原生CPU/CUDA profiler。性能候选未完成或deployment不通过时拒绝启动。
trace必须实际含混合prefill/decode、CUDA kernel及新增capture/Action CPU
范围；覆盖不足保留负面结果，不自动重跑。所有诊断结果明确禁止用作吞吐
提升证据，后续用既有 `analyze_meank_profile.py` 区分CPU嵌套、CUDA launch、
同步API、kernel和memcpy/memset，不能从CPU范围相加推导GPU idle。

准备中修正了两处诊断问题：引用对照时必须加载对照自己冻结目录内的验证器；
CPU范围仅接受user_annotation，避免把同名gpu_user_annotation计入host
耗时。44项针对性CPU检查通过，冻结校验及“普通性能未完成时禁止诊断”
负例通过。初版r1仅准备、从未启动GPU；其接续等待进程已在启动任何子任务
之前取消，旧目录保留。修正版接续目录
`artifacts/capacity-followup-20261017-r2` 的进程为2039274，出生时间
1792234339.84。它核实既有native控制队列2001447/1792232349.57结束、两组
控制发布清理终态之后，依次运行修复后KVMem的300秒测量及短诊断；各自仍
持有GPU锁、核对身份与CPU完整预算，不终止外部任务、不自动重试。现有
真实16题及native2048/4096队列未改参数。生产优化收益尚待这些公平对照
和实际trace，至少25%的目标仍未验收。

最终速度对照还需排除原生batch观测器本身的影响。工作区的
`benchmark_capacity_chain.py` 新增显式 `--no-native-observer`，仅允许独立
native单臂确认；默认及既有冻结任务仍启用观测。关闭后使用原样native CLI，
不注入worker扩展，不发observer RPC、不读取/伪造batch证明；结果明确标记
actual batch unavailable。采样、原工具等待/完整输入、两轮warmup/300秒、
CPU完整预算、实际allocator测量、抢占/headroom和退出检查保持不变。
166项CPU检查通过（1.84秒），涵盖关闭时原样launch/无RPC，以及负面资源
验收仍保留原始吞吐而不标记成功。计划与源码hash保存在
`artifacts/native-observer-overhead-plan-20261017/plan.json`：待现有2K/4K控制
完成后选择部署合格且300秒输出率最高的一组，再单独关闭观测复核。若两组
均不合格，应先审查负面证据；不得把弱原生对照包装成已达成目标。此新增
模式尚未跑GPU，未修改正在运行的r4或已排队的冻结版本。

r4追加 `ready-work-sampling-audit-20261017.json`，覆盖2567.920秒、2444个
NVML采样点。8个agent尚未到终态时GPU采样均值80.497%；少于8个agent的
采样均值61.978%，后者包含启动边界和收尾，不代表持续8路负载。226个低于
10%的采样点进一步与桥接start/finish及调度器states交叉核对：199个同时
无bridge在途回合、无原生running/waiting且无scheduler请求，26个三者均
有，另1个bridge在途但尚无原生/调度器请求。仅凭原生队列零值不能排除
KVMem停放的请求，因此上述交叉检查不可省略。这些是近邻采样相关性，
不等于226秒真实device idle，也不能将CPU分数或等待总和算成可恢复吞吐。

完整的前8个300秒窗口输出率依次为311.05、453.07、421.75、366.89、411.32、
349.80、325.77、178.31 token/s；后段agent退出导致的负载下降已单独标记。
最后一个不足300秒窗口不能与满窗对照。未见checkpoint deferral；真实
测试仍有最后1题持续推进。后续短trace重点解释“存在模型请求时的空档”，
不能把尾部没有工作到达的低利用率当作KVMem调度损失或25%提升空间。

r4真实16题已到终态：正式执行2761.839秒，15题触及192Ki容量上限、未评分，
1题自然结束且严格reward=0；不是0/16准确率。原始agent响应共868294输出，
含冷prefill/工具/低并发收尾的整段输出率314.390 token/s；原生计数器窗口
约2761.388秒、公开新增prefill624.143 token/s，包含少量release控制token，
详细口径见`final-review.json`。全程零抢占、零checkpoint deferral、零SSD
读写；任务控制器正常结束，1792234982.014完成清理且无错误。结束后GPU
进程为空、仅14MiB基础占用，CPU历史/模型检查点/retiring均归零。最后保存
的scheduler快照仍有一个close请求，因此不能宣称它证明了服务停止前所有
scheduler额度均为零；模型进程退出后的显存释放另有进程/NVML证据。

`formal-terminal-history-proof.json`把全部1154个完成模型回合按响应ID关联到
native终态记录：1154/1154均满足完整输入+接受输出-1=history_end=
checkpoint_tokens，checkpoint_kind=M且deferral=0，无遗漏、重复或长度
不一致。它证明终态历史发布长度和模型检查点覆盖；性能运行未逐回合执行
KV字节oracle，后者仍依赖此前独立GPU恢复资格，不应混淆。session的终态
记录没有普通retirement的gc计数，缺字段不能被解读成history不一致。

原生2K第一次接续在GPU启动前失败：队列将PYTHONPATH设为不存在的
`runtime/src`，导致校验导入旧RequestPlan签名。失败日志与
`native2048-r1/prelaunch-failure-review.json`保留，未启动模型或产生吞吐。
人工核实原因后仅修正控制器为`frozen/src`，从原冻结代码新准备
`artifacts/capacity-chain-g16-native2048-20261017-r2`，manifest SHA256
`7bebede5ddd5b5c84fb96d9e952059108bf6890fbc8848dc1323d06a6033efee`。
逐字段确认原源码、runtime、输入/采样、预算及模型profile hash完全相同；
4K组未尝试且保持r1。两个实际子进程已各自验证RequestPlan来源与完整冻结
身份，证据在`artifacts/native-prefill-tuning-plan-20261017-r2/import-verification.json`。

修正原生队列PID2045110/1792235220.17，2K子控制器2045111/1792235220.21，
API2045149、engine2045474已实际启动，GPU现用于原始长历史两轮warmup。
测到新原生pool为1202页、block1616、有效KV1740867 tokens；不能沿用旧
1192/1182页结论。2K CUDA Graph正常完成，是否满足4GiB headroom/零抢占及
最终吞吐仍须等待全程数据。新接续目录
`artifacts/capacity-followup-20261017-r3` PID2045117/1792235221.8，等待该
原生队列结束后执行原先尚未运行的KVMem测量及mixed-profile-r2；旧失败
队列均已退出，没有更改它们的结果或在观察超时后重启工作。


2026-10-17 原生2K长历史对照已完成，独立300秒窗口7179个有效输出，
23.93 token/s；15回合提交、7回合在窗口内结束，另外8回合正常排空。
实际GPU采样利用率98.715%、零抢占，最小采样显存余量6099MiB，部署检查通过。
原生worker观测没有纯长历史decode批次，混合批次内decode子集为1–6路；
不能将running gauge或混合批次总请求数称为8路decode。
完整结果见`artifacts/capacity-chain-g16-native2048-20261017-r2/analysis.json`。

`prefix-work-review.json`将续接工作量单列：15回合新增工具输入5059 tokens，
完整prompt共3311393 tokens；连同8个cleanup输入，排空后的local_compute
增量1365737、local_cache_hit1945664，精确相加对应3311401个输入。
正式窗口内local_compute完成计数增量706256，边界计数与客户端流式输出
时间口径不同，不应用它推导GPU kernel速度。期间无epoch重置，cache_salt
按用户和epoch保持不变，完整自有输出历史校验通过。这证明存在大量续接
prefill，尚不能仅凭聚合计数区分容量淘汰与混合模型状态对齐限制。

原生4K r1在模型启动前的第二次内存检查失败，native服务目录不存在。
其前三次连续合格采样只刚越过113834459136B门槛，后续启动检查跌回门槛下；
失败与原始采样保存在`prelaunch-failure-review.json`。新启动等待在完整预算
之外增加128MiB波动余量，最终检查与运行时8GiB余量检查仍保留；121项CPU
检查通过，覆盖余量不足、连续样本中断、取消和不合法参数。未减预算或修改
样本、采样、输入、工具等待、模型runtime和300秒计时协议。

新鲜准备的原生4K r2与release-performance r2均经过完整冻结校验；旧K测量
和旧profile未运行，保留prepared状态，不复用失败路径。独立mixed-profile r3
指向新的release r2。单次队列`artifacts/capacity-followup-20261017-r4`执行顺序
为native4096、release、mixed-profile，profiling必须等release部署检查通过；
不自动重复失败工作。控制器PID2052479/birth1792236950.54，4K子控制器
PID2052480/birth1792236950.58；本轮已实际确认存活，等待完整CPU准入预算。
两组正式比较仍需完成，不能将23.93与另一种真实16题短历史负载的314相除。


补充更强原生对照入口：`--native-shared-prefix-cache`仅允许native-only测量，
客户端适配器只省略completion POST中的cache_salt，保留全部输入token、
采样、请求ID和原始输出计数。KVMem控制字段被拒绝，不改变生产用户隔离。
实际缓存口径写入manifest/results，默认隔离模式未变；可与此前
`--no-native-observer`组合验证原生无插桩吞吐。183项相关CPU检查通过。
`artifacts/native-default-apc-plan-20261017/plan.json`约定先等当前2K/4K控制
完成，再选部署合格且最快的batch参数运行独立shared-APC原生控制。
当前仅实现并验证入口，没有运行或声称改善；若合并关闭observer与共享APC，
两因素的因果贡献仍需分别判断，不能把全部差值归因于其中一个。

原生2K的首个实际InputBatch记录进一步确认：15个正式续接请求中6个缓存命中为0，
各自重算完整200K–235K输入；其他请求的首步computed_prefill_lengths合计
1945664，与原生local_cache_hit完全一致，所有请求computed prompt总量
1365729加8个cleanup token也与local_compute计数完全相等。证据见
`request-prefix-reuse-review.json`。这将大额重算定位到具体请求，但还不能
区分哪个KV组先失去可复用页。公共前缀的潜在节省另存于shared-APC计划，
其token数不等于混合模型可直接共享的实际物理页数。


启动准入修复的最终形式：4K r2的额外128MiB门槛在600秒内未稳定达到，
任务正常报错退出且从未创建native服务目录；保留该失败。为避免把预留量
调小来碰运气，改为在所有nvidia-smi子进程检查之后执行稳定CPU准入，
再用已初始化的进程内NVML检查是否有外部GPU进程出现，最后保留原有
MemAvailable检查。这样不在合格内存采样与最终检查之间启动新的探测进程，
无需额外128MiB启动余量；原113834459136B完整历史预算和运行中8GiB底线
均保持不变。变化只涉及benchmark_capacity_chain和benchmark_ready_agents
的启动阶段，不涉及计时阶段或推理runtime。189项CPU检查通过，覆盖顺序、
等待后内存仍不足、外部GPU新进程、取消时释放NVML及缓存共享的传输约束。

全新冻结目录native4096-r3（manifest878cdd3f0134cc76c33584d965fa13772e5435e93d8b9d7e0b5504a27e9c0665）、
release-performance-r3（618a7cde3d2ca0367f02db55fcd8dfa1de5a7b0dd129455a114130766d41c26f）
和mixed-profile-r4均完整校验通过。前两者的fixture、runtime、profile、预算、
采样和工作量与前版相同，差异记录在prepare-review.json；前一队列真实PID
2052479和2052480均已消失后才启动新队列capacity-followup-20261017-r5，
控制器PID2053258/birth1792237685.87。先完成4K，然后KVMem，再独立profile，
不复用或覆盖旧失败测量。shared-APC入口仅准备，仍未用于当前冻结对照。


4K r3已通过修正后的启动准入，API PID2053290/birth1792237720.86，
engine PID2053637，1792237890进入8用户原始长历史预热。实际pool为1198页、
block1616、有效KV1735074tokens，不能沿用2K的1202页。预热实测长prefill
通常为2请求合计3232token；2K组通常1请求1616token。虽然两组long-prefill
threshold均为2048，总batch预算确实改变了实际输入批次，不能仅看CLI
threshold推断两组执行相同。

prefill计量进一步核实：2K原生所有成功worker dispatch中的实际prompt部分
为3371388tokens，与最终local_compute计数完全一致。正式300秒内按dispatch
开始时间归属的prompt部分848464tokens，即2828.213token/s；完成prompt才
累加的外侧计数为706256tokens，边界仍在处理的部分造成差异。证据在
`native2048-r2/executed-prefill-work-review.json`，包含每步明细。该数仍是
混合负载墙钟吞吐，不是独占prefill kernel速度，跨T1的最后一步另有边界限制。

候选KVMem的`batch-telemetry-source-review.json`审阅了真正冻结的runner和
StepTiming：计数来自实际InputBatch.is_prefilling_np，排除padding和队列；
原有记录不含worker request IDs，仍需要保守的长工作请求时间重叠核对，
不能把所有warmup/cleanup的8路计数算成正式容量收益。workspace runner
另有未选中的MeanK启动预加载改动，已区分，未带入冻结性能运行。


4K r3已完整结束并通过验收：300秒2096输出（6.9867token/s），12回合提交、
4回合在窗内完成，另8回合正常排空；客户端/原生计数一致、零抢占、最小
采样显存余量6119MiB。无纯decode步，混合批次decode子集分布为1路163步、
2路114步、3路13步、4路36步。12个正式续接请求中8个首步缓存命中0。
实际窗口prompt工作1063485tokens（3544.95token/s），所有阶段3972220
与原生local_compute计数精确一致。证据见`final-worker-review.json`。
较2K输入处理率提高，但有效输出和完成回合数下降；不能把prefill率提高
当作目标中的agent生成收益。二者均按真实自然输出续接，输出长度和缓存
状态也会随执行改变，批次大小不是唯一可直接归因的中间变量。

KVMem release-performance-r3已接续启动：子控制器PID2060369/birth1792239150.95，
API2060440/birth1792239198.18，1792239273进入原始预热，尚无正式结果。
队列仍为release后独立mixed-profile-r4。旧K对照167.2033token/s、94回合
及118个非cleanup请求完整历史发布已逐项核对；fixture/采样/预算/窗口及
profile（除专用端口）一致，生产runtime仅五个经过GPU资格验证的文件变化，
详见`comparison-control-review.json`。

更强原生入口增加显式`--native-gpu-memory-utilization .93`；原有.915默认不变，
所有阶段4GiB余量和零抢占验收不变，195项CPU检查通过。准备并完整校验：
`capacity-chain-native-shared930-observed-20261017-r1` manifest
6328a83b69aad1c29318e3a2a8452d7d7284747d3a1216687960c69a75afd455，
`capacity-chain-native-shared930-clean-20261017-r1` manifest
cbb65600ff2a0cb83ece3b0a8f281379e2f6ef41e284bec4d38c7258295d42ff。
两者均2K分块、8客户端、共享公共前缀、.93 GPU分配，原输入/工具等待/自然
G16/完整CPU预算/两轮warmup/300秒均不变，区别为observer开启或关闭。
这是组合增强的native配置，不能把全部差值归于APC或显存份额某一项。

新队列`native-strong-control-queue-20261017-r1` PID2061391/birth1792239263.72，
等待r5控制器正常退出及所有模型的GPU清理证据，再运行observed；只有它
通过部署检查才运行clean。独立profiling若仅trace coverage失败但已留下
干净退出证据，可以继续原生测量，保留该诊断失败而不重试。当前两项强
原生控制仅prepared，未启动，不影响正在进行的KVMem测量。


## Release 性能窗口完成：保留负结果并进入诊断

`capacity-chain-g16-release-performance-20261017-r3` 已完成300秒窗口
（wall1792239776.9477751至1792240076.9477751）及17.071秒正常排空。
客户端42616输出，142.0533token/s，102回合提交、95回合窗内完成；
原生计数精确对账，118/118非清理请求完整历史发布通过，零checkpoint deferral、
零抢占，最小采样显存余量5791MiB，显式释放后CPU历史/检查点/额度/驻留页归零。
模型停止后GPU无计算进程。部署验收通过，不代表吞吐目标通过。

相同fixture/采样/预算的旧K窗口167.2033token/s、94完成回合；本次输出率
降低15.04%，完成回合率提高1.06%，纯decode步骤的时间加权平均batch
5.861变为5.808。自然输出最终均长595.41变为451.70，不能仅据token率
把全部差异归因于生产修复。原始负结果保留，不自动重试到得到正结果。
保守核对的完整长历史8路纯decode步骤118次，约5.95秒；这是已有计数
与独占HTTP区间的交叉核对，仍非逐worker请求ID证明或持续8路占用。

进一步按窗口dispatch-start核对：新版364个混合步骤，700335个prefill
输入token，GPU event跨度累计197.981秒；纯decode1803步、92.056秒。
旧版混合385步/651349输入/183.553秒。event跨度含host提交间隙，边界
最后一步可越过T1，不能作为纯kernel时间。新版恢复p95为1.510秒，旧版
1.091秒；正式窗口98次checkpoint的host总耗时仅0.721秒，不能把主要
开销归咎于checkpoint。13个低NVML采样附近均有HTTP/调度工作，其中
若干仍有decode状态和prefill publication等待；邻近采样不是GPU-ready
或精确idle证明，需要独立trace确认。详细交叉核对保存在
`capacity-chain-g16-release-performance-20261017-r3/completed-comparison-review.json`。

队列已接续独立`capacity-chain-mixed-profile-20261017-r4`，owner2067406/
1792240105.98，API2067555。待32步CPU/CUDA诊断给出具体阻塞后再选择下一项
生产优化；更强共享APC原生observed/clean对照仍在其后按序运行。


独立diagnostic-r4在启动后的geometry检查失败，尚未发送工作请求。根因已
确认：profile入口复制了reference的旧`qualify_rolling_window.py`，其中
硬编码active32768/gen4096；实际布局及profile均为65536/32768，FP8每token
34816字节本身一致。不是模型运行或历史恢复失败。全部owner退出且GPU空闲
后补存人工`diagnostic/post-stop.json`，原失败status保留。

修复`profile_capacity_chain.py`：声明并冻结诊断入口与辅助server两份源码；
辅助server按已冻结profile核对预算；无论成功或启动检查失败，都在停止
owned model之后记录GPU退出观察。18项针对性回归通过，包括丢失辅助依赖
和启动失败清理的回归。生产runtime/工作样本/采样/CPU预算未变。
新diagnostic-r5 manifest为e95b955b210c584f9ac88f4b5386ab0b4e5c2cd0a120c8626dba2808286a03df，
controller2071804/birth1792240481.61。旧native队列在两项原生工作均未启动时
正常阻止继续，现用native-strong-control-queue-r2等待新诊断owner退出和
GPU清理证明，再执行原来prepared的observed/clean两项，未重新准备或重跑
已完成性能窗口。新队列2071806/birth1792240481.66；不自动重试失败测量。
