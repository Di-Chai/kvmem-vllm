# 逐层 prefill 优化实施记录（2026-11-04）

后续阶段隔离测量已完成，见 [八路 prefill ABBA 实测](layerwise-prefill-phase-results-20261104.md)：
P4 历史 prefill 吞吐 5804→6659 tokens/s（+14.73%），排除 Query/decode；
八路 A/A 自身及 A/B 严格输出一致性未通过，尚未切换正式实验。

本轮按[`layerwise-prefill-optimization-plan-20261104.md`](layerwise-prefill-optimization-plan-20261104.md)
继续实施。新实验目录：`artifacts/g4k-pressure-implementation-20261104-r1/`，下称`$R`。
原结果与中断目录未改写，工作区其他实验目录未动，没有外部评分、定时巡检或自动重采样。

## 控制与版本

模型、完整输入、Active=16384、Gen=4096、block32、sink41、recent0、逐层累计attention规则、
最终逐层Mean-K、采样和输出预算不变；原生batch上限16424、服务端max_num_seqs=8不变。
串行只改变诊断客户端owner数。正式计时不启用新probe，不同时运行CPU微基准。

重新核对交接文件SHA与Git状态一致，接手时实际GPU无进程。原失败目录指针仅作为协调记录，
没有把它当GPU空闲证明。分析helper清单在`$R/helper-proof.json`，原helper在本轮没有修改。
所有推理副本都从原冻结基线加最小补丁生成，保留原CUDA kernel/original-RoPE差异。

## 原生分数导出：已补模型验证，但未通过完整输出门槛

新目录`$R/native-scores-serial/`重建了同一候选，没有使用旧中断目录。
CPU预检确认加载的新`.so`确实包含`State.scores()`，不是兼容fallback。

- 四题的pressure/Query选块全部一致；前三题完整输出相同。
- `gpt4_7f6b06db`第3397个输出token首次不同（零基index=3396，6088→1910）。
- 已保存逐cohort比较和`equivalence-vs-inline.json`，没有重新采样或挑选结果。
- 首个decode几何差异在step1013，accepted prefix为prompt后3391/3392，均为5个verification rows。
  这只是执行轨迹证据，**不能证明数值根因，也不豁免不一致**。
- 该候选不进入合格基线或性能验收；预备的`profile-v2-native-baseline`已标为不运行。

详见`$R/native-scores-divergence-review.json`。
工作区原有native-score改动仍保留，但后续P2/P4冻结候选显式排除新binary和wrapper。

## P0：把真实等待与逐层计算分开

新probe在executor入口、逐层选择之前记录时间，弥补原`assembly_started`晚于选择的问题；
archive worker携带同一个事件上下文，记录native gather的wall/threadCPU。
probe仅存在新冻结诊断package，不用于下面的ABBA计时。

`profile-v2-inline-baseline`：8道原完整输入，180次prefill pressure：

| 边界 | 平均时长 |
|---|---:|
| queued → executor提交（lane与引擎调度） | 1.408 s |
| executor提交 → worker入口 | 0.322 ms |
| worker入口 → 原assembly_started（含逐层选择） | 21.36 ms |
| 逐层选择本身 | 20.69 ms |
| 其中scores导出，嵌套包含 | 3.95 ms |
| 原生plan准备 | 6.20 ms |
| KV组装主机提交 | 216.59 ms |
| producer返回 → 首次观察到CUDA fence ready | 152.99 ms |
| 观察到fence ready → commit/report完成 | 11.71 ms |

180次中，首次观察到future完成与fence ready的时间差都小于1ms（平均约8微秒）。
这说明约153ms不能直接记为GPU执行时间；但没有精确GPU完成时刻，也不能称153ms全部GPU空闲。
executor排队很短，单纯增加线程不是这一阶段的主要解法。

候选P2诊断`profile-v2-prepared-v4`同样180次：

- metadata/source admission从1,551,294次降到132,514次，约减少91.5%。
- source lookup从6120次降到180次。
- native gather仍为12,240次，线程CPU合计约11.41秒；它没有因元数据复用而消失。
- 主机KV提交诊断均值216.59→182.49ms；观察到fence ready的间隔仍约139ms。

以上不是吞吐验收：probe对每次metadata调用有额外开销，调用数减少也减少了观察器自身成本；
两次8路运行的具体batch轨迹也不同。详见两目录的`p0-summary.json`。

## P2：一次重选复用不可变源与generation admission

### 实现

- `pressure_reader.py`绑定request、accepted history、原selection plan和源记录身份。
  每次只对各层完整选中块的并集准备一次；各层的选择与顺序仍独立。
- `MemoryArchive.prepare_region_reader`在原锁下验证namespace/generation/publication，克隆一次lease，
  持有原不可变bytes引用，不复制KV。逐chunk仍通过原字节/任务准入。
- `prepared_regions.py`完成paired区域复制，沿用原gather及失败排空；复制时不持有archive锁。
- 部分/未归档块走原路径；之后才发布的源不会误用准备表。拒绝stale plan、owner、prefix或handle。
- archive在reader仍存活/复制未排空时拒绝close，scope退出释放lease。
- 首版明确排除legacy GC、historical scope、native payload arena；这些路径继续原准入协议。
- 只用于prefill pressure；Query、decode、策略、精度、CPU KV容量和两个DMA/consumer staging不变。

### 验证

- 相关CPU回归338项通过：`prepared-reader-regression-v4.xml`。
- 冻结GPU组件6项全部通过：`components-prepared-v4/audit.json`，原双读、paired异步、内联、
  prepared四条路径KV全字节一致，覆盖target/MTP、部分尾块、skip、乱序/非连续页。
- 初步完整资格`prepared-v3-serial`四题pressure、Query、完整token/text全相同；88次prefill pressure，
  2992个prepared读取chunk，每次只准备586～819个唯一源。
- v4补上了显式legacy-GC/history-scope排除保护；固定测试profile均为无GC，不改变实际已测分支。
  后续ABBA的两次v4执行又独立核验了相同完整输入/选块/输出。

### 无profiler ABBA实测：收益不稳定，不能报成功

`$R/paired-v4-abba/`，每臂为相同四题、427136个历史prefill tokens，所有四轮选块和输出均一致。
计时为client attempt至最终Query开始，包含准入/pressure/组装等待，不含最终Query和decode。

| 顺序 | 臂 | 历史prefill秒 | tokens/s |
|---|---|---:|---:|
| A1 | 合格inline基线 | 115.170 | 3708.76 |
| B1 | prepared v4 | 119.289 | 3580.68 |
| B2 | prepared v4 | 109.981 | 3883.72 |
| A2 | 合格inline基线 | 118.237 | 3612.55 |

两遍合并：基线3660.02，候选3726.05 tokens/s，均值+1.80%。
**该差异小于轮间波动，尚不能认定有稳定prefill收益。**不根据较快的B2挑选结果，没有额外重跑。
因此没有据此启动102题正式提速验收。

无profiler的现有计时中，组装提交均值为基线约138/143ms、候选约123/120ms，
有局部成本降低；但相同已记录prefill几何下，model阶段也出现约67.2～76.9秒波动。
这些只是现有计时分解，不能简单归因于频率、GIL或某个kernel。见`paired-v4-abba/phase-decomposition.json`。

## P4：prefill post-forward ready-only完成检查

作为独立候选，以合格inline基线构造，**不叠加P2或未合格native分数导出**。
复用已有restore的post-forward完成交接模式：

- `AsyncRefresh.poll_pressure_completions`只处理layerwise single-assembly、prefill、未取消的assembly阶段。
- future必须done，单遍组装marker必须存在，原authority和publication必须仍匹配。
- 调用原`step`，由原CUDA event query、原transaction commit和lane/page释放控制流程；不等待future，
  不同步CUDA，不启动prepare、Query、复制或新assembly。未完成就保持原lane/page lease。
- 在connector本轮`get_finished`反馈之前检查，可在同次反馈携带已完成pressure ACK。
- Query、decode、restore、取消、非single assembly维持原路径。
- 相关CPU合同测试129项通过：`pressure-reap-cpu.xml`，包括真实native memory transaction、
  未完成future/fence拒绝提前ACK、publication与owner隔离、取消/其他阶段不执行和无重复ACK。

### 模型资格与八路观察

- `post-forward-reap-serial/serial-equivalence.json`：四题pressure/Query和完整输出均一致，
  33次事件实际由post-forward完成检查提交。
- `eight-reap-observation`：无profiler、原first8完整输入、原预算和采样，两个新独立runtime，
  没有挑选或重采样回答；两边均180次prefill pressure、0次decode pressure。
- 八路候选180次prefill完成全部在新路径reap。`queued → assembly_started`均值
  1.414→0.432秒（此边界仍含逐层选择），pressure总逐事件延迟2.025→0.894秒。
- 并发历史prefill包络：相同865440 tokens，160.953→136.623秒，5376.97→6334.50 tokens/s，
  **观测+17.8%**。包络包含其他owner重叠Query/decode，**不是纯prefill速度**。
- accepted output量6898→4757 tokens；八路pressure/Query/完整输出严格一致均为0/8，
  最终答案文本1/8相同。因此这个A/B不能授予正确性合格的吞吐收益或准确率结论。
- 全推理墙钟194.948→157.356秒仅归档，不把生成量减少全部算成优化。

详情：`eight-reap-observation/comparison.json`与`timing-review.json`。
当前证据支持继续研究完成交接时机，但没有据此跑102题或宣布已解决Gen4K性能回退。

## 工作区控制：新候选默认关闭

冻结实验使用各自源补丁开启单项实现；为避免把未验收组合默认启用，当前工作区新增显式执行开关：

- `--layerwise-pressure-prepared-reads`：需single assembly、paired读取、CPU且无GC。
- `--layerwise-pressure-post-forward-reap`：需layerwise pressure及single assembly。
- `--layerwise-pressure-native-score-export`：需buffered history及重建的native扩展；该候选仍未通过模型门槛。

三项默认全部关闭，serve显式生成对应环境控制；旧fixture不增加额外False键，Memory driver只转发
明确设置的可选执行标记。旧预算、采样、正常8路默认值不变。不要同时启用所有标记并当成已测组合。
开关入口及当前工作区合同回归另行通过，冻结模型等价结论对应上述已保存版本，不等于当前组合102题验收。

本轮最终CPU相关回归436项通过；重建的native-score显式启用CPU精确性90项通过（与前者部分重叠，
不相加计数）；P2冻结GPU组件6项通过。旧native-score输出分歧仍保留，CPU通过不替代该模型门槛。

## 未合并的CPU复制组件研究

`prepared_copy.cpp`试验一次持有immutable bytes导出的指针，paired K/V单次native调用复制，
以区分重复buffer导出和memcpy成本。它仅在artifact目录独立编译，15个CPU字节/边界测试通过。
CPU复制组件中，无竞争约34.07→32.01ms；合成busy-Python-peer压力约398.87→223.85ms。
后者是人为GIL压力，不能代替真实引擎归因。**没有接入推理或获得模型提速结论**，
不计入已实现生产优化。见`prepared-copy-experiment.json`。

## 后续优先事项与NUMA线索

本轮不启动102题，优先补八路固定执行/自身稳定性与质量边界，不能靠随机重跑凑一致。
P4完成交接比1～4ms的score导出更值得继续；P2作为实验保留，不宣称1.8%是稳定收益。

还发现一个值得下一轮独立验证的执行环境因素：机器有两个NUMA节点，GPU0邻近node0，
CPU affinity为0-15,32-47；原进程可运行于0-63、内存可用node0-1。
没有原引擎KV页位置证据，不能直接断言原运行发生了远端内存瓶颈。

只在一个私有CPU测试进程中，源/目标先触碰于node0，使用相同gather和相同约570MB数据，
node0 CPU约38.82ms，node1 CPU约57.61ms；测试结束恢复该进程affinity，未改全局内存策略。
这证明复制对placement敏感，但不是模型收益或原运行placement的证明。
见`numa-copy-probe.json`。如继续，应先在新诊断中记录引擎真实affinity、页分布和迁移；
再以独立执行配置比较进程级GPU-local affinity，不更改模型/采样/预算，也不直接硬绑定
100GB CPU预算到约64GB的单个node。
