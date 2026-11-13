# KVMem / DuoKV 逐层 prefill 重选：优化与验证交接方案

## 0. 执行摘要

**主要问题仍在CPU组织、读取/线程交接和组装流水线，尚无证据把主要瓶颈归因于重复计算QK。**
当前attention统计已经复用模型前向kernel，归一化按层、按batch缓存。
下一轮应从“实测拆开等待→复用一次重选内的只读数据→批量处理独立层”入手，
再根据GPU归因决定是否改attention统计kernel。

本方案供下一位执行模型直接继续，不代表下面所有候选已经实现或通过。

**后续实施结果：**见[`layerwise-prefill-implementation-20261104.md`](layerwise-prefill-implementation-20261104.md)。
已落实P0边界、P2读取准备复用及P4完成检查；P2计时收益未超过波动，P4八路为未验收观察，
native分数导出有一题后段输出分歧。新工作区执行开关默认关闭，不把候选当作已完成正式验收。
本交接方案原版已保存在`artifacts/g4k-pressure-implementation-20261104-r1/plan-at-start.md`及对应SHA清单中。

### 最新接手状态

- 仓库：`/home/chaidi/kvmem-freetoken/kvmem-vllm`，main，HEAD仍为`20c292b`，未提交、未推送。
- 原八路GIL优化组`artifacts/g4k-gilopt-20261104-r1`已102/102完成。
- 本轮根目录：`artifacts/g4k-pressure-hotpath-20261104-r1`，下文简称`$H`。
- 本轮`profile-baseline-r2`（8题诊断）与`inline-serial`（4题资格）已完成。
- `native-scores-serial`在用户中断时处于模型启动阶段，status为`failed / SystemExit(143)`，
  **没有任何request attempt或response**。原driver已退出；残留server已核对命令、启动tick、
  session/group后仅发送SIGTERM清理。
- 清理记录：`$H/interrupted-native-scores-cleanup.json`。记录时GPU无compute进程，无残留实验进程组。
- 全局`artifacts/value-recall-active.json`仍指向上述失败组。**接手时重新查实际进程和GPU，不能只信指针。**
- 没有定时巡检或自动重试。不要重跑失败/完成目录，也不要恢复MAB。

## 1. 固定控制与目标

### 固定控制

冻结fixture/profile是唯一参数依据：本地Pro6000、Qwen3.8-27B NVFP4，Active=16384、Gen=4096，
block32、sink41、recent0；prefill pressure按各层自己的累计attention选择；最终Query逐层Mean-K；
decode pressure统计所有已接受输出（含thinking）。thinking=4096，输出上限=16384，
temperature=.6、top_p=.95、top_k=20、seed=20260722。原生batch上限保持**16424**。

正式性能为8路完成补位。串行客户端仅用于固定顺序的诊断/等价，服务端配置仍保持原8路上限。
不能通过降低采样行、丢弃层、共享各层分数、减少候选、缩短输入/输出或改变Gen预算换取提速。

### 目标与停止条件

1. 降低Gen4K的**单次prefill重选服务成本**，同时减少它在并发下造成的排队。
2. 报告decode之外的有效prefill吞吐，不以题/小时或整体墙钟token/s代替。
3. 不承诺未经测量的收益。建议把“稳定超过基线波动、完整输入有效prefill提升至少10%”作为一轮
   候选的工程目标，而不是宣称已经达到；小于噪声的改动仅保留低复杂度、已有正确性价值的部分。
4. 任何选块/KV字节/完整输出首次分歧：保存证据，停止扩展该候选，定位输入状态与首次差异。
5. 降低prefill重选次数需要另设临时空间、核算显存，是**不同配置实验**，不混入本方案固定Gen4K主线。

## 2. 已知证据：哪些有用，哪些不能当结论

### 2.1 实际重选成本

`$H/profile-baseline-r2`：原8题完整输入、180次prefill pressure，基线已包含此前两项GIL缓存。
细分见`$H/host-baseline.json`。以下都是每次压力事件的平均包含时长，不能直接相加：

| 范围 | 平均耗时 |
|---|---:|
| 整次逐层KV组装 | 290.5 ms |
| 17次assembler构造/页表准备 | 23.3 ms |
| 34次packed读取提交、元数据与lease准备 | 67.3 ms |
| 34次packed读取`.result()`等待 | 141.7 ms |
| archive记录的两区域gather总时长 | 112.4 ms |

gather计时包含GIL重入和线程调度，不是纯memcpy带宽；`packed.result`包含其中一部分工作。
组装线程CPU时间约126.7 ms/事件，其余含等待/调度。还需细分，不能将wall-threadCPU全叫GIL等待。

**计时口径缺口：**`async_refresh.jsonl`的`assembly_queued → assembly_started`不是纯队列等待。
`layerwise_query.install_layerwise_query()`在真正`AsyncRefresh._assemble()`前先调用
`diagnostic.prepare()`，后者做逐层pressure选择；`assembly_started`是在它完成后才记录。
下一轮必须把“等待lane / executor排队 / 逐层选择”拆开，不能沿用旧名直接判断队列瓶颈。

### 2.2 attention没有重新算一遍QK

`$H/native-attention-audit.json`与实际`native_attention_stats.json`：

- 16个目标层；`reuse:prefill=13760`、`reuse:decode=7744`；fallback=0、validation failure=0。
- `native_attention_build.py`复用前向中的QK/online softmax统计并写piece-LSE。
- `native_attention.py:score()`通过`self.reduced`让每batch每层只执行一次`_row_lse`和
  `_segment_scores`；同batch多个owner只切片/索引已归约结果。
- `action_attention.py`仅在统计不可用或verify模式才调用`page_attention`参考计算。
- pressure读已接受历史的mass/exposure，不重放历史模型前向。

可优化的是剩余的统计写出、归约/D2H、累计状态维护与CPU选择组织；不是用全层共享分数替代逐层策略。

### 2.3 已实现候选

| 候选 | 状态 | 证据/边界 |
|---|---|---|
| 原slot编号缓存、模型module表缓存 | 先前已有 | 原串行4题一致，约3.85%有效prefill改善的小样本观察 |
| prefill pressure内联CPU paired区域读取 | **本轮已实现、4题模型等价通过** | 88次pressure、64个Query层、完整输出全部一致；GPU字节6项通过 |
| 从native累计状态直接导出scores | **本轮已实现、CPU通过；模型未完成** | 89项CPU通过；本地独立扩展，未覆盖安装二进制 |

内联读取的相同427136个历史prefill tokens：已保存GIL基线120.623秒 / 3541.10 tokens/s，
本轮117.807秒 / 3625.74 tokens/s，**观测约+2.39%**。不能把它算成已通过八路吞吐验收。
原有基线是之前保存的两个两题cohort，本轮是一个四题cohort；应在正式计时用同cohort/顺序复测。

native直接导出scores避免先产生每块`[mass, exposure]` Python列表，再做Python循环除法。
16层×640块的局部CPU测量1.91→0.61 ms，1664块6.05→1.72 ms；约省1～4ms，
**不是主要瓶颈**，不要单靠这个小改动期待大幅整机收益。

### 2.4 保留的完整八路观察

先前102题GIL优化组：2506.96秒、146.47题/小时；原Gen4K为2733.93秒、134.31题/小时。
prefill次数同为2318，但输出token为67886/72930、decode刷新0/3，八路选块和完整输出仍不同。
因此这些是归档观测，不能称为正确性合格的优化收益。没有外部评分。

## 3. 实施顺序与候选设计

### P0：完善归因；先消除“等待”含义不清

复用`$H/pressure_host_probe.py`和已有Nsight/py-spy结果；新增计时只放在**新冻结诊断副本**。
每个事件绑定`request_id + command_id + history + pressure_stage`，至少覆盖：

1. scheduler悬挂、assembly queued、获得lane、executor提交、worker真正进入。
2. `LayerwisePrefillPressure.prepare`；其中历史快照/分数导出、candidate映射、native picker。
3. `_assemble`原生plan提交、assembler/page-table准备。
4. 每层/每chunk的源admission、pybind/元数据、executor等待、实际gather。
5. staging slot等待、H2D提交、consumer kernel提交、最后CUDA fence实际ready。
6. CUDA已ready到engine主线程commit，再到scheduler ACK及下一次public forward。

每段同时记录wall和threadCPU；使用既有CUDA event或有限诊断event，不在正式计时增加同步。
先以当前GIL基线与内联候选做同8题诊断，核对是否真的省了线程交接，而不是把时间挪到别处。
额外统计：每事件H2D bytes、重建/skip tokens、CPU元数据读取次数、native调用次数、
prefill/纯decode/混合batch比例与实际Q长度；均不能简单相加为总墙钟。

**产出：**一张互斥边界的时间线、一张包含调用关系的成本表、真实任务数/字节数；确定下一项最大可消除成本。

### P1：完成当前内联读取候选的配对测量

实现文件：`storage.py`、`packed_assembly.py`、`layerwise_query.py`、`layerwise_pressure_efficiency.py`。
仅CPU常驻、paired、prefill pressure启用；Query/decode/异步prefetch/NVMe维持原路径。
保留相同准入、generation/source lease、锁释放、失败排空以及两个staging的DMA/consumer栅栏。

先对其做新鲜同cohort无profiler串行A/B，再做8路小cohort诊断性能；如果增益仅2%而复杂度明显，
不要继续堆叠这一路的小修，转向P2的大量重复元数据/准入成本。

### P2：每次重选准备一次immutable reader，跨层复用

目标是67ms/事件的CPU准备，以及34个读取中重复的handle→record、lease克隆、namespace/generation
检查、sources查找。优先复用现有`RequestMemory.admitted_read`、`PackedRead`、native payload read rights，
而非重新发明无所有权的裸指针缓存。

建议设计：

- 以**当前请求、当前native selection plan、accepted history、generation**绑定一次性reader；
  收集本次所有层实际选中完整块的去重集合，单次准备源映射。
- 每层仍用自己的原顺序/候选/selected集合，只从已验证映射取对应区域。
- 仅缓存元数据和不可变源引用，不额外复制完整KV；记录union大小与临时内存上界。
- 每个实际copy保持字节准入与目标staging独占；保留部分块和未归档块原路径。
- native GC路径要持有真正payload读权，不能仅靠metadata lease；缺乏证明时先限定现有no-GC配置。
- plan取消、提交、异常时精确释放；不能跨请求/重选复用。旧slot被回收复用必须拒绝。

可选第二小步：复用已验证的页表view/模块几何，减少每层`from_runner`与`.item()`校验。
必须来自**同次assembly**的父对象并保留页表、请求allocation lease及stream所有权；不做跨轮裸缓存。

**验证重点：**乱序和不同层集合、native/普通CPU source、stale generation、取消与retire并发、
部分块、两slot busy、失败copy/DMA。要求最终GPU KV全字节一致且scratch峰值不增加。

### P3：累计attention与逐层选择的CPU批处理

第一小步已实现`State.scores()`，先完成中断的模型资格，不必重新推理已完成的baseline。
独立编译二进制见`$H/native-scores-build/`；bundle为`$H/native-scores-candidate.json`。
workspace未覆盖原`.so`，直接跑工作区代码可能仍使用兼容fallback，测量必须证明加载的是新二进制。

后续重点：

1. 检查`LayerwisePrefillPressure.prepare_accept`逐层`committed`的调用和GIL交接成本。
   候选是一个native调用处理16个**互相独立**的状态，复用rows/starts/shape解析，保留每层history/mass。
   不能平均层分数、改浮点累加顺序或降低校验精度。
2. 检查共享layer-zero历史与独立layer-zero是否重复累计同一accepted packet。
   只有证明graph transaction、commit/abort、replay排除、checkpoint与历史边界完全一致后才复用结果；
   不直接让可变对象别名共享。
3. 合并一次重选内相同config/mandatory/candidate元数据，减少Python↔pybind往返。
   native picker要保留tie-breaking、block budget/partial block/mandatory语义。
4. 如改成native批量选层，先以固定分数、实际near-tie状态对原逐层picker逐项对照；
   性能报告区分scores导出、picker、frame commit。

**目标：**少做重复的数据组织与状态转换，仍从同一批accepted前向统计产生逐层结果。

### P4：CPU gather / H2D / assembly流水线

P0证明相应成本足够大后再做：

- paired K/V区域单次native gather，复用source解析、减少两次GIL释放/重新获得；保持field-major字节布局。
- 复用`PackedTransfers.prepare_from_reader/upload_prepared`和`PackedReadAhead`做最多一块lookahead，
  先测当前小区域复制是否值得异步；不要重新引入刚省掉的大量executor交接。
- 在现有两个slot中做到“CPU填下一块 / H2D当前块 / kernel消费前一块”可重叠。
  CPU slot受DMA完成保护，GPU slot受kernel consumer完成保护，两种完成不可混同。
- 衡量最后fence到主线程commit的迟延；可借鉴`poll_restore_completions`已有模式，仅在已证明CUDA完成时
  提早reap，不让后台线程提交authority/native frame，不提前释放lane或页面。

增加lane数量/更多pinned slots不作为第一步；要有显存/主存账本及证明没有跨请求写竞争。

### P5：attention统计kernel优化（实测后决定）

独立区分：前向FA2本体、piece-LSE侧输出、清理、row归约、segment归约、stack/D2H、CPU累计。
优先复用已有`action_native_prefill_static_aligned` / aligned / CTA gate的组件测试和隔离构建：

- `src/vllm_kvmem/serving/native_attention.py`、`native_attention_build.py`。
- `scripts/benchmark_native_prefill_static_aligned.py`、`benchmark_native_prefill_stats_floor.py`。
- `docs/native-prefill-static-aligned-20261022.md`、`native-prefill-stats-floor-20261022.md`。

旧组件的三采样层、大Q收益不等于当前16层、1024 chunk、8路混合batch收益。先用实际batch geometry
和sample rows重建组件fixture，再验证完整分数/attention输出/缺失producer检测及实际分支计数。

候选包括：减少重复metadata读取、证明安全后的紧凑统计写出、在严格保持原浮点顺序时融合辅助launch。
但以下都不是免费优化：跨不同prefill chunk复用旧累计、忽略partial/causal边界、移除stale-LSE清理、
换归约树/低精度、用未插桩attention当符合策略的基线。需要逐字节oracle和首次选块分歧定位。

## 4. 实验与验收矩阵

每候选独立新目录，先单项再组合，固定顺序如下：

| 阶段 | 要证明的事情 | 通过后才做 |
|---|---|---|
| 版本/控制审计 | 原冻结基线+最小补丁；fixture/profile、采样、模型、输入哈希相同 | CPU组件 |
| CPU组件 | 原分数/顺序/双精度位、选块；generation/lease/取消/失败回滚；预算不漂移 | GPU字节 |
| GPU组件 | K/V全字节、完整/部分/skip、乱序页、MTP、双slot复用 | 完整输入 |
| 固定序完整输入 | 原版自重复稳定，pressure/Query各层候选与选择、全输出token/text一致 | 性能小测 |
| 无profiler同cohort配对 | A/B或AB/BA、相同冷/暖条件；串行纯阶段和8路实际服务分别报告 | 102题正式 |
| 102题8路 | 输入完整、政策无回退、重选次数、phase分布、峰值资源、总推理墙钟；正确性独立报告 | 最终验收 |

新增的候选一限定prefill执行路径，decode继续旧路径。若未来P2/P3/P4/P5修改到decode，再单独增加
自然长输出刷新和accepted including thinking资格；三道历史刷新题串行不再触发刷新这一事实已保存，
不能通过反复采样、改seed或缩Gen强造覆盖。

八路完成补位导致batch形状可能变化。先证明原版同控制自身稳定性，若不稳定：
保存A/A差异，使用固定分数/KV组件及固定顺序完整输入定位优化语义；八路结果标为观察，
不能靠“都有答案/最终准确率差不多”宣布选块与回答一致。
不要盲目要求重复八路跑到碰巧一致；也不要以“硬件数值波动”直接豁免所有差异。

## 5. 性能口径

- **串行有效历史prefill**：同case `client attempt → 最终Query refresh start`，tokens用对应
  `history_begin`，包含HTTP/准入/pressure/组装/等待，不含最终Query和decode。
- **八路history-prefill envelope**：首个dispatch/attempt至最后Query start；包括其他owner重叠的
  Query/decode，必须明确标注，不能称纯prefill。
- **全推理墙钟**：客户端完整dispatch到所有回答、排空完成，不含模型加载/外部评分；
  同时报prefill tokens、accepted output tokens及MTP non-prefill work，避免生成量不同的归因混淆。
- kernel/sampled CUDA step速率仅用于定位，不能替代服务吞吐。
- 先检查StepTiming是否覆盖所有输入块。此前每cohort首个1024-token chunk没有记录；
  禁止拿不完整step时长除完整输入数。已有`public_work.jsonl`可以校准wall/monotonic时钟。
- 正式计时不带Nsight、py-spy、新host probe，也不要同时运行CPU微基准干扰GIL/内存带宽。

每张最终表至少列：版本/patch哈希、完整输入与采样哈希、已执行prefill/output工作量、
prefill/decode pressure次数、实际实现分支计数、选块/输出一致性、单次重选p50/p95、
prefill phase吞吐、总墙钟、峰值内存、异常/回退、测量重复数与范围。

## 6. 代码与产物定位

### 已有但未提交的推理改动

- 先前两项：`layerwise_pressure.py` slot缓存、`window_assembly.py` modules缓存。
- 本轮内联：`storage.py`、`packed_assembly.py`、`layerwise_query.py`、`layerwise_pressure_efficiency.py`。
- 本轮native score导出：`csrc/buffered_prefill_state.cpp`、`buffered_prefill_attention.py`。
- driver：`scripts/run_layerwise_memory.py`支持显式workers，默认仍8；此前新增audit/qualification工具仍在。
- tests包含此前模块缓存/slot回归、本轮inline ownership/GPU bytes/native scores以及两个旧测试fixture修正。
- `experiments/agent_execution_observer/`是独立用户工作，保留。

### 关键证据

- 旧完整组审计：`artifacts/g4k-gilopt-review-20261104-r2/final-review.json`。
- 旧串行prefill：同目录`serial-prefill-phase-comparison.json`。
- 本轮profile：`$H/host-baseline.json`、`$H/native-attention-audit.json`。
- 内联CPU：`$H/cpu-tests-verified.xml`（242 passed）。
- 内联GPU：`$H/components-inline/audit.json`（6 passed）。
- 内联模型：`$H/inline-serial/serial-equivalence.json`（4/4、88 pressure、64 Query layers）。
- native score CPU：`$H/native-scores-cpu-tests.xml`（89 passed）、`native-scores-microbenchmark.json`。
- 中断收尾：`$H/interrupted-native-scores-cleanup.json`。

### 可复用工具

- `scripts/audit_layerwise_gilopt.py`的`compare/compare_cases`可定位首个pressure/Query/output差异。
  它的旧runtime审计仅允许原两处GIL修改，**不可直接拿来豁免本轮更多文件**；扩展显式manifest再用。
- `$H/experiment.py`：原Memory driver+严格runtime manifest，prepare/run/verify，支持patch bundle、
  profile、固定ids、workers。新实验必须使用新目录。试验性控制器仍引用部分workspace分析helper；
  正式性能/最终冻结时补齐helper哈希或复制它们，不能只记录推理package。
- `$H/components.py`：已完成GPU组件，不要在同目录重跑。
- `$H/compare_serial.py`：对四个稳定原输出做精确比较及inline执行检查；输出采用独占创建。
- `$H/analyze.py`：host profile与prefill阶段分析。
- `$H/inline-prefill-candidate.json`、`native-scores-candidate.json`：最小补丁与目标文件SHA。
  前者不含native score，后者包含其Python wrapper/新`.so`和对应source-proof。

所有冻结版本都有原有original-RoPE支持差异。**应用补丁到对应冻结基线，不要把工作区整个文件复制过去。**
`git apply`在嵌套repo下可能静默跳过路径；工具已通过`GIT_CEILING_DIRECTORIES`隔离并核对实际产物SHA。

## 7. 下一位执行模型的第一轮任务

1. 读本文件，重新核对Git、GPU、进程、指针与中断收尾记录；不启动旧实验。
2. 给当前工作区/候选补齐正式source清单；保留所有未提交改动和独立experiment目录。
3. 若继续native score资格，在**新目录**重建相同候选（上一技术中断没有请求），只跑原四题一次；
   通过逐层/输出一致性才进入测量。不要把原失败目录改成成功。
4. 增加P0缺失的worker-enter/逐层选择/真实gather与commit-lag边界，在新8题诊断中核对主要瓶颈。
5. 按数据选择P2（复用source admission）或P3（批量累计）作为主优化，一次一个候选；
   不再把数毫秒的局部导出改进当成主要进展。
6. 达到完整输入一致性后，做同cohort无profiler串行AB/BA及8路小测，再决定是否值得完整102题。
7. 最终写清实际收益、未解决的瓶颈与边界。用户未授权的新外部评分不要启动；已有回答不能因分数重采样。
