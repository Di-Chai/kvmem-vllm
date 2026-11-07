# Gen4K prefill 重选热路径优化

本轮只沿用原模型、输入、Active=16384 / Gen=4096、block32、sink41、recent0、采样和
原生 batch 上限16424。实验根目录：`artifacts/g4k-pressure-hotpath-20261104-r1/`。
此前 GIL 两项缓存是本轮基线，不把它们的收益重复计算。

后续完整实施/测试方案：
[`layerwise-prefill-optimization-plan-20261104.md`](layerwise-prefill-optimization-plan-20261104.md)。

## 实测定位

`profile-baseline-r2` 是8个原始完整输入的诊断运行，180次 prefill pressure；
`host-baseline.json` 为主机细分计时（嵌套包含关系，不能相加当墙钟）。

| 单次组装均值 | 耗时 |
|---|---:|
| 完整逐层组装 | 290.5 ms |
| 17个 assembler 的构造/页表准备 | 23.3 ms |
| 34次 packed read 的提交/元数据准备 | 67.3 ms |
| 34次 packed read 的结果等待 | 141.7 ms |
| archive 报告的两区域 gather 总时长 | 112.4 ms |

CPU提交本身约67 ms，read-result等待约142 ms。底层gather已经原生复制并释放GIL，
外层却仍然把每个读取交给archive线程池，再马上`.result()`等待；一轮17层、每层2块，
产生34次任务及额外线程交接。gather计时包含线程被调度/GIL重入，不等于纯memcpy时间。
暂不将大部分时间归因于KV assembly GPU kernel或增加staging容量。

`profile-baseline` 首次启动因诊断模块不在server PYTHONPATH上失败，没有收到请求；
失败和清理证据保留。修正到冻结package内后，使用新目录`profile-baseline-r2`运行。

## 候选一：prefill pressure 内联CPU区域读取

仅在`assembly_arena=pressure`且`pressure_stage=prefill`、CPU archive、paired读取时，
在已有的组装生产线程执行原区域读取。它原本就需要同步拿到数据才能提交H2D，因此可以
省去“提交archive任务→立即等待”的往返。

- 原代际校验、namespace、CPU source/metadata lease、字节准入与完成计数不变。
- gather时不持有archive锁，原有原生复制/并行gather及失败排空逻辑不变。
- DMA仍使用同样的两个有界pinned/device staging slot及消费栅栏。
- decode pressure、最终Query、异步预读、NVMe路径继续原执行方式。
- 在`assembly_read_stats`报告真实`layerwise_inline_read_chunks`，避免只开标记却没有执行。

冻结候选通过对原冻结基线应用最小补丁构造；保留原`packed_assembly.py`的CUDA kernel，
不混入工作区original-RoPE支持差异。见`inline-prefill-candidate.json`。

CPU相关回归242项通过。初次广测暴露两项已有`io_usage`测试预期遗漏三个零值预算字段，
补全测试期望后通过；原失败XML保留。GPU组件6项全部通过：原双读、paired异步、paired内联
三种路径全部KV字节一致，包含乱序页表、目标层0/8、MTP层16、完整/部分尾块、跳过记录。
见`components-inline/audit.json`。

`inline-serial`已完成四题：88次prefill pressure、64个最终Query层、完整输出token/text全部一致；
实际执行2992个inline read chunks。相同427136个历史prefill tokens的阶段合计
120.623→117.807秒，3541.10→3625.74 tokens/s，约+2.39%的串行观察。
这不是八路正式吞吐结论；没有重跑完整102题。

## 累计attention/逐层计算核查

这条路径已经复用前向attention的QK/softmax统计，不是在每次重选重新计算QK。
新8题诊断的native统计：16个目标层，`reuse:prefill=13760`、`reuse:decode=7744`，
没有fallback或validation failure。`NativeAttentionStats.score`以`self.reduced`按层标记，
每batch每层仅做一次row-LSE和segment归约，多个owner复用该层已归约结果。
重选只读已接受输入累计的mass/exposure；不会重放历史模型前向。

仍有两种成本：原attention kernel内部写piece-LSE及后续归约/D2H；CPU接受历史与选择准备。
当前没有证据支持把每层的分数合并或跨batch复用旧结果；那会改变逐层累计策略。
已有static-aligned特化可复用，但其资格/形状与本轮1024-token chunk、混合batch不同，
不能直接引用旧组件收益或混进本候选。

## 候选二：直接从原生累计状态导出分数

`BufferedPrefillHistory.scores()`原先先展开所有`[mass, exposure]` Python列表再逐项除法。
新增原生`State.scores()`，直接对同一组double做同样的除法和正exposure过滤，保持key顺序；
不修改累计、衰减、采样、top-k或tie-breaking。旧安装扩展可继续原字典路径。

单独编译到`native-scores-build/`，没有覆盖运行中的扩展。89项CPU回归通过，包含64步累计、
float32/float64、strided/reversed数据、NaN拒绝、浅拷贝与checkpoint、score逐double-bit核对。
在16层×640块的局部CPU测量，score导出约1.91→0.61 ms；1664块约6.05→1.72 ms。
这只是导出成本，尚不能当模型吞吐收益；见`native-scores-microbenchmark.json`。

组合候选的`native-scores-serial`在用户要求转为制定交接方案时，被中断于模型加载阶段，
status为`failed / SystemExit(143)`，没有发送请求。已按已核实的命令、启动tick、进程组清理残留server，
GPU为空；失败目录保留。模型资格未完成，下一位执行者如继续需新建目录。
收尾证据：`interrupted-native-scores-cleanup.json`。
