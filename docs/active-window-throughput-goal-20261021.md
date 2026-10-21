# 相同 active KV 条件下的吞吐目标

目标：保留完整长历史、累计 attention、Query 图传播、重选、恢复和 MTP 语义，
让 KVMem 在相同 active KV、输入块大小、模型精度、编译和并发条件下，计入
统计、归档及重选的 prefill 吞吐达到原生 vLLM 的重复测量误差范围。
当前尚未达到或证实达到此目标，不把无法隐藏的必要成本省略出最终指标。

最新状态（10 月 22 日）：单请求 K32/G16、逻辑块 2K、物理最多 14K 的候选
已测完整 256K 平均 6097.49 token/s、重复长输入 6170.25 token/s。相同实际
active 长度和物理块形状的连续阶段为 KVMem 6503.30、原生6833.78 token/s，
仍低4.84%；完整32K同形状为9409.51/9928.10，仍低5.22%。连续阶段不包含
两端重选，不能替代完整256K吞吐。更大物理块的逻辑规则和固定Q/K组件已
通过，整模型输出/选页不同且没有完成广泛准确性验证，新开关继续默认关闭。
无采样 CTA 门控已通过逐字节资格，但完整三层组件仅降低 0.95%～2.15%，
不追加实模测量，继续默认关闭。同形状 14K 时间线已确认两个块多 105.07 ms，
其中三层 FA2 统计增 37.58 ms，编译器 capture clone/copyback 占 18.89 ms；
正在验证消除这些复制的新接口，尚无该候选的性能收益。
下文保留全部历次实验与负结果。

## 当前基线与已知限制

- main HEAD `8560331`，加当前工作区已有修改；不覆盖其他任务的未提交文件。
- 已完成的流水 ABBA：完整 262,045 输入，K=32,768、G=16,384、chunk=2,048，
  native QK reuse 两组均开启；4,576.63 → 5,147.11 token/s（+12.47%）。
- 原生从空缓存处理 32K 为 9,593 token/s（同 2K chunk）。它不能直接作为
  KVMem 持续处理 256K 的同工作量分母。
- 代码将传给 FlashInfer 的 `seq_lens`、页表和 `max_seq_len` 限制到 active
  window，未选中的历史不进入模型 attention。K 是重选后的保留预算；本轮
  active 随追加增长到接近 K+G。有效 causal 配对数之比不能当成整模型耗时比。
- 初始完整请求每次 145 个 prefill 提交，最新 14K 候选为 49 次。已建立匹配实际 Q 行数、active KV
  长度及是否为末块的原生模型对照；剩余成本继续按关键路径细分。

## 验证次序

1. 记录每个提交的实际调度 Q 行数、active KV 长度、逻辑历史位置和请求身份。
   使用现有 CPU 调度数据和已提交 window offset，不为诊断增加 GPU 回读。
2. 原生完整模型运行相同 `(q_len, active_kv_len)` 的追加 prefill，覆盖首次填充、
   稳态周期和部分尾块；按 KVMem 真实分布加权。该结果是模型工作量诊断，
   不冒充独立的端到端服务吞吐。校验计时插桩本身的扰动。
3. 最新代码补做相同完整 32K 输入的原生/KVMem 对照（没有压力重选），再做
   完整 256K KVMem，以分开正常 prefill 附加成本和长期重选成本。
4. 依据关键路径证据实现优化，独立开关、先验证固定输入及事务/生命周期，
   再冻结同源码交错对照。保持采样层、密度、完整归一化域和图策略不变。
5. 单请求验证后再检查并发；最终报告完整请求成本、匹配模型基线、输出及
   选页一致性、MTP 接受、内存和回退次数。未消除的差距必须单列。

## 本轮实施记录

- `StepTiming.batch_geometry` 增加每 owner 的精确 CPU 调度形状；保持非阻塞
  CUDA event 回收，未读取 GPU 张量。
- 新候选 `prefill_native_plan_arm` 默认关闭，仅在 publication pipeline 开启、
  native scores 启用且整批为完整准备的普通 prefill 时生效。用不可变计划
  准备原生统计，移除 forward 前的 metadata 等待；capture 阶段仍核验实际
  GPU 坐标，并保留原生分数校验、失败回退和事务提交。
- 修复 publication fence 构造使用 set 丢失原请求顺序的问题，改为有序 tuple。
- 已通过本轮计时/配置/策略 52 项检查，以及原生计划/回退/发布等 77 项检查；
  两组存在重叠，不相加为独立用例总数。

## 首轮匹配结果

产物：`artifacts/active-prefill-matched-20261021-r2/comparison.json`。
完整 32K 输入两组字节相同；原生有 3 次计时/3 次关闭计时的交错重复，
KVMem 有 2 次正式 32K、2 次正式 256K。实际并发均为 1，max_num_seqs=8。

| 指标 | 原生 vLLM | KVMem 流水基线 |
| --- | ---: | ---: |
| 完整 32K 输入 / TTFT | 9,576 token/s（关闭事件计时） | 8,325 token/s |
| 完整 32K TTFT | 3.422 s | 3.936 s |
| 匹配 256K 实际形状的模型 CUDA 区间 | 40.414 s（加权诊断） | 42.793 s（实测） |
| 完整 256K 输入 / TTFT | 不适用 | 5,165 token/s |
| 完整 256K TTFT | 不适用 | 50.737 s |
| 冷准备 | 63.10 s | 75.13 s |

原生事件计时使完整 32K TTFT 增加约 0.214%；关闭计时组仍保留诊断 dispatch
wrapper。KVMem 的 32K 有 18 次提交，原生完整 32K 有 16 次，因而这行是同输入
端到端对照，不是每一步形状相同。长输入的匹配诊断逐形状单独校验。

匹配形状后模型区间剩余差距为 2.379 s（+5.89%）。完整请求还包含平均
5.342 s pressure 挂起及 1.061 s Query 挂起；这些嵌套/重叠区间不能与模型
计时直接相加。加权 40.414 s **不是**原生完整 256K 的服务 TTFT 或实测吞吐。
两次正式长请求均有 14 次 pressure、1 次 Query、145 个公开 prefill 提交；
正式请求 native reuse 零回退、零 validation failure。完整 32K 的原生和
KVMem 输出分别为 114/101 token；完整长输入 KVMem 为 138，不能把不同轨迹
用于解码效率结论。

### 提前准备原生统计：没有测出收益

产物：`artifacts/active-prefill-plan-arm-20261021-r1/comparison.json`。
同一冻结 runtime，仅 `prefill_native_plan_arm=false→true`：

- 32K：8,325 → 8,258 token/s（−0.81%）。
- 256K：5,165 → 5,163 token/s（−0.04%），TTFT 50.737 → 50.758 s。
- 所有正式请求逐 token 输出、累计分数和图摘要、15 次选页及 mandatory、MTP
  接受和逐步计算形状相同；正式回退为零。没有导出所有图边作逐边比较。

该候选继续关闭，不能将消除一个同步点等同于端到端提速。

### KV 重建批量：首轮收益约 1.9%

- 全计划目的地验证将每层容量/RoPE 限额预聚合，每次 assembly 重新获取，
  错误路径仍按原层顺序报告。真实 native Plan 的 CPU 组件中位数
  12.51 → 0.494 ms（1,024 remap × 17 层），不等同于整轮收益。
- 现有 step timing 开启时增加 publication/selection 分点和 assembly 读取
  明细，不启用完整 Action profiler、不增加 GPU 同步。
- 新配置 `assembly_chunk_blocks` 默认 32，严格限定 1～256，固定每个 arena
  的 staging 容量。它与模型 prefill chunk 是两个独立参数。

产物：`artifacts/assembly-batch-20261021-r1/`。同源码，唯一参数差异为
重建分块 32→128，模型 prefill chunk=2,048、评分提交边界不变。
45 项 GPU/CPU 资格检查通过，10 个不适用配置跳过；覆盖完整/部分页、非连续
物理页、跨块 resident V 搬移、恢复、错误 fence 及两槽复用。新增成对 GPU
资格使用普通 CPU archive，不能扩展声称验证了 native-history GC 的并发租约。

| 指标 | 32 页 | 128 页 |
| --- | ---: | ---: |
| 完整 32K 输入 / TTFT | 8,270 | 8,285 token/s |
| 完整 256K 输入 / TTFT | 5,189 | 5,287 token/s |
| 完整 256K TTFT | 50.504 | 49.564 s |
| 暖轮每次 pressure assembly | 151.48 | 86.44 ms |
| 暖轮 pressure 总挂起 | 4.770 | 3.974 s |
| 256K 显存峰值，约 1 秒采样 | 72,500 | 72,706 MiB |

完整输入吞吐 +1.90%，暖轮 +1.89%；只有一对启动、每组两次正式长输入，
仍需后续交错重复。所有正式输出、15 次选页/mandatory、累计分数与图摘要、
MTP 和计算形状完全一致，正式回退零。实际上传字节未减少，暖轮重建块数
402→102，模型 CUDA 区间基本不变。多消耗的固定槽容量每个被使用的 arena
为 204 MiB pinned CPU + 204 MiB GPU，main/refresh 都使用时分别翻倍。

新分点证实，暖轮每次 pressure 仍等待归档发布约 102 ms，而真正选页约
14 ms。原生模型剩余差距与这部分等待需继续处理，尚未达到总吞吐目标。

### 下一组整合候选

共同使用已验证的 128 页重建批量，候选同时开启以下项；默认仍关闭：

- `prefill_capture_ahead`：全批合格普通 prefill 在原生验证前先提交私有 KV
  打包/DMA，验证和完整回退均保留，成功后才入发布队列；不跨 forward 读 K。
- `cpu_direct_record_seal`：复用已有单次复制归档路径。
- `cpu_seal_hash_workers=8`：共享已有有界 gather 池，每个已准入归档任务最多
  8 个哈希分片，保持原 SHA256 和记录顺序，全部结束后才发布/回滚；默认 1。

并行哈希的独立 68 MiB CPU 组件为 39.45→12.79 ms，但 CPU 时间增多，
不能将该倍数当成完整模型吞吐提升。253 项 CPU 回归及独立代码审查已通过。
随后 6 项 GPU 资格通过，覆盖真实 DMA、取消和异常后的槽复用。

整合实验 `prefill-overlap-seal-20261021-r1` 在预热验收时失败：启动时 NVIDIA
artifactory 连通性检查超时，vLLM 将 decode 从 XQA 改为 FlashInfer native，
出现 6 次 `backend_not_observed` 回退。该轮不能纳入性能比较，已确认服务和
GPU 进程清理完成。r2 保留相同冻结 runtime、profile、输入及 driver 重试，
不放宽原生复用验收；失败产物完整保留。

计时边界说明：上述 `model_ms` 包含 sampling、MTP、后续 capture/统计及
主机提交间隙，不能把 2.379 s 差额全部解释为 attention 内核额外运算。
三层采样的最后一层是 target 第 63 层，层内提前验证只能与最后层剩余工作和
MTP 重叠；不存在可隐藏几十层前向的窗口。继续用有界时间线诊断区分这些成本。

### 提前打包＋直接归档＋八线程哈希：长输入微升，短输入退化

`prefill-overlap-seal-20261021-r2/comparison.json`，同冻结源码、每组两次正式输入：

| 指标 | 共同基线（128页重建） | 三项整合候选 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,324.50 token/s | 8,063.20 token/s（−3.14%） |
| 32K TTFT | 3.936 s | 4.064 s |
| 256K 输入 / TTFT | 5,263.73 token/s | 5,330.50 token/s（+1.27%） |
| 256K TTFT | 49.783 s | 49.160 s |
| 256K model CUDA 区间（含 capture/统计） | 42.909 s | 43.110 s |
| pressure 挂起合计 | 4.386 s | 3.579 s |
| Query 挂起合计 | 0.957 s | 0.971 s |

全部正式输出逐 token、累计分数和图摘要、15 次选页/mandatory、MTP 和实际
计算形状一致，正式原生回退为零，实际 ahead/hash/direct 分支计数均通过。
归档等待减小，但正常 prefill 成本增加，因此不能据长输入的单轮小收益默认
采用。预备的 r3 反序重复尚未运行；先定位短输入退化和剩余差距，再验证最终
候选。Nsight 诊断独立于正式吞吐组。

### 同形状 Nsight：跨步重叠丢失是主要线索

`artifacts/prefill-nsight-matched-20261021-r1/pair-analysis.json`：同一完整 32K
输入和预热，捕获两组实际相同的四步，Q=2,048，active=16,384/18,432/
20,480/22,528。冷准备和完整预热均不计入该窗口。

| 诊断窗口指标 | 原生 | KVMem |
| --- | ---: | ---: |
| GPU 活动跨度 | 893.63 ms | 1,016.06 ms |
| GPU 活动并集 | 866.06 ms | 895.01 ms |
| 活动之间的空隙 | 27.58 ms | 121.06 ms |

差额约 76% 在空隙中。按 CUDA launch correlation 追溯工作来源，稳定后三步
的前 30 ms 中，原生仍有上一轮 12.33/12.92/13.49 ms GPU 工作，覆盖下一步
CPU 准备；KVMem 仅剩 1.22/0.46/0.79 ms。仅按 CPU execute NVTX 范围判断
空隙所在步骤会误判原因，前一轮尾部等待可以使下一轮准备暴露为空闲。

另从 OSRT 的 `take_gil` 栈确认，KVMem 主线程有 38.54 ms 明确 GIL 等待，
其中 19.06 ms 与 GPU 空隙重叠；其他无栈条件等待不能全部算作 GIL。
三层带原生统计的 FA2 内核合计比匹配原生增加约 12.51 ms，其余 14 层
基本一致，辅助归约合计约 0.47 ms。各项不能与重叠区间直接求和。

首个捕获步骤有额外扰动；这些是单次 profiling 的定位证据，不是正式吞吐
收益。下一候选研究在 MTP proposal 前提交完整评分、之后完成验证，让
MTP 尾部重新覆盖下一步准备；仍在下一 forward 前完成回退，并保留入库前
完整 GPU 错误屏障。

### 融合 KV 打包：组件通过，完整吞吐基本不变

`artifacts/prefill-pack-fused-20261021-r1/comparison.json`，同冻结源码，唯一
差异为 `capture_fused_pack=false→true`。使用两次正式 32K、两次正式 256K：

| 指标 | 原打包 | 融合打包 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,313.13 | 8,350.61 token/s（+0.45%） |
| 256K 输入 / TTFT | 5,246.96 | 5,251.34 token/s（+0.084%） |
| 256K TTFT | 49.942 s | 49.901 s |
| 256K model CUDA 区间 | 42.986 s | 42.881 s |

18 项 GPU 资格通过，包括全部 BF16 位模式、FP8 scale、真实多层布局、跨
stream 未完成来源拒绝复用、graph replay 和 DMA 槽复用。指针表只占固定
1,088 B CPU+GPU；不支持的布局显式走原路径，不无界增长。全模型实际分支
计数、输出、分数与图摘要、15 次选页/mandatory、MTP、计算形状均核验通过，
正式原生评分回退为零。但完整请求收益处于波动量级，不能因独立打包组件
快数倍就宣称总吞吐明显提升。该候选继续默认关闭，不重复扩跑此小收益组。

### MTP 前评分提交候选（10 月 22 日，首轮 +1.7%）

新增 `prefill_action_before_mtp=false`，仅全批普通非末块 prefill、同帧完整
计划、publication pipeline 和原生 reuse 合格时启用。MTP 前提交完整评分、
层平均和 D2H，记录唯一私有帧的完成事件；MTP 提交后在原 capture 入口完成
验证。失败仍用当前帧原 Q/K 做原参考回退，并等待回退完成。公开 packet、
实际接受确认、图事务、入库前 KV DMA 检查都保留。

私有事件失败永久隔离该帧，禁止再次等待“偶然成功”后释槽；MTP 或后续
submit 异常也禁止确认/发布。独立审查与整合 CPU 回归 104 项通过，原有
真实 FA2/XQA、FP8、GQA、MTP 和 graph replay GPU 回归 19 项通过。新增 14 项
GPU 资格通过，覆盖正常/缺失/NaN 原生分数、reference-only/no-row 完整屏障、
单槽回退、真实 DMA 取消、部分接受和连续 graph replay；模拟 GPU 尾部证明
早事件无需等待其完成，真实 MTP 及三层统计由完整模型继续核验。

同源码对照已冻结于 `artifacts/prefill-action-before-mtp-20261022-r1/`，仅
上述开关不同，融合打包仍关闭，重建批量仍为 128 页。两组完成并通过协议
及完整语义比较：

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,303.99 | 8,497.85 token/s（+2.33%） |
| 32K TTFT | 3.946 s | 3.856 s |
| 256K 输入 / TTFT | 5,277.44 | 5,366.77 token/s（+1.69%） |
| 256K TTFT | 49.654 s | 48.827 s |
| 256K model CUDA 区间 | 42.912 s | 42.876 s |
| 256K host submit 区间 | 43.815 s | 43.044 s |

正式输出、分数与图摘要、15 次选页/mandatory、MTP 接受和实际形状相同，
早提交实际执行计数一致且无取消/回退/隔离，正式原生评分回退零。这只有
一对启动、每组两次正式输入，最终候选仍需交错重复。模型 CUDA 区间基本
不变，收益主要体现为提交重叠；不能把原先观察的 12 ms 尾部全部视作已消除
的开销。正在用独立 `prefill-mtp-phases-20261022-r1` 诊断下一步 metadata
等待是否再次破坏重叠，考虑与此前单开无收益的 plan arm 组合验证。

### 第二个等待点：下一批 metadata 回读

`artifacts/prefill-mtp-phases-20261022-r1/phase-analysis.json` 使用同样四个
实际形状，加入阶段范围，仅用于定位。稳定后三步中，提前评分确实使上一批
MTP 和 KV 打包/下载延续到下一步 CPU 准备，持续约 13.4～14.9 ms；但
`Action.prepare` 又等待 metadata 回读 11.1～12.6 ms。这次等待几乎直到上一批
全部 GPU 工作结束才返回，重新暴露后面的 CPU 准备开销。KV D2H 已使用独立
stream；不能再将它描述为尚未异步化。小 metadata D2H 紧随上一批 71.3 MB
KV D2H，独立 stream 仍共享实际传输资源。

因此组合验证此前单开无收益的 `prefill_native_plan_arm`：将坐标回读等待
推迟到同帧评分入口，允许先提交目标模型前向。实际 GPU 坐标检查、原生分数
验证、异常隔离及接受确认均保留。组合 CPU 检查 18 项、真实 FA2/两槽复用/
坐标失败 GPU 检查 3 项通过。测试包装器曾因局部类闭包持有 event 到循环 GC
而耗尽两槽，已改为实例字段并加入即时释放回归；未扩大池或修改生产回收逻辑。

同源码完整请求对照已冻结到 `artifacts/prefill-mtp-plan-20261022-r1/`，两组
均提前评分，仅 plan arm 开关不同，已完成并通过全部正式一致性检查：

| 指标 | 提前评分 | 提前评分＋plan arm |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,536.70 | 8,519.40 token/s（−0.20%） |
| 256K 输入 / TTFT | 5,353.89 | 5,401.50 token/s（+0.89%） |
| 256K TTFT | 48.945 | 48.513 s |
| 256K model CUDA 区间 | 42.888 | 42.897 s |

正式输出、分数/图摘要、15 次选页/mandatory、MTP 和实际几何一致，正式原生
回退为零；plan arm 和 early stage 真实计数通过。只有一对启动，32K 无明确
收益，256K 有小幅收益，不能把后移的等待时长全部当成净提升。

CPU 还存在一段可优化残余：四次 `action.prepare_accept` 在原生构图返回后
合计约 7.39 ms，主要候选是 GraphRow 构造与逐对象递归内存计费。此 prefill
路径没有复制全部分数字典，`multi` 也没有重复执行两遍构图。独立 CPU 组件
进一步测得每包构造 0.031 ms、递归计费 0.370 ms、元组拼接 0.002 ms；说明
trace 长尾还包含线程竞争与诊断扰动。平铺计费原型保持逐行预算字节相同，
只省约 0.19 ms/包（256K 约 0.03 s 的本体上限），暂不改生产。不能据长尾
区间宣称可消除 1～2 ms/步，也不能将所有 GIL 等待都归因于该段。

### 整合对照（已完成 ABBA）

`artifacts/prefill-integrated-20261022-r1/` 已冻结同源码。共同基线为原生 QK
复用、publication pipeline、128 页重建；候选合并提前评分、plan arm、提前
KV capture、融合 pack、直接 seal 和 8 路哈希。此前 ahead 整合让短输入退化，
但 early score 的事件现在位于 MTP 前，不会再因 ahead pack 重录到尾部；这一
组合须以新的完整结果验收，不能相乘以前各组百分比。

真实 FA2＋实际 CaptureRuntime＋KV DMA＋CPU archive 的 4 项 GPU 组合检查
通过：native/reference 回退的页分数和归档字节一致、scheduler 确认前不发布、
score/DMA 事件失败隔离、固定槽及 fused pack 共存。5 行增量仅封一页，因此
该增量只验证直接 seal；8 路哈希的并行执行由前缀归档和完整模型计数分别核验。

同一冻结源码完成 r1 关闭→开启、r2 开启→关闭的 ABBA。每长度、每组四次正式
输入；总 token / 总 TTFT 汇总见
`artifacts/prefill-integrated-20261022-r1/abba-comparison.json`：

| 指标 | 共同基线 | 整合候选 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,355.81 | 8,628.07 token/s（+3.26%） |
| 32K TTFT | 3.92158 | 3.79784 s |
| 完整 256K 输入 / TTFT | 5,271.02 | 5,506.94 token/s（+4.48%） |
| 完整 256K TTFT | 49.71430 | 47.58454 s |

首次长输入和重复长输入分别 +4.46% 和 +4.49%。8 对正式样本的输出、评分/图、
选页/mandatory、MTP 和形状检查通过，原生回退及验证失败为零；每次服务冷准备
75.15～76.16 s。GPU 约一秒采样的长输入峰值为 72,706→72,730 MiB，不能据此
声称显存降低；CPU RSS 未采样，未声称已完成 CPU 峰值内存验收。

独立整合阶段诊断 `artifacts/prefill-integrated-phases-20261022-r1/candidate/`
确认 prepare 四步 CUDA 同步为零，稳定后三步 prepare 约 1.03～1.09 ms，
与上步 MTP 尾部重叠。四步 pack GPU 区间由 5.042 降至 0.471 ms，D2H 仍约
5.334 ms。稳定后三步 GPU 空隙为 42.965 ms（旧 early-only 51.889、原生
19.179）；首个 trace 步有明显扰动，不能用整个 trace 反推正式吞吐。
验证的 184.827 ms CPU 等待主要覆盖 GPU 工作，与未提交空隙交集为零，
不能当作可消除成本。剩余未提交空隙主要落在普通模型提交区域，部分与 GIL
等待重合。下一候选将校验移至后台并冻结回退输入，检验能否进一步提前后续
模型提交；其完整对照结果见下节。

### 后台原生校验：正确性通过，但完整请求退化

`artifacts/prefill-deferred-validation-20261022-r1/` 冻结相同源码、相同整合
配置，仅改变默认关闭的 `prefill_deferred_validation`。两槽固定池冻结采样
Q、坐标、段映射、页表和 scale；后台验证原生输出，必要时在仍受 packet
生命周期保护的 KV 前缀上执行原参考评分。完成验证和预算预留后才能归档及
发布；跨流失败永久隔离，原生抢占不能绕过物理页保护。

144 项组合 CPU 检查和 21 项 GPU 检查通过，覆盖真实 FA2/FP8、部分页、
非连续页表、多 owner、后台线程、缺失分数回退、固定双槽、graph replay、
来源覆盖及事件失败。实模 303 帧冻结/验证/释放闭合，正式回退和隔离为零；
输出、评分/图、选页/mandatory、MTP 和实际几何核验一致。

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,600.26 | 8,515.68 token/s（−0.98%） |
| 32K TTFT | 3.81012 | 3.84796 s |
| 完整 256K 输入 / TTFT | 5,496.27 | 5,386.96 token/s（−1.99%） |
| 完整 256K TTFT | 47.67687 | 48.64428 s |
| 256K model CUDA 区间 | 42.68740 | 43.46230 s |

每组两次正式输入，只有一对启动，未将退化候选扩大成 ABBA。冷准备
76.15→75.14 s。池实际 GPU 常驻 19,426,800 B、pinned 常驻 290,072 B，
合计 18.803 MiB；另有 67.008 MiB scratch 预算预留，均在固定上限内。
前台 host submit 缩短，但 host gap 同步增长；这些区间包含移位后的 GPU
等待，不能相加或据此认定提前提交已带来加速。额外复制与调度成本未被
有效隐藏，因此该开关继续关闭，不加入最佳整合配置。

独立 trace 进一步把稳定后三步的主要差额定位到 embedding 与首个目标
FA2 之间（包含前三个 GDN）：原生 7.133 ms、整合 29.131 ms。部分未提交
空隙与 Python GIL 等待重合，但尚不能全部归因于后台归档。下一候选仅在
普通非末尾 prefill、一个已确认前序 packet 和可用固定槽等严格条件下，
把新归档 worker 的启动延至目标模型提交后；不会取消已运行 worker、越过
发布依赖或改变评分/归档内容。此项尚无性能结论。

### 剩余阶段的进一步核验

- pressure 后的 14 个 32-token 小块是此前暂缓的新 prompt，145 个公开输入
  区间连续覆盖 `[0,262045)`；不是重复计算旧历史。它们的 KV 必须正常归档。
  小块因评分间隔而没有采样，但当前 early stage 仍为其创建完整事件；真实
  on/off 对照只观察到约 18～31 ms/请求的此类小块区间增量，不能把先前约
  0.31 s 的形状匹配差额全归给它，也不据此承诺大收益。
- plan/MTP 候选的 Query 挂起首次为 1.244 s、重复为 0.667 s。首次私有
  22-token 前向出现三项推理期 JIT；32K 预热没有覆盖历史超过 K 的私有
  Query 路径。私有前向自身的 host 区间为 741.78→147.91 ms，未单独测编译
  子区间，因此不能将全部差额精确称为 JIT 时间。
- 重复 Query 的 selection 为 340.10 ms，其中包含式 graph_selection
  39.55 ms，C++ V2 multi 构建/传播为 15.44 ms。其余约 300 ms 还包括
  8,188 条索引的排序/pin、约 272 MiB MeanK 拼接、约 256 MiB 上传、有效页
  收集、GPU MeanK 评分、结果回读及最后 picker/unpin；现有计时无法再拆分，
  不能都归给 Python 或 GPU。选页和重建已在后台执行，但单请求的下一块须
  使用完成重选后的窗口，不能越过这个依赖。

未改变生产默认配置；未启动 DSH；完整 RULER/MAB 仍暂停。本文件随目标进展
更新，具体性能以冻结实验产物为准。

### 前台 target 提交后启动后台归档（正确性通过，净收益不明确）

新增默认关闭的 `prefill_scatter_after_target`。仅允许恰好一个 confirmed、
未消费的 bulk 包、固定双槽仍有空位、同批所有 owner 的连续不可变规划匹配，
且没有重选/恢复/末块/新请求/混合 decode 或其他生命周期操作时延期。已经
运行的 worker 不暂停；准备后再次复核，失配恢复原启动时机，旧包已同步发布
则以 `no_work_batches` 收尾。原生 target 完整提交并产生当前批次的新状态后，
才启动 worker；取样、MTP 和下一包确认保持原顺序。外层异常保护覆盖 connector
准备到 target 之后，失败保持源数据所有权并隔离。

135 项 CPU 回归和独立审查通过，含完整 runner 入口、多 owner、固定双槽、
真实 CPU archive、FIFO、确认及异常。2 项 GPU 资格使用实际 FA2、原生/参考
评分、pack/DMA 和后台 archive；延期前后评分与归档字节一致。GPU 组件用固定
successor 规划并额外提交真实 FA2，仅证明所有权与调度，不冒充完整两块模型。
另有 13 项 benchmark 审核检查，要求延期计数与启动/恢复/无工作计数闭合。

`artifacts/prefill-scatter-after-target-20261022-r1/` 冻结同源码和最佳整合配置，
后台校验关闭，两组只改变此开关。完整 32K/256K 对照已完成，每组每长度两次：

| 指标 | 原启动时机 | target 后启动 |
| --- | ---: | ---: |
| 32K 输入 / TTFT | 8,614.18 | 8,636.50 token/s（+0.26%） |
| 完整 256K 输入 / TTFT | 5,520.14 | 5,533.16 token/s（+0.24%） |
| 完整 256K TTFT | 47.47072 | 47.35901 s |
| 256K model CUDA 区间 | 42.65678 | 42.17904 s |
| 256K host between submissions 区间 | 4.66912 | 5.07775 s |

233 次延期与 233 次 target 后实际启动闭合，无隔离、无提前恢复/空任务。
正式输出、评分/图、选页/mandatory、MTP、形状及原生零回退核验均通过。
冷准备 75.14→75.15 s。model 区间缩短约 0.478 s，但 between-submissions
增长约 0.409 s，完整 TTFT 仅省约 0.112 s；这些计时的边界与嵌套仍不可任意
相加。结果主要表明等待移位，不能据局部区间宣称明显加速。单对启动的
0.24～0.26% 属波动量级，不扩跑、不加入最佳配置，该开关继续默认关闭。

### 原生 FA2 对齐分段快路径（组件小幅改善，默认关闭）

新增 `action_native_prefill_aligned_pieces=false`。启用服务固定使用一个新
FA2 模块，每帧仅更新有容量限制、地址固定的 owner 标志；完整可见逻辑分区
内部边界全部按 16 对齐、tile 基址对齐且无非对齐 split-K 时，piece 直接取
`base/16`，省去通用分段映射读取。部分 causal 尾页仍合法，其他情况沿用原
通用路径。causal mask、FP32 加法/shuffle/log 顺序、FP8 scale、全行归一化
及 heads 平均均保留；默认 FA2 ABI 和 XQA 结构布局不变。8 owners 新增
32 B GPU 与 32 B pinned，计入启动容量检查。

93 项 CPU、新增 7 项 GPU、原有 19 项 GPU 通过，独立审查无阻断；覆盖
新旧逐位一致、同 Graph 对齐/owner 切换、混合 decode 前缀、部分页、NaN
及真实非 16 split-K。编译和测试总时长不是纯内核性能。

`artifacts/native-prefill-aligned-20261022-r1/` 保存三层完整组件结果及源码。
Q=2048、64 采样行、24 Q/4 KV heads、dim256、FP8 KV、kernel 页16，bank
容量精确匹配 context65576/segments2202/width6301/rows256/owners8。每组
12 次轮换，完整组件计入 arm、标志清理/上传、三层 attention、归约、层平均
和 pinned 下载；不含接受提交、归档或构图。新旧页分数逐位相同，回退零。

| 完整 active（含当前 Q） | 同源码旧 ABI | 新模块通用分支 | 新模块对齐分支 |
| --- | ---: | ---: | ---: |
| 16K | 16.534 ms | 16.991 ms | 16.079 ms（−2.76%） |
| 32K | 32.756 ms | 34.051 ms | 32.304 ms（−1.38%） |
| 48K | 49.264 ms | 50.183 ms | 48.352 ms（−1.85%） |

32K 通用分支有 40.923 ms 离群，median 33.411 ms 仍慢于旧 ABI 的
32.749 ms。三层纯 FA2 kernel 的独立 profiler 均值分别为旧版
15.974/31.512/47.123 ms，对齐版 15.482/30.343/46.139 ms；它们来自另一
轮带 profiler 的采样，不能与上表 wall 相减归因。这里旧 ABI 也是当前
source overlay 编译的模块，没有与历史冻结旧 header 的 binary 对比。

对齐分支每块仅省约 0.45～0.91 ms，且通用分支变慢；当前不单独扩大整机
实验、不宣称模型收益，仍默认关闭。下一步先用独立稳态 pressure trace
找出阻塞下一块 prefill 的更大成本。

### 稳态 pressure 的阻塞链

`artifacts/prefill-pressure-phases-20261022-r1/`：最佳整合配置，先完成完整
256K 预热，再捕获第二次相同请求的逻辑输入 `[40960,63456)`；只包含历史
49,088 处的一次 pressure（command 16）。输出分数、实际形状、阶段完整性、
H2D correlation 和重建内核资格均通过。GPU 测量已结束并清理。详细可复算
输出为 `prefill-pressure-attribution-20261022.json`，不能当作正式吞吐组。

这次挂起共 223.205 ms，GPU 活动并集 35.407 ms，GPU 空闲 187.798 ms。
按不重叠的生命周期分点观察：

| 阶段 | 墙钟时间 |
| --- | ---: |
| 挂起至归档发布完成、组装入队 | 70.468 ms |
| 组装线程排队 | 0.965 ms |
| 等待 seal 结束并完成选页 | 29.114 ms |
| 准备和提交重建 | 111.687 ms |
| 提交后确认、commit/close/report 至 ready | 10.832 ms |

最前一段包括上次 forward/DMA 尾部约 18.90 ms，随后有接受评分/构图准备
约 7.99 ms、历史 scatter 约 7.91 ms、MeanK finalize 约 25.89 ms。
哈希的 13.62 ms 在另一线程执行，其尾部被下一阶段的 publish_sealed 包含；
不能另加在同一段上。最后重建内核在相对挂起约 211.11 ms 时已结束，因而
最后的 10.83 ms 主要是主机完成处理，不能归为 GPU 重建尾延迟。

组装读取 931 条完整历史记录的 raw-K 区域，共 518,619,136 B，V 已从当前
resident 窗口复用。实际为 8 个 CPU 读取分块、136 次重建调用（8×17 层），
不是 136 个数据分块。8 次 read_packed_records 共 42.73 ms，其中 native
gather 31.23 ms；这些读取区间 GPU 均闲置，当前双槽没有遮住这部分读取。
逐层重建 Python 调用共 22.37 ms，GPU 重建本身仅 1.90 ms；两者重叠，不能
相加。首份 GPU metadata 提交前的 CPU 准备为 18.36 ms，其中发现 resident
页对所有层容量的重复校验。

下一候选分别消除重复容量校验、保持逐通道 token 累加顺序的 MeanK SIMD，
并先用纯 CPU 组件核验 gather 工人数。不会将上述单次插桩区间直接乘以
pressure 次数承诺整机收益，也不会把已发生的 GPU 等待简单挪走算作加速。

### CPU 准备与读取组件

resident 校验已改为每次组装先取各层最小容量，再按原页顺序检查 source
extent；只读取 shape/numel，不回读 GPU。41 项相关 CPU 检查通过，包括
非均匀层容量、部分 extent、换页表后的重新计算和写入前失败。实际 931 条
native remap、17 层 metadata 的 6 轮 ABBA 组件为 6.796→0.274 ms，省
6.522 ms。产物 `artifacts/resident-validation-20261022-r1/`。这是单段 CPU
组件，不能说已消除 trace 的整个 18.36 ms 准备区间。

`artifacts/pressure-cpu-gather-20261022-r1/` 进一步测了真实 931 条记录和
8 个读取批次，保持相同 node0 首次触页的源与双目的地，8 轮交错：

| 执行 CPU 范围 | 1 worker | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| 全 64 逻辑 CPU | 59.18 ms | 44.32 ms | 34.94 ms | 33.10 ms |
| GPU 所在 node0 | 59.31 ms | 44.83 ms | 35.35 ms | 33.69 ms |

上表是完整 read_packed_records，8 workers 的纯 gather 分别为
22.30/22.42 ms。所有读回字节一致，native 库哈希全程一致。当前 8 workers
已是这组最优，暂不更改线程数或绑核。该组件使用预触页 pageable CPU 目的地，
没有 CUDA、pinned DMA 或运行中的引擎；不能把与真实服务 42.73 ms 的差额
算作收益，也不能据此对所有 NUMA 内存布局作一般结论。

### MeanK FP8 向量化（组件更快，实模无收益）

新增默认关闭 `cpu_mean_index_simd`。只在 qw3_cuda MeanK 的完整已接受页
批量 finalizer 中启用；用 AVX2 对不同通道并行，E4M3 数值精确拓宽到 FP32，
每通道从 +0 按原 token 顺序相加，后续除法和 FP16 索引转换保持原路径。
BF16、短 channel、source/output alias 或没有 AVX2 时保留标量路径；所有
job 在任何写入前完成校验，buffer exports 在释放 GIL 期间持有来源。
实际 SIMD/回退页数由 native 返回，并用独立小锁累计多 owner 指标；没有
新增线程或 GPU/pinned 大缓冲。

77 项 native/索引 CPU 检查通过，含完整 FP8 字节域、同通道正负 NaN 两种
次序、±0、partial/stride/alias、多个 owner 和异常校验；Python 配置、
部分接受、发布及分支审计组合另有 98 项通过（与其他检查可能重叠，不相加）。
resident 容量改动的 2 项真实 GPU 重建/恢复检查也通过，FP8 K/V 字节一致。

`artifacts/cpu-mean-index-simd-20261022-r1/` 保存各版编译和组件证据。
同一库、64 条 32-token 记录、17 个 raw 层/16 个评分层、4 KV heads×256，
30 轮交错：最后的精确位解码版本 14.446→7.411 ms，组件耗时 −48.7%。
较早单累加器 gather 版本慢 6.4%，四累加器 gather 版本只省 1.86 ms；
它们没有进入实模，不能择优遗漏失败实验。

`artifacts/prefill-mean-simd-20261022-r1/` 已冻结同一 native 库、Python/CUDA
与 C++ 源码，两组只变更此开关，完整 32K/256K 各两次已结束并清理 GPU。
四对输出 token、选页/mandatory、图报告、接受评分/图摘要、MTP、计算几何
均一致；候选 19,467 条完整页全部实际走 SIMD，回退零。

| 完整输入 | 参考 | SIMD | 吞吐变化 |
| --- | ---: | ---: | ---: |
| 32K | 8,584.66 token/s | 8,545.29 token/s | −0.46% |
| 256K | 5,513.62 token/s | 5,501.46 token/s | −0.22% |

长输入平均 TTFT 47.527→47.632 秒，首个长输入 48.129→48.127 秒，第二次
46.925→47.136 秒。CPU 组件提速没有变成完整服务收益，因此保持默认关闭，
不扩大 ABBA 或将它加入最佳配置。CPU 组件约 7 ms 节省有一部分原本与模型
重叠，不能直接乘块数估算 TTFT 收益。模型 CUDA、主机提交和重选区间也重叠。

冷准备分别 155.280/75.142 秒；参考首次编译新的 GPU overlay，两组都在
首次长输入中遇到重建内核首次使用。冷准备差异不是 SIMD 收益，长输入重复
结果仍未显示改善。没有新增线程、大缓冲或 GPU/pinned 内存容量。

### 历史读取单块预取（资格通过，组件无收益）

`assembly_read_prefetch` 默认关闭，仅在 CPU archive、direct packed reads、
无 raw-K GPU cache 的完整页 split assembly 中工作。当前块上传之后，使用
原有第二个 pinned 槽及 archive executor 提交下一块读取，再提交当前块的
逐层重建；最多提前一块，槽或准入忙时直接沿用同步路径，不等待整个 batch。
GPU 目的槽在真正消费时重新取得，并分别保留 CPU fill、DMA 和 GPU consumer
的完成条件；不新增线程或大缓冲区，不跨 owner/plan/generation 复用读取。

新增任务提交门避免 executor 入队后抛错留下隐藏写入者；job holder 构造失败
也不会触碰已回滚的来源。当前层失败时先 drain 自己的预取，再释放 plan
来源并走原事务失败路径。DMA 提交与 failure fence 双失败会永久隔离缓冲池，
保留端点引用。CachedArchive 的 read 入口同步复用了门；旧 SSD cache gather
路径没有扩展验证，此候选也不进入该路径。

配置、实际分支审计、归档故障、双槽生命周期和目的地校验组合 101 项 CPU
检查通过。独立复审无阻断。`artifacts/assembly-read-prefetch-20261022-r1/`
冻结实现和真实 GPU 检查，2 项 GPU 检查通过：301 个选择页、3 个完整读取块
加 7-token 部分页，涵盖非连续物理页、跨块 resident V 重排、冷恢复与延迟
CPU writer，开启/关闭的 K/V 字节一致；每次完成 2 个预取，无 abandoned。
随后使用 trace 实际选择的 1,024 页（93 skip、931 rebuild）、17 层×4 heads×
dim256、8 个 128 页分块，完成 2 轮预热和 8 轮交错组件。计时包含构造、读取、
metadata、H2D、重建、完成等待和释放；归档已 ready，不包含之前的发布/选页。
开启前后每次上传相同的 518,619,136 B，所有 17 层物理 K/V（含 skip/未用页）
逐字节一致，候选每次实际完成 7 个预取，没有忙槽、准入失败或 abandoned。

完整 wall 中位 46.155→46.965 ms（慢 1.76%），均值 45.700→47.073 ms（慢
3.00%）。没有发现收益，不进入单独实模、不启用默认。两组各自使用原来的
285,212,672 B pinned/GPU 双槽及 518,619,136 B resident V 暂存。
`component-summary-corrected.json` 修正初版摘要字段名错误：读取排队中位
0.525→1.150 ms、native gather 24.978→29.612 ms，原始逐次明细不变。
这些区间有重叠，不能加到完整 wall；没有新增大缓冲也不保证异步一定加速。

### 连续 active-window 阶段的实际参照

`artifacts/active-prefill-continuous-20261022-r3/` 使用当前最佳六项整合配置、
128 页组装，MeanK SIMD、读取预取、跨层融合、deferred validation、
scatter-after-target、aligned-pieces 均关闭。只增加可核验的连续事件关联：
每个计时器保留上一已完成 end event，不新增 CUDA event 或同步，用 9 个
model 区间加 8 个内部 gap 重建同一请求的连续跨度。call index 检出采样/
丢帧，校验 owner、offset、逻辑位置、非 terminal 和非 replay。

真实稳态阶段是先已有 32,768 KV，再追加 `32 + 7×2048 + 1952`，共 16,320
新 token，最终 active 为 49,088；第一块前和最后一块后的 pressure 均排除。
原生 3 次实测，KVMem 两个完整长请求各有 13 个匹配阶段；阶段样本共享所在
请求，不能当成 26 个独立服务重复。

| 口径 | 原生 vLLM | KVMem 当前最佳 |
| --- | ---: | ---: |
| 连续上述 16,320 token 阶段 | 6,197.87 token/s | 5,835.17 token/s（低 5.85%） |
| 完整 32K 输入 / TTFT | 9,564.08 token/s | 8,572.22 token/s |
| 完整 256K 输入 / TTFT，2 次合并 | 不适用 | 5,494.60 token/s，47.691 s |
| 重复长输入 / TTFT | 不适用 | 5,560.41 token/s，47.127 s |

原生三个连续阶段分别 6,201.95、6,194.50、6,197.18 token/s；每段内部 gap
总计约 0.51 ms。KVMem 两个请求的连续阶段分别 5,854.00 和 5,816.46，
每段 model 平均 2.778/2.796 s、内部 gap 9.877/9.477 ms。model event 区间
仍含主机发射间隙、MTP 与 capture/统计，不能解释成纯内核时间；正常 prefill
剩余差距不是仅由重选造成，也不能把约 10 ms 的内部 gap 当成全部差距。

原生完整 32K 开启计时为 9,573.39、关闭为 9,564.08 token/s（−0.097% 的
关闭组差异，处于本轮噪声量级）；两者仍保留轻量 dispatch wrapper。完整
32K 的原生/KVMem 分别 16/18 次提交，仍需区分同输入和逐步同形状参照。
最新形状加权原生 model 为 40.457 s，与此前 40.414 s 接近，仍不是完整
256K 原生 TTFT。冷准备分别 65.114/76.154 s，未计入推理吞吐。

全部服务已退出，GPU/owned processes 清理为空。跨 runtime 语义回归四对
正式请求的输出 token、选页/mandatory、图报告、接受分数/图摘要、MTP 和
几何均相同；正式 native 复用回退与验证失败均零，针题评分均 100。
`audit_semantics.py` 与 `semantic-regression.json` 保留校验字段和源码指纹；
这是语义回归，不冒充不同源码间的单项优化 A/B。当前尚未达到吞吐目标。

### 跨层批量重建：组件收益未转化为完整吞吐收益

`assembly_fused_layers=false` 为默认；可选路径使用固定层指针/stride 表，把
同一读取块的 17 次重建提交合并为一次三维 kernel。保持原 FP8/旋转计算及
resident V 搬运，部分页或未资格布局在写出前回退，表的 GPU+pinned 总容量
为 3,264 B。表绑定 owner、plan 和 restore generation，消费结束前不复用。

`artifacts/assembly-fused-layers-20261022-r1/` 冻结源码与组件结果：12 项 GPU
检查通过，覆盖异构 strides、部分页回退、非连续页、resident 重排和恢复。
真实 pressure 几何的 8 轮交错组件中，完整组装 wall 均值
52.510→48.354 ms（−7.92%），136 次逐层提交变为 8 次，全部物理 K/V 字节
相同。预取一起开启为 48.778 ms，没有进一步改善均值。

完整模型 `artifacts/prefill-fused-assembly-20261022-r1/comparison.json` 只改变
这一开关。四对正式请求的六类语义核验全部相同，正式原生评分回退/验证
失败为零。实际 220 个融合块取代 3,740 次层提交，融合回退零，metadata
容量上限符合预期。32K 无重选，仍有 8,454.60→8,584.81 token/s（+1.54%）
的组间变化，因此该变化不能归因于重建融合。

256K 首次长请求 51.460→48.206 s，重复长请求 **47.255→47.330 s**，
重复吞吐 **5,545.31→5,536.59 token/s（−0.157%）**。首轮额外准备/首次
编译不同会影响合并值，合并 5,309.09→5,485.82（+3.33%）不能当作稳定
收益。冷服务准备 75.143/76.152 s 单列，两组完整清理为空。组件约 4 ms
节省没有转化为可辨认的完整吞吐收益，不追加大规模重复，不启用默认。

### 保留逻辑 2K 事务、物理 4K 合并候选

新增默认关闭 `prefill_physical_coalescing`，服务端的
`prefill_chunk_tokens=2048` 保持不变。首版只合并一个 owner 的两个完整普通
2K 块：分别执行原 native lookahead 与 RequestPlan.action/admit；首个
forward、Query/检查点/pressure 边界、部分逻辑页、恢复中请求和容量不足
保持原分块。禁止把原 admission 裁剪成新的 3K 统计事务。

Action 逐次使用原 PrefillPlanFrame.extend 计算采样行和完整可见页域；原生
4K 只取这两份采样的并集。CPU 子视图去掉未来列，保留同一有界下载租约。
原生失败时分别用原 2K 的 length/segments 执行 reference，之后再补零并
组合；没有改变原 reference 的 softmax 宽度或每层平均顺序。

归档使用单个物理包租约和逐子块游标，每次原 Graph 预算预留成功后才追加
原 2K KV 并提交 Action。容量不足可从当前子块重试，已成功的子块不重放；
异常维持原 fail-stop 隔离，尤其 KV append 后的 Action 失败并不具备回滚
能力。全部子块 ACK、真实 DMA 槽释放成功后才算整个 owner 发布完成。迟到
取消、I/O 及释放失败均有隔离检查。GC 仍由每次原逻辑 append 调用。

GDN/MTP 的三个状态消费者使用单独 `physical_step` receipt；诊断/统计的
两个 2K last_steps 保留，不能把最后一个 2K 当成实际 4K forward 的状态。
每次成功发布记录 `logical_prefill_publication.jsonl`，包含原逻辑坐标、
采样索引和确认步骤，用于对照实际 model geometry。归约/日志/搬运均计入
完整吞吐，不跳过原 attention/图评分工作。

`artifacts/physical-prefill-qualification-20261022-r2/` 冻结 536 个源码/测试
文件；4 项真实 GPU 检查通过（4.36 s），包括 24 Q heads/4 KV heads/dim256、
FP8 scale 0.37、非连续物理页、5/27-token 逻辑段、CUDA Graph 更新 owner/
rows/segments 后污染旧 LSE 的重放，以及 fused 开/关的 reference 跨
8192/16384 softmax 宽度对照。固定 Q/K 的新旧页分数和 attention 输出
逐位相同；对独立完整 causal FP32 oracle 的页分数最大差约 2.79e-9。
此结果不保证整模型不同 GEMM/GDN 批形状逐位相同。

根代理最终调度/配置/Action/发布/native状态/分析组合 152 项 CPU 检查
通过；子模块另有扩展资格，计数存在重叠，不相加。

`artifacts/prefill-physical-coalescing-20261022-r1/` 已完成冻结同源码实模，
两组只改变这个开关，各有两次正式短输入和两次正式长输入。候选先运行，
参考随后运行；尚不是交错多服务 ABBA。两组均已退出且 GPU 清理为空。

| 完整请求口径 | 原逻辑/物理 2K | 逻辑 2K、物理最多 4K | 吞吐变化 |
| --- | ---: | ---: | ---: |
| 32K，两次合并 | 8,616.01 token/s | 9,149.51 token/s | +6.19% |
| 256K，两次合并 | 5,530.62 token/s | 5,830.95 token/s | +5.43% |
| 256K，首次长输入 TTFT | 48.062 s | 45.623 s | 首次准备仍包含在内 |
| 256K，重复长输入 TTFT | 46.699 s | 44.258 s | 吞吐 5,611.32→5,920.82（+5.52%） |

冷准备为 76.152/76.151 秒。模型 CUDA 事件区间长输入均值 42.716→40.196
秒；pressure 等待 3.320→3.373 秒、Query 等待 0.887→0.913 秒，没有把重选
成本移出测量。这些事件区间存在重叠，不能相加解释完整 TTFT。

增强后的逻辑分析器 34 项 CPU 测试通过，
`logical-comparison.json` 保存源码指纹及所有原精确比较结果。四对请求的
展开几何、冻结 stride 对应的具体采样行、原事务提交边界与有界分段证据
全部通过。32K 每次 7 个真实合并组、18→11 次物理提交；256K 每次 50 组、
145→95 次物理提交。四个正式请求合计 114 组，非首次未初始化 forward、
压力与 Query 边界、部分尾块依旧使用原块。

**准确性资格仍未完成。** 四对逐 token 输出、接受分数/图摘要及 MTP
轨迹不同；长输入选页也不同。短输入没有重选，两组仍在第 31 个生成 token
处分叉，101→113 token；长输入 138→131 token。首次 pressure 位置相同，
但 1,024 个选择页中已有 90 个不同。固定 Q/K 的 GPU 一致性不能代替完整
NVFP4 模型对批形状变化的验证。当前日志没有逐块逐页 prefill 分值，
`action.score_sum` 也不是完整累计 prefill 向量；不得据其推算分数误差。
所有针题答案得分 100，但这是同一题重复，不能作为更广泛准确性结论。

原生 vLLM 的实录形状对照现已完成：每次稳态从已有 32,768 KV 开始，追加
实际 `32 + 3×4096 + 2048 + 1952`，共 16,320 token。完整 32K 则严格复制
`2048 + 7×4096 + 2016 + 25 + 7`，另在同一原生服务内将 4K 拆回两个 2K
作为输出敏感性控制。控制组不进入主吞吐或连续阶段统计。

`artifacts/active-prefill-coalesced-20261022-r1/reference-comparison.json` 直接
引用此前完整测量的候选，不复制成一次新的 KVMem run。原生与旧候选的全部
生产源码指纹及 profile 相同，只有分析工具变化；42 项分析 CPU 检查通过，
报告保存 925 个输入指纹，原生原报告保持只读。实际模型后端中 KVMem 捕获/
统计 wrapper 与原生仍不同，源码同一份不代表相同执行分支。

| 相同工作量口径 | 原生 vLLM | KVMem 物理最多 4K |
| --- | ---: | ---: |
| 上述连续 16,320 token 阶段 | 6,521.32 token/s | 6,206.83 token/s（低 4.82%） |
| 完整 32K、完全相同实际块表 | 9,695.65 token/s（关闭事件计时） | 9,149.51 token/s（低 5.63%） |
| 按长输入实际形状加权模型区间 | 38.449 s（诊断） | 40.196 s（实际模型事件区间） |

原生连续阶段 3 次，KVMem 两个完整请求各 13 段；26 段不是 26 次独立服务
重复。原生完整 32K 打开计时为 9,723.85 token/s，约 0.29% 组间变化；
关闭计时仍有轻量 dispatch wrapper。加权模型区间差 1.746 秒（+4.54%），
不代表完整 256K 的原生 TTFT，更不能与重选嵌套时间相加解释实际 TTFT。

同一个原生服务中，仅把完整 32K 的 4K 块拆回 2K，输出稳定从 137 变为
184 token，第 4 个生成 token 即分叉；证明原生分块本身可改变生成。
但对应 KVMem 的 113/101 token 与原生也不同，不能将 KVMem 的所有差异
因此判定为正常舍入。模型/量化/MTP/seed/采样/编译配置及输入已核对，
可见执行差异包括 norm/RoPE 捕获、三层原生统计补丁、MTP 窗口包装；未保存
各层 scale/Q/K/logits，尚未定位跨系统数值差异。
当前尚未达到同吞吐或完成该候选的准确性资格，新开关仍默认关闭。

### 更大物理组的有界扩展（组件通过，实模进行中）

新增 `prefill_physical_max_chunks`，默认 2、范围 2～8。仅当原开关启用时，
按原逻辑 2K 的 native lookahead 与 RequestPlan 逐项选择最长合格前缀；
不足两块就走原 2K 路径。G16 稳态可形成 4+3 个原块，或 7 个原块，
末尾 1952 token 与 pressure 边界不变。只读 archive credit 上界在规划前
缩小物理预算，最终原 allows 检查仍只执行一次，不重复记失败或等待整 batch。

Action/发布游标保持同一共享租约和每个 2K 的 prepare→append→commit→ACK，
仅最终 child 写真实物理范围。配置检查最大组全部采样行、两个在途下载所需
桶化字节、MTP/batch/G 容量；没有暗中改变 G/K。最多 8K 需要 256 采样行，
现有 bank 足够；14K 需至少 448 行，正式使用必须显式扩大 bank 并报告显存。
按当前几何，256→512 行的持久 GPU bank 约增加 471 MB；更大物理包的 GPU
payload 还会增加占用，不能只以 pinned 槽数量不变声称没有额外内存。

规划 111 项 CPU、消费者组合 449 项 CPU、调度/配置/分析组合 163 项 CPU
均通过，计数包含重叠。`physical-prefill-multichunk-qualification-20261022-r1`
冻结 538 个源文件，12 项 GPU 检查通过（5.29 秒）：4K/8K/14K 的固定 Q/K
新旧分数与 attention 输出逐位相同，独立 FP32 oracle 最大绝对差不超过
2.79e-9；Graph 重放及 fused 开/关的逐 2K 回退也通过。此组件结论仍不能
推导完整 NVFP4 模型跨物理形状的逐位输出一致。

`artifacts/prefill-physical8k-20261022-r1/` 已冻结同源码实模，只改变最大
合并块数 2→4，其余配置一致。先测这一组，不预先把更少 forward 次数
当成收益；完整成本包括更大在途 payload、每个原 2K 提交和最终清理。

该组现已完成，两臂 cleanup 均为空。冻结分析器确认原逻辑展开、具体采样
行和事务边界通过；精确物理/输出比较仍不通过，不将其改写成逐位一致。

| 完整请求口径 | 物理最多 4K | 物理最多 8K | 变化 |
| --- | ---: | ---: | ---: |
| 32K，两次合并 | 9,158.50 token/s | 9,257.16 token/s | +1.08% |
| 256K，两次合并 | 5,794.16 token/s | 5,977.46 token/s | +3.16% |
| 256K 首次 TTFT | 45.815 s | 44.377 s | −1.438 s |
| 256K 重复 TTFT | 44.636 s | 43.300 s | −1.336 s |
| 256K 重复吞吐 | 5,870.68 token/s | 6,051.79 token/s | +3.09% |

冷准备分别 76.137/77.165 秒。长输入模型事件区间均值
40.225→39.038 秒，pressure 等待 3.548→3.327 秒，Query 等待
1.010→0.961 秒；各区间重叠，不相加。短输入输出 113→136 token，
长输入 131→127 token；相同针题均得 100 分，准确性资格限制同上。
此轮为先参考后候选，尚未交错重复，短输入约 1% 的差距可能属于波动。

`physical8k-execution-analysis.json` 保存逐日志复算和输入指纹。长输入实际
95→64 次提交，候选组 N4×18、N3×13、N2×1；短输入 11→8 次提交。
两臂短输入均为 16 次逻辑事务/1023 采样行，长输入均为 128/8151，正式
native fallback 与 validation failure 都为零。最后 pressure h=261248、
Query h=262016 不变。GPU 约 1 秒采样峰值短输入 71830→72958 MiB，长输入
73214→74342 MiB，均增加 1128 MiB。启动以来的 allocator peak 包含冷准备，
不能替代每请求峰值；本轮没有 CPU RSS 遥测。archive I/O 峰值
285212672→427819008 B、固定 capture staging 1146422304 B 也不是 CPU RSS。

候选重复长输入的 model 事件区间为 39.06183 秒，pressure 3.05335 秒，
Query 0.685354 秒。相对首次，TTFT 减少 1.076881 秒，但 model 区间反增
0.048248 秒；主要变化在 pressure/Query 准备，不能当成进一步模型提速。

新增 N3/N8 固定 Q/K 组件补齐：
`physical-prefill-multichunk-qualification-20261022-r2` 冻结 539 个文件，
8 项 GPU 检查通过（4.96 秒），覆盖 6K/16K、Graph 重放和两种旁路回退；
与原逐 2K 分数逐位相同，独立 oracle 最大绝对差仍不超过 2.79e-9。
跨运行时 N2 的四个正式请求输出、选页、摘要、MTP、几何及逻辑发布全部
一致，见 `prefill-physical8k-20261022-r1/cross-runtime-regression.json`；
这是功能回归，不能作为同源码性能 A/B。

已完成 `prefill-physical14k-20261022-r1`：最大子块数 4→7，两臂均显式
设置原生统计容量 512 行，K32/G16/逻辑 2K 不变，候选先运行。原生对照
必须随后按实际大块形状重新测量，不能沿用原 2K/4K 的原生数字声称追平。

| 完整请求口径 | 物理最多 8K、bank512 | 物理最多 14K、bank512 | 变化 |
| --- | ---: | ---: | ---: |
| 32K，两次合并 | 9,263.20 token/s | 9,409.51 token/s | +1.58% |
| 256K，两次合并 | 5,993.94 token/s | 6,097.49 token/s | +1.73% |
| 256K 首次 TTFT | 44.348 s | 43.483 s | −0.866 s |
| 256K 重复 TTFT | 43.088 s | 42.469 s | −0.619 s |
| 256K 重复吞吐 | 6,081.60 token/s | 6,170.25 token/s | +1.46% |

冷准备 76.151/75.135 秒。两臂 cleanup 为空，`logical-comparison.json`
通过逻辑规则检查，完整输出/选页仍不同。`physical14k-execution-analysis.json`
确认短输入 8→6 次实际提交（候选 N7×2），长输入 64→49（N7×16）；完整
逻辑事务/采样仍为 16/1023、128/8151。重选边界不变，native 回退、校验
失败、pool 耗尽与隔离均零，最大 pending 包为 2。

GPU 采样峰值短输入 72972→75082 MiB、长输入 74356→76468 MiB，约增加
2.06 GiB。这是模型中间张量和在途包的整体观测，不只 raw payload；两臂
native bank 实际相同 945512540 B、固定 staging 相同 1146422304 B。
Action pinned allocated 4014080→7946240 B，peak active 2359296→4784128 B。
两臂旧冻结驱动没有 CPU RSS，仍报告缺失。

后续驱动已在原 1 秒采样循环记录进程树 RSS；共享页可能重复计入，明确
不称其为 PSS，也不把各进程历史 HWM 相加成同时峰值。旧数据保持 null。
83 项诊断/来源校验 CPU 测试通过，生产代码指纹检查未放宽。
`active-prefill-physical14k-20261022-r1` 已完成实际形状匹配的原生参照，
cleanup 为空。`reference-comparison.json` 引用已有 14K KVMem 结果，未虚构
一次新 KVMem run；生产源码相同，已审查的诊断 RSS 变化单独列出指纹。

| 相同工作量口径 | 原生 vLLM | KVMem 物理最多 14K |
| --- | ---: | ---: |
| 连续 `32+14336+1952`，追加 16320 token | 6,833.78 token/s | 6,503.30 token/s（低 4.84%） |
| 完整 32K、相同六块，原生事件计时关闭 | 9,928.10 token/s | 9,409.51 token/s（低 5.22%） |
| 长输入形状加权模型区间 | 36.832 s（诊断） | 38.425 s（实测） |

原生连续阶段 3 次、KVMem 26 段来自两个请求，不能把相关阶段视为独立
重复。实际 native 完整 32K 开计时为 9930.24 token/s，与关闭计时约差
0.02%。原生冷准备 65.113 秒。模型区间仍差 1.593 秒（+4.32%），完整
KVMem 256K TTFT 42.976 秒还包含重选等工作；加权 36.832 秒不能冒充
完整原生 256K TTFT。原生 14K 形状完整输出109 token，对应 KVMem113；
原生拆回 2K 为184，输出差异资格限制继续保留。当前仍未达到同吞吐。

### 后续候选：无采样 CTA 跳过重复统计判断

只读 cubin/Nsight 核对：现用 stats CTA64/MMA_KV4 为 REG255、STACK/LOCAL0，
原生 REG252，两者 dynamic shared 均 98304 B，未发现寄存器 spill 或占用率
下降证据。静态 SASS 指令更多，但不能据静态慢分支计数断言 SFU 饱和。
stride32/GQA6 下约三分之二 Q CTA 无采样行；候选在 KV 循环外从实际 rows
位表做 CTA 一致判定，仅跳过这些块内的 CaptureSoftmax。保留有采样块
原浮点体、完整归一化、XQA 和采样密度，无新增缓冲，默认关闭。当前仅
进入实现和资格阶段，新增 barrier/寄存器也可能抵消收益，必须实测。

`action_native_prefill_cta_gate` 已实现为默认关闭、独立源 overlay/module 的
开关，与 aligned-pieces 组合有独立 key；没有新增 tensor ABI 或缓冲。
独立审查确认提前 return 是 CTA 一致的，投票在 PDL wait 之后，packed-Q
首尾对应的部分 GQA heads、decode offset、split-K 与 Graph 动态采样表均
按原索引处理。实际 cubin 增加 8 B stack，但其 STL/LDL 仅在 KV 循环前后；
不能据此声称没有任何额外成本，也不能误称每个 KV tile 都在 spill。

119 项 CPU 检查通过。GPU 资格前两轮分别遇到工具 PATH 缺 ninja 和测试
workspace 太小，均发生在候选模块执行前，失败证据已保留。仅修正测试
环境及 workspace 后，`native-prefill-cta-gate-20261022-r3` 的 15 项 GPU
检查通过（106.39 秒含四模块编译），source_unchanged=true：完整 attention
输出、LSE（包括未使用尾部）、页分数逐字节一致。覆盖无/全/稀疏/边界
采样、2K/GQA1、ragged 多 owner 与 decode offset、Graph 空→采样→换 owner
→空、非16对齐 split-K 与 padded invalid CTA，以及缺失/非有限生产结果。
原生实模分析器新增实际 dispatch/源指纹验收，并明确没有计数 device 上
跳过的 CTA；24 项相关 CPU 报告检查通过。

组件测量已完成：`native-prefill-cta-gate-component-20261022-r1`，9 种
Q/context 形状各 12 对 AB/BA，所有分数和 attention 输出逐字节相同。
14K Q / 48K context 三层完整 attention＋统计＋下载 wall 为
258.315→254.577 ms，节省 3.738 ms（1.45%）；全部形状降低 0.95%～2.15%。
按 16 个大块粗估约 60 ms，只是量级参考，不是实模结果，也不足以解释
剩余约 1.59 s 模型区间差额。此开关保持关闭，暂不追加整模型对照。
完整资格、静态 stack 与测量限制见 `native-prefill-cta-gate-20261022.md`。

下一步用已完成的 14K 候选冻结运行时进行同形状 Nsight：实际完整 32K
块表为 `[2048,14336,14336,2016,25,7]`，两边先暖完整请求，再捕获 active
16384/30720 的两个 14336-token 块。诊断扩展 80 项 CPU 检查通过；
trace 仅用于定位，不替换无 profiler 的正式吞吐结果。

`prefill-14k-nsight-matched-20261022-r1` 的 KVMem 在启动时失败，尚无 trace：
诊断入口绕过正常 serve.main，虽设置 CUDA_HOME，却未将 CUDA/bin 加 PATH；
当前继承路径没有 nvcc 且未安装 flashinfer_cubin，导致 vLLM 跳过 bmm_fp8
注册，已缓存 AOT 引用该 op 时出错。没有清缓存、关闭编译或修改安装包。
修正诊断环境并通过 83 项 CPU 回归后冻结 r2；r1 失败日志和空 cleanup 保留。
同时核验旧同形状 r1 两臂、pressure-phases r1 的实际日志均选择 FlashInfer
FP8 线性后端，不受本次路径问题影响。r2 仍需核验实际后端与 CUDA kernels。

另一只读候选暂不实施：将 refresh 的完成 fence 推迟到下一次 attention。
14K 重复长请求的 14 次 pressure submitted→ready 合计 150.588 ms，但包含
commit、close 和 report；旧 pressure trace 的该 10.832 ms 区间 GPU 活动
为零，不能把它当成可隐藏 DMA。当前按 chunk 遍历全部 17 层，第一层也要
到最后 chunk 才拿到完整上下文；提前 ACK 还会释放 resident-V gather 可能
读取的旧页，layer hook 又晚于 KV 写。现无收益证据支持引入分阶段租约及
提交改造，详见 `async-refresh-consumer-fence-review-20261022.md`。

### 14K 同形状时间线：确认额外统计与编译器 capture 拷贝

`prefill-14k-nsight-matched-20261022-r2` 两臂已完成，cleanup 均为空，实际
完整块表及所捕获 Q=14336/active=16384、30720 核验通过。实际 FP8 线性
后端均 FlashInfer；全部 34 个 FA2 的 CTA64 模板及 launch geometry 相同，
除三个选定层附带原生统计外。冷准备及 profiler 导出停顿不用于吞吐计算。

| 两块诊断窗口 | 原生 → KVMem 差额 |
| --- | ---: |
| GPU 活动跨度 2740.682→2845.755 ms | +105.073 ms |
| 活动并集 | +84.912 ms |
| 活动之间的空隙 | +20.161 ms |
| 三个选定层 FA2 核合计 | +37.582 ms |
| 其他 28 个 FA2 核合计 | +0.029 ms |
| capture clone/copyback，160 核 | +18.891 ms |
| norm/RoPE（含 capture 写出） | +6.732 ms |

只有活动并集＋空隙能组成跨度；上述 kernel 小项是另一口径，不能再次加到
并集上。KVMem D2H 合计 18.887 ms，仅 0.505 ms 与 kernel 重叠。已确认
take_gil 等待 23.896 ms 中只有 3.693 ms 落在 GPU 空隙，不能将全部空隙
归因于 Python/GIL。

每个 target norm 前紧邻三个 clone、后紧邻两个 copyback，与已载入 AOT
5710696… 的源码逐调用匹配：2 块×16 层×5 核。编译器对 persistent
raw Q/K/V 切片写入进行了 functionalization，产生完整临时拷贝。复制 extent
为 15.032 GB；读＋写约 30.065 GB 只是按张量计算的流量，不是 DRAM 实测。
不能把所有大拷贝算作可优化成本：96 条各 176160768 B 的 GDN 拷贝（合计
16.911 GB），以及约 28.7 ms generic direct_copy，两边都有。

实际 128 对 MLP gate/up GEMM 的输出 layout 和 grid 不同（188×1 / 1×188），
耗时 409.193→407.107 ms；其余 512 对模板/geometry 相同。所有 GEMM 总量
反而减少 3.548 ms，不能说两边 GEMM binary 全同，但不解释主要正差额。
复算依据与限制保存在 `matched14k-nsight-comparison.json`、
`matched14k-copy-attribution.json` 和 `matched14k-key-findings-r2.json`。

正在实现默认关闭的 `capture_opaque_projection`：保留完整 raw Q/K/V 和
原 norm/RoPE kernel，以有序副作用、forward-context owner 和实际 Q/K 输出
依赖约束 opaque op，消除编译器复制；另将源指纹纳入 additional_config 编译
key，避免误用旧 AOT。新接口必须通过真实编译＋Graph、设备 gate、MTP/Query
及 owner 隔离资格，再测完整组件与实模；暂不计为收益。


`capture-opaque-qualification-20261022-r1` 的 7 项 GPU 检查已通过
（11.35 秒，含实际 Inductor 编译），冻结 source hash 未变、cleanup 为空。
覆盖 1/37/2048/14336 行、MRoPE、整容量尾部、同名不同 owner/context、
设备 gate 0→1 和真实 CUDA Graph 重放、动态新 Query shape，以及错误发生前
不写入。原投影 q/k/v/gate 和完整 raw Q/K/V 逐字节相同。独立组件 profiler
显示旧 7 核（3 clone＋原 norm＋3 copyback）变成仅原 norm；这与实际模型
旧 trace 的 5 个额外 copy 核是不同编译形状，不能混为同一测量。

启动/source-key 检查与既有 launcher/options 共 182 项 CPU 测试通过；性能
驱动实际分支与冻结源码 key 审计 20 项通过。报告回归另发现旧 checkpoint
测试假对象缺少生产早已使用的 `authority.gc`，仅为 fixture 补 `gc=None`，
7 项报告测试通过，未改变生产退休逻辑。新接口仍默认关闭，真实 vLLM
piecewise/AOT 集成及组件/完整模型性能尚待验证。


Opaque 组件 `capture-opaque-component-20261022-r3/post-audit/final.json` 已补齐
全部必要证据。原运行在 14K 的 profiler 审计处退出（错误假设两种形状都合成
相同数量的 copy 核），有效计时与原失败状态未覆盖；后补审计使用原冻结
runtime/输入 recipe，新增 14K trace 单独保存，并记录输入重建及原计时未保存
输入 byte hash 的限制。12 对 AB/BA 完整 wall 均包含主机 dispatch：

| 17 层采集组件 | 原路径 | opaque | 变化 |
| --- | ---: | ---: | ---: |
| 2K wall | 2.728 ms | 3.167 ms | 慢 16.09% |
| 14K wall | 23.153 ms | 12.596 ms | 快 45.60% |

14K 独立 trace 旧 83 核合计 22.818 ms，其中 norm 12.146、copy 10.672；
新 17 个 norm 合计 12.181 ms、无 copy。2K 旧 51→17 核，但 CPU 供给成本
增加，完整组件反慢。14K 临时 allocated 峰值 952→728 MiB，2K 136→104 MiB。
不能将此百分比当作模型吞吐，32K/256K 同源码单开关实模已经冻结在
`prefill-capture-opaque-20261022-r1`，候选正在启动。新 source key 带来的
首次编译单列为冷准备，未关闭编译或清旧缓存。

静态 aligned 统计候选 `action_native_prefill_static_aligned` 默认关闭。
只对全部 owner 逻辑段与实际 page/split-K 边界满足 16-token 对齐、当前纯
prefill且非Graph路径选择独立模块；其他情况仍用既有 native 统计。147 项
CPU 通过。GPU r1 暴露实际 plan_info 是 TVM CPU Sequence（过严容器检查令
候选没有命中）；修正严格 CPU-int Sequence 资格后 r2 命中候选，随后仅因
测试强制 split=1 的 planner workspace 256 MiB 小于所需 347 MiB 退出。
仅扩大测试 workspace 到 512 MiB，新 r3 的 16 项 GPU 全部通过（3.70 秒），
全输出/LSE/页分数逐字节一致、source hash 未变、cleanup 为空。

只读 cubin 对比 CTA64/MMA_KV4：generic→static SASS 指令 8872→4312，CAS
32→0，LG2 74→10；REG 仍 255，stack 0→32 B，且新增 LDL 有三条在 KV
循环内。因此不能只据移除慢分支宣称提速；组件尚待与模型测试串行执行。


### Opaque 整模首轮集成与调优对照修正

`prefill-capture-opaque-20261022-r1/candidate` 的五个请求已完成，cleanup
确认没有残留进程；17 个 owner 安装成功，2906 次 host dispatch（不计 Graph
重放），native reuse 无回退，长输入每次 14 pressure＋1 Query，needle 得分
100。冷准备 322.56 秒包含首次 AOT/原生核编译及 FlashInfer 调优，独立列出。
32K 正式 TTFT 3.4355/3.4215 秒，256K 47.4613/42.0434 秒。
这些是集成结果，不能作为干净 A/B 或新路径准确性结论。

原因是新 source fingerprint 隔离编译缓存，同时改变了 vLLM 全配置 hash；
FlashInfer 也用这个 hash 区分调优缓存，因此候选重新调优，和原缓存有 38
项配置值不同，输出轨迹也不同。原 r1 reference 未运行，保留全部原始记录。
后续在各 arm 自己的目录安装同一份冻结调优 JSON，保持模型编译缓存独立；
用服务日志核验实际读取路径、旧 tactics/环境元数据不变，并报告新增 keys。
`_generation` 单独记录，不作为 tactics；两组实际最终配置也必须一致。
不会修改系统全局缓存，也不把生成轨迹差异全部归因于采集优化。

### 并行 CPU seal pack 候选

新增 `cpu_seal_pack_workers=1..8`，默认仍为 1；复用已有有界 gather 线程池，
每个完整记录仍只拷贝一次，按原记录顺序拼接并继续原 SHA/事务发布。
2× payload admission、异常排空、租约及引用生命周期保持不变；只允许
CPU direct seal、非 historical-GC payload 分支。专项及归档回归 107 项
通过，配置/报告/基准审计回归 103 项通过。组件计时尚未完成。

静态 aligned GPU 组件已完成九种形状；14K Q/48K KV 三个采样层的完整
组件从 257.992 降至 241.141 ms（−6.53%）。此前发现的 spill 并未抵消
该形状收益。按 16 个大块换算约 0.270 秒只是量级估计，不能当作已测整机
收益。仍默认关闭，后续需独立整模验证。


### 固定 FlashInfer tactics 后的 opaque 对照：尚未通过语义验收

`prefill-capture-opaque-controlled-20261022-r1` 两臂都已完成，cleanup 为空。
冻结同一份136项tactics，启动及终态实际配置语义SHA均为
`3deb508d06609a3d3a8bad055c029b8d57764afaf927b41f9fcc1d720b9d517d`，
无新增keys，环境元数据一致。驱动/seed链/真实路径等57项CPU检查通过。
结果原样保存 `comparison.json`，分析器因严格语义不一致退出1：

| 输入 | 原路径 | opaque | 吞吐变化（未验收） |
| --- | ---: | ---: | ---: |
| 32K | 9,304.12 token/s | 9,530.49 token/s | +2.43% |
| 256K | 6,081.71 token/s | 6,134.75 token/s | +0.87% |

冷准备118.21/158.29秒单列；256K每组各一次首形状编译与一次复跑，不能
隐去首轮。所有几何相同，但32K输出113→97 token；256K输出114→187，
且重选内容/图摘要不同。needle各次100不等于策略一致性或广泛准确性通过。
当前不能把上述差值作为已验收优化，也不能假定轨迹变化仅来自sampling。

进一步只读发现，相同Inductor RMSNorm→量化内核的本地best_config也不同：
一个R0_BLOCK8192/warps16→512/warps8，另一个1024/warps8→8192/warps16。
归约树改变可能使FP32舍入经FP8/FP4量化放大；这是已观察的混杂因素，
尚未认定唯一原因。采集自身Triton的fp_fusion/reflect_ftz、warps/stages
相同，尚未发现Q/K/V/gate stride错误。后续先隔离/固定共同编译内核配置，
保留原始失败，不降低比较标准。opaque仍默认关闭。

CPU pack组件已完成：常见64records seal34.430→26.841ms，但release
0.949→7.575ms；seal+release仅改善2.72%，完整读/验/释放生命周期
84.342→84.643ms。CPU时间增加51.5%，不能将封存阶段22%直接换算整机
收益。48次formal字节/SHA一致，12项故障回滚/复用验证通过，仍默认1worker。

### 静态 aligned 整模结果与剩余统计成本

`prefill-static-aligned-20261022-r1/candidate` 完成五个请求，复用上一轮
controlled reference；冻结 runtime 逐文件相同，唯一 profile 差异是静态
aligned 开关，FlashInfer 的136项实际 tactics 相同。后补分析器审计修正
仅接纳真实资格计数名称，未修改冻结执行代码或原始数据；23项CPU回归通过。
`post-audit/comparison.json` 的输出、累计分数/图摘要、选页与 mandatory、
MTP、实际计算几何全部一致，cleanup 为空。225次静态分发，无 native 旁路回退。

| 输入 | reference | 静态 aligned | 完整吞吐变化 |
| --- | ---: | ---: | ---: |
| 32K | 9,304.12 token/s | 9,455.46 token/s | +1.63% |
| 256K | 6,081.71 token/s | 6,089.57 token/s | +0.13% |

256K平均TTFT43.087410→43.031788秒，模型CUDA时间38.469669→38.092154，
而host提交间隙4.283646→4.656373秒。这只是单轮各两次请求，不能据此认定
因果或稳定收益；pressure与Query区间与上述时间重叠，不能额外相加。
候选冷准备160.287秒，首个32K暖请求包含首次静态核编译，不计正式吞吐。
候选继续默认关闭。

`native-prefill-stats-floor-20261022-r1` 用真正无统计的原生DefaultAttention、
generic统计和静态统计做九种形状×12轮、六种顺序平衡的组件对照。
14K Q/48K KV 的三个采样层完整组件分别205.626/258.195/241.354ms，
静态统计仍比无统计多35.728ms（17.38%），其中FA2核多30.558ms。
两种统计的输出/页分数及资格用LSE逐字节一致，实际原生分支和源码均核验。
这些是组件成本，不是完整模型百分比；剩余统计开销和重选等待分别继续优化。

### 14K 路径首个 pressure 的关键等待

`prefill-static-pressure-nsight-20261022-r1` 在完整256K暖请求后，捕获第二个
完整请求的[30720,63456)范围：14K、2K、1984行尾块、一次pressure、32行
replay和后续14K。五步实际几何、唯一pressure与399个冻结runtime文件通过
核验；cleanup为空。它是诊断，不能替代吞吐计时。

完整重选生命周期268.688ms，GPU活动并集34.114ms、空闲234.574ms。
尾块1984 token的GPU工作结束后，CPU才完成scatter10.081ms、MeanK
finalize22.964ms、62条记录封装61.302ms及hash16.139ms。封存读取屏障
74.104ms等待的是同一批封装/hash，不能重复相加。组装随后从历史读取
498.313MiB raw K，八次读取共53.832ms，与GPU工作没有重叠。

这说明后台线程已经存在，但这些尾部依赖未被后续模型计算覆盖。全程
readiness检查只0.952ms，主线程已有127.109ms休眠；本证据不支持把
busy-poll当主要问题。提交后的10.779ms也不是未完成DMA，包含实际commit/
report等主机工作。完整归因及范围交集见
`docs/prefill-static-pressure-nsight-20261022.md`。

下一步优先核验减少封存复制、MeanK必要依赖与静态统计元数据读取；不重试
已经完整生命周期无收益的pack8，不据嵌套区间推算全部可消除成本。

### Opaque：固定两类调优配置后通过整模语义核验

`prefill-capture-opaque-dualseed-20261022-r1` 两臂运行完成，cleanup均为空。
除了同一136项FlashInfer tactics，还按target/MTP分别冻结Inductor的
`.best_config`，只将配置安装到每个arm的独立缓存，不复制AOT图或编译产物，
不修改系统共享缓存。启动期间观察真实launcher：target12个、MTP5个双方
共同内核均实际执行同一源码和所选配置；observer在服务ready前移除，正式
分发无新增钩子。完成后再次核验本地配置和源码，父plan绑定seed哈希。
控制工具、worker和driver相关128项CPU回归通过。

只有 `capture_opaque_projection` 开关不同，`comparison.json` 四个正式
请求全部检查通过：逐token输出、累计分数与图摘要、选页/mandatory、MTP
和计算几何完全一致。32K保持113个输出token，256K保持114个；先前未固定
Inductor时的97/187不再出现。这支持编译调优是此前差异的重要混杂因素，
不等于证明所有未来形状都已资格通过。

| 完整请求口径 | 旧采集 | opaque | 吞吐变化 |
| --- | ---: | ---: | ---: |
| 32K，两次合并 | 9,325.04 token/s | 9,541.64 token/s | +2.32% |
| 256K，两次合并 | 5,788.58 token/s | 5,874.80 token/s | +1.49% |

32K平均TTFT3.513978→3.434211秒，模型CUDA3.389325→3.308075秒。
256K平均TTFT45.269284→44.604937秒，模型CUDA38.492254→38.029378秒。
两臂首次长输入TTFT48.165896/47.206198秒，第二次42.372672/42.003675秒；
第二次吞吐6,184.29→6,238.62 token/s（约+0.88%），只有各一次，不能代替
两次合并或隐去首次长输入成本。首次额外耗时尚未进一步归因。
冷准备156.291/145.267秒另列，不将单次差值归因opaque。

这是一轮candidate→reference顺序、每臂两个正式请求，并非完成交错重复
或新八并发验收。opaque维持默认关闭；可与静态统计候选继续组合验证，
不得将各自百分比相加。目标仍未达成，剩余差距继续按实际GPU与重选关键
路径定位。
