# 归档集中分配、并行复制：组件结果与实模候选

本轮在本地检查点 `587c408` 之后实施。新候选仍然生成逐条独立的 immutable
bytes，复制次数与原 direct seal 相同；优化的是分配线程、并行 memcpy 和
最终释放的组合成本，没有实现最终 owner 直接接收 scatter 的零复制设计。

旧八线程 pack 将每个 shard 的 Python bytes 分配也放在 gather worker 上，
64 条记录的完整生命周期没有收益。本候选在原 archive worker 中集中分配
不可见的私有 bytes，再让现有有界 gather 池复制不相交的记录范围；所有
范围完成后才一次性暴露输出，后续原 SHA、publication、读取和回收保持不变。

## 实现与生命周期

- `PrivateRecordPack` 不实现 buffer protocol，不提供输出字段。构造先验证
  全批输入，再按显式 payload 上限分配；源 buffer exports 持有到完成或安全中止。
- 每条记录只能被一个 range 认领，重复/相交范围全批预检查后拒绝。存在活跃
  写者或未完成记录时不能 finish，存在活跃写者时不能 abort。
- 调用方在既有 `write_from_views` 完成 record slots、generation leases、
  pending job 和 2× source/destination bytes 准入之后才构造私有输出。
- 至多 8 个 shard 使用现有 gather executor；未知 submit 结果使池失效并先
  排空，再释放源和私有输出。保留失败 Future 不应继续持有源 buffer。
- 实模开关 `cpu_seal_serial_allocate=false` 默认关闭；仅 CPU direct seal、
  无 native payload arena、`cpu_seal_pack_workers=1` 可启用。复制线程数为
  `min(8,cpu_gather_workers)`。部分页/MTP 的接受、scatter、MeanK 和原提交顺序
  均未修改。新增实际执行计数供分析器核验。

## 冻结组件

证据：`artifacts/cpu-private-pack-20261022-r1/`。两臂使用同一个新 native binary，
reference 仍调用原 pack；candidate 仅替换私有 packing。记录几何为17层、
32 token/记录、4 KV heads、dim256、原 FP8 raw K/V 字节；1,114,112 bytes/记录。
输入生成及初始 SHA 在计时外；分配、准入、封存、哈希、publication、真实
读取、字节/SHA校验和 repository retirement 全部各自计时并计入完整生命周期。
两臂共享原有 8 路 hash/gather 设置，2 次预热后每个几何 12 对 AB/BA。

| 平均区间 | 32条原 pack | 32条候选 | 64条原 pack | 64条候选 |
| --- | ---: | ---: | ---: | ---: |
| 完整封存 | 45.368 ms | 9.605 ms | 34.808 ms | 16.857 ms |
| 嵌套 pack | 38.207 ms | 2.512 ms | 21.195 ms | 3.444 ms |
| 嵌套 hash | 6.783 ms | 6.695 ms | 13.033 ms | 12.826 ms |
| 读取 | 1.620 ms | 1.477 ms | 2.923 ms | 2.430 ms |
| 字节/SHA验证 | 25.794 ms | 24.376 ms | 45.591 ms | 47.627 ms |
| retirement | 3.433 ms | 0.215 ms | 0.849 ms | 0.353 ms |
| 封存＋retirement | 48.801 ms | 9.820 ms | 35.657 ms | 17.210 ms |
| **完整生命周期** | **76.684 ms** | **36.073 ms** | **84.617 ms** | **67.701 ms** |
| 封存阶段 process CPU | 85.300 ms | 60.406 ms | 113.409 ms | 110.576 ms |

64 条记录更接近一个逻辑 2K prefill 子块：封存降低 **51.57%**、完整生命周期
降低约 **20.0%**。组件没有 scatter、MeanK、模型、DMA 或多 owner 的性能竞争，
以上百分比不能当作整模型提速。分配/首次触页/allocator 复用未进一步拆分，
不把较快 pack 全部归为纯 memcpy 带宽改善。

48 次正式试验及 12 项强制失败边界核验全部通过，原记录字节和 SHA 一致，
结束后原额度全部归还。峰值 I/O 准入仍为 2× payload：64条为142,606,336 bytes。
本组件冻结于生产开关接入之前；其 source-proof 保存该时点脚本和二进制，
后续生产接口没有修改已完成的冻结结果。

新增私有所有权、源导出释放、相交范围、未完成输出、隐式 submit、失败 Future、
真实 archive 回滚等检查与相关旧回归共125项通过。接入配置后的首次回归为
192通过、1项新 elastic 测试失败（夹具传入不属于 elastic 的 cpu_read_cache_bytes）；
修正测试参数后相关80项、扩展后的87项通过，保留原失败记录，不将重复计数相加。

## 完整模型

`prefill-serial-allocate-20261022-r1` 已按最新 opaque 候选冻结同源码两臂，只切换
`cpu_seal_serial_allocate`，固定 FlashInfer 与 Inductor 调优选择；输入 K32/G16、
完整32K/256K、逻辑2K/物理最多14K、NVFP4/FP8/MTP4及全部评分策略不变。
两臂已完成并清理，冻结分析器的全部语义检查通过。

| 完整请求口径 | 原分配路径 | 集中分配候选 | 吞吐变化 |
| --- | ---: | ---: | ---: |
| 32K，两次合并 | 9,528.74 token/s | 9,544.35 token/s | +0.16% |
| 256K，两次合并 | 5,866.20 token/s | 5,907.31 token/s | +0.70% |
| 256K平均TTFT | 44.670286 s | 44.359427 s | 减少0.310859 s |
| 首次256K TTFT | 47.202211 s | 46.771228 s | 减少0.430982 s |
| 重复256K TTFT | 42.138361 s | 41.947626 s | 减少0.190735 s |

每个32K输出113 token、256K输出114 token；四对正式请求逐token输出、累计
分数与图摘要、选页/mandatory、MTP和实际计算几何均一致。FlashInfer 136项
与Inductor target/MTP共同内核的实际调优配置核验通过，正式无旁路回退。
归档结束都为354批，候选354批实际进入集中分配，参考0批；退出时CPU authority
及I/O pending均为0。冷准备参考143.267 s、候选142.259 s单列。

整个服务生命周期的嵌套pack累计11.660→3.127 s，但多数已与模型重叠；
**不能将8.533 s差额从TTFT直接扣除**。256K model事件均值38.060→38.072 s，
host提交间隔6.207→5.875 s；pressure/query分别4.089/1.881→3.704/1.931 s，
这些视角互有覆盖，不相加。该结果支持主要改善关键路径等待，而非模型算术。

约1秒采样的256K GPU峰值两组均76,462MiB；进程树RSS峰值参考19,547,361,280B、
候选19,649,654,784B。RSS包含共享页重复计数，且是采样下界，不是精确物理峰值。
两组仍使用原固定槽及2×准入，没有新线程池。候选先、参考后，各长度只有两次，
尚未做交错多启动或八并发验收，0.70%是本次观测，默认继续关闭。
