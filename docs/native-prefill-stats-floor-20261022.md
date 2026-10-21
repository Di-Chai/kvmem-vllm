# 同输入三臂组件：原生 attention 与剩余统计成本

同 Q/K/V 的三臂结果确认：静态对齐特化降低了统计开销，但没有使它接近零。14K Q、48K 可见上下文下，三个采样层的完整成本为原生 205.626 ms、generic 统计 258.195 ms、static aligned 统计 241.354 ms。特化消除了约 32.0% 的完整统计增量，仍比原生多 35.728 ms（17.38%）。独立 profiler 的三层 FA2 核时间分别为 204.921、252.630、235.479 ms；特化后的核内增量仍为 30.558 ms。

这里的“原生”是没有统计需求的基准成本，不是满足 KVMem 语义的可替代实现，也不是所有统计实现的理论下界。当前没有据此修改默认开关。

## 对照与核验

脚本：`scripts/benchmark_native_prefill_stats_floor.py`。它使用资格通过的静态特化 r3 冻结 runtime，新增三臂控制器，没有修改生产代码或先前冻结组件脚本。

- 三臂共享同一组 BF16 Q、FP8 E4M3 K/V、物理页表、stride、scale、FlashInfer wrapper 与原始 FA2 plan。24 Q heads、4 KV heads、dim256，物理页16；三层共享输入/cache 的局限仍存在。
- native 直接调用原始已规划 FlashInfer module，不调用 Action 或 NativeAttentionStats，不清统计矩阵、不计算页分数、不做分数 D2H。另行 profiler 强制核验三个 `DefaultAttention` kernel，且没有 KVMem 统计辅助核；不能用“device 采样 gate 关闭”替代该基线。
- generic/static 包括原有 Action 元数据准备、三层 attention、全部归一化/层平均、pinned D2H 和完成等待。统计容量相同：512 rows、65,576 context、2,202 segments、6,301 width、8 owners；两臂各 945,512,540 bytes。两个 bank 在三臂计时中始终共存，因此这不是进程峰值显存对照。
- 12 轮采用六种排列的两个完整周期，每轮内部位置和有向前序对均衡；跨轮边界不宣称均衡。完整 wall/CUDA event 与 profiler 核计时分开，不将两套样本直接当作互斥时间分区。
- 正确性阶段逐层比较三臂 attention 输出和原生 row LSE，后两臂还比较完整 piece LSE（含未用区域 sentinel）与页分数，均逐字节一致。原生 row LSE 只在资格阶段额外写出；实际计时 `return_lse=False` 路径又独立验证了三层 attention 输出。全部 clone、完整 sentinel 初始化、fixture GC 均不在正式计时内。

纯 CPU 协议测试 6 项通过；实际 GPU 运行的九种形状全部通过上述资格及实际分支核验。每种形状 native 60 次原模块调用，static 60 次真实特化 dispatch，无 generic fallback。源文件起止 SHA 均相同，进程退出后 GPU 为空。

## 九种形状结果

以下完整组件均为 wall 均值；最后的核增量来自独立 profiler。全部为三个采样层合计、每批执行，不是每个 decoding token。

| Q | 可见上下文（含 Q） | native ms | generic ms | static ms | static 完整增量 | static FA2 核增量 ms | static 比 generic 快的配对 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 16K | 12.674 | 16.552 | 15.827 | 3.152（24.87%） | 1.720 | 12/12 |
| 2K | 32K | 26.432 | 32.755 | 31.850 | 5.418（20.50%） | 3.529 | 12/12 |
| 2K | 48K | 40.256 | 49.472 | 47.677 | 7.421（18.44%） | 5.160 | 12/12 |
| 8K | 16K | 35.432 | 45.965 | 43.191 | 7.759（21.90%） | 5.385 | 12/12 |
| 8K | 32K | 81.886 | 104.103 | 97.289 | 15.403（18.81%） | 12.011 | 12/12 |
| 8K | 48K | 128.635 | 161.716 | 150.985 | 22.350（17.37%） | 18.639 | 12/12 |
| 14K | 16K | 46.539 | 60.123 | 56.571 | 10.033（21.56%） | 6.843 | 12/12 |
| 14K | 32K | 126.064 | 159.387 | 149.091 | 23.027（18.27%） | 18.782 | 12/12 |
| 14K | 48K | 205.626 | 258.195 | 241.354 | 35.728（17.38%） | 30.558 | 12/12 |

证据：`artifacts/native-prefill-stats-floor-20261022-r1/{manifest.json,results.json,result.json,stdout.log,frozen/}`。`source_unchanged=true`、`frozen_source_unchanged=true`。先前 generic→static 九形状结果单独保留于 `docs/native-prefill-static-aligned-20261022.md`，不能混用不同轮次的绝对时间算增量。

## 仍在执行的工作与下一步

当前 FA2 插桩在每个 KV tile 的原 online softmax 更新之后执行。没有采样的 quad 仍需检查 Q/采样行；实际 static SASS 的主 KV 循环中保留了两组 `q_indptr`/行表加载及部分 GQA divmod，未被编译器完全移出循环。有采样的 quad 每16 token 仍执行 causal mask、原序加法和 shuffle、log，再向 global memory 写一个 FP32 piece LSE。这不是 constant memory 写入，也不是整个历史页只增加一个计数。

以本组件 14K/48K、448 采样行计算，三层各24 heads，在每个可见 piece 的 sum 均大于零时，共写 84,672,000 个 piece LSE，约323 MiB。完整域清零约378 MiB；两个后续归约各读取约378 MiB。这些是源码推导的逻辑访问字节，缓存/合并会影响实际 DRAM 流量，不能直接换算延迟。原采样行、完整 causal 域和逻辑历史分段没有改变。

后续可先研究将每 quad 的 sample/head/causal limit 移到 KV 循环外，或保留原浮点顺序而合并相邻 piece 的写出。两者均未实现：当前核已使用255寄存器，静态特化有三条循环内 local load，增加活跃元数据可能使 spill 更重；写出是否限制性能也还没有硬件计数证据。不能把这些候选的收益提前计入结果。

`35.728 ms × 16 ≈ 0.572 s` 只是16个相同大块的剩余成本量级，不是整个256K请求已测可消除时间。真实服务有其他层、不同张量/scale/缓存、MTP、子包提交、归档与重选，以及不同形状/回退资格，仍需独立整机对照。
