# 原生 prefill 页统计静态对齐特化：资格与组件结果

`action_native_prefill_static_aligned` 保持默认关闭。该候选通过正确性资格，三层完整组件耗时降低 3.34%～6.70%。14K Q、48K 可见上下文时每批节省 16.851 ms，12 对交错样本全部较快。按 16 个这种大块作量级换算约 0.270 s；这是组件量级，尚不是已测整机加速，也不能据此解释剩余模型差距的全部来源。

## 实现与回退边界

候选使用隔离的 FA2 module/overlay，保持原四张量统计 ABI，不改变 XQA。静态特化从编译结果中移除逻辑 piece 查询和 CAS 合并分支，保留原已对齐分支中的 causal mask、四值加法、shuffle、log 与写出顺序。关闭开关时继续选择原 generic module。首版禁止与 runtime aligned 或 CTA gate 同时开启，避免组合模块与资格混淆。

只有完整 host 元数据能证明安全时才选择特化：当前帧所有 owner 的逻辑分段覆盖完整可见域，内部边界均为 16 的倍数，piece 编号与物理可见 token 的 `position // 16` 一致；最终部分页允许。纯 prefill 的 owner 数、行数和实际 Q shape 必须与该冻结帧一致。K cache 页大小必须为 16 的倍数，窗口参数必须为 -1。

split-K 证明依赖安装版本的 scheduler 将 chunk 页数乘以物理 page size，以及无窗口时 `kv_start=0`。除了原 source manifest，本候选额外固定 `scheduler.cuh` 和 `variants.cuh` 的 SHA；漂移只禁用静态特化并记录原因，仍可使用原生 generic 统计。不存在 GPU 页表或 split 指针回读。

CUDA Graph 捕获、graph wrapper/plan、混合 decode、未知 host plan、未武装帧、小 Q、不可证明的布局等均回到原生 generic module。Graph 重放始终使用 generic；实际 host dispatch 计数明确不包括 replay。`disarm` 清除帧资格，下一次 `arm` 重新生成冻结证明，避免跨 owner/恢复后的旧资格误用。没有新增持久统计缓冲区。

## 正确性与编译证据

CPU 回归先完成 147 项，再补充真实 TVM host plan `Sequence` 兼容用例；专项文件最终为 29 passed、16 GPU skipped。真实 GPU 资格最终 16 项全部通过，覆盖空/全/稀疏采样、GQA 边缘行、2K Q、group=1、ragged 多 owner/无采样 owner、page16 split-K、page24 generic、内部部分页、混合 offset、小 Q、owner 不匹配，以及 Graph 更新行/owner/分段、无效填充 CTA、缺失和非有限统计拒绝。页分数、完整 LSE 缓冲区、attention 输出与 generic 路径逐字节一致，attention 输出也与原生无统计路径核验。

资格证据：`artifacts/native-prefill-static-aligned-20261022-r3/`。冻结源码未变化，退出后 GPU 进程为空。前两轮失败证据保留：r1 揭示 TVM Array 不是 Python list，现已改为只接受 host `Sequence` 并增加真实 TVM 回归；r2 的极端 split fixture workspace 不足，原 planner 在运行候选前拒绝，现仅将该 fixture 扩至 512 MiB。r3 完整通过。

实际 CTA64/MMA_KV4 编译产物对照如下，来自同机已生成库的静态检查：

| 项目 | generic | static aligned |
|---|---:|---:|
| SASS 指令行 | 8,872 | 4,312 |
| LG2 指令 | 74 | 10 |
| CAS 指令 | 32 | 0 |
| 寄存器 | 255 | 255 |
| 栈 | 0 B | 32 B |

候选有三条 LDL 位于 KV 主循环内。因此不能只根据代码缩小推断收益，也没有寄存器/occupancy 提升证据；下列实际组件时间已包含这些成本。资源报告中的静态 shared memory 不包含运行时动态 shared memory。编译产物、库 SHA、资源信息和 SASS 保存在资格 artifact 的 `compiled/`。

## 完整组件测量

固定 BF16 Q、FP8 E4M3 K/V、24 Q heads、4 KV heads、head dim 256、16-token 物理页；每 32 行采样，三层按原序平均。两臂相同容量：`max_rows=512`、context 65,576、max segments 2,202、width 6,301、max owners 8，统计资源均为 945,512,540 bytes。唯一变化是静态对齐特化；runtime aligned 和 CTA gate 均关闭。

每种形状预热后运行 12 对 AB/BA 交错样本。完整 wall 包括 Action 元数据准备、三层 FA2、行/页归约、层平均、pinned D2H 及实际完成等待。正确性复制、各形状结束后的 fixture GC 在计时外。CUDA event 区间单独保存，FA2 核时间来自另行 profiler 采样，不混入正式 wall。首次 dispatch 与正确性预热时间单列，包括首次所需的特化编译；初始化分配不在该计时内。

| Q tokens | 可见上下文（含 Q） | 完整旧版→候选 ms | 耗时降低 | 节省 ms | 较快配对 | 独立 FA2 核旧版→候选 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 16,384 | 16.240 → 15.668 | 3.52% | 0.572 | 12/12 | 15.983 → 15.221 |
| 2,048 | 32,768 | 32.732 → 31.596 | 3.47% | 1.136 | 12/12 | 31.049 → 30.164 |
| 2,048 | 49,152 | 49.154 → 47.510 | 3.34% | 1.643 | 12/12 | 47.092 → 45.231 |
| 8,192 | 16,384 | 45.861 → 43.005 | 6.23% | 2.856 | 12/12 | 43.541 → 40.513 |
| 8,192 | 32,768 | 104.061 → 97.089 | 6.70% | 6.972 | 12/12 | 100.692 → 93.606 |
| 8,192 | 49,152 | 161.737 → 150.960 | 6.66% | 10.777 | 12/12 | 157.927 → 146.945 |
| 14,336 | 16,384 | 60.061 → 56.398 | 6.10% | 3.663 | 12/12 | 56.501 → 52.704 |
| 14,336 | 32,768 | 159.167 → 148.892 | 6.46% | 10.275 | 12/12 | 154.587 → 144.067 |
| 14,336 | 49,152 | 257.992 → 241.141 | 6.53% | 16.851 | 12/12 | 252.755 → 235.518 |

全部九形状页分数和 attention 输出逐字节相同。每形状候选记录 57 次 static dispatch，generic fallback 为 0，旧版 static dispatch 为 0；模块指纹分别为 `d7b326c5139bf4d9` 和 `e0ad0e50be125b8f`。14K/48K 的配对节省范围为 16.662～17.057 ms，中位数 16.826 ms。

原始证据：`artifacts/native-prefill-static-aligned-component-20261022-r1/{manifest.json,results.json,result.json,stdout.log,frozen/}`。全部从资格通过的冻结源码运行，结束时再次验证源码 SHA；`frozen_source_unchanged=true`。可复用 `scripts/benchmark_native_prefill_static_aligned.py` 重做组件测量。

## 解释限制

三个采样层共用同一组固定随机 Q/K/V 和 scale，以保证两臂一致；真实服务每层有不同 cache，中间还有其他模型计算，缓存局部性和真实输入数值分布不同。本测量不包含其余模型层、MTP、逻辑子包接受、归档、构图、重选或跨请求调度；实际服务还有不能满足静态资格的 replay/小块/部分内部边界。

`16.851 ms × 16 ≈ 0.270 s` 仅是收益量级，既不是整轮已测加速，也不是严格上限。不能将三层组件的 6.53% 用作整模型吞吐提升；默认开关保持关闭，整机是否值得启用需单独控制变量验证。
