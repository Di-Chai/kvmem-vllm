# 原生 prefill 统计 CTA 门控：资格与组件结果

`action_native_prefill_cta_gate` 保持默认关闭。该候选通过正确性资格，完整三层组件耗时降低约 0.95%～2.15%；14K Q、48K 可见上下文时每批节省 3.738 ms。按 16 个这种大块作量级估算约 59.8 ms，不能解释当前整轮模型区间约 1.59 s 的差距。本轮不追加整模型对照，后续优先用同形状 Nsight 定位剩余开销。

## 实现边界

- 只修改隔离生成的 FA2 paged prefill kernel。在已有 PDL wait 后、VO/KV 循环前，线程按实际 `stat_rows` 扫描本 CTA 覆盖的 Q 行，通过一次 `__syncthreads_or` 产生统一条件。
- 扫描区间为 `floor(tile * CTA_TILE_Q / group)` 至 `min(qo_len, ceil((tile + 1) * CTA_TILE_Q / group))`，包括 GQA 两端只覆盖部分 heads 的 Q 行。不假定固定采样密度。
- 没有采样行的 CTA 跳过 `CaptureSoftmax`；有采样行的原函数和浮点运算顺序未修改。完整 causal 归一化、逻辑段、部分页及 split-K 合并保持原路径。
- CUDA Graph 每次读取当前 GPU 行表；不在 Python 根据本次采样内容固定分支。原 `block_valid_mask` 的统一提前退出保留在投票之前。
- 没有增加 ABI 参数或持久缓冲区，不改 XQA。`aligned_pieces` 与 CTA gate 是独立模块键；仅 aligned 模式增加它原有的设备 flag 参数。关闭 gate 时生成的 prefill header 与修改前原路径逐字相同。
- 报告新增 `prefill_cta_gate`，记录配置、模块来源及实际 host dispatch 次数。该计数不包括 CUDA Graph 重放，也不代表设备实际跳过的 CTA 数。

## 正确性与编译证据

CPU 回归 119 项通过。实际 GPU 资格 15 项通过，覆盖空/全/稀疏采样、GQA 边缘行、2K Q、group=1、ragged 多 owner、decode 前缀偏移、Graph 空→采样→换 owner→空、非 16 对齐 split-K、图规划中的无效填充 CTA，以及缺失/非有限统计的拒绝路径。两臂共享 Q/K/V 和页表；attention 输出、完整 LSE 缓冲区和页分数均按字节比较相同。

实际 CTA64、MMA_KV4 编译结果：旧版 `REG=255, STACK=0`，候选 `REG=255, STACK=8`。候选有一对 `STL.64/LDL.LU.64`，分别位于 KV 循环前和退出后，并非每个 KV tile 反复 spill。没有据寄存器数量推断性能，下面的组件测量包含了这些附加成本。

资格证据：`artifacts/native-prefill-cta-gate-20261022-r3/`，含全部相关源码冻结、命令、环境、stdout、源 SHA 与 `compiled/` 下的库 SHA、资源记录及 SASS。`source_unchanged=true`。

之前的 r1 在原生 planner 构建入口因 PATH 缺少 ninja 退出；r2 因测试 fixture 的 64 MiB workspace 小于 planner 所需 112.5 MiB 退出。两者均未执行候选 kernel，证据分别保留。r3 只修正环境和 fixture workspace，生产候选未因此改变。

## 完整组件测量

固定 BF16 Q、FP8 E4M3 K/V、24 Q heads、4 KV heads、head dim 256、16-token 物理页；每 32 行采样，三层按原序平均。`max_rows=512`、context capacity 65,576、max segments 2,202、width 6,301、max owners 8，两臂均分配 945,512,540 bytes 的持久 GPU/pinned 统计资源。aligned 优化均关闭，唯一变化是 CTA gate。

每种形状预热后运行 12 对 AB/BA 交错样本。以下 wall 均值包括 Action 准备、三层 attention、全部行/页归约、层平均、D2H 及完成等待；正确性复制在计时外。每种形状之后的 fixture GC 也在计时外。CUDA event 区间与 wall 分开保存；FA2 kernel 时间另用 profiler 采样，不混入这些正式 wall 数据。

| Q tokens | 可见上下文（含 Q） | 原版 ms | 门控 ms | 节省 ms | 耗时降低 | 较快的配对数 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 16,384 | 17.087 | 16.886 | 0.202 | 1.18% | 12/12 |
| 2,048 | 32,768 | 34.347 | 33.681 | 0.667 | 1.94% | 12/12 |
| 2,048 | 49,152 | 51.154 | 50.052 | 1.102 | 2.15% | 12/12 |
| 8,192 | 16,384 | 46.014 | 45.575 | 0.439 | 0.95% | 11/12 |
| 8,192 | 32,768 | 104.041 | 102.671 | 1.369 | 1.32% | 12/12 |
| 8,192 | 49,152 | 162.001 | 159.904 | 2.098 | 1.29% | 12/12 |
| 14,336 | 16,384 | 60.292 | 59.540 | 0.751 | 1.25% | 12/12 |
| 14,336 | 32,768 | 159.366 | 157.219 | 2.147 | 1.35% | 12/12 |
| 14,336 | 49,152 | 258.315 | 254.577 | 3.738 | 1.45% | 12/12 |

所有形状实际为 CTA64，页分数与 attention 输出逐字节相同；候选实际 dispatch 57 次、原版 gate dispatch 为 0，来源指纹不同且分配字节相同。14K/48K 配对净收益范围 3.480～4.219 ms，中位数 3.608 ms；独立 profiler 中三层 FA2 kernel 合计为 252.646→249.032 ms。

原始证据：`artifacts/native-prefill-cta-gate-component-20261022-r1/{manifest.json,results.json,component.stdout,source/}`。从冻结源码运行，结束时再次验证所有冻结源 SHA，结果 `source_unchanged=true`。可用 `scripts/benchmark_native_prefill_cta_gate.py` 重做组件测量。

## 不能直接换算整机吞吐的原因

这里使用固定随机 Q/K/V 和固定 scale，保证两臂一致，未模拟模型每层的真实张量分布。三个采样层共用同一组 Q/K/V；真实服务的层各有独立 cache，且中间穿插模型运算，缓存局部性不同。本轮还固定了采样密度和具体逻辑段布局。

组件不包含其余模型层、MTP、逻辑子包接受提交、历史归档、构图、压力/Query 重选或跨请求调度。整轮还有不同形状的小块和 replay，不能将三层 FA2 的百分比用作模型整体百分比。`3.738 ms × 16 ≈ 59.8 ms` 仅用于说明收益量级，不是已测整轮加速，也不是严格上下界。
