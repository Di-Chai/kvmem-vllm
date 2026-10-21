# 14K prefill 差距的计时范围与可加分解

本记录只复算已有数据，没有启动新推理或性能测试。对象是
`prefill-physical14k-20261022-r1/candidate` 与
`active-prefill-physical14k-20261022-r1/native`，均为物理最多 14K、逻辑
2K 的形状对照。它们在不同时间运行；相同 shape 不保证上下文内容、位置、
输出、MTP 轨迹及实际 autotune tactics 全部相同，因此时间差不是单项优化的
因果估计。后续冻结 autotune 的 opaque 实验应单独报告。

## KVMem 请求内部：可以严格相加的事件区间

来源是 [KVMem 原始事件行](../artifacts/prefill-physical14k-20261022-r1/candidate/run/server/capture/model_step_latency.jsonl)。
按 `batch_geometry[0].request_id` 分组，选取两个正式长请求的 49 条公共
prefill 行，每组 `query_tokens` 合计 262045、末行 `logical_seq_len=262045`。
所有相邻行的 `sequence`、`call_index` 连续，且
`previous_sequence/previous_call_index` 指向前行；没有漏记或跳步。

对每个请求，定义：

```
M = sum(row.model_ms) / 1000
G = sum(row.previous_end_to_model_ms for row in rows[1:]) / 1000
S = M + G
```

`S` 是本请求首个 model CUDA event 到最后一个 end event 的实际连续跨度。
`G` 使用已有完成事件之间的时间，不是另一个与 model 重叠的 host 计时。
第一个 gap 属于请求前工作，必须排除。两个正式请求结果为：

| 请求 | M：model 区间和 | G：48 个内部 gap 和 | S：连续跨度 |
| --- | ---: | ---: | ---: |
| `256k-repeat-1` | 38.374262001 s | 4.800950088 s | 43.175212089 s |
| `256k-repeat-2` | 38.475554131 s | 3.650968724 s | 42.126522855 s |
| 均值 | 38.424908066 s | 4.225959406 s | 42.650867472 s |

请求 TTFT 均值为 42.975852871 s，见
[reference-comparison.json](../artifacts/active-prefill-physical14k-20261022-r1/reference-comparison.json)
的 `full256k_previously_measured_kvmem.mean_ttft_seconds`。二者边界差为
0.324985400 s；它涵盖事件区间外的请求准备、结果传递等工作，但没有单独
测量各组成部分，不能直接命名为纯 CPU 开销或全部可消除开销。

`model_ms` 本身包含模型算子、采样、MTP、捕获，以及 GPU 等待主机提交的
空隙，不是 kernel active time。`G` 包含重选/Query、下一步维护及调度空隙。
因此不能再把 pressure/Query wall、`maintenance_ms`、host-submit 或异步
归档耗时加到 `M+G` 上。它们是相互覆盖的诊断视角，不是额外独立项。

## 同一连续阶段：模型与内部 gap 的差可以相加

两边的实测阶段都追加 16320 个 token，实际三步形状为：

```
(Q=32, active=32800)
(Q=14336, active=47136)
(Q=1952, active=49088)
```

来源是同一 `reference-comparison.json` 的 `continuous_phase_comparison`。
原生 3 段、KVMem 26 段；KVMem 每个长请求贡献 13 段，不能把这些相关阶段
当成 26 次独立整请求实验。

| 同范围计时 | 原生 | KVMem | KVMem 增量 |
| --- | ---: | ---: | ---: |
| model 区间 | 2.387720027 s | 2.493433645 s | 105.713619 ms |
| 两个内部 gap | 0.000416224 s | 0.016060170 s | 15.643946 ms |
| 实际连续跨度 | 2.388136251 s | 2.509493815 s | 121.357565 ms |

这里增量严格满足 `105.713619 + 15.643946 = 121.357565 ms`。
同样的 16320 新 token 对应 **6833.78 → 6503.30 token/s**，KVMem 低
4.84%。该跨度不包含前缀建立、阶段前后重选、HTTP 或最终生成。

另一个可以直接比较的整请求口径是相同 32768-token 输入、相同六个物理
prefill 块的完整 TTFT：原生事件计时关闭时 9928.10 token/s，KVMem
9409.51 token/s。原生计时开启是 9930.24 token/s，约差 0.02%。这只能
说明完整 32K 请求的差距，不能外推为持续处理 256K 的原生吞吐。

## 两种原生加权诊断，不应混为一个数

既有 **36.832031059 s** 保持原定义：
`comparison.json.shapes` 中每个 `(query_tokens, active_seq_len,
prefill_terminal)` 的 `native_model_ms` 中位数乘 `count` 后求和。
与 KVMem 38.424908066 s 的 model 均值比较，差 1.592877007 s、4.32%。
这是匹配 shape 的模型区间诊断，不是原生完整 256K TTFT。

现有原始事件还允许另一种诊断：按完整 family 取每次实际连续
`首 model → 末 end` 跨度，包含 family 内部 gap；每类 3 次取均值，再按
KVMem 实际出现次数加权。来源是
[native-shape-plan.json](../artifacts/active-prefill-physical14k-20261022-r1/native-shape-plan.json)
的 `families[*].target_begin_index/target_shapes/source_spans`，以及
[原生原始事件](../artifacts/active-prefill-physical14k-20261022-r1/native/probe/model_step_latency.jsonl)。
每个 family 按 target_begin_index 跳过建立前缀的行，再取
`len(target_shapes)` 行；逐一核验 Q、active、terminal 与事件连续性，排除
末尾用于保持非终止状态的 8-token guard。只有名字为 `actual-00N-1/2/3`
的正式请求进入均值，warmup 不计。

| family | 权重 | 每段新增 token | model 均值 | 内部 gap 均值 | 连续跨度均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `actual-000` | 1 | 49088 | 5.544699819 s | 0.001181707 s | 5.545881526 s |
| `actual-001` | 13 | 16320 | 2.387720027 s | 0.000416224 s | 2.388136251 s |
| `actual-002` | 1 | 768 | 0.169145794 s | 0.000034443 s | 0.169180237 s |
| `actual-003` | 1 | 29 | 0.079185941 s | 0.000036501 s | 0.079222442 s |

复算为：

```
新增 token = 49088 + 13*16320 + 768 + 29 = 262045
加权 model = 36.833391900 s
加权 family 内 gap = 0.006663563 s
加权 family 连续跨度 = 36.840055463 s
```

**36.840055463 s 仍不能称为原生端到端时间。** 它重复使用了已测 steady
family 的均值，并未连续实测 13 次窗口循环；排除了各 family 的 32K 前缀
建立，以及循环间重置/恢复和请求边界。它也不是旧 36.832031059 s 的修正值：
一个使用完整 family 的均值并含内部 gap，另一个使用逐 shape 中位数。
不得用 `262045 / 36.840055463` 标成实测原生整请求吞吐。

## 真正的循环基线还缺什么

不必重跑 256K dense attention，但现成日志无法补出没有执行的生命周期。
若在原生侧保留 32K 前缀 KV、每次追加约 16K 后重置，仍要正确恢复 GDN、
MTP、位置和 scheduler 状态，并将这些实际边界成本计入连续循环。固定
前缀反复恢复也不等于 KVMem 按历史重选得到的不同前缀。

另一种实现是前缀缓存命中的多个独立请求；必须证明实际缓存命中与 GDN
状态复用，并将请求调度/缓存开销纳入测量。它会变成暖缓存多请求负载，
不同于当前关闭 prefix caching 的单个 256K 请求，不能直接替换验收口径。
在没有新循环实测前，应并列报告上述实际连续阶段、完整 32K TTFT、KVMem
完整 256K TTFT；加权原生数只保留为诊断参照。
