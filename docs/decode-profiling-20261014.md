# MeanK + 原生 MTP：decode 性能归因，2026-10-14

本轮在同一张 RTX PRO 6000 Blackwell Server 96GB 上核对 qw3 的全部主要
KVMem 功能，并对当前 MeanK + NVFP4/FP8 混合权重 + FP8 KV + MTP(k=3)
采集真实八路 CPU/CUDA trace。完整功能核对见
[迁移核对表](qw3-parity-audit-20261014.md)。

## 已确认的原因

1. 当前 KVMem 入口强制 `compilation.mode=0` 和单个同步在途 batch。
   旧 591.92 的完整上下文路径采用默认编译、FULL_AND_PIECEWISE、async、
   native aligned GDN/APC 和 CPU offload。它们不是同一个执行配置。
2. 旧服务启动日志实际选择 `FlashInferCutlassNvFp4LinearKernel` 和
   `FlashInferFP8ScaledMMLinearKernel`。当前入口明确覆盖为
   `CutlassNvFp4LinearKernel` / `CutlassFP8ScaledMMLinearKernel`。
   此前 eager 的最佳后端不能直接当作 CUDA Graph decode 的最佳后端。
3. 相同同步/未编译/CUTLASS 参数下，KVMem 与 native trace 的主模型 GPU
   时间约 30 ms，MTP 约 5.3 ms。当前八路稳态窗口没有 MeanK 选页和装配；
   raw-K cache / resident V 优化主要作用于刷新、prefill 和 resume。
4. KVMem 的持续附加工作集中在 accepted history 的 CPU 复制、索引更新和
   元数据事务。它有可减少的成本，但不是全部 benchmark 差距。
5. 仅在原生控制中切换 async，没有显示明显吞吐变化；本负载不能据此将
   性能差距主要归因于同步 scheduling。动态工具/混合 prefill 仍需单独测量。

本轮未更改 vLLM 主模型验证算法、模型权重、KVMem scoring 或线上调度契约。
native 参数控制用于定位原因；当前 adapter 的兼容性 guard 保留。

## 匹配的 CPU/CUDA trace

使用 vLLM 0.30.0、Torch 2.13.0+cu130、FlashInfer 0.6.18.post1。冻结输入
SHA256 为 `19543eb03ad997478d92dc001aab03bd091f3b92a17c639e721e1d09fa3d3c8d`，
C8、32768 input / 4096 output、temperature=0、seed=104729、unique salts。
普通 target/MTP decode 使用 FULL_DECODE_ONLY，CUTLASS 两种量化后端、
GDN none、APC off、max_num_seqs=8、max_num_batched_tokens=4224、utilization=.82。
KVMem 参数是 K32768/G16384、CPU cache100GiB、raw GPU cache4GiB、resident V
reuse、async seal 和 min128 自适应 CPU scatter。

每个 trace 校验 48 个纯 decode CPU 迭代且每轮 generation=8；delay2。
分析第二个 CPU 迭代起点至最后一个起点的 46 个完整周期，GPU 用区间并集，
graph kernel 通过 CPU CUDA launch correlation 归因。CPU/GPU 同名 annotation
分开，CPU operator 扣除嵌套 aten 范围。此次增强的 analyzer 用 prefix-end 索引
跳过已结束范围，减少分析进程本身的 CPU 消耗。

| 阶段，ms / sampled cycle | MeanK KVMem | native 同参数、无 KVMem hooks |
| --- | ---: | ---: |
| GPU 主模型 graph | 30.209 | 30.040 |
| GPU MTP prefix graph | 1.729 | 1.725 |
| GPU MTP draft graph | 3.602 | 3.658 |
| GPU sampling | 0.660 | 0.694 |
| GPU 全设备活动区间并集 | 36.337 | 36.196 |
| GPU history D2H | 0.0355 | 无此捕获路径 |
| GPU history pack | 0.0229 | 无此捕获路径 |
| CPU authority.append inclusive | 2.233 | 无 |
| CPU capture.drain inclusive | 2.607 | 无 |

append inclusive 已包含 flush 等子范围，不能与 drain、其他 scope 或 GPU
时间相加。append exclusive 约2.022 ms；其 CPU Torch operator exclusive
每轮 copy_ 约0.700 ms，sum0.158 ms，add_0.064 ms；其余包含原生事务、
Python/pybind、分配及未标注工作。不能据此声称全部2.233 ms都可兑现为吞吐收益。

KVMem GPU kernel 热点为 NVFP4 GEMM13.35 ms、FP8 GEMM5.97 ms、
attention约7.45 ms、GDN多token kernel2.37 ms。这是所有相关阶段的 kernel
类别累计时间，与阶段 GPU union 有包含/重叠关系。FP8 GEMM 对应 checkpoint
已有 FP8 层；不是将所有 NVFP4 权重改用 FP8。

native sample 的 CPU cycle43.254 ms，KVMem41.646 ms，不能据此宣称
KVMem更快：CUPTI buffer overhead 等采样扰动不同，trace中的墙钟不是正式
吞吐。native/KVMem CUDA kernels 全部找到 launch correlation；稳态 trace
没有 select/assemble。0.059 ms pack+D2H 不包含 graph 内融合 QK 捕获，后者
已经计入 target/MTP graph。单份 GPU 差值0.14 ms也不是所有捕获开销的精确隔离值。

另对已资格测量的原生 default compilation + FlashInfer 采48轮C8 decode。
该组合 target GPU24.883 ms、MTP prefix1.531 ms、MTP draft2.897 ms、
sampling0.553 ms；每完整周期约1038个kernel，而未编译CUTLASS native约2855个。
GPU时间和kernel数量共同支持执行配置造成了实质差距；这个trace同时改变了
编译和后端，分别贡献多少以未采样的逐项控制判断，不把组合GPU差值全部归给编译。

## 未采样参数控制

正式成绩来自完全关闭 profiler/诊断注解且没有 CPU 测试/trace分析并行工作的
独占后端。精确 SSE token ID 计数，C8共同窗口各路两端裁128 token；保留所有
停顿，native running=8/waiting=0/prompt counter稳定/零抢占提供旁证。
输入、温度、长度、量化精度与 batch参数固定；参数不同的控制单列。

native 同步 CUTLASS 两轮547.22/545.29 token/s；只改 async 两轮547.37/545.49，
几乎一致。默认编译/FULL_AND_PIECEWISE + CUTLASS 两轮632.76/632.61。
后者 SSE 周期从约36.6降至32.69 ms，接受输出从约2.502增至2.585 token/user/event。
这既有执行周期变化，也有生成/接受量变化，不能把16%的token/s差全部当作CPU节省。

保持编译/async及其余参数相同，恢复默认FlashInfer，隔离复测03/04为
683.09/683.01 token/s，SSE周期约30.37 ms，每事件接受量约2.59。
同一路径32K+16K单轮为699.49 token/s；它是无KVMem hooks的直接引擎控制，
没有旧 benchmark 的 GDN align/APC/CPU offload/认证入口配置，不能当作
591.92的逐项等条件复测或 KVMem 入口成绩。

| C8 / 32K+4K，未采样 | 两轮 decode token/s | 合并窗口 token/s |
| --- | ---: | ---: |
| 原生，未编译，同步，CUTLASS | 547.22 / 545.29 | 546.25 |
| 原生，未编译，async，CUTLASS | 547.37 / 545.49 | 546.43 |
| 原生，默认编译/graph，async，CUTLASS | 632.76 / 632.61 | 632.69 |
| 原生，默认编译/graph，async，FlashInfer | 683.09 / 683.01 | 683.05 |
| 当前 KVMem，未编译，同步，CUTLASS | 450.44 / 518.83 | 482.22 |
| 当前 KVMem，未编译，同步，FlashInfer | 508.66 / 533.64 | 520.74 |

KVMem CUTLASS 首批/后续差异很大，接受输出量均约2.509 token/user/event，
两轮输出 hash multiset 相同；SSE周期44.57/38.69 ms。对应native同步约36.6 ms。
首批NVML GPU active约82%，native对照约100%，SM时钟均约2.4GHz。
NVML active不等于trace区间并集；这支持额外host停顿而不是显著降频的解释，
尚未进一步隔离首次host分配、归档和后台工作各自贡献。不能只取518.83作为全部
KVMem开销的固定估计，也不能把首批差值全叫CPU append成本。

FlashInfer在当前KVMem暖态的周期约36.77 ms、接受量约2.453；暖态吞吐仅比
CUTLASS暖态高约2.9%，两次复测不足以承诺稳定收益。保留普通部署CUTLASS配置。

| C8 / 32K+16K，单轮 | decode token/s | 整轮输出 token/s |
| --- | ---: | ---: |
| 当前 KVMem + MTP3，真实8用户入口 | 510.48 | 415.01 |
| 原生 default compilation + FlashInfer，直接引擎控制 | 699.49 | 551.25 |

两组都8/8成功、零抢占。长输出原生组的接受量/上下文路径与KVMem不同，所以
699与510的差值不是某一个组件的精确成本。32K输入没有超过K，主要隔离持续
history维护；超过K的真实语义选页/恢复和装配证据仍见此前64K测试报告。

完整配置、窗口、接受量、hash、逐阶段计时、近似窗口NVML统计见
[结构化汇总](benchmarks/qw3-audit-decode-profile-20261014.json)。
先前与CPU trace分析或回归进程重叠的样本全部保留 `eligibility.json`，不纳入
正式控制。它们包括 KVMem473/523/518和最初两轮 compiled-auto的观测。
CPU回归使用本机loopback服务；受限网络环境中的首次运行失败/被中断，随后在
允许loopback、明确关闭GPU的环境单独重跑。

SSE cycle 是客户端事件代理，不能替代独立 GPU step timer；不同配置的 output
hash 和 MTP acceptance 可能不同。32K+4K 与32K+16K控制分别报告，不能把它们
直接相减构造某个组件的开销。旧591.92另含APC/GDN align/native CPU offload及
不同native容量参数，也不是同参数逐token对照。

## 下一步

优先解除 KVMem adapter 对原生 compilation 的限制：验证 persistent Q/K/V
捕获的 side effects、provisional Query、batch邻居GDN rollback、MTP prefix
和恢复 oracle，保留原生模型验证实现。然后批量处理CPU accepted-row copy/
index/metadata，最后评估 async 的 batch-owned资源和SLO；当前C8对照不支持
把async视为最大的单项收益。

GPU MeanK index、原版精确 scorer、Q-only/early exit、prefix replay复用与提前
prefetch仍有价值，但多数主要影响TTFT。显式Query span/请求策略和FP16生产
索引模式也需补齐，才能缩小与qw3在上下文管理方式上的差异。

## 验证与复现

CPU回归593 passed / 66 skipped / 14 warnings；GPU明确关闭，loopback HTTP
测试使用允许的本机环境。三份真实CPU/CUDA trace均通过48轮/8路/无prefill
校验；模型执行逻辑未改。普通服务两轮鉴权、工具等待/恢复成功，历史272→292，
各回合active/queue/restore归零，临时会话已删除。73个package文件（含native
extension）与wheel/安装目录/源码逐字节相同；wheel SHA256：
`f84b2545fa41b8607001d97746d27ee1e564768519bec7f67d0b67e590a9ee28`。

`profile_elastic_decode.py --phase decode` 支持MeanK MTP和完整上下文native MTP，
`--direct` 使用直接引擎SSE控制；`--phase both` 保留原MeanK prefill/decode两阶段
收集。原生控制显式指定 `--worker-cls vllm_kvmem.profile_worker.NativeProfileWorker`，
只增加profiler启用时的注解，没有安装KVMem capture/selector/connector hooks。
普通基准需完全省略profiler与该worker类，不能只调用stop_profile后就当正式成绩。

本机证据在 `artifacts/qw3-audit-profile-20261014/`。配置和launch plan、冻结输入
引用、结果和trace均保留；原始trace、credentials、模型、token IDs/正文未提交。
恢复服务使用原KVMem参数，gateway18382/backend18381，profiler和恢复oracle关闭。
