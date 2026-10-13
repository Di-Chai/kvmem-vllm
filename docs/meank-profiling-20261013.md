# MeanK＋DFlash2 并发性能归因，2026-10-13

此前 **386.89 token/s 是开启 DFlash2（3 个草稿 token）的八路稳态 decode**，
不是原生 MTP。配置为 NVFP4／FP8 混合目标权重、FP8 attention KV、BF16
DFlash2 和原生 CUDA Graph，MeanK K=32768、G=16384。本次保留 DFlash，
分别采集超过 K 后的长文 prefill 和八路共同 decode，没有更改选页算法。

结论：decode 的 GPU 时间主要用于主模型验证，主机准备／历史维护也有开销；
主动 stage-out 的 GPU 打包和 D2H 搬运很小。MeanK 选页和历史重装配发生在
本次 prefill，而不在采样的稳态 decode 内。现有 trace 不能进一步确认主模型
内部究竟受显存带宽还是计算单元限制。

## 方法和边界

- RTX PRO 6000 Blackwell Server 96 GB，vLLM 0.30.0、PyTorch 2.13.0+cu130。
  模型和冻结输入同此前 64K＋4K 测试，八个独立认证用户和 session salt。
  目标 NVFP4 与 FP8 线性后端均为 CUTLASS。每用户 65536 token 输入、4096
  token 输出，CPU 历史上限 100 GiB、active admission 48 GiB。当前 MeanK 入口
  使用同步调度（`--no-async-scheduling`），普通 decode 保留 CUDA Graph。
- 此次关闭额外恢复字节核对。原 386.89 运行开启了核对，额外终态复制主要
  位于共同 decode 窗口外；普通部署也关闭该测试选项。
- 原生 torch profiler 只采 worker，关闭 stack、shape、memory、FLOPs 记录。
  delay=2、max_iterations=80、warmup=0。prefill 提前结束，decode 自动结束。
  KVMem 注解仅在显式启用 profiler 时安装，不增加 CUDA 同步。
- decode trace 包含 80 个八路迭代，每轮验证 8×4 个 query token。取第二次
  迭代起点到最后一次起点，保留 78 个完整周期。241796 个 GPU kernel／memcpy
  事件全部关联到 CUDA launch 和所属阶段，边界裁剪两项事件。
- CPU 仅统计 `user_annotation`，排除同名 GPU 注解。CUDA Graph 的 kernel
  通过 graph-launch correlation 归因；设备繁忙时间采用区间并集，排除重叠
  stream、嵌套范围和 CPU 等待的重复计数。

**采样结果不作为新吞吐成绩。** 采样和导出暂停计入此次请求耗时，带采样
整轮约 84 token/s、共同窗口约 240 token/s，不能替代未采样的 386.89 token/s，
也不能剔除暂停后重新构造一个服务吞吐。八路请求均完整成功，无抢占。

## Decode 的归因

78 个周期覆盖 3.679 秒，平均 **47.17 ms／轮**。GPU 占比以各 GPU 活动时长
总和为分母，每轮时间使用对应阶段的 GPU 区间并集。

| 阶段 | GPU 时间／轮 | GPU 活动时间占比 |
| --- | ---: | ---: |
| 主模型验证 CUDA Graph | 30.454 ms | 85.29% |
| DFlash2 草稿 CUDA Graph | 4.016 ms | 11.25% |
| sampling | 0.684 ms | 1.92% |
| DFlash graph 外 propose 工作 | 0.212 ms | 0.59% |
| DFlash context KV 投影／维护 | 0.100 ms | 0.28% |
| 历史 KV 下载 D2H | 0.031 ms | 0.086% |
| forward 后的 KV stage-out 打包 | 0.022 ms | 0.061% |

原始 Q/K 的融合捕获及 V buffer 复制位于主模型 graph 内，已包含在 30.454 ms
中，尚未单独隔离其增量成本；0.022 ms 不代表全部 KVMem 捕获开销。
其余为 runner 准备等少量 GPU 操作。全设备活动区间并集为 2.783 秒，占
窗口 75.64%；约 11.49 ms／轮没有被所记录的 GPU 活动覆盖。这不是硬件
利用率计数器，也不能把全部未覆盖时间归因于 KVMem 或视为可消除开销。

| 最重的 kernel 类别 | 累计时间／轮 | GPU 活动时间占比 |
| --- | ---: | ---: |
| 目标 NVFP4 CUTLASS GEMM | 11.95 ms | 33.46% |
| 目标 attention `kernel_mha` | 6.43 ms | 18.00% |
| 目标 FP8 CUTLASS GEMM | 5.96 ms | 16.68% |
| GDN 多 token 验证 kernel | 2.52 ms | 7.05% |

GEMM 合计约一半 GPU 活动时间。出现 FP8 GEMM 是 checkpoint 本身含 FP8 层，
不表示 NVFP4 权重全部改用 FP8 执行。GDN kernel 名称中的 `mtp` 指其多 token
验证实现；实际 speculative method 为 `dflash`，不能用 kernel 名判断启用了 MTP。

CPU 侧每轮 `capture.drain` 累计约 3.83 ms，其中八用户历史 `append` 约
3.54 ms；`flush` 排除嵌套 KVMem 子范围后的时间约 2.13 ms。这些时间部分
与 GPU 执行重叠，不能叠加到 30.45 ms 上，也不是可直接兑现的优化收益。
78 次 `cudaEventSynchronize` 合计 2.095 秒，反映主机等待设备执行，不能再次
计入独立成本。

1248 次 flush 中，9 次包含封页写入，合计 92.56 ms；其余 1239 次合计
74.95 ms。实际 `archive.write` 子范围仅 1.19 ms，cache write 0.23 ms。
源码显示 flush 还包含目录扫描、页状态推进、pin、序列化和摘要计算；尚未
精确分摊这些子步骤，不能把全部 flush 时间称为磁盘等待或历史扫描成本。

本次 decode 没有 `authority.select` 或 `window.assemble`。原未采样运行的
256 次刷新也全部位于 prefill。因此选页／重装配不是该 387 token/s 稳态
窗口的直接瓶颈。此次也没有 PCIe 搬运成为首要瓶颈的证据，但没有验证更高
并发或工具执行竞争下的带宽限制。

## Prefill 的单独成本

prefill trace 仅两个实际 batch，含八次历史装配，是有限样本。CPU 累计装配
范围 1.358 秒，其中 upload host 范围 1.063 秒；另外，独立的 MeanK select
范围为 0.658 秒。设备 upload 0.196 秒、MeanK select 0.068 秒。CPU 范围含准备、发射和等待，
嵌套范围不可相加。

这一阶段 profiler 产生 2673 次 `Command Buffer Full`，累计约 1.185 秒，
明显影响墙钟时间。长期 prefill 成本优先采用此前未采样运行的记录：
**256 次刷新累计上传 344.17 GiB，装配计时合计 58.47 秒，其中 host prepare
47.12 秒，TTFT p95 231.74 秒**。事务累计计时也不是独立可扣除的端到端时间。
完整记录见 [吞吐报告](meank-throughput-20261013.md)。

## 优化优先级

1. 保留 DFlash，围绕目标 GEMM／attention 验证成本做相同输入、相同并发的
   backend 和草稿长度对照，同时记录接受 token 数。原未采样 387 运行每用户
   每轮平均接受约 2.425 token，不能只比较 kernel 时间而忽略接受率。进一步
   用 kernel 硬件计数器区分带宽／计算限制，再确定具体 kernel 优化。
2. 减少主机准备和历史维护：研究未提交完整页队列与 resident 计数，减少每次
   append 前后两遍目录扫描；减少封页复制，评估有界后台序列化／哈希。保留
   generation、背压、不可变 buffer 所有权和 durable publication 顺序，每项
   改动都须关闭 profiler 做吞吐 A/B。
3. 单独优化长文 prefill 页复用、host prepare 和传输合并，保留原版 MeanK
   评分、选页及 dense re-RoPE 语义。收益应先体现在 TTFT 和整轮吞吐，不能
   宣称会等量提高当前稳态 decode。

本次没有更换算法、关闭历史捕获或声明加速收益。也没有对旧 MTP 592 token/s
路径采集匹配 trace，不能把全部差距归因于 DFlash 或某个 KVMem 模块。

## 复现和交付

启动配置可添加 `profiler_config`，内容为原生 vLLM 字段：

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "trace",
  "ignore_frontend": true,
  "torch_profiler_with_stack": false,
  "torch_profiler_record_shapes": false,
  "torch_profiler_with_memory": false,
  "torch_profiler_with_flops": false,
  "torch_profiler_use_gzip": true,
  "delay_iterations": 2,
  "max_iterations": 80,
  "warmup_iterations": 0,
  "capture_torch_profiler": false,
  "detailed_trace_annotation": false
}
```

相对 trace 目录按配置文件位置解析，正常部署省略该字段。采样批次独占后端。
运行 `scripts/profile_elastic_decode.py`，提供 `--launch`（已启动服务的
launch.json）、`--input`（冻结 token ID 文件）、`--capture`（worker capture
目录）、`--output`（新目录）；默认八路、4096 token 输出。再用
`scripts/analyze_meank_profile.py TRACE --phase decode --output JSON` 分析。

本地原始证据在 `artifacts/meank-profile-20261013/run-01/`，不提交大 trace、
凭据或生成正文。可公开复核的配置及阶段统计见
[结构化 profiling 汇总](benchmarks/meank-profile-20261013.json)。
CPU 回归 504 passed、50 skipped；随后新增的采样边界校验 15 项全部通过。
它们检查不支持的重复采样日程、输出余量、完整 trace、实际并发和迭代数；
现有真实 trace 同样通过校验。wheel 已重新安装并逐文件核对 68 个 Python
源文件。普通服务已关闭 profiler，两轮认证请求成功，历史从 272 增至 292
token，结束后 active 请求及预算归零，记录在 `release-smoke.json`。
