# Qwen3.8-27B 在双 A800 上的 TP=2 实测

两张 NVIDIA A800-SXM4-80GB 通过 NV8 互连。权重为 `sokada4/Qwen3.8-27B-GPTQ-Int4`，
版本 `2232b67a2ef5480e97c4450838a712de1346eccc`。KVMem 和原生均使用
vLLM 0.30.0、PyTorch 2.13.0+cu130、GPTQ Marlin、BF16 激活、FP8 E4M3 KV、
FlashInfer 和 MTP4。一次只运行一个引擎。下表中的 KVMem 通过公开的
`vllm-kvmem serve` 入口启动，冻结源码 SHA-256 为 `4c5d4f5b0f47b7dad7a44eca7d0dda0a81e39e5a993340183329b45878b73652`。
[运行配置](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-release/profile.json) · [源码清单](../artifacts/tp2-huazhong-20261111/runtime-tp-v13-manifest.json) · [上一份完整冻结版本](../artifacts/tp2-huazhong-20261111/v10-report.md)

## 实测吞吐

输入吞吐为双卡合计的输入 tokens/s。每行采用该负载实测最快的原生配置，以及与其
匹配的 KVMem 轮次。88K、16 路的两轮 KVMem 分别为 4,197.8 和 4,196.2 tokens/s。

| 每请求输入 | 同时提交请求数 | 原生 vLLM | KVMem TP=2 | 差幅 |
| --- | ---: | ---: | ---: | ---: |
| 54K | 1 | 4,663.6 | 4,233.6 | −9.2% |
| 54K | 16 | 4,676.1 | 4,443.7 | −5.0% |
| 54K | 32 | 4,674.8 | 4,477.5 | −4.2% |
| 88K | 16 | 4,045.4 | 4,197.8 | +3.8% |
| 88K | 32 | 4,046.9 | 4,220.5 | +4.3% |
| 120K | 16 | 3,585.8 | 4,081.2 | +13.8% |

输出吞吐取所有请求都在运行且无人排队的最长连续采样区间，单位为输出 tokens/s。

| 每请求输入 | 每请求输出 | 运行请求数 | 原生 vLLM | KVMem TP=2 | 差幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 512 | 16 | 967.4 | 845.6 | −12.6% |
| 4K | 512 | 32 | 1,377.9 | 1,168.9 | −15.2% |
| 54K | 16,640 | 16 | 915.8 | 921.3 | +0.6% |
| 88K | 2,048 | 16 | 602.3 | 726.4 | +20.6% |
| 88K | 4,096 | 32 | 687.3 | 935.3 | +36.1% |

在 88K、32 路的共同生成区间，KVMem 的 336 个采样点中，336 个记录了 32 个
运行请求。最长满载区间持续 91.63 秒，交付 85,696 个 token，整轮抢占 0 次。
最快原生满载区间持续 60.74 秒，交付 41,744 个 token，整轮抢占 1 次。
在同输入的纯 prefill 波次，KVMem 最多观测到 8 个运行请求。原生自动分块以及每请求 2K
分块分别观测到 1 和 4 个。

把读入 88K 输入和最后几个请求的尾段也计入，16 路的整批端到端输出速率为原生
78.93、KVMem 82.54 tokens/s（+4.6%），32 路分别为
140.34 和 154.68 tokens/s（+10.2%）。在 16 路共同生成区间，
最快满载原生组和 KVMem 的 MTP 接受率分别为 45.1% 和 46.6%，按四个草稿折算的
请求校验速率分别为每秒 214.7 和 253.5 次。
[各项最快对照](../artifacts/tp2-huazhong-20261111/results-tp2-release/strongest-controls.csv) · [所有对照及重复轮次](../artifacts/tp2-huazhong-20261111/results-tp2-release/all-controls.csv)

## 测量方式

每个请求使用独立会话和固定的输入 token。客户端同时发起请求，逐条记录输入哈希、
实际返回的 token ID 和到达时间。两组都关闭前缀缓存，使用贪心采样和固定输出配额。
计时脚本核对实际输入和输出数量及长度结束标记，比较脚本再匹配两组的客户端和输入哈希。
服务启动和预热独立计时。1K 按 1,024 个 token 计算。
KVMem 将同一输入的最后 45 个 token 标为 Query，使用 `kvmem_reselect=auto` 和
`kvmem_mandatory_parts=none`。

- **Prefill** 每个请求只生成一个 token。整批输入 token 总量除以从首个请求发出到
  最后一个请求收到首 token 的墙钟时间，包含排队、分块、KVMem 选页和首 token 交付。
  并发数表示同时提交的客户端数，调度器可以分批执行 prefill。
- **Decode** 每个客户端都已收到首 token 到第一个客户端完成之间构成共同生成区间。
  另外约每 250 ms 采样运行和排队请求数。满并发速率取所有请求都在运行、排队数为零
  的最长连续采样区间，以流中实际交付的输出 token 计数。采样间隔超过一秒或抢占
  计数发生变化时分段。
- **端到端输出速率** 全部输出 token 除以首个请求发出到最后一个请求末 token
  到达的墙钟时间，包含整批 prefill、decode 及并发逐渐降低的尾段。

两组的最大逻辑长度为 131,072，最大请求数为 32，每批预算为 8,192 个 token，
GPU 内存利用率参数为 0.90。长文请求中，KVMem 按每个请求 K=36,864、G=16,384
保留活动窗口，启用 CPU 历史归档。普通 prefill 保持 2,048-token 逻辑分块，
一个物理前向最多合并四块。原生 vLLM 保留完整历史。两边均请求编译和 CUDA Graph，
运行日志显示 FlashInfer 与当前 MTP 组合采用 piecewise graph。
测试实例有 20 核 CPU 配额和 400 GiB 主机内存上限。KVMem 的 CPU 读缓存为每个
rank 64 GiB，性能测试的会话检查点池为每个 rank 4 GiB。

原生有三种调度组合，包括同步调度加自动分块，默认异步调度加每请求 2K 分块，
以及默认异步调度加自动分块。用实际参数和实机 vLLM 解析器复核，后两组
`async_scheduling=true`，自动分块组 `long_prefill_token_threshold=0`。88K、32 路
生成 4,096 个 token 的波次，每十秒采样的每卡显存峰值为 KVMem 75.70 GiB、
原生两个异步调度组分别为 74.33 和 74.41 GiB。
[原生生效配置](../artifacts/tp2-huazhong-20261111/runs/native-tp2-default-auto/resolved-scheduler.json) · [GPU 采样](../artifacts/tp2-huazhong-20261111/results-tp2-release/gpu-resources.json)

原生 2K 分块的 88K、32 路混合负载写完逐请求记录和汇总时，SSH 监测连接中断。
核对了 32 份输入哈希、长度结束标记、流分块、131,072 个实际输出和完整采样后
补登记该轮完成，沿用其原始时间戳。
[完成记录](../artifacts/tp2-huazhong-20261111/runs/native-tp2-default-2048/cases/decode-88k-c32-r1/completed.json)

长输入下，KVMem 与原生参与注意力计算的上下文不同。在 88K 输入的 16／32 路
长回答中，与原生 2K 分块对照整条完全相同的分别为 0/16 和 0/32，
共同输出前缀中位数为 14 和 14 个 token。这轮测量给出吞吐和状态一致性，任务准确率需要单独评估。

## 54K prefill 的开销

54K 输入下，首个通过并发测试的 TP 版本在 16 路和 32 路分别为 4,147 和 4,247
tokens/s。物理分块合并、后台窗口重建与公开前向重叠等改动合计，使正式版本达到
4,444 和 4,478 tokens/s，分别变化 +7.2% 和 +5.4%。
[首轮 TP 结果](../artifacts/tp2-huazhong-20261111/comparison-baseline.json)

预热后，55,296-token 单请求首 token 用时 13.061 秒，原生同步对照为 11.857 秒。
在第 53,184 个 token，容量刷新耗时 0.371 秒，其中组装 0.248 秒。在第
55,232 个 token，最终 Query 选页耗时 0.586 秒，其中组装 0.200 秒。两次均保留
36,864 个 token，rank 0 分别从主机上传约 285 和 385 MiB。这条 54K 输入越过
第一次刷新边界后，剩余的 prefill 约 2K。88K 和 120K 的后续分块使用较短活动窗口的时间更长。
[刷新分解](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-release/54k-overhead.json)

## 归档线程的 TP 发布策略

在第一轮压测期间，工作区增加了默认开启的归档等待优化和会话状态缓存。对含有
这些改动的冻结版本 `48e633a2`，引擎及两个 GPU worker 均实测继承
`KVMEM_GIL_GATE_DRAIN_OPEN=1` 和 `KVMEM_LIFECYCLE_SNAPSHOT_CACHE=1`。
在 88K 输入、16 路、各生成 2,048 个 token 的组合中，监控记录到 4,345 个输出后
GPU 1 报 CUDA 非法内存访问，16 条流均未完整返回。调度快照显示 15 个前台请求
各有 5 个 MTP 验证行，同时两个 rank 的第 132 个私有 Query 窗口处于组装阶段。
最早的异常由 NCCL watchdog 查询 CUDA 事件时报告，具体出错的 kernel 尚未确定。
[首次异常与调度形状](../artifacts/tp2-huazhong-20261111/results-tp2-integrated/failure-v11-defaults.json)

强制 CUDA 同步后同一负载完成 16/16。正式的单开关对照恢复正常异步执行，保持
会话状态缓存开启，仅将 `KVMEM_GIL_GATE_DRAIN_OPEN` 设为 0。该轮 16 个请求
返回全部 32,768 个 token，共同生成区间为 739.6 tokens/s。在 4K 输入、32 路、
每条 512 个输出的共同区间，同一快照开关为 1 和 0 时分别为 743.4 和 1,162.2
tokens/s。公开 TP 启动配置采用后一策略，保留原有 50 ms GIL gate。单卡仍可使用
其原来的等待策略。另一个实验开关 `KVMEM_NATIVE_SCORE_BATCHED_SELECT` 在这些
轮次中使用其默认值 0。
[单开关对照](../artifacts/tp2-huazhong-20261111/results-tp2-integrated/drain-open-vs-ordinary-gate.json) ·
[16 路实际返回记录](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-v11-drain-off/cases/decode-88k-c16-r1/summary.json)

公开入口的首次复测还暴露了独立的调度边界错误。2K 父分块的最后一段被裁为
1K 时，旧校验用子段的 1K 计算步长，拒绝其正确的物理偏移。修正后，父组决定
步长，子段仍须由完整的 32-token 记录构成，组级检查只允许裁短最后一段。
对 `[2048, 2048, 1024]`，CPU 测试验证了物理偏移 `[0, 2048, 4096]` 和分段
接受数量。A800 实际 FA2 测试同时通过独立分段、合并投影和 FP32 参考值比较。
[规划回归](../artifacts/tp2-huazhong-20261111/pytest-clipped-span-after.log) ·
[A800 组件验证](../artifacts/tp2-huazhong-20261111/logs/half-tail-gpu.log)

## 跨卡选页、滚动生成与恢复

每层在每个 rank 上有 12 个 Q 头和 2 个 KV 头。公开前向复用 attention kernel 的
页分数，先做各自 head 的归一化，再跨卡求均值。协调 rank 广播有序的历史起点
和长度，各卡映射到自己的 KV 槽。后台重建和公开模型前向走分开的通信通道，调度器
收齐两个 rank 对同一命令的确认后才提交窗口。若可选会话检查点只在部分 rank 写入，
调度器等待两卡清理完成，后续请求可冷重算。

正式运行共 692 次带历史范围的刷新，全部在两卡配对，选取范围和 token 数完全一致。
归档尾部 256 条共同完成记录的接受量、拒绝量和历史终点一致。两卡均复用
prefill 分数 21,891 次、decode 分数 108,831 次，验证失败为零。每卡检查点池
峰值为 4,278,156,966 字节。数值校验中，两张卡各自首选不同页，合并后选择了全局参考的第三页。
MeanK 和 FPS4 相对完整 head 的最大绝对误差均为 5.96 × 10⁻⁸。相同逻辑选择在
两个 rank 对应物理槽 `[2, 0]` 和 `[102, 100]`，故意不一致的窗口在写入前被拒绝。
[双卡审计](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-release/rank-consistency.json) · [GPU 数值参考](../artifacts/tp2-huazhong-20261111/results-tp2-release/rank-oracle-final-source.log)

滚动测试的 16 条请求均读入 55,296 个 token，各生成 16,640 个，共交付
266,240 个。每个请求均在生成偏移 16,288 时刷新，选回 36,864 个 token。
全部 16 组跨卡选择一致。与原生 2K 分块对照，整段共同生成区间的输出速率为
915.7 对 920.2 tokens/s（+0.5%），端到端为 532.4 对
508.7 tokens/s（−4.5%）。原生和 KVMem 分别抢占 0、0 次。
[滚动生成逐卡校验](../artifacts/tp2-huazhong-20261111/results-tp2-release/rolling-validation.json)

独立的驻留与主机重建两组都通过公开 CLI 启动，实际使用 Marlin 和 FlashInfer。
在 4,127 个历史 token 上，分别逐字节核对了每卡 71,842,816 字节、
17 层工作 KV（含 1 层 MTP），以及每卡 96 个 GDN 状态张量。
驻留组匹配原始工作窗口，恢复和该进程内冷重算的共同输出前缀为 8/32。
从 CPU 的不可变 FP8 历史重建时，两卡均匹配独立 CPU 重建参考，恢复和对应冷重算
的共同前缀为 8/32。归档 K 在 RoPE 前已做 FP8 量化，重建时再旋转并写回 FP8。
该样本不建立所有回答都能逐 token 复现的保证。
[驻留校验](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-release-resident/finished.json) · [主机重建校验](../artifacts/tp2-huazhong-20261111/runs/kvmem-tp2-release-immutable/finished.json) · [实际输出比较](../artifacts/tp2-huazhong-20261111/results-tp2-release/restore-output-comparison.json) · [FP8 历史格式](fp8-raw-k-20261016.md)

GPU 实测覆盖单机 TP=2、PP=1、prefill/decode context parallel 均为 1，使用全局选页与静态 GPU 准入。其他 TP 度数要求 Q 和 KV 头均可均匀独立分片。四进程测试覆盖 CPU Gloo 协议。
最终源码专项回归 743 项通过、1 项跳过。扩大到直接受影响的 251 个测试模块，
5,339 项通过、18 项跳过、65 项失败。这些失败在开发前的工作区快照中均可见，新增失败数为零。
[专项回归](../artifacts/tp2-huazhong-20261111/pytest-release.log) · [改动前后失败项对照](../artifacts/tp2-huazhong-20261111/affected-test-comparison-release.json)

## 复跑

复制双 A800 配置，把 `model`、`tokenizer` 和 `hf_config_path` 改为 GPTQ 检查点目录，再选取新的运行目录。
[双 A800 配置](../examples/serve-qwen-mtp-tp2-a800.json)

```sh
vllm-kvmem serve --config local-tp2-a800.json --run-dir runs/serve-tp2
```

测试机的模型、编译缓存、固定输入和逐请求记录保存在 `/root/localstorage/kvmem-tp`。
在该目录中，以下命令以公开服务入口完整复跑 KVMem 和已有原生记录的匹配验收。

```sh
venv/bin/python qualified_public_run.py --run repeat-tp2 --results results-repeat-tp2 --source-file release-source.txt
```

重新采集原生自动调度、自动分块的单项对照时，在两个终端分别运行以下命令。

```sh
venv/bin/python launch.py native repeat-native-tp2 --native-default-scheduling
venv/bin/python measure.py native repeat-native-tp2 decode-88k-c32-r1 32 90112 4096
```
