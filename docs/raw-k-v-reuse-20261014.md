# GPU raw-K、resident V 与 CPU 历史维护，2026-10-14

本轮完成 qw3 中尚未移植的 GPU raw-K 缓存、resident V 复用，并接入按数据包
大小执行的 CPU scatter／MeanK 维护。原版 selector、主模型 MeanK 评分、原生
MTP(k=3) 和 CUDA Graph decode 保留。

**装配搬运有明显改善，稳态 decode 尚未获得稳定提升。** 八用户 64K＋4K 的
K/V payload H2D 从约 363 GiB 降至约 152 GiB，约减少 58%；装配路径累计时间
从 17.32 秒降至 13.72／14.67 秒，约减少 15–21%。同轮 steady decode 为
486.92／501.24 token/s，低于本次新测基线 511.98。八用户 32K＋16K 为
513.79，对照 510.40，差异约 0.66%，不能据此声称明显提升。

## 实现与资源边界

参照本机 qw3 `src/qwen_executor.cpp` 约 6320–6445 行的 raw-K cache 和
14402 行附近的 resident stage-in，核心源码未修改。

- raw-K cache 只保存完整 sealed 页的原始 BF16 K。键为 CPU authority
  namespace、slot、generation；命中也验证当前权威记录。淘汰只影响性能。
  GPU 槽按请求配额限制，等待／工具边界释放该 namespace 的缓存占用。
- 先处理本次计划的全部命中，再写入 miss，GPU 操作按执行 stream 和完成
  fence 排序。一次写入 batch 不会将同一个槽分配给两个不同记录。
- `working_k_resident` 的 V 从旧 native 页读取，先将全部移动源暂存，再执行
  所有目标写入，覆盖跨 chunk 的交叠移动。raw K 仍从 immutable 来源重新
  RoPE，不从 FP8 的旋转 K 反推。冷恢复强制重新加载 V；部分尾页走旧路径。
- resident V 暂存为 runner 共享池，每次复用等待上一批 consumer event。
  失败的部分 DMA／kernel 也保留完成 fence，未完成的装配不能提交。
- CPU cache 读取可只 gather raw-K 或 V 字段，native memcpy 释放 GIL，仍使用
  有界常驻 worker。NVMe miss 保留原 tier 的完整记录读取，再取字段；额外
  fallback slab 单独计入 I/O admission，namespace／generation／lease 均验证。
- CPU scatter 只在收到下一轮 scheduler 的 EOS／取消结果后处理上一批。
  一个 worker 处理已有 D2H 队列，当前批次提交前 join；刷新、恢复、容量反馈
  和所有 foreground drain 都作为发布屏障。下载仍限制为两个 slot。
- `cpu_scatter_min_tokens=128`：大 prefill 包放后台，小 decode 包在当前 GPU
  计算提交后维护，避免为 32-token 级别包反复交接线程及等待 D2H。

测试显式增加 **4 GiB raw-K budget**，实际槽容量 4,291,559,424 字节；共享 V
池上限为 570,425,344 字节，约 0.531 GiB。两者在 native working KV 预算之外。
CPU cache 保持 100 GiB；该限额不是整个进程 RSS 的限额，索引、staging、
checkpoint 和系统内存仍另行占用。默认配置关闭新增开关，显式测试配置见
[cached profile](../examples/qwen-nvfp4-meank-mtp-cached-elastic.json)。

## 相同负载的测量

单张 RTX PRO 6000 Blackwell Server 96 GB；vLLM 0.30.0、PyTorch
2.13.0+cu130、FlashInfer 0.6.18.post1。Qwen3.8-27B checkpoint 使用 NVFP4／
FP8 混合权重、BF16 activation、target／MTP FP8 E4M3 KV、CUTLASS linear
backend。K=32768、G=16384、max_num_seqs=8、max_num_batched_tokens=4224、
GPU utilization=0.82、active admission=48 GiB；APC off、同步调度、单 rank。

基线为本轮开始时的发布提交 `7cb52bb`，没有把上一轮较低的 486／492 成绩
作为收益分母。八个独立鉴权用户使用同一冻结输入、温度 0、seed 104729、
unique session salts 和 ignore_eos。性能运行关闭 profiler／额外恢复 oracle，
无其他模型请求；共同 decode 窗口裁掉两端各 128 输出 token，并用实际 token
IDs 和 engine running／waiting／prompt counter／preemption 指标交叉确认。
所有轮次使用同一个测量脚本和输入 hash。

| 八用户负载 | 稳态 decode，token/s | 整轮输出，token/s | TTFT p95，秒 | 完成时间，秒 |
| --- | ---: | ---: | ---: | ---: |
| 基线，64K＋4K | 511.98 | 147.89 | 154.25 | 221.56 |
| 全部包后台，64K＋4K | 470.86 | 146.50 | 152.35 | 223.67 |
| 按包大小，64K＋4K，第 1 轮 | 486.92 | 149.49 | 150.23 | 219.20 |
| 按包大小，64K＋4K，第 2 轮 | 501.24 | 150.24 | 150.06 | 218.11 |
| 基线，32K＋16K | 510.40 | 411.59 | 36.65 | 318.46 |
| 按包大小，32K＋16K | 513.79 | 416.27 | 34.18 | 314.87 |

六轮完成 48/48 请求，零抢占。另保留首轮后台 drain 并发失败的 0/8 记录，
没有计入成功样本。TTFT 来自网关内部的后端 SSE；公开 turn 接口在完成时
返回 JSON，不能将它当成客户端公开流式首 token 的延迟。

全部包后台与按包大小的第 1 轮，完成输出的 hash 多重集合一致、MTP counters
一致；共同窗口每用户每事件约 2.410 输出。平均 SSE 事件间隔由 40.94 降至
39.60 ms，支持小包交接策略改善了该路径，但这是事件时间而非 CUDA kernel
计时。基线每用户每事件约 2.483，接受率和 batch 组合也影响 token/s，不能将
不同输出轨迹的差异全部归因于 CPU 或 K/V 搬运。

两轮按包大小的 64K 整轮中位吞吐比基线高约 1.33%，32K＋16K 高约 1.14%。
这些整轮收益较小。raw-cache 命中及 V 复用主要发生在历史刷新，不减少纯
decode 每步的主模型和 MTP 工作量；本轮验证明确支持更少的装配搬运，未验证
持续工具负载、带宽竞争时的 SLO 改善。

## 正确性和复现

首轮真实并发暴露非推测 GDN 续接直接调用 drain、未等待 CPU worker 的问题。
新增复现：worker 停在 APPENDING 状态时 foreground drain 必须等待，不能
再次追加；终止包仍按 EOS prefix 裁剪并保留 batch 邻居。所有 foreground
drain 现在统一 join，worker 自己不 join 自己。

恢复 oracle 还发现只读 `_target_bytes` 观察器误带入写入优化参数的问题。
只读 views 现在不激活 raw-cache／V 装配，保持共享 staging 几何；新增了
对应回归。两次失败记录均保留。

最终 CPU 回归 587 项通过。真实 64K＋257 暂停后追加 32＋128 的 oracle
也通过：17 层 working KV（含一层 MTP）1,185,415,168 字节与 96 个 recurrent
tensor 相同，保留 65792 个已计算历史 token，恢复后前端历史为 65953。
该验证关闭 cold greedy 对照，证明状态字节而非完整任务质量。最后的只读
observer 修正不改变已测的 direct packed assembly 路径。
普通服务（关闭 profiler／oracle）也完成两轮鉴权、工具等待／恢复请求，
历史从 272 增至 292 token；每轮生成 16 token，active、queue 和 restore
预算在结束后归零。

相关 GPU 回归共 38 项通过，其中两项使用实际 17 层、4 KV head、256 head
dim、128-token 历史页和 1600-token native 页，在 FP32／BF16 RoPE 表下逐
字节比较旧装配与新装配，覆盖冷／热交替与跨 native 页移动。

MTP 的 qw3 prefix-prime KV-only 分支没有直接复制：原生 proposer 的 `_prefill`
输出参与首个 draft sampling 和下一步 hidden state，而 qw3 的对应分支丢弃
这些输出。删除原生计算会改变语义；本轮保持原生 proposer 的计算。

吞吐用 `scripts/qualify_elastic_long_output.py`；逐批装配统计用
`scripts/summarize_kvmem_assembly.py`，精确指定 capture 行区间。原始证据在
`artifacts/raw-assembly-20261014/`，不提交凭据、模型或生成正文；汇总见
[公开 JSON](benchmarks/raw-k-v-reuse-20261014.json)。
