# 八路历史 prefill 阶段隔离实测（2026-11-04）

本轮继续验证 P4：在 native post-forward 反馈前提交已完成的逐层 pressure 组装，
及时释放原 lane/page lease，并把 ACK 放入本轮反馈。未改变选块、预算或 attention 统计。
结果目录：`artifacts/g4k-prefill-phase-abba-20261104-r1/`。

**本次测得该阶段 prefill 吞吐提高 14.73%，但八路严格一致性仍未通过。**
这完成了阶段隔离诊断的性能测量，不能据此声称整个 Gen4K 回退已解决、正常连续服务提速同样多，
或已通过 102 题正确性验收。候选仍默认关闭，没有切换正式实验。

## 本轮代码与控制

- 新增 `src/vllm_kvmem/serving/prefill_phase_gate.py`：仅测量使用的有限 cohort gate。
  八个 owner 全部完成原始历史 prefill 后才放行最终 Query；pressure 始终可执行。
  默认不安装，冻结测量副本显式启用。拒绝重复 owner、提前执行 Query 或超出容量。
- 新增 `scripts/benchmark_layerwise_prefill_phase.py`：事先冻结 A1/B1/B2/A2 四轮顺序；
  每轮独立启动原八路驱动、保存全部首次回答，不重试、不外部评分。
  从原冻结 runtime 加最小补丁，保留其原有 RoPE 实现，不覆盖为当前工作区完整文件。
- 新增 `scripts/analyze_prefill_phase_abba.py`：只读分析阶段时长，并比较 A/A、B/B、A/B。
- `tests/test_prefill_phase_gate.py` 与已有 completion poll、execution flags、refresh stage
  合同测试合计 **42 passed**。

A 为原合格 inline 基线；B 为该基线加 P4，未叠加 P2 prepared reader 或未通过的 native score export。
两组具有相同的测量 gate。模型、完整输入、NVFP4、Active=16384、Gen=4096、block32、sink41、
recent0、采样、thinking 与输出上限均沿用冻结 profile；原生 batch cap=16424，max_num_seqs=8。
逐层累计 attention pressure 与最终逐层 Mean-K 不变。

每轮使用原 first8 完整输入，共 **865440 历史 tokens**；每轮 **180 次 prefill pressure**。
时间从第一条原请求派发到最后一个历史 prefill 完成，包含 pressure/组装/HTTP 等待，
排除模型加载、最终 Query 和所有生成。每轮均检查计时区间没有 Query 执行或 decode。
不启用新 profiler，不同时运行 CPU 微基准。

gate 改变了请求阶段重叠关系，因此这是**八路阶段隔离诊断**，不是普通连续补位服务吞吐。

## 实际吞吐

| 预定顺序 | 配置 | 历史 prefill 秒 | tokens/s |
|---|---|---:|---:|
| A1 | inline 基线 | 150.091 | 5766.09 |
| B1 | 加 P4 完成交接 | 130.888 | 6612.05 |
| B2 | 加 P4 完成交接 | 129.042 | 6706.66 |
| A2 | inline 基线 | 148.138 | 5842.11 |

按相同 token 总数除以两轮总时长，基线 **5803.85**，P4 **6659.02 tokens/s**：
吞吐 **+14.73%**，阶段墙钟 **-12.84%**。两轮 B 均快于两轮 A；
各臂两轮吞吐极差约 1.3%～1.4%。只有两次观测，不将其当统计置信区间。

四轮完整推理时间与输出 token 数也已保存，但输出长度不同，不据此计算固定工作量端到端收益。
例如 B1 的 prefill 比 A1 快，完整推理反而更久；这正是必须分开测量的原因。

## 收益来源与剩余成本

以下为原有异步事件计时，每臂合计 360 次 pressure 的均值；不同 owner 可重叠，不能相加为墙钟。

| 阶段 | 基线 | P4 |
|---|---:|---:|
| suspended → ready | 1.850 s | 0.839 s |
| assembly queued → started | 1.276 s | 0.405 s |
| assembly started → submitted | 0.194 s | 0.201 s |
| assembly submitted → ready | 0.145 s | 0.202 s |

P4 两轮分别有 180/180、179/180 次完成由新检查路径提交。队列等待显著减少，
请求更早恢复，记录的历史 public forward 批次数从两轮均 418 降为两轮均 313。
**收益主要在完成交接及其带来的请求调度变化；KV 组装主机提交本身没有变快。**

`assembly_started` 在逐层选择后，因此 queued 区间仍包含选块；submitted→ready 包括观察完成的
延迟，不等于纯 GPU copy。现有 step CUDA 事件计时还包含主机提交间隙、MTP、采样和 capture，
不能当作纯 attention kernel 时间。初始化期间少量批次早于 timing 注册，分析报告列出 token
覆盖率；本轮吞吐的分母使用完整阶段墙钟，不使用这些不完整分项的和。

## 一致性与验收边界

既有独立 P4 单路完整输入资格仍为四题 pressure/Query 选块与完整输出 **4/4 相同**：
`artifacts/g4k-pressure-implementation-20261104-r1/post-forward-reap-serial/serial-equivalence.json`。
这证明该组单路输入的结果，不能代替八路检验。

本轮 A1/A2、B1/B2、A1/B1 的八道题，严格 pressure、Query、完整输出 token 一致均 **0/8**。
基线自身也不复现，说明不能把全部差异归因于 P4，也不能豁免候选的一致性要求。
检查样例的差异已出现于第一次 pressure（约 20352/20384 accepted tokens），
两边候选前缀长度相同但选中块不同，之后输出也不同。动态 batch 轨迹不同；
**尚未隔离具体数值根因，不能断言仅为浮点误差或对准确率无影响。**
完整首次答案与首次差异均保留，没有随机复跑直到匹配。

最终核验四个冻结 runtime、原输入请求哈希、profile 字节一致性，以及 32 个完整结果。
GPU 已释放（0 个计算进程，14 MiB 基础占用）。没有恢复 MAB、启动巡检、外部评分或新 102 题实验。

## 后续按收益推进

1. 优先固定多 owner 的执行轨迹并检查原 attention 累计是否仍不复现，再比较同状态的块集合和输出；
   不以重采样或忽略差异替代验证。
2. P4 已能显著降低等待。下一项成本是每次约 200ms 的 KV 主机组装提交；先定位真实复制/准入占比，
   复用已有实现，单独冻结候选，不叠加未合格开关。
3. 一致性通过后，再做正常八路连续服务对照，评估收益能否转化到原 102 题。
   本轮阶段隔离结果不能直接替代该验收，也不能与原 Gen16K 总墙钟跨协议相除。

主要证据：`comparison.json`、`final-audit/summary.json`、`final-audit/*_vs_*.json`、
`completion-audit.json`，均位于本轮结果目录。NUMA 观察赶在模型释放后，未取得有效进程页位置证据；
未据此得出 NUMA 结论或改变系统策略。
