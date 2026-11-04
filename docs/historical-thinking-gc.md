# 旧 thinking 冗余回收与 LRU 容量后备

这组可选参数扩展现有 `historical_gc.policy=redundancy`。先在符合保护规则的
thinking 历史块中寻找重复内容，保留直接代表；冗余回收不足时再按 LRU 释放历史。
策略仍由 native worker 执行，退休、CPU 引用退出、索引回执和槽位复用沿用已有
historical GC 机制。已准入的 GPU 窗口不由策略直接释放。

计算方法参考远端分支的
[固定版本实现](https://github.com/zyf32453/kvmem-llama.cpp/blob/c02612d05e88833a854bb688c65d17e40a58b86a/kvmem/src/host/historical_kv_redundancy.cpp)。

新增策略开关全部默认 `false`，原有 LRU 和默认 redundancy 配置不自动切换行为。
这里只改变物理历史保留规则，不修改召回窗口预算、采样或服务端逻辑长度上限。

## 配置

以下是已有合格 CPU historical GC profile 中的替换片段。生产 profile 仍须保留
request planning、async seal/retirement、CPU history coordination、owned sessions，
并关闭旧 `repository_gc` 和 block graph。原有 redundancy 不支持的 prefill
publication/coalescing 组合仍然拒绝。

```json
{
  "historical_gc": {
    "mode": "delete",
    "policy": "redundancy",
    "poll_ms": 10,
    "scan_blocks": 256,
    "delete_blocks": 64,
    "ready_low": 0,
    "ready_high": 0,
    "policy_params": {
      "keep_mandatory": true,
      "thinking_only": true,
      "pool_pairwise": true,
      "all_history_references": false,
      "refresh_before_rank": true,
      "epoch_peak": true,
      "lru_fallback": true,
      "allow_lossy_fallback": false,
      "importance_weight": 0.1,
      "representative_similarity": 0.9,
      "candidate_blocks": 128,
      "reference_blocks": 32,
      "pair_comparisons": 512,
      "query_segments_per_step": 8,
      "feature_cache_bytes": 134217728,
      "query_cache_bytes": 8388608,
      "query_capture_bytes": 67108864
    }
  }
}
```

`ready_low/high` 表示可复用槽位数量；`0,0` 自动采用约 5%/10%。低于低水位后
尝试回收到高水位，另响应共享 CPU 字节压力、scope 保留额度和显式 ready-slot
预约。已经提交但尚未释放的容量抵扣后续删除需求，不能当作已可准入容量。

| 参数 | 行为 |
| --- | --- |
| `thinking_only` | 冗余候选必须是完整证明为 thinking 的封存块；mixed/unknown 块不按冗余删除。 |
| `pool_pairwise` | 池内候选可成为直接代表，使用有界贪心比较；不保存全历史或池内的平方邻接矩阵。 |
| `all_history_references` | 允许从非 thinking 历史抽取参考块。参考池数量仍受 `reference_blocks` 限制，不表示比较全历史。 |
| `refresh_before_rank` | 初次 Query 评分完成后、代表优先级排序前，最多取得一次最新生成观察并补评分。 |
| `epoch_peak` | 同一问题内保留策略已观察到的重要性峰值，按 slot/generation/feature version/question epoch 隔离。 |
| `lru_fallback` | 冗余阶段结束并收到提交结果后，仍有容量缺口才启动真正 LRU；范围包括非 thinking 历史。 |
| `query_capture_bytes` | 单个公开 Query 捕获槽的字节上限；默认 64 MiB，允许正整数至 256 MiB。共享 staging 预算为该值的 8 倍。 |

`allow_lossy_fallback` 是旧的低重要性后备，与 `lru_fallback` 互斥。新 LRU 后备
按最近使用时间排序，不用重要性代替冷热；它也适用于 ready-slot 预约及 Query
缺失的情况。`keep_mandatory=false` 仍不允许用于 redundancy。

## thinking 来源

分类使用实际 tokenizer 的消息头和 `<think>` / `</think>` 标记，并且只接受
已经保留 KV 的 target forward 输入行。新采样但尚未消费的输出、MTP 拒绝行和
私有重放不提交分类。输入 ID 与原始逻辑位置跟随原有有界下载包传输，每行
额外占 16 bytes；不会为策略保存无限增长的 token 副本。

只有完整覆盖整个封存块的已知 thinking 内容才标记为纯 thinking。跨消息头、
跨思考结束标记、未知模板、缺失前缀或位置不一致的块不会由冗余阶段删除。
回退重新从经过会话验证的前缀建立解析状态，旧 generation 分类不适用于新槽位。
这项分类不要求启用 action attention 或图传播。

分类还要求可信的外层消息角色边界。user/tool 正文即使引用完整 assistant 消息头，
也不会被当成 thinking；没有结构化角色边界的 raw token prompt 及其生成区保守保持
unknown。Terminal Bench 桥接会提供真实渲染对应的 locality spans。运行时
`thinking_capture.layout_supported`、`owner_stats` 与 native `thinking_pure` 用于核验
分类确实生效，而非静默退化为纯 LRU。

执行用的 Query 重放范围与策略用的 Question 来源分别保存。Terminal Bench 的工具
反馈续轮可以跳过重复 Query 重放，同时在会话前缀、问题身份、位置和 token 摘要
均验证一致时复用原 Question。显式关闭 Query、缺失来源或更换问题不会自动继承
旧 Question；单个失败请求只失效一次，不会随每个 decode 包反复增加 epoch。
桥接器在确定 token 坐标后先验证 Question 来源，再对最终计划执行完整容量校验。
实时请求和历史恢复共用这一入口，避免长工具反馈在应用原问题信息之前被误当作
超长 Query 拒绝；原始请求字段不能关闭容量校验。

## 评分与代表

特征使用最多四个 target attention 层已有的 MeanK。每个层/KV head 分别计算
余弦，再跨分组平均。公开问题 Query 每 32 tokens 一段；生成观察包括最近
8 tokens 的 fast 均值以及最多四个 32-token slow 段。

重要性对每个 KV head 取其对应 GQA Query heads 的最大点积，除以 `sqrt(d)`，
再跨层/head 平均；最后跨问题段和观察段取最大。`epoch_peak` 保留已经评分到的
峰值，不声称覆盖所有已经滑出观察窗口、但后台从未评分过的 Query。

只有存在相似度至少 `representative_similarity` 的直接保留代表，才会产生正常
冗余删除意图。合格候选的排序分数为：

```text
importance_weight × softmax(importance)
  − (1 − importance_weight) × softmax(similarity)
```

低分先删除。池内代表优先选择 mandatory、已有代表、重要性更高和更近使用的块。
候选只与已经保留的直接代表建立删除关系：A 像 B、B 像 C 不足以证明 A 像 C。
本轮计划使用的代表不会进入本轮删除集合；提交再次检查候选、代表、generation、
feature version 和 question epoch。

实现保留了远端策略的直接代表思想，但没有逐行复制远端算法。这里先刷新 Query，
再排序并作贪心代表比较；远端分支是在比较结束后、最终排序前刷新。此处选择是
为了用 O(pool) 辅助空间保留直接关系，避免存储 O(pool²) 邻接边。刷新只发生一次，
不会因持续生成新 token 而反复重启本轮。

参考池沿用本地的 mandatory、已有代表、最近使用优先级；没有照搬远端的年龄分层
抽样。候选池会逐轮轮转，池内也能产生代表，但这不等同于均匀覆盖各年龄段。

## LRU 的两个后备层次

冗余意图必须先得到提交/拒绝回执，下一次策略推进再根据实际 pending credit
检查缺口。每个意图提交前还会重新检查容量需求；预约撤回或前面的提交已满足
缩小后的目标时，取消剩余意图。提交也重新合并当前 recent 范围，避免回退后沿用
旧快照的保护边界。分片扫描或评分尚未完成时，LRU 不会抢先开始。
这里的“先去重”针对本轮有界候选池；不会在容量压力下等待全历史所有 thinking
两两比较完毕。候选池逐轮轮转，因此本轮没有找到足够直接代表时，也可能进入 LRU。

1. 优先删除全 scope 中可回收、非 mandatory、没有承担旧代表关系的最冷块。
   非 thinking、mixed 和没有 Query/特征的块都可以进入这一显式有损阶段。
2. 普通 LRU 全部耗尽后仍有缺口，才按 LRU 回收**以前轮次**留下的代表。使用独立
   `LruRepresentativeFallback` 原因计数。本轮刚建立或仍计划使用的代表始终排除，
   不会在同轮先删除重复块，再立即把它的代表也删除。

冷热排序依据本轮观察快照，属于分 scope 的近似 LRU；不是跨 scope、每次访问
立即调整的全局精确 LRU。

第二层是长期容量推进所需的有损选择。如果旧代表永久不可删除，它们可以逐渐占满
repository，导致即使启用 LRU 也无法达到水位。例如 8 个槽位中有 2 个 mandatory，
其余三组重复各删一个后只得到 3 个空槽；要求 4 个空槽时，必须在后续轮次允许
淘汰一个旧代表。若只剩真正 mandatory 或资源引用尚未退出，仍需等待，策略不能
保证任何保护配置下都立即腾出指定容量。

`shadow` 不退休块，但新模式会模拟代表的建立和后备释放，以保持后续决策关系一致。
`delete` 的 `lru_representative_retired` 是 `lru_fallback_retired` 的子集，后者也计入
`lossy_retired`；这些计数不能相加当作独立删除总数。

Checkpoint 恢复先为目标窗口取得新的准入引用，再把当前窗口已有的 GPU 引用交给
恢复计划持有到 assembly 完成。当前窗口即使已经被 GC 标记删除，也可以安全完成
这次交接；这不允许已删除块重新进入目标窗口。数据与槽位仍要等全部旧引用退出后
才能释放、复用。

有会话所有者的运行时还会在首次 append 前启用当前窗口保护：当前选中的帧和已接受
的追加尾部暂时不可删除，切换到下一个窗口后退出保护。该状态独立于 GPU residency；
旧窗口仅因异步操作而留下的 GPU 引用仍可进入退休流程。新 selection 先取得临时
保护再发布窗口，新 append 在页面变为 Live 前发布保护，避免准入与后台删除竞态。
这是保存下一轮 M 检查点所需的保护，也应用于相同测试配置的 LRU 基线。

终端保存先取得受保护的 CPU 帧；遇到后台正在提交时只延迟该请求，在下一 worker
步骤重试，取得后直接复用该帧完成模型状态保存。不会因一次短暂竞争丢掉整个会话。
保护仍可能让目标水位暂时无法达到：当前 K+G 窗口、旧 P 检查点和切换期间的保护
并集必须留有容量，不能要求先回收到目标水位才允许窗口切换。

## 工作量、内存与验证

默认候选最多 128、参考最多 32。每次策略推进的 Query 预算是所有行合计 8 次
“块 × Query 段”评分，pair 预算为 512；每轮仍分片扫描全 scope 的 metadata。
纯计算推进会通知 worker 继续有限 burst，每个 burst 最多 16 次，然后按 poll 间隔
等待。没有全历史 N² 比较，但较长公开问题、较多 head 和并发 scope 都增加成本。

每 scope 的特征和 Query 限额默认分别为 128 MiB 和 8 MiB。策略 metadata 的保守
预约为 `per_slot × capacity + 512 × pools + 65536`：原模式每槽 64 bytes；启用
LRU heap 后每槽 192 bytes；thinking 分类再加每槽 512 bytes。共享 Query staging
预留 `8 × query_capture_bytes`，默认 512 MiB；thinking 输入传输再预留 32 MiB。它们从 CPU 历史预算中先扣除，
不是声称运行时 RSS 恰好等于这些数字。仍被快照持有的数据继续计费。

启用 `thinking_only` 时，启动会检查 `max_num_batched_tokens × 采样层数 × Q heads
× head_dim × 2` 是否能放入单槽，不足则明确拒绝启动。公开 atomic Query 可以绕过
普通 prefill chunk 限制，因此只缩小 `prefill_chunk_tokens` 不能替代这项容量检查。
例如 4 层、24 个 Query heads、head_dim 256 的 BF16 捕获，在批量上限 4136 时
需要 203,292,672 bytes（193.875 MiB）；可配置 `query_capture_bytes=268435456`，
对应共享 Query staging 预留 2 GiB。未启用 thinking 的原 redundancy 保留原有行为：
运行时超槽上限会使该问题观测无效，并增加 `overflow_packets`，不会截断后冒充完整 Query。

原生策略回归覆盖直接代表非传递性、两级 LRU、回执后重新检查压力、缺 Query、
字节/槽位预约、shadow、一次刷新和问题 epoch 峰值。组件通过只能证明这些约束，
不能替代真实模型任务质量及吞吐测试；真实模型结果应记录使用的冻结源码与完整
profile，不沿用旧 LRU 的性能数字。

永久删除会失去后续可能需要的信息。MeanK 相似不保证事实可互换，LRU 后备更不
提供语义等价承诺；旧代表的最终淘汰也可能移除其过去代表的全部信息。质量变化是
这一有损容量策略的实际代价，需要与固定任务集上的释放量和延迟一起评估。
