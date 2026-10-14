# 上下文管理主路径与 qw3 的一致性核对

**后续实现更新：** 下文冻结 `3bc639b` 的审计结论。显式 token-ID Query、
请求策略、partial-tail/full-Query replay、FP16 index 和原版基础 CUDA scorer
已补齐有界适配，并新增真实模型跨轮及并发验收。
当前状态请以[新报告](explicit-query-20261014.md)和
[实现清单](implementation-status.md)为准；增量 GPU index、mandatory spans、
完整 qw3/DSH 协议及附加策略仍有缺口。

核对日期：2026-10-14。vLLM 集成基线为 `3bc639b`，参考当前
`/home/chaidi/qw3` 工作树。重新检查的 10 个 vendored C++ 核心文件均与该工作树
逐字节相同，且与上次审计的文件哈希相同。该结论不覆盖未迁移的 executor、
请求规划及 CUDA scorer。

本次以**单轮输出不超过 G**的设计为准。保留该限制，不将多 epoch 生成作为
本次需要补齐的功能；没有修改运行时、输出限制或服务配置。

结论：GPU active view、CPU/NVMe historical view 的分离已经实现；但决定何时
重选、用哪些 Query、如何保留消息，以及如何重放 Query 的完整主路径还不与
qw3 等价。原版 Store 相同不能替代请求规划的一致性。

## 已接通的核心契约

- historical authority 保留已接受的 raw K 和 V，生成的 rejected draft 不作为
  历史发布。CPU cache 有预算；被淘汰的封页通过 NVMe arena 读取，不能将
  CPU cache budget 理解为整个 historical view 永远常驻 RAM。
- GPU full-attention working window 采用 K + G 的有界容量；原版 Store 的
  sink/recent、TopK、历史排序和 remap 计划已接入。
- dense 窗口的移动 K 从 immutable raw K 重建并重新应用 RoPE；同位置 K
  可以跳过，resident V 可以复用。逻辑历史位置与 GPU dense attention 位置
  分离。原生物理页、lookahead 和 staging 仍有额外显存开销。
- target 和原生 MTP 共用历史选页计划、同步重建窗口；MeanK 只对 target
  的 full-attention 层评分。GDN 保留完整逻辑状态，provisional 执行后回滚
  同一 batch 中所有受影响请求，再提交 durable replay。
- token-ID 弹性会话入口能保留历史、释放终态 GPU 请求页，并在 exact-prefix
  匹配时恢复 active KV/GDN checkpoint。已有 64K 两回合真实恢复验证。

## 主路径中仍不一致的部分

| 项目 | qw3 | 当前 vLLM 集成 | 实际影响 |
| --- | --- | --- | --- |
| Query 来源与范围 | 显式 `kvmem_query_span` / `kvmem_query_message_range`，调用方指定用户 Query；可跨物理 chunk 捕获完整跨度。 | `bounded_runtime.py` 从当前原生调度 chunk 获取 Q；弹性网关没有角色、消息范围和 Query 字段。投机 decode 的 Query 排除未验证 draft。 | 工具输出、历史或其他输入 chunk 可能参与检索，结果取决于 chunk 划分。 |
| 重选时机与策略 | 支持 `auto/force/off`、`pressure/keep_selected` 和单独的 `semantic_chunk`，还有跨轮刷新 gate。 | `QueryPolicy.refresh` 在长 prefill 的 block 边界触发；导入长 checkpoint 后也触发；没有公开原版策略。 | 不能要求只在明确用户 Query 或预算压力时刷新，也不能按原版保持上次选页。 |
| MeanK 数值路径 | 生产 Query/index 为 FP16，归约为 FP32；专用 CUDA scorer 和增量 index。 | CPU FP32 mean index，PyTorch FP32 einsum/softmax；基础页间 softmax、层/头平均、Query 求和定义相同。 | 不能保证相同输入张量的 bitwise score 或近似平分时的 TopK 一致；刷新准备和评分效率也不同。 |
| Query checkpoint 与重放 | 对显式 Query 使用其开始前的 block-aligned checkpoint，必要时重放 partial tail + 完整 Query。 | 当前事务回滚并重放被调度的 chunk；终态会话 checkpoint/partial tail 是另一套恢复边界。 | 已有终态恢复不等于已实现任意 Query 起点的原版重放契约。 |
| prefill budget 与历史构建 | prefill budget 可独立于 semantic select budget；`start/append` 可用 `max_tokens=0` 连续 ingest，`finish` 再生成。 | `KVMemConfig.native()` 固定 `prefill_budget = select_budget`；公开 turn 要求正数输出。 | 无法直接采用原版分阶段历史构建和相同压力阈值。 |
| 必留语义与内容组织 | 请求规划可映射工具、文件证据、消息等 mandatory spans，并处理组合预算；可做 round/message expansion。 | 低层有 `mandatory_handles`，在线选择调用未传入消息或工具对应 handles；主要保留 sink/recent。 | 不能保证完整工具片段、某条消息或文件证据一定保留。 |

例如，在 qw3 **显式 Query 模式**中，一条跨两个物理 chunk 的用户问题可以
作为完整 Query 评分并重放；当前集成会分别用两个调度 chunk 触发选择。这会
改变第一段提交时的 attention window，也可能改变第二段捕获的 Q，不能用
“两边都是 MeanK”推断它们等价。qw3 自己的 opt-in semantic-chunk 路径也会
按 chunk 选择，但它不能代表原版所有请求策略。

## 历史与会话的边界

- 当前 elastic 协议是版本化 append-only token ID。没有接通原版任意历史
  编辑、分支、命名 snapshot、offline archive build/attach、readonly base +
  writable overlay 或完整 GC policy 的公开协议。NVMe arena 不代表进程重启
  后可以恢复原版 session/archive 格式。
- active KV 有界不等于逻辑历史无界：当前仍受原生 `max_model_len` 和会话配额
  限制，默认逻辑上限为 262,144。
- 当前 DSH 可选入口直接调用 18381 的原生聊天 API，没有发送 KVMem
  `session_id` / `X-Session-ID`，也没有经过 18382 的 token-ID 弹性网关。
  DSH 两轮工具测试只验证聊天/工具链；它发送完整历史，不能据此声称已验证
  跨轮 CPU KV checkpoint 复用或 qw3 请求规划。

## 尚未移植的附加策略与效率优化

Adaptive/SubBlockMeanK/PerToken、Query history/decay、action attention、
CacheBlend、多模态索引和 MRoPE 等没有完整接通；它们不应与基础 MeanK 主路径
混为一谈。增量 GPU index、原版专用 scorer、Q-only early exit、provisional
前缀精确重用和提前 Query prefetch 也仍未移植。

此前报告中的两个执行缺口已补齐：原生编译兼容和 CPU 历史批处理。
它们改善执行效率，但没有修复上表的 Query、策略、精度或消息保留差异。

若以基础 MeanK 上下文语义等价为目标，优先级应为：显式 Query/请求规划与
重选策略 → Query checkpoint/partial-tail replay → 原版索引精度和评分路径 →
独立 prefill budget、prefill-only ingest 与 mandatory spans。后续应固定同一
历史、Query 和配置，比较每次选择的页 ID、mandatory 覆盖、重放范围、历史
长度及 MTP/GDN 状态，不能仅用最终吞吐证明一致。

本次是代码与已留存证据的核对，没有新增 GPU 压测。参见
[完整功能审计](qw3-parity-audit-20261014.md)、
[最新编译和批处理验证](compiled-capture-history-20261014.md)。DSH 接入记录保存在
本地 `deploy/vllm-kvmem-dsh-20261014/README.md`，位于本仓库的父目录。
