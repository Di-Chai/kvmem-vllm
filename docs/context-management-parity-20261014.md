# 当前上下文管理与 qw3 的一致性核对

核对日期：2026-10-14。运行时代码基线为 `5043645`。参考 qw3 HEAD 为
`1410cfe284726edb4161381546fe3915110331c9`，但其工作树有未提交修改；
本次以实际文件内容为准，相关 SHA256 和 CPU 策略对照保存在
[核对数据](benchmarks/context-parity-5043645.json)。本报告替代此前针对
`3bc639b` 的上下文管理核对；历史完整审计仍保留在
[qw3 功能审计](qw3-parity-audit-20261014.md)。

**结论：分离式 KV 和有界显式 Query / 基础 MeanK 主路径已实现，完整的
请求级上下文管理还没有与 qw3 等价。** 原版 Store 和 scorer 的相同源码，
只能证明对应组件相同，不能证明 Query、重选时机、消息保留和恢复策略相同。
本轮以用户确认的单轮输出不超过 G 为边界，不将多 epoch 输出列为待补齐要求。

本次只做代码核对、原版 helper 编译和 CPU 对照，没有更改服务或新增 GPU 压测。
此前最终安装包三组 C8、48 回合和 24 次恢复的成功记录仍有效，但它们证明的是
已实现路径可以运行，不是与 qw3 的完整逐页等价验收。

## 已接入或确认相同的部分

- 10 个 vendored Store / repository / GC / NVMe C++ 核心文件逐字节相同。
  `block_attn_score_softmax_pages_kernel` 的基础 kernel body 也与当前 qw3 源码
  相同。此范围不包括两边各自的请求处理器和模型执行器。
- GPU 有界 working window 与 CPU authority / NVMe historical storage 分离；
  使用原版 sink/recent、TopK、历史顺序和 generation lease。
- 明确的 token-ID Query 只对指定内容捕获 Q，完整 partial tail + Query 原子
  调度；provisional → select → 全 batch GDN rollback → durable replay。
- 新配置采用 FP16 Q / MeanK index 和 FP32 归约；封页均值按原版 token 顺序累加。
  旧配置仍可以使用 FP32 PyTorch scorer，不能将所有入口概括为原版数值路径。
- 移动 K 从 immutable raw K 重建并 dense re-RoPE；target 和原生 MTP 共用
  选页及装配，MeanK 只对 target full-attention 层评分。
- token-ID 网关维护完整逻辑历史，鉴权和 namespace 隔离；释放回合 GPU 页，
  exact-prefix 命中后恢复 CPU 检查点并重放后缀。accepted 与 rejected draft
  历史分开，恢复和资源回收已通过真实模型并发验收。

## 影响上下文行为的剩余差异

| 项目 | qw3 实际实现 | 当前 vLLM 实际实现与影响 |
| --- | --- | --- |
| 默认 Query | 开启 query-conditioned 时，普通 chat 可从最后 user 消息自动映射 Query。 | 显式规划只接 token IDs。没有 Query 元数据的 token-ID 请求默认做容量压力刷新；chat/text 入口仍走 legacy semantic-chunk，工具/历史 chunk 可能成为评分 Query。 |
| Query 字段协议 | `kvmem_query_span` 为 `{message_index, content_start, content_end}`，偏移是消息内容的 UTF-8 字节；服务经聊天模板/tokenizer 映射。 | 同名字段为完整逻辑 prompt 的绝对 token 范围 `[begin,end]`，不是兼容原 API 的实现。客户端不能直接复制 qw3 请求。 |
| Query message range | `{message_begin,message_end}` 可选保留角色的 user/assistant/tool 整轮，使用重渲染 LCP/LCS 映射。 | 调用方提供 token message spans；范围只能含 user 消息，混合角色整轮被拒绝。完整 Query + partial tail 必须不超过 G−B，尚无跨 batch / host streaming 的长 Query 捕获。 |
| `auto` 跨轮 gate | 区分新 user Query、同一任务工具回调、证据变化、增长阈值、初始 headroom；稳定回调保持 epoch，特定配置生成 private retrieval query。 | `auto/force/off` 是简化计划及组合检查；没有对应的任务/Query 身份、刷新水位和 evidence 状态机，也没有 private-query 生成。提供 Query 就按该边界走，未提供则压力刷新。 |
| 恢复是否保留原选页 | keep-selected continuation 保存已有 A+B epoch，checkpoint 本身不能成为压力重选触发器。 | 默认 `pressure` 恢复在旧窗口大于 K 时先缩回 K，并按历史位置做压力选页。即使还有 G headroom，也可能改变此前 Query 选页。仅 `keep_selected` 分支跳过这一步。 |
| 长历史 `keep_selected` | 在实际 checkpoint / active window 上检查增量后缀及剩余额度；必要时 pressure-ingest。 | HTTP 准入以完整逻辑 prompt + 输出 ≤ K+G−B 为条件，长历史即使 active view 有空间也会被拒绝，无法普遍承接稳定工具回调。 |
| mandatory / 内容生命周期 | 请求可映射 system、root task、project policy、current Query、工具/文件证据等 spans；超额时按优先级保留结构边缘、尾部并留检索额度，原历史仍可检索。 | 低层支持 mandatory handles，在线 `select_for_queries` 调用没有传入语义 spans，主要依靠 sink/recent；不保证某条消息、工具结构或证据片段必留。 |
| 重放起点 | 可分别指定评分 Query 和 live replay suffix；消息 header 在更早 block 时，从更早 replay origin 捕获/重放。 | 只有 Query 起点向下封页对齐，尚无独立的 live replay span。已实现的 partial-tail replay 不覆盖全部原版消息/工具边界规则。 |
| P/M、旧 Query 与模板改写 | 可选 P/M checkpoint、保留 Query snapshot 和 replay checkpoints，以 token LCP、选择 generation、branch-safe tail 等决定恢复位置。 | 新配置每会话保留一个最近封页 checkpoint；旧 Query 在其前面时冷恢复，没有同等级的旧 Query snapshot / checkpoint ladder / 模板改写分支恢复。 |
| 历史 KV 尾部 | 原版支持终态 M 与 P 等多种保留/恢复路径，不需要一律回退到单个封页点。 | 新显式规划启动要求 sealed decode checkpoint，拒绝 terminal capture；回合结束 authority 会 rewind 到封页 checkpoint。之后的逻辑 tokens 仍保留，但相应 KV 下一轮重算，短轮可能重放上一完整 Query。旧模式已有 terminal capture，尚未组合成新模式的 P/M 策略。 |
| prefill budget / ingest | prefill budget 可独立于 semantic budget，公开支持 `max_tokens=0` 及 start/append/finish 历史构建。 | `prefill_budget = select_budget`，公开回合要求正输出；没有对应连续 ingest 协议。压力阈值还需预留原生 MTP allocation headroom。 |

参照代码：qw3 的 `src/qw3_server.cpp`、`src/kvmem_refresh_policy.hpp`、
`src/kvmem_request_plan.hpp`、`src/kvmem_prefix_reuse_policy.hpp` 和
`src/qwen_native_backend.cpp`；本集成的 [Query 计划](../src/vllm_kvmem/request_plan.py)、
[HTTP 规划入口](../src/vllm_kvmem/http_sessions.py)、
[会话恢复 gate](../src/vllm_kvmem/session_control.py)、
[检查点生命周期](../src/vllm_kvmem/prompt_checkpoints.py)、
[在线选择调用](../src/vllm_kvmem/bounded_runtime.py)。
证据文件记录对应来源哈希，避免将参考工作树变化与本次结论混淆。

## 已复现的具体差异

使用实际 qw3 header 编译 helper，再调用当前安装版本的 RequestPlan / SessionLedger：

1. K=32,768、G=16,384、已有 active window=34,816；同一任务增长 2,048 token，
   guided 刷新阈值 4,096。qw3 决定不重选，保留 headroom，禁止 checkpoint
   压力选页；当前默认恢复将窗口变为 32,768，`pressure_restore=true`。
   这改变 active view，并非仅多了一次搬运。
2. 同一长历史的 `keep_selected` 请求，即使实际 active + 小增量可以容纳，
   当前 HTTP 检查仍按完整 52,096 token prompt 拒绝。不能仅切换请求字段就
   声称已有完整 qw3 工具续接行为。
3. qw3 原格式 Query span 被当前 token-range parser 拒绝；包含 user、assistant、
   tool 的 Query message range 也被当前实现拒绝。两边字段名字相同不代表协议相同。
4. Query 从 token 49,152 开始，live header 从 49,149 开始。原版 replay helper
   选择 block 起点 49,024；当前只有 Query 坐标时选择 49,152。这里只比较实际
   helper 的边界决策，不声称已做两模型输出质量对照。

现有真实模型短轮记录还说明尾部重算规模：首轮 prompt=50,006、输出=17，
逻辑历史为 50,023；下一轮实际恢复 checkpoint=48,000，需先重放旧后缀
**2,023 token**，再处理新增内容。它不是只回放不足一页的尾部，也不是只
加载 32K active context。完整 tokens 没有丢失；CPU 可直接恢复的 KV prefix
与逻辑历史结束位置不同。

## 尚未接入的效率优化与扩展

以下缺口与请求语义差异分开看：增量 GPU MeanK index、one-dot/two-pass、
在线逐层评分、Q-only early exit、provisional 前缀精确复用、提前 Query
prefetch 等主要影响成本；不能以它们代替上面的语义迁移。

此外仍缺原版完整 no-copy logical page window、Adaptive/SubBlockMeanK/PerToken、
query history/decay、action attention、CacheBlend、多模态/MRoPE、任意历史编辑与
分支、offline archive/session 格式和完整 GC policy。工具等待当前立即归还 GPU
页，没有 warm-idle GPU TTL；历史 TTL 与 GPU residency TTL 的含义不同。
逻辑历史仍受 262,144 token 及配额限制，100 GiB 是 read cache 预算。

DSH 当前原生聊天连接没有该 token-ID 规划与弹性 session 协议。多 GPU、异步
多在途 batch、长期工具负载的带宽分配和逐 token SLO 仍未验收或实现。

## 语义补齐优先级与验收

1. 聊天/DSH 的准确 Query 与消息坐标、默认最后 user Query、兼容原字段协议。
2. 原版跨轮 gate 与 checkpoint finalize：稳定工具回调保持选页、实际 active
   headroom 准入、增长/evidence 触发；需要时再接 private-query 策略。
3. mandatory spans 与 capacity fit、独立 live replay span；补齐 P/M、Query
   snapshot 和 branch-safe 恢复，减少终态后缀重算。
4. 独立 prefill budget 与 prefill-only ingest；随后再迁移索引/评分效率优化。

验收需固定同一 token history、Query、K/G、block size、keep bands 和配置，
逐事件对比触发原因、候选/必留页、最终页 ID 顺序、replay 范围及保留历史位置。
已有 scorer 测试是与 PyTorch 定义的数值容差对照，已有恢复 oracle 是本实现
CPU checkpoint 的 GDN 字节对照；它们尚未构成两套模型执行器的完整逐页、
逐轮质量等价验证。CUDA kernel 原码相同也不能消除 Q/K 来源和近似同分排序差异。
