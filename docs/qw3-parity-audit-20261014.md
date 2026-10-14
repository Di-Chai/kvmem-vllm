# qw3 / KVMem 与 vLLM 迁移核对，2026-10-14

后续更新：本报告冻结的 compilation NONE 限制已在单 GPU、同步 Qwen MTP 路径解除；
CPU 历史维护也已批处理。详见[实现与测量](compiled-capture-history-20261014.md)。
其余未迁移项仍按下表跟踪，不能把此更新当作整个 qw3 已完整迁移。

本次核对当前 `/home/chaidi/qw3` 工作树和 kvmem-vllm `e9e8d6d` 的实现、
公开配置及实际执行路径。qw3 HEAD 为 `1410cfe284726edb4161381546fe3915110331c9`，
其工作树有未提交修改，所以 HEAD 不能单独标识本次参考内容。随报告保存相关文件
SHA256；本仓库引用的 10 个 qw3 C++ 核心文件与当前工作树逐字节相同。

**基础 MeanK 数学定义、原版 Store 选页、immutable KV、dense re-RoPE 和
target/MTP 同步窗口已经接入；完整 qw3 功能和所有执行细节没有全部迁移。**
尤其不能把“原版 selector 源码未修改”解释为 Query 定义、索引精度、刷新策略、
可选检索器及 HTTP 协议全部一致。当前服务对应的是有界 MeanK 主路径和一部分
semantic-chunk 生命周期适配，不是 qw3 全部模式的替代品。

`KvMemStoreConfig` 当前可提取 56 个字段，pybind 暴露其中 10 个；未暴露字段
也包含 executor 内部预算/记账，不能据此计算一个“迁移完成百分比”。

表中“已接入”表示本次在执行路径中确认；“不同”表示适配方式或数值精度不同；
“未接入”包括原代码虽已随核心导入、却没有公开配置或运行时调用的功能。
实验功能与原版限定串行的功能单列，不把它们当作当前八路 decode 缺失的必要步骤。

## 检索与上下文语义

| qw3 功能 | 当前状态 | 当前行为、差异与证据 |
| --- | --- | --- |
| 标准 attention 的全局页选择；GDN 保留完整逻辑状态 | 已接入 | `cache_groups.py`、`native_state.py`、`window_batch.py`；所有 target full-attention 层共享页计划。 |
| 基础 MeanK：content-frame Q/K、页间 softmax、平均层/头、累加 Query | 已接入，执行不同 | `retrieval.py`、`authority.select_for_queries`；原 CUDA reference 对照已有。当前是 PyTorch FP32 评分，原版生产索引/Query 为 FP16、归约 FP32。数学定义一致不保证 bitwise score 或近似平分时的排名一致。 |
| 原版 sink/recent 分配、TopK、历史顺序、remap diff | 已接入 | `config.py`、`memory.py` 调用未经修改的 `KvMemStore` / `resolve_kvmem_keep_allocation`。自动保留带随 K/B 解析；不表示所有调用采用 qw3 同一套显式参数。 |
| 显式用户 Query span / message range、role-aware request plan | 未接入 | qw3 `kvmem_request_plan.hpp`、executor `kvmem_set_query_span`；当前 `bounded_runtime.py` 取原生调度 chunk 的 Q，网关只接收 token ID 增量，没有 Query 范围或消息角色字段。 |
| Query 全跨度捕获及 block-aligned checkpoint + partial-tail replay | 部分 | 当前完整重放一次被调度的 chunk 并回滚 batch 邻居；不能指定一个跨多个 chunk 的显式 Query，并在其开始前恢复原版请求边界。 |
| 默认请求/Query 边界 reselection、force/off、pressure/keep-selected 策略 | 不同 | 当前 `admission.py` / `speculative_admission.py` 固定 K/G 容量策略，长 prefill 使用 chunk Query；qw3 `kvmem_incremental_session_api.md` 描述的调用方策略尚未暴露。 |
| max_tokens=0 prefill-only；增量 start/append/finish ingest | 未接入公开协议 | 当前 turn 必须请求正数输出；不能直接采用原版的无采样历史构建协议。 |
| semantic-chunk：provisional Q → select → rollback → durable replay | 已接入子集 | `bounded_runtime.py` 保留事务顺序；chunk 大小受 vLLM scheduling 决定，缺少原版完整请求规划和独立 prefill budget。 |
| prefill budget 与 semantic select budget 分开设置 | 未接入 | `KVMemConfig.native()` 固定 `prefill_budget = select_budget`。 |
| mandatory spans / 文件证据 / 工具与消息整体保留 | 部分，在线未接入 | `select_for_queries(..., mandatory_handles=())` 有低层接口；在线调用没有将文件/消息/工具 span 映射为 handles。sink/recent 和原版核心的 mandatory 参数不是完整请求规划。 |
| Retrieval / H2O / Recency 及 TopK / Quota 模式切换 | 未接入可配置模式 | 原版枚举仍在 vendored core；`KVMemConfig` / pybind `StoreConfig` 不暴露这些选择，当前采用基础 Retrieval/TopK。 |
| PerToken / SubBlockMeanK、sum/max、固定四原型/Adaptive 1/2/4 原型 | 未接入 | 当前仅每个物理块一个均值；原版 scorer/index executor 没有移植。 |
| Round/message semantic expansion、group MaxSim / length-normalized mass | 未接入 | 需要原版细粒度索引和语义 span；当前输入协议没有相应 metadata。 |
| Query relevance history、decay、branch preview/commit | 未接入，可选 | qw3 `kvmem_query_history.hpp` / executor；原版默认 off，并拒绝若干 continuous batching 组合。 |
| 输出 action attention、prefill attention、quota/score fusion、pressure policy | 未接入，可选实验 | qw3 `kvmem_action_attention.*` / `kvmem_attention.hpp`；原版默认 off，当前原版文档也限制串行。需要新的 batch/accepted-prefix 验证，不能直接开启后宣称多用户等价。 |
| Guided/private Query、mid-decode 多 epoch、cross-turn refresh gate、thinking compaction | 未接入 | 当前 output 上限不超过 G；没有原版自然边界探测、私有 Query 或多 epoch 请求策略。 |
| CacheBlend 选择 KV 的部分重算 | 未接入，可选实验 | qw3 `kvmem_blend.cpp` / executor blend 生命周期；当前始终使用 immutable 历史 KV 装配。 |
| DeltaNet edit retrieval | 未接入，原版归档实验 | 原版标明 experimental/not recommended，区别于保留原生 GDN 状态；不是本阶段迁移目标。 |
| 多模态内容索引、视觉身份及 MRoPE remap | 未接入 | 当前发布入口和测试是文本 token ID，指定 language-model-only；不能据此承诺图像/视频历史。 |

## 索引、Query 和装配效率

| qw3 优化 | 当前状态 | 对当前执行路径的影响 |
| --- | --- | --- |
| 增量 GPU MeanK index、封块 finalize、FP16 常驻或 CPU 分层 placement | 未接入 | 当前 `authority.append` 在 CPU 保存 raw/V 并 FP32 sum；`index_snapshot` 选择时排序、stack 全索引并 H2D。仍有逐请求 CPU copy/索引维护，属于可继续减少的 KVMem decode 工作。 |
| 精确 one-dot / two-pass scorer、bounded logits slab、Adaptive packing | 未接入 | 当前参考 scorer `query_chunk=16`，使用 PyTorch einsum/softmax；未迁移 qw3 专用 scorer kernels。主要影响刷新和 TTFT。 |
| 在线逐层消费 provisional Q | 部分方式不同 | 当前 Q 已在 GPU，避免 qw3 旧 Q 的 GPU→CPU→GPU 往返；但在整个 provisional forward 结束后 stack 全部层并评分，没有在线逐层 scorer。 |
| provisional 最末 attention 层 Q-only / early exit | 未接入 | 当前运行完整原生 provisional forward，之后恢复 GDN；不能跳过原版无关的 K/V、attention、FFN 和尾部层。主要影响长 prefill。 |
| provisional 前三层 hidden/SSM 精确重放复用 | 未接入 | qw3 可复用第一层 full-attention 之前的原生前缀；当前 durable replay 从完整模型入口重跑。 |
| resident K 同位置 skip；移动 K 从 immutable raw 重建 | 已接入 | `window_assembly.py` 使用原版 plan；FP8 移动块从 raw 重建，避免重复量化 drift。 |
| GPU raw-K working-set cache | 已接入，配额不同 | `gpu_raw_cache.py`；共享有界 slab，namespace+generation、每请求公平上限、fence；不是 qw3 单 executor 的完整 K working-set 分配方式。 |
| resident V 复用；K/V 分开 stage-in | 已接入，物理布局不同 | 当前 dense native 窗口必须先 gather 所有将复用的 V，以免后续写覆盖源；qw3 可以保留逻辑页原位置。额外共享 V slab 约 0.53125 GiB。 |
| no-copy logical page window、resident KV 只换 page list | 不同 | 当前 vLLM dense KV 页表需要搬动部分仍 resident 的值；没有原版全部 no-copy 物理窗口语义。 |
| batched remap/RoPE/FP8 materialization | 已接入 | `split_assembly.py` / `assembly.py` / `packed_assembly.py`；17 层、原生 1600-token 页真实几何已对照旧路径字节。 |
| pinned gather、固定 CPU worker、双 GPU/host staging、独立 DMA/consumer fence | 已接入 | `cached_storage.py`、`transfer.py`、C++ GIL-free gather；异步资源有边界。 |
| prepare/finish reselect 分离，提前预取并与 MTP prefix 工作重叠 | 未完整接入 | 当前选页后 assembly 内部流水化，但事务同步等待完成；没有 qw3 `kvmem_start_query_prefetch` 的提前 Query/replay prefetch 或与完整 MTP prefix 的重叠规划。 |
| 新封块增量 writeback、resident count、seal/hash 异步化 | 已接入 | `_unsubmitted`、`_resident_count`、`write_from_views`；不再每次 append 扫描全部历史或在前台打包整个封块。 |
| accepted CPU scatter 与推理重叠 | 已接入子集 | 大包 >=128 token 使用一个后台 worker；小 decode 包在当前 GPU 工作入队后前台处理。全包后台曾退化，保留下一 scheduler/EOS barrier，不能移除正确性 fence。 |
| 跨请求 accepted-row 批量 CPU scatter/索引更新 | 未接入 | 当前 drain 仍逐 request 调用 Torch copy、FP32 sum 和原生 append。完整 packet GPU 打包/D2H 已合并，但 CPU authority 更新未合并。 |
| 原版 NUMA placement、lazy host slab/recycling、malloc trim | 部分 | 当前长期复用 pinned staging 和 cache budget；没有 qw3 全套 NUMA 策略与 host allocator 控制。 |

## 投机、模型状态与多用户生命周期

| qw3 功能 | 当前状态 | 边界 |
| --- | --- | --- |
| 原生 Qwen MTP 与 target 同选页、local dense positions、raw KV 归档 | 已接入 | `mtp_runtime.py`；评分仅 target16层，target16+MTP1 一同装配；k=2/3 已资格测试。 |
| MTP prefix-prime KV-only shortcut | 未接入，契约不同 | qw3 executor 可丢弃 prefix attention/FFN 输出；vLLM `_prefill` 的最后 hidden 驱动第一个 draft，不能删掉整个原生 forward。若继续做等价裁剪，需要独立数值/accepted-prefix 验证。 |
| 全 batch GDN provisional checkpoint / rollback、speculative accepted state | 已接入 | `native_state.py`、`bounded_runtime.py`；包含未选页邻居，保持逻辑 history 与 dense attention positions 分离。 |
| DFlash2 独立 sliding context 与 bounded target | 已接入 | BF16 draft 与可选 FP8 linear quant；draft KV 仍为独立 BF16 sliding KV，不能与原生 MTP 的 full-attention 层混同。 |
| 回合结束 checkpoint、partial tail、pending last output、恢复 active KV | 已接入 | `session_worker.py` / `prompt_checkpoints.py`；真实 64K 两回合 KV/GDN 字节 oracle 已通过。 |
| arbitrary mid-decode suspend/resume、周期 checkpoint ladder、长请求多 epoch | 未完整接入 | 有低层 decode capture 设置；发布入口主要资格覆盖 terminal checkpoint，不提供任意抢占点 continuation 或原版完整 ladder 语义。 |
| 进程内 idle 会话/用户历史；共享页释放 | 已接入，策略不同 | 当前终态 GDN 也移 CPU，并归还 native 请求页；qw3 pressure suspension 可保留私有 GPU state。两者均不能据 NVMe 目录宣称会话重启持久化。 |
| warm idle / 短工具等待 TTL 保持 GPU 页 | 未接入策略 | 当前回合结束即回收，waiting_for_tool 是会话状态；session TTL 控制过期历史，不是延迟回收 GPU 的 tool TTL。 |
| 编辑历史/多分支 rewind/命名 cache snapshot 公共协议 | 部分低层，未接入 API | 有 generation-safe rewind/checkpoint；网关公开的是准确版本的 append-only token 增量，没有任意编辑分支协议。 |
| offline archive build/attach、readonly base + writable overlay、持久 checkpoint ladder | 未接入 | 当前 NvmeTier 是临时 authority arena；不能加载 qw3 原 archive/session 格式。 |
| bounded repository generations / leases / ready reserve / original background GC | 已接入低层 | 10 个核心文件相同；`repository_gc.py` 低层桥接有测试。elastic 当前默认 GC 关闭且没有完整 policy 配置，保留预算内完整历史。 |
| 原版完整 GC modes / rule/file-history policy 控制 | 未接入公开配置 | 默认 core、bool repository_gc 及派生 reserve，不等同于全部 qw3 GC CLI / 模式。 |
| 鉴权、namespace、每用户配额、版本、admission/restore bytes、队列 deadline | 已接入，本集成新增 | `elastic.py` / `elastic_gateway.py`；原 qw3 的 prefix key 本身不提供 tenant ACL。 |
| 标准 chat/message 协议、prepared message/token/vision prefix cache、公开 SSE | 未接入此弹性协议 | vLLM 原生端仍有标准 API；本次有状态网关使用 token ID 增量并在结束返回 JSON，内部 SSE TTFT 不是调用方收到首 token 的延迟。 |
| 持续工具负载的 bandwidth arbitration、per-token SLO 调度 | 尚未实现/验收 | 当前 EDF 是可进入用户间的整回合准入；decode 使用原生 scheduling，没有 PCIe/NVMe 速率分配或硬 SLO 控制器。 |
| 多 GPU / 多进程共享会话、多模型迁移 | 未接入 | 当前 runner 明确 single rank；会话模型身份和 worker owner 不可跨加载恢复。 |
| 无界逻辑历史 / 原 archive 超长历史接入 | 未接入 | 当前逻辑请求仍受 max_model_len 和 token history 配额约束；有限 active KV 不会自动取消这个原生请求长度限制。 |
| 原生异步 scheduling / torch compilation 与完整 graph 配置 | 不同，当前限制 | MeanK 契约要求一个同步在途 batch、compilation NONE；普通 decode 保留 FULL_DECODE_ONLY，语义 refresh eager。旧完整上下文 benchmark 是 async + FULL_AND_PIECEWISE。 |

## 本次核对产生的优先级

1. **语义一致性**：加入显式 Query token 范围和请求策略，区分 pressure、
   keep-selected、semantic-chunk，复用原版 Query request plan；验证跨 chunk Query、
   partial-tail replay、mandatory spans 和工具输入。明确 FP16 index/Query 与当前
   FP32 oracle 的精度模式，做 score/ranking 数值对照和真实任务质量测试。
2. **原生编译兼容**：当前入口强制 compilation NONE。原生控制保持 CUTLASS
   时启用默认 compilation/graph 配置，两轮由约546升至633 token/s；恢复
   FlashInfer 后约683。应先验证 capture/Query/replay/MTP 在原生编译下的
   所有权和 side effects，保留原生验证算法；不能只移除 guard 即用于服务。
3. **KVMem decode 成本**：将 accepted-row CPU copy/索引更新批量化，评估原版
   GPU sealed-block MeanK 更新。在真实 matched native profile 确认增量后再选择
   实现；GPU MeanK 更新本身不能消除全部 authority metadata/GIL 工作。
4. **长 prefill / resume**：原版专用 scorer、逐层 Q 消费、provisional early exit、
   第一层 full-attention 之前的精确 prefix replay 复用、提前 prefetch。
   这些改动主要提高 TTFT/整轮吞吐，不能当作当前没有 refresh 的稳态 decode 收益。
5. **运行时重叠**：双在途 batch / 异步 scheduling 需要把 capture、rejection、
   GPU窗口、GDN rollback 及 terminal fences 都改为 batch-owned；当前 guard 不可
   直接删除。恢复 native compilation 也需验证捕获 buffer 和模型数值行为。
6. **产品补齐**：short-tool TTL、持续带宽/SLO 测试、公开历史分支和 archive 协议。
   action/query-history/Adaptive/CacheBlend 应按原版兼容性条件独立移植，不能一次打开。

本轮仅增加诊断采样支持和核对报告，未改变原生主模型验证、MeanK 选页或
线上调度契约。decode 的 matched 控制、kernel/CPU 分摊和 benchmark 配置差异见
[本次性能归因](decode-profiling-20261014.md)。
