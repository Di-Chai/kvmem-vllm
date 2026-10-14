# KVMem + vLLM 实现状态

更新日期：2026-10-14。当前为显式 Query / 原版基础 FP16 MeanK 版本；此前编译与
CPU 批处理基线为 `3bc639b`。新增路径的契约和资格范围见
[Query 与真实模型验收报告](explicit-query-20261014.md)。
开发分支：`main`。

当前已实现分离式 KV 管理和有界 MeanK 主路径，但**尚未完整复现 qw3 的请求级
上下文管理语义**。单轮输出不超过 generation budget G 是保留的设计；多 epoch
生成不列为本阶段的缺失要求。此清单中的“已实现”只覆盖注明的配置及入口。

## 已实现并有验证

- [x] **分离 active / historical KV**：GPU 保存有界 working window；CPU authority
  管理已接受的历史 raw K/V，封页可进入 NVMe arena。
- [x] **K/G 预算**：选取历史最多 K，预留 G 用于生成；当前公开 MeanK 弹性入口
  限制单轮输出不超过 G。原生物理页、MTP lookahead 和 staging 还有额外开销。
- [x] **原版 Store 选页主路径**：sink/recent、TopK、历史排序、remap diff、
  generation/lease 管理；10 个 vendored 核心文件与参考 qw3 工作树一致。
- [x] **基础 MeanK 数学定义及原版 CUDA scorer**：content-frame Q/K、页间
  softmax、平均层/头并累加 Query；新配置使用 FP16 Query/index、FP32 归约。
- [x] **显式 token-ID Query 规划**：绝对 Query span、调用方提供的 user message
  range、auto/force/off、pressure/keep_selected/semantic_chunk，非法组合拒绝准入。
- [x] **Query 边界重放**：封页起点检查点、partial tail + 完整 Query 原子调度、
  仅 Query token 评分、全 batch GDN rollback 和 durable replay。
- [x] **Immutable KV 与 dense re-RoPE 装配**：移动 K 从 raw source 重建，
  避免反复旋转/量化历史；支持同位置 K 跳过和 resident V 复用。
- [x] **有界 GPU raw-K cache、packed assembly、pinned gather 和传输 fence**。
- [x] **共享 archive 读准入背压**：后台封页占满队列时有界等待并重试，保留
  generation lease 和独占 staging buffer；连续跨轮负载和真实 I/O 竞争已验证。
- [x] **长 prefill 的 chunk 事务**：provisional Q → select → rollback →
  durable replay。当前是 qw3 semantic-chunk 契约的子集。
- [x] **GDN 状态维护**：保留完整逻辑状态；provisional 执行回滚 batch 邻居，
  只发布已接受的历史，处理 rejected suffix、EOS、取消和 partial tail。
- [x] **原生 Qwen MTP**：target 和 MTP 同选页、同步重建；k=2/3 已有资格测试。
  当前编译配置使用 k=3。
- [x] **DFlash2 主路径**：独立 sliding draft context；其已验证配置与当前原生
  MTP 编译配置分开，不能据此宣称 compiled DFlash 已验收。
- [x] **原生编译及 CUDA Graph**：单 GPU、同步在途 batch、原生 safetensors
  Qwen MTP 路径已验证。
- [x] **CPU 历史维护优化**：跨请求批量 copy/MeanK 更新、向量化小包归约、
  有界异步封页、缓存 scatter views、大包 CPU scatter 与 GPU 工作重叠。
- [x] **token-ID 弹性会话**：鉴权、用户 namespace、版本化增量历史、用户配额、
  active/restore 字节准入、队列 deadline 和取消后资源释放。
- [x] **回合结束暂停与恢复**：旧配置使用终态 checkpoint；显式 Query 配置保留
  最近封页检查点并重放尾部。GPU 请求页回收，exact-prefix 命中时恢复 KV/GDN；
  压力恢复重新选取 K 页，禁止复用已归还物理页的 residency 标记。

## 部分实现，仍有行为差异

| 项目 | 已有部分 | 尚缺部分 |
| --- | --- | --- |
| MeanK 评分 | 新配置的 FP16 Q/index、原版基础 CUDA kernel、串行 token 顺序封页均值 | 增量 GPU index、one-dot/two-pass、Adaptive 和在线逐层评分；旧配置仍为 FP32/PyTorch。 |
| Query 语义 | token-ID 完整 Query、调用方提供的角色和 message range | 自动聊天模板坐标映射；超过 G−B 的完整 Query 跨 batch 累计。 |
| 重选策略 | auto/force/off、pressure/keep_selected 和独立 semantic_chunk；长 prefill 留出 MTP 空间 | keep_selected 对长逻辑历史的缓存空间快速准入；原版所有跨轮 gate 与附加策略。 |
| Query 回滚/重放 | Query 前封页检查点、partial tail + 完整 Query、CPU 检查点跨轮恢复 | 任意历史编辑、checkpoint ladder；早于已有检查点的 Query 采用冷恢复。 |
| mandatory 内容 | sink/recent 和低层 `mandatory_handles` | 在线消息、工具、文件证据 spans 的规划与映射。 |
| historical storage | CPU cache + NVMe arena、预算内历史保留 | 原版 offline archive/session 格式、重启持久恢复、完整公开 GC policy。 |
| resident KV 复用 | raw-K cache、同位置 K 跳过、resident V 复用 | 原版完整免搬动页窗口；当前 dense native 窗口仍需搬动部分 resident V。 |
| 工具等待 | 终态释放 GPU 页、CPU checkpoint、历史 TTL | 短工具等待期间暂留 GPU 页的 TTL 策略和细粒度带宽协调。 |
| DSH 接入 | 本机可选入口已通过真实 read 工具及两轮对话 | 稳定 KVMem session 标识、跨轮 CPU KV 复用和 token-ID 弹性网关接入。当前 DSH 发送完整聊天历史。 |

## 尚未实现或未接入公开运行路径

- [ ] 独立 prefill budget；当前固定等于 semantic select budget。
- [ ] `max_tokens=0` 的 prefill-only 及原版 `start/append/finish` ingest 协议。
- [ ] 原版增量 GPU MeanK index、one-dot/two-pass scorer 和在线逐层评分。
- [ ] Q-only/early exit、provisional 前缀精确重用、提前 Query prefetch。
- [ ] Adaptive、SubBlockMeanK、PerToken、其他检索/分配模式的公开配置。
- [ ] Query relevance history/decay、action attention、语义消息扩展、CacheBlend。
- [ ] 历史编辑、多分支 rewind、命名 snapshot 和公开 checkpoint ladder。
- [ ] 任意 decode 点暂停/恢复的完整公开契约。
- [ ] 多模态历史索引、视觉身份和 MRoPE remap。
- [ ] 多 GPU/多 rank、多进程共享会话及跨模型恢复。
- [ ] 无界逻辑历史：当前仍受 `max_model_len` 和会话配额约束。
- [ ] 原生异步多在途 batch 与 KVMem 编译路径的适配。
- [ ] PCIe/NVMe 带宽分配、工具竞争协调和逐 token SLO 调度。

## 已留存验证及未验收边界

| 验证 | 结果及范围 |
| --- | --- |
| 最新独占 CPU/GPU 回归 | 699 passed、6 skipped；pinned/frame 针对测试 21 passed，包含真实 GPU DMA fence 检查；新增四个真实 I/O 背压回归。 |
| 显式 Query 跨轮与并发 | 最终安装包连续完成三组 C8、50,006 token 输入，共 48 回合、48 次完整 Query/partial-tail 重放、24 次 CPU checkpoint 恢复；96 个 GDN tensor 字节检查及 GPU 页回收均通过。 |
| 新版 C8，32K 输入 + 4K 输出，不带 session | 稳态 decode 576.48 token/s、整轮 347.38 token/s；KVMem KV 历史维护开启，未计周期会话检查点。 |
| 相同输入输出，带 session，pinned 优化前 | 稳态 decode 378.21 token/s、整轮 259.13 token/s；原生确认 8 路 decode。最终快照配置的结果见新报告，不能用上一行代替完整恢复方案的吞吐。 |
| 相同输入输出，带 session，最终 pinned 快照 | 稳态 decode **503.22 token/s**、整轮 **313.07 token/s**；本轮相对 pageable 快照速率增幅 33.1%，单次、生成序列不同，不代表稳定提升保证。 |
| 同组 C8、32K 输入的有效 prefill | 262,144 输入 token / 36.65 秒 = **7,153 输入 token/s**，包含排队、历史维护和穿插 decode；不是纯 GPU prefill 吞吐。 |
| 64K 两回合恢复 | 17 层 KV、96 个 GDN state tensor 恢复字节检查通过。 |
| 8 独立用户，32K 输入 + 16K 输出 | 8 路完成，零抢占；稳态 decode 538.75 token/s，整轮 449.94 token/s。 |
| C8，64K 输入 + 4K 输出 | 最新配置稳态 decode 619.37 token/s。 |
| CPU copy + MeanK 更新组件 | 孤立交替测量耗时减少约 59.7%；不是整体 CPU 或服务吞吐提升比例。 |
| DSH 本机连接 | read 工具读取随机标记，第二轮无工具复述成功；短对话不足以验证超过 32K 的历史选页或跨轮 KV checkpoint 命中。 |

尚未验收的项目包括：与 qw3 完整请求策略逐页等价、长期多用户工具负载的
SLO、完整 agent/长历史质量、8 用户各 256K 的总容量，以及异步多 batch、
多 GPU、compiled DFlash。100 GiB CPU read cache 是配置预算，并非总进程
RAM 上限，也不等于所有用户历史始终常驻 CPU。

吞吐数据属于固定输入/输出测试，不能推断持续工具负载下的延迟保证。与无
KVMem 的原生同步编译控制相比，当前吞吐仍有差距；见对应测量报告。

## 下一阶段顺序

1. 增量 GPU index、one-dot/two-pass 与在线评分。
2. keep_selected 缓存空间准入，以及更大 Query 的有界跨 batch 捕获。
3. 独立 prefill budget、prefill-only ingest 和 mandatory spans。
4. DSH 的完整 token 坐标及会话适配、工具负载和 SLO 验收。

一致性验证应固定历史、Query 和配置，比较选中的页 ID、mandatory 覆盖、
重放范围、历史长度及 MTP/GDN 状态。吞吐相近不能代替语义等价验证。

## 关联文档

- [上下文管理一致性核对](context-management-parity-20261014.md)
- [完整 qw3 功能审计](qw3-parity-audit-20261014.md)：冻结版本的旧状态以本清单
  和后续编译/批处理更新为准。
- [原生编译、CPU 批处理与最新测量](compiled-capture-history-20261014.md)
- [MeanK 集成与配置](meank-integration-20261013.md)
- [弹性会话 API](elastic-serving.md)
- [显式 Query 与原版 scorer 配置](../examples/qwen-nvfp4-meank-explicit-query-elastic.json)
