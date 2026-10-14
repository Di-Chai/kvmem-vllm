# KVMem + vLLM 实现状态

更新日期：2026-10-14。运行时基线：`3bc639b`；本次文档补充前已提交的仓库版本：
[`9cbf943`](https://github.com/Di-Chai/kvmem-vllm/commit/9cbf9432e0ed1787fbe3c9cc445b04274d4dd95c)。
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
- [x] **基础 MeanK 数学定义**：使用模型 content-frame Q/K，页间 softmax，
  平均层/头并累加 Query。精度和 Query 来源的差异见下节。
- [x] **Immutable KV 与 dense re-RoPE 装配**：移动 K 从 raw source 重建，
  避免反复旋转/量化历史；支持同位置 K 跳过和 resident V 复用。
- [x] **有界 GPU raw-K cache、packed assembly、pinned gather 和传输 fence**。
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
- [x] **回合结束暂停与恢复**：终态 checkpoint、partial tail、pending last output、
  GPU 请求页回收；exact-prefix 匹配时恢复选中的 KV 和 GDN 状态。

## 部分实现，仍有行为差异

| 项目 | 已有部分 | 尚缺部分 |
| --- | --- | --- |
| MeanK 评分 | 基础数学定义及原版参考对照 | 原版生产 FP16 Query/index 和专用 CUDA scorer；当前为 FP32 index/PyTorch 评分，近似同分时 TopK 可能不同。 |
| Query 语义 | 从调度 chunk 捕获 Q；排除未验证 draft | 显式完整用户 Query、message range、角色、跨 chunk Query 规划。 |
| 重选策略 | chunk 边界及导入后的刷新、K/G 容量控制 | 原版 `auto/force/off`、`pressure/keep_selected` 和完整跨轮 gate。 |
| Query 回滚/重放 | 当前 chunk 的事务回滚、终态会话恢复 | 显式 Query 起点前的 checkpoint，以及 partial tail + 完整 Query 重放。 |
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
| 最新 CPU/GPU 回归 | 669 passed、6 skipped；详见编译与批处理报告。 |
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

1. 显式 Query/message range、请求规划和原版重选策略。
2. Query 边界 checkpoint 与 partial-tail/full-Query replay。
3. 原版索引精度、GPU index 和 CUDA scorer。
4. 独立 prefill budget、prefill-only ingest 和 mandatory spans。
5. DSH 的完整会话适配、工具负载及 SLO 验收。

一致性验证应固定历史、Query 和配置，比较选中的页 ID、mandatory 覆盖、
重放范围、历史长度及 MTP/GDN 状态。吞吐相近不能代替语义等价验证。

## 关联文档

- [上下文管理一致性核对](context-management-parity-20261014.md)
- [完整 qw3 功能审计](qw3-parity-audit-20261014.md)：冻结版本的旧状态以本清单
  和后续编译/批处理更新为准。
- [原生编译、CPU 批处理与最新测量](compiled-capture-history-20261014.md)
- [MeanK 集成与配置](meank-integration-20261013.md)
- [弹性会话 API](elastic-serving.md)
- [当前编译配置](../examples/qwen-nvfp4-meank-mtp-compiled-elastic.json)
