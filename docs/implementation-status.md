# KVMem + vLLM 实现状态

更新日期：2026-10-18。新增默认关闭的 Query 驱动 block 图评分，完成六种模式及同步恢复的
小规模真实模型验收，见 [图评分实现与验证](graph-retrieval.md)。本次还修复 guided Query
接受策略、M→P prefill 历史保留和联合容量准入，支持混合角色 Query 范围；具体核对和验收
见 [最新 qw3 对齐记录](qw3-alignment-20261018.md)。完整封页的 CPU 追加路径现跳过
随后会被最终 MeanK 覆盖的重复求和；对应的组件验证和同参数 FP8 八并发实验见
[FP8 多并发验证](aligned-fp8-agents-20261018.md)。以下早期性能数据保留原日期与配置。

FP8 八并发的保守主机启动余量已按实测峰值校准，完整历史预算和运行时保护
保留；新的同条件对照见 [主机启动余量校准](host-admission-calibration-20261018.md)。
新增有界临时封页缓冲池，以及默认关闭、显式配置的原生 CUDA 空闲缓存回收。
两者都保持历史格式和选页语义。GPU 缓存回收与 KVMem repository GC 不同；
不会删除历史 KV 或图节点。CUDA 策略的真实模型与部署资格按该报告单独列出，
不能将未通过显存余量门禁的候选记为性能收益。

最终同协议 FP8 八并发、300 秒测试已通过完整历史、输出对账、零抢占、显式
释放及 CPU/GPU 余量门禁。服务输出率为 145.73 → 146.95 token/s，差值仅
0.84%，完成轮次 93 → 89，未证明稳定整体加速。实际 KV pool 均为 779 blocks；
候选 CPU 可用内存最低 26.45 GiB、GPU 空闲显存最低 5.94 GiB。结论和所有失败
尝试见上述校准报告。图评分的质量结论仍限于原小规模功能实验，不宣称提升。

2026-10-14 基线为显式 Query / 原版基础 FP16 MeanK + P/M 跨轮上下文版本；此前编译与
CPU 批处理基线为 `3bc639b`。新增路径的契约和资格范围见
[跨轮上下文实现与验收](context-policy-20261014.md)；最新可选组合见
[guided Query 与历史评分](guided-history-20261014.md)。此前 Query 基线见
[历史验收报告](explicit-query-20261014.md)。
开发分支：`main`。

2026-10-16 更新：服务端 immutable raw K 已改为 FP8 E4M3FN，CPU 历史和 GPU raw-K
cache 使用相同格式；MeanK 索引仍为 FP16。格式、容量计算与本次验证范围见
[FP8 raw K 迁移](fp8-raw-k-20261016.md)。此前报告的 BF16 raw-K 数值属于旧版本。

当前已实现分离式 KV 管理和有界 MeanK 主路径，但**尚未完整复现 qw3 的请求级
上下文管理语义**。单轮输出不超过 generation budget G 是保留的设计；多 epoch
生成不列为本阶段的缺失要求。此清单中的“已实现”只覆盖注明的配置及入口。

## 已实现并有验证

- [x] **分离 active / historical KV**：GPU 保存有界 working window；CPU authority
  管理已接受的历史 raw K/V，封页可进入 NVMe arena。
- [x] **可选纯 CPU 历史存储**：`archive_backend=cpu` 保存不可变 FP8 raw-K / FP8-V，
  保留 generation、lease、发布与回收协议；无磁盘回退。100 GiB 是历史存储预算，
  不是总 RAM 上限。NIAH 入口另按完整输入预留额度；通用服务的 CPU 准入不能据此
  宣称全部完成。配置与运行证据见 [180 题 NIAH 报告](niah-20261014.md)。
- [x] **K/G 预算**：选取历史最多 K，预留 G 用于生成；当前公开 MeanK 弹性入口
  限制单轮输出不超过 G。原生物理页、MTP lookahead 和 staging 还有额外开销。
- [x] **原版 Store 选页主路径**：sink/recent、TopK、历史排序、remap diff、
  generation/lease 管理。Store 有 bounded pressure-budget 等适配，不能再称所有
  核心文件逐字节一致；最新逐文件比较见 2026-10-18 对齐记录。
- [x] **基础 MeanK 数学定义及原版 CUDA scorer**：content-frame Q/K、页间
  softmax、平均层/头并累加 Query；新配置使用 FP16 Query/index、FP32 归约。
- [x] **显式 token-ID Query 规划**：绝对 Query span、调用方提供的混合角色 message
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
  此前编译配置使用 k=3；本轮纯 CPU 历史 NIAH180 及 C8 固定输出实际使用 k=4。
- [x] **DFlash2 主路径**：独立 sliding draft context；其已验证配置与当前原生
  MTP 编译配置分开，不能据此宣称 compiled DFlash 已验收。
- [x] **原生编译及 CUDA Graph**：单 GPU、同步在途 batch、原生 safetensors
  Qwen MTP 路径已验证。
- [x] **CPU 历史维护优化**：跨请求批量 copy/MeanK 更新、向量化小包归约、
  有界异步封页、缓存 scatter views、大包 CPU scatter 与 GPU 工作重叠。
- [x] **token-ID 弹性会话**：鉴权、用户 namespace、版本化增量历史、用户配额、
  active/restore 字节准入、队列 deadline 和取消后资源释放。
- [x] **跨轮保持选页与实际 headroom**：exact-prefix 校验后恢复原 active frame；
  稳定工具续接/重复旧 Query auto 不自动重选。按实际窗口+增量+输出检查空间，
  不足时在运行容量边界压力刷新。网关全局字节预留仍为保守 K+G。
- [x] **必留 spans / capacity fit**：system/root/Query/policy 与显式工具/文件范围，
  Query 和压力选页均接入原版 qw3 helper；超额按结构边缘与优先级适配，保留检索额度。
- [x] **P/M 暂停恢复**：保留 Query 前 P 和终态 M，普通续接从 M，强制旧 Query
  可退回匹配 P；支持包含 Query 的独立 live replay span 和 branch-safe 模板尾块。
  GPU 请求页回收，检查点配额内保留 CPU 快照，失配/缓存不足可降级重放。
- [x] **Query relevance history/decay**：使用原版 C++ 逻辑回合规则；可与显式边界
  guided Query 同开（扩展原版语义），见 [实现与验收](guided-history-20261014.md)。

## 部分实现，仍有行为差异

| 项目 | 已有部分 | 尚缺部分 |
| --- | --- | --- |
| MeanK 评分 | 新配置的 FP16 Q/index、原版基础 CUDA kernel、串行 token 顺序封页均值 | 增量 GPU index、one-dot/two-pass、Adaptive 和在线逐层评分；旧配置仍为 FP32/PyTorch。 |
| Query 语义 | token-ID 完整 Query、调用方提供的角色和混合角色 message range | 与 qw3 同名字段协议不同；默认最后 user Query / 自动聊天模板映射、超过 G−B 的跨 batch 捕获。 |
| private guided Query | 私有 prompt、target Q 捕获、状态回滚；跳过前导空白，仅接受自然终止，结构化／空／截断输出回退 | direct SOURCE REQUEST 重写、private thinking、完整 prompt 和采样轨迹仍有差异。本版私有生成保持有界 greedy。 |
| 重选策略 | exact-prefix 证明下的旧 Query auto、保持选页、实际 active headroom、容量压力刷新；显式 Query 边界可启用 private guided Query | 完整任务/Query 身份、增长/evidence gate（用户要求暂缓）；不自动生成聊天 Query，也无新增 decode 中途触发。 |
| Query 回滚/重放 | partial tail + 完整 Query、包含 Query 的 live span、一个 P + 一个 M、branch-safe 恢复 | 任意分离的历史 Query/live suffix、Query tensor cache、编辑与完整 ladder；无匹配 P 时仍冷恢复。 |
| mandatory 内容 | 在线 token spans、角色映射、原版 capacity fit 和 generation-qualified handles | 聊天模板自动坐标与完整证据生命周期；不保证超大 span 整体驻留 GPU。 |
| historical storage | CPU cache + NVMe arena，或可选纯 CPU 不可变历史；预算内历史保留 | 原版 offline archive/session 格式、重启持久恢复、完整公开 GC policy；纯 CPU 服务的通用完整历史准入仍需补齐。 |
| resident KV 复用 | raw-K cache、同位置 K 跳过、resident V 复用 | 原版完整免搬动页窗口；当前 dense native 窗口仍需搬动部分 resident V。 |
| 工具等待 | 终态释放 GPU 页、CPU checkpoint、历史 TTL | 短工具等待期间暂留 GPU 页的 TTL 策略和细粒度带宽协调。 |
| DSH 接入 | 本机可选入口已通过真实 read 工具及两轮对话 | 稳定 KVMem session 标识、跨轮 CPU KV 复用和 token-ID 弹性网关接入。当前 DSH 发送完整聊天历史。 |

## 尚未实现或未接入公开运行路径

- [ ] 独立 prefill budget；当前固定等于 semantic select budget。
- [ ] `max_tokens=0` 的 prefill-only 及原版 `start/append/finish` ingest 协议。
- [ ] 原版增量 GPU MeanK index、one-dot/two-pass scorer 和在线逐层评分。
- [ ] Q-only/early exit、provisional 前缀精确重用、提前 Query prefetch。
- [ ] Adaptive、SubBlockMeanK、PerToken、其他检索/分配模式的公开配置。
- [ ] 语义消息扩展、CacheBlend。
- [ ] 历史编辑、多分支 rewind、命名 snapshot 和公开 checkpoint ladder。
- [ ] 任意 decode 点暂停/恢复的完整公开契约。
- [ ] 多模态历史索引、视觉身份和 MRoPE remap。
- [ ] 多 GPU/多 rank、多进程共享会话及跨模型恢复。
- [ ] 无界逻辑历史：当前仍受 `max_model_len` 和会话配额约束。
- [ ] 原生异步多在途 batch 与 KVMem 编译路径的适配。
- [ ] PCIe/NVMe 带宽分配、工具竞争协调和逐 token SLO 调度。

### Action attention（2026-10-14）

已补齐可选 prefill mass/exposure、按 attention 的 resident pressure 选页，
以及冷启动 Query 的 prefill 评分融合；选择器调用原 qw3 C++ 实现。8 并发、
82K 输入的两轮真实模型验收通过，覆盖压力重选、Query 重放排除和跨轮评分恢复。
另支持冻结 P 前缀模式，供独立 benchmark 问题复用输入 KV 而不继承上一题答案。
完整 256K NIAH 60 题已完成：覆盖率 **97.08%**、整题 **53/60**，旧版为
68.33%、30/60；qw3 同题 attention 对照为 98.75%、57/60。MemoryAgentBench
同批 1,116 题推理全部完成，29 次首次加载、1,087 次冻结前缀恢复，零错误/抢占，
CPU-only；1,116 题已全部评分，三类别宏平均 35.423%，qw3 对照 36.635%。独立 LongMemEval-S 102 题真实 C8 也已完成，91/102 对 qw3 的 90/102，零错误、零抢占、零截断，见 [两套记忆评测报告](longmemeval-s-attention-20261015.md)。另补齐原 qw3 的
大历史 MeanK CUDA 分块路径，支持至 32,768 页。详见 [完整路径报告](prefill-attention-20261014.md)。

已接入原版 qw3 scope parser、按响应衰减和 quota/score 选页融合，可通过
`action_attention=true` 开启。支持 tool-call/non-thinking、last/spread3/all/指定层、
采样间隔和权重；只使用 target 标准注意力层及实际接受的输出 token。
使用真实工作 KV 的抽样 token attention，保留 native compilation/CUDA Graph/MTP。
P/M 检查点携带评分，完整 M token 前缀验证通过后的 P 边界重放保留该响应的有效评分；
历史编辑则回退评分，不向新分支泄漏未来反馈。详见 [Action attention](action-attention.md)。

默认配置仍在整个 batch 的执行入口同步刷新。新增可选 `async_refresh=true`：
只暂停刷新请求，私有 Query 分步推进，MeanK／历史读取／装配后台执行，其他请求
继续 decode；详见 [请求级刷新](async-refresh-20261014.md)。它与 chunked prefill
互补，不等于启用原生异步多在途 batch；共享 GPU 和前台会话恢复仍有干扰。

## 已留存验证及未验收边界

| 验证 | 结果及范围 |
| --- | --- |
| 请求级异步刷新 | CPU697 passed / 77 skipped；最终C8两轮16请求及两个取消点通过。7路decode＋1路50K长请求中，Query窗口邻居最大输出间隔1.212→0.191秒；整轮吞吐644→548 token/s，未证明吞吐提升。见[完整结果与范围](async-refresh-20261014.md)。 |
| Action attention 跨轮与并发 | CPU 688 passed / 77 skipped，GPU 数值测试 4 passed；真实模型 C8 两轮16请求通过，每会话16/8个有效样本，第二轮恢复1094页正分数；含完整前缀校验后的 P 边界重放。 |
| Action attention 开关对照 | NVFP4、FP8 KV、MTP4、CPU历史、guided512，50,006输入+4K输出，C8、K32K/G16K；关闭/开启稳态849.29/726.35 token/s，整轮204.85/205.20 token/s。单次对照稳态下降14.48%，启动状态不同，整轮差异不可用于断言零开销；见 [报告](action-attention.md)。 |
| 纯 CPU NIAH180 | 64K/128K/256K 各60题，完整冻结输入；实际准入8/8/6路，180题零技术错误、终态历史长度及回收通过。数字覆盖率100%/92.50%/65.42%，存在漏答和一例名称绑定错误，**质量未达旧qw3参考**。无历史KV磁盘I/O；见 [报告](niah-20261014.md)。 |
| 纯 CPU 固定输出 C8，50,006输入 + 4K输出 | MTP4、B32、guided512、无保留checkpoint；稳态860.71、整轮313.49输出token/s，有效输入6146token/s；原生8路共同decode覆盖99.26%。与下行729基准配置不同，不能视作单变量优化收益。 |
| guided Query + Query relevance history | 同时开启的扩展语义：CPU 663 passed / 73 skipped，原版 C++ 回归通过；C8 五轮40回合及后续主体构建C8两轮16回合通过。预算内同输入重试不重复累计，无Query续接分数不变。详细构建范围见组合报告。 |
| guided/history 长文 C8，50,006输入 + 4K输出 | 两项实际执行后的稳态729.37、整轮199.18 token/s，原生8路共同decode覆盖99.40%；TTFT p95 108.27秒。输入不同于此前32K文本，作为独立基线。 |
| 当前上下文策略回归 | 最终 CPU 652 passed / 73 skipped；最终 worker 针对回归 38 passed / 1 skipped、launcher 87 passed。中间版本 GPU 713 passed / 6 skipped；最终 wheel 另过真实模型验收。 |
| 当前 P/M 与 mandatory 真实模型 | C8 四轮 32 回合、24 次恢复全部通过；另有 C1 五轮边界测试。最终确认顺序修复后另过单 token C1/C8 共 9 请求及 C8 五轮 40 回合（32 恢复），每用户历史 73,447；4 GiB 检查点配额零降级，GPU 请求页归零。 |
| 显式 Query 跨轮与并发 | 最终安装包连续完成三组 C8、50,006 token 输入，共 48 回合、48 次完整 Query/partial-tail 重放、24 次 CPU checkpoint 恢复；96 个 GDN tensor 字节检查及 GPU 页回收均通过。 |
| 当前正常 P/M 配置 C8，32K 输入 + 4K 输出 | 两轮稳态 **561.08 / 561.39**，整轮 **337.23 / 334.17 token/s**；8 路原生共同 decode、零抢占。关闭验收字节 oracle，实际 KVMem 和会话维护开启。 |
| 当前独立 prefill 测法 | 每路 32K 输入、1 token 输出：C1 **4,656 / 5,932**；C8 合计 **6,426 / 7,573 输入 token/s**，包含排队与历史维护。 |
| 同诊断开关的最终对照 | 稳态 560.57、整轮 301.36，对此前 503.22 / 313.07；稳态提高约 11.4%，整轮没有提升。见完整口径说明。 |
| 此前显式 Query 基线：C8，32K 输入 + 4K 输出，不带 session | 稳态 decode 576.48 token/s、整轮 347.38 token/s；KVMem KV 历史维护开启，未计周期会话检查点。 |
| 相同输入输出，带 session，pinned 优化前 | 稳态 decode 378.21 token/s、整轮 259.13 token/s；原生确认 8 路 decode。最终快照配置的结果见新报告，不能用上一行代替完整恢复方案的吞吐。 |
| 此前显式 Query 基线：带 session，周期 pinned 快照 | 稳态 decode **503.22 token/s**、整轮 **313.07 token/s**；本轮相对 pageable 快照速率增幅 33.1%，单次、生成序列不同，不代表稳定提升保证。 |
| 同组 C8、32K 输入的有效 prefill | 262,144 输入 token / 36.65 秒 = **7,153 输入 token/s**，包含排队、历史维护和穿插 decode；不是纯 GPU prefill 吞吐。 |
| 64K 两回合恢复 | 17 层 KV、96 个 GDN state tensor 恢复字节检查通过。 |
| 8 独立用户，32K 输入 + 16K 输出 | 8 路完成，零抢占；稳态 decode 538.75 token/s，整轮 449.94 token/s。 |
| C8，64K 输入 + 4K 输出 | 最新配置稳态 decode 619.37 token/s。 |
| CPU copy + MeanK 更新组件 | 孤立交替测量耗时减少约 59.7%；不是整体 CPU 或服务吞吐提升比例。 |
| DSH 本机连接 | read 工具读取随机标记，第二轮无工具复述成功；短对话不足以验证超过 32K 的历史选页或跨轮 KV checkpoint 命中。 |

尚未验收的项目包括：与 qw3 完整请求策略逐页等价、长期多用户工具负载的
SLO、完整 agent/长历史质量、8 用户各 256K 的总容量，以及异步多 batch、
多 GPU、compiled DFlash。100 GiB CPU read cache 是配置预算，并非总进程
RAM 上限；NVMe 后端下也不等于所有用户历史始终常驻 CPU。新纯 CPU 后端无磁盘
回退，需在入口为完整历史保留额度。当前 256K NIAH 按内存余量最多放行 6 路。

吞吐数据属于固定输入/输出测试，不能推断持续工具负载下的延迟保证。与无
KVMem 的原生同步编译控制相比，当前吞吐仍有差距；见对应测量报告。

## 下一阶段顺序

2026-10-15 调度协调第一版代码已接入，真实性能验收等待另一组 GPU 测试结束。
`cpu_history_coordination=true` 在 native connector 分配前按请求核算完整本轮
CPU 历史额度；工具等待保留历史，高水位额度在 worker 释放确认后归还。
`cpu_progress_reserve=true` 保留一条已驻留轨迹增长到声明历史上限的空间。
会话槽满时等待，不用 LRU 淘汰或无缓存执行绕过历史保留。该额度只覆盖历史
KV，checkpoint、暂存缓冲和进程其他内存仍有独立预算；历史不足的请求可能
等待调用方结束并释放其他会话，不保证任意数量的无限增长轨迹均可推进。

`maintenance_max_steps=8` 启用现有异步重选的协同推进：先回收已完成事务，
按等待时间和估计上传量选择准备好的请求。上传量估计复用此前实际 raw-K
缓存命中和 resident V 复用后的传输统计。独立维护工作使用 host 时间预算，
有公开 batch 时最多推进一个私有模型 quantum，空闲时可连续推进多个阶段。
单个 assembly 的 CPU/H2D 额度一直持有到 CUDA fence 完成并提交；这不是硬
GPU 时间限额，也没有将所有普通 P/M checkpoint 恢复或 D2H 工作异步化。
普通 prefill quantum、公开 Query 边界、MeanK 和原生 GPU 页分配保持原策略。

上述两个开关默认关闭。`scripts/qualify_coordination.py` 提供同配置 B0/B1/B2
比较，覆盖 8 并发 decode、prefill、长短混合、工具等待和集中返回；保存流式
token 计数、事件间隔、TTFT、跨轮实际输出前缀、原生指标和配对统计。
GPU 连续 120 秒无计算进程才启动，发现其他 GPU 作业则停止自己的进程组。
没有真实配对结果前，不将本版标记为吞吐提升或 SLO 已验收。

同日继续补齐了实验性 v2 代码（仍待真实模型验收）：
`async_restore=true` 将普通 P/M 跨轮恢复接入单个后台维护执行器，完成前
仅挂起对应请求，并保持其 CPU checkpoint、GPU 页和原生状态槽的所有权。
取消不会提前释放 DMA 目的页，也不会丢掉刚导入的历史。新分配凭据在 CPU
核对页号，避免恢复准备中的 GPU 页表回读；模型状态复用打包的 pinned CPU
快照。非阻塞 P rewind 保留跨次轮询的超时，且等旧历史读取者排空才完成。

`adaptive_prefill=true` 按已完成模型步延迟调整普通 prefill chunk 的上限，
保留正数下限、vLLM 原生调度顺序和全局预算，完整 Query 重放仍为原子操作。
`step_latency_interval` 的事件抽样不主动等待 GPU，覆盖目标模型、采样及
MTP，且可能包含主机提交空隙；它不是逐 token SLO，也不是纯 kernel 时间。
这三个新选项默认关闭。v2 比较增加 B3（异步恢复）、B4（动态 prefill），
同一对照中各组使用相同采样设置。配置、指标及待执行步骤见
[调度协调 v2](coordination-v2-20261015.md)。原始 v1 排队测试使用固定源码快照。

1. 原版 Query / 消息协议及默认 user Query；DSH 的会话与坐标适配。
2. 调用方 evidence 和增长阈值触发按用户要求暂缓；显式边界 guided Query 与历史评分已接入，继续验证质量与重选延迟。
3. Query tensor cache（区别于已实现的历史相关性分数）、任意历史编辑与 checkpoint ladder。
4. 独立 prefill budget、prefill-only ingest；再迁移增量 GPU index 等效率优化，
   并做工具负载和 SLO 验收。最新具体对照见上下文一致性报告。

一致性验证应固定历史、Query 和配置，比较选中的页 ID、mandatory 覆盖、
重放范围、历史长度及 MTP/GDN 状态。吞吐相近不能代替语义等价验证。

## 关联文档

- [本轮跨轮上下文策略与性能](context-policy-20261014.md)
- [修改前上下文管理一致性核对](context-management-parity-20261014.md)
- [完整 qw3 功能审计](qw3-parity-audit-20261014.md)：冻结版本的旧状态以本清单
  和后续编译/批处理更新为准。
- [原生编译、CPU 批处理与最新测量](compiled-capture-history-20261014.md)
- [MeanK 集成与配置](meank-integration-20261013.md)
- [弹性会话 API](elastic-serving.md)
- [显式 Query 与原版 scorer 配置](../examples/qwen-nvfp4-meank-explicit-query-elastic.json)
