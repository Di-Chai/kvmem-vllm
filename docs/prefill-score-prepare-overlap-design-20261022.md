# 在完整 KV 下载等待前准备普通 prefill 分数事务

状态：只读设计，尚未实现或测试。默认配置不变。本轮先完成 CPU 读取融合候选及新的 native/KVMem 参照，再决定是否实现本文候选。

## 证据：本次压力边界没有明显的后台唤醒空洞

来源为 `artifacts/prefill-pressure-phases-20261022-r1/trace.sqlite`、同目录 `timeline-summary.json` 和 `prefill-pressure-attribution-20261022.json`。该 trace 开启 early Action、native plan arm、capture ahead、publication pipeline、fused pack、direct seal、hash workers=8；deferred validation 和 scatter-after-target 关闭。以下时间均相对于该次 `suspended` 标记；它们是带 profiler 的单次诊断，不能直接外推吞吐收益。

| 事件 | 相对时间（ms） | 含义 |
| --- | ---: | --- |
| early native score validation 返回 | -2.454449 | 分数及其 D2H 已完整验证，MTP/KV 尾部仍运行 |
| publication worker 进入 `drain` | -0.326293 | 后台已启动，甚至早于 suspended 标记 |
| publication worker 开始 `cudaEventSynchronize` | -0.208505 | 等该包完整 KV D2H |
| 前一步 GPU/下载最后活动结束 | 18.969316 | 完整 target+MTP payload 可用 |
| publication worker 的等待返回 | 18.984680 | 与活动末端相差约 15 微秒 |
| `action.prepare_accept` | 19.247647–27.239004 | 约 7.991 ms；当前位于整包 DMA 后 |
| native scatter | 30.064894–37.973266 | 约 7.908 ms |
| native MeanK finalize | 38.410166–64.300942 | 约 25.891 ms |
| owner publication 返回 | 68.812370 | 历史、分数和图已按原序提交 |
| `assembly_queued` | 70.467671 | publication 后约 1.655 ms |
| assembly worker 启动 | 71.433030 | queued 后约 0.965 ms |
| `publish_sealed` | 72.100419–89.184006 | assembly 仍等待异步封存完成 |

因此，suspended→assembly_queued 的 70.468 ms 主要来自 GPU/DMA 尾部和随后的 CPU 工作。不是后台任务晚启动了几十毫秒，也不是把线程改成常驻等待就能消除的空洞。实际准备/发布工作已在下一调度决策到来后迅速开始。

`prepare_accept` 的约 8 ms 包含 native 累计分数 3.863 ms 和 native 构图 2.320 ms。`csrc/prefill_attention_commit.hpp` 与 `csrc/block_graph_accept.hpp` 都在主要计算区间释放 GIL；不能将整个 8 ms 归因于 Python/GIL。已有小组件成本与 profiler 区间不可混用。MeanK SIMD 的组件提速也没有在完整模型 A/B 中转化成吞吐提升。

## 当前依赖链

1. `action_attention.py:ActionRuntime.stage_before_mtp` 在 target sampling 后排入完整分数归约、层平均、分数 D2H 和校验 event。
2. `mtp_runtime.py:propose` 正常调用 `original_propose`，再将 target 与 MTP prefix captures 一起交给 `CaptureRuntime.submit`。MTP 归档层必须随 target 同步保存，不能遗漏。
3. capture ahead 提前排入完整 payload pack/D2H；`action.capture` 仍在本 forward 内等待早 score event、处理 fallback 并验证坐标。成功后入队的 `CapturedPacket` 仍为 `confirmed=False`。
4. 下一 `runner._kvmem_execute_model` 首先执行 `begin_step`，安装 EOS/cancel terminal prefix，再确认先前包。`advance_publication` 启动现有单线程 scatter future。
5. `drain_requests` 按 owner FIFO claim；当前 `_copy_source_owners` 先等整包 DMA、读 GPU header、计算实际接受/裁剪行，然后 `prepare_accept`、预算预留、raw append/scatter/MeanK，最后才提交 Action/graph 状态。
6. `AsyncRefresh.pump` 必须等该 owner `publication_ready`。归档 seal/hash 是另外的已有 future；assembly 的 `publish_sealed` 仍需等它完成。

普通非末 prefill 数学上没有 speculative rejection：安装版 vLLM `gpu/input_batch.py:_get_num_sampled_and_rejected_kernel` 对 `seq_len < prefill_len` 明确写出 `num_sampled=0,num_rejected=0`。但这不等于可以在 scheduler 确认前发布：`terminal.py:TerminalPrefix.from_request` 排除取消时仍在途的输入。既有下一决策确认边界必须保留。

## 候选：只前移 CPU prepare/reserve，不前移 commit

最早位置是现有 worker 已通过 `_claim_publications` 获得某 owner 的 FIFO claim、且 packet 已确认之后，在 `_copy_source_owners` 等待完整 KV DMA 之前。仍使用现有 worker，不增加线程，不改变 native score validation，不冻结 Q/K，不新增旁路 QK，也不采用之前无收益的跨帧 deferred validation。

`ActionRuntime.prepare_accept(packet, request_id, accepted, graph, history_begin)` 读取已完成的 score masses、Action packet 坐标、原 prefill 累计状态及 graph 状态，不读取 raw K/V。它复制 prefill 状态并构造新的分数/图候选，返回 `_PreparedActionAcceptance`；只有后续 `commit()` 才替换原分数状态和图。现有 `GraphUpdate.detach()` 可以释放 owner lock，同时保留预算 charge、binding 和 pending-owner 标记。

建议在 `submit` 成功完成原同步 score validation 后，为确切合格 packet 保存少量不可变 CPU 坐标/owner 身份。不能从后来可被覆盖的 `action.prefill_plans`、batch GPU buffer 或 live native bank 读取。可复用已经 `_validate_prefill_plan` 对真实 GPU metadata 完整校验过的 `ActionPacket.history_begin`、`candidate_count`、`prefill=(end,count,starts,local_begin)`；另存各 owner 的精确 header starts/count、authority/state 身份及 window generation。只保存整数/已有受包租约保护的对象，不新增无界矩阵或槽。

首版资格建议严格限定：

- 默认关闭的独立开关；early Action、native reuse、publication pipeline 已启用，原同步校验已经成功完成；`action_validation is None`，不与 deferred validation 叠加。
- 完整 bulk 包，全部 owner 为真正公共普通非末 prefill，全部命中既有 pipeline 计划；排除 decode、mixed/subset、私有/Query replay、末块、无 sampled Action packet 等情形。
- 当前 packet 已经下一 scheduler 决策确认；该 owner 的前包已发布，没有其他正在准备/提交的同 owner 更新。
- 快照中的 history/count/local 与原 Action packet 一致，authority/action state 身份未变，当前 authority.history_end 为本包起点，graph binding 有效。
- 起初已有 terminal prefix 的 owner 可以直接保留原路径，减少首版范围；即使起初没有 terminal，也必须处理等待期间新到的取消信息。
- 一项不满足即在任何预留发生前走原同步路径；不能降低采样、跳过校验或改变接受规则。

成功路径建议顺序：

1. 按原 `indices` 顺序计算已确认的候选 accepted 范围，调用原 `_prepare_graph_acceptance` / `prepare_accept`，成功后 detach 并放入现有 `packet.graph_updates`。
2. 观察原完整 KV DMA event；不提前 scatter、MeanK、归档或 graph/Action commit。
3. 解析实际 GPU header，重新核对 starts、history、local、scheduled 与快照一致且 rejection 为 0。快照/实际坐标不一致是错误，不能静默信任 CPU 上界。
4. 重新读取当前 terminal prefix 并计算 retained。等待 DMA 期间另一次 `begin_step` 可能安装取消边界，早时刻的 accepted 不能作为最终值。
5. 若 retained 未变且身份仍有效，复用已预留对象；否则先 abort 原预留，再按实际 header 和最新 clipping 调用原 prepare，仍在 append 之前完成。
6. 保留 `append_authority_batch` → scatter → MeanK → accepted history commit → seal 入队 → 原 Action/graph commit 的顺序。继续由 packet 全部 owner 完成后释放 DMA 槽。

这只可能将准备事务与约 19 ms 的 MTP/pack/D2H 尾部重叠。raw scatter/MeanK 仍要等完整 17 层 payload；把它们再提前需要拆分 target/MTP 数据到达和独立私有 staging，属于更大设计，不包含在本候选内。

## 事务、预算与失败边界

- 沿用原 owner 顺序和现有预算 reserve API，不新增第二套 charge；早准备只延长同一 reservation 的存活时间，不提前释放旧图，也不提前发布候选分数。
- 同一个 owner 只能有一个 pending `GraphUpdate`。每次 prepare 成功立即 detach，不能跨 DMA 等待持有 graph lock，也不能跨等待持有 runtime publication condition。
- 多 owner 的第 N 个准备失败时，前面已准备的 owner 必须按现有 `_abort_graph_updates` 统一回滚。raw history 尚未 append，原分数/parser/graph 应保持不变。
- 早准备异常发生时整包 DMA 可能还在执行，与旧路径的异常时点不同。必须保留 queued packet/source/slot 租约，阻止后续 publication，并在失败清理中观察已有 DMA；fence 失败则永久隔离，不能释放、复用或把包标成已发布。
- DMA/header/身份复核失败也必须 abort 全部尚未提交的预留。abort 只释放候选 charge/pending marker，不改变原数据。
- DMA 等待中 terminal 变化：abort 后再按原 header 重做，不能同时保留新旧两份预算预留。重新 prepare 失败依旧不能留下 raw history 或图的半更新。
- 后续 append/MeanK/commit 失败沿用现有致命 publication 处理，不新增失败后继续服务或跨帧恢复承诺。
- 较早占用共享 graph 预算可能改变多 owner 的瞬时预算竞争。必须沿原 claim/FIFO/reserve 顺序，预算不足时不能偷偷改变选页或绕过预留；资源耗尽需要独立资格测试。

## 验证与预期限制

组件应覆盖旧/新 masses、累计分数、graph edges、最终选页逐项一致；多 owner FIFO；早 event 已完成而完整 KV DMA 延迟；等待期间 terminal 从无到部分/零接受；header 与快照不符；第 N 个 owner 预算失败；DMA fence 失败；append/commit 失败；两固定槽连续复用和 packet 引用结束。graph prepare/abort 必须成对且原计数不能重复；新分支应有实际命中、早准备耗时及 terminal 导致重做的计数。

GPU 资格需证明准备确实在完整 KV event 之前运行、CPU 原状态在 event 前仍未提交、event 后字节一致；不以放大池或强制 GC 掩盖租约问题。完整模型必须做同源码、同配置的开/关 A/B，核验全部选页、输出、MTP、fallback、预算峰值和端到端 TTFT/吞吐；正常连续阶段也使用新 event 连续跨度参照。

风险是正常 prefill 中这只改变 CPU/GIL 竞争发生的时间：更早 prepare 可能与下一步输入准备争抢 GIL，且 native 主循环本身释放 GIL，不能保证将竞争完全移离首个 GDN。带 profiler 的 8 ms 不是无 profiler 的可节省时间；压力边界只是少量步骤，不应据此承诺整轮明显收益。当前证据支持测试这个小范围顺序调整，不支持重做后台线程/唤醒架构，更不支持直接将理论重叠相加成为吞吐提升。

## 2026-10-22：物理 14K 合并后的适用范围

状态仍为只读设计，未实施。当前物理 14K forward 包含七个原始 2K 逻辑子事务；每个子事务仍按原顺序执行分数/图准备、KV append、分数/图 commit 和运行时 ACK。

已完成的 `artifacts/prefill-physical14k-20261022-r1/candidate` 中，每个完整 256K 请求实际有 16 个七子块物理 group，共 112 个子块；该请求总共提交 128 个采样 prefill 逻辑块。数据来自 `run/server/capture/logical_prefill_publication.jsonl` 与 `prefill_attention.jsonl`，不是从配置推算。

在不改变累计顺序和图事务语义的前提下，可提前准备的只有每个物理 group 的**首个** 2K child，共 16 次，即全部 128 个采样逻辑块的 12.5%。最小接入位置是在已 confirmed 的 FIFO claim 中，将 `LogicalPublicationCursor` 的构造、确认和首个 child 的原 `prepare_accept`/`detach` 移到完整 KV DMA event 等待之前。先不扩展 `wait=False` 的非阻塞 claim：当前它会跳过 DMA 未完成的包，现有后台 `wait=True` 路径已经足够承载这一候选。

第二至第七个 child 必须读取前一个 child 已 commit 的累计 prefill 分数、graph binding 和预算状态，不能一次性提前准备。它们执行时完整物理包的 KV 已经到达，因此没有同包 DMA 等待可继续覆盖。不能将一次首块准备的收益乘以七。独立普通块仍可按前述资格判断，末块、部分接受、Query、恢复等不能因合并而自动纳入。

首块预留之后仍须等待原 event，再严格校验整包 header；等待期间若 terminal clipping、owner identity 或 binding 改变，应先 abort 旧预留，再按原规则重做。原 append/commit/ACK、失败隔离、两槽租约和后续子块边界均不能提前。尤其 `LogicalPublicationCursor.confirm` 目前允许后续确认裁剪未发布后缀，新增预准备状态必须显式处理这一失效，而非继续沿用过时 accepted 数量。

现有无 profiler 的重复 256K 请求中，`archive.json` 相邻完成记录的累计计数差为：native prefill commit 0.314268 秒 / 128 次，graph prepare 0.332836 秒 / 178 次（后者包含 decode）。按这些全请求平均成本估算，16 个首 child 的可前移工作约 **60～75 ms**。这只是判断优先级的量级估计，并非实测收益或严格上界；Python 准备的其余部分、不同上下文长度和 graph/decode 混合均未被该平均分离。真正可节省的时间至多是每次合格首 child 准备工作与同包剩余 DMA 等待在关键路径上的交集，还可能与下一 forward 的主线程争用。

不能把全部约 0.647 秒计数和、累计 `logical_transaction_seconds`，或整包 DMA 等待都列为该方案的收益。当前结论是暂不实施，优先验证已经定位的 FA2 统计与 compiled capture 拷贝成本。
