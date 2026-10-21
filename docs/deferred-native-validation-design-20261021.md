# 原生 attention 延迟验证备忘（2026-10-21）

10 月 22 日更新：**跨 forward 的后台验证已实现并通过正确性检查，但完整吞吐退化，继续关闭**。
新开关为 `prefill_deferred_validation`，依赖 publication pipeline、plan arm、
early MTP scoring 和 async retirement。固定两个槽复制采样 raw Q、页表、scales
及逻辑段，独立 pinned flags/event 随整帧进入原 publication 队列；CPU append
之前完成整帧验证或原参考算子回退。GPU LSE bank 仍只有一份。

144 项组合 CPU 和 21 项 GPU 检查通过，覆盖下一帧覆盖输入、两帧 Graph 重放、
FP8/GQA/部分页、原参考回退逐位一致、并发及异常边界。同源码单开关对照中，
32K 为 8,600.26→8,515.68 token/s（−0.98%），完整 256K 为
5,496.27→5,386.96 token/s（−1.99%），语义核验通过，正式回退零。
减少前台等待没有抵消快照等附加成本，不能把 CPU 等待时长直接当作净收益。
详细结果及 earlier MTP/plan arm 整合 ABBA 见
`docs/active-window-throughput-goal-20261021.md`。下文保留设计时的依据和检查清单。

## 可行范围与现有依据

对 publication pipeline 合格的普通、非末块 prefill，可研究只保存选定层的
采样 Q、小页表和评分元数据，继续读取原 native FP8 K 前缀，无须复制整个 K。
但 owner/window generation 只是身份检查，不能单独充当物理页租约。

现有代码为这种租约提供了大部分生命周期边界：

- `window_manager.py` 只包装原生 FullAttentionManager，禁止 prefix caching
  与 context parallel；普通 allocation 追加页。原生 slot mapping 按实际
  local position 定位槽位，FlashInfer `do_kv_cache_update` 只写这些槽位。
  因而严格连续的下一块只写旧上下文末尾之后的 token，包括同一物理页的未用尾部。
- `async_refresh.py:AsyncRefresh.pump` 在该 owner 的
  `publication_ready` 为真之前不启动 Query、pressure 或恢复操作；同步
  Query 分支 `bounded_runtime.py` 在重放前也调用 `drain_requests`。
- `session_worker.py:before_attention` 在 checkpoint capture/import 前先
  drain。`CaptureRuntime.finish` 在销毁 owner 前先 drain。
- 原 `scheduler.py:_request_blocks_can_be_freed` 返回 False **不足以保护强制
  preemption**：原生仍会 pop 页并按 scheduler step 延后释放。新开关明确拒绝
  native preempt；强制 prefix reset 有任何存活/退休 request 时，在修改队列前
  返回 False。正常结束仍由 worker publication 完成后的 retirement ACK 保护。
- refresh ticket 在 `window_manager.py` 中保留全部原有物理页，直到 worker
  完成并确认；不会在准备重选时提前回收旧页。
- 捕获队列现有两槽及同 owner 最多两个未发布包可约束验证积压。

这些依据只支持当前限定的原生全 attention 路径，不能推广到共享 KV、滑窗淘汰、
context parallel、draft 层或任意外部 cache 写入实现。

## 实施必须保持的安全条件

1. **取消时的 scheduler 物理页保护。** 当前 pipeline 开关不要求
   `async_archive_retirement`。该选项关闭且没有 refresh pending 时，
   `connector.py:request_finished_all_groups` 会返回不延迟释放；worker 的
   `defer_terminals` 无法独自阻止 scheduler 提前回收页。延迟验证必须要求
   async retirement，或新增等效的物理页 lease/ACK 协议。
2. **验证本身必须阻止 publication。** 原捕获包只有在验证及可能的回退 GPU
   工作、结果 D2H 全部完成后，才能标记 published。只登记后台 future、不让
   publication readiness 包含它，会让 refresh 覆写或取消释放与回退读 K 竞态。
3. **每帧独立验证状态。** `NativeAttentionStats.arm/disarm` 会复用 flags、
   plans、masses、`frame_success`。后台不能直接调用共享 bank 的 `validate()`；
   应保存该帧独立 pinned flags/event、计数快照及结果 lease，再以单一归属更新统计。
4. **Q 与页表必须真实复制。** CUDA Graph 下一次 forward 会原地覆写 persistent
   capture buffers；只保留 tensor 引用不能保护旧 Q。按原采样行复制三层 raw Q，
   复制实际 GPU 页表前缀，固定 rows、local positions、logical segments、完整
   context length、Q dtype、Q/K scales、attention scale 和 RoPE cache 所有权。
   scale 若可能原地修改，需要保存其值，不能延后读取 module 当前属性。
5. **独立回退执行接口。** 不能后台重入 `ActionRuntime._capture`，它读取 live
   batch、request indices 和当前 native bank。应由不可变 FrozenScoreInputs
   调用原评分算子，保持相同 FP8 codec、完整 causal softmax 和层平均顺序。
6. **跨 stream 依赖。** Q/页表/flags 快照必须在原 forward stream 记录完成事件；
   回退使用明确 device/stream 并等待该事件。lease 释放必须等待回退和 D2H 的
   实际 GPU 完成，不能只等待 CPU future 完成提交。
7. **入库前验证，图关闭时也一样。** `_copy_source_owners` 当前在 graph 关闭时
   直接 append KV，之后才调用 `Action.accept`。若只在 `_wait_packet/accept`
   延迟验证，失败时会出现 KV 已推进但分数未提交。必须在任何 graph reservation
   和 KV append 之前统一 resolve validation，不能依赖 graph 开关。
8. **容量及异常。** 建议两组固定快照槽，容量按 max sampled rows、Q heads、
   head dimension、页表和 flags 计算并纳入预算；采样 Q 容量为
   `layers × sampled_rows × Q_heads × head_dim × dtype_bytes`，不能固定宣称“几 MB”。
   还须预算回退 scratch 与前台模型同时运行时的峰值。容量不足走当前同步验证，
   不无界分配。失败保留源 lease 并隔离 owner，不能给出成功释放 ACK。

## 建议实施与测试顺序

首先仅支持 `reuse` 模式、整批合格普通 prefill、bounded target、async retirement、
禁用 early capture slot release；decode、末块、Query replay、verify 模式沿用同步路径。

组件验证需注入缺失 LSE/NaN 强制回退，并同时运行下一块；逐位比较冻结输入执行的
参考分数与原同步回退结果。覆盖部分物理页、非连续页表、多 owner、取消后页复用、
pressure/restore、快照池耗尽、graph 关闭、回退异常，以及下一次 graph 重放污染。
只有验证完成前不能修改历史或 ACK 释放的测试通过后，再测端到端成本。

当前没有加速承诺。保存 Q 和页表会增加 GPU 读写；即使去掉主线程等待，也可能
与前台模型争用带宽。必须使用完整请求的同源码开关对照验收净效果。

## 同一 forward 内提前提交的较小候选（已实现并完成整合 ABBA）

上述跨 forward 方案不是消除主线程等待的唯一研究方向。当前 MTP 调用顺序为：
target forward → sampling → `original_propose` → `CaptureRuntime.submit` →
Action 评分、D2H、原生验证 → 返回原生 runner 的后续提交。原生 `validate()`
的事件位于已提交的模型、MTP、三层归约、层平均和分数 D2H 之后。它的 CPU
等待包含前序 GPU 工作，不能当作验证操作本身的净成本。

匹配实验 `artifacts/active-prefill-matched-20261021-r2/` 提供了以下边界：

| 每次完整 256K 请求 | 第一次 | 第二次 |
| --- | ---: | ---: |
| 相对匹配原生形状的 model CUDA 区间增量 | 2.284 s | 2.474 s |
| 其中 114 个 2K 块的增量 | 1.620 s | 1.790 s |
| 其中 31 个部分块的增量 | 0.664 s | 0.684 s |
| Action capture 主机累计时间，含同步等待 | 20.507 s | 20.514 s |
| capture 阶段 metadata 等待 | 2.189 ms | 2.183 ms |

部分块中包含 14 个 pressure 后恢复执行的 32-token 小块，它们相对匹配形状
共多约 0.31 s。10 月 22 日核验调度和连续历史区间后更正：日志/API 中虽然
使用 replay 命名，这些是重选前被暂缓、尚未执行的新 prompt，首次执行后须
正常归档，并不是重复读取旧 KV；pressure 的 private_forward_steps 为零。
这些小块因采样间隔而没有 attention 采样，故剩余 model 区间增量不能全部
归给原生分数验证。
原生形状加权 host 提交仅 22.014 s，CUDA model 为 40.414 s，说明其提交和
等待的位置与 KVMem 不同。上表的 20.5 s 不能从 TTFT 或 model 区间直接扣除，
也不能与它们相加。当前没有单独测得验证 fence 造成的 GPU 空闲时长。

可先研究仅对整批合格的普通、非末块 prefill：在 target sampling 后、MTP
proposal 前提交完整 Action 评分、层平均、分数 D2H 和 flags D2H，记录该帧
验证事件；原 MTP proposal 提交完成后，再等待该事件、检查原有有效性并在必要
时执行原参考回退。验证和回退仍须在下一次 forward 之前结束。Q、页表和原生
bank 没有跨 forward 复用，因此无需引入前述跨帧物理页租约或 Q 快照。
此设计现已通过组件检查并纳入完整模型整合 ABBA；单开首轮 256K +1.69%，
最终六项整合相对共同基线 +4.48%，不能将单项百分比相加或相乘。

该方案必须保持 packet 私有，验证成功前不得入发布队列；MTP proposal 或
验证失败时要丢弃暂存下载、恢复候选计数，并等待相关 DMA 后才复用资源。
仍须核验原始坐标、保留相同 fallback、MTP 接受过滤和最终确认事务。private
replay、decode、混合 batch、末块及 verify 模式先保持原同步路径。错误边界
不能只依赖有限值 flags：评分的完整 GPU 事件和 KV 捕获事件都必须在任何
历史 append 前完成，graph 开启和关闭时均须成立。

另一个候选是在采样层结束后立刻执行有限值验证。实际采样层为 `[3,31,63]`，
最后一个就是 target 的最后一层；最终验证只能与该层后续投影/MLP、sampling
及 MTP 重叠，不能隐藏后续几十层。当前三层复用同一个 `row_lse` scratch，
不能直接把三个 normalizer 排在所有 segment reduction 之前，否则会覆盖前两层
的归一化结果。实现这种重排需要每层独立 scratch，或保留逐层生产/消费顺序；
还要证明图捕获与 replay 中事件的语义以及 flags 的生命周期。

只把 flags 的事件提前到 segment reduction 之前，可能漏掉后续评分 GPU 异常。
目前 graph 关闭时可先 append 再 `Action.accept`；因此不能只把等待移到 accept。
必须在统一的 append 前完成完整评分事件检查，再允许原事务提交。当前优先用
非阻塞事件量出 target/MTP/评分/捕获的结束位置，并用轨迹确认 GPU 空隙；
不根据主线程等待总和承诺加速。

## 同形状 Nsight 观察：跨步重叠的定量线索

`artifacts/prefill-nsight-matched-20261021-r1/` 的两份已完成轨迹均经过实际
StepTiming 与 NVTX 对齐：四步的 Q 都为 2,048，active context 依次为
16,384、18,432、20,480、22,528，window offset 为 0，无重选。每份轨迹都包含
68 次 paged attention，即每步 16 个 target 层及 1 个 MTP 层；KVMem 为
12 次带统计的 target 内核、52 次原 target 内核及 4 次 MTP 内核。
所有 kernel 均有 runtime launch correlation，且没有来自首个
execute NVTX 之前的 kernel launch。完整同输入 warmup 已结束并退休后才启动
采集，65.71/79.30 s 的 native/KVMem 冷准备均不在轨迹内。

GPU activity union 为 native 866.06 ms、KVMem 895.01 ms；其首末 activity
之间的空隙分别为 27.58/121.06 ms。这是单次带 profiler 的诊断区间，不能作为
非插桩吞吐或把差额全部归给 validation。首个被采集步骤中，KVMem 在前 30 ms
之后还有 30.54 ms 的空隙，而其余三步对应部分仅约 2 ms，应单列首步现象。

稳定的后三步出现了更具体的重叠差异。以下旧工作仅统计满足 runtime launch
时间早于本步 execute NVTX 开始、且 GPU activity 落在本步开始后 30 ms 内的
区间并集；没有把同时运行的内核时长相加：

| 轨迹步骤 | native 旧 GPU 工作 | KVMem 旧 GPU 工作 | native 全 GPU 空隙 | KVMem 全 GPU 空隙 |
| --- | ---: | ---: | ---: | ---: |
| 第二步，active 18,432 | 12.326 ms | 1.217 ms | 4.443 ms | 18.246 ms |
| 第三步，active 20,480 | 12.924 ms | 0.456 ms | 3.830 ms | 19.074 ms |
| 第四步，active 22,528 | 13.492 ms | 0.789 ms | 3.492 ms | 17.660 ms |

native 本步 embedding 分别在 execute 开始后 13.885、14.563、15.135 ms
执行，之前 GPU 主要执行前一步已经提交的尾部。KVMem 约少了 11–13 ms 的
这类覆盖，能解释前置阶段新增空隙的大部分。execute 是 CPU 范围，不能因
空隙位于其内部就判定为该步新增 CPU 计算，也不能把 NVTX 结束当作 GPU
结束；sampling、MTP 和捕获尾部还会继续执行。

这支持优先研究上述同一 forward 内、MTP proposal 前提交完整评分的方案。
它可能恢复已有 CPU 准备与上一轮 GPU 尾部之间的重叠，避免先承担完整的
跨帧异步回退机制。后续实测已验证该方案的小幅收益，跨帧后台校验则没有收益；
两者须分开记录。仍没有把所有差额归因于某个单一 fence；完整评分与 KV 的
append 前 GPU 错误屏障仍不可删。
