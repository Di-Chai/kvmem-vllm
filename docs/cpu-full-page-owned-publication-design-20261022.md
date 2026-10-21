# 完整历史页直接写入最终 CPU owner：实现前置设计

状态：只读设计，尚未实现或测量。不得将本文的可省复制字节换算成已获得的吞吐收益。

## 当前链路与机会

当前 CPU direct seal 已没有 whole-batch snapshot 到独立 bytes 的第二次 seal 复制。
实际链路是 `scatter_history_batch` 将 DMA 数据复制至 authority 的可复用 raw/V
staging，再由 `pack_record_pairs` 复制至最终独立 immutable bytes。SHA 与
`MemoryTier.publish_records` 使用相同 bytes，后者只发布引用。

`prefill-static-pressure-nsight-20261022-r1` 的尾包为 1984 tokens、62 条完整记录，
每条 1,114,112 bytes，共 69,074,944 bytes。pack scope 为 61.31 ms，含分配、
首次触页、复制及返回；trace 没有 CPU running/first-touch 分项，不能视作纯 memcpy
或可全部消除的成本。最终历史存储的分配和首次触页仍然必需。

## 首版资格

- 默认关闭，仅 CPU archive、原 `qw3_cuda` MeanK、原 FP8 raw-K/V 格式。
- 普通 prefill，当前接受／取消前缀已按现 publication 规则确定，完整 DMA 已完成。
- `history_begin % 32 == 0`、`accepted > 0`、`accepted % 32 == 0`，且所有目的页都是新页。
- 任一旧 partial tail、部分页、MTP 或不支持的 codec，使整个本次逻辑 append 走旧路。
- 物理合并继续逐原逻辑 2K 子块执行，不合并 attention、graph、MeanK、ACK 边界。
- 不把现有 Torch staging 永久 pin，也不缓存整份 DMA slab 作为历史 owner。

## 最小事务

1. `RequestAuthority._prepare_append` 完成原 `flush/poll` 与 resident 检查后，在
   `memory.begin_append` **之前**申请 archive reservation：共享 `n` 个 allocation
   槽、1 个 pending job、`2 * n * record_bytes` I/O 额度，并分配私有 native builders。
   此时尚无 repository handle，票据只绑定 namespace、archive generation 和确定几何。
2. 容量不足不分配 payload、不改变历史，返回原 backpressure。若 `begin_append` 返回
   `None`，所有 private builders、槽与额度必须退回；不得占住配额等待 repository。
3. `begin_append` 成功后将 spans 绑定至票据。完整 descriptor 验证后，native scatter
   从已完成 DMA 的只读源直接写 builders；布局保持完整 raw 在前、完整 V 在后。
   Builder 不提供可写 Python/NumPy/Torch buffer，避免 seal 后仍可写的旧别名。
4. Scatter 全部结束，builder 冻结为**尚未公开**的只读对象。原
   `finalize_mean_batch` 可读其 raw 的只读 NumPy view；原 token 顺序、FP32 加法、
   SIMD 选择、除法及 FP16 转换均保留。可写 index 与 payload 生命周期分开。
5. MeanK 完成后仍调用原 `memory.commit_append`。再将 reservation 绑定准确的
   namespace/repository generations，获取原 generation leases，提交原 archive executor。
   原 graph commit／logical ACK 相对 append 的顺序不变。
6. 同一 archive job 按原记录顺序做 SHA，随后 `publish_owned` 无复制移交 native
   owners，最后整批 `entry.ready=True`。提交继续使用现 `_submit_archive_job` gate。
   未 ready 的数据不得供 pressure、Query、checkpoint 或 session export 读取。

## 所有权、预算与失败

- 2× admission 不降低：62 条预留 138,149,888 bytes，64 条预留 142,606,336 bytes。
  所有 owner 的 reservation 共同受现 `_pending`、`_io_bytes`、archive slots 上限约束，
  包括尚未入 executor 的票据。CPU archive 总容量包含未公开 builders，不能另开无界池。
- 保留原 resident-record credit，直到 archive job 成功才退回。快路径的最终 owner
  不归还 `_record_buffers`；该固定 staging 池仅服务旧路径。
- DMA 源只须存活至 native scatter 返回；后续 job 不得保留 DMA 源／packet slab。
  Builder 以及 SHA read views、repository leases 则一直保持到 job 完成或安全失败清理。
- 正常容量 retry 必须发生 `begin_append` 前。现 Python `RequestMemory` 没有普通
  “撤销 append 后继续”的接口；begin 后异常沿原 fatal/quarantine，不能把已 commit
  的历史假装 rollback。后续 graph failure 也遵循现 fail-stop，不重新提交旧页。
- 取消未执行票据可退款；scatter／hash 已开始时必须等源读结束。失败 Future 保持存活
  也不能延长源 buffer 租约或导致提前复用。移交后用 native reader lease/retirement
  回收，只有真实释放完成才能复用 slot。close 必须覆盖未入队票据及 initializing builder。
- 多 owner 按现 admitted prefix 规则处理；后一个 owner 容量不足只退其私有预留，
  不重放前一个已接受的 owner，也不改变 graph reservation 顺序。

## 现有 arena 可复用范围

`MemoryArchive(native_payloads=True)` 与 `enable_history_gc()` 已分离；现
`tests/test_native_memory_storage.py` 在无 historical GC 下验证 native payload
读取、generation、read lease 和回收。因此可只启用 ownership backend，不安装
`HistoricalGcWorker`，不改 Store 选页、历史保留和 append 算法。

现 `HistoricalPayloadArena.publish_segments` 仍分配／清零／复制，不能当作 owned
transfer。需要私有 builder 和 `publish_owned`；builder 的 initializing 状态、关闭
保护及写者计数须覆盖整个存活期，不能只覆盖一次 C++ 调用。完整页每字节写满后才能
seal；未写满／失败 builder 永远不可 acquire。已 seal 的只读 view 不拥有公开读取权。

直接启用 NativeMemoryTier 会使 fallback 的 payload owner 从 bytes 变 native，
原 scatter→copy 的算法不变，但分配／回收机制改变。若要求 fallback 这层实现也不变，
需要混合 bytes/native tier，范围更大。性能归因须明确此差异；组件可在同 native owner
backend 上比较 copy 与 owned transfer，最终仍需与当前完整配置比较。

## 文件面与验收前置

最小文件面：`authority.py`、`storage.py`、`memory_storage.py`、
`native_memory_storage.py`、`historical_payload.hpp/.cpp`、`bindings.cpp`，再加默认关闭
选项及测试。不需要修改 `third_party/qw3` 或 GPU attention 算术。

实施前必须明确 reservation 的 reserve/bind/submit/abort 一次性接口及 native builder
状态机，并覆盖：全批预验证、第二 owner 准入失败、repo prepare 返回空、scatter/MeanK/
hash/submit/publish 失败、后子块失败、保留失败 Future、迟取消、读者跨 retirement、
generation 重用、关闭 initializing builder、完整 bytes/SHA/MeanK、partial fallback。
性能必须计入分配、触页、scatter、MeanK、hash、publication、读取及最终释放，不能只报
被删除的 pack scope 或把回收工作移出计时窗口。
