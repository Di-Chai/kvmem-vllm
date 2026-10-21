# 256K 单请求 prefill：瓶颈定位与批量内容索引

本轮保持 1K chunk、评分规则、K/G 预算不变，检查原生 attention、CPU
历史发布和诊断写出的成本。工作基线为 main `8560331` 加当前工作区已有修改；
各轮实际运行的是冻结副本，以 `runtime-proof.json` 为准。没有启动 DSH，
没有恢复 MAB 或完整 RULER。

**结论：确认了原生 attention 的主要 GPU 成本，并降低了 CPU 组件成本，
但这轮没有证明完整请求的吞吐提升。归档单次复制开关继续默认关闭。**

## 测量范围

- 单张 RTX PRO 6000 Blackwell Server Edition，同一条完整 262,045-token
  RULER `niah_single_1-0000` 输入；这是性能诊断样本，不是完整准确率评测。
- NVFP4 Qwen、FP8 KV、编译和 CUDA Graph、MTP(k=4)，实际并发 1。
- K=36,864，G=16,384，chunk=1,024，batch 上限 16,424；CPU 历史 100 GB，
  不用 SSD，关闭 prefix caching；sink=41 个 32-token 页，recent=0。
- pressure 使用累计 prefill attention；Query 使用 V2 multi 图传播、alpha=0.2；
  guided=0。四项 attention/图效率优化保持开启，native QK reuse 关闭。
- 首 token 延迟由客户端测量；有效输入吞吐为 262,045 / TTFT，计入预填充、
  归档等待、重选和首个输出，但不计模型冷启动。输出自然结束，共 111 token。

实验目录：[`artifacts/prefill-bottleneck-20261021-r1`](../artifacts/prefill-bottleneck-20261021-r1)。
原始请求、响应、选页、运行配置、编译输入和清理证明均保留。

## Profiling 发现

Nsight 的一个中段窗口记录了 **19.953 秒**，kernel/copy/memset 时间并集
为 **10.935 秒（54.8%）**。这是带 profiling 的记录活动占比，不是硬件利用率，
也不能直接当作无 profiling 时的 GPU 空闲率。

| GPU 工作 | 累计 kernel 时间 | 调用次数 |
| --- | ---: | ---: |
| 原生 `BatchPrefillWithPagedKVCacheKernel` | 6.786 s | 959 |
| 旁路评分 `_segment_stats` | 0.254 s | 162 |
| 全部 kernel | 10.835 s | — |

原生 prefill attention 约占累计 kernel 时间的 **62.6%**，是最大的 GPU 热点。
完整无 profiling 基线中，13 次 pressure 的 suspended-to-ready 总计约
3.95～4.33 秒，最终 Query 约 1.62～1.81 秒。不能把所有额外耗时都算成重选。

每条输入还发布约 **9.127 GB** CPU 历史。后台发布的累计 claim 时间为
12.38～14.72 秒；打包、校验和二次复制也有明显成本。这些区间互相嵌套，
并与 GPU 执行重叠，不能相加作为独占延迟。

主线程采样中还出现大量 `public_work.poll` / `Path.replace`。检查发现零 token
维护轮次被当成完全空闲，导致每次强制写进度；一次请求写了 518 条快照。
这是实际的频率控制问题，但单独修正它的实验**没有证明端到端提速**。
Python 栈采样同时包含阻塞和重新获取 GIL 的时间，不能把采样频数当作独占
文件系统耗时。

本次 Nsight 使用 `--delay=100 --duration=20`。20 秒采样完成后，尚在运行的
控制器启动 `nvidia-smi` 时继承了已结束的 Nsight 注入环境，导致该轮中断。
因此 `profile/` **只用于中段诊断，不进入吞吐或准确率对照**；进程已清理，
另存有 `external-cleanup-verification.json`。之后所有性能轮次均未启用 profiler。

## 实现的改动

1. `public_work_idle()` 区分仍在等待发布/重选的 live owner 与真正结束。
   维护轮次继续无阻塞查询完成状态，正常按一秒发布；全部请求结束时仍立即
   flush 最后不足一秒的计数。最终 submitted/completed 精确对账。
2. 新增 C++ `finalize_mean_batch()`，一次校验整批已封页，再统一释放 GIL
   完成内容索引累加。`authority.py` 按当前已接受的 append 调用它，避免每页
   Python/C++ 往返；没有完整页的 decode 不调用。原来的单页 API 保留，旧版
   扩展自动使用原路径。
3. 对已有、默认关闭的 `cpu_direct_record_seal` 路径增加 `pack_record_pairs()`，
   批量生成独立 immutable bytes，减少逐页 GIL 往返。单次复制仍需显式开启
   该开关；没有修改默认归档配置、预算预留、校验和或发布事务。

批量索引保持原来的 token 顺序 FP32 累加、FP8/BF16 转换和后续 FP16 mean
   转换。没有更改 attention 采样、图构建、选页策略、chunk 或上下文范围。

CPU 组件交错测试（32 页，16 个索引层，5 次预热、30 次测量）中，原路径
中位数分别为 **9.296/9.307 ms**，批量为 **7.195/7.196 ms**，耗时约下降
**22.6%**。这不包含模型、归档发布和线程竞争，不能外推为整轮加速。

另一个 35.65 MB copy/hash 组件测试中，原两次复制中位数为
58.36/58.56 ms，已有单次复制为 38.68/40.83 ms，新批量单次复制为
38.63/39.91 ms。主要的组件差异来自避免第二次复制；批量化本身的孤立收益
很小。真实请求中，逐页/批量单次复制的累计 pack 时间从 10.92 降到 8.41 秒，
同时 hash 时间从 10.36 降到 9.13 秒，仍未缩短请求耗时。

## 完整请求对照

最终数据见 `analysis.json`。`reference-3` 与 `batch-index-2` 使用相同的新编译
扩展，区别仅在三个 Python 调用/诊断文件；前者仍调用原单页 API。
`batch-index-1` 是首轮候选，尚未跳过空的索引批次，保持单独记录。

| 轮次 / 路径 | 冷准备 s | 首 token s | 输入 token/s | 进度快照 |
| --- | ---: | ---: | ---: | ---: |
| reference-1，原路径 | 80.17 | 75.863 | 3,454 | 518 |
| idle-fix-1，仅修正诊断频率 | 75.14 | 80.348 | 3,261 | 74 |
| reference-2，原路径复测 | 76.14 | 86.723 | 3,022 | 518 |
| batch-index-1，诊断修正＋批量索引 | 77.16 | 73.587 | 3,561 | 73 |
| reference-3，新扩展、原调用路径 | 76.15 | 75.003 | 3,494 | 518 |
| batch-index-2，同扩展、最终批量索引实现 | 75.14 | 75.812 | 3,456 | 73 |
| direct-seal-reference，再启用已有逐页单次复制 | 77.14 | 77.805 | 3,368 | 74 |
| direct-seal-batched-1，改为批量单次复制 | 77.16 | 79.210 | 3,308 | 74 |

前三轮基线的波动明显，不能选最慢基线与最快候选计算收益。同编译库的
最终对照为 **75.003 → 75.812 秒**，有效输入吞吐约 **−1.1%**；归档单次复制
内部对照为 **77.805 → 79.210 秒**，吞吐约 **−1.8%**。这些单次差异也不够
证明稳定退化，但明确不足以宣称端到端提速。

启用直接归档的两轮配置仅增加 `cpu_direct_record_seal=true`，单独放在
`analysis.json` 的 `direct_archive_comparison`，不混成相同配置的重复样本。
两次复制的 9.127 GB 第二份搬运确已消除；它主要发生在可与 GPU 重叠的后台，
削减累计 CPU 耗时不能等同于降低首 token 延迟。

所有正式轮次实际并发为 1、输出 111 token，MTP 接受/丢弃边界一致。
最终计数均为 272 次 prefill forward、262,045 个 prefill token，另有
31 次 decode forward、155 个非 prefill 验证位置；后者不是接受输出数。
5 秒采样的 GPU 显存峰值约 70.4～70.9 GiB；该采样会漏掉瞬时峰值，完整
资源数据和各轮冷启动分开保留在报告 JSON 中。

## 正确性与限制

- BF16/FP8 内容索引、最终 FP16 mean 逐位一致；覆盖所有有限 FP8 编码、
  非紧密 token/layer stride、多 owner、部分页、MTP 接受前缀和旧扩展回退。
- C++ 在全部页验证成功前不写入任何 sum；后续页无效、只读输出、错误
  stride 等均验证不会半更新前面的页。
- 批量归档验证独立所有权、源缓冲区在复制完成前保活、完成后释放、预算
  耗尽、等待期间关闭/取消，以及 pack/hash/publication 失败后的完整回滚。
- 维护轮次限频、多 owner 未全部退休、结束前最后一小段计数均有测试。
- 每轮对比全部 14 次 pressure/Query 选页及 mandatory 内容、输出 token、
  累计 attention 统计、最终公开 forward 计数。图汇总字段另外记录精确差异，
  浮点容差为 rtol=1e-6、atol=1e-6，离散字段严格相同；未导出每条图边。
- 使用相同输入和自然结束轨迹，完整请求都通过协议、资源释放和执行路径审核。
  只有一个样本，不能作为完整 memory benchmark 的准确率结论，也不代表八并发。

本轮最终相关检查共 **195 项通过**。编译输入和运行扩展保留在数据盘，工作区扩展已更新。
原工作区其他修改保持原样；没有提交或推送。

36K 是重选后的历史预算。此前已核验，这个 K/G 配置运行时 attention 可见
上下文会增长到约 53K，按 token 加权的平均可见 key 约 41.2K；原生完整
32K causal 输入的平均可见 key 约 16.4K。两者的 attention 工作量、chunk
以及 CPU 归档任务不同，不能用原生 32K 的约 10.6K token/s 直接作为同成本
预期。此前对照见 [原生上下文测试](native-context-prefill-20261021.md)。

复现入口：`scripts/analyze_prefill_bottleneck.py`、`scripts/benchmark_mean_index.py`、
`scripts/benchmark_record_seal.py`，实验目录中的 `prepare_arm.py`、冻结 runtime、
配置及两次 `build-proof*.json`。性能轮次均完成清理，最终 GPU 无计算进程。
