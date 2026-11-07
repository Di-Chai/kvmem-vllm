# 动态分配第二阶段：会话换出、SSD 暂停快照与调度优化

本阶段延续[第一阶段](dynamic-memory-implementation-20261106.md)。K/G、选块规则、mandatory span、Query replay 和刷新时机均沿用原路径。修改的是物理资源准入、迁移及请求轮换，不是模型的记忆策略。默认仍为静态 GPU 准入。

后续在同一运行时上补充的 [16 路真实文档混合实验](dynamic-memory-mixed16-20261106.md)记录了驻留数变化、短请求完成、实际 SSD 卸载和输出等价性的限制；与本文受控输入验收分开判定。

## 会话请求的暂停与恢复

`session_swap.py` 在调度器和 worker 两端保存并校验 `(cache_salt, session_id, generation)`，以及原 request、authority 和 P/M checkpoint 的对象归属。导入未完成、capture/publication 未确认、P/M 下载未完成或退休中的请求不能成为换出对象。

会话暂停期间继续持有原命名空间和 native slot。同会话后续请求按既有规则等待，不会因为当前请求被换出而取得其 checkpoint。会话池全部被活动请求占用时，动态模式等待一个可用所有者，避免产生没有受控会话归属的临时请求。

正常路径为 `capture → paused → restore → drop → runnable`。取消一个已暂停的会话请求时，也必须先恢复其精确 GPU 状态，再完成 drop 确认，最后执行原来的终结、P/M 保存与回收。无会话请求仍可以在暂停时直接丢弃快照。所有路径都保留原 sampler、RNG、token 历史和 slot；不会以重新 prefill 代替恢复。

本阶段支持单 rank、eager、同步调度及同步会话导入的 native MTP 配置。异步恢复/刷新、图模式、APC、rolling 和原本拒绝的未验证组合继续受配置检查限制。

## 暂停快照的 CPU / SSD 分层

新增 `request_checkpoint_ssd_bytes`，默认 0，保留第一阶段 CPU-only 行为。非零时：

| 配置 | 计量范围 |
| --- | --- |
| `request_checkpoint_bytes` | 暂停快照 CPU/pinned 驻留及在途暂存额度；至少容纳一个最坏完整快照 |
| `request_checkpoint_ssd_bytes` | 全部暂停快照的逻辑后备容量，包含为保证恢复推进保留的余量 |
| `cpu_read_cache_bytes` / `nvme_max_bytes` | 原有历史记录的 CPU / SSD 容量，与暂停快照独立 |
| `session_checkpoint_bytes` | 原有 P/M 会话 checkpoint 池，与暂停快照独立 |

[`serve-qwen-mtp-dynamic-ssd.json`](../examples/serve-qwen-mtp-dynamic-ssd.json) 给出本次模型页几何下的示例：512 MiB 暂存、4 GiB 暂停快照 SSD 额度、2 GiB P/M 池。其他模型应依启动校验设置容量；这些数字不是进程总 RAM 上限。

`checkpoint_storage.py` 在捕获前为完整快照保留 CPU 额度。达到高水位 85% 后按冷记录写回，目标低水位 70%；新增捕获或恢复无法装入时，先写回可回收快照，再继续。一个 I/O lane 串行处理磁盘作业，完整请求的读回也必须先取得有界 CPU 暂存。不会无界加载所有暂停请求。

写盘保留完整 CPU 源和额度直到 I/O 确认，读回直接填入目标 slab 并校验长度与 SHA-256。只有校验成功才安装 GPU 页表并开始 H2D。读失败保留磁盘源；写失败保留 CPU 源；取消和 drop 等待在途 I/O/DMA。实现会重试被中断的系统调用；回归测试覆盖短写、截断读取、checksum 错误和陈旧身份。文件按需创建且立即 unlink，进程退出后清理；这是进程内换页后备，不提供进程重启恢复。

本机实验目录位于 NVMe 的 ext4 文件系统。文件 I/O 没有绕过 OS page cache，计数表示文件接口完成的 payload 字节，不等同于设备的实际读写流量或冷盘带宽；本次未宣称测得裸 SSD 性能。

原有闲置 P/M checkpoint 池没有在本阶段改为 SSD 存储；进入 SSD 的是活动请求暂停时的完整精确快照。CPU pinned 分配器缓存、索引和 OS 文件缓存也不在上述 payload 上限内。

## GPU 吞吐优化

第一阶段固定 1.4 GB GPU KV 池的 C4 合并输出吞吐比静态准入低约 1.55%。每轮动态模式有 6 次捕获和 6 次恢复，按 native backing page 计的双向传输上界约 5.40 GB；实际传输 payload 小于该保守上界。轨迹中既有仅推进少量自身 prefill 就再次被换出的请求，也有接近结束的 decode 被换出。原始数据不能单独证明 GPU 同步或空轮询贡献了多少时间。

本阶段做了以下修改：

- 轮换时钟只统计已确认的 accepted-history 前进。空轮询、延迟 ACK 和导入/恢复的既有历史不计为计算进展。
- 恢复后的最短驻留间隔按该请求自己的实际前进次数计算，保留恢复后首个完整操作的临时页保护。
- 有空闲执行能力时允许等待者准入；正在推进的驻留请求不会仅因新请求等待而立即被换出。等待者超过有界进展轮数或所有驻留者都被资源阻塞时，触发推进。
- 单个换出对象足以满足缺口时优先选择足够且较小的对象，并尽量让接近结束的 decode 自然释放。必须换出多个对象时优先减少迁移次数。
- 公平性轮换到期时，如果可继续执行的 decode 请求即将在 32 个以内输出 token 后结束，且释放的页面足够完整恢复目标，可以额外等待至多 32 个实际进展轮次。默认硬上限为 128+32；所有驻留者阻塞或会话取消时不等待。预测释放的页面绝不提前算成可分配额度。
- 已经符合上述条件的请求结束、但仍等待原生页面释放确认时，继续沿用同一次等待及原截止点。每次检查都要求 scheduler 和资源预留中的 request 对象一致且仍实际持页；请求 ID 被复用、页面已释放或所有权变化时不再等待。这样避免终结确认间隙触发多余换出。

`request_swaps` 增加实际前进轮数、空轮询次数、驻留期覆盖、等待者延期、迁移原因和传输字节上界，便于区分请求并发、实际 GPU 驻留及迁移成本。仍然只有一个迁移指令在途，capture 前仍保留 GPU 同步，尚未声称消除全部换出成本。

## 验证与复现

`scripts/benchmark_dynamic_memory.py` 新增 `baseline` 臂和 `--baseline-runtime`，从上一阶段冻结源码启动基线；当前代码也在运行前冻结，两个来源分别保存 SHA-256 清单。`ssd` 臂单独启用暂停快照 SSD，历史后端、K/G、精度和刷新设置保持相同。`--session-stress` 在最后一个动态臂完成性能计时后运行会话资格测试。

```bash
PYTHONPATH=src:scripts python scripts/benchmark_dynamic_memory.py \
  --profile /absolute/path/to/fixed-gpu-profile.json \
  --baseline-runtime /absolute/path/to/previous/runtime \
  --output /absolute/path/to/new-results \
  --arms baseline incremental incremental baseline --levels 1 2 4
```

正确性判据是完整输出 token IDs 和文本相同；资源判据是已完成请求的 GPU 页、临时 checkpoint 及 I/O 所有权回收。仍有有效闲置会话时允许 P/M 和历史保留；显式释放全部会话后要求这些资源也归零。观察到 SSD 配置并不能算分层验证，必须记录实际快照写入和读回。

`archive_resources.json.request_checkpoints` 独立报告 CPU 驻留/承诺/峰值、SSD 占用/读写、I/O 等待和逻辑账本，不能与历史存储同名计数混合。性能采样继续排除模型启动和最终回收等待，包含 HTTP 排队、prefill 与 decode。

## 第一轮优化的实测与修正依据

`scheduler-abba-r1` 使用上一阶段冻结源码作 baseline，与第一轮新调度按 baseline → incremental → incremental → baseline 顺序同机测试。每臂预热后运行 1/2/4 并发；K/G 均为 2,048，GPU KV 池 1.4 GB，暂停快照仍为 CPU-only 2 GiB。四臂输出逐 token 一致，同模式重复的输出稳定，资源回收通过。

| C4 指标 | 上一阶段 | 第一轮调度优化 |
| --- | ---: | ---: |
| 合并输出吞吐 token/s | 10.107 | 9.737 |
| 平均完整响应时间跨度 s | 50.660 | 52.583 |
| 每轮 capture / restore 次数 | 6.5 / 6.5 | 4 / 4 |
| 双向传输 backing 字节上界 GB | 5.791 | 3.750 |
| 每轮模型批次数 | 356.5 | 382 |
| 累计模型 GPU 时间 s | 46.159 | 48.634 |

这轮吞吐下降 3.66%，不能把迁移减少称为吞吐收益。相同总 Query token 和 request-turn 数下，prefill/decode 混合批次减少，更多分开的批次增加了约 2.48 秒模型 GPU 时间。平均单请求完整延迟下降，但最后一个请求结束更晚，二者不矛盾。该结果促成上文的有界完成等待：两次公平轮换时，另一个驻留请求仅余约 30 和 4 个 token；优先等待其自然释放可以避免额外迁移及批次拆分，后续独立对照如下。

`scheduler-abba-r2` 按 incremental → baseline → baseline → incremental 反向顺序重复 1/2/4 并发，启用有界完成等待，尚未加入终结 ACK 间隙保护。全部输出和资源检查通过。C4 合并吞吐为 9.919 → 10.130 token/s（+2.13%），平均每轮暂停 6 → 3 次，保守双向传输上界 5.460 → 2.757 GB。C1 −0.30%、C2 +0.11%，两者没有发生换出，不把这种幅度称为确定的改善或退化。

| r2 C4 顺序 | 输出 token/s | 模型批次数 | capture / restore |
| --- | ---: | ---: | ---: |
| 新调度 1 | 10.108 | 370 | 3 / 3 |
| 上一阶段 1 | 9.607 | 385 | 7 / 7 |
| 上一阶段 2 | 10.251 | 358 | 5 / 5 |
| 新调度 2 | 10.152 | 370 | 3 / 3 |

两种调度每轮均处理 33,280 个模型 Query token，但基线重复的批次数和迁移次数有差异。这是实质性调度波动，不能仅看均值或把全部时间变化归因于复制。合并吞吐采用全部输出 token 除以累计 HTTP 计时；模型 GPU 时间为 CUDA event 间隔，可能包含提交空隙，不是独立 kernel 时间之和。

## 最终调度的吞吐对照

`scheduler-abba-r3` 冻结最终源码，按 baseline → incremental → incremental → baseline 顺序验证 C4。每臂先做一次单请求预热；两种实现都关闭全部恢复诊断，使用相同的 1.4 GB GPU 池和 2 GiB CPU-only 暂停快照。此处 baseline 是上一阶段动态实现，不是静态准入。四臂完整 token IDs、文本和资源回收检查全部通过。

| 最终 C4 指标 | 上一阶段动态实现 | 最终动态实现 |
| --- | ---: | ---: |
| 两次独立输出吞吐 token/s | 9.591 / 9.830 | 10.235 / 10.102 |
| 合并输出吞吐 token/s | 9.709 | 10.168 |
| 平均完整响应时间跨度 s | 52.734 | 50.354 |
| 合并请求 TTFT 中位数 s | 33.636 | 32.460 |
| 平均单请求完整延迟 s | 45.975 | 42.526 |
| 每轮 capture / restore 次数 | 6 / 6 | 3 / 3 |
| 双向 backing 传输保守上界 GB | 5.460 | 2.757 |
| 平均模型 GPU event 时间 s | 48.224 | 46.591 |

本轮合并吞吐改善 **4.73%**。两个最终调度样本都在一次实际前进轮中命中终结确认间隙保护，避免当时立刻换出另一个 owner；后续公平期限到达时仍需第三次迁移，因此总数是 3 对，而非 2 对。不同轮次还存在批次、排队、模型提交及同步变化，不能将总收益全部归因于更少的 DMA，也不能用 r2/r3 的跨轮差值单独量化确认间隙保护的收益。

本轮仅有两次同模式重复，且采用受控输入；4.73% 是这组条件下的结果，不是所有模型或负载的加速保证。C1/C2 的无换出路径由 r2 和最终主机回归覆盖。第一阶段静态吞吐 10.220 token/s 仅作历史背景，不与本轮构造未经匹配的动态/静态性能结论。逐臂批次、迁移原因、时间和源码摘要保存在各轮的 `swap-policy-attribution.json`、`analysis.json` 与 `source.json`。

真实 CUDA 字节测试已通过 3 项，包括直接 CPU 快照、完整 CUDA→SSD→CUDA 往返和恢复核验器的真实 GPU 检查；SSD 路径在原页被复用后恢复到新页，比较目标/MTP FP8、GDN 状态和 accepted counter 的原始字节。运行命令及 JUnit 保存在 `checkpoint-ssd-gpu.xml` 对应实验目录。

## 会话输出差异的独立对照

首轮 `ssd-session-r1` 的无会话测试和暂停断连测试通过；C4 的完整输出也与 CPU checkpoint 基线一致，暂停快照实际写入及读回各 1,735,753,744 bytes。会话代码生成输入则没有通过逐 token 比较，原失败状态与全部输出均保留。

该失败中，一个请求在输出 token 索引 24（从零计）已发生差异，而它自己的首次换出在约 241 个输出 token 后。随后 `session-no-swap-control-r2` 在同一输入、同一 K/G 和同一代码上把 GPU 池增至 3 GB，使捕获、暂停、恢复和 drop 全部为零：两次独立运行均与原参考完全一致，但四路并发的首个不同 token 索引分别为 2、47、47、47。因此已确认该输入存在不依赖请求换出的并发输出差异；这不单独证明其具体数值原因，也不能把该输入的换出正确性判为通过。

后续稳定负载使用 `--session-workload repeat`，仍要求完整 token IDs 一致，以及真实 P/M 命中、暂停取消、同会话等待、SSD 读写和全部释放。它不是对原失败输入的重新判分。自然代码生成输入可通过 `--session-workload code` 重现，保持原默认值和严格比较。

新增顶层 launcher 设置 `verify_request_swap=true`，显式生成 `KVMEM_VERIFY_REQUEST_SWAP=1`，在每次实际恢复后、释放 CPU 来源之前核验全部 snapshot 字节。受控 launcher 会清理继承的 `KVMEM_*` 环境变量，因此验收必须使用该配置入口，并检查 worker 的 `verification_enabled` 及逐次成功记录。它使用最多 1 MiB 的额外诊断暂存，覆盖所有记录的 KV、GDN/MTP 和 accepted-counter 字段，并写入 `request_swap_verified.jsonl`。错误保留来源和恢复所有权，不发成功确认。这个开关默认关闭，不能用于正式吞吐计时。它验证被捕获字段的字节恢复，不能代替 sampler/slot 保留规则或任意提示词的端到端稳定性。

`verify_session_restore=true` 与 `verify_session_restore_digest=true` 则验证 P/M 导入；两者的证据与活动请求换出的字节证据分开记录。原失败会话测试中已完成的 3 次 P/M 导入均通过 digest 核验，但这没有改变端到端输出测试的失败结果。

## 闲置 SSD 回收缺口

无换出对照还暴露了第一阶段历史后端的回收问题：最后一个会话释放后，逻辑记录、会话池和 GPU 页已归零，但 601,620,480 bytes 的历史 SSD extent 留在延迟回收队列。原实现只在下一次 I/O、spill 或 close 时处理这个队列，因此服务保持闲置时并不归零。这是实际物理回收缺口，不能用逻辑计数为零代替回收验证，原现场保留在 `session-no-swap-control-r2`。

现已改为一个合并的后台回收任务，最后一个请求结束后也会主动推进；释放与任务完成交错时补交一次任务，不按每条记录堆积后台任务。文件关闭完成前保留物理描述符和容量计量，关闭失败保留所有权并阻止成功确认。普通终结、异步退休、会话 eviction 和仅剩闲置会话的 shutdown 均等待相应完成屏障，再发布最终统计。专项测试覆盖慢关闭、失败关闭、提交失败、补交竞争、rollback 和无新请求的最终回收。

物理回收屏障也作用于静态 GPU 准入搭配 tiered 历史的请求和会话：终结所有权确认会等待已退休磁盘 extent 回收及空 segment 文件关闭，而非仅等待逻辑记录释放。segment 仍有有效记录时，回收的槽位供后续复用，文件在最后一条记录释放后关闭。这是有意修正的清理时序；不会修改 K/G 或模型刷新时机。

最终联合主机回归为 1,044 项通过，3 项 CUDA 测试在主机命令中跳过后另行执行，全部通过。控制器测试包含 96 个混合所有者压力组合，覆盖不同页池、MTP、延迟迁移确认、延迟终结释放、迟到请求和取消。`final-host-regressions.xml` 保存最终主机结果；`policy-source-audit.json` 保存与上一阶段冻结源码的逐文件摘要对照，选块、Query、attention、刷新和 session ledger 实现不在修改列表中。

## 完整 SSD / 会话验收结果

`ssd-session-r4/qualification.json` 与 `01-ssd/session-stress/byte-audit.json` 均为 `passed`，覆盖与最终吞吐对照相同的运行时源码。此前 r3 也已通过，r4 在补上终结确认间隙保护后重新完成全流程。配置为 K/G 2,048、1.4 GB GPU KV 池、512 MiB 暂停快照 CPU 暂存、4 GiB 暂停快照 SSD 额度；P/M 导入和活动请求恢复的诊断都实际启用。

| 检查 | 实测结果 |
| --- | --- |
| 无会话 1/4 并发完整输出 | 与 CPU checkpoint 基线逐 token、逐文本一致 |
| 稳定输入的会话流程 | 独立参考、四路并发、M 延续、P 回退、同会话等待、暂停取消和后续新请求全部通过严格比较 |
| P/M 导入 | 9 次导入对应 9 条成功 digest |
| 活动请求恢复 | 12 次全部通过字节核验；10 次来自 SSD、2 次来自 CPU；每次覆盖 131 个实际模型字段 |
| 已核验恢复总字节数 | 5,152,112,688 bytes；无缺失、重复或失败确认 |
| 会话阶段独立 SSD 流量 | 4 次写回、4 次读回，每方向 1,680,605,200 bytes |
| 暂停快照 CPU payload 峰值 | 453.28 MiB，CPU 承诺/占用计费峰值 473.35 MiB，均低于 512 MiB |
| 暂停快照已完成写回的 SSD 占用峰值 | 1,315,602,444 bytes，低于 4 GiB 逻辑额度；在途写另计 `ssd_pending_write_bytes` |
| 最终 GPU 页 | 请求占用 0；24 页可用，另有 1 个原生 null page |
| 最终 CPU / SSD / 会话 | 历史及快照逻辑占用、SSD extent/file、退休队列、I/O、P/M 池和命名空间全部归零 |

暂停会话取消的回收顺序是 `capture → restore → drop → FINISHED_ABORTED 终结记录`；请求的取消状态会先标记，物理回收延后确认。同会话后续请求的导入开始时间晚于前一个请求的退休确认。最后的释放无需提交额外请求来推动 SSD 回收，修复了上面的闲置缺口。

本轮是单臂功能验收，benchmark 通用分析器因缺少双模式重复而正确保留 `needs_review`；独立功能审计不会把它改成吞吐 ABBA 通过。诊断产生的额外复制/同步时间也不用于性能结论。早先 `ssd-session-r2` 因继承的诊断环境变量被 launcher 清理而被主动中断，原结果和中断原因保留；r3/r4 使用新增显式配置并逐一核对了 worker 状态和成功记录。

最终证据索引为 `artifacts/dynamic-memory-phase2-20261106/final-validation.json`，由同目录 `summarize-final.py` 重建。它核对最终主机/CUDA 测试、四臂性能、SSD 会话字节与生命周期资格、230 个未修改文件摘要，以及性能与功能测试冻结源码和当前工作区的一致性。原始失败、主动中断和无换出对照均保留；索引的通过范围不包含任意提示词并发时的逐 token 确定性。验证进程已退出，GPU 上没有遗留的模型进程。

复现 SSD 功能验收时，使用与示例相同资源配置且将顶层 `verify_request_swap`、`verify_session_restore`、`verify_session_restore_digest` 都设为 `true` 的本地模型 profile：

```bash
PYTHONPATH=src:scripts python scripts/benchmark_dynamic_memory.py \
  --profile /absolute/path/to/diagnostic-ssd-profile.json \
  --output /absolute/path/to/new-ssd-results \
  --arms ssd --levels 1 4 --cancel-stress --session-stress \
  --session-workload repeat
```

单臂命令因缺少性能比较预期返回 `needs_review`；功能结果看 `01-ssd/cancel-stress/result.json`、`01-ssd/session-stress/result.json`，恢复证据可由已保存的 `audit_session_swap_bytes.py --arm-dir /absolute/path/to/new-ssd-results/01-ssd` 独立核对。吞吐比较则使用前文四臂命令并关闭三个诊断开关。
