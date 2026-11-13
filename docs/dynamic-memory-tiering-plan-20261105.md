# KVMem 动态容量与分层存储实现方案

日期：2026年11月5日。现状分析对应当时工作区，包括未提交改动；基础提交为 `20c292bdbf0555a021379105bd273525416cc552`。本文保留原实现及验收方案。2026年11月6日开始的实现范围、配置与验证结果见[动态分配实现记录](dynamic-memory-implementation-20261106.md)。

可以实现按实际占用容纳更多请求，并在压力下进行 GPU、CPU、SSD 之间的迁移。当前可明确识别的一类闲置来自**请求未来容量的保守预留**，不能简单理解为每个请求已经分配了完整的 K、G 和 CPU payload。改进应保留共享内存池、按需分配物理块，将整轮容量保证替换为短期增量额度，并以可恢复的暂停和分层存储保证安全。

第一版保持 K/G 的语义、历史选块、刷新时机、KV 精度和完整输入输出要求。压力下提前刷新或缩小 K 是另一种模型策略，需要独立验证质量，不能混入存储优化的收益。

## 目前哪些资源提前分配

| 资源 | 物理分配行为 | 预留及限制 | 直接影响 |
| --- | --- | --- | --- |
| 原生 GPU KV 池 | 启动时建立共享 backing，运行中从池里分配和归还块 | 总池受 GPU 预算约束 | 归还请求的块通常只增加池内空闲页，显存不会因此退回给操作系统 |
| K，即 `select_budget` | 限制选入 GPU 的历史工作窗口，不是独立的一整块 malloc | 长请求在 bounded 路径会按最多 K+G 计算生命周期额度 | 选入历史少于 K 时，也可能扣留较大未来额度 |
| G，即 `gen_budget` | 新 token 随执行按块分配 | `ReservedKVCacheManager` 为未来增长保留信用；刷新归还物理页后，信用仍保留 | 实际 G 未满，也可能不能准入新请求 |
| CPU 历史 payload | `MemoryTier` 写入不可变记录时逐步分配；native payload 也在发布时建立 | `HistoryAdmission` 预留 prompt 加最大输出；历史 GC 路径也预留整个新 turn 及在途包络 | 已预留但尚未生成的历史占用准入额度 |
| CPU 进度保证 | 信用，不等于实际 payload | 非历史 GC 的 `HistoryAdmission` 可额外为一个 session 长到逻辑上限保留空间；空闲 session 还保留高水位信用 | 逻辑上限很大时，预留与实际占用差距可很大 |
| 仓库元数据 | repository 槽表、ready 队列和 native payload 描述符有容量规模的预分配 | `repository_blocks` 是另一道独立限制 | SSD 有空间也不代表逻辑记录槽还有空间 |
| 传输、检查点和计算暂存 | 同时存在固定池、首次使用分配和按检查点分配 | pinned buffer、CUDA Graph 地址、Query 缓存、特征缓存有独立约束 | 只计算历史 KV 会低估实际 RAM/VRAM |
| 现有 NVMe 后端 | `nvme_preallocate=true` 时尝试整 arena 的 `posix_fallocate`；否则随写入增长 | 固定 arena 槽容量与可选字节配额 | 不是 CPU 后端满后自动启动的下一级 |
| NVMe 的 CPU read cache | 按访问填充，并按总量及 namespace 配额淘汰 | 它是可丢弃副本，NVMe 已有权威记录 | 已有一部分动态缓存能力，可复用其读路径 |

GPU 全局池预分配与请求块的动态分配可以同时成立。上游也说明 GPU cache 预分配，以及普通 V1 抢占默认采用重算；本项目有额外状态和生命周期限制，不能直接照搬其抢占流程。[vLLM 官方说明](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

### 当前代码依据

| 位置 | 已确认行为 |
| --- | --- |
| [capacity.py](../src/vllm_kvmem/serving/capacity.py)，`_demand`、`_allocate_slots`、`begin_step` | attention 需求按 `min(prompt + max_tokens + speculative/guided 余量, K+G)` 再按各组原生页大小取整；另加 recurrent 和 draft 固定需求。其他请求的未使用增长额度通过 `reserved_blocks` 保留 |
| [window_manager.py](../src/vllm_kvmem/serving/window_manager.py)，`allocate_new_blocks`、`commit_refresh` | 委托原生池分配实际窗口页；仅在完成 fence 后归还刷新淘汰的页 |
| [logical_capacity.py](../src/vllm_kvmem/serving/logical_capacity.py)，`wrap_planner` | 在超长上下文或特定 recall 路径中，启动时要求 `max_num_seqs × 每请求完整窗口及状态 + null page` 能装下；普通路径不都经过这项检查 |
| [coordination.py](../src/vllm_kvmem/serving/coordination.py)，`demand`、`_usage`、`consume` | 完整 turn 预留、单 trajectory 进度预留，以及 idle 后保留高水位 |
| [historical_admission.py](../src/vllm_kvmem/serving/historical_admission.py)，`preview`、`settle` | 已分配历史加整轮 incoming、tail/checkpoint/在途余量；只认 worker 完成确认，不提前使用待 GC 空间 |
| [storage.py](../src/vllm_kvmem/core/storage.py)，`make_shared_archive` | 进程共享 arena；CPU 和 NVMe 是互斥后端；CPU 总预算还要扣除 Query 和特征预留 |
| [memory_storage.py](../src/vllm_kvmem/core/memory_storage.py)、[native_memory_storage.py](../src/vllm_kvmem/core/native_memory_storage.py) | CPU payload 随记录发布建立；现有 CPU 后端没有自动磁盘 fallback |
| [cached_storage.py](../src/vllm_kvmem/core/cached_storage.py)、[read_cache.py](../src/vllm_kvmem/core/read_cache.py) | NVMe 权威记录上方的有界 CPU 缓存，不能等同于延迟写回的 CPU 优先分层存储 |
| [scheduler.py](../src/vllm_kvmem/serving/scheduler.py)，`_preempt_request`、`_request_blocks_can_be_freed` | bounded 请求尚未结束时禁止静默释放权威 GDN 历史；deferred prefill validation 明确禁止普通原生抢占 |
| [active_residency.py](../src/vllm_kvmem/serving/active_residency.py)、[restore_admission.py](../src/vllm_kvmem/serving/restore_admission.py) | 已支持 idle GPU 页撤回、精确 session restore 及 restore lane 准入；尚不是任意运行中请求的换出调度 |
| [cache_manifest.py](../src/vllm_kvmem/adapters/cache_manifest.py)、[transfer.py](../src/vllm_kvmem/core/transfer.py)、[historical_feature_budget.py](../src/vllm_kvmem/core/historical_feature_budget.py) | 存在共享 backing、固定传输池和 feature/query 预留，必须与 payload 分开计量 |

现有 `test_future_growth_is_reserved_and_refresh_does_not_overadmit` 就覆盖了用户描述的情况：池还有六个物理空闲页，第三个请求仍因前两个请求未来增长信用而被拒绝；刷新释放页也不改变这一保证。CPU 对应测试覆盖整轮预留和释放 fence。

## 改进目标与容量边界

需要区分三种数量：已接收但等待的请求、拥有执行槽的 GPU 驻留请求、已暂停但完整状态仍在 CPU/SSD 的请求。允许保存更多请求不等于这些请求都能在同一步计算。

建议保留原生 `max_num_seqs` 作为执行槽和图形状上限，另设有界的已接收请求及暂停请求数量。若希望同一步执行更多请求，必须相应调整原生执行槽上限，并重新测量它带来的 Graph、recurrent、capture 和 host 元数据成本；不能仅修改 admission 信用。

共享 GPU 池第一版保持启动时定容，在池内动态分配。这样可以复用当前页表与 CUDA Graph 固定地址约束。动态向操作系统归还或扩张整个 GPU 池属于后续 VMM/图地址专项，不是实现请求间共享余量的前提。

一个纯说明性的例子：假设扣除模型、固定状态、暂存和全局余量后，还剩 48 GiB 可用于请求窗口；每请求 K 占 8 GiB、G 最多 4 GiB。整窗口预留最多容纳 4 个请求；若实际 G 仅 0.5 GiB，再给每个请求 0.25 GiB 的短期增长额度，理论上可容纳 5 个。真实结果还受页取整、recurrent state 和执行槽影响，不能作为本仓库的实测收益。

所有请求最终同时增长到最大值时，动态分配无法凭空扩大 GPU。此时必须减少驻留请求数、迁移状态并轮换执行。GPU、RAM、SSD 都有限时，也不能无条件保证无限请求永久保留完整历史。

## 统一资源账本

建议新增 `resource_ledger.py`，集中管理额度，但不替代原生 GPU 分配器和已有 archive lease。scheduler 单线程决定信用授予，worker 上报带 epoch 和 generation 的实际占用及完成确认。

对每种资源分别维护：

```text
live                 当前仍实际占用的资源，包含 retired 但尚未回收的对象
promised             已授予某次操作、但尚未物理分配的信用
emergency            尚未使用、仅供排空、封存、换出和恢复推进的专用余量
hard_limit           此进程可使用的硬上限

live + promised + emergency <= hard_limit
```

`inflight`、`retired_pending` 和 `dirty` 是 live 的分类指标，不能再相加重复计费。完成物理分配时，应将 promised 原子转换为 live；紧急操作使用余量时也相应转为 promised/live，完成后补回 emergency，避免重复扣费。未来可能回收的页绝不是当前 available。

资源维度至少包括原生 GPU 各组页、CPU payload、CPU pinned、checkpoint、repository 描述符、SSD extent、I/O 字节及槽位、执行槽、restore/assembly lane。GPU 各组仍共享同一原生 page pool，除各组约束外还要检查所有组的总页数，不能让每组独享一次全池额度。按真实 backing owner 计费，同一内存的 view、共享 checkpoint 或 fork 引用不重复计费；迁移产生的第二份物理副本必须计费。

容量计算从 `cache-geometry.json` 和运行时实际 tensor/storage 几何取得。GPU 原生 page 与 KVMem archive block 可以不同大小，尤其不能把 block_tokens 直接当作原生 page 大小。CPU record 要包含 target/MTP/QSA 辅助平面、索引以及该模式实际持有的 sidecar。

RAM 总上限还必须容纳非历史占用：模型加载后的常驻内存、runtime、token ID、索引、分配器保留页、SSD buffered I/O 的 page cache 等。使用内部 owner 账本控制分配，用 RSS、cgroup memory.current、MemAvailable 做外部压力保护，二者分别记录。删除 Python/native 对象不代表 RSS 立即下降；需要 slab 复用、受控回收和观测差值，不能据低估的账本继续准入。

### 用短期信用替代整轮信用

每次执行前只申请“到下一个可暂停安全点”为止的额外资源，而不申请未来完整 G 或 `max_tokens`。

```text
本次增量需求 = 该操作完成时各资源的峰值 - 当前可复用且已拥有的资源
```

普通 decode 的额度可以较小；prefill 按块或逻辑 chunk 申请。但完整 Query replay、原子 logical prefill、MTP 最坏候选与拒绝回滚、terminal tail、private probe、restore 的必要峰值必须完整覆盖。不能固定给每请求一两个小块后假设所有操作都能执行。

长期输出预测只用于优先级和提前迁移，不能替代硬边界。安全性依赖下次执行前重新申请，以及申请失败时能停在安全点。异步 worker 的陈旧快照通过 scheduler 尚未结算的 grants 补齐，不能把上报延迟当作剩余容量。

跨资源申请使用 `prepare → commit → settle/abort`：所有必要资源都有信用后才发布任务；未发布时允许全部回滚，发布后取消也必须排空。标识至少含 engine epoch、request ID、scope generation、operation ID；重复确认幂等，旧确认不能给新请求返还信用。

## CPU 优先的分层归档

建议新增 `TieredArchive`，在现有 archive 接口外观下统一 CPU 和 NVMe。复用不可变记录、generation lease、checksum、异步作业及 bounded queue；新增“记录位置”目录，将逻辑记录身份与 CPU/SSD 物理地址分离。

```text
逻辑记录 = namespace + repository slot + generation
位置目录 = CPU 副本、SSD extent、各副本状态、读者、迁移事务、校验信息

新历史 → CPU 权威记录 → 后台写入 SSD → SSD 有效且读者排空 → 释放 CPU payload
读取    → CPU 命中直接读；否则 SSD → 有界 pinned buffer → GPU 或 CPU 热缓存
```

### 迁移状态和发布顺序

| 状态 | 有效数据所在位置 | 允许回收的内容 |
| --- | --- | --- |
| `CPU_ONLY` | CPU | 不能直接丢弃 payload |
| `SPILLING` | CPU，SSD 写入尚未确认 | CPU 继续保留；写入占用单独额度 |
| `BOTH_VALID` | CPU 与 SSD | CPU 无读者后可回收 |
| `SSD_ONLY` | SSD | CPU 已回收，逻辑记录保持可查询 |
| `LOADING` | SSD，目标 buffer 由读取事务持有 | 完成前不能重用目标 buffer |
| `RETIRING` | 已禁止新逻辑读，可能仍有老读者 | 全部 lease 与 I/O fence 完成后才回收各层 |

写盘顺序为申请 extent、持有源读 lease、写入完整记录、验证成功、发布 SSD locator，再等待 CPU 老读者退出后返还 CPU 字节。ENOSPC、短写、I/O 错误、取消或线程提交失败都不得使唯一有效副本消失。迁移完成与请求完成是两个不同事件。

不能把 `NativeMemoryTier.retire_payload()` 直接当作迁移：当前 retirement 与不可再读语义有关，分层存储需要独立的 CPU residency 释放；repository 的逻辑记录仍然存在。历史 GC 的不可逆删除也应与分层降级分离。启用已有有损 GC 时保持原策略并单独标注，不把“删除了历史”计入无损分层的收益。

初版 SSD 作为同进程生命周期的后备存储。写完成保证进程后续可读，不自动宣称掉电可恢复；若后续要求进程重启恢复，再增加 durable manifest、版本兼容、fsync 顺序和恢复审计。现有 ephemeral arena 不提供这项产品承诺。

### 按需磁盘空间与元数据

采用分段 extent 池，按压力增配一小批 SSD 空间，不在启动时按所有请求最大历史预分配整个 arena。段大小通过实测选择，每次增配检查配额、实际可用空间和专用余量；小段 `fallocate` 可提前暴露空间不足。回收可复用 extent，必要时对空段删除或打洞，绝不能处理仍有读者的 extent。

仅加入 SSD payload 并不能突破 `repository_blocks`。第一阶段保留明确的每请求历史记录上限并返回独立原因；为进一步支持大量 idle/paused scope，增加稳定逻辑 handle 的分段元数据分配，按使用增长，不能对有 lease 的连续数组原地 resize。`MemoryTier.allocate` 当前将 block ID 与 CPU slot 对齐，也必须解耦。小索引、记录目录和 token 身份同样有共享硬限额。

索引可先保留在 RAM，以便选块时不扫描 SSD 上的全部 K/V；按大历史、多 session 测量它是否成为新瓶颈。若需要索引分层，作为独立分页阶段实现，不宣称“所有历史都放 SSD 后 RAM 就没有随历史增长的成本”。

### 驻留策略与 I/O 调度

CPU 优先保留近期生成、即将 restore 的窗口、活跃请求常用记录。优先降级已空闲或暂停 session 的冷记录。shared/fork 记录按真实 owner 计费，热度可以聚合；语义 mandatory 表示不能删除，不意味着必须永久驻留 CPU。

热缓存使用共享总容量加可借用的公平份额，替代把每请求最大窗口额度锁死。已在 SSD 有完整副本的 CPU cache 可以直接丢弃；CPU_ONLY 记录必须先成功写盘。对后台逐层读取维持原 record lease、层片段布局和 byte-exact 语义，不破坏现有 prepared reader 优化。

对 SSD read、SSD write、D2H、H2D 分别设置有界字节和槽位额度。关键 restore、排空写回有进度通道，预取优先级最低；读写之间设置最低服务份额，避免 restore 抢占所有槽导致写回永远无法释放 RAM。复用 [transfer_budget.py](../src/vllm_kvmem/core/transfer_budget.py) 的限速思路，但增加 SSD 及双向传输维度。

## 运行中请求暂停和恢复

当前 session checkpoint 主要覆盖 P/M 等既有边界。动态 GPU 超额准入之前，必须扩展为运行中请求的精确暂停事务，并接入 scheduler、connector、runner 的完整生命周期。

```text
RUNNING → PAUSE_REQUESTED → DRAINING → CHECKPOINTING → PAUSED
PAUSED  → RESTORE_QUEUED → RESTORING → RUNNABLE → RUNNING
```

1. scheduler 停止给 victim 发布新计算，选已完成采样/接受判定的安全边界，等待当前 kernel、capture、archive publication、private query 和 deferred validation 排空。
2. 保存一致的 accepted prefix 与执行状态；未接受 draft 不发布为历史，最后一个已采样但尚未计算 KV 的 token 及其计数必须显式记录。
3. 写好 CPU 或 SSD 的暂停 checkpoint，得到 worker 的 generation-qualified 完成确认后，才能移交或归还 GPU attention/recurrent/draft 页及 runner slot。
4. 保留公开 request 身份、请求进度、session 串行约束、已输出 token 数和连接；暂停不发送 EOS，不重新提交同一个请求。
5. 恢复前一次申请 GPU 页、状态槽、CPU/pinned 目标、I/O 和 assembly lane；恢复完成并安装完整页表后，才重新进入 runnable 队列。

checkpoint 至少需要：精确层级选块及 dense 顺序、逻辑位置和 RoPE 坐标、未满块尾部、KV codec/scale、target recurrent/GDN 状态、MTP/draft/SWA/hidden 状态、sampling RNG、grammar/stop 状态、accepted/rejected/未计算 token 计数、attention/Mean-K/graph 策略累计状态、layerwise 窗口版本，以及请求和 namespace generation。依赖 request slot 的对象须保存逻辑内容并在恢复时重绑定。

GPU 当前 dense K 可能经过 RoPE、重排或重压缩，不能假设 CPU raw K 重建一定逐字节等于暂停前值。首版精确路径保留必要的当前 GPU 字节及布局；已有 CPU authority 能替代的部分，仅在重建一致性验证后消除重复快照。GDN 和完整注意力历史也不可互相替代。

暂停应保存已确定的窗口，不主动重新选块。否则恢复虽无崩溃，模型看到的历史却变了。整请求换出通常比频繁缺页更适合当前 dense attention；让每层每步临时从 SSD 拉 KV 会引入大量随机 I/O，作为后续专项评估。

原生 preemption、`_request_blocks_can_be_freed` 的限制要替换成专门的、已完成 checkpoint 的允许路径，保留对普通未排空释放的拒绝。不能删除断言后依赖 vLLM 自动重算。deferred prefill validation、SystemOne 分支和未合格的 speculative 模式在接入前，应在动态模式配置检查中明确拒绝。

## 水位与保证进度

“快满时再降级”应实现为预测下一次峰值，并在高水位之前启动迁移；不能等 malloc、CUDA 分配或磁盘写入已经失败。

为每层设置低水位、高水位和硬上限，形成回滞，避免频繁换入换出。初始可在扣除固定成本后的可管理容量上试验 CPU 70%/85%、GPU 页池 75%/90%；这些只是调参起点，不是生产保证。高水位到硬上限的余量至少覆盖最坏不可中断步骤、在途传输、控制延迟和一个迁移事务，必须由真实字节几何校准。

| 压力 | 动作顺序 |
| --- | --- |
| GPU | 撤回可回收的 idle GPU 页；降低新准入；暂停低优先级且换出代价合适的请求；确认释放后接纳其他请求 |
| CPU | 丢弃已有 SSD 副本的冷 cache；后台写回 CPU_ONLY 冷历史；降低 prefill/capture 生产速度；资源不足时停止新计算 |
| SSD | 回收已逻辑释放且无读者的 extent，停止预取与非必要副本；停止扩大总接收量；到硬限额时执行有界等待或明确容量错误 |
| I/O 变慢 | 减少新增长和轮换频率，保持排空及少量活跃请求的通道，避免所有请求持有半套资源互等 |

设置独立 emergency reserve，普通请求不能借走。它须覆盖至少一个合格请求的下一原子步骤和一条迁移链路；exact GPU checkpoint 若必须整窗口保存，也须计入该峰值。空间不足时支持分块快照、分块 restore，避免额外申请一个完整最大窗口才能回收旧窗口。

全资源 prepare 失败时立即归还未发布 grants，不持有 CPU 目的地等待一个无限期无法得到的 SSD/GPU 资源。恢复只拉下一次计算真正必要的完整窗口及状态，不把全部历史重新加载到 RAM。

对所有资源满、I/O 永久失败等场景，明确服务契约：严格完成模式可以只在最底层 SSD 预留整轮逻辑配额，物理空间仍逐步分配；弹性模式允许更高总接收量，但必须允许有界排队和显式容量失败。两者不能同时宣称“没有任何最坏预留”且“所有已接收请求必定生成到最大输出长度”。第一版默认停止新准入、优先保证已运行请求；持续失败不得伪装正常 EOS 或静默删历史。

## 公平调度和收益控制

按 runnable、paused、restore、tool-wait 分开排队，按 session/租户设置可借用的公平份额并叠加等待老化。活跃 decode、短请求、长 prefill 都必须得到有界服务；大请求容量不满足时可以让可运行的小请求通过。

victim 选择综合近期使用、已持有 GPU 字节、恢复成本、优先级和已等待时间；设置最小驻留时间、冷却期、每秒迁移字节及迁移次数上限。相比均匀轮转，可将工作集接近的请求分批驻留，减少反复传输。暂停时间仍计入端到端延迟。

并发控制器以“满足延迟目标的完成请求数/秒”为目标。若增加驻留请求使换出字节持续上升、decode 被传输挤占或 p99 恶化，应回退驻留数，而不是把接收连接数当作吞吐提升。

可选的弹性 K/G 应留到最后：压力下改变 K、刷新时机或 G 的有效上限，会改变历史选择、private replay 次数和模型计算路径。需要独立开关、mandatory/sink/recent 下限检查和质量非劣验收；第一版动态信用不改变这些模型策略。

## 实施顺序和文件改动

各阶段都保留旧模式，只有满足前置验收才打开下一阶段的动态准入。

| 阶段 | 实现内容和主要文件 | 放行条件 |
| --- | --- | --- |
| P0 | 在 `capacity.py`、`coordination.py`、`historical_admission.py`、`cache_manifest.py`、archive 和 capture stats 增加实际占用、未来信用、pending reclaim、拒绝原因和各组页数据；冻结输入及 profile | 能解释一次“有空闲物理内存但拒绝”的完整账目，并核对 RAM/VRAM 差值 |
| P1 | 新增 `resource_ledger.py` 和压力快照；保留旧准入决策，先影子计算增量 grants；暂存、checkpoint、I/O 归入明确资源域 | 多资源并发事务、陈旧通知、取消和失败注入均不重复授信 |
| P2 | 新增 `tiered_storage.py`、记录位置目录和 SSD 分段分配器；扩展 `storage.py`、`memory_storage.py`、`native_memory_storage.py`；先保持旧 GPU 信用 | CPU↔SSD 往返一致，压力下 RAM 有界，故障不丢唯一副本 |
| P3 | 新增运行中 checkpoint 与暂停协议；接入 `scheduler.py`、`connector.py`、`session_worker.py`、`session_state.py`、`runner.py`、`restore_admission.py` | 强制暂停各安全边界并恢复通过；释放必须晚于 worker fence；取消/断连可正确终结 |
| P4 | 在 `capacity.py`、CPU 两套 admission、`archive_admission.py` 启用短期信用；同步处理 `logical_capacity.py` 的所有 owner 最大窗口检查，以及原生 `full_sequence_must_fit` 入口；拆分执行槽与保留 scope 上限 | 同时增长至上限时也能轮换推进，不能再依赖旧 lifetime 保证；单个原子操作不可能装下时提前拒绝 |
| P5 | 分段 repository 元数据和可借用 cache 配额，解除大量 idle scope 的静态元数据乘积；接入索引、特征、fork 引用和逻辑删除 | SSD 空间、CPU 字节、metadata 槽分别受控；长期多 session 无线性泄漏 |
| P6 | 自适应水位、公平队列、迁移限速、预取与正式端到端压测；必要时增加弹性 K/G 的独立候选 | 同内存、同质量和同延迟约束下有稳定收益，达到下述验收标准 |

P4 的原生适配只改变新模式的完整序列准入策略，继续使用原生逐步分配、页取整、null page、hybrid 各组限制。不能把 `full_sequence_must_fit=False` 或取消启动检查当作独立完成项。启动仍应验证一个可支持的最大原子操作及恢复工作集能完成；动态驻留上限来自账本，而非 `max_num_seqs × K+G` 的硬乘积。

CPU 增量准入须同时覆盖无 GC 与 historical GC 路径。CPU→SSD 迁移释放的是 RAM，不释放仍有效的逻辑记录描述符；必须将 `ready_records`、resident CPU bytes、total logical records 分开。readiness/GC 汇报不能沿用“只有 CPU resident 才算有效”的隐含假设。

### 配置与兼容策略

建议的新增配置概念如下，名称在实现时统一；这些不是当前可用参数：

| 配置概念 | 建议行为 |
| --- | --- |
| memory admission mode | `static` 保持原行为；`elastic` 要求分层和安全暂停能力已合格 |
| archive backend | 增加 `tiered`；现有 `cpu`、`nvme` 含义保持不变 |
| CPU / pinned / checkpoint / metadata hard limits | 显式定义包含范围，拆开 `cpu_read_cache_bytes` 的历史双重含义 |
| SSD quota 与 segment size | 有总上限、物理余量、迁移保底；按段申请空间 |
| growth horizon | 用于普通步骤，实际 grant 取原子操作所需峰值，不允许截断 Query |
| GPU / CPU watermarks | 运行时可调，必须满足 `low < high < hard` 和最坏峰值检查 |
| max accepted / paused / resident | 与原生 `max_num_seqs`、session_capacity 和 authority 上限明确映射 |
| pressure semantic policy | 默认 `preserve`；改变 K/G/选块需独立显式开启 |
| admission contract | 明确最底层整轮配额保证或弹性有界等待策略 |

首版在已验证的单 GPU、TP/PP=1、APC off、同步单 batch 范围内实现。target-only、native MTP、layerwise、historical GC、fork 分别验收；未合格组合启动时明确拒绝。后续再扩展多 rank 的全局额度和跨 rank 暂停一致性。

回退不允许直接丢弃 SSD_ONLY 记录或运行时切换成 CPU-only。最简单的首次部署回退方式是停止新准入、排空请求后重启旧配置。动态模式下可临时停止继续超额准入，但已有暂停请求仍沿分层恢复路径完成。

## 测试方案

### 账本和状态机测试

在现有测试之外新增 `test_resource_ledger.py`、`test_tiered_storage.py`、`test_request_pause_resume.py`、`test_elastic_admission.py` 和 `test_pressure_fairness.py`，使用可控完成事件和模拟时间。测试行为与不变量，不把新实现公式复制到断言中。

| 场景 | 必须验证 |
| --- | --- |
| 大 max_tokens、短实际输出 | 新请求可利用旧模式锁定的信用；结束后只保留真实历史及仍有读者的资源 |
| 所有请求同时增长 | 在真实硬上限前暂停、迁移；无 OOM、越额、死锁，最终都按契约完成或显式失败 |
| 两个请求同时争最后一页或最后一条记录 | 只能有一个完整 grant 成功；无双重授信或持有半套资源互等 |
| 原生页与 archive block 大小不同 | 各自取整，多组 page、null page、GDN、SWA、MTP 独立计费 |
| Query 大于普通增长 horizon | 全原子 Query 被准入或等待，不截断、不默默改选块 |
| MTP 全拒绝或临界 EOS | provisional、tail、多版本和最后一次 draft 峰值被覆盖，拒绝数据不成为历史 |
| 晚到快照、重复 ACK、旧 generation、request ID 重用 | 不提前释放、重复加信用或污染新请求 |
| 迁移中并发读、取消、GC、fork | 唯一有效副本、generation 和 reader lease 始终正确；fork 不重复计费共享 owner |
| SSD 短写、ENOSPC、EIO、checksum 错误 | 未确认写入不发布，CPU 源保留，错误隔离和清理可完成 |
| 线程池 submit 失败、回调延迟 | 未发布 grant 回滚；已发布工作排空；沿用 archive submission gate 保证 |
| CPU 和 GPU 同时高水位 | emergency 通道可完成一条迁移链；无资源循环等待 |
| SSD I/O 故意暂停或限速 | 有界 buffer、队列和增长；仍可取消，不无限增长主线程等待 |
| 小请求连续到来夹一个长请求 | 既无队首阻塞，也不能使长请求无限饥饿 |
| idle/tool-wait session 再次激活 | 不丢历史，匹配正确 P/M 版本，暂停状态与 session 串行锁不冲突 |
| namespace/tenant 相同 token 内容 | 不能跨 namespace 借用私有历史，资源归属和权限不随迁移改变 |
| 元数据满但 payload 未满 | 报独立容量原因；分段扩容不使现有 handle、lease、快照失效 |
| 长期 allocate/free、暂停/恢复 | 等待所有 fence 后回到规定保留量，碎片和 RSS 高水位有界 |

加入状态机随机测试：固定种子生成 grant、allocate、spill、restore、cancel、close、GC 和乱序 ACK 序列；每个动作后检查容量守恒与唯一权威。保存最小失败轨迹。native residency/目录改动增加 C++ 并发与 sanitizer 测试，保证 reader 获取和迁移发布的原子性。

### GPU 字节及模型状态验证

从小模型或冻结状态开始，以非常小的 GPU/CPU/SSD 配额强制每条迁移路径发生。覆盖完整块、partial tail、跨 page、跨 layer、不同 dtype、真实原生页几何以及页复用。

每次暂停前后比较原始 GPU KV 字节、dense 顺序、位置、GDN/recurrent、MTP/hidden/SWA、accepted counters、RNG 和策略累计状态。专门验证新请求使用旧物理页后，旧请求恢复仍得到自己的数据。CUDA Graph 模式检查固定 backing 地址、request slot 重绑定、block-table 更新和 capture 源/destination 生命周期。

模型验证分为两类，不能混为一个“输出一致”指标：

1. **隔离存储和切换正确性**：固定 batch 形状、输入、seed、采样及选块轨迹，只插入暂停/迁移，比较所有 public token、关键 logits 和状态；要求迁移数据 byte-exact。若重建路径不能等价，保留原始快照路径。
2. **真实动态调度**：请求到达和 batch 组成变化，先做 A/A 复现性测量，再评估差异和任务质量。当前仓库已有批次形状引起上游数值差异的记录，不能把所有 token 差异都判成换出损坏，也不能用这点掩盖状态不一致。[已有调查](layerwise-pressure-followup-20261105.md)

不得缩短完整输入、降低 max_tokens、关闭原质量相关功能或删历史来制造通过。弹性 K/G 候选另做长上下文检索、多轮工具使用、MAB/既有 agent 任务质量与截断率评估；正式外部评分不属于本方案阶段的自动执行内容。

### 性能与容量对照矩阵

保持模型、精度、完整输入、输出配额、硬件、GPU 池、CPU 总预算一致。CPU-only 与新增 SSD 的差异单独披露；只有 CPU 足够的基线可用于无压力纯开销比较。另加入现有 NVMe＋CPU cache 模式，区分 CPU 优先 writeback 与写穿存储的收益。

| 维度 | 取值或工作负载 |
| --- | --- |
| 模式 | 当前 static CPU；当前 NVMe＋cache；tiered 存储＋旧 GPU 预留；完整弹性信用＋安全暂停；可选弹性 K/G |
| 请求数 | 1、2、4、8、16、32，直到达到该硬件延迟/吞吐拐点；执行槽上限相应记录 |
| 历史与输出 | 短历史/短输出；长历史/短输出；长输出接近 max_tokens；混合长度 |
| G 利用率 | 低利用率、接近一半、接近满额；同时记录实际窗口增长分布，不只记录最终输出长度 |
| 到达模式 | 同时突发、平滑到达、开放环固定请求率、tool-wait 后集中唤醒 |
| 压力 | 无压力、GPU 单独不足、CPU 单独不足、两者不足、SSD 满、SSD 降速 |
| 存储热度 | 冷启动、CPU 热、SSD 热缓存、真实 SSD 冷读，明确 page cache 处理方式 |
| 功能 | target-only、native MTP、layerwise、多 owner prefill、历史 GC 分别验证；fork 在支持后加入 |

可复用当前 16K/4K 效率 fixture 和 36K/16K MAB fixture，但必须保留各自完整参数，不能混用。吞吐以真实 public 工作计数为准，private replay、restore 和预填充不能当作新生成 token。

每组预热后做至少三组配对 ABBA 或随机交错顺序；同一组复用相同到达时间和输入轨迹，记录中位数、离散度和置信区间。开放环试验报告队列是否持续增长；封闭环报告完整端到端耗时，不能只报告纯 decode 区间。至少一组高压力场景运行到稳态并覆盖多次冷/热循环，再执行不少于两小时的混合生命周期 soak。

必须收集：

- 完成请求数/秒、public output token/s、prefill token/s、TTFT、ITL/TPOT、端到端 p50/p95/p99、排队/暂停/恢复时间、失败与截断率。
- GPU pool 的 live/free/promised/emergency 页，recurrent/draft 页，进程 VRAM；CPU RSS/cgroup、payload、pinned、checkpoint、metadata、缓存、retired pending 和 allocator 残留。
- SSD 逻辑字节与实际占用、读写吞吐、放大率、I/O 队列深度、p95/p99 完成延迟、脏 CPU 字节、冷热命中率、checkpoint 重复字节。
- 每次 admission 拒绝原因、每请求借用/归还信用、swap 次数、连续无进展时间、最长等待和各租户服务份额。
- CUDA 事件实际完成时间与 CPU 观察时间分别记录，避免把完成通知延迟全部解释成传输耗时。

定义最大有效并发为：在约定 TTFT/ITL/端到端延迟目标内，队列不发散且持续完成请求的最高并发；不能用接收连接数、保存 session 数或瞬时驻留数代替。

### 放行标准

| 方面 | 要求 |
| --- | --- |
| 安全性 | 预算不越界、状态不串请求、无提前复用、无唯一副本丢失；所有故障测试能正常排空或明确终止 |
| 进度 | 最坏同时增长、CPU/GPU 双压力和 I/O 降速下无死锁；有限输入和正常存储下所有请求持续获得服务 |
| 精确性 | 冻结计算轨迹的迁移 byte-exact，状态和 public 输出一致；动态 batch 的数值波动用 A/A 基线及任务评估解释 |
| 无压力开销 | 建议初始门槛：吞吐中位数退化不超过 5%，p95 延迟退化不超过 10%；这是实现验收目标，不是现有结果 |
| 容量收益 | 在明确构造的“未来信用占满、物理页仍空闲”负载中，实际运行更多请求；公开新增 I/O 与暂停成本 |
| 真实收益 | 同质量及延迟目标下提高最大有效并发或完成率；如果只有驻留数上升而吞吐/尾延迟恶化，不启用该策略 |
| 资源回收 | soak 后排空所有 fence，回到定义的缓存保留量，内存及 extent 无持续爬升 |

最终性能门槛应在跑候选前固定，不在观察结果后临时改口径。弹性 K/G 的质量容忍度也必须事先定义，不能仅凭吞吐提升放行。

## 当前验证与后续交付

本次对现状运行了 CPU 路径回归：`test_capacity.py`、`test_window_manager.py`、`test_coordination.py`、`test_historical_admission.py`、`test_storage.py`、`test_cached_storage.py`、`test_archive_io_budget.py`、`test_active_residency.py`、`test_restore_admission.py`，82 项通过；另运行 `test_logical_capacity.py`，47 项通过，**合计 129 项通过**。执行使用现有 Python 3.12 环境、当前工作区 `PYTHONPATH=src` 以及 `CUDA_VISIBLE_DEVICES=''`。这些验证当前预留、启动限制及生命周期行为，不是新方案的 GPU、分层存储或性能验收。

实现交付应包括：功能开关与升级说明、资源几何及账本指标、故障注入结果、GPU 状态等价报告、完整 profile 和输入来源、代码/依赖版本、配对性能原始记录、质量结果和回退演练。首先交付 P0 的“实际占用与锁定信用差额”报告，用它判断主要收益来自 G 余量、CPU 整轮预留，还是执行槽和 recurrent 状态约束，再按上述依赖实施。
