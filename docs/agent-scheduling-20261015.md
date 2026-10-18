# 多轮 agent 调度优化与验收

> 2026-10-15 策略收敛：后续开发与主验收以
> [离线多 agent 调度策略](offline-agent-scheduling-policy.md) 为准。
> 下文 A0–A3 是功能诊断能力，512-token 共享上限和 8 GiB/s 限速没有完成
> 成本校准，不作为默认生产参数或默认主验收矩阵。

## 目标与当前状态

保持 Qwen NVFP4、FP8 KV、原生 MTP(4)、编译图、MeanK / Query history /
action attention 语义，使用 K=131072、G=131072、4 路实际模型并发、100 GB CPU
历史预算完成原 first16 DeepSWE。CPU 历史不落 SSD，不因容量压力丢弃或重算。

先进行同负载的对照验收，再选择配置运行完整 first16。主要指标为有效输出
token/s、整批完成时间、工具返回到首 token 的延迟、输出事件间隔 p95。
CUDA model-step 时间与用户输出间隔分别报告；不把预填充结束时一次入账的
prompt token 计数当成瞬时 prefill 速率。

当前 **尚未通过本次真实 GPU 验收**。r16 的旧桥接准备存在已验证的重复编码
瓶颈，已中止并保留原始记录。新序列先运行恢复正确性与三对 A0/P1 性能测试，
通过后用同一修复后的桥接器分别重跑两组完整 first16，避免混入桥接差异。
不将新增代码的 CPU 检查称为实际吞吐提升。

## 已实现

1. **工具往返保持原始输出 token**：桥接器比较严格解析后的 JSON 参数，
   保留数值精度、字符串内容、调用 ID、工具顺序和 reasoning。仅对已知 Claude
   Code Edit 补齐 `replace_all=false` 默认语义。重复 JSON key、非有限数及无效
   JSON 不进行等价归并；真实修改仍走验证过的公共前缀 / P 检查点路径。
   r16 离线快照的 44 次非首轮前缀失败中，43 次参数等价，1 次真实字符串变化
   仍被拒绝。证据：`artifacts/deepswe-first16-20261015-r16/prefix-transport-audit.json`。
   这是身份复核，不是已经消除 43 次 GPU prefill 的性能实测。

2. **整个 batch 共享普通 prefill 额度**：`prefill_batch_tokens` 默认 0（关闭）。
   有可运行 decode 时共享额度，并按可准入 prefill 数分配份额；没有 decode 时
   回到原有分块上限。原子 Query 不拆分，单 token 的 MTP 占位形状不改变。
   额度在原生分配前保守预留，分配失败会在下一调度步重新计算。它不是硬延迟保证。

3. **GPU active 页跨轮驻留**：`active_resident_ttl_ms` 默认 0（关闭），
   `active_resident_max_fraction` 默认 0.5，限制占用原生 KV 页池的比例。
   只有 CPU 历史发布和精确 M 检查点退休确认后的 attention 页才能驻留。
   下一请求的 namespace、generation、历史位置、active 长度以及 CPU 前缀
   校验均一致后，原生页引用转移给新请求，不重新下载 / 装配其历史 attention KV。
   GDN / 原生 MTP 模型状态仍从 CPU 检查点恢复并受完成事件保护。
   闲置页按 TTL 和容量压力释放，历史仍留在 CPU。已经转给活跃请求的页不参与
   TTL 回收。TTL 在调度活动时检查，不承诺完全空闲期间按墙钟立即释放。
   当前限定已资格验证的原生 MTP / target 路径，不扩展到 DFlash SWA 驻留。

4. **后台 KV 恢复搬运限速**：`maintenance_transfer_bytes_per_second` 默认 0，
   `maintenance_transfer_burst_bytes` 默认 64 MiB。刷新和恢复共用一个限速器，
   与已有单生产者、传输预约、请求老化策略配合。等待发生在后台生产者，GPU 无
   其他公共工作时立即放开。限速针对 packed KV H2D 提交；单次 copy 不可抢占，
   GDN 模型状态恢复仍是独立的原子操作。这不是对总 PCIe 流量或 GPU 延迟的硬保证。

新控制项默认关闭，可逐项比较，不改变基线模型配置。

## 验收路径

`scripts/qualify_agent_scheduling.py` 准备固定输入 / 工具 token、固定输出长度和
固定工具等待，跨轮保留每个 arm 自己接受的原始输出 token。支持以下配置：

| Arm | 设置 |
| --- | --- |
| A0 | 原配置：每请求普通 prefill 2048，CPU 恢复 |
| A1 | A0 + active 页 TTL 5 秒，闲置页最多使用原生池的 50% |
| P1 | A1 + 统一离线策略；实际就绪状态、资源额度和观测成本驱动队列与后台恢复顺序 |
| A2 | A1 + decode 存在时普通 prefill 总额度 512 |
| A3 | A2 + GPU 有公共工作时 packed KV 上传上限 8 GiB/s、突发 64 MiB |

默认主验收仅运行 A0 与 P1，顺序在重复之间交替。工具等待包含 0.05 / 0.25 / 1 / 6 秒，覆盖短时热恢复
和 TTL 失效；工具结果包含 512 / 16384 token。压力回收另由原生页池和实际
未来增长预约共同约束，不能借缓存驻留透支已有请求的 GPU 额度。

- 正确性：开启逐字节 KV / GDN / MTP oracle，按完整 request ID 对应每轮恢复，
  `check_coordination_restore.py` 校验所有用户 / 回合 / attention 层和 recurrent 张量。
  热恢复必须真实出现，并且历史 KV uploaded_bytes 为 0；不能只凭 HTTP 成功放行。
  P1 另测 TTL、冷恢复在途取消和热恢复后取消；独立 Ppressure 通过小闲置池及四路
  退休屏障强制回收，并要求页归还与 CPU 额度守恒。这些故障注入不计入性能。
- 性能：关闭 oracle，所有组保留相同采样。至少三组成对重复，分别比较纯 decode、
  多轮 agent、长 prefill 混合 decode。热缓存、共享额度和限速分别消融。
- 正式 first16：固定原题、工具与隐藏评分协议，重新执行全部题目；先通过原有
  Query 原子重放、短尾跨轮 token 一致性、4 路完整 128K 生成及清理预检。
- 稳定性：不得出现非法 GPU 访问、跨用户状态混用、缺失恢复证明、静默截断、
  因容量重算、死锁或基础设施故障；观察 CPU 历史、原生页额度和闲置页归还。

目标是多轮场景有效吞吐提高至少 10%，纯 decode 不出现可重复退化；这是待验证
目标，不是结果承诺。P1 未通过时保存轨迹并定位原因，不能临时更换参数后挑选
最好的一次作为主结果。正式 agent 轨迹可能随调度变化，任务质量与性能
分别报告，不把不同生成轨迹当成严格相同计算量的对照。
