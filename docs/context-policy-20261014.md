# 跨轮选页、预算、必留内容与 P/M 检查点

日期：2026-10-14。此版本在显式 token-ID Query + 原版 FP16 MeanK + 原生 Qwen MTP
路径上补齐下面四项；历史审计 `5043645` 的缺口以本报告和当前实现状态为准。

## 行为变化

| 项目 | 此前 | 当前 |
| --- | --- | --- |
| 稳定工具续接 | 默认恢复可能把 active 窗口压回 K，改变选页 | exact-prefix 校验通过后恢复同一 active frame；没有新 Query 且空间足够时保持原选页。重复提交同一旧 Query 的 auto 模式也不会自动重选。 |
| 长历史准入 | keep_selected 按完整逻辑 prompt 检查，长历史会被拒绝 | 原生会话持有者按 `saved_window + new_suffix + max_tokens <= K+G−B` 检查实际剩余空间。空间不足保留原 frame 恢复，然后在真实容量边界压力选页；检查点失效则冷构建。 |
| 必留内容 | 在线路径只依靠 sink/recent | token spans 映射到 generation-qualified 历史块；Query 与压力选页均传入 hard mandatory。超额直接调用未修改的 qw3 capacity-fit helper，保留结构边缘、按优先级分配并预留检索容量。 |
| 检查点 | 显式 Query 模式周期性保存封页快照，回合结束 rewind | 保留 Query 前 P 与回合终态 M；普通工具续接从 M 恢复，强制旧 Query 重选从匹配的 P 重放。两份快照均计入 CPU 检查点配额。 |

K=32,768、G=16,384、B=128。仍限制单轮输出不超过 G。没有新 Query 的普通
输入可使用原版 branch-safe helper 留出完整模板尾块。P/M 使用原生 accepted-state
canonicalization，target/MTP KV、96 个 GDN 状态和 CPU authority 保持同一位置。
恢复先验证完整 token prefix；不能只凭 session 名字或哈希恢复状态。

网关 `active_bytes` 仍按 K+G 和模型附加状态保守预留，避免缓存失效后的冷请求超额。
上面的“实际预算准入”指原生已获得会话所有权之后的 active headroom 判定；本版本
没有将全局字节预留改成基于可能过期的外部快照动态缩减，也没有实现带宽 SLO 调度。

## 请求字段

字段属于本项目的 token-ID API，坐标均为完整逻辑 prompt 的绝对 token 范围，
不是 qw3 同名字段的 UTF-8 消息坐标协议。

- `kvmem_mandatory_spans`: `[{"begin":12800,"end":13312,"kind":"file"}]`。
  kind 可为 system/root/query/policy/tool-call/tool-result/file/explicit。
- `kvmem_mandatory_parts`: 默认 system/root/query/policy；支持列表、布尔字典、
  `default`、`none` 或逗号分隔字符串。消息角色映射需要调用方提供 token message spans；
  tool-call 等无法从普通 assistant 角色识别的结构须显式给出 spans。
- `kvmem_replay_span`: `[begin,end]`，须包含完整评分 Query；可从更早的消息 header
  开始，评分仍只覆盖 Query。封页起点使用原版 replay helper。
- 未显式替换 mandatory 策略时跨轮继承；新 Query 替换旧 Query 必留段，保留其他 pins。
  新增的历史 pin 若不在已保存 frame 中，先重建，避免让缺失证据的旧状态继续执行。
- `kvmem_reselect=force` 配合旧 Query 可选择 P；新 Query 可从因果上位于其前面的 M
  继续。auto 对已验证相同 prefix 中的旧 Query 保持 epoch；off 不接收 Query。

超大 file/root 等 span 的“必留”遵循 qw3 容量适配规则，不代表整个超大 span 永远
常驻 GPU。被软化部分仍保留在 historical source 中，后续可检索。

## 检查点资源

显式 Query 弹性 profile 默认 `session_checkpoint_policy=pm`，周期 decode 快照关闭，
终态 M 捕获开启；可显式选择旧 sealed 策略。公开 8 用户示例的检查点额度为 4 GiB。
原 2 GiB 在此次 8 用户 P+M 验收中出现快照配额不足：请求仍成功，但部分会话回退
到 P 并多重放后缀。此降级已记录，不计为完整 M 验收通过。

退役记录提供 `checkpoint_kind`、单份/池 payload bytes、池限额和累计 deferrals。
配额不足时保留仍然有效的 P，不丢失完整 token 历史；恢复成本会增加。
100 GiB CPU history read cache 与检查点池、token 账本、临时传输和 Python 索引分别
计费，不等于进程 RSS 总上限。GPU 页退役后释放，CPU P/M 仍占受限的缓存额度。

## 验收与性能

功能验收使用真实 NVFP4 模型、target/MTP FP8 KV、MTP k=3、原生编译/CUDA Graph、
KVMem 历史维护和原版 FP16 MeanK。输入为冻结的 50,006 token；模拟工具回调，
不执行外部工具。恢复诊断对 GDN 做逐字节检查，M 还检查重建后的 target/MTP KV。
早于 M 的 P 使用自己的 GDN 状态，不拿不同位置的 M KV 作为 oracle。

8 用户四轮共 32 回合、24 次恢复（16 M / 8 P）、24 次显式 Query 重放全部通过，
零检查点配额降级，最终 requests/owned_blocks 均为 0。另有单用户 5 回合边界测试：
强制旧 Query、新 Query、超预算 file span，以及追加 20K 工具内容后的压力刷新；
历史达到 73,447 token，4 次恢复通过。所有选页事件检查 hard mandatory 被覆盖。

短回合样例中，M 恢复从旧后缀重放 2,023 token 降至 1 token；这一个 token 是尚未
作为输入计算 KV 的最后采样输出。该结果不推广到 P 回退、模板编辑或所有新 Query。

CPU 回归 652 passed / 73 skipped；中间版本 GPU 回归为 713 passed / 6 skipped，最终 worker 针对回归 38 passed / 1 skipped、launcher 87 passed，
上述 37 个真实模型回合在确认顺序修复前完成。最终 wheel 追加通过单 token 的 C1/C8
共 9 个请求，以及 C8 五轮共 40 回合：32 次恢复（24 M / 8 P）、24 次 Query 重放、
16 次容量压力刷新；每个用户历史达 73,447 token，零检查点配额降级，结束后 GPU
请求页归零。超大必留范围、稳定选页和实际 headroom 溢出均在并发中检查。
单 token 请求还暴露并修复了同一批 final prefill 中 P 捕获确认与终态 M 确认重叠的
问题：两个待确认位置分别跟踪，未确认的 P 不作为已保存检查点；四种成功/配额不足
组合均有回归覆盖。

[公开基准数据](benchmarks/context-policy-20261014.json) 留存 wheel/source 哈希、配置
和验收/性能记录。所有性能口径区分稳态 decode、整轮输出
吞吐以及输入到全部首 token 的有效 prefill。性能通过原生 `/v1/completions` 与独立 session 测量；多用户鉴权/版本隔离在前置
弹性网关的跨轮验收中检查，性能表不额外计入网关的整段 JSON 回包成本。
prefill-only 服务测量使用 max_tokens=1，
仍包含排队、CPU 历史/检查点维护、MTP 准备，不是孤立 GPU kernel 速度。

## 最终吞吐

单卡 RTX PRO 6000 Blackwell 96 GB；NVFP4 权重、FP8 target/MTP KV、MTP k=3、
编译/CUDA Graph、KVMem 和 P/M session 均开启。性能输入固定 32,768 token，
SHA256 为 `19543eb03ad997478d92dc001aab03bd091f3b92a17c639e721e1d09fa3d3c8d`。

| 测试 | 第一次 | 第二次 |
| --- | ---: | ---: |
| 正常配置 C1 prefill，输出 1 token，输入 token/s | 4,656.24 | 5,932.14 |
| 正常配置 C8 prefill，输出 1 token，合计输入 token/s | 6,426.44 | 7,572.67 |
| 正常配置 C8，32K 输入 + 4K 输出，稳态输出 token/s | 561.08 | 561.39 |
| 同组 C8 整轮输出 token/s | 337.23 | 334.17 |
| 同组 C8 首 token p95（秒） | 36.08 | 36.99 |

两个 decode 窗口均由原生数据确认 8 路稳定所有权，覆盖率 99.71% / 99.54%；
16 个请求全部生成 4,096 token。无抢占、无检查点配额降级，结束后 GPU 请求页归零。
第一轮保留实际首次形状/JIT 等成本，没有删除较慢样本；两轮不能构成置信区间。

保持恢复诊断开启的最终对照为 **560.57 稳态 / 301.36 整轮 token/s**，此前周期
pinned 检查点基线为 **503.22 / 313.07**：稳态约 +11.4%，整轮约 −3.7%。
中间版本同诊断对照为 549.44 / 300.31。P/M 策略取消了每 128 token 的周期快照，
改为保留 P 和终态 M，降低持续 decode 的快照成本；两边生成序列可能不同，不能把
全部速率变化归因到单一改动。

正常配置只关闭验收用的恢复字节 oracle，保留所有实际历史与检查点维护。其整轮
334–337 比诊断配置更快，包含少做诊断复制的收益，不能全部算作上下文策略优化。
prefill 的 6.4K–7.6K 为单输出 token 服务测法，不能直接与此前含穿插 decode 的
7,153 输入 token/s 混为一个同口径涨跌指标。32K 性能输入在共同 decode 窗口没有
历史选页刷新；50K→73K 的功能验收单独覆盖了真实压缩、重选和恢复。

## 仍未覆盖

本版本保留一个 P 和一个 M，不是任意 checkpoint ladder、命名 Query tensor cache
或历史编辑系统。完整 private-query/evidence/growth gate 尚未移植；强制旧 Query
从 P 重放，不等于缓存旧 Q 后对新增全部 evidence 再评分。

独立 live span 必须包含 Query，且完整 partial tail + replay 仍受 G−B 原子容量限制。
默认聊天 Query、qw3 消息协议兼容、DSH 自动坐标/会话适配、独立 prefill budget、
prefill-only ingest、增量 GPU index、多模态、多 GPU 和异步多在途 batch 尚未补齐。
本轮不声称两套模型执行器完整逐页/质量等价，也不证明 8×256K 长期容量或工具负载 SLO。

## 复测入口

- `scripts/qualify_context_policy.py`：通过鉴权网关检查跨轮 P/M、原选页和 mandatory。
  `--pressure-tool-tokens 20000 --oversized-file --concurrency 8` 是最终五轮验收。
  配置需开启恢复字节校验，输入 fixture 包含 `input_ids` 与 `query_span`。
- `scripts/qualify_prefill.py`：固定 token IDs，C1/C8、每路输出 1 token，保存每路首 token
  时间和服务输入吞吐；独立 session，等待 GPU 请求页归零后开始下一轮。
- `scripts/qualify_decode.py --prompt-sessions --output-tokens 4096 --concurrency 8 --rounds 2`：
  共同稳态窗口剔除每路首尾至少 64 token，并要求原生记录确认完整 8 路同时 decode。

脚本的输入 SHA256、最终安装 wheel 和源文件 SHA256 随公开基准数据保存；完整运行
日志位于本机 `artifacts/context-policy-20261014/`，不把模型或用户密钥提交到仓库。
