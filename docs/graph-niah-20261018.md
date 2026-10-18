# 图评分与 Query／累计 Attention 的 256K 对照

生产评分基线为 [25a7bd9](https://github.com/Di-Chai/kvmem-vllm/commit/25a7bd9051b65906943a9b4be1f53298305c905d)。
后续新增离线队列与逐请求租约观测，不修改三种评分算法。
图功能保持默认关闭。本页记录该基线上的评分实验，不代表对全部 agent 任务的质量结论。

## 固定协议

- qw3 `official_template_20260927` 原始完整输入中 256K 的全部 60 题：四种任务、五个深度、三个 seed。
- 逐题完整 int32 token 文件、Query 坐标、原始评分函数均冻结并校验；不截断 filler，不按答案筛题。
- 三组各 60 题，8 个客户端；每条历史按完整输入、最大输出和 generation staging 预留 CPU 信用。
  当前协议使用连续 FIFO waiting queue，任一请求完成并通过其自身退休门禁后立即补位。
  running 与 retiring 合计最多 8 条，不等待整波结束；详见 [离线队列策略](offline-inference-queue.md)。
- 同一 27B NVFP4 权重、FP8 KV、native MTP(k=4)、编译/CUDA Graph。
- B32、K36,864、G16,384、sink1,312、recent0；普通 prefill chunk1,024。
- profile 中的 `max_num_batched_tokens=4224` 会被当前启动器归一化为
  `query_replay_budget + max_num_seqs × (draft_tokens+1) = 1024 + 8×5 = 1064`。
  当前图评分实例的启动日志已核对，另外两组共享同一归一化逻辑；
  4224 不能当作本次实际 native token 预算。
- temperature1、top_p.95、top_k20、min_p0、presence0、repetition1；逐题原 seed；thinking4,096、最大输出8,192。
- 完整历史在 CPU；cache 上限 100,000,000,000 bytes；保留 16 GiB 入口 payload 余量，运行保留 8 GiB CPU 和 4 GiB GPU 可用内存。
- 模型逻辑上限 524,288，容纳完整 256K 原始输入及输出；GPU 实际窗口仍受上述 K/G 限制。

| arm | 评分 | `graph_mode` | `action_fraction` |
|---|---|---|---:|
| baseline | 原 MeanK Query | off | 0 |
| legacy | Query＋累计 prefill/output Attention 配额 | off | 0.5 |
| multi | 当前 Query 加权的一跳 locality＋attention 图 | multi | 0 |

multi 使用固定 `wL=.1`、`wA=.2`、top4、cross-message=.25，未按测试答案调权。
裸 token 输入不捏造消息边界；缺少边界时沿用已实现且有日志的 locality fallback。

三个 arm 共享相同冻结 runtime，profile 只允许 `graph_mode` 和 `action_fraction` 不同。
为隔离评分选择，三组均采集相同的 input/non-thinking output attention。
因此 baseline 是 Query-only **选页**基线，并非关闭所有 attention 采集的最低成本生产部署。

首轮统一关闭 Query history、guided Query、attention pressure 与 repository GC，保持 chronological
pressure 行为相同。图评分现阶段的支持边界不允许它与 attention pressure 同开。
所以本实验比较正常 Query 评分，不把不同 pressure 策略造成的收益归到图上。
过去完整 attention pressure 部署的 97.08%、53/60 只作历史参考，不作同配置对照。

## 验证和结果口径

测试不向模型发送独立的金标、针位置或证据标签，只发送原始输入与公开 Query 坐标。
每题检查原输入字节、返回接受 token 数、终态历史长度、唯一 Query 重放、预算、CPU 退休确认和显式释放。
输出 Attention 行数由独立 accepted-token parser 重放核对；Query replay 不得增加 prefill 观测。
图组还要求真实 Query 图评分日志、非空来源与边以及正确接受水位；报告实际改变选页的题数。

主要结果为逐题数值覆盖率宏平均和整题数字集合正确数，附逐题配对、图独有命中／丢失和
McNemar exact 双侧 p 值。固定 seed 的一次配对仍不代表多轮稳定性；不同实际 batch 和生成轨迹也会影响输出。
选页证据覆盖单独报告，不能以“证据已选中”替代最终回答正确。

吞吐为完整长输入、选页、恢复及短答案混合任务的端到端速率；不能称为纯 decode 吞吐。
首次加载、编译和矩阵算子调优不计入客户端答题窗口，但包含在全程资源监控中。
每个 arm 使用独立实例，持有现有 GPU 测试锁，退出验证自有进程组和 GPU 进程已清理。
任何技术失败停止该次尝试并保留证据，不因低分自动重试。

此前波次协议的测试入口与评分门禁通过 133 项 CPU 回归；连续队列及相关路径已通过 409 项 CPU 回归。
完整三组真实模型结果尚待完成，不能用先前 12K 功能样本或部分 256K 结果替代。

首次尝试 r1 保留为失败证据：客户端错误地用未对齐的 Query 起点检查图接受水位，
而实际公开 Query replay 从所在页的起点开始。修正门禁为页对齐的 replay 起点，并新增回归；
这没有修改生产评分路径、样本、采样参数或预算。r1 的两条回答不计入正式结果。
随后 r2 把显式关闭移到每波结束。用户要求持续补位后，进一步检查确认：非保留检查点的
独立题目会在普通退休后自动删除空会话。新协议不再发送额外 GPU 清理请求，而是检查
本请求的 scheduler 会话、CPU 历史租约与 GPU reservation 已释放，再补入下一题。
波次协议的 r2 保留为部分实验，不与后续连续协议混算。所有尝试的冻结文件与证据独立保留。

连续协议的首次启动 r3 在客户端输入校验阶段发现漏传 tokenizer，未发出模型请求；
失败触发自有进程清理并确认无 GPU 残留。参数传递已修复，新增完整输入提交阶段回归；
重新冻结的新尝试使用 r4 目录，r3 的失败证据保持原样。

## 运行期利用率检查（非最终性能结论）

r2 首波完成时，8 个客户端并发提交的请求已经全部执行；观察到的模型 decode batch 最大为 2。
45 秒只读 GPU 采样的平均利用率为 65.9%，范围 1–100%，其中 8/45 个采样低于 30%。
同窗口 179 个模型 step 中，149 个仅有一路 prefill，7 个两路 prefill，
7 个一路 prefill 加一路 decode，16 个仅有一路 decode。
这一窗口主要处于波次收尾，不能推断整轮平均利用率。

实际总 token 预算 1064，以及一波结束后才补齐下一波的质量测试协议，均限制了持续大 batch。
图状态准备在首波累计约 134 秒主机计时，Attention 捕获累计约 107 秒主机计时；
这些计时可能与设备工作重叠，不能相加或等同于 GPU idle。
该采样属于旧波次协议；新三组将统一采用连续补位，保持相同 token 预算与评分参数。
扩大总 token 预算需另外进行性能实验。
诊断原始采样及结构化说明保留在 r2 的 `gpu-diagnostic-1792310994.json`、
`status-diagnostic-20261018.json`。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  /home/chaidi/kvmem-vllm/.venv/bin/python -B scripts/benchmark_graph_niah.py prepare \
  --output artifacts/graph-niah256k-repro

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  /home/chaidi/kvmem-vllm/.venv/bin/python -B \
  artifacts/graph-niah256k-repro/frozen/benchmark_graph_niah.py run \
  --output artifacts/graph-niah256k-repro
```

输入和权重使用现有本机路径；更换模型、预算或采样参数需要独立的新协议与目录。
原始模型和大体积 token／运行文件不随 Git 提交发布。
