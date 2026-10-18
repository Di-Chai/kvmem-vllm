# 256K NIAH：输出 Action 已运行，冷输入 attention 路径尚未迁移

2026-10-14。60 个原始样本已全部取得结果，数字覆盖率 **68.33%**，整题数字集合
全对 **30/60**。这不是 qw3 完整 attention 策略的复现，也未达到其质量。
运行中服务在完成 45 题后中断；保留全部原始证据，以相同配置补跑剩余 15 题。
不能将本次记录描述为连续 60 题无故障验收。

## 找到的正确 qw3 对照

用户提到的 attention 提升记录位于
`/home/chaidi/qw3/benchmark/attention_multilayer_qa_20261011/`。
逐题检查这组 Query/attention 两个分支，确认本次使用相同的 60 个完整输入、
Query 坐标、随机种子和请求采样参数。对照的 JSON token 哈希与 int32 二进制哈希
编码不同，均按各自编码重算校验；不是只凭样本名称相同。

| 256K，60 题 | 数字覆盖率 | 数字集合全对 |
| --- | ---: | ---: |
| qw3 Query-only，同引擎配对基线 | 68.33% | 30/60 |
| qw3 Query + prefill/output attention | **98.75%** | **57/60** |
| 此前 vLLM，无 Action | 65.42% | 30/60 |
| 当前 vLLM，只有输出 Action | **68.33%** | **30/60** |

qw3 两个分支均无 CacheBlend 重算，其配对结果支持完整 attention 策略在该配置上
带来明显收益。当前 vLLM 与其 Query-only 总分相同不表示逐题输出或实现相同。
之前引用的 qw3 **97.92%、58/60** 来自另一个
`official_template_20260927` 实验，该实验用了 10% CacheBlend；不是这里的
attention 对照。此前 **65.42%、30/60** 则属于旧 vLLM，不能称作 qw3 分数。

## 关键实现缺口

qw3 每题实际有 255 次 prefill 观测、每次 32 个 Query 行，共 **8,160 行**，
跨 full-attention 层 3/31/63 聚合；每题 **13 次**中途容量回收使用 prefill attention，
最终语义选页也记录了 `attention_source=prefill` 和非零 `action_added`。
60 题共 489,600 个 sampled Query 行、780 次 attention 容量回收。

当前 vLLM 的输出 Action 路径确实工作：60 题均有接受后的非 thinking 输出样本，
总计 188 行；原版 scope parser 离线重放与实际计数逐题一致。但所有题的首次 Query
选页都记录 **0 个 Action 历史页**。输出分数在答案生成后提交，仅供后续轮次选页。
本测试是独立冷请求，后产生的分数无法影响本题已经完成的首次选页。

尚未迁移、而对冷输入 NIAH 必要的路径是：

1. **Prefill attention**：对普通输入的因果 Query 行采样，按原历史页记录 mass/exposure，
   按输入 token 距离衰减。排除私有 Query、重放、重算和未提交分支，处理重绕及检查点。
2. **Attention pressure selection**：active 空间不足时，在当前驻留页及已提交追加内容中，
   用 attention 保留页面；sink/mandatory 优先，空信号按原规则回退。
3. **冷输入语义选页回退**：没有正输出 attention 时，使用 prefill 分数填充 attention
   配额，并与全历史 MeanK Query 排序联合选择；不能简单把输入和输出原始分数相加。

因此 `action_attention=true` 不能代表完整迁移。接口中目前也没有能开启上述缺失路径
的隐藏开关。本轮只修正验证及文档，没有实现这三项，也没有声称准确率已修复。

## 其他需要对齐的配置

| 项目 | qw3 attention 参考 | 本轮 vLLM |
| --- | --- | --- |
| 主模型权重 | Q8_0 | NVIDIA NVFP4，部分层 FP8 |
| GPU KV | FP8 | FP8 |
| 推测解码 | DFlash2 | MTP(k=4) |
| selection / gen reserve | 36,864 / 16,384 | 32,768 / 16,384 |
| sink / recent | 1,312 / 0 | 1,312 / 10,496 |
| Query / attention 配额 | 50% / 50% | 75% / 25%，冷请求 attention 为空 |
| 普通 prefill chunk | 1,024 | 2,048 |
| prefill attention | 每 1,024 tokens；stride 32；half-life 8,192 | 未实现 |
| pressure selection | attention | 时间顺序策略 |
| guided Query | 配置上限 4,096；60 题日志实际未触发 | 上限 512，本轮实际执行 |
| 并发 | 原实验串行 | 8 个客户端 worker，CPU 额度最多准入 6 路 |

temperature=1、top_p=.95、top_k=20、min_p=0、presence=0、repetition=1、
thinking=4,096、最大输出=8,192 和逐题 seed 均已核对一致。
KVMem 页长均为 32。当前保留原版 MeanK CUDA 后端、编译与 CUDA Graph、
异步历史准备和 CPU-only 历史存储。

缺失策略已经由代码及日志确认；但未做同 NVFP4 模型、同预算的策略消融，不能将
全部差距直接归因于量化，也不能保证补齐后必然恢复到 98.75%。正确的下一轮验证
应先补齐这些路径，再冻结以上预算及策略，检查首次选页真正消耗了 prefill 分数，
最后运行相同 60 题，分别报告质量、prefill、混合负载 decode 和采集开销。

## 本轮分项结果与证据完整性

| 任务，各 15 题 | 当前覆盖率 | 当前全对 | qw3 attention 覆盖率 / 全对 |
| --- | ---: | ---: | ---: |
| single | 100% | 15/15 | 100% / 15/15 |
| multikey | 100% | 15/15 | 100% / 15/15 |
| multivalue | 23.33% | 0/15 | 95% / 12/15 |
| multiquery | 50% | 0/15 | 100% / 15/15 |

完整输入哈希、唯一 Query 重放、qw3 CUDA scorer、32K 上限及 Query history 提交
均通过；60 个终态历史长度等于输入 + 输出 − 1，且完成回收。
输出 attention 的接受、scope、计数和归一化检查均通过；所有首次选页的历史为零。
这些证明当前路径按记录执行，不证明与 qw3 的完整上下文策略等价。

150 个目标数字均出现在最终选中页中，147 个名称、118 个完整证据句被完整覆盖。
漏答 76 个数字，其中 74 个的名称和数字均被选中。最终页的包含关系不足以证明
prefill 时形成的 KV/recurrent 状态、证据完整性和模型行为正确。
金标及针的位置只用于离线审计，从未参与推理或选页。

## 性能及中断边界

| 运行段 | 成功题数 | 时长 | 成功输入 / 运行秒 | 成功输出 / 运行秒 |
| --- | ---: | ---: | ---: | ---: |
| 第一段，随后服务中断 | 45 | 2,291.70 s | 5,146.66 token/s | 4.79 token/s |
| 相同配置补跑 | 15 | 799.37 s | 4,918.27 token/s | 5.76 token/s |

这里是包含 prefill、选页、装配和生成的有效处理率，**不是纯 prefill kernel 或稳态
decode 吞吐**。第一段还包括未完成工作的耗时。首次准入到最后回收共 3,393.01 秒，
包含中断及重启空档；15,726,105 个成功输入、15,574 个输出对应整体 4,634.85 输入
token/s、4.59 输出 token/s。不能用该短输出混合任务替代长输出稳态基准。

成功尝试平均 HTTP 耗时 286.73 秒，含本次尝试 CPU 准入等待为 307.31 秒；
平均 TTFT 250.26 秒，平均生成阶段 34.95 秒。按所有请求生成时间加权，
单请求输出速度 7.40 token/s，含共享负载停顿；不是合计 decode 吞吐。
这些逐题延迟不包含之前失败尝试及其队列等待，不能作为完整用户重试延迟。

API 管理进程在第一段向引擎发送 SIGTERM，随后正常退出；信号来源未确认。
退出前没有捕获到模型异常，内核日志不可读，不能断言 OOM 或工具会话是原因。
45 份已完成答案原样保留，仅补跑 15 份未完成请求；低分答案没有重试。
服务源代码及配置在两段之间未变；只修复测试入口的错误取消时机以免失败继续扩散。

历史始终在 CPU，所有采样的 KV 磁盘读写量及 NVMe 容量为零，无 swap。
监控中唯一匹配 `archive` 的打开文件是 `capture/archive.tmp` 诊断 JSON，非 KV 内容。
100 GiB 历史额度下主机可用内存最低约 0.30 GiB，说明余量仍不足；不能称为生产稳定
容量验收。最终请求、待刷新任务及 GPU owned blocks 均为零。测试服务退出后恢复
此前带跨轮检查点的 `async-refresh-20261014/profile.json`。

## 产物

- 摘要：`docs/benchmarks/niah256k-attention-20261014.json`。
- 原始运行：`artifacts/niah256k-attention-20261014/`。
- 逐题结果及审计：`run/summary.json`、`run/offline-audit.json`、`run/attention-audit.json`。
- 正确 qw3 attention 配对审计：`qw3-attention-comparison.json`，由同目录
  `compare_qw3_attention.py` 校验输入、采样和实际 attention 事件。
- 中断与恢复：`attempt-01/`、`attempt-02/`、`interruption.json`、`recovery-audit.json`。
- 60 个终态归档证据：`combined-terminal-capture/archive.json`。

本次对外结论应为：**输出 Action 功能运行成功，但冷输入 attention 策略未迁移，
256K NIAH 质量验收未达到 qw3 水平。**
