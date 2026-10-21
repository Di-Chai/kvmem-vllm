# 同一完整 32K 输入：KVMem 与原生 vLLM

同机、同一份完整 32,768-token 输入、同样 2,048-token chunk 和 16,424
batch 上限，原生为 **9,593.46 token/s**，KVMem 为 **6,927.91 token/s**，
吞吐低 **27.79%**。首 token 延迟均值由 **3.41566 秒**增至 **4.72985 秒**，
增加 **1.31419 秒（38.48%）**。

此前约 3,500 token/s 来自完整 256K 输入，并非同输入的 32K KVMem 测量。
本次补齐这一缺失对照，没有修改模型实现或生产默认参数。

## 协议与结果

RTX PRO 6000 Blackwell Server Edition，Qwen3.8-27B NVFP4，FP8 KV，MTP4，
编译/CUDA Graph，单活跃请求，max sequences=8，max model length=524,288，
关闭 prefix caching、async scheduling。每个服务先完整预热一次，再顺序测两次。
KVMem 沿用 K=36,864、G=16,384、CPU 历史 100 GB、sink=1,312、recent=0，
四项性能优化全开、原生 QK 复用关闭、guided 关闭、Query V2 multi alpha=.2。
2K chunk 只用于本次对照，不改变生产的 1K 默认值。

| 路径 | Chunk | TTFT 1 / 2，秒 | 合并 input/TTFT，token/s |
|---|---:|---:|---:|
| 原生，本次 | 2,048 | 3.41543 / 3.41589 | 9,593.46 |
| KVMem，本次 | 2,048 | 4.68114 / 4.77856 | 6,927.91 |
| 原生，历史对照 | 2,048 | 3.41672 / 3.41161 | 9,597.65 |
| 原生，历史较大 chunk | 16,384 | 3.09616 / 3.11453 | 10,552.14 |

历史原生 batch 上限为 16,384；本次两组均用 16,424。复测 2K 原生速度与历史
相差不到 0.05%。自动核验实际启动参数：去除 KVMem scheduler、connector、
middleware，并统一端口与 chunk 的控制方式后，其余原生参数逐项相同。

冷准备为 KVMem 75.14 秒、原生 66.12 秒，均不计入上表。预热也不计入。
两服务冷准备期间都有 FlashInfer 大形状 autotune 内存不足后退回默认 tactic
的警告，均成功启动；正式请求期间没有推理 OOM 或新的 JIT 编译警告。
KVMem 保留原配置 step_latency_interval=1 的非阻塞 CUDA event 计时，成本计入；
未启用 Torch/Nsight profiler 或 action 分项 profiling。

输入与采样参数完全相同；输入 SHA-256：
`fc74290ef4a52d923272ec24f19dc4ff10788e909ef0137d33385b2d97a409cd`。
输入是历史 256K RULER 题裁去无关填充后的性能夹具，并非新增正式准确率题组。
每次都完整提交 32,768 token，prefix cache 不参与输入复用。

## 这次排除了重选成本

请求仍携带同一个显式 Query 和 `kvmem_reselect=force`，但历史小于 K，调度器
跳过实际重选。三次请求均为 **0 次 pressure、0 次 Query selection、0 次恢复**。
参见 `src/vllm_kvmem/scheduler.py` 中 `num_computed_tokens >= selected_budget`
的触发条件。Query 图传播配置启用，但没有选页事件，因而没有传播执行成本。

attention 统计和构图确实运行：每请求 1,023 个 prefill 采样行、16 次原生
prefill 分数提交、54 次三层累计 fused scoring 调用，形成 4,086 条图边。
接受行核验、全量输入转发计数、最终完成栅栏均通过；没有 reference scoring
回退、Python decode commit、下载池耗尽或隔离槽。

因此，这 **1.314 秒**是该条件下 KVMem 完整路径的附加成本，不能归为压力重选
或 Query 图传播，也不涉及完整 256K 输入较大的平均注意力域。

## 当前能定位到的时间

| KVMem 正式请求 | 第一次 | 第二次 |
|---|---:|---:|
| 记录的 prefill 提交跨度，秒 | 4.6312 | 4.7245 |
| 提交调用内部 host 时间，秒 | 3.6579 | 3.6840 |
| 提交调用之间 host 间隔，秒 | 0.9734 | 1.0405 |
| CUDA model 事件区间累计，秒 | 3.6664 | 3.6906 |
| CUDA maintenance 事件区间累计，秒 | 0.0495 | 0.0517 |

每个正式请求记录 18 个 prefill 批次，共 32,768 token。KVMem 在 Query/模板边界
把末段拆成 2,016、25、7 token，前面是 15 个 2K 批次；原生为 16 个 2K 批次。
模型相同不意味着调度批次完全一样，这部分差异保留在实际路径成本中。

约 1 秒的提交间隔是下一步需要关注的线索，但它是 host 时间，不代表全部 GPU
空闲，更不能直接把它全算作新增开销。CUDA model 区间也含统计、捕获与提交
间隙，并非纯模型 kernel 时间；以上 GPU 与 host 两套计时不能相加。

每请求还会持续发布约 1.144 GB KV 历史，三次共 3.433 GB；SSD 读写均为零。
三次累计 CPU pack/hash/record-copy 分别为 2.113/2.082/2.084 秒，这些后台工作
可重叠，不能相加后从 TTFT 扣除。按累计计数差分，两次正式请求的构图准备
只有 .048/.056 秒，prefill 分数提交约 .023/.020 秒；单独构图准备不足以解释差距。

实测支持把后续调查重点放在正常 prefill 的历史捕获、归档、统计和提交衔接。
本轮尚未分别隔离这些模块的净开销，不宣称约 1 秒 host 间隔全由某个模块导致，
也未产生新的端到端优化收益。

## 完成性与证据

全部六次有效请求自然结束、思考完整、答案正确。每组内部输出 token IDs 相同；
两组之间不同（KVMem 101、原生 114 个输出 token），因此不据此比较 decode
吞吐。上表指标是输入 token / 首 token 等待时间，排除之后的完整 decode。
每组只有两次正式重复，支持此次观测，不构成跨题统计置信区间。

- `artifacts/matched-32k-prefill-20261021-r2/`：冻结 runtime、profile、输入引用、
  实际 launch、逐请求结果、接受行和资源释放证明、`comparison.json`。
- 同目录 `analyze.py`：核对同输入/采样、源码哈希、实际参数、完整结果与清理后
  生成比较；`scripts/benchmark_matched_prefill.py` 是 KVMem 性能诊断入口。
- `artifacts/matched-32k-prefill-20261021-r1/` 保留初次预热：旧长输入审计错误要求
  必须重选而中止。该预热及失败尝试不进入正式结果，未修改生产语义来迎合校验。
- 17 个既有 native benchmark / pressure interval 测试通过；两服务进程与 GPU
  所有权均已清理。DSH 未启动，完整 RULER 和 MAB 未恢复。
