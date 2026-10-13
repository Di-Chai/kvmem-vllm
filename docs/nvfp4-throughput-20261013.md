# NVFP4 与 Q8 吞吐比较：2026-10-13

本实验在同一套 KVMem + vLLM + DFlash2 服务中，将目标权重从 Qwen3.8-27B
GGUF Q8_0 换为 NVIDIA 官方 NVFP4 checkpoint，复用先前短输入、长输入和多客户端
测试的输入与计数方法。auto 后端基线 84/84 请求成功；随后将 FP8 与 NVFP4
投影均改用原生 CUTLASS，矩阵及短输入重复测试共 102/102 请求成功。短 C32 的
整批吞吐从 NVFP4 auto 的 426.16 提升至 **712.80 token/s**，短 C8 共同稳态三次为
386.54–392.75 token/s。性能改善已实测，但 32K 强制长度输出仍有质量异常：auto
为 5 条，CUTLASS 为 14 条，不能把这些服务吞吐称为有效文本吞吐或质量验收。

## 模型与量化范围

使用 [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/482ca0f3832238542f8f5295dde86b5f22711d80)，
固定 revision **`482ca0f3832238542f8f5295dde86b5f22711d80`**。
模型配置的 `quant_method=modelopt`、`quant_algo=MIXED_PRECISION`，vLLM 据此选择
`modelopt_mixed`；不手动覆盖为 `modelopt_fp4`。

该 checkpoint 的量化范围如下，不能理解为所有权重均为 4 bit：

| 模块 | 数量 | 权重／输入激活 |
| --- | ---: | --- |
| 64 层 MLP 的 gate、up、down 投影及 `lm_head` | 193 | NVFP4 W4A4，group size 16 |
| 48 层线性注意力的三个投影及 16 层完整注意力的四个投影 | 208 | FP8 W8A8 |

模型卡声明 ModelOpt 0.48.0，而实际配置记录 producer
`0.47.0.dev80+g913f5e224`，本实验保存原始配置，不推断二者相同。
量化配置排除 MTP 模块；本服务仍使用独立 BF16 **DFlash2**，每次 7 个推测 token，
不是原生 MTP 路径。目标 KV 保持 `fp8_e4m3`，草稿 KV 保持 BF16。

与本地 BF16 配置比较，文本架构和 `text_config` 完全相同。词表及 chat template
文件相同，但 tokenizer 的 padding 元数据有差异，因此实测继续指定原 BF16 tokenizer，
保证输入处理条件一致。完整检查记录在
`artifacts/nvfp4-20261013/metadata-validation.json`；其中权重是否存在是检查时刻的
快照，权重下载完整性与推理结果须使用后续证据。

## 启动配置

新增示例 [qwen-nvfp4-fp8-dflash-multi-user.json](../examples/qwen-nvfp4-fp8-dflash-multi-user.json)
使用原有多用户准入与租户隔离。调整本地模型路径，并按
[多用户服务文档](multi-user-serving.md) 准备 `users.json` 后执行：

```bash
python -m vllm_kvmem serve \
  --config examples/qwen-nvfp4-fp8-dflash-multi-user.json \
  --run-dir runs/nvfp4-check --dry-run
python -m vllm_kvmem serve \
  --config examples/qwen-nvfp4-fp8-dflash-multi-user.json \
  --run-dir runs/nvfp4-serving
```

`users_file` 的相对路径以 profile 所在目录为基准；示例因此读取
`examples/users.json`，可以在自己的 profile 中改成用户配置的绝对路径。
运行目录必须是新目录。dry-run 仅验证配置，不导入 vLLM／Torch、不分配 GPU，
也不验证权重内容或 kernel 能否运行。

NVFP4 必须显式设置 `model_format: "safetensors"`；`model` 与 `hf_config_path`
均指向同一个量化 checkpoint 目录，不能让后者继续指向 BF16 配置而丢失量化元数据。
删除 GGUF 专属 `mm_proj`，设置 `q8_tensorcore: false`。launcher 使用原生
safetensors loader，仅启用 KVMem 插件，量化由 vLLM 根据 checkpoint 配置选择。
未填写 `model_format` 的既有 profile 仍默认使用 GGUF。

通用示例保持 `max_num_seqs=max_inflight=8`、显存利用率 0.72 和排队超时 120 秒。
本次对比实验另用 32 请求、0.80 显存利用率、3600 秒排队超时的 profile，二者应区分；
实验的客户端数不构成任意长短混合流量或生产延迟保证。

## 同工作负载比较方法

设备和软件沿用此前测试：单张 RTX PRO 6000 Blackwell 96GB、vLLM 0.30.0、
Torch 2.13.0+cu130；保持 eager／同步调度、单 worker、K131072／G65536／B128、
最大上下文 262144、batch token 预算 4224、长 prefill 分块 1024、`prefill_cadence=8`。
共享 CPU 读缓存 16 GiB，保留 packed assembly、直接 pinned 读取与异步归档退休。

比较基线为[先前长文测试](long-context-throughput-20261013.md)中相同 32 请求配置的
GGUF Q8 Tensor Core 结果。此前将活跃请求限制为 8 的短输入对照是另一种准入设置，
不能混作同配置基线。

每请求 temperature=0、seed=104729，输出固定 1024 token，`ignore_eos=true`，
提示关闭 thinking。短输入继续使用同一 152-token 写作提示；32768、65536、134276
token 长输入直接提交原有冻结 token IDs，不重新套模板。134276-token 输入完整保留
原始 134067-token 前缀并追加写作问题。输入哈希和基线结果哈希记录在
`artifacts/nvfp4-20261013/measurement-plan.json`。

分别报告三种口径：

- **共同稳态 decode**：逐个累计 SSE `token_ids`，核对最终 usage；每请求裁掉前后
  至少各 64 token 后取共同时间窗，保留窗内停顿，使用原生快照检查有效覆盖与 prefill。
- **整轮吞吐和 TTFT**：从客户端提交计时，包含预填充、排队和结束开销。
- **实际活跃 owner 区间**：如果 32 个客户端受容量限制而不能同时 decode，按严格
  reservation／owner 条件分析实际并行数；这些局部区间不能标为“32 路 decode”。

量化后的权重与激活、不同进程和测量时段可能改变输出 token 路径、推测接受率及运行
状态。各档一次测量只能作为本轮服务表现，不能推导纯 kernel 加速或质量等价；原 Q8
32K 单路两次相同输入输出已出现 38.17／47.22 token/s 波动，比较时保留这两个观察值。
内容检查只验证计数、明显异常和重复等，不替代任务质量评测。

## 网络调研与参数候选（查阅于 2026-10-13）

以下资料用于决定本地对照顺序。网页的吞吐不是本机成绩，不能从不同硬件、量化、
推测方法和输入内容的最高值推导出唯一“最快配置”。官方文档说明能力和参数，社区
原始测试提供可复核的案例；最终选择仍以本机相同工作负载结果为准。

| 原始来源及测量日期 | 环境／方法 | 对本次实验的意义 |
| --- | --- | --- |
| [NVIDIA 模型卡](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)，模型发布 2026-09-08 | 官方混合 NVFP4/FP8；推理验证硬件 GB300；示例 TP4、FP8 KV、seq32、batch token 32768、利用率 .85 | 是本 checkpoint 的原始使用说明，不是单卡 SM120 最快参数的证明。 |
| [vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)，页面未给统一测量日期 | 含 SM120、vLLM `0.26.1rc1.dev608+g99a10304d` 验证；NVFP4 使用真实 CUTLASS 内核；另给 DFlash2 K7 和 MTP 用法 | 单张 32GB 5090 的 eager 要求出于该例显存不足，不能直接推为本机 96GB 卡的要求。 |
| [Deep-AI-Evo 原始报告](https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark)，2026-08-15 | RTX PRO6000 96GB、vLLM .21/.26、unsloth NVFP4、MTP k2、利用率 .92；报告短 C1 132.3、C8 聚合 654.1 tok/s | GPU 相近，模型 recipe、推测方法、软件与计时不同；不能把它当本次 NVIDIA checkpoint + DFlash2 + KVMem 的同条件成绩。 |
| [Prefill Systems 原始报告](https://prefillsystems.com/benchmarks/qwen3-8-rtx-pro-6000)，2026-09-06 至 07 | RTX PRO6000、Inferact NVFP4、FP8 KV、MTP k3、最多 8 活跃请求；约 1K 输入／1K 输出的公开服务路径聚合 553 tok/s | 报告明确旧测量使用 Inferact，后来服务才换 NVIDIA；不据此声称后者已有同值测量。 |
| [SM12x DFlash2 发布测试](https://raw.githubusercontent.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2/main/BENCHMARKS.md)，证据目录 2026-08-26 | RTX5090、vLLM .27.1 加社区补丁、NVFP4 target/draft/KV；三轮预热、五轮测量；C4 代码 586.6、叙述 343.5 tok/s | 输入内容影响很大。[另一组匹配实验](https://raw.githubusercontent.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2/main/EVIDENCE.md)的 C4 代码为 616.33；不能只挑该最高值与本次写作提示比较。 |

社区 DFlash2 的[实际 compose](https://raw.githubusercontent.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2/main/compose.yaml)
及 [README](https://raw.githubusercontent.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2/main/README.md)
还明确包含 target `FULL_AND_PIECEWISE` 图、K7 捕获尺寸 `[8,16,24,32]`、eager drafter、
专用 XQA stream、autotune、seq4、batch4096、长 prefill 2048 和 BF16 GDN 状态。
这是基于旧版的 51 文件 overlay，没有 KVMem；不直接运行或移植该补丁作为本实验配置。

网络资料确定了以下验证方向；已完成的后端选择见文末实测，图执行与 MTP 对照仍未完成：

1. **先比较同精度 GEMM 后端。** [vLLM 0.30 kernel 配置](https://docs.vllm.ai/en/v0.30.0/api/vllm/config/kernel/)
   支持按量化类型覆盖后端。本地 auto 列表将 FlashInfer CUTLASS 放在 FlashInfer B12x
   之前；可以固定 NVFP4 W4A4 比较二者，FP8 注意力先保持 auto。先检查小矩阵的数值、
   启动日志的实际后端，再做相同 C1／C8／C32 测量；不能因 B12x 专为 SM120 提供就
   假定它更快。[FlashInfer #4253](https://github.com/flashinfer-ai/flashinfer/pull/4253)
   于 2026-07-31 合并，对原有 B12x 的同步更新在 PRO6000 上报告约 0.997 倍，
   也说明“更新了 kernel”不等于该形状取得加速。
2. **再检查 autotune 和图执行的收益。** [vLLM CUDA graph 设计](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
   区分完整 decode 图与混合／prefill 的 piecewise 图，代价包含额外显存和捕获时间。
   用独立原生 vLLM 的 eager／target `FULL_DECODE_ONLY` 对照量化禁图代价，draft
   保持 eager，K7 的捕获 token 尺寸取请求数的 8 倍；这组仅作诊断，不标为 KVMem 成绩。
3. **推测方法最后单独比较。** 原生 MTP k2／k3 与 DFlash2 K7 的目标调用、接受率和
   状态生命周期不同。当前 KVMem `DraftContract` 只接受 DFlash2；MTP 应先以原生
   服务对照验证，不能仅换参数就认为 KVMem 事务语义保持不变。

本地源码还有两个明确边界：KVMem `runner.py` 要求 eager，逐次 capture sink 不能
通过删除检查自动适配 CUDA graph；vLLM 0.30 `config/vllm.py` 本身也拒绝 DFlash 与
async scheduling 的组合。若图执行对照显示明显收益，需要实现并验证 KVMem 的图
捕获／重放数据生命周期，不能将未执行的图参数写成已优化。

launcher 为上述同精度对照提供可选 `linear_backend_per_quant`，例如：

```json
{"linear_backend_per_quant": {"nvfp4_w4a4": "cutlass", "fp8_w8a8": "cutlass"}}
```

它原样转为 vLLM `--kernel-config` 的同名映射，未填写时保持 vLLM auto。
允许的 key 为 `nvfp4_w4a4`、`fp8_w8a8`；前者可选 `auto`、`cutlass`、
`flashinfer_cutlass`、`flashinfer_b12x`、`b12x`，后者可选 `auto`、`cutlass`、
`flashinfer_cutlass`、`b12x`、`torch`。拒绝未知拼写、空映射、GGUF 误用以及改变
激活精度的 weight-only／emulation 路径；是否已安装所需依赖、硬件及具体层形状是否
支持仍由 vLLM 在启动时验证。NVFP4 示例采用本机 eager 路径实测的双原生 CUTLASS；
这不代表其在 CUDA Graph、其他 GPU 或模型上也优于 FlashInfer。

## 测量结果

以下为 **auto 后端基线**，不包含后续 GEMM 后端 A/B。启动日志确认 NVFP4 使用
`FlashInferCutlassNvFp4LinearKernel`，FP8 投影使用 `FlashInferFP8ScaledMMLinearKernel`。
所有请求均成功，无 HTTP 拒绝或失败，每请求最终 usage 与 SSE 精确计数均为
1024 token。Q8 列均使用相同 API／native 上限 32 的旧结果。

### 整批 HTTP 吞吐与首 token 延迟

吞吐单位 token/s，时间单位秒。整批 wall 包含 prefill、排队、decode 与响应结束。
这些数字计入 `ignore_eos` 后的强制输出；它们衡量服务处理量，不代表有用文本处理量。

| 输入 token／客户端 | 成功 | Q8 HTTP | NVFP4 HTTP | Q8 TTFT p95 | NVFP4 TTFT p95 | NVFP4 wall | 实测最高 owner（Q8 → NVFP4） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 152／1 | 1/1 | 56.51 | 29.68 | 0.55 | 0.55 | 34.50 | 1 → 1 |
| 152／8 | 8/8 | 244.95 | 223.39 | 1.83 | 1.15 | 36.67 | 8 → 8 |
| 152／32 | 32/32 | 203.85 | 426.16 | 131.58 | 39.08 | 76.89 | 14 → 18 |
| 32768／1 | 1/1 | 12.37 | 18.91 | 55.48 | 8.76 | 54.14 | 1 → 1 |
| 32768／8 | 8/8 | 16.25 | 65.31 | 462.55 | 81.51 | 125.43 | 8 → 8 |
| 32768／32 | 32/32 | 19.63 | 99.18 | 1549.03 | 272.22 | 330.38 | 10 → 12 |
| 65536／1 | 1/1 | 10.22 | 16.76 | 81.00 | 16.06 | 61.10 | 1 → 1 |
| 134276／1 | 1/1 | 4.89 | 10.92 | 186.30 | 49.08 | 93.78 | 1 → 1 |

短输入 C32 的 HTTP 吞吐为 Q8 的 **2.09 倍**，32K C32 为 **5.05 倍**。
NVFP4 原生页池为 1637 页，Q8 同配置基线为 1316 页；实际稳定 owner 数分别通过
采样确认，不能仅按空闲显存推算。旧 Q8 上限 8 的另一组短 C32 对照为
249.95 token/s，配置不同，单独保存在比较文件的 `additional_references` 中。

### 全部客户端共同稳态 decode

以下是裁掉每请求首尾至少各 64 token 后的共同窗口；不含 prefill，也不移除窗内
停顿。原生确认要求至少 90% 覆盖。32 客户端两组均不存在完整共同窗口，故为
`not_applicable`，不能报告“32 路纯 decode”。

| 输入／客户端 | Q8 共同稳态 token/s | NVFP4 共同稳态 token/s | NVFP4 窗口秒 | 原生确认覆盖 |
| --- | ---: | ---: | ---: | ---: |
| 152／1 | 60.79 | 29.83 | 29.967 | 98.92% |
| 152／8 | 267.09 | 233.69 | 29.082 | 99.41% |
| 152／32 | N/A | N/A | — | — |
| 32768／1 | 38.17 | 21.81 | 41.030 | 99.21% |
| 32768／8 | 154.74 | 199.15 | **3.063** | **90.31%** |
| 32768／32 | N/A | N/A | — | — |
| 65536／1 | 52.34 | 22.18 | 40.401 | 99.50% |
| 134276／1 | 44.37 | 22.42 | 39.786 | 99.76% |

32K C8 的共同窗口很短，不能外推为持续高并发成绩。Q8 32K 单路随后同输入、
同输出复测为 47.22 token/s；前后漂移也说明本表不是仅改变上下文长度的受控曲线。

### 严格确认的实际 owner 区间

离线分析要求连续三个样本的 reservation 集合稳定且数量等于 owner 数，所有 owner
已完成各自 prompt、处于 SSE 首尾裁剪边界内，且无观测到的 prefill、refresh 或退休。
允许原生请求总数大于 owner 数，但不把多出来的等待请求计作 decode。有效区间的
零进展时间仍留在分母中；排除原因、时间及全部区间保存在原始分析文件。

| 工作负载 | 实际 decode owner | 严格区间合计秒 | SSE token/s | 占整批 wall |
| --- | ---: | ---: | ---: | ---: |
| 短 C32 | 18 | 31.181 | 508.78 | 40.55% |
| 短 C32 | 14 | 29.926 | 410.91 | 38.92% |
| 32K C8 | 8 | 2.766 | 199.90 | 2.21% |
| 32K C8 | 4 | 25.391 | 100.31 | 20.24% |
| 32K C32 | 12 | 29.431 | 290.34 | 8.91% |
| 32K C32 | 6 | 5.024 | 142.72 | 1.52% |
| 32K C32 | 4 | 3.016 | 103.46 | 0.91% |
| 32K C32 | 2 | 19.600 | 46.89 | 5.93% |

这些局部速率不能替代整批 HTTP 速率，也不能证明每个 GPU step 都包含同样多的
请求。原生快照是约 0.25 秒采样的归档提交近似，SSE 是采样端点间实际收到的精确
token 数；两者时间边界与提交延迟不同。32K C32 仍有 178.13 秒被标为混合／prefill，
另有 42.94 秒因已准入 reservation 与已捕获 owner 不一致而排除。

短 C1 的严格原生请求验证频率从 Q8 的 17.81 降至 9.01 次/秒，平均接受量从
3.39 降至 3.24 token/次；接受长度变化较小，主要观察差异在调用节奏。
以 `区间秒 × owner数 / 请求验证次数` 估算的迭代间隔为 56.16 → 111.03 ms。
短 C8 同口径为 94.35 → 114.73 ms，但接受长度从 3.11 升至 3.38。
这不是 GPU kernel 计时，不能据此认定具体算子或 CPU 环节是根因。

### 内容、启动与资源清理

四组内容审计覆盖全部 84 请求及 86016 token，计数、时间线和输出 SHA 均一致。
短输入 41 请求、64K 单路和 134K 单路未出现 EOS／工具控制标记，摘录是技术评审
与伪代码，并在固定长度截断；这不等于完成了要求的完整评审或通过质量评测。

32K 的 41 请求中 **5 条出现工具控制语法及 EOS，共 31 个 EOS**：C8 的客户端
0、5，以及 C32 的客户端 8、11、12（零基编号）。首 EOS 在第 37–75 个零基 token
位置，随后因 `ignore_eos` 又生成 948–986 token。四条后续转入技术评审或显式
thinking；C32 客户端 11 在首尾摘录中反复生成旧 coding context 的“读取测试文件”
与工具片段。自动重复检查只覆盖周期不超过 64 token 的精确相邻重复，未命中该循环，
因此不能将自动标记为零解释为没有重复。生成的工具文本从未执行。上述输出仍保留
在原始服务计数中，**不把 EOS 后强制输出计作有用文本**，也不静默删掉异常请求重算成绩。

启动日志记录模型加载 13.17 秒、占用 22.85 GiB；引擎 profiling／KV 分配／warmup
阶段 369.60 秒，FlashInfer autotune 保存 126 个配置。此前已做独立 kernel warmup，
这些阶段数据不是清空所有缓存的完整冷启动实验，也不包含模型下载。首个短 C1 有
四条推理期 Triton JIT 日志，时间精度为一秒；都早于该请求 3.18 秒开始的稳态窗口，
因此不能把稳态单路降速全部归因于这四次编译。长输入阶段仍有 autotune 未覆盖形状
的回退警告及后续 packed assembly JIT；保留日志，不声称所有形状已预热。

短 C1 最初三个原生样本为文件尚未产生的 `FileNotFoundError`，发生在稳态前，
稳态原生覆盖仍为 98.92%、无窗口内违例；其余七批无原生采样错误。
矩阵完成后的冻结快照确认累计完成 **84 请求**，活动请求、reservation、owned blocks、
refresh／connector pending、退休请求、归档占用、待下载、待写入字节及 CPU cache
占用均为 0，空闲页 **1636/1637**。这证明本轮资源归还，不代表长时间稳定性验收。

### 证据路径

所有文件位于 `artifacts/nvfp4-20261013/`：

- `baseline-summary.json`：全部八批指标、Q8 同配置对照、源文件 SHA、结束资源快照。
- `content-review.json`：人工摘录检查范围、五条异常定位和质量限制。
- `short-c1-c8-c32/`、`32k-c1-c8-c32/`、`64k-c1/`、`134k-c1/`：各自的
  `results.json`、`cohorts-strict-reservations.json`、`analysis-summary.json`、
  `comparison-vs-q8.json` 和 `content-audit.json`。
- `short-c1-c8-c32/regression-and-scaling-evidence.json`：调用节奏分解和首轮 JIT 时间边界。
- `matrix-status.json`、`measurement-plan.json`、`frozen-manifest.json`、
  `source-change-validation.json`：完成状态、输入／基线／冻结源码和运行版本证据。
- `server.log`、`kernel-warmup.json`、`baseline-final-capture/`：实际 kernel 选择、
  启动预热与矩阵完成后的资源快照。

## GEMM 后端 A/B 结果

在相同 NVFP4 checkpoint、DFlash2、输入、输出长度、准入与显存利用率下，将
`nvfp4_w4a4`、`fp8_w8a8` 均设置为 `cutlass`。profile 唯一差异为这两个覆盖项，
移除 `--kernel-config` 后启动 argv 相同；日志确认实际选中
`CutlassNvFp4LinearKernel` 和 `CutlassFP8ScaledMMLinearKernel`。
这一节比较 **NVFP4 auto 与 NVFP4 原生 CUTLASS**，不把 Q8 混作此组控制。

### 微基准支持什么结论

NVFP4 三后端微基准使用实际 checkpoint 的第 0 层 MLP 和 `lm_head` 权重、
固定随机 BF16 激活、M=8/64。原生 CUTLASS 的 MLP eager wall 耗时相对
FlashInfer CUTLASS 改善约 1.82–3.54 倍；`lm_head` M8 则较慢，B12x 在这个
小形状更有优势。测试样本输出数值相同，但没有覆盖所有层、真实模型激活或完整
BF16 质量参考。该 NVFP4 微基准未加载服务 autotune cache，不能直接换算服务收益。

FP8 使用独占 GPU 重测的 `kernel-ab/fp8-isolated.json`，36 项数值和计时检查通过；
`fp8-results.json`、`fp8-ab-fast.json` 因进程重叠，其计时全部排除。
将服务真实 autotune cache 载入后，六个量化加 GEMM 形状的原生 CUTLASS
eager wall 耗时比 tuned FlashInfer 低，比例为 5.50–8.46 倍；但图重放隔离后的
GPU 时间中，tuned FlashInfer 在这六个形状全部更快，在包括纯 GEMM 的十二个
配对中有十一个更快。因此证据支持本机 **eager 调用路径的额外开销较大**，不能
声称 CUTLASS 的底层 GPU GEMM 普遍更快。

源码还显示当前 FlashInfer FP8 路径每次调用重新创建 cuBLAS runner 及其空算法
缓存，非负 tactic 会再次查询算法。host 提交计时与 eager／图重放差异支持这一
开销方向，但没有 profiler 分离 Python、分配、算法查询和 launch 的占比。
eager CUDA event 也包含 host 提交不及产生的 GPU 空闲，不能当作纯 kernel 时间。
本服务同时修改两个后端，不能分别归因二者贡献；图重放仅用于诊断，KVMem 服务仍是 eager。

### 相同工作负载的服务结果

四组共 84/84 请求成功，每请求 1024 token，计数与 SHA 全部通过。吞吐单位
token/s，时间单位秒；共同稳态只有通过最小时长和原生确认的窗口才显示速率。

| 输入／客户端 | auto HTTP | CUTLASS HTTP | auto TTFT p95 | CUTLASS TTFT p95 | CUTLASS wall | CUTLASS 共同稳态／窗口秒 | 实测最高 owner |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 152／1 | 29.68 | 55.49 | 0.55 | 0.45 | 18.45 | 56.91／15.727 | 1 |
| 152／8 | 223.39 | 372.51 | 1.15 | 0.70 | 21.99 | 392.75／16.249 | 8 |
| 152／32 | 426.16 | **712.80** | 39.08 | 23.24 | 45.97 | N/A | 18 |
| 32768／1 | 18.91 | 32.32 | 8.76 | 6.98 | 31.69 | 41.14／21.732 | 1 |
| 32768／8 | 65.31 | 85.78 | 81.51 | 70.67 | 95.50 | N/A：共同窗过短 | 8 |
| 32768／32 | 99.18 | **119.47** | 272.22 | 236.07 | 274.28 | N/A | 12 |
| 65536／1 | 16.76 | 26.42 | 16.06 | 15.68 | 38.76 | 44.39／20.139 | 1 |
| 134276／1 | 10.92 | 14.44 | 49.08 | 47.70 | 70.92 | 44.16／20.269 | 1 |

短 C32 HTTP 吞吐提高 **67.26%**；32K C32 提高 **20.45%**，后者包含下述
异常及 EOS 后强制生成，不是有用文本增益。实际最高 owner 与 auto 相同，短输入
为 18、32K 为 12；页池略增至 1641 页（auto 为 1637），不能假定准入波次完全一样。

严格局部区间中，短 C32 的 18-owner 合计 15.845 秒、818.42 token/s，14-owner
为 13.065 秒、620.76 token/s；32K C32 的 12-owner 合计 29.903 秒、510.89 token/s。
这些依然是无观测 prefill／refresh、reservation 稳定的局部区间，不能标为全 32 路
decode 或替代上表 HTTP 吞吐。32K C8 的共同裁剪窗口只有 0.580 秒，不满足 2 秒
门槛；严格 8-owner 仅 0.251 秒且为 `insufficient_evidence`，保留原始数字但不报告
合格 8 路 decode 速率。

### 短输入重复与调用节奏

矩阵后对 C1、C8 各追加两轮，共 18/18 请求成功；18 条输出无 EOS、工具控制标记
或自动长重复标记。初轮加重复共三次，两个并发档的输出 token 序列分布分别完全相同。

| 并发 | 初轮共同稳态 | 重复 1 | 重复 2 | 三次中位数 | 共同窗口范围（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 56.909 | 55.973 | 55.982 | 55.982 | 15.727–15.990 |
| C8 | 392.754 | 391.688 | 386.542 | 391.688 | 16.249–16.464 |

C1 的原生请求验证频率三次为 16.37–16.70 次/秒、平均接受 3.32–3.34 token/次，
估算迭代间隔 59.87–61.10 ms；auto 基线为 9.01 次/秒、3.24 token/次、111.03 ms。
C8 为 124.41–126.39 次/秒、3.05–3.09 token/次、63.29–64.30 ms；auto 为
69.73 次/秒、3.38 token/次、114.73 ms。C8 接受长度反而较短，较快的调用节奏
仍带来更高吞吐。原生统计是采样推算，不是逐 kernel profile。

三次观察范围不是置信区间；auto 只有一次控制，没有随机化测量次序。
重复 C1 的 TTFT 降至 0.10–0.12 秒，而初轮为 0.45 秒，存在首次 JIT／缓存状态
差异。两种服务首请求均有四类 JIT 警告；日志从首次 API 消息至开始监听约
402 → 53 秒，但冷／热缓存及预热历史不同，不将此称为受控冷启动加速。

### 输出质量与清理边界

累计 **102/102 请求、104448 token** 的计数及摘要一致。初轮短 C1 有两个原生
文件尚未产生的采样错误，位于稳态前；其他矩阵及重复批次无采样错误。短输入、
64K、134K 的检查未见 EOS／工具控制标记，长输入摘录为评审和截断伪代码，仍未作
任务质量验收。

32K 的 41 请求中 **14 条出现工具控制语法和 EOS，共 533 个 EOS**，多于 auto
的 5 条／31 个。C8 客户端 2、3、4 各有 65 个 EOS，C32 客户端 25、26 各有
136 个 EOS，首尾摘录呈现明显工具／控制标记循环；至少这五条不能视作正常长文。
其余异常后续可能转入显式 thinking 或技术评审。14 条中 13 条首 EOS 位于零基
token 37，另一条为 42，此后继续强制生成 981–986 token。保留全部服务计数，但
不将这些 token 算作有用文本，亦不据此直接推断某个 kernel 的数值错误。

`matrix-final-capture/` 保留 84 请求后、重复前的清理快照；
`performance-final-capture/` 则确认性能测量全部完成后的累计 **102 请求**，活动请求、
reservation、owned blocks、refresh／connector pending、归档占用、退休、待下载、
待写入字节和 CPU cache 占用均为 0，空闲页 **1640/1641**。
`performance-cleanup-validation.json` 保存该冻结快照的 SHA 和核验结果。

性能测试结束后，另用 4 个独立用户并发提交干净的中性文档，将每份不同的随机
检索值放在约 50% 位置。实际输入为 32768／32769／32768／32768 token，关闭
thinking，不提供工具，不设置 `ignore_eos`，输出上限 64 token。**4/4 回答精确
匹配各自检索值**，未返回其他用户的值，也没有工具调用；均以 `finish_reason=stop`
正常结束，实际输出为 20／19／17／19 token，reasoning token 均为 0。

证据在 `quality-probe/natural-stop-32k-c4/`，保留输入模板、位置、哈希、响应与
usage。这是独立长文检索和返回隔离的窄范围检查，未覆盖多轮会话、KVMem 刷新或
通用任务质量；不能据此宣称前述固定长度输出问题已修复。106 请求后的资源回收
另存于该目录的 `final-capture/`，不覆盖性能测试的 102 请求快照。

### A/B 证据

证据根目录为 `artifacts/nvfp4-20261013/`：

- `kernel-ab/nvfp4-three-backends/results.json`、`dispatch-audit.json`：NVFP4 样本和后端分发核查。
- `kernel-ab/fp8-isolated.json`、`fp8-analysis.json`、`timing-exclusions.json`：有效 FP8 微基准、host／图重放解释及排除项。
- `backend-ab/native-cutlass/launch-review.json`：配置、实际 kernel、内存与启动日志比较。
- `backend-ab/native-cutlass/` 下四个矩阵目录及 `short-repeat-c1-c8/`：原始
  `results.json`、严格 cohort、内容审计及 `comparison-vs-nvfp4-auto.json`。
- `backend-ab/native-cutlass/32k-c1-c8-c32/content-review.json`：14 条异常定位及人工检查范围。
- `backend-ab/native-cutlass/matrix-and-repeats-summary.json`、`short-three-runs-summary.json`：102 请求汇总、三次范围与原生节奏。
- `backend-ab/native-cutlass/performance-final-capture/`、`performance-cleanup-validation.json`：102 请求后的最终资源归还。

比较 helper 已用真实 C1／C8／C32 数据做零差回归核对，另由独立代理复核
NVFP4 控制映射、重复轮次保留及 N/A 与 strict cohort 分离；这些均为 CPU 离线验证，
不额外占用 GPU。

### 安装与配置验证

启动器及 decode 统计相关 CPU 检查共 116 项通过。新 wheel 已安装到不含 Torch／
vLLM 的独立环境，console dry-run 与实测服务的 native argv 一致，且不创建运行
目录。分发包逐文件核对记录见 `package-validation.json`，安装记录见
`wheel-installation-validation.json`；NVFP4 示例使用实测双原生 CUTLASS 后端。
