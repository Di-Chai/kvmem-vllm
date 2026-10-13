# 原版 MeanK 并发 decode 与历史恢复实测，2026-10-13

本页记录 `context_mode: "meank"` 的真实模型测试。该模式将原版 KVMem
MeanK 选页、不可变 raw KV 历史和 dense re-RoPE 装配接入 vLLM，并用
DFlash2 加速生成。此前约 592 token/s 的成绩来自完整上下文、原生 MTP(k=3)
及 CPU offload，不能用来证明 MeanK 选页路径已经达到同样性能。
本轮八用户 32K＋16K 的 MeanK 稳态 decode 为 BF16 draft 345–421 token/s、
FP8 draft 339 token/s，尚未达到旧约 592 token/s 的结果。
核心实现、预算和未接入的可选原版功能见 [实现核对](meank-integration-20261013.md)。

## 32K 输入＋16K 输出，八个独立用户

所有行均使用同一份 32768 token 输入，每用户固定生成 16384 token。
吞吐单位为八路合计的已接受输出 token/s。

| 执行配置 | 稳态 decode | 包含 prefill 的整轮吞吐 | 完成时间（秒） |
| --- | ---: | ---: | ---: |
| 旧完整上下文＋MTP(k=3)＋CPU offload | 591.92 | 505.28 | 259.40 |
| MeanK＋BF16 DFlash2，7 个草稿 token，自动线性后端 | 392.71 | 279.47 | 469.00 |
| MeanK＋BF16 DFlash2，3 个草稿 token，自动线性后端 | 399.14 | 315.66 | 415.23 |
| MeanK＋BF16 DFlash2，3 个草稿 token，CUTLASS 线性后端 | 421.12 | 325.00 | 403.30 |
| 上一配置＋缓存／异步上传窗口位置偏移 | 345.08 | 284.11 | 461.34 |
| 上一配置＋FP8 DFlash2 线性层权重 | 339.17 | 277.73 | 471.94 |

以上六行均完整成功 8/8，原生指标确认同时 decode 八路、无等待和无抢占。
MeanK 各行的有效原生采样覆盖为共同窗口的 99.86%–99.92%。旧基线使用
48 GiB CPU cache，MeanK 配置上限为 100 GiB；这次单轮每用户的历史最多
48K，没有实际填满每用户 256K 的 CPU 上限。

减少草稿长度后稳态吞吐的观测增幅仅约 1.6%；切换 CUTLASS 的那次运行比
自动后端高约 5.5%。窗口偏移缓存改动后的整轮结果反而更低，不能声称该改动
已证明性能收益。不同运行的生成内容和草稿接受比例发生变化，这些单次测量
没有将数值变化、接受率和 kernel 成本完全隔离，不能全部归因于一个参数。
BF16 的最好观测值 421.12 token/s 仍低于旧 MTP 基线；345.08 这一结果同样
保留在报告中。FP8 草稿权重的 339.17 token/s 没有显示吞吐收益，所以它仍是
可选实验参数；示例采用 BF16 DFlash2、3 个草稿 token，并对目标的 NVFP4 和
FP8 线性层显式选择 CUTLASS。

普通 decode 已保留 CUDA Graph。相比旧原生 MTP 路径，这里采用不同的
DFlash2 草稿方法，并增加 accepted raw KV 的捕获、CPU 历史写入及必要的
MeanK 刷新事务。主动 stage-out 使终态可以释放 GPU 页，其带宽与后台处理
成本仍需计入服务吞吐。当前测量没有完成各项成本的独立归因；草稿长度、
CUTLASS 和 FP8 实验的完整结果如上，不能据此承诺消除全部差距。

旧 MTP 成绩是已完成的历史测量。此次 admission 修正还将共享 native pool
的物理 padding 计入页预算；旧入口预算不能直接当作当前版本的可准入人数，
须按新导出的页几何重新计算，原始吞吐记录本身仍保留。

32K 输入恰好等于 K，该组主要检验开启 MeanK 管理时的并发生成成本。
它不能独立证明从超过 K 的历史中选择 KV 后仍有同样的吞吐；更长历史和
暂停／恢复另行验证。

## 并发变化与超过 K 的输入

最终 BF16 draft 3＋CUTLASS 配置另测 4096 token 固定输出。该组输出长度
与前表的 16384 不同，整轮吞吐不能直接横比；其稳态窗口仍使用同样的准确
token 计数和原生并发确认。

| 并发 | 每路输入 | 每路输出 | 合计稳态 decode（token/s） | 整轮吞吐（token/s） | 完成时间（秒） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32768 | 4096 | 63.17 | 55.92 | 73.25 |
| 4 | 32768 | 4096 | 232.17 | 168.69 | 97.12 |
| 8 | 32768 | 4096 | 378.42 | 191.07 | 171.50 |
| 8 | 65536 | 4096 | 386.89 | 101.41 | 323.12 |

同样 32K 输入和 4K 输出下，四路、八路合计稳态 decode 分别是单路的
3.68 倍、5.99 倍，确认原版 MeanK 管理开启时 vLLM 多并发带来了吞吐提升。
八路同窗口每用户平均约 47.30 token/s；这不是每用户时延或输出速率保证，
不同请求的草稿接受率和完成时间仍有差异。三组全部完成，均无抢占，原生
并发采样覆盖为 99.44%–99.63%。

64K 输入的八路测试同样 8/8 成功、无抢占，共同 decode 窗口原生采样覆盖
99.92%。稳态吞吐约 387 token/s，但首 token 时延 p95 达 231.74 秒；较长的
prefill 和历史刷新显著影响整轮吞吐。该结果支持超过 K 的历史压缩后仍可
共同 decode，不能当作长文请求已达到低时延 SLO 的证明。

后端记录确认八个请求各发生 32 次上下文刷新，共 256 次，其中 248 次执行
MeanK 模型注意力评分，选中的历史始终不超过 32768 token。每用户最终
checkpoint 保留 69631 个已计算 token，前端历史有 69632 个，最后一个生成
token 按正常自回归语义留待下次 forward。回合结束后请求占用的 GPU 页全部
回收，native pool 空闲 1735/1736 页，剩余一页为永久 null page。

本次 prefill 的刷新累计上传约 344.17 GiB，装配计时合计 58.47 秒，其中 host
prepare 合计 47.12 秒。这是已观测到的长文开销，说明频繁历史装配和 CPU
准备值得继续优化；这些是事务累计计时，不等于可直接从端到端延迟中扣除的
独立墙钟时间，也不代表稳态 decode 的全部成本已完成归因。

## 恢复完整 256K 历史

`terminal-256k-08b` 完成单用户测试：261632 token 初始输入，生成 258 token，
追加 126 token，再生成 128 token，前端最终保留恰好 262144 token。首次回合
的已计算 checkpoint 有 261889 token，GPU working KV 为 33537 token；该数量
包含选中的历史和最近生成内容，不是将整段 256K KV 留在 GPU。

回收 GPU 页后恢复，真实模型字节核对通过：目标 KV 共 1,098,940,416 字节、
96 个 recurrent 状态张量和 5 层 draft 状态均一致。末尾未满页和最后一个尚待
forward 的输出 token 均纳入会话恢复语义。请求完成后，GPU 请求／保留页和
预算归零；native pool 的一个永久 null page 不属于请求泄漏。

这次成功记录是**缓存命中后的重试**：原始冷启动测试 `terminal-256k-08`
在约 174.92 秒后因网关 SSE 消息 1 MiB 限制返回 400，保留了 261120 token
的安全 checkpoint。修复大 prompt 的 SSE 限制后，重试首回合用时 4.79 秒；
第二回合的恢复、追加输入、128 token 生成及字节核对合计 5.82 秒。
4.79 秒不是冷启动 256K prefill 时延，5.82 秒也不是单独的 KV 加载时延。
未做八用户各 256K 常驻，也未把这项合成容量测试当作长文任务质量验收。

## FP8 草稿权重的恢复验证

`draft_quantization: "fp8"` 对 DFlash2 采用原生 FP8 线性层量化。目标模型仍是
NVFP4／FP8 混合权重和 FP8 attention KV；DFlash2 模型 dtype 及 sliding KV
保持 BF16，FP8 线性 kernel 内部按原生后端处理输入量化和尺度。
vLLM 原生 DFlash context 投影直接切片 BF16 布局的权重，无法处理量化后
转置权重及尺度；本仓库 adapter 改为调用该层原生量化投影接口，再提取 K/V。

`terminal-fp8-draft-10` 的 4200 token 输入＋257 token 输出、追加 32 token＋128
输出已通过回收／恢复。4456 个已计算 token 对应 146,014,208 字节目标 KV，
另有 96 个 recurrent 张量及 5 层 draft 状态，全部字节一致。首回合 4.90 秒，
第二回合 3.02 秒，二者都包含生成及验证开销。此项记录仅确认恢复完整性；
FP8 草稿的独立并发测试见首表，任务质量仍未验收。

## 测量方法和环境

- 单张 RTX PRO 6000 Blackwell Server 96 GB；vLLM 0.30.0，PyTorch
  2.13.0+cu130，FlashInfer 0.6.18.post1；目标为 Qwen3.8-27B ModelOpt
  NVFP4／FP8 混合 checkpoint，目标 attention KV 为 FP8 E4M3。
- MeanK 的 K=32768、G=16384，max-seqs=8、batch token 上限 4224、GPU
  utilization=0.82；active admission 为 48 GiB，CPU raw KV cache 上限
  100 GiB。每用户独立认证和 session salt，排除跨用户 prefix 共享。
- 稳态窗口取所有请求共同 decode 的区间，每路前后裁去 128 个输出 token，
  按原生 SSE 返回的准确已接受 token ID 计数。窗口内全部停顿保留在分母中。
  同时采样 native running/waiting、prompt token 和 preemption 指标，确认
  所列并发数。指标约一秒更新，属于隔离批次的 engine 级旁证。
- 整轮吞吐为成功输出总数除以首请求发出至全部响应完成的时间，包括 prefill、
  排队、生成、回合收尾和响应传输。网关当前回合接口最后返回 JSON；测试的
  TTFT 是网关内部 native SSE 首 token 时刻，不代表客户端流式首字节时延。
- 这些 MeanK 运行开启 `verify_session_restore: true`；额外终态字节复制主要
  位于共同 decode 窗口外，但计入整轮耗时。普通部署应关闭该测试选项。
- 固定输出使用 `ignore_eos`，不是自然结束任务。没有执行长期混合工具负载、
  带宽竞争、公平性／硬 SLO 或 agent 质量验收。短 greedy 冷重算对照出现过
  `text`／`instruction` 的差异；恢复状态字节一致不能替代逐 token 冷重算等价。

普通 decode 已使用 vLLM 原生 FULL_DECODE_ONLY CUDA Graph。MeanK 查询、
选页和装配事务显式执行 eager batch，因此不能把整个服务描述成全程 eager。
当前选中历史的模式使用 DFlash2，原生 MTP 的历史 KV capture／装配仍未接入。

本轮最终非 GPU 回归为 482 passed、50 skipped；另外完成 29 项 GPU 组件测试
和原版 C++ 的 5 项 CTest。GPU 测试包括原版 CUDA MeanK kernel 数值／top-k
对照、已接受 recurrent 状态、raw Q/K capture、FP8 KV 装配和窗口位置恢复。
wheel 已重建，包内 Python 源文件与当前源码逐文件匹配。这些结果是组件、
生命周期和打包验证，不替代上面的真实模型性能与任务质量边界。

## 可复查证据和复现入口

本地原始证据保留在被 Git 忽略的 `artifacts/`，未提交模型、用户凭据或生成正文。
每次并发测试的 `summary.json` 给出指标，`results.json` 包含 token 时间线、原生
计数器差分、最终预算和采样覆盖。
可公开复核的汇总见 [结构化测量数据](benchmarks/meank-20261013.json)，只包含
配置、指标和验收状态。64K 本地目录另有 `historical-selection-summary.json`
和 `capture-evidence.json`，记录选页次数、历史长度和 GPU 资源释放。

| 证据目录（相对于仓库） | 对应结果 |
| --- | --- |
| `artifacts/elastic-long-output-20261013/c8-32k-16k/` | 旧完整上下文 MTP 基线 |
| `artifacts/meank-20261013/c8-32k-16k-02/` | BF16 draft 7，自动后端 |
| `artifacts/meank-20261013/c8-32k-16k-q4/` | BF16 draft 3，自动后端 |
| `artifacts/meank-20261013/c8-32k-16k-q4-cutlass/` | BF16 draft 3，CUTLASS |
| `artifacts/meank-20261013/c8-32k-16k-async-offsets/` | BF16 draft 3，异步窗口偏移 |
| `artifacts/meank-20261013/c8-32k-16k-fp8draft/` | FP8 draft 3，异步窗口偏移 |
| `artifacts/meank-20261013/c1-32k-4k-final/` | 最终 BF16 配置，单路 32K＋4K |
| `artifacts/meank-20261013/c4-32k-4k-final/` | 最终 BF16 配置，四路 32K＋4K |
| `artifacts/meank-20261013/c8-32k-4k-final/` | 最终 BF16 配置，八路 32K＋4K |
| `artifacts/meank-20261013/c8-64k-4k-final/` | 最终 BF16 配置，八路 64K＋4K，历史超过 K |
| `artifacts/meank-20261013/terminal-256k-08/` | 修复 SSE 限制前的冷启动失败 |
| `artifacts/meank-20261013/terminal-256k-08b/` | 单用户 256K 完整历史恢复 |
| `artifacts/meank-20261013/terminal-fp8-draft-10/` | FP8 draft 恢复字节验证 |

32K 冻结输入 token ID 的 SHA-256：
`19543eb03ad997478d92dc001aab03bd091f3b92a17c639e721e1d09fa3d3c8d`。
64K 冻结输入的 SHA-256：
`e706ca27ec4d13163f23df55c3355465f9a00fe2b47a7314cb7a684b62ebf60a`。
只有采用相同长度、输入及并发窗口口径的结果才适合直接对比。

启动 [MeanK 示例配置](../examples/qwen-nvfp4-meank-elastic.json)，再用
[`scripts/qualify_elastic_long_output.py`](../scripts/qualify_elastic_long_output.py)
指定 `--launch`、`--input`、新建的 `--output` 目录、`--concurrency 8` 和
`--output-tokens 16384`。测试期间后端须只运行这一批请求，且不得同时运行 GPU
单元测试。脚本的临时网关复用同一个已启动的模型后端，不会加载第二套模型。

恢复验证用
[`scripts/qualify_terminal_sessions.py`](../scripts/qualify_terminal_sessions.py)，
提供本人服务的 `--api-base`、`--backend-base`、`--users-file`、`--input` 和
`--capture`。`--skip-cold-control` 明确只验收真实 GPU 状态字节，结果不计为
冷启动 greedy 等价通过。要复现 256K 边界，设置 `--prompt-tokens 261632`
`--first-output-tokens 258 --append-tokens 126`，并保留第二回合的 128 token
生成空间。
