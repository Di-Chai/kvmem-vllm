# KVMem 原生编译兼容与 CPU 历史批处理，2026-10-14

本次实现以 NVFP4 target、FP8 E4M3 target/MTP KV、Qwen 自带 MTP(k=3)、
单 GPU、同步 batch 为验证范围。没有修改 vLLM 的验证／采样实现，也没有更换
MeanK 检索策略或原版 Store/selector。测量结果在本文件后半部分记录。

## 编译与捕获

公开启动器新增 `execution_mode: "compiled_graph"`，使用原生
`CompilationMode.VLLM_COMPILE` 和 `FULL_AND_PIECEWISE`。普通 decode 保留
FULL CUDA Graph；调度、MeanK 评分、装配、GDN 回滚和会话事务仍由原来的
控制路径负责。语义刷新关闭整批图重放，但模型 forward 可以继续调用原生
编译代码。当前只开放已测试的 native safetensors Qwen MTP 路径，保持一个同步
在途 batch；DFlash、异步多 batch 和多 rank 编译不在此次资格范围。

捕获 Q/K/V 的固定缓冲区在 `load_model` 阶段分配，因为原生显存 profiling 会在
KV 初始化前编译模型。后续图初始化保留其地址。查询探测直接读取这些固定输出，
不依赖编译后不会重复执行的 Python ContextVar 回调。

MTP 需要额外处理：原生编译包装器不在每次调用重新检查 Python 分支，首次编译
固定的 prefix-active 布尔值可能使后续草稿覆盖前缀捕获。本次用 GPU int32 标志
控制捕获写入，prefill 图包含开启／关闭标志，独立 draft 图保持关闭。Q/K 规范化、
RoPE 和 gate 运算保持原运算，V 捕获合并进同一捕获内核。

首次真实长文恢复测试捕获到了这一问题；修复后严格的 GPU 字节校验通过。该校验
没有降级成近似比较，也没有通过放宽容差让失败消失。

## CPU 历史维护

`batched_cpu_append: true` 将同一已确认 capture packet 中的请求集中处理：先
保留已获容量的请求前缀，在一个释放 GIL 的原生调用中复制所有 accepted 行，
再提交各自的原版 Store 事务。失败／背压不会发布未完成的行；每个请求仍有独立
namespace、generation、尾页和历史计数。EOS、取消及后续完全丢弃的 packet
保留原来的裁剪规则。后台 scatter 与前台读取之间的提交屏障仍然保留。

第一版只合并复制和索引加法，真实 decode 没有稳定提升，因此又移除了重复的
目标数组导出，并为当前 x86/Torch 2.13 的小批量 BF16 decode 实现向量化
FP32 求和。该路径限定最多 4 行且通道数为 64 的倍数，保持原 ATen 外层归约的
逐 token 加法顺序；其他版本、CPU、dtype、几何和较大 prefill 使用原 Torch 归约。
目标数组视图在封页提交时清除，避免额外持有已经转交 archive 的 raw/V 内存。

这没有把索引改为低精度，也没有修改 MeanK 的 Query 范围、softmax、层／头
归约、选页、mandatory bands 或 dense re-RoPE。之前审计中记录的 FP32 索引与
原版生产 FP16 索引差异，仍是独立的迁移项，不能把此次改动描述成全部 qw3
执行细节已经一致。

## 正确性与测量方法

- 全套 CPU/GPU 检查：669 passed、6 skipped。包含 accepted/rejected 裁剪、
  多 packet 取消、背压前缀提交、封页释放、原版 MeanK、窗口装配、循环状态和
  编译捕获重放。小批量求和使用真实 17 层／4 KV heads／256 head dimension
  几何及跨大指数的正负相消输入，与原 Torch 结果逐位相同。
- 真实 64K 输入、257 输出、暂停后追加 32 token 并输出 128：验证 17 层 KV，
  共 1,185,415,168 字节，以及 96 个 GDN 状态张量。此项是恢复完整性校验，
  不等同于冷启动生成质量或长期 agent 质量测试。
- CPU 复制＋MeanK 更新的交替顺序单独测量：8 请求、17 层、16 评分层、
  4 KV heads、D=256、每路接受 1/2/3/4 行、Torch 单线程，12 组各 100 次。
  中位数从 **0.260 ms/批降到 0.105 ms/批（−59.7%）**。此数不包含 Store
  预留、archive I/O、GPU、D2H 或网络，不能解释成整机吞吐提升 60%。
- 正式并发吞吐均关闭 profiler；测试期间没有运行 CPU 测试或 trace 分析。
  共同 decode 窗口每路头尾各裁掉 128 token，利用 native 指标确认所有请求
  同时 decode、prompt 计数稳定和零抢占。包含 prefill 的整轮吞吐单列。
- 32K 冻结输入 SHA256：
  `19543eb03ad997478d92dc001aab03bd091f3b92a17c639e721e1d09fa3d3c8d`。
  C8、temperature=0、seed=104729、每路独立 salt、APC off、GDN none、
  max_num_seqs=8、max_num_batched_tokens=4224、GPU 利用预算 0.82、K=32768、
  G=16384。首轮与热轮都保留，避免只挑吞吐较高的一轮。

## 真实并发结果

| 配置／工作负载 | 稳态 decode，token/s | 整轮，token/s |
| --- | ---: | ---: |
| 上一版 KVMem，C8，32K 输入＋16K 输出 | 510.48 | 415.01 |
| 本次编译＋向量化 CPU 维护＋auto 后端，8 独立用户，32K＋16K | **538.75** | **449.94** |
| 本次最终配置，C8，64K 输入＋4K 输出 | **619.37** | 见原始汇总 |
| 本次最终配置，C8，32K＋4K，两轮 | 549.65／555.98 | 见原始汇总 |
| 仅编译、旧 CPU 维护、CUTLASS，C8，32K＋4K | 555.67／566.53 | 见原始汇总 |
| 第一版批量复制、CUTLASS，C8，32K＋4K | 526.73／565.29 | 见原始汇总 |
| 第一版批量复制、CUTLASS，C8，64K＋4K | 540.78 | 169.91 |
| 无 KVMem、原生编译、同步调度、auto，C8，32K＋4K | 679.73／679.98 | 见原始汇总 |

实际八用户长输出测试 **8/8 成功、零抢占**，总共生成 131,072 token，
291.31 秒完成，TTFT p95 为 33.03 秒。新版稳态／整轮吞吐相对上一版提升约
5.5%／8.4%。上下文增长的共同八路窗口分段约为 529、519、565 token/s；
12K–16K 输出段已经没有完整共同八路窗口，不能把收尾阶段当作八路稳态。

这些是最终组合的收益，包含原生编译、捕获调整、CPU 维护和线性后端选择的
变化，不能全部归因于其中一项。第一版批量复制没有稳定吞吐收益；首轮还伴随
不同的生成摘要和较低的 MTP 接受率，热轮与旧路径接近，所有数据都保留。
向量化版本的 CPU 单独测量才是复制／MeanK 组件的单变量证据。

当前仍低于早期约 592 token/s 的完整上下文结果。那个结果的调度、GDN/APC、
batch/GPU 参数和上下文管理方式不同；本次新补的原生**同步**编译控制仍达到约
680 token/s，所以不能主要用“当前关闭 async”解释差距。

## 最终 decode profiling 与剩余差距

48 个真实 C8 纯 decode 迭代通过 trace 验证，去掉边界后分析 46 个完整周期。
仪器开启时每周期约 33.21 ms，其中 GPU 活动区间并集约 30.24 ms。
GPU target graph 约 25.00 ms，MTP prefill 约 1.55 ms、后续两步 draft 合计约
2.99 ms。KV 打包约 0.023 ms，D2H 约 0.034 ms。没有观察到足以解释剩余差距
的大规模捕获缓冲区 GPU 复制。target GPU 时间与此前原生编译控制接近，
不需要为本次任务改写原生主模型验证逻辑。

CPU `authority.append_batch` 约 **1.27 ms/周期**，包含各请求 Store 预留、
复制／求和和提交；`capture.drain` 约 **1.54 ms/周期**。此前 trace 的 drain
约 2.61 ms；编译和后端也已变化，因此此差异是组合测量，不能冒充严格单变量
的 CPU 总耗时归因。新 trace 中，纯 decode 不再为历史维护执行原来每请求的
Torch 转换、求和和复制组合。

同一 32K＋4K 工作负载，原生同步控制约 30.62 ms/轮、每流每事件输出约 2.60
个 token；KVMem 最终两轮约 34.44／34.33 ms、2.37／2.39 个 token。
**每轮额外耗时和较低 MTP 接受长度共同造成吞吐差距**。CPU 历史维护约 1.54 ms
仍是其中一部分；其余调度／元数据／输出处理成本，以及接受率变化的具体来源，
尚未被逐项严格隔离。不能把它们直接归咎于原生验证效率，也不能声称新版已经
复现全部原生吞吐。不同量化执行／prefill 调度产生的生成差异不等同于已完成
质量评估，后续仍需要独立质量和混合负载 SLO 测试。

正式性能数据、CPU 组件测量和经过清理的 profile 汇总见
[结构化证据](benchmarks/compiled-capture-history-20261014.json)。原始 trace、token
数据、用户凭据、模型及恢复字节快照不进入 GitHub。

## 使用

参考 [编译＋CPU 批处理配置](../examples/qwen-nvfp4-meank-mtp-compiled-elastic.json)，
填写本地模型、tokenizer 与用户文件路径，然后通过 `elastic-serve` 启动。
CPU 缓存仍为 100 GiB、active 预算为 48 GiB，8 个 active 用户，每路 K=32K、G=16K。
编译及 CPU 批处理是显式配置；需要兼容回退时使用原来的 `full_graph` 和
`batched_cpu_append: false`。目前运行中的正式服务关闭 profiler 和恢复 oracle。
