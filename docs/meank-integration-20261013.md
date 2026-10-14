截至 2026-10-14，原生编译捕获和 CPU 批处理已接通，资格范围与性能见
[编译兼容与历史维护报告](compiled-capture-history-20261014.md)。

# 原版 MeanK 与 vLLM 并发执行

`elastic-serve` 的 `context_mode: "meank"` 保留完整已接受历史，由模型真实的
pre-RoPE Q 与历史 MeanK 评分，选择历史 KV 并重新装配 GPU working context。
没有 BM25、词法页检索或 token 片段拼接。完整上下文／原生 MTP offload 路径
仍为 `context_mode: "full"`，两种模式的成绩须分别报告。

## 实现核对

| 上下文语义 | 当前实现和证据 |
| --- | --- |
| MeanK 分数 | 每页 raw K 均值；QK / sqrt(D)；先屏蔽固定保留页，再按 layer/query/head 做页间 softmax；平均 head/layer，累加 query。GPU 测试与原版 CUDA kernel 对照，包含 2048 页。 |
| 选页和保留带 | 直接使用未经修改的 qw3 `KvMemStore`；自动 sink/recent、top-k 及历史顺序由原 C++ 核心决定。低层 mandatory 参数可用，在线入口尚未接入消息／工具／文件证据 span。32K K 对应 1024 sink／4096 recent token。 |
| 历史权威 | 每用户独立的 immutable BF16 raw K＋FP8 V；仅写已接受 token；完整块异步 stage-out，尾部暂存到封块。CPU 缓存可淘汰，底层归档仍为权威。 |
| GPU working set | 选中历史按原顺序紧密排列，raw K 在新 dense 位置重新施加 RoPE，V 原字节加载；用完成 fence 后提交 residency。普通、packed、direct 路径均有 FP8 字节对照。 |
| 混合模型状态 | GDN 保留完整逻辑历史状态；临时查询后恢复所有 batch 邻居，再重放正式 forward。DFlash 使用独立的逻辑位置和原生 sliding KV；原生 MTP 的 raw KV 与 target 按同一个 MeanK 计划装配。 |
| decode | 原生 vLLM scheduler、batching、attention、sampling；普通 decode 保留 FULL graph；`compiled_graph` 接通原生编译与 PIECEWISE prefill。MeanK 查询／装配事务在图外控制，模型 forward 可以继续使用编译代码。 |
| 暂停／恢复 | 回合结束保留完整已接受历史、精确部分尾页、GDN accepted row、DFlash 尾部和待计算的最后一个输出 token；回收请求 GPU 页。恢复只装配 checkpoint working KV，并处理追加输入。 |
| 用户隔离 | 服务端生成用户／会话 namespace 和 cache salt；客户端只提交准确 token ID 增量与版本；会话锁、字节 admission 和截止时间队列控制并发。 |

源码快照和 CUDA 参考 kernel 的哈希见 `third_party/qw3/PROVENANCE.json`。
原始 C++ 文件保持原样；vLLM adapter 在本仓库中实现，不改写安装的 vLLM 文件。

此处核对的是原版 MeanK 主路径。qw3 后续可选的 action-attention fusion 和
query-history 加权模式没有作为本入口的可配置功能接入。分块 prefill 的查询批次
由 vLLM 原生调度决定；没有声称不同分块、量化或 batch 下生成 token 都完全相同。
当前索引和 scorer 使用 FP32，而 qw3 生产索引/Query 使用 FP16；当前 Query 是调度
chunk，尚未提供原版显式 Query span 与完整请求策略。所有功能及性能优化的逐项
状态见 [完整迁移核对](qw3-parity-audit-20261014.md)。

## 启动和预算

复制 `examples/qwen-nvfp4-meank-elastic.json`，填写本地模型、tokenizer、DFlash2
以及用户 key 文件，先运行 `elastic-serve --dry-run` 检查具体参数。
回合 API 与 [弹性入口](elastic-serving.md) 相同，MeanK 模式拒绝 `active_ranges`。
超过 K 的完整历史由后端做 MeanK KV 选择，不由网关裁剪 token IDs。

单卡测试配置为 NVFP4 target、FP8 attention KV、DFlash2、K=32768、
G=16384、8 路、CPU 缓存上限 100 GiB。`draft_tokens` 可设置 1／3／7；它控制
候选验证长度，所有接受仍由原生目标模型验证。通用入口默认 7，当前 NVFP4
MeanK 示例显式采用 3 和 CUTLASS 线性后端；性能应按实际工作负载测量。
`draft_quantization: "fp8"` 可在线量化 DFlash2 线性层权重；draft 模型 dtype 及
sliding KV 仍为 BF16，线性 kernel 内部按原生 FP8 后端处理输入量化和尺度。
目标模型权重与目标 KV 精度不受此选项影响。FP8 draft 的 context
投影通过原生量化层接口执行，包含正确权重布局和尺度；真实模型的回合恢复已通过
KV、GDN 与 draft 状态字节核对，但本次八路长输出测量没有显示吞吐收益，
所以示例保留 BF16 draft 权重。

active admission 根据真实共享页池计算，包含每组页取整、GDN speculation rows
和增长余量。V2 的 KV descriptors 共用一份 allocation，不能把各 descriptor 的
容量相加；每次页分配也要计入共享 pool 的 padding。当前例子用 48 GiB active
预算，物理 native KV pool 约 52.7 GiB，原生 allocator 继续执行硬上限。

CPU 权威格式为每 token 49152 B，256K 约 12 GiB／用户，8 用户约 96 GiB。
100 GiB 是按需使用的 CPU cache 上限，不是预分配 100 GiB；模型状态、索引、
pin staging 和进程开销另占内存。单用户允许 256K token 的总历史，须给追加输入
和生成预留空间；没有宣称八用户的 256K 历史都已经做过常驻压测。

## 验证边界

已完成八用户各 32K 输入＋16K 输出、八用户各 64K 输入＋4K 输出的 MeanK
并发测试，以及单用户最终 256K 历史的 GPU 页回收和恢复。64K 测试中，八个
请求各发生 32 次刷新，历史选中量始终不超过 32K。完整吞吐表、冷／热请求区别和证据路径见
[MeanK 实测报告](meank-throughput-20261013.md)。32K 输入本身没有超过 K，
所以其并发数字须与长历史选页测试分别解读；原生 MTP 的旧约 592 token/s
不能直接计为新 MeanK 路径的成绩。

`verify_session_restore: true` 是测试选项，读取回收前和恢复后的真实模型 GPU
KV、GDN 和 draft 状态并核对字节。普通运行应关闭，以免回合结束产生额外复制。
`scripts/qualify_terminal_sessions.py` 保留冷启动 greedy 对照结果；短测试发现一处
措辞不同（`text`／`instruction`），尽管恢复状态字节一致。该结果不是冷启动逐
token 等价验收，也不是执行真实工具任务的质量验收。

2026-10-14 起，MeanK 模式也支持 `draft_method: "mtp"`，从目标 checkpoint
加载 Qwen 自带的一层 MTP，继承 target FP8 KV 配置。主模型 MeanK 索引选页，
target／MTP raw KV 一同归档和 re-RoPE；新测量见
[qw3 优化与 MTP 报告](decode-optimization-20261014.md)。旧完整上下文 MTP
成绩仍不能当作新路径验收。持续工具负载下的带宽公平性、
硬 SLO、跨进程重启恢复、多卡以及完整 agent 任务质量仍需进一步验收。
