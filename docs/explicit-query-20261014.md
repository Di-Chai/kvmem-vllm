# 显式 Query、跨轮恢复与原版 MeanK 评分

日期：2026-10-14。本次补齐 token-ID 请求的 Query 规划、Query 边界重放、
FP16 MeanK index 和原版基础 CUDA scorer；继续保留单轮输出不超过 G。
这是单 GPU、同步 Qwen MTP 的有界适配，不代表 qw3 所有协议和可选策略均已迁移。

## 请求契约

使用 [新配置](../examples/qwen-nvfp4-meank-explicit-query-elastic.json)，启用
`request_planning: true` 和 `mean_k_backend: "qw3_cuda"`。
默认旧配置保持原来的 semantic-chunk 路径。新版入口支持：

| 字段 | 含义 |
| --- | --- |
| `kvmem_query_span: [begin, end]` | 完整逻辑 prompt 的绝对 token 坐标，左闭右开；不包含想排除的工具结果或模板。 |
| `kvmem_message_spans` | 调用方提供 `{role, begin, end}` 的有序、不重叠消息跨度。 |
| `kvmem_query_message_range` | 在上述消息列表中选择左闭右开范围；仅允许 user 消息。不能同时指定 token span。 |
| `kvmem_reselect` | `auto` 为默认；有明确 Query 且历史达到 K 时执行 Query 选择。`force` 要求提供 Query；`off` 禁止提供 Query，仍可进行容量压力刷新。 |
| `kvmem_prefill_window` | `pressure` 默认仅做容量压力刷新和明确 Query 选择；`keep_selected` 保持选页并在准入时检查空间；`semantic_chunk` 单独使用调度 chunk 作为 Query，不能与显式 Query 混用。 |

弹性会话首轮示例（token IDs 由本地 tokenizer 生成）：

```json
{
  "append_token_ids": [101, 102, 103],
  "max_tokens": 128,
  "kvmem_query_span": [1, 3],
  "kvmem_reselect": "force",
  "waiting_for_tool": true
}
```

发送至 `POST /v1/kvmem/sessions/{id}/turn`，并携带用户 Bearer key。
下一轮带 `version: 1` 和新增的 `append_token_ids`；Query 坐标以整个历史为基准，
包含之前实际接受的输出。网关完整保留 token 历史，不截断为 32K。
原生 `/v1/completions` 也支持这些字段，但 prompt 必须为 token-ID 数组；
HTTP middleware 将规划放入原生支持的 `kv_transfer_params.kvmem_request_plan`。

`keep_selected` 当前采用保守准入：完整 prompt + 输出额度必须不超过 K+G−B。
它尚未开放“逻辑历史很长但缓存 active view 仍有空间”的快速准入。

## Query 与恢复的执行顺序

1. 将 Query 起点向下对齐到 B=128 的封页边界，保存 CPU 模型检查点。
2. 前面的普通历史只使用压力策略，不把工具输入或历史 chunk 当作用户 Query。
3. 原子调度 partial tail + 完整 Query；预算不足时等待，容量不足时返回错误，
   不截短 Query。原生 batch 容量为 G + `max_num_seqs × (MTP k+1)`，
   并禁用原生固定长 prefill 阈值以避免拆开 Query。
4. provisional 执行捕获 target Q，仅对显式 Query 内的 token 评分；partial tail
   参与重放但不参与 Query 评分。保存并回滚同一 batch 的全部 GDN 状态，包括邻居。
5. 原版 Store 选页；target 16 层及 MTP 1 层共同从 immutable raw KV 装配、
   dense re-RoPE，然后进行 durable replay。历史只发布实际接受的 token。
6. 回合结束归还原生 GPU 页，CPU 保留最近的封页检查点。下一轮先检查完整 token
   prefix 和用户 namespace，再恢复所选 KV/GDN，并重放检查点后尚未保留的尾部。

重选提交时同时更新该边界检查点的 KV frame，保留其 Query 前 GDN state。
如果短回合在下一个封页 decode 检查点之前结束，恢复后重放旧 Query 必须使用
原 durable replay 已提交的选页，不能恢复更早的压力选页。窗口更新不需要再次
复制模型状态。周期模型快照改用有额度的 pinned CPU buffer，并等待 DMA fence
后才允许源 GDN 页继续执行；CPU 快照的大小和生命周期仍由检查点池记账。

Query 重放长度上限为 G−B，包括起点之前不足一页的 partial tail。
因此 G=16,384 时允许的完整重放上限是 16,256 token。
启用规划要求 G 至少为 3B，留出压力重放和 MTP 空间。
本版本通过将 Query 合并为单个调度段保证完整捕获；不支持任意大 Query 的跨 batch
累计。压力刷新提前留下重放和 MTP allocation lookahead 所需的空间。
共享此 profile 的旧 chat/text 入口仍按小 chunk 执行，保留 MTP headroom；
扩大显式 Query 的 batch 容量不应使它们填满旧窗口。

K 限制的是被选中的历史前缀；Query 重放和后续生成都占用 K+G working window
内的追加空间。若累计追加填满该空间，会进行容量压力刷新；单轮输出额度仍
不能超过 G。本版没有独立的 prefill budget。

跨轮压力恢复可能把旧 active view 重新缩至 K。新分配的物理页上必须重新装配
所有所选 K/V；旧请求的同位置 K skip 和 resident V 标记均不可复用。
检查点池分别验证完整历史位置和本次已经提交的新页布局，避免将旧布局误作恢复契约。
Query 起点如果早于已有检查点，则采用冷恢复，不能使用因果上位于 Query 之后的状态。

## 原版数值路径

- 每个封页按 token 顺序累加 content-frame BF16 K，使用 FP32 累加器，除以页长后
  发布 FP16 MeanK；target 层参与评分，MTP 层仅归档和装配。
- Q 与 index 为 FP16；点积、页间 softmax 和分数归约使用原版基础 CUDA kernel。
  其 kernel body 与参考 qw3 文件逐字相同；[来源和哈希](../csrc/qw3_mean_k_provenance.json)
  随源码和 wheel 一同保留。
- 将 eligible 页压紧后评分，再散回历史页顺序，排除 sink/recent 等必留页的
  softmax 质量。每次支持最多 8,192 个 eligible 页，覆盖当前 256K/128 配置。
- 10 个 vendored C++ 核心文件仍与参考 qw3 工作树逐字节一致。

CUDA scorer 首次运行使用 Torch 扩展编译，需要匹配的 CUDA toolkit、C++ 编译器
和 Ninja；`[vllm]` 安装项包含 Ninja。wheel 包含 `.cu/.cuh` 和 provenance，
运行不依赖工作树外的 qw3 文件。封页索引当前由 CPU 维护后上传，增量 GPU index、
one-dot/two-pass、Adaptive、在线逐层评分仍未移植。
相同 kernel 不意味着不同模型执行器捕获的 Q/K 或最终生成结果逐位相同；
浮点原子归约顺序也可能影响近似同分的排名。

## 真实模型验收

硬件为 RTX PRO 6000 Blackwell Server 96GB；vLLM 0.30.0、Torch 2.13.0+cu130，
NVFP4 混合 checkpoint、target/MTP FP8 KV、MTP(k=3)、原生编译与 CUDA Graph。
K=32,768，G=16,384，CPU read cache 配额 100 GiB。
完整数值记录见 [公开验收数据](benchmarks/explicit-query-20261014.json)。

真实测试包括 50,006 token 输入，1,964 token 完整 Query，以及模拟工具等待后第二轮
1,301 token Query；两个 Query 起点都包含 partial tail。每轮严格检查输出数量和
完整历史长度，按原生 response ID 对应评分/重放记录和 CPU 恢复记录。
多用户分别使用独立鉴权和 session，检查状态 API 不暴露其他用户的历史。

decode 测量分开报告不带 session 的内部后端请求，以及携带 session ID、包含
每 128 token CPU 检查点维护的请求。它们都启用 KVMem KV 历史维护；鉴权和
跨轮并发由另一个弹性网关测试验证。开启检查点的速率才覆盖本次恢复方案的
主要 decode 成本。可用 `scripts/qualify_decode.py --prompt-sessions` 复测。

| C8，32K 输入 + 每路 4K 输出 | 稳态 decode token/s | 整轮 token/s |
| --- | ---: | ---: |
| 不带 session，仍维护 KVMem KV 历史 | 576.48 | 347.38 |
| 带每 128 token 会话检查点，pageable 快照 | 378.21 | 259.13 |
| 带相同检查点，最终 pinned 快照 | **503.22** | **313.07** |

三组全部完成，原生采样均确认 8 个稳定 decode owner。最终 pinned 组的有效
窗口为 60.85 秒，原生覆盖率 99.53%，服务抢占计数为零。完整恢复方案仍有
周期状态快照和 fence 开销；576.48 不能替代它的吞吐。
该输入长度恰好为 K，稳态窗口没有历史重选；超过 K 的 Query 选择/装配由
50K 功能测试覆盖，不能将此速率推断为 256K 历史或频繁重选的吞吐。
本轮 pinned 相对 pageable 的稳态速率增幅为 33.1%，但每组只有一次测量，
输出 token hash 均有差异，不能承诺重复运行或其他负载有相同增幅。

同一 pinned 组从首个请求提交到 8 路全部拿到首 token 为 36.65 秒，
262,144 个输入 token 对应有效 prefill **7,153 输入 token/s**。这个口径包含
排队、KVMem 历史维护及穿插的 decode，尚未隔离纯 GPU prefill 时间。
首 token 延迟 p95 为 34.90 秒。稳态 decode 的 503.22 使用共同生成窗口，
整轮 313.07 则将预填充、排队及收尾计入 104.67 秒的总耗时；两者的分子
分别为窗口内的 30,619 输出 token 和整轮的 32,768 输出 token。

长回合组为每路输出 257 + 129 token，短回合组为 17 + 17；每组各有 8 个
独立用户、16 个成功回合、16 次完整 Query 重放、8 次 CPU 恢复，最后 GPU
owned blocks 和 native requests 均归零。短回合恢复点为 48,000，使用已提交的
Query 选页 frame；长回合还覆盖了 decode 后的检查点及压力恢复。
最终安装包连续完成两组长回合和一组短回合，共 48 个成功回合、48 次完整
Query 重放和 24 次 CPU 恢复。共享 archive 队列出现 23 次准入延期，均可
完成；结束时在途 I/O 字节、GPU owned blocks 和 native requests 均为零。

独占回归为 699 passed、6 skipped；另有 pinned/frame 针对测试 21 passed，
包含真实 GPU DMA fence 验证，decode 测量工具回归 28 passed。非法规划拒绝、
auto message range、keep_selected/off 两轮恢复和 off 的容量压力刷新均通过。
旧 text completion 入口也完成 50,006 token 输入 + 32 输出，17 个 semantic
Query chunk 均不超过 1,024 token。首次 scorer JIT 编译会增加冷启动延迟。

复测完整 Query/恢复时，将配置复制后设置 `verify_session_restore: true` 再
启动服务，运行 `scripts/qualify_explicit_query.py --concurrency 8`；短回合再加
`--first-output-tokens 17 --second-output-tokens 17`。日常配置默认关闭恢复 oracle，
本机当前验收服务仍保留 oracle，以便继续检查。测试数据只使用合成历史。

验收结果和最终安装包复测数据在上述记录中列出。此测试验证固定请求的恢复和
并发执行；长期工具负载 SLO、agent 质量和全部 qw3 策略的逐页等价仍需独立验收。
压力恢复的字节 oracle 覆盖 96 个 GDN state tensor；旧 active view 已改变，
不能把旧 KV 布局的字节比较当作该压力恢复的有效 oracle。

## 剩余边界

- DSH 原生聊天入口仍使用旧协议，没有自动生成完整 token 坐标，也没有接入此
  弹性会话的跨轮 CPU 恢复。此处不以 DSH 两轮聊天代替检查点验收。
- 独立 prefill budget、prefill-only start/append/finish、mandatory 消息/工具/文件
  spans、任意历史编辑和 checkpoint ladder 尚未接入。
- 当前只支持单 GPU、同步在途 batch、文本 native Qwen MTP；compiled DFlash、
  多 GPU、多模态和异步多 batch 不在本次资格范围。
- 100 GiB 是 read cache 额度，不是进程总 RAM 上限；historical view 仍包含
  NVMe 封页，并受 262,144 logical token、会话数和其他存储配额约束。

测试中发现并修复了压力恢复仍校验旧布局、无 Query 长 prefill 的 MTP 空间不足，
以及短回合检查点仍保留重选前 KV frame 的问题。长短回合连续负载还暴露了
后台封页占满共享 archive 队列时，恢复读错误地使引擎退出的问题；现在保留
generation lease 和独占 staging buffer，在有界 deadline 内等待队列释放后
重试，不扩大队列额度。四个真实线程/I/O 回归覆盖 job 数和字节额度的竞争。
完整 GPU 回归在独占 GPU 时执行，避免与模型服务的显存占用混淆。
