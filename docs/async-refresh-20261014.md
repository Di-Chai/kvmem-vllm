# 请求级 KVMem 刷新

`async_refresh=true` 将正在重选的请求从公共 batch 中暂时移出。其他可运行请求继续走
vLLM 的连续批处理、原生编译、CUDA Graph 和 MTP。该开关并不是 vLLM 的
`async_scheduling`；原生多在途 batch 仍关闭。

## 事务与资源所有权

1. Scheduler 为当前请求保留旧物理页和 Query 重放空间；发送刷新命令时从
   `num_scheduled_tokens` 删除该请求，因此原生 token 计数、采样及 CPU 历史均不前进。
   请求注册和 block-table 更新仍交给 worker。
2. worker 只备份该请求的 GDN 状态。Query 探测按 `refresh_quantum_tokens` 分段；
   guided Query 每次推进一个 token。每个公共模型迭代前至多执行一个私有小步，多个
   刷新请求轮转；私有模型 forward 不会与公共 forward 同时写捕获缓冲区。
3. MeanK 评分、历史封页发布、CPU 读取和 KV 装配交给单独线程与 CUDA stream。
   使用原版 MeanK、既有 Query history 和 Action attention 选页规则；不发布私有
   token，不更改公开单轮 G 限制。
4. 刷新与前台会话恢复使用分开的 staging / raw-K cache。配置的 raw-K 总额度一分为二；
   两套 staging 额外需要显存。下一次后台装配必须等上一事务 GPU fence 完成并释放
   scratch，不能仅凭 CPU future 已返回就复用。
5. worker 非阻塞轮询 completion，完成后更新当前请求的 dense 映射并回执。
   Scheduler 原子提交所有 full-attention group、回收多余页，再恢复公开 Query 重放。
   公开 Query 仍遵守完整 Query 原子重放契约。
6. 取消会延后 native request index、authority 和页回收，直到私有 forward／DMA
   完成。取消和首次命令同批到达也保留所有权；已完成刷新但未重放的取消清理准备状态。

## 范围

当前资格范围为单 GPU、原生 Qwen MTP、FP8 KV、显式 request planning 和有界非 GC
历史。`repository_gc=true` 与异步刷新组合拒绝启动。参数默认关闭，保持旧配置行为；
`refresh_quantum_tokens` 取 1–2048，默认 256。

这消除了“必须等一个请求完成整个刷新事务，公共 batch 才能继续”的依赖，**不保证
其他请求零延迟**。GPU 计算和 PCIe 带宽仍共享，私有小步仍占用模型线程；公开原子
Query 重放、前台 P/M 会话恢复、CPU scatter 的提交屏障也仍可能造成短暂停顿。
较小 quantum 可缩短单次私有占用，但会增加刷新开销。尚未实现按 SLO 分配带宽或
抢占任意已提交 GPU kernel。

## 验证入口

- `tests/test_async_refresh.py`：单请求调度移除、恢复只重放一次、取消与页租约、
  首次分派取消、轮转、公共 capture 保护及 scratch fence。
- `scripts/qualify_action_attention.py`：真实模型跨轮、历史评分和并发恢复。
- `scripts/qualify_async_refresh.py`：7 路持续 decode 与 1 路长上下文刷新；流式
  token ID 与最终 usage 对账，报告刷新窗口内每路进展和最大 token 事件间隔。
- `scripts/qualify_refresh_cancel.py`：在 Query 探测和后台装配中取消，核对 GPU 页
  释放后立即执行新请求。

## 真实模型测量

RTX PRO 6000 Blackwell Server Edition（97,887 MiB），vLLM 0.30.0，27B NVFP4，
FP8 KV，原生 MTP(k=4)，compiled graph，K=32768/G=16384，CPU 历史预算
100 GiB。启用 guided Query（上限512）、Query history、Action attention
（non-thinking，interval16，spread3，fraction0.25）。CPU KV 的磁盘读写均为0。

相同 token-ID 输入和固定输出长度：7 路各 8192 输入＋8192 输出，全部进入 decode
后加入 1 路 50006 输入＋129 输出。同步与异步各跑一轮，并先各执行单用户两轮预热。
两次长请求的 guided Query 都实际生成1个私有 token；其他私有 Query 探测行也不进入
公开历史。8 路最终历史长度逐一核对为 `prompt + output - 1`。

| 指标 | 同步刷新 | 异步刷新，quantum256 |
| --- | ---: | ---: |
| 压力选页期间其他7路合计输出 | 0 token | 292 token |
| 跨越压力选页窗口的最大输出事件间隔 | 0.488秒 | 0.086秒 |
| Query 重选期间其他7路合计输出 | 0 token | 492 token |
| 跨越 Query 重选窗口的最大输出事件间隔 | 1.212秒 | 0.191秒 |
| 长请求 Query 刷新事务耗时 | 1.075秒 | 2.253秒 |
| 长请求首 token 延迟 | 8.869秒 | 10.579秒 |
| 整个混合负载输出吞吐 | 644.47 token/s | 548.26 token/s |

异步版本每个邻居都在两个刷新窗口内取得进展，Query 窗口每路输出67–72个 token，
窗口结束时7路仍在 decode。输出事件可能携带多个 MTP 接受 token；间隔是 SSE
token-ID 事件间隔，不是逐 token GPU kernel 时间。同步窗口内没有任何邻居输出。

**本次证明了请求隔离和停顿缩短，没有证明总吞吐提高。** 小步推进增加私有 forward
的调度开销，刷新自身更慢。两轮生成的 token 序列不同，因此整轮吞吐还受 MTP
接受率和 batch 形状影响；不能将全部吞吐差异都归因于异步开销。这里是一次配对
实验，不是重复统计结论，也没有重新跑180题 NIAH 质量评估或证明逐位输出一致。

取消验收分别在 Query 探测和后台装配阶段断开连接：约0.150/0.378秒后记录 fence
完成，最终 `owned_blocks=0`，后续无会话请求各成功生成32 token。测试顺带发现并
修复了原有 Action attention 对 session binding 的强制索引：无会话／未缓存会话
现在使用原生完整 prompt 初始化 scope，且不将生成 token 当作 prompt。
混合性能数据采集在该补丁之前，两组计时请求均有 session binding。

完整 CPU 回归：697 passed / 77 skipped（GPU 测试未包含在这个 CPU 命令中）。
最终代码的8用户两轮验收也通过，共16请求，耗时201.30秒；每用户两轮分别16/8个
有效 Action 样本，第二轮恢复1094页历史分数，16个请求的历史终点均严格核对。
结束后刷新队列、connector pending 和请求 GPU 占用全部归零，原生 CUDA Graph
仍实际执行；此用例用于跨轮／并发正确性，不作为稳态 decode benchmark。
机器可读数据见 [benchmark JSON](benchmarks/async-refresh-20261014.json)，原始
响应、token 到达时间及刷新事件位于 `artifacts/async-refresh-20261014/`。

可用配置示例：[异步刷新弹性服务](../examples/qwen-nvfp4-async-refresh-elastic.json)。
示例使用 tool-call scope、guided上限32；本表压力测试采用 non-thinking、guided上限512。
