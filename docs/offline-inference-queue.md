# 离线推理：持续补位与逐请求退休

目标是一批独立输入持续占用可用推理槽。外层队列决定哪些任务获得 CPU 历史额度和在途槽；
vLLM 继续负责已准入请求的 continuous batching、chunked prefill 和 decode。
本层不重新实现 GPU batch，也不按整波等待或为了补位丢弃历史。

实现为 `src/vllm_kvmem/offline_queue.py`，首个使用方为 256K 图评分对照。

## 固定策略

1. 全部任务进入 FIFO waiting queue。仅保存任务描述，尚未获准的输入不发送到模型。
2. 在途任务数未到上限，且完整历史预留额度足够时准入。预留包括输入、最大输出和 staging；
   不通过截断或重新计算满足内存约束。
3. HTTP 输出结束后进入 retiring；独立的退休线程等待该任务自己的资源释放确认。
   此阶段仍占用在途槽与 CPU 信用，避免把“输出完成”误当作 DMA、历史或检查点释放。
4. 任意一条退休确认完成，立即归还它的额度和槽位，准入 waiting queue 下一条。
   一个慢请求或慢退休不会成为其他请求的整波屏障。
5. 技术失败立即通知拥有服务的控制器，停止补位；未确认释放的额度仍记为占用。
   不自动重试低分或失败样本，不把仍在执行的请求报告为已回收。

模型提交与退休使用不同的线程池，退休不排在长 prefill 工作队列后面。
在途上限同时覆盖 running 和 retiring，所以不能用退休线程池绕过 8 用户与内存限制。
`queue.json` 和 `queue_events.jsonl` 记录 waiting/running/retiring/completed、额度和峰值。

## 独立样本的释放确认

这批 NIAH 输入各自拥有独立 namespace，但不需要跨题保留检查点。
原有非保留配置会在普通退休确认后自动删除空会话，无需额外提交一 token 的 GPU 请求。

补位前同时检查：

- worker 已发布本请求的终态完整历史和 `retirement_ack_ready_at`；
- scheduler 的资源快照时间不早于该确认；
- 该 namespace 已不在 `session_owners`，也不在 CPU `lease_views` 中；
- 该 native request 的 GPU reservation 已消失。

其他用户仍在运行、仍有历史或图元数据时，以上检查照常可以通过。
只在整批结束时使用全局零资源检查。活跃／退休／idle 的会话和 CPU 租约观测用于确认所有权；
这些新增字段不改变 vLLM 或 KVMem 的模型、选页和资源分配策略。

## 本次范围

该入口实现独立离线推理任务的持续补位。agent 跨轮持久会话不能使用上述非保留配置；
工具等待不等于任务结束，必须继续持有 CPU 历史，并分别管理 GPU 驻留与工具就绪队列。
后者应复用本层的任务／资源准入思想和已有 KVMem 会话能力，不能把独立题目的自动释放当作
完整 agent 生命周期调度已经完成。

持续补位也不能单独保证 GPU 满载。当前 Query 启动路径仍把 native 总 token 预算限制为
`query_replay_budget + max_num_seqs × (draft_tokens+1)`；应在独立实验中分开评估总预算、
每请求 chunk 与 CPU 图维护成本，避免把多个改动的收益混在一起。

CPU 验证共 408 项通过，覆盖队列补位／失败、逐所有权释放门禁、会话生命周期、CPU 准入、
恢复协调、服务启动参数以及图评分原有证据门禁。真实模型的连续补位与三组质量对照单独记录。
