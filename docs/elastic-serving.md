# NVFP4／FP8 弹性会话入口

**新增原版 MeanK 模式：** `context_mode: "meank"` 自动选择并重装配历史 KV，
保留完整 CPU 历史，使用 NVFP4＋FP8 target KV＋DFlash2＋decode CUDA Graph。
配置见 `examples/qwen-nvfp4-meank-elastic.json`，实现核对、预算和验证边界见
[MeanK 集成说明](meank-integration-20261013.md)。下文描述默认 `context_mode: "full"`
的原生 MTP offload 路径；其中显式 `active_ranges` 只适用于该模式。

新增 `vllm-kvmem elastic-serve`：原生 vLLM 负责完整上下文计算、CUDA Graph、
MTP 与混合状态 offload；KVMem 网关负责用户身份、完整 token 历史、会话版本、
active 字节预算、恢复并发和排队截止时间。此路径没有启用旧的 eager capture、
DFlash2 scheduler 或 MeanK 稀疏 KV 选择。

实际验证环境为 vLLM 0.30.0 V2 runner、Qwen3.8-27B NVIDIA ModelOpt NVFP4、
单 RTX PRO 6000 Blackwell Server 96GB。模型注意力 KV 和 MTP KV 继承全局
FP8 e4m3；GDN state 保持原生精度。测试结果见
[2026-10-13 测量](elastic-throughput-20261013.md)。这是实验入口，非生产支持声明。

## 配置与启动

复制 `examples/qwen-nvfp4-mtp-elastic.json`，替换模型、tokenizer 和 users 文件。
路径相对配置文件解析，支持 `~`。users 文件格式沿用多用户入口：

```json
{"users":[{"id":"alice","api_key":"replace-with-a-long-random-key"}]}
```

只需目标 checkpoint 内置的 MTP，不加载独立 DFlash draft。先检查启动参数：

```sh
.venv/bin/python -m vllm_kvmem elastic-serve \
  --config elastic-profile.json --run-dir runs/elastic-01 --dry-run
```

去掉 `--dry-run` 启动。已有运行目录拒绝覆盖。安装的 `vllm-kvmem` 命令也支持
同样参数。后端仅绑定 `127.0.0.1`，网关绑定 profile 中的 host／port；一套会话
账本使用一个网关进程。保持后端为内部端口，网关不会转发后端管理接口。

本机已构建、安装并核验的 wheel 位于
`artifacts/elastic-serving-20261013/dist/`。本轮仍为 `0.1.0.dev1` 开发快照；安装
同版本新构建需要显式重装，例如 `uv pip install --reinstall --no-deps WHEEL_PATH`。
vLLM／CUDA 依赖应先按现有安装说明配置。

## 回合 API

`POST /v1/kvmem/sessions/{session_id}/turn` 使用 Bearer 用户 key。
同名 session 在不同用户下完全独立；缓存盐由服务器生成，客户端不能覆盖。

```json
{
  "version": 0,
  "append_token_ids": [123, 456],
  "max_tokens": 256,
  "waiting_for_tool": true
}
```

第一次发送 `version:0` 和完整已套模板的 prompt token IDs。响应包含新的
version、历史长度和 `native_response`（原生文本、准确的 output token IDs、usage）。
下一轮只追加工具结果／用户消息的 token IDs，并提交最新 version。服务自动保留
输入和已接受的输出 token，不把 MTP 未接受的候选写入历史。

调用方负责 tokenizer／chat template 和实际工具执行；本入口返回原始生成文本，
不解析或执行工具命令。工具返回内容须按模型的实际模板编码为增量 token。
使用准确输出 token IDs，避免重新编码生成文本导致 thinking／工具标记变化。
当前回合接口返回完整 JSON，内部使用原生 SSE 取得准确计数和首 token 时间；
它没有向调用方提供流式输出。长回合的可见响应时间为 `turn_seconds`。

一个用户同时最多运行一个回合；同会话的并发或旧 version 返回409。无法在排队
截止时间内进入 active／restore 预算返回429。上游错误或断连不提交半截历史，
下次仍使用原 version。`ignore_eos:true` 是固定长度性能测试选项，正常工具回合
应保留默认 false。

`turn_timeout_seconds` 是从回合验证完成后起算的绝对时间上限，包含 active 排队、
恢复排队及生成；超时返回504。排队时也检查客户端断连。超时／断连后先等待
上游请求关闭，再释放 active 额度，所以取消清理可能越过截止时刻；这不是硬 SLO 保证。

`GET /v1/kvmem/status` 返回本用户会话和聚合预算指标。
`DELETE /v1/kvmem/sessions/{session_id}` 删除本用户空闲 token 历史；活动回合拒绝删除。

## 内存和调度语义

- `active_bytes`：根据后端实际页几何，保守预留输入、输出增长、MTP 和 GDN state。
  这是回合 admission 预算，原生 KV allocator 仍负责物理硬上限和在途复制页租约。
- `cpu_kv_bytes`：原生 CPU KV offload 池的物理限额；采用 LRU。完成块异步复制，
  GDN checkpoint 与 MTP 尾块由原生接口处理。缓存未命中时完整重算，不静默删历史。
- `token_history_bytes`／`token_history_bytes_per_user`：CPU 完整 token 历史 payload
  及待提交输出的额度；另有全局／单用户 session 数限制。
- `restore_bytes`／`max_restore_inflight`：保守限制同时恢复的字节和请求数量，收到
  首个已接受 token 后释放恢复额度；active 额度保持到整个生成结束或取消。
- `queue_timeout_seconds`：全局最大 admission 等待时间。每回合可设更短的同名
  deadline；满足字节预算且用户未占用 active 的候选按最早截止时间进入。
- `session_ttl_seconds`：回收空闲 token 历史，活动回合不被 TTL 清理。工具等待时
  active 租约已经释放；GPU prefix cache 可作为无请求所有权的可淘汰缓存保留。

CPU KV 缓存和完整 token 账本有不同生命周期：删除／TTL 回收账本不会立即逐 session
清除原生 LRU KV，KV 池始终受全局容量限制。指标中的 CPU cache usage 是在途传输
pin 比例，不能拿它表示全部已缓存内容。token payload 额度也不是进程 RSS 限额，
还需为 Python 索引、请求体、native 元数据和传输池留内存。

本机示例使用40GiB active、48GiB CPU KV、8GiB restore、4路恢复及16路 active。
48GiB CPU offload 使用 `/dev/shm`；需满足实际 shm／主存容量。刚完成生成不等于
后台复制已结束，页复用和缓存清理由原生 completion fence 保护。

## 显式有限 active context

默认 `active_context_tokens:0` 保留完整上下文。启用例如8192后，超限回合必须明确
提交 `active_ranges`；未提供选择范围就返回429，服务不会自动裁剪历史。

```json
{
  "version": 1,
  "append_token_ids": [789],
  "active_ranges": [[0, 128], [16000, 16256], [26000, 32769]],
  "max_tokens": 256
}
```

范围为当前完整历史加本次 append 后的 token 下标，左闭右开，必须有序且不重叠。
选出的 token 总数必须不超过配置限额。上例仅展示字段语义，真实范围须结合会话长度
和任务确定。响应 `input_context` 列出完整／active 输入长度和所用范围。

完整 CPU token 历史始终保留。选中片段拼接为本轮原生输入，只有准确相同的选择
prefix 才能命中缓存；上下文改变时重建相应状态，不套用不匹配的旧 GDN checkpoint。
这是一种显式回合上下文选择，可能改变任务答案；没有自动 MeanK 打分、任意历史
原位 sparse KV 装配或与完整上下文等价的保证。

## 验证边界

此次覆盖真实模型 CPU 恢复、完整／有限上下文、32客户端、正常停止检索和取消／
deadline 回收。工具等待为回合之间的模拟等待。持续负载下的工具 CPU／磁盘／
传输竞争、通用任务质量、硬 SLO、跨 worker／重启持久化和 NVMe 历史层尚未验收。

## 显式 Query profile 的 P/M 上下文策略

`qwen-nvfp4-meank-explicit-query-elastic.json` 默认启用 P/M：稳定续接保持选页，
依据实际 active window 与新增输入/输出判定 headroom；不足时才做容量压力刷新。
`kvmem_mandatory_spans` 和 `kvmem_mandatory_parts` 管理必留内容，超额使用原版 qw3
容量适配；`kvmem_replay_span` 可涵盖 Query 前的消息 header。

此 MeanK 路径的 CPU historical authority/read cache、NVMe arena、P/M 检查点和
完整 token 账本分别有预算。8 用户 P+M 示例预留 4 GiB 检查点；配额不足会保留可用的
旧检查点并增加重放，而不会截断逻辑历史。网关全局 active 字节仍按 K+G 保守预留。
字段、实际测试和剩余协议限制见 [跨轮上下文报告](context-policy-20261014.md)。
