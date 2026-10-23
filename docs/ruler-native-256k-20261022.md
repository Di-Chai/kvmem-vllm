# 原生 vLLM 完整 256K 上下文 RULER 对照

2026-10-22 启动。使用刚完成的 KVMem RULER 同一套冻结输入，不重新抽题或生成数据。

- 实验目录：`artifacts/ruler-v1-256k-native-20261022-r1`，实际文件保存在数据盘。
- 参考组：`artifacts/ruler-v1-256k-query-replay-20261022-r1`，260/260 完成，官方任务均分 **85.75**。
- 本组：纯原生 vLLM，保留完整输入 KV，不使用 KVMem 历史归档、评分、构图、图传播或重选。
- 不启动 DSH，不恢复已暂停的 MAB。

## 保持一致的条件

13 类任务，每类 20 题，共 260 题。逐题复用原 `.i32` 输入 ID 和 SHA256、任务顺序、随机种子（42 + 题内序号）、参考答案及官方评分函数。名义输入长度 256K；原数据实际为 247,313～262,130 token，不补齐、不截断。只评分 thinking 结束后的最终答案。

| 项目 | 两组共同设置 |
|---|---|
| GPU | RTX PRO 6000 Blackwell Server Edition，约 96 GB |
| 模型 | 相同本地 NVFP4 权重与 tokenizer |
| KV cache | FP8 E4M3 |
| 推测生成 | 原生 MTP，k=4 |
| 编译 | mode=3，FULL_AND_PIECEWISE CUDA Graph |
| 引擎最大序列数 | 8 |
| 引擎 batch token 上限 | 16,424 |
| GPU memory utilization | 0.72 |
| prefix caching / async scheduling | 均关闭 |
| 输出上限 / thinking budget | 8,192 / 4,096 token |
| 采样 | temperature=1、top_p=0.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1 |
| max_model_len | 524,288，预留 256K 输入后的输出空间 |

软件环境：vLLM 0.30.0、PyTorch 2.13.0+cu130、FlashInfer 0.6.18.post1、Transformers 5.17.0。

## 必须单列的执行差异

1. KVMem 组选择 36,864 token 历史并预留 16,384 token 生成空间；原生组所有输入 KV 可见，无 KVMem budget 或重选。
2. KVMem 逻辑 prefill chunk=2,048。原生组设置 `long_prefill_token_threshold=2048`；保留同一 batch token 上限。两组实际物理分块和混合批次调度仍可不同。
3. 原生组使用原生 hybrid KV block/state 策略，不加载 KVMem 对状态管理的修改。
4. 引擎最大序列数保留 8。客户端实际并发按启动日志的 KV 容量准入：每路预留完整最长输入、完整输出 allowance、MTP lookahead、页对齐和两页余量；全局再预留 5% 和一个 batch。最终并发写入 `run/admission.json`。不通过显存过度准入制造抢占和重算。
5. 第一题单独执行，核验完整 256K 输入及输出 token 计数，计入准确性结果及总耗时；其余题采用连续 FIFO。
6. 同一随机种子不能保证不同上下文及合批过程的生成逐位相同。最终逐题比较质量，并单列生成长度、MTP 接受统计及并发差异。

原生启动命令由参考组实际 `launch.json` 派生，移除 KVMem scheduler、connector、middleware 和编译捕获标记。服务直接启动 vLLM CLI，`VLLM_PLUGINS=''`，清除 `KVMEM_*`、父进程的 `VLLM_*`、`PYTHONPATH` 等配置后仅恢复必要的原生设置。冻结运行目录只包含标准库客户端和控制脚本，不包含 KVMem 模型运行代码。

## 核验与结果文件

相关 CPU 校验 **94 项通过**，覆盖完整上下文准入、容量不足、元数据及种子复用、原生启动参数剥离、环境隔离和官方评分协议。

本次实测启动冷准备 83.14 秒，原生 KV 容量 1,295,299 token、原生 attention block=1,616 token。准入每路预留 274,720 token，实际并发 **4**，引擎最大序列数仍为 8。14:43 首题 `niah_single_1-0000` 完成：完整输入 262,045 token，输入/输出计数通过，官方题分 100，抢占计数 0。随后已进入 4 路并发；这只是首题核验，不是完整套件结论。

- `native-launch.json`、`launch-proof.json`、`runtime-proof.json`：命令、差异与冻结代码摘要。
- `run/server.log`：实际原生缓存容量、原生后端、模型加载和 MTP 日志。
- `run/admission.json`：实际并发及容量计算。
- `run/protocol-qualification.json`：首题完整输入与输出计数核验。
- `run/answers/progress.json`：完成数量及任务分布。
- `run/answers/cases/*/{request,response,result}.json`：逐题采样设置、原始输出、token ID、TTFT、总耗时、评分。
- `run/resources.jsonl`：5 秒采样的原生服务指标、CPU 可用内存和 GPU 显存/利用率；保留 prefill、decode、MTP 接受与抢占计数。
- `run/answers/{scores,completed}.json` 与 `report.md`：完整 260 题结束后才生成官方总分及参考组对照。
- `run/cleanup.json`：实验结束时仅清理本实验进程，记录 GPU 释放状态。

若出现抢占、输入/输出计数不符或其他推理异常，则保留失败记录并停止，不把不完整结果算作完整测试。全轮 token/s 包含长 prefill、生成及首题核验，不能当作纯 decode 吞吐。两组并发不同，不能直接把总耗时差全部归因于 KVMem。
