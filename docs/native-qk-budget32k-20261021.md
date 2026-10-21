# 完整 256K 输入、32K KVMem budget：原生 QK 复用吞吐

本次按用户确认测 **完整原始 256K 输入、KVMem 保留预算 32K**，并非只输入 32K。
开启原生 QK 复用后，合并 prefill 吞吐由 **4,716.68 降至 4,277.33 token/s**，
下降 **9.31%**；平均首 token 等待由 **55.557 增至 61.264 秒**，增加 **5.707 秒**。
两次开启组都慢于相应关闭组。这是单请求、每组两遍的观测，不是八并发结论。

原生复用确实执行，正式请求没有回退；选页和输出相同。当前实现未在这一场景
带来端到端收益，生产默认继续关闭。没有启动 DSH，也未恢复完整 RULER/MAB。

## 实验设置

- 同一 RTX PRO 6000 Blackwell Server Edition；Qwen3.8-27B NVFP4、FP8 KV、
  MTP(k=4)、编译/CUDA Graph；单活跃请求，max sequences=8。
- K=32,768、G=16,384、chunk=2,048、batch=16,424；sink=1,312、recent=0。
  K 是重选后的保留预算；新增输入使用 G 余量，活跃上下文会接近 48K。
- CPU 历史预算 100 GB，无 SSD，关闭 prefix caching；guided 关闭。
- pressure 仍用累计 attention，Query 用 V2 multi 图传播，alpha=.2；采样层、
  密度、归一化域、图参数不变，四项既有性能优化均开启。
- 两组冻结 runtime 文件哈希完全相同，profile 唯一差异为
  `kvmem.action_native_scores="off"` 或 `"reuse"`。关闭组仍用融合旁路评分。
- 每个服务先跑一次完整 32K 预热，再依次跑两次完整 256K。顺序为 off/off，
  然后 reuse/reuse，未交错启动服务。加载和预热不计入正式吞吐。
- 输入是同一原始 RULER 256K 档样本，渲染后 **262,045 token**，SHA-256：
  `ed7bbc143b62c931d0d9ddf79e4101dafdb7361f65ef18cf3065a5ee793e0778`。
  该单题用于性能诊断，不能代替完整准确率 benchmark。
- 沿用此前 prefill 测试的采样参数：seed=42，temperature=1，top_p=.95，
  top_k=20，最大输出 8,192、thinking budget=4,096；实际均自然输出 138 token。
  最大输出与 KVMem 的 G=16K 容量预算是不同参数。

## 实测结果

吞吐为完整输入 token 数 / 客户端首 token 等待时间，包含持续 prefill、统计、
归档、重选与提交间隔；不含随后完整 decode。没有用日志的一秒滑动吞吐作比较。

| 路径 | 第一次 TTFT | 第二次 TTFT | 合并输入吞吐 |
|---|---:|---:|---:|
| 原生复用关闭 | 56.597 s | 54.517 s | 4,716.68 token/s |
| 原生复用开启 | 60.228 s | 62.299 s | 4,277.33 token/s |

两次单独吞吐分别为 off 4,630.03 / 4,806.63，以及 reuse 4,350.88 / 4,206.23。
冷准备分别为 off **75.158 秒**、reuse **98.186 秒**，没有混入上表。
每组的第一次长请求仍可能触发短预热未覆盖的重组内核 JIT；第二次未见新的
JIT 警告。两遍单独列出，不能宣称所有首次形状准备都被 32K 预热消除。

本次是完整 256K 输入在 32K 保留预算下的速度，不能直接拿它与原生仅输入
32K 的约 9.6K token/s 解释为同等注意力计算量。

## 实际复用及一致性

当前环境的 18 项 host 契约测试、19 项 GPU 测试均通过，覆盖 FA2/XQA、GQA、
FP8 scale、causal/部分/非连续页、MTP 部分接受、CUDA Graph 重放及旧数据污染。
首次检查因测试进程没有虚拟环境的 Ninja 路径而失败；补齐 PATH/CUDA_HOME 后
完整重跑通过，未因此改动内核。使用的 source fingerprint 为 `e23711467a3dafc0`。

正式开启组每请求实际记录：

- **384 次 `reuse:prefill`**、6 次 `reuse:decode`；两次合计 768 / 12。
- **0 次 fallback、0 次 producer validation failure**。
- 预热首批三个采样层有 `prefill:unprepared_forward` 回退；只发生于预热，
  保留在全局累计报告中，没有将其冒充原生复用。
- 统计持久缓冲为 **473,800,368 GPU 字节（约 451.85 MiB）**，另有
  352,684 字节 pinned host 缓冲；总计 474,153,052 字节。

每个正式请求均发生 **14 次 pressure 和 1 次 Query 重选**。两组相应的全部
15 个选页事件，历史边界、选页范围集合及 mandatory 内容均一致。
输出 token IDs 在两组和两次重复之间也相同，都是 138 token，答案正确。
MTP 每请求均为 32 次 draft、128 个提议 token、109 个接受 draft token，
平均接受长度 4.40625。因此，本次观察到的差异没有生成轨迹/MTP 接受长度混杂。

接受行样本、完整逻辑历史、独立 owner 释放和最终完成栅栏均通过。
这些核验不宣称新旧页分数逐位相同；数学语义与允许的浮点误差仍按原协议处理。

## 当前时间证据

两次正式请求均记录 145 个 public prefill 批次、262,045 个实际输入 token。
下表均为两次均值：

| 记录项 | 复用关闭 | 复用开启 |
|---|---:|---:|
| CUDA model 事件区间累计 | 42.735 s | 42.992 s |
| Host 提交调用内部时间 | 42.125 s | 43.618 s |
| Host 提交调用之间的间隔 | 13.154 s | 17.412 s |
| 14 次 pressure 挂起到就绪合计 | 4.193 s | 4.412 s |
| 最后 Query 挂起到就绪 | 1.030 s | 0.963 s |

重复 QK 已省去，但本次记录的 model 区间没有下降，提交间隔反而增加约
4.258 秒。原生复用仍有元数据准备、LSE 写出/归约、验证和相应调度成本。
这些记录支持进一步检查复用路径的准备及提交衔接，但尚未把增量精确归因于
其中某个调用，不能据此断言某一个同步点独自造成全部回退。

CUDA model 区间含统计、捕获及提交间隙，并非纯模型 kernel 时间；host 间隔
也不是 GPU 空闲证明。重选时间嵌套在完整请求内，上述各行不可相加。

## 可复核产物

`artifacts/native-qk-budget32k-20261021-r1/` 链接到数据盘，包含：

- `off/`、`reuse/`：冻结 profile/runtime、实际 launch、逐请求结果、接受行、
  native 分项计数、资源释放和服务 cleanup。
- `component-tests.txt`：37 项组件测试结果。
- `comparison.json`：吞吐、选页集合、输出 token 哈希、MTP 和阶段时间。
- `analyze.py`、`finish-analysis.py`、`analysis-proof.json`：离线分析和哈希。
- `final-state.json`：全部请求完成、所有 owned session 释放、GPU 空闲。

本轮仅扩展 `scripts/benchmark_matched_prefill.py`，允许显式 32K budget 和
off/reuse 对照，校验短输入跳过重选、长输入真实重选以及实际原生复用。
未修改评分内核、生产默认预算或默认复用开关。

后续已实现同请求的有界 prefill 发布流水。固定原生复用开启的新一轮 ABBA
对照中，吞吐由 4,576.63 提升至 5,147.11 token/s（+12.47%），见
[发布流水实验](prefill-publication-pipeline-20261021.md)。该比较只切换流水开关，
与本文 off/reuse 对照属于不同实验，不能混合计算模块收益。
