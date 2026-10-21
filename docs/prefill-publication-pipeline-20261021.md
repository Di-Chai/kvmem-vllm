# 有界 prefill 发布流水：完整 256K、K32K 性能对照

实现同一请求的下一块 prefill 与上一块 CPU 历史发布重叠后，正式交错对照的
输入吞吐由 **4,576.63 增至 5,147.11 token/s（+12.47%）**。平均首 token
等待由 **57.257 降至 50.911 秒（−11.08%，减少 6.346 秒）**。

两组均启用原生 QK 复用，仅切换 `prefill_publication_pipeline`。本轮不比较
原生复用与旁路 QK，也不把此前不同运行时段的数字作为加速基线。

## 实现与边界

- 新开关 `kvmem.prefill_publication_pipeline` 默认关闭。要求 async CPU scatter、
  split capture DMA、action/prefill attention；暂不与 early capture slot release 组合。
- 对普通非末块 prefill，允许一个已确认的前序 packet 尚未发布时继续执行。
  原有两个 bulk 下载槽保持固定容量；两个 packet 积压时保留回压。
- 前台保存不可变逻辑页布局和独立的已规划采样位置，不读取后台正在修改的
  store 或累计分数。保留部分历史页的原始分段，沿用原采样策略。
- 缓存绑定 owner、请求/窗口 generation、offset、Action 恢复 epoch；Query
  replay、decode 和末块沿用原来的等待。真实 GPU 坐标仍须与计划完全一致。
- 后台按 owner FIFO 完成原事务：图预算预留、历史 append、评分/图提交、容量
  发布。MTP/EOS/cancel 仍由下一次 scheduler decision 确认；refresh、restore、
  checkpoint 和结束继续等待所需历史完成。
- 原生统计的 metadata D2H 和 validate 同步保留。本轮收益来自发布流水与布局
  规划解耦；没有实施多个原生统计帧的延迟验证，也没有修改评分 kernel。

主要实现位于 `src/vllm_kvmem/capture_runtime.py`、`action_attention.py`、
`prefill_planning.py`；`runner.py` 增加模型提交前的非阻塞诊断计数。

## 协议

RTX PRO 6000 Blackwell，Qwen3.8-27B NVFP4、FP8 KV、MTP(k=4)、编译/CUDA Graph。
单活跃请求，完整原始输入 **262,045 token**，K=32,768、G=16,384、chunk=2,048、
batch=16,424；sink=1,312、recent=0，CPU 历史 100 GB、无 SSD、关闭 prefix caching。
K 是重选后的保留预算，活跃上下文可随新增输入接近 K+G=48K。

pressure 使用累计 attention；Query 使用 V2 multi 图传播、alpha=.2；guided
关闭。采样层、密度、精度、归一化域和四项既有优化设置均不变。
输入、随机种子及采样参数沿用上一轮原生 QK 诊断。

先独立完成一次开启组 smoke（32K 预热＋256K），再按 **关闭—开启—开启—关闭**
启动四个服务。每个服务先 32K 预热，再测两次完整 256K，合计每组四次正式样本。
所有组冻结同一份 runtime，配置唯一差异为新开关。

吞吐定义为原始输入 token 数 / 客户端 TTFT，计入统计、归档、提交衔接和重选，
不含模型加载、预热及首 token 后的 decode。首个长请求仍有短预热未覆盖的
重组形状 JIT；每轮的第一、第二遍分别保留，避免将形状准备隐去。

## 正式结果

| 顺序 | 流水 | TTFT 第一次 / 第二次 | 输入吞吐第一次 / 第二次 |
|---|---|---:|---:|
| off-a | 关闭 | 58.370 / 56.301 s | 4,489.36 / 4,654.37 token/s |
| on-a | 开启 | 51.290 / 50.651 s | 5,109.06 / 5,173.49 token/s |
| on-b | 开启 | 51.292 / 50.411 s | 5,108.91 / 5,198.18 token/s |
| off-b | 关闭 | 58.290 / 56.068 s | 4,495.53 / 4,673.74 token/s |

合并吞吐使用总输入 / 总 TTFT。八次均自然输出 138 token，单题答案均正确。
这是单题重复的性能验证，不代替完整 RULER/LME/MAB 准确率测试，也不是八并发结论。

| 每请求均值 | 关闭 | 开启 |
|---|---:|---:|
| TTFT | 57.257 s | 50.911 s |
| Host 提交间隔累计 | 13.936 s | 6.885 s |
| Host 提交调用内部 | 43.053 s | 43.772 s |
| CUDA model 事件区间 | 42.508 s | 42.886 s |
| 14 次 pressure 挂起到就绪 | 4.349 s | 5.439 s |
| Query 挂起到就绪 | 0.936 s | 1.054 s |

模型事件区间基本持平，主要缩短了块间提交等待。pressure 计时仍包含等待后台
追上的阶段；其观测时间增加，完整 TTFT 已计入这项成本。各行存在重叠，不能
相加；host 间隔不是 GPU 空闲证明，model 事件区间也不是纯 kernel 时间。

四轮冷准备分别为 75.141 / 75.136 / 76.145 / 76.155 秒，未计入吞吐。

## 一致性、资源和测试

- 每个正式请求均为 145 个 prefill 批次、8,151 个 prefill 采样行，14 次 pressure
  和 1 次 Query 重选。八次对应选页范围、历史边界、mandatory 内容全部一致。
- 八次输出 token IDs 完全相同。MTP 均为 32 次 draft、128 个候选、109 个接受，
  平均接受长度 4.40625；不存在生成轨迹或接受长度差异造成的吞吐混杂。
- 每请求实际 `reuse:prefill=384`、`reuse:decode=6`，正式请求零回退；仅预热首批
  保留原有三个 `unprepared_forward` 回退。统计生产者无验证失败。
- 每个开启服务（含预热和两遍正式请求）记录 270 次提交时仍有旧 packet 待发布，
  其中 242 次后台正在发布。owner 和 bulk packet 峰值均为 2；全部最终释放。
- 固定 pinned capture staging 两组均为 **1,146,422,304 bytes**。正式请求每秒
  采样的 GPU 占用峰值均为 **72,500 MiB**。这不是更细粒度临时显存峰值证明。
- 进程树 RSS 采样峰值为关闭 **17.628 GiB**、开启 **17.765 GiB**，约增加
  140 MiB。RSS 可能重复计算共享页，不能视为精确的物理内存净增量。
- **44 项新增测试全部通过**：不可变页布局和累计分数逐位对照、延迟提交、
  部分页/重排、MTP 相关尾部排除、Query replay、恢复 epoch、多 owner、取消、
  两槽耗尽、图预算失败回滚、原生验证失败重算、配置和 benchmark 审计。
- 既有发布、Action、native host、图事务、预算、调度、恢复及检查点回归通过；
  CPU 测试环境中一项要求 GPU 的会话状态测试按条件跳过。真实模型测试保留
  编译/CUDA Graph、FP8 KV 和 MTP。

组件逐位比较针对固定输入的计划/提交；实模没有逐页导出全部分数和图边，
不将选页与输出一致扩展为“实模所有中间数值均已逐位核验”。

## 产物与复现

- `artifacts/prefill-pipeline-20261021-smoke-r1/`：先行 smoke，TTFT 50.955 s，
  与上一轮参考的全部 15 次选页及输出一致，不进入正式收益计算。
- `artifacts/prefill-pipeline-20261021-r1/`：ABBA 四组冻结源码、配置、逐请求
  结果、资源采样、`comparison.json`、组件 JUnit XML 和最终清理证明。
- `scripts/benchmark_prefill_publication_pipeline.py`：`prepare --root <新目录>`
  冻结实验；使用生成的 `<目录>/driver.py run --root <目录>` 执行。
  `--smoke --repeats 1` 仅准备先行验证。继承的两个历史辅助脚本明确记录来源并核验哈希。

本轮在 main 当前工作区实现，保留原有其他修改；新开关默认关闭。测试服务均已
释放，未启动 DSH，未恢复完整 RULER/MAB。
