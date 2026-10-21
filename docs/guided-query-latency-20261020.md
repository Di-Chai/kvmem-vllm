# Guided Query 私有生成加速（2026-10-20）

本次优化私有 Query 的执行开销，未改变 private prompt、512 token 上限、
前缀过滤、自然终止要求、采样层或后续 V2 multi 图评分规则。
公开回答仍使用 DFlash2(k=7)；私有 Query 仍是 target 贪心生成，未接入 DFlash。

## 改动

- 新增 `kvmem.guided_query_backend`：`eager` 为原参考路径和配置默认值，
  `cuda_graph` 为新路径。当前 DSH 服务显式启用新路径。
- 私有单 token forward、greedy argmax、评分 Q 写出和位置／长度／slot 更新进入
  一个 CUDA Graph。只在分支开始准备 native 元数据，复用独立输入缓冲区。
- 当前适配 Qwen GDN 非投机单 token 状态与 FlashInfer XQA。原生元数据中的
  DFlash draft 层不进入 target-only 私有 forward。其他布局明确回退 eager。
- 图只存活于一次串行 refresh，不跨 owner 或请求保留。采样 Q 每次复制后再重放，
  避免后一个 token 覆盖旧行。已有 scratch 预留和 K+G 容量规则不变。
- warmup 只快照实际会修改的 GDN 基础行和接受计数器，包含异常分支的 stream fence。
  全行、其他 owner 和五层 draft 的外层回滚验证保留。
- warmup 快照上限 512 MiB，分配前检查；图建立后新增 live allocation 超过
  256 MiB 则释放并回退。当前模型实测分别为 153.38 MiB 和 32.19 MiB。
  资源在私有分支结束时释放，CUDA allocator 可保留已经释放的 reserved 内存。
- 记录实际执行模式、回退原因、重放次数、图／快照字节和准备时间。
  验证模式额外从相同初始状态执行 eager，逐 token hash 和评分 Q 做完全相等比较。
  GPU 时间仅在验证模式采样，普通服务不执行该对照或采样。

## 真实 GPU 测量

NVFP4 target、FP8 KV、DFlash2(k=7) 公开生成、FULL CUDA Graph；
K=36,864，G=16,384，sink=1,024，recent=2,048，CPU 历史 100 GB，无 SSD。
所有测量保持相同 private prompt 和自然生成长度。

| 输入与私有生成 | eager | 新路径 | 收益 |
| --- | ---: | ---: | ---: |
| 完整 DSH 58,737 token 输入，私有生成 135 token，同状态配对三次均值 | 14.486 s | 2.883 s | 5.02 倍，耗时下降 80.1% |
| 原长轨迹的精确因果前缀 58,655 token，私有生成 217 token，一次配对 | 22.776 s | 4.403 s | 5.17 倍，耗时下降 80.7% |

第一项重建全部历史请求的实际 accepted token 前缀，不重新编码模型已生成的代码。
第二项原完整请求有 193,209 token；只保留原私有 Query 边界前的精确前缀，
省去该 Query 之后的公开工具步骤，属于私有组件测量。

新路径计时包括私有 prompt forward、缓冲准备、warmup、图捕获、生成、Q 收集和图清理。
不包含外层公共探测、选页／图传播、窗口组装和外层检查点回滚。
配对模式整个 `guided_seconds` 还包含旧路径 oracle 和验证副本，不能当作普通服务开销。
两路径按“新路径后旧路径”的固定顺序从相同状态执行，未做端到端开关对照。

最终版本另一次 135 token 配对为 14.224 → 3.050 s。
普通服务关闭 oracle 后，三次完整 `guided_seconds` 为 3.466、2.888、2.901 s：
首轮包含约 0.62 s 的懒加载初始化，后两次平均 2.895 s。普通运行没有旧路径重复生成。
CUDA 事件采样显示单步图执行约 18.3–18.4 ms；剩余主要是 target 串行模型计算。
目前不保证 100–200 token 都在 1–2 秒内完成。私有分支的投机生成是后续方向，
不能把公开回答的 DFlash 吞吐套用于私有分支。

## 验证

- 308 项 CPU／配置／策略测试通过；最终独立 CUDA 组件测试 16 项通过。
- 两个 owner、每个三轮，覆盖 Query、工具追加后新 Query、后缀编辑和跨轮 P/M 恢复。
  5 次发布私有 Query，全部 token hash 和评分 Q 与 eager 完全相等。
- GDN 全状态和接受计数器、五层已提交 DFlash KV 字节回滚通过；其他 owner 隔离通过。
  接受输出 attention 计数与独立 token parser oracle 相同，私有行和拒绝行没有混入。
- GPU 组件覆盖旧 Q 不被新 replay 覆盖、元数据地址变化拒绝重放、内存预算耗尽回退，
  以及 warmup 抛错后等待 GPU 完成再恢复状态。
- 所有测试自建 namespace 显式释放。公开生成上限、历史和上下文预算未调整。

## 证据与复现

- 三次相同状态配对：`artifacts/guided-speed-20261020-r3/dsh-latency/results.json`
- 最终两 owner 验证：`artifacts/guided-speed-20261020-r4/qualification/results.json`
- 最终 135 token／GPU 时间：`artifacts/guided-speed-20261020-r4/dsh-latency/results.json`
- 217 token 前缀：`artifacts/guided-speed-20261020-r4/long-prefix-latency/results.json`
- 普通服务三次完整计时：`artifacts/guided-speed-20261020-r4/production-latency/results.json`
- 当前服务：`artifacts/guided-speed-20261020-r4/profile.json`、`process.json`、`source-proof.json`

`scripts/qualify_guided_dflash.py` 验证多 owner 状态和接受行隔离；
`scripts/qualify_guided_latency.py` 可对保存的输入执行配对验证或普通服务测量。
本次没有重跑八并发端到端吞吐，不能把私有组件的 5 倍提速视为整轮提速。
