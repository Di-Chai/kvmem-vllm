# 2026-10-18 开发基线

本次提交固定此前在当前工作区完成的 KVMem 与 vLLM 集成，作为下一轮评分质量对照的代码基线。
包含完整 CPU 历史、FP8 raw K/V、GPU active view、Query 规划、MeanK、input/output attention、
guided/history 语义、P/M 检查点、异步恢复与刷新、压力驻留、离线就绪请求调度及分项观测。
Query 驱动的一跳 block 图评分已集成，默认关闭。

实现范围与剩余差异见 [实现状态](implementation-status.md)、[qw3 对齐](qw3-alignment-20261018.md)
和 [图评分](graph-retrieval.md)。不同候选的验证边界仍以各报告冻结的运行时为准；本次代码提交
不把所有可选配置的 CPU 测试概括为真实模型、任意组合的完整验收。

## 已有证据与本次提交检查

- 当前源码的图评分、上下文、恢复、调度和启动器 CPU 回归重新执行：**630 passed**。
- 现有 Debug 构建的 C++ 回归重新执行：**5/5 passed**；333 个待提交 Python 文件 AST 检查通过。
- 先前完整 FP8 八并发固定 300 秒对照已完成，零抢占，同样 779 个 native KV blocks。
  服务窗口输出率为 145.73 → 146.95 token/s，没有证明稳定的整体吞吐提升。
  已确认的是 CPU 组件降耗、有界 staging 复用及 CPU/GPU 内存余量通过。
- 图评分原小规模真实模型实验六组均为 6/6，属于功能正确性和天花板样本验证；没有证明质量提升。
  当前源码重放固定选页输入与其结果一致。

完整性能口径、失败尝试及资源边界见 [主机余量与 FP8 验收报告](host-admission-calibration-20261018.md)。
本机原始模型、输入与大体积运行证据仍在忽略的 `artifacts/`，不包含于 Git 提交。

## 后续评分质量对照

下一轮首先比较 Query、Query＋累计 attention、Query 驱动 block 图评分。
沿用 qw3 冻结的 256K 大海捞针原始完整输入和评分器，同一模型、K/G、保留规则、Query 坐标、
采样参数与逐题 seed；不根据首轮答案筛题或重试低分样本。

首轮受控对照保持 Query history、guided Query 和 attention pressure 策略一致，避免把它们的
差异归因于图传播。图评分暂不支持的组合不在运行中静默降级；与完整 attention pressure
部署的比较另作明确标记。报告逐题覆盖率、整题正确数、证据选页、图实际生效及成本。
过去不同源码或不同 pressure 设置的 97.08% 结果仅作历史参照，不作为严格的同配置对照。

本节为实验约束，不代表这些较长真实样本已完成新的三组测试。
