# Guided Query 与跨轮相关性分数

本轮按用户要求暂不迁移调用方 `evidence.changed` 和跨轮增长阈值触发。
新增显式 Query 边界的 private guided rewrite，以及 Query relevance history，
允许同时启用。这是对 qw3 组合限制的扩展，不宣称该组合与原版逐步等价。

## 执行语义

`guided_query_tokens: 32` 在既有显式 Query 选页点启用内部检索 Query。
先执行完整公开 Query 的 provisional pass，再根据当前 active view 内可见的
任务、进展和工具结果生成最多 32 个内部 token；只捕获生成文本的 target Q。
模板沿用 qw3 的无 thinking 私有检索指令。空生成回退原公开 Query；warm M 恰好
落在新 Query 边界且没有额外临时空间时，也明确回退公开 Query，不能超出K+G。
回退原因写入 `guided_fallback`。不会调用
外部检索、BM25 或用 token 字符串近似模型注意力。

内部 prompt 和生成使用 native allocator 提前预留的临时页，包含在 K+G 中。
所有邻居的 GDN 状态均有 probe 前快照；选页后恢复，再装配并重放公开输入。
私有 token 不进入历史 KV authority、会话 token 列表、用户流或 MTP 采样状态。
私有生成采用 greedy target forward；公开 prefill/decode 保持原生编译、CUDA
Graph 和 MTP(k=3)。私有生成本身是同步 eager，因此会增加发生重选的首 token 延迟。

`query_history: true` 保存的是历史页相关性分数，不是长期保存 Q 张量。
归一化、衰减、严格父回合选择、abort/snapshot 规则直接使用未修改的
`third_party/qw3/src/kvmem_query_history.hpp`。有效观测时：

```
本轮候选分数(page) = decay × 父回合分数(page)
                      + 本轮 MeanK 分数(page) / 本轮正分数总和
```

合并后的分数送入原版 Store TopK；必留页仍优先由 mandatory 策略保证。
与 guided 同开时，本轮 MeanK 分数来自内部 Query 的 Q。每次 preview 都从
同一个不可变父回合计算，反复选页不会反复累计；完整公开 prefill 接受后才
commit。一轮没有新 Query 或没有正分数时不衰减旧证据。重试同一完整输入
从严格父回合重新算；分支根据完整 token 公共前缀查找祖先。

分数按逻辑历史页起点保存，映射回被 lease 保护的当前 generation 后选页。
不同用户的 authority 和分数对象独立；旧物理 slot 的分数不会直接给新页使用。
普通工具续接只保留历史分数，不因此触发新选页。容量压力仍走原有压力策略。
本轮不引入新的证据变化、增长阈值或 decode 中途语义触发。

## 配置与边界

可直接参考 [组合配置](../examples/qwen-nvfp4-guided-history-elastic.json)，弹性 profile 增加：

```json
{
  "request_planning": true,
  "query_history": true,
  "query_history_decay": 0.5,
  "guided_query_tokens": 32
}
```

两项默认关闭。历史分数要求配置有界 prompt sessions，未绑定 session 的原生
请求仍可执行普通 MeanK，但不跨轮累计分数。guided 只作用于历史超过 K 的
既有显式 Query 选页；不自动补聊天模板 Query，不替代 DSH 的坐标／会话适配。
历史不足 K 时，不需要 guided 选页，但仍可学习公开 Query 的相关性分数。

私有 prompt 预留 256 token，再加 guided 输出上限并按 B 对齐。B=128、上限32
时预留384 token；完整公开 Query replay 上限从 G−B 调整为 G−B−384，即
G=16,384 时为15,872 token。超预算或超模型位置范围的请求在入口拒绝。

qw3 原版禁止 query history 与 initial/boundary guided Query 同开；本版的扩展
定义为一次公开输入只提交一次 guided relevance。没有合并公开 Q 与私有 Q 的
两次独立奖励，也没有把私有文本当作一个新用户回合。

仍缺的独立能力包括 Q 张量持久缓存、任意历史编辑／完整 checkpoint ladder、
默认聊天 Query/DSH 协议适配、独立 prefill budget、prefill-only ingest、增量 GPU
MeanK index 等；完整状态以 [实现清单](implementation-status.md) 为准。

## 验证

本轮日志位于 `artifacts/guided-history-20261014/`。测试同时检查实际选页受到
历史分数影响、重复 preview/输入重试不重复奖励、无 Query 不衰减、分支 abort、
用户隔离、极端浮点分数、private batch 不覆盖公开输入和临时页准入。

真实模型使用 Qwen 27B NVFP4、target/MTP FP8 KV、MTP(k=3)、原生编译与
CUDA Graph，K=32K、G=16K，恢复字节 oracle 开启。

- CPU：663 passed / 73 skipped；原版 C++ query-history 测试通过，导入头文件哈希一致。
- C8 五轮：40 回合、32 次 P/M 恢复、24 次 Query 重放、40 次逻辑输入提交；
  每用户历史增长到73,447 token，覆盖20K工具追加后的容量压力，最终GPU页归零。
  21次产生内部 Query，3次空生成回退公开 Query。无 Query 回合的分数哈希不变。
  该组在后续准入／预算内重试修正前完成；最终构建另行复测。
- 最终构建预算内测试：13,055 token输入，原输入、相同输入重新生成、无 Query
  续接三次请求均成功；提交轮次分别为1、1、2，观测标记为true、true、false，
  最后两次分数哈希相同。相同输入重新生成时不承诺Q浮点结果逐位一致。
- 最终主体构建C8两轮：16回合、16次guided Query重放、8次字节校验恢复，
  GPU页归零；16次内部Query均非空。内部生成耗时中位3.98秒，最大5.44秒。
- 同构建C8，50,006输入＋每路4,096输出：稳态729.37输出token/s，整轮199.18，
  全部8路成功；共同窗口37.67秒，原生8路稳定覆盖99.40%，TTFT p95为108.27秒。
  每路确实经历guided选页与历史评分，之后进入稳态decode；不是只打开配置但
  未触发选页的测试。输入与此前32K性能文本不同，不据此宣称优化收益。
  这些GPU数据在最后补充“warm M无临时空间回退”保护前测得；保护的正常分支
  不改变这些请求的执行。最终服务另过C1两轮各1输出token的guided/历史评分/
  恢复冒烟，2次重放、1次字节校验恢复，GPU页归零；边界针对回归36 passed。

本机最终测试服务使用组合profile，运行目录为
`artifacts/guided-history-20261014/server-06`；token-ID弹性入口18382、原生后端18381。
服务保留恢复字节oracle用于验收。它仍不是DSH聊天协议的自动兼容入口。

并发测试发现并修复了两个生命周期错误：评分不能在 `submit` 中提前 drain
刚采样的批次，否则会抢在邻居 MTP terminal clip 前发布多余KV；预算内重试
不能用上一请求的 observation 标记抑制新Q捕获。另修正临时页准入中的
逻辑B=128与原生GDN对齐物理页大小的区别。取消未完成的输入会丢弃候选分数及Q。

结构化结果见 [验收记录](benchmarks/guided-history-20261014.json)。功能通过不代表
已证明 agent 质量提升、长期工具负载 SLO 或与 qw3 的生成结果等价。
