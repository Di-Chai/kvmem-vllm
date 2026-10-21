# 逻辑 2K 统计事务不变，物理 prefill 合并 4K/8K

状态：只读可行性设计；未修改实现、未进行 GPU 或性能测试。参照是当前最佳实验的逻辑 2,048-token 普通 prefill，不是仓库历史默认 1,024。完整输入、K/G、采样、重选和原生 QK 复用均保持。

## 判断

值得作为下一阶段的结构性候选，潜在收益来源比减少一个 CPU 小循环更广：更大的模型 GEMM/attention 工作批次、更少的模型/采样/MTP proposal 调度和主线程提交。但它不是简单调大 chunk 参数；统计、历史归档、图事务仍必须按原逻辑块执行，不能将这些工作量减半作为收益。历史 `prefill-chunk-native-20261021.md` 的 1K→2K 单请求观测为吞吐 +21.29%，同时改变了全部选页，不是本候选收益预测或资格证据。

最小首版应只在单 owner、普通非末 prefill、同一 active view 内，合并两段连续且完整的 2K 逻辑块。原有 32-token 桥接、1,952/1,984-token 尾块、部分块、Query、检查点和末块继续原路。4K 合格后再扩 8K。不能按全局 `history % 2048` 重新切块：pressure 后相位及原 scheduler 的 MTP headroom/页对齐规则决定原事务边界。

## 保持的是哪一种等价

1. **策略算法等价**：给定相同的每个采样 Q 行页概率及相同接受前缀，按相同逻辑事务、相同次序运行原累计/图算法，应得到相同累计分数、exposure、图边、图 revision、选页及 checkpoint 状态。这部分可以要求固定输入下逐位一致。
2. **attention 数学等价**：同一 Q 行只看其原来可见的全部 active 前缀，每 head 在完整 causal 域归一化，再平均 heads。把若干 Q 行放到同一物理 forward，不应改变数学函数。
3. **真实模型浮点结果不保证逐位一致**：4K/8K 可能改变 NVFP4 GEMM tactic、FA2 tile/split、归约树、GDN chunk/state 数值路径及 MTP prefill 的批次形状。同精度、同模型选项不等于原 Q/K/hidden state 的每一位相同。必须报告分数误差、选页差异、输出轨迹和 MTP 接受差异，不能自动归因于策略 bug，也不能忽略质量影响。

## 逻辑计划与采样

`scheduler.py:_reserve_prefill_lookahead`、`RequestPlan.admit/action` 仍确定原来的逻辑块列表。候选只将已证明可连续执行的一小段列表合成物理 token count，不改变其中的边界。遇到 pressure、Query replay、generation/prefill checkpoint、mandatory metadata 变化、guided 生命周期边界、终止/取消或 active-view generation 变化，必须停止合并。只对最终大 count 调用一次 `admit` 不能证明中间边界均未被跳过。

`PrefillPlanFrame.extend` 已能在 CPU publication 尚未完成时模拟精确 logical records 和 sampled_end，可按每个原 2K 子块顺序调用。每份结果都保存：

- `history_begin/count/local_begin`、原 `rows`、原全局 `source_positions`；
- 该时刻精确的 `starts/counts/history`，包括部分历史页和重排后的 active view；
- 原 `sampled_end`、采样 interval、near-capacity 判定结果及 owner/window generation。

不能对 4K/8K 仅调用一次 `PrefillHistory.rows` 再均分结果。它使用逻辑 end、interval、near_capacity，并从每个块末尾反向按 stride 取样；短块会改变采样相位。首版完整2K、stride32通常是每块64行；合并后的物理 row index 是各逻辑偏移加原 row，密度、层数均不减少。

## 页分数与未来页

一份物理 forward 的后半部分 K 已存在于 cache，并不使前半部分 Q 可以读取它。必须保持原 causal mask，完整前缀归一化不能改为每个2K独立softmax，也不能截断旧历史。FP8 K scale、Q codec、RoPE dense position、head mean 及三层相加/除法顺序沿用原路。

当前 native bank 每真实 owner 只有一套 ranges/piece mapping，所有该 owner 的 sampled rows 共用。最小实现可对完整物理范围建立原生统计布局，输出所有 sampled rows 后，将每个逻辑子块投影到其**当时的精确逻辑 segments**。首版可要求每个子块的 `(starts,counts,history)` 恰为最终布局的前缀，逐项验证后只取对应行/列；不满足则保留原物理2K路径。

不能把最终所有历史列原封不动交给较早事务，即使未来列概率恰为0。`PrefillHistory.accept` 会基于传入 history 建立新的 values 集合并做 decay，未来零页可能提前进入集合；exposure 按每行 causal 可见 segment 去重累加，不能提前增加。图准备的 target 域、预算 scratch 和未来页校验也应与原事务一致。

通用 partial-page 情况中，前事务的历史尾页/新后缀分段可在后事务合并。简单按逻辑页ID求和不一定保留原浮点相加次序。若未来扩展，应使用所有逻辑边界的公共细分，再以原顺序归约成各事务的原 segments；需要单独预算及数值资格。首版前缀一致性限定可避免这部分复杂度。

`native_attention.py:_row_lse` 目前按整行 piece width 做归约。较早行的未来pieces应保持-inf；数学分母不变，但更大 width 的归约树可能改变浮点舍入，因此不能默认原生统计逐位相同。也不能将未来 masked pieces 当成缺失producer触发错误fallback。

参考评分回退应按原逻辑块调用 `page_attention`，传原 Q rows、原 logical positions、原可见 length 和 segments，保留原64行分块和完整 causal 域；不额外评分未来行，不以降采样适配容量。原生正常路径仍只做物理forward的一次QK，子块评分来自已产生的统计。

## 接受与发布：一份物理 DMA，多份原事务

当前 `ActionPacket` 每owner只有一份 `prefill`，需要有界的逻辑子包序列。`CaptureRuntime` 的一个物理 `CapturedPacket` 可以持有2或4份子包及每owner子块游标，但依然只占原固定下载槽。不能将同一个 owner 作为数个独立 owner 交给 `append_authority_batch`，该接口明确禁止重复authority。

下一 scheduler 决策依然先安装 terminal prefix，再授权发布。普通非末 prefill 的 native rejection 为0，不免除取消裁剪。对实际 retained 前缀R，逻辑子块偏移o、长度c的接受数是 `min(c,max(0,R-o))`：完整子块按原 full-chunk rule计分，部分子块只推进原允许的历史/图边界且不提交其 sampled prefill rows，后续0行子块不能计分。不得对整份8K做一次“非全接受则四块都不计分”。

保持原事务顺序的最小方案是在同一个被claim的物理包内逐子块执行：

`原 prepare_accept / budget reserve → 原 raw append/scatter/MeanK → 原 Action/graph commit → 下一逻辑子块`。

这样 decay、每块1/rows权重、exposure、图partial source、revision与历史前缀按原序推进。`GraphUpdate` 当前同owner只允许一份pending reservation，不能一次prepare四份后再直接commit。逐子块事务也避免同时保留四份全图候选扩大预算峰值。后块失败时已经完成的前块是合法已提交前缀；必须记录游标、隔离失败物理包，禁止重放前块或留下半个逻辑事务。不能对物理包突然新增“全部四块原子提交”的更强语义而又声称预算/失败行为未变。

全部子块完成前，该物理包仍不具备 `publication_ready`，原槽、score tensor alias和GPU source租约均保留；pressure/restore/finish/checkpoint仍等待整个包。最小版本可保持原每owner最多一个前驱物理包的pipeline限制，实际token lookahead从2K变4K/8K须纳入预算。对外不暴露中间统计/归档状态给下一次选页；合并范围本来就不能包含需要中间状态的边界。

## MTP、GDN与编译资格

保留实际MTP(k=4)、compiled/PW/full graph、FP8 KV和全部17层归档。`mtp_runtime.py` 的prefix-only capture与 `propose` 后target+MTP合并下载继续生效。安装版autoregressive speculator将target input左移一位，物理块末尾用 `next_prefill_tokens`；原2K块之间的同一next token在合并块中成为普通内部token。该关系数学上相容，但应逐行核验MTP prefix坐标/内容，不能仅比较target16层。

较少的物理步骤会减少中间无输出prefill的sampling/proposal调用。逻辑统计不计draft/rejected rows，所有真正输出的MTP接受过滤保持。显式seed的sampler读取seed和position，仍需核验终点draft状态、GDN/recurrent state、最终next-token logits及真实接受长度；无显式seed的全局随机数路径不能未经资格就宣称调用次数变化无影响。

物理4K/8K要重新核验target/MTP shape dispatch、capture gates、FP8 scale/cache读取方式、RoPE位置及GDN状态缓存。不得关闭MTP或换精度来取得吞吐改善。正常编译/graph预热单列冷准备成本；临时tensor地址/缓存specialization不能无限增长。

## 现有容量与还需检查的预算

当前单服务上限 `max_num_tokens=16424,max_num_reqs=8`，native bank已有 `max_rows=256,context=65576,max_segments=2202,width=6301`，3采样层、24Qheads、4KVheads、dim256。

| 物理新token | stride32采样行 | 完整17层FP8 rawK+V payload（不含小header） | 最坏2202列层平均score矩阵 | download pool按2次幂槽 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 64 | 68 MiB | 563712 B | 1 MiB |
| 4096 | 128 | 136 MiB | 1127424 B | 2 MiB |
| 8192 | 256 | 272 MiB | 2254848 B | 4 MiB |

- **256行是整batch总和，不是每owner。** 单owner8K已用满；两个8K会触发 `sample_row_capacity`。首版单owner避免多并发隐性fallback；之后必须按实际各逻辑块rows之和控制物理合并，而非降低密度或盲目扩大bank。
- native预分配LSE为 `3×256×24×6301×4=464560128 B`，mass为6764544 B。若maxbatch/context/rows都不变，单owner4K/8K不必增大这两项启动容量。行清零、归约和D2H实际工作增加，需要全组件计入。
- Action metadata实际GPU→CPU坐标三行int64是`24×Q`，2K/4K/8K分别48/96/192 KiB。现固定pool64MiB、128slots需要按两物理包在途、子包aliases和其他owner总和验证。矩阵切片必须共享原lease，不用额外逐子块host拷贝掩盖成本。
- 参考回退 `PackedActionMetadata` 总8MiB（host/device各半），单个64行逻辑块最坏payload为`8×2202+16×64=18640 B`，host+device共37280 B；必须维持consumer event生命周期和max_leases=128，不在失败时无界分配。
- raw DMA已有两固定maxbatch槽，4K/8K在其单槽容量内，但实际GPU pack输出/临时activation及同owner在途token峰值增大。不能只看启动预分配显存。每8K有256个32-token逻辑record；两包及partial tail、seal在途记录、archive admission/GC索引和100GB历史预算都要同时核验。
- graph每次仍最多原64sample逻辑事务；保留原reservation顺序与上限，不一次预留4份全图。native `ranges`/row映射仍按真实owner，不能把4个逻辑事务伪装成4个owner避开ABI限制。
- `serve.py` 的64sample guard当前约束物理chunk。新配置必须显式区分逻辑统计chunk=2048和物理merge上限，并验证每逻辑块≤64及整batch≤256；直接放宽guard不能满足上述契约。

## 函数触点和建议顺序

1. scheduler/`RequestPlan`：有界逻辑计划列表与物理合并资格，所有既有边界仍原位；`serve.py/options`区分两个参数并保持默认关闭。
2. `prefill_planning.py` / `ActionRuntime._prepare_prefill_pipeline`：顺序extend，保存每子块精确采样与分段，物理row映射。
3. `native_attention.py` / `ActionRuntime._capture`：一次native QK、多逻辑score视图与原层平均顺序；先用前缀布局资格，保留逐逻辑参考fallback与原验证。
4. `ActionPacket` / `CaptureRuntime` / authority：一份DMA包的逻辑游标与逐块原事务，terminal裁剪、失败前缀、预算和租约。
5. 诊断/报告：区分physical forwards与logical score/graph/archive commits，保留原logical history与checkpoint计数，实际fallback/峰值不可省略。

先做固定矩阵的CPU逐位等价测试（decay/exposure、future columns、partial acceptance、图partial、预算失败第2/4块、恢复/多owner），再做固定Q/K/FP8/causal的GPU分数测试和MTP prefix/GDN state资格。最后运行相同完整256K任务、相同逻辑选页策略的2K/4K物理A/B；不仅比较模型kernel，还计入score投影、逻辑提交、DMA/归档、重选、临时显存与冷编译。4K通过并有完整成本收益后再试8K。

正常情况下，逻辑统计事务数和归档字节不会减少；主要节约来自模型批次效率和物理dispatch次数。若完整吞吐无收益，保持关闭。若真实页分数/选页变化，需要把可证明的策略等价与真实模型数值轨迹变化分别报告，再做质量资格；不能以数学等价替代真实验收。
