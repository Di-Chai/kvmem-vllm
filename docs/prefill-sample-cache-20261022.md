# 统计采样行共享缓存：独立组件试验

状态：组件试验，服务没有开关请求该模块，原默认/静态统计源码与overlay均未改变。
本次没有启动新的完整模型吞吐测试。

前一轮 `__ldg` 只读加载仍把metadata读取留在KV循环内。本候选在原paged
FA2 kernel的PDL wait之后、VO/KV循环之前，用整个CTA预读采样编号到
`int32_t[CTA_TILE_Q]`共享数组，随后CaptureSoftmax读取该数组。Q/head映射、
causal mask、softmax更新、FP32加法顺序、log、piece-LSE写出和归约全部保留。
源码逆向还原测试确认除整数读取和参数传递外整个variant文本不变。

新overlay由旧静态overlay和两段源码转换共同寻址，旧`d7b326c5139bf4d9`
保持不变；候选`580559a87a9bd775`。没有修改已安装FlashInfer或手动清理缓存。
候选直接调用独立生成的`KVMemStaticCachedAttention`。沿用组件专用的host
metadata guard：无graph、无mixed decode、无滑动窗口，Q indptr/采样表为
连续整数CUDA张量且不与输出/workspace/statistics写域重叠；同一stream上的
metadata生命周期仍由原frame保证。没有使用未经证明的restrict。

## 编译与资格

5项CPU源码协议检查通过。冻结编译及组件均exit0，manifest内源码未变化，
每阶段退出后GPU无计算进程。原库SHA为
`6b212e768fc58ba402dda17d7ab17af1ac346dc028d7544b2adccf2858e151d7`，候选为
`9fcb92d5dd467843d59efbed94e95c92aa3b4c0ea8dd2c7595c7da207f73bad1`。

已用cuobjdump保存两库的完整resource usage。与实际服务大块相关的causal
CTA64/NUM_MMA_KV=4模板：寄存器仍255，静态共享内存1024→1280B，栈32→40B；
KV=2模板寄存器254→252、栈均0；KV=8模板栈448→456B。额外共享内存和spill
可能抵消收益，因此不将“源码减少global读取”当作必然提速，直接测完整组件。
没有新硬件counter采集，不用静态计数冒充实际DRAM流量。

7种形状全部通过原三臂资格：逐层attention输出、native row LSE、完整容量
piece LSE（包括未用区域sentinel）以及最终页分数均逐字节相同。24Q heads、
4KV heads、dim256、BF16 Q、FP8 K/V、causal、非连续物理页、K scale0.8、
原512-row bank和完整归一化域不变。独立profiler确认每个形状实际运行3个
候选FA2 kernel；没有generic回退。只证明这些组件形状，不扩展为已验证所有
GQA/partial历史/Graph/多owner服务生命周期。

## 完整组件性能

每个形状2轮预热、12对AB/BA，计入原metadata arm、3层attention、归一化、
层平均、pinned分数D2H和完成事件。正确性clone及独立profiling不在计时内。
两臂的分数缓冲容量相同。以下为全部样本wall均值，每次执行3个采样层合计：
三个采样层共享fixture的Q/K/V，没有真实模型层间工作，缓存局部性与服务不同。

| 新Q行 | 可见KV（含Q） | 原静态统计 | 共享缓存 | 耗时变化 | 更快配对 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 16384 | 16.529ms | 16.272ms | -1.56% | 12/12 |
| 2048 | 32768 | 33.247ms | 32.674ms | -1.72% | 12/12 |
| 2048 | 49152 | 49.767ms | 48.952ms | -1.64% | 11/12 |
| 8192 | 32768 | 97.543ms | 95.912ms | -1.67% | 11/12 |
| 8192 | 49152 | 151.101ms | 148.290ms | -1.86% | 12/12 |
| 14336 | 32768 | 149.085ms | 146.527ms | -1.72% | 12/12 |
| 14336 | 49152 | 241.322ms | 237.000ms | -1.79% | 12/12 |

这个1.6%～1.9%是三个采样层的attention＋统计组件节省，不是整模型吞吐。
14K/48K节省约4.32ms，仍有LSE写出、归约、下载等成本。没有将它与旧native
floor的绝对时间跨轮相减，没有与归档或GPU页均值的百分比相加。

证据：`artifacts/prefill-sample-cache-20261022-r1/`的`prepared.json`、
`{reference,candidate}.resources.txt`、`component.json`、编译/运行命令与日志、
两阶段result/cleanup和frozen manifest。模块保持在实验脚本中，未接入服务。
