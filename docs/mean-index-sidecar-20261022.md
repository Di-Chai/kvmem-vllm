# GPU 有序页均值：独立原型资格与完整组件成本

状态：实验脚本，不由服务路径导入。没有生产开关，没有计入完整模型吞吐收益。

核验读取的是实际 capture pack 后 FP8 raw K 字节，非量化前 K 或带 RoPE 的
attention K cache。每个通道按 token0..31 顺序累加FP32，禁用融合；原CPU除法
与FP16转换保留。只处理命名证明过的16层target前缀，17层archive中的MTP排除。
非有限值触发整批原CPU finalizer，保持NaN符号/载荷等原行为。

资格 `artifacts/mean-index-sidecar-20261022-r2/`：12项GPU检查通过6.12s，覆盖
FP8码值/非有限值回退、目标层排除、真实32/1984/2048/14336行捕获、真实异步
direct seal、旧flag/slot复用、忙槽拒绝和event失败隔离；和原CPU sum/mean逐位一致。
CPU协议等相关34项已通过。r1的8通过/4失败保留：失败为测试夹具没有像服务factory
一样设置 `archive.async_seal=True`，数学比较通过；r2修复后整组重跑通过。

## 测量

固定一个GPU/pinned槽，4种实际形状，每种2轮预热、8轮CPU/GPU交错比较。
`index_only` 包含GPU核、分数D2H、CPU结果拷贝、原除法/FP16及ticket释放；
完整组件包括公共raw/V下载、按原逻辑2K append、真实异步归档/hash8/publication、
owner/archive关闭和真实retirement。输入准备与正确性clone不计入测量。

以下为全部8次样本的**算术均值**，未剔除离群值：

| 输入行 | CPU index | GPU index | CPU完整生命周期 | GPU完整生命周期 | 生命周期变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.246ms | 0.156ms | 3.301ms | 3.312ms | +0.32% |
| 1984 | 16.297ms | 3.184ms | 86.785ms | 62.721ms | -27.73% |
| 2048 | 16.677ms | 3.330ms | 56.248ms | 51.882ms | -7.76% |
| 14336 | 110.755ms | 12.718ms | 285.061ms | 273.535ms | -4.04% |

index四形状均8/8对更快，完整生命周期分别4/8、8/8、4/8、7/8对更快。
完整2048行CPU/GPU中位数为51.162/63.117ms，方向与均值不同；14K最后一对
为290.274/375.642ms，未丢弃。归档分配、首次触页和线程时序有显著波动，
不能只选中位数或单独index的80%～89%下降作为完整收益。

组件固定容量16,424行，单槽GPU和pinned各33,652,800B，总67,305,600B。
假设服务采用两槽需134,611,200B，但本原型没有建立生产lease pool，也没有
实际为模型配置这份空间。没有与模型同时占用GPU，无法测出模型竞争的影响。
本轮CPU集中分配候选没有和本原型组合，不能将各自收益相加。

`qualification-result.json`和`component-result.json`均exit0、冻结文件未变化、
退出后GPU空闲。所有记录digest和sum/mean逐位核验通过。完整结果包括原始
每对样本及旧实际调用计数，在`component.json`。

## 尚未完成的服务接入

现原型采用一个显式consumer，组件中临时替换finalizer，不能用于并发服务。
接入需要：显式owner/packet/窗口generation绑定；MTP接受前缀和部分历史页
过滤/回退；DMA、CPU消费和所有引用结束后的槽归还；有界预算与异常持久隔离；
取消/恢复/结束路径清理；原提交点前一次性消费。

可进一步评估将sum/非有限flag附在现有bulk DMA payload末尾，复用已合格的
slot/event/packet生命周期，避免另建并行缓冲池。这是下一版设计方向，尚未
实现，不能据此提前声称已减少真实模型的CPU等待或已保持完整CUDA Graph生命周期。
