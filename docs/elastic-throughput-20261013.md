# 弹性会话性能：2026-10-13

补充的8用户、每路32K输入＋16K输出实测见
[长输出吞吐](elastic-long-output-throughput-20261013.md)：共同decode约592token/s，
包含冷预填充的整轮HTTP约505token/s，8路全部成功且无抢占。

完整上下文的工具边界暂停／CPU恢复已实现，并加入全局 active 字节预算、用户
token 历史配额、TTL、截止时间 admission、恢复字节／并发限流，以及显式有限
active context。模型计算保持原生 NVFP4＋FP8＋MTP k3＋CUDA Graph。
入口和使用方法见 [elastic serving](elastic-serving.md)。

## 固定设置与统计范围

RTX PRO 6000 Blackwell Server 96GB，vLLM0.30.0／Torch2.13.0+cu130／
FlashInfer0.6.18.post1；同一份已校验 Qwen3.8-27B NVFP4 checkpoint。
目标与MTP注意力 KV均为FP8 e4m3，GDN state采用原生精度。TP1、batch8192、
max_num_seqs32、GPU utilization0.85、async scheduling、FULL_AND_PIECEWISE。
原生 prefix caching 和 aligned GDN checkpoint 用于 offload，connector 要求
LBHNC KV 布局。没有启用旧 eager capture／DFlash2／MeanK 重建路径。

网关 active 预算40GiB，CPU KV缓存48GiB，恢复预算8GiB／最多4路；完整上下文
最多16个 active 回合，有限上下文最多32个。每用户同时一回合，队列最大128，
admission deadline120秒。初始历史32768个冻结token，各性能回合输出512token、
temperature0／seed104729／ignore_eos。CPU恢复前明确清空GPU prefix cache，
保留CPU cache，并核对 CPU→GPU 原生加载字节，避免把GPU缓存命中当作CPU恢复。

以下HTTP吞吐包括排队、预填充／恢复、生成和JSON响应。TTFT为网关排队加内部
原生SSE首个已接受token的时间，不是非流式调用方看到响应的时间。单路完整
上下文重复3次，其余档各一次；不将样本视为普遍最优配置或持续SLO证明。

## 完整上下文结果

恢复时实际输入为初始32768token加512已生成token和2个新增token。

| 客户端 | 冷处理HTTP token/s | CPU恢复HTTP token/s | 冷TTFT p95秒 | 恢复TTFT p95秒 |
| --- | ---: | ---: | ---: | ---: |
| 1，三次中位数 | 58.62 | 110.22 | 3.31 | 0.46 |
| 8 | 120.70 | 405.83 | 26.94 | 3.41 |
| 32 | 131.38 | 546.28 | 110.02 | 20.87 |

32客户端两轮各生成16384token，冷处理wall124.71秒，恢复wall29.99秒，整轮
吞吐约4.16倍。32客户端不代表网关同时运行32回合，本配置的active峰值为16。
CPU实际加载合计38,889,586,688字节，与32份可复用历史一致。等待阶段原生
running／waiting／请求KV使用率归零；已预分配GPU显存仍由引擎保留。

在较短的自然停止检索回合，四份约32K文档的模型TTFT约3.21–3.23秒，恢复后
约0.33秒。每份都准确返回旧文档随机值和新工具结果，正常stop，未返回其他
客户端随机值。另一次128-output恢复与冷重算比较的全部token IDs完全一致。
这些是窄范围检查，不代表通用任务质量或所有并发情形的数值相等。

初版16GiB CPU缓存可以承载8份历史，但32份历史造成LRU淘汰／重新prefill，
一条恢复请求触及120秒排队上限；客户端gather关闭其余在途调用。该不完整
32客户端结果保留并排除。48GiB缓存及恢复限流后，32客户端全部完成。

## 有限 active context 结果

CPU初始逻辑历史仍为32768token；调用方显式选出7168个token，恢复时保留
同一选择prefix并追加已生成token／新输入，模型实际读取7682个token。选择
限额8192；未选择部分不参与这一轮模型计算。收益包括输入变短，不能解释为
完整32K注意力计算的等价加速。

| 客户端 | 初次HTTP token/s | CPU恢复HTTP token/s | 初次TTFT p95秒 | 恢复TTFT p95秒 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.57 | 107.83 | 0.58 | 0.28 |
| 8 | 398.33 | 583.07 | 4.40 | 2.11 |
| 32 | 601.52 | 986.01 | 17.42 | 9.05 |

32客户端初次／恢复wall27.24／16.62秒，原生最高运行数32；恢复实际加载
10,368,319,488字节。采样active reservation最高35,861,299,200字节，低于40GiB；
恢复峰值4路，符合配置。四份约30.9K文档另做显式选择后的正常停止检索，
旧随机值所在片段被明确包含，4／4精确返回旧值和新工具结果。此夹具没有检验
自动选择是否能找到任意历史信息，也没有证明被省略内容对其他任务无影响。

## 归档开启后的纯 decode

额外绕过网关admission，直接测同一个原生后端的计算路径。每请求1024输出token，
裁掉各路前后64token的共同稳态；每请求独立随机cache_salt，排除prefix共享和
旧历史命中。原生running／waiting和无新增prefill计数佐证，窗口覆盖96.1%–98.2%。

| 模型输入token | 并发 | 合计稳态decode token/s | 整轮HTTP token/s |
| --- | ---: | ---: | ---: |
| 152 | 8 | 834.01 | 812.48 |
| 152 | 32 | 2376.28 | 2205.40 |
| 32768 | 8 | 604.79 | 202.62 |
| 32768 | 32 | 1171.41 | 240.64 |

四档均确认完整目标并发窗口、无抢占。此前未启用KVMem的原生k3全FP8同档
decode为940.56／2255.27／629.28／1178.63。长C32单次观察差约−0.6%，计算快路径
保留；短C8本轮下降约11.3%，短C32则提高约5.4%。布局、APC／GDN状态处理与
归档一起改变，没有拆分归因或做重复短文对照，不宣称所有负载都无开销。

## 生命周期、安装包与证据

真实模型检查通过：同会话并发409；另一个同用户回合超过排队deadline返回429；
取消长请求后active／restore／queue及原生请求页回到空闲；取消的版本零历史未
提交，随后64-output新请求成功。此检查不是持续故障注入或硬SLO验收。

主性能数据共248个成功请求、167936个准确计数输出token：完整上下文86请求，
有限上下文82请求，原生hot decode80请求。正常停止的工具续接为8个案例、各两回合。
host／HTTP回归101测试通过。wheel构建、安装、64份Python源码逐文件hash核验通过；
最终安装入口的两种模式还独立验证32K历史、CPU恢复、版本冲突与空闲回收。

所有证据位于 `artifacts/elastic-serving-20261013/`：`summary.json`核验输出摘要
及统计来源；`optimized-qualification/`、`bounded-qualification-v2/`和`hot-*/`
保存原始请求、响应、指标；`lifecycle-results.json`保存回收检查；
`package-manifest.json`／`installed-verification.json`／`final-installed-smoke.json`
保存安装与最终服务验证。失败初始化、错误夹具及16GiB缓存的未完成批次均保留。

工具等待由回合之间模拟，工具结果由测试输入提供，未执行完整Agent任务。
CPU KV为原生有界LRU缓存；CPU完整token账本受独立额度和TTL控制。NVMe历史层、
跨worker／重启续接、自动MeanK选择、真实工具资源竞争下的持续p99与硬SLO未验收。
