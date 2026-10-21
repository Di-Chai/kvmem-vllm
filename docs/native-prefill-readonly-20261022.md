# 静态统计 metadata 只读加载试验

本候选在编译检查后停止，没有进入 GPU 正确性或性能测量，也没有接入服务配置。`__ldg` 改变了读取的缓存属性，但没有消除目标循环中的重复读取，未达到预先约定的继续条件。

## 范围和资格

仅在组件构建参数 `static_readonly=True` 下生成独立源码 overlay，将静态对齐类的一处 `stats.rows[p.q_indptr[batch] + q]` 改为两次 `__ldg`，并使用独立类名。原浮点计算、归约顺序、采样行、归一化域、写地址和已有静态 header 均未修改。没有新增 buffer、`restrict` 或循环外预计算。

组件入口在调用前以 CPU 张量元数据核验 `q_indptr`、采样行表的连续布局，以及它们与 workspace、attention 输出、row LSE 和统计 LSE 的地址范围不重叠。首版拒绝 CUDA Graph、混合 decode 和滑动窗口；同一 producer stream 上的 frame 在 kernel 完成前不变仍是调用方的生命周期要求，地址检查不能防止其他线程擅自写入这些张量。

相关 CPU 协议、边界与构建测试共 **68 项通过**。这只验证源码变换、范围资格和证据验收逻辑，不代表新内核已经通过数值资格。

## 编译结果

旧 static overlay 保持 `d7b326c5139bf4d9`，候选为 `45fa6f8562de521f`。两份库及源码指纹在 `prepared.json`，冻结文件复核未变化。

同一 CTA64 的 SASS 结果：

| 项目 | 原 static | readonly 候选 |
|---|---:|---:|
| 寄存器数 | 255 | 255 |
| Stack bytes | 32 | 32 |
| 静态 shared bytes | 1,024 | 1,024 |
| 主 KV 循环 metadata loads | 4 | 4 |
| 主 KV 循环 local loads | 3 | 3 |
| 整核 global-load 指令数 | 35 | 35 |

两组 metadata 读取仍留在主 KV 循环中。原位置为 `0x7860/0x78d0`、`0x8700/0x8770`；候选位置为 `0x7860/0x78e0`、`0x8710/0x87a0`。候选仅把这四条 `LDG.E` 改为 `LDG.E.CONSTANT`，没有减少次数。循环回边分别为 `0xcf70` 和 `0xcfa0`，共同回到 `0x2470`。

预设继续条件要求 metadata load 次数下降且寄存器、stack、循环 local loads 不增加。当前未满足第一项，因此停止。**没有测得候选更快或更慢，不能将此结果表述为性能回退或数值失败。**

首次准备因进程 PATH 未含虚拟环境的 `ninja` 而失败，保留 `compile.log`；补全工具路径后同一冻结源码编译成功，记录在 `compile-retry.log`。未删除或覆盖原失败证据。

证据目录：`artifacts/native-prefill-readonly-20261022-r1/`，包括 `manifest.json`、`prepared.json`、`sass-verdict.json`、`compiled/{reference,candidate}.{resources.txt,cta64.sass}` 和 `frozen/`。可复算 JSON 记录每条目标指令、库和 SASS 的 SHA，以及 `performance_run=false`。

## 后续硬件计数的限制

本机安装 Nsight Compute 2025.3.1，但只读检查显示驱动 `RmProfilingAdminOnly=1`，当前进程的 permitted/effective/bounding capabilities 均为零。因此本轮没有采集 DRAM/L1 sectors、long-scoreboard 等硬件计数，也没有使用 sudo、改变驱动权限或尝试启动 Ncu GPU 采集。安装文档另列出部分不需硬件计数权限的指令级源码指标；本轮同样未采集，不能用静态 SASS 指令数替代实际动态次数或 stall 归因。

现有三臂组件已测得的 14K/48K 三层 static FA2 额外 30.558 ms，仍不能进一步可靠拆成写出流量、spill 和指令开销。本限制不影响已有完整模型和组件结果，也不阻止其他独立优化继续验证。
