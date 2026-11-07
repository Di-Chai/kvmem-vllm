"""Build the Chinese experiment report from saved measurements."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

from analyze_gpu import spearman
from protocol import quantile


def read(path):return json.loads(path.read_text())
def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def fmt(value,digits=2):return f'{float(value):.{digits}f}' if value not in (None,'') else '—'
def md_table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def run(args):
    root=args.root;analysis=root/'analysis';data=args.dataset
    summary=read(analysis/'summary.json');candidates=csv_rows(analysis/'candidates.csv')
    groups=read(analysis/'groups.json');census=read(data/'summary.json')
    by=defaultdict(Counter)
    for line in (data/'census.jsonl').read_text().splitlines():
        row=json.loads(line)
        for key in ('eligible_units_80chars','earlier_byte_duplicate','earlier_whitespace_duplicate'):
            by[row['content_type']][key]+=row.get(key,0)
    type_names={'task_instruction':'用户任务约束','tool_call':'工具调用','tool_result':'工具返回','thinking':'Thinking','assistant_text':'其他助手正文'}
    corpus_table=md_table(['信息类型','合格单元数','字节级重复数','字节级重复比例','归一化重复比例'],[
        [type_names[k],v['eligible_units_80chars'],v['earlier_byte_duplicate'],
         fmt(v['earlier_byte_duplicate']/v['eligible_units_80chars']*100)+'%',
         fmt(v['earlier_whitespace_duplicate']/v['eligible_units_80chars']*100)+'%'] for k,v in by.items()])
    category_names={'exact_duplicate':'完全重复','partial_overlap':'词法部分重叠','error_observation':'真实错误记录',
        'code_observation':'代码观察','thinking_without_replacement':'无替代证据的 Thinking','user_constraint_control':'用户约束'}
    group_table=md_table(['候选类型','数量','Attention 平均 Δ','Transformer 平均 Δ','最终层平均 Δ'],[
        [category_names[g['category']],g['n'],fmt(100*g['attention_mean'])+'%',fmt(100*g['transformer_mean'])+'%',fmt(100*g['final_layer_mean'])+'%']
        for g in groups])
    cost=read(root/'judge-r1/graph-observer/cpu-costs.json')
    observer=read(root/'judge-r1/graph-observer/summary.json')
    coverage=Counter(r['result']['body']['answers']['coverage']['choice'] for r in observer if r['result']['success'])
    edges=Counter();unit_count=0
    for p in (data/'snapshots').glob('*.json'):
        s=read(p);unit_count+=len(s['units']);edges.update(e['relation'] for e in s['edges'])
    recovery=read(root/'recovery-r1/results.json');prefill=read(root/'paired-r1/recovery/prefill.json')
    recovery_table=md_table(['历史 tokens','注意力 KV','普通 CPU P50/P95 ms','锁页 CPU P50/P95 ms','Ceph 暖读 P50/P95 ms','Ceph 客户端冷读建议 P50/P95 ms','完整前缀重算 P50/P95 s'],[
        [r['tokens'],str(r['bytes']//1024**2)+' MiB',
         *[fmt(r['summaries'][m]['p50_ms'])+'/'+fmt(r['summaries'][m]['p95_ms']) for m in ('pageable_cpu','pinned_cpu','file_warm','file_client_cache_advised_cold')],
         fmt(quantile([v['prefill_plus_first_token_seconds'] for v in prefill if v['tokens']==r['tokens']],.5),3)+'/'+
         fmt(quantile([v['prefill_plus_first_token_seconds'] for v in prefill if v['tokens']==r['tokens']],.95),3)] for r in recovery])
    raw=read(root/'analysis-judge-r1/summary.json')
    judge_table=md_table(['判断版本','一致','遗漏','矛盾','不确定','解释'],[
        ['初版',*[raw['choices'][k] for k in ('consistent','omission','contradiction','uncertain')],'引用差异导致事实保留误报'],
        ['引用归一化修正版',*[summary['choices'][k] for k in ('consistent','omission','contradiction','uncertain')],'跨字段对照失败，分数仍未验证']])
    correlation_table=md_table(['计算变化量','与平均 1−P(一致) 的 Spearman ρ'],[
        [name,fmt(spearman([float(r[key]) for r in candidates],[float(r['inconsistency_probability']) for r in candidates]),3)]
        for key,name in [('attention_mean','16 个 Attention 层平均'),('transformer_mean','64 个 Transformer 层平均'),('final_layer_mean','最终 Transformer 层')]])
    quad=summary['quadrants']
    quadrant_table=md_table(['描述性分组','六项均判一致','至少一项未判一致'],[
        ['低于样本中位数',len(quad.get('lower_delta__all_consistent',[])),len(quad.get('lower_delta__not_all_consistent',[]))],
        ['达到或高于样本中位数',len(quad.get('higher_delta__all_consistent',[])),len(quad.get('higher_delta__not_all_consistent',[]))]])
    serving=read(root/'serving-r1/summary.json') if (root/'serving-r1/summary.json').exists() else []
    native=read(root/'serving-r1/native-systemone/summary.json') if (root/'serving-r1/native-systemone/summary.json').exists() else []
    finalized=(root/'run-completion.json').exists() and read(root/'run-completion.json').get('completed')
    state='本轮 GPU/离线测量和原始记录审计已完成' if finalized else '配对、块数、恢复和语义诊断完成；服务矩阵仍在执行，本文为当前快照'
    serving_table=md_table(['输入','请求类型','并发','完成/提交','TTFT P50/P95 s','排队 P95 s','总输出 tokens/s'],[
        [str(r['input_length']//1024)+'K',r['kind'],r['concurrency'],f"{r['completed']}/{r['submitted']}",
         fmt(r['ttft_p50_seconds'])+'/'+fmt(r['ttft_p95_seconds']),fmt(r['queue_p95_ms']/1000 if r['queue_p95_ms'] is not None else None),
         fmt(r['aggregate_output_tokens_per_wall_second'])]
        for r in serving if r['output_length']==512 and ((r['kind']=='observer' and r['concurrency'] in (1,4,16)) or (r['kind']=='pair' and r['concurrency']==16))])
    native_table=md_table(['输入','并发','完成/提交','HTTP P50/P95 s','总标签/s'],[
        [str(r['target_input']//1024)+'K',r['concurrency'],f"{r['completed']}/{r['submitted']}",
         fmt(r['http_latency_p50_seconds'])+'/'+fmt(r['http_latency_p95_seconds']),fmt(r['labels_per_wall_second'],3)]
        for r in native if r['concurrency'] in (1,4,16)]) if native else '原生分类接口矩阵尚未完成。'
    worst_context=next((r for r in serving if r['input_length']==32768 and r['output_length']==512 and
                        r['concurrency']==16 and r['kind']=='pair'),None)
    capacity_note=(f"在本次配置下，32K 输入、512-token 输出、16 对并发的整波耗时约 {fmt(worst_context['wall_seconds'],0)} 秒，"
        f"总输出吞吐约 {fmt(worst_context['aggregate_output_tokens_per_wall_second'])} tokens/s；"
        f"首 token P95 约 {fmt(worst_context['ttft_p95_seconds'],0)} 秒，其中排队 P95 约 {fmt(worst_context['queue_p95_ms']/1000,0)} 秒。"
        "这组完整上下文基线显示，单卡同时接收 32 个长上下文请求会产生很高的排队成本。"
        if worst_context and worst_context['completed']==worst_context['submitted'] else '')
    semantic_p50=quantile([float(r['judge_seconds']) for r in candidates],.5)
    semantic_p95=quantile([float(r['judge_seconds']) for r in candidates],.95)
    generated=datetime.now().astimezone().isoformat(timespec='seconds')
    report=f'''# DuoKV 4090 初步验证报告

实验日期：2026 年 11 月 5 日。状态：**{state}**。记录生成时间：{generated}。

本次验证确认了历史重复信息的存在，打通了真实历史注意力 KV 的配对排除、逐层测量及恢复成本测量。**当前 System One 语义判断器未通过全部对照，不能用其标签批准删除。** Graph 是否比简单规则筛选得更准，以及联合指标能否降低 False-safe，仍需独立人工标注。

| 假设 | 本次证据 | 尚不能得出的结论 |
| --- | --- | --- |
| H1 候选识别 | 存在可复核的重复信息，Graph、映射和排序已实现 | Graph 优于简单规则、语义替代 Precision/Recall |
| H2 删除影响估计 | 真实注意力 KV 排除与逐层测量可重复；语义判断器暴露对照失败 | 六探针足以检出重要损失、联合规则降低 False-safe |
| H3 系统可行性 | CPU/Ceph 恢复组件与全上下文服务成本可测 | 本地 SSD 恢复、混合状态完整恢复、满足生产延迟目标 |

按[六项实验清单](../graph-cleanup-4090-20261105-r4/requested-experiments.txt)对 [Proposal](https://app.notion.com/p/P3-Graph-guided-Output-aware-KV-Cleanup-System-One-40fba38265ed81839659f5b61ca7766b)执行离线验证；没有重新运行 Agent，也没有改变线上 KV 删除策略。下述源数据统计来自 64 条 DeepSWE Trace；GPU 配对实验覆盖 24 个候选、15 个任务、16 个重放上下文、23 个不同文本簇。

## 实验环境与配对范围

使用 NVIDIA RTX 4090 24 GB、vLLM 0.32.0，模型目录为 `Qwen3.8-27B-GPTQ-Int4`，实际配置是 Qwen3.5 系混合架构：48 个循环层、16 个全注意力层，共 64 层。权重为 GPTQ Int4，计算使用 BF16，注意力 KV 使用 FP8 E4M3。版本证据见[环境记录](paired-r1/environment.json)和[模型与源代码指纹](model-source-metadata.json)。

20 个候选使用早期完整 Step 前缀，4 个错误候选使用明确标记的后期窗口并保留原任务约束。重放输入为 12,299–26,118 tokens；原 Agent 的框架消息和工具 schema 未复现，因此这些是研究用重放上下文。所有选择证据都限定在冻结检查点之前。

GPU 候选仅从 seed 1001 选取，其余三个 seed 进入语料统计。完全重复组的四个候选来自两个任务；共享文本或上下文的候选不视为独立任务样本。

Full 与 Reduced 的输入 token IDs、Query 绝对位置和删除前历史 KV 完全一致。Reduced 只在本次影子计算的注意力页表中排除选定的历史页，原位置编码和其余历史 KV 保持不变。**原完整前缀的循环状态仍然保留，也可能携带候选信息。** 该实验不是对历史文本删改后重新 Prefill 的替代实验。

## Experiment 1 历史重复与替代候选

排除 5,349 个框架 reminder 单元后，64 条 Trace 共包含 211,582 个合格结构单元。其中归一化后至少 80 字符的 150,138 个单元作为重复率分母。同一 Trace 内、相对更早信息的字节级重复为 13,756 个，即 **9.16%**；空白归一化重复为 15,128 个，即 **10.08%**。

{corpus_table}

这些是单元计数比例，不能换算成可回收 token 比例或安全释放的显存。重复单元绝对数量以 Thinking 最多；重复比例以工具调用最高。原始统计见[逐 Trace 分类型统计](../graph-cleanup-4090-20261105-r4/census.jsonl)。

在选定的 24 对早晚信息中，原生 Observer 给出 full={coverage['full']}、partial={coverage['partial']}、none={coverage['none']}。其中 4 个 exact pair 均被判 full；7 个词法重叠 pair 中 full=1、partial=4、none=2。这是模型辅助覆盖判断，说明词法重叠不能直接当作完全替代；**它不是整个语料的 Supersession Ratio，也未经人工 Precision 验证**。见[关系判断记录](judge-r1/graph-observer/summary.json)。

## Experiment 2 Graph 与映射

29 个快照包含 {unit_count:,} 个单元出现、{sum(edges.values()):,} 条边：精确重复 {edges['redundant_with']}、词法重叠 {edges['lexical_overlap']}、实体关联 {edges['entity_association']}、实际工具 ID 关联 {edges['tool_result_of']}。同一单元可出现在多个快照中；实体关联只是依赖代理，未被标为真实语义依赖。

在服务器上，对每个快照重复构图 5 次，构图 P50/P95 为 **{fmt(cost['build_p50_ms'])}/{fmt(cost['build_p95_ms'])} ms**，排序 P50 为 **{fmt(cost['ranking_p50_ms'])} ms**。图规模为 62–315 个单元；当前新 Step 更新采用全图重建。见[服务器构图计时](judge-r1/graph-observer/cpu-costs.json)。

24 个候选均完成原文本字符位置、真实 tokenizer offset、逻辑内核页和 16 层实际 slot mapping 的对应检查。选定范围为 1–17 个 32-token 内核页，即 32–544 tokens；没有触及其他信息单元的正文，但向外取整会包含部分标签或边界 token。模型分配器的注意力块实际按 1,568 tokens 对齐，**32-token 内核页不等于可独立释放的分配器块**。

已导出 LRU 代理、冗余排序、Graph 规则、Graph 传播及真实 Query-to-KV 注意力质量排序。注意力来自固定探针的最多 8 个 Query 行、24 个 heads，跨 16 个注意力层平均；它不代表原 Agent 的全部历史访问。见[排序与 Top10 重合](analysis/ranking.csv)和[映射审计](analysis/audit.json)。没有独立标签，因此 Candidate Precision/Recall 和 Edge Precision/Recall 保持未测；尚不能证明 Graph 的额外筛选收益。

## Experiment 3 逐层输出变化

每个候选固定相同的 148 个 Query tokens。Attention Output 取 gate 和输出投影之前的值，每 token 拼接 24 个 heads 后计算相对 L2；Transformer Output 取延迟残差与 hidden 的 float32 和。两类观测点分别统计，不混为一个数。每层保存逐 token 的相对变化、Full 范数和差值范数，并导出 Mean/P95/Max。

零删除的自定义路径与原生路径差异为 **0**，重复零删除差异为 **0**；c01、c09、c21 的重复 Reduced 测量差异为 **0**。8 次重复出现的 Full 输入也得到完全相同的生成 token IDs。48 个正式答案均为完整结构化 JSON，没有截断。所有候选的 16 层历史 KV 哈希配对一致。

{group_table}

最终层的候选平均变化范围为 **2.66%–6.22%**。前 3 个循环层位于第一个注意力排除点之前，变化为 0。图中值是各 token 相对 L2 的平均，不能解释成任务错误率。

![逐层输出变化](analysis/figures/layer-output-change.png)

对 c08、c15、c16 进一步使用嵌套页集合，分别排除 1、2、4、8 和全部页；18 个前向均完成，全部页结果与初始实验逐数一致。三条曲线均不是严格单调，删除数量不能直接替代影响或安全判断。

![同候选块数扫描](analysis/figures/page-count-sweep.png)

单个候选排除的注意力读集合为 1–17 MiB；影子实验保留所有存储以便审计，**实际释放字节数为 0**。原始数据见[候选指标](analysis/candidates.csv)、[逐层指标](analysis/layers.csv)及[块数扫描](dose-r1/summary.json)。

## Experiment 4 System One 语义验证

同一模型先分别从 Full/Reduced 的真实 KV 状态生成六项任务状态答案，再由原生 `/v1/systemone` 返回六个分类标签。该接口是分类器，不直接生成长 JSON 答案。问题覆盖目标、进度、已知结论、未决问题、约束与失败经验、下一步行动。

生成 Prompt 要求每字段简述约 30 个英文词，总输出上限 768 tokens；实际回答均未达到上限。这些答案是任务状态摘要，Full 本身也可能遗漏原文中的细节。因此配对摘要不同不等于证明模型已无法访问原事实，六探针也不代表覆盖了全部重要历史信息。

初版判断出现引用格式误报：c01 的目标、进度、结论和未决问题四个字段，移除引用编号后文字完全相同，初版却把其中后三项判为遗漏。修正版去除纯引用括号，要求跨整个 Reduced JSON 检查事实是否保留；两版结果均完整保存。

{judge_table}

修正版的 8 组对照有 7 组通过，**跨字段对照失败**：Full 在 constraints 写“不引入第三方依赖”，Reduced 把同一约束写入 goal，分类器仍以约 0.616 概率选择 omission。该请求 label mass 约 0.994，说明输出遵循标签格式并不能证明语义判断正确。原始请求与结果见[失败对照](judge-r2/control-field_moved.json)。

失败后继续运行同一判断器，只为记录 24 个候选的诊断分布；没有把失败改记为通过。修正版六维判断延迟 P50/P95 为 **{fmt(semantic_p50,3)}/{fmt(semantic_p95,3)} s**。这 34 个 omission 标签不能当作 34 次真实信息损失，110 个 consistent 标签也不能当作安全删除证明。Critical Omission、False-safe 与人工一致率均未估计。

进一步检查了 72 对去掉引用后完全相同的字段：初版将 7 对判为遗漏，修正版仍将 1 对判为遗漏。后者是 c17 的 progress，连引用在内的原始文字都完全相同。见[相同字段审计](analysis/identical-field-audit.json)。这项机械一致性检查再次说明该判断器尚不足以担当删除安全门。

![判断器敏感性](analysis/figures/judge-sensitivity.png)

## Experiment 5 两类指标的关系

以修正版平均 `1−P(consistent)` 为诊断分数，24 个候选的相关性如下。候选共享任务和上下文，且判断器未通过全部对照，因此只做描述性分析。

{correlation_table}

以 Transformer 平均变化的样本中位数 **{fmt(summary['exploratory_median_delta']*100,3)}%** 仅作可视化分组，可得到四类情况。此值不是删除阈值。

{quadrant_table}

![计算变化与判断器分数](analysis/figures/complementarity.png)

两类分数没有表现为同一个信号，但这还不能证明互补收益。尤其不能从低相关直接推出联合规则更安全；High-risk Recall、联合分类效果和 False-safe 需要独立标注后才能计算。所有层的相关性见[逐层相关表](analysis/correlations.csv)。

## Experiment 6 恢复与服务性能

### 真实 KV 恢复组件

归档保存本次真实 FP8 注意力 K/V，每个 token 跨 16 层占 32 KiB。复制和文件读分别预热 2 次、测量 20 次，目标缓冲预先分配；锁页内存分配和物理页安装不计入。每种方法的恢复内容均通过哈希核对。

{recovery_table}

数据目录是 **Ceph 网络文件系统**。冷读项仅对当前实验文件使用客户端 `DONTNEED` 建议，服务端缓存未清空；它不是本地 SSD 或物理盘冷读结果。CPU/文件复制也没有包含循环状态和在线页表接入，不能称为完整请求恢复延迟。

完整前缀从原始位置 0 重算，1K/4K/8K/16K 各 3 次，注意力 KV 全部逐字节一致。计时包含 Prefill 和第一个 token，循环状态没有单独归档核验。只保存候选片段能否精确重建原 KV，仍未被证明。见[恢复原始数据](recovery-r1/results.json)和[重算验证](paired-r1/recovery/prefill.json)。

![恢复组件](analysis/figures/recovery-components.png)

### 生成式 Observer 与 Full Reduced Pair

矩阵为输入 4K/8K/16K/32K × 输出 256/512/1024 × 并发 1/2/4/8/16 × 两类请求，共 120 格、1,116 个请求。当前保存 **{len(serving)}/120 格**，完成 {sum(r['completed'] for r in serving)} 个请求，失败 {sum(r['failed'] for r in serving)} 个请求。

使用原生全上下文 vLLM、eager 模式、FP8 KV，关闭 prefix caching 和投机解码；最大长度 34,816、最大活跃序列 16、每轮 Prefill 1,024 tokens、显存预算 90%。这是完整上下文基线，没有启用项目的定制多分支或有界 KV 路径。缓存规划容量约 112,482 tokens，最长请求理论驻留并发约 3.23。

Observer 的 C 表示 C 个 HTTP 请求；Pair 的 C 表示 C 对，即 2C 个请求。吞吐压测用真实 Trace tokens 裁剪或重复到固定长度，强制生成指定输出长度；Pair 的 Reduced 输入少 32 tokens。这一矩阵测量请求服务成本，不把它当作真实 KV 删除的质量实验。

以下选取 512 输出 tokens 的代表配置；完整数据见[服务矩阵 CSV](analysis/serving.csv)。

{serving_table}

{capacity_note}

TTFT 为客户端收到首个 token 的时间，包含排队；Queueing 来自原生请求指标。总输出吞吐为整波输出 tokens/整波 wall time；单请求 decode 速率单独保存。Prefill 保存原生时间累计量，并导出请求时间归一化速率，以及从提交到所有首 token 到齐这一窗口内的有效输入速率，避免把重叠请求时长当作设备总吞吐。每格仅 1 波，P95 是本波描述值，尤其 C=1 时不能视为稳定尾延迟估计。

![生成式服务吞吐](analysis/figures/serving-throughput.png)

### 原生 System One 分类接口

另测 4 种实际输入长度 × 5 种并发，共 20 格、124 个请求。每个请求只输出 1 个原生 Observer 分类标签；输入长度通过接口的实际 usage 校准为精确的 4K/8K/16K/32K。六维一致性判断的延迟已在 Experiment 4 单独报告。当前保存 **{len(native)}/20 格**。

已完成 {sum(r['completed'] for r in native)} 个原生分类请求，失败 {sum(r['failed'] for r in native)} 个。

{'32K 的首次预校准在 32,767 与 32,769 tokens 之间跳动并按设计停止；失败记录保留。后续用末尾空白修正分词边界，并逐请求核对实际输入为 32,768 tokens。前 15 格原生结果保留，最后 5 格在相同服务配置重启后完成。启动和校准请求不计入正式矩阵。' if finalized and (root/'native-calibration-restart.json').exists() else ''}

{native_table}

{'![原生分类服务](analysis/figures/native-serving.png)' if native else ''}

此接口没有 256/512/1024-token 输出模式，不能与上方长答案吞吐混为一项。原始计数、响应、前后 Prometheus 指标和 GPU 采样均保留。单卡并发增加能否满足“开销可接受”，需结合目标延迟预算判断；本次没有预设生产 SLO。

## 可复核材料与下一步

优先完成[人工语义复核表](analysis/human-semantic-review.csv)：144 个维度为空白，未填入伪造的人类标签。[隐藏 Full/Reduced 身份的答案包](analysis/review-blinded.jsonl)与[解盲键](analysis/review-key.json)分开保存。先核对事实保留和引用支持，再估计 False-safe、候选 Precision/Recall 及联合规则收益。

当前最明确的方法问题是判断器把引用范围和字段位置变化混入“信息丢失”。需要将事实比较与引用依据检查分开，再用独立样本校准。在线清理还需验证混合模型循环状态、1,568-token 分配器块及实际页回收路径；本次影子排除不承担这项结论。

实验代码位于 `experiments/graph_cleanup/`，10 项本地协议和数学校验通过。早期内核布局、Query batch shape 及服务参数失败均另存；只有通过零删除对照的正式配对数据进入汇总。压测为校准判断器暂停过一次，已完成格保留，未完成格另存后重跑，未将两种负载同时运行的时间计入结果。

{'运行结束后的证据包括[权重 SHA-256](model-weight-sha256.json)、[分词器 SHA-256](tokenizer-sha256.json)、[原始文件清单](manifest.sha256.json)、[下载校验](download-verification.json)和[请求记录审计](verification.json)。大体积真实 KV 归档保留在实验服务器，未下载的文件及哈希单列于下载校验记录。' if finalized else ''}

{'[原服务恢复记录](restored-service.json)确认此前的 System One 服务已恢复；实验服务已退出。' if finalized else ''}
'''
    (root/'REPORT.zh-CN.md').write_text(report)
    print(json.dumps(dict(report=str(root/'REPORT.zh-CN.md'),serving_cells=len(serving),native_cells=len(native),finalized=bool(finalized))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    run(p.parse_args())
