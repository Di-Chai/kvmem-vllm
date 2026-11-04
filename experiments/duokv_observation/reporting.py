"""Publication-safe descriptive plots and Chinese reports from saved tables."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
from core import evidence, normalized, read_json
from analysis_stats import mean, cluster_interval


def fmt(value, percent=False):
    if value is None:
        return 'NA'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f'{value:,}'
    return f'{value:.2%}' if percent else f'{value:.4f}'


def table(rows, columns):
    text='| '+' | '.join(label for key,label in columns)+' |\n'
    text+='| '+' | '.join('---' for _ in columns)+' |\n'
    for row in rows:
        text+='| '+' | '.join(fmt(row.get(key)) if not isinstance(row.get(key),str) else row[key] for key,label in columns)+' |\n'
    return text


def make_cases(root, units, edges, gaps, relations):
    by_id={u['unit_id']:u for u in units}
    cases=[]
    # Revalidate the supplied bundle's known substantive example against raw
    # spans. This is an analyst-inspected observation, never human ground truth.
    wasmi=[u for u in units if u['trajectory_id']=='deepswe/seed-1001/wasmi-trap-coredumps']
    src=next((u for u in wasmi if u['content_type']=='tool_result' and 'assert_eq!(mem::size_of::<Error>(), 8)' in u['text']),None)
    dst=next((u for u in wasmi if u['content_type']=='assistant_text' and 'Since there\'s a size test' in u['text']),None)
    if src and dst:
        mentions=sorted({u['step'] for u in wasmi if 'size_of::<Error>' in u['text'] or ('8 bytes' in u['text'] and 'Error' in u['text'])})
        pos=src['text'].index('assert_eq!(mem::size_of::<Error>(), 8)')
        cases.append(dict(case_id='wasmi_error_size_constraint',source=evidence(src,(max(0,pos-100),min(len(src['text']),pos+150))),target=evidence(dst),
            observation='Early Error size invariant explicitly constrains final design prose.',
            level='explicit',validation_status='assistant_inspected_trace_case_not_human_ground_truth',
            birth_to_late_reference_distance=dst['step']-src['step'], matching_reference_steps=mentions,
            caveat='Intervening mentions exist; birth-to-final distance is not a dormant interval. Final design prose is not implemented or tested success.',
            original_bundle_evidence='analysis/retention/content-reuse-evidence.json#/persistent_constraint'))
    dna=[u for u in units if u['trajectory_id']=='terminalbench/dna-assembly']
    srcs=[u for u in dna if u['message_index']==12 and u['content_type']=='terminal_observation'
          and u['text'].startswith('=== FINAL ASSEMBLY STRUCTURE ===')]
    dst=next((u for u in dna if u['message_index']==15 and u['content_type']=='analysis' and '714' in u['text']),None)
    if srcs and dst:
        cases.append(dict(case_id='dna_fragment_lengths',sources=[evidence(u) for u in srcs],target=evidence(dst),
            level='semantic',observation='Later analysis repeats fragment lengths found in earlier terminal observations.',
            validation_status='assistant_inspected_trace_case_not_human_ground_truth',
            caveat='Only these factual values are visibly carried forward. Entire old observation is not shown to be superseded.',
            original_bundle_evidence='analysis/retention/content-reuse-evidence.json#/information_repeated_in_shorter_analysis'))
    seen=set()
    for gap in sorted([g for g in gaps if g['level']=='explicit' and g['gap_since_last_observed_use']>=16],key=lambda g:(by_id[g['unit_id']]['content_type'] not in {'task_instruction','tool_result','terminal_observation'},-g['gap_since_last_observed_use'])):
        if gap['task_id'] in seen:
            continue
        edge=next(e for e in edges if e['source_unit_id']==gap['unit_id'] and e['target_step']==gap['target_step'] and e['level']=='explicit')
        cases.append(dict(case_id=f'long_gap_{len(seen)+1}',observation='Distinctive text is observed again after a gap in detected reuse.',
            gap_since_last_observed_reuse=gap['gap_since_last_observed_use'],edge=edge,
            validation_status='automated_candidate_for_review',caveat='Detector silence is not proof of semantic nonuse; source attribution may be ambiguous.'))
        seen.add(gap['task_id'])
        if len(seen)>=8:
            break
    return cases


def counterfactual_design(units,edges,gaps,relations):
    by_id={u['unit_id']:u for u in units}
    candidates=[]
    seen=defaultdict(set)
    for gap in sorted(gaps,key=lambda g:g['gap_since_last_observed_use'],reverse=True):
        if gap['level']!='explicit' or gap['gap_since_last_observed_use']<16:
            continue
        u=by_id[gap['unit_id']]
        group='dormant_but_reused'
        if u['task_id'] in seen[group] or len(seen[group])>=12:
            continue
        checkpoint=gap['target_step']-gap['gap_since_last_observed_use']+1
        candidates.append(dict(candidate_id=f'{group}:{len(seen[group])+1}',category=group,
            trajectory_id=u['trajectory_id'],task_id=u['task_id'],seed=u['seed'],
            checkpoint_step=checkpoint,unit_ids=[u['unit_id']],evidence=evidence(u),
            future_reuse_step=gap['target_step'],observed_gap=gap['gap_since_last_observed_use'],
            selection_uses_future_for_offline_validation_only=True,requires_human_review=True))
        seen[group].add(u['task_id'])
    for relation in relations:
        group='superseded' if relation['relation'] in {'fully_superseded','partially_superseded'} else ('uncertain' if relation['relation']=='uncertain' else None)
        if not group or relation['followup_steps']<4 or relation['task_id'] in seen[group] or len(seen[group])>=12:
            continue
        u=by_id[relation['source_unit_id']]
        candidates.append(dict(candidate_id=f'{group}:{len(seen[group])+1}',category=group,
            trajectory_id=u['trajectory_id'],task_id=u['task_id'],seed=u['seed'],
            checkpoint_step=relation['event_step'],unit_ids=[u['unit_id']],evidence=evidence(u),
            relation_id=relation['relation_id'],event_id=relation['event_id'],
            classification=relation['relation'],requires_human_review=True))
        seen[group].add(u['task_id'])
    design=dict(schema_version=1,status='design_only_no_rollouts_executed',
        arms=['full_history','remove_selected_history','retain_selected_history'],
        arm_definitions={
            'full_history':'All history available at checkpoint; unrestricted reference arm.',
            'remove_selected_history':'Apply recorded policy to remaining history after removing reviewed selected spans; rebuild the full model prefix and KV cache.',
            'retain_selected_history':'Apply the same policy and total memory budget as remove arm, pin selected spans; evict a matched token budget of other eligible history where necessary.'},
        additional_control='Sham deletion with identical reconstruct/re-prefill work but no content deletion; optional matched random deletion.',
        runner_interface=dict(required_inputs=['candidate_id','task_id','environment_snapshot_uri','environment_snapshot_sha256',
            'checkpoint_conversation','checkpoint_step','model_revision','tokenizer_revision','sampling_seed','policy_config',
            'selected_raw_message_spans','memory_budget_tokens','stop_condition','independent_task_verifier'],
            required_outputs=['arm','sampling_seed','task_success_or_unknown','tool_calls','new_trajectory_uri',
                'input_tokens','output_tokens','prefill_tokens','ttft_ms','wall_latency_ms','recompute_tokens','recompute_time_ms','failure_reason']),
        protocol=[
            'Restore exactly the same checkpoint filesystem, tool/process state and conversation for all arms. Do not replay recorded future actions.',
            'If checkpoint snapshots are unavailable, rerun from the same initial environment and record prefix divergence; report this separately from checkpoint-matched trials.',
            'Review candidates and dependency closures. Preserve valid message/tool-call/result structure; use recorded span-level replacements, never malformed tool JSON.',
            'Rebuild prefix, re-tokenize and re-prefill in every arm. Continue NEW model inference and allow trajectories to diverge.',
            'Use paired sampling seeds (initial pilot five per candidate), identical budgets/deadlines and independent benchmark verifier.',
            'Score success as unknown when verifier missing. Log crashes, timeouts, costs and divergent tool actions; do not discard failed trials.',
            'Analyze paired outcomes and costs with task-cluster bootstrap; report category and benchmark separately.',
            'Estimate variance in pilot before fixing confirmatory sample size; no success threshold derived from these observational labels.'],
        unavailable_assets=['checkpoint environment snapshots','model and tokenizer artifacts','unified independent grading results','complete per-step KV/cost telemetry'],
        candidates_by_category=dict(Counter(c['category'] for c in candidates)),
        causal_claims_permitted=False)
    return candidates,design


def make_figures(out,demand,curves,gaps,event_summary,cross_tasks,baselines):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/duokv-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    colors={'deepswe':'#276FBF','terminalbench':'#D97B2B'}
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,level in zip(axes,['explicit','semantic','potential_dependency']):
        panel_max=0.
        for family in colors:
            rows=[r for r in curves if r['family']==family and r['content_type']=='ALL' and r['level']==level]
            ax.step([0]+[r['distance'] for r in rows],[0]+[r['cumulative_reuse'] for r in rows],where='post',label=family,color=colors[family])
            panel_max=max(panel_max,max((r['cumulative_reuse'] for r in rows if r['distance']<=80),default=0.))
        ax.set(xlim=(0,80),ylim=(0,min(1,max(.002,panel_max*1.2))),xlabel='Steps since creation',ylabel='1 - KM survival',title=level.replace('_',' ').title())
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Observed rule candidates; panel-specific Y scales; trace ends censored (independent censoring not established)')
    fig.savefig(out/'figures/future-demand-cdf.png');plt.close(fig)
    kinds=['thinking','analysis','plan','tool_result','terminal_observation','task_instruction','tool_call','assistant_text']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,family in zip(axes,colors):
        arrays=[[r['gap_since_last_observed_use'] for r in gaps if r['family']==family and r['content_type']==kind and r['level']=='explicit'] for kind in kinds]
        keep=[(k,a) for k,a in zip(kinds,arrays) if a]
        if keep:
            ax.boxplot([a for k,a in keep],orientation='horizontal',tick_labels=[f'{k} (n={len(a):,})' for k,a in keep],showfliers=False)
            ax.set_xscale('log')
        ax.set(title=family,xlabel='Gap since previous detected use (log steps)')
        ax.axvline(16,ls='--',color='#b22',alpha=.6)
    fig.suptitle('Explicit text reuse candidates; one source / target step; outliers hidden')
    fig.savefig(out/'figures/reuse-distance-by-type.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,family in zip(axes,colors):
        rows=[r for r in event_summary if r['family']==family]
        labels=[r['event_type'].replace('_',' ') for r in rows];base=np.zeros(len(rows))
        for key,color in [('fully_superseded','#277E66'),('partially_superseded','#8EBF87'),('not_superseded','#729BCC'),('uncertain','#C5C8CC')]:
            values=np.array([r[key]/r['assessed_event_source_pairs'] for r in rows])
            ax.barh(labels,values,left=base,label=key.replace('_',' '),color=color);base+=values
        ax.set(title=family,xlim=(0,1),xlabel='Fraction of assessed event-source pairs')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.4),ncol=2)
    fig.suptitle('Information replacement candidates; this plot does not measure causal value decay')
    fig.savefig(out/'figures/event-value-transition.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    eligible=[r for r in event_summary if r['task_macro_mean'] is not None]
    for i,row in enumerate(eligible):
        x=row['task_macro_mean'];lo=row['ci_low'];hi=row['ci_high']
        ax.plot(x,i,'o',color=colors[row['family']])
        if lo is not None and hi is not None:
            ax.hlines(i,lo,hi,color=colors[row['family']],lw=2)
    ax.set_yticks(range(len(eligible)),[r['family']+' / '+r['event_type']+f' (tasks={r["cluster_tasks"]})' for r in eligible])
    ax.axvline(0,color='#777',ls='--');ax.set(xlabel='Task mean difference-in-differences in observed reuse / Step',title='Matched complete windows; task bootstrap 95% intervals; observational, not causal')
    fig.savefig(out/'figures/event-demand-change-controls.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    labels=[r['task_id'].replace('-',' ') for r in cross_tasks];a=[r['stable_demand_ratio'] or 0 for r in cross_tasks];b=[r['trajectory_dependent_demand_ratio'] or 0 for r in cross_tasks]
    ax.barh(labels,a,label='Observed reuse in all 4 seeds',color='#276FBF');ax.barh(labels,b,left=a,label='Observed reuse in 1-3 seeds',color='#D97B2B')
    ax.set(xlim=(0,1),xlabel='Fraction of entity/constraint groups present in all 4 seeds',title='Cross-seed observation proxies; 16 task clusters')
    ax.legend(loc='lower right');fig.savefig(out/'figures/cross-seed-comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,family in zip(axes,colors):
        for baseline in ['random','lru','frequency','query_relevance','execution_state']:
            rows=[r for r in baselines if r['family']==family and r['content_type']=='ALL' and r['target']=='explicit' and r['baseline']==baseline]
            ax.plot([r['window'] for r in rows],[r['pr_auc_average_precision'] or 0 for r in rows],marker='o',label=baseline)
        ax.set(title=family,xlabel='Future window (steps)',ylabel='Out-of-fold average precision')
        ax.set_xticks([1,4,16]);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Rule-label predictability on sampled history; task-grouped folds')
    fig.savefig(out/'figures/baseline-comparison.png');plt.close(fig)


def make_reports(out,config,verification,quality,audit,demand,sensitivity,dormant,events,cross,baselines,kv,cases,candidates):
    def save(name,text):
        (out/name).write_text(text.strip()+'\n',encoding='utf-8')
    common='本报告使用离线确定性规则。Explicit / Semantic / Potential Dependency 分开统计，所有自动关系均为候选，不是人工确认 Ground Truth，也不是因果重要性。未观察到复用不等于不需要。'
    main=[r for r in demand if r['content_type']=='ALL' and r['window']==4]
    subtype_path=out/'subtype-reuse-statistics.csv'
    subtype_rows=list(csv.DictReader(subtype_path.open())) if subtype_path.exists() else []
    subtype_rows=[r for r in subtype_rows if r['window']=='4' and r['level']=='explicit']
    qcounts=Counter()
    for q in quality:
        for k in ('units','main_units','retry_units','parser_retry_turns'):
            qcounts[k]+=q[k]
    missing=[q for q in quality if not q['final_saved_response_available']]
    fresh=[q for q in quality if q['attempt_source']=='fresh_restart']
    save('data_quality_report.md',f'''# 数据质量与完整性

原数据包：`{config['bundle']}`。运行前、运行后均通过包内 `verify_bundle.py`，校验原文件哈希、原消息逐项一致性、标签坐标和示例 token 连续性。原数据未改写。

{table([verification],[('trajectories','轨迹'),('messages','消息'),('assistant_turns','回复轮'),('executor_events','审计事件'),('labels','原标签')])}

DeepSWE：16 个任务 × 4 Seeds = 64 轨迹；Terminal-Bench：86 轨迹。DeepSWE 的五条 fresh restart 如下：

{table(fresh,[('task_id','任务'),('seed','Seed'),('attempt_source','来源')])}

缺失末轮回复：

{table(missing,[('task_id','任务'),('saved_assistant_turns','已保存回复轮'),('main_steps','排除 Retry 后 Step')])}

总共生成 {qcounts['units']:,} 个非空结构片段，其中主分析 {qcounts['main_units']:,} 个；{qcounts['parser_retry_turns']:,} 轮 Parser Retry 单独识别。所有 {audit['unit_span_checks']:,} 个片段均通过原消息字符切片重建校验。

## 坐标与去重

原包没有 message_id，因此稳定生成 `trajectory_id:m{{message_index}}`，不冒充原生 ID；原 `message_index`、`turn`、文件行号和 raw source pointer 同时保留。原始轮次字段为 `turn_id`；主分析 `step` 是保留回复轮的顺序号，初始输入为 0。工具反馈沿用对应回复 Step，同一 Step 内关系不计“未来”。一次 Step 仍可能有多个工具调用，不能等同命令次数。

片段按已有字段、段落、代码围栏、终端提示及最多 1,400 字符的行边界划分，是结构化语义代理，不保证每片段为不可分割命题。重叠标签以 task_instruction / execution_status 等子标签优先，避免重复统计。Terminal-Bench 可解析的 analysis、plan、command 映射到 original JSON 字符串的原始区间；另存 decoded 坐标和转义类型。无法可靠映射的正文保留 assistant_text；计数见 trajectory-quality.csv。框架规则、工具定义、非任务初始文本不进入需求分母。

消息没有可验证的产生时间戳，`timestamp=null`，不根据文件顺序伪造墙钟时间。执行器审计保留原始 at 与 session-only 归属，未强行对齐到轮次。可信追加的 Executor status 中的 command ordinal 可对齐同轮命令；多命令批次不任意分配测试结果。

## 排除与解释边界

主分析排除 Parser Retry 的 assistant 内容和反馈，同时压缩 Step 轴；敏感性分析保留全部回复轮和 Parser Feedback。缺失回复、fresh restart 另外提供排除敏感性；没有拼接被替代的中断尝试。所有轨迹结束都右删失，task_complete / completion_declaration 不表示正确性成功。

两 benchmark 的配置、任务与停止条件不同，不能将差异归因于 KV 策略。150 条轨迹并非随机代表性样本。原包 KV 特征只覆盖两条示例，全文没有推算其余轨迹的精确 token 或 32-token block 标签。

`structured/analysis-audit.json` 记录自动校验；`structured/executor-audit-unmapped.jsonl` 保留全部执行审计。
''')
    save('future_demand_report.md',f'''# Experiment 1：Future Demand

{common}

## 主要结果（未来 4 Step）

{table(main,[('family','Benchmark'),('level','层级'),('units','信息片段'),('observed_reused_units','观察到复用'),('never_observed_reuse_ratio','未观察到比例'),('full_window_units','完整窗口分母'),('full_window_hits','窗口内阳性'),('future_reuse_probability','比例'),('long_range_reused_units','≥16 Step 再用片段')])}

## 识别规则与证据

Explicit：后续 assistant 字段与历史片段共享具有至少 4 个非停用内容词、长度≥36 字符的 8-token 片段；采用内容哈希 winnowing 以限制通用短语碰撞。标点/空白可不同，原文双方 excerpt 均保留。该规则检测明确文本再现，不证明因果依赖，通用模板重复可能造成假阳性。

Semantic：仅检测实体/数值事实的重述，且目标含推理或行动提示。它覆盖很窄，不能据低比率判断真正的语义复用少。Potential Dependency：至少两个具体实体同时出现，且目标有推理/行动提示；它本身不是复用确认，绝不并入 Explicit / Semantic。单关键词或单路径匹配不构成复用。

来源归因选择每个匹配证据最近的历史片段，并记录 competing_prior_occurrences。相同信息在多个旧片段存在时，无法证明究竟使用了哪个原 KV；这会低估较早片段的长期需求。只把 assistant 推理/文本/调用作为需求目标；重复工具输出本身不构成 agent 使用。每来源片段、目标 Step、层级至多计一次；层级可能在不同证据上重叠，不能简单相加当作互斥总量。

## 指标定义

| 指标 | 定义、分母与排除 |
| --- | --- |
| Future Reuse Probability(k) | 产生后 k Step 内至少一次相应层级复用的片段 / 能观察完整 k Step 的片段；k=1,4,16；截断窗口全部排除，包括提前出现阳性的截断片段。 |
| First Reuse Distance | 首次观测目标 Step − 产生 Step；中位数分母仅为发生过相应复用的片段。 |
| Gap since Last Use | 当前目标 Step − 上次观测使用 Step，首次使用以产生 Step 为起点。Explicit 与 Semantic 任一层级的中间使用都会中断二者的休眠间隔；Potential 单独计算。不是单纯产生到最终引用的距离。 |
| Reuse Frequency | 每片段不同目标 Step 的观测复用次数，主表均值包括 0。 |
| Long-range Reuse Ratio | 至少一个使用间隔≥16 Step 的片段 / 至少复用一次的片段；8/16/32 阈值敏感性另表。 |
| Never Observed Reuse Ratio | 到保存轨迹末尾未检测到复用的片段 / 全部纳入片段；全部为首次复用右删失，不能称为永不需要。 |
| KM CDF | 时间至首次观测复用的 1−Kaplan–Meier；未发生者在末 Step 删失，同距事件先于删失。任务结束可能信息性删失，独立删失假设未得到验证，曲线仅描述该记录过程。 |
| Dormant Future Demand | 产生后 q=4/8/16 Step 没有检测到使用且后续 4 Step 完整可观测的片段中，下一窗口复用的比例。 |

每行同时给出分子、分母、剔除窗口数、零随访数量、轨迹数、任务数。置信区间按 task_id 聚类 bootstrap 1,000 次，同一 DeepSWE 任务四个 seed 一起抽样；任务均值与按片段加权的比例分别保存。不同类型规则可检测性不同，不能把低观测率等同低真实需求。

## 代码、错误与执行输出的补充比较

以下 subtype 由行结构和错误标记作互斥规则划分（错误优先于代码），不是人工语义标注；完整CSV同时保存分子和短随访排除数。子类型可能混有终端回显代码，不能解释成纯程序源码或纯运行错误。

{table(subtype_rows,[('family','Benchmark'),('subtype','结构子类型'),('units','片段数'),('observed_reused_units','已复用'),('median_first_reuse_distance','首次距离中位数'),('long_range_reused_units','长间隔片段'),('long_range_ratio_among_reused','占已复用比例')])}

![Future demand](figures/future-demand-cdf.png)

![Reuse distances](figures/reuse-distance-by-type.png)

`future-reuse-statistics.csv`、`unit-reuse-statistics.csv`、`reuse-gaps.csv`、`dormant-future-demand.csv`、`future-demand-survival.csv` 与 `structured/reuse-evidence.jsonl` 可完整追溯。`sensitivity-statistics.csv` 提供含 Retry、排除 fresh restart、排除不完整轨迹三种口径。
''')
    save('event_driven_decay_report.md',f'''# Experiment 2：Event-driven Value Decay

{common}

**目前能支持“局部信息被新结果更新”的候选观察，不能确认信息的因果价值已经消失。**

## 观测结果

{table(events,[('family','Benchmark'),('event_type','事件'),('observed_events','事件数'),('assessed_event_source_pairs','候选关系分母'),('fully_superseded','完整替代'),('partially_superseded','部分替代'),('not_superseded','未替代'),('uncertain','不确定'),('matched_control_pairs','匹配对照数')])}

## 可验证事件和覆盖限制

DeepSWE 依靠 tool_use_id → tool_result，Edit / Write 必须有实际成功确认；Bash 的测试/构建事件要求命令类型与输出摘要一致。返回 running/background/timeout 的消息不视为确定完成。Terminal-Bench 使用当轮追加的 shell completion confirmed + command ordinal，只有单命令批次可赋予具体测试/构建结果，且必须同时有结果摘要；单独 exit=0 不能证明测试通过。Terminal-Bench shell 内文件写入未做推测性的 Code Modification 标注。

独立 Hypothesis Verification 与框架验证 Task Completion 在当前规则中不可可靠识别，因此没有伪造事件或把数量 0 解释为现实中不存在。模型自行宣称修复、完成、验证成功均不能升级为真实事件。调用执行成功也不等于任务验收成功。

## 关系与残留信息

候选集合包括上一条相同确切工具作用域的结果，以及每个事件作用域实体、每种内容类型最近的一条历史片段；不是对所有 B×E 穷举，因此替代比率分母仅为“被评估的事件—片段对”。同作用域**只读观察**的完整文本承接、确认 Edit 的 old_string 替换、相同确切测试命令由失败转成功可产生替代候选。重复的“文件已更新”回执不携带上次修改内容，不能仅凭字面相同判为完整替代；对应工具结果的确认也不自动替代旧 Thinking。

Fully Superseded 表示规则发现完整**表示内容**被承接/更新，不代表可安全删除；Partially Superseded 保留未覆盖的诊断/约束；用户任务约束默认不因局部事件被解除；其余保留 Uncertain。`residual_lexical_fraction` 仅在可度量的确切文本覆盖下提供（1−覆盖字符比）；`residual_semantic_information_fraction` 始终 null。未完成依赖目前是 unknown，不能宣称都已结束。

Superseded Ratio=(Fully+Partially)/全部候选对；另外提供去掉 Uncertain 的条件比例，不能只展示后者。完整替代、部分替代比例分母同前。Post-event Reuse 分母为候选对，观察时间各异，附 followup_steps；公平窗口比较仅纳入事件前后各 4 Step 完整、且片段在事件前 4 Step 已产生的样本。

特别注意来源归因偏差：当同一信息在新结果里出现后，检测器会将后续匹配归给最近的新片段，因此旧 unit 的 Post-event Reuse 可能机械下降，而信息内容仍然被使用。这里测量的是特定旧片段的观测归因，不是该语义事实的整体需求；不能用这一下降独立验证 H2。

## 事件前后与对照

前窗口为 [t−4,t)，后窗口为 (t,t+4]，排除事件 Step 本身的机械引用；需求为该片段 Explicit 或 Semantic 的不同目标 Step 数 / 4。对照在同一轨迹挑选相同 content_type、年龄差≤2 Step 的历史片段和伪事件时点，该前后窗口内不能有同种事件；选择不使用未来复用结果。没有合格对照时不强制匹配。工具完成事件过密的轨迹往往没有这类对照。

DiD = 处理片段(后−前) − 对照片段(后−前)，先在任务内平均，再按任务聚类 bootstrap。参考对照不能消除任务阶段、代码状态和难度混杂，不能作因果衰减估计。`event-control-comparison.csv` 保留全部可核对的对照与排除信息。

{table(events,[('family','Benchmark'),('event_type','事件'),('matched_control_pairs','匹配对照'),('mean_pre_reuse_rate','事件前复用率'),('mean_post_reuse_rate','事件后复用率'),('task_macro_mean','任务平均DiD'),('ci_low','95%下限'),('ci_high','95%上限'),('cluster_tasks','任务数')])}

![Matched observed demand changes](figures/event-demand-change-controls.png)

Event-to-Decay Latency **不可识别，保留 null**；另提供最后一次观测复用距离事件的偏移，它既不等于价值消失时间，也不支持“零后续使用=立即无价值”。失败后是否重新使用需要查后续证据，不会仅因失败事件就自动赋值为重新重要。

![Replacement candidates](figures/event-value-transition.png)

`value-evolution-examples.jsonl` 包含四种关系典型候选及原文定位；`structured/event-source-relations.jsonl` 保存完整证据、置信度、残留、事件后使用与对照。全部自动关系尚未经人工 Ground Truth 标注。
''')
    cross_macro={key:cluster_interval([r.get(key) for r in cross]) for key in ['stable_demand_ratio','trajectory_dependent_demand_ratio','reuse_consistency','supersession_consistency','event_specific_supersession_consistency']}
    save('cross_seed_report.md',f'''# Experiment 3：Cross-seed Demand Consistency

{common}

统计单位为 16 个任务；64 条轨迹不是 64 个独立任务。以路径、函数/类符号、错误类型、精确任务约束片段作实体对齐；相同实体的上下文可能不同，所以这是对齐代理，并非语义等价的已确认信息。

{table(cross,[('task_id','任务'),('four_seed_groups','四 Seed 均出现的组'),('excluded_missing_seed_groups','缺失 Seed 组'),('stable_demand_ratio','稳定需求比例'),('trajectory_dependent_demand_ratio','路径相关比例'),('reuse_consistency','复用一致性')])}

## 定义与分母

Stable Demand Ratio：在四 seed 均出现且四 seed 都有 Explicit/Semantic 来源片段复用的实体/约束组数 / 四 seed 均出现的组数。

Trajectory-dependent Demand Ratio：同一分母中只在 1–3 seed 观察到复用的组比例。0 seed 复用不属于此类，仍是未观察到；未出现的 seed 视为 unknown，不能当不需要。一个组的部分片段复用不代表整组内容都被需要。

Cross-seed Reuse Consistency：可比 seed 两两观测复用二元状态一致率，先实体组、再任务平均。Supersession Consistency 将至少一条部分/完整替代候选视为观测阳性；仅有 Not Superseded 的已评估组才可为阴性；含 Uncertain 且无阳性、或从未评估的 seed 为unknown，排除其配对，绝不把不确定当未替代。两者仍可能受双零一致影响。另在 `cross-seed-supersession-by-event.csv` 对每个实体组与事件类型，仅比较至少有两seed具有已知候选状态的组，提供 event_specific_consistency、暴露seed、unknown seed和可比seed对数。暴露仅表示候选检测覆盖，不保证真正的执行机会、语义和上下文完全相同，不能直接作因果一致性结论。

以下为 16 任务等权均值与 task-cluster bootstrap 95% 区间：

{table([dict(metric=k,**v) for k,v in cross_macro.items()],[('metric','指标'),('task_macro_mean','均值'),('ci_low','下限'),('ci_high','上限'),('cluster_tasks','任务数')])}

fresh restart 的五个任务已逐组标记；`cross-seed-no-restart-tasks.csv` 排除整个含重跑任务，保留其余 11 个四-seed完整任务作敏感性分析，避免把三-seed任务混入四-seed分母。

![Cross seed](figures/cross-seed-comparison.png)

`cross-seed-demand.csv` 保存每个对齐组各 seed 的全部 unit_id；可沿 `structured/units.jsonl` 与 `structured/reuse-evidence.jsonl` 回到原文。要判断哪些完整约束真正由任务决定，需要人工确认组内语义一致及独立删除实验。
''')
    base4=[r for r in baselines if r['window']==4 and r['content_type']=='ALL' and r['target']=='explicit']
    prediction_notes=[]
    for family in ('deepswe','terminalbench'):
        rr=[r for r in base4 if r['family']==family and r['pr_auc_average_precision'] is not None]
        if rr:
            best=max(rr,key=lambda r:r['pr_auc_average_precision'])
            prediction_notes.append(f'{family} 在该窗口的最高点估计为 {best["baseline"]}，AP={best["pr_auc_average_precision"]:.4f}，标签阳性率={best["prevalence"]:.4f}；此处没有建立基线间差异的显著性。')
    save('prediction_report.md',f'''# Experiment 4：Future Demand Predictability

{common}

目前评估的是对规则观测标签的预测能力，尤其 Query Relevance 与规则标签共享词面特征，可能产生测量上的循环优势；不能把性能解释为真实 KV 因果需求可预测。

## Explicit，未来 4 Step

{table(base4,[('family','Benchmark'),('baseline','基线'),('n','评估窗口'),('positives','阳性'),('precision','Precision'),('recall','Recall'),('f1','F1'),('pr_auc_average_precision','PR-AUC(AP)'),('brier','Brier'),('ece_10_bins','ECE')])}

{' '.join(prediction_notes)}

## 时间与任务隔离

按 benchmark 内 task_id 哈希顺序轮流分配 5 folds；同任务全部 seed 始终在同一 fold。每个 held-out fold 的校准和阈值仅用其他任务。`structured/task-folds.json`、`prediction-models.jsonl` 列出 train/test task 名单，已检查交集为零。

在第 t Step 结束时，从所有已产生的合格历史片段中等概率无放回抽至多 12 个，每隔 8 Step 一个检查点；采样仅由 ID、t 和固定种子决定，不依赖未来标签。只评估 t+k 不超过保存末 Step 的完整窗口，k=1,4,16。指标是此“检查点等权、每点均匀抽历史”总体上的结果，非所有 B×t 组合的无偏片段加权估计；保存 inclusion_probability 供其他加权研究。

LRU=1/(1+距最近观测使用的年龄)，无使用则按产生时间；Frequency=min(1,log(1+此前使用次数)/log21)；Query Relevance 为当前可见反馈词集合与历史片段 Jaccard，无反馈时使用当前 assistant 字段；Execution State 为固定权重的 relevance、recency、frequency、用户约束标记、最近失败×错误类型组合；Random 为 ID/t 固定随机分数。代码和样本文件保存全部公式输入。没有调用未来一轮 Query，也没有用全轨迹出现频率作历史频率。

Explicit 与狭义 Semantic 单独建目标；**High-confidence Dependency 没有可靠确认标签，保留 NA，样本数 0，不用 Potential 的未观测项制造负例。** Semantic 阳性稀少的分组，AP/Recall 可能不可估；训练无阳性时不生成肯定预测。

## 指标

Precision=TP/预测阳性数（预测为空约定0）；Recall=TP/实际阳性数（无阳性 NA）；F1 为二者调和均值。阈值仅在训练任务上最大化 F1。PR-AUC 使用并列分数组合的非插值 Average Precision，按原始规则分数排序，非梯形面积。概率校准按训练 raw-score 的10个固定区间估计并加2个训练先验伪计数；Brier=平均(p−y)²，ECE=10个概率箱的样本占比×|平均概率−阳性率|。样本数、阳性数、任务数与内容类型结果均在 CSV；任务 AP 的区间仅纳入存在阳性的任务，cluster_tasks 给出实际分母。

![Baselines](figures/baseline-comparison.png)

`future-demand-prediction.csv` 为逐预测结果；`baseline-comparison.csv`、`prediction-calibration.csv` 为聚合。`structured/prediction-examples.jsonl` 保存可重建的源片段、检查点、当前可见 Query IDs 和特征时间上界。
''')
    save('kv_snapshot_case_study.md',f'''# Experiment 5：既有 KV Snapshot 再分析

读取了 `analysis/retention/block-trends.csv`（299 个内容块）、`retention-observations.json` 与 `evidence/selected-capture-snapshots.jsonl`（5 条原始快照）。仅分析 seed-1001 / wasmi-trap-coredumps 和 dna-assembly。

DeepSWE 仅第31、39轮；Terminal-Bench 仅第10、12、16轮。selected_fraction 是已记录策略选中的 token 比例，既非因果重要性，也不是语义需求概率。eligible=false 的观测保留 null；不插值，不把无资格当作0。

没有完整字符→token映射，因此按 birth_turn + content_type 聚合既有 block，将对应语义片段的未来4个主分析 Step需求与该组 token 加权选择比例关联。多个 block 不重复作为独立语义需求样本；映射是粗粒度的，不能称为精确单块对应。只有整组 eligible 且未来窗口完整、且有语义片段匹配的观测进入相关统计，另列包含/排除固定 sink 的结果。

{table(kv,[('trajectory_id','案例'),('include_sink','包含sink'),('matched_aggregate_observations','有效聚合观测'),('snapshots','快照数'),('spearman','Spearman')])}

相关系数仅描述两个案例内的重复稀疏观测；没有独立样本量或有效显著性检验，恒定值/不足三观测时为 NA，不能推广到150条轨迹或归因某项KV策略。

已核验的早期 Error 大小约束→最终设计引用，以及 dna 片段长度→较短 analysis 的事实承接，见 `structured/trace-case-studies.jsonl`。前者不证明最后设计实施成功；后者不证明长工具结果全部被摘要覆盖。两者均不支持“删旧留新必然安全”。

`kv-snapshot-correlation.csv` 保存每次聚合的原 block IDs、语义 unit IDs、资格、sink标志与随访窗口；`kv-correlation-summary.csv` 保存描述性相关。
''')
    longrows=[r for r in demand if r['window']==4 and r['content_type']!='ALL' and r['level']=='explicit' and r['long_range_reused_units']]
    longrows=sorted(longrows,key=lambda r:r['long_range_reused_units'],reverse=True)[:12]
    relation_counts=audit['relationship_counts']
    save('experiment_summary.md',f'''# DuoKV Observation Study：第一轮结果

完成日期：2026-11-03。已完成 Phase A 的未来需求、事件关系与跨 Seed 定量分析，Phase B 的五个规则基线，Phase C 的两条 KV 案例关联，以及 Phase D 的候选样本和反事实实验接口设计。没有启动 LLM，没有执行删除重跑。

**核心结论：观察到了历史信息较晚再次出现的证据，以及执行结果更新部分旧信息的候选现象；现有轨迹不足以证明信息可安全删除，或事件导致了因果价值衰减。**

## 1. 是否存在长期暂时不用、但未来仍需要的信息？

存在“至少16个主分析 Step未检测到使用后又出现明确文本复用”的候选，计数如下。这里“未检测到”不等于真正未使用，“文本复用”不等于因果必要。来源和目标原文、最近一次观测使用距离保存在 `reuse-gaps.csv` 与证据 JSONL。

{table([r for r in main if r['level']=='explicit'],[('family','Benchmark'),('long_range_reused_units','长期间隔再用片段'),('long_range_ratio_among_reused','占已复用片段比例'),('observed_reused_units','已复用片段分母')])}

已核验案例：wasmi 的早期 `size_of::<Error>() == 8` 约束在最终设计中明确影响表示选择。中间存在相关提及，因此不能把产生到末轮的距离当作完整休眠期。自动长间隔候选另列8个不同任务供复核。

## 2. 哪些类型容易出现长期需求？

按“出现长间隔复用的片段数量”列出下表，并同时给出相对于该类型已复用片段的比例。不能仅以数量或词面识别率对真实需求强弱排名；类型长度、拆分粒度和规则召回率不同。用户任务约束有可追溯的长期引用案例，但仍需控制这些因素。

{table(longrows,[('family','Benchmark'),('content_type','内容类型'),('long_range_reused_units','长间隔片段'),('observed_reused_units','已复用分母'),('long_range_ratio_among_reused','比例')])}

## 3. 是否存在由执行事件驱动的信息价值下降？

规则得到 Fully Superseded {relation_counts.get('fully_superseded',0):,} 对、Partially Superseded {relation_counts.get('partially_superseded',0):,} 对，Uncertain {relation_counts.get('uncertain',0):,} 对。它们是被选候选集合中的**表示更新/承接关系**，不是价值下降的 Ground Truth。相同测试命令由失败到成功可更新“当前失败状态”，但此前诊断、约束、修复依据仍可能有用。Event-to-Decay Latency 与真实残留语义比例目前不可识别。

事件前后各4 Step的完整窗口与同轨迹同内容类型、相近年龄、无同类事件窗口的对照均已保存；无合格对照不补造。任务结束/截断未当作衰减。观察性对照仍存在执行阶段混杂，H2 的因果版本需要反事实验证。

## 4. 同任务不同执行路径是否有共同需求？

完成16任务的四Seed实体/约束组对齐，报告稳定、路径相关和未观测三类。缺失seed为unknown；按任务而非64条轨迹聚类。实体一致并不保证语义等价，跨seed替代的一致性还需控制事件暴露，不能从双零一致推出共同规律。

## 5. 当前执行状态能否预测未来需求？

完成 Random / LRU / Frequency / Current Query Relevance / Execution State，按任务5-fold评估1、4、16 Step窗口，并分别提供内容类型、PR-AUC、F1、校准。结果只支持对规则观测标签的可预测性描述。Semantic覆盖很窄；High-confidence Dependency标签不足，指标为NA。当前相关性与词面标注共享特征，需独立人工标签防止循环论证。

## 6. 是否足以指导 KV 召回、分层存储和主动回收？

可以用于生成后续策略验证候选：长期再用片段测试召回/分层；有替代证据的片段测试主动回收；Uncertain作为难例和保留对照。暂不能据此制定安全删除阈值或承诺性能收益。两条示例仅5张KV快照，selected_fraction不是因果重要性；尚无150轨迹的KV级结论。

## 后续验证与交付

准备了 {len(candidates)} 个删除验证候选（每类最多12个不同任务），含可追溯字符坐标、检查点与来源。`counterfactual-design.json` 明确 Full / Remove / Retain 三组定义、相同环境与种子、重新推理与prefill、独立验收、执行行为和token/TTFT/latency/recompute成本；没有用原输出文本差异冒充删除实验。

数据完整性前后检查通过；{audit['unit_span_checks']:,} 个片段全部原文重建通过；没有人工确认标签，也没有删除因果实验。完整代码、CSV、JSONL、图片与SHA-256清单均在本目录，复跑见 README.md。

用户贴文的最终问题列表在第3条中途截断；本总结依据前文完整研究目标补充回答跨Seed、预测与KV指导问题。
''')
    save('README.md',f'''# DuoKV 可重复离线分析

先读 [experiment_summary.md](experiment_summary.md)。本目录为研究输出，原包只读；代码在 `code/`，主仓库实现位于 `experiments/duokv_observation/`。

## 复跑

需要 Python 3.12+、numpy 与 matplotlib（精确版本见 code/requirements.txt）。分析无需 GPU、模型、tokenizer、网络或外部标注服务。

```bash
python3 -m venv /tmp/duokv-env
/tmp/duokv-env/bin/pip install -r code/requirements.txt
MPLCONFIGDIR=/tmp/duokv-mpl /tmp/duokv-env/bin/python code/run_study.py \\
  --bundle '{config['bundle']}' \\
  --output /tmp/duokv-reproduction
/tmp/duokv-env/bin/python -m unittest discover -s code -p 'test_*.py' -v
/tmp/duokv-env/bin/python code/validate_output.py --output /tmp/duokv-reproduction --write-audit
```

依赖安装可能需要联网；安装完成后分析完全离线。当前机器运行使用 `/tmp/duokv-study-env/bin/python`，依赖从已有本地缓存复制，未改系统环境。输出目录必须与输入分离；只允许写空目录或已有 study_config.json 的本实验输出。固定种子和任务分组，结构化统计可重复；图片文件元信息不作为数值一致性的依据。

## 数据与口径

- `structured/units.jsonl`：信息片段、原始消息索引、Unicode字符区间、JSON转义映射、任务/seed/工具、text SHA-256。
- `structured/reuse-evidence.jsonl` 与 `retry-reuse-evidence.jsonl`：分层复用候选、双方证据、目标Step、来源歧义。
- `structured/events.jsonl` 与 `event-source-relations.jsonl`：可验证事件、替代关系、Uncertain、残留与后续复用。
- `structured/executor-audit-unmapped.jsonl`：原始session审计，不虚构轮次归属。
- `future-reuse-statistics.csv`、`unit-reuse-statistics.csv`、`reuse-gaps.csv`、`future-demand-survival.csv`、`dormant-future-demand.csv`：需求与删失统计。
- `sensitivity-statistics.csv`、`long-range-sensitivity.csv`：Retry、fresh restart、不完整轨迹和阈值敏感性。
- `supersession-statistics.csv`、`event-control-comparison.csv`、`value-evolution-examples.jsonl`：事件关系与对照。
- `cross-seed-demand.csv`、`cross-seed-task-statistics.csv`：对齐代理与任务聚合；缺失seed不作负例。
- `future-demand-prediction.csv`、`baseline-comparison.csv`、`prediction-calibration.csv` 及 structured/prediction-*：分组评估、逐样本特征与校准。
- `kv-snapshot-correlation.csv`、`kv-correlation-summary.csv`：仅两条示例的粗粒度映射与相关。
- `counterfactual-candidates.jsonl`、`counterfactual-design.json`：第二阶段候选与接口方案，未执行推理。
- `figures/`：六幅PNG；各报告包含图表引用。
- `output-manifest.json`：输出文件SHA-256；`study_config.json` 与 `structured/analysis-audit.json`：参数及校验。

所有指标的公式、分母、排除规则与局限分别在各报告给出。所有自动标签均为候选；高置信依赖、真衰减时刻、真实语义残留无可靠标注时保持NA/null。人工审阅可按 edge_id/relation_id 另建 reviewed_label、reviewer、evidence、adjudication 字段，保留原候选与未审记录；本次没有把自动分类当人工Ground Truth。

## 查回原文

用 unit_id 在 units.jsonl 查到 trace_path、trace_line、field_pointer 和 character_span；读取包内对应JSONL行，按JSON Pointer取字符串并截取 [start,end)。若 encoding=json_string_interior，再解码JSON字符串转义即可得到完整text；code/core.py 的 verify_unit 实现了逐条校验。原包工具也可直接显示指定turn：

```bash
python3 '{config['bundle']}/tools/read_trace.py' --family deepswe --seed 1001 --task wasmi-trap-coredumps --from-turn 3 --to-turn 3
```

## 覆盖边界

不是全语义自动理解：切分为结构代理、词面复用有误检/漏检、来源使用最近匹配归因；任务结束删失可能信息性；事件对照非随机；训练目标与相似度共享特征；KV映射缺精确字符/token对齐。详见七份报告。第二阶段需环境快照、模型/分词器和独立验收，不能仅靠本包完成因果重跑。
''')
