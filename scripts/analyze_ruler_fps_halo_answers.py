#!/usr/bin/env python3
"""CPU-only answer and budgeted-group audit for the completed FPS halo cohort."""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import analyze_ruler_fps_answers as fps
import ruler_protocol as protocol
from analyze_ruler_overlap_answers import read,sha,require,verify_files,span_coverage,distribution,ranks

_spec=importlib.util.spec_from_file_location('halo_analysis_controller',Path(__file__).with_name('test_ruler_fps_halo_answers.py'))
campaign=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(campaign)
ARM=campaign.ARM
REFERENCES=(*fps.REFERENCES,campaign.SCORE_MODE)
DEFAULT_ROOT=campaign.DEFAULT_ROOT
SCOPE=fps.SCOPE


def completed_gate(root):
    status=read(root/'status.json');arm_status=read(root/ARM/'status.json')
    require(status.get('status')=='completed' and status.get('cases')==12 and
        arm_status.get('status')=='completed' and arm_status.get('cases')==12,
        'Final halo analysis requires all twelve completed cases')
    audit=read(root/'execution-audit.json')
    required=('unchanged_frozen_runtime','original_inputs_seeds_sampling_verified','native_picker_replay_verified',
        'native_group_picker_replay_verified','independent_group_geometry_verified','official_scores_recomputed',
        'no_post_picker_neighbors','before_picker_expansion','pressure_cumulative_attention',
        'gpu_cleanup_verified','reference_answers_unchanged')
    require(audit.get('status')=='passed' and audit.get('cases')==12 and all(audit.get(k) is True for k in required) and
        audit.get('guided') is False and audit.get('graph_alpha')==0 and audit.get('propagation_applied') is False and
        audit.get('query_score_mode')==campaign.SCORE_MODE and audit.get('query_fps_halo_blocks')==1,
        'Final halo execution audit has not passed')
    cleanup=read(root/ARM/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],'GPU cleanup incomplete')
    return {name:sha(root/name) for name in ('status.json','execution-audit.json',ARM+'/cleanup.json')}


def selection_details(inputs,ranges,target):
    actual=fps.describe_selection(inputs,ranges,target)
    group=inputs['query_group_selection'];counts=dict(zip(inputs['starts'],inputs['counts']))
    baseline_ranges=[(start,counts[start]) for start in group['baseline_selected_starts']]
    baseline=fps.describe_selection(inputs,baseline_ranges,target)
    ranking=ranks(inputs['raw_scores'],inputs['score_mask'])
    ranks_by_start=dict(zip(inputs['starts'],ranking))
    selected=set(group['selected_starts'])
    covered=set(group['covered_anchor_starts'])
    for block in actual['target_blocks']:
        block['supporting_covered_groups']=[dict(anchor=anchor,anchor_score=score,
            anchor_rank=ranks_by_start[anchor],members=members)
            for anchor,score,members in zip(group['anchor_starts'],group['group_scores'],group['group_members'])
            if block['start'] in members and anchor in covered]
    require(selected=={start for start,_ in ranges},'Group evidence and selected ranges differ')
    return dict(actual=actual,same_seed_block_picker=baseline,
        group_summary={key:value for key,value in group.items() if key not in ('groups','group_scores','group_members','anchor_starts')},
        added_value_blocks=[b['start'] for b in actual['target_blocks'] if b['value_tokens'] and b['start'] in group['added_starts']],
        displaced_value_blocks=[b['start'] for b in baseline['target_blocks'] if b['value_tokens'] and b['start'] in group['displaced_starts']],
        note='Same-seed block picker is a selection-only replay, not a generated answer. Covered anchors mean '
            'their entire group lies in the final selected set, not an explicit picker-choice log. '
            'A selected member score/cutoff is not the group-admission threshold.')


def seed_comparison(old_inputs,new_inputs):
    aligned=all(old_inputs[key]==new_inputs[key] for key in
        ('starts','counts','score_mask','transport_mask','mandatory_starts','budget'))
    delta=[abs(float(a)-float(b)) for a,b in zip(old_inputs['raw_scores'],new_inputs['raw_scores'])] if aligned else None
    return dict(index_and_masks_equal=aligned,raw_seeds_exactly_equal=aligned and
        old_inputs['raw_scores']==new_inputs['raw_scores'],max_abs_difference=max(delta) if delta else None,
        note='Historical and current runs may have different captured operands; group-only selection replay uses current seeds.')


def analyze_case(root,row,target,fixture,profile,source,old_fixture,old_profile,old_snapshots,scorer):
    case=root/ARM/'cases'/row['id'];manifest=read(case/'hashes.json')
    require(set((*campaign.CASE_FILES,'query-inputs.jsonl')).issubset(manifest),'Halo case evidence incomplete')
    verify_files(case,manifest)
    request,response,result,evidence=(read(case/name) for name in campaign.CASE_FILES)
    fps.validate_request(request,row,fixture)
    require(request['profile_sha256']==sha(root/ARM/'profile.json'),'Case profile changed')
    answer=fps.checked_answer(result,response,row,fixture,scorer)
    queries=[q for q in evidence['bounded'] if q['selection']=='query']
    graphs=[g for g in evidence['graph'] if g['phase']=='query'];inputs=evidence['query_inputs']
    require(len(queries)==len(graphs)==len(inputs)==1,'Expected exactly one public Query and group export')
    query=queries[0]
    require(query['query_span']==row['served_query'] and all(q['guided_query_tokens']==0 for q in evidence['bounded']),
        'Public Query or guidance changed')
    exported=[json.loads(line) for line in (case/'query-inputs.jsonl').read_text().splitlines() if line.strip()]
    require(exported==inputs,'Standalone group export differs')
    pressure=[q for q in evidence['bounded'] if q['selection']=='pressure']
    pressure_graph=[g for g in evidence['graph'] if g['phase']=='pressure']
    require(len(pressure)==len(pressure_graph)==result['pressure_count']>0 and
        all(g['seed_source']=='prefill' and g['pressure_policy']=='attention' and not g['propagation_applied'] for g in pressure_graph),
        'Prefill cumulative attention changed')
    replay=campaign.audit_selection(query,graphs[0],inputs[0],profile)
    recorded=deepcopy(result['selection_audit'])
    replay['propagation'].pop('scoring_seconds',None);recorded['propagation'].pop('scoring_seconds',None)
    require(replay==recorded,'Final native group replay differs from saved audit')
    selection=selection_details(inputs[0],query['selected_history_ranges'],target)
    require(result['value_covered']==(selection['actual']['value']['state']=='full'),'Saved coverage differs')
    previous=fps.analyze_case(source,row,target,old_fixture,old_profile,old_snapshots,scorer)
    controls=deepcopy(previous['controls'])
    controls[campaign.SCORE_MODE]={key:previous[key] for key in ('correct','official_case_score','answer',
        'thinking_complete','completion_tokens','request_seconds','finish_reason')}
    controls[campaign.SCORE_MODE].update(key=previous['selection']['key'],value=previous['selection']['value'],
        selection=previous['selection'],cache_stats=previous['cache_stats'],evidence_sha256=previous['evidence_sha256'])
    old_inputs=read(source/campaign.SCORE_MODE/'cases'/row['id']/'evidence.json')['query_inputs'][0]
    old_query=next(q for q in read(source/campaign.SCORE_MODE/'cases'/row['id']/'evidence.json')['bounded'] if q['selection']=='query')
    old_starts={start for start,_ in old_query['selected_history_ranges']}
    new_starts={start for start,_ in query['selected_history_ranges']}
    actual=selection['actual']['value']
    outcome=actual['state']+'_value_answer_'+('correct' if answer['correct'] else 'incorrect')
    stats=query['query_key_fps_scoring']
    for key in ('cache_build_seconds','scoring_seconds','seconds','cache_bytes','cache_capacity_bytes','cache_hit_blocks','cache_rebuilt_blocks'):
        require(math.isfinite(stats[key]) and stats[key]>=0,'Invalid FPS timing/cache stats')
    require(stats['cache_bytes']<=stats['cache_capacity_bytes'],'FPS cache budget exceeded')
    return dict(id=row['id'],expected=row['outputs'],input_tokens=row['served_input_tokens'],
        value_span=target['token_span'],key_span=target['key_token_span'],**answer,outcome=outcome,
        selection=selection,controls=controls,cache_stats=stats,
        query_refresh_seconds=query['refresh_finished_at']-query['refresh_started_at'],pressure_count=len(pressure),
        historical_fps_change=dict(added_starts=sorted(new_starts-old_starts),displaced_starts=sorted(old_starts-new_starts),
            seeds=seed_comparison(old_inputs,inputs[0])),evidence_sha256=manifest,selection_audit=replay)


def summarize(rows):
    arms={}
    for arm in (*REFERENCES,ARM):
        entries=rows if arm==ARM else [row['controls'][arm] for row in rows]
        coverage=[row['selection']['actual'] for row in rows] if arm==ARM else entries
        arms[arm]=dict(cases=len(rows),correct=sum(e['correct'] for e in entries),
            mean_case_score=statistics.mean(e['official_case_score'] for e in entries),
            full_key_coverage=sum(e['key']['state']=='full' for e in coverage),
            full_value_coverage=sum(e['value']['state']=='full' for e in coverage),
            completion_tokens=sum(e['completion_tokens'] for e in entries),
            request_seconds=distribution(e['request_seconds'] for e in entries))
    changes={arm:dict(improved=[r['id'] for r in rows if r['correct'] and not r['controls'][arm]['correct']],
        regressed=[r['id'] for r in rows if not r['correct'] and r['controls'][arm]['correct']]) for arm in REFERENCES}
    costs={}
    for arm in (campaign.SCORE_MODE,ARM):
        stats=[row['cache_stats'] if arm==ARM else row['controls'][arm]['cache_stats'] for row in rows]
        costs[arm]={field:distribution(item[field] for item in stats)
            for field in ('cache_build_seconds','scoring_seconds','seconds','cache_bytes')}
    return dict(arms=arms,answer_changes=changes,costs=costs,
        same_seed_block_picker_full_value=sum(r['selection']['same_seed_block_picker']['value']['state']=='full' for r in rows),
        actual_group_full_value=sum(r['selection']['actual']['value']['state']=='full' for r in rows),
        correct_without_full_value=[r['id'] for r in rows if r['correct'] and r['selection']['actual']['value']['state']!='full'],
        full_value_but_wrong=[r['id'] for r in rows if not r['correct'] and r['selection']['actual']['value']['state']=='full'],
        added_blocks_same_seed=sum(len(r['selection']['group_summary']['added_starts']) for r in rows),
        displaced_blocks_same_seed=sum(len(r['selection']['group_summary']['displaced_starts']) for r in rows),
        cache_hit_blocks=sum(r['cache_stats']['cache_hit_blocks'] for r in rows),
        cache_rebuilt_blocks=sum(r['cache_stats']['cache_rebuilt_blocks'] for r in rows),
        charged_tokens=distribution(r['selection_audit']['charged_block_tokens'] for r in rows),
        query_refresh_seconds=distribution(r['query_refresh_seconds'] for r in rows),
        timing_limit='Recorded host intervals include CPU archive/cache work and packing/H2D/GPU/synchronization. '
            'Different generation lengths, CPU load and captures prevent attributing all wall-time changes to FPS optimization.')


def build_analysis(root,*,completed_only=False):
    root=root.resolve();completion=None if completed_only else completed_gate(root)
    fixture,profile,targets=read(root/'fixture.json'),read(root/ARM/'profile.json'),read(root/'target-spans.json')
    require(tuple(row['id'] for row in fixture['rows'])==campaign.CASE_IDS and fixture['query_fps_halo_blocks']==1,
        'Original halo cohort changed')
    verify_files(root,read(root/'prepared-proof.json'))
    proof=read(root/'reference-proof.json');source=Path(proof['root'])
    require(str(source)==fixture['fps_reference'],'FPS reference binding differs')
    verify_files(source,proof['files'])
    verified,snapshots=campaign.reference_proof(source,fixture,profile)
    require(verified['files']==proof['files'] and verified['profile_difference']==proof['profile_difference'] and
        snapshots==read(root/'reference-answers.json'),'Frozen previous answers or parameters changed')
    old_fixture=read(source/'fixture.json');old_profile=read(source/campaign.SCORE_MODE/'profile.json')
    old_snapshots=read(source/'reference-answers.json')
    scorer=protocol.load_scorer(Path(fixture['upstream_path']));rows=[]
    for row in fixture['rows']:
        if completed_only and not (root/ARM/'cases'/row['id']/'hashes.json').exists():continue
        require(sha(Path(fixture['input_directory'])/(row['id']+'.i32'))==row['served_sha256'],'Original input changed')
        rows.append(analyze_case(root,row,targets[row['id']],fixture,profile,source,old_fixture,old_profile,old_snapshots,scorer))
    if not completed_only:
        require(len(rows)==12,'Expected all twelve answers')
        scores=read(root/ARM/'scores.json')
        require(scores['cases']==12 and math.isclose(scores['mean_case_score'],statistics.mean(r['official_case_score'] for r in rows),abs_tol=1e-12) and
            scores['full_value_coverage']==sum(r['selection']['actual']['value']['state']=='full' for r in rows),'Final scores differ')
        verify_files(root/'runtime',read(root/'runtime-proof.json')['files'])
    native=fps.native_reference(fixture,Path(fixture['overlap_reference']),scorer)
    return dict(version=1,status='inspection' if completed_only else 'completed',at=time.time(),root=str(root),scope=SCOPE,
        query_score_mode=campaign.SCORE_MODE,query_fps_halo_blocks=1,query_fps_optimized=fixture['query_fps_optimized'],
        graph_alpha=0,propagation_applied=False,atomic_groups=True,post_picker_expansion=False,
        profile_difference=proof['profile_difference'],reference_answers_unchanged=True,completion_audit=completion,
        generation_performed_by_analysis=False,external_judge=False,native_full_context=native,
        comparison_limit=fixture['comparison_limit'],source_sha256={str(Path(p).resolve()):sha(Path(p)) for p in
            (__file__,campaign.__file__,fps.__file__)},
        evidence_sha256={name:sha(root/name) for name in ('fixture.json','prepared-proof.json','runtime-proof.json',
            'reference-proof.json','reference-answers.json')},summary=summarize(rows) if rows else None,cases=rows)


def format_report(analysis):
    rows,summary=analysis['cases'],analysis['summary'];n=len(rows)
    names={'mean_k':'MeanK','token_sum':'逐 token 页求和','overlap_16_8':'16/8 窗口',
        'key_fps_4':'FPS-4 原单块召回',ARM:'FPS-4＋预算内左右一块原子组'}
    short=lambda ids:'、'.join(i.rsplit('-',1)[-1] for i in ids) or '无'
    lines=['# FPS-4 原子邻块组：实际回答与预算核验','',SCOPE,'',
        '保留完整原始输入、每题种子/采样、32-token block、K=36,864、G=16,384、sink=1,312、recent=0、'
        'NVFP4/FP8 KV/MTP4、单路串行、chunk=2,048。prefill 仍累计 attention，Query 图传播关闭。'
        '每个 anchor 构造同一连续历史/消息内左一块＋自身＋右一块，组分数仅是 anchor 原 FPS 分数；'
        '组在预算内整体选取，共享 block 只计一次。选完不追加邻块。','',
        f"本次 query_fps_optimized={str(analysis['query_fps_optimized']).lower()}；这是明确冻结的实现选择。"
        '原始种子与历史轮次若有差异会逐题记录，不把所有变化归因于分组或成本优化。','',
        '| 配置 | 实际答对 | key 完整召回 | value 完整召回 | 平均分 |','|---|---:|---:|---:|---:|']
    for arm,item in summary['arms'].items():
        lines.append(f"| {names[arm]} | {item['correct']}/{n} | {item['full_key_coverage']}/{n} | {item['full_value_coverage']}/{n} | {item['mean_case_score']:.2f} |")
    native=analysis['native_full_context']
    if native:lines+=['',f"历史完整上下文原生 vLLM 同12题实际答对 {native['correct']}/12，原答案和输入哈希已核验；并发不同，仅作质量参照。"]
    change=summary['answer_changes']['key_fps_4']
    lines+=['',f"相对上一轮 FPS-4：改善 {short(change['improved'])}；退步 {short(change['regressed'])}。",'',
        '| 题号 | 旧 FPS 答案 | 本次答案 | 旧→新 value 覆盖 token | 同种子单块→原子组完整覆盖 | 换入/换出 block（同种子） |',
        '|---|---|---|---|---|---:|']
    for row in rows:
        old=row['controls']['key_fps_4'];sel=row['selection'];v=sel['actual']['value'];ov=old['value'];g=sel['group_summary']
        mark=lambda ok:'对' if ok else '错'
        lines.append(f"| {row['id'].rsplit('-',1)[-1]} | {mark(old['correct'])} | {mark(row['correct'])} | "
            f"{ov['selected_tokens']}/{ov['tokens']}→{v['selected_tokens']}/{v['tokens']} | "
            f"{sel['same_seed_block_picker']['value']['state']}→{v['state']} | {len(g['added_starts'])}/{len(g['displaced_starts'])} |")
    lines+=['','同种子单块对照仅在本次真实种子上重放选页，没有生成另一组答案。'
        'covered anchors 表示最终集合完整包含该组，不是 picker 显式选择该 anchor 的日志。'
        '成员页原始分数低也可随高分 anchor 入选，成员最低分不作为组录取阈值。','',
        f"答对但目标值不完整：{short(summary['correct_without_full_value'])}；完整召回但答错：{short(summary['full_value_but_wrong'])}。"
        '回答正确率与完整召回率分别计算，不能互相替代。']
    for row in rows:
        value=row['selection']['actual']['value']
        if row['correct'] and value['state']!='full':
            lines+=['',f"{row['id'].rsplit('-',1)[-1]}仅保留{value['selected_tokens']}/{value['tokens']}个目标值token，"
                '计入答对，不计入完整召回；原因不能仅从答案推断。']
    lines+=['','| 成本区间 | 旧 FPS 中位数/P90（秒） | 本次中位数/P90（秒） |','|---|---:|---:|']
    for field,label in (('cache_build_seconds','历史K读取＋FPS缓存构建'),('scoring_seconds','打包/H2D/GPU评分/同步'),('seconds','完整FPS区间')):
        old,new=summary['costs']['key_fps_4'][field],summary['costs'][ARM][field]
        lines.append(f"| {label} | {old['median']:.2f}/{old['p90']:.2f} | {new['median']:.2f}/{new['p90']:.2f} |")
    lines+=['','这些是主机计时区间，不是纯GPU内核耗时。旧组曾与CPU诊断部分重叠；整请求还受不同生成长度影响，'
        '不能把端到端时间差全部归因于FPS优化。冷索引和首次Query成本与热缓存复用应分开判断。','',
        '| 配置 | 整请求中位数/P90（秒） | 总生成token | 缓存中位/峰值（GiB） |','|---|---:|---:|---:|']
    for arm in ('key_fps_4',ARM):
        item=summary['arms'][arm];size=summary['costs'][arm]['cache_bytes']
        lines.append(f"| {names[arm]} | {item['request_seconds']['median']:.2f}/{item['request_seconds']['p90']:.2f} | "
            f"{item['completion_tokens']} | {size['median']/1024**3:.3f}/{size['max']/1024**3:.3f} |")
    lines+=['',f"本次缓存命中{summary['cache_hit_blocks']}个block，重建{summary['cache_rebuilt_blocks']}个block。"
        f"实际最大预算计费{summary['charged_tokens']['max']:.0f} tokens，原K预算未扩张。",'',
        '所有旧答案和新请求/响应/结果/证据哈希已核验；独立按原逻辑位置、空洞、完整消息归属重建组，'
        '逐项检查anchor分数，重放原生组picker、固定保留、原32-token预算和同种子单块baseline。'
        '最终执行审计与GPU清理通过。各题目标页排名、支持它的完整覆盖组、历史FPS换页差异、答案与计时详见 [analysis.json](analysis.json)。','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--inspect-completed',action='store_true')
    args=parser.parse_args(argv);analysis=build_analysis(args.root,completed_only=args.inspect_completed)
    if args.inspect_completed:print(json.dumps(analysis,ensure_ascii=False,allow_nan=False));return
    root=args.root.resolve();out=root/'analysis.json';report=root/'report.md'
    require(not out.exists() and not report.exists(),'Final analysis already exists; do not overwrite')
    text=format_report(analysis)
    with out.open('x') as destination:destination.write(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with report.open('x') as destination:destination.write(text)
    print(json.dumps(dict(cases=len(analysis['cases']),summary=analysis['summary'],analysis=str(out),report=str(report)),ensure_ascii=False))


if __name__=='__main__':main()
