#!/usr/bin/env python3
"""Read-only FPS-4 answer analysis; publish only after all twelve audited cases.

No model generation, GPU operations or external judge. Historical answers are
read from the frozen reference snapshots, not inferred from retrieval coverage.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

from analyze_ruler_overlap_answers import read, sha, require, verify_files, span_coverage, ranks, distribution
import analyze_block_graph as offline
import ruler_protocol as protocol
from vllm_kvmem.graph_scoring import score_graph

# Import by absolute filename: pytest has a test module with the same basename.
_CONTROLLER = Path(__file__).with_name('test_ruler_fps_answers.py')
_spec = importlib.util.spec_from_file_location('fps_campaign_analysis_controller', _CONTROLLER)
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)
ARM = campaign.ARM
REFERENCES = ('mean_k', 'token_sum', 'overlap_16_8')
DEFAULT_ROOT = campaign.DEFAULT_ROOT
DEFAULT_FIXED_CAPTURE = Path(__file__).resolve().parents[1]/'artifacts/query-key-fps-fixed-operands-20261023-r1'
SCOPE = '原先 9 道 MeanK 失败题＋3 道成功对照，共 12 题；不是完整 RULER 准确率。'
TIMING_LIMIT = ('这是尚未优化的 lazy 冷索引质量实验。cache_build_seconds 包含 CPU 容量检查、'
    '历史 K 读取、FPS 与发布；scoring_seconds 是包含 CPU 打包、H2D、GPU 评分、同步与调度的主机计时区间，'
    '并非纯 GPU 内核时间。该模式仍默认关闭。'
    '整请求时间还包含 prefill、重选与生成，输出长度变化需单列；不能由点积数量或这些区间直接推算吞吐提升。')


def completed_gate(root):
    status = read(root/'status.json')
    require(status.get('status') == 'completed' and status.get('cases') == 12,
        'Final analysis requires all twelve completed cases')
    arm_status = read(root/ARM/'status.json')
    require(arm_status.get('status') == 'completed' and arm_status.get('cases') == 12,
        'FPS arm is not complete')
    audit = read(root/'execution-audit.json')
    required = ('unchanged_frozen_runtime', 'original_inputs_seeds_sampling_verified',
        'native_picker_replay_verified', 'official_scores_recomputed', 'no_post_picker_neighbors',
        'pressure_cumulative_attention', 'gpu_cleanup_verified', 'reference_answers_unchanged',
        'overlap_reference_unchanged')
    require(audit.get('status') == 'passed' and audit.get('cases') == 12 and
        all(audit.get(name) is True for name in required) and audit.get('guided') is False and
        audit.get('graph_alpha') == 0 and audit.get('propagation_applied') is False and
        audit.get('query_score_mode') == ARM, 'Final execution audit has not passed the FPS contract')
    cleanup = read(root/ARM/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'GPU cleanup incomplete')
    return dict(execution_audit_sha256=sha(root/'execution-audit.json'),
        cleanup_sha256=sha(root/ARM/'cleanup.json'), status_sha256=sha(root/'status.json'))


def checked_answer(result, response, row, fixture, scorer):
    answer = protocol.final_answer(response['text'], starts_in_thinking=fixture['starts_in_thinking'])
    scoring = scorer[0][protocol.TASK_FAMILIES[row['task']]]
    score = scoring['metric_fn']([scorer[1](answer['text'], scoring)], [row['outputs']])
    require(result['official_case_score'] == score and result['answer'] == answer['text'] and
        result['thinking_complete'] == answer['thinking_complete'], 'Saved answer differs from official scoring')
    require(result['completion_tokens'] == len(response['output_ids']) and
        result['seconds'] == response['seconds'], 'Saved answer token count or timing differs')
    return dict(correct=score == 100, official_case_score=score, answer=answer['text'],
        thinking_complete=answer['thinking_complete'], completion_tokens=len(response['output_ids']),
        request_seconds=response['seconds'], finish_reason=response['finish_reason'])


def validate_request(request, row, fixture):
    require(request['sampling'] == dict(fixture['sampling'], seed=row['seed']) and
        request['input_sha256'] == row['served_sha256'], 'Original input, seed or sampling changed')
    if 'query_span' in request:
        require(request['query_span'] == row['served_query'], 'Original public Query span changed')


def describe_selection(inputs, ranges, target):
    """Describe actual alpha-zero scores; callers separately audit each scorer."""
    starts, counts, raw = inputs['starts'], inputs['counts'], inputs['raw_scores']
    selected = {begin for begin, _ in ranges}
    mandatory = set(inputs['mandatory_starts'])
    eligible = [bool(allowed) and start not in mandatory for start, allowed in zip(starts, inputs['score_mask'])]
    ranking = ranks(raw, eligible)
    optional_selected = [i for i, start in enumerate(starts) if start in selected and eligible[i]]
    cutoff = min((raw[i] for i in optional_selected), default=None)
    cutoff_rank = max((ranking[i] for i in optional_selected), default=None)
    blocks = []
    for i,(start,count) in enumerate(zip(starts,counts)):
        value = max(0, min(start+count, target['token_span'][1])-max(start, target['token_span'][0]))
        key = max(0, min(start+count, target['key_token_span'][1])-max(start, target['key_token_span'][0]))
        if value or key:
            blocks.append(dict(start=start, count=count, value_tokens=value, key_tokens=key,
                optional_rank=ranking[i], score=raw[i], selected=start in selected,
                score_allowed=bool(inputs['score_mask'][i]), transport_allowed=bool(inputs['transport_mask'][i]),
                mandatory=start in mandatory, cutoff_ratio=raw[i]/cutoff if cutoff and eligible[i] else None))
    require(sum(block['value_tokens'] for block in blocks) == target['token_span'][1]-target['token_span'][0] and
        sum(block['key_tokens'] for block in blocks) == target['key_token_span'][1]-target['key_token_span'][0],
        'Target key/value not fully present in accepted history')
    return dict(key=span_coverage(ranges,target['key_token_span']),
        value=span_coverage(ranges,target['token_span']), target_blocks=blocks,
        optional_cutoff=cutoff, optional_cutoff_rank=cutoff_rank,
        optional_selected_blocks=len(optional_selected), optional_candidates=sum(eligible),
        selected_tokens=sum(count for _,count in ranges), selected_blocks=len(selected),
        selected_nonmandatory_unscored=[start for start,allowed in zip(starts,eligible)
            if start in selected and start not in mandatory and not allowed],
        rank_definition='1 + strictly higher raw scores among score_mask blocks excluding mandatory starts; '
            'ties share rank. Cutoff is the lowest score actually picked in that same set. Native picker decides inclusion.')


def verify_snapshots(root,fixture,profile):
    proofs = [read(root/name) for name in ('reference-proof.json','overlap-reference-proof.json')]
    require(proofs[0]['root'] == fixture['answer_reference'] and
        proofs[1]['root'] == fixture['overlap_reference'], 'Reference binding changed')
    snapshots = read(root/'reference-answers.json')
    require(set(snapshots) == set(REFERENCES), 'Expected all three frozen answer references')
    for proof in proofs:
        verify_files(Path(proof['root']),proof['files'])
    previous_profile = read(Path(proofs[1]['root'])/'overlap_16_8/profile.json')
    require(profile == campaign.profile_for_fps(previous_profile), 'Paired execution profile changed')
    delta = campaign.profile_difference(previous_profile,profile)
    require(delta == proofs[1]['profile_difference'], 'Prepared profile difference changed')
    for arm in REFERENCES:
        require(set(snapshots[arm]) == {row['id'] for row in fixture['rows']}, 'Reference cohort changed')
        proof = proofs[1] if arm == 'overlap_16_8' else proofs[0]
        for row in fixture['rows']:
            saved = snapshots[arm][row['id']]
            case = Path(proof['root'])/arm/'cases'/row['id']
            require(Path(saved['root']) == case, 'Reference answer source changed')
            for name,key in (('response.json','response'),('result.json','result')):
                relative = str((case/name).relative_to(proof['root']))
                require(proof['files'][relative] == saved['sha256'][name] == sha(case/name) and
                    read(case/name) == saved[key], 'Frozen answer snapshot differs from original evidence')
    return snapshots,proofs,delta


def control_answer(arm,snapshot,row,fixture,target,scorer):
    entry = checked_answer(snapshot['result'],snapshot['response'],row,fixture,scorer)
    case = Path(snapshot['root'])
    request,evidence = read(case/'request.json'),read(case/'evidence.json')
    validate_request(request,row,fixture)
    queries = [q for q in evidence['bounded'] if q['selection'] == 'query']
    require(len(queries) == 1 and queries[0]['query_span'] == row['served_query'], 'Reference public Query differs')
    query = queries[0]
    expected_method = 'content_token_softmax_page_sum' if arm == 'token_sum' else arm
    require(query['query_scoring_method'] == expected_method and query['guided_query_tokens'] == 0,
        'Reference scorer or guidance changed')
    graph = [g for g in evidence['graph'] if g['phase'] == 'query']
    require(len(graph) == 1 and graph[0]['alpha'] == 0, 'Reference unexpectedly used graph propagation')
    ranges = query['selected_history_ranges']
    entry.update(key=span_coverage(ranges,target['key_token_span']),
        value=span_coverage(ranges,target['token_span']), original_sha256=snapshot['sha256'])
    require(snapshot['result']['value_covered'] == (entry['value']['state'] == 'full'),
        'Reference target coverage differs')
    if arm == 'overlap_16_8':
        inputs = evidence['query_inputs']
        require(len(inputs) == 1 and inputs[0]['seed_source'] == arm and inputs[0]['alpha'] == 0,
            'Reference overlap scores differ')
        entry['selection'] = describe_selection(inputs[0],ranges,target)
        entry['window_stats'] = query['query_window_scoring']
    return entry


def analyze_case(root,row,target,fixture,profile,snapshots,scorer):
    case = root/ARM/'cases'/row['id']
    manifest = read(case/'hashes.json')
    require(set((*campaign.CASE_FILES,'query-inputs.jsonl')).issubset(manifest), 'Case evidence is incomplete')
    verify_files(case,manifest)
    request,response,result,evidence = (read(case/name) for name in campaign.CASE_FILES)
    validate_request(request,row,fixture)
    require(request['profile_sha256'] == sha(root/ARM/'profile.json'), 'Case used another profile')
    entry = checked_answer(result,response,row,fixture,scorer)
    queries = [q for q in evidence['bounded'] if q['selection'] == 'query']
    graphs = [g for g in evidence['graph'] if g['phase'] == 'query']
    inputs = evidence['query_inputs']
    require(len(queries) == len(graphs) == len(inputs) == 1, 'Expected one public Query and complete graph export')
    exported = [json.loads(line) for line in (case/'query-inputs.jsonl').read_text().splitlines() if line.strip()]
    require(exported == inputs, 'Standalone Query export differs')
    query = queries[0]
    require(query['query_span'] == row['served_query'] and
        all(q['guided_query_tokens'] == 0 for q in evidence['bounded']), 'Public Query or guidance differs')
    pressure = [q for q in evidence['bounded'] if q['selection'] == 'pressure']
    pressure_graph = [g for g in evidence['graph'] if g['phase'] == 'pressure']
    require(len(pressure) == result['pressure_count'] == len(pressure_graph) > 0 and
        all(g['seed_source'] == 'prefill' and g['pressure_policy'] == 'attention' and
            not g['propagation_applied'] for g in pressure_graph), 'Prefill cumulative attention differs')
    replay = campaign.audit_selection(query,graphs[0],inputs[0],profile)
    recorded = deepcopy(result['selection_audit'])
    replay['propagation'].pop('scoring_seconds',None); recorded['propagation'].pop('scoring_seconds',None)
    require(replay == recorded, 'Saved selection audit differs from native replay')
    selection = describe_selection(inputs[0],query['selected_history_ranges'],target)
    require(result['value_covered'] == (selection['value']['state'] == 'full'), 'Saved value coverage differs')
    controls = {arm:control_answer(arm,snapshots[arm][row['id']],row,fixture,target,scorer) for arm in REFERENCES}
    stats = query['query_key_fps_scoring']
    for key in ('cache_build_seconds','scoring_seconds','seconds','cache_bytes','cache_capacity_bytes',
            'cache_hit_blocks','cache_rebuilt_blocks'):
        require(math.isfinite(stats[key]) and stats[key] >= 0, 'Invalid FPS timing/cache statistics')
    require(stats['cache_bytes'] <= stats['cache_capacity_bytes'] and
        stats['seconds']+1e-6 >= stats['cache_build_seconds']+stats['scoring_seconds'],
        'FPS timing intervals or cache capacity are inconsistent')
    if selection['value']['state'] == 'full':
        outcome = 'full_value_answer_correct' if entry['correct'] else 'full_value_answer_incorrect'
    else:
        outcome = selection['value']['state']+'_value_answer_'+('correct' if entry['correct'] else 'incorrect')
    return dict(id=row['id'],expected=row['outputs'],input_tokens=row['served_input_tokens'],
        value_span=target['token_span'],key_span=target['key_token_span'],**entry,
        selection=selection,controls=controls,outcome=outcome,cache_stats=stats,
        query_refresh_seconds=query['refresh_finished_at']-query['refresh_started_at'],
        pressure_count=len(pressure),evidence_sha256=manifest,selection_audit=replay)


def native_reference(fixture,overlap_root,scorer):
    path = overlap_root/'native-context-comparison.json'
    if not path.exists():
        return None
    comparison = read(path)
    require(comparison['same_inputs_seeds_sampling'] is True and comparison['cases'] == 12,
        'Native context comparison does not cover the original cohort')
    native = Path(comparison['native_root'])
    require(read(native/'status.json')['status'] == 'completed', 'Historical native context run is incomplete')
    cleanup = read(native/'run/cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'Historical native GPU cleanup incomplete')
    old_rows = {row['id']:row for row in comparison['rows']}
    require(set(old_rows) == {row['id'] for row in fixture['rows']}, 'Native comparison cohort changed')
    checked = []
    for row in fixture['rows']:
        previous = old_rows[row['id']]
        for name,digest in previous['evidence_sha256'].items():
            require(sha(Path(name)) == digest, 'Native comparison source hash changed')
        case = native/'run/answers/cases'/row['id']
        for name in ('request.json','response.json','result.json'):
            require(str(case/name) in previous['evidence_sha256'], 'Native comparison lacks original answer hashes')
        request,response,result = (read(case/name) for name in ('request.json','response.json','result.json'))
        validate_request(request,row,fixture)
        require(request['input_tokens'] == row['served_input_tokens'] == response['usage']['prompt_tokens'],
            'Native answer did not receive the original full input')
        answer = checked_answer(result,response,row,fixture,scorer)
        require(answer['official_case_score'] == previous['native_score'] and
            previous['input_sha256'] == row['served_sha256'] and previous['seed'] == row['seed'],
            'Native context comparison differs from original answer')
        checked.append(dict(id=row['id'],**answer,input_tokens=row['served_input_tokens'],
            evidence_sha256=previous['evidence_sha256']))
    return dict(root=str(native),comparison_path=str(path),comparison_sha256=sha(path),
        status_sha256=sha(native/'status.json'),cleanup_sha256=sha(native/'run/cleanup.json'),
        cases=checked,correct=sum(row['correct'] for row in checked),same_inputs_seeds_sampling=True,
        actual_concurrency=comparison['native_actual_concurrency'],current_actual_concurrency=1,
        limit='Historical full-input actual answers. Native concurrency differs from current serial requests; '
            'not a throughput comparison. Full input delivery is verified, not inferred from retrieval ranks.')


def fixed_capture_reference(directory,case_ids):
    if directory is None or not directory.exists():
        return None
    summaries = []
    for case_id in case_ids:
        path = directory/case_id/'summary.json'
        if not path.exists():
            continue
        item = read(path)
        require(item['id'] == case_id and item['mode'] == ARM and item['graph_alpha'] == 0 and
            item['generation_performed'] is False and item['selection_uses_targets'] is False and
            item['source_stat_unchanged'] is True, 'Invalid fixed-operand FPS diagnostic')
        summaries.append(dict(path=str(path.resolve()),sha256=sha(path),measurement=item))
    if not summaries:
        return None
    return dict(cases=summaries,limit='Fixed captured Q/K selection diagnosis only; no new model answers. '
        'Cold CPU/mmap timings, possibly concurrent with model startup, cannot be compared with production GPU scoring or throughput.')


def summarize(rows):
    arms = {}
    for arm in (*REFERENCES,ARM):
        entries = rows if arm == ARM else [row['controls'][arm] for row in rows]
        coverage = [row['selection'] for row in rows] if arm == ARM else entries
        arms[arm] = dict(cases=len(rows),correct=sum(e['correct'] for e in entries),
            mean_case_score=statistics.mean(e['official_case_score'] for e in entries),
            full_key_coverage=sum(e['key']['state'] == 'full' for e in coverage),
            full_value_coverage=sum(e['value']['state'] == 'full' for e in coverage),
            completion_tokens=sum(e['completion_tokens'] for e in entries),
            request_seconds=distribution(e['request_seconds'] for e in entries))
    changes = {arm:dict(improved=[row['id'] for row in rows if row['correct'] and not row['controls'][arm]['correct']],
        regressed=[row['id'] for row in rows if not row['correct'] and row['controls'][arm]['correct']]) for arm in REFERENCES}
    return dict(arms=arms,answer_changes=changes,
        full_value_but_wrong=[row['id'] for row in rows if not row['correct'] and row['selection']['value']['state']=='full'],
        correct_without_full_value=[row['id'] for row in rows if row['correct'] and row['selection']['value']['state']!='full'],
        cache_build_seconds=distribution(row['cache_stats']['cache_build_seconds'] for row in rows),
        scoring_seconds=distribution(row['cache_stats']['scoring_seconds'] for row in rows),
        combined_fps_seconds=distribution(row['cache_stats']['seconds'] for row in rows),
        query_refresh_seconds=distribution(row['query_refresh_seconds'] for row in rows),
        cache_bytes=distribution(row['cache_stats']['cache_bytes'] for row in rows),
        cache_hit_blocks=sum(row['cache_stats']['cache_hit_blocks'] for row in rows),
        cache_rebuilt_blocks=sum(row['cache_stats']['cache_rebuilt_blocks'] for row in rows),
        input_tokens=sum(row['input_tokens'] for row in rows),timing_limit=TIMING_LIMIT,
        quantile_definition='P90 linearly interpolates at (n-1)*0.9')


def build_analysis(root,*,completed_only=False,fixed_capture=DEFAULT_FIXED_CAPTURE):
    root = root.resolve()
    completion = None if completed_only else completed_gate(root)
    fixture,profile,targets = read(root/'fixture.json'),read(root/ARM/'profile.json'),read(root/'target-spans.json')
    require(tuple(row['id'] for row in fixture['rows']) == campaign.CASE_IDS and
        fixture['graph_alpha'] == 0 and fixture['query_score_mode'] == ARM, 'Frozen FPS cohort or policy differs')
    verify_files(root,read(root/'prepared-proof.json'))
    snapshots,proofs,delta = verify_snapshots(root,fixture,profile)
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    rows = []
    for row in fixture['rows']:
        if completed_only and not (root/ARM/'cases'/row['id']/'hashes.json').exists():
            continue
        require(sha(Path(fixture['input_directory'])/(row['id']+'.i32')) == row['served_sha256'],
            'Original complete input changed')
        rows.append(analyze_case(root,row,targets[row['id']],fixture,profile,snapshots,scorer))
    if not completed_only:
        require(len(rows) == 12, 'Expected all twelve immutable answers')
        scores = read(root/ARM/'scores.json')
        require(scores['cases'] == 12 and math.isclose(scores['mean_case_score'],
            statistics.mean(row['official_case_score'] for row in rows),abs_tol=1e-12) and
            scores['full_value_coverage'] == sum(row['selection']['value']['state']=='full' for row in rows),
            'Final aggregate score or coverage differs')
        verify_files(root/'runtime',read(root/'runtime-proof.json')['files'])
    native = native_reference(fixture,Path(proofs[1]['root']),scorer)
    return dict(version=1,status='inspection' if completed_only else 'completed',at=time.time(),
        root=str(root),scope=SCOPE,query_score_mode=ARM,graph_alpha=0,propagation_applied=False,
        generation_performed_by_analysis=False,external_judge=False,
        answers_source='Actual saved generations; target coverage or rank never implies an answer.',
        comparison_limit='Identical frozen inputs, seeds and sampling; overlap/no-graph profile changes '
            'only query_score_mode and port. Historical actual answers are reused. Runtime changes and '
            'output trajectories prevent assigning all timing differences to the scorer.',
        profile_difference=delta,reference_answers_unchanged=True,completion_audit=completion,
        source_sha256={str(Path(path).resolve()):sha(Path(path)) for path in
            (__file__,_CONTROLLER,Path(__file__).with_name('analyze_ruler_overlap_answers.py'))},
        replay_source_sha256={str(Path(path).resolve()):sha(Path(path)) for path in
            (offline.__file__,score_graph.__code__.co_filename)},
        evidence_sha256={name:sha(root/name) for name in ('fixture.json','prepared-proof.json',
            'runtime-proof.json','reference-proof.json','overlap-reference-proof.json','reference-answers.json')},
        native_full_context=native,fixed_capture=fixed_capture_reference(fixed_capture,[row['id'] for row in rows]),
        summary=summarize(rows) if rows else None,cases=rows)


def format_report(analysis):
    rows,summary = analysis['cases'],analysis['summary']; n=len(rows)
    names = {'mean_k':'原 MeanK','token_sum':'逐 token 页求和','overlap_16_8':'16/8 窗口 MeanK',ARM:'每 block 4 个原始 Key（FPS）'}
    mark = lambda value:'对' if value else '错'
    short = lambda ids:'、'.join(case.rsplit('-',1)[-1] for case in ids) or '无'
    lines = ['# FPS-4：实际回答、召回与成本分析','',SCOPE,'',
        '全部使用原始 256K 输入及相同每题种子/采样；K=36,864、G=16,384、32-token block、'
        'sink=1,312、recent=0、NVFP4/FP8 KV/MTP4，单请求串行、chunk=2,048。'
        'prefill pressure 继续累计 attention；末尾公开 Query 使用每块 4 个原始 Key，'
        '每层/Query token/head 内先取代表 Key 的最高分，再做 block softmax。四组都关闭图传播（α=0），未追加邻块。','',
        '下面是各组模型实际生成的答案，不从召回覆盖或排名推断能否答对。','',
        '| 配置 | 正确题数 | key 完整召回 | value 完整召回 | 平均得分 |',
        '|---|---:|---:|---:|---:|']
    for arm,item in summary['arms'].items():
        lines.append(f"| {names[arm]} | {item['correct']}/{n} | {item['full_key_coverage']}/{n} | {item['full_value_coverage']}/{n} | {item['mean_case_score']:.2f} |")
    native = analysis['native_full_context']
    if native:
        lines += ['',f"历史原生 vLLM 完整上下文在这 12 题实际答对 {native['correct']}/12；"
            f"已核验原始输入、种子、采样与原始回答哈希。该历史组并发 {native['actual_concurrency']}，本组串行，"
            '这里只作答案参照，不作吞吐对照。']
    for arm,change in summary['answer_changes'].items():
        lines += ['',f"相对{names[arm]}：改善 {short(change['improved'])}；退步 {short(change['regressed'])}。"]
    lines += ['', '| 题号 | MeanK | token 求和 | 16/8 | FPS | FPS key/value 覆盖 token | 目标页可选排名 / 实际末位排名 |',
        '|---|---|---|---|---|---|---|']
    for row in rows:
        sel=row['selection']; key,value=sel['key'],sel['value']
        positions='；'.join(f"{b['start']}：{b['optional_rank']}" for b in sel['target_blocks'])
        lines.append(f"| {row['id'].rsplit('-',1)[-1]} | {mark(row['controls']['mean_k']['correct'])} | "
            f"{mark(row['controls']['token_sum']['correct'])} | {mark(row['controls']['overlap_16_8']['correct'])} | "
            f"{mark(row['correct'])} | {key['selected_tokens']}/{key['tokens']}；{value['selected_tokens']}/{value['tokens']} | "
            f"{positions} / {sel['optional_cutoff_rank']} |")
    lines += ['', '可选排名在 score_mask 允许且排除固定保留页的集合内计算，并列分数同名次；'
        '“末位”来自实际选中页的最低分，完整 cutoff 分数及比值见 JSON。是否入选仍以原生 picker 为准。'
        '跨页目标会列出各页，选中其中一页不等于目标值完整。', '',
        f"目标值完整召回但仍答错：{short(summary['full_value_but_wrong'])}；"
        f"目标值未完整召回但答对：{short(summary['correct_without_full_value'])}。"
        '回答正确率与完整召回率分别统计；答对不代表目标内容完整保留，完整召回也不保证答对。']
    partial_correct = [row for row in rows if row['correct'] and row['selection']['value']['state'] != 'full']
    if partial_correct:
        details = '；'.join(f"{row['id'].rsplit('-',1)[-1]} 仅保留 "
            f"{row['selection']['value']['selected_tokens']}/{row['selection']['value']['tokens']} 个目标值 token"
            for row in partial_correct)
        lines += ['', details+'，但最终实际答案得分为 100。这些题计入答对题数，不计入完整召回题数。'
            '缺失内容为何仍能回答正确，不能仅由选页覆盖或最终答案确定。']
    failures=[row for row in rows if not row['correct']]
    if failures:
        lines += ['', '| 未答对题号 | 目标值缺失 token 数 | thinking 是否结束 | 实际最终答案（摘要） |', '|---|---:|---|---|']
        for row in failures:
            answer=row['answer'].replace('|','\\|').replace('\n',' ')[:180] or '（空答案）'
            lines.append(f"| {row['id'].rsplit('-',1)[-1]} | {len(row['selection']['value']['missing_positions'])} | "
                f"{'是' if row['thinking_complete'] else '否'} | {answer} |")
    lines += ['',TIMING_LIMIT,'','| 正式模型进程内区间 | 中位数（秒） | P90（秒） |','|---|---:|---:|']
    for field,label in (('cache_build_seconds','缓存构建/读取/FPS'),('scoring_seconds','代表 K 搬运＋评分'),
            ('combined_fps_seconds','FPS 完整区间'),('query_refresh_seconds','整个 Query 重选')):
        item=summary[field];lines.append(f"| {label} | {item['median']:.2f} | {item['p90']:.2f} |")
    size=summary['cache_bytes']
    lines += ['',f"每请求缓存中位 {size['median']/1024**3:.3f} GiB，最大 {size['max']/1024**3:.3f} GiB；"
        f"累计命中 {summary['cache_hit_blocks']} 个 block，重建 {summary['cache_rebuilt_blocks']} 个 block。"
        '每题独立 owner、一次公开 Query；零命中时测的是首次构建成本，未测重复 Query 的热缓存性能。', '',
        '| 配置 | 整请求中位数（秒） | P90（秒） | 总生成 token |','|---|---:|---:|---:|']
    for arm,item in summary['arms'].items():
        lines.append(f"| {names[arm]} | {item['request_seconds']['median']:.2f} | {item['request_seconds']['p90']:.2f} | {item['completion_tokens']} |")
    fixed=analysis['fixed_capture']
    if fixed:
        lines += ['', '另有固定历史 Q/K 的离线诊断，只分析相同操作数的召回排序，不生成答案。'
            '其 CPU/mmap 冷读取与构建耗时可能和模型启动重叠，不能与上表正式进程的 GPU 评分直接比较；'
            '首题与该 CPU 诊断曾并跑，因此逐题时间也不完全处于相同 CPU 负载下，不能据此作因果归因。'
            '具体目标页排名与文件哈希保存在 JSON 的 fixed_capture 中。']
    lines += ['', '已核验所有完成回答的原始证据哈希、三组旧答案快照、原输入/种子/采样、α=0 的 identity 与'
        '逐项原始种子不变、原生 picker 重放一致、原 block 长度和预算、未追加邻块。最终执行审计和 GPU 清理通过。'
        '审计通过说明本次执行符合定义，不能单凭正确率高低宣称实现有错或算法必然更优。','',
        '详细实际答案、覆盖位置、每页分数/排名/cutoff、分阶段耗时、生成 token 数、缓存容量与哈希见 [analysis.json](analysis.json)。','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--fixed-capture',type=Path,default=DEFAULT_FIXED_CAPTURE)
    parser.add_argument('--inspect-completed',action='store_true',help='Read completed cases only; print without writing')
    args=parser.parse_args(argv)
    analysis=build_analysis(args.root,completed_only=args.inspect_completed,fixed_capture=args.fixed_capture)
    if args.inspect_completed:
        print(json.dumps(analysis,ensure_ascii=False,allow_nan=False));return
    root=args.root.resolve();output=root/'analysis.json';report=root/'report.md'
    require(not output.exists() and not report.exists(),'Analysis artifacts already exist; do not overwrite')
    text=format_report(analysis)
    with output.open('x') as destination:
        destination.write(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with report.open('x') as destination:
        destination.write(text)
    print(json.dumps(dict(cases=len(analysis['cases']),summary=analysis['summary'],
        analysis=str(output),report=str(report)),ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
