#!/usr/bin/env python3
"""CPU-only, answer-blind selection replay for the completed 16/8 cohort.

No model generation or external judge is used. The optional inspection command
prints completed cases without publishing files; final artifacts require the
whole campaign's immutable evidence, execution audit and GPU cleanup to pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import analyze_block_graph as offline
from vllm_kvmem.graph_scoring import score_graph


ARM = 'overlap_16_8'
REFERENCES = ('mean_k', 'token_sum')
DEFAULT_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/ruler-overlap-16-8-20261023-r1')
COMPONENT_PATH = Path(__file__).resolve().parents[1]/'artifacts/overlap-window-component-20261023.json'
COMPARISON_LIMIT = ('此前 MeanK 和逐 token 对照使用 α=0；本次同时使用 16/8 窗口评分与 '
    'V2 multi sum、α=0.2。因此答案变化是这套组合的结果，不能全部归因于窗口评分。')
SCOPE = '原先 9 道 MeanK 失败题＋3 道成功对照，共 12 题；不是完整 RULER 准确率。'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(directory, manifest):
    for name, expected in manifest.items():
        require(sha(directory/name) == expected, f'Immutable evidence changed: {directory/name}')


def span_coverage(ranges, span):
    begin, end = span
    require(0 <= begin < end, 'Invalid target span')
    pieces = sorted((max(begin, start), min(end, start+count))
        for start, count in ranges if start < end and start+count > begin)
    merged = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    selected = sum(hi-lo for lo, hi in merged)
    missing = [position for position in range(begin, end)
        if not any(lo <= position < hi for lo, hi in merged)]
    return dict(tokens=end-begin, selected_tokens=selected,
        state='full' if selected == end-begin else 'partial' if selected else 'none',
        missing_positions=missing)


def ranks(scores, eligible):
    """Competition ranks inside transport; actual inclusion uses native picker."""
    ordered = sorted((float(score) for score, keep in zip(scores, eligible) if keep), reverse=True)
    first = {}
    for index, value in enumerate(ordered, 1):
        first.setdefault(value, index)
    return [first[float(score)] if keep else None for score, keep in zip(scores, eligible)]


def replay_selection(inputs, actual_ranges, value_span, key_span):
    alpha = inputs.get('alpha')
    require(inputs.get('seed_source') == ARM and inputs.get('mode') == 'multi' and
        inputs.get('scoring_version') == 'v2' and inputs.get('aggregation') == 'sum' and
        alpha in (0, .2), 'Recorded Query is not requested overlap/V2 multi SUM alpha=0 or .2')
    config, snapshot = offline._validate_record(inputs)
    memory, handles = offline._native_history(config, inputs['starts'], inputs['counts'])
    starts, counts, raw = inputs['starts'], inputs['counts'], inputs['raw_scores']
    actual = {start for start, _ in actual_ranges}
    try:
        offline._validate_native_masks(inputs, memory, handles)
        mandatory = tuple(handles[start].slot for start in inputs['mandatory_starts'])
        def select(scores):
            slots = memory.preview_selection({handles[start].slot:score for start, score in zip(starts, scores)},
                mandatory, action=False, prefill_scores=None, max_tokens=inputs['budget'])
            return {start for start, handle in handles.items() if handle.slot in slots}
        seed_selected = select(raw)
        if 'baseline_selected_starts' in inputs:
            require(seed_selected == set(inputs['baseline_selected_starts']), 'Captured seed-only selection differs')
        with score_graph(raw, starts, inputs['transport_mask'], snapshot, mode='multi',
                scoring_version='v2', aggregation='sum', alpha=alpha,
                cross_message_weight=inputs['cross_message_weight'], block_tokens=config.block_tokens) as output:
            propagated = list(output.scores)
            actual_replay = select(propagated)
            require(actual_replay == actual, 'Observed selection differs from native SUM picker replay')
            diagnostics = dict(output.diagnostics)
            if alpha == 0:
                require(propagated == raw and actual == seed_selected and diagnostics['identity'],
                    'Disabled propagation changed seeds or native selection')
    finally:
        require(memory.close(), 'Offline native selection metadata did not release')
    expected_ranges = sorted((start, count) for start, count in zip(starts, counts) if start in actual)
    require(expected_ranges == sorted(tuple(v) for v in actual_ranges), 'Selected block lengths or duplicate ranges differ')
    seed_ranges = [(start, count) for start, count in zip(starts, counts) if start in seed_selected]
    before_ranks, after_ranks = ranks(raw, inputs['transport_mask']), ranks(propagated, inputs['transport_mask'])
    blocks = []
    for index, (start, count) in enumerate(zip(starts, counts)):
        value_tokens = max(0, min(start+count, value_span[1])-max(start, value_span[0]))
        key_tokens = max(0, min(start+count, key_span[1])-max(start, key_span[0]))
        if value_tokens or key_tokens:
            blocks.append(dict(start=start, count=count, value_tokens=value_tokens, key_tokens=key_tokens,
                score_allowed=inputs['score_mask'][index], transport_allowed=inputs['transport_mask'][index],
                seed_score=raw[index], propagated_score=propagated[index],
                seed_rank=before_ranks[index], propagated_rank=after_ranks[index],
                selected_before_graph=start in seed_selected, selected_actual=start in actual))
    require(sum(block['value_tokens'] for block in blocks) == value_span[1]-value_span[0],
        'Target value not fully present in accepted history')
    return dict(target_blocks=blocks, pregraph_value=span_coverage(seed_ranges, value_span),
        actual_value=span_coverage(actual_ranges, value_span),
        pregraph_key=span_coverage(seed_ranges, key_span), actual_key=span_coverage(actual_ranges, key_span),
        graph_displaced_value_blocks=[b['start'] for b in blocks if b['value_tokens'] and
            b['selected_before_graph'] and not b['selected_actual']],
        graph_added_value_blocks=[b['start'] for b in blocks if b['value_tokens'] and
            not b['selected_before_graph'] and b['selected_actual']],
        selected_tokens=sum(count for _, count in actual_ranges),
        selected_blocks=len(actual), transport_selected_blocks=sum(start in actual and keep
            for start, keep in zip(starts, inputs['transport_mask'])),
        rank_definition='1 + number of strictly higher scores within transport_mask; ties share rank',
        native_picker_replay_equal=True, graph_alpha=alpha, propagation=diagnostics)


def analyze_case(root, row, target, reference):
    directory = root/ARM/'cases'/row['id']
    manifest = read(directory/'hashes.json')
    verify_files(directory, manifest)
    result, evidence, request = (read(directory/name) for name in ('result.json', 'evidence.json', 'request.json'))
    queries = [event for event in evidence['bounded'] if event['selection'] == 'query']
    require(len(queries) == 1 and len(evidence['query_inputs']) == 1, 'Expected one public Query')
    query, inputs = queries[0], evidence['query_inputs'][0]
    exported = [json.loads(line) for line in (directory/'query-inputs.jsonl').read_text().splitlines() if line.strip()]
    require(exported == evidence['query_inputs'], 'Standalone Query dump differs from saved evidence')
    require(result['selection_audit']['native_picker_replay_equal'] and
        result['selection_audit']['no_post_picker_neighbors'], 'Case execution audit did not pass')
    require(query['query_scoring_method'] == ARM and query['guided_query_tokens'] == 0 and
        query['query_span'] == row['served_query'], 'Original public Query changed')
    require(request['input_sha256'] == row['served_sha256'] and request['sampling']['seed'] == row['seed'],
        'Original input or seed changed')
    selection = replay_selection(inputs, query['selected_history_ranges'], target['token_span'], target['key_token_span'])
    require(result['value_covered'] == (selection['actual_value']['state'] == 'full'), 'Saved target coverage differs')
    controls = {}
    for arm in REFERENCES:
        case = reference/arm/'cases'/row['id']
        hashes = read(case/'hashes.json'); verify_files(case, hashes)
        previous = read(case/'result.json')
        controls[arm] = {key:previous[key] for key in ('official_case_score', 'value_covered',
            'record_covered', 'completion_tokens', 'seconds')}
        controls[arm]['correct'] = previous['official_case_score'] == 100
    correct = result['official_case_score'] == 100
    if selection['actual_value']['state'] == 'full':
        outcome = 'full_value_retrieved_answer_correct' if correct else 'full_value_retrieved_answer_incorrect'
    elif selection['pregraph_value']['state'] == 'full':
        outcome = 'graph_displaced_complete_value'
    elif selection['actual_value']['state'] == 'partial':
        outcome = 'partial_value_retrieval'
    else:
        outcome = 'value_not_retrieved'
    return dict(id=row['id'], expected=row['outputs'], answer=result['answer'], correct=correct,
        official_case_score=result['official_case_score'], value_span=target['token_span'],
        key_span=target['key_token_span'], controls=controls, outcome=outcome,
        selection=selection, completion_tokens=result['completion_tokens'],
        request_seconds=result['seconds'], finish_reason=result['finish_reason'],
        query_refresh_seconds=query['refresh_finished_at']-query['refresh_started_at'],
        window_stats=query['query_window_scoring'], pressure_count=result['pressure_count'],
        evidence_sha256=manifest, execution_audit_passed=True)


def distribution(values):
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), 'Cannot summarize empty data')
    position = (len(ordered)-1)*.9
    lo, hi = math.floor(position), math.ceil(position)
    return dict(count=len(ordered), median=statistics.median(ordered),
        p90=ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo),
        min=ordered[0], max=ordered[-1], sum=math.fsum(ordered))


def summarize(rows):
    total = len(rows)
    results = {}
    for arm in (*REFERENCES, ARM):
        entries = rows if arm == ARM else [row['controls'][arm] for row in rows]
        results[arm] = dict(cases=total, correct=sum(item['correct'] for item in entries),
            mean_case_score=statistics.mean(item['official_case_score'] for item in entries),
            full_value_coverage=(sum(row['selection']['actual_value']['state'] == 'full' for row in rows)
                if arm == ARM else sum(item['value_covered'] for item in entries)),
            completion_tokens=sum(item['completion_tokens'] for item in entries),
            request_seconds=distribution(item['request_seconds' if arm == ARM else 'seconds'] for item in entries))
    return dict(arms=results,
        pregraph_full_value_coverage=sum(row['selection']['pregraph_value']['state'] == 'full' for row in rows),
        graph_displaced_complete_value=[row['id'] for row in rows if row['outcome'] == 'graph_displaced_complete_value'],
        graph_added_any_value_blocks=[row['id'] for row in rows if row['selection']['graph_added_value_blocks']],
        window_score_seconds=distribution(row['window_stats']['seconds'] for row in rows),
        query_refresh_seconds=distribution(row['query_refresh_seconds'] for row in rows),
        cache_bytes=distribution(row['window_stats']['cache_bytes'] for row in rows),
        cache_hit_blocks=sum(row['window_stats']['cache_hit_blocks'] for row in rows),
        cache_rebuilt_blocks=sum(row['window_stats']['cache_rebuilt_blocks'] for row in rows),
        fallback_blocks=sum(row['window_stats']['fallback_blocks'] for row in rows),
        quantile_definition='P90 uses linear interpolation at (n-1)*0.9',
        timing_limit='Window seconds include cold archive reads, cached-mean construction, transfers and '
            'scoring; not a separately measured cache-build or GPU-kernel cost. Request timing also '
            'depends on different output lengths and does not isolate scoring overhead.')


def compare_previous_overlap(root, rows):
    """Compare actual generated answers, keeping seed drift separate from propagation."""
    path = root/'overlap-reference-proof.json'
    if not path.exists():
        return None
    proof = read(path)
    previous = Path(proof['root'])
    verify_files(previous, proof['files'])
    cases = []
    for row in rows:
        old_case = previous/ARM/'cases'/row['id']
        old_result = read(old_case/'result.json')
        old_inputs = read(old_case/'evidence.json')['query_inputs'][0]
        new_inputs = read(root/ARM/'cases'/row['id']/'evidence.json')['query_inputs'][0]
        require(old_inputs['alpha'] == .2 and new_inputs['alpha'] == 0,
            'Paired answer comparison requires previous alpha=.2 and current alpha=0')
        aligned = all(old_inputs[key] == new_inputs[key] for key in
            ('starts', 'counts', 'score_mask', 'transport_mask', 'mandatory_starts', 'budget'))
        old_query = next(q for q in read(old_case/'evidence.json')['bounded'] if q['selection'] == 'query')
        old_selection = replay_selection(old_inputs, old_query['selected_history_ranges'],
            row['value_span'], row['key_span'])
        delta = ([abs(float(a)-float(b)) for a,b in zip(old_inputs['raw_scores'],new_inputs['raw_scores'])]
            if aligned else None)
        cases.append(dict(id=row['id'], previous_correct=old_result['official_case_score'] == 100,
            current_correct=row['correct'], previous_coverage=old_result['value_covered'],
            current_coverage=row['selection']['actual_value']['state'] == 'full',
            previous_seed_only_coverage=old_selection['pregraph_value']['state'] == 'full',
            index_and_masks_equal=aligned, raw_seeds_exactly_equal=aligned and
                old_inputs['raw_scores'] == new_inputs['raw_scores'],
            raw_seed_max_abs_difference=max(delta) if delta else None,
            previous_completion_tokens=old_result['completion_tokens'],
            current_completion_tokens=row['completion_tokens']))
    return dict(root=str(previous), profile_difference=proof['profile_difference'],
        previous_correct=sum(c['previous_correct'] for c in cases),
        current_correct=sum(c['current_correct'] for c in cases),
        improved=[c['id'] for c in cases if c['current_correct'] and not c['previous_correct']],
        regressed=[c['id'] for c in cases if c['previous_correct'] and not c['current_correct']],
        all_index_and_masks_equal=all(c['index_and_masks_equal'] for c in cases),
        all_raw_seeds_exactly_equal=all(c['raw_seeds_exactly_equal'] for c in cases), cases=cases)


def format_report(analysis):
    rows, summary = analysis['cases'], analysis['summary']
    alpha = analysis.get('graph_alpha', .2)
    n = len(rows)
    lines = ['# 16/8 重叠窗口召回：完整回答与选页分析', '', SCOPE, '',
        '**对照限制：'+analysis.get('comparison_limit', COMPARISON_LIMIT)+'**', '',
        '保持原始 256K 输入、种子和采样，K=36,864、G=16,384、32-token KV block、'
        'sink=1,312、recent=0、NVFP4/FP8 KV/MTP4；prefill pressure 继续累计 attention，'
        f'末尾原始 Query 使用窗口均值评分，图传播 α={alpha:g}。没有追加邻块，也没有重试低分答案。', '',
        '| 配置 | 正确题数 | 完整召回目标值 | 平均得分 |',
        '|---|---:|---:|---:|']
    names = {'mean_k':'原 MeanK，α=0', 'token_sum':'逐 token 页求和，α=0',
        ARM:'16/8 窗口，关闭传播' if alpha == 0 else '16/8 窗口＋sum 图，α=.2'}
    for arm, value in summary['arms'].items():
        lines.append(f"| {names[arm]} | {value['correct']}/{n} | {value['full_value_coverage']}/{n} | {value['mean_case_score']:.2f} |")
    previous = analysis.get('previous_overlap')
    if previous:
        lines += ['', f"与之前同样 16/8 的实际生成相比：开启 sum 图传播时正确 {previous['previous_correct']}/{n}，"
            f"本次正确 {previous['current_correct']}/{n}。改善题号："
            + ('、'.join(previous['improved']) or '无') + '；退步题号：'
            + ('、'.join(previous['regressed']) or '无') + '。',
            '两轮全部原始窗口种子分数逐项相同。' if previous['all_raw_seeds_exactly_equal'] else
            '两轮部分题目的历史索引或评分域不同，不能将其视为固定同一组种子的传播消融。'
            if not previous['all_index_and_masks_equal'] else
            '两轮窗口种子存在数值差异，逐题差值已单列记录；不把全部答案差异直接归因于传播。']
    lines += ['', '以下“传播前”只是在本次实际种子分数上重放原生 picker，未生成另一组答案。'
        '即使传播前完整覆盖目标值，也不能据此断言模型一定答对。排名只在 transport_mask 范围内比较，'
        '并列分数同名次；实际入选始终以原生 picker 为准。', '',
        '| 题号 | 原 MeanK | 原逐 token | 本次 16/8 | 目标值覆盖：传播前→实际 | 目标页排名：种子→最终 |',
        '|---|---:|---:|---:|---|---|']
    coverage_names = {'full':'完整', 'partial':'部分', 'none':'无'}
    def mark(correct):
        return '对' if correct else '错'
    for row in rows:
        selection = row['selection']
        positions = '；'.join(f"{block['start']}：{block['seed_rank']}→{block['propagated_rank']}"
            for block in selection['target_blocks'] if block['value_tokens'])
        before, after = selection['pregraph_value'], selection['actual_value']
        coverage = f"{coverage_names[before['state']]}({before['selected_tokens']}/{before['tokens']})→" \
            f"{coverage_names[after['state']]}({after['selected_tokens']}/{after['tokens']})"
        lines.append(f"| {row['id'].rsplit('-',1)[-1]} | {mark(row['controls']['mean_k']['correct'])} | "
            f"{mark(row['controls']['token_sum']['correct'])} | {mark(row['correct'])} | {coverage} | {positions} |")
    lines += ['', f"传播前完整覆盖目标值 {summary['pregraph_full_value_coverage']}/{n}；实际图后覆盖 "
        f"{summary['arms'][ARM]['full_value_coverage']}/{n}。"]
    displaced = summary['graph_displaced_complete_value']
    if displaced:
        lines += ['', '以下题在传播前完整覆盖、传播后丢失了至少一个目标值页：'+
            '、'.join(case.rsplit('-',1)[-1] for case in displaced)+'。这说明当前 sum 图会改变预算边缘的排序；'
            '窗口提供两侧评分依据，但没有强制共同入选。']
    failed = [row for row in rows if not row['correct']]
    if failed:
        lines += ['', '| 未答对题号 | 实际缺失目标值 token 数 | 原因定位 |', '|---|---:|---|']
        reasons = {'graph_displaced_complete_value':'传播前覆盖完整，图后目标值页被挤出',
            'partial_value_retrieval':'仍只选回部分目标值', 'value_not_retrieved':'未选回目标值',
            'full_value_retrieved_answer_incorrect':'目标值已完整召回，回答仍不正确'}
        for row in failed:
            missing = len(row['selection']['actual_value']['missing_positions'])
            lines.append(f"| {row['id'].rsplit('-',1)[-1]} | {missing} | {reasons[row['outcome']]} |")
    seconds = summary['window_score_seconds']; refresh = summary['query_refresh_seconds']; size = summary['cache_bytes']
    lines += ['', f"窗口统计完整区间中位数 {seconds['median']:.2f} 秒，P90 {seconds['p90']:.2f} 秒；"
        f"整次 Query 重选中位数 {refresh['median']:.2f} 秒，P90 {refresh['p90']:.2f} 秒。"
        '窗口区间包括原始 K 读取、均值缓存构建、搬运与评分，不能当作纯内核时间或单独缓存构建时间。', '',
        f"每请求均值缓存占用中位数 {size['median']/1024**3:.3f} GiB，峰值 {size['max']/1024**3:.3f} GiB；"
        f"累计缓存命中 {summary['cache_hit_blocks']} 个 block、重建 {summary['cache_rebuilt_blocks']} 个 block，"
        f"MeanK 回退 {summary['fallback_blocks']} 个 block。",
        '本组每题独立 owner、每题仅一次公开 Query；如果命中为零，测到的是冷缓存成本，尚不能代表同一会话多次 Query 的复用收益。', '',
        '| 配置 | 整请求耗时中位数（秒） | P90（秒） | 总生成 token |', '|---|---:|---:|---:|']
    for arm, item in summary['arms'].items():
        lines.append(f"| {names[arm]} | {item['request_seconds']['median']:.2f} | "
            f"{item['request_seconds']['p90']:.2f} | {item['completion_tokens']} |")
    component = analysis.get('fixed_capture_component')
    if component:
        cold, warm = (next(run for run in component['measurement']['runs'] if run['run'] == name)
            for name in ('cold', 'warm'))
        lines += ['', f"固定捕获 Q/K 的独立组件测量（题 {component['measurement']['case'].rsplit('-',1)[-1]}）："
            f"首次构建＋评分 {cold['wall_seconds']:.2f} 秒，已有均值缓存时 {warm['wall_seconds']:.2f} 秒。"
            '它不包括整段模型 prefill、图传播、picker 和答案生成；冷缓存测试读取保存的 Q/K，'
            '不能与模型进程内首次构建的约 40 秒直接视为相同测量，也不能据此宣称端到端吞吐提升。'
            f"[组件证据]({component['path']})。"]
    lines += ['', '整请求耗时包括 prefill、Query 重选及生成；不同输出长度影响耗时，不能全部归因于评分方案。', '',
        '全部案例的原始 request/response/result/evidence 哈希已核验，图传播与原生 picker 已离线重放一致，'
        '执行审计与 GPU 清理通过。上述低分属于本配置的质量结果；现有证据没有显示评分顺序、预算或 '
        'picker 实现错误。答案正确率不能替代实现审计，审计通过也不保证策略质量提高。', '',
        '完整逐题分数、覆盖位置、传播前后目标页分数与排名、缓存统计及证据哈希见 [analysis.json](analysis.json)。', '']
    return '\n'.join(lines)


def build_analysis(root, *, completed_only=False):
    root = root.resolve()
    status = read(root/'status.json')
    if not completed_only:
        require(status['status'] == 'completed', 'Final analysis requires a completed campaign')
        require(read(root/'execution-audit.json')['status'] == 'passed', 'Final execution audit has not passed')
        cleanup = read(root/ARM/'cleanup.json')
        require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'GPU cleanup incomplete')
    fixture, targets, proof = (read(root/name) for name in ('fixture.json', 'target-spans.json', 'reference-proof.json'))
    verify_files(root, read(root/'prepared-proof.json'))
    reference = Path(proof['root'])
    verify_files(reference, proof['files'])
    rows = []
    for row in fixture['rows']:
        if completed_only and not (root/ARM/'cases'/row['id']/'hashes.json').exists():
            continue
        rows.append(analyze_case(root, row, targets[row['id']], reference))
    require(bool(rows), 'No complete immutable cases are available')
    if not completed_only:
        require(len(rows) == len(fixture['rows']) == 12, 'Expected all twelve cases')
        scores = read(root/ARM/'scores.json')
        require(scores['cases'] == len(rows) and math.isclose(scores['mean_case_score'],
            statistics.mean(row['official_case_score'] for row in rows), abs_tol=1e-12), 'Final aggregate score differs')
        frozen = read(root/'runtime-proof.json')
        verify_files(root/'runtime', frozen['files'])
    component = None
    if COMPONENT_PATH.exists():
        measurement = read(COMPONENT_PATH)
        require({item['run'] for item in measurement['runs']} == {'cold', 'warm'}, 'Unexpected fixed-capture component runs')
        require(all(item['method'] == ARM for item in measurement['runs']), 'Fixed component uses another scorer')
        component = dict(path=str(COMPONENT_PATH.resolve()), sha256=sha(COMPONENT_PATH), measurement=measurement)
    alpha = read(root/ARM/'profile.json')['kvmem']['graph_alpha']
    require(all(row['selection']['graph_alpha'] == alpha for row in rows), 'Profile/record alpha differs')
    previous_overlap = compare_previous_overlap(root, rows)
    comparison_limit = (COMPARISON_LIMIT if alpha else
        '本次关闭图传播（α=0），与更早 MeanK/逐 token 组均不传播，但评分方法不同。'
        '12 题是失败样本富集的诊断集。')
    return dict(version=1, status='inspection' if completed_only else 'completed', at=time.time(),
        root=str(root), scope=SCOPE, graph_alpha=alpha, comparison_limit=comparison_limit,
        generation_performed=False, generation_performed_by_analysis=False,
        answers_source='Actual model generations from each campaign; not inferred from target coverage.',
        external_judge=False, analysis_source_sha256=sha(Path(__file__)),
        source_proof={name:sha(root/name) for name in ('fixture.json', 'prepared-proof.json', 'reference-proof.json')},
        replay_source_proof={str(path):sha(path) for path in (
            Path(offline.__file__).resolve(), Path(score_graph.__code__.co_filename).resolve())},
        fixed_capture_component=component, previous_overlap=previous_overlap,
        selection_analysis='Seed-only preview reuses actual seeds; no model answer is inferred from this preview.',
        summary=summarize(rows), cases=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--inspect-completed', action='store_true', help='Print in-memory analysis only; never publish files')
    args = parser.parse_args(argv)
    analysis = build_analysis(args.root, completed_only=args.inspect_completed)
    if args.inspect_completed:
        print(json.dumps(analysis, ensure_ascii=False, allow_nan=False))
        return
    root = args.root.resolve()
    output, report = root/'analysis.json', root/'report.md'
    require(not output.exists() and not report.exists(), 'Analysis artifacts already exist; do not silently overwrite')
    report_text = format_report(analysis)
    with output.open('x') as target:
        target.write(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with report.open('x') as target:
        target.write(report_text)
    print(json.dumps(dict(cases=len(analysis['cases']), summary=analysis['summary'],
        analysis=str(output), report=str(report)), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
