"""Causal, rule-only candidate slices. No inference or trace commands execute.

Candidate slices are workflow proposals, NOT semantic stages or ground truth.
Only original messages in each selected Step are inspected. Exporter outcomes,
Observer predictions and review references are never read as rule inputs.
"""
import argparse
from collections import Counter
import copy
import csv
import html
import json
from pathlib import Path
import re
import shlex
import time

from observer import digest, extract_step, resolve, save


VERSION = 'rule-candidates-v1'
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
INTENT = re.compile(r'\b(?:next(?:\s+up)?|move\s+on\s+to|switch\s+to|start\s+task)\b', re.I)
EXECUTION = {'verification_execution', 'runtime_execution'}
LABELS = {
    'inspection': '读取/检查', 'modification_attempt': '修改尝试',
    'verification_execution': '执行测试/构建/类型检查',
    'runtime_execution': '执行程序', 'vcs_mutation': '版本库变更尝试',
    'task_bookkeeping': '任务计划/状态登记', 'assistant_only': '仅模型回复',
    'unknown_tool': '其他工具', 'mixed': '多类实际工具行为',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean_offsets(text):
    """Strip ANSI for matching while retaining original Unicode coordinates."""
    output, offsets, previous = [], [], 0
    for match in ANSI.finditer(text):
        output.append(text[previous:match.start()])
        offsets.extend(range(previous, match.start()))
        previous = match.end()
    output.append(text[previous:])
    offsets.extend(range(previous, len(text)))
    return ''.join(output), offsets


def citation(source, pointer, text, begin=0, end=None):
    end = min(len(text), begin + 240) if end is None else end
    return dict(source={**source, 'pointer': pointer, 'character_span': [begin, end]},
                quote=text[begin:end])


def command_facts(command):
    """Bounded lexical classification, not a shell parser or success detector."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|()')
        lexer.whitespace_split = True
        lexer.commenters = ''
        tokens = list(lexer)
    except ValueError:
        return dict(family='unknown_tool', test_runner=False, pipeline=False,
                    note='Shell tokenization failed; no fallback execution or text-wide command matching.')
    groups, current = [], []
    for token in tokens:
        if token and all(c in ';&|()' for c in token):
            if current:
                groups.append(current)
                current = []
        else:
            current.append(token)
    if current:
        groups.append(current)
    families, test_runner = set(), False
    for group in groups:
        while group and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', group[0]):
            group = group[1:]
        if not group:
            continue
        executable = Path(group[0]).name
        args = group[1:]
        if executable == 'git':
            # Skip common global options without scanning commit-message bodies.
            while args and args[0].startswith('-'):
                args = args[2:] if args[0] in ('-C', '-c', '--git-dir', '--work-tree') else args[1:]
            operation = args[0] if args else ''
            families.add('vcs_mutation' if operation in ('add', 'commit', 'switch', 'checkout', 'merge', 'reset')
                         else 'inspection')
        elif executable in ('npx', 'npm', 'pnpm', 'yarn', 'cargo', 'go', 'pytest', 'python', 'python3', 'ruff', 'tsc'):
            is_test = (executable == 'pytest'
                or executable in ('cargo', 'go') and args[:1] == ['test']
                or executable in ('npm', 'pnpm', 'yarn') and args[:1] == ['test']
                or executable in ('npx', 'pnpm', 'yarn') and args[:1] in (['vitest'], ['jest'])
                or executable in ('python', 'python3') and args[:2] == ['-m', 'pytest'])
            is_check = (is_test or executable in ('tsc', 'ruff')
                or executable == 'npx' and args[:1] in (['tsc'], ['eslint'], ['prettier'])
                or executable in ('cargo', 'go') and args[:1] in (['build'], ['check'], ['vet'])
                or executable in ('npm', 'pnpm', 'yarn') and args[:2] in (['run', 'build'], ['run', 'typecheck']))
            test_runner |= is_test
            families.add('verification_execution' if is_check else 'runtime_execution')
        elif executable in ('node', 'deno', 'bun'):
            families.add('runtime_execution')
        elif executable in ('ls', 'cat', 'grep', 'rg', 'find', 'wc', 'pwd', 'head', 'tail', 'sed', 'echo'):
            families.add('inspection')
        else:
            families.add('unknown_tool')
    # Presentation filters do not change the principal command family.
    if len(families) > 1:
        families.discard('inspection')
    family = next(iter(families)) if len(families) == 1 else 'mixed' if families else 'unknown_tool'
    return dict(family=family, test_runner=test_runner, pipeline=any('|' in t and set(t) <= set('|&;()') for t in tokens))


def extract_events(trajectory, step, records):
    events, calls, returned = [], {}, set()
    families, read_targets, modified_targets = set(), [], []

    def emit(kind, citations, **attributes):
        event = dict(event_id=f'E{step:04}-{len(events)+1:03}', step=step, kind=kind,
                     attributes=attributes, citations=citations)
        events.append(event)
        return event

    for line, record in records:
        require(record['turn'] == step, 'Other/future Step in event extraction')
        source = dict(trace_path=trajectory['trace_path'], trace_line=line,
                      message_index=record['message_index'], step=step)
        original = record['original']
        require(isinstance(original['content'], list), 'This prototype expects content-block arrays')
        for index, block in enumerate(original['content']):
            pointer = f'/original/content/{index}'
            kind = block.get('type')
            if kind == 'tool_use':
                require(original['role'] == 'assistant' and block['id'] not in calls, 'Invalid tool request')
                name, inputs = block['name'], block['input']
                if name in ('Read', 'Grep', 'Glob'):
                    family = 'inspection'
                elif name in ('Write', 'Edit', 'MultiEdit'):
                    family = 'modification_attempt'
                elif name in ('TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet'):
                    family = 'task_bookkeeping'
                else:
                    family = 'unknown_tool'
                facts = command_facts(inputs.get('command', '')) if name == 'Bash' else {}
                family = facts.get('family', family)
                families.add(family)
                cites = [citation(source, pointer + '/name', name)]
                for key in ('file_path', 'command', 'subject', 'taskId', 'status'):
                    if isinstance(inputs.get(key), str):
                        cites.append(citation(source, pointer + '/input/' + key, inputs[key]))
                event = emit('tool_call', cites, call_id=block['id'], tool=name,
                             workflow_family=family, **({'command_analysis': facts} if facts else {}))
                calls[block['id']] = dict(block=block, event=event, source=source, pointer=pointer, facts=facts)
                target = inputs.get('file_path')
                if isinstance(target, str):
                    (modified_targets if family == 'modification_attempt' else read_targets).append(target)
            elif kind == 'tool_result':
                call_id = block['tool_use_id']
                require(call_id in calls and call_id not in returned, 'Cross-Step/duplicate tool return')
                returned.add(call_id)
                call = calls[call_id]
                name, inputs = call['block']['name'], call['block']['input']
                text = block.get('content', '')
                if not isinstance(text, str):
                    emit('nontext_tool_return', [], call_id=call_id, semantic_status='unknown')
                    continue
                plain, offsets = clean_offsets(text)

                def result_cite(begin=0, end=None):
                    if end is None:
                        end = min(len(plain), begin + 240)
                    a = offsets[begin] if begin < len(offsets) else len(text)
                    b = offsets[end-1]+1 if end > begin else a
                    return citation(source, pointer + '/content', text, a, b)

                failed_tool = block.get('is_error') is True or '<tool_use_error>' in plain[:160]
                emit('tool_return', [result_cite()], call_id=call_id, tool=name,
                     error_flag=block.get('is_error'), error_marker_seen=failed_tool,
                     successful_tool_return_is_not_goal_completion=True)
                if failed_tool:
                    emit('tool_error_observed', [result_cite()], call_id=call_id,
                         scope='tool_attempt_not_entire_goal')
                if name == 'TaskCreate' and not failed_tool:
                    match = re.match(r'Task #([^\s]+) created successfully:', plain)
                    if match:
                        emit('task_created', [result_cite(0, match.end()), *call['event']['citations']],
                             task_id=match[1], subject=inputs.get('subject'),
                             task_creation_does_not_start_execution=True)
                if name == 'TaskUpdate' and not failed_tool:
                    match = re.match(r'Updated task #([^\s]+) status', plain)
                    if match and str(inputs.get('taskId')) == match[1] and 'status' in inputs:
                        emit('task_status_declared', [result_cite(0, match.end()), *call['event']['citations']],
                             task_id=match[1], declared_status=inputs['status'],
                             semantic_completion_proven=False)
                if name == 'Bash' and call['facts']['test_runner']:
                    summaries = list(re.finditer(r'^\s*Tests\s*:?.*$', plain, re.M))
                    if not summaries:
                        summaries = list(re.finditer(r'^=+[^\n]*\d+\s+(?:failed|passed)[^\n]*=+\s*$', plain, re.M))
                    for match in summaries:
                        counts = {label: int(count) for count, label in
                                  re.findall(r'(\d+)\s+(failed|passed|skipped|todo)', match[0])}
                        if counts:
                            emit('test_summary_observed', [result_cite(match.start(), match.end())],
                                 call_id=call_id, **counts, scope='reported_test_summary_only',
                                 semantic_completion_proven=False,
                                 pipeline_exit_status_not_used_as_test_success=call['facts']['pipeline'])
                if name == 'Bash' and call['facts']['family'] == 'vcs_mutation':
                    match = re.search(r'^\[[^\]\n]+ [0-9a-f]{7,40}\].*$', plain, re.M)
                    if match:
                        emit('commit_output_observed', [result_cite(match.start(), match.end())],
                             semantic_completion_proven=False)
            elif original['role'] == 'assistant' and kind in ('thinking', 'text'):
                field = 'thinking' if kind == 'thinking' else 'text'
                text = block.get(field, '')
                match = INTENT.search(text[:320])
                if match:
                    emit('intent_cue', [citation(source, pointer + '/' + field, text)],
                         matched=match[0], reliability='agent_text_hint_not_executed_action')
    require(set(calls) == returned, 'Selected Step has unobserved tool returns; cannot call it closed')
    effective = families - {'task_bookkeeping'} if len(families) > 1 else families
    mode = next(iter(effective)) if len(effective) == 1 else 'mixed' if effective else 'assistant_only'
    if mode == 'assistant_only':
        line, record = records[0]
        block = next((b for b in record['original']['content'] if b.get('type') == 'text'), None)
        if block:
            i = record['original']['content'].index(block)
            source = dict(trace_path=trajectory['trace_path'], trace_line=line,
                          message_index=record['message_index'], step=step)
            emit('assistant_only_reply', [citation(source, f'/original/content/{i}/text', block['text'])],
                 semantic_completion_proven=False)
    return dict(step=step, events=events, workflow_mode=mode,
                workflow_families=sorted(families), read_targets=sorted(set(read_targets)),
                modified_targets=sorted(set(modified_targets)), tool_calls=len(calls),
                tool_returns=len(returned))


class CandidateSlicer:
    def __init__(self, max_steps=12):
        require(max_steps > 0, 'max_steps must be positive')
        self.max_steps = max_steps
        self.tasks, self.active = {}, set()
        self.previous = None
        self.slices, self.decisions = [], []

    def observe(self, facts):
        facts = copy.deepcopy(facts)
        turn, events = facts['step'], facts['events']
        require(self.previous is None or turn == self.previous['step']+1, 'Gap, duplicate or reordered Step')
        old_active = set(self.active)
        starts = []
        for event in events:
            attrs = event['attributes']
            if event['kind'] == 'task_created':
                self.tasks[attrs['task_id']] = dict(subject=attrs['subject'], declared_status='pending')
            elif event['kind'] == 'task_status_declared':
                tid, status = attrs['task_id'], attrs['declared_status']
                task = self.tasks.setdefault(tid, dict(subject=None))
                task['declared_status'] = status
                if status == 'in_progress':
                    self.active.add(tid)
                    if tid not in old_active:
                        starts.append(event)
                elif status in ('completed', 'deleted', 'pending'):
                    self.active.discard(tid)
        reasons = []

        def reason(rule, explanation, evidence, category):
            reasons.append(dict(rule=rule, explanation=explanation,
                                evidence_event_ids=[e['event_id'] for e in evidence], category=category))

        if self.previous is None:
            reason('stream_start', '输入起点，不代表原轨迹阶段起点。', [], 'stream_bookkeeping')
        else:
            previous = self.previous
            if starts:
                reason('new_task_in_progress_marker', '新的显式任务开始标记；仍待语义裁定。', starts, 'explicit_task_marker')
            created = [e for e in events if e['kind'] == 'task_created']
            if created:
                reason('task_plan_created', '实际创建任务清单，不代表已执行其中任务。', created, 'explicit_task_marker')
            failures = [e for e in previous['events'] if e['kind'] == 'tool_error_observed'
                        or e['kind'] == 'test_summary_observed' and e['attributes'].get('failed', 0) > 0]
            if failures:
                reason('after_observed_failure', '前一步出现失败；新片段可能仍延续同一修复目标。', failures, 'observed_result')
            if facts['workflow_mode'] in EXECUTION and facts['workflow_mode'] != previous['workflow_mode']:
                reason('execution_workflow_started', '开始测试/构建/程序执行；不是必然的新目标。',
                       [e for e in events if e['kind'] == 'tool_call'], 'workflow_heuristic')
            elif facts['workflow_mode'] == 'modification_attempt' and previous['workflow_mode'] in EXECUTION:
                reason('modification_after_execution', '执行程序后转入修改，可能是修复或新实现。',
                       [e for e in events if e['kind'] == 'tool_call'], 'workflow_heuristic')
            intent = [e for e in events if e['kind'] == 'intent_cue']
            if (facts['modified_targets'] and previous['modified_targets']
                    and not set(facts['modified_targets']) & set(previous['modified_targets']) and intent):
                reason('intent_and_changed_edit_target', '转向措辞与修改对象变化同时出现；弱候选，不按换文件单独切片。',
                       intent + [e for e in events if e['kind'] == 'tool_call'], 'weak_joint_hint')
            if facts['workflow_mode'] == 'assistant_only' and previous['workflow_mode'] != 'assistant_only':
                reason('assistant_only_reply', '进入无工具的回复，不能据此认定任务完成。',
                       [e for e in events if e['kind'] == 'assistant_only_reply'], 'workflow_heuristic')
            if len(self.slices[-1]['steps']) >= self.max_steps:
                reason('budget_limit', '片段步数预算切分，不是语义边界。', [], 'budget_only')
        if reasons:
            if self.slices:
                self.slices[-1]['closed_when_observed_step'] = turn
                self.slices[-1]['close_reason'] = 'next_candidate_started'
            self.slices.append(dict(candidate_id=f'C{len(self.slices)+1:03}', steps=[], start_reasons=reasons,
                                    semantic_stage_id=None, semantic_status='not_assessed',
                                    close_reason=None, closed_when_observed_step=None))
        current = self.slices[-1]
        current['steps'].append(turn)
        facts.update(candidate_id=current['candidate_id'], candidate_boundary_before=bool(reasons) and self.previous is not None,
                     start_reasons=reasons, active_task_ids=sorted(self.active),
                     active_task_hints={tid: copy.deepcopy(self.tasks[tid]) for tid in sorted(self.active)},
                     available_through_step=turn,
                     review_flags=(['multiple_active_task_markers'] if len(self.active) > 1 else [])
                         + (['multiple_tool_families'] if facts['workflow_mode'] == 'mixed' else []))
        self.decisions.append(copy.deepcopy(facts))
        self.previous = copy.deepcopy(facts)
        return copy.deepcopy(facts)


def verify_citations(events, raw):
    count = 0
    for event in events:
        for cite in event['citations']:
            source = cite['source']
            record = raw[source['trace_line']-1]
            text = resolve(record, source['pointer'])
            begin, end = source['character_span']
            require(record['turn'] == source['step'] == event['step']
                    and record['message_index'] == source['message_index']
                    and text[begin:end] == cite['quote'], 'Invalid raw citation')
            count += 1
    return count


def render(output, decisions, slices):
    by_id = {c['candidate_id']: [d for d in decisions if d['candidate_id'] == c['candidate_id']] for c in slices}
    fields = ['window_id', 'candidate_id', 'steps', 'task_hints', 'workflow_modes', 'start_rules',
              'modified_targets', 'observed_test_summaries', 'semantic_status']
    summary_rows = []
    for candidate in slices:
        rows = by_id[candidate['candidate_id']]
        hints = {tid: value.get('subject') for r in rows for tid, value in r['active_task_hints'].items()}
        summary_rows.append(dict(window_id=candidate['window_id'], candidate_id=candidate['candidate_id'],
            steps=f'{candidate["steps"][0]}–{candidate["steps"][-1]}', task_hints=json.dumps(hints, ensure_ascii=False),
            workflow_modes=' / '.join(dict.fromkeys(LABELS[r['workflow_mode']] for r in rows)),
            start_rules=' / '.join(r['rule'] for r in candidate['start_reasons']),
            modified_targets='; '.join(sorted({p for r in rows for p in r['modified_targets']})),
            observed_test_summaries=json.dumps([e['attributes'] for r in rows for e in r['events']
                                               if e['kind'] == 'test_summary_observed'], ensure_ascii=False),
            semantic_status='未评估，不是最终阶段'))
    with (output/'slices.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    intro = ('规则候选切片预览：仅结构/工作流候选，不是语义阶段、准确率或完成判定。'
             '任务标题和状态来自 Agent 的工具操作，不是独立验收。'
             'JSONL 中逐 Step 决策为因果快照；完整片段表只供事后查看。')
    parts = ['<!doctype html><meta charset="utf-8"><title>规则候选切片</title>',
        '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap;max-width:1000px}summary{cursor:pointer}</style>',
        '<h1>规则候选切片</h1><p>'+html.escape(intro)+'</p>',
        '<table><tr><th>候选</th><th>Step</th><th>声明任务提示</th><th>工具工作流</th><th>触发规则</th><th>事件与原文坐标</th></tr>']
    md = ['# 规则候选切片预览', '', intro, '', '| 候选 | Step | 工具工作流 | 触发规则 |', '|---|---|---|---|']
    for candidate, row in zip(slices, summary_rows):
        rows = by_id[candidate['candidate_id']]
        cells = [row[k] for k in ('candidate_id', 'steps', 'task_hints', 'workflow_modes', 'start_rules')]
        details = ''.join('<details><summary>Step '+str(r['step'])+'：'+html.escape(LABELS[r['workflow_mode']])+
                          '</summary><pre>'+html.escape(json.dumps(r, ensure_ascii=False, indent=2))+'</pre></details>' for r in rows)
        parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'<td>'+details+'</td></tr>')
        md.append(f'| {row["candidate_id"]} | {row["steps"]} | {row["workflow_modes"]} | {row["start_rules"]} |')
    (output/'preview.html').write_text('\n'.join(parts)+'</table>')
    (output/'preview.md').write_text('\n'.join(md)+'\n')


def run(bundle, dataset, output, max_steps=12):
    started = time.perf_counter()
    require(not output.exists(), 'Output already exists; use a new directory')
    manifest = json.loads((dataset.parent/'manifest.json').read_text())
    require(digest(dataset) == manifest['dataset_sha256'], 'Frozen dataset hash differs')
    require(digest(bundle/'index.json') == manifest['source_hashes']['index.json'], 'Source index hash differs')
    selected = [json.loads(line) for line in dataset.read_text().splitlines()]
    index = {r['trajectory_id']: r for r in json.loads((bundle/'index.json').read_text())}
    raw_by_trace, source_hashes, slicers = {}, {}, {}
    decisions, events = [], []
    cited = 0
    for step in selected:
        item = index[step['trajectory_id']]
        path = bundle/item['trace_path']
        if item['trace_path'] not in raw_by_trace:
            source_hashes[item['trace_path']] = digest(path)
            require(source_hashes[item['trace_path']] == manifest['source_hashes'][item['trace_path']], 'Trace hash differs')
            raw_by_trace[item['trace_path']] = [json.loads(line) for line in path.read_text().splitlines()]
        raw = raw_by_trace[item['trace_path']]
        records = [(i, r) for i, r in enumerate(raw, 1)
                   if r.get('record_type') == 'message' and r.get('turn') == step['step']]
        require(records[0][1]['original']['role'] == 'assistant'
                and all(r['original']['role'] != 'assistant' for _, r in records[1:]), 'Step includes another assistant reply')
        reconstructed = extract_step(item, step['step'], records)
        require(all(reconstructed[k] == step[k] for k in reconstructed), 'Frozen Step/source differs')
        slicer = slicers.setdefault(step['window_id'], CandidateSlicer(max_steps))
        facts = extract_events(item, step['step'], records)
        cited += verify_citations(facts['events'], raw)
        decision = slicer.observe(facts)
        # Candidate IDs are globally unique, while each window retains independent state.
        prefix = f'W{list(slicers).index(step["window_id"])+1:03}-'
        decision['candidate_id'] = prefix + decision['candidate_id']
        decision.update(window_id=step['window_id'], trajectory_id=step['trajectory_id'])
        decisions.append(decision)
        events.extend(dict(e, window_id=step['window_id'], trajectory_id=step['trajectory_id']) for e in facts['events'])
    slices = []
    for ordinal, (wid, slicer) in enumerate(slicers.items(), 1):
        for current in slicer.slices:
            candidate = copy.deepcopy(current)
            candidate.update(candidate_id=f'W{ordinal:03}-'+candidate['candidate_id'], window_id=wid,
                             posthoc_display_only=True,
                             available_through_step=candidate['closed_when_observed_step'] or slicer.previous['step'])
            if candidate['close_reason'] is None:
                candidate['close_reason'] = 'input_exhausted_not_task_completed'
            slices.append(candidate)
    require(digest(dataset) == manifest['dataset_sha256'], 'Dataset changed during extraction')
    require(all(digest(bundle/name) == value for name, value in source_hashes.items()), 'Source changed during extraction')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('steps.jsonl', decisions), ('events.jsonl', events), ('slices.jsonl', slices)]:
        (output/name).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    render(output, decisions, slices)
    summary = dict(version=VERSION, model_calls=0, no_trace_commands_executed=True,
        steps=len(decisions), windows=len(slicers), candidate_slices=len(slices),
        candidate_boundaries=sum(d['candidate_boundary_before'] for d in decisions),
        event_counts=dict(Counter(e['kind'] for e in events)), citations_exactly_verified=cited,
        tool_calls=sum(d['tool_calls'] for d in decisions), tool_returns=sum(d['tool_returns'] for d in decisions),
        max_steps_per_slice=max_steps, dataset_sha256=digest(dataset), source_sha256=source_hashes,
        source_index_sha256=digest(bundle/'index.json'), slicer_sha256=digest(__file__),
        rule_selection='Heuristic demonstration developed with this already-reviewed pilot; not held-out validation.',
        input_scope='Full original messages for selected Steps, not the previous bounded model text; not equal-budget comparison.',
        causal_scope='Only current/past selected messages enter rules. No history hydration outside each window.',
        model_semantic_stages_assessed=False, human_gold_accuracy=None, boundary_f1=None,
        local_end_to_end_seconds=time.perf_counter()-started)
    summary['output_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    save(output/'manifest.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-steps-per-slice', type=int, default=12)
    args = parser.parse_args()
    run(args.bundle, args.dataset, args.output, args.max_steps_per_slice)


if __name__ == '__main__':
    main()
