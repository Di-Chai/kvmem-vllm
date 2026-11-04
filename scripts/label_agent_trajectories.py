"""Label saved agent histories using protocol fields and recorded harness events.

This is a read-only export of the source traces. It does not infer the semantic
purpose of prose, commands, or tool output, and does not invent token offsets.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import zipfile


LABELS = {
    'task_instruction': '原始任务全文',
    'framework_rules': '框架规则',
    'tool_definitions': '工具定义',
    'thinking': 'Thinking',
    'assistant_text': '普通 assistant 正文',
    'user_text': '未进一步细分的 user 正文',
    'analysis': '显式 analysis',
    'plan': '显式 plan',
    'tool_call': '工具调用请求',
    'command_request': '终端命令请求',
    'tool_result': '工具返回',
    'parser_feedback': '解析错误反馈',
    'parser_warning': '格式警告',
    'terminal_observation': '终端观察',
    'execution_status': '执行状态',
    'progress_operation': '结构化任务进度操作',
    'completion_declaration': '完成声明',
    'completion_confirmation': '完成确认提示',
}
CONFIRMATION = (
    'Are you sure you want to mark the task as complete? This will trigger '
    "your solution to be graded and you won't be able to make any further "
    'corrections. If so, include "task_complete": true in your JSON response again.'
)
CANARY = re.compile(r'^(<!--.*canary.*-->|#.*canary.*)$', re.I)
CONTROL = re.compile(r'(?:C-[cdz]\s*)+')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotation(kind, path, evidence, *, span=None, **extra):
    result = dict(kind=kind, label=LABELS[kind], path=path, evidence=evidence, **extra)
    if span is not None:
        result['character_span'] = list(span)
    return result


def strip_canary(text):
    lines = text.split('\n')
    index = 0
    while index < len(lines) and CANARY.match(lines[index].strip()):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return '\n'.join(lines[index:])


def feedback_labels(text):
    """Recognize the frozen harness's framing, not arbitrary semantic keywords.

    Used only on the trusted recorded feedback positions of this harness.
    Nested subranges can overlap the enclosing terminal observation.
    """
    result = []
    if text.startswith('Previous response had parsing errors:\n'):
        return [annotation('parser_feedback', '/original/content',
                           'frozen_harness_feedback_frame', span=(0, len(text)))]
    terminal_start = 0
    if text.startswith('Previous response had warnings:\n'):
        points = [text.find(marker) for marker in
                  ('New Terminal Output:', 'Current terminal state:', 'Current Terminal Screen:')]
        terminal_start = min((p for p in points if p >= 0), default=len(text))
        result.append(annotation('parser_warning', '/original/content',
                                 'frozen_harness_feedback_frame', span=(0, terminal_start)))
    terminal_text = text[terminal_start:]
    if terminal_text.startswith(('New Terminal Output:', 'Current terminal state:',
                                 'Current Terminal Screen:')):
        end = len(text)
        if text.endswith(CONFIRMATION):
            end -= len(CONFIRMATION)
            result.append(annotation('completion_confirmation', '/original/content',
                                     'frozen_harness_completion_frame', span=(end, len(text))))
        result.append(annotation('terminal_observation', '/original/content',
                                 'frozen_harness_terminal_frame', span=(terminal_start, end)))
        status = text.rfind('\n[Executor status]\n', terminal_start, end)
        if status >= 0:
            result.append(annotation('execution_status', '/original/content',
                                     'frozen_executor_appended_status', span=(status + 1, end)))
    if not result:
        result.append(annotation('user_text', '/original/content', 'wire_message_role'))
    return result


def load_parser(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Parser source differs from the frozen run manifest')
    spec = importlib.util.spec_from_file_location('_trace_label_frozen_parser', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Parser(module.TerminusJSONPlainParser):
        # The frozen terminalbench_harness.py applies precisely these checks.
        def _parse_commands(self, data, warnings):
            for i, value in enumerate(data):
                if isinstance(value, dict) and isinstance(value.get('keystrokes'), str) and '\x00' in value['keystrokes']:
                    return [], f'Command {i+1}: keystrokes contains a literal NUL, which cannot be sent through the terminal transport. Use textual shell escapes to represent binary bytes.'
                if isinstance(value, dict) and isinstance(value.get('duration'), (int, float)):
                    if not math.isfinite(value['duration']) or value['duration'] < 0:
                        return [], f'Command {i+1}: duration must be finite and nonnegative'
            parsed, error = super()._parse_commands(data, warnings)
            indices = [i+1 for i, value in enumerate(data) if isinstance(value, dict)
                       and isinstance(value.get('keystrokes'), str) and CONTROL.fullmatch(value['keystrokes'])]
            warnings[:] = [w for w in warnings if not any(
                w.startswith(f'Command {i} should end with newline') for i in indices)]
            return parsed, error
    return Parser()


def label_assistant(message, parser, following=None):
    result = [annotation('assistant_text', '/original/content', 'wire_content_field')]
    if message.get('reasoning_content'):
        result.append(annotation('thinking', '/original/reasoning_content', 'wire_reasoning_content_field'))
    parsed = parser.parse_response(message.get('content') or '')
    derived = dict(analysis=parsed.analysis, plan=parsed.plan,
                   commands=[dict(keystrokes=c.keystrokes, duration=min(c.duration, 60))
                             for c in parsed.commands],
                   task_complete=parsed.is_task_complete, error=parsed.error,
                   warning=parsed.warning)
    observed_error = None
    if following is not None:
        kinds = {a['kind'] for a in feedback_labels(following)}
        if 'parser_feedback' in kinds:
            observed_error = True
        elif 'terminal_observation' in kinds:
            observed_error = False
    mismatch = observed_error is not None and observed_error != bool(parsed.error)
    derived['replay_matches_observed_feedback'] = None if observed_error is None else not mismatch
    if not parsed.error and not mismatch:
        for key in ('analysis', 'plan'):
            if derived[key]:
                result.append(annotation(key, '/parsed/' + key, 'sha_verified_frozen_parser'))
        for i, command in enumerate(derived['commands']):
            mode = 'wait' if not command['keystrokes'] else (
                'control_key' if CONTROL.fullmatch(command['keystrokes']) else 'terminal_input')
            result.append(annotation('command_request', f'/parsed/commands/{i}',
                                     'sha_verified_frozen_parser', action=mode))
        if derived['task_complete']:
            result.append(annotation('completion_declaration', '/parsed/task_complete',
                                     'sha_verified_frozen_parser'))
    return result, derived, mismatch


def label_deepswe(message, *, first_user=False, task_text='', calls=None):
    calls = calls if calls is not None else {}
    result = []
    content = message.get('content', '')
    blocks = content if isinstance(content, list) else [dict(type='text', text=content)]
    for index, block in enumerate(blocks):
        prefix = f'/original/content/{index}' if isinstance(content, list) else '/original/content'
        kind = block.get('type')
        if kind == 'thinking':
            result.append(annotation('thinking', prefix + '/thinking', 'content_block.type'))
        elif kind == 'tool_use':
            calls[block['id']] = block['name']
            result.append(annotation('tool_call', prefix, 'content_block.type',
                                     call_id=block['id'], tool=block['name']))
            if block['name'] in ('TaskCreate', 'TaskUpdate'):
                result.append(annotation('progress_operation', prefix, 'tool_use.name',
                                         call_id=block['id'], tool=block['name']))
        elif kind == 'tool_result':
            result.append(annotation('tool_result', prefix, 'content_block.type',
                                     call_id=block['tool_use_id'], tool=calls.get(block['tool_use_id']),
                                     is_error=block.get('is_error')))
        elif kind == 'text':
            text = block.get('text', '')
            path = prefix + '/text' if isinstance(content, list) else prefix
            result.append(annotation('assistant_text' if message['role'] == 'assistant' else 'user_text',
                                     path, 'wire_message_role'))
            if first_user and task_text:
                # The CLI strips the prompt file's final line break. Match its
                # content exactly; never normalize words or internal whitespace.
                candidate = task_text if task_text in text else task_text.rstrip('\r\n')
                if candidate and text.count(candidate) == 1:
                    begin = text.index(candidate)
                    result.append(annotation('task_instruction', path, 'exact_task_artifact_content_match',
                        span=(begin, begin + len(candidate)),
                        source_transform='none' if candidate == task_text else 'remove_trailing_line_breaks'))
    return result


def outcome(labels, parsed, following):
    kinds = {a['kind'] for a in labels}
    next_kinds = {a['kind'] for a in feedback_labels(following)} if following is not None else set()
    if 'parser_feedback' in next_kinds:
        return 'format_retry'
    if 'completion_confirmation' in next_kinds:
        return 'completion_confirmation'
    if 'tool_call' in kinds:
        return 'tool_call_then_return' if following is not None else 'tool_request_no_followup'
    if 'terminal_observation' in next_kinds:
        modes = [a['action'] for a in labels if a['kind'] == 'command_request']
        if not modes:
            return 'observation_only'
        if all(mode == 'wait' for mode in modes):
            return 'wait_then_observation'
        return 'command_then_observation'
    if following is None:
        return 'completion_declaration_no_followup' if parsed and parsed.get('task_complete') else 'no_followup_observed'
    return 'other_followup'


def longest_run(rows, target):
    best = current = 0
    start = best_start = None
    for row in rows:
        if row['outcome'] == target:
            if current == 0:
                start = row['turn']
            current += 1
            if current > best:
                best, best_start = current, start
        else:
            current = 0
    return dict(length=best, first_turn=best_start)


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def export(manifest_path, output, parser_path):
    sampled = read_json(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = {'deepswe_seed1001': 'deepswe', 'terminalbench_thinking': 'terminalbench'}
    summary = dict(schema_version=1, scope='Previously sampled 16 DeepSWE and 86 Terminal Bench histories',
                   coordinates='JSON pointers; character spans are Unicode code points, half-open; no token offsets inferred',
                   counting='One final saved complete request per session, plus its saved response; no repeated prefix counting',
                   groups={}, trajectories=[], validation=dict(parser_feedback_mismatches=[]))
    all_rows, sources, examples = [], [], []
    for key, family in groups.items():
        directory = output / family
        directory.mkdir()
        label_counts, outcomes, transitions = Counter(), Counter(), Counter()
        audits_by_session = defaultdict(list)
        parser = None
        selected = sampled[key]['selected']
        if family == 'terminalbench':
            root = Path(selected[0]['request']).parents[3]
            frozen = read_json(root / 'manifest.json')
            parser = load_parser(parser_path, frozen['harness_repair']['upstream_sources']['terminus_json_plain_parser.py'])
            audit = root / 'formal-harness.jsonl'
            with audit.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    event = json.loads(line)
                    if event.get('event') == 'command':
                        audits_by_session[event['session']].append((line_number, event))
            summary['parser'] = dict(path=str(parser_path), sha256=sha(parser_path),
                                     harness_overlay=str(root / 'runtime/scripts/terminalbench_harness.py'),
                                     harness_overlay_sha256=sha(root / 'runtime/scripts/terminalbench_harness.py'))
        for selection in selected:
            request_path = Path(selection['request'])
            request = read_json(request_path)
            messages = [(dict(original=m), dict(file=str(request_path), pointer=f'/messages/{i}'))
                        for i, m in enumerate(request['messages'])]
            response_path = request_path.parent / 'response.json'
            source = dict(family=family, session=selection['session'], request=str(request_path),
                          request_sha256=sha(request_path))
            if response_path.exists():
                response = read_json(response_path)
                assistant = response['choices'][0]['message'] if family == 'terminalbench' else dict(
                    role='assistant', content=response['content'])
                messages.append((dict(original=assistant), dict(file=str(response_path),
                                pointer='/choices/0/message' if family == 'terminalbench' else '')))
                source.update(response=str(response_path), response_sha256=sha(response_path))
            root = request_path.parents[3]
            task_id = request.get('kvmem_query_provenance', {}).get('task_id', selection['session'])
            task_file = root / ('dataset/tasks/' + task_id + '/instruction.md' if family == 'terminalbench'
                                else 'cases/' + task_id + '/prompt.md')
            task_text = strip_canary(task_file.read_text()) if family == 'terminalbench' else task_file.read_text()
            source.update(task_id=task_id, task_file=str(task_file), task_sha256=sha(task_file))
            sources.append(source)
            calls, local_counts, rows = {}, Counter(), []
            turn, first_user = 0, True
            destination = directory / (task_id + '.jsonl')
            with destination.open('w') as stream:
                write_line(stream, dict(record_type='provenance', **source))
                for field, kind in [('system', 'framework_rules'), ('tools', 'tool_definitions')]:
                    if request.get(field):
                        item = dict(record_type='context', original=request[field],
                                    source=dict(file=str(request_path), pointer='/' + field),
                                    labels=[annotation(kind, '/original', 'wire_request_field')])
                        write_line(stream, item)
                        label_counts[kind] += 1
                        local_counts[kind] += 1
                for index, (item, location) in enumerate(messages):
                    message = item['original']
                    following = None
                    following_message = None
                    if message['role'] == 'assistant':
                        turn += 1
                        if index + 1 < len(messages) and messages[index+1][0]['original']['role'] == 'user':
                            following_message = messages[index+1][0]['original']
                            following = following_message.get('content')
                            if not isinstance(following, str):
                                following = json.dumps(following, ensure_ascii=False)
                    if family == 'deepswe':
                        item['labels'] = label_deepswe(message, first_user=first_user,
                                                       task_text=task_text, calls=calls)
                    elif message['role'] == 'assistant':
                        item['labels'], item['parsed'], mismatch = label_assistant(message, parser, following)
                        if mismatch:
                            summary['validation']['parser_feedback_mismatches'].append(
                                dict(task_id=task_id, turn=turn, source=location))
                    elif first_user:
                        text = message['content']
                        item['labels'] = [annotation('user_text', '/original/content', 'wire_message_role')]
                        if task_text and text.count(task_text) == 1:
                            start = text.index(task_text)
                            item['labels'].append(annotation('task_instruction', '/original/content',
                                'exact_task_artifact_match', span=(start, start + len(task_text))))
                            item['labels'].append(annotation('framework_rules', '/original/content',
                                'frozen_initial_prompt_prefix', span=(0, start)))
                    else:
                        item['labels'] = feedback_labels(message.get('content') or '')
                    if message['role'] == 'user':
                        first_user = False
                    item.update(record_type='message', message_index=index, turn=turn,
                                role=message['role'], source=location)
                    for label in item['labels']:
                        label_counts[label['kind']] += 1
                        local_counts[label['kind']] += 1
                    if message['role'] == 'assistant':
                        state = outcome(item['labels'], item.get('parsed'), following)
                        if family == 'deepswe' and state == 'tool_call_then_return':
                            requested = {a['call_id'] for a in item['labels'] if a['kind'] == 'tool_call'}
                            returned = {b['tool_use_id'] for b in following_message.get('content', [])
                                        if isinstance(b, dict) and b.get('type') == 'tool_result'}
                            item['observed_result_call_ids'] = sorted(returned)
                            if not requested.issubset(returned):
                                state = 'tool_call_without_complete_observed_returns'
                        item['outcome'] = state
                        row = dict(family=family, task_id=task_id, session=selection['session'], turn=turn,
                                   message_index=index, labels='|'.join(dict.fromkeys(a['kind'] for a in item['labels'])),
                                   outcome=state, command_requests=sum(a['kind'] == 'command_request' for a in item['labels']),
                                   tool_calls=sum(a['kind'] == 'tool_call' for a in item['labels']),
                                   has_observed_followup=following is not None)
                        rows.append(row)
                        outcomes[state] += 1
                        if len(rows) > 1:
                            transitions[(rows[-2]['outcome'], state)] += 1
                    write_line(stream, item)
                if family == 'terminalbench':
                    session_id = request['kvmem_query_provenance']['session_id']
                    env_session = session_id.removesuffix('__agent') + '__env'
                    for line_number, event in audits_by_session.get(env_session, []):
                        write_line(stream, dict(record_type='execution_event', original=event,
                                   source=dict(file=str(audit), line=line_number),
                                   labels=[annotation('execution_status', '/original', 'recorded_executor_event')],
                                   attribution='session only; no invented per-turn/per-command output alignment'))
                        label_counts['execution_status'] += 1
                        local_counts['execution_status'] += 1
            all_rows.extend(rows)
            record = dict(family=family, task_id=task_id, session=selection['session'],
                          labelled_file=str(destination.resolve()), turns=turn, messages=len(messages),
                          recorded_request_directories=selection['turns'], label_counts=dict(local_counts),
                          outcomes=dict(Counter(row['outcome'] for row in rows)),
                          longest_format_retry_run=longest_run(rows, 'format_retry'))
            summary['trajectories'].append(record)
            if family == 'deepswe' and not any(e['family'] == family for e in examples):
                examples.append(dict(family=family, task_id=task_id, first_turns=rows[:4]))
            if family == 'terminalbench' and task_id in ('extract-elf', 'make-mips-interpreter', 'break-filter-js-from-html'):
                examples.append(dict(family=family, task_id=task_id, first_turns=rows[:5], last_turns=rows[-4:]))
        summary['groups'][family] = dict(trajectories=len(selected),
            turns=sum(r['turns'] for r in summary['trajectories'] if r['family'] == family),
            labels=dict(label_counts), outcomes=dict(outcomes),
            transitions=[dict(before=a, after=b, count=n) for (a, b), n in transitions.most_common()])
    summary['validation']['source_requests_read'] = len(sources)
    changed = []
    checked = 0
    for source in sources:
        for field in ('request', 'response', 'task'):
            path_field = 'task_file' if field == 'task' else field
            if path_field in source:
                checked += 1
                if sha(source[path_field]) != source[field + '_sha256']:
                    changed.append(source[path_field])
    summary['validation']['source_hashes_rechecked'] = checked
    summary['validation']['sources_modified'] = bool(changed)
    summary['validation']['changed_sources'] = changed
    summary['validation']['task_instruction_matches'] = sum(
        record['label_counts'].get('task_instruction', 0) for record in summary['trajectories'])
    summary['examples'] = examples
    (output / 'sampled-selection.json').write_text(json.dumps(
        {key: dict(selected=sampled[key]['selected']) for key in groups}, ensure_ascii=False, indent=2) + '\n')
    (output / 'input-manifest.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
    (output / 'labels.json').write_text(json.dumps(dict(labels=LABELS,
        notes=['Labels may overlap: progress_operation is also a tool_call; status is part of an observation.',
               'Replayed JSON fields are derived fields, not spans in the original JSON string.',
               'Completion declarations do not assert success; no-followup does not assert termination.',
               'Tool and terminal output is not semantically classified.',
               'Execution events have session attribution only; a stale exit_code is not proof of completion.']),
        ensure_ascii=False, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'turns.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with zipfile.ZipFile(output / 'labelled-traces.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(output))
    print(json.dumps(dict(output=str(output.resolve()), groups=summary['groups'],
                          validation=summary['validation']), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terminal-parser', type=Path, required=True)
    args = parser.parse_args()
    export(args.manifest, args.output, args.terminal_parser)


if __name__ == '__main__':
    main()
