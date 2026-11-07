"""Protocol/field slicing of saved DeepSWE and Terminal Bench messages.

The 17 requested types describe structural origin, not semantic execution stages.
Frozen parser replay sees only the current reply. Session-only executor audit
records are exported separately, never assigned an invented Step or timestamp.
"""
import argparse
from collections import Counter
import csv
import html
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

from observer import digest, resolve, save


TYPES = {
    'framework_rules': '框架规则', 'tool_definitions': '工具定义',
    'task_instruction': '原始任务', 'thinking': 'Thinking',
    'assistant_text': '普通正文', 'analysis': '显式分析', 'plan': '显式计划',
    'tool_call': '工具调用', 'command_request': '命令请求',
    'progress_operation': '任务进度操作', 'tool_result': '工具返回',
    'terminal_observation': '终端观察', 'parser_feedback': '解析错误反馈',
    'parser_warning': '格式警告', 'execution_status': '执行状态',
    'completion_declaration': '完成声明', 'completion_confirmation': '完成确认提示',
}
DEFAULT_TRACES = ['deepswe/seed-1004/superjson-error-stack-serialization', 'terminalbench/dna-assembly']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rules(bundle):
    manifest = json.loads((bundle/'MANIFEST.json').read_text())
    expected = {entry['path']: entry['sha256'] for entry in manifest['files']}
    paths = ['methodology/label_agent_trajectories.py', 'methodology/frozen/terminus_json_plain_parser.py']
    for path in paths:
        require(digest(bundle/path) == expected[path], f'Frozen method hash differs: {path}')
    # Import definitions without creating bytecode files inside the source bundle.
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('_observer_structure_frozen_rules', bundle/paths[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        parser = module.load_parser(bundle/paths[1], expected[paths[1]])
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module, parser, expected


def top_level_fields(text):
    """Find raw JSON value lexemes, not decoded-string character offsets.

    Handles nested objects, escapes and duplicate keys. Caller chooses the last
    occurrence, matching json.loads/frozen-parser duplicate-key behavior.
    """
    decoder = json.JSONDecoder()
    match = re.match(r'\s*\{', text)
    require(match is not None, 'Not a JSON object')
    cursor, fields = match.end(), []
    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        require(cursor < len(text), 'Incomplete JSON object')
        if text[cursor] == '}':
            cursor += 1
            require(not text[cursor:].strip(), 'Trailing non-whitespace JSON content')
            return fields
        key, end = decoder.raw_decode(text, cursor)
        require(isinstance(key, str), 'Non-string JSON key')
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        require(cursor < len(text) and text[cursor] == ':', 'Missing colon')
        cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        value, end = decoder.raw_decode(text, cursor)
        fields.append(dict(key=key, begin=cursor, end=end, value=value))
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        require(cursor < len(text) and text[cursor] in ',}', 'Invalid JSON separator')
        if text[cursor] == ',':
            cursor += 1


def partition(text, annotations, fallback):
    """Non-overlapping raw-string regions with specific children overriding parents."""
    normalized = []
    for ordinal, annotation in enumerate(annotations):
        begin, end = annotation.get('character_span', [0, len(text)])
        require(0 <= begin <= end <= len(text), 'Annotation span outside raw string')
        kind = annotation['kind']
        normalized.append(dict(begin=begin, end=end, kind=kind, ordinal=ordinal,
                               specificity={'user_text': 0, 'terminal_observation': 1,
                                            'execution_status': 3}.get(kind, 2)))
    boundaries = sorted({0, len(text), *(p for a in normalized for p in (a['begin'], a['end']))})
    output = []
    for begin, end in zip(boundaries, boundaries[1:]):
        if begin == end:
            continue
        active = [a for a in normalized if a['begin'] <= begin and end <= a['end']]
        winner = max(active, key=lambda a: (a['specificity'], -(a['end']-a['begin']), a['ordinal'])) if active else None
        kind = winner['kind'] if winner else fallback
        if kind == 'user_text':
            kind = fallback
        also = sorted({a['kind'] for a in active if a['kind'] != kind and a['kind'] != 'user_text'})
        if output and output[-1]['kind'] == kind and output[-1]['also_types'] == also and output[-1]['end'] == begin:
            output[-1]['end'] = end
        else:
            output.append(dict(kind=kind, begin=begin, end=end, also_types=also))
    require(''.join(text[r['begin']:r['end']] for r in output) == text, 'Partition lost raw text')
    return output


class StructureAdapter:
    def __init__(self, rules, parser):
        self.rules, self.parser = rules, parser
        self.calls = {}
        self.first_user = True
        self.parse_audits = []

    def record_blocks(self, item, record, line, task_text):
        blocks = []
        location = dict(trace_path=item['trace_path'], trace_line=line,
                        message_index=record.get('message_index'), step=record.get('turn'))

        def add(kind, pointer, value=None, span=None, also=None, layer='original_field', **metadata):
            raw = resolve(record, pointer)
            if span is not None:
                require(isinstance(raw, str) and 0 <= span[0] <= span[1] <= len(raw), 'Invalid raw span')
                raw = raw[span[0]:span[1]]
            if value is not None:
                require(raw == value, 'Block does not equal located source value')
            source = dict(location, pointer=pointer, character_span=span)
            blocks.append(dict(family=item['family'], trajectory_id=item['trajectory_id'],
                role=record.get('role'),
                structural_type=kind, supported_type=kind in TYPES, also_types=also or [],
                layer=layer, source=source, content=raw,
                available_through_step=record.get('turn', 0) if record['record_type'] != 'execution_event' else None,
                semantic_stage_id=None, semantic_status='not_assessed', **metadata))

        record_type = record['record_type']
        if record_type == 'context':
            kind = {'/system': 'framework_rules', '/tools': 'tool_definitions'}.get(record['source']['pointer'])
            require(kind is not None, 'Unknown context source field')
            add(kind, '/original', basis='request_field_not_reconstructed_rendered_tokens')
        elif record_type == 'execution_event':
            add('execution_status', '/original', layer='session_audit',
                attribution='session_only_no_invented_step_alignment')
        elif record_type == 'message' and item['family'] == 'deepswe':
            original = record['original']
            labels = self.rules.label_deepswe(original, first_user=self.first_user,
                                               task_text=task_text, calls=self.calls)
            grouped = {}
            for label in labels:
                grouped.setdefault(label['path'], []).append(label)
            for pointer, annotations in grouped.items():
                value = resolve(record, pointer)
                if isinstance(value, str):
                    for region in partition(value, annotations, 'unclassified_input'):
                        add(region['kind'], pointer, span=[region['begin'], region['end']],
                            also=region['also_types'], basis='content_block_role_or_exact_task_match')
                else:
                    types = {a['kind'] for a in annotations}
                    kind = 'progress_operation' if 'progress_operation' in types else annotations[0]['kind']
                    if kind == 'tool_result' and isinstance(value, dict) and isinstance(value.get('content'), str):
                        add(kind, pointer+'/content', span=[0, len(value['content'])],
                            basis='structured_tool_return', call_id=value.get('tool_use_id'),
                            tool=self.calls.get(value.get('tool_use_id')), is_error=value.get('is_error'))
                    else:
                        add(kind, pointer, also=sorted(types-{kind}), basis='structured_content_block',
                            call_id=value.get('id') if isinstance(value, dict) else None)
            if original['role'] == 'user':
                self.first_user = False
        elif record_type == 'message' and item['family'] == 'terminalbench':
            original = record['original']
            if original['role'] == 'assistant':
                if original.get('reasoning_content'):
                    add('thinking', '/original/reasoning_content', span=[0, len(original['reasoning_content'])],
                        basis='wire_reasoning_content_not_keyword_detection')
                # Never pass following feedback: old export label gating used it.
                _, parsed, _ = self.rules.label_assistant(original, self.parser, following=None)
                core = {k: v for k, v in parsed.items() if k != 'replay_matches_observed_feedback'}
                stored = record.get('parsed', {})
                require(all(stored.get(k) == v for k, v in core.items()), 'Frozen current-reply parser replay differs')
                self.parse_audits.append(dict(step=record['turn'], trace_line=line,
                    accepted=not bool(parsed['error']), parser_error=parsed['error'], warning=parsed['warning'],
                    derived_fields_match_saved_core=True, following_feedback_not_read=True))
                content = original.get('content') or ''
                if parsed['error']:
                    add('assistant_text', '/original/content', span=[0, len(content)],
                        basis='parse_failed_no_executable_fields_invented', parser_error=parsed['error'])
                else:
                    self._tb_body(content, parsed, record, add)
            elif self.first_user:
                text = original['content']
                candidate = self.rules.strip_canary(task_text)
                require(candidate and text.count(candidate) == 1, 'Initial task artifact match is not unique')
                begin = text.index(candidate)
                annotations = [dict(kind='framework_rules', character_span=[0, begin]),
                               dict(kind='task_instruction', character_span=[begin, begin+len(candidate)])]
                for region in partition(text, annotations, 'unclassified_input'):
                    add(region['kind'], '/original/content', span=[region['begin'], region['end']],
                        also=region['also_types'], basis='frozen_prefix_and_exact_task_artifact')
                self.first_user = False
            else:
                text = original.get('content') or ''
                for region in partition(text, self.rules.feedback_labels(text), 'unclassified_input'):
                    add(region['kind'], '/original/content', span=[region['begin'], region['end']],
                        also=region['also_types'], basis='frozen_feedback_position_and_framing')
        return blocks

    def _tb_body(self, content, parsed, record, add):
        raw_json, _ = self.parser._extract_json_content(content)
        fields = None
        try:
            fields = top_level_fields(raw_json)
        except (ValueError, json.JSONDecodeError):
            pass
        if fields is None:
            # Auto-repaired parsing may have no exact raw JSON field boundaries.
            add('unmapped_wire_container', '/original/content', span=[0, len(content)],
                basis='parser_accepted_after_repair_but_no_exact_wire_value_map')
            for key, kind in [('analysis', 'analysis'), ('plan', 'plan'), ('task_complete', 'completion_declaration')]:
                if parsed[key]:
                    add(kind, '/parsed/'+key, layer='parser_projection',
                        basis='current_reply_frozen_parser', no_raw_wire_character_span=True,
                        declaration_is_not_evaluation_success=kind == 'completion_declaration')
            for index, command in enumerate(parsed['commands']):
                add('command_request', f'/parsed/commands/{index}', layer='parser_projection',
                    basis='current_reply_frozen_parser', no_raw_wire_character_span=True,
                    command_requested_not_execution_confirmed=True)
            return
        offset = content.index(raw_json)
        last = {field['key']: field for field in fields}
        selected = []
        for key, kind in [('analysis', 'analysis'), ('plan', 'plan'), ('commands', 'command_request'),
                          ('task_complete', 'completion_declaration')]:
            value = parsed[key]
            if key not in last or not value:
                continue
            field = last[key]
            selected.append(dict(kind=kind, begin=offset+field['begin'], end=offset+field['end'],
                                 parsed_key=key, decoded_value=value))
        cursor = 0
        for field in sorted(selected, key=lambda r: r['begin']):
            if cursor < field['begin']:
                self._tb_remainder(content, cursor, field['begin'], offset, offset+len(raw_json), add)
            add(field['kind'], '/original/content', span=[field['begin'], field['end']],
                layer='raw_json_value_lexeme', basis='current_reply_frozen_parser_and_lexical_value_map',
                decoded_value=field['decoded_value'], decoded_source_pointer='/parsed/'+field['parsed_key'],
                raw_span_is_encoded_json_not_decoded_string=True,
                declaration_is_not_evaluation_success=field['kind'] == 'completion_declaration',
                command_requested_not_execution_confirmed=field['kind'] == 'command_request')
            cursor = field['end']
        if cursor < len(content):
            self._tb_remainder(content, cursor, len(content), offset, offset+len(raw_json), add)

    @staticmethod
    def _tb_remainder(text, begin, end, json_begin, json_end, add):
        cuts = sorted({begin, end, *(p for p in (json_begin, json_end) if begin < p < end)})
        for a, b in zip(cuts, cuts[1:]):
            kind = ('protocol_unclassified' if json_begin <= a and b <= json_end or not text[a:b].strip()
                    else 'assistant_text')
            add(kind, '/original/content', span=[a, b], basis='raw_remainder_not_semantic_reclassification')


def check_block(block, raw):
    source = block['source']
    value = resolve(raw[source['trace_line']-1], source['pointer'])
    if source['character_span'] is not None:
        begin, end = source['character_span']
        require(isinstance(value, str) and 0 <= begin <= end <= len(value), 'Invalid source coordinate')
        value = value[begin:end]
    require(value == block['content'], 'Original field/citation differs')


def render(output, blocks, flows, summaries):
    parts = ['<!doctype html><meta charset="utf-8"><title>17类结构切片</title>',
        '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap;max-width:1100px}.aux{color:#777}summary{cursor:pointer}</style>',
        '<h1>17类协议/字段结构切片</h1><p>结构归属≠语义阶段。原文位置与 parser 派生值明确区分；未调用模型或执行轨迹命令。辅助 protocol/unclassified 类型仅保留剩余内容，不是新增语义类别。</p>',
        '<h2>类型样例（点击查看原文）</h2>']
    gallery = {}
    for kind in TYPES:
        sample = next((b for b in blocks if b['structural_type'] == kind), None)
        if sample:
            gallery[kind] = sample
            content = sample.get('decoded_value', sample['content'])
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            parts.append('<details><summary>'+html.escape(kind+' — '+TYPES[kind]+' — '+sample['block_id'])+
                         '</summary><pre>'+html.escape(json.dumps(sample['source'], ensure_ascii=False))+
                         '\n'+html.escape(text)+'</pre></details>')
    for summary in summaries:
        tid = summary['trajectory_id']
        parts.append('<h2>'+html.escape(tid)+'</h2><p>'+html.escape(json.dumps(summary, ensure_ascii=False))+'</p>')
        parts.append('<h3>按 Step 查看结构链</h3><table><tr><th>Step</th><th>按消息顺序的结构类型</th></tr>')
        for flow in flows:
            if flow['trajectory_id'] == tid:
                parts.append('<tr><td>'+str(flow['step'])+'</td><td>'+html.escape(' → '.join(flow['type_sequence']))+'</td></tr>')
        parts.append('</table><h3>所有片段与坐标</h3>')
        for block in blocks:
            if block['trajectory_id'] != tid:
                continue
            content = block.get('decoded_value', block['content'])
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            label = f'{block["block_id"]} Step {block["source"]["step"]} {block["structural_type"]}'
            parts.append('<details'+(' class="aux"' if not block['supported_type'] else '')+'><summary>'+
                html.escape(label)+' — '+html.escape(text[:120].replace('\n',' '))+'</summary><pre>'+
                html.escape(json.dumps({k:v for k,v in block.items() if k not in ('content','decoded_value')}, ensure_ascii=False, indent=2))+
                '\n\n'+html.escape(text)+'</pre></details>')
    (output/'preview.html').write_text('\n'.join(parts))
    save(output/'type-gallery.json', gallery)


def run(bundle, trajectories, output):
    started = time.perf_counter()
    require(not output.exists(), 'Output exists; use a new directory')
    rules, parser, expected = load_rules(bundle)
    index = {item['trajectory_id']: item for item in json.loads((bundle/'index.json').read_text())}
    blocks, audits, flows, summaries, input_hashes = [], [], [], [], {}

    def checked(path):
        actual = digest(bundle/path)
        require(actual == expected[path], 'Bundle file hash differs: '+path)
        input_hashes[path] = actual

    checked('index.json')
    for tid in trajectories:
        item = index[tid]
        for key in ('trace_path', 'task_path', 'raw_request_path', 'raw_response_path'):
            if item.get(key):
                checked(item[key])
        raw = [json.loads(line) for line in (bundle/item['trace_path']).read_text().splitlines()]
        request = json.loads((bundle/item['raw_request_path']).read_text())
        originals = list(request['messages'])
        if item.get('raw_response_path'):
            response = json.loads((bundle/item['raw_response_path']).read_text())
            originals.append(dict(role='assistant', content=response['content']) if item['family'] == 'deepswe'
                             else response['choices'][0]['message'])
        require([r['original'] for r in raw if r['record_type']=='message'] == originals, 'Saved raw messages differ')
        adapter = StructureAdapter(rules, parser)
        local_blocks, local_audits = [], []
        task = (bundle/item['task_path']).read_text()
        for line, record in enumerate(raw, 1):
            if record['record_type'] == 'provenance':
                continue
            if record['record_type'] == 'context':
                require(record['original'] == resolve(request, record['source']['pointer']), 'Context/raw request differs')
            extracted = adapter.record_blocks(item, record, line, task)
            for block in extracted:
                check_block(block, raw)
                destination = local_audits if block['layer'] == 'session_audit' else local_blocks
                block['block_id'] = f'T{len(summaries)+1:02}-'+('A' if block['layer']=='session_audit' else 'B')+f'{len(destination)+1:04}'
                destination.append(block)
        for turn in sorted({b['source']['step'] for b in local_blocks if b['source']['step'] is not None}):
            current = [b for b in local_blocks if b['source']['step'] == turn and b['supported_type']]
            flows.append(dict(trajectory_id=tid, step=turn, block_ids=[b['block_id'] for b in current],
                              type_sequence=[b['structural_type'] for b in current],
                              note='Field order, not a semantic stage or token-offset reconstruction'))
        summaries.append(dict(trajectory_id=tid, family=item['family'], assistant_steps=item['saved_assistant_turns'],
            message_blocks=len(local_blocks), core_type_counts=dict(Counter(b['structural_type'] for b in local_blocks if b['supported_type'])),
            auxiliary_counts=dict(Counter(b['structural_type'] for b in local_blocks if not b['supported_type'])),
            session_only_executor_audit_blocks=len(local_audits), parser_replay=adapter.parse_audits))
        blocks.extend(local_blocks)
        audits.extend(local_audits)
    require(all(digest(bundle/path) == value for path,value in input_hashes.items()), 'Inputs changed during slicing')
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('blocks.jsonl',blocks),('session-audit.jsonl',audits),('step-flows.jsonl',flows)]:
        (output/name).write_text(''.join(json.dumps(value,ensure_ascii=False)+'\n' for value in values))
    with (output/'blocks.csv').open('w') as stream:
        fields = ['block_id','trajectory_id','step','type','also_types','layer','trace_line','pointer','character_span','excerpt']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for block in blocks:
            source=block['source'];content=block.get('decoded_value',block['content'])
            text=content if isinstance(content,str) else json.dumps(content,ensure_ascii=False)
            writer.writerow(dict(block_id=block['block_id'],trajectory_id=block['trajectory_id'],step=source['step'],
                type=block['structural_type'],also_types=json.dumps(block['also_types']),layer=block['layer'],
                trace_line=source['trace_line'],pointer=source['pointer'],character_span=json.dumps(source['character_span']),excerpt=text[:320]))
    render(output, blocks, flows, summaries)
    report=dict(structural_types=TYPES, covered_types=sorted({b['structural_type'] for b in blocks if b['supported_type']}),
        trajectories=summaries, model_calls=0, no_trace_commands_executed=True,
        raw_coordinate_blocks_verified=len(blocks), session_only_audit_values_verified=len(audits),
        no_token_offsets_inferred=True, no_dependency_or_redundancy_judgments=True,
        parsed_assistant_labels_use_following_feedback=False, input_sha256=input_hashes,
        frozen_method_sha256={name:expected[name] for name in ('methodology/label_agent_trajectories.py','methodology/frozen/terminus_json_plain_parser.py')},
        script_sha256=digest(__file__), local_end_to_end_seconds=time.perf_counter()-started)
    report['output_sha256']={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    save(output/'manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','output_sha256','structural_types')},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--trajectory',action='append')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.bundle,args.trajectory or DEFAULT_TRACES,args.output)


if __name__=='__main__':
    main()
