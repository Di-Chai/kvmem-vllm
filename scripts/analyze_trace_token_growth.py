"""Count exclusive token labels on exact saved, append-only agent histories.

No model execution. Historical model output is never re-tokenized. Complete
input-fragment encodings are checked against the recorded prompt fingerprints.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from scripts.label_agent_trajectories import feedback_labels, strip_canary


NAMES = {
    'framework_rules': '框架规则', 'tool_definitions': '工具定义',
    'task_instruction': '原始任务', 'initial_context': '首轮补充上下文',
    'thinking': 'Thinking', 'assistant_text': '普通正文',
    'analysis': 'Analysis', 'plan': 'Plan', 'tool_call': '工具调用',
    'command_request': '命令请求', 'progress_operation': '任务进度操作',
    'tool_result': '工具返回', 'terminal_observation': '终端观察',
    'parser_feedback': '解析错误反馈', 'parser_warning': '格式警告',
    'execution_status': '执行状态', 'completion_declaration': '完成声明',
    'completion_confirmation': '完成确认提示', 'user_text': '普通 user 文本',
    'protocol': '消息／JSON 格式', 'boundary': '跨标签边界 token',
}
GROUPS = {
    'prefix': ['framework_rules', 'tool_definitions', 'task_instruction', 'initial_context'],
    'thinking': ['thinking'],
    'prose': ['assistant_text', 'analysis', 'plan'],
    'actions': ['tool_call', 'command_request', 'progress_operation', 'completion_declaration'],
    'observations': ['tool_result', 'terminal_observation', 'parser_feedback', 'parser_warning',
                     'execution_status', 'completion_confirmation', 'user_text'],
    'structure': ['protocol', 'boundary'],
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def utf8_points(text):
    result = [0]
    for character in text:
        result.append(result[-1] + len(character.encode()))
    return result


def paint(labels, starts, ends, spans):
    """Paint only complete tokens; a token split by a label boundary is explicit.

    Spans are applied from outer to inner. Byte coordinates refer to the actual
    encoded input fragment or the byte-level decoded accepted output IDs.
    """
    for begin, end, kind in spans:
        if begin >= end:
            continue
        first = bisect_right(ends, begin)
        last = bisect_left(starts, end)
        for index in range(first, last):
            labels[index] = kind if begin <= starts[index] and ends[index] <= end else 'boundary'
    return labels


def runs(labels, offset=0):
    result = []
    for index, kind in enumerate(labels):
        if result and result[-1]['kind'] == kind:
            result[-1]['end'] += 1
            result[-1]['tokens'] += 1
        else:
            result.append(dict(kind=kind, begin=offset + index, end=offset + index + 1, tokens=1))
    return result


def top_fields(text):
    """Read actual JSON value intervals without dumping/re-encoding values."""
    decoder = json.JSONDecoder()
    begin = text.find('{')
    if begin < 0:
        return None
    index = begin + 1
    fields = []
    try:
        while True:
            while index < len(text) and text[index].isspace():
                index += 1
            if text[index] == '}':
                return begin, index + 1, fields
            key, index = decoder.raw_decode(text, index)
            while text[index].isspace():
                index += 1
            if text[index] != ':':
                return None
            index += 1
            while text[index].isspace():
                index += 1
            start = index
            value, index = decoder.raw_decode(text, index)
            fields.append((key, start, index, value))
            while text[index].isspace():
                index += 1
            if text[index] == ',':
                index += 1
            elif text[index] != '}':
                return None
    except (ValueError, IndexError, TypeError):
        return None


def byte_decoder(tokenizer):
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    original, extra = list(values), 0
    for byte in range(256):
        if byte not in values:
            original.append(byte)
            values.append(256 + extra)
            extra += 1
    inverse = dict(zip(map(chr, values), original))
    added = set(tokenizer.get_added_vocab().values())

    @lru_cache(maxsize=16384)
    def piece(token):
        text = tokenizer.convert_ids_to_tokens(token)
        return text.encode() if token in added else bytes(inverse[c] for c in text)
    return piece


def output_labels(ids, tokenizer, piece, family, response, parsed):
    chunks = [piece(token) for token in ids]
    starts, ends, position = [], [], 0
    for chunk in chunks:
        starts.append(position)
        position += len(chunk)
        ends.append(position)
    raw = b''.join(chunks)
    close_id = tokenizer.convert_tokens_to_ids('</think>')
    close = ids.index(close_id) if close_id in ids else len(ids)
    labels = ['thinking' if i < close else 'assistant_text' for i in range(len(ids))]
    spans = []
    body_begin = ends[close] if close < len(ids) else len(raw)
    if family == 'deepswe':
        calls = [b for b in response['content'] if b.get('type') == 'tool_use']
        frames = list(re.finditer(rb'<tool_call>.*?</tool_call>', raw[body_begin:], re.S))
        if len(frames) != len(calls):
            raise ValueError('Accepted tool frame count differs from structured response')
        for frame, call in zip(frames, calls):
            kind = 'progress_operation' if call['name'] in ('TaskCreate', 'TaskUpdate') else 'tool_call'
            spans.append((body_begin + frame.start(), body_begin + frame.end(), kind))
    elif not parsed.get('error'):
        body = raw[body_begin:].decode('utf-8')
        found = top_fields(body)
        if found is not None:
            byte_at = utf8_points(body)
            begin, end, fields = found
            spans.append((body_begin + byte_at[begin], body_begin + byte_at[end], 'protocol'))
            for key, begin, end, value in fields:
                kind = key if key in ('analysis', 'plan') else (
                    'command_request' if key == 'commands' and value else
                    'completion_declaration' if key == 'task_complete' and value else None)
                if kind:
                    spans.append((body_begin + byte_at[begin], body_begin + byte_at[end], kind))
    paint(labels, starts, ends, spans)
    structural = {tokenizer.convert_tokens_to_ids(marker) for marker in
                  ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')}
    for index, token in enumerate(ids):
        if token in structural:
            labels[index] = 'protocol'
    return labels


def input_labels(text, encoding, rendered, shift, family, initial, task):
    points = utf8_points(text)
    starts = [points[lo] for lo, hi in encoding['offset_mapping']]
    ends = [points[hi] for lo, hi in encoding['offset_mapping']]
    labels = ['protocol'] * len(starts)
    raw = text.encode()
    spans = []
    control = rendered['control_end'] + shift
    if initial and control > 0:
        header = b'<|im_start|>system\n'
        closing = b'<|im_end|>\n'
        body_begin = len(header) if raw.startswith(header) else 0
        body_end = control - len(closing) if raw[:control].endswith(closing) else control
        spans.append((body_begin, body_end, 'framework_rules'))
        begin = raw.find(b'<tools>')
        end = raw.find(b'</tools>', begin)
        if begin >= 0 and end >= 0:
            spans.append((begin + len(b'<tools>'), end, 'tool_definitions'))
    for span in rendered['spans']:
        begin, end = span['content_begin'] + shift, span['content_end'] + shift
        if begin < 0 or end <= 0:
            continue
        content = raw[begin:end].decode()
        if initial and span['role'] == 'user':
            spans.append((begin, end, 'initial_context'))
            candidate = task if task in content else task.rstrip('\r\n')
            if not candidate or content.count(candidate) != 1:
                raise ValueError('Initial task has no unique exact content match')
            pos = content.index(candidate)
            lo = begin + len(content[:pos].encode())
            if family == 'terminalbench':
                spans.append((begin, lo, 'framework_rules'))
            spans.append((lo, lo + len(candidate.encode()), 'task_instruction'))
        elif span['role'] == 'tool':
            spans.append((begin, end, 'tool_result'))
        elif span['role'] == 'user':
            for label in feedback_labels(content):
                if 'character_span' in label:
                    lo, hi = label['character_span']
                    spans.append((begin + len(content[:lo].encode()), begin + len(content[:hi].encode()), label['kind']))
                else:
                    spans.append((begin, end, label['kind']))
    return paint(labels, starts, ends, spans)


def analyze(selection, tokenizer, destination):
    # Protocol and conversion helpers do not execute inference or native KV code.
    from benchmarks.deepswe.claude_bridge import Protocol
    from scripts.dsh_bridge import to_anthropic
    family, task_id = selection['family'], selection['task_id']
    session = Path(selection['request']).parent.parent
    root = Path(selection['request']).parents[3]
    library = root / ('protocol/protocol.so' if family == 'deepswe' else 'protocol.so')
    protocol = Protocol(library)
    task = Path(selection['task_file']).read_text()
    if family == 'terminalbench':
        task = strip_canary(task)
    labelled = destination.parent / 'context-trace-labels-20261031-r2' / family / (task_id + '.jsonl')
    records = [json.loads(line) for line in labelled.open()]
    assistants = [r for r in records if r.get('record_type') == 'message' and r.get('role') == 'assistant']
    decoder = byte_decoder(tokenizer)
    previous_ids, previous_output, previous_render, previous_labels = [], [], None, []
    turns, sources = [], []
    proofs = Counter()
    for index, folder in enumerate(sorted(p for p in session.iterdir() if p.name.isdigit()), 1):
        request, plan, response, native = [read(folder / (name + '.json')) for name in ('request', 'plan', 'response', 'native')]
        payload = to_anthropic(request) if family == 'terminalbench' else request
        rendered = protocol(request=payload)
        prefix = []
        if index > 1:
            if not plan['exact_output_prefix_reused']:
                raise ValueError('This exporter requires append-only exact accepted prefixes')
            first = next(s for s in rendered['spans'] if s['index'] == len(previous_render['messages']) + 1)
            boundary = first['begin']
            fragment = '\n' + rendered['prompt'].encode()[boundary:].decode()
            shift = 1 - boundary
            prefix = previous_ids + previous_output
        else:
            fragment, shift = rendered['prompt'], 0
        encoding = tokenizer(fragment, add_special_tokens=False, return_offsets_mapping=True)
        ids = prefix + encoding['input_ids']
        assert len(ids) == plan['prompt_tokens'], (task_id, index, len(ids), plan['prompt_tokens'])
        proofs['prompt_counts'] += 1
        if 'token_sha256' in plan:
            assert digest(json_bytes(ids)) == plan['token_sha256'], (task_id, index, 'full prompt SHA')
            proofs['full_prompt_sha256'] += 1
        provenance = plan['kvmem_locality_provenance']
        assert digest(json_bytes(encoding['input_ids'])) == provenance['token_ids_sha256']
        assert provenance.get('token_offset', 0) == len(prefix)
        proofs['actual_input_fragment_sha256'] += 1
        extra_labels = input_labels(fragment, encoding, rendered, shift, family, index == 1, task)
        labels = previous_labels + extra_labels
        assert len(labels) == len(ids)
        if index > 1:
            turns[-1]['following_input_tokens'] = len(extra_labels)
            turns[-1]['events'] += [dict(event, origin='following_input') for event in runs(extra_labels, len(prefix))]
            turns[-1]['counts_added'] = dict(Counter(turns[-1]['counts_added']) + Counter(extra_labels))
            turns[-1]['counts_after'] = dict(Counter(labels))
            turns[-1]['context_after'] = len(ids)
            turns[-1]['after_boundary'] = 'next_request_input'
        if family == 'deepswe':
            output = native['choices'][0]['token_ids']
        else:
            output = read(folder / 'accepted.json')['token_ids']
            assert output == native['token_ids']
        assert len(output) == response['usage'].get('completion_tokens', response['usage'].get('output_tokens'))
        proofs['accepted_output_counts'] += 1
        out_labels = output_labels(output, tokenizer, decoder, family, response, assistants[index-1].get('parsed', {}))
        context = labels + out_labels
        row = dict(turn=index, prompt_tokens=len(ids), output_tokens=len(output),
                   following_input_tokens=0, context_after=len(context),
                   after_boundary='last_saved_output', counts_before=dict(Counter(labels)),
                   counts_added=dict(Counter(out_labels)), counts_after=dict(Counter(context)),
                   outcome=assistants[index-1].get('outcome'),
                   events=[dict(event, origin='model_output') for event in runs(out_labels, len(ids))])
        turns.append(row)
        if index == 1:
            initial = dict(tokens=len(ids), counts=dict(Counter(labels)), events=runs(labels))
        previous_ids, previous_output, previous_render = ids, output, rendered
        previous_labels = context
        for name in ('request', 'plan', 'response', 'native'):
            path = folder / (name + '.json')
            sources.append(dict(path=str(path), sha256=digest(path.read_bytes())))
        print(json.dumps(dict(task=task_id, turn=index, input=len(ids), output=len(output)), ensure_ascii=False), flush=True)
    assert all(sum(r['counts_added'].values()) == r['output_tokens'] + r['following_input_tokens'] for r in turns)
    assert all(sum(r['counts_after'].values()) == r['context_after'] for r in turns)
    assert all(s['sha256'] == digest(Path(s['path']).read_bytes()) for s in sources)
    return dict(family=family, task_id=task_id, initial=initial, rounds=turns,
                proofs=dict(proofs), protocol_sha256=digest(library.read_bytes()), sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    selected = read(args.manifest)
    traces = [analyze(next(s for s in selected if s['task_id'] == task), tokenizer, args.output)
              for task in ('wasmi-trap-coredumps', 'dna-assembly')]
    data = dict(schema_version=1, names=NAMES, groups=GROUPS, traces=traces,
        counting='Exclusive labels on actual model tokens; boundary-crossing tokens are a separate class. JSON value ranges include their original quotes/escapes. No text fragments are independently tokenized.',
        context='Logical full context, not resident/selected GPU KV. Round end is the next saved request input, or the final saved model output.',
        tokenizer=str(args.tokenizer), tokenizer_sha256=digest((args.tokenizer/'tokenizer.json').read_bytes()))
    (args.output/'token-growth.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with (args.output/'token-growth.csv').open('w', newline='') as stream:
        keys = ['family', 'task_id', 'turn', 'prompt_tokens', 'output_tokens', 'following_input_tokens', 'context_after', 'outcome']
        writer = csv.DictWriter(stream, fieldnames=keys + ['added_' + k for k in NAMES] + ['cumulative_' + k for k in NAMES])
        writer.writeheader()
        for trace in traces:
            for row in trace['rounds']:
                item = {key: trace[key] if key in ('family', 'task_id') else row[key] for key in keys}
                item.update({'added_' + k: row['counts_added'].get(k, 0) for k in NAMES})
                item.update({'cumulative_' + k: row['counts_after'].get(k, 0) for k in NAMES})
                writer.writerow(item)
    print(json.dumps(dict(output=str(args.output), proofs=[t['proofs'] for t in traces]), ensure_ascii=False))


if __name__ == '__main__':
    main()
