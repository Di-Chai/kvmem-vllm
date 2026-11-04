"""Read-only trace normalization and conservative, auditable observation rules.

No recorded commands are executed. No model, tokenizer, or network is used.
All spans are Unicode offsets, never token/block offsets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

VERSION = 'duokv-observation-v1'
TYPES = {'thinking', 'analysis', 'plan', 'tool_call', 'tool_result',
         'terminal_observation', 'execution_status', 'task_instruction',
         'assistant_text', 'parser_feedback', 'parser_warning'}
ASSISTANT_TYPES = {'thinking', 'analysis', 'plan', 'tool_call', 'assistant_text'}
STOP = set('the a an and or to of in is it this that for with from as be we i you can will should would have has are was not on at by our let now need use using then there if so all also file code output command result return true false none null self app test tests'.split())
STOP.update('issue problem response responses json text before after keep keeps having try trying approach different think seems just want write writing open read reading cat echo python python3 tool tools assistant analysis plan need needs make sure look like get got see seen check checking going done instead again still been being do does did my your they them their which what how why could must only already first next same into about than more some other each very up out even an such through here new function functions'.split())
WORD = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?')
PATH = re.compile(r'(?<![\w:])(?:/?(?:[\w.@+-]+/)+[\w.@+-]+|[\w+-]+\.(?:py|rs|ts|tsx|js|go|java|cpp|hpp|c|h|toml|json|yaml|yml|sql|fasta|csv|txt|md))(?![\w])')
SYMBOL = re.compile(r'\b(?:def|class|fn|function|struct|enum)\s+([A-Za-z_]\w{3,})|\b([A-Za-z_]\w{3,})(?=\()')
ERROR = re.compile(r'\b(?:\w*(?:Error|Exception)|error\[E\d+\])\b')
REFERENCE = re.compile(r'\b(?:previous(?:ly)?|earlier|above|as (?:seen|shown|noted)|recall|remember|we (?:saw|found|established)|from (?:the|our) (?:output|test|result)|since|because|must|constraint|invariant)\b', re.I)
ACTION = re.compile(r'\b(?:fix|change|modify|replace|implement|preserve|keep|ensure|use|need|add|remove|write|call|check|verify|read|run)\w*\b', re.I)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            yield n, json.loads(line)


def resolve(obj, pointer):
    for p in pointer.split('/')[1:]:
        p = p.replace('~1', '/').replace('~0', '~')
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(text):
    # Read tool line numbers are presentation, not program text.
    text = re.sub(r'(?m)^\s*\d+\s*(?:→|\t)\s?', '', text)
    return ' '.join(text.split())


def entities(text):
    out = set()
    for m in PATH.finditer(text):
        if '/' in m.group() and not (m.group().startswith('/') or '.' in m.group().rsplit('/', 1)[-1] or m.group().count('/') >= 2):
            continue  # Avoid prose such as forward/reverse.
        s = m.group().removeprefix('/app/').removeprefix('./')
        if s not in {'dev/null', 'tmp', 'app'} and len(s) > 4:
            out.add('path:' + s)
    for m in SYMBOL.finditer(text):
        s = m.group(1) or m.group(2)
        if s.lower() not in STOP and s not in {'print', 'println', 'assert', 'isinstance', 'range'}:
            out.add('symbol:' + s)
    for m in ERROR.finditer(text):
        if m.group().lower() not in {'error', 'exception'}:
            out.add('error:' + m.group())
    return sorted(out)


def words(text):
    return {m.group().lower() for m in WORD.finditer(text)} - STOP


def structural_spans(text, start=0, end=None, limit=1400):
    """Paragraph, code fence, terminal prompt, and bounded line chunks.

    This is a structural proxy for semantic units; no claim of atomic meaning.
    Every character in the requested range occurs exactly once, including space.
    """
    end = len(text) if end is None else end
    if end <= start:
        return []
    cuts = {start, end}
    for m in re.finditer(r'\n\s*\n|(?m:^```[^\n]*\n)|(?m:^root@[^\n]*[#$])', text[start:end]):
        cuts.add(start + m.end() if m.group().startswith('\n') else start + m.start())
    ordered = sorted(cuts)
    spans = []
    for a, b in zip(ordered, ordered[1:]):
        while b - a > limit:
            cut = text.rfind('\n', a + limit // 2, a + limit)
            if cut < 0:
                cut = text.rfind(' ', a + limit // 2, a + limit)
            cut = cut + 1 if cut >= 0 else a + limit
            spans.append((a, cut))
            a = cut
        if b > a:
            spans.append((a, b))
    return spans


def json_string_fields(text):
    """Locate string leaves inside a valid JSON object, allowing an outer fence.

    Returns decoded leaf text and exact encoded substring coordinates. Invalid
    JSON falls back to assistant_text, even if the supplied tolerant parser
    accepted it; we never invent original coordinates.
    """
    start = text.find('{')
    if start < 0:
        return {}
    decoder = json.JSONDecoder()
    result = {}

    def ws(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i

    def walk(i, path):
        i = ws(i)
        if text[i] == '{':
            i = ws(i + 1)
            if text[i] == '}':
                return i + 1
            while True:
                key, j = decoder.raw_decode(text, i)
                j = ws(j)
                if not isinstance(key, str) or text[j] != ':':
                    raise ValueError('object')
                i = ws(walk(j + 1, path + '/' + key.replace('~', '~0').replace('/', '~1')))
                if text[i] == '}':
                    return i + 1
                if text[i] != ',':
                    raise ValueError('separator')
                i = ws(i + 1)
        elif text[i] == '[':
            i = ws(i + 1)
            if text[i] == ']':
                return i + 1
            n = 0
            while True:
                i = ws(walk(i, path + '/' + str(n)))
                n += 1
                if text[i] == ']':
                    return i + 1
                if text[i] != ',':
                    raise ValueError('separator')
                i = ws(i + 1)
        else:
            value, j = decoder.raw_decode(text, i)
            if isinstance(value, str):
                result[path] = (value, i, j)
            return j

    try:
        walk(start, '')
    except (ValueError, IndexError, TypeError):
        return {}
    return result


def decoded_boundaries(encoded):
    """Map decoded Unicode boundaries to an encoded JSON string's interior."""
    out = [1]
    i = 1
    while i < len(encoded) - 1:
        if encoded[i] == '\\':
            if encoded[i + 1] == 'u':
                code = int(encoded[i + 2:i + 6], 16)
                i += 6
                if 0xD800 <= code <= 0xDBFF and encoded[i:i + 2] == '\\u':
                    low = int(encoded[i + 2:i + 6], 16)
                    if 0xDC00 <= low <= 0xDFFF:
                        i += 6
            else:
                i += 2
        else:
            i += 1
        out.append(i)
    return out


def string_leaves(obj, path):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from string_leaves(v, path + '/' + k.replace('~', '~0').replace('/', '~1'))
    elif isinstance(obj, list):
        for n, v in enumerate(obj):
            yield from string_leaves(v, path + '/' + str(n))


def normalize_trace(root, item):
    records = list(read_jsonl(root / item['trace_path']))
    messages = [(n, r) for n, r in records if r['record_type'] == 'message']
    retry_turns = {r['turn'] for _, r in messages if r.get('outcome') == 'format_retry'}
    step_map = {}
    step = 0
    for turn in range(item['saved_assistant_turns'] + 1):
        if turn and turn not in retry_turns:
            step += 1
        step_map[turn] = step
    units = []
    quality = Counter()
    calls = {}
    for line, row in messages:
        mid = item['trajectory_id'] + ':m' + str(row['message_index'])
        base = dict(trajectory_id=item['trajectory_id'], family=item['family'], task_id=item['task_id'],
                    seed=item['seed'], attempt_source=item['attempt_source'], message_id=mid,
                    message_index=row['message_index'], turn_id=row['turn'], step=step_map[row['turn']],
                    role=row['role'], retry=row['turn'] in retry_turns, trace_path=item['trace_path'],
                    trace_line=line, raw_source=row['source'], timestamp=None,
                    timestamp_status='not_available_at_message_granularity')
        raw = row['original']
        fragments = []
        if item['family'] == 'terminalbench' and row['role'] == 'assistant':
            if raw.get('reasoning_content'):
                fragments.append(('thinking', '/original/reasoning_content', raw['reasoning_content'], 0,
                                  len(raw['reasoning_content']), {}, None))
            fields = json_string_fields(raw.get('content') or '')
            parsed = row.get('parsed', {})
            valid = not parsed.get('error') and parsed.get('replay_matches_observed_feedback') is not False
            wanted = {'/analysis': 'analysis', '/plan': 'plan'}
            wanted.update({f'/commands/{i}/keystrokes': 'tool_call' for i, _ in enumerate(parsed.get('commands', []))})
            accepted = 0
            if valid and fields:
                for p, kind in wanted.items():
                    if p not in fields:
                        continue
                    val, a, b = fields[p]
                    expected = resolve(parsed, p)
                    if val != expected:
                        quality['parsed_field_original_mismatch'] += 1
                        continue
                    extra = dict(tool='terminal', field=p)
                    if kind == 'tool_call':
                        extra['command_index'] = int(p.split('/')[2]) + 1
                    fragments.append((kind, '/original/content', val, 0, len(val), extra,
                                      (a, b, decoded_boundaries(raw['content'][a:b]), p)))
                    accepted += 1
            if not accepted and raw.get('content'):
                text = raw['content']
                fragments.append(('assistant_text', '/original/content', text, 0, len(text), {}, None))
                quality['assistant_json_fallback'] += int(valid)
        else:
            # Nested labels override parent labels, preventing double counting.
            labels = row['labels']
            by_path = defaultdict(list)
            for a in labels:
                if a['kind'] in {'progress_operation', 'completion_declaration'}:
                    continue
                value = resolve(row, a['path'])
                if a['kind'] in {'tool_call', 'tool_result'}:
                    extra = {k: a.get(k) for k in ('tool', 'call_id', 'is_error')}
                    if a['kind'] == 'tool_call':
                        calls[a['call_id']] = (row, value)
                        value, sub = value.get('input', {}), '/input'
                    else:
                        value, sub = value.get('content', ''), '/content'
                    for p, text in string_leaves(value, a['path'] + sub):
                        # Tool-use IDs and serialization scaffolding are not content.
                        if p.endswith('/type'):
                            continue
                        fragments.append((a['kind'], p, text, 0, len(text),
                                          dict(extra, field=p.rsplit('/', 1)[-1]), None))
                elif isinstance(value, str):
                    aa, bb = a.get('character_span', [0, len(value)])
                    by_path[a['path']].append((aa, bb, a['kind'], value))
            priority = {'task_instruction': 6, 'execution_status': 5, 'framework_rules': 4,
                        'parser_feedback': 4, 'parser_warning': 4, 'completion_confirmation': 4}
            for p, parts in by_path.items():
                cuts = sorted({v for a, b, _, _ in parts for v in (a, b)})
                for a, b in zip(cuts, cuts[1:]):
                    covering = [x for x in parts if x[0] <= a and x[1] >= b]
                    if covering:
                        chosen = max(covering, key=lambda x: priority.get(x[2], 1))
                        extra = {}
                        if item['family'] == 'terminalbench' and chosen[2] in {'terminal_observation', 'execution_status'}:
                            extra['tool'] = 'terminal'
                        elif chosen[2] in {'parser_feedback', 'parser_warning'}:
                            extra['tool'] = 'frozen_response_parser'
                        fragments.append((chosen[2], p, chosen[3], a, b, extra, None))
        for kind, p, text, a, b, extra, mapping in fragments:
            for begin, end in structural_spans(text, a, b):
                content = text[begin:end]
                if not content.strip():
                    continue
                # IDs use the per-message counter, not length-limited history.
                uid = f'{mid}:u{quality["message_units:" + mid]}'
                quality['message_units:' + mid] += 1
                raw_begin, raw_end = begin, end
                extra_map = {}
                if mapping:
                    ra, rb, bounds, dp = mapping
                    assert len(bounds) == len(text) + 1
                    raw_begin, raw_end = ra + bounds[begin], ra + bounds[end]
                    extra_map = dict(decoded_field_pointer='/parsed' + dp,
                                     decoded_character_span=[begin, end], encoding='json_string_interior')
                entity = entities(content)
                if ERROR.search(content) or re.search(r'(?m)^\s*(?:Traceback|FAILED|error:)', content):
                    subtype = 'error_log'
                elif re.search(r'(?m)^\s*(?:\d+[\t→ ]+)?(?:def |class |fn |pub |impl |import |function |const |struct )', content):
                    subtype = 'code'
                elif kind in {'tool_result', 'terminal_observation', 'execution_status'}:
                    subtype = 'execution_output'
                elif kind == 'task_instruction':
                    subtype = 'user_constraint'
                else:
                    subtype = 'prose_or_command'
                u = dict(base, unit_id=uid, content_type=kind, subtype=subtype,
                         field_pointer=p, character_span=[raw_begin, raw_end], text=content,
                         text_sha256=digest(content), entities=entity, **extra_map, **extra)
                u['main_eligible'] = kind in TYPES and not u['retry'] and kind not in {'parser_feedback', 'parser_warning'}
                units.append(u)
    quality = {k: v for k, v in quality.items() if not k.startswith('message_units:')}
    return units, messages, records, step_map, calls, dict(quality)


def shingles(text):
    tokens = list(WORD.finditer(text))
    candidates = []
    for i in range(max(0, len(tokens) - 7)):
        group = tokens[i:i + 8]
        key = tuple(m.group().lower() for m in group)
        if len(set(key) - STOP) < 4:
            continue
        a, b = group[0].start(), group[-1].end()
        if b - a < 36:
            continue
        # Standard local-minimum winnowing: even a short distinctive passage
        # retains an anchor. Matching sufficiently long passages share minima.
        h = hashlib.blake2s(' '.join(key).encode(), digest_size=4).digest()
        candidates.append((h, key, (a, b)))
    result = {}
    if candidates:
        width = min(4, len(candidates))
        for i in range(len(candidates)-width+1):
            _, key, span = min(candidates[i:i+width])
            result[key] = span
    return result


def numeric_facts(text):
    """Narrow entity/value statement anchors, not unrestricted semantic matching."""
    out = {}
    for m in re.finditer(r'\b([A-Za-z_][\w.-]{2,})\s*(?::|=|is|are|has|length(?: is)?)\s*(\d{2,}(?:\.\d+)?)(?:\s*(bp|bytes|tokens|lines|items|ms|seconds))?\b', text, re.I):
        if m.group(1).lower() not in STOP:
            out[(m.group(1).lower(), m.group(2), (m.group(3) or '').lower())] = m.span()
    return out


def evidence(unit, span=None):
    a, b = span or (0, min(500, len(unit['text'])))
    return dict(unit_id=unit['unit_id'], message_id=unit['message_id'], trace_path=unit['trace_path'],
                trace_line=unit['trace_line'], field_pointer=unit['field_pointer'],
                unit_character_span=unit['character_span'], excerpt_span_in_unit=[a, b],
                excerpt=unit['text'][a:b], turn_id=unit['turn_id'])


def detect_reuse(units, include_retry=False):
    """Nearest observable source attribution; same-turn edges are excluded.

    Explicit: distinctive multiword span, with overlap text evidence.
    Semantic: entity/value fact restatement with a reasoning/action cue.
    Potential: >=2 specific entities plus a reasoning/action cue. A shared
    keyword or path alone never creates a reuse edge.
    All are rule candidates, not annotated ground truth or causal dependency.
    """
    eligible = [u for u in units if (u['content_type'] in TYPES if include_retry else u['main_eligible'])]
    index, fact_index, entity_index = {}, {}, defaultdict(dict)
    edges = []
    counts = Counter()
    groups = defaultdict(list)
    for u in eligible:
        groups[u['turn_id']].append(u)
    for turn, batch in sorted(groups.items()):
        prepared = [(u, shingles(u['text']), numeric_facts(u['text'])) for u in batch]
        for target, anchors, facts in prepared:
            if target['role'] != 'assistant' or target['content_type'] not in ASSISTANT_TYPES:
                continue  # Repeated tool output alone is not agent reuse.
            proposals = {}
            for key, span in anchors.items():
                if key not in index:
                    continue
                source, srcspan, ambiguity = index[key]
                if source['turn_id'] >= turn:
                    continue
                old = proposals.get(source['unit_id'])
                if old is None or span[1] - span[0] > old[2][1] - old[2][0]:
                    proposals[source['unit_id']] = (source, srcspan, span, 'explicit',
                                                    'distinctive_eight_token_span', ambiguity, 'medium')
            if REFERENCE.search(target['text']) or ACTION.search(target['text']):
                for key, span in facts.items():
                    if key in fact_index:
                        source, srcspan, ambiguity = fact_index[key]
                        if source['unit_id'] not in proposals and source['turn_id'] < turn:
                            proposals[source['unit_id']] = (source, srcspan, span, 'semantic',
                                'same_entity_value_fact_with_reasoning_or_action', ambiguity, 'medium')
                matches = defaultdict(list)
                candidates = {}
                for ent in target['entities']:
                    for source in entity_index.get(ent, {}).values():
                        uid = source['unit_id']
                        matches[uid].append(ent)
                        candidates[uid] = source
                for uid, shared in matches.items():
                    source = candidates[uid]
                    if uid not in proposals and len(shared) >= 2 and source['turn_id'] < turn:
                        proposals[uid] = (source, None, None, 'potential_dependency',
                            'multiple_specific_entities_with_reasoning_or_action', 0, 'low')
            for source, srcspan, tgtspan, level, rule, ambiguity, confidence in proposals.values():
                dt = turn - source['turn_id'] if include_retry else target['step'] - source['step']
                if dt <= 0:
                    continue
                row = dict(edge_id=f'{source["unit_id"]}->{target["unit_id"]}:{level}',
                           trajectory_id=source['trajectory_id'], family=source['family'], task_id=source['task_id'],
                           seed=source['seed'], source_unit_id=source['unit_id'], target_unit_id=target['unit_id'],
                           source_message_id=source['message_id'], target_message_id=target['message_id'],
                           source_turn=source['turn_id'], target_turn=turn, source_step=source['step'], target_step=target['step'],
                           distance=dt, level=level, rule=rule, confidence=confidence,
                           validation_status='automated_candidate_unreviewed',
                           competing_prior_occurrences=ambiguity, attribution='latest_matching_unit',
                           source_evidence=evidence(source, srcspan or (0,len(source['text']))),
                           target_evidence=evidence(target, tgtspan or (0,len(target['text']))),
                           shared_entities=sorted(set(source['entities']) & set(target['entities'])),
                           includes_retry=include_retry)
                edges.append(row)
        # Update after the whole turn: never attribute a result back into its query.
        for u, anchors, facts in prepared:
            for key, span in anchors.items():
                prior = index.get(key)
                index[key] = (u, span, 0 if prior is None else prior[2] + 1)
            for key, span in facts.items():
                prior = fact_index.get(key)
                fact_index[key] = (u, span, 0 if prior is None else prior[2] + 1)
            for ent in u['entities']:
                # Last source per type limits repetition inflation without using future data.
                entity_index[ent][u['content_type']] = u
    # A source counts at most once per target Step and level (not per assistant field).
    unique = {}
    for row in edges:
        key = (row['source_unit_id'], row['target_turn'], row['level'])
        old = unique.get(key)
        if old is None or len(row['target_evidence']['excerpt']) > len(old['target_evidence']['excerpt']):
            unique[key] = row
    return list(unique.values())


def verify_unit(root, unit, record_cache):
    path = unit['trace_path']
    if path not in record_cache:
        record_cache[path] = dict(read_jsonl(root / path))
    row = record_cache[path][unit['trace_line']]
    raw = resolve(row, unit['field_pointer'])
    a, b = unit['character_span']
    actual = raw[a:b]
    if unit.get('encoding') == 'json_string_interior':
        actual = json.loads('"' + actual + '"')
    assert actual == unit['text'], (unit['unit_id'], actual[:50], unit['text'][:50])
    assert digest(actual) == unit['text_sha256']
