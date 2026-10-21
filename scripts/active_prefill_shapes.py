"""Pure CPU native schedules derived from observed physical prefill geometry.

Prefix establishment is synthetic and outside each selected continuous phase.
Actual public chunks, active lengths, terminal flags and refresh boundaries
are retained. No scheduler or model is invoked here.
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(row):
    return [row['query_tokens'], row['active_seq_len'], row['prefill_terminal']]


def _segments(geometry, refresh):
    _require(bool(geometry), 'Actual physical geometry is empty')
    events = {row['history_begin']: row for row in refresh}
    _require(len(events) == len(refresh), 'Duplicate refresh boundary')
    current = offset = 0
    result, part = [], []
    seen = set()
    for index, row in enumerate(geometry):
        q = row['query_tokens']
        _require(type(q) is int and 0 < q <= 16384, 'Unsupported actual physical chunk')
        _require(row.get('prefill') is True, 'Shape family contains a non-prefill row')
        _require(type(row.get('prefill_terminal')) is bool, 'Missing actual terminal geometry')
        if current in events:
            seen.add(current)
            selected = events[current]['selected_tokens']
            _require(type(selected) is int and 0 <= selected <= current, 'Invalid selected active prefix')
            if part:
                result.append(part)
                part = []
            offset = current - selected
        _require(not any(current < boundary < current + q for boundary in events),
                 'Actual physical chunk crosses a refresh boundary')
        _require(row['logical_seq_len'] == current + q and row['offset'] == offset and
                 row['active_seq_len'] == current + q - offset,
                 'Actual physical geometry is not contiguous with refresh history')
        _require(not row['prefill_terminal'] or index == len(geometry)-1,
                 'Terminal geometry occurs before the last public row')
        part.append((index, row))
        current += q
    _require(geometry[-1]['prefill_terminal'], 'Actual physical input is incomplete')
    _require(seen == set(events), 'Refresh boundary has no following public geometry')
    if part:
        result.append(part)
    return result


def _prefix_chunks(tokens):
    full, tail = divmod(tokens, 2048)
    return [2048]*full + ([tail] if tail else [])


def observed_native_plan(geometry, refresh, full32_geometry):
    """Build only exact observed families, including the complete 32K control.

    Families are split even when a refresh retains the same offset. A synthetic
    native prefix establishes the active history, and an eight-token guard
    preserves nonterminal rows. Duplicate complete schedules share a family.
    Steady patterns are discovered at exact 32768 -> 49088 active boundaries;
    a shape which crosses either endpoint cannot stand in for that phase.
    """
    segments = _segments(geometry, refresh)
    _require(len(_segments(full32_geometry, [])) == 1 and
             full32_geometry[-1]['logical_seq_len'] == 32768,
             'Full32K native control must have one complete unreselected input')
    families, by_schedule, steady = {}, {}, []
    for part in segments:
        first, last = part[0][1], part[-1][1]
        prefix = _prefix_chunks(first['active_seq_len']-first['query_tokens'])
        target_shapes = [_shape(row) for _, row in part]
        chunks = prefix + [row['query_tokens'] for _, row in part]
        if not last['prefill_terminal']:
            chunks.append(8)
        key = (tuple(chunks), len(prefix), tuple(map(tuple, target_shapes)))
        name = by_schedule.get(key)
        if name is None:
            name = f'actual-{len(families):03d}'
            by_schedule[key] = name
            families[name] = dict(chunks=chunks, target_begin_index=len(prefix),
                target_shapes=target_shapes, source_spans=[])
        families[name]['source_spans'].append(dict(row_begin=part[0][0],row_end=part[-1][0]+1,
            history_begin=first['logical_seq_len']-first['query_tokens'],
            history_end=last['logical_seq_len'],offset=first['offset']))
        for start, (_, row) in enumerate(part):
            if row['active_seq_len']-row['query_tokens'] != 32768:
                continue
            selected = []
            for _, candidate in part[start:]:
                if candidate['prefill_terminal'] or candidate['active_seq_len'] > 49088:
                    break
                selected.append([candidate['query_tokens'],candidate['active_seq_len']])
                if candidate['active_seq_len'] == 49088:
                    if selected not in steady:
                        steady.append(selected)
                    break
    _require(bool(steady), 'No complete observed 32K -> 49088 steady phase')
    available = set()
    for item in families.values():
        cumulative = 0
        for index, q in enumerate(item['chunks']):
            cumulative += q
            available.add((q,cumulative,index == len(item['chunks'])-1))
    _require(all(tuple(_shape(row)) in available for row in geometry),
             'Generated native families do not cover every actual physical shape')
    full32_chunks = [row['query_tokens'] for row in full32_geometry]
    return dict(mode='observed_physical_geometry',families=families,
        steady_patterns=steady,full32_chunks=full32_chunks,
        logical2k_control_chunks=[chunk for count in full32_chunks
                                 for chunk in ([2048]*(count//2048) if count > 2048 and count%2048 == 0 else [count])],
        max_chunk=max(row['query_tokens'] for row in [*geometry,*full32_geometry]),
        scope='Native prefixes establish matching active lengths. Target chunk shapes are observed; '
              'weighted timings remain a diagnostic, not measured full256K native TTFT.')


def matching_steady_patterns(steps, patterns, span_function, refresh=()):
    """Find exact observed sequences; the span validator checks CPU/event links."""
    phases = []
    for pattern in patterns:
        expected = [tuple(shape) for shape in pattern]
        for index in range(len(steps)-len(expected)+1):
            selected = steps[index:index+len(expected)]
            actual = [(row['batch_geometry'][0]['query_tokens'],row['batch_geometry'][0]['active_seq_len'])
                      if len(row.get('batch_geometry',())) == 1 else None for row in selected]
            if actual == expected:
                begin = selected[0]['batch_geometry'][0]['logical_seq_len']-expected[0][0]
                end = selected[-1]['batch_geometry'][0]['logical_seq_len']
                _require(not any(begin < event['history_begin'] < end for event in refresh),
                         'Matched steady phase crosses an actual refresh boundary')
                phases.append(dict(shape_pattern=pattern,**span_function(selected,expected)))
    return dict(status=('no_complete_steady_phase' if not phases else
                        'available' if all(p['status']=='available' for p in phases) else 'unavailable'),
                matched_phases=len(phases),phases=phases)


def apply_observed_native_cap(argv, plan):
    """Extend only this diagnostic's legacy 2K cap from its frozen shape proof.

    The inherited native_argv helper deliberately accepts only 2K. Keeping it
    unchanged preserves older diagnostics and frozen runtime behavior.
    """
    result = list(argv)
    maximum = plan['max_chunk']
    _require(type(maximum) is int and 2048 <= maximum <= 16384 and maximum%2048 == 0,
             'Unsupported observed native cap')
    _require(all(0 < count <= maximum for item in plan['families'].values() for count in item['chunks']) and
             all(0 < count <= maximum for count in plan['full32_chunks']), 'Native cap cannot realize its schedule')
    option = '--long-prefill-token-threshold'
    _require(result.count(option) == 1 and result[result.index(option)+1] == '2048',
             'Expected the original native diagnostic 2K threshold')
    budget = '--max-num-batched-tokens'
    _require(result.count(budget) == 1 and int(result[result.index(budget)+1]) >= maximum,
             'Native batch budget cannot realize observed chunks')
    result[result.index(option)+1] = str(maximum)
    return result


def compare_output_tokens(left, right):
    """Exact token comparison, without attributing shape differences to policy."""
    common = 0
    for a,b in zip(left,right):
        if a != b:
            break
        common += 1
    equal = len(left) == len(right) == common
    return dict(equal=equal,left_tokens=len(left),right_tokens=len(right),
        common_prefix_tokens=common,first_difference=None if equal else common)
