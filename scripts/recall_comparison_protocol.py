"""Frozen RULER cohort and event contract for five-method recall comparison."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import ruler_protocol as ruler

METHODS = ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass')
VALUE_METHODS = ('exact_mass', 'block_value_norm', 'block_deletion_error')
PREFIX_PILOT_METHODS = ('mean_k', 'block_value_norm', 'block_deletion_error')
ALL_METHODS = (*METHODS, *VALUE_METHODS[1:])
CANDIDATES = METHODS[:4]
PRESSURE_WINDOW = 128
EVENT_TYPES = ('prefill_pressure', 'decode_pressure', 'query')
VERSION = 'recall-comparison-w128-v1'
SUPPORTED_CONCURRENCY = (1, 2, 4, 8)
OFFICIAL_GENERATION_LIMITS = {
    'niah': 128, 'variable_tracking': 30, 'common_words_extraction': 120,
    'freq_words_extraction': 50, 'qa': 32,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def execution_rows(rows):
    order = {task: i for i, task in enumerate(ruler.TASK_FAMILIES)}
    return sorted(rows, key=lambda row: (row['index'], order[row['task']]))


def official_task_limits():
    """Generation limits from the pinned classic RULER v1 data constants."""
    return {task: OFFICIAL_GENERATION_LIMITS[family] for task, family in ruler.TASK_FAMILIES.items()}


def validate_generation_limits(fixture):
    limits = fixture.get('task_max_tokens')
    if limits is None:
        return
    require(isinstance(limits, dict) and limits == official_task_limits() and
            all(type(value) is int for value in limits.values()),
            'Task output limits differ from pinned RULER official generation limits')
    require(fixture.get('generation_limit_policy') == 'ruler_v1_official_per_task' and
            fixture.get('starts_in_thinking') is False and
            fixture['sampling'].get('thinking_token_budget') is None,
            'Official short output limits require the declared non-thinking protocol')
    require(fixture['sampling']['max_tokens'] == max(limits.values()),
            'Default output limit must be the maximum official per-task limit')


def case_sampling(fixture, row):
    """Resolve an audited per-task response cap without touching GPU KV budgets."""
    validate_generation_limits(fixture)
    sampling = dict(fixture['sampling'], seed=row['seed'])
    if fixture.get('task_max_tokens') is not None:
        require(row['task'] in fixture['task_max_tokens'], 'Task has no frozen output limit')
        sampling['max_tokens'] = fixture['task_max_tokens'][row['task']]
    return sampling


def validate_fixture(fixture):
    require(fixture['cases'] == 260 and fixture['samples_per_task'] == 20 and
            fixture['target_length'] == 262144, 'Expected the original 260-case 256K cohort')
    ruler.validate_cohort(fixture['rows'], 20)
    require(len({row['id'] for row in fixture['rows']}) == 260, 'Duplicate sample IDs')
    for task in ruler.TASK_FAMILIES:
        require({row['index'] for row in fixture['rows'] if row['task'] == task} == set(range(20)),
                'Original sample indices changed')
    for row in fixture['rows']:
        require(Path(row['id']).name == row['id'] and row['id'] not in ('.', '..'), 'Unsafe sample ID')
        begin, end = row['served_query']
        require(0 <= begin < end <= row['served_input_tokens'], 'Invalid original Query span')
    validate_generation_limits(fixture)
    return fixture


def campaign_methods(fixture):
    if fixture.get('answer_prefix_pilot'):
        selected_methods=fixture.get('pilot_selected_methods',list(PREFIX_PILOT_METHODS))
        require(isinstance(selected_methods,list) and
                (selected_methods==list(PREFIX_PILOT_METHODS) or
                 len(selected_methods)==1 and selected_methods[0] in PREFIX_PILOT_METHODS) and
                fixture.get('methods') == selected_methods and
                fixture.get('pressure_attention_hybrid') is True and
                fixture.get('value_recall_reference') is True and
                fixture.get('full_value_comparison') is False and
                fixture.get('answer_prefix_policy') == 'official_after_assistant_template_v1' and
                fixture.get('starts_in_thinking') is False,
                'Answer-prefix pilot requires its declared method matrix and hybrid protocol')
        wanted = set(fixture['qualification_ids'])
        selected = [row for row in fixture['rows'] if row['id'] in wanted]
        counts = Counter(row['task'] for row in selected)
        count = fixture.get('pilot_samples_per_task')
        require(type(count) is int and 0 < count <= 20 and
                counts == {task: count for task in ruler.TASK_FAMILIES} and
                len(wanted) == len(selected) and
                all(row['index'] < count for row in selected),
                'Pilot must use the same first N original cases of every task')
        return tuple(selected_methods)
    if fixture.get('value_recall_reference'):
        if fixture.get('pressure_attention_hybrid'):
            require(fixture['methods'] == list(VALUE_METHODS[1:]) and
                    fixture.get('full_value_comparison') is False,
                    'Hybrid campaign must declare its two Query methods separately')
            return VALUE_METHODS[1:]
        require(fixture['methods'] == list(VALUE_METHODS), 'Value-reference method matrix changed')
        require(fixture.get('full_value_comparison') is True, 'Value comparison must preserve the full 260-case cohort')
        return VALUE_METHODS
    require(fixture['methods'] == list(METHODS), 'Frozen method matrix changed')
    return METHODS


def validate_profile(profile):
    k = profile['kvmem']
    require(type(profile['max_num_seqs']) is int and profile['max_num_seqs'] in SUPPORTED_CONCURRENCY,
            'Recall comparisons require explicitly frozen owners in (1, 2, 4, 8)')
    require(profile.get('recall_experiment', {}).get('concurrent_owners', 1) == profile['max_num_seqs'],
            'Recall owner declaration differs from the frozen engine concurrency')
    require(k['block_tokens'] == 32 and k['select_budget'] > 0 and k['gen_budget'] > 0,
            'Invalid common physical block or KV budgets')
    fixed = (k.get('sink_blocks', 0) + k.get('recent_blocks', 0)) * k['block_tokens']
    require(fixed < k['select_budget'], 'No optional historical block budget remains')
    require(k.get('graph_alpha', 0) == 0 and k.get('guided_query_tokens', 0) == 0 and
            not k.get('query_history', False) and k.get('query_fps_halo_blocks', 0) == 0 and
            k.get('query_mean_quota_fraction', 0) == 0,
            'Graph, Guided Query, Halo, Query history and hybrid quota must be disabled')
    require(not k.get('repository_gc', False), 'History deletion is outside this comparison')
    return profile


def method_environment(method):
    require(method in ALL_METHODS, 'Unknown recall method')
    return dict(KVMEM_RECALL_METHOD=method, KVMEM_PRESSURE_QUERY_WINDOW=str(PRESSURE_WINDOW))


def method_profile(base, method):
    require(method in ALL_METHODS, 'Unknown recall method')
    profile = deepcopy(base)
    settings = dict(profile.get('recall_experiment') or {})
    settings.update(method=method, pressure_query_window=PRESSURE_WINDOW)
    # Value kernels have a separate larger scratch tiling allowance. Keep the
    # original representative scorers within their native bound; no candidates,
    # Query rows, per-owner budgets or score definitions change.
    if method not in VALUE_METHODS[1:] and settings.get('block_batch', 32) > 128:
        settings['block_batch'] = 128
    profile['recall_experiment'] = settings
    return profile


def source_manifest(runtime):
    runtime = Path(runtime)
    files = []
    for directory in ('src/vllm_kvmem', 'csrc', 'scripts', 'third_party/qw3'):
        files.extend(path for path in (runtime / directory).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.cpp', '.hpp', '.h', '.cu', '.cuh'))
    for name in ('pyproject.toml', 'setup.py', 'CMakeLists.txt'):
        if (runtime / name).is_file():
            files.append(runtime / name)
    return {str(path.relative_to(runtime)): sha(path) for path in sorted(set(files))}


def verify_sources(runtime, expected):
    require(source_manifest(runtime) == expected, 'Frozen implementation sources changed')


def binary_manifest(runtime, extra_paths=()):
    """Bind target-host native builds separately from portable source hashes."""
    runtime = Path(runtime).resolve()
    paths = {path for path in (runtime / 'src/vllm_kvmem').rglob('*.so') if path.is_file()}
    paths.update(Path(path).resolve() for path in extra_paths)
    result = {}
    for path in sorted(paths):
        require(path.is_file(), 'Missing compiled binary: ' + str(path))
        try:
            name = str(path.relative_to(runtime))
        except ValueError:
            name = str(path.resolve())
        result[name] = sha(path)
    return result


def verify_binaries(runtime, expected):
    extra = [Path(name) for name in expected if Path(name).is_absolute()]
    require(binary_manifest(runtime, extra) == expected, 'Frozen compiled binaries changed')


def events_for_response(events, response, session):
    import qualify_block_graph as q
    return [event for event in events if q.native_matches(response, event) or
            event.get('session_id') == session or event.get('sample_id') == session]


def audit_events(events, method, profile, *, query_span=None, required_types=()):
    """No mixed-method fallback may pass as a pure-method result."""
    require(method in ALL_METHODS and events, 'Missing recall events')
    counts = Counter()
    actual_methods = set()
    settings = profile.get('recall_experiment', {})
    hybrid = any(settings.get(stage + '_pressure_method') == 'accumulated_attention'
                 for stage in ('prefill', 'decode'))
    for event in events:
        require(all(key in event for key in ('sample_id', 'event_type', 'configured_method',
            'actual_method', 'query_source', 'candidate_block_count', 'selected_block_ids',
            'fallback_reason', 'query_begin', 'query_end', 'accepted_through', 'selected_tokens', 'budget')),
            'Incomplete recall event contract')
        kind = event['event_type']
        require(kind in EVENT_TYPES, 'Unknown reselection event type')
        accumulated = kind.endswith('_pressure') and settings.get(kind.removesuffix('_pressure') + '_pressure_method') == 'accumulated_attention'
        expected_method = 'accumulated_attention' if accumulated else method
        require(event['configured_method'] == method and event['actual_method'] == expected_method,
                'Mixed-method configuration or actual method mismatch')
        if hybrid:
            require(event.get('configuration_kind') == 'pressure_attention_query_recall' and
                    event.get('configured_event_method') == expected_method and
                    all(event.get(stage + '_pressure_method') == settings.get(stage + '_pressure_method', 'same_method')
                        for stage in ('prefill', 'decode')),
                    'Stage-specific configuration was not declared in the event')
        if not accumulated and profile.get('recall_experiment',{}).get('include_mandatory_in_softmax'):
            require(event.get('attention_normalization_domain')=='candidate_and_mandatory' and
                    event.get('mandatory_in_normalization') is True and
                    event.get('score_mask_count')==event['candidate_block_count'],
                    'Value reference excluded candidate or mandatory attention tokens')
        require(event['fallback_reason'] in (None, ''), 'Recall fallback is not a pure-method run')
        require(isinstance(event['query_source'], str) and event['query_source'], 'Missing Query provenance')
        begin, end, accepted = event['query_begin'], event['query_end'], event['accepted_through']
        require(all(type(x) is int for x in (begin, end, accepted)) and 0 <= begin < end and accepted >= 0,
                'Query used an invalid token range')
        if accumulated:
            require(end <= accepted and begin == 0 and
                    event.get('attention_normalization_domain') == 'sampled_causal_active_windows',
                    'Cumulative pressure attention used unaccepted provenance')
            if event.get('attention_source') == 'prefill':
                require(event['query_source'] == 'accepted_prefill_accumulation' and
                    event.get('prefill_attention_samples', 0) > 0 and
                    event.get('prefill_attention_chunks', 0) > 0 and
                    event.get('prefill_score_normalization') == 'decayed_mass_over_exposure',
                    'Missing accepted cumulative prefill attention provenance')
            else:
                require(kind == 'decode_pressure' and event.get('attention_source') == 'output' and
                        event['query_source'] == 'accepted_output_accumulation' and
                        event.get('output_score_normalization') == 'existing_action_decay_and_sample_mean',
                        'Missing accepted cumulative output attention provenance')
        elif kind.endswith('_pressure'):
            require(end <= accepted, 'Pressure Query used an unaccepted token')
            require(end == accepted and begin == max(0, end - PRESSURE_WINDOW),
                    'Pressure Query is not the last 128 accepted tokens')
        else:
            processed, public_end = event.get('query_processed_through'), event.get('public_input_end')
            require(type(processed) is int and type(public_end) is int and end <= processed <= public_end,
                    'User Query exceeded its processed public-input boundary')
            if query_span is not None:
                require([begin, end] == list(query_span), 'User Query coordinates differ from frozen input')
        budget, selected = event['budget'], event['selected_tokens']
        require(type(budget) is int and type(selected) is int and
                0 < selected <= budget <= profile['kvmem']['select_budget'], 'Active KV budget violation')
        blocks = event['selected_block_ids']
        require(isinstance(blocks, list) and blocks and
                len({json.dumps(x, sort_keys=True) for x in blocks}) == len(blocks),
                'Missing or duplicate selected blocks')
        require(type(event['candidate_block_count']) is int and event['candidate_block_count'] > 0,
                'Missing candidate domain size')
        require(not event.get('propagation_applied', False) and not event.get('guided', False),
                'An excluded recall enhancement executed')
        counts[kind] += 1
        actual_methods.add(event['actual_method'])
    require(set(required_types) <= counts.keys(), 'Not all required reselection entrances executed')
    return dict(passed=True, method=method, window=PRESSURE_WINDOW, event_counts=dict(counts),
                fallback_count=0, actual_methods=sorted(actual_methods))


def score_case(row, response, scorer, *, starts_in_thinking):
    answer = ruler.final_answer(response['text'], starts_in_thinking=starts_in_thinking)
    config = scorer[0][ruler.TASK_FAMILIES[row['task']]]
    prediction = scorer[1](answer['text'], config)
    matches = [ref.lower() in prediction.lower() for ref in row['outputs']]
    fraction = float(any(matches)) if ruler.TASK_FAMILIES[row['task']] == 'qa' else sum(matches) / len(matches)
    raw = 100. * fraction
    official = config['metric_fn']([prediction], [row['outputs']])
    require(round(raw, 2) == official, 'Unrounded substring score disagrees with frozen official metric')
    return dict(answer=answer['text'], thinking_complete=answer['thinking_complete'],
        official_case_score=official, case_score_unrounded=raw, case_coverage_fraction=fraction,
        full_reference_recall=(any(matches) if ruler.TASK_FAMILIES[row['task']] == 'qa' else all(matches)))


def compare_results(rows, arms):
    """Paired complete-run oracle, with the same task weights as fixed methods."""
    require(set(arms) == set(METHODS), 'Need all five methods, including the separate diagnostic')
    ids = {row['id'] for row in rows}
    require(len(ids) == len(rows), 'Duplicate comparison sample')
    indexed = {}
    for method, values in arms.items():
        indexed[method] = {row['id']: row for row in values}
        require(len(values) == len(indexed[method]) and set(indexed[method]) == ids,
                'Missing, duplicate or unexpected paired results')
        for row in rows:
            value = indexed[method][row['id']]
            require(value.get('status') == 'completed' and value.get('method') == method,
                    'Unqualified result in comparison')
            require(value['input_sha256'] == row['served_sha256'] and value['task'] == row['task'] and
                    value['index'] == row['index'], 'Paired result identity/input mismatch')
            score = value['official_case_score']
            require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 100,
                    'Invalid official case score')
    tasks = sorted({row['task'] for row in rows})
    by_task = {task: [row['id'] for row in sorted(rows, key=lambda r: r['index']) if row['task'] == task]
               for task in tasks}
    fraction_of = lambda method, ident: indexed[method][ident].get('case_coverage_fraction',
        indexed[method][ident].get('case_score_unrounded', indexed[method][ident]['official_case_score']) / 100.)
    # Match the frozen scorer's summation and rounding order, including thirds.
    table = {task: {method: round(sum(fraction_of(method, ident)
                    for ident in by_task[task]) / len(by_task[task]) * 100, 2)
                    for method in METHODS} for task in tasks}
    macros = {method: statistics.fmean(table[task][method] for task in tasks) for method in METHODS}
    pairs = {}
    for i, left in enumerate(CANDIDATES):
        for right in CANDIDATES[i + 1:]:
            per_task = {}
            for task, group in by_task.items():
                delta = [fraction_of(left, ident) - fraction_of(right, ident)
                         for ident in group]
                per_task[task] = dict(wins=sum(x > 0 for x in delta), losses=sum(x < 0 for x in delta),
                                     ties=sum(x == 0 for x in delta))
            pairs[left + '__vs__' + right] = dict(left=left, right=right, per_task=per_task,
                **{name: sum(value[name] for value in per_task.values()) for name in ('wins', 'losses', 'ties')})
    unique = {method: [ident for ident in sorted(ids) if indexed[method][ident]['full_reference_recall'] and
                all(not indexed[other][ident]['full_reference_recall'] for other in CANDIDATES if other != method)]
              for method in CANDIDATES}
    oracle_per_task = {task: round(sum(max(fraction_of(method, ident)
        for method in CANDIDATES) for ident in group) / len(group) * 100, 2) for task, group in by_task.items()}
    best = max(macros[method] for method in CANDIDATES)
    oracle = statistics.fmean(oracle_per_task.values())
    return dict(cases=len(rows), per_task=table, macro_scores=macros, pairwise=pairs,
        unique_successes={method: dict(count=len(values), sample_ids=values,
            per_task={task: sum(ident in values for ident in group) for task, group in by_task.items()})
            for method, values in unique.items()},
        potential=dict(candidate_methods=list(CANDIDATES), best_fixed_score=best,
            best_fixed_methods=[method for method in CANDIDATES if macros[method] == best],
            per_request_oracle_score=oracle, gain_points=oracle - best, oracle_per_task=oracle_per_task,
            weighting='Mean over samples within each task, then equal mean over tasks.',
            scope='Post-hoc best whole independent request; not an implemented router or event-wise splice.'),
        diagnostic='ExactMass is reported separately and is not assumed to be a task-accuracy upper bound.',
        success_definition='All official reference targets recalled; QA uses any accepted reference. Not strict exact match.')
