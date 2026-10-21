"""Whole-model native shape reference plus matched complete-input controls.

Shape-weighted timing is a diagnostic counterfactual, never an end-to-end
256K throughput result. Native/KVMem full32K TTFT is a separate direct control;
KVMem full256K retains all statistics and reselections for the real workload.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from benchmark_prefill_publication_pipeline import (read, save, sha, require, rows, digest, runtime_origins,
    owned_process_memory, sampled_process_memory, PROCESS_MEMORY_SCOPE)
from active_prefill_shapes import (observed_native_plan, matching_steady_patterns,
                                  apply_observed_native_cap, compare_output_tokens)


FAMILIES = {
    'regular': [2048]*23 + [1984, 8],
    'steady': [2048]*16 + [32] + [2048]*7 + [1952, 8],
    'history-tail': [2048]*16 + [32, 736, 8],
    'query-tail': [2048]*16 + [22, 7],
}

# This is the uninterrupted post-refresh public phase, before the next
# pressure boundary. The first 32K and the following guard are excluded.
STEADY_SHAPES = [(32, 32800), *((2048, n) for n in range(34848, 47137, 2048)),
                 (1952, 49088)]
CONTINUITY_FIELDS = ('call_index', 'previous_sequence', 'previous_call_index',
                     'previous_end_to_model_ms')


def continuous_phase_span(steps, expected_shapes=STEADY_SHAPES):
    """Actual first-model to last-end span, not a weighted counterfactual.

    Retained end events link only completed rows. Interior links include all
    elapsed maintenance/host idle time. The first link is deliberately omitted,
    so prefix establishment and the preceding pressure are outside this phase.
    """
    require(len(steps) == len(expected_shapes) and bool(steps), 'Continuous phase length differs')
    owner = offset = previous_shape = None
    for row, expected in zip(steps, expected_shapes):
        geometry = row.get('batch_geometry', ())
        require(len(geometry) == 1, 'Continuous phase requires one owner')
        shape = geometry[0]
        require(shape.get('prefill') is True and not shape.get('prefill_terminal', False) and
                (shape['query_tokens'], shape['active_seq_len']) == expected,
                'Continuous phase shape/order or terminal status differs')
        require(row.get('decode_requests', 0) == 0 and row.get('atomic_prefill_tokens', 0) == 0,
                'Continuous phase contains decode or private replay')
        current_offset = shape['logical_seq_len'] - shape['active_seq_len']
        require(shape.get('offset', current_offset) == current_offset, 'Continuous phase offset disagrees')
        if previous_shape is not None:
            require(shape['request_id'] == owner and current_offset == offset and
                    shape['logical_seq_len'] == previous_shape['logical_seq_len'] + shape['query_tokens'],
                    'Continuous phase owner, offset, or logical continuity changed')
        else:
            owner, offset = shape['request_id'], current_offset
        previous_shape = shape
    missing = sorted({field for row in steps for field in (*CONTINUITY_FIELDS, 'model_ms') if field not in row})
    if missing:
        return dict(status='unavailable', reason='historical_or_incomplete_timing_fields', missing_fields=missing,
                    request_id=owner, steps=len(steps))
    models, gaps = [], []
    previous = None
    for row in steps:
        require(type(row.get('sequence')) is int and type(row['call_index']) is int,
                'Continuous phase sequence/call index is invalid')
        model = row['model_ms']
        require(type(model) in (int, float) and math.isfinite(model) and model >= 0,
                'Continuous phase model interval is invalid')
        models.append(model)
        if previous is not None:
            require(row['sequence'] == previous['sequence'] + 1 and
                    row['call_index'] == previous['call_index'] + 1 and
                    row['previous_sequence'] == previous['sequence'] and
                    row['previous_call_index'] == previous['call_index'],
                    'Continuous phase contains a dropped, sampled, or unlinked step')
            gap = row['previous_end_to_model_ms']
            require(type(gap) in (int, float) and math.isfinite(gap) and gap >= 0,
                    'Continuous phase CUDA gap is invalid')
            gaps.append(gap)
        previous = row
    span = (sum(models) + sum(gaps)) / 1000
    require(span > 0, 'Continuous phase elapsed span must be positive')
    tokens = sum(q for q, _ in expected_shapes)
    return dict(status='available', request_id=owner, steps=len(steps), new_input_tokens=tokens,
        first_sequence=steps[0]['sequence'], last_sequence=steps[-1]['sequence'], offset=offset,
        model_seconds=sum(models)/1000, interior_gap_seconds=sum(gaps)/1000,
        model_start_to_last_end_seconds=span, new_tokens_per_second=tokens/span,
        scope='Actual uninterrupted public phase including interior maintenance/host gaps; '
              'excludes prefix establishment, preceding/following pressure, HTTP and final generation. '
              'Not full-request TTFT or full256K native throughput.')


def continuous_steady_phases(steps, patterns=None, refresh=()):
    if patterns is not None:
        return matching_steady_patterns(steps, patterns, continuous_phase_span, refresh)
    phases = []
    for index in range(len(steps)-len(STEADY_SHAPES)+1):
        selected = steps[index:index+len(STEADY_SHAPES)]
        actual = [(row['batch_geometry'][0]['query_tokens'], row['batch_geometry'][0]['active_seq_len'])
                  if len(row.get('batch_geometry', ())) == 1 else None for row in selected]
        if actual == STEADY_SHAPES:
            phases.append(continuous_phase_span(selected))
    return dict(status=('no_complete_steady_phase' if not phases else
                        'available' if all(p['status'] == 'available' for p in phases) else 'unavailable'),
                matched_phases=len(phases), phases=phases)


def reconstruct_geometry(steps, refresh, input_tokens):
    """Exact CPU geometry from serial public rows and committed offset changes."""
    events = {event['history_begin']: event for event in refresh}
    require(len(events) == len(refresh), 'Duplicate refresh boundary')
    current = offset = 0
    result = []
    for step in steps:
        if current in events:
            offset = current-events[current]['selected_tokens']
        count = step['prefill_tokens']
        require(count > 0 and not any(current < pos < current+count for pos in events),
                'A public batch crossed a refresh boundary')
        row = dict(query_tokens=count, active_seq_len=current+count-offset,
                   logical_seq_len=current+count, offset=offset, prefill=True,
                   prefill_terminal=(current+count == input_tokens))
        actual = step.get('batch_geometry')
        if actual is not None:
            recorded = dict(actual[0]) if len(actual) == 1 else {}
            if 'logical_seq_len' in recorded and 'active_seq_len' in recorded:
                recorded.setdefault('offset', recorded['logical_seq_len']-recorded['active_seq_len'])
                recorded.setdefault('prefill_terminal', recorded['logical_seq_len'] == input_tokens)
            require(len(actual) == 1 and all(recorded.get(key) == value for key, value in row.items()),
                    'Recorded CPU geometry disagrees with committed history reconstruction')
        result.append(row)
        current += count
    require(current == input_tokens, 'Incomplete original public input')
    return result


def reference_inputs(repo, reference=None):
    """Read one qualified reference without confusing suite order with shape order."""
    import benchmark_prefill_publication_pipeline as pipeline
    explicit = reference is not None
    reference = Path(reference) if explicit else Path('artifacts/prefill-pipeline-20261021-r1/on-a')
    reference = (reference if reference.is_absolute() else repo / reference).resolve()
    profile = read(reference / 'profile.json')
    require(profile['kvmem']['action_native_scores'] == 'reuse' and profile['kvmem']['prefill_publication_pipeline'],
            'Expected the qualified native-reuse pipeline baseline')
    overrides = {}
    if not explicit:
        # Preserve the original diagnostic protocol when no reference is given.
        profile['kvmem']['prefill_native_plan_arm'] = False
        overrides['prefill_native_plan_arm'] = False
    suite = read(reference / 'suite.json')['cases']
    matches = [index for index, item in enumerate(suite) if item['id'] == '256k-repeat-1']
    require(len(matches) == 1, 'Reference suite must identify exactly one 256k-repeat-1 case')
    groups = pipeline.prefill_groups(rows(reference / 'run/server/capture/model_step_latency.jsonl'))
    require(len(groups) == len(suite), 'Reference prefill grouping differs from suite case order')

    def case_path(ident):
        path = reference / 'requests' / ident / 'case.json'
        items = [item for item in suite if item['id'] == ident]
        require(len(items) == 1, 'Reference suite case is missing or duplicated: ' + ident)
        expected = items[0].get('case_sha256')
        require(expected is None or sha(path) == expected, 'Reference case changed: ' + ident)
        return path

    source32 = repo / 'artifacts/matched-32k-prefill-20261021-r2/kvmem/requests/32k-repeat-1/case.json'
    source32_ident = None
    if explicit:
        for ident in ('32k-repeat-1', 'warmup-32k'):
            if (reference / 'requests' / ident / 'case.json').is_file():
                source32 = case_path(ident)
                source32_ident = ident
                break
    source256 = case_path('256k-repeat-1')
    cases32 = read(source32)
    require(cases32['row']['served_input_tokens'] == 32768, 'Formal 32K case must contain all 32768 tokens')
    cases256 = read(source256)
    refresh = read(reference / 'requests/256k-repeat-1/evidence.json')['refresh']
    geometry = reconstruct_geometry(groups[matches[0]], refresh,
                                    cases256['row']['served_input_tokens'])
    shape_plan = None
    if profile['kvmem'].get('prefill_physical_coalescing', False):
        require(explicit and source32_ident is not None, 'Coalesced reference requires its actual full32K case')
        require(read(reference/'status.json')['status'] == 'completed', 'Coalesced reference is not complete')
        index32 = next(i for i,item in enumerate(suite) if item['id'] == source32_ident)
        require(all(len(step.get('batch_geometry',())) == 1 for step in [*groups[matches[0]],*groups[index32]]),
                'Coalesced reference requires recorded physical geometry for every public row')
        geometry32 = reconstruct_geometry(groups[index32],
            read(reference/'requests'/source32_ident/'evidence.json')['refresh'],32768)
        shape_plan = observed_native_plan(geometry,refresh,geometry32)
        proof_names = ['profile.json','status.json','suite.json','run/server/capture/model_step_latency.jsonl',
            'requests/256k-repeat-1/case.json','requests/256k-repeat-1/evidence.json',
            f'requests/{source32_ident}/case.json',f'requests/{source32_ident}/evidence.json']
        shape_plan['source_evidence_sha256'] = {name:sha(reference/name) for name in proof_names}
        shape_plan['source_reference'] = str(reference)
    output_controls = None
    if shape_plan is not None:
        output_controls = dict(samples=[],unavailable=[])
        paths = [('kvmem-coalesced-reference',reference)]
        if reference.name == 'candidate':
            other = reference.parent/'reference'
            mode = (read(other/'profile.json')['kvmem'].get('prefill_physical_coalescing',False)
                    if (other/'profile.json').exists() else False)
            paths.append(('kvmem-coalesced-baseline' if mode else 'kvmem-logical2k-reference',other))
        for label,arm in paths:
            if not (arm/'status.json').exists() or read(arm/'status.json')['status'] != 'completed':
                output_controls['unavailable'].append(dict(label=label,path=str(arm),reason='not_completed'))
                continue
            require(read(arm/'runtime-proof.json')['files'] == read(reference/'runtime-proof.json')['files'],
                    'Output control uses different frozen runtime source')
            if label == 'kvmem-logical2k-reference':
                require(not read(arm/'profile.json')['kvmem'].get('prefill_physical_coalescing',False),
                        'Logical2K output reference unexpectedly enables physical coalescing')
            for item in read(arm/'suite.json')['cases']:
                if not item['id'].startswith('32k-repeat-'):
                    continue
                directory = arm/'requests'/item['id']
                definition = read(directory/'case.json')
                require(definition['row']['served_sha256'] == cases32['row']['served_sha256'] and
                        definition['sampling'] == cases32['sampling'],'Output control input/sampling differs')
                response = directory/'response.json'
                if not response.exists():
                    output_controls['unavailable'].append(dict(label=label,path=str(response),reason='missing_response'))
                    continue
                output_controls['samples'].append(dict(label=label,id=item['id'],path=str(response),
                    response_sha256=sha(response),output_ids=read(response)['output_ids']))
    return dict(reference=reference, profile=profile, cases32=cases32, cases256=cases256, geometry=geometry,
                launch=read(reference / 'run/server/launch.json'), preparation_overrides=overrides,
                case_sources={'full32k': str(source32), 'full256k': str(source256)},
                geometry_suite_index=matches[0], native_shape_plan=shape_plan,output_controls=output_controls)


def prepare(root, repo, reference=None, *, flashinfer_autotune_seed=None,
            flashinfer_autotune_hashes=(), inductor_seed_sources=None, inductor_seed_hashes=None):
    require((flashinfer_autotune_seed is None) == (not flashinfer_autotune_hashes),
            'FlashInfer seed and allowed hashes must be supplied together')
    require((inductor_seed_sources is None) == (inductor_seed_hashes is None),
            'Inductor module sources and allowed hashes must be supplied together')
    inputs = reference_inputs(repo, reference)
    reference, profile = inputs['reference'], inputs['profile']
    cases32, cases256, geometry = inputs['cases32'], inputs['cases256'], inputs['geometry']
    shape_plan = inputs['native_shape_plan']
    output_controls = inputs['output_controls']
    families = ({name: item['chunks'] for name,item in shape_plan['families'].items()}
                if shape_plan is not None else FAMILIES)
    provenance = dict(source_reference=str(reference), preparation_overrides=inputs['preparation_overrides'],
                      case_sources=inputs['case_sources'], geometry_suite_index=inputs['geometry_suite_index'])
    names = set(read(reference / 'runtime-proof.json')['files'])
    names.update(str(p.relative_to(repo)) for p in (repo / 'src/vllm_kvmem').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.cuh', '.so'))
    names.update(('scripts/benchmark_active_prefill.py', 'scripts/native_prefill_shape_probe.py',
                  'scripts/active_prefill_shapes.py', 'scripts/benchmark_native_prefill.py',
                  'scripts/prefill_autotune_control.py', 'scripts/prefill_inductor_control.py',
                  'scripts/prefill_inductor_worker.py', 'scripts/prefill_inductor_serve.py'))
    origins, inherited = runtime_origins(repo, reference, names)
    hashes = {name: sha(origin) for name, origin in origins.items()}
    root.mkdir(parents=True, exist_ok=False)
    proof = dict(files=hashes, source=str(repo), inherited_utilities=inherited,
                 head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), dirty=True)
    for arm in ('native', 'kvmem'):
        target = root / arm
        target.mkdir()
        for name, origin in origins.items():
            destination = target / 'runtime' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            require(sha(destination) == hashes[name], 'Source changed while freezing: ' + name)
        save(target / 'runtime-proof.json', proof)
        save(target / 'profile.json', profile)
    require(all(sha(origin) == hashes[name] for name, origin in origins.items()), 'Source changed during freeze')
    # The native argv comes from the actual qualified launch, not a manually
    # reconstructed approximation of dtype/backend/MTP/graph settings.
    save(root / 'native/reference-launch.json', inputs['launch'])
    inputs = root / 'inputs'
    inputs.mkdir()
    full = Path(cases256['input_path']).read_bytes()
    require(sha(Path(cases256['input_path'])) == cases256['row']['served_sha256'], 'Original input changed')
    definitions = []

    def add_native(ident, family, warmup=False, instrumented=True):
        directory = root / 'native/requests' / ident
        directory.mkdir(parents=True)
        if family in ('full32k','logical2k-control'):
            definition = deepcopy(cases32)
            chunks = (shape_plan['logical2k_control_chunks'] if family == 'logical2k-control' else
                      shape_plan['full32_chunks'] if shape_plan is not None else None)
        elif family == 'flush':
            definition = deepcopy(cases256)
            chunks = [1]
        else:
            definition = deepcopy(cases256)
            chunks = families[family]
        if chunks is not None and family not in ('full32k','logical2k-control'):
            prompt = inputs / (family + '.i32')
            if not prompt.exists():
                prompt.write_bytes(full[:4*sum(chunks)])
            definition.update(input_path=str(prompt), sampling={**cases256['sampling'], 'max_tokens':8 if family != 'flush' else 1})
            definition['row'] = dict(served_input_tokens=sum(chunks), served_sha256=sha(prompt))
        definition.update(id=ident, request_id='active-probe-' + ident, family=family,
                          instrumented=instrumented, warmup=warmup, chunks=chunks)
        save(directory / 'case.json', definition)
        definitions.append(dict(id=ident, request_id=definition['request_id'], family=family,
            input_tokens=definition['row']['served_input_tokens'], instrumented=instrumented,
            chunks=chunks, warmup=warmup, case_sha256=sha(directory / 'case.json')))

    for family in families:
        add_native('warmup-' + family, family, warmup=True)
    add_native('warmup-full32k', 'full32k', warmup=True)
    for repeat in range(1, 4):
        for family in families:
            add_native(f'{family}-{repeat}', family)
        # Alternate instrumentation order to avoid consistently assigning the
        # later/warm-cache request to one side of the perturbation check.
        enabled_order = (True, False) if repeat % 2 else (False, True)
        for enabled in enabled_order:
            add_native(f'full32k-{repeat}-' + ('timed' if enabled else 'untimed'),
                       'full32k', instrumented=enabled)
        if shape_plan is not None and repeat <= 2:
            add_native(f'logical2k-control-{repeat}','logical2k-control')
    add_native('flush', 'flush', warmup=True, instrumented=False)
    save(root / 'native/suite.json', dict(cases=definitions,
        native_prefill_chunk=shape_plan['max_chunk'] if shape_plan is not None else 2048))
    if shape_plan is not None:
        save(root/'native-shape-plan.json',shape_plan)
        save(root/'output-control-reference.json',output_controls)
    kvmem = []
    for ident, case, warmup in [('warmup-32k', cases32, True), ('32k-repeat-1', cases32, False),
            ('32k-repeat-2', cases32, False), ('256k-repeat-1', cases256, False), ('256k-repeat-2', cases256, False)]:
        target = root / 'kvmem/requests' / ident
        target.mkdir(parents=True)
        save(target / 'case.json', case)
        kvmem.append(dict(id=ident, warmup=warmup, case_sha256=sha(target / 'case.json')))
    save(root / 'kvmem/suite.json', dict(cases=kvmem, prefix_caching=False, serial=True))
    save(root / 'kvmem/manifest.json', dict(profile_sha256=sha(root / 'kvmem/profile.json'),
        scope='Qualified complete full32K and full256K, native QK reuse+bounded publication pipeline; '
              'optimizations follow the recorded profile and preparation overrides; no reduced statistics, no DSH.',
        prefill_native_plan_arm=profile['kvmem'].get('prefill_native_plan_arm', False), **provenance))
    save(root / 'native/manifest.json', dict(profile_sha256=sha(root / 'native/profile.json'),
        suite_sha256=sha(root / 'native/suite.json'),
        scope='Instrumented native shape reference, without KVMem capture/statistics/archive hooks.', **provenance))
    for arm in ('native', 'kvmem'):
        if flashinfer_autotune_seed is not None:
            from prefill_autotune_control import freeze_seed
            freeze_seed(root / arm, flashinfer_autotune_seed, flashinfer_autotune_hashes)
        if inductor_seed_sources is not None:
            from prefill_inductor_control import freeze_seed as freeze_inductor_seed
            freeze_inductor_seed(root / arm, inductor_seed_sources, inductor_seed_hashes)
    save(root / 'reference-geometry.json', dict(source=str(reference), input_tokens=cases256['row']['served_input_tokens'],
        suite_case_id='256k-repeat-1', suite_case_index=provenance['geometry_suite_index'],
        rows=geometry, shape_counts=[dict(query_tokens=q, active_seq_len=k, prefill_terminal=t, count=count)
            for (q,k,t),count in sorted(Counter((x['query_tokens'],x['active_seq_len'],x['prefill_terminal']) for x in geometry).items())]))
    driver = root / 'driver.py'
    shutil.copy2(Path(__file__), driver)
    # The root driver imports only stdlib accounting helpers from this folder.
    shutil.copy2(repo / 'scripts/benchmark_prefill_publication_pipeline.py', root / 'benchmark_prefill_publication_pipeline.py')
    shutil.copy2(repo / 'scripts/active_prefill_shapes.py', root / 'active_prefill_shapes.py')
    seed_helpers = ('prefill_autotune_control.py', 'prefill_inductor_control.py')
    for name in seed_helpers:
        shutil.copy2(repo / 'scripts' / name, root / name)
        require(sha(root / name) == hashes['scripts/' + name], 'Seed helper changed during freeze')
    require(all(sha(origin) == hashes[name] for name, origin in origins.items()), 'Source changed during preparation')
    save(root / 'plan.json', dict(driver_sha256=sha(driver),
        helper_sha256=sha(root / 'benchmark_prefill_publication_pipeline.py'), runtime_sha256=digest(hashes),
        shape_helper_sha256=sha(root/'active_prefill_shapes.py'),
        native_shape_plan_sha256=sha(root/'native-shape-plan.json') if shape_plan is not None else None,
        output_control_reference_sha256=sha(root/'output-control-reference.json') if shape_plan is not None else None,
        reference_geometry_sha256=sha(root/'reference-geometry.json'),
        native_suite_sha256=sha(root/'native/suite.json'),
        seed_helper_sha256={name: sha(root / name) for name in seed_helpers},
        flashinfer_autotune_seed_sha256=sha(flashinfer_autotune_seed) if flashinfer_autotune_seed is not None else None,
        flashinfer_seed_manifest_sha256={name: read(root / name / 'manifest.json').get(
            'flashinfer_seed_manifest_sha256') for name in ('native', 'kvmem')},
        inductor_seed_manifest_sha256={name: read(root / name / 'manifest.json').get(
            'inductor_seed_manifest_sha256') for name in ('native', 'kvmem')},
        **provenance, native_shape_rounds=3, native_full32k_timed=3,
        native_logical2k_controls=2 if shape_plan is not None else 0,
        native_full32k_untimed=3, kvmem_full32k=2, kvmem_full256k=2,
        note='Native instrumentation-disabled requests retain tiny diagnostic dispatch wrappers. '
             'Weighted model intervals are not measured end-to-end 256K throughput.'))
    print(root)


def assert_frozen(root, arm):
    require(sha(root / 'driver.py') == read(root / 'plan.json')['driver_sha256'], 'Frozen driver changed')
    require(sha(root / 'benchmark_prefill_publication_pipeline.py') == read(root / 'plan.json')['helper_sha256'],
            'Frozen helper changed')
    require(sha(root/'active_prefill_shapes.py') == read(root/'plan.json')['shape_helper_sha256'],
            'Frozen shape helper changed')
    expected_shapes = read(root/'plan.json').get('native_shape_plan_sha256')
    if expected_shapes is not None:
        require(sha(root/'native-shape-plan.json') == expected_shapes,'Frozen native shape plan changed')
        require(sha(root/'output-control-reference.json') == read(root/'plan.json')['output_control_reference_sha256'],
                'Frozen output control reference changed')
    require(sha(root/'reference-geometry.json') == read(root/'plan.json')['reference_geometry_sha256'],
            'Frozen reference geometry changed')
    require(sha(root/'native/suite.json') == read(root/'plan.json')['native_suite_sha256'],
            'Frozen native suite changed')
    target = root / arm
    proof = read(target / 'runtime-proof.json')['files']
    require(digest(proof) == read(root / 'plan.json')['runtime_sha256'], 'Runtime manifest differs')
    require(all(sha(target / 'runtime' / name) == value for name, value in proof.items()), 'Frozen source changed')
    assert_seed_plan(root)


def assert_seed_plan(root):
    """Bind both optional seed chains to the original preparation plan."""
    plan = read(root / 'plan.json')
    if 'seed_helper_sha256' not in plan:
        return  # Historical artifacts predate controlled autotuning.
    from prefill_autotune_control import validate_seed
    from prefill_inductor_control import validate_seed as validate_inductor_seed
    require(set(plan['seed_helper_sha256']) == {'prefill_autotune_control.py', 'prefill_inductor_control.py'},
            'Missing frozen seed helper')
    for name, expected in plan['seed_helper_sha256'].items():
        require(name in ('prefill_autotune_control.py', 'prefill_inductor_control.py'), 'Unknown seed helper')
        require(sha(root / name) == expected, 'Frozen seed helper changed')
    for name in ('native', 'kvmem'):
        target = root / name
        manifest = read(target / 'manifest.json')
        seed = validate_seed(target)
        validate_inductor_seed(target)
        require((seed['seed_sha256'] if seed else None) == plan.get('flashinfer_autotune_seed_sha256'),
                'FlashInfer seed differs from frozen plan')
        for key in ('flashinfer_seed_manifest_sha256', 'inductor_seed_manifest_sha256'):
            require(manifest.get(key) == plan.get(key, {}).get(name), 'Seed manifest differs from frozen plan: ' + key)
            require(plan.get(key, {}).get('native') == plan.get(key, {}).get('kvmem'),
                    'Native and KVMem frozen seed chains differ: ' + key)


def audit_seed_pair(native, kvmem):
    """Compare actual choices, not merely the requested seed configuration."""
    for kind in ('flashinfer', 'inductor'):
        pair = [native[kind], kvmem[kind]]
        if all(seed is None for seed in pair):
            continue
        require(all(seed is not None for seed in pair), 'Only one arm used controlled ' + kind)
        if kind == 'flashinfer':
            require(all(seed.get('loaded_files_semantically_identical') is True and
                        seed.get('actual_configs_sha256') is not None for seed in pair),
                    'FlashInfer actual configuration audit is incomplete')
            require(pair[0]['seed_sha256'] == pair[1]['seed_sha256'] and
                    pair[0]['actual_configs_sha256'] == pair[1]['actual_configs_sha256'],
                    'Actual native/KVMem FlashInfer choices differ')
            continue
        require(pair[0]['seed_manifest_sha256'] == pair[1]['seed_manifest_sha256'],
                'Native/KVMem Inductor seed chains differ')
        require(all(seed.get('launch_verified') is True and seed.get('common_configurations_unchanged') is True
                    and seed.get('formal_dispatch_observer') is False for seed in pair),
                'Inductor needs successful startup-only launch audits')
        require(set(pair[0]['modules']) == set(pair[1]['modules']), 'Inductor module roles differ')
        for role in pair[0]['modules']:
            entries = [{item['key']: item for item in seed['modules'][role]['common_kernels']} for seed in pair]
            common = set(entries[0]) & set(entries[1])
            require(bool(common), 'No executed common Inductor kernels: ' + role)
            for key in common:
                require(all(entries[0][key][field] == entries[1][key][field] for field in
                    ('source_sha256', 'function_source_sha256', 'configs_hash', 'selected_config')),
                    'Actual native/KVMem Inductor choice differs: ' + role + '/' + key)


def native_client(target):
    import numpy as np
    from benchmark_native_prefill import http
    case = read(target / 'case.json')
    native = target.parents[1]
    port = read(native / 'native-launch.json')['port']
    source = Path(case['input_path'])
    require(sha(source) == case['row']['served_sha256'], 'Frozen input changed')
    tokens = np.fromfile(source, dtype='<i4').tolist()
    require(len(tokens) == case['row']['served_input_tokens'], 'Prompt was truncated')
    body = dict(case['sampling'], model='kvmem-graph-v2', prompt=tokens,
        request_id=case['request_id'], return_token_ids=True, stream=True,
        stream_options={'include_usage': True})
    save(target / 'request.json', dict(request_id=case['request_id'], sampling=case['sampling'],
        input_tokens=len(tokens), input_sha256=sha(source), instrumented=case['instrumented']))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    response = dict(output_ids=[], text='', chunks=[], response_id=None, usage=None, finish_reason=None)
    done = False
    with http().open(request, timeout=1200) as stream:
        for line in stream:
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done = True
                break
            event = json.loads(payload)
            require(not event.get('error'), 'Native API error: ' + str(event.get('error')))
            response['response_id'] = event.get('id') or response['response_id']
            response['usage'] = event.get('usage') or response['usage']
            for choice in event.get('choices', []):
                ids = choice.get('token_ids') or []
                if ids:
                    response['output_ids'].extend(ids)
                    response['chunks'].append([time.monotonic()-start, len(ids)])
                response['text'] += choice.get('text') or ''
                response['finish_reason'] = choice.get('finish_reason') or response['finish_reason']
    require(done and response['usage'] and response['output_ids'], 'Incomplete native stream')
    require(response['usage']['prompt_tokens'] == len(tokens) and
            response['usage']['completion_tokens'] == len(response['output_ids']), 'Native token accounting failed')
    response.update(seconds=time.monotonic()-start, ttft_seconds=response['chunks'][0][0])
    save(target / 'response.json', response)
    save(target / 'result.json', dict(status='completed', family=case['family'],
        instrumented=case['instrumented'], input_tokens=len(tokens), input_sha256=sha(source),
        completion_tokens=len(response['output_ids']), ttft_seconds=response['ttft_seconds'],
        input_tokens_per_second=len(tokens)/response['ttft_seconds'], seconds=response['seconds'],
        finish_reason=response['finish_reason'], output_sha256=digest(response['output_ids'])))


def run_native(root):
    assert_frozen(root, 'native')
    target = root / 'native'
    runtime = target / 'runtime'
    sys.path.insert(0, str(runtime / 'scripts'))
    import benchmark_block_graph as lifecycle
    from benchmark_native_prefill import (native_argv, http, idle, metrics,
        install_native_seeds, audit_native_seeds, native_environment_receipt)
    from native_agent_launch import _environment
    plan = read(target / 'suite.json')
    for item in plan['cases']:
        require(sha(target / 'requests' / item['id'] / 'case.json') == item['case_sha256'], 'Case changed')
    probe = target / 'probe'
    probe.mkdir(exist_ok=False)
    argv = native_argv(read(target / 'reference-launch.json'), 18854, prefill_chunk=2048)
    if (root/'native-shape-plan.json').exists():
        shape_plan = read(root/'native-shape-plan.json')
        require(shape_plan['max_chunk'] == plan['native_prefill_chunk'],'Native cap differs from frozen geometry')
        argv = apply_observed_native_cap(argv,shape_plan)
    env, overrides = _environment(os.environ, allow_long=True)
    env.update(PYTHONPATH=str(runtime / 'scripts'), PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='4',
        FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1',
        NATIVE_PREFILL_PROBE_PLAN=str(target / 'suite.json'), NATIVE_PREFILL_PROBE_OUTPUT=str(probe))
    seed_receipts = install_native_seeds(target, env)
    worker_cls = ('prefill_inductor_worker.SeededNativeProbeWorker' if seed_receipts['inductor'] is not None
                  else 'native_prefill_shape_probe.NativeProbeWorker')
    argv.extend(('--scheduler-cls', 'native_prefill_shape_probe.NativeShapeScheduler', '--worker-cls', worker_cls))
    save(target / 'native-launch.json', dict(argv=argv, port=18854, environment_overrides=overrides,
        environment=native_environment_receipt(env), seed_installations=seed_receipts,
        startup_inductor_audit=seed_receipts['inductor'] is not None, formal_inductor_observer=False,
        connector_absent=True, prefix_caching=False, instrumented_native=True,
        model_kernels_unmodified=True, diagnostic_scheduler_only_caps_chunk_boundaries=True,
        timing_disabled_keeps_dispatch_wrappers=True))
    server = worker = None
    with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            require(not lifecycle.gpu_processes(), 'Foreign GPU owner')
            start = time.monotonic()
            with (target / 'server.log').open('x') as log:
                server = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            save(target / 'server-process.json', dict(pid=server.pid, command=argv))
            save(target / 'status.json', dict(status='loading', at=time.time()))
            deadline = time.monotonic()+1200
            while time.monotonic() < deadline:
                require(server.poll() is None, 'Native server exited during loading')
                try:
                    with http().open('http://127.0.0.1:18854/health', timeout=1) as result:
                        if result.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Native startup exceeded 20 minutes')
            audit_native_seeds(target, 'ready')
            cold = time.monotonic()-start
            results = []
            for item in plan['cases']:
                directory = target / 'requests' / item['id']
                before = metrics(18854)
                require(idle(before), 'Unexpected concurrent native request')
                save(directory / 'initial-metrics.json', dict(metrics=before, at=time.time()))
                save(target / 'status.json', dict(status='testing', case_id=item['id'],
                    completed_cases=len(results), cold_prepare_seconds=cold, at=time.time()))
                command = [sys.executable, '-B', str(runtime / 'scripts/benchmark_active_prefill.py'),
                           'native-client', '--root', str(directory)]
                with (directory / 'client.log').open('x') as log:
                    worker = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                save(directory / 'client-process.json', dict(pid=worker.pid, command=command))
                deadline = time.monotonic()+1200
                with (directory / 'resources.jsonl').open('x') as stream:
                    while worker.poll() is None:
                        require(server.poll() is None and time.monotonic() < deadline, 'Native request failed/timed out')
                        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                                      '--format=csv,noheader,nounits'], text=True).strip()
                        stream.write(json.dumps(dict(at=time.time(), gpu=gpu,
                            **owned_process_memory(server.pid),
                            process_tree_memory_scope=PROCESS_MEMORY_SCOPE))+'\n')
                        stream.flush()
                        time.sleep(1)
                require(worker.returncode == 0, 'Native client failed: '+str(directory / 'client.log'))
                after = metrics(18854)
                require(idle(after), 'Native request did not retire')
                save(directory / 'final-metrics.json', dict(metrics=after, at=time.time()))
                results.append(dict(id=item['id'], warmup=item['warmup'], result=read(directory / 'result.json')))
            audit_native(target)
            audit_native_seeds(target, 'final')
            save(target / 'status.json', dict(status='completed', cold_prepare_seconds=cold,
                 protocol_passed=True, results=results, at=time.time()))
        except BaseException as error:
            save(target / 'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining = lifecycle.stop_owned_groups((worker, server))
            save(target / 'cleanup.json', dict(remaining_owned_groups=remaining,
                                              gpu_processes=sorted(lifecycle.gpu_processes())))
            require(not remaining, 'Owned native process survived cleanup')


def audit_native(target):
    from native_prefill_shape_probe import case_for_request
    definitions = read(target / 'suite.json')['cases']
    observed = defaultdict(list)
    for row in rows(target / 'probe/model_step_latency.jsonl'):
        require(len(row['batch_geometry']) == 1, 'Unexpected concurrent native batch')
        shape = row['batch_geometry'][0]
        item = case_for_request(shape['request_id'], definitions)
        require(item['instrumented'], 'Timing leaked into disabled control')
        if shape['prefill']:
            observed[item['id']].append((shape, row))
    warmup_incomplete = []
    for item in definitions:
        if not item['instrumented']:
            continue
        found = observed[item['id']]
        if item['warmup'] and sum(shape['query_tokens'] for shape, _ in found) != item['input_tokens']:
            warmup_incomplete.append(item['id'])
            continue
        require(sum(shape['query_tokens'] for shape, _ in found) == item['input_tokens'],
                'Missing native prefill timing rows: '+item['id'])
        cumulative = 0
        for shape, _ in found:
            cumulative += shape['query_tokens']
            require(shape['active_seq_len'] == cumulative and shape['logical_seq_len'] == cumulative,
                    'Native context geometry is not contiguous')
            require(shape['prefill_terminal'] is (cumulative == item['input_tokens']),
                    'Native timing terminal status differs from complete-prompt boundary')
        if item['chunks'] is not None:
            require([shape['query_tokens'] for shape, _ in found] == item['chunks'],
                    'Native scheduler did not realize requested chunk geometry: '+item['id'])
    save(target / 'probe/audit.json', dict(passed=True, full_original_inputs=True,
         exact_chunk_schedule=True, requests_with_timing=len(observed),
         warmup_incomplete_timing=warmup_incomplete, formal_timing_complete=True))
    return observed


def run_kvmem(root):
    assert_frozen(root, 'kvmem')
    target = root / 'kvmem'
    command = [sys.executable, '-B', str(target / 'runtime/scripts/benchmark_matched_prefill.py'),
               'run', '--root', str(target)]
    with (target / 'controller.log').open('x') as log:
        result = subprocess.run(command, cwd=target, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'KVMem controller failed; inspect controller.log')


def analyze(root):
    import statistics
    assert_frozen(root,'native')
    native = root / 'native'
    shape_plan = read(root/'native-shape-plan.json') if (root/'native-shape-plan.json').exists() else None
    steady_patterns = shape_plan['steady_patterns'] if shape_plan is not None else None
    sys.path.insert(0, str(native / 'runtime/scripts'))
    observed = audit_native(native)
    # Old artifacts have no seed files and retain their original accounting.
    native_seed_audit = None
    if 'seed_helper_sha256' in read(root / 'plan.json'):
        from benchmark_native_prefill import audit_native_seeds
        native_seed_audit = audit_native_seeds(native, 'analysis')
    definitions = read(native / 'suite.json')['cases']
    bank = defaultdict(list)
    full32 = {True: [], False: []}
    native_continuous = []
    output_samples = []
    for item in definitions:
        if item['warmup']:
            continue
        if item['family'] in ('full32k','logical2k-control'):
            if shape_plan is not None:
                output_samples.append(dict(label='native-'+item['family'],id=item['id'],
                    output_ids=read(native/'requests'/item['id']/'response.json')['output_ids']))
        if item['family'] == 'logical2k-control':
            continue
        if item['family'] == 'full32k':
            full32[item['instrumented']].append(read(native / 'requests' / item['id'] / 'result.json'))
        elif item['instrumented']:
            if item['family'] == 'steady' or shape_plan is not None:
                phases = continuous_steady_phases([row for _, row in observed[item['id']]],steady_patterns)
                if shape_plan is None:
                    require(phases['matched_phases'] == 1, 'Native steady phase not found: '+item['id'])
                if phases['matched_phases']:
                    native_continuous.append(dict(id=item['id'], **phases))
            for shape, row in observed[item['id']]:
                bank[(shape['query_tokens'],shape['active_seq_len'],shape['prefill_terminal'])].append(row)
    baseline = []
    for target in read(root / 'reference-geometry.json')['shape_counts']:
        key = (target['query_tokens'], target['active_seq_len'],target['prefill_terminal'])
        require(bank.get(key), 'Missing native shape samples: '+str(key))
        baseline.append(dict(**target, samples=len(bank[key]),
            native_model_ms=statistics.median(row['model_ms'] for row in bank[key]),
            native_step_ms=statistics.median(row['step_ms'] for row in bank[key]),
            native_host_submit_ms=statistics.median(row['host_submit_ms'] for row in bank[key])))
    result = dict(scope='Whole-model matched-shape diagnostic, not end-to-end 256K native throughput',
        native_shape_mode=shape_plan['mode'] if shape_plan is not None else 'original_2k_families',
        continuous_steady_prefill_native=native_continuous,
        native_process_memory={item['id']:sampled_process_memory(
            rows(native/'requests'/item['id']/'resources.jsonl'))
            for item in definitions if not item['warmup']},
        shapes=baseline,
        weighted_native_model_seconds=sum(s['count']*s['native_model_ms'] for s in baseline)/1000,
        weighted_native_step_seconds=sum(s['count']*s['native_step_ms'] for s in baseline)/1000,
        weighted_native_host_submit_seconds=sum(s['count']*s['native_host_submit_ms'] for s in baseline)/1000,
        full32k_native={('timed' if key else 'timing_disabled'):dict(samples=value,
            pooled_prefill_tokens_per_second=sum(x['input_tokens'] for x in value)/sum(x['ttft_seconds'] for x in value),
            mean_ttft_seconds=statistics.mean(x['ttft_seconds'] for x in value)) for key,value in full32.items()},
        caveats=['CUDA intervals include launch gaps and MTP; they are not a sum of pure kernel times.',
                 'Same shapes do not make native request lifecycle/selection/history costs equivalent.',
                 'Native shape requests allow 8 outputs so MTP runs; generation trajectories are not a quality comparison.',
                 'Timing-disabled native retains diagnostic dispatch wrappers.'])
    if shape_plan is not None:
        references = read(root/'output-control-reference.json')
        samples = output_samples+references['samples']
        result['physical_shape_output_diagnostic'] = dict(
            scope='Independent full32K token-output control; excluded from throughput, weighted shapes and steady samples.',
            unavailable=references['unavailable'],samples=[dict(label=s['label'],id=s['id'],
                tokens=len(s['output_ids']),output_sha256=digest(s['output_ids'])) for s in samples],
            pairs=[dict(left=left['label']+'/'+left['id'],right=right['label']+'/'+right['id'],
                        **compare_output_tokens(left['output_ids'],right['output_ids']))
                   for i,left in enumerate(samples) for right in samples[i+1:]],
            caveat='Output differences establish sensitivity to execution shape or request trajectory; '
                   'they do not alone identify an attention-statistics or selection implementation bug.')
    if (root / 'kvmem/status.json').exists() and read(root / 'kvmem/status.json')['status'] == 'completed':
        import benchmark_prefill_publication_pipeline as pipeline
        arm = root / 'kvmem'
        groups = pipeline.prefill_groups(rows(arm / 'run/server/capture/model_step_latency.jsonl'))
        cases = read(arm / 'suite.json')['cases']
        require(len(groups) == len(cases), 'KVMem prefill grouping differs')
        result['kvmem'] = []
        for group, item in zip(groups, cases):
            if item['warmup']:
                continue
            directory = arm / 'requests' / item['id']
            sample = read(directory / 'result.json')
            evidence = read(directory / 'evidence.json')
            shapes = reconstruct_geometry(group, evidence['refresh'], sample['input_tokens'])
            require(all('batch_geometry' in step for step in group), 'KVMem exact CPU geometry missing')
            entry = dict(id=item['id'], **sample, timing=pipeline.timing(group),
                         **sampled_process_memory(rows(directory/'resources.jsonl')))
            entry['continuous_steady_prefill'] = continuous_steady_phases(group,steady_patterns,evidence['refresh'])
            if shape_plan is not None and sample['input_tokens'] == 32768:
                require([s['query_tokens'] for s in shapes] == shape_plan['full32_chunks'],
                        'KVMem full32K physical schedule differs from its native control')
            if sample['input_tokens'] == 262045:
                entry['native_shape_estimate_model_seconds'] = sum(
                    statistics.median(r['model_ms'] for r in bank[(s['query_tokens'],s['active_seq_len'],s['prefill_terminal'])]) for s in shapes)/1000
            result['kvmem'].append(entry)
    if native_seed_audit is not None:
        result['autotune_control'] = {'native': native_seed_audit, 'kvmem': None, 'pair_verified': False}
        if 'kvmem' in result:
            from prefill_autotune_control import audit_seed as audit_flashinfer
            from prefill_inductor_control import audit_seed as audit_inductor
            other = dict(flashinfer=audit_flashinfer(root / 'kvmem'), inductor=audit_inductor(root / 'kvmem'))
            audit_seed_pair(native_seed_audit, other)
            result['autotune_control'].update(kvmem=other,
                pair_verified=any(value is not None for value in native_seed_audit.values()))
    save(root / 'comparison.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('shapes','full32k_native')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run-native', 'run-kvmem', 'native-client', 'analyze'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--reference', type=Path,
                        help='Qualified reference arm for prepare; relative paths are resolved against --repo')
    parser.add_argument('--flashinfer-autotune-seed', type=Path)
    parser.add_argument('--flashinfer-autotune-hash', action='append', default=[])
    parser.add_argument('--inductor-seed-sources', type=json.loads)
    parser.add_argument('--inductor-seed-hashes', type=json.loads)
    args = parser.parse_args()
    if args.command != 'prepare' and (args.flashinfer_autotune_seed is not None or args.flashinfer_autotune_hash
            or args.inductor_seed_sources is not None or args.inductor_seed_hashes is not None):
        parser.error('Seed source/configuration flags belong to prepare; runs use the frozen plan')
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, args.repo.resolve(), args.reference,
            flashinfer_autotune_seed=args.flashinfer_autotune_seed,
            flashinfer_autotune_hashes=args.flashinfer_autotune_hash,
            inductor_seed_sources=args.inductor_seed_sources, inductor_seed_hashes=args.inductor_seed_hashes)
    elif args.command == 'run-native':
        run_native(root)
    elif args.command == 'run-kvmem':
        run_kvmem(root)
    elif args.command == 'native-client':
        native_client(root)
    else:
        analyze(root)


if __name__ == '__main__':
    main()
