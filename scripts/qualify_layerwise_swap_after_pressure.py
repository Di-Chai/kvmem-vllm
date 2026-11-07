"""One fresh delayed-arrival qualification using an unchanged frozen runtime.

Only client arrival timing changes. The second cold prefix waits until the
first has at least 98,304 CUDA-confirmed public-prefill tokens and an observed
heterogeneous per-layer pressure selection. No model launch occurs at prepare.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def tree(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def records(path, *, live=False):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if not live:
                raise
    return result


def heterogeneous(event):
    layers = event.get('layers', ())
    return (event.get('event') == 'pressure_selected' and
            event.get('actual_method') == 'accumulated_attention_per_layer' and
            event.get('fallback_reason') is None and
            [row['layer'] for row in layers] == list(range(16)) and
            len({tuple(row['selected_starts']) for row in layers}) > 1)


def prepare(root, reference, threshold=98304):
    if root.exists():
        raise ValueError('Preserve prior attempts; choose a fresh directory')
    if type(threshold) is not int or threshold < 98304:
        raise ValueError('Arrival threshold must cover prior pressure selections')
    if not read(reference/'serial-comparison.json')['passed']:
        raise ValueError('Original serial full-input qualification must have passed')
    original = reference/'incremental_tight'
    fixture, profile = read(original/'fixture.json'), read(original/'profile.json')
    if (fixture['cases'] != 2 or len(fixture['rows']) != 2 or
            [profile['kvmem'][k] for k in ('select_budget','gen_budget')] != [36864,16384] or
            not profile['verify_request_swap'] or not profile['kvmem']['request_swap_layerwise']):
        raise ValueError('Reference does not match the qualified two-prefix layerwise protocol')
    proof = read(original/'runtime-proof.json')
    if tree(original/'runtime-v1') != proof:
        raise ValueError('Reference frozen runtime changed')
    root.mkdir(parents=True)
    shutil.copytree(original/'runtime-v1',root/'runtime-v1',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    fixture = deepcopy(fixture)
    for row, binding in zip(fixture['rows'],fixture['bindings']):
        if row['id'] != binding['id']:
            raise ValueError('Request binding order differs from frozen input order')
        for field, record, digest, directory in (
                ('source',row,row['source_cache_sha256'],'inputs'),
                ('request',binding,binding['sha256'],'bindings')):
            origin = Path(record[field])
            if sha(origin) != digest:
                raise ValueError('Original input or sampling binding changed')
            target = root/directory/origin.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(origin,target)
            record[field] = str(target)
    for name, value in (('fixture.json',fixture),('profile.json',profile),
                        ('runtime-proof.json',proof),('status.json',{'status':'prepared'})):
        save(root/name,value)
    files = {str(path.relative_to(root)):sha(path) for path in root.rglob('*')
             if path.is_file() and 'runtime-v1' not in path.parts and path.name != 'status.json'}
    reference_files = {str(reference/'serial-comparison.json'):sha(reference/'serial-comparison.json')}
    baseline = reference/'static_serial'
    if read(baseline/'status.json')['status'] != 'completed':
        raise ValueError('Read-only serial reference is incomplete')
    for case in fixture['case_ids']:
        for name in ('request.json','response.json','result.json','events.json'):
            path = baseline/'run/answers/cases'/case/name
            reference_files[str(path)] = sha(path)
    shutil.copy2(__file__,root/'launch.py')
    files['launch.py'] = sha(root/'launch.py')
    save(root/'plan.json',dict(reference=str(reference),baseline=str(baseline),
        reference_files=reference_files,files=files,threshold=threshold,workers=2,
        delayed_case=fixture['rows'][1]['id'],first_case=fixture['rows'][0]['id'],
        source_runtime=str(original/'runtime-v1'),source_runtime_sha256=sha(original/'runtime-proof.json'),
        scope='Fresh answers; unchanged r2 runtime, inputs, budgets and sampling; only second arrival delayed. '
              'r2 static answers are a read-only reference, not a newly generated baseline.',
        automatic_retry=False,external_grading=False,imported_answers=0))
    save(root/'prepared-proof.json',{'plan_sha256':sha(root/'plan.json')})


def verify(root):
    if sha(root/'plan.json') != read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Frozen arrival plan changed')
    plan = read(root/'plan.json')
    for path,digest in plan['files'].items():
        if sha(root/path) != digest:
            raise ValueError('Frozen follow-up control changed: '+path)
    for path,digest in plan['reference_files'].items():
        if sha(path) != digest:
            raise ValueError('Read-only reference changed: '+path)
    if tree(root/'runtime-v1') != read(root/'runtime-proof.json'):
        raise ValueError('Frozen follow-up runtime changed')
    return plan


def arrival_evidence(capture, threshold):
    try:
        work = read(capture/'public_work.json')
    except (FileNotFoundError,json.JSONDecodeError):
        return None
    if work.get('completed',{}).get('public_prefill_tokens',0) < threshold:
        return None
    pressure = [row for row in records(capture/'layerwise_query.jsonl',live=True)
                if heterogeneous(row)]
    if not pressure:
        return None
    return dict(public_work=work,pressure=pressure[-1],at=time.time(),
                scope='Only the first request was submitted before this release')


def run(root):
    plan = verify(root)
    runtime = root/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('delayed_memory',runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    def frozen(path):
        if path != root:
            raise ValueError('Unexpected follow-up run root')
        verify(root)
        return read(root/'fixture.json'),read(root/'profile.json')
    original_source = memory.source
    def delayed_source(row):
        if row['id'] == plan['delayed_case']:
            deadline = time.monotonic()+1200
            capture = root/'run/server/capture'
            while True:
                evidence = arrival_evidence(capture,plan['threshold'])
                if evidence is not None:
                    save(root/'arrival-release.json',evidence)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('First prefix never reached confirmed post-pressure arrival boundary')
                time.sleep(0.2)
        elif row['id'] != plan['first_case']:
            raise ValueError('Unexpected additional input')
        return original_source(row)
    memory.verify,memory.source = frozen,delayed_source
    memory.run(root,workers=2)


def analyze(root):
    plan = verify(root)
    if read(root/'status.json')['status'] != 'completed':
        raise ValueError('Both fresh full-input answers must complete')
    capture = root/'run/server/capture'
    restores = records(capture/'request_swap_verified.jsonl')
    events = records(capture/'layerwise_query.jsonl')
    covered = []
    for restore in restores:
        previous = [event for event in events if heterogeneous(event) and
                    event['request_id'] == restore['request_id'] and
                    event['accepted_history'] <= restore['history_end'] and
                    event['monotonic_at'] <= restore['at']]
        if restore['history_end'] > 36864+16384 and previous:
            event = max(previous,key=lambda item:item['accepted_history'])
            covered.append(dict(request_id=restore['request_id'],operation_id=restore['operation_id'],
                history_end=restore['history_end'],verified_bytes=restore['bytes'],
                pressure_history=event['accepted_history'],layers=len(event['layers']),
                distinct_layer_selections=len({tuple(layer['selected_starts']) for layer in event['layers']})))
    worker = read(capture/'archive_resources.json')['request_checkpoints']
    byte_checks = (bool(restores) and len({tuple(row['identity']) for row in restores}) == len(restores) and
        worker['verification_enabled'] and worker['verification_failures'] == 0 and
        worker['restores'] == worker['verification_successes'] == len(restores) and
        not worker['requests'] and worker.get('layerwise_deep_checks',0) > 0)
    cleanup = read(root/'run/cleanup.json')
    clean = not cleanup['gpu_processes'] and not cleanup['remaining_owned_groups']
    sys.path[:0] = [str(root/'runtime-v1/scripts'),str(root/'runtime-v1/src')]
    from audit_layerwise_gilopt import compare
    comparison = compare(Path(plan['baseline']),root,read(root/'fixture.json')['case_ids'])
    report = dict(covered_swaps=covered,restores=len(restores),byte_checks_passed=bool(byte_checks),
        released=clean,post_pressure_map_coverage_passed=bool(covered),
        swap_qualification_passed=bool(covered) and bool(byte_checks) and clean,
        full_output_and_selection_equivalence=comparison['passed'],comparison=comparison,
        passed=bool(covered) and bool(byte_checks) and clean and comparison['passed'],
        scope='Post-pressure heterogeneous layer-map swap coverage and byte recovery. '
              'Read-only prior serial baseline; arrival changes can change batch shapes. '
              'Do not equate byte correctness with output equivalence or pure-prefill performance.')
    save(root/'qualification.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','verify','analyze'))
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--reference',type=Path,
                        default=REPO/'artifacts/layerwise-incremental-20261106-r2')
    parser.add_argument('--threshold',type=int,default=98304)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root,args.reference.resolve(),args.threshold)
    elif args.action == 'run':
        run(root)
    elif args.action == 'analyze':
        print(json.dumps(analyze(root)))
    else:
        verify(root)


if __name__ == '__main__':
    main()
