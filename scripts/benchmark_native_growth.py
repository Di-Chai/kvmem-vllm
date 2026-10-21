"""Measure one native request growing from an empty prefix through 32K to 64K.

The existing native harness supplies process ownership, fixed compiler choices,
nonblocking CUDA events and exact chunk audits. No model or KVMem runtime edits.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import shutil
import statistics
import sys

if Path(__file__).name == 'growth-driver.py':
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'native/runtime/scripts'))

import benchmark_active_prefill as b


CHUNKS = [2048, 14336, 14336, 2048, 14336, 2048, 14336, 2048]


def phase(rows, begin, end):
    """An actual linked interval, including its interior gaps, never subtraction
    of separately measured requests. The final 64K step includes native sampling.
    """
    chosen = []
    for row in rows:
        b.require(len(row['batch_geometry']) == 1, 'Expected one owner')
        shape = row['batch_geometry'][0]
        start = shape['active_seq_len'] - shape['query_tokens']
        if start < end and shape['active_seq_len'] > begin:
            b.require(start >= begin and shape['active_seq_len'] <= end,
                      'A physical chunk crosses the requested phase boundary')
            chosen.append(row)
    b.require(bool(chosen), 'No phase rows')
    cursor, owner, previous = begin, None, None
    model = gaps = 0.0
    for row in chosen:
        s = row['batch_geometry'][0]
        b.require(s['prefill'] and row['decode_requests'] == 0 and s.get('offset', 0) == 0,
                  'Phase is not ordinary native prefill')
        b.require(s['logical_seq_len'] == s['active_seq_len'] == cursor + s['query_tokens'],
                  'Native phase lost context continuity')
        b.require(owner is None or owner == s['request_id'], 'Owner changed within phase')
        owner = s['request_id']
        cursor += s['query_tokens']
        value = row['model_ms']
        b.require(math.isfinite(value) and value >= 0, 'Invalid model time')
        model += value
        if previous is not None:
            b.require(row['sequence'] == previous['sequence'] + 1 and
                      row['call_index'] == previous['call_index'] + 1 and
                      row['previous_sequence'] == previous['sequence'] and
                      row['previous_call_index'] == previous['call_index'],
                      'Unlinked or missing timing interval')
            gap = row['previous_end_to_model_ms']
            b.require(gap is not None and math.isfinite(gap) and gap >= 0, 'Invalid interior gap')
            gaps += gap
        previous = row
    b.require(cursor == end, 'Phase is incomplete')
    seconds = (model + gaps) / 1000
    b.require(seconds > 0, 'Empty timing interval')
    return dict(begin_tokens=begin, end_tokens=end, new_tokens=end-begin,
                model_seconds=model/1000, interior_gap_seconds=gaps/1000,
                seconds=seconds, tokens_per_second=(end-begin)/seconds,
                includes_terminal_sampling=bool(chosen[-1]['batch_geometry'][0]['prefill_terminal']),
                shapes=[r['batch_geometry'][0] for r in chosen])


def prepare(root, repo, reference, seeds):
    seed_maps = b.read(seeds)
    source = b.reference_inputs(repo, reference)
    fi_seed = reference / 'flashinfer_seed/seed.json'
    b.prepare(root, repo, reference, flashinfer_autotune_seed=fi_seed,
              flashinfer_autotune_hashes=['419d38c7e8f695e7103d7ef1d584956000dea1b628af6e682a5d5fc8b7307e59'],
              inductor_seed_sources=seed_maps['sources'], inductor_seed_hashes=seed_maps['hashes'])
    # Specialize the new, never-run artifact before sealing its final plan.
    # The inherited KVMem arm is frozen provenance only and will not execute.
    native = root / 'native'
    shutil.copy2(native / 'suite.json', native / 'unused-shape-suite.json')
    original = Path(source['cases256']['input_path'])
    b.require(b.sha(original) == source['cases256']['row']['served_sha256'], 'Source input changed')
    prompt = root / 'inputs/growth64.i32'
    prompt.write_bytes(original.read_bytes()[:65536*4])
    b.require(prompt.stat().st_size == 65536*4, 'Source is shorter than 64K')
    cases = []

    def add(ident, warmup=False, instrumented=True, flush=False):
        directory = native / 'requests' / ident
        directory.mkdir()
        case = deepcopy(source['cases256'])
        path, chunks = prompt, CHUNKS
        if flush:
            path = root / 'inputs/growth-flush.i32'
            path.write_bytes(prompt.read_bytes()[:4])
            chunks = [1]
        case.update(id=ident, request_id='native-growth-' + ident, family='growth64',
                    input_path=str(path), chunks=chunks, instrumented=instrumented, warmup=warmup,
                    sampling={**case['sampling'], 'max_tokens': 1 if flush else 8},
                    row=dict(served_input_tokens=sum(chunks), served_sha256=b.sha(path)))
        b.save(directory / 'case.json', case)
        cases.append(dict(id=ident, request_id=case['request_id'], family='growth64',
                          input_tokens=sum(chunks), chunks=chunks, warmup=warmup,
                          instrumented=instrumented, case_sha256=b.sha(directory/'case.json')))

    add('growth-warmup', warmup=True)
    for repeat in range(1, 4):
        for timed in ((True, False) if repeat % 2 else (False, True)):
            add(f'growth-{repeat}-' + ('timed' if timed else 'untimed'), instrumented=timed)
    add('growth-flush', warmup=True, instrumented=False, flush=True)
    b.save(native/'suite.json', dict(cases=cases, native_prefill_chunk=max(CHUNKS)))
    manifest = b.read(native/'manifest.json')
    manifest.update(suite_sha256=b.sha(native/'suite.json'),
                    scope='Native single request 0→32K→48K→64K; no context reset, no prefix caching.')
    b.save(native/'manifest.json', manifest)
    driver = root/'growth-driver.py'
    shutil.copy2(__file__, driver)
    plan = b.read(root/'plan.json')
    plan.update(native_suite_sha256=b.sha(native/'suite.json'),
                growth_driver_sha256=b.sha(driver), experiment_kind='native_32k_to_64k',
                native_shape_rounds=0, native_full32k_timed=0, native_full32k_untimed=0,
                native_logical2k_controls=0, kvmem_full32k=0, kvmem_full256k=0,
                native_growth_timed=3, native_growth_untimed=3,
                input_sha256=b.sha(prompt), source_input_sha256=b.sha(original),
                chunks=CHUNKS, boundary_tokens=[0,32768,49152,65536],
                phase_scope='First model event after boundary through final end event, including interior gaps. '
                            'No earlier prefix prefill or boundary-before-phase gap; final step includes native '
                            'sampling/MTP. Whole-request TTFT is reported separately. '
                            'KVMem provenance arm is not executed.')
    b.save(root/'plan.json', plan)
    b.assert_frozen(root, 'native')


def analyze(root):
    b.assert_frozen(root, 'native')
    b.require(b.sha(root/'growth-driver.py') == b.read(root/'plan.json')['growth_driver_sha256'],
              'Growth analyzer changed')
    native = root/'native'
    b.require(b.read(native/'status.json')['status'] == 'completed', 'Native run did not complete')
    observed = b.audit_native(native)
    summaries = []
    grouped = defaultdict(list)
    for item in b.read(native/'suite.json')['cases']:
        if item['warmup']:
            continue
        result = b.read(native/'requests'/item['id']/'result.json')
        grouped['timed' if item['instrumented'] else 'untimed'].append(result)
        if item['instrumented']:
            rows = [row for _, row in observed[item['id']]]
            summaries.append(dict(id=item['id'], ttft_seconds=result['ttft_seconds'],
                phases={label:phase(rows, begin, end) for label,begin,end in
                        [('prefix32',0,32768),('append32_to48',32768,49152),
                         ('append48_to64',49152,65536),('append32_to64',32768,65536),
                         ('full64',0,65536)]}))
    aggregate = {}
    for label in summaries[0]['phases']:
        entries = [row['phases'][label] for row in summaries]
        times = [entry['seconds'] for entry in entries]
        aggregate[label] = dict(repeats=len(entries), new_tokens=entries[0]['new_tokens'],
            mean_seconds=statistics.mean(times), min_seconds=min(times), max_seconds=max(times),
            pooled_tokens_per_second=sum(e['new_tokens'] for e in entries)/sum(times),
            mean_model_seconds=statistics.mean(e['model_seconds'] for e in entries),
            mean_interior_gap_seconds=statistics.mean(e['interior_gap_seconds'] for e in entries))
    ttft = {key:dict(mean_seconds=statistics.mean(r['ttft_seconds'] for r in values),
                     pooled_tokens_per_second=sum(r['input_tokens'] for r in values)/
                                               sum(r['ttft_seconds'] for r in values),
                     values_seconds=[r['ttft_seconds'] for r in values])
            for key,values in grouped.items()}
    from benchmark_native_prefill import audit_native_seeds
    seeds = audit_native_seeds(native, 'analysis')
    b.save(root/'growth-results.json', dict(phase_aggregates=aggregate, full64_ttft=ttft,
        requests=summaries, cold_prepare_seconds=b.read(native/'status.json')['cold_prepare_seconds'],
        instrumentation_ttft_change_percent=100*(ttft['timed']['mean_seconds']/ttft['untimed']['mean_seconds']-1),
        seed_audits=seeds, scope=b.read(root/'plan.json')['phase_scope'],
        cleanup=b.read(native/'cleanup.json')))
    print(aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','analyze'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--seeds', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, Path(__file__).resolve().parents[1], args.reference.resolve(), args.seeds.resolve())
    else:
        b.require(b.sha(root/'growth-driver.py') == b.read(root/'plan.json')['growth_driver_sha256'],
                  'Growth controller changed')
        if args.command == 'run':
            b.run_native(root)
        analyze(root)


if __name__ == '__main__':
    main()
