"""Run frozen NIAH256K/LME/MAB with guidance off or on, without changing gold."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace as NS

from tokenizers import Tokenizer

import benchmark_niah as niah
import guided_benchmark_protocol as protocol
import qualify_block_graph as q
import qualify_graph_niah as old
from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask

ROOT = Path(__file__).resolve().parents[1]
NIAH_REFERENCE = Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927')
MEMORY_REFERENCE = Path('/home/chaidi/qw3/benchmark/attention_multilayer_prefix_reuse_20261012_r2')
SUITES = ('niah', 'longmemeval_s', 'memoryagentbench')


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.config.read_text())
    configurations = {}
    for suite in SUITES:
        # Preserve the actual per-suite C profiles, including synchronous
        # frozen-P import for MAB and asynchronous maintenance for cold inputs.
        source = args.config.parent/f'{suite}-C.json'
        common = json.loads(source.read_text())
        import graph_v2_protocol
        graph_v2_protocol.validate_profile(common, 'C', suite)
        common['port'] = args.port
        common['kvmem'].update(guided_reselect='boundary',
            guided_query_backend='eager', guided_trigger_tokens=0)
        pair = {}
        for arm in ('off', 'guided'):
            pair[arm] = deepcopy(common)
            pair[arm]['kvmem']['guided_query_tokens'] = 512 if arm == 'guided' else 0
            from vllm_kvmem.serve import build_launch
            build_launch(pair[arm], ROOT, root/f'{suite}-{arm}/server')
            q.save(root/f'{suite}-{arm}.json', pair[arm])
        protocol.validate_pair(pair['off'], pair['guided'], suite)
        configurations[suite] = pair
        expected = json.loads(source.read_text())
        expected['port'] = args.port
        actual = deepcopy(pair['off'])
        for name in ('guided_reselect', 'guided_query_backend', 'guided_trigger_tokens'):
            actual['kvmem'].pop(name)
        protocol.require(actual == expected, 'The historical C baseline changed')
        q.save(root/f'{suite}-baseline.json', dict(source=str(source.resolve()),
            sha256=protocol.sha(source), permitted_changes=['port', 'guided options']))
    settings = NS(reference=args.niah_reference, config=root/'niah-guided.json',
        output=root/'dataset/niah', lengths=[262144], depths=[10, 30, 50, 70, 90],
        seeds=[1000, 2000, 3000], max_output=8192, workers=8,
        cpu_headroom_gib=16, resume=False)
    rows = niah.prepare(settings, niah.normalized_config(configurations['niah']['guided']))
    protocol.require(len(rows) == 60, 'Expected all 60 frozen 256K cases')
    original = (args.niah_reference/'run.py').read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    fixture = dict(rows=rows, workers=8, input_directory=str(settings.output/'inputs'),
        tokenizer_sha256=protocol.sha(Path(base['tokenizer'])/'tokenizer.json'),
        scorer_source=ast.get_source_segment(original, node),
        source_scorer_sha256=protocol.sha(args.niah_reference/'run.py'),
        source_cohort_sha256=protocol.sha(args.niah_reference/'cohort.json'),
        cpu_payload_credits_bytes=100000000000-16*1024**3,
        request_timeout_seconds=3600,
        sampling=dict(max_tokens=8192, temperature=1., top_p=.95, top_k=20, min_p=0,
                      presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096))
    q.save(root/'niah-fixture.json', fixture)
    reference = json.loads((args.memory_reference/'cohort.json').read_text())
    import benchmark_memory as memory
    for suite, count in (('longmemeval_s', 102), ('memoryagentbench', 1116)):
        rows = [r for r in reference['cases'] if r['suite'] == suite]
        protocol.require(len(rows) == count, 'Frozen cohort size differs')
        # Validate each original compressed input before scheduling model work.
        for row in rows:
            memory.source(row)
        q.save(root/f'{suite}-fixture.json', dict(suite=suite, rows=rows,
            reference=str(args.memory_reference.resolve()),
            source_cohort_sha256=protocol.sha(args.memory_reference/'cohort.json')))
    q.save(root/'manifest.json', dict(schema=protocol.SCHEMA, prepared_at=time.time(),
        arms=['off', 'guided'], cases=dict(niah=60, longmemeval_s=102, memoryagentbench=1116),
        scope='Existing frozen cohorts; MAB three-category subset, not the full official four-category dataset.',
        original_input_intact=True, gold_used_for_selection=False,
        raw_query_protocol='qw3 explicit raw-completion boundary guidance; agent lifecycle qualified separately',
        baseline='Previously qualified quality C: V2 multi Query alpha .2, cumulative attention pressure, MTP4',
        profile_hashes={p.name:protocol.sha(p) for p in root.glob('*.json') if p.name.endswith(('-off.json','-guided.json'))}))
    print(json.dumps(dict(prepared=str(root), cases_per_arm=1278)), flush=True)


def run_niah(args, profile, fixture):
    tree = ast.parse(fixture['scorer_source'])
    protocol.require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
                     'Expected frozen original scoring function')
    scope = {'re': re}
    exec(compile(tree, 'frozen-niah-scorer', 'exec'), scope)
    tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    queue = OfflineQueue([OfflineTask(r['id'], r, old.history_bytes(r, fixture, profile))
                          for r in fixture['rows']], max_inflight=8,
                         history_capacity_bytes=fixture['cpu_payload_credits_bytes'])
    def finish(task, pending):
        row = task.payload
        result, response, tokens = pending
        directory = args.output/'cases'/row['id']
        terminal = old.wait_retired(args.capture, response['response_id'])
        protocol.require(terminal['history_end'] == len(tokens)+len(response['output_ids'])-1,
                         'Accepted input/output history differs')
        release = old.wait_owner_released(args.capture, result['session'], response, terminal)
        proof = protocol.audit_case(args.capture, response['response_id'], profile,
                                    row['served_query'], require_pressure=True)
        action = [e for e in q.records(args.capture/'action_attention.jsonl') if q.native_matches(response, e)]
        expected = q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),
                                             profile, tokens, response['output_ids'])
        protocol.require(len(action) == 1 and action[0]['samples'] == expected,
                         'Private or rejected rows contaminated output attention')
        needles = [{**n, 'selected':old.covered(n['start'], n['end'],
                    proof['query']['selected_history_ranges'])} for n in row['needles']]
        q.save(directory/'evidence.json', dict(**proof, terminal=terminal, action=action, needles=needles))
        q.save(directory/'release.json', release)
        result.update(status='completed', evaluation=scope['score'](response['text'], row['expected']),
            completion_tokens=len(response['output_ids']), seconds=response['seconds'],
            ttft_seconds=response['ttft_seconds'], all_needles_selected=all(n['selected'] for n in needles),
            guided_query=proof['query'])
        q.save(directory/'result.json', result)
        return result
    observed = []
    def on_result(result):
        observed.append(result)
        q.save(args.output/'progress.json', dict(completed=len(observed), expected=len(fixture['rows'])))
        print(json.dumps(dict(completed=len(observed), id=result['id'],
                              exact=result['evaluation']['normalized_exact_numeric_set'])), flush=True)
    started = time.monotonic()
    results = queue.run(lambda task:old.infer_case(args, fixture, profile, task.payload, tokenizer),
        finish, on_result=on_result, on_state=lambda state:q.save(args.output/'queue.json', state),
        on_failure=lambda error:q.save(args.output/'failure.json', dict(error=repr(error))))
    q.wait_released(args.capture, [r['session'] for r in results], timeout=180)
    return dict(status='completed', cases=len(results), seconds=time.monotonic()-started,
        exact=sum(r['evaluation']['normalized_exact_numeric_set'] for r in results),
        coverage=statistics.fmean(r['evaluation']['normalized_coverage'] for r in results),
        results=results)


def run(args):
    root = args.campaign.resolve()
    args.config = root/f'{args.suite}-{args.arm}.json'
    args.output = root/f'{args.suite}-{args.arm}/answers'
    args.capture = root/f'{args.suite}-{args.arm}/server/capture'
    args.fixture = root/f'{args.suite}-fixture.json'
    args.output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.config.read_text())
    fixture = json.loads(args.fixture.read_text())
    protocol.validate_profile(profile, args.arm, args.suite)
    if args.suite == 'niah':
        result = run_niah(args, profile, fixture)
    else:
        module = __import__('benchmark_memory' if args.suite == 'memoryagentbench' else 'benchmark_longmemeval')
        settings = NS(config=args.config, output=args.output, capture=args.capture,
            reference=Path(fixture['reference']), guided_comparison=True,
            workers=8, active_groups=8, cpu_headroom_gib=16, timeout=3600,
            group_limit=0, per_group_limit=0, case_id=[], prepare_only=False, resume=False)
        started = time.monotonic()
        module.run(settings)
        q.wait_released(args.capture, [], timeout=180)
        result = dict(status='completed', cases=len(fixture['rows']), seconds=time.monotonic()-started)
    q.save(args.output/'completed.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18846)
    p.add_argument('--niah-reference', type=Path, default=NIAH_REFERENCE)
    p.add_argument('--memory-reference', type=Path, default=MEMORY_REFERENCE)
    p = sub.add_parser('run')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--arm', choices=('off', 'guided'), required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)
