"""Fresh V2 A/C/D qualification client; eight original 256K cases by default."""
import argparse
import ast
import json
from pathlib import Path
import re
import statistics
import time

from tokenizers import Tokenizer

from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask
import graph_v2_protocol as protocol
import qualify_block_graph as q
import qualify_graph_niah as old

save, sha, require = old.save, protocol.sha, protocol.require


def audit_launch(profile, capture, arm, suite='niah'):
    protocol.validate_profile(profile, arm, suite)
    proof = old.audit_launch(profile, capture)
    launch = json.loads((capture.parent/'launch.json').read_text())
    argv = launch['vllm_argv']
    transfer = json.loads(argv[argv.index('--kv-transfer-config') + 1])
    actual = transfer['kv_connector_extra_config']['kvmem']
    for name in ('graph_mode', 'graph_scoring_version', 'graph_alpha', 'pressure_policy',
                 'action_fraction', 'query_history', 'step_latency_interval',
                 'action_attention', 'prefill_attention'):
        require(actual[name] == profile['kvmem'][name], 'Actual worker setting differs: '+name)
    proof.update(protocol=protocol.SCHEMA, arm=arm, worker_policy={name: actual[name] for name in
        ('graph_mode', 'graph_scoring_version', 'graph_alpha', 'pressure_policy')})
    return proof


def finish_case(args, profile, row, scorer, tokenizer, pending):
    result, response, tokens = pending
    directory = args.output/'cases'/row['id']
    terminal = old.wait_retired(args.capture, response['response_id'])
    require(terminal['history_end'] == len(tokens)+len(response['output_ids'])-1,
        'Full accepted logical input/output history differs')
    require(not terminal.get('checkpoint_tokens'), 'Independent input retained a checkpoint')
    release = old.wait_owner_released(args.capture, result['session'], response, terminal)
    save(directory/'release.json', release)
    evidence = protocol.audit_case(args.capture, response['response_id'], profile,
        row['served_query'], require_pressure=True)
    actions = [event for event in q.records(args.capture/'action_attention.jsonl')
               if q.native_matches(response, event)]
    require(len(actions) == 1, 'Missing terminal attention sampling evidence')
    from types import SimpleNamespace
    expected = q.expected_output_samples(SimpleNamespace(convert_tokens_to_ids=tokenizer.token_to_id),
                                          profile, tokens, response['output_ids'])
    require(actions[0]['samples'] == expected and actions[0]['prefill_samples'] ==
        evidence['query']['prefill_attention_samples'], 'Query/replay polluted accepted attention samples')
    needles = [{**needle, 'selected': old.covered(needle['start'], needle['end'],
        evidence['query']['selected_history_ranges'])} for needle in row['needles']]
    save(directory/'evidence.json', dict(evidence, terminal=terminal, action=actions[0], needles=needles))
    result.update(status='completed', evaluation=scorer(response['text'], row['expected']),
        completion_tokens=len(response['output_ids']), seconds=response['seconds'],
        ttft_seconds=response['ttft_seconds'], all_needles_selected=all(n['selected'] for n in needles),
        score_domain_version=protocol.SCORE_DOMAIN, phases=evidence['phases'], seed_sources=evidence['seed_sources'],
        graph_added_blocks=sum(event.get('added_blocks', 0) for event in evidence['graph']),
        hit_output_cap=len(response['output_ids']) == row.get('max_tokens', 8192))
    save(directory/'result.json', result)
    return result


def run_niah(args, profile, fixture):
    require(len(fixture['rows']) in (8, 180), 'V2 NIAH must be smoke8 or the full original180')
    require(fixture['workers'] == 8, 'Eight-owner protocol cannot be silently reduced')
    require(sha(Path(profile['tokenizer'])/'tokenizer.json') == fixture['tokenizer_sha256'], 'Tokenizer changed')
    tree = ast.parse(fixture['scorer_source'])
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == 'score',
        'Expected only the frozen original NIAH scoring function')
    scope = {'re': re}
    exec(compile(tree, 'frozen-original-niah-score', 'exec'), scope)
    tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    queue = OfflineQueue([OfflineTask(row['id'], row, old.history_bytes(row, fixture, profile))
        for row in fixture['rows']], max_inflight=8,
        history_capacity_bytes=fixture['cpu_payload_credits_bytes'])
    observed = []

    def on_result(result):
        state = queue.states[result['id']]
        result.update(cpu_queue_seconds=state['admitted_at']-state['queued_at'],
            retirement_seconds=state['completed_at']-state['inference_done_at'])
        save(args.output/'cases'/result['id']/'result.json', result)
        observed.append(result)
        save(args.output/'progress.json', dict(completed=len(observed), expected=len(fixture['rows'])))

    def on_state(state):
        save(args.output/'queue.json', state)
        with (args.output/'queue_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), **{k:v for k,v in state.items() if k != 'tasks'}))+'\n')

    begin = time.monotonic()
    results = queue.run(lambda task: old.infer_case(args, fixture, profile, task.payload, tokenizer),
        lambda task, pending: finish_case(args, profile, task.payload, scope['score'], tokenizer, pending),
        on_result=on_result, on_state=on_state,
        on_failure=lambda error: save(args.output/'client-failure.json', dict(error=repr(error))))
    released = q.wait_released(args.capture, [row['session'] for row in results], timeout=180)
    archive = released['archive']
    require(archive.get('disk_read_bytes', 0) == archive.get('disk_write_bytes', 0) ==
            archive.get('nvme_capacity_bytes', 0) == 0, 'CPU authority unexpectedly used disk KV')
    end = time.monotonic()
    steps = [event for event in q.records(args.capture/'model_step_latency.jsonl')
             if begin <= event['started_at'] <= end]
    concurrency = protocol.audit_actual_concurrency(steps)
    require(len(results) == len(fixture['rows']) and queue.peak_inflight == 8 and
            queue.reserved_bytes == queue.inflight == 0, 'Incomplete cohort or leaked admission credits')
    return dict(status='passed', protocol=protocol.SCHEMA, arm=args.arm, suite='niah',
        cases=len(results), fixture_sha256=sha(args.fixture), all_owned_sessions_released=True,
        protocol_passed=True, quality_improvement_established=False,
        wall_seconds=end-begin, measurement_started_at=begin, measurement_finished_at=end,
        cpu_admission_peak=queue.peak_inflight,
        eight_owner_admission_qualified=queue.peak_inflight == 8,
        cpu_admission_peak_bytes=queue.peak_bytes, actual_concurrency=concurrency,
        admission_protocol='Continuous FIFO with eight owner slots; per-owner retirement before refill, no wave barriers.',
        macro_coverage=statistics.fmean(row['evaluation']['normalized_coverage'] for row in results),
        exact=sum(row['evaluation']['normalized_exact_numeric_set'] for row in results),
        graph_changed_cases=sum(row['graph_added_blocks'] > 0 for row in results),
        output_tokens_per_wall_s=sum(row['completion_tokens'] for row in results)/(end-begin),
        throughput_scope='Whole campaign including long prefill, Query, output and retirement; not decode TPS.',
        score_domain_version=protocol.SCORE_DOMAIN, final_archive=archive,
        step_summary=q.summarize_steps(args.capture, begin, end), results=sorted(results, key=lambda row:row['id']))


def run_reference_suite(args, profile, fixture):
    from types import SimpleNamespace
    suite = fixture['suite']
    module = __import__('benchmark_memory' if suite == 'memoryagentbench' else 'benchmark_longmemeval')
    options = SimpleNamespace(config=args.config, capture=args.capture, output=args.output/'dataset',
        reference=Path(fixture['reference']), workers=8, active_groups=8, cpu_headroom_gib=16,
        timeout=3600, group_limit=0, per_group_limit=0, case_id=[], prepare_only=False, resume=False)
    begin = time.monotonic()
    module.run(options)
    rows = [json.loads(path.read_text()) for path in (options.output/'cases').glob('*/result.json')]
    expected = 1116 if suite == 'memoryagentbench' else 102
    require(len(rows) == expected and all(row['status'] == 'completed' for row in rows), 'Reference suite incomplete')
    released = q.wait_released(args.capture, [], timeout=180)
    steps = [event for event in q.records(args.capture/'model_step_latency.jsonl') if event['started_at'] >= begin]
    return dict(status='passed', protocol=protocol.SCHEMA, arm=args.arm, suite=suite, cases=expected,
        fixture_sha256=sha(args.fixture), all_owned_sessions_released=True,
        score_domain_version=protocol.SCORE_DOMAIN, actual_concurrency=protocol.audit_actual_concurrency(steps,
        require_eight=False), quality_improvement_established=False,
        evaluation_scope='Original benchmark answers retained for the separate frozen evaluator; no surrogate quality claim.',
        wall_seconds=time.monotonic()-begin, final_archive=released['archive'], results=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output', 'fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=protocol.ARMS, required=True)
    args = parser.parse_args(argv)
    profile, fixture = json.loads(args.config.read_text()), json.loads(args.fixture.read_text())
    require(fixture.get('schema') == protocol.SCHEMA, 'Unknown benchmark fixture version')
    protocol.validate_profile(profile, args.arm, fixture['suite'])
    require(not args.output.exists(), 'Use a fresh client directory; failed cases are never retried')
    args.output.mkdir(parents=True)
    try:
        save(args.output/'launch-audit.json', audit_launch(profile, args.capture, args.arm, fixture['suite']))
        result = (run_niah if fixture['suite'] == 'niah' else run_reference_suite)(args, profile, fixture)
        save(args.output/'results.json', result)
    except BaseException as error:
        save(args.output/'client-failure.json', dict(error=repr(error), at=time.time()))
        raise
    print(json.dumps({k:v for k,v in result.items() if k not in ('results', 'final_archive')}), flush=True)


if __name__ == '__main__':
    main()
