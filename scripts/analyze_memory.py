"""Audit frozen MemoryAgentBench outputs and run qw3's unchanged evaluators.

No model answer is regenerated. Missing external judgments remain unscored.
Use --external only with the original judge credentials in the environment.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from benchmark_niah import save, sha

FROZEN = Path('/home/chaidi/qw3/benchmark/qw3_memory_comparison_20261004/frozen')
PYTHON = '/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python'
OFFICIAL_REPO = None


def key(record):
    return f"{record['split']}|{record['dataset_row']}|{record['question_index']}"


def binding(record):
    fields = ['answer', 'gold_answer', 'raw_question', 'question_type', 'source',
              'prompt_sha256', 'token_sha256']
    data = {k: record.get(k) for k in fields}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def write_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    temporary.replace(path)


def analyze(args):
    manifest = json.loads((args.campaign/'manifest.json').read_text())
    reference = Path(manifest['reference'])
    assert sha(reference/'cohort.json') == manifest['cohort_sha256']
    out = args.campaign/'scoring'
    records, expected, evidence = {}, {}, []
    for row in manifest['rows']:
        expected[row['id']] = row
        directory = args.campaign/'cases'/row['id']
        if not (directory/'result.json').exists(): continue
        record = json.loads((directory/'result.json').read_text())
        if record['status'] != 'completed': continue
        response = json.loads((directory/'response.json').read_text())
        assert record['token_sha256'] == row['token_sha256']
        assert record['prompt_sha256'] == row['prompt_sha256']
        assert record['query_sha256'] == row['query_sha256']
        assert record['prompt_tokens'] == row['input_tokens']
        assert record['raw_completion'] == response['choices'][0]['text']
        assert record['usage'] == response['usage']
        assert record['answer'] == record['raw_completion'].split('</think>', 1)[-1].strip()
        identity = key(record)
        assert identity not in records
        records[identity] = record
        evidence.append({'id':row['id'], 'result_sha256':sha(directory/'result.json')})
    assert records, 'No completed answers to score'
    bindings = {k:binding(r) for k,r in records.items()}
    prior = json.loads((out/'judge-answer-manifest.json').read_text()) if (out/'judge-answer-manifest.json').exists() else {}
    assert all(bindings.get(k) == v for k,v in prior.items()), 'A previously scored answer changed'
    save(out/'judge-answer-manifest.json',bindings)
    write_lines(out/'rows/selected/results.jsonl',records.values())
    subprocess.run([PYTHON, '-P', str(FROZEN/'score_memoryagentbench_official_local.py'),
                    '--results-dir',str(out.resolve()),'--allow-partial'] +
                   (['--official-repo',str(OFFICIAL_REPO)] if OFFICIAL_REPO is not None else []),check=True)

    # A byte-identical answer/input can reuse its original judgment, but the
    # cached outcome must never be applied just because the question matches.
    judgments = {r['key']:r for r in lines(out/'special_judgments.jsonl')}
    reused = json.loads((out/'reused-judgments.json').read_text()) if (out/'reused-judgments.json').exists() else {}
    reference_judgments = {r['key']:r for r in lines(reference/'results/attention/memoryagentbench/special_judgments.jsonl')}
    judge_fields = ('judge_model','judge_thinking','judge_max_tokens','metric')
    def compatible(judgment):
        original = reference_judgments[judgment['key']]
        return all(judgment.get(field)==original.get(field) for field in judge_fields)
    assert all(compatible(j) for j in judgments.values()), 'A cached judgment uses a different judge configuration'
    for arm in ('attention','query'):
        old = reference/'results'/arm/'memoryagentbench'
        old_bindings = json.loads((old/'judge-answer-manifest.json').read_text())
        for judgment in lines(old/'special_judgments.jsonl'):
            identity = judgment['key']
            if (identity not in judgments and identity in bindings and
                    old_bindings.get(identity) == bindings[identity] and compatible(judgment)):
                judgments[identity] = judgment
                reused[identity] = {'path':str(old/'special_judgments.jsonl'),
                    'file_sha256':sha(old/'special_judgments.jsonl'),'answer_input_sha256':bindings[identity]}
    write_lines(out/'special_judgments.jsonl',judgments.values())
    save(out/'reused-judgments.json',reused)
    targets = {k:r for k,r in records.items() if r['source'].startswith(('longmemeval_','infbench_'))}
    if args.external:
        failures = []
        for identity,record in targets.items():
            if identity in judgments: continue
            dest = out/'per_question'/hashlib.sha256(identity.encode()).hexdigest()[:24]
            write_lines(dest/'rows/selected/results.jsonl',[record])
            budget = reference_judgments[identity].get('judge_max_tokens',1024)
            command = [PYTHON,'-P',str(FROZEN/'judge_memoryagentbench_special.py'),
                '--results-dir',str(dest.resolve()),'--allow-partial','--model','deepseek-v4-pro',
                '--longmem-max-tokens',str(budget)]
            if OFFICIAL_REPO is not None:
                command += ['--official-repo',str(OFFICIAL_REPO)]
            with (dest/'judge.log').open('w') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append(identity)
            else:
                values = lines(dest/'special_judgments.jsonl')
                assert len(values) == 1 and values[0]['key'] == identity
                judgments[identity] = values[0]
                write_lines(out/'special_judgments.jsonl',judgments.values())
        save(out/'judge-failures.json',failures)

    sources, categories = defaultdict(list), defaultdict(list)
    old_comparison = json.loads((reference/'comparison.json').read_text())
    pending = []
    scored_records = lines(out/'rows/selected/results.jsonl')
    for record in scored_records:
        identity = key(record)
        if identity in targets:
            if identity not in judgments:
                pending.append(identity); continue
            value = float(judgments[identity]['score'])
        else:
            value = float(record['official_local_metrics']['substring_exact_match'])
        sources[record['source']].append(value)
        categories[record['split']]  # Filled with equally weighted source means below.
    for source,values in sources.items():
        category = next(r['split'] for r in records.values() if r['source']==source)
        categories[category].append(statistics.fmean(values))
    reference_sources = old_comparison['summary']['attention']['memoryagentbench']['by_source']
    completed_sources = Counter(r['source'] for r in records.values())
    source_results = {
        source: {'completed':completed_sources[source], 'scored':len(sources[source]),
                 'expected':reference_value['scored'],
                 'complete':len(sources[source])==reference_value['scored'],
                 'score':statistics.fmean(sources[source]) if sources[source] else None,
                 'metric':reference_value['metric']}
        for source,reference_value in reference_sources.items()}
    # A selectively reused subset of API judgments is not an unbiased estimate
    # of its category. Publish category means only after every source is scored.
    category_sources = defaultdict(set)
    for record in records.values(): category_sources[record['split']].add(record['source'])
    category_results = {category: statistics.fmean(values)
                        if len(records)==1116 and all(source_results[s]['complete']
                            for s in category_sources[category]) else None
                        for category,values in categories.items() if values}
    complete = len(records)==1116 and not pending
    report = {'expected':1116,'completed':len(records),'scored':len(records)-len(pending),
        'unscored_judgments':len(pending),'pending_judgment_keys':pending,
        'exact_answer_judgments_reused':len(reused),'complete':complete,
        'by_source':source_results,
        'by_category':category_results,
        'three_category_macro':statistics.fmean(statistics.fmean(v) for v in categories.values()) if complete and len(categories)==3 else None,
        'reference':{arm:old_comparison['summary'][arm]['memoryagentbench'] for arm in ('attention','query')},
        'note':'Complete means the frozen qw3 1116-question cohort, not the 3671-question full benchmark. Missing API judgments are unscored, never incorrect. A reused-judgment subset is selected by exact answer identity and is not an unbiased source accuracy estimate. Reference combines cold and frozen-prefix execution; model quantization/runtime differ.',
        'scorer_sha256':{name:sha(FROZEN/name) for name in ('score_memoryagentbench_official_local.py','judge_memoryagentbench_special.py')},
        'answer_bindings_checked':len(evidence)}
    ledger = json.loads((args.campaign/'cpu_admission.json').read_text())
    times = [e['at'] for e in ledger['events']]
    elapsed = max(times)-min(times)
    new_input = sum(r['prompt_tokens']-(r['prefix_tokens'] if r['prefix_reused'] else 0) for r in records.values())
    report['performance'] = {'wall_seconds':elapsed,'peak_admitted_groups':ledger['peak'],
        'client_workers':manifest['workers'],'new_input_tokens':new_input,
        'logical_prompt_tokens':sum(r['prompt_tokens'] for r in records.values()),
        'output_tokens':sum(r['usage']['completion_tokens'] for r in records.values()),
        'new_input_tokens_per_second':new_input/elapsed,
        'output_tokens_per_second':sum(r['usage']['completion_tokens'] for r in records.values())/elapsed,
        'cold_mean_seconds':statistics.fmean(r['seconds'] for r in records.values() if not r['prefix_reused']),
        'warm_mean_seconds':statistics.fmean(r['seconds'] for r in records.values() if r['prefix_reused']) if any(r['prefix_reused'] for r in records.values()) else None,
        'scope':'End-to-end mixed workload. Restored prefix tokens are excluded from new-input throughput; neither number is isolated GPU prefill/decode speed.'}
    history_groups = defaultdict(list)
    for record in records.values(): history_groups[record['id'].rsplit('_q',1)[0]].append(record)
    for group in history_groups.values():
        assert len({r['prefix_sha256'] for r in group})==1
        assert len({r['attention']['prefill_attention_samples'] for r in group})==1
        assert all(r['attention']['attention_source']=='prefill' and
                   r['attention']['action_history_pages']==0 for r in group)
        assert sum(not r['prefix_reused'] for r in group)==1
    report['restore_audit'] = {'history_groups':len(history_groups),
        'cold_requests':sum(not r['prefix_reused'] for r in records.values()),
        'restored_requests':sum(r['prefix_reused'] for r in records.values()),
        'identical_prefix_and_prefill_sample_count_per_group':True,
        'zero_output_history_at_every_query':True}
    resources = lines(args.campaign/'resources.jsonl')
    if resources:
        archive = [r['archive'] for r in resources if 'archive' in r]
        def maximum(field): return max((r.get(field) or 0 for r in archive),default=0)
        report['resources'] = {'samples':len(resources),
            'minimum_available_host_bytes':min(r['meminfo']['MemAvailable'] for r in resources),
            'maximum_swap_used_bytes':max(r['meminfo']['SwapTotal']-r['meminfo']['SwapFree'] for r in resources),
            'maximum_cpu_authority_bytes':maximum('cpu_authority_bytes'),
            'maximum_history_disk_read_bytes':maximum('disk_read_bytes'),
            'maximum_history_disk_write_bytes':maximum('disk_write_bytes'),
            'maximum_nvme_capacity_bytes':maximum('nvme_capacity_bytes'),
            'maximum_preemption_counter':max((float(line.rsplit(' ',1)[-1])
                for r in resources for line in r.get('metrics',[])
                if line.startswith('vllm:num_preemptions_total{')),default=None),
            'archive_backends':sorted({r['archive_backend'] for r in archive if r.get('archive_backend')})}
        assert report['resources']['archive_backends']==['cpu']
        assert all(report['resources'][name]==0 for name in (
            'maximum_history_disk_read_bytes','maximum_history_disk_write_bytes','maximum_nvme_capacity_bytes'))
    save(out/'audit.json',{'results':evidence,'bindings':bindings})
    save(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reference','pending_judgment_keys','scorer_sha256')},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--external',action='store_true')
    analyze(parser.parse_args())
