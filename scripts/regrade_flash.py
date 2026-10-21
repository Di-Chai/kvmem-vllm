"""Regrade frozen LME and MAB answers with the current official Flash judge.

The plan names immutable answer files. A fresh run has a fresh request cache;
historical judge verdicts are never imported. Identical requests within this run
share a response. Model inference and the service-restoration controller are
independent of this process.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS


MODEL = 'deepseek-flash'
DOC = 'https://api-docs.deepseek.com/updates/'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))


def check_flash(model):
    if model != MODEL and not model.startswith('deepseek-v4.1-flash'):
        raise RuntimeError('Unexpected judge response model: '+model)


class TruncatedFlashVerdict(ValueError):
    """An incomplete response may use the official JSON repair stage."""


def complete_mab_call(flash, body):
    """Apply the same bounded truncation retry policy to every MAB arm.

    Preserve the original prompt/thinking settings and every failed response.
    An incomplete verdict is never interpreted as an incorrect model answer.
    """
    original_hash = digest(body)
    attempts = []
    request = dict(body)
    while True:
        raw, ident = flash.request(request)
        choice = raw['choices'][0]
        content = (choice['message'].get('content') or '').strip()
        attempts.append(dict(request_sha256=ident, max_tokens=request['max_tokens'],
                             finish_reason=choice['finish_reason'], usage=raw.get('usage')))
        complete = choice['finish_reason'] == 'stop' and bool(content)
        if len(attempts) > 1 or not complete:
            save(flash.root/'mab-truncation-retries'/(original_hash+'.json'),
                 dict(complete=complete, policy='length-only-4x-cap-16384-v1', attempts=attempts))
        if complete:
            return content
        if choice['finish_reason'] != 'length':
            raise RuntimeError('Incomplete Flash MAB verdict; inspect call '+ident)
        if request['max_tokens'] >= 16384:
            raise TruncatedFlashVerdict('Truncated Flash MAB verdict; inspect call '+ident)
        request = {**request, 'max_tokens':min(16384, request['max_tokens']*4)}


def publish_mab_completed(root, completed, pending, active):
    """Publish finished arms without waiting for the remaining inference."""
    report = dict(judge=MODEL, status='partial' if pending or active else 'completed',
                  completed=sorted(completed), pending=sorted(pending), active=sorted(active),
                  arms={arm:{k:v for k,v in row.items() if k!='values'}
                        for arm,row in completed.items()}, pairs={}, at=time.time())
    for left, right in [('guided','all_on'), ('all_on','historical_C'),
                        ('guided','historical_C'), ('all_off','all_on'),
                        ('all_off','historical_C'), ('all_off','guided')]:
        if left not in completed or right not in completed:
            continue
        lhs, rhs = completed[left], completed[right]
        assert lhs['values'].keys() == rhs['values'].keys()
        delta = [v-rhs['values'][k] for k,v in lhs['values'].items()]
        report['pairs'][left+'_vs_'+right] = dict(
            macro_delta=lhs['three_category_macro']-rhs['three_category_macro'],
            improved=sum(x>1e-12 for x in delta), regressed=sum(x < -1e-12 for x in delta),
            tied=sum(abs(x)<=1e-12 for x in delta))
    save(root/'mab-comparison-completed.json',report)


def bind_cases(cases, expected):
    records = {}
    bindings = {}
    for path in sorted(cases.glob('*/result.json')):
        row = read(path)
        assert row.get('status', 'completed') == 'completed', path
        assert row['answer'] == row['raw_completion'].split('</think>', 1)[-1].strip(), path
        ident = row['id']
        assert ident not in records
        records[ident] = row
        bindings[ident] = dict(path=str(path.resolve()), sha256=sha(path))
    assert len(records) == expected, (cases, len(records), expected)
    return records, bindings


def unchanged(binding):
    assert sha(Path(binding['path'])) == binding['sha256'], 'Source answer changed'


class Flash:
    def __init__(self, root, judge, key):
        self.root = root
        self.judge = judge
        self.client = judge.Client(key)
        self.guard = threading.Lock()
        self.locks = {}
        self.returned_model = None

    def request(self, body):
        assert body['model'] == MODEL
        ident = digest(body)
        with self.guard:
            lock = self.locks.setdefault(ident, threading.Lock())
        with lock:
            path = self.root/'calls'/ident
            cached = path.with_suffix('.response.json')
            if cached.exists():
                assert read(path.with_suffix('.request.json')) == body
                raw = read(cached)['response']
            else:
                save(path.with_suffix('.request.json'), body)
                raw, timing = self.client.request('/chat/completions', body)
                save(cached, dict(response=raw, **timing))
            model = raw.get('model', '').lower()
            check_flash(model)
            with self.guard:
                if self.returned_model is None:
                    self.returned_model = model
                assert self.returned_model == model, 'Judge version changed during this run'
            return raw, ident


def lme(root, plan, flash):
    judge = flash.judge
    check = judge.prompt_function()
    parameters = {**judge.PARAMETERS, 'model': MODEL}
    calibration = root/'calibration.json'
    if not calibration.exists():
        checks = []
        fixtures = read(Path(plan['evaluation'])/'evaluators/historical_lme_calibration.json')['cases']
        for fixture in fixtures:
            ref = fixture['fixture']
            body = {**parameters, 'messages':[dict(role='user', content=check(
                ref['question_type'], ref['question'], ref['answer'], fixture['hypothesis'],
                abstention=ref['question_id'].endswith('_abs')))]}
            raw, ident = flash.request(body)
            checks.append(dict(expected=fixture['expected'], observed=judge.verdict(raw), request_sha256=ident))
        save(calibration, dict(passed=all(x['expected']==x['observed'] for x in checks), cases=checks))
    assert read(calibration)['passed'], 'Flash calibration failed'
    jobs = []
    expected = None
    for arm, cases in plan['lme'].items():
        records, bindings = bind_cases(Path(cases), 102)
        metadata = {k:{f:r[f] for f in ('question_id','question_type','question','gold_answer')}
                    for k,r in records.items()}
        if expected is None:
            expected = metadata
        assert metadata == expected, 'LME question/reference metadata differs: '+arm
        snapshot = root/'lme'/arm/'bindings.json'
        if snapshot.exists():
            assert read(snapshot) == bindings
        save(snapshot, bindings)
        for ident, row in records.items():
            jobs.append((arm, ident, row, bindings[ident]))
    # Interleave methods without telling the judge which method produced an answer.
    jobs.sort(key=lambda x:digest(['flash-regrade-20261021',x[0],x[1]]))

    def grade(job):
        arm, ident, row, binding = job
        unchanged(binding)
        body = {**parameters, 'messages':[dict(role='user', content=check(
            row['question_type'], row['question'], row['gold_answer'], row['answer'],
            abstention=row['question_id'].endswith('_abs')))]}
        raw, request_hash = flash.request(body)
        result = dict(id=ident, correct=judge.verdict(raw), question_type=row['question_type'],
                      request_sha256=request_hash, source=binding, response_model=raw['model'])
        save(root/'lme'/arm/'scores'/(ident+'.json'), result)
        return arm, result

    values = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(grade, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            arm, row = future.result()
            values[arm][row['id']] = row
            save(root/'lme-progress.json', dict(completed=i, expected=len(jobs), by_arm={k:len(v) for k,v in values.items()}))
    report = dict(status='completed', judge=MODEL, returned_model=flash.returned_model,
                  prompt_sha256=sha(judge.OFFICIAL), arms={}, pairs={})
    for arm, rows in values.items():
        assert len(rows) == 102
        correct = sum(x['correct'] for x in rows.values())
        by_type = defaultdict(list)
        for row in rows.values():
            by_type[row['question_type']].append(row['correct'])
        report['arms'][arm] = dict(correct=correct, count=102, accuracy=correct/102,
                                  by_type={k:statistics.fmean(v) for k,v in by_type.items()})
    for arm in values:
        if arm == 'all_off':
            continue
        delta = {k:int(v['correct'])-int(values[arm][k]['correct']) for k,v in values['all_off'].items()}
        report['pairs']['all_off_vs_'+arm] = dict(improved=sum(x>0 for x in delta.values()),
            regressed=sum(x<0 for x in delta.values()), tied=sum(x==0 for x in delta.values()),
            changes={k:v for k,v in delta.items() if v})
    save(root/'lme-comparison.json', report)
    return report


def mab_arm(root, plan, arm, answers, flash):
    output = root/'mab'/arm
    records, bindings = bind_cases(answers/'cases', 1116)
    snapshot = output/'bindings.json'
    if snapshot.exists():
        assert read(snapshot) == bindings
    save(snapshot, bindings)
    reference = Path(read(answers/'manifest.json')['reference'])
    original = {r['key']:r for r in lines(reference/'results/attention/memoryagentbench/special_judgments.jsonl')}
    assert all(x.get('judge_max_tokens') == 1024 for x in original.values() if x['metric']=='llm_judge_correct')
    scored = output/'rows/selected/results.jsonl'
    if not (output/'official_local_summary.json').exists():
        write_lines(scored, records.values())
        with (output/'local.log').open('w') as log:
            subprocess.run([sys.executable,'-P',str(Path(plan['evaluation'])/'evaluators/score_memoryagentbench_official_local.py'),
                '--results-dir',str(output),'--official-repo',str(Path(plan['evaluation'])/'official'),
                '--allow-partial'],check=True,stdout=log,stderr=subprocess.STDOUT)
    module = load(Path(plan['evaluation'])/'evaluators/judge_memoryagentbench_special_flash.py', 'flash_mab_'+arm)
    args = NS(results_dir=output, official_repo=Path(plan['evaluation'])/'official',
              base_url='https://api.deepseek.com', model=MODEL, thinking='enabled',
              infbench_thinking='disabled', longmem_max_tokens=1024, allow_partial=True,
              limit=None, timeout=60, max_retries=4)
    module.parse_args = lambda:args

    def call(session, ignored_args, api_key, prompt, max_tokens=512, thinking='enabled'):
        body = dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,
                    max_tokens=max_tokens,stream=False,thinking=dict(type=thinking))
        return complete_mab_call(flash, body)

    module.call = call
    # The original evaluator checks this variable before invoking the injected
    # transport. The actual credential remains only in Flash.client.
    assert os.environ.get('DEEPSEEK_API_KEY') == 'transport-managed'
    assert module.main() == 0
    special = {r['key']:r for r in lines(output/'special_judgments.jsonl')}
    assert len(special)==316
    by_source = defaultdict(list)
    source_category = {}
    values = {}
    for row in lines(scored):
        ident = module.key(row)
        value = float(special[ident]['score']) if ident in special else float(row['official_local_metrics']['substring_exact_match'])
        values[ident] = value
        by_source[row['source']].append(value)
        source_category[row['source']] = row['split']
    categories = defaultdict(list)
    for source, vals in by_source.items():
        categories[source_category[source]].append(statistics.fmean(vals))
    assert len(categories)==3 and len(values)==1116
    report = dict(complete=True, count=1116, judge=MODEL, response_model=flash.returned_model,
                  three_category_macro=statistics.fmean(statistics.fmean(v) for v in categories.values()),
                  by_source={k:dict(count=len(v),score=statistics.fmean(v)) for k,v in by_source.items()},
                  by_category={k:statistics.fmean(v) for k,v in categories.items()}, values=values)
    save(output/'summary.json',report)
    return report


def run(args):
    root = args.output.resolve()
    plan = read(root/'plan.json')
    campaign = Path(plan['campaign'])
    identity = read(root/'identity.json')
    for name, expected in identity['frozen_files'].items():
        assert sha(root/name)==expected, 'Frozen evaluator/adapter changed'
    judge = load(Path(plan['evaluation'])/'evaluators/historical_lme_score.py', 'flash_lme')
    judge.OFFICIAL = Path(plan['evaluation'])/'evaluators/historical_lme_evaluate_qa.py'
    assert judge.API_BASE == 'https://api.deepseek.com'
    flash = Flash(root,judge,args.credential_file.read_text().strip())
    assert flash.client.key
    models,_ = flash.client.request('/models')
    available = [m['id'] for m in models['data']]
    assert MODEL in available
    save(root/'model-availability.json',dict(available=available,checked_at=time.time(),documentation=DOC,
         documented_release='DeepSeek-V4.1-Flash-1017',note='API model field may expose an alias rather than a dated revision.'))
    save(root/'status.json',dict(status='grading_lme',at=time.time()))
    lme_report = lme(root,plan,flash)
    save(root/'status.json',dict(status='grading_mab',lme_completed=True,at=time.time()))
    save(campaign/'grading-status.json',dict(status='grading_flash',lme_completed=True,output=str(root),at=time.time()))
    os.environ['DEEPSEEK_API_KEY']='transport-managed'
    pending = dict(plan['mab'])
    completed = {}
    # At most three background arms plus the new arm: four API callers total.
    with ThreadPoolExecutor(max_workers=4) as pool:
        active = {}
        while pending or active:
            for arm, source in list(pending.items()):
                answers = Path(source)
                if len(list((answers/'cases').glob('*/result.json'))) != 1116:
                    continue
                # Wait for the active client's release/audit before scoring it.
                if arm=='all_off' and read(campaign/'memoryagentbench-off/status.json')['status']!='completed':
                    continue
                active[pool.submit(mab_arm,root,plan,arm,answers,flash)] = arm
                del pending[arm]
            for future,arm in list(active.items()):
                if not future.done():
                    continue
                completed[arm]=future.result()
                del active[future]
                save(root/'mab-progress.json',dict(completed=sorted(completed),pending=sorted(pending),active=sorted(active.values())))
                publish_mab_completed(root,completed,pending,active.values())
            assert read(campaign/'status.json')['status']!='failed', 'Inference controller failed'
            if pending or active:
                time.sleep(10)
    os.environ.pop('DEEPSEEK_API_KEY',None)
    report = dict(status='completed',judge=MODEL,response_model=flash.returned_model,
                  lme=lme_report,mab=completed,niah=read(campaign/'niah-comparison.json'),
                  scope='Existing model answers regraded; no old Flash or Pro judgments reused.')
    save(root/'comparison.json',report)
    save(campaign/'comparison-flash.json',report)
    save(campaign/'comparison.json',report)
    save(campaign/'grading-status.json',dict(status='completed',judge=MODEL,output=str(root),at=time.time()))
    while not (campaign/'service-restore-status.json').exists():
        assert read(campaign/'status.json')['status']!='failed'
        time.sleep(15)
    assert read(campaign/'service-restore-status.json')['status']=='running','Service restoration failed'
    save(root/'status.json',dict(status='completed',at=time.time()))
    save(campaign/'completion-status.json',dict(status='completed',judge=MODEL,comparison=str(root/'comparison.json'),at=time.time()))
    args.credential_file.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--credential-file',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as error:
            save(args.output/'status.json',dict(status='failed',error=str(error),at=time.time()))
            raise
