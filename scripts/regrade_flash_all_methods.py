"""Extend the frozen Flash regrade to all historical MAB methods.

Reuse only this run's exact-request Flash cache. Existing answers, completed
verdicts, prompts, scoring formulas, and the bounded truncation rule stay fixed.
An interrupted inference arm remains pending rather than receiving a score.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time


def main(args):
    root = args.output.resolve()
    spec = importlib.util.spec_from_file_location('frozen_flash', root/'runtime/regrade_flash.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    identity = f.read(root/'mab-all-methods-identity.json')
    for name, expected in identity['frozen_files'].items():
        assert f.sha(root/name) == expected, ('Frozen file changed', name)
    plan = f.read(root/'mab-all-methods-plan.json')
    campaign = Path(plan['campaign'])
    judge = f.load(Path(plan['evaluation'])/'evaluators/historical_lme_score.py', 'judge_transport')
    judge.OFFICIAL = Path(plan['evaluation'])/'evaluators/historical_lme_evaluate_qa.py'
    assert judge.API_BASE == 'https://api.deepseek.com'
    flash = f.Flash(root, judge, args.credential_file.read_text().strip())
    os.environ['DEEPSEEK_API_KEY'] = 'transport-managed'
    completed, errors = {}, {}
    pending = dict(plan['mab'])
    for arm in list(pending):
        output = root/'mab'/arm
        summary = output/'summary.json'
        if not summary.exists():
            continue
        result = f.read(summary)
        assert result['complete'] and result['count'] == 1116 and result['judge'] == f.MODEL
        bindings = f.read(output/'bindings.json')
        assert len(bindings) == 1116
        for binding in bindings.values():
            f.unchanged(binding)
        f.check_flash(result['response_model'])
        completed[arm] = result
        del pending[arm]

    with ThreadPoolExecutor(max_workers=4) as pool:
        active = {}
        while pending or active:
            for arm, source in list(pending.items()):
                answers = Path(source)
                if len(list((answers/'cases').glob('*/result.json'))) != 1116:
                    continue
                if arm == 'all_off':
                    try:
                        state = f.read(campaign/'memoryagentbench-off/status.json')['status']
                    except (OSError, ValueError):
                        continue
                    if state != 'completed':
                        continue
                active[pool.submit(f.mab_arm, root, plan, arm, answers, flash)] = arm
                del pending[arm]
            for future, arm in list(active.items()):
                if future.done():
                    try:
                        completed[arm] = future.result()
                    except Exception as error:
                        errors[arm] = repr(error)
                    del active[future]
            f.publish_mab_completed(root, completed, pending, active.values())
            status = dict(status='grading_all_mab' if active else 'waiting_for_inference',
                          completed=sorted(completed), pending=sorted(pending),
                          active=sorted(active.values()), errors=errors, at=time.time())
            f.save(root/'mab-progress.json', status)
            f.save(root/'status.json', status)
            f.save(campaign/'grading-status.json', dict(status='grading_flash',
                   completed_mab=sorted(completed), pending_mab=sorted(pending),
                   errors=errors, output=str(root), at=time.time()))
            if errors and not active:
                raise RuntimeError('MAB grading failures: '+json.dumps(errors))
            if pending or active:
                time.sleep(10)
    report = dict(status='completed', judge=f.MODEL, response_model=flash.returned_model,
                  lme=f.read(root/'lme-comparison.json'), mab=completed,
                  niah=f.read(campaign/'niah-comparison.json'),
                  scope='All nine MAB arms; original answers, this-run Flash judgments only.')
    f.save(root/'comparison.json', report)
    f.save(campaign/'comparison-flash.json', report)
    f.save(campaign/'comparison.json', report)
    f.save(root/'status.json', dict(status='completed', at=time.time()))
    f.save(campaign/'grading-status.json', dict(status='completed', judge=f.MODEL,
                                               output=str(root), at=time.time()))
    try:
        restored = f.read(campaign/'service-restore-status.json')['status'] == 'running'
    except (OSError, ValueError):
        restored = False
    if restored:
        f.save(campaign/'completion-status.json', dict(status='completed', judge=f.MODEL,
                                                     comparison=str(root/'comparison.json'), at=time.time()))
        args.credential_file.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    args = parser.parse_args()
    with (args.output/'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main(args)
        except BaseException as error:
            temporary = args.output/'status.failed.tmp'
            temporary.write_text(json.dumps(dict(status='failed', error=repr(error), at=time.time()))+'\n')
            temporary.replace(args.output/'status.json')
            raise
