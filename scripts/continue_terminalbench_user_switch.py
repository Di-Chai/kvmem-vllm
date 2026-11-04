"""Overlap a retained CPU final review with the untouched LRU suite startup."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import psutil

try:
    from . import run_terminalbench_user_switch as prior
except ImportError:
    import run_terminalbench_user_switch as prior

base=prior.base
read,save,sha,require=base.read,base.save,base.sha,base.require


def validate_adoption(pair,transition,live_probe=base.process_alive):
    adopted=read(transition/'finalizer-adoption.json')
    require(not live_probe(adopted['previous_controller']), 'Previous switch controller remains active')
    require(sha(transition/'audit.json')==adopted['audit_sha256']
            and sha(transition/'authorization.json')==adopted['authorization_sha256'], 'Bound transition evidence changed')
    authorization=read(transition/'authorization.json');audit=read(transition/'audit.json')
    require(authorization.get('human_authorization')==prior.AUTHORIZATION and audit.get('passed') is True
            and audit.get('original_completed_evidence_preserved') is True
            and audit['summary'].get('scored')==82 and audit['summary'].get('passed')==53
            and audit['summary'].get('cancelled_by_user')==3
            and {r['trial'] for r in audit['user_cancelled']}==prior.AUTHORIZED,
            'Missing successful authorized-stop integrity audit')
    arm=pair/'thinking'
    for name,digest in audit['source_hashes'].items():
        require(sha(arm/name)==digest, 'Original candidate evidence changed: '+name)
    for key in ('original_pair','original_suite','formal_identity'):
        require(not live_probe(authorization[key]), 'Original benchmark process remains active')
    cleanup=read(arm/'cleanup.json')
    require(cleanup.get('remaining_processes')==cleanup.get('remaining_owned_gpu_pids')==[], 'Original GPU cleanup incomplete')
    base.unattempted_lru(pair)
    return adopted,audit


def check_final_review(pair,transition,audit):
    arm=pair/'thinking'
    require(read(arm/'status.json').get('status')=='completed', 'Retained candidate final review did not complete')
    result=read(arm/'results.json');measurement=read(arm/'execution-audit.json')
    require(result.get('measurement')==measurement and measurement.get('passed') is True
            and result.get('scored')==82 and result.get('passed')==53 and result.get('errors')==7,
            'Retained candidate final review changed original scores or coverage')
    for name,digest in audit['source_hashes'].items():
        require(sha(arm/name)==digest, 'Final review changed raw evidence: '+name)
    state=read(arm/'status.json')
    save(arm/'status.json',dict(state,user_cancelled=3,completion_reason='user_requested_switch_to_lru'))
    save(transition/'candidate-final-review-completed.json',dict(at=time.time(),passed=True,
        scored=82,user_cancelled=3,execution_audit_sha256=sha(arm/'execution-audit.json')),exclusive=True)


def run(pair,transition):
    adopted,audit=validate_adoption(pair,transition)
    require(not (transition/'overlapped-controller-started.json').exists(), 'Overlapped controller already attempted')
    save(transition/'overlapped-controller-started.json',dict(at=time.time(),pid=psutil.Process().pid),exclusive=True)
    started=read(transition/'before/pair-status.json')['started_at'];owner=psutil.Process()
    metadata=dict(kind='user_authorized_switch_with_cpu_review_overlap',directory=str(transition),
        thinking_scored=82,thinking_user_cancelled=3,original_scores_preserved=True,no_task_retries=True)
    save(pair/'pair-process.json',dict(pid=owner.pid,created=owner.create_time(),argv=owner.cmdline(),cwd=owner.cwd(),at=time.time(),recovery=metadata))
    save(pair/'pair-status.json',dict(status='running',arm='lru',started_at=started,at=time.time(),recovery=metadata))
    arm=pair/'thinking'
    if read(arm/'status.json').get('status')!='completed':
        save(arm/'status.json',dict(status='needs_final_review',at=time.time(),user_cancelled=3,
                                   completion_reason='user_requested_switch_to_lru'))
    lru=pair/'lru';runtime=lru/'runtime'
    with (pair/'lru-driver.log').open('xb') as log:
        child=subprocess.Popen([base.PYTHON,'-B',str(runtime/'scripts/run_terminalbench_suite.py'),'run',str(lru)],
            cwd=runtime,env=base.runtime_env(runtime),stdout=log,stderr=subprocess.STDOUT)
        record=dict(arm='lru',pid=child.pid,created=psutil.Process(child.pid).create_time(),at=time.time(),recovery=metadata)
        save(pair/'pair-child.json',record);save(transition/'lru-launched.json',record,exclusive=True)
        review_done=False;review_error=None
        while child.poll() is None:
            if not review_done and review_error is None and not base.process_alive(adopted['finalizer']):
                try:
                    check_final_review(pair,transition,audit);review_done=True
                    save(pair/'thinking-completed.json',dict(at=time.time(),status=read(arm/'status.json'),recovery=metadata),exclusive=True)
                except Exception as error:
                    review_error=repr(error)
                    save(transition/'candidate-final-review-error.json',dict(at=time.time(),error=review_error))
            time.sleep(5)
        code=child.returncode
    require(code==0 and read(lru/'status.json').get('status')=='completed','LRU ended unsuccessfully; preserve all attempts, no retry')
    save(pair/'lru-completed.json',dict(at=time.time(),status=read(lru/'status.json'),recovery=metadata),exclusive=True)
    require(review_error is None,'Candidate final review requires attention: '+str(review_error))
    if not review_done:
        require(not base.process_alive(adopted['finalizer']), 'Candidate CPU final review still active')
        check_final_review(pair,transition,audit)
    analyzer=pair/'setup-audit-continuation-code/compare_terminalbench_scored_pairs.py'
    require(sha(analyzer)==prior.known.ANALYSIS_SHA,'Qualified comparison wrapper changed')
    spec=importlib.util.spec_from_file_location('_user_switch_quality',analyzer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    comparison=module.compare(lru,arm);comparison['user_interruption']=read(arm/'user-interruption.json')
    comparison['caveats'].append('The human ended three candidate tasks before verification. They remain unsuccessful, unscored full-cohort attempts. This asymmetric stopping rule and excluded scores prevent attributing timing or overall success differences solely to GC policy.')
    output=pair/'accuracy-comparison-with-user-stop.json';save(output,comparison,exclusive=True)
    save(pair/'pair-status.json',dict(status='completed',at=time.time(),started_at=started,seconds=time.time()-started,
        result=str(output),recovery=metadata,quality_coverage='partial_with_three_user_cancelled_candidate_trials'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair-root',type=Path,required=True)
    p.add_argument('--transition',type=Path,required=True);a=p.parse_args()
    pair=a.pair_root.resolve();transition=a.transition.resolve()
    try:run(pair,transition)
    except BaseException as error:
        save(transition/'overlapped-controller-error.json',dict(at=time.time(),error=repr(error)))
        previous=read(pair/'pair-status.json');record=read(pair/'pair-process.json')
        if record.get('pid')==psutil.Process().pid:
            save(pair/'pair-status.json',dict(status='failed',at=time.time(),error=repr(error),
                started_at=previous.get('started_at'),recovery=record.get('recovery')))
        raise
