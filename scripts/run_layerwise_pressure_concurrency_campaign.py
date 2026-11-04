"""Finite 2/4-way qualification, then only 26 new independent-pressure answers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(root):
    sys.path[:0]=[str(root/'qualification-c4/runtime-v1/scripts'),str(root/'qualification-c4/runtime-v1/src')]
    import recall_comparison_protocol as p
    import run_layerwise_query_diagnostic as driver
    plan=p.read(root/'campaign-plan.json');digest=p.sha(root/'campaign-plan.json')
    proof=p.read(root/'campaign-controller-proof.json')
    if (p.sha(Path(__file__))!=proof['controller_sha256'] or
            p.sha(root/'analyze_reference.py')!=proof['analyzer_sha256']):
        raise ValueError('Finite campaign controller changed')
    def frozen():
        if p.sha(root/'campaign-plan.json')!=digest:raise ValueError('Frozen campaign plan changed')
    def pointer(stage,path):
        frozen()
        p.save(root/'status.json',dict(status='running',stage=stage,active_root=str(path),scheduled_inspections_enabled=False,at=time.time()))
        p.save(root.parent/'value-recall-active.json',dict(root=str(path),campaign=str(path),queue_root=str(root),
            status=str(path/'status.json'),process=str(path/'process.json'),experiment_type=stage,at=time.time()))
    def alive(pid):
        f=Path('/proc')/str(pid)/'stat'
        return f.exists() and f.read_text().rsplit(')',1)[1].split()[0] not in ['Z','X']
    def qualification(path,n):
        if p.sha(path/'prepared.json')!=plan['qualification_prepared_sha256'][str(n)]:
            raise ValueError('Frozen qualification binding changed')
        driver.verify(path)
        state=p.read(path/'status.json')
        if state['status']!='completed':raise ValueError('Qualification did not complete: '+str(path))
        summary=p.read(path/'analysis/qualification.json')
        perf=p.read(path/'concurrency-performance.json')
        attempts=sorted((path/'formal/sequential/server-attempts').iterdir())
        work=p.read(attempts[-1]/'server/capture/public_work.json')
        peak=max(c['prefill_requests'] for c in work['completed']['batch_composition'])
        if summary['cases']!=n or not summary['methods_inputs_query_and_budgets_verified'] or peak<n:
            raise ValueError('Concurrent owner isolation or actual native batching is unproven')
        return dict(perf,workers=n,actual_max_prefill_requests=peak)
    def launch(path,label):
        frozen();driver.verify(path);pointer(label,path)
        command=[sys.executable,'-B',str(path/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),'--root',str(path)]
        with (root/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=path/'runtime-v1',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        p.save(root/'child.json',dict(pid=child.pid,stage=label,root=str(path),command=command,at=time.time()))
        if child.wait()!=0:raise RuntimeError('Finite stage failed; preserve all evidence: '+label)
        if p.read(path/'status.json')['status']!='completed':raise ValueError('Stage lacks completion proof')
    with (root/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # C4 was already launched interactively. This is a one-time workflow
        # dependency wait, not a scheduled inspection or retry loop.
        current=root/'qualification-c4';pointer('qualification-c4',current)
        while True:
            state=p.read(current/'status.json');process=p.read(current/'process.json')
            if state['status']=='failed':raise RuntimeError('C4 qualification failed; preserve attempts')
            running=alive(process['pid'])
            if state['status']=='completed' and not running:break
            if not running:raise RuntimeError('Qualification controller exited without completion')
            time.sleep(2)
        a=qualification(root/'qualification-c2',2);b=qualification(current,4)
        chosen=4 if b['input_tokens_per_second']>=1.05*a['input_tokens_per_second'] else 2
        p.save(root/'concurrency-selection.json',dict(chosen_workers=chosen,
            qualification={'c2':a,'c4':b},selection_uses_accuracy=False,
            policy='Choose C4 only if its whole-inference input throughput is at least 5% higher than C2.',at=time.time()))
        formal=root/f'formal-c{chosen}'/'independent'
        if p.sha(formal/'prepared.json')!=plan['formal_prepared_sha256'][str(chosen)]:
            raise ValueError('Frozen formal campaign binding changed')
        launch(formal,'formal-independent')
        pointer('analysis',formal)
        # The analyzer itself is frozen independently of the running model.
        sys.path.insert(0,str(root))
        from analyze_reference import analyze
        report=analyze(formal,Path(plan['baseline_root']),root/'analysis')
        p.save(root/'status.json',dict(status='completed',formal_cases=26,chosen_workers=chosen,
            baseline_answers_reused=26,macro_scores=report['macro_scores'],scheduled_inspections_enabled=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    try:run(root)
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error),at=time.time()),indent=2)+'\n')
        raise
