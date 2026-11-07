"""Compact read-only progress for the long owned serving matrix."""
import json
from pathlib import Path
import re
import time
from urllib.request import build_opener,ProxyHandler


def metrics(text):
    values={}
    for line in text.splitlines():
        if line.startswith('#'):continue
        match=re.match(r'([^ {]+)(?:\{[^}]*\})?\s+([-+\d.eE]+)(?:\s|$)',line)
        if match:values[match[1]]=values.get(match[1],0)+float(match[2])
    return values


root=Path.cwd();result={'time':time.time()}
state=root/'pipeline-status.json'
if state.exists():result['stage']=json.loads(state.read_text())['stage']
for folder,key,pattern in [('serving-r1','generative',r'l(\d+)-o(\d+)-c(\d+)-(observer|pair)'),
                           ('serving-r1/native-systemone','native',r'l(\d+)-c(\d+)')]:
    base=root/folder;summary=base/'summary.json'
    rows=json.loads(summary.read_text()) if summary.exists() else []
    result[key]={'cells':len(rows),'completed_requests':sum(r['completed'] for r in rows),'failed_requests':sum(r['failed'] for r in rows)}
    if rows:result[key]['last']=rows[-1]['name']
    if not base.exists():continue
    active=[p for p in base.iterdir() if p.is_dir() and re.fullmatch(pattern,p.name) and not (p/'summary.json').exists()]
    if active:
        path=max(active,key=lambda p:p.stat().st_mtime);match=re.fullmatch(pattern,path.name)
        expected=int(match[2])*int(match[3])*(2 if match[4]=='pair' else 1) if key=='generative' else int(match[2])
        current={'name':path.name,'expected_output_tokens_or_labels':expected}
        before=path/'metrics-before.txt'
        if before.exists():
            current['elapsed_approx_seconds']=round(time.time()-before.stat().st_mtime,1)
            try:
                opener=build_opener(ProxyHandler({}))
                with opener.open('http://127.0.0.1:18996/metrics',timeout=3) as response:
                    now=metrics(response.read().decode())
                past=metrics(before.read_text())
                current['emitted_tokens_or_labels']=now.get('vllm:generation_tokens_total',0)-past.get('vllm:generation_tokens_total',0)
                current['running']=now.get('vllm:num_requests_running')
                current['waiting']=now.get('vllm:num_requests_waiting')
                current['preemptions']=now.get('vllm:num_preemptions_total',0)-past.get('vllm:num_preemptions_total',0)
            except Exception as exc:current['metrics_error']=repr(exc)
        result[key]['active']=current
print(json.dumps(result))
