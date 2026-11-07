"""Paired native-label format probes on already frozen past-step requests."""
import argparse
import copy
import json
from pathlib import Path
import time
from urllib.request import Request,build_opener,ProxyHandler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:18993')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    decisions={r['step']:r for r in (json.loads(l) for l in (args.results/'decisions.jsonl').open())}
    opener=build_opener(ProxyHandler({}));results=[]
    for step in (6,10,16):
        original=json.loads((args.results/'calls'/(decisions[step]['call_id']+'.request.json')).read_text())
        for variant in ('original','explicit_letter'):
            payload=copy.deepcopy(original)
            payload['questions']={'stage_link':payload['questions']['stage_link']}
            if variant=='explicit_letter':
                payload['questions']['stage_link']['instructions']+= (
                    ' Return ONLY the single option LETTER (A, B, C, etc.). '
                    'Do not return an option name, a stage ID such as S001, JSON, or an explanation.')
            ident=f'step-{step}-{variant}'
            (args.output/(ident+'.request.json')).write_text(json.dumps(payload,indent=2)+'\n')
            req=Request(args.base_url+'/v1/systemone',data=json.dumps(payload).encode(),
                        headers={'Content-Type':'application/json'})
            begin=time.perf_counter()
            with opener.open(req,timeout=180) as f:response=json.load(f)
            seconds=time.perf_counter()-begin
            (args.output/(ident+'.response.json')).write_text(json.dumps(response,indent=2)+'\n')
            row=dict(step=step,variant=variant,seconds=seconds,answer=response['answers']['stage_link'],
                     diagnostics=response['diagnostics']['stage_link'],usage=response['usage'])
            results.append(row);print(json.dumps(row),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
