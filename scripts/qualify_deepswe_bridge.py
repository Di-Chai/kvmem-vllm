"""Real eight-client tool round trips; separate from official DeepSWE scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main(a):
    tasks=a.tasks.read_text().splitlines()[:8]
    def run(task):
        http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        p={'model':'kvmem-deepswe','max_tokens':2048,'stream':False,
           'system':'You are a local protocol test assistant. Follow the user exactly.',
           'tools':[{'name':'Bash','description':'Run a shell command.',
             'input_schema':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}],
           'messages':[{'role':'user','content':'Call Bash exactly once with command "printf READY". After its result, reply with just READY.'}]}
        responses=[]
        for turn in range(2):
            request=urllib.request.Request(a.base+'/task/'+task+'/v1/messages',
                data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
            start=time.monotonic()
            with http.open(request,timeout=900) as response:r=json.load(response)
            responses.append({'seconds':time.monotonic()-start,'response':r})
            calls=[b for b in r['content'] if b['type']=='tool_use']
            if turn==0:
                assert len(calls)==1 and calls[0]['name']=='Bash' and 'READY' in calls[0]['input']['command'],r
                p['messages'] += [{'role':'assistant','content':r['content']},
                                 {'role':'user','content':[{'type':'tool_result','tool_use_id':calls[0]['id'],'content':'READY'}]}]
            else:
                assert not calls and 'READY' in ''.join(b.get('text','') for b in r['content']),r
        return {'task':task,'turns':responses}
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,tasks))
    a.output.write_text(json.dumps({'passed':True,'clients':8,'results':results},indent=2))
    print(json.dumps({'passed':True,'clients':8,'turns':16}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:18483')
    p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
