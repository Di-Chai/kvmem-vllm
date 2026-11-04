"""Real CLI tool/history regression, using harmless commands and a fixture API."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading


def run(binary, output):
    output.mkdir(parents=True, exist_ok=False)
    child = output/'child'; child.mkdir()
    commands = [
        f'cd {output} && pwd',
        f'cd {child} && pwd',
        f'printf "before\\n"; cd {child} && pwd',
        f'cd {output} && pwd',
        "printf '%s\\n' " + shlex.quote(f'literal cd {output} && stays'),
        "printf '%s\\n' " + shlex.quote(r'literal \\; stays'),
    ]
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('count_tokens'):
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"input_tokens":100}'); return
            index = len(requests); requests.append(request)
            tool = index < len(commands)
            block = (dict(type='tool_use', id='bash_fixture_'+str(index), name='Bash',
                          input=dict(command=commands[index], description='Isolated harmless transport check'))
                     if tool else dict(type='text', text='Fixture complete.'))
            message = dict(id='msg_fixture_'+str(index), type='message', role='assistant',
                model='local-fixture', content=[], stop_reason=None, stop_sequence=None,
                usage=dict(input_tokens=100, output_tokens=0))
            events = [('message_start',dict(type='message_start',message=message)),
                ('content_block_start',dict(type='content_block_start',index=0,
                    content_block={**block, **({'input':{}} if tool else {'text':''})})),
                ('content_block_delta',dict(type='content_block_delta',index=0,
                    delta=(dict(type='input_json_delta',partial_json=json.dumps(block['input']))
                           if tool else dict(type='text_delta',text=block['text'])))),
                ('content_block_stop',dict(type='content_block_stop',index=0)),
                ('message_delta',dict(type='message_delta',delta=dict(
                    stop_reason='tool_use' if tool else 'end_turn',stop_sequence=None),
                    usage=dict(output_tokens=20))), ('message_stop',dict(type='message_stop'))]
            data=''.join('event: '+event+'\ndata: '+json.dumps(value)+'\n\n'
                         for event,value in events).encode()
            self.send_response(200); self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    env={name:os.environ[name] for name in ['PATH','LD_LIBRARY_PATH','LANG'] if name in os.environ}
    env.update(CLAUDE_CONFIG_DIR=str(output/'config'),
        ANTHROPIC_BASE_URL=f'http://127.0.0.1:{server.server_port}',
        ANTHROPIC_API_KEY='fixture-only',ANTHROPIC_AUTH_TOKEN='fixture-only',
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',DISABLE_COMPACT='1')
    try:
        with (output/'client.log').open('x') as log:
            result=subprocess.run([str(binary),'-p','--verbose','--output-format','stream-json',
                '--permission-mode','acceptEdits','--allowedTools','Bash',
                '--setting-sources','project,local','--settings','{}','--model','local-fixture','--',
                'Run the six supplied harmless pwd/printf commands in this temporary fixture workspace.'],
                cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    finally:
        server.shutdown();server.server_close();thread.join()
        (output/'requests.json').write_text(json.dumps(requests,ensure_ascii=True,indent=2)+'\n')
    latest=requests[-1] if requests else {}
    blocks=[b for m in latest.get('messages',[]) for b in m.get('content',[]) if isinstance(b,dict)]
    tools={b['id']:b for b in blocks if b.get('type')=='tool_use'}
    results={b['tool_use_id']:b for b in blocks if b.get('type')=='tool_result'}
    exact=[tools.get('bash_fixture_'+str(i),{}).get('input',{}).get('command')==cmd
           for i,cmd in enumerate(commands)]
    errors=[i for i in range(len(commands)) if results.get('bash_fixture_'+str(i),{}).get('is_error')
            or 'bash_fixture_'+str(i) not in results]
    literal=results.get('bash_fixture_4',{}).get('content','')
    report=dict(client_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),exit=result.returncode,
        requests=len(requests),exact_command_history=exact,tool_errors=errors,
        literal_preserved=f'literal cd {output} && stays' in str(literal),
        passed=result.returncode==0 and len(requests)==7 and all(exact) and not errors
            and f'literal cd {output} && stays' in str(literal),
        scope='Real CLI, six Bash executions and subsequent serialized history; fixture API, no formal scores.')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.binary.resolve(),args.output.resolve())))
