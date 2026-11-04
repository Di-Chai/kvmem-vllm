import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid
import pytest
from scripts.terminalbench_harness import controls, single_line, standalone_heredoc, install

@pytest.mark.parametrize('text,expected', [('C-c',['C-c']),('C-c\n',['C-c']),('C-cC-c',['C-c','C-c']),('C-d',['C-d']),('echo C-c\n',None),('cat <<EOF\nC-c\nEOF\n',None),('',None)])
def test_exact_controls(text,expected):
    assert controls(text)==expected

@pytest.mark.parametrize('text,expected',[('echo hi\n',True),('echo hi',False),('echo hi\nsleep 5\n',False),('C-c\n',False)])
def test_single_line_only(text,expected):assert single_line(text)==expected


@pytest.mark.parametrize('text', [
    "python3 << 'EOF'\nprint('ok')\nEOF\n",
    'python3 - <<"PY_END"\nprint("EOF")\nPY_END\n',
    "cat > /tmp/plain-text <<'DATA'\n$HOME `false`\nDATA\n",
    "cat <<'EMPTY'\nEMPTY\n",
])
def test_standalone_heredoc_accepts_only_complete_literal_command(text):
    assert standalone_heredoc(text)


@pytest.mark.parametrize('text', [
    "python3 <<'EOF'\nprint('still entering')\n",
    "python3 <<'EOF'\nprint('ok')\nEOF",
    "python3 <<'EOF'\nprint('ok')\nEOF\nsleep 10\n",
    "python3 <<'EOF'\nEOF\nsleep 10\nEOF\n",
    "python3 <<'EOF' | cat\nprint('ok')\nEOF\n",
    "true && python3 <<'EOF'\nprint('ok')\nEOF\n",
    "false; python3 <<'EOF'\nprint('ok')\nEOF\n",
    "python3 <<EOF\n$(sleep 10)\nEOF\n",
    "cat <<-'EOF'\n\tEOF\n",
    "cat <<'A' <<'B'\nA\nB\n",
    "python3 <<'EOF'\nprint('ok')\n EOF\n",
    "if true <<'EOF'\nEOF\n",
    "for value <<'EOF'\nEOF\n",
    "python3 <<'EOF'\nprint('ok')\nEOF\n\n",
    "python3 <<'EOF'\n\x00\nEOF\n",
    "cat <<'EOF'\n\x03sleep 10\nEOF\n",
    "cat <<'EOF'\n\x04\nEOF\n",
    "cat <<'EOF'\n\x1b[200~EOF\nEOF\n",
    "cat <<'EOF'\nEOX\x7fF\nEOF\n",
    "python3 <<'EOF'\r\nprint('ok')\r\nEOF\r\n",
])
def test_standalone_heredoc_rejects_ambiguous_or_multiple_commands(text):
    assert not standalone_heredoc(text)

class LocalEnvironment:
    session_id='isolated-harness-test'
    def __init__(self,socket):self.socket=socket
    async def exec(self,command,user=None,timeout_sec=10,**kw):
        command=command.replace('tmux ',f'tmux -L {self.socket} ')
        process=await asyncio.create_subprocess_shell(command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        out,err=await asyncio.wait_for(process.communicate(),timeout_sec)
        return SimpleNamespace(return_code=process.returncode,stdout=out.decode(),stderr=err.decode())


class DockerEnvironment:
    session_id='isolated-recorded-docker-harness-test'
    def __init__(self,container):self.container=container
    async def exec(self,command,user=None,timeout_sec=10,**kw):
        p=await asyncio.create_subprocess_exec('docker','exec',self.container,'bash','-c',command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        out,err=await asyncio.wait_for(p.communicate(),timeout_sec)
        return SimpleNamespace(return_code=p.returncode,stdout=out.decode(),stderr=err.decode())
    async def upload_file(self,source_path,target_path):
        subprocess.run(['docker','cp',str(source_path),self.container+':'+target_path],check=True,capture_output=True)


def test_real_tmux(tmp_path):
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
    socket='kb-test-'+uuid.uuid4().hex
    install(tmp_path/'audit.jsonl',recorder_transport=os.getenv('KVMEM_HARNESS_DUPLEX')=='1')
    parser=TerminusJSONPlainParser();warnings=[]
    parsed,error=parser._parse_commands([{'keystrokes':'C-c','duration':.1},{'keystrokes':'pwd\n','duration':1}],warnings)
    assert not error and not warnings
    for v in (-1,float('nan'),float('inf')):
        assert parser._parse_commands([{'keystrokes':'','duration':v}],[])[1]
    assert 'literal NUL' in parser._parse_commands([{'keystrokes':'echo \x00\n','duration':1}],[])[1]
    assert not parser._parse_commands([{'keystrokes':"printf '\\0'\n",'duration':1}],[])[1]
    image=os.getenv('KVMEM_HARNESS_TEST_IMAGE');container=None
    if image:
        container=subprocess.check_output(['docker','run','-d','--network','none','--cpus','1','--memory','512m','--entrypoint','sleep','-v',str(tmp_path)+':'+str(tmp_path),image,'180'],text=True).strip()
    async def exercise():
        environment=DockerEnvironment(container) if container else LocalEnvironment(socket)
        session=TmuxSession('test',environment,tmp_path/'pane.log',None,tmp_path/'recording.cast' if container else None)
        await session.start()
        agent=Terminus2.__new__(Terminus2);agent._limit_output_length=lambda x:x
        async def execute(text,duration):
            return await agent._execute_commands([SimpleNamespace(keystrokes=text,duration_sec=duration)],session)
        if os.getenv('KVMEM_HARNESS_DUPLEX')=='1':
            import hashlib
            # Long real multiline writes must survive the recording PTY,
            # including non-ASCII bytes and repeated full-size paste buffers.
            for index, lines in enumerate((180,700,4000)):
                body=''.join(f'{n:05d} 中文 λ '+('abcdef0123456789'*8)+'\n' for n in range(lines))
                command=f"cat > /tmp/duplex-{index}.txt <<'KB_END'\n"+body+'KB_END\n'
                _,out=await execute(command,2)
                result=await environment.exec(f'sha256sum /tmp/duplex-{index}.txt')
                # The 584KB case takes more than two seconds to echo/readline
                # under one CPU. Observe completion without resending input.
                deadline=time.monotonic()+20
                while result.return_code and time.monotonic()<deadline:
                    await asyncio.sleep(.2)
                    result=await environment.exec(f'sha256sum /tmp/duplex-{index}.txt')
                assert result.return_code==0 and result.stdout.startswith(hashlib.sha256(body.encode()).hexdigest())
            # Resize must remain deliverable while recording is active.
            await environment.exec('tmux resize-window -t test -x 100 -y 35')
            await execute('',.5)
        start=time.monotonic();_,out=await execute('printf "finished\\n"\n',3)
        assert time.monotonic()-start<2 and 'completion confirmed, exit=0' in out
        _,out=await execute('false\n',3)
        assert 'exit=1' in out
        # A busy Docker daemon can time out a read-only status probe. Preserve
        # the command attempt and report uncertainty; never abort the trial.
        old_exec=environment.exec
        async def timeout_probes(command,**kwargs):
            if command.startswith('cat /tmp/.kb-terminal-'):
                raise RuntimeError('Command timed out after 5 seconds')
            return await old_exec(command,**kwargs)
        environment.exec=timeout_probes
        try:
            _,out=await execute('printf "probe-recovery\\n"\n',.1)
            assert 'completion unconfirmed' in out and 'observation unavailable' in out
        finally:
            environment.exec=old_exec
        _,out=await execute('printf "probe-alive\\n"\n',.2)
        assert 'probe-alive' in out
        start=time.monotonic();await execute('',.6)
        assert time.monotonic()-start>=.6
        _,out=await execute('sleep 15\n',.3)
        assert 'completion unconfirmed' in out
        _,out=await execute('C-c\n',.1)
        assert 'completion confirmed' in out
        _,out=await execute('printf "alive\\n"\n',3)
        assert 'alive' in out and 'completion confirmed' in out
        start=time.monotonic();await execute('echo first\nsleep .7\necho last\n',1)
        assert time.monotonic()-start>=1
        # A displayed prompt is not completion. Only Bash's private hook after
        # this entire command returns may acknowledge the quoted heredoc.
        start=time.monotonic()
        _,out=await execute("python3 <<'KB_DOC'\nimport time\nprint('root@fake:/app# ', flush=True)\ntime.sleep(.8)\nraise SystemExit(7)\nKB_DOC\n",4)
        elapsed=time.monotonic()-start
        assert .8 <= elapsed < 3.5 and 'completion confirmed, exit=7' in out
        # A suffix is another top-level command, so keep the observation time.
        start=time.monotonic()
        _,out=await execute("cat <<'KB_DOC'\none\nKB_DOC\nsleep .3\nprintf 'suffix-done\\n'\n",1)
        assert time.monotonic()-start>=1 and 'completion unconfirmed' in out
        # An incomplete delimiter must not be mistaken for an idle shell even
        # when Bash still has its previous prompt's sequence marker on disk.
        start=time.monotonic()
        _,out=await execute("cat <<'KB_DOC'\npartial\n",.4)
        assert time.monotonic()-start>=.4 and 'completion unconfirmed' in out
        _,out=await execute("KB_DOC\n",.5)
        assert 'partial' in out
        _,out=await execute('bash --noprofile --norc\n',.2)
        assert 'completion unconfirmed' in out
        await execute('exit\n',.3)
        # Hook state preserved; literal C-c in a command is not normalized.
        _,out=await execute('echo C-c\n',1)
        assert 'C-c' in out
        _,out=await execute('python3 -c \"import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(5)\"\n',.3)
        _,out=await agent._execute_commands([SimpleNamespace(keystrokes='C-c',duration_sec=.1),SimpleNamespace(keystrokes='echo SHOULD_NOT_SEND\n',duration_sec=1)],session)
        assert 'following commands were NOT sent' in out
        await asyncio.sleep(3)
        _,out=await execute('declare -f curl wget\n',1)
        assert '--speed-time 120' in out and '--tries=3' in out
        await session.environment.exec('tmux kill-session -t test')
    try:asyncio.run(exercise())
    finally:
        if container:subprocess.run(['docker','rm','-f',container],capture_output=True,check=True)
        else:subprocess.run(['tmux','-L',socket,'kill-server'],capture_output=True)
    events=[json.loads(l) for l in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert any(e.get('original')=='C-c\n' and e['sent']==['C-c'] for e in events)


def test_source_preserves_prompt_hooks_and_explicit_network_options():
    from scripts.terminalbench_harness import hook_source
    source=hook_source('/tmp/kb-test-state')
    script='''PROMPT_COMMAND=('printf old-hook')
'''+source+'''
curl() { command printf '%s\\n' --connect-timeout 15 --speed-limit 1 --speed-time 120 --retry 2 --retry-delay 2 --retry-max-time 300 "$@"; }
printf '%s\\n' "${PROMPT_COMMAND[@]}"
curl --connect-timeout 7 'https://example.invalid/a b'
'''
    result=subprocess.run(['bash','-c',script],check=True,capture_output=True,text=True).stdout
    assert '__kb_prompt\nprintf old-hook\n' in result
    assert result.index('15')<result.index('7') and 'https://example.invalid/a b' in result


@pytest.mark.parametrize('error',[TimeoutError(),RuntimeError('Command timed out after 5 seconds')])
def test_probe_timeout_is_unknown_not_completion(error):
    from scripts.terminalbench_harness import probe
    async def execute(**kwargs):raise error
    session=SimpleNamespace(_kb_path='/tmp/probe',_session_name='test',_user=None,
                            environment=SimpleNamespace(exec=execute))
    result=asyncio.run(probe(session))
    assert result['seq'] is None and not result['shell'] and not result['dead']
    assert 'observation_error' in result


@pytest.mark.parametrize('error',[RuntimeError('container missing'),asyncio.CancelledError()])
def test_probe_preserves_other_errors_and_cancellation(error):
    from scripts.terminalbench_harness import probe
    async def execute(**kwargs):raise error
    session=SimpleNamespace(_kb_path='/tmp/probe',_session_name='test',_user=None,
                            environment=SimpleNamespace(exec=execute))
    with pytest.raises(type(error)):asyncio.run(probe(session))


@pytest.mark.parametrize('failure',[None,TimeoutError('qualification deadline')])
def test_recorder_setup_keeps_merged_output_exit_code_and_timeout_evidence(failure):
    from scripts.terminalbench_harness import prepare_recorder
    events=[];calls=[]
    async def execute(command,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            return SimpleNamespace(return_code=0,stdout='#!/usr/bin/python3\n',stderr=None)
        if failure is not None:raise failure
        return SimpleNamespace(return_code=1,stdout='Traceback: unsupported recorder version',stderr=None)
    async def upload(**kwargs):pass
    session=SimpleNamespace(_user=None,environment=SimpleNamespace(exec=execute,upload_file=upload))
    with pytest.raises(RuntimeError,match='deadline' if failure else 'unsupported recorder version'):
        asyncio.run(prepare_recorder(session,lambda **row:events.append(row)))
    assert all(row['timeout_sec']==30 for row in calls)
    assert events[-1]['stage']=='uploaded_adapter' and events[-1]['timeout_sec']==30
    if failure:
        assert 'qualification deadline' in events[-1]['error']
    else:
        assert events[-1]['return_code']==1 and events[-1]['stderr'] is None
        assert 'unsupported recorder version' in events[-1]['stdout']


def test_recorder_setup_quotes_interpreter_without_reinterpreting_shell_text():
    import shlex
    from scripts.terminalbench_harness import prepare_recorder
    commands = []; events = []; uploaded = []
    interpreter = ['/opt/Python runtime/bin/python3', '-I', 'literal$(do-not-run)`literal`']
    async def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=0, stderr=None,
            stdout='#!'+shlex.join(interpreter)+'\n' if len(commands)==1 else 'asciinema 2.0.2\n')
    async def upload(**kwargs): uploaded.append(kwargs)
    session = SimpleNamespace(_user=None, environment=SimpleNamespace(
        session_id='qualification', exec=execute, upload_file=upload))
    command = asyncio.run(prepare_recorder(session, lambda **row: events.append(row)))
    assert shlex.split(command) == [*interpreter, uploaded[0]['target_path']]
    assert commands[1].endswith(' && '+command+' --version')
    assert events[-1]['event'] == 'recorder_prepared' and events[-1]['interpreter'] == interpreter
