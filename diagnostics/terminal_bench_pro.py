"""Harbor 0.24 external adapter: local Docker/DSH, remote C1 KVMem model.

The task instruction and verifier are owned by Harbor. This adapter does not
inspect reference solutions or tests. Docker runs without any GPU devices.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

import httpx
from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions


class Options(AgentOptions):
    gc_mode: str
    source_root: str
    dsh_root: str
    tokenizer: str
    python: str


class DshGcAgent(BaseAgent):
    options_model = Options

    @staticmethod
    def name():
        return 'dsh-kvmem-gc'

    def version(self):
        return '0.1.0-rc.7-tbpro-1'

    async def setup(self, environment):
        if self.options.gc_mode not in ('off','delete'):
            raise ValueError('gc_mode must match the remote native off/delete profile')
        if environment.task_env_config.gpus:
            raise ValueError('Local task containers must not request a GPU')
        result=await environment.exec('pwd')
        self.workspace=result.stdout.strip()
        if not self.workspace.startswith('/') or self.workspace=='/':
            raise ValueError('A real task WORKDIR is required')
        result=await environment._run_docker_compose_command(['ps','-q','main'])
        self.container=result.stdout.strip()
        if not re.fullmatch('[a-f0-9]{12,64}',self.container):
            raise ValueError('Cannot identify the owned Harbor Docker task container')

    async def run(self, instruction, environment, context):
        source=Path(self.options.source_root).resolve()
        dsh=Path(self.options.dsh_root).resolve()
        bridge_dir=self.logs_dir/'bridge'
        task_file=self.logs_dir/'instruction.txt'
        task_file.write_text(instruction)
        await environment.upload_file(task_file,'/tmp/kvmem-tbpro-instruction.txt')
        run_root='/logs/agent/dsh'
        prepare=['python',str(source/'diagnostics/dsh_cloud/prepare.py'),
            '--run-root',run_root,'--harness',str(dsh),'--context-window','262144',
            '--select-budget','36864','--gen-budget','16384','--max-output-tokens','16384',
            '--max-output-segments','1','--stream-idle-timeout-ms','1800000',
            '--turn-timeout-ms','3600000','--case-name','task',
            '--task-file','/tmp/kvmem-tbpro-instruction.txt']
        result=await environment.exec(shlex.join(prepare),timeout_sec=60)
        (self.logs_dir/'prepare.txt').write_text(result.stdout+'\n'+(result.stderr or ''))
        if result.return_code:
            raise RuntimeError('DSH task preparation failed')
        # The official task files and WORKDIR stay in their original locations.
        program='from pathlib import Path; p=Path("/logs/agent/dsh/task/workspace"); p.rmdir(); p.symlink_to('+repr(self.workspace)+', target_is_directory=True)'
        result=await environment.exec(shlex.join(['python','-c',program]),timeout_sec=20)
        if result.return_code:
            raise RuntimeError('Cannot bind DSH to the official task WORKDIR')
        close=['python',str(source/'diagnostics/dsh_cloud/submit.py'),run_root+'/task','--close']
        result=await environment.exec(shlex.join(close),timeout_sec=20)
        if result.return_code:
            raise RuntimeError('Cannot queue the one-task DSH driver exit')
        owner='tbpro-'+str(self.context_id).replace('-','')+'-'+self.options.gc_mode
        command=[self.options.python,'-B','-u','-m','scripts.dsh_bridge',
            '--tokenizer-protocol','--disable-thinking','--reasoning-effort','low',
            '--pin-system','--select-budget','36864','--owner-namespace',owner,
            '--tokenizer',self.options.tokenizer,'--backend','http://127.0.0.1:8186',
            '--model','kvmem-gc-qwen38','--output',str(bridge_dir),'--port','18387',
            '--gen-budget','16384','--max-model-len','262144','--max-sessions','1',
            '--max-inflight','1','--max-output-segments','1','--require-query-provenance',
            '--single-query-task',
            '--temperature','0','--seed','20261024',
            '--artifact-gc-root',self.workspace,'--artifact-gc-container',self.container,
            '--artifact-gc-mode',self.options.gc_mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',
            PYTHONPATH=str(source/'src')+os.pathsep+str(source),PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
        env.pop('VLLM_PLUGINS',None)
        with socket.socket() as port:
            port.bind(('127.0.0.1',18387))
        log=(self.logs_dir/'bridge.log').open('x')
        child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        try:
            async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
                for _ in range(120):
                    if child.poll() is not None:
                        raise RuntimeError('Local bridge exited during startup; see bridge.log')
                    try:
                        response=await client.get('http://127.0.0.1:18387/health')
                        if response.status_code==200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(.5)
                else:
                    raise RuntimeError('Local bridge did not become ready')
            launch=['python',str(source/'diagnostics/dsh_cloud/launch.py'),
                '--run-root',run_root,'--case','task','--execute']
            result=await environment.exec(shlex.join(launch),timeout_sec=3600,
                env={'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void','PYTHONDONTWRITEBYTECODE':'1'})
            (self.logs_dir/'driver.txt').write_text(result.stdout+'\n'+(result.stderr or ''))
            context.metadata=dict(gc_mode=self.options.gc_mode,driver_exit=result.return_code,
                task_seconds=time.monotonic()-started,workspace=self.workspace,
                native_k=36864,native_g=16384,logical_context=262144,
                max_model_requests=1,local_gpu_devices=False,
                task_instruction_unchanged=True)
            usages=[]
            for path in bridge_dir.glob('*/*/native.json'):
                usages.append(json.loads(path.read_text())['usage'])
            context.n_input_tokens=sum(x['prompt_tokens'] for x in usages)
            context.n_output_tokens=sum(x['completion_tokens'] for x in usages)
        finally:
            try:
                # Freeze live-scope GC evidence before ordinary owner cleanup frees.
                async with httpx.AsyncClient(trust_env=False,timeout=180) as client:
                    snapshot=await client.post('http://127.0.0.1:8186/collective_rpc',json={
                        'method':'dsh_artifact_gc_snapshot'})
                    snapshot.raise_for_status()
                    (self.logs_dir/'gc-before-cleanup.json').write_text(json.dumps(snapshot.json(),indent=2))
                    releases=[]
                    for directory in bridge_dir.iterdir() if bridge_dir.exists() else ():
                        if not directory.is_dir() or not directory.name.startswith(owner+'-'):
                            continue
                        payload=dict(model='kvmem-gc-qwen38',prompt=[1],session_id=directory.name,
                            cache_salt=directory.name,max_tokens=1,temperature=0,kvmem_reselect='off',
                            kvmem_mandatory_parts='none',kvmem_mandatory_spans=[],
                            kv_transfer_params={'kvmem_release_session':True})
                        response=await client.post('http://127.0.0.1:8186/v1/completions',json=payload)
                        releases.append(dict(owner=directory.name,status=response.status_code,response=response.json()))
                    (self.logs_dir/'owner-cleanup.json').write_text(json.dumps(releases,indent=2))
            finally:
                child.terminate()
                try:
                    await asyncio.to_thread(child.wait,15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    await asyncio.to_thread(child.wait)
                log.close()
