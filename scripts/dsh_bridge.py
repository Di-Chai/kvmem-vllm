"""Local DSH chat transport with exact accepted-token history and opt-in control-frame pins.

The DSH OpenAI-compatible provider sends its stable session as prompt_cache_key
using cacheRetention=long and supportsLongCacheRetention=true. Native prefix
caching remains disabled. The key only identifies an owned KVMem session.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
import codecs
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory, backend_request, canonical
from scripts.dsh_query_epoch import apply_query_provenance


def to_anthropic(body):
    """Preserve message/tool content while adapting the two wire vocabularies."""
    messages=[];systems=[]
    for message in body['messages']:
        role=message['role'];content=message.get('content') or ''
        if role in ('system','developer'):
            if not isinstance(content,str):
                content='\n'.join(p['text'] for p in content if p['type']=='text')
            systems.append(content);continue
        if role=='tool':
            messages.append(dict(role='user',content=[dict(type='tool_result',
                tool_use_id=message['tool_call_id'],content=content)]));continue
        blocks=[dict(type='text',text=content)] if isinstance(content,str) and content else []
        if isinstance(content,list):
            for p in content:
                if p['type']!='text':raise ValueError('This local model route accepts text only')
                blocks.append(dict(type='text',text=p['text']))
        if role=='assistant':
            reasoning=message.get('reasoning_content') or message.get('reasoning')
            if reasoning:blocks.insert(0,dict(type='thinking',thinking=reasoning,signature='local'))
            for call in message.get('tool_calls',[]):
                f=call['function'];blocks.append(dict(type='tool_use',id=call['id'],
                    name=f['name'],input=json.loads(f['arguments'])))
        if role not in ('assistant','user'):raise ValueError('Unsupported chat role')
        messages.append(dict(role=role,content=blocks))
    tools=[]
    for tool in body.get('tools',[]):
        f=tool['function'];tools.append(dict(name=f['name'],description=f.get('description',''),
            input_schema=f.get('parameters',dict(type='object',properties={}))))
    return dict(model=body['model'],system='\n\n'.join(systems),messages=messages,tools=tools)


def to_openai(response):
    message=dict(role='assistant',content=''.join(b['text'] for b in response['content'] if b['type']=='text') or None)
    thinking=''.join(b['thinking'] for b in response['content'] if b['type']=='thinking')
    if thinking:message['reasoning_content']=thinking
    calls=[dict(id=b['id'],type='function',function=dict(name=b['name'],arguments=canonical(b['input'])))
           for b in response['content'] if b['type']=='tool_use']
    if calls:message['tool_calls']=calls
    stop=response['stop_reason']
    usage=response['usage'];prompt=usage['input_tokens'];output=usage['output_tokens']
    return dict(id='chatcmpl-'+uuid.uuid4().hex,object='chat.completion',created=int(time.time()),
        model=response['model'],choices=[dict(index=0,message=message,
            finish_reason='tool_calls' if calls else 'length' if stop=='max_tokens' else 'stop')],
        usage=dict(prompt_tokens=prompt,completion_tokens=output,total_tokens=prompt+output))


def session_owner(key,payload,namespace='dsh'):
    # DSH title generation uses the conversation's session key too. Distinct
    # system/tool frames need distinct owners so an auxiliary call cannot
    # replace the agent's accepted-token ledger or graph checkpoints.
    frame=canonical(dict(system=payload.get('system',''),tools=payload.get('tools',[])))
    return namespace+'-'+hashlib.sha256((key+'\0'+frame).encode()).hexdigest()[:40]


class AcceptedTextDecoder:
    """Qwen ByteLevel decoding, retaining UTF-8 bytes across native requests."""
    def __init__(self,tokenizer):
        decoder=json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
        if decoder.get('type')!='ByteLevel':raise ValueError('Continuation requires a verified ByteLevel tokenizer')
        values=list(range(33,127))+list(range(161,173))+list(range(174,256))
        original=list(values);extra=0
        for byte in range(256):
            if byte not in values:original.append(byte);values.append(256+extra);extra+=1
        inverse=dict(zip(map(chr,values),original));added=set(tokenizer.get_added_vocab().values())
        @lru_cache(maxsize=8192)
        def token_bytes(token):
            piece=tokenizer.convert_ids_to_tokens(token)
            return piece.encode() if token in added else bytes(inverse[c] for c in piece)
        self.token_bytes=token_bytes;self.decoder=codecs.getincrementaldecoder('utf-8')('replace')

    def feed(self,ids):
        return self.decoder.decode(b''.join(self.token_bytes(i) for i in ids),final=False)

    def finish(self):return self.decoder.decode(b'',final=True)


class QwenStream:
    """Emit only irrevocable prefixes of the frozen full-response parser.

    The renderer opens thinking, or closes an empty frame in off mode. Hold
    split control tags and the one
    trailing reasoning newline that the parser removes at </think>. Tool
    arguments stay buffered for the authoritative schema-aware parser, but a
    canonical function header can report real model activity before its body.
    A malformed announced call fails closed instead of executing partial JSON.
    """
    markers=('</think>','<tool_call>','<|im_end|>','<|endoftext|>')

    def __init__(self,protocol,tools):
        self.protocol=protocol;self.tools=tools;self.raw=''
        self.sent={'reasoning_content':'','content':''};self.tool=None
        if getattr(protocol,'kwargs',{}).get('enable_thinking') is False:
            self.markers=('<think>',*self.markers)

    def _delta(self,message):
        delta={}
        for field,previous in self.sent.items():
            current=message.get(field) or ''
            if not current.startswith(previous):
                raise ValueError('Streamed content differs from the authoritative Qwen parser')
            if len(current)>len(previous):delta[field]=current[len(previous):]
            self.sent[field]=current
        return delta

    def feed(self,text):
        self.raw+=text
        safe=self.raw
        # Do not leak a tag which can become a complete control token next.
        held=max((n for marker in self.markers for n in range(1,len(marker))
                  if safe.endswith(marker[:n])),default=0)
        if held:safe=safe[:-held]
        close=safe.find('</think>')
        implicit=getattr(self.protocol,'implicit_tool_reasoning_end',False)
        first_tool=safe.find('<tool_call>') if implicit else -1
        if first_tool>=0 and (close<0 or first_tool<close):
            close=-1;tool_start=first_tool
        else:
            tool_start=safe.find('<tool_call>',close+len('</think>')) if close>=0 else -1
        parse_text=safe[:tool_start] if tool_start>=0 else safe
        for ending in ('<|endoftext|>','<|im_end|>'):
            parse_text=parse_text.removesuffix(ending)
        parsed=self.protocol(op='parse',text=parse_text,tools=self.tools)
        reasoning=parsed['reasoning']
        if close<0:reasoning=reasoning.removesuffix('\n')
        delta=self._delta(dict(reasoning_content=reasoning,content=parsed['content']))
        # Once visible reasoning/text has established TTFT, keep tool frames
        # provisional until full parsing. Otherwise a length-truncated write
        # would leave an announced call which cannot be retracted from SSE.
        if tool_start>=0 and self.tool is None and not any(self.sent.values()):
            header=re.match(r'<tool_call>\s*<function=([^>\n]+)>',safe[tool_start:])
            if header and any(t.get('function',t).get('name')==header[1] for t in self.tools):
                self.tool=dict(id='toolu_'+uuid.uuid4().hex,name=header[1])
                delta['tool_calls']=[dict(index=0,id=self.tool['id'],type='function',
                    function=dict(name=self.tool['name'],arguments=''))]
        return delta

    def finish(self,response):
        # Use the same IDs in the wire response and the accepted-token ledger.
        calls=[b for b in response['content'] if b['type']=='tool_use']
        if self.tool is not None:
            if not calls or calls[0]['name']!=self.tool['name']:
                if response['stop_reason']=='max_tokens':
                    raise ValueError(f"Output limit ({response['usage']['output_tokens']} tokens) reached before completing tool {self.tool['name']}; no tool was executed. Split large writes into smaller calls.")
                raise ValueError('Incomplete or invalid streamed Qwen tool call')
            calls[0]['id']=self.tool['id']
        output=to_openai(response);message=output['choices'][0]['message']
        delta=self._delta(message)
        if message.get('tool_calls'):
            delta['tool_calls']=[]
            for i,call in enumerate(message['tool_calls']):
                if i==0 and self.tool is not None:
                    delta['tool_calls'].append(dict(index=i,function=dict(arguments=call['function']['arguments'])))
                else:delta['tool_calls'].append(dict(index=i,**call))
        return output,delta


async def native_events(result):
    """Read SSE records without depending on HTTP/TCP chunk boundaries."""
    fields=[]
    async for line in result.aiter_lines():
        if line=='':
            if fields:
                data='\n'.join(fields);fields=[]
                if data=='[DONE]':return
                event=json.loads(data)
                if event.get('error'):raise ValueError('Model stream failed: '+str(event['error'])[:2000])
                yield event
        elif line.startswith('data:'):fields.append(line[5:].lstrip(' '))
    raise ValueError('Model stream ended without [DONE]')


def apply_mandatory_policy(prepared, protocol):
    """Keep only the explicitly enabled audited control-frame pin."""
    tokens, plan, rendered, _, _, _, locations = prepared
    plan['kvmem_mandatory_parts'] = 'none'
    if not getattr(protocol, 'pin_system', False):
        plan['kvmem_mandatory_spans'] = []
        return
    control_end = rendered.get('control_end')
    end = locations.get((-1, control_end))
    expected = [dict(begin=0, end=end, kind='system')]
    if (type(control_end) is not int or control_end <= 0 or
            locations.get((-1, 0)) != 0 or type(end) is not int or not 0 < end <= len(tokens) or
            plan.get('kvmem_mandatory_spans') != expected):
        raise ValueError('System pin lacks exact accepted-prefix token coordinates')
    # Encode only the audited immutable frame, never accepted generated history.
    frame = rendered['prompt'].encode()[:control_end].decode()
    if protocol.tokenizer.encode(frame, add_special_tokens=False) != tokens[:end]:
        raise ValueError('System pin differs from the actual accepted control-frame IDs')


def validate_system_pin_budget(data, protocol):
    """Conservative bridge admission for the explicitly declared backend profile.

    Native `system` spans can be capacity-fitted. This guard therefore leaves
    the full declared replay budget and two alignment/speculation blocks free,
    rather than relying on a pin's priority to prevent fitting. Runtime profile
    equality and actual worker coverage must still be qualified separately.
    """
    if not getattr(protocol, 'pin_system', False):
        return None
    from vllm_kvmem.request_plan import FIELDS, RequestPlan
    cfg = protocol.system_pin_policy
    block = cfg['block_tokens']
    plan = RequestPlan.parse({k:v for k,v in data.items() if k in FIELDS},
        len(data['prompt']), block, cfg['gen_budget'])
    if (data.get('max_tokens', 0) <= 0 or data['max_tokens'] > cfg['gen_budget'] or
            plan.probe_tokens or plan.pressure_probe_tokens):
        raise ValueError('System-pin admission requires the declared generation and replay profile')
    allowed_reselect=('auto','force') if getattr(protocol,'artifact_gc_force_allowed',False) else ('auto',)
    maintenance=(getattr(protocol,'artifact_gc_maintenance_allowed',False) and
        plan.query_mode=='raw' and plan.reselect in ('force','auto') and plan.query is not None and
        plan.query[0]==(plan.query[1]-1)//block*block and
        plan.query_id=='artifact-maintenance-'+str(plan.query[1]) and
        (plan.reselect=='auto' or plan.query[1]==len(data['prompt'])))
    single_task=(getattr(protocol,'single_query_task',False) and plan.query_mode=='agent' and
        plan.query is None and plan.reselect=='auto' and bool(plan.query_id))
    if ((plan.query_mode != 'agent' and not maintenance) or plan.reselect not in allowed_reselect or
            (plan.query is None and not single_task) or not plan.query_id):
        raise ValueError('System-pin admission requires a verified agent Query epoch')
    replay = plan.replay_end - plan.replay_begin if plan.query is not None else 0
    if replay > cfg['query_replay_budget']:
        raise ValueError('System-pin admission: complete Query exceeds the declared replay budget')
    if len(plan.mandatory) != 1 or plan.mandatory[0][0] != 0 or plan.mandatory[0][2] != 'system':
        raise ValueError('System-pin admission requires exactly one complete system/tool frame')
    begin, end, _ = plan.mandatory[0]
    pinned = set(range(begin // block, (end + block - 1) // block))
    pinned.update(range(cfg['sink_blocks']))
    reserve = (cfg['query_replay_budget'] // block + cfg['recent_blocks'] +
               cfg['rounding_reserve_blocks'])
    required = (len(pinned) + reserve) * block
    if required > cfg['select_budget']:
        raise ValueError('Insufficient selected capacity for complete system/tool frame: '
            f'requires at least {required} tokens including replay/sink/recent/alignment reserve, '
            f'declared K={cfg["select_budget"]}; no model request was sent')
    return dict(system_tokens=end, pinned_blocks=len(pinned), reserved_blocks=reserve,
                required_select_tokens=required, declared_select_budget=cfg['select_budget'],
                actual_query_replay_tokens=replay)


def restore_trajectory(planner,owner,directories):
    """Rebuild only completed turns using their actual accepted IDs.

    This permits a bridge update without discarding active conversations or
    re-encoding past generated code. A newer directory wins for a given turn;
    an incomplete attempt has no accepted.json and is never restored.
    """
    turns={}
    for directory in directories:
        if (directory/owner).exists() and hasattr(getattr(planner,'protocol',None),'provenance'):
            manifest=directory/'protocol.json'
            if not manifest.is_file() or json.loads(manifest.read_text())!=planner.protocol.provenance:
                raise ValueError('Accepted history protocol differs from the current tokenizer/template')
        for path in (directory/owner).glob('[0-9][0-9][0-9][0-9][0-9]'):
            if (path/'accepted.json').exists() and (path/'response.json').exists():
                turns[int(path.name)]=path
    state=Trajectory()
    for turn,path in sorted(turns.items()):
        if turn!=state.turns+1:raise ValueError('Accepted history has a missing turn; refusing an inexact restore')
        body=json.loads((path/'request.json').read_text());payload=to_anthropic(body)
        prepared=planner.prepare(payload,state);saved=json.loads((path/'plan.json').read_text())
        apply_query_provenance(body, prepared, required=getattr(getattr(planner, 'protocol', None), 'pin_system', False))
        if len(prepared[0])!=saved['prompt_tokens'] or prepared[5]!=saved['exact_output_prefix_reused']:
            raise ValueError('Accepted history does not match its recorded prompt prefix')
        apply_mandatory_policy(prepared, planner.protocol)
        if getattr(getattr(planner, 'protocol', None), 'pin_system', False) and (
                saved.get('kvmem_mandatory_spans') != prepared[1]['kvmem_mandatory_spans'] or
                saved.get('kvmem_mandatory_parts') != 'none'):
            raise ValueError('Accepted history differs from the explicit system-pin policy')
        wire=json.loads((path/'response.json').read_text())
        content=to_anthropic(dict(model=body['model'],messages=[wire['choices'][0]['message']]))['messages'][0]['content']
        ids=json.loads((path/'accepted.json').read_text())['token_ids']
        if len(ids)!=wire['usage']['completion_tokens']:raise ValueError('Accepted history has inconsistent token usage')
        planner.commit(state,prepared,payload,dict(content=content),ids)
    return state


def apply_sampling_overrides(data,args,turn):
    """Apply an explicit shared experiment preset; preserve ordinary callers."""
    if getattr(args,'seed_per_turn',False) and getattr(args,'seed',None) is None:
        raise ValueError('Per-turn seed scheduling requires an explicit base seed')
    for key in ('temperature','seed','top_p','top_k','min_p','presence_penalty','repetition_penalty'):
        value=getattr(args,key,None)
        if value is not None:
            data[key]=value+turn if key=='seed' and getattr(args,'seed_per_turn',False) else value


def create_app(args,tokenizer):
    max_inflight=getattr(args,'max_inflight',0)
    if type(max_inflight) is not int or max_inflight<0:
        raise ValueError('max_inflight must be a nonnegative integer')
    pin_system=getattr(args,'pin_system',False)
    if type(pin_system) is not bool:
        raise ValueError('pin_system must be boolean')
    if pin_system and not getattr(args,'tokenizer_protocol',False):
        raise ValueError('--pin-system requires --tokenizer-protocol')
    if pin_system and not getattr(args,'require_query_provenance',False):
        raise ValueError('--pin-system requires --require-query-provenance')
    disable_thinking=getattr(args,'disable_thinking',False)
    if type(disable_thinking) is not bool:
        raise ValueError('disable_thinking must be boolean')
    if disable_thinking and not getattr(args,'tokenizer_protocol',False):
        raise ValueError('--disable-thinking requires --tokenizer-protocol')
    admission=asyncio.BoundedSemaphore(max_inflight) if max_inflight else None
    if getattr(args,'tokenizer_protocol',False):
        from scripts.dsh_model_protocol import TokenizerProtocol
        protocol=TokenizerProtocol(tokenizer,args.tokenizer,
            reasoning_effort=getattr(args,'reasoning_effort','low'),
            enable_thinking=not disable_thinking, pin_system=pin_system,
            select_budget=getattr(args,'select_budget',None), gen_budget=args.gen_budget)
    else:
        protocol=Protocol(args.protocol)
    artifact_root=getattr(args,'artifact_gc_root',None)
    artifact_mode=getattr(args,'artifact_gc_mode',None)
    artifact_container=getattr(args,'artifact_gc_container',None)
    if (artifact_mode is not None or artifact_container is not None) and artifact_root is None:
        raise ValueError('Artifact container/mode requires the explicit artifact experiment')
    if artifact_root is not None and (args.backend != 'http://127.0.0.1:8186' or not pin_system or max_inflight != 1):
        raise ValueError('Artifact GC experiment requires the owned loopback C1 pinned-system backend')
    protocol.artifact_gc_force_allowed=artifact_root is not None
    protocol.artifact_gc_maintenance_allowed=artifact_root is not None and artifact_mode=='delete'
    protocol.single_query_task=getattr(args,'single_query_task',False)
    parser_url=getattr(args,'tool_parser_url',None)
    if parser_url is not None:
        if parser_url!='http://127.0.0.1:18389' or not disable_thinking:
            raise ValueError('Remote official parser requires the owned loopback no-thinking route')
        with httpx.Client(trust_env=False,timeout=180) as parser_client:
            check=parser_client.get(parser_url+'/health')
            check.raise_for_status()
            manifest=check.json()
            if (not manifest.get('cpu_only') or manifest.get('provenance',{}).get('files')!=protocol.provenance['files'] or
                    manifest.get('provenance',{}).get('chat_template_kwargs')!=protocol.kwargs or
                    manifest.get('provenance',{}).get('tool_parser')!=protocol.provenance['tool_parser']):
                raise ValueError('Remote parser differs from the audited local tokenizer assets')
        def parse_remote(text,tools):
            with httpx.Client(trust_env=False,timeout=180) as parser_client:
                parsed=parser_client.post(parser_url+'/parse',json=dict(text=text,tools=tools))
                parsed.raise_for_status()
                return parsed.json()
        protocol.parse=parse_remote
    artifact_decoder=AcceptedTextDecoder(tokenizer) if artifact_root is not None else None
    artifact_armed={}
    artifact_pending={}
    app=FastAPI();planner=Planner(protocol,tokenizer,args.gen_budget)
    client=httpx.AsyncClient(trust_env=False,timeout=3600)
    owners={};restored=set();args.output.mkdir(parents=True,exist_ok=True)
    if hasattr(protocol,'provenance'):
        manifest=args.output/'protocol.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=protocol.provenance:
                raise ValueError('Output directory belongs to another tokenizer/template protocol')
        elif next(args.output.glob('*/[0-9][0-9][0-9][0-9][0-9]/accepted.json'),None) is not None:
            raise ValueError('Existing accepted history has no tokenizer/template protocol manifest')
        else:
            manifest.write_text(json.dumps(protocol.provenance,indent=2)+'\n')
    def log(value):
        with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    @app.get('/health')
    async def health():
        result=await client.get(args.backend+'/health')
        return JSONResponse(dict(ok=result.status_code==200),status_code=result.status_code)
    @app.get('/v1/models')
    async def models():
        return dict(object='list',data=[dict(id=args.model,object='model',owned_by='local')])
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        body=await request.json()
        try:
            key=body.get('prompt_cache_key') or request.headers.get('x-session-id')
            if not isinstance(key,str) or not key or len(key)>1024:
                raise ValueError('A stable DSH prompt_cache_key or X-Session-ID is required')
            if body.get('model')!=args.model:raise ValueError('Unknown local model')
            limit=body.get('max_tokens',body.get('max_completion_tokens',args.gen_budget))
            segments=getattr(args,'max_output_segments',1)
            if type(limit) is not int or not 1<=limit<=args.gen_budget*segments:
                raise ValueError(f'Requested output must be between 1 and {args.gen_budget*segments} tokens')
            payload=to_anthropic(body)
            owner=session_owner(key,payload,getattr(args,'owner_namespace','dsh'))
            if owner not in owners:
                if len(owners)>=args.max_sessions:raise ValueError('Local DSH session capacity exhausted; restart/release before adding owners')
                owners[owner]=(asyncio.Lock(),Trajectory())
        except (ValueError,KeyError,TypeError) as error:
            return JSONResponse(dict(error=dict(message=str(error),type='invalid_request_error')),status_code=400)
        async def generate(emit=None):
            lock,state=owners[owner]
            async with lock, AsyncExitStack() as permit:
                # Owners also include title/auxiliary frames. A deployment-wide
                # permit spans the entire reply, including every continuation
                # segment and upstream stream cleanup on cancellation.
                if admission is not None:
                    await permit.enter_async_context(admission)
                if owner not in restored:
                    sources=[*getattr(args,'restore_from',[]),args.output]
                    state=await asyncio.to_thread(restore_trajectory,planner,owner,sources)
                    owners[owner]=(lock,state);restored.add(owner)
                    if state.turns:log(dict(at=time.time(),owner=owner,restored_turns=state.turns))
                started=time.time();prepared=await asyncio.to_thread(planner.prepare,payload,state)
                tokens,plan,rendered,*_=prepared
                query_source = apply_query_provenance(body, prepared,
                    required=getattr(args, 'require_query_provenance', False))
                apply_mandatory_policy(prepared, planner.protocol)
                remaining_context=args.max_model_len-len(tokens)
                if remaining_context<=0:
                    raise ValueError('Complete history leaves no output space within the logical context limit')
                # Preserve every input/accepted ID. The public output cap is a
                # maximum, so even the first native segment can use a smaller
                # positive remainder. Full tool parsing remains authoritative.
                output_limit=min(limit,remaining_context)
                data=backend_request(model=args.model,tokens=tokens,session=owner,
                    max_tokens=min(output_limit,args.gen_budget),plan=plan,backend_kind='kvmem')
                single_task=None
                if protocol.single_query_task:
                    # DSH may include framework context as additional wire-user
                    # frames. Its source proof identifies the human Query; the
                    # native role alone must not be used to count human inputs.
                    if (query_source.get('status')!='verified-user-source' or
                            not query_source.get('message_id') or plan.get('kvmem_query_span') is None):
                        raise ValueError('Single-task benchmark requires exactly its original verified user Query')
                    if state.turns and (not prepared[5] or plan.get('kvmem_query_id')!=state.plan.get('kvmem_query_id')):
                        raise ValueError('Single-task continuation changed the original user Query or accepted prefix')
                    if not state.turns and len(tokens)>args.select_budget-32:
                        raise ValueError('The complete initial task must fit the declared selected capacity')
                    single_task=dict(original_query_span=plan.get('kvmem_query_span'),
                        source_user_ordinal=query_source['user_ordinal'],
                        wire_user_frames=sum(m.get('role')=='user' for m in body['messages']),
                        policy='stream_initial_task_without_historical_scoring',model_input_unchanged=True)
                    data.pop('kvmem_query_span',None)
                    data.pop('kvmem_replay_span',None)
                    data['kvmem_reselect']='auto'
                maintenance=None
                maintenance_deferred=None
                if artifact_root is not None and artifact_armed.get(owner) and artifact_mode is not None:
                    if artifact_mode=='delete' and artifact_pending.get(owner):
                        if len(tokens)<=args.select_budget-256:
                            from diagnostics.dsh_artifact_gc import maintenance_boundary
                            maintenance=maintenance_boundary(data,artifact_mode)
                        else:
                            maintenance_deferred='Use ordinary bounded-window selection beyond the complete-history maintenance bound'
                elif (artifact_root is not None and artifact_armed.get(owner) and
                        plan.get('kvmem_query_id') != state.plan.get('kvmem_query_id') and
                        plan.get('kvmem_query_span') is not None):
                    data['kvmem_reselect']='force'
                if disable_thinking:
                    data.pop('thinking_token_budget',None)
                pin_admission=validate_system_pin_budget(data, planner.protocol)
                for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','seed'):
                    if k in body:data[k]=body[k]
                apply_sampling_overrides(data,args,state.turns)
                directory=args.output/owner/f'{state.turns+1:05d}'
                if directory.exists() and (directory/'request.json').exists():
                    directory.rename(directory.with_name(directory.name+'-attempt-'+uuid.uuid4().hex[:8]))
                directory.mkdir(parents=True,exist_ok=True)
                (directory/'request.json').write_text(json.dumps(body,ensure_ascii=False))
                (directory/'plan.json').write_text(json.dumps({k:v for k,v in data.items() if k!='prompt'}|
                    dict(prompt_tokens=len(tokens),exact_output_prefix_reused=prepared[5],
                         query_source=query_source,requested_output_cap=limit,
                         effective_output_cap=output_limit,logical_remaining_tokens=remaining_context)|
                    (dict(system_pin_admission=pin_admission) if pin_admission is not None else {})|
                    (dict(artifact_gc_maintenance=maintenance) if maintenance is not None else {})|
                    (dict(artifact_gc_deferred=maintenance_deferred) if maintenance_deferred is not None else {})|
                    (dict(single_query_task=single_task) if single_task is not None else {}),indent=2))
                if artifact_root is not None:
                    from diagnostics.dsh_artifact_gc import verified_candidates
                    armed_ids=artifact_armed.setdefault(owner,set())
                    candidates=await asyncio.to_thread(verified_candidates,artifact_root,args.output,owner,state,prepared,body,artifact_decoder.token_bytes,armed_ids,container=artifact_container)
                    for candidate in candidates:
                        armed=await client.post(args.backend+'/collective_rpc',json=dict(method='dsh_artifact_gc_arm',kwargs=dict(plan=candidate)))
                        armed.raise_for_status()
                        armed_ids.add(candidate['span_id'])
                    if candidates:
                        (directory/'artifact-gc-plans.json').write_text(json.dumps(candidates,indent=2)+'\n')
                data.update(stream=True,stream_options=dict(include_usage=True))
                wire=QwenStream(planner.protocol,rendered['tools']);output_ids=[];usage=None;finish=None
                decoder=AcceptedTextDecoder(tokenizer) if segments>1 else None
                metrics=None;request_id=None;first_token=None;first_emitted=None;chunks=0
                segment_reports=[]
                for segment in range(segments):
                    begin=len(output_ids);segment_usage=None;segment_metrics=None;finish=None
                    segment_data=dict(data,prompt=tokens+output_ids,max_tokens=min(args.gen_budget,output_limit-begin))
                    if segment:
                        segment_data['kvmem_reselect']='auto'
                        segment_data.pop('thinking_token_budget',None)
                        # The unchanged auto Query is suppressed only after the
                        # native ledger proves this exact accepted prefix.
                    segment_pin_admission=validate_system_pin_budget(segment_data, planner.protocol)
                    async with client.stream('POST',args.backend+'/v1/completions',json=segment_data) as result:
                        if result.status_code!=200:
                            raise ValueError('Model request failed: '+(await result.aread()).decode()[:2000])
                        async for native in native_events(result):
                            request_id=native.get('id',request_id)
                            if native.get('usage') is not None:segment_usage=native['usage']
                            if native.get('metrics') is not None:segment_metrics=native['metrics']
                            for choice in native.get('choices',[]):
                                if choice.get('index',0)!=0:raise ValueError('Unexpected model choice')
                                ids=choice.get('token_ids')
                                if not isinstance(ids,list):raise ValueError('Model stream omitted actual accepted token IDs')
                                if ids and first_token is None:first_token=time.time()
                                output_ids.extend(ids);chunks+=1
                                text=decoder.feed(ids) if decoder is not None else choice.get('text','')
                                if emit is None:wire.raw+=text
                                else:
                                    delta=wire.feed(text)
                                    if delta:
                                        if first_emitted is None:first_emitted=time.time()
                                        await emit(delta)
                                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                    if (segment_usage is None or finish is None or
                            segment_usage['completion_tokens']!=len(output_ids)-begin or
                            segment_usage['completion_tokens']>segment_data['max_tokens']):
                        raise ValueError('Incomplete model stream or accepted-token/usage mismatch')
                    report=dict(index=segment,id=request_id,prompt_tokens=len(segment_data['prompt']),
                        output_tokens=len(output_ids)-begin,finish_reason=finish,metrics=segment_metrics)
                    if segment_pin_admission is not None:
                        report['system_pin_admission']=segment_pin_admission
                        report['mandatory_spans']=segment_data['kvmem_mandatory_spans']
                    segment_reports.append(report)
                    (directory/f'segment-{segment:03d}.json').write_text(json.dumps(dict(**report,
                        token_ids=output_ids[begin:]),ensure_ascii=False))
                    if finish!='length' or len(output_ids)>=output_limit:break
                    if len(output_ids)==begin:raise ValueError('Continuation made no progress')
                if decoder is not None:
                    tail_text=decoder.finish()
                    if emit is None:wire.raw+=tail_text
                    else:
                        delta=wire.feed(tail_text)
                        if delta:await emit(delta)
                usage=dict(prompt_tokens=len(tokens),completion_tokens=len(output_ids),total_tokens=len(tokens)+len(output_ids))
                metrics=segment_reports[0]['metrics'] if len(segment_reports)==1 else dict(segments=segment_reports)
                # Preserve a failed full-parser input too, so a long coding
                # response can be diagnosed without repeating GPU generation.
                (directory/'native.json').write_text(json.dumps(dict(id=request_id,text=wire.raw,
                    token_ids=output_ids,usage=usage,finish_reason=finish,metrics=metrics),ensure_ascii=False))
                response=planner.response(wire.raw,rendered['tools'],body['model'],usage,finish)
                if emit is not None:output,tail=wire.finish(response)
                else:output=to_openai(response)
                async def commit():
                    await asyncio.to_thread(planner.commit,state,prepared,payload,response,output_ids)
                    (directory/'response.json').write_text(json.dumps(output,ensure_ascii=False))
                    # The completed accepted-ID file is the recovery commit marker.
                    (directory/'accepted.json').write_text(json.dumps(dict(token_ids=output_ids,metrics=metrics)))
                committed=asyncio.create_task(commit())
                try:
                    await asyncio.shield(committed)
                except asyncio.CancelledError:
                    # Cancelling to_thread cannot stop its state mutation. Keep
                    # both owner and global admission until the completed reply
                    # and its recovery marker are committed together.
                    with anyio.CancelScope(shield=True):
                        await asyncio.shield(committed)
                    raise
                if artifact_root is not None:
                    snapshot=await client.post(args.backend+'/collective_rpc',json=dict(method='dsh_artifact_gc_snapshot'))
                    snapshot.raise_for_status()
                    snapshot_data=snapshot.json()
                    from diagnostics.dsh_artifact_gc import artifact_handoff_needed
                    artifact_pending[owner]=artifact_handoff_needed(snapshot_data,owner)
                    (directory/'artifact-gc-state.json').write_text(json.dumps(snapshot_data,indent=2)+'\n')
                log(dict(at=time.time(),owner=owner,turn=state.turns,request_id=request_id,
                    prompt_tokens=len(tokens),output_tokens=len(output_ids),
                    query_span=plan.get('kvmem_query_span'),
                    mandatory_spans=plan['kvmem_mandatory_spans'],mandatory_parts=plan['kvmem_mandatory_parts'],
                    exact_output_prefix_reused=prepared[5],seconds=time.time()-started,
                    tool_calls=len(output['choices'][0]['message'].get('tool_calls',[])),
                    transport='native_sse',native_chunks=chunks,metrics=metrics,
                    native_segments=len(segment_reports),segment_reports=segment_reports,
                    requested_output_cap=limit,effective_output_cap=output_limit,
                    first_native_token_seconds=None if first_token is None else first_token-started,
                    first_visible_delta_seconds=None if first_emitted is None else first_emitted-started))
                if emit is not None and tail:await emit(tail)
                return output
        if not body.get('stream'):
            try:return await generate()
            except (ValueError,httpx.HTTPError) as error:
                log(dict(at=time.time(),owner=owner,error=str(error)))
                return JSONResponse(dict(error=dict(message=str(error),type='api_error')),status_code=502)
        async def stream():
            queue=asyncio.Queue(maxsize=32)
            job=asyncio.create_task(generate(queue.put))
            base=dict(id='chatcmpl-'+uuid.uuid4().hex,created=int(time.time()),model=body['model'],object='chat.completion.chunk')
            def event(delta,finish=None,**extra):
                return 'data: '+json.dumps(dict(**base,choices=[dict(index=0,delta=delta,finish_reason=finish)],**extra))+'\n\n'
            pending=None
            try:
                yield event(dict(role='assistant'))
                while not job.done() or not queue.empty():
                    pending=asyncio.create_task(queue.get())
                    done,_=await asyncio.wait({pending,job},timeout=2,return_when=asyncio.FIRST_COMPLETED)
                    if pending.done():yield event(pending.result())
                    else:
                        pending.cancel();await asyncio.gather(pending,return_exceptions=True)
                        if not pending.cancelled():yield event(pending.result())
                        if not done:yield ': waiting for model\n\n'
                    pending=None
                result=job.result()
                yield event({},result['choices'][0]['finish_reason'],usage=result['usage'])
                yield 'data: [DONE]\n\n'
            except (ValueError,httpx.HTTPError) as error:
                log(dict(at=time.time(),owner=owner,error=str(error)))
                yield 'data: '+json.dumps(dict(error=dict(message=str(error),type='api_error')))+'\n\n'
            finally:
                # ASGI 2.3 disconnects cancel the surrounding AnyIO scope.
                # Cancel both children before the first checkpoint; otherwise a
                # cancelled pending.get cleanup can strand the generation job
                # and its global admission permit. Finish upstream close under
                # a shield before another owner can acquire the permit.
                if pending is not None:pending.cancel()
                if not job.done():job.cancel()
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(job,*(() if pending is None else (pending,)),
                                         return_exceptions=True)
        return StreamingResponse(stream(),media_type='text/event-stream')
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    protocol=p.add_mutually_exclusive_group(required=True)
    protocol.add_argument('--protocol',type=Path,help='Existing frozen qw3 protocol library')
    protocol.add_argument('--tokenizer-protocol',action='store_true',help='Use the audited current Qwen3.8 tokenizer chat template')
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--reasoning-effort',choices=('low','medium','xhigh'),default='low')
    p.add_argument('--select-budget',type=int,
        help='Explicit backend K for --pin-system; backend must use block32/sink1/recent1/replay128')
    p.add_argument('--pin-system',action='store_true',
        help='Preserve the complete audited system/tool frame; requires tokenizer protocol and sufficient K')
    p.add_argument('--disable-thinking',action='store_true',
        help='Close the official tokenizer think frame before generation; incompatible with thinking-enabled history')
    p.add_argument('--backend',default='http://127.0.0.1:18846');p.add_argument('--model',default='kvmem-dsh-dflash2')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18381)
    p.add_argument('--gen-budget',type=int,default=16384);p.add_argument('--max-model-len',type=int,default=262144)
    p.add_argument('--max-sessions',type=int,default=32)
    p.add_argument('--max-inflight',type=int,default=0,help='Maximum concurrent logical replies across owners; 0 preserves unlimited admission')
    p.add_argument('--owner-namespace',default='dsh')
    p.add_argument('--max-output-segments',type=int,choices=range(1,5),default=1)
    p.add_argument('--require-query-provenance',action='store_true')
    p.add_argument('--artifact-gc-root',type=Path,help='Opt-in verified-file GC experiment workspace; owned loopback backend only')
    p.add_argument('--artifact-gc-container',help='Read actual workspace files in this owned local Docker task container')
    p.add_argument('--artifact-gc-mode',choices=('off','delete'),help='Explicit paired experiment; delete adds bounded real-input maintenance handoffs')
    p.add_argument('--single-query-task',action='store_true',help='Verified one-instruction benchmark: stream the full initial task in normal prefill chunks, without historical Query scoring')
    p.add_argument('--temperature',type=float,help='Explicit sampling override for controlled paired experiments')
    p.add_argument('--seed',type=int,help='Explicit sampling seed for controlled paired experiments')
    p.add_argument('--seed-per-turn',action='store_true',help='Use base seed plus committed reply index in both groups')
    p.add_argument('--top-p',type=float)
    p.add_argument('--top-k',type=int)
    p.add_argument('--min-p',type=float)
    p.add_argument('--presence-penalty',type=float)
    p.add_argument('--repetition-penalty',type=float)
    p.add_argument('--tool-parser-url',help='Owned remote CPU service using the exact official vLLM parser')
    p.add_argument('--restore-from',type=Path,action='append',default=[]);args=p.parse_args()
    from transformers import AutoTokenizer
    import uvicorn
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    uvicorn.run(create_app(args,tokenizer),host='127.0.0.1',port=args.port)
