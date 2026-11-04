"""Run against the installed Harbor implementation with a fake provider only."""
import asyncio
import hashlib
import inspect
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip('harbor')
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
import harbor.llms.lite_llm as provider
from scripts.terminalbench_length_finish import install


@pytest.mark.parametrize('thinking', [False, True])
def test_length_recovery_keeps_reasoning_only_when_configured(tmp_path,monkeypatch,thinking):
    monkeypatch.setattr(LiteLLM,'call',LiteLLM.call)
    monkeypatch.setattr(Terminus2,'_query_llm',Terminus2._query_llm)
    monkeypatch.setattr(LiteLLM,'_kvmem_length_finish_repaired',False,raising=False)
    sources={Path(inspect.getfile(cls)).name:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in (LiteLLM,Terminus2)}
    install(tmp_path/'hook.json',sources)
    calls=[]
    async def completion(**kwargs):
        calls.append(kwargs)
        count=len(calls)
        return provider.litellm.ModelResponse(model='openai/fixture',choices=[dict(index=0,
            finish_reason='length' if count==1 else 'stop',message=dict(role='assistant',
                content='unchanged truncated text' if count==1 else 'complete response',
                reasoning_content='original reasoning' if count==1 else 'next reasoning'))],
            usage=dict(prompt_tokens=10,completion_tokens=8,total_tokens=18))
    monkeypatch.setattr(provider.litellm,'acompletion',completion)
    monkeypatch.setattr(LiteLLM,'_extract_usage_info',lambda *_:None)
    llm=LiteLLM('openai/fixture',model_info=dict(max_input_tokens=10000000,max_output_tokens=131072,
        input_cost_per_token=0,output_cost_per_token=0,litellm_provider='openai',mode='chat'),api_key='fixture')
    chat=Chat(llm,interleaved_thinking=thinking)
    agent=SimpleNamespace(logger=logging.getLogger('length-test'),_llm=llm,_llm_call_kwargs={},_api_request_times=[],
        _parser=SimpleNamespace(parse_response=lambda _:SimpleNamespace(warning=None)),options=SimpleNamespace(enable_summarize=False))
    agent._query_llm=MethodType(Terminus2._query_llm,agent)
    response=asyncio.run(agent._query_llm(chat=chat,prompt='original question'))
    assert response.content=='complete response' and len(calls)==2
    assert chat.messages[0]==dict(role='user',content='original question')
    expected=dict(role='assistant',content='unchanged truncated text')
    if thinking:expected['reasoning_content']='original reasoning'
    assert chat.messages[1]==expected
    assert 'NONE of the actions' in chat.messages[2]['content']
    assert calls[1]['messages'][1]==expected
    # This is the original agent's recovery conversation, never task reexecution.
    assert len(agent._api_request_times)==1
