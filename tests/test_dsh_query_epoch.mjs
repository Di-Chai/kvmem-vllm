import test from 'node:test';
import assert from 'node:assert/strict';
import { kvmemQueryEpochPayload } from '../scripts/dsh_query_epoch.mjs';

const model={id:'kvmem-dsh-dflash2',baseUrl:'http://127.0.0.1:18385/v1'};
const message=(id,kind,text)=>({id,role:'user',source:{kind},content:[{type:'text',text}]});
const options=()=>({sessionId:'s',messages:[message('u1','user','任务 中文'),
  {id:'tool',role:'user',source:{kind:'tool'},content:[{type:'tool-result',content:[]}]},
  message('p','plugin','notice') ]});
const wire=()=>({prompt_cache_key:'s',messages:[{role:'system',content:'system'},
  {role:'user',content:'任务 中文'},{role:'tool',content:'result'},
  {role:'user',content:'notice'}]});

test('plugin and tool messages do not replace the manual Query',()=>{
  const body=wire(),before=JSON.stringify(body.messages);
  kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,options())(body);
  assert.equal(body.kvmem_query_provenance.query_user_ordinal,0);
  assert.equal(body.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(JSON.stringify(body.messages),before);
});

test('identical text sent manually again starts a distinct Query',()=>{
  const o=options();o.messages.push(message('u2','user','任务 中文'));
  const b=wire();b.messages.push({role:'user',content:'任务 中文'});
  kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,o)(b);
  assert.equal(b.kvmem_query_provenance.query_user_ordinal,2);
  assert.equal(b.kvmem_query_provenance.query_message_id,'u2');
});

test('wire mismatch and wrong session fail before attaching proof',()=>{
  const callback=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,options());
  const a=wire();a.messages[1].content='different';assert.throws(()=>callback(a));
  const b=wire();b.prompt_cache_key='other';assert.throws(()=>callback(b));
  assert.equal(a.kvmem_query_provenance,undefined);assert.equal(b.kvmem_query_provenance,undefined);
});

test('scope is the owned local route and each callback owns its source identity',()=>{
  assert.equal(kvmemQueryEpochPayload('other',model,options()),undefined);
  assert.equal(kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',{...model,baseUrl:'https://example.com'},options()),undefined);
  const a=options(),b=options();b.messages[0].id='other-id';
  const one=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,a);
  const two=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,b);
  const x=wire(),y=wire();two(y);one(x);
  assert.equal(x.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(y.kvmem_query_provenance.query_message_id,'other-id');
});

test('an explicit deployment route replaces all three default match fields',()=>{
  const route={provider:'gc-local',model:'kvmem-gc-qwen38',baseURL:'http://127.0.0.1:18387/v1/'};
  const customModel={id:route.model,baseUrl:'http://127.0.0.1:18387/v1'};
  const body=wire();
  kvmemQueryEpochPayload(route.provider,customModel,options(),route)(body);
  assert.equal(body.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(kvmemQueryEpochPayload(route.provider,customModel,options()),undefined);
  assert.equal(kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,options(),route),undefined);
  for (const [provider,target] of [
    ['other',customModel],
    [route.provider,{...customModel,id:'other'}],
    [route.provider,{...customModel,baseUrl:'http://127.0.0.1:18388/v1'}],
  ]) assert.equal(kvmemQueryEpochPayload(provider,target,options(),route),undefined);
});

test('invalid explicit routes fail closed instead of using default configuration',()=>{
  const valid={provider:'gc-local',model:'local',baseURL:'http://127.0.0.1:18387/v1'};
  for (const route of [
    null, [], 'default', {}, {...valid,provider:''}, {...valid,model:' '},
    {...valid,baseURL:'file:///tmp/provider'},
    {...valid,baseURL:'http://user:password@127.0.0.1:18387/v1'},
    {...valid,baseURL:valid.baseURL+'?query=value'},
    {...valid,baseURL:valid.baseURL+'#fragment'},
    {...valid,baseURL:'http:/127.0.0.1:18387/v1'},
    {...valid,baseURL:'http://127.0.0.1:99999/v1'},
  ]) assert.throws(()=>kvmemQueryEpochPayload('other',model,options(),route),TypeError);
});

test('custom-route callbacks preserve wire and session validation',()=>{
  const route={provider:'gc-local',model:'local',baseURL:'http://127.0.0.1:18387/v1'};
  const callback=kvmemQueryEpochPayload(route.provider,{id:route.model,baseUrl:route.baseURL},options(),route);
  const wrongText=wire();wrongText.messages[1].content='different';
  const wrongSession=wire();wrongSession.prompt_cache_key='different';
  assert.throws(()=>callback(wrongText),/provenance differs/);
  assert.throws(()=>callback(wrongSession),/session identity differs/);
  assert.equal(wrongText.kvmem_query_provenance,undefined);
  assert.equal(wrongSession.kvmem_query_provenance,undefined);
});
