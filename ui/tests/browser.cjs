/* Model-free integration checks for the packaged console, not inference tests. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '../../src/vllm_kvmem/apps/webui/static');
const profile = {model:'/models/main',tokenizer:'/models/tokenizer',hf_config_path:'/models/main',model_format:'safetensors',served_model_name:'fixture',host:'127.0.0.1',port:8000,max_model_len:32768,max_num_seqs:2,max_num_batched_tokens:1024,gpu_memory_utilization:0.6,prefill_chunk_tokens:128,query_replay_budget:1024,draft_method:'mtp',draft_tokens:1,execution_mode:'eager',kvmem:{select_budget:2048,gen_budget:2048,cpu_read_cache_bytes:2147483648,session_capacity:8,repository_blocks:1024,block_tokens:32,sink_blocks:1,recent_blocks:1,unexposed_fixture:true}};
let state = 'stopped', active = null, instance = 'fixture-1', failNext = false;
const posts = [], errors = [], timers = new Set();
const later = (fn, ms) => { const timer = setTimeout(() => {timers.delete(timer); fn();}, ms); timers.add(timer); };
const status = () => ({state,instance,active_profile:active,active_config:active?{parameters:{Backend:'vllm',Model:active.model}}:null,url:active?'http://127.0.0.1:8000':null,error:state==='failed'?'Fixture loading failed':null});
function json(res, value, code=200) {res.writeHead(code,{'Content-Type':'application/json'});res.end(JSON.stringify(value));}
const server = http.createServer(async (req,res) => {
  try {
    const url = new URL(req.url,'http://localhost'); let body;
    if(req.method==='POST') {let raw='';for await(const piece of req)raw+=piece;body=JSON.parse(raw||'{}');posts.push({path:url.pathname,body,headers:req.headers});}
    if(url.pathname==='/api/launcher/info')return json(res,{token:'fixture-token',backend:'vllm',profile,status:status(),capabilities:{}});
    if(url.pathname==='/api/launcher/status')return json(res,status());
    if(url.pathname==='/api/launcher/preview')return json(res,{profile:body.profile,vllm_argv:['python','-m','vllm'],environment:{}});
    if(url.pathname==='/api/launcher/start') {active=body.profile;state='loading';instance='fixture-'+Date.now();json(res,status());later(()=>{state=failNext?'failed':'ready';failNext=false;},900);return;}
    if(url.pathname==='/api/launcher/stop') {state='stopped';return json(res,status());}
    if(url.pathname==='/api/launcher/logs')return json(res,{protocol:1,revision:'one',text:'INFO fixture backend log\n',backend:'vllm',model:'fixture',session:instance,running:state==='ready',available:true,started_at:Date.now()});
    if(url.pathname==='/api/launcher/hardware')return json(res,{schema_version:1,scope:'system',sampled_at_ms:Date.now(),gpus:[]});
    if(url.pathname==='/api/monitor/snapshot')return json(res,{schema_version:1,backend:'vllm',server_instance:instance,sampled_at_ms:Date.now(),active:[],recent:[],hardware:{sampled_at_ms:Date.now(),gpus:[]},kvmem:{},request_scope:'console',native_metrics:{scope:'engine',values:{'vllm:num_requests_running':0}}});
    if(url.pathname==='/props')return json(res,{model_name:'fixture',default_generation_settings:{n_ctx:32768},kvmem:{},execution:{}});
    if(url.pathname==='/slots')return json(res,[]);
    if(url.pathname==='/api/chat/context')return json(res,{state,instance,backend:'vllm',model:'fixture',context:32768,limit:4096,effective:{max_tokens:128,enable_thinking:true},sources:{},capabilities:{thinking:true,budget:false,vision:false,preserve:false,efforts:[]}});
    if(url.pathname.startsWith('/api/chat/requests/'))return json(res,{id:url.pathname.split('/').at(-1),state:'answering'});
    if(url.pathname==='/api/chat/release'||url.pathname==='/api/chat/cancel')return json(res,{ok:true});
    if(url.pathname==='/api/chat/completions') {
      assert.equal(req.headers['x-kvmem-client'],body.client_id);assert.ok(body.conversation_id);assert.equal(req.headers['x-kvmem-token'],'fixture-token');
      res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-store'});
      res.write('data: '+JSON.stringify({choices:[],kvmem:{history_mode:body.messages.length>1?'canonical_rebuild':'new'}})+'\n\n');
      later(()=>{if(!res.destroyed)res.write('data: '+JSON.stringify({choices:[{delta:{content:'Fixture answer **rendered**.'}}]})+'\n\n');},150);
      const slow=body.messages.at(-1)?.content==='slow';
      later(()=>{if(!res.destroyed){res.write('data: '+JSON.stringify({choices:[{delta:{},finish_reason:'stop'}],usage:{completion_tokens:5}})+'\n\n');res.end('data: [DONE]\n\n');}},slow?10000:1300);return;
    }
    let name = url.pathname;
    if(name==='/'||name==='/monitor/'||name==='/chat/')name='/monitor/index.html';
    if(name==='/chat/view/')name='/chat/view/index.html';
    const file=path.resolve(root,'.'+name); if(!file.startsWith(root+path.sep)||!fs.existsSync(file))return json(res,{error:'not found'},404);
    const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.woff2':'font/woff2','.txt':'text/plain'};
    res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream'});fs.createReadStream(file).pipe(res);
  } catch(error) {errors.push(error.stack);if(!res.headersSent)json(res,{error:error.message},500);else res.end();}
});
(async()=>{
  let browser;
  try {
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    const origin='http://127.0.0.1:'+server.address().port;
    browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{}),args:['--no-sandbox']});
    const context=await browser.newContext({viewport:{width:1440,height:1050}}),page=await context.newPage();
    page.on('pageerror',e=>errors.push(e.stack));
    await page.goto(origin+'/monitor/');
    await page.locator('[data-profile="model"]').fill('/models/edited');
    await page.getByRole('tab',{name:'缓存与历史',exact:true}).click();
    await page.locator('[data-profile="kvmem.select_budget"]').fill('4096');
    await page.locator('[data-action="preview"]').click();
    await page.locator('dialog pre').filter({hasText:'vllm_argv'}).waitFor();
    const preview=posts.find(p=>p.path.endsWith('/preview')).body.profile;
    assert.equal(preview.model,'/models/edited');assert.equal(preview.kvmem.select_budget,4096);assert.equal(preview.kvmem.unexposed_fixture,true);
    await page.getByRole('button',{name:'关闭',exact:true}).click();
    await page.getByRole('button',{name:'另存为模板',exact:true}).click();
    await page.getByRole('textbox',{name:'模板名称',exact:true}).fill('Fixture profile');
    await page.getByRole('button',{name:'保存',exact:true}).click();
    await page.getByRole('button',{name:'Fixture profile',exact:true}).waitFor();
    await page.locator('[data-action="start"]').click();
    await page.locator('[data-runtime-state]').filter({hasText:'加载中'}).waitFor();
    assert.ok(await page.locator('[data-profile="kvmem.select_budget"]').isDisabled());
    assert.equal(await page.locator('[data-action="stop"]').isDisabled(),false);
    await page.locator('[data-runtime-state]').filter({hasText:'已就绪'}).waitFor({timeout:7000});
    await page.locator('[data-page="logs"]').click();await page.locator('[data-log-rows]').filter({hasText:'fixture backend log'}).waitFor();
    await page.locator('[data-page="chat"]').click();
    const frame=page.frameLocator('#chat-frame');
    await frame.getByPlaceholder('输入消息…').fill('first turn');await frame.getByRole('button',{name:'发送',exact:true}).click();
    await frame.getByRole('button',{name:'停止生成',exact:true}).waitFor();
    await page.locator('[data-page="monitor"]').click();
    await page.waitForTimeout(1500);await page.locator('[data-page="chat"]').click();
    await frame.getByRole('button',{name:'发送',exact:true}).waitFor();await frame.locator('strong').filter({hasText:'rendered'}).waitFor();
    await frame.getByPlaceholder('输入消息…').fill('second turn');await frame.getByRole('button',{name:'发送',exact:true}).click();
    await frame.getByText('本轮已从完整历史重新构建上下文。',{exact:true}).waitFor();
    await frame.getByRole('button',{name:'发送',exact:true}).waitFor();
    const completions=posts.filter(p=>p.path.endsWith('/completions'));
    assert.equal(completions.length,2);assert.equal(completions[0].body.client_id,completions[1].body.client_id);assert.equal(completions[0].body.conversation_id,completions[1].body.conversation_id);assert.equal(completions[1].body.messages.length,3);
    const otherContext=await browser.newContext(),otherPage=await otherContext.newPage();
    await otherPage.goto(origin+'/chat/');const otherFrame=otherPage.frameLocator('#chat-frame');
    await otherFrame.getByPlaceholder('输入消息…').fill('separate browser');await otherFrame.getByRole('button',{name:'发送',exact:true}).click();
    await otherFrame.getByRole('button',{name:'停止生成',exact:true}).waitFor();
    await otherFrame.getByRole('button',{name:'发送',exact:true}).waitFor();
    assert.notEqual(posts.filter(p=>p.path.endsWith('/completions')).at(-1).body.client_id,completions[0].body.client_id);
    await otherContext.close();
    await frame.getByRole('button',{name:'参数',exact:true}).click();
    assert.equal(await frame.getByRole('spinbutton',{name:'思考预算 · tokens',exact:true}).count(),0);
    await frame.getByRole('button',{name:'关闭面板',exact:true}).last().click();
    await frame.getByPlaceholder('输入消息…').fill('slow');await frame.getByRole('button',{name:'发送',exact:true}).click();await frame.getByRole('button',{name:'停止生成',exact:true}).click();
    await frame.getByRole('button',{name:'发送',exact:true}).waitFor();assert.ok(posts.some(p=>p.path.endsWith('/cancel')&&p.body.client_id===completions[0].body.client_id));
    await page.reload();await page.locator('[data-page="chat"]').click();await frame.getByRole('button',{name:'first turn',exact:true}).waitFor();
    await frame.locator('.history-menu summary').first().click();page.once('dialog',dialog=>dialog.accept());await frame.getByRole('button',{name:'删除对话',exact:true}).click();
    await page.waitForTimeout(100);assert.ok(posts.some(p=>p.path.endsWith('/release')&&p.body.conversation_id===completions[0].body.conversation_id));
    await page.locator('[data-page="config"]').click();await page.locator('[data-action="stop"]').click();failNext=true;
    await page.locator('[data-action="start"]').click();await page.locator('[data-runtime-state]').filter({hasText:'启动失败'}).waitFor({timeout:7000});
    assert.equal(await page.locator('[data-action="start"]').isDisabled(),false);await page.locator('[data-service-detail]').filter({hasText:'Fixture loading failed'}).waitFor();
    await page.locator('[data-language-select]').selectOption('en');await page.getByRole('heading',{name:'Launch configuration',exact:true}).waitFor();
    assert.match(await page.locator('#monitor-note-prefill').textContent(),/including tokenization, queueing and network waits/);
    await page.locator('[data-theme-select]').selectOption('blue');await page.locator('[data-density-select]').selectOption('compact');
    assert.equal(await page.locator('#app').getAttribute('data-theme'),'blue');
    await page.screenshot({path:process.env.UI_SCREENSHOT||'/tmp/kvmem-ui-fixture.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS: profile preservation, templates, loading/failure controls, logs, persistent streaming navigation, chat session identity, history, cancellation, release, capability settings and appearance.');
  } finally {for(const timer of timers)clearTimeout(timer);if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
