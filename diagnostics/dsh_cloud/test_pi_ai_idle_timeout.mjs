#!/usr/bin/env node
// CPU-only: official DSH adapter/SDK, loopback fake HTTP, no agent or tools.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';

const args = process.argv.slice(2);
const option = name => args[args.indexOf(name) + 1];
assert(args.includes('--adapter') && args.includes('--output'),
  'Usage: node test_pi_ai_idle_timeout.mjs --adapter ABSOLUTE/lib/index.js --output FRESH.json');
const adapterPath = resolve(option('--adapter'));
const { apply, PiAiAdapter } = await import(pathToFileURL(adapterPath));
const output = resolve(option('--output'));
const provider = 'gc-cpu-watchdog';
const model = 'qwen-cpu-fixture';
const toolArguments = JSON.stringify({ path: 'fixture.txt', content: 'accepted content, not a real file' });
const messages = [{ role: 'user', content: [{ type: 'text', text: 'CPU fixture only' }] }];
let scenario;
const socketSet = new Set();
const server = createServer(async (request, response) => {
  const current = scenario;
  current.requests++;
  let body = '';
  for await (const piece of request) body += piece;
  current.body = JSON.parse(body);
  assert.equal(request.url, '/v1/chat/completions');
  response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
  response.flushHeaders();
  const chunk = (delta, finish_reason = null) => response.write('data: ' + JSON.stringify({
    id: 'cpu-fixed-id', object: 'chat.completion.chunk', created: 0, model,
    choices: [{ index: 0, delta, finish_reason }],
  }) + '\n\n');
  const timers = [];
  response.on('close', () => {
    current.closed = true;
    for (const timer of timers) clearTimeout(timer);
    current.closeResolve();
  });
  chunk({ role: 'assistant' });
  chunk({ content: 'Preparing.' });
  // Only transport events arrive while the complete tool is unavailable.
  if (current.noise) timers.push(setInterval(() => {
    current.noiseSent++;
    if (current.noise === 'comment') response.write(': waiting for model\n\n');
    else chunk(current.noise === 'empty-delta' ? {} : { content: '' });
  }, 30));
  if (current.visible) timers.push(setInterval(() => chunk({ content: '.' }), 60));
  if (current.abortAfter) timers.push(setTimeout(() => current.controller.abort('CPU caller cancel'), current.abortAfter));
  if (!current.stall) timers.push(setTimeout(() => {
    chunk({ tool_calls: [{ index: 0, id: 'cpu-tool-id', type: 'function',
      function: { name: 'write', arguments: toolArguments } }] });
    chunk({}, 'tool_calls');
    response.write('data: ' + JSON.stringify({ id: 'cpu-fixed-id', model, choices: [],
      usage: { prompt_tokens: 7, completion_tokens: 19, total_tokens: 26 } }) + '\n\n');
    response.end('data: [DONE]\n\n');
  }, current.finishAfter));
});
server.on('connection', socket => {
  socketSet.add(socket);
  socket.on('close', () => socketSet.delete(socket));
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
const baseURL = 'http://127.0.0.1:' + server.address().port + '/v1';
function makeAdapter(timeout) {
  let adapter;
  // Official apply() resolves the actual public profile and creates PiAiAdapter.
  // Settings hot reload is outside this one immutable call's scope.
  apply({
    get() { return undefined; },
    inject() {},
    logger: { warn() {}, error() {} },
    llm: {
      registerConfigurableProviders() { return { replace() {} }; },
      registerModelDiscovery() {},
      registerAdapter(routes, instance) {
        assert.deepEqual(routes, [provider]);
        adapter = instance;
        return { replace() {} };
      },
    },
  }, { providers: { [provider]: {
    api: 'openai-completions', baseURL,
    headers: { Authorization: 'Bearer cpu-unused' },
    streamIdleTimeoutMs: timeout,
    compat: { supportsDeveloperRole: false, maxTokensField: 'max_tokens', thinkingFormat: 'deepseek' },
    models: [{ id: model, contextWindow: 32768, maxTokens: 8192 }],
  } } });
  assert(adapter instanceof PiAiAdapter);
  return adapter;
}
const cases = [];
async function run(name, settings) {
  const current = scenario = { name, requests: 0, closed: false, noiseSent: 0,
    finishAfter: 450, controller: new AbortController(), ...settings };
  current.closedPromise = new Promise(resolve => current.closeResolve = resolve);
  const adapter = makeAdapter(current.idle);
  const chunks = [];
  const started = performance.now();
  let error;
  try {
    for await (const chunk of adapter.stream({
      provider, model, system: 'CPU integration fixture, no tools will be executed.',
      messages, maxTokens: 8192, signal: AbortSignal.any([current.controller.signal, AbortSignal.timeout(3000)]),
      tools: [{ name: 'write', description: 'Fixture only',
        parameters: { type: 'object', properties: { path: { type: 'string' },
          content: { type: 'string' } }, required: ['path', 'content'] } }],
    })) chunks.push(structuredClone(chunk));
  } catch (caught) {
    error = { name: caught.name, code: caught.code, message: caught.message };
  }
  await Promise.race([current.closedPromise, delay(3000).then(() => { throw Error(name + ': upstream did not close'); })]);
  assert.equal(current.requests, 1);
  const report = {
    name, idle_ms: current.idle, elapsed_ms: Math.round(performance.now() - started),
    error: error ?? null, upstream_closed: current.closed, requests: current.requests,
    noise_sent: current.noiseSent, chunks, body: current.body,
  };
  if (current.expectedError) {
    if (current.expectedError === 'ABORTED') {
      // pi-ai returns caller abort as a terminal finish; the watchdog itself
      // throws TIMEOUT. Assert the official distinction rather than erasing it.
      assert.equal(error, undefined, JSON.stringify(report));
      assert.equal(chunks.at(-1)?.reason?.kind, 'aborted', JSON.stringify(report));
      assert.equal(chunks.at(-1)?.reason?.failure?.code, 'ABORTED');
      assert(report.elapsed_ms < current.idle);
    } else assert.equal(error?.code, current.expectedError, JSON.stringify(report));
    assert(!chunks.some(c => c.type === 'tool-call-delta' ||
      (c.type === 'block-start' && c.blockType === 'tool-call')),
      name + ': no partial tool may become visible');
  } else {
    assert.equal(error, undefined, JSON.stringify(report));
    assert.equal(chunks.at(-1)?.type, 'finish');
    const blocks = chunks.filter(c => c.type === 'block-end').map(c => c.block);
    assert.equal(blocks.at(-1).type, 'tool-call');
    assert.equal(blocks.at(-1).arguments, toolArguments);
    assert.equal(chunks.filter(c => c.type === 'text-delta').map(c => c.text).join(''),
      blocks[0].text);
  }
  cases.push(report);
  return report;
}
try {
  // Warm lazy SDK imports before short real timers. No implementation is copied.
  const baseline = await run('fast-baseline', { idle: 2000, finishAfter: 10 });
  for (const noise of ['comment', 'empty-delta', 'empty-content']) {
    const result = await run(noise + '-cannot-reset-watchdog',
      { idle: 200, noise, expectedError: 'TIMEOUT' });
    assert(result.noise_sent >= 2);
  }
  const extended = await run('finite-extension-completes-buffered-tool',
    { idle: 1200, noise: 'empty-content' });
  assert(extended.elapsed_ms > 200);
  assert.deepEqual(extended.chunks, baseline.chunks,
    'Empty content must not change durable blocks, replay metadata, usage or finish');
  assert.deepEqual(extended.body, baseline.body,
    'Changing streamIdleTimeoutMs must not change model request/history');
  await run('visible-progress-rearms-watchdog', { idle: 200, visible: true });
  await run('silent-stall-still-times-out', { idle: 300, stall: true, expectedError: 'TIMEOUT' });
  await run('caller-abort-closes-upstream', { idle: 1200, stall: true, abortAfter: 80, expectedError: 'ABORTED' });
  const report = { passed: true, cases: cases.length, model_requests: 0, tool_executions: 0,
    adapter_path: adapterPath,
    adapter_sha256: createHash('sha256').update(await readFile(adapterPath)).digest('hex'),
    note: 'Official adapter + pi-ai SDK, loopback fake HTTP only. No transport heartbeat fix is enabled.',
    results: cases };
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ passed: true, cases: cases.length, output }));
} finally {
  server.close();
  for (const socket of socketSet) socket.destroy();
}
