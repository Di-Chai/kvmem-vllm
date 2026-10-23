#!/usr/bin/env node
// CPU-only observational harness for an existing model-generated static page.
// This script never writes game source or sends model requests.
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i]?.startsWith('--') || !process.argv[i + 1]) {
    throw new Error('Expected --name value pairs; see README.md');
  }
  args[process.argv[i].slice(2)] = process.argv[i + 1];
}
for (const name of ['root', 'output', 'tool-root']) {
  if (!args[name]) throw new Error('Missing --' + name);
}
const sourceRoot = await fs.realpath(args.root);
const output = path.resolve(args.output);
const toolRoot = path.resolve(args['tool-root']);
if (output === sourceRoot || output.startsWith(sourceRoot + path.sep)) {
  throw new Error('Validation output must be outside the game source directory');
}
await fs.mkdir(output, { recursive: true });
const require = createRequire(path.join(toolRoot, 'package.json'));
const { chromium } = require('playwright');
const { PNG } = require('pngjs');
const plan = args.plan ? JSON.parse(await fs.readFile(args.plan, 'utf8')) : {};
const entry = args.entry || 'index.html';
const width = plan.viewport?.width || 1280;
const height = plan.viewport?.height || 800;
const holdMs = Math.min(plan.hold_ms || 700, 10000);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const digest = data => crypto.createHash('sha256').update(data).digest('hex');
const report = {
  schema: 'dsh-static-game-validation-v1',
  started_at: new Date().toISOString(), source_root: sourceRoot, entry,
  limitations: [
    'Only existing static artifacts are executed. No game source is modified.',
    'Screenshot change alone does not prove movement, shooting, or correctness.',
    'Behavior claims require explicit state probes, action checks and source review.',
    'This evaluates a DSH task artifact, not semantic GC quality or model throughput.',
  ],
  browser: {}, requests: [], http_errors: [], console_errors: [], console_warnings: [],
  page_errors: [], request_failures: [], blocked_requests: [], actions: [],
  artifact_sha256: {}, plan,
};
const mime = {
  '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8',
  '.mjs':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8',
  '.json':'application/json', '.png':'image/png', '.jpg':'image/jpeg',
  '.jpeg':'image/jpeg', '.svg':'image/svg+xml', '.ico':'image/x-icon',
  '.woff':'font/woff', '.woff2':'font/woff2', '.mp3':'audio/mpeg',
  '.wav':'audio/wav', '.ogg':'audio/ogg', '.wasm':'application/wasm',
};
const server = http.createServer(async (req, res) => {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405); res.end(); return;
    }
    const requested = path.resolve(sourceRoot, '.' + pathname);
    const resolved = await fs.realpath(requested.endsWith(path.sep) ? path.join(requested, 'index.html') : requested);
    if (resolved !== sourceRoot && !resolved.startsWith(sourceRoot + path.sep)) {
      res.writeHead(403); res.end(); return;
    }
    const st = await fs.stat(resolved);
    const file = st.isDirectory() ? path.join(resolved, 'index.html') : resolved;
    const final = await fs.realpath(file);
    if (!final.startsWith(sourceRoot + path.sep)) {
      res.writeHead(403); res.end(); return;
    }
    const data = await fs.readFile(final);
    report.artifact_sha256[path.relative(sourceRoot, final)] = digest(data);
    res.writeHead(200, {'Content-Type': mime[path.extname(final)] || 'application/octet-stream', 'Cache-Control':'no-store'});
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch (error) {
    res.writeHead(error.code === 'ENOENT' ? 404 : 500); res.end();
  }
});
let browser, context;
try {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = 'http://127.0.0.1:' + server.address().port;
  const entryURL = new URL(entry.replace(/^\/+/, ''), origin + '/').href;
  if (new URL(entryURL).origin !== origin) throw new Error('Entry must be a relative local path');
  const browserArgs = [
    '--no-sandbox', '--enable-automation', '--disable-dev-shm-usage', '--disable-gpu',
    '--disable-gpu-compositing', '--disable-accelerated-video-decode',
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    '--disable-vulkan',
  ];
  browser = await chromium.launch({headless: true, args: browserArgs, timeout: 60000});
  report.browser = {
    version: browser.version(), default_executable_path_hint: chromium.executablePath(),
    playwright_version: require('playwright/package.json').version,
    launch_args: browserArgs, rendering: 'forced-software', viewport: {width, height},
  };
  context = await browser.newContext({
    viewport: {width, height}, serviceWorkers: 'block', reducedMotion:'reduce',
  });
  await context.route('**/*', async route => {
    const u = new URL(route.request().url());
    if (u.origin === origin || ['data:', 'blob:', 'about:'].includes(u.protocol)) {
      await route.continue(); return;
    }
    // Explicit asset host exceptions are recorded and never allow POST/API calls.
    if ((plan.allowed_asset_hosts || []).includes(u.hostname)
        && route.request().method() === 'GET'
        && ['script','stylesheet','image','font','media'].includes(route.request().resourceType())) {
      await route.continue(); return;
    }
    report.blocked_requests.push({url:u.href, method:route.request().method(), resource_type:route.request().resourceType()});
    await route.abort('blockedbyclient');
  });
  await context.routeWebSocket('**/*', socket => {
    report.blocked_requests.push({url:socket.url(), method:'WEBSOCKET', resource_type:'websocket'});
    socket.close();
  });
  const page = await context.newPage();
  const browserCDP = await browser.newBrowserCDPSession();
  report.browser.actual_command_line = (await browserCDP.send('Browser.getBrowserCommandLine')).arguments;
  await browserCDP.detach();
  page.on('console', msg => {
    if (msg.type() === 'error') report.console_errors.push(msg.text());
    if (msg.type() === 'warning') report.console_warnings.push(msg.text());
  });
  page.on('pageerror', error => report.page_errors.push(String(error)));
  page.on('requestfailed', req => report.request_failures.push({url:req.url(), failure:req.failure()}));
  page.on('response', res => {
    const response = {url:res.url(), status:res.status()};
    report.requests.push(response);
    if (response.status >= 400 && response.status <= 599) report.http_errors.push(response);
  });
  await page.addInitScript(() => {
    window.__dshValidationEvents = [];
    for (const kind of ['keydown','keyup','mousedown','mouseup','click','mousemove','pointerlockchange','input','change']) {
      document.addEventListener(kind, event => {
        if (window.__dshValidationEvents.length < 500) window.__dshValidationEvents.push({
          kind, key:event.key, code:event.code, button:event.button, trusted:event.isTrusted,
          at:performance.now(), pointer_locked:Boolean(document.pointerLockElement),
        });
      }, true);
    }
  });
  const response = await page.goto(entryURL, {waitUntil:'load', timeout:30000});
  report.main_status = response?.status();
  await sleep(Math.min(plan.initial_wait_ms || 1200, 10000));
  const renderer = await page.evaluate(() => {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return {available:false};
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    const value = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return {available:true, renderer:String(value)};
  });
  report.browser.webgl = renderer;
  if (renderer.available && !/swiftshader|software|llvmpipe/i.test(renderer.renderer)) {
    throw new Error('Refusing non-software WebGL renderer: ' + renderer.renderer);
  }
  const probe = async () => {
    const state = await page.evaluate(() => ({
      title:document.title,
      text:document.body?.innerText?.slice(0, 24000),
      canvas:[...document.querySelectorAll('canvas')].map(c => ({
        width:c.width,height:c.height,client_width:c.clientWidth,client_height:c.clientHeight,
      })),
      buttons:[...document.querySelectorAll('button,[role=button],input[type=button]')].map(e => ({
        text:(e.innerText || e.value || e.getAttribute('aria-label') || '').slice(0,200),
        id:e.id, disabled:Boolean(e.disabled),
      })),
      pointer_locked:Boolean(document.pointerLockElement),
      events:window.__dshValidationEvents?.slice() || [],
    }));
    if (plan.probe_expression) {
      try {
        state.probe = await page.evaluate(expression => Function('"use strict"; return (' + expression + ')')(), plan.probe_expression);
      } catch (error) {
        state.probe_error = String(error);
      }
    }
    return state;
  };
  const pixelDelta = (a, b) => {
    const aa = PNG.sync.read(a), bb = PNG.sync.read(b);
    if (aa.width !== bb.width || aa.height !== bb.height) return {same_dimensions:false};
    let changed = 0, sum = 0;
    for (let i = 0; i < aa.data.length; i += 4) {
      let pixel = false;
      for (let j = 0; j < 3; j++) {
        const difference = Math.abs(aa.data[i+j] - bb.data[i+j]);
        sum += difference; if (difference > 8) pixel = true;
      }
      if (pixel) changed++;
    }
    return {same_dimensions:true, changed_pixels:changed,
      changed_fraction:changed/(aa.width*aa.height), mean_absolute_rgb_difference:sum/(aa.width*aa.height*3)};
  };
  const read = (obj, key) => key.split('.').reduce((value, part) => value?.[part], obj);
  const check = (spec, before, after) => {
    const a = read(before.probe, spec.path), b = read(after.probe, spec.path);
    let pass = false;
    if (spec.op === 'changed') pass = a !== undefined && b !== undefined && JSON.stringify(a) !== JSON.stringify(b);
    else if (spec.op === 'decreased') pass = typeof a === 'number' && typeof b === 'number' && b < a;
    else if (spec.op === 'increased') pass = typeof a === 'number' && typeof b === 'number' && b > a;
    else if (spec.op === 'equals') pass = b !== undefined && JSON.stringify(b) === JSON.stringify(spec.value);
    else throw new Error('Unknown check operation: ' + spec.op);
    return {...spec, before:a, after:b, pass};
  };
  const actions = plan.actions || [
    {name:'idle-control', type:'wait', ms:700},
    {name:'start-if-present', type:'start-if-present'},
    {name:'focus-game', type:'focus-game'},
    {name:'move-forward', type:'key-hold', key:'w', ms:holdMs},
    {name:'move-right', type:'key-hold', key:'d', ms:holdMs},
    {name:'primary-fire', type:'mouse-click', button:'left'},
    {name:'reload', type:'key-press', key:'r'},
    {name:'settle', type:'wait', ms:700},
  ];
  report.initial = await probe();
  await page.screenshot({path:path.join(output, 'initial.png')});
  for (let index = 0; index < actions.length; index++) {
    const action = actions[index];
    const prefix = String(index+1).padStart(2,'0') + '-' + String(action.name || action.type).replace(/[^a-zA-Z0-9_-]/g,'_');
    const before = await probe();
    const beforeImage = await page.screenshot({path:path.join(output, prefix+'-before.png')});
    const result = {action, before, started_at:new Date().toISOString()};
    let dialogHandler, dialogTimer, dialogDone, dialogDeadline;
    const pendingDialogs = [];
    try {
      if (action.expect_dialog !== undefined) {
        const expected = action.expect_dialog;
        if (action.type !== 'click' || !expected || Array.isArray(expected)
            || expected.type !== 'confirm' || typeof expected.accept !== 'boolean'
            || Object.keys(expected).some(key => !['type','accept'].includes(key))) {
          throw new Error('expect_dialog requires a click and {type:"confirm",accept:boolean}');
        }
        result.dialogs = [];
        dialogDeadline = Date.now() + 3000;
        let finishDialog;
        dialogDone = new Promise(resolve => { finishDialog = resolve; });
        dialogHandler = dialog => {
          const matches = dialog.type() === expected.type;
          const record = {type:dialog.type(), message:dialog.message(),
            decision:matches && result.dialogs.length === 0 && expected.accept ? 'accept' : 'dismiss',
            matches_expected_type:matches, handled:false};
          result.dialogs.push(record);
          const pending = (async () => {
            try {
              if (record.decision === 'accept') await dialog.accept();
              else await dialog.dismiss();
              record.handled = true;
            } catch (error) { record.error = String(error); }
            finally { finishDialog(); }
          })();
          pendingDialogs.push(pending);
        };
        page.on('dialog', dialogHandler);
      }
      if (action.type === 'wait') await sleep(Math.min(action.ms || 700,10000));
      else if (action.type === 'start-if-present') {
        const button = page.getByRole('button', {name:/start|play|开始|进入|战斗/i}).first();
        if (await button.count() && await button.isVisible()) await button.click({timeout:3000});
        else result.skipped = 'No visible start button found';
      } else if (action.type === 'focus-game') {
        const canvas = page.locator('canvas').first();
        if (await canvas.count() && await canvas.isVisible()) await canvas.click({position:{x:80,y:80},timeout:3000});
        else await page.locator('body').click({position:{x:80,y:80},timeout:3000});
      } else if (action.type === 'click') await page.locator(action.selector).first().click({timeout:3000});
      else if (action.type === 'fill' || action.type === 'select-option') {
        if (typeof action.selector !== 'string' || !action.selector.trim()
            || typeof action.value !== 'string') {
          throw new Error(action.type + ' requires a selector and a string value');
        }
        // Use strict locators: an ambiguous form selector must fail visibly.
        const field = page.locator(action.selector);
        if (action.type === 'fill') await field.fill(action.value, {timeout:3000});
        else result.selected_values = await field.selectOption({value:action.value}, {timeout:3000});
      } else if (action.type === 'reload') {
        const reloaded = await page.reload({waitUntil:'load', timeout:30000});
        result.reload_status = reloaded?.status();
      } else if (action.type === 'key-hold') {
        await page.keyboard.down(action.key);
        try { await sleep(Math.min(action.ms || holdMs,10000)); }
        finally { await page.keyboard.up(action.key); }
      } else if (action.type === 'key-press') await page.keyboard.press(action.key);
      else if (action.type === 'mouse-click') await page.mouse.click(action.x || width/2, action.y || height/2,{button:action.button || 'left'});
      else if (action.type === 'mouse-move') await page.mouse.move(action.x,action.y,{steps:action.steps || 10});
      else throw new Error('Unknown action type: '+action.type);
      if (dialogDone) {
        await Promise.race([dialogDone, new Promise((_, reject) => {
          dialogTimer = setTimeout(() => reject(new Error('Expected confirm dialog was not observed within 3000 ms')),
            Math.max(0, dialogDeadline - Date.now()));
        })]);
      }
      await sleep(Math.min(action.settle_ms ?? 150,10000));
      if (dialogHandler) {
        await Promise.all(pendingDialogs);
        if (result.dialogs.length !== 1) throw new Error('Expected exactly one confirm dialog; observed ' + result.dialogs.length);
        const observed = result.dialogs[0];
        if (!observed.matches_expected_type) throw new Error('Expected confirm dialog; observed ' + observed.type);
        if (!observed.handled) throw new Error('Failed to handle expected confirm dialog: ' + observed.error);
      }
    } catch (error) { result.error = String(error); }
    finally {
      clearTimeout(dialogTimer);
      if (dialogHandler) page.off('dialog', dialogHandler);
      await Promise.all(pendingDialogs);
    }
    result.after = await probe();
    const afterImage = await page.screenshot({path:path.join(output, prefix+'-after.png')});
    result.screenshot_delta = pixelDelta(beforeImage, afterImage);
    result.screenshot_before_sha256 = digest(beforeImage);
    result.screenshot_after_sha256 = digest(afterImage);
    result.checks = (action.checks || []).map(spec => check(spec,before,result.after));
    result.finished_at = new Date().toISOString();
    report.actions.push(result);
  }
  report.final = await probe();
  await page.screenshot({path:path.join(output,'final.png')});
  const actionChecks = report.actions.flatMap(x => x.checks || []);
  report.summary = {
    boot_pass:report.main_status === 200 && !report.page_errors.length,
    no_console_errors:!report.console_errors.length,
    no_blocked_dependencies:!report.blocked_requests.length,
    no_failed_requests:!report.request_failures.length,
    no_http_errors:!report.http_errors.length,
    action_errors:report.actions.filter(x => x.error).length,
    state_checks_total:actionChecks.length,
    state_checks_passed:actionChecks.filter(x => x.pass).length,
    specified_state_checks_pass:actionChecks.length ? actionChecks.every(x => x.pass) : null,
    default_observations_are_not_gameplay_assertions:true,
  };
  report.status = report.summary.boot_pass && report.summary.no_console_errors
    && report.summary.no_blocked_dependencies && report.summary.no_failed_requests
    && report.summary.no_http_errors
    && !report.summary.action_errors && actionChecks.every(x => x.pass) ? 'observational-checks-passed' : 'issues-observed';
} catch (error) {
  report.status = 'harness-or-page-failure';
  report.error = error.stack || String(error);
} finally {
  await context?.close().catch(()=>{});
  await browser?.close().catch(()=>{});
  await new Promise(resolve => server.close(resolve));
  report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(output,'report.json'), JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify({status:report.status, summary:report.summary, output},null,2));
process.exitCode = report.status === 'observational-checks-passed' ? 0 : 1;
