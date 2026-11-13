/* vLLM profile editor. The controller owns validation and process lifecycle. */
'use strict';
(() => {
  const root = document.querySelector('[data-launch-root]');
  if (!root) return;
  const tr = (zh, en) => window.monitorI18n.language === 'en' ? en : zh;
  const DRAFT = 'kvmem-vllm-profile-v1', TEMPLATES = 'kvmem-vllm-templates-v1';
  let info, runtime, profile = {}, draft = {}, busy = false, locked = false, activeTab = 0, timer;
  const groups = [
    ['模型与服务', 'Model & service', [
      ['model', '本地模型路径', 'Local model path', 'text', true],
      ['tokenizer', '本地分词器路径', 'Local tokenizer path', 'text'],
      ['hf_config_path', '模型配置路径', 'Model configuration path', 'text'],
      ['model_format', '模型格式', 'Model format', ['safetensors', 'gguf']],
      ['served_model_name', '对外模型名', 'Served model name', 'text'],
      ['host', '监听地址', 'Listen address', ['127.0.0.1', '0.0.0.0']],
      ['port', 'API 端口', 'API port', 'number', false, 1, 65535],
    ]],
    ['缓存与历史', 'Cache & history', [
      ['kvmem.select_budget', '历史 KV 预算 K（tokens）', 'History KV budget K (tokens)', 'number', false, 0],
      ['kvmem.gen_budget', '生成预留 G（tokens）', 'Generation reserve G (tokens)', 'number', false, 1],
      ['kvmem.cpu_read_cache_bytes', 'CPU 读取缓存（GiB）', 'CPU read cache (GiB)', 'gib', false, 0],
      ['kvmem.cpu_history_cache_bytes_per_request', '每请求 CPU 历史（GiB）', 'CPU history per request (GiB)', 'gib', false, 0],
      ['kvmem.session_checkpoint_bytes', '会话检查点（GiB）', 'Session checkpoints (GiB)', 'gib', false, 0],
      ['kvmem.session_capacity', '保留会话数', 'Retained session capacity', 'number', false, 1],
      ['kvmem.repository_blocks', '历史仓库块数', 'History repository blocks', 'number', false, 1],
      ['kvmem.block_tokens', '每块 token 数', 'Tokens per block', 'number', false, 1],
      ['kvmem.sink_blocks', '起始保留块数', 'Sink blocks', 'number', false, 0],
      ['kvmem.recent_blocks', '最近保留块数', 'Recent blocks', 'number', false, 0],
    ]],
    ['硬件与执行', 'Hardware & execution', [
      ['max_model_len', '逻辑上下文上限', 'Logical context limit', 'number', false, 1],
      ['max_num_seqs', '最大并发数', 'Maximum concurrent requests', 'number', false, 1],
      ['gpu_memory_utilization', '显存使用比例', 'GPU memory utilization', 'fraction', false, 0.01, 1],
      ['max_num_batched_tokens', '批处理 token 预算', 'Batch token budget', 'number', false, 1],
      ['prefill_chunk_tokens', '预填充分块 tokens（0 关闭）', 'Prefill chunk tokens (0 disables)', 'number', false, 0],
      ['query_replay_budget', 'Query replay 预算', 'Query replay budget', 'number', false, 1],
      ['execution_mode', '执行模式', 'Execution mode', ['eager', 'full_graph', 'compiled_graph']],
    ]],
    ['推测解码', 'Speculative decoding', [
      ['draft_method', '草稿方式', 'Draft method', ['dflash', 'mtp']],
      ['draft_model', '草稿模型路径', 'Draft model path', 'text'],
      ['draft_tokens', '草稿 token 数', 'Draft tokens', 'number', false, 1, 7],
      ['experimental_dflash_query_planning', 'DFlash Query planning', 'DFlash query planning', 'checkbox'],
    ]],
  ];
  const node = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const get = (obj, path) => path.split('.').reduce((v, k) => v?.[k], obj);
  function put(obj, path, value) { const keys = path.split('.'); let at = obj; for (const key of keys.slice(0, -1)) at = at[key] ??= {}; if (value === undefined) delete at[keys.at(-1)]; else at[keys.at(-1)] = value; }
  function read(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
  function clean(value) { const result = structuredClone(value); for (const key of ['api_key', 'api_keys']) delete result[key]; return result; }
  function store(key, value) { localStorage.setItem(key, JSON.stringify(clean(value))); }
  function message(text, error = false) { const e = root.querySelector('[data-launch-message]'); if (e) { e.textContent = text; e.classList.toggle('l-error', error); } }
  async function api(path, data) {
    const response = await fetch('/api/launcher/' + path, { method: data === undefined ? 'GET' : 'POST', cache: 'no-store', headers: data === undefined ? {} : { 'Content-Type': 'application/json', 'X-KVMem-Token': info.token }, body: data === undefined ? undefined : JSON.stringify(data), signal: AbortSignal.timeout(60000) });
    const result = await response.json(); if (!response.ok) throw Error(typeof result.error === 'string' ? result.error : result.error?.message || 'HTTP ' + response.status); return result;
  }
  function collect() {
    const result = structuredClone(profile);
    root.querySelectorAll('[data-profile]').forEach(input => {
      let value = input.type === 'checkbox' ? input.checked : input.value;
      if (input.type !== 'checkbox' && value === '') value = undefined;
      else if (input.type === 'number') value = Number(value) * (input.dataset.unit === 'gib' ? 1024 ** 3 : 1);
      put(result, input.dataset.profile, value);
    });
    return result;
  }
  function saveDraft() {
    if (locked) return;
    draft = collect();
    try { store(DRAFT, draft); root.querySelector('[data-save-state]').textContent = tr('草稿已保存到此浏览器', 'Draft saved in this browser'); }
    catch { root.querySelector('[data-save-state]').textContent = tr('草稿保存失败，请导出备份', 'Could not save draft; export a backup'); }
  }
  function field(parent, spec) {
    const [key, zh, en, type, required, min, max] = spec;
    const label = node('label', 'l-field'); label.append(node('span', '', tr(zh, en)));
    const input = node(Array.isArray(type) ? 'select' : 'input'); input.dataset.profile = key;
    input.setAttribute('aria-label', tr(zh, en));
    if (Array.isArray(type)) {
      const value = get(profile, key); input.append(new Option(tr('使用默认值', 'Use default'), ''));
      for (const item of [...new Set([...type, ...(value && !type.includes(value) ? [value] : [])])]) input.append(new Option(item, item));
    } else { input.type = type === 'gib' || type === 'fraction' ? 'number' : type; if (type === 'gib') input.dataset.unit = 'gib'; if (type === 'gib' || type === 'fraction') input.step = 'any'; if (min !== undefined) input.min = min; if (max !== undefined) input.max = max; }
    const value = get(profile, key);
    if (type === 'checkbox') input.checked = value === true;
    else input.value = value == null ? '' : type === 'gib' ? value / 1024 ** 3 : value;
    input.required = !!required; input.disabled = locked; input.oninput = saveDraft; input.onchange = () => { saveDraft(); dependencies(); };
    label.append(input); if (key === 'model' || key === 'tokenizer' || key === 'draft_model') label.classList.add('l-wide'); parent.append(label);
  }
  function dependencies() {
    const method = root.querySelector('[data-profile="draft_method"]')?.value;
    const input = root.querySelector('[data-profile="draft_model"]');
    if (input) { input.disabled = locked || method === 'mtp'; input.closest('label').title = method === 'mtp' ? tr('MTP 使用模型内置草稿组件', 'MTP uses the model’s built-in draft component') : ''; }
  }
  function button(text, action, cls = '') { const b = node('button', 's-button ' + cls, text); b.type = 'button'; b.onclick = action; return b; }
  function dialog(title) {
    const d = node('dialog', 'l-dialog'), heading = node('div', 'l-heading');
    const close = button(tr('关闭', 'Close'), () => d.close()); heading.append(node('h2', 's-title', title), close); d.append(heading); root.append(d); d.addEventListener('close', () => d.remove(), { once: true }); d.showModal(); return d;
  }
  function download(value) { const url = URL.createObjectURL(new Blob([JSON.stringify(clean(value), null, 2) + '\n'], { type: 'application/json' })); const a = node('a'); a.href = url; a.download = 'kvmem-vllm-profile.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  function editJson() {
    const d = dialog(tr('完整配置 JSON', 'Complete profile JSON'));
    d.append(node('p', 'l-muted', tr('保留当前配置中的所有高级选项。应用只更新草稿，启动前仍会校验。', 'All advanced settings are retained. Apply updates the draft; launch still validates it.')));
    const area = node('textarea', 'l-profile-json'); area.rows = 20; area.value = JSON.stringify(collect(), null, 2); area.spellcheck = false; area.setAttribute('aria-label', 'Profile JSON'); area.disabled = locked;
    const error = node('p', 'l-error'); d.append(area, error);
    if (!locked) d.append(button(tr('应用到草稿', 'Apply to draft'), () => { try { const next = JSON.parse(area.value); if (!next || typeof next !== 'object' || Array.isArray(next)) throw Error(tr('配置必须为 JSON 对象', 'Profile must be a JSON object')); draft = clean(next); store(DRAFT, draft); render(draft); d.close(); } catch (e) { error.textContent = e.message; } }));
  }
  function templates(parent) {
    parent.append(node('h2', 's-title', tr('我的模板', 'My templates')), node('p', 'l-muted', tr('保存在此浏览器。套用模板后仍需手动启动。', 'Stored in this browser. Apply a template, then start the service.')));
    const items = read(TEMPLATES, []);
    if (!items.length) parent.append(node('p', 'l-muted', tr('暂无模板', 'No saved templates')));
    for (const item of items) {
      const row = node('div', 'l-preset'), apply = button(item.name, () => { draft = structuredClone(item.profile); store(DRAFT, draft); render(draft); }); apply.disabled = locked;
      const remove = button(tr('删除', 'Delete'), () => { if (!confirm(tr('删除此模板？', 'Delete this template?'))) return; store(TEMPLATES, items.filter(x => x.id !== item.id)); render(collect()); }); remove.disabled = locked; row.append(apply, remove); parent.append(row);
    }
    const save = button(tr('另存为模板', 'Save as template'), () => {
      const d = dialog(tr('另存为模板', 'Save as template')), form = node('form'), name = node('input'); name.placeholder = tr('模板名称', 'Template name'); name.required = true; name.maxLength = 100; name.setAttribute('aria-label', tr('模板名称', 'Template name'));
      const submit = button(tr('保存', 'Save')); submit.type = 'submit'; form.append(name, submit); d.append(form); form.onsubmit = e => { e.preventDefault(); try { const items = read(TEMPLATES, []); items.push({ id: crypto.randomUUID(), name: name.value.trim(), profile: clean(collect()) }); store(TEMPLATES, items); d.close(); render(collect()); } catch (e) { message(e.message, true); } }; name.focus();
    }); save.disabled = locked; parent.append(save);
  }
  function render(next) {
    profile = structuredClone(next || {}); root.replaceChildren();
    const heading = node('div', 'l-heading'), intro = node('div', 'l-intro');
    intro.append(node('h1', '', locked ? tr('当前启动参数', 'Current launch parameters') : tr('启动配置', 'Launch configuration')), node('p', 'l-muted', 'vLLM · KVMem'));
    const saved = node('span', 'l-save-state'); saved.dataset.saveState = ''; saved.textContent = locked ? tr('服务运行中，停止后可编辑', 'Stop the service to edit its configuration') : tr('草稿保存在此浏览器', 'Drafts stay in this browser'); intro.append(saved);
    const actions = node('div', 'l-actions'), preview = button(tr('检查配置', 'Check configuration'), previewProfile); preview.dataset.action = 'preview'; preview.disabled = locked || busy;
    actions.append(preview, button(tr('导出配置', 'Export profile'), () => download(collect()))); heading.append(intro, actions);
    const notice = node('p', 'l-notice'); notice.dataset.launchMessage = ''; notice.setAttribute('role', 'status');
    const layout = node('div', 'l-layout'), main = node('div', 'l-main'), side = node('aside', 'l-side'), form = node('form'); form.onsubmit = e => e.preventDefault();
    const section = node('section', 's-card'), tabs = node('div', 'l-tabs'); tabs.setAttribute('role', 'tablist');
    groups.forEach((group, index) => { const b = button(tr(group[0], group[1]), () => { activeTab = index; render(collect()); }); b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', String(index === activeTab)); tabs.append(b); });
    const fields = node('div', 'l-fields'); fields.setAttribute('role', 'tabpanel'); groups[activeTab][2].forEach(spec => field(fields, spec));
    section.append(tabs, fields); form.append(section); main.append(form);
    const advanced = node('section', 's-card'); advanced.append(node('h2', 's-title', tr('高级配置', 'Advanced configuration')), node('p', 'l-muted', tr('未列出的原生配置会完整保留。K 和 G 是驻留 KV 与生成空间预算，逻辑上下文另行设置。', 'Unlisted native settings are retained. K and G budget resident KV and generation workspace; logical context is configured separately.')), button(tr('查看 / 编辑完整 JSON', 'View / edit complete JSON'), editJson));
    const file = node('input'); file.type = 'file'; file.accept = '.json,application/json'; file.hidden = true; file.onchange = async () => { try { const f = file.files[0]; if (!f) return; if (f.size > 1024 * 1024) throw Error(tr('配置文件过大', 'Profile file too large')); const value = JSON.parse(await f.text()); const next = value.profile || value; if (!next || typeof next !== 'object' || Array.isArray(next)) throw Error(tr('配置必须为 JSON 对象', 'Profile must be a JSON object')); draft = clean(next); store(DRAFT, draft); render(draft); } catch (e) { message(e.message, true); } };
    const importButton = button(tr('导入配置', 'Import profile'), () => file.click()); importButton.disabled = locked; advanced.append(importButton, file); main.append(advanced);
    const service = node('section', 's-card'), badge = node('span', 's-badge'); badge.dataset.runtimeState = ''; service.append(node('h2', 's-title', tr('服务状态', 'Service status')), badge);
    const state = node('p', 'l-muted'); state.dataset.serviceDetail = ''; service.append(state);
    const progress = node('div', 'l-vllm-progress'); progress.dataset.progress = ''; progress.setAttribute('role', 'progressbar'); progress.setAttribute('aria-label', tr('正在加载模型', 'Loading model')); progress.hidden = true; service.append(progress);
    const controls = node('div', 'l-actions'); for (const [action, zh, en] of [['start', '启动服务', 'Start service'], ['stop', '停止服务', 'Stop service']]) { const b = button(tr(zh, en), () => act(action), action === 'start' ? 'l-primary' : ''); b.dataset.action = action; controls.append(b); } service.append(controls);
    const log = button(tr('查看后台日志', 'View backend logs'), () => document.querySelector('[data-page="logs"]').click()); service.append(log);
    const presets = node('section', 's-card l-presets'); templates(presets); side.append(service, presets); layout.append(main, side); root.append(heading, notice, layout); dependencies(); paintStatus();
  }
  function paintStatus() {
    if (!runtime) return;
    const labels = { stopped: ['未启动', 'Stopped'], starting: ['加载中', 'Loading'], ready: ['已就绪', 'Ready'], exited: ['已退出', 'Exited'], error: ['启动失败', 'Failed'] };
    const badge = root.querySelector('[data-runtime-state]'); if (!badge) return;
    badge.textContent = tr(...(labels[runtime.state] || [runtime.state, runtime.state]));
    const detail = root.querySelector('[data-service-detail]'); detail.textContent = runtime.error || (runtime.url ? runtime.url : runtime.state === 'starting' ? tr('正在加载，请查看后台日志。', 'Loading; see backend logs for progress.') : tr('配置编辑不会改变正在运行的服务。', 'Editing a draft does not change the running service.'));
    root.querySelector('[data-progress]').hidden = !(busy || runtime.state === 'starting');
    root.querySelectorAll('[data-action]').forEach(b => b.disabled = busy || (b.dataset.action === 'stop' ? !['starting', 'ready'].includes(runtime.state) : locked));
    const chat = document.querySelector('[data-chat-link]'); chat.href = '/chat/'; chat.setAttribute('aria-disabled', String(runtime.state !== 'ready'));
  }
  function updateStatus(status) {
    if (!status) return;
    status = { ...status, state: ({ loading: 'starting', failed: 'error' })[status.state] || status.state };
    const wasLocked = locked, previous = runtime?.instance; runtime = status; locked = ['starting', 'ready'].includes(status.state);
    if (info && (locked !== wasLocked || (locked && previous !== status.instance))) { if (locked && !wasLocked) draft = collect(); render(locked ? status.active_profile || draft : draft); }
    paintStatus(); window.dispatchEvent(new CustomEvent('kvmem-runtime-status', { detail: status }));
  }
  async function previewProfile() {
    if (locked || busy || !root.querySelector('form').reportValidity()) return;
    const d = dialog(tr('检查配置', 'Check configuration')), result = node('pre', 'l-result'); result.textContent = tr('正在校验…', 'Validating…'); d.append(result);
    try { const value = await api('preview', { profile: collect() }); result.textContent = JSON.stringify(value, null, 2); d.append(node('p', 'l-muted', tr('校验不加载模型。显存需求和推理结果需启动后确认。', 'Validation does not load the model. Memory use and inference are verified after startup.'))); } catch (e) { result.textContent = e.message; result.classList.add('l-error'); }
  }
  async function act(action) {
    if (busy || (action === 'start' && (locked || !root.querySelector('form').reportValidity()))) return;
    const payload = action === 'start' ? { profile: collect() } : {};
    if (action === 'start') saveDraft(); busy = true; paintStatus();
    try { updateStatus(await api(action, payload)); message(action === 'start' ? tr('启动已提交，正在加载模型。', 'Launch submitted; the model is loading.') : tr('服务已停止。', 'Service stopped.')); }
    catch (e) { message(e.message, true); try { updateStatus(await api('status')); } catch {} }
    finally { busy = false; paintStatus(); }
  }
  async function poll() { try { if (!busy) updateStatus(await api('status')); } catch (e) { message(tr('暂时无法读取服务状态：', 'Could not read service state: ') + e.message, true); } timer = setTimeout(poll, document.hidden ? 5000 : 2000); }
  window.addEventListener('kvmem-language-change', () => { if (info) render(collect()); });
  window.addEventListener('pagehide', () => clearTimeout(timer));
  async function init() {
    try { info = await api('info'); draft = read(DRAFT, info.profile || {}); render(draft); updateStatus(info.status); if (!location.hash && !location.pathname.startsWith('/chat') && !locked) document.querySelector('[data-page="config"]').click(); poll(); }
    catch (e) { root.append(node('section', 's-card', tr('无法连接本地控制台：', 'Cannot connect to the local console: ') + e.message)); }
  }
  init();
})();
