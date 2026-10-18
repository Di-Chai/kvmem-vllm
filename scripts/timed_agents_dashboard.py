"""Read-only live view for fixed-window agent measurements.

Usage: python scripts/timed_agents_dashboard.py --run-root EXPERIMENT --port 50123
No model, benchmark, or GPU process is started or controlled by this server.
"""
import argparse
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class JsonLines:
    """Consume complete new lines once, retaining partial trailing writes."""
    def __init__(self, path, consume):
        self.path, self.consume = Path(path), consume
        self.offset = 0
        self.identity = None
        self.partial = b''
        self.invalid_lines = 0
        self.reset_detected = False
        self.backlog_bytes = 0

    def update(self):
        try:
            with self.path.open('rb') as stream:
                stat = stream.seek(0, 2)
                identity = (self.path.stat().st_dev, self.path.stat().st_ino)
                if self.identity is not None and (identity != self.identity or stat < self.offset):
                    # Keep existing aggregates visible, but never silently mix a
                    # restarted producer with their previous cumulative values.
                    self.reset_detected = True
                    return
                self.identity = identity
                stream.seek(self.offset)
                block = stream.read(8 * 1024 * 1024)
                self.offset = stream.tell()
                self.backlog_bytes = max(0, stat-self.offset)
        except OSError:
            return
        lines = (self.partial+block).split(b'\n')
        self.partial = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected JSON object')
                self.consume(row)
            except (ValueError, TypeError, KeyError, OverflowError):
                self.invalid_lines += 1


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def window_times(window):
    return (window.get('t0_monotonic', window.get('begin')),
            window.get('t1_monotonic', window.get('end')))


class Case:
    def __init__(self, path):
        self.path = Path(path)
        self.window = {}
        self.tokens = defaultdict(int)
        self.accepted_tokens = 0
        self.recent_tokens = deque()
        self.latest_event = None
        self.phases = {}
        self.events = JsonLines(self.path/'events.jsonl', self.event)

    def event(self, row):
        timestamp = row.get('observed_monotonic', row.get('monotonic'))
        if finite(timestamp):
            self.latest_event = max(self.latest_event or timestamp, timestamp)
        if row.get('event') == 'agent_phase':
            self.phases[str(row['agent'])] = {k: row.get(k) for k in ('phase', 'epoch', 'turn')}
        if row.get('event') != 'stream_chunk' or row.get('kind') != 'workload':
            return
        begin, end = window_times(self.window)
        count = row.get('accepted_tokens', len(row.get('token_ids', [])))
        if (finite(timestamp) and finite(begin) and finite(end) and begin <= timestamp < end
                and type(count) is int and count >= 0):
            # Only scalar counts are retained; output token IDs are discarded.
            self.tokens[int(math.floor(timestamp-begin))] += count
            self.accepted_tokens += count
            self.recent_tokens.append((timestamp, count))
            while self.recent_tokens and self.recent_tokens[0][0] < timestamp-10.:
                self.recent_tokens.popleft()

    def update(self):
        self.window = read_json(self.path/'window.json')
        self.events.update()


class Arm:
    def __init__(self, path):
        self.path = Path(path)
        self.gpu = []
        self.public = []
        self.gpu_reader = JsonLines(self.path/'gpu_samples.jsonl', self.gpu.append)
        self.work_reader = JsonLines(self.path/'server/capture/public_work.jsonl', self.public.append)
        self.cases = {}

    def update(self):
        self.gpu_reader.update()
        self.work_reader.update()


def public_interval(rows, begin, end):
    valid = [r for r in rows if finite(r.get('at')) and r['at'] <= end and 'submitted' in r]
    baseline = max((r for r in valid if r['at'] <= begin), key=lambda r:r['at'], default=None)
    latest = max(valid, key=lambda r:r['at'], default=None)
    if baseline is None or latest is None or latest['at'] <= baseline['at']:
        return {'available': False}
    if baseline.get('counter_id') != latest.get('counter_id'):
        return {'available': False, 'counter_reset': True}
    names = ('public_prefill_tokens', 'public_batches', 'prefill_request_batches', 'decode_request_batches')
    try:
        delta = {k: latest['submitted'][k]-baseline['submitted'][k] for k in names}
        if any(v < 0 for v in delta.values()):
            return {'available': False, 'counter_reset': True}
    except (KeyError, TypeError):
        return {'available': False}
    seconds = latest['at']-baseline['at']
    batches = delta['public_batches']
    return dict(available=True, observed_seconds=seconds,
                submitted_public_prefill_tokens_per_second=delta['public_prefill_tokens']/seconds,
                prefill_requests_per_batch=delta['prefill_request_batches']/batches if batches else None,
                decode_requests_per_batch=delta['decode_request_batches']/batches if batches else None,
                batches=batches, last_public_batch=latest.get('last_public_batch'),
                scope='已提交 public prefill；异步提交观察区间，非精确 GPU 完成时间。批次数为执行频次，非时间加权并发。')


class Dashboard:
    def __init__(self, root, clock=time.monotonic, wall_clock=time.time):
        self.root = Path(root).resolve()
        self.clock, self.wall_clock = clock, wall_clock
        self.arms = {}
        self.lock = threading.Lock()

    def active(self, status, manifest):
        pair, arm = status.get('pair'), status.get('arm')
        if type(pair) is int and arm in ('A0', 'P1'):
            path = self.root/f'pair-{pair:02d}-{arm}'
        else:
            choices = [p for p in self.root.glob('pair-*-*') if p.is_dir()
                       and re.fullmatch(r'pair-\d+-(A0|P1)', p.name)]
            path = max(choices, key=lambda p:p.stat().st_mtime, default=None)
        if path is None:
            return None, None
        mode = status.get('mode')
        modes = manifest.get('workload', {}).get('modes', ['saturated'])
        if mode not in ('saturated', 'tools'):
            existing = [path/name for name in modes if (path/name).is_dir()]
            mode = max(existing, key=lambda p:p.stat().st_mtime).name if existing else modes[0]
        return path, mode

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        now, wall = self.clock(), self.wall_clock()
        status = read_json(self.root/'status.json')
        manifest = read_json(self.root/'manifest.json')
        result = dict(observed_wall=wall, run_name=self.root.name, status=status,
                      settings=manifest.get('workload', {}), phase='等待实验',
                      arm=None, pair=None, mode=None, remaining_seconds=None,
                      output_tokens=0, output_tokens_per_second=None,
                      recent_output_tokens_per_second=None, series=[], agents={},
                      gpu=None, public_work={'available': False}, warnings=[], comparison=read_json(self.root/'comparison.json'))
        pipeline = read_json(self.root/'pipeline-status.json')
        qualification_phase = pipeline.get('phase')
        if (qualification_phase in ('correctness', 'correctness-hot', 'correctness-pressure')
                and pipeline.get('status') in ('running', 'failed')):
            qualification = read_json(self.root/qualification_phase/'status.json')
            failed = (pipeline['status'] == 'failed' or qualification.get('status') == 'failed')
            scope = {'correctness': '恢复', 'correctness-hot': '热复用',
                     'correctness-pressure': '压力恢复'}[qualification_phase]
            label = scope+('校验失败' if failed else '正确性校验')
            result.update(phase=label+' · '+str(qualification.get('arm', qualification.get('status', '准备中'))),
                          qualification=qualification, status=pipeline)
            return result
        path, mode = self.active(status, manifest)
        if path is None:
            result['phase'] = {'prepared': '等待启动', 'loading': '模型加载', 'failed': '实验异常',
                               'waiting_for_gpu': '等待 GPU 空闲'}.get(status.get('status'), '等待实验')
            return result
        arm = self.arms.setdefault(path, Arm(path))
        arm.update()
        case = arm.cases.setdefault(mode, Case(path/mode))
        case.update()
        progress = read_json(path/mode/'progress.json')
        completed = read_json(path/mode/'results.json')
        match = re.fullmatch(r'pair-(\d+)-(A0|P1)', path.name)
        result.update(arm=match[2], pair=int(match[1]), mode=mode,
                      agents=progress.get('agents') or case.phases,
                      completed_requests=progress.get('completed_requests', 0),
                      final_results=completed if completed.get('status') == 'completed' else None)
        readers = [case.events, arm.gpu_reader, arm.work_reader]
        if any(r.reset_detected for r in readers):
            result['warnings'].append('检测到记录文件重置，当前统计不可用于验收。')
        if any(r.invalid_lines for r in readers):
            result['warnings'].append('部分监控记录暂不可解析。')
        if any(r.backlog_bytes for r in readers):
            result['warnings'].append('正在追赶监控记录，当前累计值尚未完整。')
        gpu = max((r for r in arm.gpu if finite(r.get('at'))), key=lambda r:r['at'], default=None)
        if gpu is not None:
            result['gpu'] = dict(utilization_percent=gpu.get('utilization'), memory_mib=gpu.get('memory_mib'),
                                 sample_age_seconds=max(0., now-gpu['at']))
        begin, end = window_times(case.window)
        if not (finite(begin) and finite(end) and end > begin):
            result['phase'] = '模型加载' if status.get('status') == 'loading' else '预热与历史发布'
            if status.get('status') in ('failed', 'interrupted'):
                result['phase'] = '实验异常 / 已停止'
            return result
        stop = min(max(now, begin), end)
        elapsed = stop-begin
        result.update(window=case.window, elapsed_seconds=elapsed,
                      remaining_seconds=max(0., end-now), output_tokens=case.accepted_tokens,
                      output_tokens_per_second=case.accepted_tokens/elapsed if elapsed else None,
                      phase='测量中' if now < end else '收尾与清理')
        if completed.get('status') == 'completed':
            result['phase'] = '测量完成'
        elif status.get('status') in ('failed', 'interrupted'):
            result['phase'] = '实验异常 / 已停止'
        if case.latest_event is not None:
            result['event_age_seconds'] = max(0., now-case.latest_event)
        span = min(10., elapsed)
        lower = stop-span
        while case.recent_tokens and case.recent_tokens[0][0] < lower:
            case.recent_tokens.popleft()
        recent = sum(count for timestamp, count in case.recent_tokens if lower <= timestamp < stop)
        result['recent_output_tokens_per_second'] = recent/span if span else None
        result['public_work'] = public_interval(arm.public, max(begin, stop-10.), stop)
        gpu_rows = [r for r in arm.gpu if finite(r.get('at')) and finite(r.get('utilization'))]
        work = [r for r in arm.public if finite(r.get('at')) and 'submitted' in r]
        gpu_index = work_index = 0
        for bucket in range(int(math.ceil(elapsed))):
            point_time = begin+bucket
            seconds = min(1., elapsed-bucket)
            while gpu_index+1 < len(gpu_rows) and gpu_rows[gpu_index+1]['at'] <= point_time:
                gpu_index += 1
            sampled = gpu_rows[gpu_index] if gpu_rows else None
            utilization = (sampled['utilization'] if sampled and 0 <= point_time-sampled['at'] <= 3. else None)
            while work_index+1 < len(work) and work[work_index+1]['at'] <= point_time:
                work_index += 1
            prefill = None
            if work_index+1 < len(work):
                left, right = work[work_index], work[work_index+1]
                width = right['at']-left['at']
                if (left['at'] <= point_time <= right['at'] and 0 < width <= 3.
                        and left.get('counter_id') == right.get('counter_id')):
                    delta = right['submitted'].get('public_prefill_tokens', 0)-left['submitted'].get('public_prefill_tokens', 0)
                    if delta >= 0:
                        prefill = delta/width
            result['series'].append(dict(second=bucket, output_tokens_per_second=case.tokens[bucket]/seconds,
                                         gpu_utilization_percent=utilization,
                                         submitted_public_prefill_tokens_per_second=prefill))
        return result


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KVMem 固定窗口测试</title><style>
:root{color-scheme:dark;font:16px system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;background:#0b1418;color:#dce9ee}body{max-width:1240px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}p{color:#9eb1ba;line-height:1.6;margin:8px 0}.top{display:flex;justify-content:space-between;align-items:center;gap:20px}.badge{padding:10px 17px;background:#203d44;border-radius:20px;color:#82e0c4;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}.card,.panel{background:#121f25;border:1px solid #263840;border-radius:18px;padding:21px}.label{font-size:14px;color:#9eb1ba}.value{font-size:32px;font-weight:650;margin:9px 0 3px}.small{font-size:13px;color:#8da4af}.panel{margin-bottom:18px}canvas{width:100%;height:300px;display:block;margin:16px 0}.legend{display:flex;gap:25px;flex-wrap:wrap}.legend span:before{content:"";display:inline-block;width:20px;height:3px;margin:0 7px 4px 0;background:var(--c)}table{width:100%;text-align:left;border-collapse:collapse}td,th{padding:10px 6px;border-bottom:1px solid #263840}th{color:#9eb1ba;font-weight:500}.warnings{color:#ffcc88}.bar{height:5px;background:#24363f;border-radius:4px;margin-top:16px}.bar>div{height:5px;background:#77d9bd;border-radius:4px}#connection{color:#92a5af;font-size:13px}@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}body{padding:15px}.top{align-items:flex-start}h1{font-size:22px}.value{font-size:26px}}</style>
<div class="top"><div><h1>KVMem 多 Agent 性能测试</h1><p id="identity">读取实验状态…</p></div><div class="badge" id="phase">等待实验</div></div><div class="bar"><div id="bar" style="width:0"></div></div>
<div class="cards"><div class="card"><div class="label">窗口有效输出吞吐</div><div class="value" id="mean">—</div><div class="small">token/s · 窗口累计平均</div></div><div class="card"><div class="label">最近 10 秒输出</div><div class="value" id="recent">—</div><div class="small">token/s · 包含所有停顿</div></div><div class="card"><div class="label">GPU 利用率</div><div class="value" id="gpu">—</div><div class="small" id="gpuage">等待采样</div></div><div class="card"><div class="label">测量剩余</div><div class="value" id="remaining">—</div><div class="small" id="count">有效输出 0 token</div></div></div>
<div class="panel"><div class="legend"><span style="--c:#70b5ff">Decode 输出 token/s · 左轴</span><span style="--c:#79d8bc">GPU 利用率 % · 右轴</span></div><canvas id="chart"></canvas><p class="small">按一秒统计。只计测量窗口内收到的有效输出，预热与清理输出不计入；MTP 一次输出事件可包含多个 token。GPU 采样缺失时保留空档。</p></div>
<div class="panel"><div class="legend"><span style="--c:#e4bb76">已提交 public prefill · token/s</span></div><canvas id="prefill" style="height:180px"></canvas><p class="small">来自实际 public 批次的累计提交计数差分，按观察区间显示；不包含恢复的历史 KV。此曲线不表示精确 GPU 完成时间。</p><div id="batch" class="small">等待批次统计</div></div>
<div class="panel"><div class="label">Agent 进度</div><table><thead><tr><th>Agent</th><th>当前状态</th><th>任务循环</th><th>回合</th></tr></thead><tbody id="agents"></tbody></table><p id="warnings" class="warnings"></p><p id="connection"></p></div>
<script>
const $=id=>document.getElementById(id),num=(v,n=1)=>v==null?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});let last=null;
const phases={initializing:'初始化',warming:'预热',warmup_barrier:'等待开始',awaiting_first_token:'等待首 token',streaming:'生成中',tool_wait:'工具等待',ready:'等待提交',cleanup:'清理当前任务',stopped:'本轮已收尾'};
function draw(id,rows,main,second=null){let canvas=$(id),r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;let c=canvas.getContext('2d');c.scale(d,d);let w=r.width,h=r.height,L=52,R=second?45:16,T=17,B=30,pw=w-L-R,ph=h-T-B;let vals=rows.map(x=>x[main]).filter(x=>x!=null),max=Math.max(1,...vals)*1.08, duration=Math.max(1,rows.length);c.font='12px system-ui';c.lineWidth=1;
for(let i=0;i<=4;i++){let y=T+ph*i/4;c.strokeStyle='#2a3c45';c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.fillStyle='#91a8b3';c.textAlign='right';c.fillText(num(max*(1-i/4),0),L-9,y+4);if(second){c.textAlign='left';c.fillText(String(100-i*25),w-R+9,y+4)}}
function line(key,color,m){c.strokeStyle=color;c.lineWidth=2;let on=false;c.beginPath();for(let i=0;i<rows.length;i++){let v=rows[i][key];if(v==null){on=false;continue}let x=L+rows[i].second/duration*pw,y=T+ph*(1-v/m);if(on)c.lineTo(x,y);else c.moveTo(x,y);on=true}c.stroke()}
line(main,main.includes('prefill')?'#e4bb76':'#70b5ff',max);if(second)line(second,'#79d8bc',100);c.fillStyle='#91a8b3';c.textAlign='left';c.fillText('0 秒',L,h-6);c.textAlign='right';c.fillText(num(duration,0)+' 秒',w-R,h-6)}
function render(s){last=s;$('identity').textContent=s.run_name+' · '+(s.arm?'第 '+(s.pair+1)+' 对 / '+s.arm+' / '+(s.mode==='tools'?'工具等待负载':'持续供给负载'):'等待启动');$('phase').textContent=s.phase;$('mean').textContent=num(s.output_tokens_per_second);$('recent').textContent=num(s.recent_output_tokens_per_second);$('gpu').textContent=s.gpu?num(s.gpu.utilization_percent,0)+'%':'—';$('gpuage').textContent=s.gpu?'采样距今 '+num(s.gpu.sample_age_seconds)+' 秒':'等待采样';$('remaining').textContent=s.remaining_seconds==null?'—':num(Math.ceil(s.remaining_seconds),0)+' 秒';$('count').textContent='有效输出 '+num(s.output_tokens,0)+' token';let total=s.window?.duration_seconds||300;$('bar').style.width=Math.min(100,100*(s.elapsed_seconds||0)/total)+'%';draw('chart',s.series,'output_tokens_per_second','gpu_utilization_percent');draw('prefill',s.series,'submitted_public_prefill_tokens_per_second');let b=s.public_work;$('batch').textContent=b.available?'最近观察区间：每批 decode 请求 '+num(b.decode_requests_per_batch,2)+'，prefill 请求 '+num(b.prefill_requests_per_batch,2)+'；已提交 public prefill '+num(b.submitted_public_prefill_tokens_per_second,0)+' token/s。':'等待批次统计';let tbody=$('agents');tbody.replaceChildren();Object.entries(s.agents).sort((a,b)=>Number(a[0])-Number(b[0])).forEach(([id,a])=>{let tr=document.createElement('tr');[String(Number(id)+1),phases[a.phase]||a.phase,(a.epoch??0)+1,(a.turn??0)+1].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td)});tbody.appendChild(tr)});$('warnings').textContent=s.warnings.join(' ');$('connection').textContent='实时更新 · '+new Date(s.observed_wall*1000).toLocaleTimeString('zh-CN')+(s.event_age_seconds>10?' · 最近负载活动 '+num(s.event_age_seconds,0)+' 秒前':'')}
async function poll(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json())}catch(e){$('connection').textContent='连接暂不可用，正在重试';}finally{setTimeout(poll,1000)}}window.addEventListener('resize',()=>{if(last)render(last)});poll();</script></html>'''


def serve(root, host='127.0.0.1', port=50123):
    dashboard = Dashboard(root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
            elif path == '/api/status':
                try:
                    payload = dashboard.snapshot()
                    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                except Exception:
                    # A partially published producer record must not turn the
                    # loading page into an unusable 500 response.
                    body = json.dumps(dict(phase='等待监控记录', run_name=dashboard.root.name,
                        observed_wall=time.time(), output_tokens=0, output_tokens_per_second=None,
                        recent_output_tokens_per_second=None, remaining_seconds=None, agents={},
                        gpu=None, public_work={'available': False}, series=[],
                        warnings=['监控读取暂不可用，正在重试。']), ensure_ascii=False).encode()
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50123)
    args = parser.parse_args()
    if not 0 < args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    serve(args.run_root, args.host, args.port)


if __name__ == '__main__':
    main()
