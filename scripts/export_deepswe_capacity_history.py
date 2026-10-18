"""Restore exact recorded DeepSWE prompts for an optional capacity diagnostic.

No inference, tools, KV files or synthetic padding. Full per-turn token hashes
must match the recorded bridge plans, including accepted output IDs. Run CPU
replay outside formal GPU measurement windows.
"""
import argparse
import array
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def select_turns(run, count, minimum, per_task):
    state = read(run/'run.json')
    if state['status'] != 'window_complete' or state.get('cleanup_errors') != []:
        raise ValueError('Export only a completed, cleaned fixed window')
    rows = [json.loads(line) for line in (run/'bridge/requests.jsonl').read_text().splitlines()]
    completed = {(r['task'], r['turn']) for r in rows if r.get('event') == 'finish'
                 and state['model_ready_at'] <= r['at'] <= state['window_deadline']}
    candidates = [r for r in rows if r.get('event') == 'start' and
                  (r['task'], r['turn']) in completed and r['prompt_tokens'] >= minimum]
    counts, chosen = Counter(), []
    for r in sorted(candidates, key=lambda r: (-r['prompt_tokens'], r['task'], r['turn'])):
        if counts[r['task']] >= per_task:
            continue
        counts[r['task']] += 1
        chosen.append(dict(task=r['task'], turn=r['turn'], prompt_tokens=r['prompt_tokens']))
        if len(chosen) == count:
            return chosen
    raise ValueError(f'Need{count} completed long turns, found{len(chosen)}; never pad or truncate')


def replay_task(run, task, selected, planner, state, emit):
    wanted = {r['turn']: r for r in selected}
    provenance = {}
    for turn in range(1, max(wanted)+1):
        folder = run/'bridge'/task/f'{turn:05d}'
        data = {}
        for name in ('request', 'plan', 'response', 'native'):
            path = folder/(name+'.json')
            raw = path.read_bytes()
            provenance[str(path)] = hashlib.sha256(raw).hexdigest()
            data[name] = json.loads(raw)
        prepared = planner.prepare(data['request'], state)
        tokens, plan = prepared[:2]
        # canonical() in the frozen bridge uses these exact JSON options.
        token_json = json.dumps(tokens, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if (len(tokens) != data['plan']['prompt_tokens'] or
                hashlib.sha256(token_json.encode()).hexdigest() != data['plan']['token_sha256']):
            raise ValueError(f'Exact prompt token hash mismatch:{task}/{turn}')
        if plan != {k: v for k, v in data['plan'].items() if k.startswith('kvmem_')}:
            raise ValueError(f'Request selection plan changed:{task}/{turn}')
        if turn in wanted:
            if len(tokens) != wanted[turn]['prompt_tokens']:
                raise ValueError('Event and plan prompt counts differ')
            output = data['native']['choices'][0]['token_ids']
            if not output or type(output[0]) is not int or not 0 <= output[0] < 2**31:
                raise ValueError('Need a recorded first continuation token')
            emit(wanted[turn], tokens, plan, dict(data['plan'], recorded_continuation_token_id=output[0]))
        # Recorded accepted output IDs are authoritative. Re-encoding assistant
        # text would silently change future prefixes and invalidate comparison.
        planner.commit(state, prepared, data['request'], data['response'],
                       data['native']['choices'][0]['token_ids'])
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--bridge-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--minimum-input-tokens', type=int, default=212992)
    p.add_argument('--per-task', type=int, default=2)
    a = p.parse_args()
    if min(a.count, a.minimum_input_tokens, a.per_task) <= 0:
        raise ValueError('Positive counts and lengths required')
    run = a.run.resolve()
    chosen = select_turns(run, a.count, a.minimum_input_tokens, a.per_task)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'selected.json').write_text(json.dumps(chosen, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('frozen_export_bridge', a.bridge_source)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    from transformers import AutoTokenizer
    profile = read(run/'profile.json')
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    planner = bridge.Planner(bridge.Protocol(run/'protocol/protocol.so'), tokenizer,
                             profile['kvmem']['gen_budget'])
    samples = []
    def emit(row, tokens, plan, original_plan):
        values = array.array('I', tokens)
        if values.itemsize != 4:
            raise ValueError('Require32-bit token IDs')
        if sys.byteorder != 'little':
            values.byteswap()
        name = f"{row['task']}-{row['turn']:05d}.i32"
        raw = values.tobytes()
        (a.output/name).write_bytes(raw)
        samples.append(dict(row, file=name, sha256=hashlib.sha256(raw).hexdigest(),
            continuation_token_id=original_plan['recorded_continuation_token_id'],
            request_plan=plan, original_sampling={k: original_plan.get(k) for k in
            ('temperature', 'top_p', 'top_k', 'seed', 'thinking_token_budget', 'max_tokens')}))
    provenance = {}
    for task in sorted({r['task'] for r in chosen}):
        provenance.update(replay_task(run, task, [r for r in chosen if r['task'] == task],
                                     planner, bridge.Trajectory(), emit))
    if not all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == sha for f, sha in provenance.items()):
        raise ValueError('Source measurements changed during export')
    sources = [a.bridge_source, run/'protocol/protocol.so', run/'profile.json',
               run/'run.json', run/'bridge/requests.jsonl', Path(__file__)]
    provenance.update({str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    result = dict(format='deepswe_capacity_history_v1', capacity_diagnostic_only=True,
        scope='Exact recorded completed DeepSWE turns. No padding, cropping, re-encoding accepted output, KV files or accuracy claim. Different replay clients must use independent cache salts.',
        source_run=str(run), tokenizer=profile['tokenizer'], selected=chosen,
        samples=samples, source_sha256=provenance)
    (a.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(exported=len(samples), tokens=sum(r['prompt_tokens'] for r in samples))))


if __name__ == '__main__':
    main()
