"""Observe recorded history selection and repeated feedback, without inferring causality."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def overlap(begin, end, ranges):
    return sum(max(0, min(end, start + count) - max(begin, start)) for start, count in ranges)


def analyze(trace, labels_root):
    first = Path(trace['sources'][0]['path'])
    root, session = first.parents[3], first.parents[1]
    ids = {}
    for folder in sorted(p for p in session.iterdir() if p.name.isdigit()):
        ids[read(folder / 'native.json')['id']] = int(folder.name)
    capture = root / ('server/capture' if trace['family'] == 'deepswe' else 'model/server/capture')
    snapshots = []
    with (capture / 'bounded_refresh.jsonl').open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            turn = next((turn for request_id, turn in ids.items()
                         if row['request_id'] == request_id or row['request_id'].startswith(request_id + '-')), None)
            if turn is None:
                continue
            ranges = sorted(row['selected_history_ranges'])
            assert all(a + n <= b for (a, n), (b, m) in zip(ranges, ranges[1:]))
            assert sum(n for a, n in ranges) == row['selected_tokens']
            snapshots.append(dict(turn=turn, history_begin=row['history_begin'],
                selected_tokens=row['selected_tokens'], ranges=ranges,
                source=dict(path=str(capture/'bounded_refresh.jsonl'), line=line_number),
                selection=row['selection'], attention_source=row.get('attention_source'),
                mandatory_fit=row.get('mandatory_fit')))
    snapshots.sort(key=lambda s: (s['turn'], s['history_begin']))
    events = [dict(event, birth_turn=0, origin='initial_input') for event in trace['initial']['events']]
    events += [dict(event, birth_turn=row['turn']) for row in trace['rounds'] for event in row['events']]
    records = [json.loads(line) for line in (labels_root/trace['family']/(trace['task_id']+'.jsonl')).open()]
    model = {r['turn']: r for r in records if r.get('record_type') == 'message' and r.get('role') == 'assistant'}
    blocks = []
    for index, event in enumerate(events):
        if event['kind'] in ('protocol', 'boundary'):
            continue
        block = dict(event, id=f"{trace['family']}:b{index}")
        turn = event['birth_turn']
        if trace['family'] == 'deepswe' and turn:
            calls = [b for b in model[turn]['original']['content'] if b.get('type') == 'tool_use']
            block['action_sources'] = [dict(tool=b['name'], input=b['input']) for b in calls]
        elif turn:
            block['round_analysis'] = model[turn].get('parsed', {}).get('analysis', '')
        observations = []
        for snapshot in snapshots:
            eligible = event['end'] <= snapshot['history_begin']
            selected = overlap(event['begin'], event['end'], snapshot['ranges']) if eligible else None
            observations.append(dict(turn=snapshot['turn'], history_begin=snapshot['history_begin'],
                eligible=eligible, selected_tokens=selected,
                selected_fraction=selected/event['tokens'] if eligible else None))
        block['observations'] = observations
        block['in_fixed_sink'] = event['end'] <= 41 * 32
        block['overlaps_fixed_sink'] = event['begin'] < 41 * 32
        blocks.append(block)
    grouped = defaultdict(list)
    for block in blocks:
        grouped[(block['birth_turn'], block['kind'])].append(block)
    rows = []
    for (birth, kind), members in grouped.items():
        tokens = sum(b['tokens'] for b in members)
        observations = []
        for i, snapshot in enumerate(snapshots):
            eligible = all(b['observations'][i]['eligible'] for b in members)
            kept = sum(b['observations'][i]['selected_tokens'] for b in members) if eligible else None
            observations.append(dict(turn=snapshot['turn'], fraction=kept/tokens if eligible else None,
                                     selected_tokens=kept, eligible=eligible))
        rows.append(dict(birth_turn=birth, kind=kind, tokens=tokens, observations=observations))
    feedbacks = []
    for record in records:
        if record.get('record_type') != 'message' or record.get('role') != 'user':
            continue
        if any(a['kind'] == 'parser_feedback' for a in record['labels']):
            text = record['original']['content']
            feedbacks.append(dict(after_turn=record['turn'], sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  text=text, source=record['source']))
    repeated = Counter(f['sha256'] for f in feedbacks)
    retries = [row for row in trace['rounds'] if row['outcome'] == 'format_retry']
    return dict(family=trace['family'], task_id=trace['task_id'], snapshots=snapshots,
        blocks=blocks, round_kind_coverage=rows, feedbacks=feedbacks,
        duplicate_feedback_counts=dict(repeated),
        retry_tokens=dict(Counter({}) + sum((Counter(r['counts_added']) for r in retries), Counter())),
        retry_total_tokens=sum(r['context_after']-r['prompt_tokens'] for r in retries),
        caveats=['Selected coverage measures the recorded policy decision, not causal task importance.',
                 'Snapshots are sparse. No interpolation or claim of continued selection between snapshots.',
                 'Fixed sink retention is not evidence of model demand.',
                 'Unselected may include unavailable or already retired history; it does not prove disuse.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--growth', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    growth = read(args.growth)
    traces = [analyze(trace, args.labels) for trace in growth['traces']]
    (args.output/'retention-observations.json').write_text(json.dumps(
        dict(metric='Recorded active-window selected fraction; not causal importance', traces=traces),
        ensure_ascii=False, indent=2) + '\n')
    with (args.output/'retention-by-round-kind.csv').open('w', newline='') as stream:
        fields = ['family','task_id','birth_turn','kind','tokens','snapshot_turn','eligible','selected_tokens','selected_fraction']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trace in traces:
            for row in trace['round_kind_coverage']:
                for observation in row['observations']:
                    writer.writerow(dict(family=trace['family'], task_id=trace['task_id'],
                        birth_turn=row['birth_turn'], kind=row['kind'], tokens=row['tokens'],
                        snapshot_turn=observation['turn'], eligible=observation['eligible'],
                        selected_tokens=observation['selected_tokens'], selected_fraction=observation['fraction']))
    print(json.dumps(dict(output=str(args.output), traces=[dict(task_id=t['task_id'],
        snapshots=len(t['snapshots']), blocks=len(t['blocks']), retry_tokens=t['retry_tokens'],
        retry_total_tokens=t['retry_total_tokens'], duplicate_feedback_counts=t['duplicate_feedback_counts'])
        for t in traces]), ensure_ascii=False))


if __name__ == '__main__':
    main()
