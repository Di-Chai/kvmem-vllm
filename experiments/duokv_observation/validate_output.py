"""Independently validate saved products; no analysis generation or model calls."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from core import read_json, read_jsonl, verify_unit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def validate(out):
    manifest=read_json(out/'output-manifest.json')
    for entry in manifest['files']:
        path=out/entry['path']
        assert path.stat().st_size==entry['bytes'],entry['path']
        assert sha(path)==entry['sha256'],entry['path']
    bundle=Path(manifest['config']['bundle'])
    units={}
    cache={}
    last_path=None
    for _,unit in read_jsonl(out/'structured/units.jsonl'):
        if unit['trace_path']!=last_path:
            cache={}
            last_path=unit['trace_path']
        verify_unit(bundle,unit,cache)
        assert unit['unit_id'] not in units
        units[unit['unit_id']]=unit
    def evcheck(ev):
        u=units[ev['unit_id']]
        assert ev['message_id']==u['message_id']
        assert ev['trace_line']==u['trace_line']
        a,b=ev['excerpt_span_in_unit']
        assert 0<=a<=b<=len(u['text'])
        assert ev['excerpt']==u['text'][a:b],ev['unit_id']
    counts={}
    uses=defaultdict(lambda:defaultdict(set))
    for name in ['reuse-evidence','retry-reuse-evidence']:
        n=0
        for _,edge in read_jsonl(out/f'structured/{name}.jsonl'):
            a,b=units[edge['source_unit_id']],units[edge['target_unit_id']]
            assert a['trajectory_id']==b['trajectory_id']==edge['trajectory_id']
            assert a['turn_id']<b['turn_id']
            assert b['role']=='assistant'
            if name=='reuse-evidence':
                assert a['main_eligible'] and b['main_eligible'] and a['step']<b['step']
                assert edge['distance']==b['step']-a['step']
                uses[a['unit_id']][edge['level']].add(b['step'])
            else:
                assert edge['distance']==b['turn_id']-a['turn_id']
            evcheck(edge['source_evidence']);evcheck(edge['target_evidence']);n+=1
        counts[name]=n
    events={}
    times=defaultdict(set)
    for _,event in read_jsonl(out/'structured/events.jsonl'):
        assert event['event_id'] not in events
        events[event['event_id']]=event
        times[(event['trajectory_id'],event['event_type'])].add(event['step'])
        for ev in event['evidence']:
            evcheck(ev)
        if event['source_evidence']:
            evcheck(event['source_evidence'])
        assert event['task_success'] is None
    relations=0;controls=0
    for _,relation in read_jsonl(out/'structured/event-source-relations.jsonl'):
        u=units[relation['source_unit_id']];event=events[relation['event_id']]
        assert u['trajectory_id']==event['trajectory_id']
        assert u['step']<event['step']==relation['event_step']
        assert relation['event_to_decay_latency'] is None
        assert relation['residual_semantic_information_fraction'] is None
        evcheck(relation['source_evidence'])
        alluses=uses[u['unit_id']]['explicit']|uses[u['unit_id']]['semantic']
        assert sorted(s for s in alluses if s>event['step'])==relation['post_event_reuse_steps']
        control=relation['control'];w=relation['window']
        if control:
            cu=units[control['unit_id']];t=control['pseudo_event_step']
            assert cu['trajectory_id']==u['trajectory_id'] and cu['content_type']==u['content_type']
            assert cu['step']<=t-w
            assert all(abs(t-x)>w for x in times[(u['trajectory_id'],event['event_type'])])
            assert abs((t-cu['step'])-(event['step']-u['step']))<=2
            controls+=1
        relations+=1
    samples=0
    for _,ex in read_jsonl(out/'structured/prediction-examples.jsonl'):
        u=units[ex['unit_id']];t=ex['checkpoint_step'];k=ex['window']
        assert ex['features_max_step']==t and u['step']<=t
        for uid in ex['query_unit_ids']:
            assert units[uid]['step']<=t
        for level in ['explicit','semantic']:
            assert ex['labels'][level]==int(any(t<s<=t+k for s in uses[u['unit_id']][level]))
        assert ex['labels']['high_confidence_dependency'] is None
        alluses=uses[u['unit_id']]['explicit']|uses[u['unit_id']]['semantic']
        assert ex['prior_reuse_count']==sum(s<=t for s in alluses)
        assert ex['last_observed_use_step']==max([u['step']]+[s for s in alluses if s<=t])
        samples+=1
    for _,model in read_jsonl(out/'structured/prediction-models.jsonl'):
        assert not set(model['train_tasks'])&set(model['test_tasks'])
    with (out/'kv-snapshot-correlation.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['eligible']=='False':
                assert row['selected_fraction']==''
            if row['full_future_window']=='False':
                assert row['future_reuse_fraction']==''
    return dict(status='passed',verified_manifest_files=len(manifest['files']),
        original_unit_reconstruction_checks=len(units),reuse_edge_checks=counts,
        events=len(events),event_source_relations=relations,matched_controls=controls,
        temporal_prediction_samples=samples,human_ground_truth_labels=0,
        raw_input_manifest_sha256=sha(bundle/'MANIFEST.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--write-audit',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    audit=validate(out)
    if args.write_audit:
        rel='structured/output-validation.json';path=out/rel
        path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        manifest=read_json(out/'output-manifest.json')
        manifest['files']=[r for r in manifest['files'] if r['path']!=rel]+[
            dict(path=rel,bytes=path.stat().st_size,sha256=sha(path))]
        manifest['files'].sort(key=lambda r:r['path'])
        (out/'output-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':
    main()
