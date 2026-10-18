#!/usr/bin/env python3
"""Replay bounded, recorded Query inputs through six fixed-budget selectors.

This is a counterfactual *selection* analysis. The recorded Mean-K vector and
attention graph remain fixed for every arm; no model is run and no answer
correctness or closed-loop quality is inferred from these rankings.
"""
import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from vllm_kvmem.block_graph import GraphPartial, GraphRow
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.graph_scoring import score_graph
from vllm_kvmem.memory import RequestMemory


MAX_NODES = 2048
MAX_RECORDS = 256
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 256 * 1024
ARMS = ('mean_k', 'collect', 'legacy_attention', 'locality', 'attention', 'multi')


class _Ready:
    def query(self):
        return True


def _pairs(value):
    if value is None:
        return {}
    items = value.items() if isinstance(value, dict) else value
    result = {int(key): float(score) for key, score in items}
    if any(not math.isfinite(score) or score < 0 for score in result.values()):
        raise ValueError('Invalid recorded legacy attention score')
    return result


def _snapshot(record, block_tokens):
    value = record['graph']
    rows = tuple(GraphRow(int(row['source']),tuple(row['targets']),tuple(row['weights']),
        int(row['samples']),int(row.get('prefill_samples',0)),int(row.get('output_samples',0)))
        for row in value.get('rows',()))
    partial = value.get('partial')
    if partial is not None:
        partial = GraphPartial(int(partial['source']),tuple(partial['targets']),tuple(partial['sums']),
            int(partial['samples']),int(partial.get('prefill_samples',0)),int(partial.get('output_samples',0)))
    spans = tuple(tuple(span) for span in value.get('message_spans',()))
    ends_known = value.get('message_end_known')
    return SimpleNamespace(rows=rows,partial=partial,message_spans=spans,
        message_end_known=tuple((True,)*len(spans) if ends_known is None else ends_known),
        accepted_through=int(value['accepted_through']),
        completed_through=int(value['completed_through']),revision=int(value.get('revision',0)),
        epoch=int(value.get('runtime_epoch',value.get('epoch',0))),block_tokens=block_tokens,
        closed=False,bytes=int(value.get('bytes',0)))


def _validate_record(record):
    if record.get('version') != 1:
        raise ValueError('Unsupported recorded graph input version')
    starts,counts,raw,eligible = (record[key] for key in ('starts','counts','raw_scores','eligible'))
    if not 0 < len(starts) <= MAX_NODES or not len(starts) == len(counts) == len(raw) == len(eligible):
        raise ValueError('Recorded graph index is empty, oversized or shape mismatched')
    cfg = KVMemConfig(**record['config'])
    block = cfg.block_tokens
    if starts != list(range(0,len(starts)*block,block)):
        raise ValueError('Offline native replay requires contiguous original history; holes are unsupported')
    if any(type(n) is not int or n != block for n in counts[:-1]) or not 0 < counts[-1] <= block:
        raise ValueError('Recorded history has an incomplete non-tail block')
    if any(type(p) is not int or p not in starts for p in record['mandatory_starts']):
        raise ValueError('Recorded mandatory identity is missing from the index')
    mandatory = set(record['mandatory_starts'])
    expected = [start >= cfg.sink_blocks*block and i < len(starts)-cfg.recent_blocks and start not in mandatory
                for i,start in enumerate(starts)]
    if eligible != expected:
        raise ValueError('Recorded eligible mask differs from native keep bands or mandatory identity')
    budget = record['budget']
    if type(budget) is not int or not 0 < budget <= cfg.select_budget or budget % block:
        raise ValueError('Recorded runtime budget must be block aligned and fit the original K')
    snapshot = _snapshot(record,block)
    if snapshot.accepted_through < starts[-1]+counts[-1]:
        raise ValueError('Recorded graph accepted watermark precedes its Query index')
    if len(snapshot.rows) > MAX_NODES or (snapshot.partial is not None and len(snapshot.partial.targets)>MAX_NODES):
        raise ValueError('Recorded graph exceeds bounded replay capacity')
    # Production capture is strictly causal. Future-directed operator examples
    # must remain isolated math fixtures, never presented as recorded attention.
    for row in snapshot.rows:
        if (row.source % block or row.samples <= 0 or len(row.targets) > 4 or
                row.prefill_samples+row.output_samples != row.samples or
                any(type(target) is not int or target < 0 or target % block or target >= row.source
                    for target in row.targets)):
            raise ValueError('Recorded attention graph has invalid or noncausal logical identities')
    return cfg,snapshot


def _native_history(config, starts, counts):
    """Recreate selector metadata only, respecting original K/G while appending."""
    memory = RequestMemory('offline-graph-query',config)
    handles = {}
    remaining, local = starts[-1]+counts[-1],0
    try:
        while remaining:
            room = config.select_budget+config.gen_budget-local
            if not room:
                plan = memory.select({},action=False)
                local = plan.total_window_tokens
                memory.attach_completion(_Ready())
                assert memory.commit_selection()
                del plan
                room = config.select_budget+config.gen_budget-local
            count = min(remaining,room)
            spans = memory.begin_append(count,local)
            if spans is None:
                raise BufferError('Recorded repository capacity cannot reconstruct its history')
            handles.update((span.history_begin,span.handle) for span in spans)
            memory.attach_completion(_Ready())
            assert memory.commit_append(count)
            remaining -= count
            local += count
        if set(handles) != set(starts):
            raise ValueError('Reconstructed native handles do not match recorded logical blocks')
        return memory,handles
    except BaseException:
        memory.close()
        raise


def _quality(record, selected, baseline):
    evidence = record.get('evidence')
    if evidence is None:
        return dict(required_evidence_coverage=None, required_dependency_coselection=None,
            displaced_mean_k_evidence=None, duplicate_blocks=None, outdated_blocks=None,
            irrelevant_blocks=None, source_span_complete=None, dependency_span_complete=None,
            full_evidence_coselection=None,answer_correctness=None)
    required = set(evidence.get('required_starts',()))
    dependencies = tuple(tuple(pair) for pair in evidence.get('dependency_pairs',()))
    groups = evidence.get('duplicate_groups')
    duplicate = (sum(max(0,len(selected&set(group['starts']))-1) for group in groups)
                 if groups is not None else
                 len(selected & set(evidence['duplicate_starts'])) if 'duplicate_starts' in evidence else None)
    spans = {span['name']:set(span['block_starts']) for span in evidence.get('spans',())}
    source = spans.get('source')
    dependency = spans.get('dependency')
    return dict(required_evidence_coverage=(len(selected & required)/len(required) if required else None),
        required_dependency_coselection=(sum(a in selected and b in selected for a,b in dependencies)/len(dependencies)
                                        if dependencies else None),
        displaced_mean_k_evidence=len((baseline-selected)&required) if required else None,
        duplicate_blocks=duplicate,
        outdated_blocks=len(selected & set(evidence['outdated_starts'])) if 'outdated_starts' in evidence else None,
        irrelevant_blocks=len(selected & set(evidence['irrelevant_starts'])) if 'irrelevant_starts' in evidence else None,
        source_span_complete=source <= selected if source else None,
        dependency_span_complete=dependency <= selected if dependency else None,
        full_evidence_coselection=(source|dependency) <= selected if source and dependency else None,
        answer_correctness=None)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()


def unique_token_span(tokens, needle):
    if not needle:
        raise ValueError('Evidence needle is empty')
    matches = [index for index in range(len(tokens)-len(needle)+1)
               if tokens[index:index+len(needle)] == needle]
    if len(matches) != 1:
        raise ValueError(f'Evidence needle must occur exactly once; found {len(matches)} matches')
    return matches[0],matches[0]+len(needle)


def annotate_fixture(records, fixture, traces, tokenizer):
    """Derive auditable evidence labels; never edit recorded inputs or fixture.

    Native request identities are joined to the retained client trace. Exact
    prompt hashes, including actual first-turn output IDs on continuation, must
    match before labels from the frozen initial archive may be attached.
    """
    if (fixture.get('kind') != 'block_graph_qualification_v1' or
            fixture.get('sha256') != canonical_digest({k:v for k,v in fixture.items() if k!='sha256'})):
        raise ValueError('Fixture kind or immutable hash does not match')
    agents = {agent['index']:agent for agent in fixture['agents']}
    if len(agents) != len(fixture['agents']):
        raise ValueError('Fixture has duplicate agent identities')
    by_turn = {}
    for trace in traces:
        key = trace.get('agent'),trace.get('turn')
        if key in by_turn or key[0] not in agents or key[1] not in (0,1,2):
            raise ValueError('Trace has duplicate, foreign or unsupported agent turn')
        by_turn[key] = trace
    rebuilt,agent_labels,proofs = {},{},[]
    for index,agent in agents.items():
        initial = agent['initial_tokens']
        if agent['first_input_sha256'] != canonical_digest(initial):
            raise ValueError('Fixture initial token fingerprint differs')
        first = by_turn.get((index,0))
        if first is None:
            raise ValueError('Trace is missing the exact first-turn identity')
        before_query = initial + first['output_ids'] + agent['tool_tokens']
        prompts = (initial,before_query+agent['header']+agent['second_question']+agent['footer'],
                   before_query+agent['header']+agent['edited_question']+agent['footer'])
        for turn,prompt in enumerate(prompts):
            trace = by_turn.get((index,turn))
            if trace is None:
                continue
            if trace.get('input_sha256') != canonical_digest(prompt) or trace.get('input_tokens') != len(prompt):
                raise ValueError('Trace prompt fingerprint does not match frozen fixture and accepted output')
            rebuilt[index,turn] = prompt
        project,ledger = f'Project Meridian-{index:02d}',f'ledger-{71+index}'
        deployment,recovery,_ = agent['expected_codes']
        statements = dict(
            source=f'{project} depends on access record {ledger}. When asked about this project, '
                   'resolve its deployment or recovery code through that record.\n',
            dependency=f'Validated access record {ledger}: the deployment code is {deployment}; '
                       f'the recovery code is {recovery}. Both exact strings are authoritative.\n')
        located = {}
        for name,text in statements.items():
            needle = list(tokenizer.encode(text,add_special_tokens=False))
            if needle != agent[name+'_tokens']:
                raise ValueError('Local tokenizer encoding differs from frozen evidence needle')
            begin,end = unique_token_span(initial,needle)
            if end > agent['archive_prefix_tokens']:
                raise ValueError('Evidence needle is outside the frozen archive prefix')
            located[name] = (begin,end)
            proofs.append(dict(agent=index,name=name,text=text,token_ids=needle,
                token_ids_sha256=canonical_digest(needle),begin=begin,end=end,
                first_input_sha256=agent['first_input_sha256']))
        agent_labels[index] = located
    annotated,joins = [],[]
    for number,record in enumerate(records):
        matches = [trace for trace in traces if trace.get('response_id') and
            (record['request_id'] == trace['response_id'] or
             record['request_id'].startswith(trace['response_id']+'-'))]
        if len(matches) != 1:
            raise ValueError('Recorded Query cannot be joined uniquely to a client response')
        trace = matches[0]
        index,turn = trace['agent'],trace['turn']
        origin = by_turn[index,0]['response_id']
        if record['authority_id'] != origin and not record['authority_id'].startswith(origin+'-'):
            raise ValueError('Recorded authority belongs to another fixture session')
        prompt = rebuilt[index,turn]
        block = record['config']['block_tokens']
        if block != 32:
            raise ValueError('Real fixture annotation requires the unchanged B32 geometry')
        positions = set(record['starts'])
        available = dict(zip(record['starts'],record['counts']))
        spans = []
        for name,(begin,end) in agent_labels[index].items():
            covered = list(range(begin//block*block,((end-1)//block+1)*block,block))
            if not set(covered) <= positions or any(start+available[start] < min(end,start+block) for start in covered):
                raise ValueError('Recorded Query index is missing part of a labeled evidence span')
            spans.append(dict(name=name,begin=begin,end=end,block_starts=covered))
        required = set().union(*(set(span['block_starts']) for span in spans))
        observed = {}
        for start,count in zip(record['starts'],record['counts']):
            if start in required or count != block or start+block > len(prompt):
                continue
            tokens = tuple(prompt[start:start+block])
            observed.setdefault(tokens,[]).append(start)
        groups = [dict(starts=starts,token_ids=list(tokens),token_ids_sha256=canonical_digest(list(tokens)))
                  for tokens,starts in observed.items() if len(starts)>1]
        groups.sort(key=lambda group:group['starts'][0])
        source = next(span for span in spans if span['name']=='source')['block_starts']
        dependency = next(span for span in spans if span['name']=='dependency')['block_starts']
        derived = deepcopy(record)
        derived['evidence'] = dict(required_starts=sorted(required),spans=spans,
            dependency_pairs=[[s,t] for s in source for t in dependency],duplicate_groups=groups)
        derived['annotation'] = dict(kind='exact_frozen_fixture_evidence',fixture_sha256=fixture['sha256'],
            agent=index,turn=turn,trace_input_sha256=trace['input_sha256'],
            original_record_sha256=canonical_digest(record),
            scope='Exact source/ledger statement blocks and exact full-block duplicate tokens only; '
                  'outdated/irrelevant labels and counterfactual answer correctness are unknown')
        annotated.append(derived)
        joins.append(dict(record=number,request_id=record['request_id'],authority_id=record['authority_id'],
            agent=index,turn=turn,trace_response_id=trace['response_id'],
            trace_input_sha256=trace['input_sha256'],original_record_sha256=canonical_digest(record),
            labeled_record_sha256=canonical_digest(derived),required_starts=sorted(required),
            duplicate_groups=len(groups)))
    return annotated,dict(version=1,fixture_sha256=fixture['sha256'],needle_proofs=proofs,joins=joins,
        limitations=['Exact evidence coverage is a selector metric, not generated-answer correctness.',
            'All arms reuse one recorded trajectory; generated continuations and graphs are not regenerated.',
            'Duplicate counts measure extra selected copies of byte-identical token blocks; semantic redundancy is unknown.',
            'Outdated and irrelevant content are unlabelled and remain null.'])


def summarize_records(results):
    summaries = {}
    metrics = ('required_evidence_coverage','required_dependency_coselection',
        'source_span_complete','dependency_span_complete','full_evidence_coselection',
        'displaced_mean_k_evidence','duplicate_blocks','outdated_blocks','irrelevant_blocks','answer_correctness')
    for arm in ARMS:
        data = [record['arms'][arm] for record in results]
        summary = dict(records=len(data),added_blocks=sum(len(row['added_starts']) for row in data),
                       displaced_blocks=sum(len(row['displaced_starts']) for row in data))
        for metric in metrics:
            values = [row['quality'].get(metric) for row in data]
            known = [value for value in values if value is not None]
            summary[metric] = dict(labeled_records=len(known),
                mean=math.fsum(known)/len(known) if known else None,
                total=math.fsum(known) if known else None)
        summaries[arm] = summary
    return summaries


def analyze_record(record, *, locality_weight=.1, attention_weight=.2,
                   cross_message_weight=.25, legacy_fraction=.25):
    config,snapshot = _validate_record(record)
    starts,raw,eligible = (record[key] for key in ('starts','raw_scores','eligible'))
    if not math.isfinite(legacy_fraction) or not 0 <= legacy_fraction <= 1:
        raise ValueError('Legacy comparison fraction must lie in [0,1]')
    modes = dict(mean_k='off',collect='collect',legacy_attention='off',
                 locality='locality',attention='attention',multi='multi')
    results,baseline = {},None
    for arm in ARMS:
        legacy = arm == 'legacy_attention'
        cfg = replace(config,action_attention=legacy,action_fraction=legacy_fraction if legacy else 0.)
        memory,handles = _native_history(cfg,starts,record['counts'])
        try:
            propagated = score_graph(raw,starts,eligible,snapshot,mode=modes[arm],
                locality_weight=locality_weight,attention_weight=attention_weight,
                cross_message_weight=cross_message_weight,block_tokens=cfg.block_tokens)
            if legacy:
                memory.set_action_scores(_pairs(record.get('legacy_action_scores',{})))
            slots = memory.preview_selection({handles[start].slot:score
                for start,score in zip(starts,propagated.scores)},
                tuple(handles[start].slot for start in record['mandatory_starts']),
                action=legacy,prefill_scores=_pairs(record.get('prefill_scores',{})) if legacy else None,
                max_tokens=record['budget'])
            slot_positions = {handle.slot:start for start,handle in handles.items()}
            selected = {slot_positions[slot] for slot in slots}
            if baseline is None:
                baseline = selected
            counts = dict(zip(starts,record['counts']))
            used = sum(counts[start] for start in selected)
            if used > record['budget']:
                raise RuntimeError('Native replay exceeded its fixed token budget')
            observed = [(row.source,target,weight) for row in snapshot.rows
                        for target,weight in zip(row.targets,row.weights)]
            retained_mass = math.fsum(weight for _,_,weight in observed)
            diagnostics = dict(propagated.diagnostics)
            # CPU propagation time excludes attention capture, graph building,
            # model inference, restoration and request scheduling.
            results[arm] = dict(selected_starts=sorted(selected),selected_tokens=used,
                added_starts=sorted(selected-baseline),displaced_starts=sorted(baseline-selected),
                retained_raw_mean_k_mass=math.fsum(score for start,score in zip(starts,raw) if start in selected),
                legacy_fraction=legacy_fraction if legacy else 0.,
                observed_edge_coselection=(sum(a in selected and b in selected for a,b,_ in observed)/len(observed)
                                          if observed else None),
                observed_mass_coselection=(math.fsum(w for a,b,w in observed if a in selected and b in selected)/retained_mass
                                          if retained_mass else None),
                diagnostics=diagnostics,quality=_quality(record,selected,baseline))
        finally:
            assert memory.close()
    return dict(version=1,authority_id=record.get('authority_id'),request_id=record.get('request_id'),
        captured_mode=record.get('mode'),input_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
        annotation=record.get('annotation'),
        history_blocks=len(starts),budget=record['budget'],block_tokens=config.block_tokens,
        labeled_fixture=record.get('evidence') is not None,
        observed_source_coverage=len(snapshot.rows)/len(starts),
        mean_retained_attention_mass=(math.fsum(row.retained_mass for row in snapshot.rows)/len(snapshot.rows)
                                      if snapshot.rows else 0.),
        arms=results)


def causal_fixture():
    """Ground-truth labels are explicitly synthetic, not measured task quality."""
    return dict(version=1,authority_id='synthetic-causal',request_id='synthetic-query',mode='synthetic',
        starts=[0,4,8,12,16,20],counts=[4]*6,raw_scores=[.01,.10,.15,.12,.02,.60],
        eligible=[True]*6,mandatory_starts=[],budget=12,
        config=dict(block_tokens=4,select_budget=12,gen_budget=12,repository_blocks=16,
                    sink_blocks=0,recent_blocks=0,action_attention=True,action_fraction=0.,action_combine='quota'),
        graph=dict(rows=[dict(source=20,targets=[4],weights=[.9],samples=4,prefill_samples=3,output_samples=1)],
                   partial=None,message_spans=[],message_end_known=[],accepted_through=24,
                   completed_through=24,revision=1),
        legacy_action_scores={0:.7,4:.3},prefill_scores={4:1.},
        evidence=dict(required_starts=[4,20],dependency_pairs=[[20,4]],
                      duplicate_starts=[],outdated_starts=[16],irrelevant_starts=[12]))


def load_records(paths, limit):
    if not 1 <= limit <= MAX_RECORDS:
        raise ValueError('Offline record limit must be between 1 and 256')
    records,sources = [],[]
    for filename in paths:
        path = Path(filename)
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError('Recorded input file exceeds the offline analysis limit')
        payload = path.read_bytes()
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest()))
        for line in payload.splitlines():
            if not line.strip():
                continue
            if len(line) > MAX_RECORD_BYTES:
                raise ValueError('Recorded Query exceeds its bounded export size')
            if len(records) >= limit:
                return records,sources
            records.append(json.loads(line))
    return records,sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',default=[],help='Recorded block_graph_inputs.jsonl; repeatable')
    parser.add_argument('--synthetic',action='store_true',help='Include the explicitly labeled causal fixture')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-records',type=int,default=64)
    parser.add_argument('--locality-weight',type=float,default=.1)
    parser.add_argument('--attention-weight',type=float,default=.2)
    parser.add_argument('--cross-message-weight',type=float,default=.25)
    parser.add_argument('--legacy-fraction',type=float,default=.25)
    parser.add_argument('--fixture',type=Path,help='Unmodified real-model block graph fixture for exact evidence labels')
    parser.add_argument('--traces',type=Path,help='Retained qualification/traces.json for request/agent and prompt hash joins')
    parser.add_argument('--tokenizer',type=Path,help='Local tokenizer directory; no network or model inference')
    args = parser.parse_args(argv)
    if not args.input and not args.synthetic:
        parser.error('Provide recorded --input or explicit --synthetic')
    records,sources = load_records(args.input,args.max_records)
    annotation = None
    if any((args.fixture,args.traces,args.tokenizer)):
        if not all((args.fixture,args.traces,args.tokenizer)) or args.synthetic:
            parser.error('Use --fixture, --traces and --tokenizer together with recorded inputs only')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()),local_files_only=True,
                                                  trust_remote_code=False)
        fixture = json.loads(args.fixture.read_text())
        traces = json.loads(args.traces.read_text())
        records,annotation = annotate_fixture(records,fixture,traces,tokenizer)
        annotation['files'] = {name:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for name,path in (('fixture',args.fixture),('traces',args.traces))}
        annotation['recorded_inputs'] = sources
        annotation['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        annotation['tokenizer'] = dict(path=str(args.tokenizer.resolve()),class_name=type(tokenizer).__name__,
            files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (args.tokenizer/'tokenizer.json',args.tokenizer/'tokenizer_config.json',args.tokenizer/'config.json')
                   if path.is_file()})
    if args.synthetic:
        if len(records) == args.max_records:
            parser.error('Increase --max-records to include the synthetic fixture')
        records.append(causal_fixture())
    results = [analyze_record(record,locality_weight=args.locality_weight,
        attention_weight=args.attention_weight,cross_message_weight=args.cross_message_weight,
        legacy_fraction=args.legacy_fraction) for record in records]
    output = dict(version=1,kind='fixed_recorded_input_selection_analysis',sources=sources,
        limitations=['All arms reuse the same recorded Mean-K and attention observations.',
                     'No model inference, answer correctness, end-to-end throughput or closed-loop quality is measured.',
                     'Evidence metrics are null unless an explicitly labeled fixture was supplied.',
                     'Legacy attention comparison uses the stated fraction; K/G, keep bands and runtime budget are unchanged.'],
        weights=dict(locality=args.locality_weight,attention=args.attention_weight,
                     cross_message=args.cross_message_weight,legacy_fraction=args.legacy_fraction),
        summary=summarize_records(results),records=results)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    if annotation is not None:
        (args.output_dir/'annotated-inputs.jsonl').write_text(''.join(json.dumps(record,allow_nan=False)+'\n' for record in records))
        (args.output_dir/'annotation-provenance.json').write_text(json.dumps(annotation,indent=2,allow_nan=False)+'\n')
    if args.synthetic:
        (args.output_dir/'synthetic-input.jsonl').write_text(json.dumps(causal_fixture(),allow_nan=False)+'\n')
    with (args.output_dir/'summary.csv').open('w',newline='') as stream:
        columns=['record','arm','selected_tokens','added_blocks','displaced_blocks',
                 'retained_raw_mean_k_mass','observed_mass_coselection','required_evidence_coverage',
                 'required_dependency_coselection','source_span_complete','dependency_span_complete',
                 'full_evidence_coselection','displaced_mean_k_evidence','duplicate_blocks',
                 'outdated_blocks','irrelevant_blocks','answer_correctness','propagation_ms']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for index,record in enumerate(results):
            for arm,row in record['arms'].items():
                writer.writerow(dict(record=index,arm=arm,selected_tokens=row['selected_tokens'],
                    added_blocks=len(row['added_starts']),displaced_blocks=len(row['displaced_starts']),
                    retained_raw_mean_k_mass=row['retained_raw_mean_k_mass'],
                    observed_mass_coselection=row['observed_mass_coselection'],
                    required_evidence_coverage=row['quality']['required_evidence_coverage'],
                    required_dependency_coselection=row['quality']['required_dependency_coselection'],
                    **{key:row['quality'][key] for key in ('source_span_complete','dependency_span_complete',
                        'full_evidence_coselection','displaced_mean_k_evidence','duplicate_blocks',
                        'outdated_blocks','irrelevant_blocks','answer_correctness')},
                    propagation_ms=1000*row['diagnostics']['scoring_seconds']))
    with (args.output_dir/'arm-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','metric','labeled_records','mean','total'])
        writer.writeheader()
        for arm,summary in output['summary'].items():
            for metric,value in summary.items():
                if isinstance(value,dict):
                    writer.writerow(dict(arm=arm,metric=metric,**value))
    print(json.dumps(dict(records=len(results),arms=len(ARMS),output=str(args.output_dir.resolve()))))
    return output


if __name__ == '__main__':
    main()
