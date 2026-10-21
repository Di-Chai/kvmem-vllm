"""Separate physical shape changes from original logical policy evidence.

The ordinary variant comparison remains intact, including every failed exact
equality check. This adds logical expansion and sampling-boundary checks; it
does not relabel numerical or output differences as identical semantics.
"""
import argparse
from bisect import bisect_left
from pathlib import Path

import benchmark_prefill_variant as variant
import benchmark_prefill_publication_pipeline as pipeline
from benchmark_active_prefill import reconstruct_geometry
from qualify_block_graph import native_matches
from benchmark_prefill_publication_pipeline import read, save, rows, require, sha


SAMPLING_KEYS = ('end','query_samples','samples','chunks','pages')


def sampling_index(trace, label):
    """Require actual, unique fresh-request commits before comparing them."""
    require(bool(trace), label+' sampling evidence is missing')
    indexed = {}
    last_end = samples = chunks = 0
    for row in trace:
        require(all(type(row.get(key)) is int and row[key] > 0 for key in SAMPLING_KEYS),
                label+' sampling report has invalid integer counters')
        end = row['end']
        require(end not in indexed, label+' has duplicate sampling boundaries')
        require(end > last_end and row['samples'] == samples+row['query_samples'] and
                row['chunks'] == chunks+1, label+' sampling reports are not an ordered complete commit sequence')
        indexed[end] = row
        last_end,samples,chunks = end,row['samples'],row['chunks']
    return indexed


def audit_logical_sampling(publications, baseline_trace, candidate_trace, profile):
    """Check row phase against frozen stride and original commit boundaries.

    Baseline reports establish whether each original logical boundary sampled;
    they do not record row indices. The frozen policy's end-anchored stride
    establishes those indices, which the candidate publication records directly.
    Segment counts are checked as bounded complete partitions, not equated to
    unique historical page counts after active-view reordering.
    """
    base = sampling_index(baseline_trace,'Reference')
    candidate = sampling_index(candidate_trace,'Candidate')
    kv = profile['kvmem']
    stride,block,interval = (kv['prefill_attention_stride'],kv['block_tokens'],
                            kv['prefill_attention_interval'])
    capacity = kv['select_budget']+kv['gen_budget']
    require(type(stride) is int and stride >= 0 and type(block) is int and block > 0 and
            type(interval) is int and interval > 0 and type(capacity) is int and capacity > 0,
            'Frozen sampling/block geometry is invalid')
    require(bool(publications),'No physical publications support the logical sampling audit')
    checks = dict(sampling_boundaries_identical=baseline_trace == candidate_trace,
                  sampled_row_indices_identical=True, logical_reports_identical=True,
                  segment_evidence_valid=True)
    checked_spans = checked_rows = 0
    seen_ends = set()
    reference_ends = sorted(base)
    for publication in publications:
        previous = None
        for span in publication['spans']:
            count,begin,local = span['count'],span['history_begin'],span['local_begin']
            end = begin+count
            require(end not in seen_ends,'Duplicate logical sampling span')
            seen_ends.add(end)
            recorded = span.get('sampled_row_indices')
            require(isinstance(recorded,list) and all(type(row) is int for row in recorded) and
                    all(0 <= row < count for row in recorded) and
                    all(a < b for a,b in zip(recorded,recorded[1:])),
                    'Logical publication has malformed sampled row indices')
            require(type(span.get('sampled_rows')) is int and span['sampled_rows'] == len(recorded),
                    'Logical publication sample count differs from its row indices')
            old,new = base.get(end),candidate.get(end)
            at = bisect_left(reference_ends,end)
            previous_end = reference_ends[at-1] if at else 0
            near_capacity = capacity-local-count <= count+block
            sampled = not (previous_end and end-previous_end < interval and not near_capacity)
            expected = list(reversed(range(count-1,-1,-(stride or count)))) if sampled else []
            checks['sampled_row_indices_identical'] &= recorded == expected
            checks['logical_reports_identical'] &= (old == new and (old is not None) == bool(expected) and
                (old is None or old['query_samples'] == len(expected)))
            # No sampled rows means no successful PrefillHistory commit/report.
            checks['logical_reports_identical'] &= (new is not None) == bool(recorded)
            segments = span.get('segments')
            require(type(segments) is int and segments > 0 and type(local) is int and local >= 0,
                    'Logical publication has invalid segment metadata')
            complete = (local+count+block-1)//block <= segments <= local+count
            if new is not None:
                complete &= new['pages'] <= segments
            if previous is not None:
                # Qualified groups start on a logical block boundary; their
                # second full 2K suffix appends exactly count/block new pieces.
                complete &= count % block == 0 and segments == previous+count//block
            checks['segment_evidence_valid'] &= complete
            previous = segments
            checked_spans += 1
            checked_rows += len(recorded)
    return dict(checks=checks,checked_logical_spans=checked_spans,checked_sampled_rows=checked_rows,
        scope='Candidate row indices versus frozen end-anchored stride and original reference commit '
              'boundaries; bounded segment counts, not segment identities or per-page numerical equality.')


def expand_logical_geometry(geometry, publications, maximum=8):
    require(type(maximum) is int and 2 <= maximum <= 8,'Invalid frozen physical group bound')
    indexed = {}
    for item in publications:
        begin = item['history_begin']
        require(begin not in indexed, 'Duplicate physical publication boundary')
        spans = item['spans']
        require(item['count'] == item['retained'] == 2048*len(spans) and
                item['confirmation_step'] > item['scheduler_step'],
                'Benchmark requires fully confirmed ordinary physical groups')
        require(2 <= len(spans) <= maximum and all(span['count'] == 2048 for span in spans),
                'A physical group must preserve its bounded original 2K spans')
        for i, span in enumerate(spans):
            require(span['history_begin'] == begin+2048*i and
                    span['local_begin'] == item['local_begin']+2048*i,
                    'Physical publication has a discontinuous logical plan')
        indexed[begin] = item
    result, current, used = [], 0, set()
    for shape in geometry:
        record = indexed.get(current)
        if record is None:
            result.append(dict(shape))
        else:
            require(not shape['prefill_terminal'] and shape['query_tokens'] == record['count'] and
                    record['local_begin'] == current-shape['offset'],
                    'Published group differs from the actual physical forward')
            for span in record['spans']:
                result.append(dict(shape, query_tokens=span['count'],
                    active_seq_len=span['local_begin']+span['count'],
                    logical_seq_len=span['history_begin']+span['count']))
            used.add(current)
        current += shape['query_tokens']
    require(used == set(indexed), 'Physical publication has no matching model forward')
    return result


def analyze(root):
    base, candidate = root/'reference', root/'candidate'
    profiles = {name:read(arm/'profile.json') for name,arm in (('base',base),('candidate',candidate))}
    changed = variant.differences(profiles['base'],profiles['candidate'])
    allowed = {'kvmem.prefill_physical_coalescing','kvmem.prefill_physical_max_chunks'}
    require(bool(changed) and all(row['path'] in allowed and row['after_present'] for row in changed),
            'Only the explicit physical coalescing flag/bound may change')
    patch = {'kvmem':{row['path'].split('.')[-1]:row['after'] for row in changed}}
    result = variant.compare(base,candidate,expected_overrides=patch)
    enabled = {name:profile['kvmem'].get('prefill_physical_coalescing',False) for name,profile in profiles.items()}
    require(all(type(value) is bool for value in enabled.values()) and enabled['candidate'],
            'Candidate must explicitly enable physical coalescing')
    maximum = {name:profile['kvmem'].get('prefill_physical_max_chunks',2) for name,profile in profiles.items()}
    suite = read(base/'suite.json')['cases']
    groups = {name:pipeline.prefill_groups(rows(arm/'run/server/capture/model_step_latency.jsonl'))
              for name,arm in (('base',base),('candidate',candidate))}
    logical_pairs = []
    total_groups = 0
    for pair, item in zip(result['pairs'], (item for item in suite if not item['warmup'])):
        index = suite.index(item)
        geometries, traces, publications = {}, {}, {}
        for name,arm in (('base',base),('candidate',candidate)):
            directory = arm/'requests'/item['id']
            response, evidence = read(directory/'response.json'), read(directory/'evidence.json')
            measured = read(directory/'result.json')
            capture = arm/'run/server/capture'
            geometries[name] = reconstruct_geometry(groups[name][index], evidence['refresh'],
                                                     measured['input_tokens'])
            publications[name] = [row for row in rows(capture/'logical_prefill_publication.jsonl')
                                  if native_matches(response,row)]
            traces[name] = [{key:row[key] for key in SAMPLING_KEYS}
                for row in rows(capture/'prefill_attention.jsonl') if native_matches(response,row)]
        require(bool(publications['base']) == enabled['base'],
                'Reference publication evidence differs from its physical coalescing mode')
        require(publications['candidate'], 'Enabled candidate did not execute physical coalescing')
        original = expand_logical_geometry(geometries['base'],publications['base'],maximum['base'])
        expanded = expand_logical_geometry(geometries['candidate'], publications['candidate'],maximum['candidate'])
        sampling = audit_logical_sampling(publications['candidate'],traces['base'],traces['candidate'],
                                          profiles['base'])
        baseline_sampling = (audit_logical_sampling(publications['base'],traces['base'],traces['base'],
                                                    profiles['base']) if enabled['base'] else None)
        checks = dict(expanded_geometry_identical=expanded == original,**sampling['checks'])
        if baseline_sampling is not None:
            checks['reference_sampling_valid'] = all(baseline_sampling['checks'].values())
        total_groups += len(publications['candidate'])
        logical_pairs.append(dict(id=pair['id'], checks=checks,
            physical_groups=len(publications['candidate']),
            reference_physical_groups=len(publications['base']),
            candidate_group_sizes=sorted({len(row['spans']) for row in publications['candidate']}),
            original_steps=len(geometries['base']), actual_steps=len(geometries['candidate']),
            expanded_steps=len(expanded), original_sampling=traces['base'],
            candidate_sampling=traces['candidate'],sampling_audit=sampling,
            reference_sampling_audit=baseline_sampling))
    result['logical_policy_evidence'] = dict(pairs=logical_pairs, formal_physical_groups=total_groups,
        all_checks_passed=all(all(pair['checks'].values()) for pair in logical_pairs),
        scope='Shape expansion, sampling rows/transaction boundaries and bounded segment-count evidence only. '
              'Exact returned token IDs, selected ranges and reported score/graph summary checks remain '
              'in the original pairs. No per-page score or graph-edge weight equality is established; '
              'numerical differences are not silently accepted.')
    result['logical_analyzer'] = dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))
    save(root/'logical-comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root.resolve())
    import json
    print(json.dumps(dict(exact_physical_comparison=result['all_checks_passed'],
        logical_policy_checks=result['logical_policy_evidence']['all_checks_passed'],
        summary=result['summary']),indent=2))
