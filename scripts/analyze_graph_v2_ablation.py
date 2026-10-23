"""Join real four-arm selected pages and replay one immutable known snapshot.

No model or judge is called. Native replay verifies every actual selected page;
counterfactual arms share the exact seed/graph/masks and have no quality labels.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import benchmark_graph_v2_ablation as campaign
from graph_v2_protocol import require, sha
from qualify_block_graph import digest, native_matches, records


def full_selected_starts(record, query):
    """Working-record ranges are [logical begin, count], never [begin, end]."""
    directory = dict(zip(record['starts'], record['counts']))
    ranges = query['selected_history_ranges']
    require(ranges and all(isinstance(r, (tuple, list)) and len(r) == 2 and
        type(r[0]) is int and type(r[1]) is int and directory.get(r[0]) == r[1]
        for r in ranges), 'Actual selected range is missing, partial, or outside the frozen Query directory')
    selected = sorted(r[0] for r in ranges)
    require(len(set(selected)) == len(selected), 'Duplicate actual selected block')
    used = sum(r[1] for r in ranges)
    require(used == query['selected_tokens'] and used <= record['budget'], 'Actual native selected-token budget differs')
    require(query['history_begin'] == record['accepted_boundary'] == query['replay_span'][0],
        'Query replay boundary differs from the captured scoring directory')
    return selected


def derived_record(record, arm):
    require(record.get('version') == 2 and record.get('phase') == 'query', 'Require an actual versioned Query snapshot')
    require(record.get('aggregation', 'sum') == 'sum', 'Historical four-arm ablation requires sum aggregation')
    value = deepcopy(record)
    version, mode, known = campaign.POLICY[arm]
    value.update(scoring_version=version, mode=mode, alpha=.2,
        locality_weight=0., attention_weight=.2, cross_message_weight=.25)
    if not known:
        value['graph'].update(message_spans=[], message_end_known=[],
            locality_provenance=dict(source='unknown', reason='explicit_graph_only_metadata_ablation'))
    value.pop('evidence', None)
    return value


def replay(record, arm):
    # Reuse the existing native replay path, including its full baseline and
    # generation/mask validation. There is no second Python page picker here.
    import analyze_block_graph as native
    from vllm_kvmem.graph_scoring import score_graph
    original_hash = digest(record)
    derived = derived_record(record, arm)
    whole = native.analyze_record(derived)
    selected = whole['arms'][campaign.POLICY[arm][1]]
    cfg, snapshot = native._validate_record(derived)
    with score_graph(derived['raw_scores'], derived['starts'], derived['transport_mask'], snapshot,
            mode=campaign.POLICY[arm][1], scoring_version=campaign.POLICY[arm][0], alpha=.2,
            locality_weight=0., attention_weight=.2, cross_message_weight=.25,
            block_tokens=cfg.block_tokens) as result:
        vector = list(result.scores)
    require(digest(record) == original_hash, 'Replay mutated the shared captured snapshot')
    return dict(selected, score_vector=vector, score_vector_sha256=digest(vector),
        original_record_sha256=original_hash, derived_record_sha256=digest(derived),
        allowed_transform=dict(version=campaign.POLICY[arm][0], mode=campaign.POLICY[arm][1],
                               alpha=.2 if campaign.POLICY[arm][0] == 'v2' else None,
                               legacy_attention_weight=.2 if campaign.POLICY[arm][0] == 'v1' else None,
                               known_boundaries=campaign.POLICY[arm][2]))


def audit_live_case(capture, response, case, profile, arm, input_value):
    matches = lambda event:native_matches(response, event)
    selections = [r for r in records(capture/'bounded_refresh.jsonl') if matches(r)]
    queries = [r for r in selections if r.get('selection') == 'query']
    pressure = [r for r in selections if r.get('selection') == 'pressure']
    require(len(queries) == 1 and len(pressure) >= 2, 'Fixed long input lacks one Query and multiple pressure selections')
    query = queries[0]
    require(query['query_span'] == input_value['plan']['kvmem_query_span'] and
        query.get('mean_k_backend') == 'qw3_cuda' and query.get('query_history_enabled') is False and
        query.get('action_history_pages') == 0, 'Public Query changed or inherited output feedback')
    inputs = [r for r in records(capture/'block_graph_inputs.jsonl') if matches(r) and r.get('phase') == 'query']
    require(len(inputs) == 1, 'Missing or duplicate complete Query replay record; bounded export may have refused it')
    record = inputs[0]
    version, mode, known = campaign.POLICY[arm]
    require(record['scoring_version'] == version and record['mode'] == mode and
        query.get('pressure_policy') == 'recency', 'Actual scoring/pressure policy differs from the declared arm')
    require(record['config']['select_budget'] == profile['kvmem']['select_budget'] and
        record['config']['gen_budget'] == profile['kvmem']['gen_budget'], 'Captured native K/G changed')
    if known:
        require(record['graph']['message_spans'] and record['graph']['locality_provenance'] ==
            input_value['plan']['kvmem_locality_provenance'], 'Known arm lost the real producer metadata')
    else:
        require(not record['graph']['message_spans'], 'Unknown arm inherited message boundaries')
    actual = full_selected_starts(record, query)
    replayed = replay(record, arm)
    require(replayed['selected_starts'] == actual, 'Complete actual native selected pages differ from own-snapshot replay')
    return dict(id=case['id'], arm=arm, input_sha256=case['input_sha256'],
        record_sha256=digest(record), query=query, pressure_events=len(pressure),
        actual_selected_starts=actual, replay=replayed, actual_selection_matches_native_replay=True,
        response_sha256=digest(response), answer_correctness=None), record


def compare(left, right):
    a, b = set(left['selected_starts']), set(right['selected_starts'])
    require(len(left['score_vector']) == len(right['score_vector']), 'Counterfactual score domains differ')
    return dict(added_starts=sorted(b-a), displaced_starts=sorted(a-b),
        selected_jaccard=len(a & b)/len(a | b) if a | b else 1.,
        score_l1=math.fsum(abs(x-y) for x,y in zip(left['score_vector'],right['score_vector'])),
        answer_correctness=None, retrieval_accuracy=None)


def effective_boundaries(record):
    from vllm_kvmem.graph_scoring import locality_boundary
    block = record['config']['block_tokens']
    graph = record['graph']
    available = {start:(score,keep) for start,score,keep in
        zip(record['starts'],record['raw_scores'],record['transport_mask'])}
    pairs = []
    for source, (score, keep) in available.items():
        if not keep or score <= 0: continue
        for target in (source-block, source+block):
            if target not in available or not available[target][1]: continue
            coefficient, fallback = locality_boundary(source,target,block,graph['message_spans'],.25,
                                                       graph['message_end_known'])
            if coefficient == .25 and not fallback:
                pairs.append(dict(source=source,target=target,positive_seed=score))
    return dict(positive_seed_cross_boundary_pairs=pairs, covered=bool(pairs))


def analyze(root):
    manifest = campaign.validate(root)
    require(Path(__file__).resolve() == root/'frozen/analyze_graph_v2_ablation.py', 'Run the frozen analyzer')
    require(json.loads((root/'status.json').read_text())['status'] == 'passed', 'All four real model arms must pass first')
    # The analyzer itself is frozen, and its native selector comes from the
    # same hashed runtime as all real model arms.
    sys.path.insert(0, str(root/campaign.ARMS[0]/'runtime/src'))
    fixture = json.loads((root/'fixture.json').read_text())
    live, canonical = [], {}
    for arm in campaign.ARMS:
        directory = root/arm
        results = json.loads((directory/'qualification/results.json').read_text())
        require(results['status'] == 'passed' and results['all_owned_sessions_released'], 'An arm failed or leaked an owner')
        profile = json.loads((directory/'profile.json').read_text())
        for case in fixture['cases']:
            row = json.loads((directory/'qualification'/case['id']/'response.json').read_text())
            value = json.loads(Path(case['input']).read_text())
            proof, record = audit_live_case(directory/'server/capture', row['row'], case, profile, arm, value)
            live.append(proof)
            if arm == manifest['canonical_arm']:
                canonical[case['id']] = record
    comparisons = []
    for case in fixture['cases']:
        record = canonical[case['id']]
        variants = {arm:replay(record, arm) for arm in campaign.ARMS}
        pairs = [(campaign.ARMS[i],campaign.ARMS[i+1]) for i in range(3)]
        deltas = {left+'__'+right:compare(variants[left],variants[right]) for left,right in pairs}
        coverage = effective_boundaries(record)
        normalization = variants['v2-attention']['diagnostics']
        norm_covered = normalization['attention_external_mass'] > variants['v1-attention']['diagnostics']['attention_external_mass']
        boundary_delta = deltas['v2-multi-unknown__v2-multi-known']['score_l1']
        comparisons.append(dict(id=case['id'], canonical_record_sha256=digest(record),
            variants=variants, comparisons=deltas, boundary_evidence=coverage,
            normalization_effect_covered=norm_covered,
            boundary_effect_covered=coverage['covered'] and boundary_delta > 0))
    qualified = all(row['normalization_effect_covered'] and row['boundary_effect_covered'] for row in comparisons)
    report = dict(schema=campaign.SCHEMA, status='passed' if qualified else 'inconclusive',
        real_model_requests=len(live), own_snapshot_full_native_selection_matches=len(live),
        canonical_snapshots=len(canonical), arms=list(campaign.ARMS), live=live, canonical_replay=comparisons,
        quality_gold=None, answer_correctness=None, retrieval_accuracy=None,
        scope='Real model requests and exact own-snapshot native selection closure; fixed canonical counterfactuals isolate algorithms, not answer quality',
        note='No selected-page change is required. Missing effective normalization or boundary coverage is inconclusive, never fabricated success.')
    output = root/'analysis'
    require(not output.exists(), 'Never overwrite previous ablation analysis')
    output.mkdir()
    campaign.save(output/'results.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('live','canonical_replay')}), flush=True)
    require(qualified, 'Real fixed-input operator/boundary effect was not covered; see retained inconclusive report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    analyze(parser.parse_args().campaign.resolve())


if __name__ == '__main__':
    main()
