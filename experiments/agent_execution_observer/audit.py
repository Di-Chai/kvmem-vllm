"""Read-only audit of frozen inputs and native decisions; never calls inference.

Reconstruct each request from raw current/past records and preceding responses.
Exact citation coordinates establish provenance, not semantic support.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from observer import (Memory, clipped, digest, extract_step, request_for, resolve,
                      save, task_instruction, validate)


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def normalize_boundaries(predictions):
    """Keep v1 output intact, correcting only unresolved -> existing-ID transitions."""
    previous, known, output = {}, {}, []
    for prediction in predictions:
        p = dict(prediction)
        wid = p['window_id']
        p['raw_stage_boundary'] = p['stage_boundary']
        p['boundary_normalization'] = None
        if (wid in previous and previous[wid] is None
                and p['stage_id'] in known.get(wid, set())
                and p['stage_boundary'] is True):
            p['stage_boundary'] = None
            p['boundary_normalization'] = 'v1_unresolved_previous_to_existing_id'
        previous[wid] = p['stage_id']
        if p['stage_id'] is not None:
            known.setdefault(wid, set()).add(p['stage_id'])
        output.append(p)
    return output


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return dict(count=len(values), minimum=values[0], mean=statistics.mean(values),
                median=statistics.median(values),
                p95=values[max(0, math.ceil(.95 * len(values)) - 1)], maximum=values[-1])


def diagnostics(predictions):
    grouped = {}
    for p in predictions:
        for qid, diagnostic in p['diagnostics'].items():
            grouped.setdefault(qid, []).append(diagnostic)
    return {qid: dict(label_mass=distribution([d['label_mass'] for d in rows]),
                     argmax_is_label_true=sum(d['argmax_is_label'] for d in rows),
                     argmax_is_label_false=sum(not d['argmax_is_label'] for d in rows),
                     label_mass_below_0_1=sum(d['label_mass'] < .1 for d in rows),
                     label_mass_below_0_5=sum(d['label_mass'] < .5 for d in rows))
            for qid, rows in grouped.items()}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(bundle, dataset, results, output):
    """Fail closed on any provenance, causal reconstruction or hash mismatch."""
    manifest = json.loads((dataset.parent / 'manifest.json').read_text())
    run = json.loads((results / 'run.json').read_text())
    summary = json.loads((results / 'summary.json').read_text())
    rows = load_lines(dataset)
    predictions = load_lines(results / 'decisions.jsonl')
    check(summary['status'] == 'completed', 'Run is not complete')
    check(len(rows) == len(predictions) == summary['steps'], 'Incomplete decision coverage')
    check(digest(dataset) == manifest['dataset_sha256'] == run['dataset_sha256'],
          'Dataset hash mismatch')
    hashes = {}
    for relative, expected in manifest['source_hashes'].items():
        actual = digest(bundle / relative)
        check(actual == expected, f'Source hash mismatch: {relative}')
        hashes[relative] = actual
    index = json.loads((bundle / 'index.json').read_text())
    trajectories = {r['trajectory_id']: r for r in index}
    records = {}
    memories, normalized, correction_rows = {}, [], []
    candidate_count = selected_count = request_count = 0
    matched_tool_returns = unreturned_tool_calls = 0
    original_keys = {'trajectory_id', 'window_id', 'step', 'stage_id', 'activity',
                     'outcome', 'selected_evidence', 'evidence_support_validated',
                     'status_revision', 'unresolved'}
    for step, prediction in zip(rows, predictions):
        tid, turn = step['trajectory_id'], step['step']
        item = trajectories[tid]
        if tid not in records:
            records[tid] = load_lines(bundle / item['trace_path'])
        raw = records[tid]
        current = [(i, r) for i, r in enumerate(raw, 1)
                   if r.get('record_type') == 'message' and r.get('turn') == turn]
        check(current[0][1]['original']['role'] == 'assistant'
              and all(r['original']['role'] != 'assistant' for _, r in current[1:]),
              'Step contains another assistant reply')
        following = [i for i, r in enumerate(raw, 1) if r.get('record_type') == 'message'
                     and r['original']['role'] == 'assistant' and r['turn'] > turn]
        check(not following or current[-1][0] < min(following), 'Tool return crosses next assistant reply')
        calls = {block['id'] for _, r in current for block in r['original']['content']
                 if block['type'] == 'tool_use'}
        returned = [block['tool_use_id'] for _, r in current for block in r['original']['content']
                    if block['type'] == 'tool_result']
        check(set(returned) <= calls and len(returned) == len(set(returned)),
              'Tool result refers to another Step or is duplicated')
        matched_tool_returns += len(returned)
        unreturned_tool_calls += len(calls - set(returned))
        reconstructed = extract_step(item, turn, current)
        # The instruction is taken only from the initial task record.
        initial = [(i, r) for i, r in enumerate(raw, 1) if r.get('turn') == 0]
        reconstructed.update(window_id=step['window_id'],
                             task_instruction=clipped(task_instruction(initial), 5000))
        check(reconstructed == step, f'Raw-source Step reconstruction differs: {tid} {turn}')
        for candidate in step['evidence'].values():
            src = candidate['source']
            record = raw[src['trace_line'] - 1]
            begin, end = src['character_span']
            check(src['trace_path'] == item['trace_path'] and src['step'] == turn
                  and record['turn'] == turn
                  and record['message_index'] == src['message_index'], 'Citation source mismatch')
            text = resolve(record, src['pointer'])
            check(0 <= begin <= end <= len(text) and text[begin:end] == candidate['quote'],
                  f'Citation span mismatch: {tid} {turn}')
            candidate_count += 1
        memory = memories.setdefault(step['window_id'], Memory())
        ident = hashlib.sha256(step['window_id'].encode()).hexdigest()[:12] + f'-{turn:04}'
        check(prediction['call_id'] == ident, 'Call/decision identity mismatch')
        payload = json.loads((results / 'calls' / f'{ident}.request.json').read_text())
        response = json.loads((results / 'calls' / f'{ident}.response.json').read_text())
        expected = request_for(step, memory, run['model'], run['mode'])
        check(json.dumps(payload, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False),
              f'Request contains different state/instructions or option order: {ident}')
        check(all(s['step'] < turn for s in payload['state']['execution_memory']['recent_steps']),
              'Recent memory contains current/future Step')
        validate(response, payload)
        result = memory.apply(step, response)
        check(all(prediction[k] == result[k] for k in original_keys),
              f'Decision differs from recorded native response: {ident}')
        check(prediction['native_answers'] == response['answers']
              and prediction['diagnostics'] == response['diagnostics']
              and prediction['usage'] == response['usage'], 'Response metadata differs')
        if result['stage_boundary'] != prediction['stage_boundary']:
            check(result['stage_boundary'] is None and prediction['stage_boundary'] is True,
                  'Unexpected client boundary discrepancy')
            correction_rows.append(dict(call_id=ident, step=turn, raw=True, normalized=None))
        normalized.append(dict(prediction, stage_boundary=result['stage_boundary'],
                               raw_stage_boundary=prediction['stage_boundary']))
        if prediction['selected_evidence'] is not None:
            check(prediction['selected_evidence'] == step['evidence'].get(response['answers']['evidence']['choice']),
                  'Selected evidence mismatch')
            selected_count += 1
        request_count += 1
    check(len(list((results / 'calls').glob('*.request.json'))) == request_count,
          'Unexpected extra/missing requests')
    check(len(list((results / 'calls').glob('*.response.json'))) == request_count,
          'Unexpected extra/missing responses')
    check(sum(p['usage']['input_tokens'] for p in predictions) == summary['input_tokens'],
          'Summary token mismatch')
    check(sum(p['usage']['output_tokens'] for p in predictions) == summary['label_reads'],
          'Summary label-read mismatch')
    check(math.isclose(sum(p['seconds'] for p in predictions), summary['total_seconds'], abs_tol=1e-8),
          'Summary latency mismatch')
    check(not (results / 'failure.json').exists(), 'Failure artifact present')
    check(json.loads((results / 'memory.json').read_text()) == {k: v.view() for k, v in memories.items()},
          'Final memory differs from causal replay')
    report = dict(status='passed', steps=len(rows), exact_request_reconstructions=request_count,
                  raw_step_reconstructions=len(rows), exact_evidence_candidates=candidate_count,
                  same_step_tool_returns_verified=matched_tool_returns,
                  tool_calls_without_observed_returns=unreturned_tool_calls,
                  selected_evidence_coordinates_valid=selected_count,
                  evidence_abstentions=len(rows) - selected_count,
                  semantic_evidence_support='Not implied by coordinate validation; see case review.',
                  memory_windows=len(memories), source_sha256=hashes,
                  dataset_sha256=digest(dataset), runtime_observer_sha256=run['observer_sha256'],
                  original_boundaries=summary['stage_boundaries'],
                  normalized_boundaries=sum(p['stage_boundary'] is True for p in normalized),
                  boundary_corrections=correction_rows,
                  uncertain_boundary_count=sum(p['stage_boundary'] is None for p in normalized),
                  omitted_or_clipped_steps=sum(bool(s['omitted_characters'] or s['current_serialization_clipped'])
                                               for s in rows),
                  left_censored_windows=sum(w['left_censored'] for w in manifest['windows']),
                  diagnostics_by_question=diagnostics(predictions),
                  latency_seconds=distribution([p['seconds'] for p in predictions]),
                  input_tokens_per_step=distribution([p['usage']['input_tokens'] for p in predictions]),
                  total_input_tokens=summary['input_tokens'], total_label_reads=summary['label_reads'],
                  human_gold_accuracy=None, human_gold_boundary_f1=None)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'audit.json', report)
    with (output / 'diagnostics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['window_id', 'step', 'question', 'choice',
            'label_mass', 'argmax_is_label', 'conditional_max_probability', 'native_confidence'])
        writer.writeheader()
        for p in predictions:
            for qid, diagnostic in p['diagnostics'].items():
                answer = p['native_answers'][qid]
                writer.writerow(dict(window_id=p['window_id'], step=p['step'], question=qid,
                    choice=answer['choice'], **diagnostic,
                    conditional_max_probability=max(answer['probabilities'].values()),
                    native_confidence=answer['confidence']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_run(args.bundle, args.dataset, args.results, args.output)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('source_sha256', 'diagnostics_by_question')}, ensure_ascii=False))
    print(json.dumps(report['diagnostics_by_question'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
