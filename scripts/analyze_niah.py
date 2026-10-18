"""Offline audit of frozen NIAH results; gold never enters inference requests."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from tokenizers import Tokenizer

from benchmark_niah import covered, save


def audit(campaign, reference, tokenizer_path, capture=None):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    summary = json.loads((campaign / 'summary.json').read_text())
    originals = {r['id']: r for r in json.loads((reference / 'results.json').read_text())}
    metadata = {r['id']: r for r in manifest['rows']}
    admissions = json.loads((campaign / 'cpu_admission.json').read_text())
    active, concurrency = set(), {}
    for event in admissions['events']:
        if event['event'] == 'admit':
            active.add(event['case'])
        else:
            active.discard(event['case'])
        for length in {metadata[case]['nominal_input_tokens'] for case in active}:
            count = sum(metadata[case]['nominal_input_tokens'] == length for case in active)
            concurrency[length] = max(concurrency.get(length, 0), count)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    native = json.loads((capture / 'archive.json').read_text()) if capture else None
    terminal_checked = 0
    pairs, evidence, failures, bindings = [], [], [], []
    for result in summary['results']:
        if result['status'] != 'completed':
            continue
        case = result['id']
        row, old = metadata[case], originals[case]
        source = campaign / 'inputs' / (case + '.i32')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == row['served_sha256'] == old['token_sha256'] == result['served_sha256']
        tokens = np.fromfile(source, dtype='<i4').tolist()
        assert len(tokens) == row['input_tokens'] == result['served_input_tokens']
        events = json.loads((campaign / 'cases' / case / 'events.json').read_text())
        response = json.loads((campaign / 'cases' / case / 'response.json').read_text())
        metrics = response['metrics']
        if native is not None:
            completed = [r for r in native['completed'] if r['request_id'].startswith(response['id'] + '-')]
            assert len(completed) == 1, 'Native terminal evidence missing or evicted'
            terminal = completed[0]
            expected_end = response['usage']['prompt_tokens'] + response['usage']['completion_tokens'] - 1
            assert terminal['history_end'] == expected_end
            assert terminal['accepted_tokens'] == expected_end
            assert terminal.get('retirement_finished_at')
            terminal_checked += 1
        queries = [e for e in events['bounded_refresh.jsonl'] if e['selection'] == 'query']
        assert len(queries) == 1
        query = queries[0]
        assert query['query_span'] == row['served_query']
        assert query['mean_k_backend'] == 'qw3_cuda'
        assert query['selected_tokens'] <= manifest['active_budget']
        assert len(events['query_history.jsonl']) == 1
        assert events['query_history.jsonl'][0]['observed']
        old_seconds = old['ingest_seconds'] + old['answer_seconds']
        pairs.append({'id': case, 'length': row['nominal_input_tokens'],
                      'old_request_seconds': old_seconds, 'new_request_seconds': result['seconds'],
                      'old_output_tokens': old['usage']['completion_tokens'],
                      'cpu_queue_seconds': result['cpu_queue_seconds'],
                      'new_with_cpu_admission_seconds': result['seconds'] + result['cpu_queue_seconds'],
                      'new_ttft_seconds': metrics['time_to_first_token_ms'] / 1000,
                      'new_generation_seconds': metrics['generation_time_ms'] / 1000,
                      'new_mean_itl_seconds': metrics['mean_itl_ms'] / 1000,
                      'input_tokens': row['input_tokens'],
                      'output_tokens': response['usage']['completion_tokens'],
                      'pressure_refreshes': result['pressure_refreshes'],
                      'assembly_seconds': sum(e['assembly_seconds'] for e in events['bounded_refresh.jsonl']),
                      'assembly_host_prepare_seconds': sum(e['assembly_host_prepare_seconds'] for e in events['bounded_refresh.jsonl']),
                      'guided_seconds': result['guided_seconds'],
                      'old_coverage': old['normalized_coverage'],
                      'new_coverage': result['evaluation']['normalized_coverage'],
                      'old_exact': old['normalized_exact_numeric_set'],
                      'new_exact': result['evaluation']['normalized_exact_numeric_set']})
        predicted = set(result['evaluation']['predicted_numbers'])
        for needle in row['needles']:
            segment = tokens[needle['start']:needle['end']]
            decoded = tokenizer.decode(segment, skip_special_tokens=False)
            # Needle text is ASCII. Prefix lengths preserve tokenizer boundaries.
            assert decoded.isascii()
            offsets = [len(tokenizer.decode(segment[:i], skip_special_tokens=False))
                       for i in range(len(segment) + 1)]
            item = {'case': case, 'length': row['nominal_input_tokens'],
                    'key': needle['key'], 'value': needle['value'],
                    'sentence_selected': covered(needle['start'], needle['end'], query['selected_history_ranges']),
                    'number_returned': needle['value'] in predicted}
            for field in ('key', 'value'):
                begin = decoded.index(needle[field])
                end = begin + len(needle[field])
                overlap = [i for i in range(len(segment)) if offsets[i] < end and offsets[i + 1] > begin]
                assert overlap
                span = [needle['start'] + overlap[0], needle['start'] + overlap[-1] + 1]
                item[field + '_span'] = span
                item[field + '_selected'] = covered(*span, query['selected_history_ranges'])
            evidence.append(item)
        if not result['evaluation']['normalized_exact_numeric_set']:
            failures.append({'id': case, 'coverage': result['evaluation']['normalized_coverage'],
                             'expected': row['expected'], 'predicted': sorted(predicted),
                             'final_answer': result['evaluation']['final_answer']})
        elif row['task'] == 'niah_multiquery':
            answer = result['evaluation']['final_answer']
            mismatches, unverified = [], []
            for key, expected in zip(row['keys'], row['expected']):
                lines = [line for line in answer.splitlines() if key in line]
                if len(lines) == 1 and sum(k in lines[0] for k in row['keys']) == 1:
                    found = [value for value in row['expected'] if value in lines[0].replace(',', '')]
                    if found and found != [expected]:
                        mismatches.append({'key': key, 'expected': expected, 'found': found})
                    elif not found:
                        unverified.append(key)
                else:
                    unverified.append(key)
            bindings.append({'id': case, 'length': row['nominal_input_tokens'],
                             'mismatches': mismatches, 'unverified': unverified})

    groups = []
    for length in sorted({p['length'] for p in pairs}):
        matched = [p for p in pairs if p['length'] == length]
        needles = [e for e in evidence if e['length'] == length]
        missing = [e for e in needles if not e['number_returned']]
        case_ids = {p['id'] for p in matched}
        times = [e['at'] for e in admissions['events'] if e['case'] in case_ids]
        # Each phase includes boundary overlap with adjacent lengths, and the
        # 64K preflight/main gap. Do not sum phase spans or call them isolated runs.
        group = {'length': length, 'matched_cases': len(matched),
                 'peak_admitted_concurrency': concurrency[length],
                 'old_mean_seconds': statistics.mean(p['old_request_seconds'] for p in matched),
                 'new_mean_seconds': statistics.mean(p['new_request_seconds'] for p in matched),
                 'new_mean_with_cpu_admission_seconds': statistics.mean(p['new_with_cpu_admission_seconds'] for p in matched),
                 'old_mean_output_tokens': statistics.mean(p['old_output_tokens'] for p in matched),
                 'new_mean_output_tokens': statistics.mean(p['output_tokens'] for p in matched),
                 'new_mean_ttft_seconds': statistics.mean(p['new_ttft_seconds'] for p in matched),
                 'new_mean_generation_seconds': statistics.mean(p['new_generation_seconds'] for p in matched),
                 'new_median_itl_seconds': statistics.median(p['new_mean_itl_seconds'] for p in matched),
                 'cpu_queue_max_seconds': max(p['cpu_queue_seconds'] for p in matched),
                 'mean_pressure_refreshes': statistics.mean(p['pressure_refreshes'] for p in matched),
                 'assembly_seconds_sum': sum(p['assembly_seconds'] for p in matched),
                 'assembly_host_prepare_seconds_sum': sum(p['assembly_host_prepare_seconds'] for p in matched),
                 'guided_seconds_sum': sum(p['guided_seconds'] for p in matched),
                 'old_serial_request_seconds': sum(p['old_request_seconds'] for p in matched),
                 'current_phase_span_seconds': max(times) - min(times),
                 'new_faster_cases': sum(p['new_request_seconds'] < p['old_request_seconds'] for p in matched),
                 'old_coverage': statistics.mean(p['old_coverage'] for p in matched),
                 'new_coverage': statistics.mean(p['new_coverage'] for p in matched),
                 'old_exact': sum(p['old_exact'] for p in matched),
                 'new_exact': sum(p['new_exact'] for p in matched),
                 'needle_count': len(needles),
                 'sentence_selected': sum(e['sentence_selected'] for e in needles),
                 'key_selected': sum(e['key_selected'] for e in needles),
                 'value_selected': sum(e['value_selected'] for e in needles),
                 'missing_numbers': len(missing),
                 'missing_with_key_and_value_selected': sum(e['key_selected'] and e['value_selected'] for e in missing),
                 'missing_with_value_selected_only': sum(not e['key_selected'] and e['value_selected'] for e in missing),
                 'missing_value_unselected': sum(not e['value_selected'] for e in missing),
                 'numeric_exact_but_explicit_key_binding_mismatch': sum(bool(b['mismatches']) for b in bindings if b['length'] == length)}
        groups.append(group)
    resources = [json.loads(line) for line in (campaign / 'resources.jsonl').read_text().splitlines() if line]
    if len(pairs) == manifest['cases']:
        last_release = max(e['at'] for e in admissions['events'] if e['event'] == 'release')
        resources = [r for r in resources if r['wall_time'] <= last_release]
    storage_samples = [r['archive'] for r in resources if r.get('archive')]
    memory = [r['meminfo']['MemAvailable'] for r in resources if r.get('meminfo')]
    phase_limit = ('Whole-campaign span includes an interruption and restart; per-case latencies '
                   'cover successful attempts only.' if admissions.get('interruption_included') else
                   'Phase spans include adjacent-length overlap and the 64K preflight/main pause.'
                   if len({p['length'] for p in pairs}) > 1 else
                   'Single-length campaign span includes all admission and final retirement waits.')
    output = {'completed': len(pairs), 'expected': manifest['cases'], 'groups': groups,
              'terminal_histories_verified': terminal_checked if native is not None else None,
              'comparison_limits': ['Old Q8 + CacheBlend 10%, new NVFP4 without recomputation.',
                  'Old ingest+answer latency excludes startup/reset; new server already running.',
                  'New request time excludes CPU admission; both timings provided, neither includes executor backlog.',
                  phase_limit],
              'storage_audit': {'samples': len(storage_samples),
                  'all_cpu': bool(storage_samples) and all(r['archive_backend'] == 'cpu' for r in storage_samples),
                  'max_disk_read_bytes': max(r['disk_read_bytes'] for r in storage_samples),
                  'max_disk_write_bytes': max(r['disk_write_bytes'] for r in storage_samples),
                  'max_nvme_capacity_bytes': max(r['nvme_capacity_bytes'] for r in storage_samples),
                  'minimum_available_memory_bytes': min(memory),
                  'max_live_accepted_history_tokens': max(sum(r.get('progress', {}).get('accepted_history', {}).values()) for r in resources),
                  'max_observed_histories_at_least_262101_tokens': max(sum(n >= 262101 for n in r.get('progress', {}).get('accepted_history', {}).values()) for r in resources),
                  'max_observed_archive_used_slots': max(r.get('progress', {}).get('archive_used_slots', 0) for r in resources),
                  'any_swap': any(r.get('meminfo', {}).get('SwapTotal', 0) for r in resources),
                  'any_archive_file_descriptors': any(r.get('archive_file_descriptors') for r in resources)},
              'failures': failures, 'pairs': pairs, 'evidence': evidence,
              'supplementary_explicit_bindings': bindings}
    save(campaign / 'offline-audit.json', output)
    print(json.dumps({k: v for k, v in output.items() if k not in ('pairs', 'evidence', 'failures', 'supplementary_explicit_bindings')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--capture', type=Path, help='Optional live capture with retained terminal evidence')
    args = parser.parse_args()
    audit(args.campaign, args.reference, args.tokenizer, args.capture)
