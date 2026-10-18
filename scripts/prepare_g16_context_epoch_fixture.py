"""Prepare a separate natural context-epoch contract from unchanged canonical data.

No tokenizer, model, GPU, runtime edits, source reselection or source replay.
The original strict16 rejection remains immutable and part of the evidence.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import freeze_g16_long_history_fixture as strict


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'g16_long_history_context_epoch_v1'
WORKLOAD = dict(concurrency=8, warmup_turns=2, epoch_turns=16,
    duration_seconds=300, mode='tools', generation_mode='natural', generation_limit=16384)
CONTRACT = dict(
    version=EXPERIMENT,
    maximum_epoch_turns=16,
    source_records_per_task=16,
    request_generation_ceiling=16384,
    kvmem_select_budget=65536,
    kvmem_gen_budget=32768,
    logical_context_ceiling=262144,
    admission='Each arm uses its own complete accepted output IDs; admit the next full prompt only '
              'when prompt_tokens+16384<=262144.',
    epoch_end='End before an inadmissible next complete prompt or after16 completed turns; '
              'await the current agent request and one-token namespace cleanup, then restart the same '
              'unaltered initial source prompt as a new independent epoch.',
    retirement='Per-agent awaited cleanup; no global barrier or per-epoch global-quiescence claim. '
               'Existing warmup barrier and final normal drain remain unchanged.',
    source_tail='All16 original records and full external-message tokens remain frozen. Unissued tail '
                'turns may remain unexecuted in a context-limited epoch; no partial request is sent.',
    warmup='Two complete natural warmup requests are mandatory per agent before T0. If actual own '
           'outputs leave insufficient room, the run fails without retry, replacement, cap reduction or clipping.',
    memory='All admitted history remains intact until logical epoch termination. No pressure-driven '
           'history deletion, SSD, recomputation or reduced CPU admission estimate.',
    source_outputs='Unmodified source accepted output IDs remain provenance only. They are never '
                   'substituted for model output or used as fixed quotas.',
    scoring='All cold epoch restarts, cleanup latency and prefill inside the fixed window remain charged; '
            'cleanup tokens stay separately reported and excluded from effective workload tokens.',
    termination='Natural EOS and length terminations remain separately visible. T1 stops new offers; '
                'all already offered requests drain without cancellation.')


def sha(path):
    return strict.base.sha(path)


def ids_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def prepare(args):
    strict.guard(args)
    started = time.time()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh independent output directory')
    verification = strict.base.read(source/'verification.json')
    if verification.get('all_pass') is not True:
        raise ValueError('Original canonical audit was not completed')
    original_hashes = verification['files_sha256']
    if any(sha(source/name) != digest for name, digest in original_hashes.items()):
        raise ValueError('Original strict qualification artifacts changed')
    original = strict.base.read(source/'rejected-canonical-fixture.json')
    rejected = strict.base.read(source/'qualification.json')
    evidence = strict.base.read(source/'canonical-evidence.json')
    if rejected['status'] != 'rejected_no_gpu_run' or not rejected['fullplan_query_consistency_verified']:
        raise ValueError('Require preserved strict16 rejection and complete canonical qualification')
    expected_failures = [
        ('koota-composite-trait-aspects', 180, 520),
        ('koota-composite-trait-aspects', 181, 871),
        ('wasmi-trap-coredumps', 102, 15),
    ]
    actual_failures = [(r.get('task'), r.get('source_turn'), r.get('overflow_tokens'))
                       for r in rejected['failures']]
    if actual_failures != expected_failures:
        raise ValueError('Original failure reasons changed; do not silently relax additional failures')
    plans = original['agent_plans']
    if [(p['source_task'], p['source_turn']) for p in plans] != strict.EXPECTED:
        raise ValueError('Fixed eight source tasks/starts changed')
    if original['protocol'] != strict.LIMITS:
        raise ValueError('K64/G32/ctx256/request16K/CPU contracts changed')
    reference_path = args.eos_reference.resolve()
    reference = strict.base.read(reference_path)
    if (reference.get('provenance', {}).get('tokenizer_file_sha256') !=
            original['provenance']['tokenizer_file_sha256']):
        raise ValueError('EOS reference uses a different tokenizer identity')
    eos = reference.get('eos')
    if type(eos) is not int or eos < 0:
        raise ValueError('EOS must come from the previously validated frozen natural fixture')
    warmup_proofs = []
    for plan, proof in zip(plans, evidence):
        if (proof['task'] != plan['source_task'] or len(plan['turns']) !=16 or
                len(proof['turns']) !=16):
            raise ValueError('Complete16 source rows and evidence must remain present')
        initial = len(plan['initial_tokens'])
        if initial+strict.CAP > strict.CONTEXT:
            raise ValueError('Complete canonical initial prompt plus request ceiling does not fit')
        if any(t['canonical_prompt_tokens']+strict.CAP > strict.CONTEXT for t in proof['turns'][:2]):
            raise ValueError('Reference source-output two-warmup replay does not fit')
        headroom = strict.CONTEXT-strict.CAP-initial-len(plan['turns'][1]['tool_tokens'])
        if headroom < 1:
            raise ValueError('Even a one-token first completion leaves no room for second warmup')
        warmup_proofs.append(dict(task=plan['source_task'], source_turn=plan['source_turn'],
            canonical_initial_tokens=initial, full_second_tool_tokens=len(plan['turns'][1]['tool_tokens']),
            maximum_first_own_output_tokens_allowing_second_warmup=headroom,
            source_first_output_tokens=proof['turns'][0]['source_accepted_output_count'],
            source_second_prompt_tokens=proof['turns'][1]['canonical_prompt_tokens'],
            source_two_warmups_fit=True, actual_own_output_two_warmups_not_yet_measured=True))
    strict.guard(args)
    fixture = copy.deepcopy(original)
    fixture['scope'] = ('Independent natural external-message performance replay with logical context epochs; '
                        '16 maximum source turns per epoch, not strict16 completion qualification.')
    fixture['eos'] = eos
    fixture['experiment_kind'] = EXPERIMENT
    fixture['context_epoch_contract'] = CONTRACT
    fixture['selection'].update(users=8, turns=16,
        manifest_order='longest eligible eight distinct tasks from original manifest first16; frozen before canonical render',
        source_task_and_start_revision=False)
    fixture['semantics'].update(
        epoch='Maximum16 source turns, ending earlier only at the predeclared full logical context guard.',
        eos='Own natural EOS or actual request ceiling16384; preserve and report all length terminations.',
        strict16='Original strict16 canonical qualification remains rejected and immutable.')
    extra_paths = [source/name for name in original_hashes]
    extra_paths.extend((source/'verification.json', reference_path, Path(__file__).resolve()))
    fixture['provenance']['source_file_sha256'].update({str(p): sha(p) for p in extra_paths})
    if fixture['agent_plans'] != original['agent_plans'] or fixture['protocol'] != original['protocol']:
        raise AssertionError('Canonical initial IDs, external tokens/waits, Query, source outputs or budgets changed')
    fingerprint = ids_sha(fixture['agent_plans'])
    qualification = dict(schema='g16_context_epoch_cpu_preparation_v1',
        status='prepared_context_epoch_cpu_only_no_gpu_run', strict16_status='rejected_unchanged',
        started_at=started, finished_at=time.time(), preparation_seconds=time.time()-started,
        parent_rejected_directory=str(source), parent_rejected_hashes=original_hashes,
        changed_contract_only=CONTRACT, workload=WORKLOAD, limits=strict.LIMITS,
        preserved_checks=verification['checks'],
        unchanged_agent_plans_sha256=fingerprint, canonical_agent_plans_equal=True,
        source_fullplan_query_consistency_verified=True,
        source_accepted_outputs_all_preserved=True, original_selection_all_candidates_preserved=True,
        source_two_warmup_proofs=warmup_proofs,
        actual_two_warmup_status='requires_run_observation; fail whole run if any arm cannot finish',
        original_strict16_failures=rejected['failures'],
        capacity_lower_bound_reference=rejected['initial_lower_bound'],
        capacity_claim_limits=[
            '1205-page initial full-history lower bound is compared to the recorded1192-page reference only; '
            'remeasure the actual allocator capacity of every tuned native arm, including .915.',
            'Eight offered clients do not prove eight simultaneous decoding requests.',
            'The current harness permits zero accepted output tokens; use zero as any generic growth lower bound. '
            'The1205-page initial bound itself does not rely on a positive output length.'],
        comparability_limits=[
            'This independent task cohort/request16K protocol is not directly comparable to the first8 natural32000 controls.',
            'Different own output lengths can change per-arm epoch endpoints, external-turn exposure, cold-prefill frequency '
            'and request mix. Report those alongside throughput and actual decode batches.',
            'No claim that all16 source turns execute, that complete tasks finish, or that this is on-policy tool quality.',
            'Epoch cleanup is per agent, with no global quiescence barrier at every reset.'],
        required_analysis=[
            'Per-agent epoch_ended reason, completed_turns, retained_history_tokens, next_prompt_tokens and cap.',
            'Per-task/source-turn exposure, prompt lengths, output lengths, finish reasons and restarted-epoch count.',
            'Accepted workload token rate in[T0,T1), actual decode batch observations, prefill/cleanup cost, '
            'preemptions, CPU history/SSD/GC checks, native reconciliation and normal final drain.'],
        runner_compatibility=dict(
            existing_workload='frozen/scripts/timed_agent_workload.py workload and agent_plans schema unchanged',
            fixture_required_keys=['eos', 'agent_plans', 'provenance'],
            new_manifest_experiment_kind=EXPERIMENT,
            manifest_workload=WORKLOAD,
            wrapper='Use a fresh dedicated preparation/verification wrapper. Existing benchmark_rolling_agents '
                    'hard-codes32000 and originalfirst8 selection and must not be relabeled or modified in frozen controls.',
            query='Retain explicit complete Query/mandatory plan; validate G32768 separately from request cap16384 '
                  'and require replay span to fit the existing query_replay_budget.',
            cpu='Use eight full262144-token history budgets plus staging/index and32GiB extra, '
                'requiring113834459136 bytes available; runtime8GiB floor remains.',
            eos_reference=dict(path=str(reference_path), sha256=sha(reference_path), eos=eos)),
        static_sources=[
            dict(path='artifacts/natural-rolling-reverse-20261017-r1/frozen/scripts/timed_agent_workload.py',
                 lines=[422, 428, 437, 458, 501, 518, 538, 546, 550, 586, 613],
                 meaning='Existing natural context guard, full own IDs, strict two-warmup barrier, '
                         'per-agent cleanup/restart, and final normal drain.'),
            dict(path='artifacts/natural-rolling-reverse-20261017-r1/frozen/scripts/benchmark_rolling_agents.py',
                 lines=[23, 60, 76, 84, 88, 94],
                 meaning='Original preparation checks initial+cap and feasible second warmup, keeps16 source records '
                         'and uses full per-user256K CPU history admission; it never requires all16 source-replay '
                         'prompts plus cap to fit.')])
    output.mkdir(parents=True, exist_ok=False)
    strict.write(output/'fixture.json', fixture)
    qualification['fixture_sha256'] = sha(output/'fixture.json')
    strict.write(output/'qualification.json', qualification)
    regression = dict(status='passed', scope='Lightweight canonical-content and metadata regression only; no tokenizer/GPU work',
        agent_plans_byte_serialization_sha256=fingerprint,
        unchanged_agent_plans=fixture['agent_plans']==original['agent_plans'],
        unchanged_protocol=fixture['protocol']==original['protocol'],
        all_original_strict_artifact_hashes_unchanged=all(sha(source/name)==digest for name,digest in original_hashes.items()),
        files_sha256={name:sha(output/name) for name in ('fixture.json', 'qualification.json')},
        preparation_entrypoint_sha256=sha(Path(__file__)), at=time.time())
    if not all(regression[k] for k in ('unchanged_agent_plans', 'unchanged_protocol',
                                       'all_original_strict_artifact_hashes_unchanged')):
        raise AssertionError('Canonical preservation regression failed')
    strict.write(output/'verification.json', regression)
    print(json.dumps(dict(status=qualification['status'], fixture_sha256=qualification['fixture_sha256'],
        unchanged_agent_plans_sha256=fingerprint, source_two_warmup_proofs=warmup_proofs), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/g16-long-history-canonical-20261017-r1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eos-reference', type=Path,
        default=ROOT/'artifacts/natural-rolling-reverse-20261017-r1/fixture.json')
    parser.add_argument('--not-before', type=float, default=1792212435.281)
    parser.add_argument('--guard-window', type=Path, action='append', default=[])
    args = parser.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise RuntimeError('PYTHONDONTWRITEBYTECODE=1 is required')
    prepare(args)


if __name__ == '__main__':
    main()
