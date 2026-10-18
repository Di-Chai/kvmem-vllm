"""Read-only verification for the independent natural context-epoch fixture.

Only JSON and individual artifact hashes; no tokenizer, GPU, full directory
hashing or source replay. The caller must run this outside formal windows and
separately verify full source provenance, runtime identity and current Query fit.
"""
import argparse
import json
from pathlib import Path

import prepare_g16_context_epoch_fixture as preparation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def verify_prepared(path):
    """Return (fixture, qualification, verification), failing closed on changes.

The caller's frozen-source manifest must also cover this verifier and its
preparation/strict/helper imports. A verifier cannot attest its own code without
that external identity. capacity-premises.json must be hashed into the caller's
manifest alongside fixture.json, qualification.json and verification.json.
"""
    path = Path(path).resolve()
    verification = read(path/'verification.json')
    require(verification.get('status') == 'passed', 'Context-epoch preparation regression did not pass')
    require(set(verification.get('files_sha256', {})) == {'fixture.json', 'qualification.json'},
            'Unexpected context-epoch verification file list')
    for name, digest in verification['files_sha256'].items():
        require(preparation.sha(path/name) == digest, f'Prepared artifact changed: {name}')
    require(verification.get('preparation_entrypoint_sha256') ==
            preparation.sha(Path(preparation.__file__)), 'Preparation entrypoint identity changed')
    for field in ('unchanged_agent_plans', 'unchanged_protocol', 'all_original_strict_artifact_hashes_unchanged'):
        require(verification.get(field) is True, f'Missing preservation regression: {field}')
    fixture, qualification = read(path/'fixture.json'), read(path/'qualification.json')
    require(qualification.get('schema') == 'g16_context_epoch_cpu_preparation_v1' and
            qualification.get('status') == 'prepared_context_epoch_cpu_only_no_gpu_run' and
            qualification.get('strict16_status') == 'rejected_unchanged', 'Wrong qualification contract')
    require(fixture.get('experiment_kind') == preparation.EXPERIMENT,
            'Not the independent natural context-epoch experiment')
    require(fixture.get('context_epoch_contract') == qualification.get('changed_contract_only') == preparation.CONTRACT,
            'Context-epoch admission, completion, retirement or warmup contract changed')
    require(fixture.get('protocol') == qualification.get('limits') == preparation.strict.LIMITS,
            'Request cap, K/G, context or full CPU-history budgets changed')
    require(qualification.get('workload') == preparation.WORKLOAD, 'Context-epoch workload changed')
    require(qualification.get('fixture_sha256') == verification['files_sha256']['fixture.json'],
            'Qualification does not identify this exact fixture')
    plans = fixture.get('agent_plans', [])
    require([(p.get('source_task'), p.get('source_turn')) for p in plans] == preparation.strict.EXPECTED,
            'Fixed eight source task identities or starts changed')
    require(all(len(p.get('turns', [])) == 16 for p in plans), 'All16 complete source records must remain present')
    digest = preparation.ids_sha(plans)
    require(digest == qualification.get('unchanged_agent_plans_sha256') ==
            verification.get('agent_plans_byte_serialization_sha256'), 'Canonical agent-plan fingerprint changed')
    require(qualification.get('canonical_agent_plans_equal') is True and
            qualification.get('source_fullplan_query_consistency_verified') is True and
            qualification.get('source_accepted_outputs_all_preserved') is True and
            qualification.get('original_selection_all_candidates_preserved') is True,
            'Required canonical/source-preservation proofs are missing')
    selection = fixture.get('selection', {})
    require(selection.get('users') == 8 and selection.get('turns') == 16 and
            selection.get('source_task_and_start_revision') is False,
            'Selection or maximum source-turn contract changed')
    source = Path(qualification['parent_rejected_directory'])
    strict_verification = read(source/'verification.json')
    require(strict_verification.get('all_pass') is True and
            strict_verification.get('files_sha256') == qualification.get('parent_rejected_hashes'),
            'Original strict16 evidence identity changed')
    for name, expected in qualification['parent_rejected_hashes'].items():
        require(preparation.sha(source/name) == expected, f'Original strict16 artifact changed: {name}')
    old_fixture = read(source/'rejected-canonical-fixture.json')
    old_qualification = read(source/'qualification.json')
    require(plans == old_fixture['agent_plans'], 'Canonical contents differ from the rejected strict16 parent')
    require(old_qualification.get('status') == 'rejected_no_gpu_run' and
            old_qualification.get('failures') == qualification.get('original_strict16_failures'),
            'Original strict16 rejection was removed or reclassified')
    for key, value in old_fixture['selection'].items():
        require(selection.get(key) == value, f'Original eligibility/negative-selection evidence changed: {key}')
    inherited = old_fixture['provenance']['source_file_sha256']
    current = fixture.get('provenance', {}).get('source_file_sha256', {})
    require(all(current.get(name) == value for name, value in inherited.items()),
            'Original source provenance map was altered')
    require(fixture['provenance']['tokenizer_file_sha256'] == old_fixture['provenance']['tokenizer_file_sha256'],
            'Tokenizer identity changed')
    for name in ('rejected-canonical-fixture.json', 'qualification.json', 'canonical-evidence.json', 'selection.json'):
        require(current.get(str(source/name)) == qualification['parent_rejected_hashes'][name],
                'Parent artifact identity is absent from current provenance')
    eos_reference = qualification['runner_compatibility']['eos_reference']
    require(type(fixture.get('eos')) is int and fixture['eos'] >= 0 and
            fixture['eos'] == eos_reference.get('eos'), 'Validated EOS changed')
    require(preparation.sha(Path(eos_reference['path'])) == eos_reference['sha256'] ==
            current.get(eos_reference['path']), 'Frozen EOS reference identity changed')
    for plan, proof in zip(plans, qualification['source_two_warmup_proofs']):
        initial = len(plan['initial_tokens'])
        limit = preparation.strict.CAP
        ceiling = preparation.strict.CONTEXT
        headroom = ceiling-limit-initial-len(plan['turns'][1]['tool_tokens'])
        require(initial+limit <= ceiling and headroom >= 1,
                'Canonical initial request or minimum second warmup no longer fits')
        require(proof['task'] == plan['source_task'] and proof['source_turn'] == plan['source_turn'] and
                proof['maximum_first_own_output_tokens_allowing_second_warmup'] == headroom and
                proof['source_two_warmups_fit'] is True and
                proof['actual_own_output_two_warmups_not_yet_measured'] is True and
                proof['source_second_prompt_tokens']+limit <= ceiling,
                'Source-vs-actual warmup qualification changed')
    require(len(qualification['source_two_warmup_proofs']) == 8, 'Need eight source warmup proofs')
    premises = read(path/'capacity-premises.json')
    required_premises = dict(schema='g16_native_capacity_lower_bound_premises_v1',
        native_prefix_caching_enabled=True, native_cache_salt_expression="f'{session}:epoch:{epoch}'",
        native_cache_namespace='per_agent_per_epoch', cross_agent_prefix_sharing=False,
        cross_epoch_prefix_sharing=False, initial_full_history_attention_pages=1085,
        minimum_active_recurrent_state_pages=120, initial_total_pages_lower_bound=1205,
        reference_native_usable_pages=1192, per_arm_native_allocator_capacity_must_be_remeasured=True,
        changes_to_native_apc_or_salt=False)
    require(all(premises.get(k) == v for k, v in required_premises.items()),
            'Native APC isolation or conditional allocator-bound premise changed')
    return fixture, qualification, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--not-before', type=float, default=1792212435.281)
    parser.add_argument('--guard-window', type=Path, action='append', default=[])
    args = parser.parse_args()
    preparation.strict.guard(args)
    fixture, qualification, verification = verify_prepared(args.path)
    print(json.dumps(dict(status='passed', experiment_kind=fixture['experiment_kind'],
        qualification_status=qualification['status'],
        files_sha256=verification['files_sha256'],
        capacity_premises_sha256=preparation.sha(args.path/'capacity-premises.json'))))


if __name__ == '__main__':
    main()
