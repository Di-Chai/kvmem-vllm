"""Freeze four real-input graph ablations; prepare never starts CUDA or HTTP.

Every arm receives two complete, previously accepted Protocol turn-1 inputs.
Cold requests fix the inputs across arms; this is not another multi-turn
restore test, a throughput benchmark, or an answer-quality experiment.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import benchmark_block_graph as owned
import benchmark_graph_v2 as shared
from graph_v2_protocol import require, sha
from qualify_block_graph import digest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'kvmem-graph-v2-fixed-ablation-1'
ARMS = ('v1-attention', 'v2-attention', 'v2-multi-unknown', 'v2-multi-known')
POLICY = {'v1-attention': ('v1', 'attention', False),
          'v2-attention': ('v2', 'attention', False),
          'v2-multi-unknown': ('v2', 'multi', False),
          'v2-multi-known': ('v2', 'multi', True)}
HELPERS = tuple(dict.fromkeys((*shared.HELPERS, 'analyze_block_graph.py',
    'benchmark_graph_v2_ablation.py', 'qualify_graph_v2_ablation.py', 'analyze_graph_v2_ablation.py')))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def arm_plan(plan, arm):
    require(arm in ARMS, 'Unknown fixed-input arm')
    result = deepcopy(plan)
    if not POLICY[arm][2]:
        for field in ('kvmem_locality_spans', 'kvmem_locality_provenance'):
            result.pop(field, None)
    return result


def plan_contract(tokens, plan, profile):
    from vllm_kvmem.request_plan import RequestPlan
    kv = profile['kvmem']
    parsed = RequestPlan.parse(plan, len(tokens), kv['block_tokens'], kv['gen_budget'])
    return dict(token_ids_sha256=digest(tokens), query=parsed.query, replay=parsed.replay,
        replay_begin=parsed.replay_begin, replay_end=parsed.replay_end,
        mandatory=parsed.mandatory, explicit_mandatory=parsed.explicit_mandatory)


def profiles(base, port):
    common = deepcopy(base)
    common.update(port=port, served_model_name='graph-v2-fixed-ablation')
    kv = common['kvmem']
    for key in ('graph_alpha', 'graph_locality_weight', 'graph_attention_weight', 'action_pressure'):
        kv.pop(key, None)
    kv.update(pressure_policy='recency', graph_trace=True)
    result = {}
    for arm, (version, mode, _) in POLICY.items():
        profile = deepcopy(common)
        profile['kvmem'].update(graph_scoring_version=version, graph_mode=mode)
        if version == 'v1':
            profile['kvmem'].update(graph_locality_weight=0., graph_attention_weight=.2)
        else:
            profile['kvmem']['graph_alpha'] = .2
        validate_profile(profile, arm)
        result[arm] = profile
    return result


def validate_profile(profile, arm):
    version, mode, _ = POLICY[arm]
    kv = profile['kvmem']
    require(profile.get('model_format') == 'safetensors' and
        profile.get('execution_mode') == 'compiled_graph' and
        profile.get('draft_method') == 'mtp' and profile.get('draft_tokens') == 4,
        'Keep native safetensors compiled MTP4')
    require(profile.get('max_num_seqs', 0) >= 2 and kv.get('session_capacity', 0) >= 2,
        'Two independent owners must fit')
    require((kv.get('block_tokens'), kv.get('select_budget'), kv.get('gen_budget'),
             kv.get('sink_blocks'), kv.get('recent_blocks')) == (32, 36864, 16384, 41, 0),
        'Keep the original block/K/G/sink/recent geometry')
    require(kv.get('request_planning') and kv.get('archive_backend') == 'cpu' and
        kv.get('mean_k_backend') == 'qw3_cuda' and not kv.get('repository_gc') and
        not kv.get('query_history') and kv.get('guided_query_tokens', 0) == 0 and
        kv.get('action_fraction') == 0, 'Keep aligned Mean-K, mandatory and complete CPU history')
    require(kv.get('pressure_policy') == 'recency' and kv.get('graph_trace') is True and
        kv.get('graph_mode') == mode and kv.get('graph_scoring_version') == version,
        'Arm must use declared Query scoring with common recency pressure and bounded export')
    require(kv.get('action_attention') and kv.get('prefill_attention') and
        kv.get('action_scope') == 'non-thinking' and kv.get('graph_cross_message_weight') == .25,
        'Keep common accepted attention capture and boundary coefficient')
    if version == 'v1':
        require('graph_alpha' not in kv and kv.get('graph_attention_weight') == .2 and
            kv.get('graph_locality_weight') == 0, 'V1 uses only legacy attention weight .2')
    else:
        require(kv.get('graph_alpha') == .2 and not {'graph_locality_weight', 'graph_attention_weight'} & kv.keys(),
            'V2 uses only shared alpha .2')


def validate_profiles(configs):
    require(set(configs) == set(ARMS), 'All four arms are required')
    stripped = []
    for arm, value in configs.items():
        validate_profile(value, arm)
        item = deepcopy(value)
        for key in ('graph_scoring_version', 'graph_mode', 'graph_alpha', 'graph_locality_weight', 'graph_attention_weight'):
            item['kvmem'].pop(key, None)
        stripped.append(item)
    require(all(item == stripped[0] for item in stripped), 'A non-ablation profile setting differs')


def freeze_inputs(source, root, profile):
    """Use real accepted IDs and producer evidence, never re-render or clip."""
    original = json.loads((source/'manifest.json').read_text())
    require(json.loads((source/'status.json').read_text())['status'] == 'passed' and
        json.loads((source/'qualification/results.json').read_text())['status'] == 'passed',
        'Source must be a completed real structured session qualification')
    for name, expected in original['runtime_sha256'].items():
        require(sha(source/'runtime'/name) == expected, 'Original producer runtime changed: '+name)
    for key, name in (('profile_sha256','profile.json'), ('fixture_sha256','fixture.json')):
        require(sha(source/name) == original[key], 'Original structured session configuration changed')
    names = ['manifest.json', 'profile.json', 'fixture.json', 'status.json',
        'qualification/manifest.json', 'qualification/results.json', 'qualification/cleanup.json',
        'runtime/benchmarks/deepswe/claude_bridge.py', 'runtime/protocol/protocol.so']
    cases = []
    eos = set()
    for owner in range(2):
        prefix = f'qualification/agent-{owner}-turn-'
        input_name, previous_name, response_name = prefix+'1-input.json', prefix+'0-input.json', prefix+'0-response.json'
        names += [input_name, previous_name, response_name]
        row = json.loads((source/input_name).read_text())
        previous = json.loads((source/previous_name).read_text())
        response = json.loads((source/response_name).read_text())
        accepted = response['row']['output_ids']
        require(accepted and row['tokens'][:len(previous['tokens'])+len(accepted)] == previous['tokens']+accepted,
            'Fixed continuation lost the real accepted assistant prefix')
        eos.add(accepted[-1])
        calls = {part['id'] for part in response['response']['content'] if part['type'] == 'tool_use'}
        results = {part['tool_use_id'] for part in row['payload']['messages'][-1]['content'] if part['type'] == 'tool_result'}
        require(len(calls) >= 2 and calls == results, 'Frozen tool results do not bind actual model tool-use IDs')
        require(row['plan'].get('kvmem_locality_spans') and
            row['plan']['kvmem_locality_provenance'].get('history') == 'actual_accepted_ids' and
            row['plan']['kvmem_locality_provenance'].get('source') == 'deepswe_protocol',
            'Missing real producer locality provenance')
        contract = plan_contract(row['tokens'], row['plan'], profile)
        require(all(plan_contract(row['tokens'], arm_plan(row['plan'], arm), profile) == contract for arm in ARMS),
            'Graph-only ablation changed token/Query/replay/mandatory contract')
        require(len(row['tokens']) > 36864+3*16384 and len(row['tokens'])+257 <= profile['max_model_len'],
            'Keep intact long inputs capable of multiple pressure selections')
        cases.append(dict(id=f'owner-{owner}-turn-1', owner=owner,
            input=str(root/'source'/input_name), input_sha256=sha(source/input_name),
            contract=contract, real_tool_use_ids=sorted(calls),
            previous_response_sha256=sha(source/response_name)))
    require(len(eos) == 1, 'Source owners disagree on the accepted message EOS')
    for name in names:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    fixture = dict(schema=SCHEMA, cases=cases, original_source=str(source), concurrency=2,
        bridge=str(root/'source/runtime/benchmarks/deepswe/claude_bridge.py'),
        bridge_sha256=sha(root/'source/runtime/benchmarks/deepswe/claude_bridge.py'),
        protocol_sha256=sha(root/'source/runtime/protocol/protocol.so'),
        eos_token_id=eos.pop(), output_tokens=257,
        sampling=dict(thinking_token_budget=0, temperature=0, min_tokens=0, seed=1000),
        scope='Two cold complete frozen turn-1 requests per arm; historical accepted IDs are fixed, fresh outputs never feed another arm',
        quality_gold=None)
    fixture['sha256'] = digest(fixture)
    return fixture


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root/'status.json', dict(status='preparing'))
    try:
        source = args.source.resolve()
        base = json.loads((args.profile or source/'profile.json').read_text())
        configs = profiles(base, args.port)
        validate_profiles(configs)
        from vllm_kvmem.serve import build_launch
        for arm, profile in configs.items():
            build_launch(profile, ROOT, root/arm/'server')
            save(root/(arm+'.json'), profile)
        fixture = freeze_inputs(source, root, configs[ARMS[0]])
        save(root/'fixture.json', fixture)
        jobs, runtime = [], None
        for arm in ARMS:
            target = root/arm
            owned.prepare(SimpleNamespace(output=target, fixture=root/'fixture.json', arm=arm,
                profile=root/(arm+'.json'), qualifier=ROOT/'scripts/qualify_graph_v2_ablation.py',
                synchronous=False, port=args.port, client_timeout_seconds=7200))
            for name in HELPERS:
                shutil.copyfile(ROOT/'scripts'/name, target/'runtime/scripts'/name)
            hashes = {str(p.relative_to(target/'runtime')):sha(p) for p in (target/'runtime').rglob('*') if p.is_file()}
            require(runtime is None or runtime == hashes, 'Runtime changed during four-arm freeze')
            runtime = hashes
            manifest = json.loads((target/'manifest.json').read_text())
            manifest['runtime_sha256'] = hashes
            save(target/'manifest.json', manifest)
            jobs.append(dict(id=arm, manifest_sha256=sha(target/'manifest.json')))
        (root/'frozen').mkdir()
        for name in HELPERS:
            shutil.copyfile(ROOT/'scripts'/name, root/'frozen'/name)
            require(sha(root/'frozen'/name) == runtime['scripts/'+name], 'Controller changed during freeze')
        save(root/'manifest.json', dict(schema=SCHEMA, jobs=jobs, fixture_sha256=sha(root/'fixture.json'),
            runtime_sha256=runtime, model_identity=shared.model_identity(base),
            frozen_sha256={p.name:sha(p) for p in (root/'frozen').iterdir()},
            source_sha256={str(p.relative_to(root)):sha(p) for p in (root/'source').rglob('*') if p.is_file()},
            canonical_arm='v2-multi-known', allowed_differences=['graph scoring version/operator/weight', 'graph-only metadata'],
            quality_gold=None, gpu_started=False))
        save(root/'status.json', dict(status='prepared'))
        print(json.dumps(dict(prepared=str(root), gpu_started=False, cases=8)), flush=True)
    except BaseException as error:
        save(root/'status.json', dict(status='failed', error=repr(error)))
        raise


def validate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    require(manifest['schema'] == SCHEMA and [j['id'] for j in manifest['jobs']] == list(ARMS), 'Unknown ablation campaign')
    require(sha(root/'fixture.json') == manifest['fixture_sha256'], 'Frozen input fixture changed')
    for group, base in (('frozen_sha256', root/'frozen'), ('source_sha256', root)):
        for name, expected in manifest[group].items():
            require(sha(base/name) == expected, 'Frozen dependency changed: '+name)
    configs = {}
    for job in manifest['jobs']:
        directory = root/job['id']
        require(sha(directory/'manifest.json') == job['manifest_sha256'], 'Frozen job manifest changed')
        saved = json.loads((directory/'manifest.json').read_text())
        require(saved['runtime_sha256'] == manifest['runtime_sha256'], 'Arms use different runtimes')
        for name, expected in saved['runtime_sha256'].items():
            require(sha(directory/'runtime'/name) == expected, 'Frozen runtime changed: '+name)
        require(sha(directory/'profile.json') == saved['profile_sha256'] and
            sha(directory/'fixture.json') == saved['fixture_sha256'] == manifest['fixture_sha256'], 'Job profile/fixture changed')
        configs[job['id']] = json.loads((directory/'profile.json').read_text())
    validate_profiles(configs)
    identity = manifest['model_identity']
    for group in ('config_sha256', 'tokenizer_sha256'):
        for path, expected in identity[group].items():
            require(sha(path) == expected, 'Model/tokenizer config changed')
    for path, expected in identity['weight_stat_identity'].items():
        stat = Path(path).stat()
        require(dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == expected, 'Model weights changed')
    return manifest


def run(args):
    root = args.output.resolve()
    require(Path(__file__).resolve() == root/'frozen/benchmark_graph_v2_ablation.py', 'Run the frozen controller')
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate(root)
        require(json.loads((root/'status.json').read_text())['status'] == 'prepared', 'Use a fresh campaign; never retry a failed arm')
        save(root/'status.json', dict(status='running'))
        process = None
        try:
            with owned.cleanup_on_termination() as begin_cleanup:
                for job in manifest['jobs']:
                    target = root/job['id']
                    command = [sys.executable, '-B', str(target/'runtime/scripts/benchmark_graph_v2_ablation.py'),
                               'arm-run', '--output', str(target)]
                    with (target/'controller.log').open('x') as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
                        require(process.wait() == 0, 'Arm failed; retain logs: '+job['id'])
                    require(not owned.stop_owned_groups((process,)), 'Owned arm did not stop')
                    process = None
                begin_cleanup()
            save(root/'status.json', dict(status='passed', analysis='pending'))
        except BaseException as error:
            save(root/'status.json', dict(status='failed', error=repr(error)))
            raise
        finally:
            if process is not None:
                require(not owned.stop_owned_groups((process,)), 'Owned arm did not stop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'run', 'arm-run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/graph-v2-sessions-20261018/sync-r1')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--port', type=int, default=18678)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args)
    elif args.action == 'run': run(args)
    elif args.action == 'validate':
        manifest = validate(args.output.resolve())
        print(json.dumps(dict(status='validated', jobs=manifest['jobs'])))
    else:
        require(Path(__file__).resolve() == args.output.resolve()/'runtime/scripts/benchmark_graph_v2_ablation.py', 'Run the frozen arm controller')
        owned.run(args)


if __name__ == '__main__':
    main()
