"""Run two complete fixed Protocol inputs on one declared owned model arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

import benchmark_graph_v2_ablation as campaign
import analyze_graph_v2_ablation as analysis
import qualify_block_graph as q
import qualify_graph_niah as launch
from graph_v2_protocol import require, sha


def validate_fixture(fixture, profile):
    require(fixture.get('schema') == campaign.SCHEMA and fixture.get('sha256') ==
        q.digest({k:v for k,v in fixture.items() if k != 'sha256'}), 'Frozen ablation fixture changed')
    require(len(fixture['cases']) == fixture['concurrency'] == 2 and fixture['quality_gold'] is None,
        'Require exactly two unlabeled fixed original owners')
    require(sha(fixture['bridge']) == fixture['bridge_sha256'], 'Frozen producer transport changed')
    for case in fixture['cases']:
        require(sha(case['input']) == case['input_sha256'], 'Frozen complete input changed')
        value = json.loads(Path(case['input']).read_text())
        expected = case['contract']
        for arm in campaign.ARMS:
            require(q.digest(campaign.plan_contract(value['tokens'],campaign.arm_plan(value['plan'],arm),profile)) ==
                q.digest(expected), 'Token/Query/replay/mandatory identity changed')


def run(args):
    profile, fixture = json.loads(args.config.read_text()), json.loads(args.fixture.read_text())
    campaign.validate_profile(profile, args.arm)
    validate_fixture(fixture,profile)
    require(not args.output.exists(), 'Use a fresh qualification output')
    args.output.mkdir(parents=True)
    proof = launch.audit_launch(profile,args.capture)
    effective = json.loads((args.capture.parent/'launch.json').read_text())
    argv = effective['vllm_argv']
    actual = json.loads(argv[argv.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    campaign.validate_profile(dict(profile,kvmem=actual),args.arm)
    campaign.save(args.output/'launch-audit.json',proof)
    spec = importlib.util.spec_from_file_location('frozen_ablation_bridge',fixture['bridge'])
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    sessions = [f'graph-fixed-{i}-{uuid.uuid4().hex}' for i in range(2)]
    failures, rows, releases = [], [], []

    def execute(index):
        case = fixture['cases'][index]
        value = json.loads(Path(case['input']).read_text())
        directory = args.output/case['id']
        directory.mkdir()
        plan = campaign.arm_plan(value['plan'],args.arm)
        payload = bridge.backend_request(model=profile['served_model_name'],tokens=value['tokens'],
            session=sessions[index],max_tokens=fixture['output_tokens'],plan=plan,backend_kind='kvmem')
        payload.update(fixture['sampling'],stream=False)
        campaign.save(directory/'request.json',{k:v for k,v in payload.items() if k != 'prompt'} |
            dict(input_tokens=len(value['tokens']),input_sha256=q.digest(value['tokens'])))
        request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        started = time.monotonic()
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=1800) as stream:
            response = json.load(stream)
        choice, = response['choices']
        output = choice.get('token_ids')
        require(isinstance(output,list) and output and all(type(v) is int and v >= 0 for v in output), 'Missing actual accepted output IDs')
        require(choice['prompt_token_ids'] == value['tokens'] and response['usage']['prompt_tokens'] == len(value['tokens']) and
            response['usage']['completion_tokens'] == len(output) <= fixture['output_tokens'], 'Original prompt or output accounting changed')
        row = dict(id=case['id'],response_id=response['id'],session=sessions[index],output_ids=output,
            input_sha256=q.digest(value['tokens']),input_tokens=len(value['tokens']),
            text=choice['text'],seconds=time.monotonic()-started,finish_reason=choice['finish_reason'])
        campaign.save(directory/'response.json',dict(native=response,row=row))
        rows.append(row)
        return row

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(execute,index) for index in range(2)]
            for future in futures:
                try: future.result()
                except Exception as error: failures.append(repr(error))
        require(not failures, '; '.join(failures))
        archive = q.wait_terminal_publication(args.capture,rows)
        for row in rows:
            case = next(c for c in fixture['cases'] if c['id'] == row['id'])
            value = json.loads(Path(case['input']).read_text())
            completed = [event for event in archive['completed'] if q.native_matches(row,event)]
            require(completed and completed[-1].get('terminal_prefix',{}).get('history_end') ==
                len(value['tokens'])+len(row['output_ids'])-1, 'Terminal accepted history differs')
            evidence, _ = analysis.audit_live_case(args.capture,row,case,profile,args.arm,value)
            campaign.save(args.output/case['id']/'evidence.json',evidence)
    except BaseException as error:
        failures.append(repr(error))
    finally:
        for session in sessions:
            try:
                releases.append(q.stream_request(profile,session,[fixture['eos_token_id']],1,0,release=True,timeout=300))
            except Exception as error: failures.append('release: '+repr(error))
        cleared = None
        if len(releases) == 2:
            try: cleared = q.wait_released(args.capture,set(sessions))
            except Exception as error: failures.append('retirement: '+repr(error))
        campaign.save(args.output/'cleanup.json',dict(releases=releases,state=cleared,errors=failures))
        result = dict(status='failed' if failures else 'passed',arm=args.arm,cases=len(rows),
            all_owned_sessions_released=cleared is not None,errors=failures,results=rows,
            fixture_sha256=sha(args.fixture),quality_gold=None,answer_correctness=None,retrieval_accuracy=None,
            scope='Real model fixed-input cold requests plus complete own-snapshot native selection closure; no quality or throughput claim')
        campaign.save(args.output/'results.json',result)
    require(not failures, '; '.join(failures))
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=campaign.ARMS,required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
