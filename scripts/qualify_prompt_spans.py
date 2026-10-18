"""CPU acceptance of real DeepSWE renderer/tokenizer graph metadata (no inference).

Use a protocol.so produced by freeze_deepswe_protocol.py. This validates exact
production inputs and boundaries; it does not measure model or retrieval quality.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def qualify(protocol, tokenizer_path):
    from transformers import AutoTokenizer
    from vllm_kvmem.request_plan import RequestPlan
    spec = importlib.util.spec_from_file_location('qualify_locality_bridge', ROOT/'benchmarks/deepswe/claude_bridge.py')
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    planner = bridge.Planner(bridge.Protocol(protocol), tokenizer)
    payload = dict(model='cpu-protocol-acceptance', max_tokens=64, system='检查消息边界，保留原始文本。',
        tools=[dict(name='Bash', description='Run a command', input_schema=dict(type='object',
            properties=dict(command=dict(type='string'))))],
        messages=[dict(role='user',content='重复内容🙂'), dict(role='assistant',content=[
            dict(type='tool_use',id='one',name='Bash',input=dict(command='one')),
            dict(type='tool_use',id='two',name='Bash',input=dict(command='two'))]),
            dict(role='user',content=[dict(type='tool_result',tool_use_id='one',content='重复内容🙂'),
                                     dict(type='tool_result',tool_use_id='two',content='')])])
    tokens, plan, rendered, *_ = planner.prepare(payload, bridge.Trajectory())
    expected = tokenizer.encode(rendered['prompt'], add_special_tokens=False)
    if tokens != expected:
        raise AssertionError('Metadata producer changed production token IDs')
    original = {k:v for k,v in plan.items() if k not in ('kvmem_locality_spans','kvmem_locality_provenance')}
    before, after = [RequestPlan.parse(p, len(tokens), 32, planner.gen_budget) for p in (original, plan)]
    for field in ('query','replay','mandatory','explicit_mandatory'):
        if getattr(before, field) != getattr(after, field):
            raise AssertionError('Metadata producer changed ' + field)
    spans = plan['kvmem_locality_spans']
    if sum(s['role'] == 'tool' for s in spans) != 2:
        raise AssertionError('Independent tool-result boundaries were lost')
    if plan['kvmem_locality_provenance']['unknown_spans']:
        raise AssertionError('Qualification template has unresolved complete message boundaries')
    return dict(schema_version=1, passed=True, scope='CPU real renderer/tokenizer metadata only; no inference',
        token_ids_unchanged=True, query_replay_mandatory_unchanged=True, prompt_tokens=len(tokens),
        source_messages=len(rendered['spans']), locality_spans=spans,
        provenance=plan['kvmem_locality_provenance'],
        protocol_sha256=hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
        tokenizer_json_sha256=hashlib.sha256((Path(tokenizer_path)/'tokenizer.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a prior qualification record')
    result = qualify(args.protocol, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], prompt_tokens=result['prompt_tokens'],
                         locality_spans=len(result['locality_spans']))))


if __name__ == '__main__':
    main()
